#!/usr/bin/env python3
"""Local inference server using pretrained Faster R-CNN (COCO - already detects people)."""
import io
import base64
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import boto3
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

coco_core_agri_classes = {
    # Humans (highest priority)
    1: "person",

    # Vehicles / moving hazards
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    8: "truck",

    # Large animals (real farm risk)
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",

    # Small / fast unpredictable
    16: "bird",

    # Basic road interaction (edge of fields, crossings)
    10: "traffic light",
    13: "stop sign",

    # Generic obstacles (proxies for "stuff in the way")
    62: "chair",
    64: "potted plant",
    27: "backpack"
}

CONFIDENCE_THRESHOLD = 0.5

print("Loading pretrained Faster R-CNN...")
model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Model loaded on {device}")

print("Loading MiDaS depth model...")
midas = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small')
midas.eval().to(device)
midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms').small_transform
print("MiDaS loaded")


@app.route('/detect', methods=['POST'])
def detect():
    import time
    t0 = time.time()

    data = request.get_json()
    image_data = base64.b64decode(data['image'])
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    img_w, img_h = image.size
    t_decode = time.time()

    # Run Faster R-CNN
    img_tensor = F.to_tensor(image).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(img_tensor)[0]
    t_rcnn = time.time()

    # Run MiDaS depth estimation
    import cv2
    import numpy as np
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    midas_input = midas_transforms(img_cv).to(device)
    with torch.no_grad():
        depth = midas(midas_input)
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1), size=(img_h, img_w), mode='bilinear', align_corners=False
    ).squeeze().cpu().numpy()
    t_midas = time.time()

    # Normalize depth to 0-1 (higher = closer)
    depth_min, depth_max = depth.min(), depth.max()
    depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-6)

    detections = []
    for i in range(len(preds['boxes'])):
        if preds['labels'][i].item() in coco_core_agri_classes and preds['scores'][i].item() > CONFIDENCE_THRESHOLD:
            box = [int(x) for x in preds['boxes'][i].tolist()]
            x1, y1, x2, y2 = box

            # Sample depth in lower half of bbox (feet area, more reliable)
            mid_y = (y1 + y2) // 2
            region = depth_norm[mid_y:y2, x1:x2]
            rel_depth = float(region.mean()) if region.size > 0 else 0.0

            if rel_depth > 0.6:
                proximity = 'VERY CLOSE'
            elif rel_depth > 0.3:
                proximity = 'NEARBY'
            else:
                proximity = 'FAR'

            detections.append({
                'bbox': box,
                'confidence': float(preds['scores'][i].item()),
                'class': coco_core_agri_classes[preds['labels'][i].item()],
                'depth': round(rel_depth, 3),
                'proximity': proximity
            })

    # Encode depth map as base64 PNG for visualization
    depth_colored = cv2.applyColorMap((depth_norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    _, depth_png = cv2.imencode('.png', depth_colored)
    depth_b64 = base64.b64encode(depth_png).decode()
    t_end = time.time()

    timing = {
        'decode_ms': round((t_decode - t0) * 1000),
        'rcnn_ms': round((t_rcnn - t_decode) * 1000),
        'midas_ms': round((t_midas - t_rcnn) * 1000),
        'postprocess_ms': round((t_end - t_midas) * 1000),
        'total_ms': round((t_end - t0) * 1000),
    }
    print(f"[detect] {img_w}x{img_h} | decode={timing['decode_ms']}ms rcnn={timing['rcnn_ms']}ms midas={timing['midas_ms']}ms post={timing['postprocess_ms']}ms total={timing['total_ms']}ms")

    return jsonify({
        'detections': detections,
        'image_size': {'width': img_w, 'height': img_h},
        'depth_map': depth_b64,
        'timing': timing
    })


bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

BEDROCK_PROMPT = """Analyze this image from an autonomous tractor's camera. Respond ONLY with JSON in this exact format, no other text:

{
  "image_quality": {
    "issues": ["list of quality issues if any, e.g. too dark (e.g. night), blurry, overexposed, dust, lense flair. If it is good, just return sufficient, if it is obscured, warn that the detection results may be unreliable."]
    "sufficient": true/false, # if not sufficient, this should be false
  },
  "obstacles": [
    {"type": "person/animal/vehicle/rock/tree/fence/ditch/other", "severity": "critical/warning/info", "description": "brief description of the obstacle and risk. Max 8 words."}
  ],
  "ground_assessment": {
    "surface_type": "asphalt/gravel/grass/mud/water/snow/ice/mixed",
    "safety_to_traverse": "safe/caution/unsafe",
    "hazards": ["list of hazards affecting traversability, e.g. waterlogged, muddy, icy, flooded, soft ground, steep slope"]
  },
  "summary": "brief overall safety assessment for the tractor"
}

Severity levels:
- critical: immediate danger, tractor must stop (e.g. person, child, large animal)
- warning: obstacle that requires path adjustment (e.g. rock, ditch, fallen tree)
- info: minor obstacle, tractor can likely handle (e.g. small branch, puddle)"""


@app.route('/detect-bedrock', methods=['POST'])
def detect_bedrock():
    data = request.get_json()
    image_b64 = data['image']
    media_type = data.get('media_type', 'image/jpeg')

    print("Received image for Bedrock analysis, invoking model...")

    try:
        response = bedrock.converse(
            modelId='amazon.nova-pro-v1:0',
            messages=[{
                'role': 'user',
                'content': [
                    {'image': {'format': media_type.split('/')[-1], 'source': {'bytes': base64.b64decode(image_b64)}}},
                    {'text': BEDROCK_PROMPT}
                ]
            }]
        )

        text = response['output']['message']['content'][0]['text']
        # Extract JSON from response (handle markdown code blocks)
        if '```' in text:
            parts = text.split('```')
            if len(parts) >= 2:
                text = parts[1].strip()
                if text.lower().startswith('json'):
                    text = text[4:].strip()
        
        result = json.loads(text)
        return jsonify(result)

    except Exception as e:
        print(f"Bedrock API error: {str(e)}")
        return jsonify({
            "error": "Bedrock analysis failed",
            "details": str(e)
        }), 502




@app.route('/quality', methods=['POST'])
def quality():
    data = request.get_json()
    image_data = base64.b64decode(data['image'])
    arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())

    issues = []
    if blur < 100:
        issues.append('Blurry')
    if brightness < 40:
        issues.append('Too dark')
    elif brightness > 220:
        issues.append('Overexposed')
    if contrast < 20:
        issues.append('Low contrast')
    if w < 320 or h < 240:
        issues.append('Low resolution')

    return jsonify({
        'sufficient': len(issues) == 0,
        'issues': issues,
        'metrics': {
            'blur': round(blur, 1),
            'brightness': round(brightness, 1),
            'contrast': round(contrast, 1),
            'resolution': f'{w}x{h}'
        }
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
