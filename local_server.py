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

app = Flask(__name__)
CORS(app)

# COCO class index 1 = person
PERSON_CLASS = 1
CONFIDENCE_THRESHOLD = 0.5

print("Loading pretrained Faster R-CNN...")
model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"Model loaded on {device}")


@app.route('/detect', methods=['POST'])
def detect():
    data = request.get_json()
    image_data = base64.b64decode(data['image'])
    image = Image.open(io.BytesIO(image_data)).convert('RGB')

    img_tensor = F.to_tensor(image).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(img_tensor)[0]

    detections = []
    for i in range(len(preds['boxes'])):
        if preds['labels'][i].item() == PERSON_CLASS and preds['scores'][i].item() > CONFIDENCE_THRESHOLD:
            detections.append({
                'bbox': [int(x) for x in preds['boxes'][i].tolist()],
                'confidence': float(preds['scores'][i].item())
            })

    return jsonify({
        'detections': detections,
        'image_size': {'width': image.size[0], 'height': image.size[1]}
    })


bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

BEDROCK_PROMPT = """Analyze this image from an autonomous tractor's camera. Respond ONLY with JSON in this exact format, no other text:

{
  "image_quality": {
    "sufficient_for_human_detection": true/false,
    "issues": ["list of quality issues if any, e.g. too dark, blurry, overexposed"]
  },
  "obstacles": [
    {"type": "person/animal/vehicle/rock/tree/fence/ditch/other", "severity": "critical/warning/info", "description": "brief description of the obstacle and risk"}
  ],
  "soil_assessment": {
    "condition": "dry/wet/muddy/frozen/waterlogged",
    "traversability": "good/moderate/poor/impassable",
    "concerns": ["list of concerns, e.g. risk of getting stuck, erosion"]
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
        text = text.split('```')[1].removeprefix('json').strip()
    result = json.loads(text)

    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
