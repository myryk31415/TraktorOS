#!/usr/bin/env python3
"""Local inference server using pretrained Faster R-CNN (COCO - already detects people)."""
import io
import base64
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
