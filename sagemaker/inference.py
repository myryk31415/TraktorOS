import json
import torch
import io
import base64
from PIL import Image
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

model = None

def model_fn(model_dir):
    """Load the PyTorch model"""
    global model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    model.load_state_dict(torch.load(f'{model_dir}/model.pth', map_location=device))
    model.to(device)
    model.eval()
    
    return model

def input_fn(request_body, content_type):
    """Deserialize input data"""
    if content_type == 'application/json':
        data = json.loads(request_body)
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        return image
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(image, model):
    """Run inference"""
    device = next(model.parameters()).device
    
    # Transform image
    img_tensor = F.to_tensor(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        predictions = model(img_tensor)
    
    return predictions[0], image.size

def output_fn(prediction, content_type):
    """Serialize predictions"""
    pred, img_size = prediction
    
    detections = []
    for i in range(len(pred['boxes'])):
        score = pred['scores'][i].item()
        
        if score > 0.5:  # Confidence threshold
            box = pred['boxes'][i].tolist()
            detections.append({
                'bbox': [int(x) for x in box],
                'confidence': float(score)
            })
    
    return json.dumps({
        'detections': detections,
        'image_size': {
            'width': img_size[0],
            'height': img_size[1]
        }
    })
