import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import boto3

class TractorDataset(Dataset):
    """Dataset for tractor human detection"""
    def __init__(self, data_dir, annotations_file):
        self.data_dir = data_dir
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        # Load image
        img_path = os.path.join(self.data_dir, ann['image'])
        image = Image.open(img_path).convert('RGB')
        image = F.to_tensor(image)
        
        # Parse boxes and labels
        boxes = torch.tensor(ann['boxes'], dtype=torch.float32)
        labels = torch.ones(len(boxes), dtype=torch.int64)  # All humans
        
        target = {
            'boxes': boxes,
            'labels': labels
        }
        
        return image, target

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(device)
    
    # Load dataset
    train_dataset = TractorDataset(
        args.train_dir,
        os.path.join(args.train_dir, 'annotations.json')
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=0.0005
    )
    
    # Training loop
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
        
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
    
    # Save model
    model_path = os.path.join(args.model_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # SageMaker parameters
    parser.add_argument('--train-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    # Hyperparameters
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--learning-rate', type=float, default=0.005)
    
    args = parser.parse_args()
    train(args)
