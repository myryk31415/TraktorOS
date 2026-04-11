# TraktorOS - Human Detection System

A demo system for detecting humans in autonomous tractor field of view using PyTorch on AWS SageMaker.

## Architecture

Three modes are available:

**Hosted mode (EC2)** — pretrained model on AWS, no local setup needed:
```
User → Frontend → EC2 (t3.xlarge) → Pretrained Faster R-CNN + MiDaS
                                      http://34.210.69.60:5000
```

**Local mode (pretrained)** — no AWS needed, great for offline dev:
```
User → Frontend → Local Flask Server → Pretrained Faster R-CNN + MiDaS
```

**SageMaker mode (custom trained)** — full AWS pipeline:
```
User → Frontend → API Gateway → Lambda → SageMaker Endpoint → PyTorch Model
                                    ↓
                                   S3 (Training Data)
```

**Components:**
- **Frontend**: HTML/CSS/JS for image upload and bounding box visualization
- **Local Server**: Flask server running pretrained Faster R-CNN (COCO, already detects people)
- **API Gateway + Lambda**: Lightweight proxy to SageMaker
- **SageMaker**: Train and deploy PyTorch models
- **S3**: Store training data and model artifacts
- **Model**: Faster R-CNN for human detection

## Project Structure

```
.
├── local_server.py          # Local inference server (pretrained model)
├── requirements.txt         # Local server dependencies
├── frontend/
│   ├── index.html           # Web interface
│   ├── app.js               # Frontend logic
│   └── style.css            # Styling
├── backend/
│   ├── lambda_function.py   # Lambda proxy to SageMaker
│   ├── requirements.txt     # Lambda dependencies
│   └── template.yaml        # SAM template
├── sagemaker/
│   ├── train.py             # Training script
│   ├── inference.py         # Inference handler
│   └── requirements.txt     # Model dependencies
├── scripts/
│   ├── upload_training_data.py   # Upload data to S3
│   ├── train_sagemaker.py        # Start training job
│   └── deploy_sagemaker.py       # Deploy model endpoint
└── data/
    └── training/            # Your training data goes here
        ├── images/
        └── annotations.json
```

## Quick Start (Local Pretrained Model)

No AWS account needed. Uses Faster R-CNN pretrained on COCO (already detects people).

```bash
pip install -r requirements.txt
python3 local_server.py
```

Then open `frontend/index.html` in your browser, select "Local (Pretrained)", and upload an image.

## SageMaker Setup (Custom Trained Model)

### Prerequisites

1. **AWS Account** with SageMaker access
2. **AWS CLI** configured with credentials
3. **Python 3.9+** with pip
4. **AWS SAM CLI** ([Installation Guide](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html))

### Step 1: Prepare Training Data

Create your training data directory:

```bash
mkdir -p data/training/images
```

Add your images and create `data/training/annotations.json`:

```json
[
  {
    "image": "images/tractor_field_001.jpg",
    "boxes": [[100, 150, 200, 400], [300, 200, 380, 450]]
  },
  {
    "image": "images/tractor_field_002.jpg",
    "boxes": [[50, 100, 150, 350]]
  }
]
```

**Box format**: `[x1, y1, x2, y2]` (top-left and bottom-right coordinates)

### Step 2: Upload Training Data to S3

```bash
python3 scripts/upload_training_data.py
```

This creates an S3 bucket and uploads your training data.

### Step 3: Train Model on SageMaker

```bash
pip install sagemaker boto3
python3 scripts/train_sagemaker.py
```

This will:
- Start a SageMaker training job (takes 30-60 minutes)
- Use GPU instances (ml.p3.2xlarge)
- Save trained model to S3

### Step 4: Deploy Model to SageMaker Endpoint

```bash
python3 scripts/deploy_sagemaker.py
```

This creates a SageMaker endpoint for real-time inference (takes 5-10 minutes).

### Step 5: Deploy API Gateway + Lambda

```bash
cd backend
sam build
sam deploy --guided
```

Configuration:
- Stack name: `traktoros-api`
- Region: Your preferred region
- SAGEMAKER_ENDPOINT: Use the endpoint name from Step 4

### Step 6: Configure Frontend

Update `frontend/app.js` with your API endpoint:

```javascript
const API_ENDPOINT = 'https://YOUR_API_ID.execute-api.REGION.amazonaws.com/prod/detect';
```

### Step 7: Test

Open `frontend/index.html` in your browser and upload a test image!

## How It Works

1. **Training Phase**:
   - Upload labeled images to S3
   - SageMaker trains Faster R-CNN on your data
   - Model saved to S3

2. **Inference Phase**:
   - User uploads image via web interface
   - API Gateway → Lambda (lightweight proxy)
   - Lambda → SageMaker endpoint (runs PyTorch model)
   - Bounding boxes returned and displayed

## Cost Estimates

- **S3**: ~$0.023/GB/month for storage
- **SageMaker Training**: ~$3.06/hour (ml.p3.2xlarge)
- **SageMaker Endpoint**: ~$0.269/hour (ml.m5.xlarge)
- **Lambda**: ~$0.20 per 1M requests
- **API Gateway**: ~$3.50 per 1M requests

**Tip**: Stop the SageMaker endpoint when not in use to save costs!

## Customization

### Change Model Hyperparameters

Edit `scripts/train_sagemaker.py`:

```python
hyperparameters={
    'epochs': 20,           # More epochs for better accuracy
    'batch-size': 8,        # Larger batch if you have more GPU memory
    'learning-rate': 0.001  # Lower for fine-tuning
}
```

### Use Different Instance Types

**Training** (in `train_sagemaker.py`):
- `ml.p3.2xlarge` - 1 GPU, good for small datasets
- `ml.p3.8xlarge` - 4 GPUs, faster training

**Inference** (in `deploy_sagemaker.py`):
- `ml.t2.medium` - Cheapest, slower inference
- `ml.m5.xlarge` - Balanced (recommended)
- `ml.p2.xlarge` - GPU inference for high throughput

## Troubleshooting

**"No module named sagemaker"**: Run `pip install sagemaker boto3`

**"Role not found"**: The script will create the IAM role automatically

**Training fails**: Check CloudWatch logs in SageMaker console

**Endpoint not responding**: Verify endpoint is "InService" in SageMaker console

**CORS errors**: Check API Gateway CORS settings in `template.yaml`

## Next Steps

- [ ] Collect more tractor field images
- [ ] Add data augmentation for better training
- [ ] Implement real-time video processing
- [ ] Add distance estimation using camera calibration
- [ ] Store detection logs in DynamoDB
- [ ] Add emergency stop integration
- [ ] Deploy frontend to S3 + CloudFront

## License

MIT
