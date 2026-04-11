# TraktorOS - Human Detection System

A demo system for detecting humans and obstacles in an autonomous tractor's field of view using PyTorch on AWS.

**Live demo: [http://34.210.69.60](http://34.210.69.60)**

## Architecture

**Hosted mode (EC2)** — pretrained models on AWS, no local setup needed:
```
User → nginx (:80) → Static Frontend
                   → Flask API (:5000) → Faster R-CNN (object detection)
                                       → MiDaS (depth estimation)
                                       → Amazon Bedrock (scene analysis)
```

**Local mode** — for offline development:
```
User → frontend/index.html (file://) → localhost:5000 → Same Flask API
```

**SageMaker mode (custom trained)** — full AWS pipeline:
```
User → Frontend → API Gateway → Lambda → SageMaker Endpoint → Custom PyTorch Model
                                    ↓
                                   S3 (Training Data)
```

## Project Structure

```
.
├── local_server.py              # Flask API (detection + depth + Bedrock)
├── requirements.txt             # Python dependencies
├── nginx.conf                   # nginx: frontend + API proxy
├── traktoros.service            # systemd service for Flask API
├── frontend/
│   ├── index.html               # Web interface
│   ├── app.js                   # Frontend logic
│   ├── style.css                # Styling
│   └── icons/                   # UI icons
├── .github/workflows/
│   └── deploy.yml               # CI/CD: deploy to EC2 on push
├── scripts/
│   ├── detect_horizon.py        # Horizon line detection
│   ├── analyze_quality.py       # Image quality analysis
│   ├── evaluate_fasterrcnn.py   # Model evaluation on COCO
│   ├── coco_dataset.py          # COCO dataset loader
│   ├── upload_training_data.py  # Upload training data to S3
│   ├── train_sagemaker.py       # Start SageMaker training job
│   └── deploy_sagemaker.py      # Deploy SageMaker endpoint
├── backend/
│   ├── lambda_function.py       # Lambda proxy to SageMaker
│   └── template.yaml            # SAM template
├── sagemaker/
│   ├── train.py                 # Training script
│   └── inference.py             # Inference handler
├── models/                      # BRISQUE quality model files
└── data/training/               # Training data and annotations
```

## Quick Start

### Live demo

Go to [http://34.210.69.60](http://34.210.69.60) and upload an image.

### Run locally

```bash
pip install -r requirements.txt
python3 local_server.py
```

Open `frontend/index.html` in your browser. The frontend auto-detects `file://` and routes requests to `localhost:5000`.

## Detection Features

- **Object detection**: Faster R-CNN pretrained on COCO — detects people, vehicles, animals, and farm-relevant objects
- **Depth estimation**: MiDaS monocular depth — classifies detections as VERY CLOSE/NEARBY/FAR
- **Tractor path overlay**: Configurable corridor with horizon line to highlight objects in the tractor's path
- **Image quality check**: Blur, brightness, contrast, and resolution analysis
- **Scene analysis**: Amazon Bedrock (Nova Pro) for soil assessment, obstacle severity, and safety summary

## SageMaker Setup (Custom Model Training)

### Prerequisites

1. AWS Account with SageMaker access
2. AWS CLI configured
3. Python 3.9+
4. AWS SAM CLI

### Train and deploy

```bash
# Upload training data to S3
python3 scripts/upload_training_data.py

# Start training job (~30-60 min on ml.p3.2xlarge)
python3 scripts/train_sagemaker.py

# Deploy model endpoint (~5-10 min)
python3 scripts/deploy_sagemaker.py

# Deploy API Gateway + Lambda
cd backend && sam build && sam deploy --guided
```

Then update `frontend/app.js` with your API Gateway URL and select "SageMaker" mode in the UI.

## Deployment

Pushes to `main` auto-deploy to EC2 via GitHub Actions.

The workflow pulls latest code, installs deps if changed, updates nginx, and restarts the Flask API via systemd.

### GitHub Actions secrets

- `EC2_SSH_KEY`: Contents of the EC2 SSH private key

### Manual EC2 setup

```bash
ssh -i ~/.ssh/traktoros-key.pem ec2-user@34.210.69.60
sudo dnf install -y git python3-pip nginx
git clone https://github.com/myryk31415/TraktorOS.git ~/TraktorOS
cd ~/TraktorOS
pip3 install -r requirements.txt
sudo cp nginx.conf /etc/nginx/conf.d/traktoros.conf
sudo cp traktoros.service /etc/systemd/system/
chmod 711 /home/ec2-user
sudo systemctl enable --now nginx traktoros
```

## License

MIT
