#!/usr/bin/env python3
"""
Train PyTorch model on SageMaker
"""
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from datetime import datetime

# Configuration
REGION = 'us-east-1'
BUCKET_NAME = 'Traktoros-training-data'
ROLE_NAME = 'SageMakerExecutionRole'

def train_model(role_arn, training_data_s3):
    """Start SageMaker training job"""
    
    session = sagemaker.Session()
    
    print("Creating PyTorch estimator...")
    estimator = PyTorch(
        entry_point='train.py',
        source_dir='sagemaker',
        role=role_arn,
        framework_version='2.0.1',
        py_version='py39',
        instance_count=1,
        instance_type='ml.p3.2xlarge',  # GPU instance
        hyperparameters={
            'epochs': 10,
            'batch-size': 4,
            'learning-rate': 0.005
        },
        output_path=f's3://{BUCKET_NAME}/model/'
    )
    
    print(f"Starting training job...")
    print(f"Training data: {training_data_s3}")
    print("This may take 30-60 minutes...")
    
    estimator.fit({'training': training_data_s3})
    
    return estimator.model_data

def main():
    print("🚜 Traktoros SageMaker Training")
    print("=" * 50)
    
    # Initialize clients
    session = boto3.Session(region_name=REGION)
    iam_client = session.client('iam')
    
    # Get role
    print("\n1. Getting IAM role...")
    try:
        role = iam_client.get_role(RoleName=ROLE_NAME)
        role_arn = role['Role']['Arn']
        print(f"✓ Using role: {role_arn}")
    except:
        print(f"✗ Role {ROLE_NAME} not found. Please run deploy_sagemaker.py first.")
        return
    
    # Training data location
    training_data_s3 = f's3://{BUCKET_NAME}/training/'
    
    print(f"\n2. Starting training...")
    model_data = train_model(role_arn, training_data_s3)
    
    print(f"\n✓ Training complete!")
    print(f"\nModel saved to: {model_data}")
    print(f"\nNext step: Run deploy_sagemaker.py to deploy the model")

if __name__ == '__main__':
    main()
