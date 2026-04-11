#!/usr/bin/env python3
"""
Deploy PyTorch model to SageMaker endpoint
"""
import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from datetime import datetime

# Configuration
REGION = 'us-east-1'
BUCKET_NAME = 'Traktoros-training-data'
ROLE_NAME = 'SageMakerExecutionRole'  # You'll need to create this

def get_or_create_role(iam_client, role_name):
    """Get or create SageMaker execution role"""
    try:
        role = iam_client.get_role(RoleName=role_name)
        return role['Role']['Arn']
    except:
        print(f"Creating IAM role {role_name}...")
        
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [{
                "Effect": "Allow",
                "Principal": {"Service": "sagemaker.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }]
        }
        
        role = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=str(trust_policy)
        )
        
        # Attach policies
        iam_client.attach_role_policy(
            RoleName=role_name,
            PolicyArn='arn:aws:iam::aws:policy/AmazonSageMakerFullAccess'
        )
        
        return role['Role']['Arn']

def deploy_model(model_data_url, role_arn):
    """Deploy model to SageMaker endpoint"""
    
    print("Creating PyTorch model...")
    pytorch_model = PyTorchModel(
        model_data=model_data_url,
        role=role_arn,
        entry_point='inference.py',
        source_dir='sagemaker',
        framework_version='2.0.1',
        py_version='py39',
    )
    
    endpoint_name = f'Traktoros-human-detection-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    
    print(f"Deploying to endpoint: {endpoint_name}")
    print("This may take 5-10 minutes...")
    
    predictor = pytorch_model.deploy(
        instance_type='ml.m5.xlarge',
        initial_instance_count=1,
        endpoint_name=endpoint_name
    )
    
    return endpoint_name

def main():
    print("🚜 Traktoros SageMaker Deployment")
    print("=" * 50)
    
    # Initialize clients
    session = boto3.Session(region_name=REGION)
    iam_client = session.client('iam')
    
    # Get or create role
    print("\n1. Setting up IAM role...")
    role_arn = get_or_create_role(iam_client, ROLE_NAME)
    print(f"✓ Using role: {role_arn}")
    
    # Model location (after training)
    model_data_url = f's3://{BUCKET_NAME}/model/model.tar.gz'
    
    print(f"\n2. Deploying model from {model_data_url}")
    endpoint_name = deploy_model(model_data_url, role_arn)
    
    print(f"\n✓ Deployment complete!")
    print(f"\nEndpoint name: {endpoint_name}")
    print(f"\nUpdate your Lambda function with this endpoint name.")

if __name__ == '__main__':
    main()
