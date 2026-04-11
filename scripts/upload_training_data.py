#!/usr/bin/env python3
"""
Upload training data to S3 for SageMaker training
"""
import boto3
import os
import json
from pathlib import Path

# Configuration
BUCKET_NAME = 'Traktoros-training-data'  # Change this to your bucket name
REGION = 'us-east-1'  # Change to your region

def create_bucket_if_not_exists(s3_client, bucket_name, region):
    """Create S3 bucket if it doesn't exist"""
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"✓ Bucket {bucket_name} already exists")
    except:
        print(f"Creating bucket {bucket_name}...")
        if region == 'us-east-1':
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={'LocationConstraint': region}
            )
        print(f"✓ Bucket created")

def upload_directory(s3_client, local_dir, bucket_name, s3_prefix):
    """Upload a directory to S3"""
    local_path = Path(local_dir)
    
    if not local_path.exists():
        print(f"✗ Directory {local_dir} does not exist")
        return
    
    files = list(local_path.rglob('*'))
    total_files = len([f for f in files if f.is_file()])
    
    print(f"Uploading {total_files} files from {local_dir}...")
    
    uploaded = 0
    for file_path in files:
        if file_path.is_file():
            relative_path = file_path.relative_to(local_path)
            s3_key = f"{s3_prefix}/{relative_path}"
            
            s3_client.upload_file(
                str(file_path),
                bucket_name,
                s3_key
            )
            uploaded += 1
            print(f"  [{uploaded}/{total_files}] {relative_path}")
    
    print(f"✓ Uploaded {uploaded} files")

def main():
    print("🚜 Traktoros Training Data Upload")
    print("=" * 50)
    
    # Initialize S3 client
    s3_client = boto3.client('s3', region_name=REGION)
    
    # Create bucket
    create_bucket_if_not_exists(s3_client, BUCKET_NAME, REGION)
    
    # Upload training data
    # Expected structure:
    # data/training/
    #   ├── images/
    #   │   ├── image1.jpg
    #   │   └── image2.jpg
    #   └── annotations.json
    
    training_dir = 'data'
    if os.path.exists(training_dir):
        upload_directory(s3_client, training_dir, BUCKET_NAME, 'data')
    else:
        print(f"\n⚠ Training directory '{training_dir}' not found")
        print("Please create it with the following structure:")
        print("  data/training/")
        print("    ├── images/")
        print("    │   ├── image1.jpg")
        print("    │   └── image2.jpg")
        print("    └── annotations.json")
        print("\nAnnotations format:")
        print(json.dumps([{
            "image": "images/image1.jpg",
            "boxes": [[x1, y1, x2, y2]]
        }], indent=2))
        return
    
    print(f"\n✓ All data uploaded to s3://{BUCKET_NAME}/data/")
    print(f"\nUse this S3 path in your SageMaker training job:")
    print(f"  s3://{BUCKET_NAME}/data/")

if __name__ == '__main__':
    main()
