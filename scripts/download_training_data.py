#!/usr/bin/env python3
"""Download training data from S3."""
import boto3
import os

BUCKET_NAME = 'Traktoros-training-data'
REGION = 'us-east-1'
LOCAL_DIR = 'data/training'

s3 = boto3.client('s3', region_name=REGION)
paginator = s3.get_paginator('list_objects_v2')

for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix='data/'):
    for obj in page.get('Contents', []):
        key = obj['Key']
        local_path = os.path.join(LOCAL_DIR, key.removeprefix('data/'))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        print(f"Downloading {key} -> {local_path}")
        s3.download_file(BUCKET_NAME, key, local_path)

print("Done.")
