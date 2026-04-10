import json
import base64
import boto3
import os

# SageMaker endpoint name (set via environment variable)
ENDPOINT_NAME = os.environ.get('SAGEMAKER_ENDPOINT', 'traktoros-human-detection')

# Initialize SageMaker runtime client
sagemaker_runtime = boto3.client('sagemaker-runtime')

def lambda_handler(event, context):
    """
    Lightweight Lambda proxy to SageMaker endpoint
    """
    try:
        # Parse request
        body = json.loads(event['body'])
        
        # Forward to SageMaker endpoint
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='application/json',
            Body=json.dumps({'image': body['image']})
        )
        
        # Parse SageMaker response
        result = json.loads(response['Body'].read().decode())
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(result)
        }
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': str(e)
            })
        }
