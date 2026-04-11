#!/bin/bash

echo "🚜 Traktoros Deployment Script"
echo "================================"

# Check if AWS SAM CLI is installed
if ! command -v sam &> /dev/null; then
    echo "❌ AWS SAM CLI not found. Please install it first:"
    echo "   https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html"
    exit 1
fi

# Build the SAM application
echo "📦 Building SAM application..."
cd backend
sam build

# Deploy
echo "🚀 Deploying to AWS..."
sam deploy --guided

echo "✅ Deployment complete!"
echo ""
echo "Next steps:"
echo "1. Copy the API endpoint URL from the output above"
echo "2. Update frontend/app.js with the API_ENDPOINT value"
echo "3. Open frontend/index.html in your browser to test"
