#!/bin/bash
set -e

echo "=== BtF Law Checker — Railway Deploy ==="
echo ""

# Check Railway CLI
if ! command -v railway &> /dev/null; then
    echo "Installing Railway CLI..."
    npm install -g @railway/cli
fi

# Login check
if ! railway whoami &> /dev/null 2>&1; then
    echo "Logging into Railway..."
    railway login
fi

# Create project
echo "Creating Railway project..."
railway init -n btf-law-checker

# Deploy
echo "Deploying..."
railway up --detach

# Set API key
echo ""
read -sp "Enter your ANTHROPIC_API_KEY: " api_key
echo ""
railway vars set ANTHROPIC_API_KEY="$api_key"

# Generate domain
echo "Generating public URL..."
railway domain

echo ""
echo "=== Deploy complete! ==="
echo "Run 'railway logs' to monitor."
