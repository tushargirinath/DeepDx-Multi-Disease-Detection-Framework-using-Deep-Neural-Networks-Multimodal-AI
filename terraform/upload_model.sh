#!/bin/bash
# =====================================================================
#  DeepDx — Model Weights Upload Script (Bash)
#  Uploads models/best_model.weights.h5 to your provisioned S3 bucket
# =====================================================================
set -euo pipefail

echo "============================================================"
echo "  DeepDx Model Weights S3 Uploader (Mumbai ap-south-1)"
echo "============================================================"

# Check if local weights exist
if [ ! -f "../models/best_model.weights.h5" ]; then
    echo "[ERROR] Model weights file not found at ../models/best_model.weights.h5"
    exit 1
fi

# Detect bucket name from Terraform output if available
S3_BUCKET=$(terraform output -raw s3_bucket_name 2>/dev/null || true)

if [ -z "$S3_BUCKET" ]; then
    read -rp "Please enter your S3 bucket name: " S3_BUCKET
fi

if [ -z "$S3_BUCKET" ]; then
    echo "[ERROR] No bucket name provided. Aborting."
    exit 1
fi

echo "--> Uploading ../models/best_model.weights.h5 to s3://$S3_BUCKET/models/best_model.weights.h5..."
aws s3 cp ../models/best_model.weights.h5 "s3://$S3_BUCKET/models/best_model.weights.h5" --region ap-south-1

echo "============================================================"
echo "[SUCCESS] Model weights uploaded to s3://$S3_BUCKET/models/best_model.weights.h5"
echo "To sync on your EC2 instance, run: /opt/deepdx/sync_model.sh"
echo "============================================================"
