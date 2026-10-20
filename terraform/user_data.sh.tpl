#!/bin/bash
# =====================================================================
#  DeepDx Automated Cloud-Init User Data Script
#  Executed automatically by EC2 upon first boot
# =====================================================================
set -euo pipefail

# Redirect all output to log for debugging
exec > >(tee /var/log/deepdx-init.log|logger -t deepdx-init -s 2>/dev/console) 2>&1

echo "=========================================================="
echo " Starting DeepDx EC2 Automated Provisioning"
echo "=========================================================="

# 1. Update system packages
export DEBIAN_FRONTEND=noninteractive
apt-get update && apt-get upgrade -y

# 2. Install prerequisites & AWS CLI
apt-get install -y \
    curl \
    git \
    ca-certificates \
    apt-transport-https \
    gnupg \
    lsb-release \
    unzip

# Install AWS CLI v2 if not present
if ! command -v aws &> /dev/null; then
    echo "--> Installing AWS CLI v2..."
    curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscliv2.zip"
    unzip -q /tmp/awscliv2.zip -d /tmp
    /tmp/aws/install
    rm -rf /tmp/aws /tmp/awscliv2.zip
fi

# 3. Install Docker Engine and Docker Compose plugin
echo "--> Installing Docker Engine..."
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

systemctl enable --now docker
usermod -aG docker ubuntu

# 4. Clone DeepDx Repository
APP_DIR="/opt/deepdx"
echo "--> Setting up application in $APP_DIR..."
mkdir -p "$APP_DIR"

git clone -b "${git_branch}" https://github.com/tushargirinath/DeepDx-Multi-Disease-Detection-Framework-using-Deep-Neural-Networks-Multimodal-AI.git "$APP_DIR"
cd "$APP_DIR"

# 5. Fetch trained CNN weights from S3 using IAM Instance Profile
echo "--> Fetching model weights from S3 bucket: ${s3_bucket_name}..."
mkdir -p models
aws s3 cp "s3://${s3_bucket_name}/models/best_model.weights.h5" models/best_model.weights.h5

# Verify weights downloaded properly
if [ -f "models/best_model.weights.h5" ]; then
    echo "--> Model weights verified successfully: $(ls -lh models/best_model.weights.h5)"
else
    echo "--> WARNING: Model weights not found in S3 yet. Falling back to local/demo mode."
fi

# 6. Generate production .env file
echo "--> Writing production .env configuration..."
cat << 'EOF' > .env
GROQ_API_KEY=${groq_api_key}
GROQ_MODEL=${groq_model}
HOST=0.0.0.0
PORT=8000
CNN_WEIGHTS_PATH=models/best_model.weights.h5
CLASS_NAMES_PATH=class_names.json
ENABLE_BIOGPT=false
ENABLE_MEDPALM=false
EOF

# 7. Start DeepDx via Docker Compose
echo "--> Building and starting DeepDx container..."
docker compose up -d --build

echo "=========================================================="
echo " DeepDx Provisioning Complete!"
echo " Application is listening on port 8000"
echo "=========================================================="
