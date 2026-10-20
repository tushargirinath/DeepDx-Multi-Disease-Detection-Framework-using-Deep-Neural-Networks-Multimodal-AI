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

# 5. Model weights sync script & initial check
mkdir -p "$APP_DIR/models"

# Create a helper script to sync model weights anytime from S3
cat << 'EOF_SYNC' > "$APP_DIR/sync_model.sh"
#!/bin/bash
set -e
cd /opt/deepdx
echo "--> Downloading model weights from s3://${s3_bucket_name}/models/best_model.weights.h5..."
if aws s3 cp "s3://${s3_bucket_name}/models/best_model.weights.h5" models/best_model.weights.h5; then
    echo "--> Model weights synced successfully: $(ls -lh models/best_model.weights.h5)"
    echo "--> Restarting DeepDx container to load updated weights..."
    docker compose restart
    echo "--> DeepDx is running with live model weights!"
else
    echo "--> ERROR: Failed to download weights from S3. Please ensure best_model.weights.h5 is uploaded to ${s3_bucket_name}/models/"
    exit 1
fi
EOF_SYNC

chmod +x "$APP_DIR/sync_model.sh"
chown ubuntu:ubuntu "$APP_DIR/sync_model.sh"

# Attempt to fetch model weights if they exist in S3 already
echo "--> Checking S3 for model weights: ${s3_bucket_name}..."
if aws s3 cp "s3://${s3_bucket_name}/models/best_model.weights.h5" models/best_model.weights.h5 2>/dev/null; then
    echo "--> Model weights found and downloaded successfully."
else
    echo "--> Model weights not yet in S3. The app will boot in standby mode."
    echo "--> Once you push weights to S3, run '/opt/deepdx/sync_model.sh' on EC2 to load them."
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

chown -R ubuntu:ubuntu "$APP_DIR"

# 7. Start DeepDx via Docker Compose
echo "--> Building and starting DeepDx container..."
docker compose up -d --build

echo "=========================================================="
echo " DeepDx EC2 Provisioning Complete!"
echo " Application is listening on port 8000"
echo "=========================================================="
