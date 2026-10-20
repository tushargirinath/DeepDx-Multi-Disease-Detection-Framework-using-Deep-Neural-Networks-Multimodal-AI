#!/usr/bin/env bash
# =====================================================================
#  DeepDx — Automated AWS EC2 Provisioning Script
#  Target OS: Ubuntu 22.04 LTS / 24.04 LTS (x86_64)
#  Run: bash deploy_ec2.sh
# =====================================================================
set -euo pipefail

echo "========================================================"
echo " Starting DeepDx EC2 Provisioning & Docker Setup"
echo "========================================================"

# 1. Update system packages
echo "--> Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# 2. Install prerequisites & Git LFS
echo "--> Installing prerequisites and Git LFS..."
sudo apt-get install -y curl git git-lfs ca-certificates apt-transport-https gnupg lsb-release

# 3. Initialize Git LFS
echo "--> Initializing Git LFS..."
git lfs install

# 4. Install official Docker Engine & Docker Compose
echo "--> Setting up Docker apt repository..."
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
echo "--> Installing Docker and Compose plugin..."
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# 5. Enable and start Docker service
echo "--> Enabling Docker service..."
sudo systemctl enable --now docker

# 6. Add active user to docker group
CURRENT_USER=$(whoami)
echo "--> Adding user '$CURRENT_USER' to docker group..."
sudo usermod -aG docker "$CURRENT_USER"

# 7. Pull Git LFS model weights if running inside the deepdx repo
if [ -f "models/best_model.weights.h5" ]; then
    echo "--> Pulling Git LFS model weights..."
    git lfs pull
fi

echo "========================================================"
echo " EC2 Provisioning Complete!"
echo " Docker Version: $(docker --version)"
echo " Compose Version: $(docker compose version)"
echo ""
echo " Next Steps:"
echo " 1. Create or edit .env file (add your GROQ_API_KEY)"
echo " 2. Run: docker compose up -d --build"
echo " 3. Verify: curl http://localhost:8000/api/health"
echo "========================================================"
