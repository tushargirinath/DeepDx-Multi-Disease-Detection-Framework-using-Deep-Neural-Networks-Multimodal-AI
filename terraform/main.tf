# =====================================================================
#  DeepDx — Production AWS Infrastructure via Terraform
#  Target Region: Mumbai (ap-south-1)
#  Target Instance: m7i-flex.large (4th Gen Intel Xeon, 8 GB RAM)
#
#  Components Provisioned:
#  1. S3 Bucket (Private, AES256 Encrypted, Versioned) for Model Weights
#  2. IAM Role & Instance Profile (Zero-Secret EC2 -> S3 Access)
#  3. Security Group (Ports 22, 80, 443, 8000)
#  4. EC2 Instance (m7i-flex.large, Ubuntu 24.04 LTS, 30 GB gp3)
#  5. Cloud-Init User Data Script for Bootstrapping
# =====================================================================

terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "DeepDx"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

# Unique suffix for S3 bucket naming
resource "random_string" "suffix" {
  length  = 6
  special = false
  upper   = false
}

# =====================================================================
#  1. S3 BUCKET FOR MODEL WEIGHTS (Empty bucket - push done afterwards)
# =====================================================================

resource "aws_s3_bucket" "model_bucket" {
  bucket        = "${var.project_name}-models-${var.environment}-${random_string.suffix.result}"
  force_destroy = false
}

# Enforce strict private access (no public exposure)
resource "aws_s3_bucket_public_access_block" "model_bucket_block" {
  bucket = aws_s3_bucket.model_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Enable AES256 server-side encryption at rest
resource "aws_s3_bucket_server_side_encryption_configuration" "model_bucket_encryption" {
  bucket = aws_s3_bucket.model_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Enable versioning for clinical model rollback capability
resource "aws_s3_bucket_versioning" "model_bucket_versioning" {
  bucket = aws_s3_bucket.model_bucket.id

  versioning_configuration {
    status = "Enabled"
  }
}

# =====================================================================
#  2. IAM ROLE & INSTANCE PROFILE (Zero-Secret EC2 -> S3 Access)
# =====================================================================

# EC2 Trust Policy
resource "aws_iam_role" "ec2_role" {
  name = "${var.project_name}-ec2-s3-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

# Policy allowing read access to the specific model S3 bucket
resource "aws_iam_policy" "s3_read_policy" {
  name        = "${var.project_name}-s3-read-policy-${var.environment}"
  description = "Allows DeepDx EC2 to pull trained model weights from S3"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ListBucket"
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.model_bucket.arn
        ]
      },
      {
        Sid    = "GetObject"
        Effect = "Allow"
        Action = [
          "s3:GetObject"
        ]
        Resource = [
          "${aws_s3_bucket.model_bucket.arn}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "attach_s3_read" {
  role       = aws_iam_role.ec2_role.name
  policy_arn = aws_iam_policy.s3_read_policy.arn
}

resource "aws_iam_instance_profile" "ec2_profile" {
  name = "${var.project_name}-ec2-profile-${var.environment}"
  role = aws_iam_role.ec2_role.name
}

# =====================================================================
#  3. SECURITY GROUP (Firewall)
# =====================================================================

resource "aws_security_group" "deepdx_sg" {
  name        = "${var.project_name}-sg-${var.environment}"
  description = "Allow SSH, Web, and DeepDx application traffic"

  # SSH access (Port 22)
  ingress {
    description = "SSH access"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  # DeepDx Console & API (Port 8000)
  ingress {
    description = "DeepDx Diagnostic Console Web and API"
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # HTTP (Port 80)
  ingress {
    description = "HTTP web traffic"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # HTTPS (Port 443)
  ingress {
    description = "HTTPS encrypted web traffic"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # All outbound traffic (Required for Groq API, Docker Hub, Ubuntu updates)
  egress {
    description = "Allow all outbound traffic"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-sg"
  }
}

# =====================================================================
#  4. EC2 INSTANCE (m7i-flex.large)
# =====================================================================

# Get latest official Ubuntu 24.04 LTS AMI for x86_64
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical official owner ID

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# Optional Key Pair if public key provided
resource "aws_key_pair" "deployer" {
  count      = var.ssh_public_key != "" ? 1 : 0
  key_name   = "${var.project_name}-key-${var.environment}"
  public_key = var.ssh_public_key
}

# EC2 Instance
resource "aws_instance" "deepdx_app" {
  ami                  = data.aws_ami.ubuntu.id
  instance_type        = var.instance_type
  iam_instance_profile = aws_iam_instance_profile.ec2_profile.name

  vpc_security_group_ids = [aws_security_group.deepdx_sg.id]
  key_name               = var.ssh_public_key != "" ? aws_key_pair.deployer[0].key_name : null

  root_block_device {
    volume_size           = var.root_volume_size_gb
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true
    tags = {
      Name = "${var.project_name}-root-ebs"
    }
  }

  # Cloud-init User Data to automatically bootstrap the service
  user_data = templatefile("${path.module}/user_data.sh.tpl", {
    s3_bucket_name = aws_s3_bucket.model_bucket.id
    groq_api_key   = var.groq_api_key
    groq_model     = var.groq_model
    git_branch     = var.git_branch
  })

  tags = {
    Name = "${var.project_name}-app-server"
  }

  depends_on = [
    aws_iam_role_policy_attachment.attach_s3_read
  ]
}
