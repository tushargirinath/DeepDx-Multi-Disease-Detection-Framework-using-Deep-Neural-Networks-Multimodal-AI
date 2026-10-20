# =====================================================================
#  DeepDx Terraform Variables
# =====================================================================

variable "aws_region" {
  description = "AWS region for all provisioned resources (Default: Mumbai ap-south-1)"
  type        = string
  default     = "ap-south-1"
}

variable "environment" {
  description = "Environment identifier (e.g. production, staging, dev)"
  type        = string
  default     = "production"
}

variable "project_name" {
  description = "Project name prefix for AWS resource naming"
  type        = string
  default     = "deepdx"
}

variable "instance_type" {
  description = "EC2 instance type for DeepDx inference (4th Gen Intel Xeon, 8 GB RAM)"
  type        = string
  default     = "m7i-flex.large"
}

variable "root_volume_size_gb" {
  description = "Size of the EC2 root EBS volume in GiB"
  type        = number
  default     = 30
}

variable "model_weights_path" {
  description = "Path to the local CNN weights file to upload to S3"
  type        = string
  default     = "../models/best_model.weights.h5"
}

variable "groq_api_key" {
  description = "Groq Cloud API Key for sub-second clinical LLM reasoning synthesis"
  type        = string
  sensitive   = true
  default     = ""
}

variable "groq_model" {
  description = "Groq model identifier"
  type        = string
  default     = "openai/gpt-oss-120b"
}

variable "ssh_public_key" {
  description = "Optional SSH public key string (e.g. ssh-rsa AAAA...) to inject for terminal access"
  type        = string
  default     = ""
}

variable "git_branch" {
  description = "Git branch to clone on the EC2 instance (e.g. feature/phase-1 or main)"
  type        = string
  default     = "feature/phase-1"
}

variable "allowed_ssh_cidr" {
  description = "CIDR block permitted to connect via SSH (Port 22). Restrict to your IP for production."
  type        = string
  default     = "0.0.0.0/0"
}
