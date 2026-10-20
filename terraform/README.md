# DeepDx — AWS Terraform Deployment

Infrastructure-as-Code (IaC) configuration to deploy DeepDx in the **Mumbai region (`ap-south-1`)** on an **`m7i-flex.large`** EC2 instance with a dedicated S3 bucket for model weights.

---

## Architecture Overview

```
                                      AWS Mumbai (ap-south-1)
 +-----------------------------------------------------------------------------------------+
 |                                                                                         |
 |   +------------------------------------+          +----------------------------------+  |
 |   |          S3 Bucket                 |          |          EC2 Instance            |  |
 |   |   (Private, AES256 Encrypted)      |          |       (m7i-flex.large)           |  |
 |   |                                    |          |   Ubuntu 24.04 LTS (30 GB gp3)   |  |
 |   |   models/best_model.weights.h5     |<---------|   Docker + Groq LLM API          |  |
 |   |   (Uploaded manually or by script) | IAM Role |   Listening on Port 8000         |  |
 |   +------------------------------------+          +----------------------------------+  |
 |                                                                    ^                    |
 +--------------------------------------------------------------------|--------------------+
                                                                      | Port 8000 (HTTP)
                                                                 User Browser
```

### Components Provisioned by Terraform:
1. **S3 Bucket**: Private, AES-256 encrypted at rest, versioning enabled, with all public access strictly blocked.
2. **IAM Role & Instance Profile**: EC2 instance accesses the S3 bucket seamlessly using IAM instance profile credentials (no access keys or secrets stored on the instance).
3. **Security Group**: Ingress rules for Port 22 (SSH), Port 80 (HTTP), Port 443 (HTTPS), and Port 8000 (DeepDx Diagnostic Console).
4. **EC2 Instance**: `m7i-flex.large` (2 vCPUs, 8 GB RAM, 4th Gen Intel Xeon Sapphire Rapids) with 30 GB gp3 encrypted root EBS volume.
5. **Cloud-Init User Data**: Automatically provisions Docker, Docker Compose, AWS CLI, clones the repo, configures `.env`, and sets up the container.

---

## File Structure

| File | Purpose |
|------|---------|
| `main.tf` | Defines AWS provider, S3 bucket, IAM role/policy, Security Group, and EC2 instance. |
| `variables.tf` | Variable declarations (Region default: `ap-south-1`, Instance default: `m7i-flex.large`). |
| `outputs.tf` | Outputs S3 bucket name, EC2 public IP, application URL, and post-apply upload commands. |
| `terraform.tfvars.example` | Template configuration file to copy to `terraform.tfvars`. |
| `user_data.sh.tpl` | Cloud-init shell script executed on first boot to bootstrap the server. |
| `upload_model.bat` | Windows batch script to upload `best_model.weights.h5` to S3 post-apply. |
| `upload_model.sh` | Bash script to upload `best_model.weights.h5` to S3 post-apply. |

---

## Deployment Workflow

### Step 1: Configure Variables
Copy the example variables file:
```bash
cp terraform.tfvars.example terraform.tfvars
```
Edit `terraform.tfvars` and set:
- `groq_api_key`: Your Groq Cloud API key.
- `ssh_public_key`: (Optional) Your public SSH key if you plan to SSH into the instance.

### Step 2: Initialize & Review Plan (Do NOT apply until ready)
```bash
terraform init
terraform plan
```

### Step 3: Apply Infrastructure
```bash
terraform apply
```

### Step 4: Push Model Weights to S3 (After Apply)
Once Terraform provisions the S3 bucket, upload the 105 MB model weights using either:

**Option A — Windows Helper Script:**
```cmd
upload_model.bat
```

**Option B — Linux / Mac Helper Script:**
```bash
chmod +x upload_model.sh
./upload_model.sh
```

**Option C — Direct AWS CLI Command:**
```bash
aws s3 cp ../models/best_model.weights.h5 s3://<s3_bucket_name>/models/best_model.weights.h5 --region ap-south-1
```

### Step 5: Sync on EC2 (If EC2 booted before weights were uploaded)
If the EC2 instance finished booting before you uploaded the weights to S3, simply SSH into the server and run:
```bash
/opt/deepdx/sync_model.sh
```
This downloads the weights from S3 and restarts the DeepDx Docker container.
