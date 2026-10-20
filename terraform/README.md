# DeepDx — AWS Production Infrastructure (Terraform)

Production Infrastructure-as-Code (IaC) configuration deploying DeepDx in the **Mumbai region (`ap-south-1`)** on an **`m7i-flex.large`** EC2 instance, with a dedicated S3 model bucket, a permanent **Elastic IP**, and **automated HTTPS via Caddy reverse proxy**.

---

## Live Architecture Overview

```
                                  AWS Mumbai (ap-south-1)
 +-----------------------------------------------------------------------------------------------+
 |                                                                                               |
 |   +------------------------------------+          +----------------------------------------+  |
 |   |          S3 Model Bucket           |          |         EC2 Compute Instance           |  |
 |   |     (AES-256 Encrypted, Private)   |          |           (m7i-flex.large)             |  |
 |   |                                    |          |       Ubuntu 24.04 (30 GB gp3)         |  |
 |   |   models/best_model.weights.h5     |<---------|  - Docker Container (Port 8000)        |  |
 |   |   (Decoupled upload / rollback)    | IAM Role |  - Caddy Reverse Proxy (Ports 80/443)  |  |
 |   +------------------------------------+          |  - Groq LLM API (openai/gpt-oss-120b)  |  |
 |                                                   +----------------------------------------+  |
 |                                                                       ^                       |
 |                                                                       |                       |
 |                                                              +-----------------+              |
 |                                                              |   Elastic IP    |              |
 |                                                              |   65.2.235.24   |              |
 |                                                              +-----------------+              |
 +-----------------------------------------------------------------------|-----------------------+
                                                                         |
                                                   HTTPS (Port 443 with TLS Padlock 🔒)
                                                                         |
                                                              https://deepdx.myvnc.com
```

### Components Provisioned by Terraform:
1. **S3 Bucket (`deepdx-models-production-dbmbzc`)**: Private, AES-256 encrypted at rest, versioning enabled, all public access strictly blocked.
2. **IAM Role & Instance Profile**: EC2 instance accesses S3 seamlessly via IAM role (zero access keys stored on disk). Includes `AmazonSSMManagedInstanceCore` for secure remote management.
3. **Security Group**: Ingress rules for Port 22 (SSH), Port 80 (HTTP), Port 443 (HTTPS), and Port 8000 (Console API).
4. **EC2 Instance (`m7i-flex.large`)**: 2 vCPUs, 8 GB RAM, 4th Gen Intel Xeon Sapphire Rapids, 30 GB gp3 encrypted root volume.
5. **Elastic IP (`65.2.235.24`)**: Permanent static IPv4 attached to EC2 that never changes across reboots.
6. **Cloud-Init Bootstrapping**: Automatically installs Docker Engine, AWS CLI v2, clones repository, and launches container.
7. **Caddy Reverse Proxy**: Handles SSL certificate procurement via Let's Encrypt for `deepdx.myvnc.com` and terminates TLS on port 443.

---

## File Structure

| File | Purpose |
|---|---|
| `main.tf` | AWS provider, S3 bucket, IAM role/policies, Security Group, EC2 instance, and Elastic IP. |
| `variables.tf` | Region default (`ap-south-1`), instance default (`m7i-flex.large`), volume size, Groq model. |
| `outputs.tf` | Outputs S3 bucket name, EC2 static IP, live URL, and upload commands. |
| `terraform.tfvars.example` | Template configuration variables file. |
| `user_data.sh.tpl` | Cloud-init shell script executed on first boot to bootstrap the instance. |
| `upload_model.bat` | One-click Windows batch script to push model weights to S3. |
| `upload_model.sh` | Bash script for model weights upload on Linux / Mac / WSL. |

---

## Deployment Workflow

### 1. Initialize & Validate
```bash
terraform init
terraform validate
```

### 2. Review Execution Plan
```bash
terraform plan
```

### 3. Provision Infrastructure
```bash
terraform apply
```

### 4. Push Model Weights to S3
Once Terraform creates the bucket, upload the 105.7 MB trained CNN weights:
```cmd
upload_model.bat    # On Windows
./upload_model.sh   # On Linux / Mac
```

### 5. Domain & HTTPS Reverse Proxy
Point any domain (or free dynamic DNS like No-IP or DuckDNS) to your **Elastic IP (`65.2.235.24`)**.  
Caddy on the EC2 instance automatically issues and renews Let's Encrypt SSL certificates for HTTPS access.
