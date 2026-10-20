# =====================================================================
#  DeepDx Terraform Outputs
# =====================================================================

output "s3_bucket_name" {
  description = "Name of the S3 bucket created for model weights"
  value       = aws_s3_bucket.model_bucket.id
}

output "s3_bucket_arn" {
  description = "ARN of the S3 bucket"
  value       = aws_s3_bucket.model_bucket.arn
}

output "ec2_instance_id" {
  description = "AWS EC2 Instance ID (m7i-flex.large)"
  value       = aws_instance.deepdx_app.id
}

output "ec2_public_ip" {
  description = "Permanent Static Public IPv4 address (Elastic IP)"
  value       = aws_eip.deepdx_eip.public_ip
}

output "deepdx_app_url" {
  description = "Live URL to access the DeepDx Diagnostic Console"
  value       = "http://${aws_eip.deepdx_eip.public_ip}:8000/"
}

output "upload_model_command_windows" {
  description = "PowerShell / Command Prompt command to push model weights to S3 after apply"
  value       = "aws s3 cp ..\\models\\best_model.weights.h5 s3://${aws_s3_bucket.model_bucket.id}/models/best_model.weights.h5 --region ${var.aws_region}"
}

output "upload_model_command_linux" {
  description = "Bash command to push model weights to S3 after apply"
  value       = "aws s3 cp ../models/best_model.weights.h5 s3://${aws_s3_bucket.model_bucket.id}/models/best_model.weights.h5 --region ${var.aws_region}"
}

output "ssh_command" {
  description = "SSH command to connect to the EC2 server (if ssh_public_key was provided)"
  value       = "ssh -i <path-to-private-key.pem> ubuntu@${aws_eip.deepdx_eip.public_ip}"
}
