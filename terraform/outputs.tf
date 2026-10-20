# =====================================================================
#  DeepDx Terraform Outputs
# =====================================================================

output "s3_bucket_name" {
  description = "Name of the S3 bucket hosting the DeepDx model weights"
  value       = aws_s3_bucket.model_bucket.id
}

output "s3_model_weights_uri" {
  description = "Full S3 URI to the trained CNN model weights"
  value       = "s3://${aws_s3_bucket.model_bucket.id}/models/best_model.weights.h5"
}

output "ec2_instance_id" {
  description = "AWS EC2 Instance ID"
  value       = aws_instance.deepdx_app.id
}

output "ec2_public_ip" {
  description = "Public IPv4 address of the DeepDx EC2 instance"
  value       = aws_instance.deepdx_app.public_ip
}

output "deepdx_app_url" {
  description = "Live URL to access the DeepDx Diagnostic Console"
  value       = "http://${aws_instance.deepdx_app.public_ip}:8000/"
}

output "s3_manual_upload_command" {
  description = "Command to manually update weights in S3 whenever you retrain"
  value       = "aws s3 cp models/best_model.weights.h5 s3://${aws_s3_bucket.model_bucket.id}/models/best_model.weights.h5"
}
