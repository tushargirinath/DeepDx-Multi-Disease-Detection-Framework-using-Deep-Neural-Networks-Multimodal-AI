@echo off
REM =====================================================================
REM  DeepDx — Model Weights Upload Script for Windows
REM  Uploads models/best_model.weights.h5 to your provisioned S3 bucket
REM =====================================================================
echo.
echo ============================================================
echo   DeepDx Model Weights S3 Uploader (Mumbai ap-south-1)
echo ============================================================
echo.

REM Check if local weights exist
if not exist "..\models\best_model.weights.h5" (
    echo [ERROR] Model weights file not found at ..\models\best_model.weights.h5
    echo Please make sure the weights file exists before uploading.
    exit /b 1
)

echo [1/3] Reading S3 bucket name from Terraform output...
for /f "tokens=*" %%i in ('terraform output -raw s3_bucket_name 2^>nul') do set S3_BUCKET=%%i

if "%S3_BUCKET%"=="" (
    echo [WARNING] Could not automatically detect S3 bucket from Terraform.
    set /p S3_BUCKET="Please enter your S3 bucket name: "
)

if "%S3_BUCKET%"=="" (
    echo [ERROR] No bucket name provided. Aborting.
    exit /b 1
)

echo.
echo [2/3] Uploading models/best_model.weights.h5 to s3://%S3_BUCKET%/models/best_model.weights.h5 ...
aws s3 cp "..\models\best_model.weights.h5" "s3://%S3_BUCKET%/models/best_model.weights.h5" --region ap-south-1

if %ERRORLEVEL% equ 0 (
    echo.
    echo ============================================================
    echo [3/3] Upload Successful!
    echo S3 URI: s3://%S3_BUCKET%/models/best_model.weights.h5
    echo.
    echo Next step on your EC2 instance:
    echo SSH into your EC2 and run: /opt/deepdx/sync_model.sh
    echo ============================================================
) else (
    echo.
    echo [ERROR] Upload failed with error code %ERRORLEVEL%. Please check AWS credentials and region.
)
