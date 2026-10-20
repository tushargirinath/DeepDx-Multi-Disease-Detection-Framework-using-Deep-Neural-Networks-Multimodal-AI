# Multi-Disease Detection Framework (DeepDx)
# Production Dockerfile driven directly by pyproject.toml via uv

ARG PYTHON_VERSION=3.11-slim
FROM python:${PYTHON_VERSION}

# Copy the fast standalone uv binary from its official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    HOST=0.0.0.0 \
    PORT=8000

# Install essential system dependencies:
# - libgomp1: required for TensorFlow CPU OpenMP multi-threading on Debian
# - curl: required for container healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Step 1: Copy pyproject.toml and install dependencies directly using uv (layer cached)
COPY pyproject.toml .
RUN uv pip install --system --no-cache -r pyproject.toml

# Step 2: Copy application code, configuration, and frontend static assets
COPY app/ ./app/
COPY frontend/ ./frontend/
COPY class_names.json .

# Step 3: Copy model weights (packaged directly for self-contained execution)
COPY models/ ./models/

# Expose FastAPI application port
EXPOSE 8000

# Container healthcheck using existing FastAPI /api/health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Start Uvicorn serving both REST API and static frontend
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
