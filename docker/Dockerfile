# Multi-stage Docker build for TBFusionAI

# ============================================================================
# Stage 1: Base Image with System Dependencies
# ============================================================================
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create application user and base directory
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

WORKDIR /app

# ============================================================================
# Stage 2: Dependencies
# ============================================================================
FROM base as dependencies

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --resume-retries 20 --timeout 300 -r requirements.txt

# ============================================================================
# Stage 3: Application
# ============================================================================
FROM dependencies as application

# Copy application code as the appuser
COPY --chown=appuser:appuser . .

# Create necessary directories and set permissions as root
USER root
RUN mkdir -p \
    artifacts/dataset \
    artifacts/trained_models \
    artifacts/preprocessed_data \
    artifacts/labeled_data \
    artifacts/reports \
    artifacts/logs \
    frontend/static/uploads && \
    chown -R appuser:appuser /app/artifacts /app/frontend

# Switch to application user for security
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Default command - Start FastAPI directly
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]