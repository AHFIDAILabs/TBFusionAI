# ============================================================================
# Stage 1: Builder (Compiles requirements)
# ============================================================================
FROM python:3.10-slim-bookworm AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies to a local user directory
RUN pip install --upgrade pip && \
    pip install --user --no-cache-dir -r requirements.txt

# ============================================================================
# Stage 2: Runtime (The final production image)
# ============================================================================
FROM python:3.10-slim-bookworm AS runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/appuser/.local/bin:${PATH}" \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HUB_CACHE=/app/.cache/huggingface/hub

# Install runtime-only system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create application user and directory structure
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/artifacts/trained_models \
             /app/artifacts/logs \
             /app/artifacts/reports \
             /app/.cache/huggingface && \
    chown -R appuser:appuser /app

WORKDIR /app

# CRITICAL: Copy Python packages FIRST (before trying to use them)
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# NOW download Wav2Vec2 model (transformers is available now)
RUN python -c "from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC; \
    Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h'); \
    Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')" && \
    chown -R appuser:appuser /app/.cache

# Copy model artifacts (Fail fast if missing)
COPY --chown=appuser:appuser artifacts/ /app/artifacts/

# Copy the rest of the application code
COPY --chown=appuser:appuser . .

# Final permission check
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Cloud Run uses dynamic ports, but usually defaults to 8080
EXPOSE 8080

# Healthcheck for Cloud Run
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/api/v1/health || exit 1

# Start the application
CMD sh -c "python main.py serve 0.0.0.0 ${PORT:-8080}"



# # ============================================================================
# # Stage 1: Builder (Compiles requirements)
# # ============================================================================
# FROM python:3.10-slim-bookworm AS builder

# WORKDIR /app

# # Install build dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     curl \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt .

# # Install dependencies to a local user directory
# RUN pip install --upgrade pip && \
#     pip install --user --no-cache-dir -r requirements.txt

# # ============================================================================
# # Stage 2: Runtime (The final production image)
# # ============================================================================
# FROM python:3.10-slim-bookworm AS runtime

# # Set environment variables
# ENV PYTHONUNBUFFERED=1 \
#     PYTHONDONTWRITEBYTECODE=1 \
#     PATH="/home/appuser/.local/bin:${PATH}"

# # Install runtime-only system dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libsndfile1 \
#     ffmpeg \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

# # 1. Create application user and directory structure
# RUN useradd -m -u 1000 appuser && \
#     mkdir -p /app/artifacts/trained_models /app/artifacts/logs /app/artifacts/reports \
#     chown -R appuser:appuser /app && \
#     python -c from transformers import Wav2Vec2Processor; Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')

# WORKDIR /app

# # 2. Copy the installed python packages from the builder stage
# COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# # 3. Explicitly copy model artifacts (Fail fast if missing)
# # This confirms /app/artifacts/trained_models/cost_sensitive_ensemble_model.joblib is included
# COPY --chown=appuser:appuser artifacts/ /app/artifacts/

# # 4. Copy the rest of the application code
# COPY --chown=appuser:appuser . .

# # Final permission check for the appuser
# RUN chown -R appuser:appuser /app/artifacts

# USER appuser

# # Cloud Run uses dynamic ports, but usually defaults to 8080
# EXPOSE 8080

# # Healthcheck adjusted for the dynamic $PORT environment variable
# HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
#     CMD curl -f http://localhost:${PORT:-8080}/api/v1/health || exit 1

# # Add before CMD
# ENV HF_HOME=/app/.cache/huggingface
# ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# # Start the application using the shell form to expand the $PORT variable
# CMD sh -c "python main.py serve 0.0.0.0 ${PORT:-8080}"


# # ============================================================================
# # Stage 1: Builder (Compiles requirements)
# # ============================================================================
# FROM python:3.10-slim-bookworm AS builder

# WORKDIR /app

# # Install only what's needed to compile wheels
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential \
#     curl \
#     git \
#     && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt .

# # Install dependencies into a specific folder to copy later
# RUN pip install --upgrade pip && \
#     pip install --user --no-cache-dir -r requirements.txt


# # ============================================================================
# # Stage 2: Runtime (The final production image)
# # ============================================================================
# FROM python:3.10-slim-bookworm AS runtime

# # Set environment variables
# ENV PYTHONUNBUFFERED=1 \
#     PYTHONDONTWRITEBYTECODE=1 \
#     PATH="/home/appuser/.local/bin:${PATH}"

# # Install ONLY the runtime libraries (no build tools here)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libsndfile1 \
#     ffmpeg \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

# # Create application user
# RUN useradd -m -u 1000 appuser && \
#     mkdir -p /app/artifacts/logs && \
#     chown -R appuser:appuser /app

# WORKDIR /app

# # Copy the installed python packages from the builder stage
# COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local
# # Copy application code
# COPY --chown=appuser:appuser . .

# # Final directory setup
# RUN mkdir -p artifacts/dataset artifacts/trained_models artifacts/reports artifacts/logs && \
#     chown -R appuser:appuser /app/artifacts

# USER appuser

# EXPOSE 8000

# HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
#     CMD curl -f http://localhost:8000/api/v1/health || exit 1

# # Change this:
# # CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# # To this (it tells uvicorn to use $PORT, or default to 8000 if not set):
# # CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

# # Do NOT do this: CMD ["uvicorn", "src.api.main:app", ...]
# # DO THIS:
# # CMD ["sh", "-c", "python main.py serve --host 0.0.0.0 --port ${PORT:-8000}"]
# # WRONG (Exec form - variables won't expand)
# # CMD ["python", "main.py", "serve", "--host", "0.0.0.0", "--port", "${PORT}"]

# # RIGHT (Shell form - variables expand correctly)
# # CMD sh -c "python main.py serve --host 0.0.0.0 --port ${PORT:-8080}"
# # Simple and direct: No flags, just the values
# CMD sh -c "python main.py serve 0.0.0.0 ${PORT:-8080}"