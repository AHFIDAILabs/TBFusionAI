# ============================================================================
# Stage 1: Builder — install deps and pre-download Wav2Vec2
# ============================================================================
FROM python:3.10-slim-bookworm AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only torch from PyTorch's official index, then the rest of requirements.
# Splitting the install avoids pip trying to resolve GPU torch variants for other packages.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        "torch==2.0.1+cpu" \
        "torchaudio==2.0.2+cpu" \
        --extra-index-url https://download.pytorch.org/whl/cpu

RUN grep -vE "^torch(audio)?==" requirements.txt > /tmp/req-no-torch.txt && \
    pip install --no-cache-dir -r /tmp/req-no-torch.txt

# Pre-download the exact model class used at inference time (Wav2Vec2Model, not ForCTC)
RUN python -c "\
from transformers import Wav2Vec2Processor, Wav2Vec2Model; \
Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h'); \
Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h')"

# ============================================================================
# Stage 2: Runtime — lean production image
# ============================================================================
FROM python:3.10-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    HF_HUB_CACHE=/app/.cache/huggingface/hub

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy pre-downloaded HuggingFace model cache from builder
COPY --from=builder /root/.cache/huggingface /app/.cache/huggingface

RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/artifacts/trained_models \
             /app/artifacts/logs \
             /app/artifacts/reports && \
    chown -R appuser:appuser /app

WORKDIR /app

# Copy trained model artifacts (baked into image for production;
# overridden by volume mount in docker-compose for local dev)
COPY --chown=appuser:appuser artifacts/ /app/artifacts/

COPY --chown=appuser:appuser . .

RUN chown -R appuser:appuser /app

USER appuser

# Default port for local/docker-compose. Cloud Run overrides with $PORT env var.
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/api/v1/health || exit 1

CMD sh -c "python main.py serve 0.0.0.0 ${PORT:-8000}"
