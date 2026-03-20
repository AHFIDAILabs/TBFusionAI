# ============================================================================
# Stage 1: Builder (Compiles requirements)
# ============================================================================
FROM python:3.10-slim-bookworm AS builder

WORKDIR /app

# Install only what's needed to compile wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies into a specific folder to copy later
RUN pip install --upgrade pip && \
    pip install --user --no-cache-dir -r requirements.txt


# ============================================================================
# Stage 2: Runtime (The final production image)
# ============================================================================
FROM python:3.10-slim-bookworm AS runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/appuser/.local/bin:${PATH}"

# Install ONLY the runtime libraries (no build tools here)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/artifacts/logs && \
    chown -R appuser:appuser /app

WORKDIR /app

# Copy the installed python packages from the builder stage
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local
# Copy application code
COPY --chown=appuser:appuser . .

# Final directory setup
RUN mkdir -p artifacts/dataset artifacts/trained_models artifacts/reports artifacts/logs && \
    chown -R appuser:appuser /app/artifacts

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Change this:
# CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# To this (it tells uvicorn to use $PORT, or default to 8000 if not set):
# CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

# Do NOT do this: CMD ["uvicorn", "src.api.main:app", ...]
# DO THIS:
# CMD ["sh", "-c", "python main.py serve --host 0.0.0.0 --port ${PORT:-8000}"]
# WRONG (Exec form - variables won't expand)
# CMD ["python", "main.py", "serve", "--host", "0.0.0.0", "--port", "${PORT}"]

# RIGHT (Shell form - variables expand correctly)
CMD sh -c "python main.py serve --host 0.0.0.0 --port ${PORT:-8080}"