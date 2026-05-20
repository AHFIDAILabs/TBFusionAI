"""
FastAPI dependencies for TBFusionAI.

Provides:
- Predictor instance management (singleton, retry-capable)
- Configuration management (cached)
- Request validation for audio files
- Service-level error handling
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, UploadFile, status

from src.config import Config, get_config
from src.logger import get_logger
from src.models.predictor import TBPredictor

logger = get_logger(__name__)

_predictor_instance: Optional[TBPredictor] = None
_predictor_load_attempted: bool = False
_predictor_load_error: Optional[str] = None


def get_predictor() -> TBPredictor:
    """Get or create TBPredictor instance. Raises 503 if unavailable."""
    predictor = get_predictor_optional()
    if predictor:
        return predictor

    error_detail = _predictor_load_error or "Models not loaded."
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=f"Prediction service unavailable: {error_detail}",
    )


def get_predictor_optional() -> Optional[TBPredictor]:
    """
    Attempt to initialise the predictor. Retries on every call until success
    so a transient failure at startup does not require a server restart.
    """
    global _predictor_instance, _predictor_load_attempted, _predictor_load_error

    if _predictor_instance is not None:
        return _predictor_instance

    _predictor_load_attempted = True
    try:
        base_dir = Path(os.getenv("APP_HOME", Path(__file__).parent.parent.parent))
        logger.info(f"Initializing TBPredictor (base dir: {base_dir.absolute()})")
        _predictor_instance = TBPredictor()
        _predictor_load_error = None
        logger.info("TBPredictor initialized successfully")
        return _predictor_instance
    except Exception as e:
        _predictor_load_error = str(e)
        logger.error(f"TBPredictor failed to load: {str(e)}", exc_info=True)
        logger.info("Tip: ensure .joblib files are in artifacts/trained_models/")
        return None


@lru_cache()
def get_app_config() -> Config:
    """Returns the cached application configuration."""
    return get_config()


async def validate_audio_file(audio_file: UploadFile) -> UploadFile:
    """Validates the uploaded audio file for format and size."""
    config = get_app_config()

    if not audio_file or not audio_file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Valid audio file is required.",
        )

    file_ext = f".{audio_file.filename.split('.')[-1].lower()}"
    allowed_formats = list(config.api.allowed_audio_formats)

    if file_ext not in allowed_formats:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format {file_ext}. Use: {', '.join(allowed_formats)}",
        )

    audio_file.file.seek(0, 2)
    file_size = audio_file.file.tell()
    audio_file.file.seek(0)

    if file_size > config.api.max_upload_size:
        max_mb = config.api.max_upload_size / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {max_mb}MB limit.",
        )

    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )

    logger.info(
        f"File validated: {audio_file.filename} ({file_size} bytes, {file_ext})"
    )
    return audio_file


class RateLimiter:
    """In-memory rate limiter. Wire into routes via Depends(prediction_rate_limiter)."""

    def __init__(self, calls: int = 10, period: int = 60):
        self.calls = calls
        self.period = period
        self.requests: dict = {}

    async def __call__(self, request_id: str) -> None:
        import time

        current_time = time.time()
        self.requests = {
            k: v
            for k, v in self.requests.items()
            if current_time - v["time"] < self.period
        }
        if request_id in self.requests:
            if self.requests[request_id]["count"] >= self.calls:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded. Please try again later.",
                )
            self.requests[request_id]["count"] += 1
        else:
            self.requests[request_id] = {"time": current_time, "count": 1}


prediction_rate_limiter = RateLimiter(calls=15, period=60)
