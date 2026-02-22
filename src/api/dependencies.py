"""
FastAPI dependencies for dependency injection.

Provides:
- Predictor instance management WITH GRACEFUL DEGRADATION
- Configuration management
- Request validation
- Error handling
"""

from functools import lru_cache
from typing import Optional

from fastapi import Depends, HTTPException, UploadFile, status

from src.config import Config, get_config
from src.logger import get_logger
from src.models.predictor import TBPredictor

logger = get_logger(__name__)


# Global predictor instance
_predictor_instance: Optional[TBPredictor] = None
_predictor_load_attempted: bool = False
_predictor_load_error: Optional[str] = None


def get_predictor() -> TBPredictor:
    """
    Get or create TBPredictor instance.

    Returns:
        TBPredictor: Cached predictor instance

    Raises:
        HTTPException: If predictor initialization fails
    """
    global _predictor_instance, _predictor_load_attempted, _predictor_load_error

    # If already loaded successfully, return it
    if _predictor_instance is not None:
        return _predictor_instance

    # If we already tried and failed, raise the cached error
    if _predictor_load_attempted and _predictor_load_error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=_predictor_load_error,
        )

    # Try to load for the first time
    if not _predictor_load_attempted:
        _predictor_load_attempted = True
        try:
            logger.info("Initializing TBPredictor...")
            _predictor_instance = TBPredictor()
            logger.info("✓ TBPredictor initialized successfully")
            return _predictor_instance

        except Exception as e:
            error_msg = (
                f"Models not ready. Please run training pipelines first: {str(e)}"
            )
            _predictor_load_error = error_msg
            logger.warning(f"⚠ TBPredictor initialization failed: {str(e)}")
            logger.info(
                "💡 To fix: Run 'python main.py run-pipeline' or 'docker exec tbfusionai-api python main.py run-pipeline'"
            )

            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=error_msg
            )

    # Should never reach here, but just in case
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Predictor service unavailable",
    )


def get_predictor_optional() -> Optional[TBPredictor]:
    """
    Get predictor instance without raising exceptions.
    Used for health checks and status endpoints.

    Returns:
        Optional[TBPredictor]: Predictor instance or None if not loaded
    """
    global _predictor_instance, _predictor_load_attempted, _predictor_load_error

    if _predictor_instance is not None:
        return _predictor_instance

    if not _predictor_load_attempted:
        _predictor_load_attempted = True
        try:
            logger.info("Attempting to initialize TBPredictor...")
            _predictor_instance = TBPredictor()
            logger.info("✓ TBPredictor initialized successfully")
            return _predictor_instance
        except Exception as e:
            _predictor_load_error = str(e)
            logger.warning(f"⚠ Models not available: {str(e)}")
            logger.info("💡 Run training pipelines to enable predictions")
            return None

    return None


@lru_cache()
def get_app_config() -> Config:
    """
    Get cached configuration instance.

    Returns:
        Config: Application configuration
    """
    return get_config()


async def validate_audio_file(audio_file: UploadFile) -> UploadFile:
    """
    Validate uploaded audio file.

    Args:
        file: Uploaded audio file

    Returns:
        UploadFile: Validated file

    Raises:
        HTTPException: If validation fails
    """
    config = get_app_config()

    # Check if file is provided
    if not audio_file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="No audio file provided"
        )

    # Check file extension
    file_ext = f".{audio_file.filename.split('.')[-1].lower()}"
    if file_ext not in config.api.allowed_audio_formats:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported audio format. Allowed formats: {config.api.allowed_audio_formats}",
        )

    # Check file size
    audio_file.file.seek(0, 2)  # Seek to end
    file_size = audio_file.file.tell()
    audio_file.file.seek(0)  # Reset to beginning

    if file_size > config.api.max_upload_size:
        max_size_mb = config.api.max_upload_size / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds maximum allowed size of {max_size_mb}MB",
        )

    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty"
        )

    logger.info(f"Audio file validated: {audio_file.filename} ({file_size} bytes)")

    return audio_file


class RateLimiter:
    """
    Simple rate limiter for API endpoints.

    Note: For production, use Redis-based rate limiting.
    """

    def __init__(self, calls: int = 100, period: int = 60):
        """
        Initialize rate limiter.

        Args:
            calls: Number of allowed calls
            period: Time period in seconds
        """
        self.calls = calls
        self.period = period
        self.requests = {}

    async def __call__(self, request_id: str) -> None:
        """
        Check rate limit for request.

        Args:
            request_id: Unique request identifier (e.g., IP address)

        Raises:
            HTTPException: If rate limit exceeded
        """
        import time

        current_time = time.time()

        # Clean old entries
        self.requests = {
            k: v
            for k, v in self.requests.items()
            if current_time - v["time"] < self.period
        }

        # Check rate limit
        if request_id in self.requests:
            if self.requests[request_id]["count"] >= self.calls:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Maximum {self.calls} requests per {self.period} seconds.",
                )
            self.requests[request_id]["count"] += 1
        else:
            self.requests[request_id] = {"time": current_time, "count": 1}


# Create rate limiter instances
prediction_rate_limiter = RateLimiter(calls=10, period=60)
batch_rate_limiter = RateLimiter(calls=5, period=60)
