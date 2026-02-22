"""
Logging configuration for TBFusionAI project.
Uses loguru for enhanced logging capabilities.
"""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logger(
    log_file: Optional[Path] = None,
    level: str = "INFO",
    rotation: str = "500 MB",
    retention: str = "10 days",
    compression: str = "zip",
) -> None:
    """
    Configure loguru logger with file and console handlers.

    Args:
        log_file: Path to log file. If None, uses default location.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        rotation: When to rotate log file
        retention: How long to keep log files
        compression: Compression format for old logs
    """
    # Remove default handler
    logger.remove()

    # Console handler with custom format
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>",
        level=level,
    )

    # File handler if log_file specified
    if log_file is None:
        log_file = (
            Path(__file__).parent.parent / "artifacts" / "logs" / "tbfusionai.log"
        )

    # Create log directory if it doesn't exist
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        log_file,
        rotation=rotation,
        retention=retention,
        compression=compression,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level=level,
        enqueue=True,  # Thread-safe logging
    )

    logger.info(f"Logger initialized. Log file: {log_file}")


def get_logger(name: str):
    """
    Get logger instance with specified name.

    Args:
        name: Name for the logger (typically __name__)

    Returns:
        Configured logger instance
    """
    return logger.bind(name=name)


# Initialize logger on module import
setup_logger()
