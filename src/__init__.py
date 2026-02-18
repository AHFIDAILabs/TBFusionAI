"""
TBFusionAI: AI-powered TB detection using cough sound analysis.

This package provides end-to-end ML pipelines for:
- Data ingestion and preprocessing
- Audio feature extraction
- Model training and evaluation
- Ensemble strategies for FN/FP reduction
- Real-time inference via API
"""

__version__ = "1.0.0"
__author__ = "AHFID AI Labs"
__email__ = "aiteam@ahfid.org"

from src.config import get_config
from src.logger import get_logger

__all__ = ["get_config", "get_logger"]