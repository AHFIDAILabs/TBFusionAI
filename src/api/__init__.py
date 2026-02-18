"""
API module for TBFusionAI.

FastAPI-based REST API for TB prediction service.
Provides endpoints for:
- Health checks
- Audio-based predictions
- Feature-based predictions
- Model information
"""

from src.api.main import app

__all__ = ["app"]