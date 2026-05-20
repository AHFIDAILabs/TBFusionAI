"""
Pydantic schemas for API request/response validation.

Defines data models for:
- Clinical features
- Prediction requests/responses
- Error responses
"""

from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator


class ClinicalFeatures(BaseModel):
    """Clinical features for TB prediction."""

    age: int = Field(..., ge=0, le=120, description="Patient age in years")
    sex: str = Field(..., description="Patient sex (Male/Female)")
    reported_cough_dur: int = Field(..., ge=0, description="Cough duration in days")
    tb_prior: str = Field(..., description="Prior TB history (Yes/No)")
    hemoptysis: str = Field(..., description="Presence of hemoptysis (Yes/No)")
    weight_loss: str = Field(..., description="Recent weight loss (Yes/No)")
    fever: str = Field(..., description="Presence of fever (Yes/No)")
    night_sweats: str = Field(..., description="Presence of night sweats (Yes/No)")

    @field_validator("sex")
    @classmethod
    def validate_sex(cls, v: str) -> str:
        if v.lower() not in ("male", "female"):
            raise ValueError("Sex must be either Male or Female")
        return v.capitalize()

    @field_validator("tb_prior", "hemoptysis", "weight_loss", "fever", "night_sweats")
    @classmethod
    def validate_binary_fields(cls, v: str) -> str:
        if v.lower() not in ("yes", "no"):
            raise ValueError("Field must be either Yes or No")
        return v.capitalize()

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 45,
                "sex": "Male",
                "reported_cough_dur": 21,
                "tb_prior": "No",
                "hemoptysis": "No",
                "weight_loss": "Yes",
                "fever": "Yes",
                "night_sweats": "No",
            }
        }
    }


class PredictionResponse(BaseModel):
    """Response model for TB prediction."""

    prediction: str = Field(
        ..., description="Prediction result (Probable TB / Unlikely TB)"
    )
    prediction_class: int = Field(..., description="Prediction class (0 or 1)")
    probability: float = Field(
        ..., ge=0.0, le=1.0, description="Prediction probability"
    )
    confidence: float = Field(..., ge=0.0, description="Confidence score")
    confidence_level: str = Field(
        ..., description="Confidence level (High/Medium/Uncertain)"
    )
    recommendation: str = Field(..., description="Clinical recommendation")
    disclaimer: str = Field(..., description="Disclaimer text")
    date: str = Field(..., description="Prediction timestamp")
    model_info: Dict[str, Union[str, float]] = Field(
        ..., description="Model information"
    )
    spectrogram_base64: Optional[str] = Field(
        None, description="Base64 encoded spectrogram"
    )

    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "example": {
                "prediction": "Probable TB",
                "prediction_class": 1,
                "probability": 0.85,
                "confidence": 0.35,
                "confidence_level": "High",
                "recommendation": "High probability of TB detected. Immediate clinical follow-up recommended.",
                "disclaimer": "This is an AI-assisted screening tool and NOT a diagnostic device.",
                "date": "2024-01-15 10:30:00",
                "model_info": {
                    "strategy": "Soft Voting + Cost-Sensitive Threshold",
                    "threshold": 0.45,
                },
            }
        },
    }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""

    predictions: List[Dict[str, Union[int, float, str]]] = Field(
        ..., description="List of feature dictionaries for batch prediction"
    )


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""

    predictions: List[PredictionResponse] = Field(
        ..., description="List of predictions"
    )
    total_count: int = Field(..., description="Total number of predictions")
    tb_positive_count: int = Field(..., description="Number of TB positive predictions")
    tb_negative_count: int = Field(..., description="Number of TB negative predictions")
    processing_time: float = Field(..., description="Processing time in seconds")


class ModelInfoResponse(BaseModel):
    """Response model for model information."""

    model_version: str = Field(..., description="Model version")
    ensemble_strategy: str = Field(..., description="Ensemble strategy")
    base_models: List[str] = Field(..., description="List of base models")
    optimal_threshold: float = Field(..., description="Optimal prediction threshold")
    feature_count: int = Field(..., description="Total number of features")
    clinical_features: List[str] = Field(..., description="List of clinical features")
    performance: Dict[str, Union[int, float]] = Field(
        ..., description="Model performance metrics"
    )

    model_config = {"protected_namespaces": ()}


class HealthCheckResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether models are loaded")
    timestamp: str = Field(..., description="Current timestamp")

    model_config = {"protected_namespaces": ()}


class ErrorResponse(BaseModel):
    """Response model for errors."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")


class AudioMetrics(BaseModel):
    """Response model for audio quality metrics."""

    duration: float = Field(..., description="Audio duration in seconds")
    rms: float = Field(..., description="Root Mean Square energy")
    zcr: float = Field(..., description="Zero Crossing Rate")
    flatness: float = Field(..., description="Spectral flatness")
    snr: float = Field(..., description="Signal-to-Noise Ratio in dB")
    clipping_ratio: float = Field(0.0, description="Ratio of clipped samples")
    silence_ratio: float = Field(0.0, description="Ratio of silent segments")


class FeatureImportanceResponse(BaseModel):
    """Response model for feature importance."""

    features: List[Dict[str, Union[str, float]]] = Field(
        ..., description="List of features with importance scores"
    )
    top_n: int = Field(..., description="Number of top features returned")
