"""
Pydantic schemas for API request/response validation.

Defines data models for:
- Clinical features
- Prediction requests
- Prediction responses
- Error responses
"""

from datetime import datetime
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


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
    
    @validator('sex')
    def validate_sex(cls, v):
        """Validate sex field."""
        if v not in ['Male', 'Female', 'male', 'female']:
            raise ValueError('Sex must be either Male or Female')
        return v.capitalize()
    
    @validator('tb_prior', 'hemoptysis', 'weight_loss', 'fever', 'night_sweats')
    def validate_binary_fields(cls, v):
        """Validate binary yes/no fields."""
        if v not in ['Yes', 'No', 'yes', 'no']:
            raise ValueError('Field must be either Yes or No')
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
                "night_sweats": "No"
            }
        }
    }

class PredictionResponse(BaseModel):
    """Response model for TB prediction."""
    
    prediction: str = Field(..., description="Prediction result (Probable TB / Unlikely TB)")
    prediction_class: int = Field(..., description="Prediction class (0 or 1)")
    probability: float = Field(..., ge=0.0, le=1.0, description="Prediction probability")
    confidence: float = Field(..., ge=0.0, description="Confidence score")
    confidence_level: str = Field(..., description="Confidence level (High/Medium/Uncertain)")
    recommendation: str = Field(..., description="Clinical recommendation")
    disclaimer: str = Field(..., description="Disclaimer text")
    date: str = Field(..., description="Prediction timestamp")
    model_info: Dict[str, Union[str, float]] = Field(..., description="Model information")
    spectrogram_base64: Optional[str] = Field(None, description="Base64 encoded spectrogram")

    model_config = {
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
                    "threshold": 0.45
                }
            }
        }
    }

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    
    predictions: List[Dict[str, Union[int, float, str]]] = Field(
        ..., 
        description="List of feature dictionaries for batch prediction"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "predictions": [
                    {
                        "age": 45,
                        "sex": "Male",
                        "reported_cough_dur": 21,
                        "tb_prior": "No",
                        "hemoptysis": "No",
                        "weight_loss": "Yes",
                        "fever": "Yes",
                        "night_sweats": "No",
                        **{f"feat_{i}": 0.0 for i in range(768)}
                    }
                ]
            }
        }
    }

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_count: int = Field(..., description="Total number of predictions")
    tb_positive_count: int = Field(..., description="Number of TB positive predictions")
    tb_negative_count: int = Field(..., description="Number of TB negative predictions")
    processing_time: float = Field(..., description="Processing time in seconds")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "predictions": [],
                "total_count": 10,
                "tb_positive_count": 3,
                "tb_negative_count": 7,
                "processing_time": 2.5
            }
        }
    }

class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    
    model_version: str = Field(..., description="Model version")
    ensemble_strategy: str = Field(..., description="Ensemble strategy")
    base_models: List[str] = Field(..., description="List of base models")
    optimal_threshold: float = Field(..., description="Optimal prediction threshold")
    feature_count: int = Field(..., description="Total number of features")
    clinical_features: List[str] = Field(..., description="List of clinical features")
    performance: Dict[str, Union[int, float]] = Field(..., description="Model performance metrics")

    model_config = {
        "json_schema_extra": {
            "example": {
                "model_version": "1.0.0",
                "ensemble_strategy": "Soft Voting + Cost-Sensitive Threshold",
                "base_models": ["CatBoost", "XGBoost", "LightGBM"],
                "optimal_threshold": 0.45,
                "feature_count": 786,
                "clinical_features": ["age", "sex", "reported_cough_dur"],
                "performance": {
                    "recall": 0.95,
                    "f1": 0.92
                }
            }
        }
    }

class HealthCheckResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether models are loaded")
    timestamp: str = Field(..., description="Current timestamp")

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "model_loaded": True,
                "timestamp": "2024-01-15T10:30:00"
            }
        }
    }

class ErrorResponse(BaseModel):
    """Response model for errors."""
    
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")

    model_config = {
        "json_schema_extra": {
            "example": {
                "error": "ValidationError",
                "message": "Invalid input data",
                "detail": "Age must be between 0 and 120",
                "timestamp": "2024-01-15T10:30:00"
            }
        }
    }

class AudioMetrics(BaseModel):
    """Response model for audio quality metrics."""
    
    duration: float = Field(..., description="Audio duration in seconds")
    rms: float = Field(..., description="Root Mean Square energy")
    zcr: float = Field(..., description="Zero Crossing Rate")
    flatness: float = Field(..., description="Spectral flatness")
    snr: float = Field(..., description="Signal-to-Noise Ratio in dB")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "duration": 3.5,
                "rms": 0.045,
                "zcr": 0.12,
                "flatness": 0.35,
                "snr": 15.2
            }
        }
    }

class FeatureImportanceResponse(BaseModel):
    """Response model for feature importance."""
    
    features: List[Dict[str, Union[str, float]]] = Field(
        ..., 
        description="List of features with importance scores"
    )
    top_n: int = Field(..., description="Number of top features returned")
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "features": [
                    {"feature": "feat_245", "importance": 0.045},
                    {"feature": "age", "importance": 0.032},
                    {"feature": "reported_cough_dur", "importance": 0.028}
                ],
                "top_n": 5
            }
        }
    }