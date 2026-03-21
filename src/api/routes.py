"""
API routes for TBFusionAI - Updated with graceful degradation.

Defines endpoints for:
- Health checks (works without models)
- Predictions (requires models)
- Model information
- Setup status and instructions
"""

import io
import time
from datetime import datetime
from typing import Dict, List

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse

from src.api.dependencies import (
    get_app_config,
    get_predictor,
    get_predictor_optional,
    validate_audio_file,
)
from src.api.schemas import (
    AudioMetrics,
    BatchPredictionRequest,
    BatchPredictionResponse,
    ClinicalFeatures,
    ErrorResponse,
    FeatureImportanceResponse,
    HealthCheckResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from src.config import Config
from src.logger import get_logger
from src.models.predictor import TBPredictor

logger = get_logger(__name__)

# Create API router
router = APIRouter()


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="Health Check",
    description="Check API health and model status",
)
async def health_check(config: Config = Depends(get_app_config)) -> HealthCheckResponse:
    """
    Check API health status - works even without models.

    Returns:
        HealthCheckResponse: Health status information
    """
    # Try to get predictor without raising exceptions
    predictor = get_predictor_optional()
    model_loaded = predictor is not None

    return HealthCheckResponse(
        status="healthy" if model_loaded else "ready_for_setup",
        version=config.api.app_version,
        model_loaded=model_loaded,
        timestamp=datetime.now().isoformat(),
    )


@router.get(
    "/status",
    summary="Detailed Status",
    description="Get detailed system status and setup instructions",
)
async def get_status(config: Config = Depends(get_app_config)) -> Dict:
    """
    Get detailed status including setup instructions.

    Returns:
        Dictionary with status and instructions
    """
    predictor = get_predictor_optional()

    # Check which pipeline stages are complete
    dataset_exists = (config.paths.dataset_path / "raw_data").exists()
    processed_exists = (
        config.paths.preprocessed_path / "longitudinal_wav2vec2_embeddings.csv"
    ).exists()
    models_exist = (
        config.paths.models_path / "cost_sensitive_ensemble_model.joblib"
    ).exists()

    status_info = {
        "api_version": config.api.app_version,
        "model_loaded": predictor is not None,
        "pipeline_status": {
            "data_ingested": dataset_exists,
            "data_processed": processed_exists,
            "models_trained": models_exist,
        },
        "timestamp": datetime.now().isoformat(),
    }

    # Add setup instructions if models not loaded
    if not models_exist:
        status_info["setup_required"] = True
        status_info["instructions"] = {
            "message": "Models not found. Run the ML pipeline to train models.",
            "commands": {
                "docker": "docker exec tbfusionai-api python main.py run-pipeline",
                "local": "python main.py run-pipeline",
            },
            "steps": [
                "1. Data Ingestion - Downloads CODA TB dataset (~15 min)",
                "2. Data Processing - Extracts audio features (~60 min)",
                "3. Model Training - Trains multiple models (~30 min)",
                "4. Model Evaluation - Creates ensemble (~5 min)",
            ],
            "estimated_time": "~2 hours total for complete pipeline",
        }
    else:
        status_info["setup_required"] = False
        status_info["message"] = "System ready for predictions"

    return status_info


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict TB from Audio",
    description="Make TB prediction from audio file and clinical features",
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def predict_from_audio(
    audio_file: UploadFile = File(..., description="Audio file (WAV, MP3, OGG)"),
    age: int = Form(..., ge=0, le=120, description="Patient age"),
    sex: str = Form(..., description="Patient sex (Male/Female)"),
    reported_cough_dur: int = Form(..., ge=0, description="Cough duration in days"),
    tb_prior: str = Form(..., description="Prior TB (Yes/No)"),
    hemoptysis: str = Form(..., description="Hemoptysis (Yes/No)"),
    weight_loss: str = Form(..., description="Weight loss (Yes/No)"),
    fever: str = Form(..., description="Fever (Yes/No)"),
    night_sweats: str = Form(..., description="Night sweats (Yes/No)"),
    generate_spectrogram: bool = Form(True, description="Generate spectrogram"),
    validate_quality: bool = Form(False, description="Validate audio quality"),
    predictor: TBPredictor = Depends(get_predictor),
    validated_file: UploadFile = Depends(validate_audio_file),
) -> PredictionResponse:
    """
    Make TB prediction from audio file.
    """
    try:
        logger.info(f"Received prediction request for file: {audio_file.filename}")

        # Read audio file
        audio_bytes = await audio_file.read()
        audio_io = io.BytesIO(audio_bytes)

        # Prepare clinical features
        clinical_features = {
            "age": age,
            "sex": sex,
            "reported_cough_dur": reported_cough_dur,
            "tb_prior": tb_prior,
            "hemoptysis": hemoptysis,
            "weight_loss": weight_loss,
            "fever": fever,
            "night_sweats": night_sweats,
        }

        # Make prediction
        result = await predictor.predict_from_audio(
            audio_io, clinical_features, generate_spectrogram, validate_quality
        )

        logger.info(f"✓ Prediction completed: {result['prediction']}")

        return PredictionResponse(**result)

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@router.post(
    "/predict/features",
    response_model=PredictionResponse,
    summary="Predict TB from Features",
    description="Make TB prediction from pre-extracted features",
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def predict_from_features(
    features: Dict[str, float], predictor: TBPredictor = Depends(get_predictor)
) -> PredictionResponse:
    """
    Make TB prediction from features.
    """
    try:
        logger.info("Received prediction request from features")

        # Make prediction
        result = await predictor.predict_from_features(features)

        logger.info(f"✓ Prediction completed: {result['prediction']}")

        return PredictionResponse(**result)

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Get Model Information",
    description="Get information about the loaded model",
    responses={503: {"model": ErrorResponse}},
)
async def get_model_info(
    predictor: TBPredictor = Depends(get_predictor),
) -> ModelInfoResponse:
    """
    Get model information.
    """
    try:
        info = predictor.get_model_info()
        return ModelInfoResponse(**info)
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}",
        )


@router.get(
    "/model/feature-importance",
    response_model=FeatureImportanceResponse,
    summary="Get Feature Importance",
    description="Get feature importance rankings from ensemble model",
    responses={503: {"model": ErrorResponse}},
)
async def get_feature_importance(
    top_n: int = 10, predictor: TBPredictor = Depends(get_predictor)
) -> FeatureImportanceResponse:
    """
    Get feature importance.
    """
    try:
        importance_dict = predictor.ensemble_model.get_feature_importance(
            predictor.metadata["feature_columns"]
        )

        # Convert to list format
        features = [
            {"feature": name, "importance": float(importance)}
            for name, importance in list(importance_dict.items())[:top_n]
        ]

        return FeatureImportanceResponse(features=features, top_n=top_n)
    except Exception as e:
        logger.error(f"Failed to get feature importance: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get feature importance: {str(e)}",
        )


@router.post(
    "/audio/metrics",
    response_model=AudioMetrics,
    summary="Get Audio Metrics",
    description="Calculate audio quality metrics",
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def get_audio_metrics(
    audio_file: UploadFile = File(..., description="Audio file"),
    predictor: TBPredictor = Depends(get_predictor),
    validated_file: UploadFile = Depends(validate_audio_file),
) -> AudioMetrics:
    """
    Calculate audio metrics.
    """
    try:
        logger.info(f"Calculating metrics for: {audio_file.filename}")

        # Read audio file
        audio_bytes = await audio_file.read()
        audio_io = io.BytesIO(audio_bytes)

        # Calculate metrics
        metrics = predictor.audio_preprocessor.calculate_audio_metrics(audio_io)

        # FIXED: Ensure all keys expected by AudioMetrics schema and tests are present
        return AudioMetrics(
            duration=metrics.get("duration", 0.0),
            rms=metrics.get("rms", 0.0),
            zcr=metrics.get("zcr", 0.0),
            flatness=metrics.get("flatness", 0.0),
            snr=metrics.get("snr", 0.0),
            clipping_ratio=metrics.get("clipping_ratio", 0.0),
            silence_ratio=metrics.get("silence_ratio", 0.0),
        )

    except Exception as e:
        logger.error(f"Failed to calculate metrics: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate metrics: {str(e)}",
        )


# """
# API routes for TBFusionAI - Updated with graceful degradation.

# Defines endpoints for:
# - Health checks (works without models)
# - Predictions (requires models)
# - Model information
# - Setup status and instructions
# """

# import io
# import time
# from datetime import datetime
# from typing import Dict, List

# from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
# from fastapi.responses import HTMLResponse, JSONResponse

# from src.api.dependencies import (
#     get_app_config,
#     get_predictor,
#     get_predictor_optional,
#     validate_audio_file,
# )
# from src.api.schemas import (
#     AudioMetrics,
#     BatchPredictionRequest,
#     BatchPredictionResponse,
#     ClinicalFeatures,
#     ErrorResponse,
#     FeatureImportanceResponse,
#     HealthCheckResponse,
#     ModelInfoResponse,
#     PredictionResponse,
# )
# from src.config import Config
# from src.logger import get_logger
# from src.models.predictor import TBPredictor

# logger = get_logger(__name__)

# # Create API router
# router = APIRouter()


# @router.get(
#     "/health",
#     response_model=HealthCheckResponse,
#     summary="Health Check",
#     description="Check API health and model status",
# )
# async def health_check(config: Config = Depends(get_app_config)) -> HealthCheckResponse:
#     """
#     Check API health status - works even without models.

#     Returns:
#         HealthCheckResponse: Health status information
#     """
#     # Try to get predictor without raising exceptions
#     predictor = get_predictor_optional()
#     model_loaded = predictor is not None

#     return HealthCheckResponse(
#         status="healthy" if model_loaded else "ready_for_setup",
#         version=config.api.app_version,
#         model_loaded=model_loaded,
#         timestamp=datetime.now().isoformat(),
#     )


# @router.get(
#     "/status",
#     summary="Detailed Status",
#     description="Get detailed system status and setup instructions",
# )
# async def get_status(config: Config = Depends(get_app_config)) -> Dict:
#     """
#     Get detailed status including setup instructions.

#     Returns:
#         Dictionary with status and instructions
#     """
#     predictor = get_predictor_optional()

#     # Check which pipeline stages are complete
#     dataset_exists = (config.paths.dataset_path / "raw_data").exists()
#     processed_exists = (
#         config.paths.preprocessed_path / "longitudinal_wav2vec2_embeddings.csv"
#     ).exists()
#     models_exist = (
#         config.paths.models_path / "cost_sensitive_ensemble_model.joblib"
#     ).exists()

#     status_info = {
#         "api_version": config.api.app_version,
#         "model_loaded": predictor is not None,
#         "pipeline_status": {
#             "data_ingested": dataset_exists,
#             "data_processed": processed_exists,
#             "models_trained": models_exist,
#         },
#         "timestamp": datetime.now().isoformat(),
#     }

#     # Add setup instructions if models not loaded
#     if not models_exist:
#         status_info["setup_required"] = True
#         status_info["instructions"] = {
#             "message": "Models not found. Run the ML pipeline to train models.",
#             "commands": {
#                 "docker": "docker exec tbfusionai-api python main.py run-pipeline",
#                 "local": "python main.py run-pipeline",
#             },
#             "steps": [
#                 "1. Data Ingestion - Downloads CODA TB dataset (~15 min)",
#                 "2. Data Processing - Extracts audio features (~60 min)",
#                 "3. Model Training - Trains multiple models (~30 min)",
#                 "4. Model Evaluation - Creates ensemble (~5 min)",
#             ],
#             "estimated_time": "~2 hours total for complete pipeline",
#         }
#     else:
#         status_info["setup_required"] = False
#         status_info["message"] = "System ready for predictions"

#     return status_info


# @router.post(
#     "/predict",
#     response_model=PredictionResponse,
#     summary="Predict TB from Audio",
#     description="Make TB prediction from audio file and clinical features",
#     responses={
#         400: {"model": ErrorResponse},
#         413: {"model": ErrorResponse},
#         500: {"model": ErrorResponse},
#         503: {"model": ErrorResponse},
#     },
# )
# async def predict_from_audio(
#     audio_file: UploadFile = File(..., description="Audio file (WAV, MP3, OGG)"),
#     age: int = Form(..., ge=0, le=120, description="Patient age"),
#     sex: str = Form(..., description="Patient sex (Male/Female)"),
#     reported_cough_dur: int = Form(..., ge=0, description="Cough duration in days"),
#     tb_prior: str = Form(..., description="Prior TB (Yes/No)"),
#     hemoptysis: str = Form(..., description="Hemoptysis (Yes/No)"),
#     weight_loss: str = Form(..., description="Weight loss (Yes/No)"),
#     fever: str = Form(..., description="Fever (Yes/No)"),
#     night_sweats: str = Form(..., description="Night sweats (Yes/No)"),
#     generate_spectrogram: bool = Form(True, description="Generate spectrogram"),
#     validate_quality: bool = Form(False, description="Validate audio quality"),
#     predictor: TBPredictor = Depends(get_predictor),
#     validated_file: UploadFile = Depends(validate_audio_file),
# ) -> PredictionResponse:
#     """
#     Make TB prediction from audio file.

#     Args:
#         audio_file: Audio file
#         age: Patient age
#         sex: Patient sex
#         reported_cough_dur: Cough duration
#         tb_prior: Prior TB history
#         hemoptysis: Hemoptysis presence
#         weight_loss: Weight loss
#         fever: Fever presence
#         night_sweats: Night sweats presence
#         generate_spectrogram: Whether to generate spectrogram
#         predictor: TBPredictor instance
#         validated_file: Validated audio file

#     Returns:
#         PredictionResponse: Prediction result
#     """
#     try:
#         logger.info(f"Received prediction request for file: {audio_file.filename}")

#         # Read audio file
#         audio_bytes = await audio_file.read()
#         audio_io = io.BytesIO(audio_bytes)

#         # Prepare clinical features
#         clinical_features = {
#             "age": age,
#             "sex": sex,
#             "reported_cough_dur": reported_cough_dur,
#             "tb_prior": tb_prior,
#             "hemoptysis": hemoptysis,
#             "weight_loss": weight_loss,
#             "fever": fever,
#             "night_sweats": night_sweats,
#         }

#         # Make prediction
#         result = await predictor.predict_from_audio(
#             audio_io, clinical_features, generate_spectrogram, validate_quality
#         )

#         logger.info(f"✓ Prediction completed: {result['prediction']}")

#         return PredictionResponse(**result)

#     except ValueError as e:
#         logger.error(f"Validation error: {str(e)}")
#         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
#     except Exception as e:
#         logger.error(f"Prediction failed: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Prediction failed: {str(e)}",
#         )


# @router.post(
#     "/predict/features",
#     response_model=PredictionResponse,
#     summary="Predict TB from Features",
#     description="Make TB prediction from pre-extracted features",
#     responses={
#         400: {"model": ErrorResponse},
#         500: {"model": ErrorResponse},
#         503: {"model": ErrorResponse},
#     },
# )
# async def predict_from_features(
#     features: Dict[str, float], predictor: TBPredictor = Depends(get_predictor)
# ) -> PredictionResponse:
#     """
#     Make TB prediction from features.

#     Args:
#         features: Dictionary of all features (clinical + audio)
#         predictor: TBPredictor instance

#     Returns:
#         PredictionResponse: Prediction result
#     """
#     try:
#         logger.info("Received prediction request from features")

#         # Make prediction
#         result = await predictor.predict_from_features(features)

#         logger.info(f"✓ Prediction completed: {result['prediction']}")

#         return PredictionResponse(**result)

#     except ValueError as e:
#         logger.error(f"Validation error: {str(e)}")
#         raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
#     except Exception as e:
#         logger.error(f"Prediction failed: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Prediction failed: {str(e)}",
#         )


# @router.get(
#     "/model/info",
#     response_model=ModelInfoResponse,
#     summary="Get Model Information",
#     description="Get information about the loaded model",
#     responses={503: {"model": ErrorResponse}},
# )
# async def get_model_info(
#     predictor: TBPredictor = Depends(get_predictor),
# ) -> ModelInfoResponse:
#     """
#     Get model information.

#     Args:
#         predictor: TBPredictor instance

#     Returns:
#         ModelInfoResponse: Model information
#     """
#     try:
#         info = predictor.get_model_info()
#         return ModelInfoResponse(**info)
#     except Exception as e:
#         logger.error(f"Failed to get model info: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to get model info: {str(e)}",
#         )


# @router.get(
#     "/model/feature-importance",
#     response_model=FeatureImportanceResponse,
#     summary="Get Feature Importance",
#     description="Get feature importance rankings from ensemble model",
#     responses={503: {"model": ErrorResponse}},
# )
# async def get_feature_importance(
#     top_n: int = 10, predictor: TBPredictor = Depends(get_predictor)
# ) -> FeatureImportanceResponse:
#     """
#     Get feature importance.

#     Args:
#         top_n: Number of top features to return
#         predictor: TBPredictor instance

#     Returns:
#         FeatureImportanceResponse: Feature importance data
#     """
#     try:
#         importance_dict = predictor.ensemble_model.get_feature_importance(
#             predictor.metadata["feature_columns"]
#         )

#         # Convert to list format
#         features = [
#             {"feature": name, "importance": float(importance)}
#             for name, importance in list(importance_dict.items())[:top_n]
#         ]

#         return FeatureImportanceResponse(features=features, top_n=top_n)
#     except Exception as e:
#         logger.error(f"Failed to get feature importance: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to get feature importance: {str(e)}",
#         )


# @router.post(
#     "/audio/metrics",
#     response_model=AudioMetrics,
#     summary="Get Audio Metrics",
#     description="Calculate audio quality metrics",
#     responses={
#         400: {"model": ErrorResponse},
#         413: {"model": ErrorResponse},
#         500: {"model": ErrorResponse},
#         503: {"model": ErrorResponse},
#     },
# )
# async def get_audio_metrics(
#     audio_file: UploadFile = File(..., description="Audio file"),
#     predictor: TBPredictor = Depends(get_predictor),
#     validated_file: UploadFile = Depends(validate_audio_file),
# ) -> AudioMetrics:
#     """
#     Calculate audio metrics.

#     Args:
#         audio_file: Audio file
#         predictor: TBPredictor instance
#         validated_file: Validated audio file

#     Returns:
#         AudioMetrics: Audio quality metrics
#     """
#     try:
#         logger.info(f"Calculating metrics for: {audio_file.filename}")

#         # Read audio file
#         audio_bytes = await audio_file.read()
#         audio_io = io.BytesIO(audio_bytes)

#         # Calculate metrics
#         metrics = predictor.audio_preprocessor.calculate_audio_metrics(audio_io)

#         return AudioMetrics(**metrics)

#     except Exception as e:
#         logger.error(f"Failed to calculate metrics: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to calculate metrics: {str(e)}",
#         )
