"""
API routes for TBFusionAI.

Defines endpoints for:
- Health checks (works without models)
- Predictions (requires models)
- Model information
- Audio quality metrics
"""

import io
import uuid
from datetime import datetime
from typing import Dict, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies import (
    get_app_config,
    get_predictor,
    get_predictor_optional,
    validate_audio_file,
)
from src.api.participant_schemas import (
    ParticipantErrorResponse,
    ParticipantListItem,
    ParticipantListResponse,
)
from src.api.participant_store import ParticipantStore
from src.api.schemas import (
    AudioMetrics,
    ClinicalFeatures,
    ErrorResponse,
    FeatureImportanceResponse,
    HealthCheckResponse,
    ModelInfoResponse,
    PredictionResponse,
)
from src.config import Config
from src.db.engine import get_db
from src.logger import get_logger
from src.models.predictor import TBPredictor

logger = get_logger(__name__)

router = APIRouter()


@router.get(
    "/health",
    summary="Health Check",
    description="Check API health and model status",
)
async def health_check(config: Config = Depends(get_app_config)) -> HealthCheckResponse:
    """Check API health status — works even without models."""
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
    """Get detailed status including setup instructions."""
    predictor = get_predictor_optional()

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

    if not models_exist:
        status_info["setup_required"] = True
        status_info["instructions"] = {
            "message": "Models not found. Run the ML pipeline to train models.",
            "commands": {
                "docker": "docker exec tbfusionai-api python main.py run-pipeline",
                "local": "python main.py run-pipeline",
            },
        }
    else:
        status_info["setup_required"] = False
        status_info["message"] = "System ready for predictions"

    return status_info


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict TB from Audio",
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def predict_from_audio(
    audio_file: UploadFile = File(..., description="Audio file (WAV, MP3, OGG, WebM)"),
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
    """Make TB prediction from audio file and clinical features."""
    try:
        logger.info(f"Processing prediction for: {audio_file.filename}")

        audio_bytes = await audio_file.read()

        # Normalise string fields to Title case so encode_features() matches correctly
        clinical_features = {
            "age": age,
            "sex": sex.strip().capitalize(),
            "reported_cough_dur": reported_cough_dur,
            "tb_prior": tb_prior.strip().capitalize(),
            "hemoptysis": hemoptysis.strip().capitalize(),
            "weight_loss": weight_loss.strip().capitalize(),
            "fever": fever.strip().capitalize(),
            "night_sweats": night_sweats.strip().capitalize(),
        }

        result = await predictor.predict_from_audio(
            audio_bytes,
            clinical_features,
            generate_spectrogram,
            validate_quality,
        )

        return PredictionResponse(**result)

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction Error: {str(e)}",
        )


@router.post("/predict/features", response_model=PredictionResponse)
async def predict_from_features(
    features: Dict[str, int | float | str],
    predictor: TBPredictor = Depends(get_predictor),
) -> PredictionResponse:
    """Make prediction from pre-extracted features."""
    try:
        result = await predictor.predict_from_features(features)
        return PredictionResponse(**result)
    except Exception as e:
        logger.error(f"Feature prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info(
    predictor: TBPredictor = Depends(get_predictor),
) -> ModelInfoResponse:
    """Get model information."""
    return ModelInfoResponse(**predictor.get_model_info())


@router.get("/model/feature-importance", response_model=FeatureImportanceResponse)
async def get_feature_importance(
    top_n: int = 5, predictor: TBPredictor = Depends(get_predictor)
) -> FeatureImportanceResponse:
    """Get feature importance rankings from the model."""
    try:
        importance = predictor.get_feature_importance(top_n=top_n)
        return FeatureImportanceResponse(features=importance, top_n=top_n)
    except Exception as e:
        logger.error(f"Failed to get feature importance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/audio/metrics", summary="Get Audio Metrics")
async def get_audio_metrics(
    audio_file: UploadFile = File(..., description="Audio file"),
    predictor: TBPredictor = Depends(get_predictor),
    validated_file: UploadFile = Depends(validate_audio_file),
) -> AudioMetrics:
    """Calculate audio quality metrics."""
    try:
        audio_bytes = await audio_file.read()
        metrics = predictor.audio_preprocessor.calculate_audio_metrics(audio_bytes)

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
        logger.error(f"Audio metrics failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/participants",
    summary="Save Participant",
    description="Run TB prediction and save participant record with embedded result",
    responses={
        400: {"model": ErrorResponse},
        422: {"model": ParticipantErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def save_participant(
    audio_file: UploadFile = File(..., description="Audio file (WAV, MP3, OGG, WebM)"),
    age: str = Form(..., description="Patient age in years"),
    sex: str = Form(..., description="Patient sex (Male/Female)"),
    reported_cough_dur: str = Form(..., description="Cough duration in days"),
    tb_prior: str = Form(default="No", description="Prior TB history (Yes/No)"),
    hemoptysis: str = Form(default="No", description="Hemoptysis (Yes/No)"),
    weight_loss: str = Form(default="No", description="Weight loss (Yes/No)"),
    fever: str = Form(default="No", description="Fever (Yes/No)"),
    night_sweats: str = Form(default="No", description="Night sweats (Yes/No)"),
    generate_spectrogram: bool = Form(True),
    validate_quality: bool = Form(False),
    predictor: TBPredictor = Depends(get_predictor),
    validated_file: UploadFile = Depends(validate_audio_file),
    db: AsyncSession = Depends(get_db),
) -> JSONResponse:
    """Run prediction and save participant record with embedded prediction result."""
    field_errors: Dict[str, str] = {}

    age_int = None
    try:
        age_f = float(age)
        if age_f != int(age_f):
            field_errors["age"] = "must be a whole number"
        elif int(age_f) <= 0:
            field_errors["age"] = "must be greater than 0"
        else:
            age_int = int(age_f)
    except (ValueError, TypeError):
        field_errors["age"] = "must be a positive integer"

    dur_int = None
    try:
        dur_f = float(reported_cough_dur)
        if dur_f != int(dur_f):
            field_errors["coughDuration"] = "must be a whole number"
        elif int(dur_f) < 0:
            field_errors["coughDuration"] = "must be 0 or greater"
        else:
            dur_int = int(dur_f)
    except (ValueError, TypeError):
        field_errors["coughDuration"] = "must be a non-negative integer"

    if field_errors:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=ParticipantErrorResponse(
                fields=field_errors,
                timestamp=datetime.now().isoformat(),
            ).model_dump(),
        )

    def to_bool(v: str) -> bool:
        return v.strip().lower() in ("yes", "true", "1")

    try:
        audio_bytes = await audio_file.read()

        clinical_features = {
            "age": age_int,
            "sex": sex.strip().capitalize(),
            "reported_cough_dur": dur_int,
            "tb_prior": tb_prior.strip().capitalize(),
            "hemoptysis": hemoptysis.strip().capitalize(),
            "weight_loss": weight_loss.strip().capitalize(),
            "fever": fever.strip().capitalize(),
            "night_sweats": night_sweats.strip().capitalize(),
        }

        prediction_result = await predictor.predict_from_audio(
            audio_bytes,
            clinical_features,
            generate_spectrogram,
            validate_quality,
        )

    except ValueError as e:
        logger.error(f"Validation error in save_participant: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction failed in save_participant: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}",
        )

    # DB save is best-effort — a missing/unconfigured DATABASE_URL must not block
    # the prediction result from reaching the user.
    record = None
    try:
        store = ParticipantStore(db)
        record = await store.save(
            audio_bytes=audio_bytes,
            audio_filename=audio_file.filename or "recording.webm",
            age=age_int,
            sex=sex.strip().capitalize(),
            cough_duration=dur_int,
            prior_tb_history=to_bool(tb_prior),
            hemoptysis=to_bool(hemoptysis),
            weight_loss=to_bool(weight_loss),
            fever=to_bool(fever),
            night_sweats=to_bool(night_sweats),
            prediction_result=prediction_result,
        )
    except Exception as e:
        logger.warning(f"Participant DB save skipped (DB unavailable?): {e}")

    return JSONResponse(
        content={
            "participant": record,
            "prediction": PredictionResponse(**prediction_result).model_dump(),
        }
    )


def _participant_to_item(p) -> ParticipantListItem:
    return ParticipantListItem(
        participantId=str(p.id),
        timestamp=p.created_at.isoformat(),
        audioFilename=p.audio_filename,
        age=p.age,
        sex=p.sex,
        coughDuration=p.cough_duration,
        priorTBHistory=p.prior_tb_history,
        hemoptysis=p.hemoptysis,
        weightLoss=p.weight_loss,
        fever=p.fever,
        nightSweats=p.night_sweats,
        prediction={
            "result": p.prediction,
            "predictionClass": p.prediction_class,
            "probability": p.probability,
            "confidenceLevel": p.confidence_level,
            "recommendation": p.recommendation,
        },
    )


@router.get(
    "/participants",
    response_model=ParticipantListResponse,
    summary="List Participants",
    description="Return a paginated list of saved participant records.",
)
async def list_participants(
    limit: int = 50,
    offset: int = 0,
    prediction: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
) -> ParticipantListResponse:
    """List participants with optional filter by prediction result."""
    if limit > 200:
        limit = 200
    store = ParticipantStore(db)
    items, total = await store.list(limit=limit, offset=offset, prediction=prediction)
    return ParticipantListResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=[_participant_to_item(p) for p in items],
    )


@router.get(
    "/participants/{participant_id}",
    response_model=ParticipantListItem,
    summary="Get Participant",
    description="Return a single participant record by ID.",
    responses={404: {"model": ErrorResponse}},
)
async def get_participant(
    participant_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> ParticipantListItem:
    """Fetch a single participant record by UUID."""
    store = ParticipantStore(db)
    participant = await store.get_by_id(participant_id)
    if participant is None:
        raise HTTPException(status_code=404, detail="Participant not found")
    return _participant_to_item(participant)
