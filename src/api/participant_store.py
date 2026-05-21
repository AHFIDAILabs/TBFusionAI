"""PostgreSQL-backed participant store.

Audio bytes are persisted as BYTEA in the database — required for Cloud Run
where the container filesystem is ephemeral and cannot be relied on for storage.
"""

from typing import Dict

from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import Participant
from src.logger import get_logger

logger = get_logger(__name__)


class ParticipantStore:
    def __init__(self, db: AsyncSession):
        self._db = db

    async def save(
        self,
        audio_bytes: bytes,
        audio_filename: str,
        age: int,
        sex: str,
        cough_duration: int,
        prior_tb_history: bool,
        hemoptysis: bool,
        weight_loss: bool,
        fever: bool,
        night_sweats: bool,
        prediction_result: Dict,
    ) -> Dict:
        participant = Participant(
            audio_filename=audio_filename,
            audio_data=audio_bytes,
            age=age,
            sex=sex,
            cough_duration=cough_duration,
            prior_tb_history=prior_tb_history,
            hemoptysis=hemoptysis,
            weight_loss=weight_loss,
            fever=fever,
            night_sweats=night_sweats,
            prediction=prediction_result.get("prediction", ""),
            prediction_class=prediction_result.get("prediction_class", 0),
            probability=prediction_result.get("probability", 0.0),
            confidence_level=prediction_result.get("confidence_level", ""),
            recommendation=prediction_result.get("recommendation", ""),
        )

        self._db.add(participant)
        await self._db.commit()
        await self._db.refresh(participant)

        logger.info(f"Participant saved: {participant.id}")

        return {
            "participantId": str(participant.id),
            "timestamp": participant.created_at.isoformat(),
            "age": participant.age,
            "sex": participant.sex,
            "coughDuration": participant.cough_duration,
            "priorTBHistory": participant.prior_tb_history,
            "hemoptysis": participant.hemoptysis,
            "weightLoss": participant.weight_loss,
            "fever": participant.fever,
            "nightSweats": participant.night_sweats,
            "prediction": {
                "result": participant.prediction,
                "predictionClass": participant.prediction_class,
                "probability": participant.probability,
                "confidenceLevel": participant.confidence_level,
                "recommendation": participant.recommendation,
            },
        }
