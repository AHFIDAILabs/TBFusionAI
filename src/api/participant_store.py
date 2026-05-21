"""PostgreSQL-backed participant store.

Audio bytes are persisted as BYTEA in the database — required for Cloud Run
where the container filesystem is ephemeral and cannot be relied on for storage.
"""

import uuid
from typing import Dict, List, Optional, Tuple

from sqlalchemy import func, select
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

    async def list(
        self,
        limit: int = 50,
        offset: int = 0,
        prediction: Optional[str] = None,
    ) -> Tuple[List[Participant], int]:
        stmt = select(Participant).order_by(Participant.created_at.desc())
        count_stmt = select(func.count()).select_from(Participant)

        if prediction:
            stmt = stmt.where(Participant.prediction == prediction)
            count_stmt = count_stmt.where(Participant.prediction == prediction)

        total = await self._db.scalar(count_stmt)
        result = await self._db.execute(stmt.limit(limit).offset(offset))
        return list(result.scalars().all()), total or 0

    async def get_by_id(self, participant_id: uuid.UUID) -> Optional[Participant]:
        result = await self._db.execute(
            select(Participant).where(Participant.id == participant_id)
        )
        return result.scalar_one_or_none()
