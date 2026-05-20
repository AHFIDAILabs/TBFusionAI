"""Pydantic schemas for participant records."""

from typing import Dict

from pydantic import BaseModel


class ParticipantPrediction(BaseModel):
    """Prediction snapshot embedded in a participant record."""

    result: str
    predictionClass: int
    probability: float
    confidenceLevel: str
    recommendation: str


class ParticipantRecord(BaseModel):
    """Full participant record as stored and returned."""

    participantId: str
    timestamp: str
    coughSound: str
    age: int
    coughDuration: int
    priorTBHistory: bool
    hemoptysis: bool
    weightLoss: bool
    fever: bool
    nightSweats: bool
    prediction: ParticipantPrediction


class ParticipantErrorResponse(BaseModel):
    """Field-keyed validation error response."""

    error: str = "ValidationError"
    fields: Dict[str, str]
    timestamp: str
