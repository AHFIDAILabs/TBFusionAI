import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, Integer, LargeBinary, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Participant(Base):
    __tablename__ = "participants"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # Audio
    audio_filename: Mapped[str] = mapped_column(String, nullable=False)
    audio_data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    # Clinical features
    age: Mapped[int] = mapped_column(Integer, nullable=False)
    sex: Mapped[str] = mapped_column(String(10), nullable=False)
    cough_duration: Mapped[int] = mapped_column(Integer, nullable=False)
    prior_tb_history: Mapped[bool] = mapped_column(Boolean, nullable=False)
    hemoptysis: Mapped[bool] = mapped_column(Boolean, nullable=False)
    weight_loss: Mapped[bool] = mapped_column(Boolean, nullable=False)
    fever: Mapped[bool] = mapped_column(Boolean, nullable=False)
    night_sweats: Mapped[bool] = mapped_column(Boolean, nullable=False)

    # Prediction snapshot
    prediction: Mapped[str] = mapped_column(String, nullable=False)
    prediction_class: Mapped[int] = mapped_column(Integer, nullable=False)
    probability: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_level: Mapped[str] = mapped_column(String, nullable=False)
    recommendation: Mapped[str] = mapped_column(String, nullable=False)
