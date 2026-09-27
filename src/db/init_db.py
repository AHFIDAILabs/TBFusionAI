from sqlalchemy import text

from src.db.engine import get_engine
from src.db.models import Base
from src.logger import get_logger

logger = get_logger(__name__)


async def init_db() -> None:
    """Create all tables if they don't exist. Safe to call on every startup."""
    try:
        engine = get_engine()
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            await conn.execute(
                text(
                    "ALTER TABLE participants ADD COLUMN IF NOT EXISTS "
                    "patient_id VARCHAR(20) UNIQUE"
                )
            )
            await conn.execute(text("""
                    WITH numbered AS (
                        SELECT id, ROW_NUMBER() OVER (ORDER BY created_at ASC) AS rn
                        FROM participants
                        WHERE patient_id IS NULL
                    )
                    UPDATE participants
                    SET patient_id = 'TB-2026-' || LPAD(numbered.rn::text, 4, '0')
                    FROM numbered
                    WHERE participants.id = numbered.id
                    """))
            await conn.execute(
                text("CREATE SEQUENCE IF NOT EXISTS patient_id_seq START WITH 45")
            )
        logger.info("Database tables ready")
    except Exception as e:
        logger.error(
            f"Database init failed — participant saving will be unavailable: {e}"
        )
