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
        logger.info("Database tables ready")
    except Exception as e:
        logger.error(
            f"Database init failed — participant saving will be unavailable: {e}"
        )
