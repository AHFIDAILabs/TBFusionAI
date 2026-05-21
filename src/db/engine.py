from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.config import get_config

_engine = None
_session_factory = None


def _build_engine():
    global _engine, _session_factory
    if _engine is None:
        url = get_config().database.url
        _engine = create_async_engine(url, pool_pre_ping=True, echo=False)
        _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    return _engine, _session_factory


def get_engine():
    engine, _ = _build_engine()
    return engine


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    _, factory = _build_engine()
    async with factory() as session:
        yield session
