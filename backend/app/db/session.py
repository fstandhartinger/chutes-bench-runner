"""Database session management."""
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings

settings = get_settings()

engine = create_async_engine(
    settings.async_database_url,
    echo=False,
    pool_pre_ping=True,
    pool_size=settings.effective_db_pool_size,
    max_overflow=settings.effective_db_max_overflow,
    pool_timeout=settings.db_pool_timeout_seconds,
    pool_recycle=settings.db_pool_recycle_seconds,
    # Without these, an asyncpg connect/query can hang indefinitely when the peer
    # disappears (container network rebuild, Postgres restart/recovery). That is
    # the entry point of the 2026-07-21 worker deadlock: the hung connect was
    # cancelled by a wait_for, and the cancellation could not land because
    # SQLAlchemy tears the connection down inside an asyncio.shield().
    # Bounding the driver means the hang usually never starts.
    connect_args={
        "timeout": settings.db_connect_timeout_seconds,
        "command_timeout": settings.db_command_timeout_seconds,
    },
)

async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""
    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session dependency."""
    async with async_session_maker() as session:
        try:
            yield session
        finally:
            await session.close()





























