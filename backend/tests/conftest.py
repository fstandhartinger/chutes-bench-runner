"""Test fixtures and configuration."""
import asyncio
import os
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest

# Set environment variables before importing app modules
os.environ["DATABASE_URL"] = "postgresql://test:test@localhost/test"
os.environ["CHUTES_API_KEY"] = "test-key"
os.environ["SKIP_MODEL_SYNC"] = "true"

from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.api.deps import get_session
from app.db.session import Base
from app.main import app


# Use in-memory SQLite for tests
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Create event loop for tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
async def test_engine():
    """Create test database engine."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.fixture(scope="function")
async def test_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    session_maker = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with session_maker() as session:
        yield session


@pytest.fixture(scope="function")
def client(test_session) -> Generator[TestClient, None, None]:
    """Create test client with overridden database."""

    async def override_get_db():
        yield test_session

    app.dependency_overrides[get_session] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def mock_chutes_client():
    """Create mock Chutes client."""
    mock = MagicMock()
    mock.list_models = AsyncMock(
        return_value=[
            {
                "slug": "test-model",
                "name": "Test Model",
                "tagline": "A test model",
                "user": "testuser",
                "logo": None,
                "chute_id": "test-id",
                "instance_count": 1,
            }
        ]
    )
    mock.run_inference = AsyncMock(
        return_value={
            "choices": [{"message": {"content": "A"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1},
            "model": "test-model",
        }
    )
    mock.get_completion_text = AsyncMock(
        return_value=("A", {"usage": {"prompt_tokens": 10, "completion_tokens": 1}})
    )
    return mock
