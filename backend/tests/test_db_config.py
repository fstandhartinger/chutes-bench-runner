"""Tests for database pool configuration defaults."""

from app.core.config import get_settings


def test_db_pool_defaults_track_worker_concurrency(monkeypatch):
    monkeypatch.delenv("DB_POOL_SIZE", raising=False)
    monkeypatch.delenv("DB_MAX_OVERFLOW", raising=False)
    monkeypatch.setenv("WORKER_MAX_CONCURRENT", "3")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.effective_db_pool_size == 5
    assert settings.effective_db_max_overflow == 3


def test_db_pool_overrides_take_precedence(monkeypatch):
    monkeypatch.setenv("DB_POOL_SIZE", "9")
    monkeypatch.setenv("DB_MAX_OVERFLOW", "4")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.effective_db_pool_size == 9
    assert settings.effective_db_max_overflow == 4
