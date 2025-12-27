"""Application configuration."""
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    database_url: str

    # Chutes API
    chutes_api_key: str
    chutes_api_base_url: str = "https://llm.chutes.ai/v1"
    chutes_models_api_url: str = "https://api.chutes.ai"

    # Backend
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    backend_url: str = "http://localhost:8000"

    # Admin
    admin_secret: Optional[str] = None

    # Worker
    worker_poll_interval: int = 5
    worker_max_concurrent: int = 1

    # Chutes IDP (future)
    chutes_client_id: Optional[str] = None
    chutes_client_secret: Optional[str] = None
    chutes_idp_url: str = "https://api.chutes.ai"

    # PDF
    pdf_font_path: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

    @property
    def async_database_url(self) -> str:
        """Convert database URL to async version."""
        url = self.database_url
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://", 1)
        if url.startswith("postgres://"):
            return url.replace("postgres://", "postgresql+asyncpg://", 1)
        return url

    @property
    def sync_database_url(self) -> str:
        """Convert database URL to sync version for Alembic."""
        url = self.database_url
        if "asyncpg" in url:
            return url.replace("postgresql+asyncpg://", "postgresql://", 1)
        return url


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

