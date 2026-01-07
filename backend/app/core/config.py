"""Application configuration."""
from functools import lru_cache
from typing import Optional
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

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
    chutes_models_cache_ttl_seconds: int = 600
    chutes_max_tokens_margin: int = 2048
    chutes_min_output_tokens: int = 16384
    chutes_max_output_tokens_cap: Optional[int] = 65535
    chutes_inference_timeout_seconds: int = 300
    chutes_rate_limit_sleep_seconds: int = 30

    # Backend
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    backend_url: str = "http://localhost:8000"

    # Maintenance
    maintenance_mode: bool = False
    maintenance_message: str = "Currently under maintenance as a new version is being deployed."

    # Admin
    admin_secret: Optional[str] = None

    # Worker
    worker_poll_interval: int = 5
    worker_max_concurrent: int = 3
    worker_item_concurrency: int = 4
    worker_item_timeout_seconds: int = 900
    worker_item_attempts: int = 5
    worker_stale_run_minutes: int = 5
    worker_stale_check_interval: int = 60
    worker_heartbeat_seconds: int = 60
    worker_exclusive_benchmarks: list[str] = []
    worker_disabled: bool = False

    # Startup
    skip_model_sync: bool = False

    # Chutes IDP
    chutes_client_id: Optional[str] = None
    chutes_client_secret: Optional[str] = None
    chutes_idp_url: str = "https://idp.chutes.ai"
    
    # Frontend URL (for redirects and CORS)
    frontend_url: str = "http://localhost:3000"

    # PDF
    pdf_font_path: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

    # Benchmark data cache
    bench_data_dir: str = "/tmp/chutes-bench-data"

    # Sandy Sandbox
    sandy_base_url: str = "https://sandy.94.130.222.43.nip.io"
    sandy_api_key: Optional[str] = None

    # AA-LCR judge model (LLM-based equality checker)
    aa_lcr_judge_model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507-TEE"

    # τ²-Bench (optional override for user simulator model)
    tau2_user_model: Optional[str] = None

    # Signed export keys (base64 or PEM)
    bench_signing_private_key: Optional[str] = None
    bench_signing_public_key: Optional[str] = None

    @property
    def async_database_url(self) -> str:
        """Convert database URL to async version.
        
        Also converts sslmode to ssl for asyncpg compatibility:
        - sslmode=require -> ssl=require
        """
        url = self.database_url
        if url.startswith("postgresql://"):
            url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
        elif url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql+asyncpg://", 1)
        
        # Convert sslmode to ssl for asyncpg compatibility
        url = url.replace("sslmode=", "ssl=")
        # Remove channel_binding (asyncpg does not accept it)
        parts = urlsplit(url)
        if parts.query:
            params = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True) if k != "channel_binding"]
            query = urlencode(params)
            url = urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))
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
