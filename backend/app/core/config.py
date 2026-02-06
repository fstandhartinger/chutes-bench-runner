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
    chutes_image_token_estimate: int = 1024
    chutes_inference_timeout_seconds: int = 300
    chutes_rate_limit_sleep_seconds: int = 30

    # Janus Gateway
    janus_gateway_base_url: str = "https://janus-gateway-bqou.onrender.com/v1"
    janus_gateway_api_key: Optional[str] = None

    # Artificial Analysis (public site scraping + optional LLM mapping)
    artificial_analysis_base_url: str = "https://artificialanalysis.ai"
    artificial_analysis_sitemap_url: str = "https://artificialanalysis.ai/sitemap.xml"
    artificial_analysis_timeout_seconds: int = 10
    artificial_analysis_cache_ttl_seconds: int = 3600
    artificial_analysis_llm_fallback: bool = True
    artificial_analysis_mapper_model: str = "deepseek-ai/DeepSeek-V3.2-TEE"

    # Gremium API
    enable_gremium_provider: bool = False
    gremium_api_base_url: str = "https://chutes-model-gremium.onrender.com"
    gremium_provider_default: str = "gremium-openai"
    gremium_api_key: Optional[str] = None
    gremium_timeout_seconds: int = 600
    gremium_item_timeout_seconds: int = 1800
    gremium_item_attempts: int = 3

    # RLM API (Recursive Language Model gateway)
    enable_rlm_provider: bool = False
    rlm_api_base_url: str = "https://chutes-rlm.onrender.com"
    rlm_api_key: Optional[str] = None
    rlm_timeout_seconds: int = 600

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
    worker_item_timeout_seconds: int = 1200
    worker_item_attempts: int = 5
    worker_stale_run_minutes: int = 5
    worker_stale_check_interval: int = 60
    worker_heartbeat_seconds: int = 60
    worker_exclusive_benchmarks: list[str] = []
    worker_disabled: bool = False
    worker_only_auth_mode: Optional[str] = None
    worker_only_api_key: Optional[str] = None

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

    # K2 Vendor Verifier (K2VV)
    k2vv_dataset_url: str = "https://statics.moonshot.cn/k2vv/tool-calls.tar.gz"
    k2vv_reference_model: str = "kimi-k2-0905-preview"
    k2vv_reference_results_path: Optional[str] = None
    k2vv_request_overrides_json: Optional[str] = None
    k2vv_timeout_seconds: int = 600

    # Sandy Sandbox
    sandy_base_url: str = "https://sandy.example.com"
    sandy_api_key: Optional[str] = None
    sandy_volume_root: str = "/var/lib/sandy/volumes"
    sandy_docker_upstream: Optional[str] = None

    # AA-LCR judge model (LLM-based equality checker)
    aa_lcr_judge_model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507-TEE"
    # HLE judge model (LLM-based judge)
    hle_judge_model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507-TEE"
    # AA-Omniscience judge model
    aa_omniscience_judge_model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507-TEE"
    # GDPval judge model (LLM-based evaluator against reference docs)
    gdpval_judge_model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507-TEE"
    # CritPt external evaluation API
    critpt_eval_url: str = "https://artificialanalysis.ai/api/v2/critpt/evaluate"
    critpt_api_key: Optional[str] = None
    # GDPval reference context limit (characters)
    gdpval_reference_char_limit: int = 60000

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
