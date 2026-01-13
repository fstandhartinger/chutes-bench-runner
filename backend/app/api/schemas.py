"""API request/response schemas."""
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


def _serialize_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat().replace("+00:00", "Z")


class APIModel(BaseModel):
    model_config = ConfigDict(
        from_attributes=True,
        json_encoders={datetime: _serialize_datetime},
    )


# Model schemas
class ModelResponse(APIModel):
    """Model response schema."""
    id: str
    slug: str
    name: str
    provider: str = "chutes"
    tagline: Optional[str] = None
    user: Optional[str] = None
    logo: Optional[str] = None
    chute_id: Optional[str] = None
    instance_count: int = 0
    is_active: bool = True


class ModelsListResponse(BaseModel):
    """List of models response."""
    models: list[ModelResponse]
    total: int


# Benchmark schemas
class BenchmarkInfo(APIModel):
    """Benchmark info schema."""
    name: str
    display_name: str
    description: Optional[str] = None
    category: Optional[str] = None
    is_enabled: bool = True
    supports_subset: bool = True
    requires_setup: bool = False
    setup_notes: Optional[str] = None
    default_selected: Optional[bool] = None
    total_items: int = 0
    avg_item_latency_ms: Optional[float] = None


class BenchmarksListResponse(BaseModel):
    """List of benchmarks response."""
    benchmarks: list[BenchmarkInfo]


# Run schemas
class CreateRunRequest(BaseModel):
    """Create run request schema."""
    model_id: str = Field(
        ...,
        description="Bench runner model UUID, Chutes chute_id, or model slug (e.g. zai-org/GLM-4.7-TEE).",
    )
    subset_pct: int = Field(
        default=100,
        ge=1,
        le=100,
        description=(
            "Percentage of items to sample (1-100). Common values: 1, 5, 10, 25, 50, 100. "
            "Ignored when subset_count is provided."
        ),
    )
    subset_count: Optional[int] = Field(
        default=None,
        ge=1,
        description=(
            "Fixed number of items to sample. Takes precedence over subset_pct when set. "
            "Useful for Affine environments where 1% can still be large."
        ),
    )
    subset_seed: Optional[str] = Field(
        default=None,
        max_length=128,
        description=(
            "Optional seed used to deterministically select items. "
            "Use the same seed across runs to align samples between models."
        ),
    )
    selected_benchmarks: Optional[list[str]] = Field(
        default=None,
        description="Optional list of benchmark names; use /api/benchmarks to discover valid values.",
    )
    config: Optional[dict[str, Any]] = Field(
        default=None,
        description="Optional per-run configuration overrides.",
    )
    provider: Optional[str] = Field(
        default=None,
        description="Inference provider (chutes, gremium-openai, gremium-anthropic).",
    )


class BenchmarkRunBenchmarkResponse(APIModel):
    """Benchmark within a run response."""
    id: str
    benchmark_name: str
    status: str
    total_items: int
    completed_items: int
    sampled_items: int
    score: Optional[float] = None
    metrics: Optional[dict[str, Any]] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class RunResponse(APIModel):
    """Run response schema."""
    id: str
    model_id: str
    model_slug: str
    provider: str
    subset_pct: int
    subset_count: Optional[int] = None
    subset_seed: Optional[str] = None
    status: str
    selected_benchmarks: Optional[list[str]] = None
    overall_score: Optional[float] = None
    error_message: Optional[str] = None
    provider_metadata: Optional[dict[str, Any]] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime
    benchmarks: list[BenchmarkRunBenchmarkResponse] = []


class RunsListResponse(BaseModel):
    """List of runs response."""
    runs: list[RunResponse]
    total: int


class ItemResultResponse(APIModel):
    """Item result response schema."""
    id: str
    item_id: str
    item_hash: Optional[str] = None
    prompt: Optional[str] = None
    response: Optional[str] = None
    expected: Optional[str] = None
    is_correct: Optional[bool] = None
    score: Optional[float] = None
    judge_output: Optional[dict[str, Any]] = None
    latency_ms: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    error: Optional[str] = None
    test_code: Optional[str] = None
    item_metadata: Optional[dict[str, Any]] = None
    created_at: datetime


class ItemResultsResponse(BaseModel):
    """Paginated item results response."""
    items: list[ItemResultResponse]
    total: int
    limit: int
    offset: int


class ErrorBreakdownEntry(BaseModel):
    """Error message breakdown entry."""
    message: str
    count: int


class BenchmarkSummaryResponse(BaseModel):
    """Aggregated benchmark stats across all items in a run."""
    total_items: int
    correct: int
    incorrect: int
    errors: int
    avg_latency_ms: Optional[float] = None
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost_usd: Optional[float] = None
    output_cost_usd: Optional[float] = None
    total_cost_usd: Optional[float] = None
    pricing_input_per_million_usd: Optional[float] = None
    pricing_output_per_million_usd: Optional[float] = None
    error_breakdown: list[ErrorBreakdownEntry] = []


class RunBenchmarkDetailsResponse(BaseModel):
    """Benchmark details with item results and summary stats."""
    benchmark: BenchmarkRunBenchmarkResponse
    items: ItemResultsResponse
    summary: BenchmarkSummaryResponse


class RunEventResponse(APIModel):
    """Run event response schema."""
    id: str
    event_type: str
    benchmark_name: Optional[str] = None
    message: Optional[str] = None
    data: Optional[dict[str, Any]] = None
    created_at: datetime


class CancelRunResponse(BaseModel):
    """Cancel run response."""
    success: bool
    message: str


class SyncModelsResponse(BaseModel):
    """Sync models response."""
    synced: int
    message: str


class SignedExportVerifyResponse(BaseModel):
    """Signed export verification response."""
    valid: bool
    signature_valid: bool
    hash_match: bool
    errors: list[str]
    run_id: Optional[str] = None
    model_slug: Optional[str] = None
    subset_pct: Optional[int] = None
    subset_count: Optional[int] = None
    subset_seed: Optional[str] = None
    overall_score: Optional[float] = None
    exported_at: Optional[str] = None
    benchmark_count: Optional[int] = None
    public_key_fingerprint: Optional[str] = None


class PublicKeyResponse(BaseModel):
    """Public key response."""
    algorithm: str
    public_key: str
    public_key_fingerprint: str


class MaintenanceStatusResponse(BaseModel):
    """Maintenance status response."""
    maintenance_mode: bool
    message: str


class WorkerHeartbeatInfo(APIModel):
    """Worker heartbeat info."""
    worker_id: str
    hostname: Optional[str] = None
    running_runs: int
    max_concurrent_runs: int
    item_concurrency: int
    last_seen: datetime


class WorkerTimeseriesPoint(APIModel):
    """Aggregated worker counts for charting."""
    timestamp: datetime
    worker_count: int
    running_runs: int
    queued_runs: int = 0


class TokenUsageWindow(APIModel):
    """Token usage summary for a time window."""
    window_hours: int
    input_tokens: int
    output_tokens: int


class TokenUsageStats(APIModel):
    """Token usage summary windows."""
    last_24h: TokenUsageWindow
    last_7d: TokenUsageWindow


class SandyMetricsPoint(APIModel):
    """Sandy system metrics time series point."""
    timestamp: datetime
    cpu_ratio: Optional[float] = None
    memory_ratio: Optional[float] = None
    disk_ratio: Optional[float] = None


class SandyResourcesResponse(BaseModel):
    """Current Sandy resource usage snapshot."""
    canCreateSandbox: bool
    rejectReason: Optional[str] = None
    limits: dict[str, float | int | None]
    priorityBreakdown: dict[str, int]
    cpu_percent: Optional[float] = None
    memory_percent: Optional[float] = None
    memory_total_gb: Optional[float] = None
    memory_available_gb: Optional[float] = None
    disk_used_ratio: Optional[float] = None
    cpuCount: Optional[int] = None
    load1: Optional[float] = None
    load5: Optional[float] = None
    load15: Optional[float] = None
    memoryTotalBytes: Optional[int] = None
    memoryUsedBytes: Optional[int] = None
    memoryAvailableBytes: Optional[int] = None
    diskTotalBytes: Optional[int] = None
    diskUsedBytes: Optional[int] = None
    diskFreeBytes: Optional[int] = None
    diskUsedRatio: Optional[float] = None


class SandySandboxStatsResponse(APIModel):
    """Per-sandbox resource stats."""
    sandbox_id: str
    container_id: Optional[str] = None
    cpu_ratio: Optional[float] = None
    cpu_cores_used: Optional[float] = None
    cpu_cores_total: Optional[int] = None
    memory_usage_bytes: Optional[int] = None
    memory_limit_bytes: Optional[int] = None
    memory_ratio: Optional[float] = None
    disk_bytes: Optional[int] = None
    updated_at: Optional[datetime] = None


class RunSummaryResponse(APIModel):
    """Lightweight run summary."""
    id: str
    model_slug: str
    provider: str
    subset_pct: int
    subset_count: Optional[int] = None
    subset_seed: Optional[str] = None
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    benchmarks: Optional[list[str]] = None


class OpsOverviewResponse(BaseModel):
    """Ops overview response."""
    workers: list[WorkerHeartbeatInfo]
    timeseries: list[WorkerTimeseriesPoint]
    queue_counts: dict[str, int]
    queued_runs: list[RunSummaryResponse]
    running_runs: list[RunSummaryResponse]
    completed_runs: list[RunSummaryResponse]
    worker_config: dict[str, int]
    token_stats: Optional[TokenUsageStats] = None
