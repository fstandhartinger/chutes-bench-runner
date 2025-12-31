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
    is_enabled: bool = True
    supports_subset: bool = True
    requires_setup: bool = False
    setup_notes: Optional[str] = None
    total_items: int = 0
    avg_item_latency_ms: Optional[float] = None


class BenchmarksListResponse(BaseModel):
    """List of benchmarks response."""
    benchmarks: list[BenchmarkInfo]


# Run schemas
class CreateRunRequest(BaseModel):
    """Create run request schema."""
    model_id: str
    subset_pct: int = Field(default=100, ge=1, le=100)
    selected_benchmarks: Optional[list[str]] = None
    config: Optional[dict[str, Any]] = None


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
    subset_pct: int
    status: str
    selected_benchmarks: Optional[list[str]] = None
    overall_score: Optional[float] = None
    error_message: Optional[str] = None
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
    overall_score: Optional[float] = None
    exported_at: Optional[str] = None
    benchmark_count: Optional[int] = None
    public_key_fingerprint: Optional[str] = None


class PublicKeyResponse(BaseModel):
    """Public key response."""
    algorithm: str
    public_key: str
    public_key_fingerprint: str
