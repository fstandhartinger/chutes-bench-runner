"""Benchmark run and result entities."""
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text, JSON
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base

JSON_TYPE = JSONB().with_variant(JSON, "sqlite")


class RunStatus(str, Enum):
    """Status of a benchmark run."""
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


class BenchmarkRunStatus(str, Enum):
    """Status of a benchmark within a run."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    NEEDS_SETUP = "needs_setup"


class BenchmarkRun(Base):
    """A benchmark run."""

    __tablename__ = "benchmark_runs"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    model_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("models.id"), nullable=False)
    model_slug: Mapped[str] = mapped_column(String(255), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), default="chutes")
    subset_pct: Mapped[int] = mapped_column(Integer, default=100)
    subset_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    subset_seed: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(String(50), default=RunStatus.QUEUED.value)
    auth_mode: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    auth_session_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    auth_api_key: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    config: Mapped[Optional[dict]] = mapped_column(JSON_TYPE, nullable=True)
    provider_metadata: Mapped[Optional[dict]] = mapped_column(JSON_TYPE, nullable=True)
    selected_benchmarks: Mapped[Optional[list]] = mapped_column(JSON_TYPE, nullable=True)
    overall_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    canceled_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    code_version: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    git_sha: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    model: Mapped["Model"] = relationship("Model", lazy="joined")
    benchmarks: Mapped[list["BenchmarkRunBenchmark"]] = relationship(
        "BenchmarkRunBenchmark", back_populates="run", lazy="selectin"
    )
    events: Mapped[list["RunEvent"]] = relationship("RunEvent", back_populates="run", lazy="dynamic")

    def __repr__(self) -> str:
        return f"<BenchmarkRun(id={self.id}, model={self.model_slug}, status={self.status})>"


class BenchmarkRunBenchmark(Base):
    """A benchmark within a run."""

    __tablename__ = "benchmark_run_benchmarks"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    run_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("benchmark_runs.id"), nullable=False)
    benchmark_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("benchmarks.id"), nullable=False)
    benchmark_name: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(50), default=BenchmarkRunStatus.PENDING.value)
    total_items: Mapped[int] = mapped_column(Integer, default=0)
    completed_items: Mapped[int] = mapped_column(Integer, default=0)
    sampled_items: Mapped[int] = mapped_column(Integer, default=0)
    sampled_item_ids: Mapped[Optional[list]] = mapped_column(JSON_TYPE, nullable=True)
    metrics: Mapped[Optional[dict]] = mapped_column(JSON_TYPE, nullable=True)
    score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    run: Mapped["BenchmarkRun"] = relationship("BenchmarkRun", back_populates="benchmarks")
    benchmark: Mapped["Benchmark"] = relationship("Benchmark", lazy="joined")
    item_results: Mapped[list["BenchmarkItemResult"]] = relationship(
        "BenchmarkItemResult", back_populates="run_benchmark", lazy="dynamic"
    )

    def __repr__(self) -> str:
        return f"<BenchmarkRunBenchmark(run_id={self.run_id}, benchmark={self.benchmark_name})>"


class BenchmarkItemResult(Base):
    """Result for a single benchmark item."""

    __tablename__ = "benchmark_item_results"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    run_benchmark_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), ForeignKey("benchmark_run_benchmarks.id"), nullable=False
    )
    item_id: Mapped[str] = mapped_column(String(255), nullable=False)
    item_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    response: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    expected: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_correct: Mapped[Optional[bool]] = mapped_column(nullable=True)
    score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    judge_output: Mapped[Optional[dict]] = mapped_column(JSON_TYPE, nullable=True)
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    input_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    test_code: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    item_metadata: Mapped[Optional[dict]] = mapped_column(JSON_TYPE, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    run_benchmark: Mapped["BenchmarkRunBenchmark"] = relationship(
        "BenchmarkRunBenchmark", back_populates="item_results"
    )

    def __repr__(self) -> str:
        return f"<BenchmarkItemResult(item_id={self.item_id}, is_correct={self.is_correct})>"


class RunEvent(Base):
    """Progress and log events for a run."""

    __tablename__ = "run_events"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    run_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("benchmark_runs.id"), nullable=False)
    event_type: Mapped[str] = mapped_column(String(50), nullable=False)
    benchmark_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    data: Mapped[Optional[dict]] = mapped_column(JSON_TYPE, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    run: Mapped["BenchmarkRun"] = relationship("BenchmarkRun", back_populates="events")

    def __repr__(self) -> str:
        return f"<RunEvent(run_id={self.run_id}, type={self.event_type})>"


# Import Model to resolve forward reference
from app.models.model import Model  # noqa: E402, F401
from app.models.benchmark import Benchmark  # noqa: E402, F401
