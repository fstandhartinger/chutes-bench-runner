"""Model comparison entities for RLM vs Base model benchmark comparisons."""
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base

JSON_TYPE = JSONB().with_variant(JSON, "sqlite")


class ComparisonStatus(str, Enum):
    """Status of a model comparison."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ModelComparison(Base):
    """A comparison between a base model and an RLM variant."""

    __tablename__ = "model_comparisons"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))

    # Base model run
    base_run_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("benchmark_runs.id"), nullable=False)
    base_model_slug: Mapped[str] = mapped_column(String(255), nullable=False)

    # RLM model run
    rlm_run_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("benchmark_runs.id"), nullable=False)
    rlm_model_slug: Mapped[str] = mapped_column(String(255), nullable=False)

    # Comparison metadata
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    subset_seed: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    benchmarks: Mapped[Optional[list]] = mapped_column(JSON_TYPE, nullable=True)

    # Status
    status: Mapped[str] = mapped_column(String(50), default=ComparisonStatus.PENDING.value)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Results (populated when both runs complete)
    results: Mapped[Optional[dict]] = mapped_column(JSON_TYPE, nullable=True)
    markdown_report: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Relationships
    base_run: Mapped["BenchmarkRun"] = relationship("BenchmarkRun", foreign_keys=[base_run_id], lazy="joined")
    rlm_run: Mapped["BenchmarkRun"] = relationship("BenchmarkRun", foreign_keys=[rlm_run_id], lazy="joined")

    def __repr__(self) -> str:
        return f"<ModelComparison(id={self.id}, base={self.base_model_slug}, rlm={self.rlm_model_slug})>"


# Import BenchmarkRun to resolve forward reference
from app.models.run import BenchmarkRun  # noqa: E402, F401
