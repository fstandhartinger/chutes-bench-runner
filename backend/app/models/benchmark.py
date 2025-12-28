"""Benchmark and BenchmarkItem entities."""
from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class Benchmark(Base):
    """Benchmark definition entity."""

    __tablename__ = "benchmarks"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    version: Mapped[str] = mapped_column(String(50), default="1.0.0")
    adapter_class: Mapped[str] = mapped_column(String(255), nullable=False)
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    supports_subset: Mapped[bool] = mapped_column(Boolean, default=True)
    requires_setup: Mapped[bool] = mapped_column(Boolean, default=False)
    setup_notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    config: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    total_items: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    items: Mapped[list["BenchmarkItem"]] = relationship("BenchmarkItem", back_populates="benchmark")

    def __repr__(self) -> str:
        return f"<Benchmark(name={self.name})>"


class BenchmarkItem(Base):
    """Individual benchmark item/task (optional catalog)."""

    __tablename__ = "benchmark_items"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    benchmark_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("benchmarks.id"), nullable=False)
    item_id: Mapped[str] = mapped_column(String(255), nullable=False)
    item_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    data: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    benchmark: Mapped["Benchmark"] = relationship("Benchmark", back_populates="items")

    def __repr__(self) -> str:
        return f"<BenchmarkItem(benchmark_id={self.benchmark_id}, item_id={self.item_id})>"


