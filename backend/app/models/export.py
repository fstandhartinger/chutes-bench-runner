"""Export entity for CSV/PDF reports."""
from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.session import Base


class Export(Base):
    """Export metadata and content."""

    __tablename__ = "exports"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    run_id: Mapped[str] = mapped_column(UUID(as_uuid=False), ForeignKey("benchmark_runs.id"), nullable=False)
    format: Mapped[str] = mapped_column(String(10), nullable=False)  # csv or pdf
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    content_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # file path or S3 URL
    content_size: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    generated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    run: Mapped["BenchmarkRun"] = relationship("BenchmarkRun", lazy="joined")

    def __repr__(self) -> str:
        return f"<Export(run_id={self.run_id}, format={self.format})>"


from app.models.run import BenchmarkRun  # noqa: E402, F401
























