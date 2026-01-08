"""Worker heartbeat models for ops visibility."""
from datetime import datetime
from uuid import uuid4

from sqlalchemy import DateTime, Integer, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class WorkerHeartbeat(Base):
    """Latest heartbeat per worker instance."""

    __tablename__ = "worker_heartbeats"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    worker_id: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    hostname: Mapped[str] = mapped_column(String(255), nullable=True)
    running_runs: Mapped[int] = mapped_column(Integer, default=0)
    max_concurrent_runs: Mapped[int] = mapped_column(Integer, default=0)
    item_concurrency: Mapped[int] = mapped_column(Integer, default=0)
    last_seen: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class WorkerHeartbeatLog(Base):
    """Time-series heartbeat samples for visualization."""

    __tablename__ = "worker_heartbeat_logs"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    worker_id: Mapped[str] = mapped_column(String(128), nullable=False)
    hostname: Mapped[str] = mapped_column(String(255), nullable=True)
    running_runs: Mapped[int] = mapped_column(Integer, default=0)
    max_concurrent_runs: Mapped[int] = mapped_column(Integer, default=0)
    item_concurrency: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
