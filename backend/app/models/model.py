"""Model entity representing Chutes models."""
from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import DateTime, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class Model(Base):
    """Chutes model entity."""

    __tablename__ = "models"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    slug: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    provider: Mapped[str] = mapped_column(String(32), default="chutes", index=True)
    tagline: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    user: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    logo: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    chute_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    instance_count: Mapped[int] = mapped_column(default=0)
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<Model(slug={self.slug}, name={self.name})>"





























