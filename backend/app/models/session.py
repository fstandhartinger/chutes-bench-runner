"""User session model for Chutes IDP authentication."""
from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import DateTime, String, Text, Boolean
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class UserSession(Base):
    """User session for Chutes IDP authentication.
    
    Stores OAuth tokens and user info server-side.
    The session_id is stored in an httpOnly cookie.
    """

    __tablename__ = "user_sessions"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=lambda: str(uuid4()))
    session_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    
    # User info from Chutes IDP
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    username: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # OAuth tokens (encrypted in production)
    access_token: Mapped[str] = mapped_column(Text, nullable=False)
    refresh_token: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    token_expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Scopes granted
    scopes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # Space-separated
    
    # Session metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_used_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    def __repr__(self) -> str:
        return f"<UserSession(session_id={self.session_id[:8]}..., username={self.username})>"
    
    def has_scope(self, scope: str) -> bool:
        """Check if session has a specific scope."""
        if not self.scopes:
            return False
        return scope in self.scopes.split()
    
    def can_invoke_chutes(self) -> bool:
        """Check if session has chutes:invoke scope."""
        return self.has_scope("chutes:invoke")




























