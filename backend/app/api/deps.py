"""API dependencies."""
from typing import Annotated, Optional

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.session import get_db

Settings = get_settings()


async def get_session() -> AsyncSession:
    """Get database session dependency."""
    async for session in get_db():
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]


async def verify_admin_secret(
    x_admin_secret: Optional[str] = Header(None, alias="X-Admin-Secret"),
) -> None:
    """Verify admin secret for protected endpoints."""
    settings = get_settings()
    if not settings.admin_secret:
        return  # No admin secret configured, allow access
    
    if x_admin_secret != settings.admin_secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin secret",
        )


AdminDep = Annotated[None, Depends(verify_admin_secret)]


async def get_bearer_token(
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> str:
    """Extract bearer token from Authorization header."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header",
        )
    return token.strip()


ApiKeyDep = Annotated[str, Depends(get_bearer_token)]
























