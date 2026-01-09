"""Authentication service for Chutes IDP OAuth2 flow."""
import base64
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Tuple

import httpx
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.session import UserSession

logger = get_logger(__name__)
settings = get_settings()

# IDP endpoints
IDP_BASE_URL = "https://idp.chutes.ai"
IDP_AUTHORIZE_URL = f"{IDP_BASE_URL}/idp/authorize"
IDP_TOKEN_URL = f"{IDP_BASE_URL}/idp/token"
IDP_USERINFO_URL = f"{IDP_BASE_URL}/idp/userinfo"
IDP_REVOKE_URL = f"{IDP_BASE_URL}/idp/token/revoke"

# Session settings
SESSION_LIFETIME_DAYS = 30
SESSION_COOKIE_NAME = "bench_session"


def generate_pkce() -> Tuple[str, str]:
    """Generate PKCE code_verifier and code_challenge (S256)."""
    code_verifier = secrets.token_urlsafe(32)
    sha256_hash = hashlib.sha256(code_verifier.encode()).digest()
    code_challenge = base64.urlsafe_b64encode(sha256_hash).rstrip(b"=").decode("ascii")
    return code_verifier, code_challenge


def generate_state() -> str:
    """Generate a random state for CSRF protection."""
    return secrets.token_urlsafe(16)


def generate_session_id() -> str:
    """Generate a random session ID."""
    return secrets.token_urlsafe(32)


def build_authorization_url(
    redirect_uri: str,
    state: str,
    code_challenge: str,
    scopes: str = "openid profile chutes:invoke account:read",
) -> str:
    """Build the Chutes IDP authorization URL."""
    client_id = settings.chutes_client_id
    if not client_id:
        raise ValueError("CHUTES_CLIENT_ID not configured")
    
    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": scopes,
        "state": state,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
    }
    
    query = "&".join(f"{k}={v.replace(' ', '%20')}" for k, v in params.items())
    return f"{IDP_AUTHORIZE_URL}?{query}"


async def exchange_code_for_tokens(
    code: str,
    redirect_uri: str,
    code_verifier: str,
) -> Optional[dict]:
    """Exchange authorization code for tokens."""
    client_id = settings.chutes_client_id
    client_secret = settings.chutes_client_secret
    
    if not client_id:
        raise ValueError("CHUTES_CLIENT_ID not configured")
    
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "code_verifier": code_verifier,
    }
    
    if client_secret:
        data["client_secret"] = client_secret
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(IDP_TOKEN_URL, data=data)
        
        if response.status_code != 200:
            logger.error(
                "Token exchange failed",
                status=response.status_code,
                body=response.text,
            )
            return None
        
        return response.json()


async def refresh_access_token(refresh_token: str) -> Optional[dict]:
    """Refresh an access token using a refresh token."""
    client_id = settings.chutes_client_id
    client_secret = settings.chutes_client_secret
    
    if not client_id:
        raise ValueError("CHUTES_CLIENT_ID not configured")
    
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
    }
    
    if client_secret:
        data["client_secret"] = client_secret
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(IDP_TOKEN_URL, data=data)
        
        if response.status_code != 200:
            logger.error(
                "Token refresh failed",
                status=response.status_code,
                body=response.text,
            )
            return None
        
        return response.json()


async def fetch_userinfo(access_token: str) -> Optional[dict]:
    """Fetch user info from Chutes IDP."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(
            IDP_USERINFO_URL,
            headers={"Authorization": f"Bearer {access_token}"},
        )
        
        if response.status_code != 200:
            logger.error(
                "Userinfo fetch failed",
                status=response.status_code,
                body=response.text,
            )
            return None
        
        return response.json()


async def revoke_token(token: str) -> bool:
    """Revoke an access or refresh token."""
    client_id = settings.chutes_client_id
    client_secret = settings.chutes_client_secret
    
    data = {
        "token": token,
        "client_id": client_id,
    }
    
    if client_secret:
        data["client_secret"] = client_secret
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(IDP_REVOKE_URL, data=data)
        return response.status_code == 200


async def create_session(
    db: AsyncSession,
    user_id: str,
    username: str,
    access_token: str,
    refresh_token: Optional[str],
    expires_in: Optional[int],
    scopes: str,
) -> UserSession:
    """Create a new user session."""
    session_id = generate_session_id()
    
    token_expires_at = None
    if expires_in:
        token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
    
    session = UserSession(
        session_id=session_id,
        user_id=user_id,
        username=username,
        access_token=access_token,
        refresh_token=refresh_token,
        token_expires_at=token_expires_at,
        scopes=scopes,
        expires_at=datetime.utcnow() + timedelta(days=SESSION_LIFETIME_DAYS),
    )
    
    db.add(session)
    await db.commit()
    await db.refresh(session)
    
    logger.info("Created session", session_id=session_id[:8], username=username)
    return session


async def get_session(db: AsyncSession, session_id: str) -> Optional[UserSession]:
    """Get a session by session_id."""
    result = await db.execute(
        select(UserSession).where(
            UserSession.session_id == session_id,
            UserSession.is_active == True,
            UserSession.expires_at > datetime.utcnow(),
        )
    )
    session = result.scalar_one_or_none()
    
    if session:
        # Update last_used_at
        session.last_used_at = datetime.utcnow()
        await db.commit()
    
    return session


async def get_valid_access_token(db: AsyncSession, session: UserSession) -> Optional[str]:
    """Get a valid access token, refreshing if needed."""
    # Check if token is still valid (with 5 min buffer)
    if session.token_expires_at:
        if session.token_expires_at > datetime.utcnow() + timedelta(minutes=5):
            return session.access_token
        
        # Try to refresh
        if session.refresh_token:
            tokens = await refresh_access_token(session.refresh_token)
            if tokens:
                session.access_token = tokens["access_token"]
                if "refresh_token" in tokens:
                    session.refresh_token = tokens["refresh_token"]
                if "expires_in" in tokens:
                    session.token_expires_at = datetime.utcnow() + timedelta(seconds=tokens["expires_in"])
                await db.commit()
                return session.access_token
        
        # Token expired and can't refresh
        return None
    
    return session.access_token


async def delete_session(db: AsyncSession, session_id: str) -> bool:
    """Delete a session (logout)."""
    result = await db.execute(
        select(UserSession).where(UserSession.session_id == session_id)
    )
    session = result.scalar_one_or_none()
    
    if session:
        # Revoke tokens
        if session.access_token:
            await revoke_token(session.access_token)
        if session.refresh_token:
            await revoke_token(session.refresh_token)
        
        # Mark as inactive
        session.is_active = False
        await db.commit()
        
        logger.info("Deleted session", session_id=session_id[:8])
        return True
    
    return False


async def cleanup_expired_sessions(db: AsyncSession) -> int:
    """Clean up expired sessions."""
    result = await db.execute(
        delete(UserSession).where(
            UserSession.expires_at < datetime.utcnow()
        )
    )
    await db.commit()
    return result.rowcount




























