"""Authentication routes for Chutes IDP OAuth2 flow."""
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import RedirectResponse

from app.api.deps import SessionDep
from app.core.config import get_settings
from app.core.logging import get_logger
from app.services import auth_service

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter(prefix="/auth", tags=["auth"])

# In-memory storage for PKCE state (in production, use Redis or encrypted cookies)
# This is acceptable for our use case as we're running single-instance
_auth_state_store: dict[str, dict] = {}


@router.get("/login")
async def login(
    request: Request,
    return_to: Optional[str] = Query(None, description="URL to return to after login"),
):
    """
    Initiate the Chutes IDP OAuth2 login flow.
    
    Generates PKCE challenge and redirects to Chutes IDP authorization page.
    """
    if not settings.chutes_client_id:
        raise HTTPException(
            status_code=501,
            detail="Chutes IDP not configured. Set CHUTES_CLIENT_ID and CHUTES_CLIENT_SECRET.",
        )
    
    # Generate PKCE and state
    state = auth_service.generate_state()
    code_verifier, code_challenge = auth_service.generate_pkce()
    
    # Determine redirect URI based on request
    # Use frontend URL for callback since Next.js will handle the OAuth callback
    redirect_uri = f"{settings.frontend_url}/api/auth/callback"
    
    # Store state for verification (keyed by state)
    _auth_state_store[state] = {
        "code_verifier": code_verifier,
        "return_to": return_to or "/",
        "redirect_uri": redirect_uri,
    }
    
    # Build authorization URL
    auth_url = auth_service.build_authorization_url(
        redirect_uri=redirect_uri,
        state=state,
        code_challenge=code_challenge,
    )
    
    logger.info("Initiating OAuth login", state=state[:8])
    return RedirectResponse(url=auth_url)


@router.get("/callback")
async def callback(
    db: SessionDep,
    response: Response,
    code: Optional[str] = None,
    state: Optional[str] = None,
    error: Optional[str] = None,
    error_description: Optional[str] = None,
):
    """
    Handle the OAuth2 callback from Chutes IDP.
    
    Exchanges the authorization code for tokens and creates a session.
    """
    if error:
        logger.error("OAuth callback error", error=error, description=error_description)
        raise HTTPException(
            status_code=400,
            detail=f"Authorization failed: {error} - {error_description or 'Unknown error'}",
        )
    
    if not code or not state:
        raise HTTPException(status_code=400, detail="Missing code or state parameter")
    
    # Verify state and get stored data
    stored = _auth_state_store.pop(state, None)
    if not stored:
        raise HTTPException(status_code=400, detail="Invalid or expired state")
    
    code_verifier = stored["code_verifier"]
    redirect_uri = stored["redirect_uri"]
    return_to = stored.get("return_to", "/")
    
    # Exchange code for tokens
    tokens = await auth_service.exchange_code_for_tokens(
        code=code,
        redirect_uri=redirect_uri,
        code_verifier=code_verifier,
    )
    
    if not tokens or "access_token" not in tokens:
        raise HTTPException(status_code=400, detail="Failed to exchange code for tokens")
    
    # Fetch user info
    userinfo = await auth_service.fetch_userinfo(tokens["access_token"])
    if not userinfo:
        raise HTTPException(status_code=400, detail="Failed to fetch user info")
    
    # Create session
    session = await auth_service.create_session(
        db=db,
        user_id=userinfo.get("sub", ""),
        username=userinfo.get("username", userinfo.get("sub", "Unknown")),
        access_token=tokens["access_token"],
        refresh_token=tokens.get("refresh_token"),
        expires_in=tokens.get("expires_in"),
        scopes=tokens.get("scope", ""),
    )
    
    logger.info("OAuth login successful", username=session.username)
    
    # Return session info and cookie instruction
    # The frontend will set the cookie and redirect
    return {
        "success": True,
        "session_id": session.session_id,
        "user": {
            "id": session.user_id,
            "username": session.username,
        },
        "has_invoke_scope": session.can_invoke_chutes(),
        "return_to": return_to,
    }


@router.get("/me")
async def get_current_user(
    request: Request,
    db: SessionDep,
):
    """
    Get the current authenticated user.
    
    Returns user info if authenticated, 401 otherwise.
    """
    session_id = request.cookies.get(auth_service.SESSION_COOKIE_NAME)
    
    if not session_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    session = await auth_service.get_session(db, session_id)
    
    if not session:
        raise HTTPException(status_code=401, detail="Session expired or invalid")
    
    # Check if we have a valid access token
    access_token = await auth_service.get_valid_access_token(db, session)
    
    return {
        "authenticated": True,
        "user": {
            "id": session.user_id,
            "username": session.username,
        },
        "has_invoke_scope": session.can_invoke_chutes(),
        "has_valid_token": access_token is not None,
    }


@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    db: SessionDep,
):
    """
    Log out the current user.
    
    Revokes tokens and deletes the session.
    """
    session_id = request.cookies.get(auth_service.SESSION_COOKIE_NAME)
    
    if session_id:
        await auth_service.delete_session(db, session_id)
    
    # Clear the session cookie
    response.delete_cookie(
        key=auth_service.SESSION_COOKIE_NAME,
        path="/",
        secure=True,
        httponly=True,
        samesite="lax",
    )
    
    return {"success": True, "message": "Logged out successfully"}


@router.get("/status")
async def auth_status(request: Request, db: SessionDep):
    """
    Check authentication status without requiring auth.
    
    Returns whether the user is authenticated and IDP is configured.
    """
    idp_configured = bool(settings.chutes_client_id)
    
    session_id = request.cookies.get(auth_service.SESSION_COOKIE_NAME)
    authenticated = False
    user = None
    has_invoke_scope = False
    
    if session_id:
        session = await auth_service.get_session(db, session_id)
        if session:
            authenticated = True
            user = {
                "id": session.user_id,
                "username": session.username,
            }
            has_invoke_scope = session.can_invoke_chutes()
    
    return {
        "idp_configured": idp_configured,
        "authenticated": authenticated,
        "user": user,
        "has_invoke_scope": has_invoke_scope,
    }













