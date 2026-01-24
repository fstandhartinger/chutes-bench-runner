"""Janus Gateway inference client wrapper."""
from __future__ import annotations

from typing import Optional

from app.core.config import get_settings
from app.services.chutes_client import ChutesClient

_client: Optional[ChutesClient] = None


def _build_client(
    api_key: Optional[str] = None,
    user_access_token: Optional[str] = None,
) -> ChutesClient:
    settings = get_settings()
    resolved_key = api_key or settings.janus_gateway_api_key or settings.chutes_api_key
    client = ChutesClient(api_key=resolved_key, user_access_token=user_access_token)
    client.base_url = settings.janus_gateway_base_url
    client.provider = "janus"
    return client


def get_janus_client(
    user_access_token: Optional[str] = None,
    api_key: Optional[str] = None,
) -> ChutesClient:
    """Get Janus Gateway client."""
    if user_access_token or api_key:
        return _build_client(api_key=api_key, user_access_token=user_access_token)

    global _client
    if _client is None:
        _client = _build_client()
    return _client
