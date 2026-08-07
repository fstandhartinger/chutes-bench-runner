"""OpenRouter model catalog and OpenAI-compatible inference client."""
from __future__ import annotations

from typing import Any

import httpx

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.chutes_client import ChutesClient, InferenceHTTPError

logger = get_logger(__name__)
settings = get_settings()


def _price_per_million(value: Any) -> float | None:
    """Convert OpenRouter's per-token decimal string to USD per million."""
    try:
        return float(value) * 1_000_000
    except (TypeError, ValueError):
        return None


class OpenRouterClient(ChutesClient):
    """Chutes-compatible client backed by OpenRouter's OpenAI API."""

    provider = "openrouter"
    provider_display_name = "OpenRouter"

    def __init__(self, api_key: str | None = None):
        # Avoid ChutesClient's intentional CHUTES_API_KEY fallback: crossing
        # credentials between providers would turn a missing OpenRouter key into
        # a misleading 401 and could expose the wrong credential to a sandbox.
        super().__init__(api_key="openrouter-key-placeholder")
        self.api_key = api_key or settings.openrouter_api_key
        self.user_access_token = None
        self.base_url = settings.openrouter_api_base_url.rstrip("/")
        self.models_api_url = self.base_url
        self._is_user_token_mode = False

    async def _get_client(self) -> httpx.AsyncClient:
        if not self.api_key:
            raise InferenceHTTPError(
                status_code=401,
                response_text="OPENROUTER_API_KEY is not configured",
                provider_name=self.provider_display_name,
            )
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(300.0, connect=10.0),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://github.com/fstandhartinger/chutes-bench-runner",
                    "X-Title": "Chutes Bench Runner",
                },
            )
        return self._client

    async def _fetch_catalog(self) -> list[dict[str, Any]]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else None
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
            headers=headers,
        ) as client:
            response = await client.get(f"{self.base_url}/models")
            response.raise_for_status()
            payload = response.json()
        entries = payload.get("data") if isinstance(payload, dict) else None
        return [entry for entry in (entries or []) if isinstance(entry, dict)]

    async def get_target_model(self) -> dict[str, Any]:
        target = settings.openrouter_model_slug
        for entry in await self._fetch_catalog():
            if entry.get("id") == target:
                return entry
        raise RuntimeError(f"Configured OpenRouter model is not listed: {target}")

    async def list_models(self) -> list[dict[str, Any]]:
        """Expose only the verified benchmark target, not OpenRouter's full catalog."""
        entry = await self.get_target_model()
        return [
            {
                "slug": entry["id"],
                "name": entry.get("name") or entry["id"],
                "tagline": entry.get("description"),
                "user": "OpenRouter",
                "logo": None,
                "chute_id": None,
                "instance_count": 1,
                "is_llm": True,
            }
        ]

    async def _fetch_llm_models(
        self,
    ) -> tuple[dict[str, int], dict[str, int], dict[str, tuple[float, float]]]:
        entries = await self._fetch_catalog()
        output_limits: dict[str, int] = {}
        context_limits: dict[str, int] = {}
        pricing_map: dict[str, tuple[float, float]] = {}
        for entry in entries:
            model_id = entry.get("id")
            if not isinstance(model_id, str) or not model_id:
                continue
            context_length = entry.get("context_length")
            top_provider = entry.get("top_provider") or {}
            max_output = (
                top_provider.get("max_completion_tokens")
                if isinstance(top_provider, dict)
                else None
            )
            pricing = entry.get("pricing") or {}
            prompt_price = _price_per_million(
                pricing.get("prompt") if isinstance(pricing, dict) else None
            )
            completion_price = _price_per_million(
                pricing.get("completion") if isinstance(pricing, dict) else None
            )
            if isinstance(context_length, int) and context_length > 0:
                context_limits[model_id] = context_length
            if isinstance(max_output, int) and max_output > 0:
                output_limits[model_id] = max_output
            if prompt_price is not None and completion_price is not None:
                pricing_map[model_id] = (prompt_price, completion_price)
        return output_limits, context_limits, pricing_map

    async def get_model_pricing(
        self, *identifiers: str | None
    ) -> tuple[float, float] | None:
        await self._get_llm_model_limits()
        for identifier in identifiers:
            if identifier and self._llm_pricing_cache:
                pricing = self._llm_pricing_cache.get(identifier)
                if pricing:
                    return pricing
        return None


_client: OpenRouterClient | None = None


def get_openrouter_client(api_key: str | None = None) -> OpenRouterClient:
    """Return an explicit-key client or the process-wide OpenRouter client."""
    global _client
    if api_key:
        return OpenRouterClient(api_key=api_key)
    if _client is None:
        _client = OpenRouterClient()
    return _client
