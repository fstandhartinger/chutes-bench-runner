"""RLM inference client for benchmark runs.

This client connects to the Chutes RLM gateway which uses Recursive Language Model
techniques for long-context handling. The RLM gateway exposes OpenAI-compatible
endpoints that handle large contexts via external slicing rather than full prompt
injection.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Optional, Tuple

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.chutes_client import (
    InferenceHTTPError,
    InferenceNetworkError,
    _is_retryable_exception,
)

logger = get_logger(__name__)
settings = get_settings()
RLM_MODEL_ALIASES = {
    "rlm-gpt-4o": "gpt-4o",
    "rlm-claude-3-5-sonnet": "claude-3-5-sonnet-20240620",
}


def _truncate_text(value: str, limit: int = 1000) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "...(truncated)"


class RLMClient:
    """Client for the Chutes RLM gateway API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        self.provider = "rlm"
        self.base_url = (base_url or settings.rlm_api_base_url).rstrip("/")
        self.api_key = api_key or settings.rlm_api_key or settings.chutes_api_key
        self._client: Optional[httpx.AsyncClient] = None
        self._rate_limit_until: float = 0.0

    def _endpoint(self) -> str:
        return f"{self.base_url}/v1/chat/completions"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _resolve_model_slug(self, model_slug: str) -> str:
        """Map synthetic RLM slugs to upstream model identifiers."""
        if model_slug in RLM_MODEL_ALIASES:
            return RLM_MODEL_ALIASES[model_slug]
        if model_slug.startswith("rlm-"):
            return model_slug[len("rlm-") :]
        return model_slug

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client and not self._client.is_closed:
            return self._client
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.rlm_timeout_seconds, connect=10.0)
        )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def get_api_base_url(self) -> str:
        return f"{self.base_url}/v1"

    def get_api_key(self) -> Optional[str]:
        return self.api_key

    async def get_model_max_output_length(self, model_identifier: str) -> Optional[int]:
        return None

    async def get_model_context_length(self, model_identifier: str) -> Optional[int]:
        # RLM handles long contexts via chunking, so we report a high context length
        return 1_000_000

    async def get_metadata(self) -> dict[str, Any]:
        client = await self._get_client()
        try:
            response = await client.get(
                f"{self.base_url}/health",
                headers=self._headers(),
            )
        except httpx.HTTPError as exc:
            raise InferenceNetworkError(str(exc)) from exc
        if response.status_code >= 400:
            raise InferenceHTTPError(
                status_code=response.status_code,
                response_text=_truncate_text(response.text or ""),
            )
        try:
            return response.json()
        except Exception:
            return {}

    @retry(
        retry=retry_if_exception(_is_retryable_exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def run_inference(
        self,
        model_slug: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise InferenceHTTPError(status_code=401, response_text="Missing RLM API key")

        now = time.monotonic()
        if now < self._rate_limit_until:
            await asyncio.sleep(self._rate_limit_until - now)

        payload = {
            "model": self._resolve_model_slug(model_slug),
            "messages": messages,
        }
        payload.update(kwargs)

        client = await self._get_client()
        try:
            response = await client.post(
                self._endpoint(),
                json=payload,
                headers=self._headers(),
            )
        except httpx.TimeoutException as exc:
            raise InferenceNetworkError(str(exc)) from exc
        except httpx.TransportError as exc:
            raise InferenceNetworkError(str(exc)) from exc

        if response.status_code >= 400:
            if response.status_code in (429, 503):
                retry_after = response.headers.get("retry-after")
                delay = settings.chutes_rate_limit_sleep_seconds
                if retry_after:
                    try:
                        delay = max(delay, int(float(retry_after)))
                    except ValueError:
                        pass
                self._rate_limit_until = max(self._rate_limit_until, time.monotonic() + max(delay, 1))
            raise InferenceHTTPError(
                status_code=response.status_code,
                response_text=_truncate_text(response.text or ""),
            )

        try:
            result = response.json()
        except Exception as exc:
            raise InferenceHTTPError(
                status_code=response.status_code,
                response_text=f"Invalid JSON response: {_truncate_text(response.text or '')}",
            ) from exc

        return result

    async def get_completion_text(
        self,
        model_slug: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return await self.get_completion_messages(model_slug, messages, **kwargs)

    async def get_completion_messages(
        self,
        model_slug: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        attempt = 0
        max_attempts = kwargs.pop("response_attempts", 3)
        delay_seconds = 1

        metadata: dict[str, Any] = {
            "usage": {},
            "model": model_slug,
            "finish_reason": None,
        }

        while attempt < max_attempts:
            attempt += 1
            try:
                response = await self.run_inference(model_slug, messages, **kwargs)
            except InferenceHTTPError as exc:
                metadata["response_error"] = str(exc)
                metadata["response_error_code"] = exc.status_code
                metadata["response_error_detail"] = exc.response_text
                if attempt < max_attempts:
                    await asyncio.sleep(delay_seconds)
                    delay_seconds = min(delay_seconds * 2, 10)
                    continue
                return "", metadata
            except InferenceNetworkError as exc:
                metadata["response_error"] = str(exc)
                if attempt < max_attempts:
                    await asyncio.sleep(delay_seconds)
                    delay_seconds = min(delay_seconds * 2, 10)
                    continue
                return "", metadata

            text, response_metadata = _extract_message_text(response)
            metadata["usage"] = response.get("usage", {}) if isinstance(response, dict) else {}
            metadata["model"] = response.get("model")
            metadata.update(response_metadata)
            metadata["response_attempts"] = attempt

            if text:
                return text, metadata
            if attempt < max_attempts:
                await asyncio.sleep(delay_seconds)
                delay_seconds = min(delay_seconds * 2, 10)

        return "", metadata


def _extract_message_text(response: dict[str, Any]) -> Tuple[str, dict[str, Any]]:
    choices = response.get("choices")
    metadata: dict[str, Any] = {
        "choices_count": len(choices) if isinstance(choices, list) else 0,
    }
    if isinstance(response.get("error"), dict):
        error_payload = response.get("error") or {}
        error_message = error_payload.get("message") or str(error_payload)
        metadata["response_error"] = f"RLM error: {error_message}"
        if error_payload.get("type"):
            metadata["response_error_type"] = error_payload.get("type")
        if error_payload.get("code"):
            metadata["response_error_code"] = error_payload.get("code")
    if not choices or not isinstance(choices, list):
        metadata["response_error"] = "No choices in response"
        return "", metadata

    choice = choices[0] if choices else None
    if not isinstance(choice, dict):
        metadata["response_error"] = "Choice payload is not a dict"
        return "", metadata

    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    content = message.get("content")
    reasoning = message.get("reasoning") or message.get("analysis")
    text = ""
    source = None

    if content:
        text = content
        source = "content"
    elif reasoning:
        text = reasoning
        source = "reasoning"

    if not text and isinstance(choice.get("text"), str):
        text = choice.get("text") or ""
        source = source or "text"

    metadata.update(
        {
            "response_source": source,
            "content_length": len(content) if isinstance(content, str) else 0,
            "reasoning_length": len(reasoning) if isinstance(reasoning, str) else 0,
            "finish_reason": choice.get("finish_reason"),
        }
    )

    if not text:
        metadata["response_error"] = (
            "Empty response content from RLM (content and reasoning are missing)"
        )

    return text or "", metadata
