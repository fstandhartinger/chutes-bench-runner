"""Chutes API client for model listing and inference."""
import asyncio
from typing import Any, Optional, Tuple

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# IDP endpoint for user token inference
IDP_INFERENCE_URL = "https://idp.chutes.ai/v1"


class InferenceHTTPError(RuntimeError):
    """Raised when Chutes returns a non-2xx response."""

    def __init__(self, status_code: int, response_text: str, request_id: Optional[str] = None):
        message = f"HTTP {status_code} from Chutes: {response_text.strip() or 'Empty response body'}"
        if request_id:
            message = f"{message} (request_id={request_id})"
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text
        self.request_id = request_id


class InferenceNetworkError(RuntimeError):
    """Raised when network or transport errors occur during inference."""


def _truncate_text(value: str, limit: int = 1000) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "...(truncated)"


def _is_retryable_exception(exc: Exception) -> bool:
    if isinstance(exc, InferenceHTTPError):
        return exc.status_code in (408, 429) or exc.status_code >= 500
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(exc, InferenceNetworkError):
        return True
    return False


class ChutesClient:
    """Client for Chutes API operations.
    
    Supports two modes:
    1. API key mode: Uses CHUTES_API_KEY for inference (default)
    2. User token mode: Uses user's OAuth access token for inference (BYOC)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        user_access_token: Optional[str] = None,
    ):
        """Initialize the Chutes client.
        
        Args:
            api_key: Chutes API key (fallback if no user token)
            user_access_token: User's OAuth access token for BYOC
        """
        self.api_key = api_key or settings.chutes_api_key
        self.user_access_token = user_access_token
        self.base_url = settings.chutes_api_base_url
        self.models_api_url = settings.chutes_models_api_url
        self._client: Optional[httpx.AsyncClient] = None
        self._is_user_token_mode = user_access_token is not None

    @property
    def using_user_token(self) -> bool:
        """Check if client is using user's token."""
        return self._is_user_token_mode

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            # Use user token if available, otherwise use API key
            auth_token = self.user_access_token or self.api_key
            
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(300.0, connect=10.0),
                headers={
                    "Authorization": f"Bearer {auth_token}",
                    "Content-Type": "application/json",
                },
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def list_models(self) -> list[dict[str, Any]]:
        """
        List all available models from Chutes API.
        
        Returns:
            List of model dictionaries with slug, name, tagline, etc.
        """
        # Use a separate client without auth for public API
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
        ) as client:
            all_items: list[dict[str, Any]] = []
            page = 0
            limit = 500

            while True:
                url = f"{self.models_api_url}/chutes/?include_public=true&page={page}&limit={limit}&include_schemas=false"
                logger.debug("Fetching models", url=url, page=page)

                response = await client.get(url)
                response.raise_for_status()
                data = response.json()

                items = data.get("items", [])
                all_items.extend(items)

                if len(items) < limit or len(all_items) >= data.get("total", 0):
                    break
                page += 1

        # Transform to consistent format
        models = [
            {
                "slug": item.get("name"),
                "name": item.get("name"),
                "tagline": item.get("tagline"),
                "user": item.get("user", {}).get("username") if item.get("user") else None,
                "logo": item.get("logo"),
                "chute_id": item.get("chute_id"),
                "instance_count": len(item.get("instances", [])),
            }
            for item in all_items
        ]

        logger.info("Fetched models", count=len(models))
        return models

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(_is_retryable_exception),
    )
    async def run_inference(
        self,
        model_slug: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Run inference against a Chutes model.
        
        When using a user's OAuth token (BYOC mode), requests go through the IDP
        endpoint with a Host header override to route to the LLM gateway.
        
        Args:
            model_slug: The model identifier/slug
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0 for deterministic)
            max_tokens: Maximum tokens to generate
            stop: Optional stop sequences
            **kwargs: Additional parameters
            
        Returns:
            Response dict with 'choices', 'usage', etc.
        """
        payload: dict[str, Any] = {
            "model": model_slug,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if stop:
            payload["stop"] = stop
        payload.update(kwargs)

        logger.debug(
            "Running inference",
            model=model_slug,
            message_count=len(messages),
            user_token_mode=self._is_user_token_mode,
        )

        try:
            if self._is_user_token_mode:
                # Use IDP endpoint with Host header for user token auth
                # This routes through the IDP which validates the user's token
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(300.0, connect=10.0),
                ) as client:
                    response = await client.post(
                        f"{IDP_INFERENCE_URL}/chat/completions",
                        json=payload,
                        headers={
                            "Authorization": f"Bearer {self.user_access_token}",
                            "Host": "llm.chutes.ai",
                            "Content-Type": "application/json",
                        },
                    )
            else:
                # Use standard API with API key
                client = await self._get_client()
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise InferenceNetworkError(f"Network error contacting Chutes: {exc}") from exc

        if response.status_code >= 400:
            request_id = response.headers.get("x-request-id") or response.headers.get("x-requestid")
            raise InferenceHTTPError(
                status_code=response.status_code,
                response_text=_truncate_text(response.text or ""),
                request_id=request_id,
            )

        try:
            result = response.json()
        except Exception as exc:
            raise InferenceHTTPError(
                status_code=response.status_code,
                response_text=f"Invalid JSON response: {_truncate_text(response.text or '')}",
            ) from exc

        logger.debug(
            "Inference complete",
            model=model_slug,
            usage=result.get("usage"),
        )
        return result

    def _extract_message_text(self, response: dict[str, Any]) -> Tuple[str, dict[str, Any]]:
        choices = response.get("choices")
        metadata: dict[str, Any] = {
            "choices_count": len(choices) if isinstance(choices, list) else 0,
        }
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
                "Empty response content from Chutes (content and reasoning are missing)"
            )

        return text or "", metadata

    async def get_completion_text(
        self,
        model_slug: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> tuple[str, dict[str, Any]]:
        """
        Convenience method to get completion text.
        
        Returns:
            Tuple of (response_text, metadata_dict)
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        attempt = 0
        max_attempts = kwargs.pop("response_attempts", 3)
        delay_seconds = 1

        # Initialize defaults
        metadata = {
            "usage": {},
            "model": model_slug,
            "finish_reason": None,
        }

        while attempt < max_attempts:
            attempt += 1
            try:
                response = await self.run_inference(model_slug, messages, **kwargs)
            except Exception as e:
                logger.error("Inference failed", model=model_slug, error=str(e))
                raise

            text = ""
            response_metadata: dict[str, Any] = {}
            if response and isinstance(response, dict):
                text, response_metadata = self._extract_message_text(response)
                metadata["usage"] = response.get("usage", {})
                metadata["model"] = response.get("model")

            metadata.update(response_metadata)
            metadata["response_attempts"] = attempt

            if text:
                return text, metadata

            if attempt < max_attempts:
                await asyncio.sleep(delay_seconds)
                delay_seconds = min(delay_seconds * 2, 10)

        return "", metadata


# Singleton client instance for API key mode
_client: Optional[ChutesClient] = None


def get_chutes_client(
    user_access_token: Optional[str] = None,
    api_key: Optional[str] = None,
) -> ChutesClient:
    """Get Chutes client.
    
    Args:
        user_access_token: If provided, returns a new client using the user's
                          OAuth token for BYOC mode. Otherwise returns the
                          singleton API key client.
    
    Returns:
        ChutesClient configured for the appropriate auth mode.
    """
    if user_access_token or api_key:
        # Always create a new client for explicit auth mode
        return ChutesClient(api_key=api_key, user_access_token=user_access_token)
    
    # Use singleton for API key mode
    global _client
    if _client is None:
        _client = ChutesClient()
    return _client
