"""Chutes API client for model listing and inference."""
import asyncio
import re
import time
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


def _is_max_tokens_error(exc: InferenceHTTPError) -> bool:
    text = (exc.response_text or "").lower()
    return any(
        phrase in text
        for phrase in (
            "max_tokens",
            "max_completion_tokens",
            "completion tokens",
            "maximum context length",
            "context length",
            "max output",
            "max_output",
        )
    )


def _is_context_length_error(exc: InferenceHTTPError) -> bool:
    text = (exc.response_text or "").lower()
    return "context length" in text and "input" in text and "longer" in text


def _extract_max_tokens_limit(response_text: str) -> Optional[int]:
    if not response_text:
        return None
    patterns = (
        r"supports at most\\s+(\\d+)\\s+completion tokens",
        r"supports at most\\s+(\\d+)\\s+tokens",
        r"max[_\\s-]?completion[_\\s-]?tokens[^\\d]*(\\d+)",
        r"max[_\\s-]?output[^\\d]*(\\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            try:
                value = int(match.group(1))
            except ValueError:
                continue
            if value > 0:
                return value
    return None


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
        self._llm_output_cache: Optional[dict[str, int]] = None
        self._llm_context_cache: Optional[dict[str, int]] = None
        self._llm_pricing_cache: Optional[dict[str, tuple[float, float]]] = None
        self._llm_cache_at: float = 0.0
        self._chutes_pricing_cache: Optional[dict[str, tuple[float, float]]] = None
        self._chutes_pricing_cache_at: float = 0.0

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
        models = []
        for item in all_items:
            pricing = item.get("current_estimated_price") or {}
            per_million_tokens = pricing.get("per_million_tokens") if isinstance(pricing, dict) else None
            models.append(
                {
                    "slug": item.get("name"),
                    "name": item.get("name"),
                    "tagline": item.get("tagline"),
                    "user": item.get("user", {}).get("username") if item.get("user") else None,
                    "logo": item.get("logo"),
                    "chute_id": item.get("chute_id"),
                    "instance_count": len(item.get("instances", [])),
                    "is_llm": per_million_tokens is not None,
                }
            )

        logger.info("Fetched models", count=len(models))
        return models

    async def _fetch_llm_models(
        self,
    ) -> tuple[dict[str, int], dict[str, int], dict[str, tuple[float, float]]]:
        url = f"{self.base_url}/models"
        headers = None
        auth_token = self.user_access_token or self.api_key
        if auth_token:
            headers = {"Authorization": f"Bearer {auth_token}"}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                data = response.json()
        except Exception as exc:
            logger.warning("Failed to fetch LLM model limits", error=str(exc))
            return {}, {}, {}

        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return {}, {}, {}

        output_limits: dict[str, int] = {}
        context_limits: dict[str, int] = {}
        pricing_map: dict[str, tuple[float, float]] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            max_output_length = item.get("max_output_length")
            context_length = item.get("max_model_len") or item.get("context_length")
            raw_pricing = item.get("pricing") if isinstance(item.get("pricing"), dict) else {}
            raw_price = item.get("price") if isinstance(item.get("price"), dict) else {}
            input_price = raw_pricing.get("prompt")
            output_price = raw_pricing.get("completion")
            if input_price is None or output_price is None:
                input_price = (
                    raw_price.get("input", {}).get("usd")
                    if isinstance(raw_price.get("input"), dict)
                    else input_price
                )
                output_price = (
                    raw_price.get("output", {}).get("usd")
                    if isinstance(raw_price.get("output"), dict)
                    else output_price
                )
            if not isinstance(max_output_length, int):
                max_output_length = None
            if not isinstance(context_length, int):
                context_length = None
            if not isinstance(input_price, (int, float)):
                input_price = None
            if not isinstance(output_price, (int, float)):
                output_price = None
            identifiers: set[str] = set()
            for key in ("id", "name", "model", "slug"):
                value = item.get(key)
                if isinstance(value, str) and value:
                    identifiers.add(value)
            chute_id = item.get("chute_id")
            if isinstance(chute_id, str) and chute_id:
                identifiers.add(chute_id)
            if identifiers:
                for identifier in identifiers:
                    if max_output_length is not None:
                        output_limits[identifier] = max_output_length
                    if context_length is not None:
                        context_limits[identifier] = context_length
                    if input_price is not None and output_price is not None:
                        pricing_map[identifier] = (float(input_price), float(output_price))
        return output_limits, context_limits, pricing_map

    async def _fetch_chutes_pricing_map(self) -> dict[str, tuple[float, float]]:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
            all_items: list[dict[str, Any]] = []
            page = 0
            limit = 500

            while True:
                url = (
                    f"{self.models_api_url}/chutes/?include_public=true"
                    f"&page={page}&limit={limit}&include_schemas=false"
                )
                response = await client.get(url)
                response.raise_for_status()
                data = response.json()
                items = data.get("items", [])
                all_items.extend(items)
                if len(items) < limit or len(all_items) >= data.get("total", 0):
                    break
                page += 1

        pricing_map: dict[str, tuple[float, float]] = {}
        for item in all_items:
            if not isinstance(item, dict):
                continue
            pricing = item.get("current_estimated_price") or {}
            per_million = pricing.get("per_million_tokens") if isinstance(pricing, dict) else None
            if not isinstance(per_million, dict):
                continue
            input_price = (
                per_million.get("input", {}).get("usd")
                if isinstance(per_million.get("input"), dict)
                else None
            )
            output_price = (
                per_million.get("output", {}).get("usd")
                if isinstance(per_million.get("output"), dict)
                else None
            )
            if not isinstance(input_price, (int, float)) or not isinstance(output_price, (int, float)):
                continue
            name = item.get("name")
            if isinstance(name, str) and name:
                pricing_map[name] = (float(input_price), float(output_price))
            chute_id = item.get("chute_id")
            if isinstance(chute_id, str) and chute_id:
                pricing_map[chute_id] = (float(input_price), float(output_price))
        return pricing_map

    async def _get_llm_model_limits(self) -> dict[str, int]:
        ttl_seconds = settings.chutes_models_cache_ttl_seconds
        now = time.monotonic()
        if (
            self._llm_output_cache
            and (now - self._llm_cache_at) < ttl_seconds
            and self._llm_pricing_cache is not None
        ):
            return self._llm_output_cache

        output_limits, context_limits, pricing_map = await self._fetch_llm_models()
        if output_limits or context_limits or pricing_map:
            if output_limits:
                self._llm_output_cache = output_limits
            if context_limits:
                self._llm_context_cache = context_limits
            self._llm_pricing_cache = pricing_map
            self._llm_cache_at = now
            return self._llm_output_cache or {}
        return self._llm_output_cache or {}

    async def get_model_max_output_length(self, model_identifier: str) -> Optional[int]:
        limits = await self._get_llm_model_limits()
        if not limits:
            return None
        return limits.get(model_identifier)

    async def get_model_context_length(self, model_identifier: str) -> Optional[int]:
        await self._get_llm_model_limits()
        if not self._llm_context_cache:
            return None
        return self._llm_context_cache.get(model_identifier)

    async def get_model_pricing(
        self, *identifiers: Optional[str]
    ) -> Optional[tuple[float, float]]:
        chutes_pricing = await self._get_chutes_pricing_map()
        for identifier in identifiers:
            if identifier and identifier in chutes_pricing:
                return chutes_pricing[identifier]
        return None

    async def _get_chutes_pricing_map(self) -> dict[str, tuple[float, float]]:
        ttl_seconds = settings.chutes_models_cache_ttl_seconds
        now = time.monotonic()
        if self._chutes_pricing_cache and (now - self._chutes_pricing_cache_at) < ttl_seconds:
            return self._chutes_pricing_cache
        try:
            pricing_map = await self._fetch_chutes_pricing_map()
        except Exception as exc:
            logger.warning("Failed to fetch Chutes pricing", error=str(exc))
            return self._chutes_pricing_cache or {}
        self._chutes_pricing_cache = pricing_map
        self._chutes_pricing_cache_at = now
        return pricing_map

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // 4)

    def _compute_safe_max_tokens(self, max_output_length: int, input_tokens: int) -> int:
        margin = settings.chutes_max_tokens_margin
        if max_output_length <= 0:
            return 1
        margin = max(0, min(margin, max_output_length))
        available = max_output_length - input_tokens - margin
        if available <= 0:
            return 1
        return max(1, available)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception(_is_retryable_exception),
        reraise=True,
    )
    async def run_inference(
        self,
        model_slug: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: Optional[float] = None,
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
                timeout_config = (
                    httpx.Timeout(timeout, connect=10.0)
                    if timeout is not None
                    else httpx.Timeout(300.0, connect=10.0)
                )
                async with httpx.AsyncClient(
                    timeout=timeout_config,
                ) as client:
                    request_kwargs: dict[str, Any] = {
                        "json": payload,
                        "headers": {
                            "Authorization": f"Bearer {self.user_access_token}",
                            "Host": "llm.chutes.ai",
                            "Content-Type": "application/json",
                        },
                    }
                    if timeout is not None:
                        request_kwargs["timeout"] = timeout
                    response = await client.post(
                        f"{IDP_INFERENCE_URL}/chat/completions",
                        **request_kwargs,
                    )
            else:
                # Use standard API with API key
                client = await self._get_client()
                request_kwargs = {"json": payload}
                if timeout is not None:
                    request_kwargs["timeout"] = timeout
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    **request_kwargs,
                )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            detail = str(exc) or repr(exc)
            raise InferenceNetworkError(
                f"Network error contacting Chutes ({exc.__class__.__name__}): {detail}"
            ) from exc

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
        if isinstance(response.get("error"), dict):
            error_payload = response.get("error") or {}
            error_message = error_payload.get("message") or str(error_payload)
            metadata["response_error"] = f"Chutes error: {error_message}"
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
        requested_max_tokens = kwargs.get("max_tokens")
        applied_max_tokens = requested_max_tokens
        if "timeout" not in kwargs:
            kwargs["timeout"] = settings.chutes_inference_timeout_seconds

        # Initialize defaults
        metadata = {
            "usage": {},
            "model": model_slug,
            "finish_reason": None,
        }

        min_output_tokens = settings.chutes_min_output_tokens
        max_tokens_cap = settings.chutes_max_output_tokens_cap
        max_output_length = await self.get_model_max_output_length(model_slug)
        context_length = await self.get_model_context_length(model_slug)
        safe_max_tokens: Optional[int] = None
        desired_max_tokens = requested_max_tokens
        if max_output_length:
            if desired_max_tokens is None or desired_max_tokens < min_output_tokens:
                desired_max_tokens = min_output_tokens

        input_tokens = self._estimate_tokens(prompt)
        if system_prompt:
            input_tokens += self._estimate_tokens(system_prompt)
        if context_length and input_tokens >= context_length:
            metadata["response_error"] = (
                f"Prompt length {input_tokens} exceeds model context length {context_length}"
            )
            metadata["context_length"] = context_length
            metadata["input_tokens_estimate"] = input_tokens
            return "", metadata

        if max_output_length:
            safe_max_tokens = self._compute_safe_max_tokens(max_output_length, input_tokens)
            if max_tokens_cap:
                safe_max_tokens = min(safe_max_tokens, max_tokens_cap)
            metadata["max_output_length"] = max_output_length
            if context_length:
                metadata["context_length"] = context_length
            if safe_max_tokens > 0 and desired_max_tokens is not None:
                applied_max_tokens = min(desired_max_tokens, safe_max_tokens)
            else:
                applied_max_tokens = safe_max_tokens or desired_max_tokens
        else:
            applied_max_tokens = desired_max_tokens

        if max_tokens_cap and applied_max_tokens is not None:
            applied_max_tokens = min(applied_max_tokens, max_tokens_cap)

        if applied_max_tokens is not None:
            kwargs["max_tokens"] = applied_max_tokens
            metadata["max_tokens"] = applied_max_tokens
            if applied_max_tokens != requested_max_tokens:
                logger.debug(
                    "Adjusting max_tokens for model",
                    model=model_slug,
                    requested=requested_max_tokens,
                    applied=applied_max_tokens,
                    max_output_length=max_output_length,
                )

        while attempt < max_attempts:
            attempt += 1
            try:
                response = await self.run_inference(model_slug, messages, **kwargs)
            except InferenceHTTPError as e:
                if _is_context_length_error(e):
                    metadata["response_error"] = "Prompt exceeds model context length"
                    metadata["response_error_code"] = e.status_code
                    metadata["response_error_detail"] = e.response_text
                    return "", metadata
                if (
                    attempt < max_attempts
                    and _is_max_tokens_error(e)
                    and isinstance(kwargs.get("max_tokens"), int)
                ):
                    current_max = kwargs.get("max_tokens")
                    limit = _extract_max_tokens_limit(e.response_text)
                    if limit and limit > 0:
                        reduced = min(current_max, limit)
                    else:
                        reduced = max(512, int(current_max * 0.5))
                    if reduced < current_max:
                        kwargs["max_tokens"] = reduced
                        applied_max_tokens = reduced
                        metadata["max_tokens"] = reduced
                        logger.warning(
                            "Reducing max_tokens after model limit error",
                            model=model_slug,
                            requested=current_max,
                            applied=reduced,
                        )
                        await asyncio.sleep(delay_seconds)
                        delay_seconds = min(delay_seconds * 2, 10)
                        continue
                logger.error("Inference failed", model=model_slug, error=str(e))
                raise
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

            finish_reason = metadata.get("finish_reason")
            if (
                finish_reason == "length"
                and attempt < max_attempts
                and isinstance(kwargs.get("max_tokens"), int)
            ):
                current_max = kwargs["max_tokens"]
                max_tokens_ceiling = safe_max_tokens
                if max_tokens_cap:
                    if max_tokens_ceiling:
                        max_tokens_ceiling = min(max_tokens_ceiling, max_tokens_cap)
                    else:
                        max_tokens_ceiling = max_tokens_cap
                bumped: Optional[int] = None
                if max_tokens_ceiling and max_tokens_ceiling > current_max:
                    bumped = min(current_max * 2, max_tokens_ceiling)
                elif max_tokens_ceiling is None:
                    bumped = current_max * 2
                if bumped and bumped > current_max:
                    kwargs["max_tokens"] = bumped
                    applied_max_tokens = bumped
                    metadata["max_tokens"] = bumped
                    logger.info(
                        "Retrying after truncation",
                        model=model_slug,
                        previous=current_max,
                        applied=bumped,
                    )
                    await asyncio.sleep(delay_seconds)
                    delay_seconds = min(delay_seconds * 2, 10)
                    continue

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
