"""One-token provider checks run before benchmark items are attempted."""
from __future__ import annotations

from typing import Any

from app.services.inference_client import InferenceClient


class ProviderPreflightError(RuntimeError):
    """The provider did not prove both inference and token accounting work."""


def _usage_tokens(usage: dict[str, Any]) -> tuple[Any, Any]:
    input_tokens = usage.get("prompt_tokens")
    if input_tokens is None:
        input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("completion_tokens")
    if output_tokens is None:
        output_tokens = usage.get("output_tokens")
    return input_tokens, output_tokens


async def preflight_provider(
    client: InferenceClient,
    model_slug: str,
    *,
    timeout_seconds: float = 20.0,
) -> dict[str, Any]:
    """Require a real one-token answer and provider-reported token counts."""
    response = await client.run_inference(
        model_slug,
        [{"role": "user", "content": "Reply with Y."}],
        temperature=0.0,
        max_tokens=1,
        timeout=timeout_seconds,
    )
    if not isinstance(response, dict):
        raise ProviderPreflightError("provider returned a non-object response")
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ProviderPreflightError("provider response contained no choices")
    usage = response.get("usage")
    if not isinstance(usage, dict):
        raise ProviderPreflightError("provider response contained no usage object")
    input_tokens, output_tokens = _usage_tokens(usage)
    if not isinstance(input_tokens, int) or input_tokens <= 0:
        raise ProviderPreflightError("provider response contained no input token count")
    if not isinstance(output_tokens, int) or output_tokens <= 0:
        raise ProviderPreflightError("provider response contained no output token count")
    return {
        "model": response.get("model") or model_slug,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "finish_reason": choices[0].get("finish_reason")
        if isinstance(choices[0], dict)
        else None,
    }
