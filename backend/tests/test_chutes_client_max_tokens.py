"""Tests for ChutesClient max_tokens handling."""

import pytest

from app.services.chutes_client import ChutesClient


@pytest.mark.asyncio
async def test_get_completion_messages_respects_explicit_max_tokens(monkeypatch) -> None:
    client = ChutesClient(api_key="test")
    seen: dict[str, int] = {}

    async def fake_run_inference(model_slug: str, messages, **kwargs):  # type: ignore[no-untyped-def]
        seen["max_tokens"] = int(kwargs.get("max_tokens") or 0)
        return {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "model": model_slug,
        }

    async def fake_max_output_length(_model_slug: str) -> int:
        return 32000

    async def fake_context_length(_model_slug: str):
        return None

    monkeypatch.setattr(client, "run_inference", fake_run_inference)
    monkeypatch.setattr(client, "get_model_max_output_length", fake_max_output_length)
    monkeypatch.setattr(client, "get_model_context_length", fake_context_length)

    text, _meta = await client.get_completion_messages(
        "test-model",
        [{"role": "user", "content": "hi"}],
        max_tokens=2048,
    )
    assert text == "ok"
    assert seen["max_tokens"] == 2048


@pytest.mark.asyncio
async def test_get_completion_text_respects_explicit_max_tokens(monkeypatch) -> None:
    client = ChutesClient(api_key="test")
    seen: dict[str, int] = {}

    async def fake_run_inference(model_slug: str, messages, **kwargs):  # type: ignore[no-untyped-def]
        seen["max_tokens"] = int(kwargs.get("max_tokens") or 0)
        return {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "model": model_slug,
        }

    async def fake_max_output_length(_model_slug: str) -> int:
        return 32000

    async def fake_context_length(_model_slug: str):
        return None

    monkeypatch.setattr(client, "run_inference", fake_run_inference)
    monkeypatch.setattr(client, "get_model_max_output_length", fake_max_output_length)
    monkeypatch.setattr(client, "get_model_context_length", fake_context_length)

    text, _meta = await client.get_completion_text(
        "test-model",
        "hi",
        max_tokens=2048,
    )
    assert text == "ok"
    assert seen["max_tokens"] == 2048


@pytest.mark.asyncio
async def test_min_output_tokens_is_only_used_when_max_tokens_is_missing(monkeypatch) -> None:
    client = ChutesClient(api_key="test")
    seen: dict[str, int] = {}

    async def fake_run_inference(model_slug: str, messages, **kwargs):  # type: ignore[no-untyped-def]
        seen["max_tokens"] = int(kwargs.get("max_tokens") or 0)
        return {
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "model": model_slug,
        }

    async def fake_max_output_length(_model_slug: str) -> int:
        return 32000

    async def fake_context_length(_model_slug: str):
        return None

    monkeypatch.setattr(client, "run_inference", fake_run_inference)
    monkeypatch.setattr(client, "get_model_max_output_length", fake_max_output_length)
    monkeypatch.setattr(client, "get_model_context_length", fake_context_length)

    # No max_tokens provided: should fall back to min_output_tokens.
    await client.get_completion_messages(
        "test-model",
        [{"role": "user", "content": "hi"}],
        min_output_tokens=1234,
    )
    assert seen["max_tokens"] == 1234

