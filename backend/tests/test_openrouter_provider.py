"""OpenRouter catalog, preflight, and Sandy agent configuration tests."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock

import httpx
import pytest

from app.benchmarks.agent_provider_config import build_openrouter_agent_setup
from app.services.openrouter_client import OpenRouterClient
from app.services.provider_preflight import ProviderPreflightError, preflight_provider

MODEL = "deepseek/deepseek-v4-flash-0731"


@pytest.mark.asyncio
async def test_openrouter_catalog_fields_are_normalized(monkeypatch) -> None:
    client = OpenRouterClient(api_key="test-openrouter-key")
    monkeypatch.setattr(
        client,
        "_fetch_catalog",
        AsyncMock(
            return_value=[
                {
                    "id": MODEL,
                    "name": "DeepSeek: DeepSeek V4 Flash 0731",
                    "context_length": 1_048_576,
                    "top_provider": {"max_completion_tokens": 65_536},
                    "pricing": {
                        "prompt": "0.00000009",
                        "completion": "0.00000018",
                    },
                }
            ]
        ),
    )

    assert await client.get_model_context_length(MODEL) == 1_048_576
    assert await client.get_model_max_output_length(MODEL) == 65_536
    assert await client.get_model_pricing(MODEL) == pytest.approx((0.09, 0.18))
    assert (await client.list_models())[0]["slug"] == MODEL


@pytest.mark.asyncio
async def test_real_shaped_openrouter_chat_usage_survives_preflight() -> None:
    response_payload = {
        "id": "gen-test",
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": None, "reasoning": "Y"},
                "finish_reason": "length",
            }
        ],
        "usage": {
            "prompt_tokens": 87,
            "completion_tokens": 1,
            "total_tokens": 88,
            "cost": 0.0000123354,
            "completion_tokens_details": {"reasoning_tokens": 1},
        },
    }

    async def respond(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/chat/completions"
        assert json.loads(request.content)["max_tokens"] == 1
        return httpx.Response(200, json=response_payload)

    client = OpenRouterClient(api_key="test-openrouter-key")
    client._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        result = await preflight_provider(client, MODEL)
    finally:
        await client.close()

    assert result == {
        "model": MODEL,
        "input_tokens": 87,
        "output_tokens": 1,
        "finish_reason": "length",
    }


@pytest.mark.asyncio
async def test_preflight_rejects_response_without_exact_usage() -> None:
    client = AsyncMock()
    client.run_inference.return_value = {
        "choices": [{"message": {"content": "Y"}, "finish_reason": "stop"}],
    }
    with pytest.raises(ProviderPreflightError, match="no usage object"):
        await preflight_provider(client, MODEL)


@pytest.mark.asyncio
async def test_preflight_accepts_responses_style_usage_names() -> None:
    client = AsyncMock()
    client.run_inference.return_value = {
        "model": MODEL,
        "choices": [{"message": {"reasoning": "Y"}, "finish_reason": "length"}],
        "usage": {"input_tokens": 87, "output_tokens": 1, "total_tokens": 88},
    }

    result = await preflight_provider(client, MODEL)

    assert result["input_tokens"] == 87
    assert result["output_tokens"] == 1


@pytest.mark.parametrize(
    ("agent", "expected_rlm"),
    [
        ("codex", None),
        ("chutescoder-baseline", False),
        ("chutescoder", True),
    ],
)
def test_openrouter_agent_config_is_secret_free(agent, expected_rlm) -> None:
    secret = "openrouter-secret-that-must-not-be-written"
    setup = build_openrouter_agent_setup(
        agent=agent,
        model=MODEL,
        api_base_url="https://openrouter.ai/api/v1",
        api_key=secret,
        context_window=1_048_576,
        max_output_tokens=65_536,
    )

    assert secret not in setup.config_toml
    assert secret not in setup.model_catalog_json
    assert secret not in setup.install_command()
    assert setup.env_vars["OPENROUTER_API_KEY"] == secret
    assert 'wire_api = "responses"' in setup.config_toml
    assert "model_context_window = 1048576" in setup.config_toml
    catalog = json.loads(setup.model_catalog_json)
    assert catalog["models"][0]["max_context_window"] == 1_048_576
    if expected_rlm is None:
        assert "[rlm]" not in setup.config_toml
    else:
        assert f"enabled = {str(expected_rlm).lower()}" in setup.config_toml


def test_openrouter_rejects_unsupported_sandy_agent() -> None:
    with pytest.raises(ValueError, match="support these Sandy agents"):
        build_openrouter_agent_setup(
            agent="claude-code",
            model=MODEL,
            api_base_url="https://openrouter.ai/api/v1",
            api_key="test",
            context_window=1_048_576,
            max_output_tokens=65_536,
        )
