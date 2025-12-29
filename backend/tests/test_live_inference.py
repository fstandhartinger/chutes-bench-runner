"""Live inference tests against Chutes models (opt-in)."""
import os
import pytest

from app.services.chutes_client import get_chutes_client
from app.benchmarks.adapters.hle import HLEAdapter
from app.benchmarks.adapters.mmlu_pro import MMLUProAdapter

LIVE_TESTS_ENABLED = os.getenv("RUN_LIVE_TESTS") == "1"
MODEL_SLUG = os.getenv("LIVE_MODEL_SLUG", "zai-org/GLM-4.7-TEE")


def _get_api_key() -> str | None:
    return os.getenv("LIVE_CHUTES_API_KEY")


@pytest.mark.skipif(not LIVE_TESTS_ENABLED, reason="Set RUN_LIVE_TESTS=1 to run live inference tests.")
@pytest.mark.asyncio
async def test_live_hle_responses_not_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    api_key = _get_api_key()
    if not api_key:
        pytest.skip("LIVE_CHUTES_API_KEY not set")

    hf_token = os.getenv("LIVE_HF_TOKEN")
    if hf_token:
        monkeypatch.setenv("HF_TOKEN", hf_token)

    client = get_chutes_client(api_key=api_key)
    adapter = HLEAdapter(client, MODEL_SLUG)
    await adapter.preload()

    if len(adapter._items) <= 3:
        pytest.skip("HLE dataset not available; set HF_TOKEN with access to cais/hle")

    ids = []
    async for item_id in adapter.enumerate_items():
        ids.append(item_id)
        if len(ids) >= 3:
            break

    results = [await adapter.evaluate_item(item_id) for item_id in ids]
    assert all(r.response for r in results)
    assert all(r.error is None for r in results)


@pytest.mark.skipif(not LIVE_TESTS_ENABLED, reason="Set RUN_LIVE_TESTS=1 to run live inference tests.")
@pytest.mark.asyncio
async def test_live_mmlu_responses_not_empty() -> None:
    api_key = _get_api_key()
    if not api_key:
        pytest.skip("LIVE_CHUTES_API_KEY not set")

    client = get_chutes_client(api_key=api_key)
    adapter = MMLUProAdapter(client, MODEL_SLUG)
    await adapter.preload()

    ids = []
    async for item_id in adapter.enumerate_items():
        ids.append(item_id)
        if len(ids) >= 3:
            break

    results = [await adapter.evaluate_item(item_id) for item_id in ids]
    assert all(r.response for r in results)
    assert all(r.error is None for r in results)
