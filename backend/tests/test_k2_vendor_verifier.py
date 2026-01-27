"""Tests for K2 Vendor Verifier metrics."""
import pytest

from app.benchmarks.adapters.kimi_vendor_verifier import (
    KimiVendorVerifierAdapter,
    compute_tool_call_metrics,
)
from app.benchmarks.base import ItemResult


class DummyClient:
    provider = "dummy"


@pytest.mark.asyncio
async def test_k2vv_metrics_summary():
    results = [
        ItemResult(
            item_id="1",
            metadata={
                "candidate_finish_reason": "tool_calls",
                "reference_finish_reason": "tool_calls",
                "tool_calls_valid": True,
            },
        ),
        ItemResult(
            item_id="2",
            metadata={
                "candidate_finish_reason": "tool_calls",
                "reference_finish_reason": "stop",
                "tool_calls_valid": False,
            },
        ),
        ItemResult(
            item_id="3",
            metadata={
                "candidate_finish_reason": "stop",
                "reference_finish_reason": "tool_calls",
                "tool_calls_valid": None,
            },
        ),
        ItemResult(
            item_id="4",
            metadata={
                "candidate_finish_reason": "stop",
                "reference_finish_reason": "stop",
                "tool_calls_valid": None,
            },
        ),
    ]

    metrics = compute_tool_call_metrics(results)
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tn"] == 1
    assert metrics["tool_call_precision"] == pytest.approx(0.5)
    assert metrics["tool_call_recall"] == pytest.approx(0.5)
    assert metrics["tool_call_f1"] == pytest.approx(0.5)
    assert metrics["count_finish_reason_tool_calls"] == 2
    assert metrics["count_successful_tool_call"] == 1
    assert metrics["schema_accuracy"] == pytest.approx(0.5)

    adapter = KimiVendorVerifierAdapter(DummyClient(), "test-model")
    post = await adapter.postprocess(results)
    assert post["score_override"] == pytest.approx(0.5)
