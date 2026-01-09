"""Tests for AA-Omniscience adapter helpers."""
import pytest

from app.benchmarks.adapters.aa_omniscience import AAOmniscienceAdapter
from app.benchmarks.base import ItemResult
from app.services.chutes_client import ChutesClient


def test_grade_letter_parsing():
    adapter = AAOmniscienceAdapter(ChutesClient(api_key="test"), "test-model")
    assert adapter._parse_grade_letter("A") == "A"
    assert adapter._parse_grade_letter("B: INCORRECT") == "B"
    assert adapter._parse_grade_letter("PARTIAL ANSWER") == "C"
    assert adapter._parse_grade_letter("NOT ATTEMPTED") == "D"


@pytest.mark.asyncio
async def test_postprocess_omniscience_index():
    adapter = AAOmniscienceAdapter(ChutesClient(api_key="test"), "test-model")
    results = [
        ItemResult(item_id="1", metadata={"grade": "CORRECT"}),
        ItemResult(item_id="2", metadata={"grade": "INCORRECT"}),
        ItemResult(item_id="3", metadata={"grade": "PARTIAL ANSWER"}),
        ItemResult(item_id="4", metadata={"grade": "NOT ATTEMPTED"}),
    ]
    metrics = await adapter.postprocess(results)
    assert metrics["grade_correct"] == 1
    assert metrics["grade_incorrect"] == 1
    assert metrics["grade_partial"] == 1
    assert metrics["grade_abstain"] == 1
    assert metrics["score_override"] == pytest.approx(0.0)
