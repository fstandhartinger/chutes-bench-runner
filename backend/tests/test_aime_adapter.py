from unittest.mock import AsyncMock

import pytest

from app.benchmarks.adapters.aime import AIME2025Adapter


class DummyClient:
    def __init__(self, response_text: str):
        self.get_completion_text = AsyncMock(
            return_value=(
                response_text,
                {
                    "usage": {"prompt_tokens": 12, "completion_tokens": 4},
                    "finish_reason": "stop",
                },
            )
        )


@pytest.mark.asyncio
async def test_aime_evaluate_item_is_single_pass_exact_match() -> None:
    client = DummyClient("ANSWER: 123")
    adapter = AIME2025Adapter(client, "test-model")
    adapter._items = [{"id": "0", "problem": "demo", "answer": "123", "level": "AIME"}]

    result = await adapter.evaluate_item("0")

    assert result.is_correct is True
    assert result.score == 1.0
    assert result.error is None
    assert client.get_completion_text.await_count == 1
    _, kwargs = client.get_completion_text.await_args
    assert kwargs["max_tokens"] == 64
    assert kwargs["min_output_tokens"] == 0
    assert kwargs["temperature"] == 0.0


@pytest.mark.asyncio
async def test_aime_postprocess_uses_single_run_score() -> None:
    adapter = AIME2025Adapter(DummyClient("ANSWER: 1"), "test-model")

    metrics = await adapter.postprocess([])

    assert metrics["aime_runs"] == 1
    assert metrics["aime_temperatures"] == [0.0]
