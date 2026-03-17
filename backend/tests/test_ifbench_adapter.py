from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.benchmarks.adapters.ifbench import IFBENCH_DATASET, IFBENCH_REPEATS, IFBenchAdapter
from app.benchmarks.ifbench_eval import evaluation_lib
from app.benchmarks.ifbench_eval.instructions_registry import INSTRUCTION_DICT


class DummyClient:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.get_completion_text = AsyncMock(side_effect=self._next_response)
        self.get_model_max_output_length = AsyncMock(return_value=32768)

    async def _next_response(self, *args, **kwargs):
        response = self._responses.pop(0)
        return response, {
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            "finish_reason": "stop",
            "response_attempts": 1,
        }


@pytest.mark.asyncio
async def test_ifbench_preload_uses_allenai_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    async def fake_load_dataset(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return [
            {
                "key": "abc",
                "prompt": "Prompt",
                "instruction_id_list": ["count:keywords_multiple"],
                "kwargs": [{"keyword1": "alpha"}],
            }
        ]

    monkeypatch.setattr(
        "app.benchmarks.adapters.ifbench.load_dataset_with_retry",
        fake_load_dataset,
    )

    adapter = IFBenchAdapter(DummyClient(["alpha"]), "mock-model")
    await adapter.preload()

    assert seen["args"] == (IFBENCH_DATASET,)
    assert seen["kwargs"] == {"split": "train", "token": None}
    assert adapter._items[0]["id"] == "abc"


@pytest.mark.asyncio
async def test_ifbench_evaluate_item_uses_loose_prompt_accuracy_across_five_repeats(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = DummyClient(
        [
            "resp-1",
            "resp-2",
            "resp-3",
            "resp-4",
            "resp-5",
        ]
    )
    adapter = IFBenchAdapter(client, "mock-model")
    adapter._items = [
        {
            "id": "7",
            "prompt": "Test prompt",
            "instruction_id_list": ["count:keywords_multiple"],
            "kwargs": [{"keyword1": "alpha"}],
        }
    ]

    strict_outputs = iter(
        [
            SimpleNamespace(follow_all_instructions=True, follow_instruction_list=[True, True]),
            SimpleNamespace(follow_all_instructions=False, follow_instruction_list=[True, False]),
            SimpleNamespace(follow_all_instructions=False, follow_instruction_list=[False, False]),
            SimpleNamespace(follow_all_instructions=True, follow_instruction_list=[True, True]),
            SimpleNamespace(follow_all_instructions=True, follow_instruction_list=[True, True]),
        ]
    )
    loose_outputs = iter(
        [
            SimpleNamespace(follow_all_instructions=True, follow_instruction_list=[True, True]),
            SimpleNamespace(follow_all_instructions=True, follow_instruction_list=[True, True]),
            SimpleNamespace(follow_all_instructions=False, follow_instruction_list=[False, True]),
            SimpleNamespace(follow_all_instructions=True, follow_instruction_list=[True, True]),
            SimpleNamespace(follow_all_instructions=True, follow_instruction_list=[True, True]),
        ]
    )
    monkeypatch.setattr(
        evaluation_lib,
        "test_instruction_following_strict",
        lambda *args, **kwargs: next(strict_outputs),
    )
    monkeypatch.setattr(
        evaluation_lib,
        "test_instruction_following_loose",
        lambda *args, **kwargs: next(loose_outputs),
    )

    result = await adapter.evaluate_item("7")

    assert result.error is None
    assert result.is_correct is True
    assert result.score == pytest.approx(0.8)
    assert result.input_tokens == 55
    assert result.output_tokens == 35
    assert result.judge_output["strict_prompt_accuracy"] == pytest.approx(0.6)
    assert result.judge_output["loose_prompt_accuracy"] == pytest.approx(0.8)
    assert result.judge_output["loose_instruction_accuracy"] == pytest.approx(0.9)
    assert client.get_completion_text.await_count == IFBENCH_REPEATS
    _, kwargs = client.get_completion_text.await_args
    assert kwargs["temperature"] == 0.0
    assert kwargs["max_tokens"] == 16384


@pytest.mark.asyncio
async def test_ifbench_postprocess_uses_repeat_averaged_loose_prompt_accuracy() -> None:
    adapter = IFBenchAdapter(DummyClient(["alpha"] * IFBENCH_REPEATS), "mock-model")

    metrics = await adapter.postprocess(
        [
            SimpleNamespace(
                error=None,
                score=0.8,
                judge_output={
                    "strict_prompt_accuracy": 0.6,
                    "loose_prompt_accuracy": 0.8,
                    "strict_instruction_followed_count": 8,
                    "strict_instruction_total": 10,
                    "loose_instruction_followed_count": 9,
                    "loose_instruction_total": 10,
                },
            ),
            SimpleNamespace(
                error=None,
                score=0.4,
                judge_output={
                    "strict_prompt_accuracy": 0.2,
                    "loose_prompt_accuracy": 0.4,
                    "strict_instruction_followed_count": 4,
                    "strict_instruction_total": 10,
                    "loose_instruction_followed_count": 6,
                    "loose_instruction_total": 10,
                },
            ),
        ]
    )

    assert metrics["ifbench_dataset"] == IFBENCH_DATASET
    assert metrics["ifbench_repeats"] == IFBENCH_REPEATS
    assert metrics["loose_prompt_accuracy"] == pytest.approx(0.6)
    assert metrics["strict_prompt_accuracy"] == pytest.approx(0.4)
    assert metrics["loose_instruction_accuracy"] == pytest.approx(0.75)
    assert metrics["strict_instruction_accuracy"] == pytest.approx(0.6)
    assert metrics["accuracy_override"] == pytest.approx(0.6)
    assert metrics["correct_count_override"] == pytest.approx(1.2)
    assert metrics["score_override"] == pytest.approx(0.6)


def test_ifbench_uses_allenai_instruction_registry() -> None:
    assert "count:keywords_multiple" in INSTRUCTION_DICT
