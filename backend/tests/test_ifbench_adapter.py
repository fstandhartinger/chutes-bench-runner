import asyncio

from app.benchmarks.adapters.ifbench import IFBenchAdapter
from app.benchmarks.base import ItemResult


def test_ifbench_postprocess_reports_strict_and_loose_metrics() -> None:
    adapter = IFBenchAdapter(None, "mock")
    results = [
        ItemResult(
            item_id="1",
            judge_output={
                "strict_follow_all_instructions": True,
                "strict_follow_instruction_list": [True, True],
                "loose_follow_all_instructions": True,
                "loose_follow_instruction_list": [True, True],
            },
        ),
        ItemResult(
            item_id="2",
            judge_output={
                "strict_follow_all_instructions": False,
                "strict_follow_instruction_list": [True, False],
                "loose_follow_all_instructions": True,
                "loose_follow_instruction_list": [True, True],
            },
        ),
    ]

    metrics = asyncio.run(adapter.postprocess(results))

    assert metrics["strict_prompt_accuracy"] == 0.5
    assert metrics["strict_instruction_accuracy"] == 0.75
    assert metrics["loose_prompt_accuracy"] == 1.0
    assert metrics["loose_instruction_accuracy"] == 1.0
