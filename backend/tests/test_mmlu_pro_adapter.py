import asyncio

from app.benchmarks.adapters.mmlu_pro import MMLUProAdapter


def test_mmlu_pro_few_shot_uses_cot_content() -> None:
    adapter = MMLUProAdapter(None, "mock")
    adapter._items = [
        {
            "id": "0",
            "question": "Test question?",
            "options": ["A1", "B1", "C1", "D1"],
            "answer": "B",
            "category": "science",
        }
    ]
    adapter._few_shot_by_category = {
        "science": [
            {
                "question": "Example question?",
                "options": ["A", "B", "C", "D"],
                "answer": "C",
                "cot_content": "A: Let's think step by step. The answer is (C).",
                "category": "science",
            }
        ]
    }

    class FakeClient:
        async def get_completion_text(self, *args, **kwargs):
            prompt = args[1]
            assert "Let's think step by step" in prompt
            return "Answer: B", {"usage": {"prompt_tokens": 1, "completion_tokens": 1}}

    adapter.client = FakeClient()
    result = asyncio.run(adapter.evaluate_item("0"))
    assert result.is_correct is True
