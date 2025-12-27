"""LiveCodeBench benchmark adapter."""
import time
from typing import Any, AsyncIterator, Optional

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.core.logging import get_logger

logger = get_logger(__name__)


@register_adapter("livecodebench")
class LiveCodeBenchAdapter(BenchmarkAdapter):
    """
    LiveCodeBench adapter.
    
    A coding benchmark with live programming problems.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._items: list[dict[str, Any]] = []

    def get_name(self) -> str:
        return "livecodebench"

    def get_display_name(self) -> str:
        return "LiveCodeBench"

    def requires_setup(self) -> bool:
        return True

    def get_setup_notes(self) -> Optional[str]:
        return "LiveCodeBench requires local code execution environment for verification."

    async def get_total_items(self) -> int:
        if not self._items:
            await self.preload()
        return len(self._items)

    async def preload(self) -> None:
        """Load LiveCodeBench dataset."""
        if self._items:
            return

        try:
            from datasets import load_dataset

            logger.info("Loading LiveCodeBench dataset")
            dataset = load_dataset("livecodebench/code_generation_lite", split="test")
            
            self._items = []
            for i, item in enumerate(dataset):
                self._items.append({
                    "id": str(i),
                    "question": item.get("question_content", ""),
                    "starter_code": item.get("starter_code", ""),
                    "difficulty": item.get("difficulty", ""),
                    "test_cases": item.get("test", ""),
                })
            
            logger.info(f"Loaded {len(self._items)} LiveCodeBench items")
        except Exception as e:
            logger.error("Failed to load LiveCodeBench", error=str(e))
            self._items = []

    async def enumerate_items(self) -> AsyncIterator[str]:
        if not self._items:
            await self.preload()
        for item in self._items:
            yield item["id"]

    async def evaluate_item(self, item_id: str) -> ItemResult:
        """Evaluate a single LiveCodeBench item."""
        if not self._items:
            await self.preload()

        item = next((i for i in self._items if i["id"] == item_id), None)
        if not item:
            return ItemResult(item_id=item_id, error=f"Item {item_id} not found")

        starter = item.get("starter_code", "")
        starter_section = f"Starter code:\n{starter}" if starter else ""
        prompt = f"""Solve the following coding problem. Provide a complete Python solution.

Problem:
{item["question"]}

{starter_section}

Solution:
```python
"""

        try:
            start_time = time.time()
            response_text, metadata = await self.client.get_completion_text(
                self.model_slug,
                prompt,
                system_prompt="You are an expert programmer. Write clean, correct Python code.",
                max_tokens=2048,
                temperature=0.0,
            )
            latency_ms = int((time.time() - start_time) * 1000)

            # Note: Full evaluation requires code execution
            # For now, we just capture the response
            return ItemResult(
                item_id=item_id,
                item_hash=self.compute_item_hash(item["question"]),
                prompt=prompt,
                response=response_text.strip(),
                expected="[Code execution required]",
                is_correct=None,  # Cannot determine without execution
                score=None,
                latency_ms=latency_ms,
                input_tokens=metadata.get("usage", {}).get("prompt_tokens"),
                output_tokens=metadata.get("usage", {}).get("completion_tokens"),
                metadata={"difficulty": item.get("difficulty")},
                judge_output={"note": "Code execution required for scoring"},
            )

        except Exception as e:
            logger.error("LiveCodeBench evaluation failed", item_id=item_id, error=str(e))
            return ItemResult(item_id=item_id, prompt=prompt, error=str(e))

