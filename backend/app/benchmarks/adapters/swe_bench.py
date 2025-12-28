"""SWE-Bench Pro benchmark adapter."""
import time
from typing import Any, AsyncIterator, Optional

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.core.logging import get_logger

logger = get_logger(__name__)


@register_adapter("swe_bench_pro")
class SWEBenchProAdapter(BenchmarkAdapter):
    """
    SWE-Bench Pro adapter.
    
    Software engineering benchmark with real GitHub issues.
    https://github.com/scaleapi/SWE-bench_Pro-os
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._items: list[dict[str, Any]] = []

    def get_name(self) -> str:
        return "swe_bench_pro"

    def get_display_name(self) -> str:
        return "SWE-Bench Pro"

    def requires_setup(self) -> bool:
        return True

    def get_setup_notes(self) -> Optional[str]:
        return (
            "SWE-Bench Pro requires:\n"
            "1. Docker for isolated execution\n"
            "2. Git credentials for cloning repos\n"
            "3. The official SWE-Bench harness\n"
            "See: https://github.com/scaleapi/SWE-bench_Pro-os"
        )

    def supports_subset(self) -> bool:
        return True

    async def get_total_items(self) -> int:
        if not self._items:
            await self.preload()
        return len(self._items)

    async def preload(self) -> None:
        """Load SWE-Bench Pro dataset."""
        if self._items:
            return

        try:
            from datasets import load_dataset

            logger.info("Loading SWE-Bench Pro dataset")
            # SWE-bench dataset
            try:
                dataset = load_dataset("princeton-nlp/SWE-bench", split="test")
            except Exception:
                # Try lite version
                dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
            
            self._items = []
            for i, item in enumerate(dataset):
                self._items.append({
                    "id": str(i),
                    "instance_id": item.get("instance_id", ""),
                    "repo": item.get("repo", ""),
                    "problem_statement": item.get("problem_statement", ""),
                    "hints_text": item.get("hints_text", ""),
                    "patch": item.get("patch", ""),
                })
            
            logger.info(f"Loaded {len(self._items)} SWE-Bench Pro items")
        except Exception as e:
            logger.error("Failed to load SWE-Bench Pro", error=str(e))
            self._items = []

    async def enumerate_items(self) -> AsyncIterator[str]:
        if not self._items:
            await self.preload()
        for item in self._items:
            yield item["id"]

    async def evaluate_item(self, item_id: str) -> ItemResult:
        """Evaluate a single SWE-Bench item."""
        if not self._items:
            await self.preload()

        item = next((i for i in self._items if i["id"] == item_id), None)
        if not item:
            return ItemResult(item_id=item_id, error=f"Item {item_id} not found")

        hints = item.get("hints_text", "")
        prompt = f"""You are a software engineer fixing a bug in a GitHub repository.

Repository: {item.get("repo", "unknown")}

Issue Description:
{item["problem_statement"]}

{f"Hints: {hints}" if hints else ""}

Provide a git diff patch that fixes this issue. Format your response as a unified diff.

```diff
"""

        try:
            start_time = time.time()
            response_text, metadata = await self.client.get_completion_text(
                self.model_slug,
                prompt,
                system_prompt="You are an expert software engineer. Generate precise git patches to fix bugs.",
                max_tokens=4096,
                temperature=0.0,
            )
            latency_ms = int((time.time() - start_time) * 1000)

            # Full evaluation requires applying patch and running tests
            return ItemResult(
                item_id=item_id,
                item_hash=self.compute_item_hash(item["problem_statement"]),
                prompt=prompt,
                response=response_text.strip(),
                expected="[Patch execution required]",
                is_correct=None,
                score=None,
                latency_ms=latency_ms,
                input_tokens=metadata.get("usage", {}).get("prompt_tokens"),
                output_tokens=metadata.get("usage", {}).get("completion_tokens"),
                metadata={
                    "instance_id": item.get("instance_id"),
                    "repo": item.get("repo"),
                },
                judge_output={"note": "Full evaluation requires SWE-Bench harness"},
            )

        except Exception as e:
            logger.error("SWE-Bench evaluation failed", item_id=item_id, error=str(e))
            return ItemResult(item_id=item_id, prompt=prompt, error=str(e))


