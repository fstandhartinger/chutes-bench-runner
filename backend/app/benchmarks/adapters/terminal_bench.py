"""Terminal-Bench Hard benchmark adapter."""
import time
from typing import Any, AsyncIterator, Optional

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.core.logging import get_logger

logger = get_logger(__name__)


@register_adapter("terminal_bench_hard")
class TerminalBenchHardAdapter(BenchmarkAdapter):
    """
    Terminal-Bench Hard adapter.
    
    Challenging terminal/CLI interaction benchmark.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._items: list[dict[str, Any]] = []

    def get_name(self) -> str:
        return "terminal_bench_hard"

    def get_display_name(self) -> str:
        return "Terminal-Bench Hard"

    def requires_setup(self) -> bool:
        return True

    def get_setup_notes(self) -> Optional[str]:
        return "Terminal-Bench requires Docker or isolated shell environment for execution."

    def supports_subset(self) -> bool:
        return True

    async def get_total_items(self) -> int:
        if not self._items:
            await self.preload()
        return len(self._items)

    async def preload(self) -> None:
        """Load Terminal-Bench dataset."""
        if self._items:
            return

        try:
            logger.info("Loading Terminal-Bench Hard dataset")
            # Terminal-Bench may need custom loading
            # Using placeholder items for now
            self._items = [
                {"id": "0", "task": "List all files in current directory sorted by size", "expected_cmd": "ls -lS"},
                {"id": "1", "task": "Find all Python files containing 'import os'", "expected_cmd": "grep -r 'import os' --include='*.py'"},
                {"id": "2", "task": "Count lines in all .txt files", "expected_cmd": "wc -l *.txt"},
                {"id": "3", "task": "Show disk usage of current directory", "expected_cmd": "du -sh ."},
                {"id": "4", "task": "Find processes using port 8080", "expected_cmd": "lsof -i :8080"},
            ]
            logger.info(f"Loaded {len(self._items)} Terminal-Bench Hard items")
        except Exception as e:
            logger.error("Failed to load Terminal-Bench Hard", error=str(e))
            self._items = []

    async def enumerate_items(self) -> AsyncIterator[str]:
        if not self._items:
            await self.preload()
        for item in self._items:
            yield item["id"]

    async def evaluate_item(self, item_id: str) -> ItemResult:
        """Evaluate a single Terminal-Bench item."""
        if not self._items:
            await self.preload()

        item = next((i for i in self._items if i["id"] == item_id), None)
        if not item:
            return ItemResult(item_id=item_id, error=f"Item {item_id} not found")

        prompt = f"""You are a Linux shell expert. Given the task, provide the exact shell command to accomplish it.
Only output the command, nothing else.

Task: {item["task"]}

Command:"""

        try:
            start_time = time.time()
            response_text, metadata = await self.client.get_completion_text(
                self.model_slug,
                prompt,
                system_prompt="You are a Linux shell expert. Output only the command, no explanations.",
                max_tokens=256,
                temperature=0.0,
            )
            latency_ms = int((time.time() - start_time) * 1000)

            response_cmd = response_text.strip().split("\n")[0].strip()
            expected_cmd = item.get("expected_cmd", "")
            
            # Simple command matching (full eval requires execution)
            is_correct = response_cmd.lower() == expected_cmd.lower()

            return ItemResult(
                item_id=item_id,
                item_hash=self.compute_item_hash(item["task"]),
                prompt=prompt,
                response=response_cmd,
                expected=expected_cmd,
                is_correct=is_correct,
                score=1.0 if is_correct else 0.0,
                latency_ms=latency_ms,
                input_tokens=metadata.get("usage", {}).get("prompt_tokens"),
                output_tokens=metadata.get("usage", {}).get("completion_tokens"),
                judge_output={"note": "Full evaluation requires command execution"},
            )

        except Exception as e:
            logger.error("Terminal-Bench evaluation failed", item_id=item_id, error=str(e))
            return ItemResult(item_id=item_id, prompt=prompt, error=str(e))

