"""Terminal-Bench Hard benchmark adapter."""
import time
from typing import Any, AsyncIterator, Optional

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.core.logging import get_logger
from app.services.sandy_service import SandyService

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
        self.sandy = SandyService()

    def get_name(self) -> str:
        return "terminal_bench_hard"

    def get_display_name(self) -> str:
        return "Terminal-Bench Hard"

    def requires_setup(self) -> bool:
        return True

    def get_setup_notes(self) -> Optional[str]:
        return "Terminal-Bench requires a sandbox for command execution."

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
Only output the command within a markdown code block.

Examples:
Task: List all files
Command:
```bash
ls -a
```

Task: Find files larger than 100MB
Command:
```bash
find . -size +100M
```

Task: {item["task"]}
Command:"""

        try:
            start_time = time.time()
            response_text, metadata = await self.client.get_completion_text(
                self.model_slug,
                prompt,
                system_prompt="You are a Linux shell expert. Output ONLY the final shell command within a markdown code block. No explanation, no thinking, no prose.",
                max_tokens=512,
                temperature=0.0,
            )
            latency_ms = int((time.time() - start_time) * 1000)

            # Extract command robustly
            response_cmd = self.extract_python_code(response_text)
            
            # If the extracted "code" still looks like prose (e.g. no command-like symbols or too many words)
            # try to find the last line that actually looks like a command
            if len(response_cmd.split()) > 10 and "```" not in response_text:
                # Heuristic: models sometimes put the command on the last line or after some prose
                lines = [l.strip() for l in response_cmd.split("\n") if l.strip()]
                for line in reversed(lines):
                    # Basic command heuristic: starts with common utility or doesn't look like a sentence
                    if any(line.startswith(cmd) for cmd in ["ls", "find", "grep", "wc", "du", "lsof", "ps", "cat", "echo"]):
                        response_cmd = line
                        break
                else:
                    # If still not found, take the first non-empty line
                    if lines:
                        response_cmd = lines[0]
            
            # Create sandbox and run command
            sandbox_id = await self.sandy.create_sandbox()
            if not sandbox_id:
                return ItemResult(item_id=item_id, prompt=prompt, error="Could not create sandbox")
            
            try:
                # Basic execution check
                execution_result = await self.sandy.execute_command(sandbox_id, response_cmd)
                
                # In a real benchmark, we'd compare output or state with expected
                # For now, we'll use a mix of execution success and string matching
                expected_cmd = item.get("expected_cmd", "")
                is_correct = response_cmd.lower() == expected_cmd.lower() or (execution_result.get("success", False) and execution_result.get("exit_code") == 0)
                
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
                    judge_output={
                        "stdout": execution_result.get("stdout"),
                        "stderr": execution_result.get("stderr"),
                        "exit_code": execution_result.get("exit_code")
                    }
                )
            finally:
                await self.sandy.terminate_sandbox(sandbox_id)

        except Exception as e:
            logger.error("Terminal-Bench evaluation failed", item_id=item_id, error=str(e))
            return ItemResult(item_id=item_id, prompt=prompt, error=str(e))






