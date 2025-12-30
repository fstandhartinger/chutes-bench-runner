"""Terminal-Bench Hard benchmark adapter."""
import re
import shlex
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

        def looks_like_command(candidate: str) -> bool:
            stripped = candidate.strip()
            if not stripped:
                return False
            if stripped.startswith("sudo "):
                stripped = stripped[5:].lstrip()
            if stripped.startswith("./"):
                return True
            first = stripped.split()[0]
            common = {
                "ls", "find", "grep", "wc", "du", "lsof", "ps", "cat", "echo", "awk", "sed",
                "head", "tail", "sort", "uniq", "chmod", "chown", "mkdir", "rm", "mv", "cp",
                "tar", "zip", "curl", "wget", "git", "python", "pip", "node", "npm", "yarn",
                "bash", "sh",
            }
            if first in common:
                return True
            if "/" in first:
                return True
            return False

        def normalize_candidate(candidate: str) -> str:
            cleaned = candidate.strip().strip("`")
            cleaned = re.sub(r"^(?:\d+\.|\d+\)|[-*])\s*", "", cleaned)
            cleaned = re.sub(r"(?i)^(?:command|cmd|shell)\s*[:\-]\s*", "", cleaned)
            if cleaned.count("`") >= 2:
                match = re.search(r"`([^`]+)`", cleaned)
                if match:
                    cleaned = match.group(1)
            elif "`" in cleaned:
                cleaned = cleaned.split("`", 1)[0]
            cleaned = cleaned.rstrip(" .,:;")
            return cleaned.strip()

        def extract_command(response_text: str) -> str:
            cleaned = response_text or ""
            cleaned = re.sub(r"(?i)<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()

            candidates: list[tuple[str, int]] = []

            def add_candidate(raw: str, weight: int) -> None:
                candidate = normalize_candidate(raw)
                if candidate:
                    candidates.append((candidate, weight))

            # Markdown code block or raw code extraction.
            extracted = self.extract_python_code(cleaned)
            if extracted:
                weight = 3 if "```" in cleaned else 1
                for line in extracted.splitlines():
                    if line.strip():
                        add_candidate(line, weight)

            # Inline code blocks.
            for match in re.findall(r"`([^`]+)`", cleaned):
                add_candidate(match, 2)

            # "Command:" or similar labels.
            for match in re.findall(r"(?i)(?:command|run|use|execute|try)\s*[:\-]\s*([^\n]+)", cleaned):
                add_candidate(match, 2)

            # Raw lines as fallback candidates.
            for line in cleaned.splitlines():
                if line.strip():
                    add_candidate(line, 1)

            best_candidate = ""
            best_score = -1
            for idx, (candidate, weight) in enumerate(candidates):
                if not looks_like_command(candidate):
                    continue
                tokens = candidate.split()
                score = weight * 10
                if len(tokens) > 1:
                    score += 4
                if any(ch in candidate for ch in ("|", ">", "<", "--", "-")):
                    score += 2
                score += min(len(candidate), 80) // 4
                score += idx / 1000
                if score > best_score:
                    best_score = score
                    best_candidate = candidate

            return best_candidate

        system_prompt = (
            "You are a Linux shell expert. Output ONLY the final shell command "
            "within a markdown code block. No explanation, no thinking, no prose."
        )

        try:
            start_time = time.time()
            response_text, metadata = await self.client.get_completion_text(
                self.model_slug,
                prompt,
                system_prompt=system_prompt,
                max_tokens=512,
                temperature=0.0,
            )
            latency_ms = int((time.time() - start_time) * 1000)

            if not response_text:
                item_metadata = {**metadata, "system_prompt": system_prompt}
                return ItemResult(
                    item_id=item_id,
                    item_hash=self.compute_item_hash(item["task"]),
                    prompt=prompt,
                    response="",
                    error=self.format_empty_response_error(metadata),
                    latency_ms=latency_ms,
                    metadata=item_metadata,
                )

            # Extract command robustly
            response_cmd = extract_command(response_text)

            if not response_cmd or not looks_like_command(response_cmd):
                item_metadata = {**metadata, "system_prompt": system_prompt}
                return ItemResult(
                    item_id=item_id,
                    item_hash=self.compute_item_hash(item["task"]),
                    prompt=prompt,
                    response=response_text.strip(),
                    expected=item.get("expected_cmd", ""),
                    error="Could not extract a shell command from the model response",
                    latency_ms=latency_ms,
                    input_tokens=metadata.get("usage", {}).get("prompt_tokens"),
                    output_tokens=metadata.get("usage", {}).get("completion_tokens"),
                    metadata=item_metadata,
                )
            
            def tokenize_command(command: str) -> tuple[str, set[str], set[str], list[str]]:
                try:
                    tokens = shlex.split(command)
                except ValueError:
                    tokens = command.split()
                if not tokens:
                    return "", set(), set(), []
                base = tokens[0]
                short_flags: set[str] = set()
                long_flags: set[str] = set()
                args: list[str] = []
                for token in tokens[1:]:
                    if token.startswith("--"):
                        if "=" in token:
                            flag, value = token.split("=", 1)
                            long_flags.add(flag)
                            if value:
                                args.append(value)
                            continue
                        long_flags.add(token)
                        continue
                    if token.startswith("-") and len(token) > 1:
                        if len(token) <= 4:
                            short_flags.update(token[1:])
                            continue
                        if token[1:].islower() and len(token) > 4:
                            long_flags.add(token)
                            continue
                        short_flags.update(token[1:])
                        continue
                    args.append(token)
                return base, short_flags, long_flags, args

            def commands_match(response_cmd: str, expected_cmd: str) -> bool:
                resp_base, resp_short, resp_long, resp_args = tokenize_command(response_cmd)
                exp_base, exp_short, exp_long, exp_args = tokenize_command(expected_cmd)
                if not resp_base or not exp_base or resp_base != exp_base:
                    return False
                if not exp_short.issubset(resp_short):
                    return False
                if not exp_long.issubset(resp_long):
                    return False
                for arg in exp_args:
                    if arg not in resp_args:
                        return False
                return True

            # Create sandbox and run command
            sandbox_id = await self.sandy.create_sandbox()
            if not sandbox_id:
                return ItemResult(item_id=item_id, prompt=prompt, error="Could not create sandbox")
            
            try:
                # Basic execution check
                execution_result = await self.sandy.execute_command(sandbox_id, response_cmd)
                
                # In a real benchmark, we'd compare output or state with expected
                expected_cmd = item.get("expected_cmd", "")
                is_correct = commands_match(response_cmd, expected_cmd) or commands_match(
                    response_cmd, f"sudo {expected_cmd}"
                )
                
                item_metadata = {
                    **metadata,
                    "expected_cmd": expected_cmd,
                    "system_prompt": system_prompt,
                }
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
                    test_code=f"Command to execute: {response_cmd}",
                    metadata=item_metadata,
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
            meta = locals().get("metadata") or {}
            item_metadata = {**meta, "system_prompt": system_prompt}
            return ItemResult(
                item_id=item_id, 
                prompt=prompt, 
                response=locals().get("response_text", ""), 
                error=str(e),
                metadata=item_metadata,
            )
