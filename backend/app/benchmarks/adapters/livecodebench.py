"""LiveCodeBench benchmark adapter."""
import json
import time
from typing import Any, AsyncIterator, Optional

from datasets import load_dataset

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.core.logging import get_logger
from app.services.sandy_service import SandyService

logger = get_logger(__name__)


@register_adapter("livecodebench")
class LiveCodeBenchAdapter(BenchmarkAdapter):
    """
    LiveCodeBench adapter.

    Uses the official LiveCodeBench code_generation dataset and runs provided
    public/private test cases against the model output.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._items: list[dict[str, Any]] = []
        self.sandy = SandyService()

    def get_name(self) -> str:
        return "livecodebench"

    def get_display_name(self) -> str:
        return "LiveCodeBench"

    def requires_setup(self) -> bool:
        return False

    async def get_total_items(self) -> int:
        if not self._items:
            await self.preload()
        return len(self._items)

    async def preload(self) -> None:
        """Load LiveCodeBench dataset."""
        if self._items:
            return

        try:
            import os

            logger.info("Loading LiveCodeBench dataset")
            hf_token = os.environ.get("HF_TOKEN")
            dataset = load_dataset(
                "livecodebench/code_generation",
                split="test",
                token=hf_token,
            )

            self._items = []
            for i, item in enumerate(dataset):
                question = item.get("question_content") or ""
                public_tests = item.get("public_test_cases") or "[]"
                private_tests = item.get("private_test_cases") or "[]"
                try:
                    public_cases = json.loads(public_tests) if isinstance(public_tests, str) else public_tests
                except json.JSONDecodeError:
                    public_cases = []
                try:
                    private_cases = json.loads(private_tests) if isinstance(private_tests, str) else private_tests
                except json.JSONDecodeError:
                    private_cases = []

                if question:
                    self._items.append(
                        {
                            "id": str(i),
                            "question": question,
                            "starter_code": item.get("starter_code") or "",
                            "difficulty": item.get("difficulty") or "",
                            "public_tests": public_cases,
                            "private_tests": private_cases,
                            "metadata": item.get("metadata") or {},
                        }
                    )

            logger.info("Loaded %s LiveCodeBench items", len(self._items))
        except Exception as e:
            logger.error("Failed to load LiveCodeBench", error=str(e))
            self._items = []

    async def enumerate_items(self) -> AsyncIterator[str]:
        if not self._items:
            await self.preload()
        for item in self._items:
            yield item["id"]

    async def _run_io_tests(
        self,
        sandbox_id: str,
        code: str,
        tests: list[dict[str, Any]],
    ) -> tuple[bool, Optional[dict[str, Any]]]:
        if not tests:
            return True, None

        await self.sandy.write_file(sandbox_id, "main.py", code)

        for idx, test in enumerate(tests, start=1):
            input_text = test.get("input", "")
            expected = (test.get("output", "") or "").strip()
            await self.sandy.write_file(sandbox_id, "input.txt", input_text)
            result = await self.sandy.execute_command(
                sandbox_id,
                "bash -lc 'python3 main.py < input.txt'",
                timeout_ms=60000,
            )
            stdout = (result.get("stdout") or "").strip()
            if result.get("exit_code") != 0:
                return False, {
                    "test_index": idx,
                    "error": result.get("stderr") or result.get("error"),
                }
            if stdout != expected:
                return False, {
                    "test_index": idx,
                    "expected": expected,
                    "received": stdout,
                }
        return True, None

    async def evaluate_item(self, item_id: str) -> ItemResult:
        """Evaluate a single LiveCodeBench item."""
        if not self._items:
            await self.preload()

        item = next((i for i in self._items if i["id"] == item_id), None)
        if not item:
            return ItemResult(item_id=item_id, error=f"Item {item_id} not found")

        starter = item.get("starter_code") or ""
        starter_section = f"Starter code:\n{starter}" if starter else ""
        prompt = f"""Solve the following coding problem. Provide a complete Python solution.

Problem:
{item["question"]}

{starter_section}

Solution:
```python
"""

        system_prompt = (
            "Output ONLY the final Python code within a markdown code block. "
            "Do NOT use <think> tags. Do NOT provide any explanations or prose. Just the code."
        )
        try:
            start_time = time.time()
            response_text, metadata = await self.client.get_completion_text(
                self.model_slug,
                prompt,
                system_prompt=system_prompt,
                max_tokens=16384,
                temperature=0.0,
            )
            latency_ms = int((time.time() - start_time) * 1000)

            if not response_text:
                item_metadata = {
                    **metadata,
                    "difficulty": item.get("difficulty"),
                    "system_prompt": system_prompt,
                }
                return ItemResult(
                    item_id=item_id,
                    item_hash=self.compute_item_hash(item["question"]),
                    prompt=prompt,
                    response="",
                    error=self.format_empty_response_error(metadata),
                    latency_ms=latency_ms,
                    metadata=item_metadata,
                )

            extracted_code = self.extract_python_code(response_text)
            sandbox_id = await self.sandy.create_sandbox()
            if not sandbox_id:
                return ItemResult(
                    item_id=item_id,
                    item_hash=self.compute_item_hash(item["question"]),
                    prompt=prompt,
                    response=response_text.strip(),
                    error="Could not create sandbox",
                    latency_ms=latency_ms,
                    metadata={"system_prompt": system_prompt},
                )

            try:
                tests = list(item.get("public_tests", [])) + list(item.get("private_tests", []))
                is_correct, failure = await self._run_io_tests(sandbox_id, extracted_code, tests)
                error = None
                if not is_correct:
                    if failure:
                        if "error" in failure:
                            error = failure["error"]
                        else:
                            error = "Output mismatch"
                error = self.format_truncation_error(metadata, error)

                item_metadata = {
                    **metadata,
                    "difficulty": item.get("difficulty"),
                    "system_prompt": system_prompt,
                    "test_count": len(tests),
                    "metadata": item.get("metadata"),
                }
                return ItemResult(
                    item_id=item_id,
                    item_hash=self.compute_item_hash(item["question"]),
                    prompt=prompt,
                    response=response_text.strip(),
                    expected="[All tests passed]",
                    is_correct=is_correct,
                    score=1.0 if is_correct else 0.0,
                    latency_ms=latency_ms,
                    input_tokens=metadata.get("usage", {}).get("prompt_tokens"),
                    output_tokens=metadata.get("usage", {}).get("completion_tokens"),
                    test_code=extracted_code,
                    judge_output={
                        "failure": failure,
                        "test_count": len(tests),
                    },
                    error=error,
                    metadata=item_metadata,
                )
            finally:
                await self.sandy.terminate_sandbox(sandbox_id)

        except Exception as e:
            logger.error("LiveCodeBench evaluation failed", item_id=item_id, error=str(e))
            meta = locals().get("metadata") or {}
            item_metadata = {
                **meta,
                "difficulty": item.get("difficulty"),
                "system_prompt": system_prompt,
            }
            return ItemResult(
                item_id=item_id,
                prompt=prompt,
                response=locals().get("response_text", "") or "",
                error=str(e),
                metadata=item_metadata,
            )
