"""LiveCodeBench benchmark adapter."""
import asyncio
import json
import time
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from huggingface_hub import hf_hub_url

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.benchmarks.utils import download_hf_file_async, download_http_file_async, load_dataset_with_retry
from app.core.logging import get_logger
from app.services.sandy_service import SandyService

logger = get_logger(__name__)

LIVECODEBENCH_TOTAL_ITEMS = 164
_LIVECODEBENCH_LOCK = asyncio.Lock()
_LIVECODEBENCH_JSONL: Optional[Path] = None
_LIVECODEBENCH_ITEMS: Optional[list[dict[str, Any]]] = None


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
        self._jsonl_path: Optional[Path] = None
        self.sandy = SandyService()

    def get_name(self) -> str:
        return "livecodebench"

    def get_display_name(self) -> str:
        return "LiveCodeBench"

    def requires_setup(self) -> bool:
        return False

    def _looks_like_code(self, code: str) -> bool:
        if not code:
            return False
        for line in code.splitlines():
            stripped = line.strip()
            if stripped.startswith("if __name__"):
                return True
            if stripped.startswith(("def ", "class ", "import ", "from ")):
                return True
        return False

    async def get_total_items(self) -> int:
        if self._items:
            return len(self._items)
        return LIVECODEBENCH_TOTAL_ITEMS

    def _parse_item(self, index: int, item: dict[str, Any]) -> Optional[dict[str, Any]]:
        question = item.get("question_content") or ""
        if not question:
            return None
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
        return {
            "id": str(index),
            "question": question,
            "starter_code": item.get("starter_code") or "",
            "difficulty": item.get("difficulty") or "",
            "public_tests": public_cases,
            "private_tests": private_cases,
            "metadata": item.get("metadata") or {},
        }

    async def get_items_for_evaluation(self, subset_pct: int, seed: str) -> tuple[int, list[str]]:
        total_items = await self.get_total_items()
        if subset_pct >= 100:
            await self.preload()
            return len(self._items), [item["id"] for item in self._items]

        target = max(1, int(total_items * subset_pct / 100))
        self._items = []
        try:
            import os

            hf_token = os.environ.get("HF_TOKEN")
            dataset = await load_dataset_with_retry(
                "livecodebench/code_generation",
                split="test",
                streaming=True,
                token=hf_token,
            )
            for i, item in enumerate(dataset):
                if len(self._items) >= target:
                    break
                parsed = self._parse_item(i, item)
                if parsed:
                    self._items.append(parsed)
        except Exception as e:
            logger.error("Failed to stream LiveCodeBench subset", error=str(e))
            self._items = []

        return total_items, [item["id"] for item in self._items]

    async def preload(self) -> None:
        """Load LiveCodeBench dataset."""
        if self._items:
            return

        try:
            global _LIVECODEBENCH_ITEMS, _LIVECODEBENCH_JSONL
            async with _LIVECODEBENCH_LOCK:
                if _LIVECODEBENCH_ITEMS is not None:
                    self._items = _LIVECODEBENCH_ITEMS
                    self._jsonl_path = _LIVECODEBENCH_JSONL
                    return

                logger.info("Loading LiveCodeBench dataset")
                jsonl_path = await self._ensure_jsonl()
                if not jsonl_path:
                    raise RuntimeError("Could not download LiveCodeBench dataset file")

                items: list[dict[str, Any]] = []
                with jsonl_path.open("r", encoding="utf-8") as handle:
                    for i, line in enumerate(handle):
                        if not line.strip():
                            continue
                        item = json.loads(line)
                        parsed = self._parse_item(i, item)
                        if parsed:
                            items.append(parsed)

                logger.info("Loaded %s LiveCodeBench items", len(items))
                self._items = items
                self._jsonl_path = jsonl_path
                _LIVECODEBENCH_ITEMS = items
                _LIVECODEBENCH_JSONL = jsonl_path
        except Exception as e:
            logger.error("Failed to load LiveCodeBench", error=str(e))
            self._items = []
            raise

    async def _ensure_jsonl(self) -> Optional[Path]:
        if self._jsonl_path:
            return self._jsonl_path
        try:
            import os

            hf_token = os.environ.get("HF_TOKEN")
            for attempt in range(1, 4):
                try:
                    self._jsonl_path = await download_hf_file_async(
                        "livecodebench/code_generation",
                        "test.jsonl",
                        repo_type="dataset",
                        token=hf_token,
                        cache_subdir="livecodebench",
                    )
                    break
                except Exception as exc:
                    logger.warning(
                        "Failed to download LiveCodeBench JSONL",
                        attempt=attempt,
                        error=str(exc) or exc.__class__.__name__,
                    )
                    if attempt >= 3:
                        raise
                    await asyncio.sleep(min(2**attempt, 10))

            if not self._jsonl_path:
                url = hf_hub_url(
                    "livecodebench/code_generation",
                    "test.jsonl",
                    repo_type="dataset",
                )
                headers = {"Authorization": f"Bearer {hf_token}"} if hf_token else None
                self._jsonl_path = await download_http_file_async(
                    url,
                    cache_subdir="livecodebench",
                    filename="test.jsonl",
                    headers=headers,
                    timeout_seconds=900,
                )
            return self._jsonl_path
        except Exception as e:
            logger.error("Failed to download LiveCodeBench JSONL", error=str(e))
            return None

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
    ) -> tuple[bool, Optional[dict[str, Any]], list[dict[str, Any]]]:
        if not tests:
            return True, None, []

        await self.sandy.write_file(sandbox_id, "main.py", code)

        test_runs: list[dict[str, Any]] = []
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
            stderr = (result.get("stderr") or "").strip()
            test_log = {
                "test_index": idx,
                "input": input_text,
                "expected": expected,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": result.get("exit_code"),
            }
            if result.get("exit_code") != 0:
                test_log["status"] = "error"
                test_runs.append(test_log)
                return False, {
                    "test_index": idx,
                    "error": result.get("stderr") or result.get("error"),
                }, test_runs
            if stdout != expected:
                test_log["status"] = "failed"
                test_log["received"] = stdout
                test_runs.append(test_log)
                return False, {
                    "test_index": idx,
                    "expected": expected,
                    "received": stdout,
                }, test_runs
            test_log["status"] = "passed"
            test_runs.append(test_log)
        return True, None, test_runs

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
Return ONLY Python code (no markdown, no explanations). End the response with <END>.

Problem:
{item["question"]}

{starter_section}
"""

        system_prompt = (
            "Output ONLY Python code. No markdown, no explanations, no prose. "
            "Do NOT use <think> tags. End the response with <END>."
        )
        try:
            start_time = time.time()
            response_text, metadata = await self.client.get_completion_text(
                self.model_slug,
                prompt,
                system_prompt=system_prompt,
                max_tokens=16384,
                stop=["<END>"],
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
            if not self._looks_like_code(extracted_code):
                item_metadata = {
                    **metadata,
                    "difficulty": item.get("difficulty"),
                    "system_prompt": system_prompt,
                }
                error = self.format_truncation_error(
                    metadata,
                    "Model response did not contain Python code",
                )
                return ItemResult(
                    item_id=item_id,
                    item_hash=self.compute_item_hash(item["question"]),
                    prompt=prompt,
                    response=response_text.strip(),
                    expected="[All tests passed]",
                    is_correct=False,
                    score=0.0,
                    latency_ms=latency_ms,
                    input_tokens=metadata.get("usage", {}).get("prompt_tokens"),
                    output_tokens=metadata.get("usage", {}).get("completion_tokens"),
                    error=error,
                    metadata=item_metadata,
                )
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
                is_correct, failure, test_runs = await self._run_io_tests(sandbox_id, extracted_code, tests)
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
                    "public_test_count": len(item.get("public_tests", [])),
                    "private_test_count": len(item.get("private_tests", [])),
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
                        "test_runs": test_runs,
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
