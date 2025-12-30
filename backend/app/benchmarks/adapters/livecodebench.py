"""LiveCodeBench benchmark adapter."""
import time
from typing import Any, AsyncIterator, Optional

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.core.logging import get_logger
from app.services.sandy_service import SandyService

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
        self.sandy = SandyService()

    def get_name(self) -> str:
        return "livecodebench"
    
    def get_display_name(self) -> str:
        return "LiveCodeBench"

    def requires_setup(self) -> bool:
        return True

    def get_setup_notes(self) -> Optional[str]:
        return "LiveCodeBench requires a sandbox for code execution and dataset access (HF_TOKEN if gated)."

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
            import os

            logger.info("Loading LiveCodeBench dataset")
            hf_token = os.environ.get("HF_TOKEN")
            
            # Try multiple coding benchmark sources
            sources = [
                ("livecodebench/code_generation_lite", "test", ["question_content", "starter_code"]),
                ("codeparrot/apps", "test", ["question", "starter_code"]),
                ("openai_humaneval", "test", ["prompt", "test"]),
            ]
            
            dataset = None
            field_map = {}
            for source_name, split, fields in sources:
                try:
                    logger.info(f"Trying to load from {source_name}")
                    kwargs = {"token": hf_token} if hf_token else {}
                    dataset = load_dataset(source_name, split=split, **kwargs)
                    field_map = {"question": fields[0], "starter_code": fields[1]}
                    break
                except Exception as e:
                    logger.warning(f"Could not load {source_name}: {e}")
                    continue
            
            if dataset is None:
                logger.warning("No LiveCodeBench dataset sources available")
                self._items = []
                return
            
            self._items = []
            for i, item in enumerate(dataset):
                question = item.get(field_map["question"], "") or ""
                starter = item.get(field_map["starter_code"], "") if field_map["starter_code"] else ""
                test = item.get("test", "") or item.get("canonical_solution", "")
                if question:
                    self._items.append({
                        "id": str(i),
                        "question": question,
                        "starter_code": starter or "",
                        "difficulty": item.get("difficulty", ""),
                        "test": test,
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

        system_prompt = "Output ONLY the final Python code within a markdown code block. Do NOT use <think> tags. Do NOT provide any explanations or prose. Just the code."
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

            # Robust handling of empty or None response
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

            # Extract code from response robustly
            extracted_code = self.extract_python_code(response_text)
            
            # Prepare execution code
            test_code = item.get("test", "")
            full_code = f"{extracted_code}\n\n{test_code}"
            
            # Execute in sandbox
            execution_result = await self.sandy.run_python_code(full_code)
            
            is_correct = execution_result.get("success", False) and execution_result.get("exit_code") == 0
            error = None
            if not is_correct:
                error = execution_result.get("stderr") or execution_result.get("error")
            error = self.format_truncation_error(metadata, error)

            item_metadata = {
                **metadata,
                "difficulty": item.get("difficulty"),
                "system_prompt": system_prompt,
                "test_code": test_code,
                "full_code_executed": full_code,
                "extracted_code": extracted_code,
            }
            return ItemResult(
                item_id=item_id,
                item_hash=self.compute_item_hash(item["question"]),
                prompt=prompt,
                response=response_text.strip() if response_text else "",
                expected="[Tests passed]",
                is_correct=is_correct,
                score=1.0 if is_correct else 0.0,
                latency_ms=latency_ms,
                input_tokens=metadata.get("usage", {}).get("prompt_tokens"),
                output_tokens=metadata.get("usage", {}).get("completion_tokens"),
                test_code=full_code,
                metadata=item_metadata,
                judge_output={
                    "stdout": execution_result.get("stdout"),
                    "stderr": execution_result.get("stderr"),
                    "exit_code": execution_result.get("exit_code"),
                    "test_code": test_code,
                    "extracted_code": extracted_code,
                    "system_prompt": system_prompt,
                },
                error=error
            )

        except Exception as e:
            logger.error("LiveCodeBench evaluation failed", item_id=item_id, error=str(e))
            # Safely capture what we have
            res = locals().get("response_text", "")
            meta = locals().get("metadata") or {}
            item_metadata = {
                **meta,
                "difficulty": item.get("difficulty"),
                "system_prompt": system_prompt,
            }
            return ItemResult(
                item_id=item_id, 
                prompt=prompt, 
                response=res if res is not None else "", 
                error=str(e),
                metadata=item_metadata,
            )
