"""AA-LCR (Adversarial Attacks on Large Code Reasoners) benchmark adapter."""
import time
from typing import Any, AsyncIterator, Optional

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.core.logging import get_logger
from app.services.sandy_service import SandyService

logger = get_logger(__name__)


@register_adapter("aa_lcr")
class AALCRAdapter(BenchmarkAdapter):
    """
    AA-LCR benchmark adapter.
    
    Adversarial attacks on code reasoning capabilities.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._items: list[dict[str, Any]] = []
        self.sandy = SandyService()

    def get_name(self) -> str:
        return "aa_lcr"

    def get_display_name(self) -> str:
        return "AA-LCR"

    def requires_setup(self) -> bool:
        return True

    def get_setup_notes(self) -> Optional[str]:
        return "AA-LCR requires a sandbox for code execution and access to a code dataset (HF_TOKEN if gated)."

    async def get_total_items(self) -> int:
        if not self._items:
            await self.preload()
        return len(self._items)

    async def preload(self) -> None:
        """Load AA-LCR dataset."""
        if self._items:
            return

        try:
            from datasets import load_dataset
            import os

            logger.info("Loading AA-LCR dataset")
            hf_token = os.environ.get("HF_TOKEN")
            
            dataset = None
            # Try loading from code reasoning datasets - prioritizing smaller ones
            sources = [
                ("openai_humaneval", "test", {}),
                ("codeparrot/apps", "test", {}),
                ("deepmind/code_contests", "test", {}),
            ]
            
            for source_name, split, kwargs in sources:
                try:
                    logger.info(f"Trying to load from {source_name}")
                    if hf_token:
                        kwargs["token"] = hf_token
                    # For larger datasets, try to only load a slice if possible
                    if source_name == "deepmind/code_contests":
                        dataset = load_dataset(source_name, split=f"{split}[:100]", **kwargs)
                    else:
                        dataset = load_dataset(source_name, split=split, **kwargs)
                    break
                except Exception as e:
                    logger.warning(f"Could not load {source_name}: {e}")
                    continue
            
            if dataset is None:
                logger.warning("No AA-LCR dataset sources available")
                self._items = []
                return
            
            self._items = []
            for i, item in enumerate(dataset):
                problem = item.get("prompt") or item.get("description") or item.get("question") or item.get("problem") or ""
                if problem:
                    solution = item.get("canonical_solution") or ""
                    test = item.get("test", "") or ""
                    if not solution:
                        solutions = item.get("solutions", {})
                        if isinstance(solutions, dict) and "solution" in solutions:
                            solution = solutions["solution"][0] if solutions["solution"] else ""
                    self._items.append({
                        "id": str(i),
                        "problem": problem,
                        "solution": solution,
                        "test": test
                    })
            
            logger.info(f"Loaded {len(self._items)} AA-LCR items")
        except Exception as e:
            logger.error("Failed to load AA-LCR", error=str(e))
            self._items = []

    async def enumerate_items(self) -> AsyncIterator[str]:
        if not self._items:
            await self.preload()
        for item in self._items:
            yield item["id"]

    async def evaluate_item(self, item_id: str) -> ItemResult:
        """Evaluate a single AA-LCR item."""
        if not self._items:
            await self.preload()

        item = next((i for i in self._items if i["id"] == item_id), None)
        if not item:
            return ItemResult(item_id=item_id, error=f"Item {item_id} not found")

        prompt = f"""Solve the following programming problem with Python code.

Problem:
{item["problem"]}

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

            if not response_text:
                item_metadata = {**metadata, "system_prompt": system_prompt}
                return ItemResult(
                    item_id=item_id,
                    item_hash=self.compute_item_hash(item["problem"]),
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
            # Include problem text if it contains function definitions
            prompt_code = ""
            if "def " in item["problem"] and item["problem"].strip().startswith("def "):
                prompt_code = item["problem"]
                
            full_code = f"{prompt_code}\n\n{extracted_code}\n\n{test_code}"
            
            # Execute in sandbox
            execution_result = await self.sandy.run_python_code(full_code)
            
            is_correct = execution_result.get("success", False) and execution_result.get("exit_code") == 0
            error = None
            if not is_correct:
                error = execution_result.get("stderr") or execution_result.get("error")
            error = self.format_truncation_error(metadata, error)

            item_metadata = {**metadata, "system_prompt": system_prompt}
            return ItemResult(
                item_id=item_id,
                item_hash=self.compute_item_hash(item["problem"]),
                prompt=prompt,
                response=response_text.strip(),
                expected="[Tests passed]",
                is_correct=is_correct,
                score=1.0 if is_correct else 0.0,
                latency_ms=latency_ms,
                input_tokens=metadata.get("usage", {}).get("prompt_tokens"),
                output_tokens=metadata.get("usage", {}).get("completion_tokens"),
                test_code=full_code,
                judge_output={
                    "stdout": execution_result.get("stdout"),
                    "stderr": execution_result.get("stderr"),
                    "exit_code": execution_result.get("exit_code"),
                    "system_prompt": system_prompt
                },
                error=error,
                metadata=item_metadata
            )

        except Exception as e:
            logger.error("AA-LCR evaluation failed", item_id=item_id, error=str(e))
            # Safely capture what we have
            res = locals().get("response_text", "")
            meta = locals().get("metadata") or {}
            item_metadata = {**meta, "system_prompt": system_prompt}
            return ItemResult(
                item_id=item_id, 
                prompt=prompt, 
                response=res if res is not None else "", 
                error=str(e),
                metadata=item_metadata
            )
