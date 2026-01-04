"""AIME 2025 benchmark adapter."""
import os
import re
import time
from typing import Any, AsyncIterator, Optional

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.benchmarks.utils import load_dataset_with_retry
from app.core.logging import get_logger

logger = get_logger(__name__)


@register_adapter("aime_2025")
class AIME2025Adapter(BenchmarkAdapter):
    """
    AIME 2025 benchmark adapter.
    
    American Invitational Mathematics Examination problems.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._items: list[dict[str, Any]] = []

    def get_name(self) -> str:
        return "aime_2025"

    def get_display_name(self) -> str:
        return "AIME 2025"

    def supports_parallel_items(self) -> bool:
        return True

    async def get_total_items(self) -> int:
        if not self._items:
            await self.preload()
        return len(self._items)

    async def preload(self) -> None:
        """Load AIME 2025 dataset."""
        if self._items:
            return

        try:
            logger.info("Loading AIME/Competition Math dataset")
            hf_token = os.environ.get("HF_TOKEN")
            dataset = await load_dataset_with_retry(
                "AI-MO/aimo-validation-aime",
                split="train",
                token=hf_token,
            )
            
            self._items = []
            for i, item in enumerate(dataset):
                problem = str(item.get("problem") or item.get("question") or "")
                answer = str(item.get("answer") or item.get("solution") or "")
                if problem:
                    self._items.append({
                        "id": str(i),
                        "problem": problem,
                        "answer": answer,
                        "level": item.get("level", item.get("type", "")),
                    })
            
            logger.info(f"Loaded {len(self._items)} AIME items")
        except Exception as e:
            logger.error("Failed to load AIME", error=str(e))
            self._items = []
            raise

    async def enumerate_items(self) -> AsyncIterator[str]:
        if not self._items:
            await self.preload()
        for item in self._items:
            yield item["id"]

    async def evaluate_item(self, item_id: str) -> ItemResult:
        """Evaluate a single AIME item."""
        if not self._items:
            await self.preload()

        item = next((i for i in self._items if i["id"] == item_id), None)
        if not item:
            return ItemResult(item_id=item_id, error=f"Item {item_id} not found")

        prompt = (
            "Solve the following math competition problem. AIME answers are always integers from 0 to 999.\n"
            "Provide your final answer as a single integer on a new line prefixed with \"ANSWER:\".\n\n"
            f"Problem: {item['problem']}\n\n"
            "Answer:"
        )

        system_prompt = (
            "You are a test-taking assistant. Output ONLY the final answer line in the format "
            "'ANSWER: <integer>' with no extra text."
        )
        try:
            start_time = time.time()
            response_text, metadata = await self.client.get_completion_text(
                self.model_slug,
                prompt,
                system_prompt=system_prompt,
                max_tokens=65536,
                temperature=0.0,
            )
            latency_ms = int((time.time() - start_time) * 1000)

            if not response_text or response_text is None:
                item_metadata = {
                    **metadata,
                    "level": item.get("level"),
                    "system_prompt": system_prompt,
                }
                return ItemResult(
                    item_id=item_id,
                    item_hash=self.compute_item_hash(item["problem"]),
                    prompt=prompt,
                    response="",
                    expected=str(item.get("answer", "")),
                    is_correct=False,
                    score=0.0,
                    error=self.format_empty_response_error(metadata),
                    latency_ms=latency_ms,
                    input_tokens=metadata.get("usage", {}).get("prompt_tokens"),
                    output_tokens=metadata.get("usage", {}).get("completion_tokens"),
                    metadata=item_metadata,
                )

            response_str = str(response_text)
            model_answer = ""

            answer_matches = re.findall(r"ANSWER:\s*(\d+)", response_str, re.IGNORECASE)
            if answer_matches:
                model_answer = answer_matches[-1]

            if not model_answer:
                boxed_match = re.search(r"\\boxed\{(\d+)\}", response_str)
                if boxed_match:
                    model_answer = boxed_match.group(1)

            if not model_answer:
                clean_text = re.sub(r"(?i)<think>.*?</think>", "", response_str, flags=re.DOTALL).strip()
                numbers = re.findall(r"\b\d+\b", clean_text)
                if numbers:
                    model_answer = numbers[-1]
            
            # Clean expected answer - ensure it's not None
            expected = str(item.get("answer", "")).strip()
            if expected.startswith("\\boxed{"):
                expected = expected[7:-1]
            expected = re.sub(r"[^\d]", "", expected)
            
            # Compare as integers if possible
            try:
                is_correct = int(model_answer) == int(expected)
            except (ValueError, TypeError):
                is_correct = model_answer == expected

            error = None
            if not is_correct:
                error = self.format_truncation_error(metadata, None)

            item_metadata = {
                **metadata,
                "level": item.get("level"),
                "system_prompt": system_prompt,
                "parsed_answer": model_answer or None,
            }
            return ItemResult(
                item_id=item_id,
                item_hash=self.compute_item_hash(item["problem"]),
                prompt=prompt,
                response=response_text.strip(),
                expected=expected,
                is_correct=is_correct,
                score=1.0 if is_correct else 0.0,
                latency_ms=latency_ms,
                input_tokens=metadata.get("usage", {}).get("prompt_tokens"),
                output_tokens=metadata.get("usage", {}).get("completion_tokens"),
                error=error,
                metadata=item_metadata,
            )

        except Exception as e:
            logger.error("AIME evaluation failed", item_id=item_id, error=str(e))
            # Safely capture what we have
            res = locals().get("response_text", "")
            meta = locals().get("metadata") or {}
            item_metadata = {
                **meta,
                "level": item.get("level"),
                "system_prompt": system_prompt,
            }
            return ItemResult(
                item_id=item_id, 
                prompt=prompt, 
                response=res if res is not None else "", 
                error=str(e),
                metadata=item_metadata,
            )
