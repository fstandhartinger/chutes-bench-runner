"""AIME 2025 benchmark adapter."""
import re
import time
from typing import Any, AsyncIterator, Optional

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
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

    async def get_total_items(self) -> int:
        if not self._items:
            await self.preload()
        return len(self._items)

    async def preload(self) -> None:
        """Load AIME 2025 dataset."""
        if self._items:
            return

        try:
            from datasets import load_dataset

            logger.info("Loading AIME 2025 dataset")
            # Try to load from common AIME dataset sources
            try:
                dataset = load_dataset("AI-MO/aime24", split="test")
            except Exception:
                # Fallback to a simpler dataset structure
                dataset = load_dataset("hendrycks/competition_math", split="test")
            
            self._items = []
            for i, item in enumerate(dataset):
                if "problem" in item:
                    self._items.append({
                        "id": str(i),
                        "problem": item.get("problem", ""),
                        "answer": item.get("answer", ""),
                        "level": item.get("level", ""),
                    })
            
            logger.info(f"Loaded {len(self._items)} AIME items")
        except Exception as e:
            logger.error("Failed to load AIME 2025", error=str(e))
            self._items = []

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

        prompt = f"""Solve the following math competition problem. AIME answers are always integers from 0 to 999.
Show your reasoning step by step, then provide your final answer as a single integer on a new line prefixed with "ANSWER:".

Problem: {item["problem"]}

Solution:"""

        try:
            start_time = time.time()
            response_text, metadata = await self.client.get_completion_text(
                self.model_slug,
                prompt,
                system_prompt="You are an expert mathematician solving competition problems. Show your work and provide the final integer answer.",
                max_tokens=2048,
                temperature=0.0,
            )
            latency_ms = int((time.time() - start_time) * 1000)

            # Extract answer
            answer_match = re.search(r"ANSWER:\s*(\d+)", response_text, re.IGNORECASE)
            model_answer = answer_match.group(1) if answer_match else ""
            
            # Clean expected answer
            expected = str(item["answer"]).strip()
            if expected.startswith("\\boxed{"):
                expected = expected[7:-1]
            expected = re.sub(r"[^\d]", "", expected)
            
            is_correct = model_answer == expected

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
                metadata={"level": item.get("level")},
            )

        except Exception as e:
            logger.error("AIME evaluation failed", item_id=item_id, error=str(e))
            return ItemResult(item_id=item_id, prompt=prompt, error=str(e))

