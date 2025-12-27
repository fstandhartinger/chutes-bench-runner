"""MMLU-Pro benchmark adapter."""
import time
from typing import Any, AsyncIterator, Optional

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.core.logging import get_logger

logger = get_logger(__name__)


@register_adapter("mmlu_pro")
class MMLUProAdapter(BenchmarkAdapter):
    """
    MMLU-Pro benchmark adapter.
    
    Uses datasets library to load MMLU-Pro and evaluates via Chutes API.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._dataset: Optional[Any] = None
        self._items: list[dict[str, Any]] = []

    def get_name(self) -> str:
        return "mmlu_pro"

    def get_display_name(self) -> str:
        return "MMLU-Pro"

    async def get_total_items(self) -> int:
        """Get total items from dataset."""
        if not self._items:
            await self.preload()
        return len(self._items)

    async def preload(self) -> None:
        """Load MMLU-Pro dataset."""
        if self._items:
            return

        try:
            from datasets import load_dataset

            logger.info("Loading MMLU-Pro dataset")
            dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
            
            self._items = []
            for i, item in enumerate(dataset):
                self._items.append({
                    "id": str(i),
                    "question": item["question"],
                    "options": item["options"],
                    "answer": item["answer"],
                    "category": item.get("category", ""),
                })
            
            logger.info(f"Loaded {len(self._items)} MMLU-Pro items")
        except Exception as e:
            logger.error("Failed to load MMLU-Pro", error=str(e))
            raise

    async def enumerate_items(self) -> AsyncIterator[str]:
        """Yield all item IDs."""
        if not self._items:
            await self.preload()
        for item in self._items:
            yield item["id"]

    async def evaluate_item(self, item_id: str) -> ItemResult:
        """Evaluate a single MMLU-Pro item."""
        if not self._items:
            await self.preload()

        # Find item
        item = None
        for i in self._items:
            if i["id"] == item_id:
                item = i
                break

        if not item:
            return ItemResult(
                item_id=item_id,
                error=f"Item {item_id} not found",
            )

        # Format prompt
        options_str = "\n".join(
            f"{chr(65 + i)}. {opt}" for i, opt in enumerate(item["options"])
        )
        prompt = f"""Answer the following multiple choice question. Respond with only the letter of the correct answer (A, B, C, D, etc.).

Question: {item["question"]}

{options_str}

Answer:"""

        try:
            start_time = time.time()
            response_text, metadata = await self.client.get_completion_text(
                self.model_slug,
                prompt,
                system_prompt="You are an expert test taker. Answer multiple choice questions with only the letter of the correct answer.",
                max_tokens=10,
                temperature=0.0,
            )
            latency_ms = int((time.time() - start_time) * 1000)

            # Parse response
            answer_letter = response_text.strip().upper()[:1]
            expected = item["answer"].upper()[:1]
            is_correct = answer_letter == expected

            return ItemResult(
                item_id=item_id,
                item_hash=self.compute_item_hash(item["question"]),
                prompt=prompt,
                response=response_text.strip(),
                expected=expected,
                is_correct=is_correct,
                score=1.0 if is_correct else 0.0,
                latency_ms=latency_ms,
                input_tokens=metadata.get("usage", {}).get("prompt_tokens"),
                output_tokens=metadata.get("usage", {}).get("completion_tokens"),
                metadata={"category": item.get("category")},
            )

        except Exception as e:
            logger.error("MMLU-Pro evaluation failed", item_id=item_id, error=str(e))
            return ItemResult(
                item_id=item_id,
                prompt=prompt,
                error=str(e),
            )

