"""GPQA Diamond benchmark adapter."""
import time
from typing import Any, AsyncIterator, Optional

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.core.logging import get_logger

logger = get_logger(__name__)


@register_adapter("gpqa_diamond")
class GPQADiamondAdapter(BenchmarkAdapter):
    """
    GPQA Diamond benchmark adapter.
    
    Graduate-level science questions requiring expert-level knowledge.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._items: list[dict[str, Any]] = []

    def get_name(self) -> str:
        return "gpqa_diamond"

    def get_display_name(self) -> str:
        return "GPQA Diamond"

    async def get_total_items(self) -> int:
        if not self._items:
            await self.preload()
        return len(self._items)

    async def preload(self) -> None:
        """Load GPQA Diamond dataset."""
        if self._items:
            return

        try:
            from datasets import load_dataset

            logger.info("Loading GPQA Diamond dataset")
            dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
            
            self._items = []
            for i, item in enumerate(dataset):
                self._items.append({
                    "id": str(i),
                    "question": item["Question"],
                    "correct_answer": item["Correct Answer"],
                    "incorrect_answers": [
                        item.get("Incorrect Answer 1", ""),
                        item.get("Incorrect Answer 2", ""),
                        item.get("Incorrect Answer 3", ""),
                    ],
                })
            
            logger.info(f"Loaded {len(self._items)} GPQA Diamond items")
        except Exception as e:
            logger.error("Failed to load GPQA Diamond", error=str(e))
            raise

    async def enumerate_items(self) -> AsyncIterator[str]:
        if not self._items:
            await self.preload()
        for item in self._items:
            yield item["id"]

    async def evaluate_item(self, item_id: str) -> ItemResult:
        """Evaluate a single GPQA Diamond item."""
        if not self._items:
            await self.preload()

        item = next((i for i in self._items if i["id"] == item_id), None)
        if not item:
            return ItemResult(item_id=item_id, error=f"Item {item_id} not found")

        # Shuffle options deterministically
        import hashlib
        import random
        
        options = [item["correct_answer"]] + [a for a in item["incorrect_answers"] if a]
        seed = int(hashlib.sha256(item_id.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        rng.shuffle(options)
        
        correct_idx = options.index(item["correct_answer"])
        correct_letter = chr(65 + correct_idx)

        options_str = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options))
        prompt = f"""Answer the following graduate-level question. Respond with only the letter of the correct answer.

Question: {item["question"]}

{options_str}

Answer:"""

        try:
            start_time = time.time()
            response_text, metadata = await self.client.get_completion_text(
                self.model_slug,
                prompt,
                system_prompt="You are an expert scientist. Answer the question with only the letter of the correct answer.",
                max_tokens=10,
                temperature=0.0,
            )
            latency_ms = int((time.time() - start_time) * 1000)

            answer = response_text.strip().upper()[:1]
            is_correct = answer == correct_letter

            return ItemResult(
                item_id=item_id,
                item_hash=self.compute_item_hash(item["question"]),
                prompt=prompt,
                response=response_text.strip(),
                expected=correct_letter,
                is_correct=is_correct,
                score=1.0 if is_correct else 0.0,
                latency_ms=latency_ms,
                input_tokens=metadata.get("usage", {}).get("prompt_tokens"),
                output_tokens=metadata.get("usage", {}).get("completion_tokens"),
            )

        except Exception as e:
            logger.error("GPQA evaluation failed", item_id=item_id, error=str(e))
            return ItemResult(item_id=item_id, prompt=prompt, error=str(e))

