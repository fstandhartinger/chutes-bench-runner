"""Humanity's Last Exam (HLE) benchmark adapter."""
import time
from typing import Any, AsyncIterator, Optional

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.core.logging import get_logger

logger = get_logger(__name__)


@register_adapter("hle")
class HLEAdapter(BenchmarkAdapter):
    """
    Humanity's Last Exam benchmark adapter.
    
    A challenging benchmark of expert-level questions across many domains.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._items: list[dict[str, Any]] = []

    def get_name(self) -> str:
        return "hle"

    def get_display_name(self) -> str:
        return "Humanity's Last Exam"

    def requires_setup(self) -> bool:
        return True

    def get_setup_notes(self) -> Optional[str]:
        return "HLE dataset may require authentication. Check Hugging Face access."

    async def get_total_items(self) -> int:
        if not self._items:
            await self.preload()
        return len(self._items)

    async def preload(self) -> None:
        """Load HLE dataset."""
        if self._items:
            return

        try:
            from datasets import load_dataset

            logger.info("Loading Humanity's Last Exam dataset")
            # HLE is hosted on HuggingFace
            dataset = load_dataset("cais/hle", split="test")
            
            self._items = []
            for i, item in enumerate(dataset):
                self._items.append({
                    "id": str(i),
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "subject": item.get("subject", ""),
                    "source": item.get("source", ""),
                })
            
            logger.info(f"Loaded {len(self._items)} HLE items")
        except Exception as e:
            logger.error("Failed to load HLE", error=str(e))
            # Return empty list - benchmark will be marked as needs_setup
            self._items = []

    async def enumerate_items(self) -> AsyncIterator[str]:
        if not self._items:
            await self.preload()
        for item in self._items:
            yield item["id"]

    async def evaluate_item(self, item_id: str) -> ItemResult:
        """Evaluate a single HLE item."""
        if not self._items:
            await self.preload()

        item = next((i for i in self._items if i["id"] == item_id), None)
        if not item:
            return ItemResult(item_id=item_id, error=f"Item {item_id} not found")

        prompt = f"""Answer the following question. Provide a clear, concise answer.

Question: {item["question"]}

Answer:"""

        try:
            start_time = time.time()
            response_text, metadata = await self.client.get_completion_text(
                self.model_slug,
                prompt,
                system_prompt="You are an expert in all fields. Answer questions accurately and concisely.",
                max_tokens=512,
                temperature=0.0,
            )
            latency_ms = int((time.time() - start_time) * 1000)

            # HLE uses exact match or semantic similarity
            expected = item["answer"].strip().lower()
            response_clean = response_text.strip().lower()
            
            # Simple exact match for now
            is_correct = expected in response_clean or response_clean in expected

            return ItemResult(
                item_id=item_id,
                item_hash=self.compute_item_hash(item["question"]),
                prompt=prompt,
                response=response_text.strip(),
                expected=item["answer"],
                is_correct=is_correct,
                score=1.0 if is_correct else 0.0,
                latency_ms=latency_ms,
                input_tokens=metadata.get("usage", {}).get("prompt_tokens"),
                output_tokens=metadata.get("usage", {}).get("completion_tokens"),
                metadata={"subject": item.get("subject")},
            )

        except Exception as e:
            logger.error("HLE evaluation failed", item_id=item_id, error=str(e))
            return ItemResult(item_id=item_id, prompt=prompt, error=str(e))

