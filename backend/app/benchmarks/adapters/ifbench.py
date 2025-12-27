"""IFBench (Instruction Following) benchmark adapter."""
import time
from typing import Any, AsyncIterator, Optional

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.core.logging import get_logger

logger = get_logger(__name__)


@register_adapter("ifbench")
class IFBenchAdapter(BenchmarkAdapter):
    """
    IFBench adapter.
    
    Instruction Following benchmark evaluating model's ability to follow complex instructions.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._items: list[dict[str, Any]] = []

    def get_name(self) -> str:
        return "ifbench"

    def get_display_name(self) -> str:
        return "IFBench"

    async def get_total_items(self) -> int:
        if not self._items:
            await self.preload()
        return len(self._items)

    async def preload(self) -> None:
        """Load IFBench dataset."""
        if self._items:
            return

        try:
            from datasets import load_dataset

            logger.info("Loading IFBench dataset")
            # IFEval/IFBench dataset
            dataset = load_dataset("google/IFEval", split="train")
            
            self._items = []
            for i, item in enumerate(dataset):
                self._items.append({
                    "id": str(i),
                    "prompt": item.get("prompt", ""),
                    "instruction_id_list": item.get("instruction_id_list", []),
                    "kwargs": item.get("kwargs", {}),
                })
            
            logger.info(f"Loaded {len(self._items)} IFBench items")
        except Exception as e:
            logger.error("Failed to load IFBench", error=str(e))
            self._items = []

    async def enumerate_items(self) -> AsyncIterator[str]:
        if not self._items:
            await self.preload()
        for item in self._items:
            yield item["id"]

    async def evaluate_item(self, item_id: str) -> ItemResult:
        """Evaluate a single IFBench item."""
        if not self._items:
            await self.preload()

        item = next((i for i in self._items if i["id"] == item_id), None)
        if not item:
            return ItemResult(item_id=item_id, error=f"Item {item_id} not found")

        prompt = item["prompt"]

        try:
            start_time = time.time()
            response_text, metadata = await self.client.get_completion_text(
                self.model_slug,
                prompt,
                system_prompt="Follow the instructions precisely and completely.",
                max_tokens=2048,
                temperature=0.0,
            )
            latency_ms = int((time.time() - start_time) * 1000)

            # IFBench requires checking specific instruction conditions
            # Full evaluation needs the IFEval checker
            instruction_ids = item.get("instruction_id_list", [])
            
            return ItemResult(
                item_id=item_id,
                item_hash=self.compute_item_hash(prompt),
                prompt=prompt,
                response=response_text.strip(),
                expected=str(instruction_ids),
                is_correct=None,  # Requires IFEval checker
                score=None,
                latency_ms=latency_ms,
                input_tokens=metadata.get("usage", {}).get("prompt_tokens"),
                output_tokens=metadata.get("usage", {}).get("completion_tokens"),
                metadata={"instruction_ids": instruction_ids},
                judge_output={"note": "IFEval checker required for scoring"},
            )

        except Exception as e:
            logger.error("IFBench evaluation failed", item_id=item_id, error=str(e))
            return ItemResult(item_id=item_id, prompt=prompt, error=str(e))

