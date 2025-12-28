"""τ²-Bench Telecom benchmark adapter."""
import time
from typing import Any, AsyncIterator, Optional

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.core.logging import get_logger

logger = get_logger(__name__)


@register_adapter("tau_bench_telecom")
class TauBenchTelecomAdapter(BenchmarkAdapter):
    """
    τ²-Bench Telecom adapter.
    
    Telecommunications domain agent benchmark.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._items: list[dict[str, Any]] = []

    def get_name(self) -> str:
        return "tau_bench_telecom"

    def get_display_name(self) -> str:
        return "τ²-Bench Telecom"

    def requires_setup(self) -> bool:
        return True

    def get_setup_notes(self) -> Optional[str]:
        return "τ²-Bench requires the official evaluation harness and environment setup."

    def supports_subset(self) -> bool:
        return True

    async def get_total_items(self) -> int:
        if not self._items:
            await self.preload()
        return len(self._items)

    async def preload(self) -> None:
        """Load τ²-Bench Telecom dataset."""
        if self._items:
            return

        try:
            logger.info("Loading τ²-Bench Telecom dataset")
            # Placeholder items - actual dataset requires official setup
            self._items = [
                {
                    "id": "0",
                    "scenario": "Customer wants to upgrade their mobile plan",
                    "context": "Current plan: Basic 5GB. Customer asking about unlimited options.",
                    "expected_action": "offer_plan_upgrade",
                },
                {
                    "id": "1",
                    "scenario": "Customer reporting network outage",
                    "context": "Customer in downtown area, no signal since morning.",
                    "expected_action": "check_network_status",
                },
                {
                    "id": "2",
                    "scenario": "Customer billing dispute",
                    "context": "Customer charged for international roaming they didn't use.",
                    "expected_action": "review_billing",
                },
            ]
            logger.info(f"Loaded {len(self._items)} τ²-Bench Telecom items")
        except Exception as e:
            logger.error("Failed to load τ²-Bench Telecom", error=str(e))
            self._items = []

    async def enumerate_items(self) -> AsyncIterator[str]:
        if not self._items:
            await self.preload()
        for item in self._items:
            yield item["id"]

    async def evaluate_item(self, item_id: str) -> ItemResult:
        """Evaluate a single τ²-Bench item."""
        if not self._items:
            await self.preload()

        item = next((i for i in self._items if i["id"] == item_id), None)
        if not item:
            return ItemResult(item_id=item_id, error=f"Item {item_id} not found")

        prompt = f"""You are a telecom customer service agent. Analyze the scenario and determine the best action.

Scenario: {item["scenario"]}

Context: {item["context"]}

What action should you take? Choose from:
- offer_plan_upgrade
- check_network_status
- review_billing
- transfer_to_specialist
- schedule_technician

Action:"""

        try:
            start_time = time.time()
            response_text, metadata = await self.client.get_completion_text(
                self.model_slug,
                prompt,
                system_prompt="You are a helpful telecom customer service agent. Provide the best action for each scenario.",
                max_tokens=64,
                temperature=0.0,
            )
            latency_ms = int((time.time() - start_time) * 1000)

            response_action = response_text.strip().lower().replace(" ", "_").split("\n")[0]
            expected = item.get("expected_action", "")
            is_correct = expected in response_action or response_action in expected

            return ItemResult(
                item_id=item_id,
                item_hash=self.compute_item_hash(item["scenario"]),
                prompt=prompt,
                response=response_text.strip(),
                expected=expected,
                is_correct=is_correct,
                score=1.0 if is_correct else 0.0,
                latency_ms=latency_ms,
                input_tokens=metadata.get("usage", {}).get("prompt_tokens"),
                output_tokens=metadata.get("usage", {}).get("completion_tokens"),
                judge_output={"note": "Full evaluation requires τ²-Bench harness"},
            )

        except Exception as e:
            logger.error("τ²-Bench evaluation failed", item_id=item_id, error=str(e))
            return ItemResult(item_id=item_id, prompt=prompt, error=str(e))


