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
            logger.warning("τ²-Bench Telecom dataset not configured")
            self._items = []
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

Examples:
Scenario: User wants to cancel their subscription.
Action: cancel_subscription

Scenario: User reports slow internet speeds.
Action: check_network_status

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
                system_prompt="You are a telecom customer service agent. Output ONLY the action name from the provided list. No explanation, no thinking.",
                max_tokens=512,  # Allow for potential <think> blocks
                temperature=0.0,
            )
            latency_ms = int((time.time() - start_time) * 1000)

            if not response_text:
                item_metadata = {
                    **metadata,
                    "scenario_id": item.get("scenario_id"),
                }
                return ItemResult(
                    item_id=item_id,
                    item_hash=self.compute_item_hash(item["scenario"]),
                    prompt=prompt,
                    response="",
                    error=self.format_empty_response_error(metadata),
                    latency_ms=latency_ms,
                    metadata=item_metadata,
                )

            # Extract action more robustly
            clean_response = response_text.strip().lower()
            if "</think>" in clean_response:
                clean_response = clean_response.split("</think>")[-1].strip()
            
            # Look for the predefined action names in the response
            actions = [
                "offer_plan_upgrade",
                "check_network_status",
                "review_billing",
                "transfer_to_specialist",
                "schedule_technician"
            ]
            
            response_action = ""
            for action in actions:
                if action in clean_response or action.replace("_", " ") in clean_response:
                    response_action = action
                    break
            
            if not response_action:
                # Fallback to the old method
                response_action = clean_response.replace(" ", "_").split("\n")[0]
            
            expected = item.get("expected_action", "")
            is_correct = expected == response_action or expected in response_action

            item_metadata = {
                **metadata,
                "scenario_id": item.get("scenario_id"),
            }
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
                metadata=item_metadata,
                judge_output={"note": "Full evaluation requires τ²-Bench harness"},
            )

        except Exception as e:
            logger.error("τ²-Bench evaluation failed", item_id=item_id, error=str(e))
            meta = locals().get("metadata") or {}
            item_metadata = {
                **meta,
                "scenario_id": item.get("scenario_id"),
            }
            return ItemResult(
                item_id=item_id, 
                prompt=prompt, 
                response=locals().get("response_text", ""), 
                error=str(e),
                metadata=item_metadata,
            )



