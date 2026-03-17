"""AllenAI IFBench benchmark adapter."""
import os
import re
import time
from typing import Any, AsyncIterator, Optional

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.ifbench_eval import evaluation_lib
from app.benchmarks.registry import register_adapter
from app.benchmarks.utils import load_dataset_with_retry
from app.core.logging import get_logger

logger = get_logger(__name__)

IFBENCH_DATASET = "allenai/IFBench_test"
IFBENCH_REPEATS = 5
IFBENCH_DEFAULT_MAX_TOKENS = 8192
IFBENCH_MAX_TOKENS_CAP = 16384


def _strip_reasoning_chains(response_text: str) -> str:
    cleaned = re.sub(r"(?i)<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
    if cleaned.lower().startswith("<think>"):
        cleaned = re.sub(r"(?i)^<think>", "", cleaned).strip()
    return cleaned


@register_adapter("ifbench")
class IFBenchAdapter(BenchmarkAdapter):
    """AllenAI IFBench adapter aligned with Artificial Analysis scoring."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._items: list[dict[str, Any]] = []
        self._max_tokens: Optional[int] = None

    def get_name(self) -> str:
        return "ifbench"

    def get_display_name(self) -> str:
        return "IFBench"

    def supports_parallel_items(self) -> bool:
        return True

    async def get_total_items(self) -> int:
        if not self._items:
            await self.preload()
        return len(self._items)

    async def _resolve_max_tokens(self) -> int:
        if self._max_tokens is not None:
            return self._max_tokens

        try:
            limit = await self.client.get_model_max_output_length(self.model_slug)
        except Exception as exc:
            logger.warning(
                "Failed to resolve model max output length for IFBench",
                model=self.model_slug,
                error=str(exc),
            )
            limit = None
        if isinstance(limit, int) and limit > 0:
            self._max_tokens = min(limit, IFBENCH_MAX_TOKENS_CAP)
        else:
            self._max_tokens = IFBENCH_DEFAULT_MAX_TOKENS
        return self._max_tokens

    async def preload(self) -> None:
        """Load the AllenAI IFBench dataset."""
        if self._items:
            return

        try:
            logger.info("Loading IFBench dataset", dataset=IFBENCH_DATASET)
            hf_token = os.environ.get("HF_TOKEN")
            dataset = await load_dataset_with_retry(
                IFBENCH_DATASET,
                split="train",
                token=hf_token,
            )
            self._items = []
            for i, item in enumerate(dataset):
                self._items.append(
                    {
                        "id": str(item.get("key", i)),
                        "prompt": item.get("prompt", ""),
                        "instruction_id_list": item.get("instruction_id_list", []),
                        "kwargs": item.get("kwargs", []),
                    }
                )
            logger.info("Loaded %s IFBench items", len(self._items))
        except Exception as exc:
            logger.error("Failed to load IFBench", dataset=IFBENCH_DATASET, error=str(exc))
            self._items = []
            raise

    async def enumerate_items(self) -> AsyncIterator[str]:
        if not self._items:
            await self.preload()
        for item in self._items:
            yield item["id"]

    async def evaluate_item(self, item_id: str) -> ItemResult:
        """Evaluate one IFBench prompt with AA-style 5-repeat loose prompt scoring."""
        if not self._items:
            await self.preload()

        item = next((candidate for candidate in self._items if candidate["id"] == item_id), None)
        if not item:
            return ItemResult(item_id=item_id, error=f"Item {item_id} not found")

        prompt = item.get("prompt", "")
        instruction_ids = item.get("instruction_id_list", [])
        kwargs_list = item.get("kwargs", [])
        max_tokens = await self._resolve_max_tokens()
        repeat_metadata: list[dict[str, Any]] = []
        strict_repeat_results: list[bool] = []
        loose_repeat_results: list[bool] = []
        strict_instruction_followed = 0
        strict_instruction_total = 0
        loose_instruction_followed = 0
        loose_instruction_total = 0
        total_latency_ms = 0
        input_tokens = 0
        output_tokens = 0
        first_response = ""

        try:
            for repeat_index in range(IFBENCH_REPEATS):
                started_at = time.time()
                response_text, metadata = await self.client.get_completion_text(
                    self.model_slug,
                    prompt,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    min_output_tokens=0,
                )
                total_latency_ms += int((time.time() - started_at) * 1000)
                input_tokens += int(metadata.get("usage", {}).get("prompt_tokens") or 0)
                output_tokens += int(metadata.get("usage", {}).get("completion_tokens") or 0)

                cleaned_response = _strip_reasoning_chains(response_text or "")
                repeat_metadata.append(
                    {
                        "repeat_index": repeat_index,
                        "finish_reason": metadata.get("finish_reason"),
                        "response_attempts": metadata.get("response_attempts"),
                    }
                )
                if not cleaned_response:
                    item_metadata = {
                        **metadata,
                        "dataset": IFBENCH_DATASET,
                        "instruction_id_list": instruction_ids,
                        "repeat_index": repeat_index,
                        "repeat_count": IFBENCH_REPEATS,
                    }
                    return ItemResult(
                        item_id=item_id,
                        item_hash=self.compute_item_hash(prompt),
                        prompt=prompt,
                        response="",
                        error=self.format_empty_response_error(metadata),
                        latency_ms=total_latency_ms,
                        metadata=item_metadata,
                    )

                if repeat_index == 0:
                    first_response = cleaned_response

                example = evaluation_lib.InputExample(
                    key=int(item_id) if item_id.isdigit() else item_id,
                    instruction_id_list=instruction_ids,
                    prompt=prompt,
                    kwargs=[dict(values) for values in kwargs_list],
                )
                prompt_to_response = {prompt: cleaned_response}
                strict_output = evaluation_lib.test_instruction_following_strict(
                    example,
                    prompt_to_response,
                )
                loose_output = evaluation_lib.test_instruction_following_loose(
                    example,
                    prompt_to_response,
                )

                strict_repeat_results.append(strict_output.follow_all_instructions)
                loose_repeat_results.append(loose_output.follow_all_instructions)
                strict_instruction_followed += sum(strict_output.follow_instruction_list)
                strict_instruction_total += len(strict_output.follow_instruction_list)
                loose_instruction_followed += sum(loose_output.follow_instruction_list)
                loose_instruction_total += len(loose_output.follow_instruction_list)

            strict_prompt_accuracy = sum(strict_repeat_results) / IFBENCH_REPEATS
            loose_prompt_accuracy = sum(loose_repeat_results) / IFBENCH_REPEATS
            strict_instruction_accuracy = (
                strict_instruction_followed / strict_instruction_total if strict_instruction_total else 0.0
            )
            loose_instruction_accuracy = (
                loose_instruction_followed / loose_instruction_total if loose_instruction_total else 0.0
            )

            judge_output = {
                "strict_repeat_results": strict_repeat_results,
                "loose_repeat_results": loose_repeat_results,
                "strict_prompt_accuracy": strict_prompt_accuracy,
                "loose_prompt_accuracy": loose_prompt_accuracy,
                "strict_instruction_followed_count": strict_instruction_followed,
                "strict_instruction_total": strict_instruction_total,
                "strict_instruction_accuracy": strict_instruction_accuracy,
                "loose_instruction_followed_count": loose_instruction_followed,
                "loose_instruction_total": loose_instruction_total,
                "loose_instruction_accuracy": loose_instruction_accuracy,
            }
            item_metadata = {
                "dataset": IFBENCH_DATASET,
                "instruction_id_list": instruction_ids,
                "max_tokens": max_tokens,
                "repeat_count": IFBENCH_REPEATS,
                "repeat_metadata": repeat_metadata,
            }
            return ItemResult(
                item_id=item_id,
                item_hash=self.compute_item_hash(prompt),
                prompt=prompt,
                response=first_response,
                expected="follow_all_instructions_loose",
                is_correct=loose_repeat_results[0] if loose_repeat_results else False,
                score=loose_prompt_accuracy,
                latency_ms=total_latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                judge_output=judge_output,
                metadata=item_metadata,
            )
        except Exception as exc:
            logger.error("IFBench evaluation failed", item_id=item_id, error=str(exc))
            return ItemResult(
                item_id=item_id,
                prompt=prompt,
                response=first_response,
                error=str(exc),
                metadata={
                    "dataset": IFBENCH_DATASET,
                    "instruction_id_list": instruction_ids,
                    "repeat_count": IFBENCH_REPEATS,
                },
            )

    async def postprocess(self, results: list[ItemResult]) -> dict[str, Any]:
        strict_prompt_total = 0.0
        loose_prompt_total = 0.0
        prompt_count = 0
        strict_instruction_followed = 0
        strict_instruction_total = 0
        loose_instruction_followed = 0
        loose_instruction_total = 0

        for result in results:
            if result.error:
                continue
            judge_output = result.judge_output or {}
            strict_prompt_total += float(judge_output.get("strict_prompt_accuracy") or 0.0)
            loose_prompt_total += float(judge_output.get("loose_prompt_accuracy") or result.score or 0.0)
            strict_instruction_followed += int(
                judge_output.get("strict_instruction_followed_count") or 0
            )
            strict_instruction_total += int(judge_output.get("strict_instruction_total") or 0)
            loose_instruction_followed += int(
                judge_output.get("loose_instruction_followed_count") or 0
            )
            loose_instruction_total += int(judge_output.get("loose_instruction_total") or 0)
            prompt_count += 1

        strict_prompt_accuracy = strict_prompt_total / prompt_count if prompt_count else 0.0
        loose_prompt_accuracy = loose_prompt_total / prompt_count if prompt_count else 0.0

        return {
            "ifbench_dataset": IFBENCH_DATASET,
            "ifbench_repeats": IFBENCH_REPEATS,
            "ifbench_scoring": "loose_prompt_accuracy",
            "strict_prompt_accuracy": strict_prompt_accuracy,
            "strict_instruction_accuracy": (
                strict_instruction_followed / strict_instruction_total
                if strict_instruction_total
                else 0.0
            ),
            "loose_prompt_accuracy": loose_prompt_accuracy,
            "loose_instruction_accuracy": (
                loose_instruction_followed / loose_instruction_total
                if loose_instruction_total
                else 0.0
            ),
            "score_override": loose_prompt_accuracy,
        }
