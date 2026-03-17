"""IFBench (Instruction Following) benchmark adapter."""
import os
import time
from typing import Any, AsyncIterator, Optional

import nltk

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.ifeval import evaluation_lib
from app.benchmarks.registry import register_adapter
from app.benchmarks.utils import load_dataset_with_retry
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
        self._nltk_ready = False

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

    def _ensure_nltk(self) -> None:
        if self._nltk_ready:
            return
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        try:
            nltk.data.find("tokenizers/punkt_tab/english")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        self._nltk_ready = True

    async def preload(self) -> None:
        """Load IFBench dataset."""
        if self._items:
            return

        try:
            logger.info("Loading IFBench dataset")
            hf_token = os.environ.get("HF_TOKEN")
            dataset = await load_dataset_with_retry(
                "google/IFEval",
                split="train",
                token=hf_token,
            )
            self._items = []
            for i, item in enumerate(dataset):
                self._items.append(
                    {
                        "id": str(i),
                        "prompt": item.get("prompt", ""),
                        "instruction_id_list": item.get("instruction_id_list", []),
                        "kwargs": item.get("kwargs", []),
                    }
                )
            logger.info("Loaded %s IFBench items", len(self._items))
        except Exception as e:
            logger.error("Failed to load IFBench", error=str(e))
            self._items = []
            raise

    async def enumerate_items(self) -> AsyncIterator[str]:
        if not self._items:
            await self.preload()
        for item in self._items:
            yield item["id"]

    async def evaluate_item(self, item_id: str) -> ItemResult:
        """Evaluate a single IFBench item using official IFEval checks."""
        if not self._items:
            await self.preload()
        self._ensure_nltk()

        item = next((i for i in self._items if i["id"] == item_id), None)
        if not item:
            return ItemResult(item_id=item_id, error=f"Item {item_id} not found")

        prompt = item.get("prompt", "")
        instruction_ids = item.get("instruction_id_list", [])
        kwargs_list = item.get("kwargs", [])

        try:
            start_time = time.time()
            response_text, metadata = await self.client.get_completion_text(
                self.model_slug,
                prompt,
                temperature=0.0,
                max_tokens=4096,
                min_output_tokens=0,
            )
            latency_ms = int((time.time() - start_time) * 1000)

            if not response_text:
                item_metadata = {
                    **metadata,
                    "instruction_id_list": instruction_ids,
                }
                return ItemResult(
                    item_id=item_id,
                    item_hash=self.compute_item_hash(prompt),
                    prompt=prompt,
                    response="",
                    error=self.format_empty_response_error(metadata),
                    latency_ms=latency_ms,
                    metadata=item_metadata,
                )

            inp = evaluation_lib.InputExample(
                key=int(item_id),
                instruction_id_list=instruction_ids,
                prompt=prompt,
                kwargs=kwargs_list,
            )
            prompt_to_response = {prompt: response_text}
            strict_output = evaluation_lib.test_instruction_following_strict(inp, prompt_to_response)
            loose_output = evaluation_lib.test_instruction_following_loose(inp, prompt_to_response)

            item_metadata = {
                **metadata,
                "instruction_id_list": instruction_ids,
                "strict_follow_instruction_list": strict_output.follow_instruction_list,
                "loose_follow_instruction_list": loose_output.follow_instruction_list,
            }
            return ItemResult(
                item_id=item_id,
                item_hash=self.compute_item_hash(prompt),
                prompt=prompt,
                response=response_text.strip(),
                expected="follow_all_instructions",
                is_correct=strict_output.follow_all_instructions,
                score=1.0 if strict_output.follow_all_instructions else 0.0,
                latency_ms=latency_ms,
                input_tokens=metadata.get("usage", {}).get("prompt_tokens"),
                output_tokens=metadata.get("usage", {}).get("completion_tokens"),
                judge_output={
                    "strict_follow_all_instructions": strict_output.follow_all_instructions,
                    "strict_follow_instruction_list": strict_output.follow_instruction_list,
                    "loose_follow_all_instructions": loose_output.follow_all_instructions,
                    "loose_follow_instruction_list": loose_output.follow_instruction_list,
                },
                metadata=item_metadata,
            )
        except Exception as e:
            logger.error("IFBench evaluation failed", item_id=item_id, error=str(e))
            meta = locals().get("metadata") or {}
            item_metadata = {
                **meta,
                "instruction_id_list": instruction_ids,
            }
            return ItemResult(
                item_id=item_id,
                prompt=prompt,
                response=locals().get("response_text", "") or "",
                error=str(e),
                metadata=item_metadata,
            )

    async def postprocess(self, results: list[ItemResult]) -> dict[str, Any]:
        strict_prompt_total = 0
        strict_prompt_correct = 0
        loose_prompt_total = 0
        loose_prompt_correct = 0
        strict_instruction_total = 0
        strict_instruction_correct = 0
        loose_instruction_total = 0
        loose_instruction_correct = 0

        for result in results:
            if result.error:
                continue
            judge_output = result.judge_output or {}
            strict_list = list(judge_output.get("strict_follow_instruction_list") or [])
            loose_list = list(judge_output.get("loose_follow_instruction_list") or [])
            strict_prompt = judge_output.get("strict_follow_all_instructions")
            loose_prompt = judge_output.get("loose_follow_all_instructions")

            if isinstance(strict_prompt, bool):
                strict_prompt_total += 1
                if strict_prompt:
                    strict_prompt_correct += 1
            if isinstance(loose_prompt, bool):
                loose_prompt_total += 1
                if loose_prompt:
                    loose_prompt_correct += 1

            strict_instruction_total += len(strict_list)
            strict_instruction_correct += sum(1 for followed in strict_list if followed)
            loose_instruction_total += len(loose_list)
            loose_instruction_correct += sum(1 for followed in loose_list if followed)

        return {
            "strict_prompt_accuracy": (
                strict_prompt_correct / strict_prompt_total if strict_prompt_total else 0.0
            ),
            "strict_instruction_accuracy": (
                strict_instruction_correct / strict_instruction_total
                if strict_instruction_total
                else 0.0
            ),
            "loose_prompt_accuracy": (
                loose_prompt_correct / loose_prompt_total if loose_prompt_total else 0.0
            ),
            "loose_instruction_accuracy": (
                loose_instruction_correct / loose_instruction_total if loose_instruction_total else 0.0
            ),
        }
