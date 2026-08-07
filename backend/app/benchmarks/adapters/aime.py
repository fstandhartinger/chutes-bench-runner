"""AIME 2025 benchmark adapter."""

import os
import re
import time
from collections.abc import AsyncIterator
from typing import Any

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.benchmarks.utils import load_dataset_with_retry
from app.core.logging import get_logger

logger = get_logger(__name__)

AIME_2025_DATASET = "opencompass/AIME2025"
AIME_2025_REVISION = "a6ad95f611d72cf628a80b58bd0432ef6638f958"
AIME_2025_EXAMS = ("AIME2025-I", "AIME2025-II")
AIME_2025_TASK_IDS = tuple(
    f"2025-{exam}-{problem:02d}" for exam in ("I", "II") for problem in range(1, 16)
)


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

    def get_item_timeout_seconds(self, item_id: str | None = None) -> int | None:
        return 300

    async def get_total_items(self) -> int:
        if not self._items:
            await self.preload()
        return len(self._items)

    async def preload(self) -> None:
        """Load the two pinned 15-problem AIME 2025 exams."""
        if self._items:
            return

        try:
            logger.info(
                "Loading pinned AIME 2025 dataset",
                dataset=AIME_2025_DATASET,
                revision=AIME_2025_REVISION,
            )
            hf_token = os.environ.get("HF_TOKEN")
            self._items = []
            for exam_config in AIME_2025_EXAMS:
                dataset = await load_dataset_with_retry(
                    AIME_2025_DATASET,
                    exam_config,
                    split="test",
                    revision=AIME_2025_REVISION,
                    token=hf_token,
                )
                exam = exam_config.rsplit("-", 1)[-1]
                if len(dataset) != 15:
                    raise RuntimeError(
                        f"{exam_config} identity mismatch at {AIME_2025_REVISION}: "
                        f"expected 15 problems, loaded {len(dataset)}"
                    )
                for problem_number, item in enumerate(dataset, start=1):
                    task_id = f"2025-{exam}-{problem_number:02d}"
                    problem = str(item.get("question") or item.get("problem") or "")
                    answer = str(item.get("answer") or item.get("solution") or "")
                    if not problem or not answer:
                        raise RuntimeError(
                            f"{task_id} is missing its problem or answer at {AIME_2025_REVISION}"
                        )
                    self._items.append(
                        {
                            "id": str(len(self._items)),
                            "task_id": task_id,
                            "problem": problem,
                            "answer": answer,
                            "level": f"AIME 2025 {exam}",
                            "dataset_repository": AIME_2025_DATASET,
                            "dataset_revision": AIME_2025_REVISION,
                        }
                    )

            loaded_ids = tuple(item["task_id"] for item in self._items)
            if loaded_ids != AIME_2025_TASK_IDS:
                raise RuntimeError(
                    "AIME 2025 identity mismatch: expected the ordered 30-problem "
                    f"I/II manifest, loaded {loaded_ids}"
                )
            logger.info("Loaded pinned AIME 2025 items", items=len(self._items))
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
            'Provide your final answer as a single integer on a new line prefixed with "ANSWER:".\n\n'
            f"Problem: {item['problem']}\n\n"
            "Answer:"
        )

        system_prompt = (
            "You are a test-taking assistant. Output ONLY the final answer line in the format "
            "'ANSWER: <integer>' with no extra text."
        )
        try:
            # Clean expected answer - ensure it's not None
            expected = str(item.get("answer", "")).strip()
            if expected.startswith("\\boxed{"):
                expected = expected[7:-1]
            expected = re.sub(r"[^\d]", "", expected)
            start_time = time.time()
            response_text, metadata = await self.client.get_completion_text(
                self.model_slug,
                prompt,
                system_prompt=system_prompt,
                max_tokens=64,
                min_output_tokens=0,
                temperature=0.0,
            )
            latency_ms = int((time.time() - start_time) * 1000)
            usage = metadata.get("usage", {}) if isinstance(metadata, dict) else {}

            model_answer = ""
            response_str = str(response_text or "")
            answer_matches = re.findall(r"ANSWER:\s*(\d+)", response_str, re.IGNORECASE)
            if answer_matches:
                model_answer = answer_matches[-1]

            if not model_answer:
                boxed_match = re.search(r"\\boxed\{(\d+)\}", response_str)
                if boxed_match:
                    model_answer = boxed_match.group(1)

            if not model_answer:
                clean_text = re.sub(
                    r"(?i)<think>.*?</think>", "", response_str, flags=re.DOTALL
                ).strip()
                numbers = re.findall(r"\b\d+\b", clean_text)
                if numbers:
                    model_answer = numbers[-1]

            try:
                is_correct = int(model_answer) == int(expected)
            except (ValueError, TypeError):
                is_correct = model_answer == expected

            score = 1.0 if is_correct else 0.0

            error = None
            if score == 0.0:
                error = self.format_truncation_error({}, None)

            item_metadata = {
                "level": item.get("level"),
                "system_prompt": system_prompt,
                "parsed_answer": model_answer or None,
                "finish_reason": metadata.get("finish_reason"),
                "usage": usage,
            }
            return ItemResult(
                item_id=item_id,
                item_hash=self.compute_item_hash(item["problem"]),
                prompt=prompt,
                response=response_str.strip(),
                expected=expected,
                is_correct=is_correct,
                score=score,
                latency_ms=latency_ms,
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
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

    async def postprocess(self, results: list[ItemResult]) -> dict[str, Any]:
        scores = [result.score for result in results if result.score is not None]
        mean_score = sum(scores) / len(scores) if scores else 0.0
        return {
            "aime_runs": 1,
            "aime_temperatures": [0.0],
            "score_override": mean_score,
        }
