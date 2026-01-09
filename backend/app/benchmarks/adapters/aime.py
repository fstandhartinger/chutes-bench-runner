"""AIME 2025 benchmark adapter."""
import os
import re
import time
from typing import Any, AsyncIterator, Optional

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.benchmarks.utils import load_dataset_with_retry
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

    def supports_parallel_items(self) -> bool:
        return True

    def _get_run_temperatures(self) -> list[float]:
        temps = [0.0, 0.3, 0.6]
        run_count = 8
        schedule: list[float] = []
        idx = 0
        while len(schedule) < run_count:
            schedule.append(temps[idx % len(temps)])
            idx += 1
        return schedule

    async def get_total_items(self) -> int:
        if not self._items:
            await self.preload()
        return len(self._items)

    async def preload(self) -> None:
        """Load AIME 2025 dataset."""
        if self._items:
            return

        try:
            logger.info("Loading AIME/Competition Math dataset")
            hf_token = os.environ.get("HF_TOKEN")
            dataset = await load_dataset_with_retry(
                "AI-MO/aimo-validation-aime",
                split="train",
                token=hf_token,
            )
            
            self._items = []
            for i, item in enumerate(dataset):
                problem = str(item.get("problem") or item.get("question") or "")
                answer = str(item.get("answer") or item.get("solution") or "")
                if problem:
                    self._items.append({
                        "id": str(i),
                        "problem": problem,
                        "answer": answer,
                        "level": item.get("level", item.get("type", "")),
                    })
            
            logger.info(f"Loaded {len(self._items)} AIME items")
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
            "Provide your final answer as a single integer on a new line prefixed with \"ANSWER:\".\n\n"
            f"Problem: {item['problem']}\n\n"
            "Answer:"
        )

        system_prompt = (
            "You are a test-taking assistant. Output ONLY the final answer line in the format "
            "'ANSWER: <integer>' with no extra text."
        )
        try:
            temperatures = self._get_run_temperatures()
            run_results: list[dict[str, Any]] = []
            correct_count = 0
            last_response = ""
            total_latency_ms = 0
            total_prompt_tokens = 0
            total_completion_tokens = 0

            # Clean expected answer - ensure it's not None
            expected = str(item.get("answer", "")).strip()
            if expected.startswith("\\boxed{"):
                expected = expected[7:-1]
            expected = re.sub(r"[^\d]", "", expected)

            for run_index, temp in enumerate(temperatures, start=1):
                start_time = time.time()
                response_text, metadata = await self.client.get_completion_text(
                    self.model_slug,
                    prompt,
                    system_prompt=system_prompt,
                    max_tokens=16384,
                    temperature=temp,
                )
                latency_ms = int((time.time() - start_time) * 1000)
                total_latency_ms += latency_ms
                usage = metadata.get("usage", {}) if isinstance(metadata, dict) else {}
                try:
                    total_prompt_tokens += int(usage.get("prompt_tokens") or 0)
                    total_completion_tokens += int(usage.get("completion_tokens") or 0)
                except (TypeError, ValueError):
                    pass

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
                    clean_text = re.sub(r"(?i)<think>.*?</think>", "", response_str, flags=re.DOTALL).strip()
                    numbers = re.findall(r"\b\d+\b", clean_text)
                    if numbers:
                        model_answer = numbers[-1]

                try:
                    run_correct = int(model_answer) == int(expected)
                except (ValueError, TypeError):
                    run_correct = model_answer == expected

                if run_correct:
                    correct_count += 1

                run_results.append(
                    {
                        "run": run_index,
                        "temperature": temp,
                        "answer": model_answer or None,
                        "correct": run_correct,
                        "finish_reason": metadata.get("finish_reason"),
                        "usage": usage,
                        "error": metadata.get("response_error"),
                    }
                )
                last_response = response_text or ""

            avg_latency_ms = int(total_latency_ms / len(temperatures)) if temperatures else 0
            score = correct_count / len(temperatures) if temperatures else 0.0
            is_correct = correct_count >= (len(temperatures) / 2) if temperatures else False

            error = None
            if score == 0.0:
                error = self.format_truncation_error({}, None)

            item_metadata = {
                "level": item.get("level"),
                "system_prompt": system_prompt,
                "parsed_answers": [run["answer"] for run in run_results],
                "runs": run_results,
                "run_count": len(temperatures),
                "correct_count": correct_count,
            }
            return ItemResult(
                item_id=item_id,
                item_hash=self.compute_item_hash(item["problem"]),
                prompt=prompt,
                response=last_response.strip(),
                expected=expected,
                is_correct=is_correct,
                score=score,
                latency_ms=avg_latency_ms,
                input_tokens=total_prompt_tokens,
                output_tokens=total_completion_tokens,
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
        if scores:
            mean_score = sum(scores) / len(scores)
        else:
            mean_score = 0.0
        return {
            "aime_runs": 8,
            "aime_temperatures": self._get_run_temperatures(),
            "score_override": mean_score,
        }
