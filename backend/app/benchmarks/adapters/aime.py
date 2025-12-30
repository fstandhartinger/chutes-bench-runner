"""AIME 2025 benchmark adapter."""
import re
import time
from typing import Any, AsyncIterator, Optional

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
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

    async def get_total_items(self) -> int:
        if not self._items:
            await self.preload()
        return len(self._items)

    def requires_setup(self) -> bool:
        return True

    def get_setup_notes(self) -> Optional[str]:
        return "AIME dataset requires access to competition math datasets. Set HF_TOKEN if using gated datasets."

    async def preload(self) -> None:
        """Load AIME 2025 dataset."""
        if self._items:
            return

        try:
            from datasets import load_dataset
            import os

            logger.info("Loading AIME/Competition Math dataset")
            hf_token = os.environ.get("HF_TOKEN")
            dataset = None
            
            # Try multiple sources in order of preference
            sources = [
                ("AI-MO/aimo-validation-aime", "train", {"token": hf_token} if hf_token else {}),
                ("lighteval/MATH", "test", {}),
                ("hendrycks/competition_math", "test", {}),
            ]
            
            for source_name, split, kwargs in sources:
                try:
                    logger.info(f"Trying to load from {source_name}")
                    dataset = load_dataset(source_name, split=split, **kwargs)
                    break
                except Exception as e:
                    logger.warning(f"Could not load {source_name}: {e}")
                    continue
            
            if dataset is None:
                logger.error("No AIME dataset source available")
                self._items = []
                return
            
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

        prompt = f"""Solve the following math competition problem. AIME answers are always integers from 0 to 999.
Show your reasoning step by step, then provide your final answer as a single integer on a new line prefixed with "ANSWER:".

Example:
Problem: What is 2+2?
Solution:
2+2 is 4.
ANSWER: 4

Problem: {item["problem"]}

Solution:"""

        system_prompt = "You are an expert mathematician. Solve the problem step by step. Always end your response with 'ANSWER: ' followed by the integer result."
        try:
            start_time = time.time()
            response_text, metadata = await self.client.get_completion_text(
                self.model_slug,
                prompt,
                system_prompt=system_prompt,
                max_tokens=16384,  # Increased for complex AIME problems
                temperature=0.0,
            )
            latency_ms = int((time.time() - start_time) * 1000)

            if not response_text or response_text is None:
                item_metadata = {
                    **metadata,
                    "level": item.get("level"),
                    "system_prompt": system_prompt,
                }
                return ItemResult(
                    item_id=item_id,
                    item_hash=self.compute_item_hash(item["problem"]),
                    prompt=prompt,
                    response="",
                    expected=str(item.get("answer", "")),
                    is_correct=False,
                    score=0.0,
                    error=self.format_empty_response_error(metadata),
                    latency_ms=latency_ms,
                    input_tokens=metadata.get("usage", {}).get("prompt_tokens"),
                    output_tokens=metadata.get("usage", {}).get("completion_tokens"),
                    metadata=item_metadata,
                )

            # Extract answer - try multiple patterns
            # Ensure response_text is a string
            response_str = str(response_text)
            model_answer = ""
            
            # 1. Look for ANSWER: XXX
            answer_match = re.search(r"ANSWER:\s*(\d+)", response_str, re.IGNORECASE)
            if answer_match:
                model_answer = answer_match.group(1)
            
            # 2. Look for \boxed{XXX}
            if not model_answer:
                boxed_match = re.search(r"\\boxed\{(\d+)\}", response_str)
                if boxed_match:
                    model_answer = boxed_match.group(1)
            
            # 3. Look for the last integer in the response if it's short
            if not model_answer:
                # Clean response of thinking tags for better extraction
                clean_text = re.sub(r"(?i)<think>.*?</think>", "", response_str, flags=re.DOTALL).strip()
                numbers = re.findall(r"\b\d+\b", clean_text)
                if numbers:
                    model_answer = numbers[-1]
            
            # Clean expected answer - ensure it's not None
            expected = str(item.get("answer", "")).strip()
            if expected.startswith("\\boxed{"):
                expected = expected[7:-1]
            expected = re.sub(r"[^\d]", "", expected)
            
            # Compare as integers if possible
            try:
                is_correct = int(model_answer) == int(expected)
            except (ValueError, TypeError):
                is_correct = model_answer == expected

            error = None
            if not is_correct:
                error = self.format_truncation_error(metadata, None)

            item_metadata = {
                **metadata,
                "level": item.get("level"),
                "system_prompt": system_prompt,
            }
            return ItemResult(
                item_id=item_id,
                item_hash=self.compute_item_hash(item["problem"]),
                prompt=prompt,
                response=response_text.strip(),
                expected=expected,
                is_correct=is_correct,
                score=1.0 if is_correct else 0.0,
                latency_ms=latency_ms,
                input_tokens=metadata.get("usage", {}).get("prompt_tokens"),
                output_tokens=metadata.get("usage", {}).get("completion_tokens"),
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
