"""MMLU-Pro benchmark adapter."""
import time
from typing import Any, AsyncIterator, Optional

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.benchmarks.utils import load_dataset_with_retry
from app.core.logging import get_logger

logger = get_logger(__name__)


@register_adapter("mmlu_pro")
class MMLUProAdapter(BenchmarkAdapter):
    """
    MMLU-Pro benchmark adapter.
    
    Uses datasets library to load MMLU-Pro and evaluates via Chutes API.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._dataset: Optional[Any] = None
        self._items: list[dict[str, Any]] = []
        self._few_shot_by_category: dict[str, list[dict[str, Any]]] = {}

    def get_name(self) -> str:
        return "mmlu_pro"

    def get_display_name(self) -> str:
        return "MMLU-Pro"

    def supports_parallel_items(self) -> bool:
        return True

    async def get_total_items(self) -> int:
        """Get total items from dataset."""
        if not self._items:
            await self.preload()
        return len(self._items)

    async def preload(self) -> None:
        """Load MMLU-Pro dataset."""
        if self._items:
            return

        try:
            import os

            logger.info("Loading MMLU-Pro dataset")
            hf_token = os.environ.get("HF_TOKEN")
            dataset = await load_dataset_with_retry(
                "TIGER-Lab/MMLU-Pro",
                split="test",
                token=hf_token,
            )
            
            self._items = []
            for i, item in enumerate(dataset):
                self._items.append({
                    "id": str(i),
                    "question": item["question"],
                    "options": item["options"],
                    "answer": item["answer"],
                    "category": item.get("category", ""),
                })
            
            logger.info(f"Loaded {len(self._items)} MMLU-Pro items")
            try:
                validation = await load_dataset_with_retry(
                    "TIGER-Lab/MMLU-Pro",
                    split="validation",
                    token=hf_token,
                )
                few_shot_by_category: dict[str, list[dict[str, Any]]] = {}
                for item in validation:
                    category = item.get("category", "")
                    if category not in few_shot_by_category:
                        few_shot_by_category[category] = []
                    if len(few_shot_by_category[category]) >= 5:
                        continue
                    few_shot_by_category[category].append(
                        {
                            "question": item["question"],
                            "options": item["options"],
                            "answer": item["answer"],
                            "category": category,
                        }
                    )
                self._few_shot_by_category = few_shot_by_category
                logger.info("Loaded MMLU-Pro few-shot examples", categories=len(few_shot_by_category))
            except Exception as e:
                logger.warning("Failed to load MMLU-Pro few-shot examples", error=str(e))
        except Exception as e:
            logger.error("Failed to load MMLU-Pro", error=str(e))
            raise

    async def enumerate_items(self) -> AsyncIterator[str]:
        """Yield all item IDs."""
        if not self._items:
            await self.preload()
        for item in self._items:
            yield item["id"]

    async def evaluate_item(self, item_id: str) -> ItemResult:
        """Evaluate a single MMLU-Pro item."""
        if not self._items:
            await self.preload()

        # Find item
        item = None
        for i in self._items:
            if i["id"] == item_id:
                item = i
                break

        if not item:
            return ItemResult(
                item_id=item_id,
                error=f"Item {item_id} not found",
            )

        def _format_options(options: list[str]) -> str:
            return "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options))

        category = item.get("category", "general knowledge")
        few_shot = self._few_shot_by_category.get(category, [])
        few_shot_prompt = ""
        if few_shot:
            blocks: list[str] = []
            for example in few_shot[:5]:
                blocks.append(
                    "Question: {question}\nOptions:\n{options}\nAnswer: {answer}".format(
                        question=example["question"],
                        options=_format_options(example["options"]),
                        answer=str(example["answer"]).strip().upper()[:1],
                    )
                )
            few_shot_prompt = "\n\n".join(blocks) + "\n\n"

        options_str = _format_options(item["options"])
        prompt = (
            "The following are multiple choice questions (with answers) about "
            f"{category}. Think step by step and then output the answer in the format "
            "of \"Answer: X\" at the end.\n\n"
            f"{few_shot_prompt}"
            f"Question: {item['question']}\n"
            f"Options:\n{options_str}\n\n"
            "Answer:"
        )

        system_prompt = None
        try:
            start_time = time.time()
            response_text, metadata = await self.client.get_completion_text(
                self.model_slug,
                prompt,
                max_tokens=4096,
                min_output_tokens=0,
                temperature=0.0,
                timeout=60,
                response_attempts=2,
            )
            latency_ms = int((time.time() - start_time) * 1000)

            # Robust handling of empty or None response
            if not response_text or response_text is None:
                item_metadata = {
                    **metadata,
                    "category": category,
                    "system_prompt": system_prompt,
                    "few_shot_count": len(few_shot[:5]),
                }
                return ItemResult(
                    item_id=item_id,
                    item_hash=self.compute_item_hash(item["question"]),
                    prompt=prompt,
                    response="",
                    expected=str(item.get("answer", ""))[:1].upper(),
                    is_correct=False,
                    score=0.0,
                    error=self.format_empty_response_error(metadata),
                    latency_ms=latency_ms,
                    input_tokens=metadata.get("usage", {}).get("prompt_tokens"),
                    output_tokens=metadata.get("usage", {}).get("completion_tokens"),
                    metadata=item_metadata,
                )

            answer_letter = self.extract_choice_letter(response_text, "ABCDEFGHIJ")
            expected = str(item.get("answer", "")).upper()[:1]
            is_correct = answer_letter == expected
            error = None
            if not answer_letter:
                error = "Could not parse answer letter"
            error = self.format_truncation_error(metadata, error if not is_correct else error)

            item_metadata = {
                **metadata,
                "category": category,
                "system_prompt": system_prompt,
                "parsed_answer": answer_letter or None,
                "few_shot_count": len(few_shot[:5]),
            }
            return ItemResult(
                item_id=item_id,
                item_hash=self.compute_item_hash(item["question"]),
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
            # Extract more details for HTTP errors
            error_detail = str(e)
            if hasattr(e, 'response'):
                try:
                    error_detail = f"{e}: {e.response.text}"
                except Exception:
                    pass
            logger.error("MMLU-Pro evaluation failed", item_id=item_id, error=error_detail)
            # Safely capture what we have
            res = locals().get("response_text", "")
            meta = locals().get("metadata") or {}
            item_metadata = {
                **meta,
                "category": item.get("category"),
                "system_prompt": system_prompt,
            }
            return ItemResult(
                item_id=item_id,
                prompt=prompt,
                response=res if res is not None else "", 
                error=error_detail,
                metadata=item_metadata,
            )
