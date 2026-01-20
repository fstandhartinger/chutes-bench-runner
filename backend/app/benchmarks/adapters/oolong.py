"""OOLONG benchmark adapter.

Based on the OOLONG paper (arxiv:2511.02817), used in the RLM paper.
Tests long-context reasoning and aggregation capabilities requiring
semantic classification and aggregation across nearly all dataset entries.

Uses the oolongbench/oolong-synth dataset from HuggingFace.

This adapter efficiently loads items by iterating through the streaming dataset
once and caching the items needed for evaluation.
"""
import asyncio
import hashlib
import os
import random
import time
from typing import Any, AsyncIterator, Optional

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.core.logging import get_logger

logger = get_logger(__name__)

# Total items in the oolong-synth test split (from dataset info)
OOLONG_SYNTH_TOTAL_ITEMS = 5200


def _compute_numeric_score(expected: str, predicted: str) -> float:
    """
    Compute score for numeric answers using 0.75^|y-ŷ| formula.

    From RLM paper: "Numerical answers as score(ŷ)=0.75^|y-ŷ|"
    """
    try:
        expected_num = float(expected)
        predicted_num = float(predicted.strip())
        diff = abs(expected_num - predicted_num)
        return 0.75 ** diff
    except (ValueError, TypeError):
        return 0.0


def _is_exact_match(expected: str, predicted: str) -> bool:
    """Check for exact match (case-insensitive, stripped)."""
    return expected.strip().lower() == predicted.strip().lower()


def _extract_answer(response: str) -> str:
    """Extract the answer from model response, handling common formats."""
    if not response:
        return ""

    import re

    # Remove thinking blocks
    cleaned = re.sub(r"(?i)<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    if not cleaned:
        return ""

    # Look for explicit "Answer: X" pattern
    answer_match = re.search(r"(?im)^\s*(?:answer|final answer)\s*[:\-]\s*(.+?)$", cleaned)
    if answer_match:
        return answer_match.group(1).strip()

    # Look for "The answer is X" pattern
    is_match = re.search(r"(?i)\b(?:the answer is|answer is)\s*[:\-]?\s*(.+?)(?:\.|$)", cleaned)
    if is_match:
        return is_match.group(1).strip()

    # Return last line if short enough (likely just the answer)
    lines = [l.strip() for l in cleaned.splitlines() if l.strip()]
    if lines:
        last_line = lines[-1]
        # If last line is reasonably short, it's likely the answer
        if len(last_line) < 100:
            return last_line

    # Return the full cleaned response
    return cleaned


def _load_items_by_index_sync(
    target_indices: set[int],
    hf_token: Optional[str],
) -> dict[str, dict[str, Any]]:
    """
    Load specific items by index using non-streaming mode with direct indexing.

    Non-streaming mode allows O(1) access to items by index, which is much faster
    than iterating through a streaming dataset. The dataset is cached by HuggingFace
    after first download.
    """
    from datasets import load_dataset

    logger.info(
        "Loading OOLONG dataset (non-streaming, cached by HuggingFace)",
        target_count=len(target_indices),
    )

    # Load in non-streaming mode - HuggingFace will cache it locally
    dataset = load_dataset(
        "oolongbench/oolong-synth",
        split="test",
        token=hf_token,
    )

    logger.info("OOLONG dataset loaded, extracting items by index")

    cache: dict[str, dict[str, Any]] = {}

    for i, idx in enumerate(sorted(target_indices)):
        item = dataset[idx]
        cache[str(idx)] = {
            "id": str(idx),
            "context_window_text": item["context_window_text"],
            "question": item["question"],
            "answer": str(item["answer"]),
            "answer_type": item.get("answer_type", ""),
            "task": item.get("task", ""),
            "task_group": item.get("task_group", ""),
            "context_len": item.get("context_len", 0),
            "dataset": item.get("dataset", ""),
            "num_labels": item.get("num_labels", 0),
        }

        if (i + 1) % 100 == 0:
            logger.info(
                "OOLONG item extraction progress",
                extracted=i + 1,
                target=len(target_indices),
            )

    logger.info("OOLONG items extracted", cached=len(cache), target=len(target_indices))
    return cache


@register_adapter("oolong")
class OolongAdapter(BenchmarkAdapter):
    """
    OOLONG benchmark adapter.

    Evaluates long-context reasoning and aggregation capabilities.
    Tasks require examining and transforming chunks of input semantically,
    then aggregating these chunks to form a final answer.

    Processing costs scale linearly with input length.

    Uses streaming mode efficiently by iterating once and caching needed items.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._item_cache: dict[str, dict[str, Any]] = {}
        self._preloaded: bool = False

    def get_name(self) -> str:
        return "oolong"

    def get_display_name(self) -> str:
        return "OOLONG"

    def supports_parallel_items(self) -> bool:
        # Enable parallel items since we pre-cache everything
        return True

    async def get_total_items(self) -> int:
        """Get total items from dataset (known constant)."""
        return OOLONG_SYNTH_TOTAL_ITEMS

    async def preload(self) -> None:
        """No-op - items are loaded in get_items_for_evaluation."""
        pass

    async def enumerate_items(self) -> AsyncIterator[str]:
        """Yield all item IDs."""
        for i in range(OOLONG_SYNTH_TOTAL_ITEMS):
            yield str(i)

    async def get_items_for_evaluation(
        self,
        subset_pct: int,
        seed: str,
        subset_count: Optional[int] = None,
    ) -> tuple[int, list[str]]:
        """
        Override to efficiently load only the items we need.

        Instead of loading items one-by-one with random access (which is O(n^2)
        on streaming datasets), we:
        1. Determine which item IDs we need based on subset sampling
        2. Iterate through the streaming dataset once, caching only needed items
        """
        total_items = OOLONG_SYNTH_TOTAL_ITEMS

        # Generate all item IDs
        all_item_ids = [str(i) for i in range(total_items)]

        # Get the subset we'll evaluate
        items_to_evaluate = self.get_deterministic_subset(
            all_item_ids, subset_pct, seed, subset_count
        )

        # Convert to indices for efficient lookup
        target_indices = {int(item_id) for item_id in items_to_evaluate}

        # Load items using non-streaming mode with direct indexing
        if not self._preloaded:
            hf_token = os.environ.get("HF_TOKEN")
            self._item_cache = await asyncio.to_thread(
                _load_items_by_index_sync,
                target_indices,
                hf_token,
            )
            self._preloaded = True
            logger.info(
                "OOLONG preload complete",
                cached_items=len(self._item_cache),
                target_items=len(items_to_evaluate),
            )

        return total_items, items_to_evaluate

    async def _get_item(self, item_id: str) -> Optional[dict[str, Any]]:
        """Get item by ID from cache."""
        return self._item_cache.get(item_id)

    async def evaluate_item(self, item_id: str) -> ItemResult:
        """Evaluate a single OOLONG item."""
        item = await self._get_item(item_id)

        if not item:
            return ItemResult(
                item_id=item_id,
                error=f"Item {item_id} not found",
            )

        # Construct prompt following OOLONG format
        # Input: context_window_text + "\n" + question
        prompt = (
            f"{item['context_window_text']}\n\n"
            f"Question: {item['question']}\n\n"
            f"Analyze the text above and provide your answer. "
            f"Only give the final answer without explanation."
        )

        try:
            start_time = time.time()
            response_text, metadata = await self.client.get_completion_text(
                self.model_slug,
                prompt,
                max_tokens=512,
                min_output_tokens=0,
                temperature=0.0,
                timeout=180,  # Longer timeout for long context
                response_attempts=2,
            )
            latency_ms = int((time.time() - start_time) * 1000)

            if not response_text or response_text is None:
                item_metadata = {
                    **metadata,
                    "task": item["task"],
                    "task_group": item["task_group"],
                    "answer_type": item["answer_type"],
                    "context_len": item["context_len"],
                }
                return ItemResult(
                    item_id=item_id,
                    item_hash=self.compute_item_hash(item["question"]),
                    prompt=prompt,
                    response="",
                    expected=item["answer"],
                    is_correct=False,
                    score=0.0,
                    error=self.format_empty_response_error(metadata),
                    latency_ms=latency_ms,
                    input_tokens=metadata.get("usage", {}).get("prompt_tokens"),
                    output_tokens=metadata.get("usage", {}).get("completion_tokens"),
                    metadata=item_metadata,
                )

            # Extract answer from response
            extracted_answer = _extract_answer(response_text)
            expected = item["answer"]
            answer_type = item["answer_type"]

            # Score based on answer type
            # From RLM paper: "Numerical answers as score(ŷ)=0.75^|y-ŷ| and other answers as exact match"
            if answer_type == "NUMERIC":
                score = _compute_numeric_score(expected, extracted_answer)
                is_correct = score >= 0.75  # Consider correct if within ~1 unit
            else:
                is_correct = _is_exact_match(expected, extracted_answer)
                score = 1.0 if is_correct else 0.0

            item_metadata = {
                **metadata,
                "task": item["task"],
                "task_group": item["task_group"],
                "answer_type": answer_type,
                "context_len": item["context_len"],
                "extracted_answer": extracted_answer,
            }

            return ItemResult(
                item_id=item_id,
                item_hash=self.compute_item_hash(item["question"]),
                prompt=prompt,
                response=response_text.strip(),
                expected=expected,
                is_correct=is_correct,
                score=score,
                latency_ms=latency_ms,
                input_tokens=metadata.get("usage", {}).get("prompt_tokens"),
                output_tokens=metadata.get("usage", {}).get("completion_tokens"),
                metadata=item_metadata,
            )

        except Exception as e:
            error_detail = str(e)
            if hasattr(e, 'response'):
                try:
                    error_detail = f"{e}: {e.response.text}"
                except Exception:
                    pass
            logger.error("OOLONG evaluation failed", item_id=item_id, error=error_detail)
            return ItemResult(
                item_id=item_id,
                prompt=prompt,
                response="",
                error=error_detail,
                metadata={
                    "task": item.get("task"),
                    "task_group": item.get("task_group"),
                    "context_len": item.get("context_len"),
                },
            )

    async def postprocess(self, results: list[ItemResult]) -> dict[str, Any]:
        """Compute metrics broken down by task type and context length."""
        metrics: dict[str, Any] = {}
        scores: list[float] = []

        # Group by task group
        by_task_group: dict[str, list[ItemResult]] = {}
        # Group by answer type
        by_answer_type: dict[str, list[ItemResult]] = {}
        # Group by context length (bucketed)
        by_context_len: dict[str, list[ItemResult]] = {}

        for result in results:
            if result.score is not None:
                scores.append(result.score)
            if result.metadata:
                task_group = result.metadata.get("task_group", "unknown")
                answer_type = result.metadata.get("answer_type", "unknown")
                context_len = result.metadata.get("context_len", 0)

                if task_group not in by_task_group:
                    by_task_group[task_group] = []
                by_task_group[task_group].append(result)

                if answer_type not in by_answer_type:
                    by_answer_type[answer_type] = []
                by_answer_type[answer_type].append(result)

                # Bucket context length
                if context_len < 10000:
                    bucket = "small"
                elif context_len < 50000:
                    bucket = "medium"
                else:
                    bucket = "large"

                if bucket not in by_context_len:
                    by_context_len[bucket] = []
                by_context_len[bucket].append(result)

        # Accuracy by task group
        for task_group, group_results in sorted(by_task_group.items()):
            correct = sum(1 for r in group_results if r.is_correct)
            total = len(group_results)
            metrics[f"accuracy_{task_group}"] = correct / total if total > 0 else 0.0

        # Accuracy by answer type
        for answer_type, type_results in sorted(by_answer_type.items()):
            correct = sum(1 for r in type_results if r.is_correct)
            total = len(type_results)
            # For numeric, use average score instead
            if answer_type == "NUMERIC":
                avg_score = sum(r.score or 0.0 for r in type_results) / total if total > 0 else 0.0
                metrics[f"avg_score_{answer_type}"] = avg_score
            else:
                metrics[f"accuracy_{answer_type}"] = correct / total if total > 0 else 0.0

        # Accuracy by context length bucket
        for bucket, bucket_results in sorted(by_context_len.items()):
            correct = sum(1 for r in bucket_results if r.is_correct)
            total = len(bucket_results)
            metrics[f"accuracy_ctx_{bucket}"] = correct / total if total > 0 else 0.0

        # Overall average score (numeric answers use 0.75^diff)
        if scores:
            avg_score = sum(scores) / len(scores)
            metrics["avg_score_overall"] = avg_score
            metrics["score_override"] = avg_score

        return metrics
