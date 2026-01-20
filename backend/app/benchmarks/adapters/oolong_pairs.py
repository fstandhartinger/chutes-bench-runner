"""OOLONG-Pairs benchmark adapter.

Based on the RLM paper (arxiv:2512.24601v1), this is a custom extension
of OOLONG that requires aggregating pairs of chunks to construct the final answer.

Processing costs scale quadratically with input length due to pairwise relationships.
Uses the oolongbench/oolong-real dataset from HuggingFace with D&D transcripts.

IMPORTANT: This adapter uses streaming mode to avoid memory issues.
"""
import asyncio
import os
import re
import time
from typing import Any, AsyncIterator, Optional

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.core.logging import get_logger

logger = get_logger(__name__)

# Total items in the oolong-real dnd test split (from dataset info)
OOLONG_PAIRS_TOTAL_ITEMS = 6072


def _compute_f1_score(expected: str, predicted: str) -> tuple[float, float, float]:
    """
    Compute F1 score between expected and predicted answers.

    Returns (precision, recall, f1)
    """
    # Tokenize on whitespace and punctuation
    def tokenize(text: str) -> set[str]:
        if not text:
            return set()
        # Lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text.lower())
        return set(tokens)

    expected_tokens = tokenize(expected)
    predicted_tokens = tokenize(predicted)

    if not expected_tokens or not predicted_tokens:
        return (0.0, 0.0, 0.0)

    # True positives: tokens in both
    tp = len(expected_tokens & predicted_tokens)

    # Precision: tp / predicted tokens
    precision = tp / len(predicted_tokens) if predicted_tokens else 0.0

    # Recall: tp / expected tokens
    recall = tp / len(expected_tokens) if expected_tokens else 0.0

    # F1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return (precision, recall, f1)


def _extract_answer(response: str) -> str:
    """Extract the answer from model response, handling common formats."""
    if not response:
        return ""

    # Remove thinking blocks
    cleaned = re.sub(r"(?i)<think>.*?</think>", "", response, flags=re.DOTALL).strip()
    if not cleaned:
        return ""

    # Look for explicit "Answer: X" pattern
    answer_match = re.search(r"(?im)^\s*(?:answer|final answer)\s*[:\-]\s*(.+?)$", cleaned, re.MULTILINE)
    if answer_match:
        return answer_match.group(1).strip()

    # Look for "The answer is X" pattern
    is_match = re.search(r"(?i)\b(?:the answer is|answer is)\s*[:\-]?\s*(.+?)(?:\.|$)", cleaned)
    if is_match:
        return is_match.group(1).strip()

    # Return last paragraph if reasonably sized
    paragraphs = [p.strip() for p in cleaned.split('\n\n') if p.strip()]
    if paragraphs:
        last_para = paragraphs[-1]
        if len(last_para) < 500:
            return last_para

    # Return the full cleaned response
    return cleaned


async def _load_streaming_dataset() -> Any:
    """Load oolong-real dnd dataset in streaming mode."""
    from datasets import load_dataset
    hf_token = os.environ.get("HF_TOKEN")
    return await asyncio.to_thread(
        load_dataset,
        "oolongbench/oolong-real",
        "dnd",
        split="test",
        streaming=True,
        token=hf_token,
    )


def _get_item_by_index_sync(dataset: Any, target_idx: int) -> Optional[dict[str, Any]]:
    """Get a specific item from streaming dataset by index (synchronous)."""
    from itertools import islice
    try:
        item = next(islice(dataset, target_idx, target_idx + 1), None)
        if item:
            question = item["question"].lower()
            is_pairwise = any(kw in question for kw in [
                "between", "relationship", "interact", "compare",
                "versus", "vs", "together", "both", "each other",
                "difference", "similar", "conflict", "alliance"
            ])
            return {
                "id": str(target_idx),
                "context_window_text": item["context_window_text"],
                "question": item["question"],
                "answer": str(item["answer"]),
                "question_type": item.get("question_type", ""),
                "campaign": item.get("campaign", ""),
                "episodes": item.get("episodes", []),
                "is_pairwise": is_pairwise,
            }
    except Exception:
        pass
    return None


@register_adapter("oolong_pairs")
class OolongPairsAdapter(BenchmarkAdapter):
    """
    OOLONG-Pairs benchmark adapter.

    Evaluates long-context reasoning requiring pairwise analysis.
    Tasks require aggregating pairs of chunks from the input
    to construct the final answer.

    Processing costs scale quadratically with input length.

    Uses the D&D transcripts from oolongbench/oolong-real dataset,
    which contains questions about interactions and relationships
    between characters (inherently pairwise).

    Uses streaming mode to avoid memory issues.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._dataset: Optional[Any] = None
        self._item_cache: dict[str, dict[str, Any]] = {}

    def get_name(self) -> str:
        return "oolong_pairs"

    def get_display_name(self) -> str:
        return "OOLONG-Pairs"

    def supports_parallel_items(self) -> bool:
        # Disable parallel items due to streaming dataset limitations
        return False

    def get_item_timeout_seconds(self) -> Optional[int]:
        """Longer timeout for quadratic complexity tasks."""
        return 300  # 5 minutes

    async def get_total_items(self) -> int:
        """Get total items from dataset (known constant)."""
        return OOLONG_PAIRS_TOTAL_ITEMS

    async def preload(self) -> None:
        """Initialize streaming dataset connection."""
        if self._dataset is not None:
            return

        try:
            logger.info("Initializing OOLONG-Pairs streaming dataset (oolong-real dnd)")
            self._dataset = await _load_streaming_dataset()
            logger.info("OOLONG-Pairs streaming dataset ready")
        except Exception as e:
            logger.error("Failed to initialize OOLONG-Pairs dataset", error=str(e))
            raise

    async def enumerate_items(self) -> AsyncIterator[str]:
        """Yield all item IDs."""
        for i in range(OOLONG_PAIRS_TOTAL_ITEMS):
            yield str(i)

    async def _get_item(self, item_id: str) -> Optional[dict[str, Any]]:
        """Get item by ID, using cache or loading from stream."""
        if item_id in self._item_cache:
            return self._item_cache[item_id]

        try:
            target_idx = int(item_id)
            dataset = await _load_streaming_dataset()
            item = await asyncio.to_thread(_get_item_by_index_sync, dataset, target_idx)
            if item:
                self._item_cache[item_id] = item
            return item
        except Exception as e:
            logger.error("Failed to load item from streaming dataset", item_id=item_id, error=str(e))
            return None

    async def evaluate_item(self, item_id: str) -> ItemResult:
        """Evaluate a single OOLONG-Pairs item."""
        item = await self._get_item(item_id)

        if not item:
            return ItemResult(
                item_id=item_id,
                error=f"Item {item_id} not found",
            )

        # Construct prompt following OOLONG format
        prompt = (
            f"{item['context_window_text']}\n\n"
            f"Based on the above transcript, answer the following question.\n\n"
            f"Question: {item['question']}\n\n"
            f"Provide a comprehensive answer based only on the information "
            f"in the transcript. Be specific and cite relevant details."
        )

        try:
            start_time = time.time()
            response_text, metadata = await self.client.get_completion_text(
                self.model_slug,
                prompt,
                max_tokens=1024,  # Allow longer responses for comprehensive answers
                min_output_tokens=0,
                temperature=0.0,
                timeout=300,  # 5 minutes for long context
                response_attempts=2,
            )
            latency_ms = int((time.time() - start_time) * 1000)

            if not response_text or response_text is None:
                item_metadata = {
                    **metadata,
                    "question_type": item["question_type"],
                    "campaign": item["campaign"],
                    "is_pairwise": item["is_pairwise"],
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

            # Extract answer and compute F1 score
            extracted_answer = _extract_answer(response_text)
            expected = item["answer"]

            precision, recall, f1 = _compute_f1_score(expected, extracted_answer)

            # Consider correct if F1 >= 0.5 (reasonable overlap)
            is_correct = f1 >= 0.5

            item_metadata = {
                **metadata,
                "question_type": item["question_type"],
                "campaign": item["campaign"],
                "is_pairwise": item["is_pairwise"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "extracted_answer": extracted_answer[:500] if extracted_answer else None,  # Truncate for storage
            }

            return ItemResult(
                item_id=item_id,
                item_hash=self.compute_item_hash(item["question"]),
                prompt=prompt,
                response=response_text.strip(),
                expected=expected,
                is_correct=is_correct,
                score=f1,  # Use F1 as the score
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
            logger.error("OOLONG-Pairs evaluation failed", item_id=item_id, error=error_detail)
            return ItemResult(
                item_id=item_id,
                prompt=prompt,
                response="",
                error=error_detail,
                metadata={
                    "question_type": item.get("question_type"),
                    "campaign": item.get("campaign"),
                    "is_pairwise": item.get("is_pairwise"),
                },
            )

    async def postprocess(self, results: list[ItemResult]) -> dict[str, Any]:
        """Compute metrics broken down by question type and pairwise status."""
        metrics: dict[str, Any] = {}

        # Collect F1 scores
        all_f1_scores: list[float] = []
        pairwise_f1_scores: list[float] = []
        non_pairwise_f1_scores: list[float] = []

        # Group by question type
        by_question_type: dict[str, list[ItemResult]] = {}
        # Group by campaign
        by_campaign: dict[str, list[ItemResult]] = {}

        for result in results:
            f1 = result.metadata.get("f1", 0.0) if result.metadata else 0.0
            all_f1_scores.append(f1)

            if result.metadata:
                is_pairwise = result.metadata.get("is_pairwise", False)
                if is_pairwise:
                    pairwise_f1_scores.append(f1)
                else:
                    non_pairwise_f1_scores.append(f1)

                question_type = result.metadata.get("question_type", "unknown")
                if question_type not in by_question_type:
                    by_question_type[question_type] = []
                by_question_type[question_type].append(result)

                campaign = result.metadata.get("campaign", "unknown")
                if campaign not in by_campaign:
                    by_campaign[campaign] = []
                by_campaign[campaign].append(result)

        # Overall F1
        if all_f1_scores:
            avg_f1 = sum(all_f1_scores) / len(all_f1_scores)
            metrics["avg_f1"] = avg_f1
            metrics["max_f1"] = max(all_f1_scores)
            metrics["min_f1"] = min(all_f1_scores)
            # Use average F1 as the benchmark score (per spec)
            metrics["score_override"] = avg_f1

        # Pairwise vs non-pairwise
        if pairwise_f1_scores:
            metrics["avg_f1_pairwise"] = sum(pairwise_f1_scores) / len(pairwise_f1_scores)
            metrics["count_pairwise"] = len(pairwise_f1_scores)
        if non_pairwise_f1_scores:
            metrics["avg_f1_non_pairwise"] = sum(non_pairwise_f1_scores) / len(non_pairwise_f1_scores)
            metrics["count_non_pairwise"] = len(non_pairwise_f1_scores)

        # F1 by question type
        for qtype, type_results in sorted(by_question_type.items()):
            f1_scores = [r.metadata.get("f1", 0.0) for r in type_results if r.metadata]
            if f1_scores:
                # Sanitize question type for metric name
                safe_qtype = re.sub(r'[^a-zA-Z0-9]', '_', qtype.lower())[:20]
                metrics[f"avg_f1_{safe_qtype}"] = sum(f1_scores) / len(f1_scores)

        # F1 by campaign (top 5 by count)
        campaigns_by_count = sorted(by_campaign.items(), key=lambda x: -len(x[1]))[:5]
        for campaign, campaign_results in campaigns_by_count:
            f1_scores = [r.metadata.get("f1", 0.0) for r in campaign_results if r.metadata]
            if f1_scores:
                safe_campaign = re.sub(r'[^a-zA-Z0-9]', '_', campaign.lower())[:15]
                metrics[f"avg_f1_campaign_{safe_campaign}"] = sum(f1_scores) / len(f1_scores)

        return metrics
