"""OOLONG-Pairs benchmark adapter.

Based on the RLM paper (arxiv:2512.24601v1), this is a custom extension
of OOLONG that requires aggregating pairs of chunks to construct the final answer.

Processing costs scale quadratically with input length due to pairwise relationships.
Uses the oolongbench/oolong-real dataset from HuggingFace with D&D transcripts.
"""
import re
import time
from typing import Any, AsyncIterator, Optional

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.benchmarks.utils import load_dataset_with_retry
from app.core.logging import get_logger

logger = get_logger(__name__)


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
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._dataset: Optional[Any] = None
        self._items: list[dict[str, Any]] = []

    def get_name(self) -> str:
        return "oolong_pairs"

    def get_display_name(self) -> str:
        return "OOLONG-Pairs"

    def supports_parallel_items(self) -> bool:
        return True

    def get_item_timeout_seconds(self) -> Optional[int]:
        """Longer timeout for quadratic complexity tasks."""
        return 300  # 5 minutes

    async def get_total_items(self) -> int:
        """Get total items from dataset."""
        if not self._items:
            await self.preload()
        return len(self._items)

    async def preload(self) -> None:
        """Load OOLONG-Real dataset from HuggingFace."""
        if self._items:
            return

        try:
            import os

            logger.info("Loading OOLONG-Pairs dataset (oolong-real with dnd config)")
            hf_token = os.environ.get("HF_TOKEN")

            # Load the D&D config from oolong-real
            # This contains questions about D&D transcripts that require
            # understanding relationships between characters/events
            dataset = await load_dataset_with_retry(
                "oolongbench/oolong-real",
                "dnd",
                split="test",
                token=hf_token,
            )

            self._items = []
            for i, item in enumerate(dataset):
                # Filter for questions that likely require pairwise reasoning
                # (questions about relationships, comparisons, interactions)
                question = item["question"].lower()
                question_type = item.get("question_type", "")

                # Include all items but tag those that are explicitly pairwise
                is_pairwise = any(kw in question for kw in [
                    "between", "relationship", "interact", "compare",
                    "versus", "vs", "together", "both", "each other",
                    "difference", "similar", "conflict", "alliance"
                ])

                self._items.append({
                    "id": str(i),
                    "context_window_text": item["context_window_text"],
                    "question": item["question"],
                    "answer": str(item["answer"]),
                    "question_type": question_type,
                    "campaign": item.get("campaign", ""),
                    "episodes": item.get("episodes", []),
                    "is_pairwise": is_pairwise,
                })

            logger.info(f"Loaded {len(self._items)} OOLONG-Pairs items")

        except Exception as e:
            logger.error("Failed to load OOLONG-Pairs dataset", error=str(e))
            raise

    async def enumerate_items(self) -> AsyncIterator[str]:
        """Yield all item IDs."""
        if not self._items:
            await self.preload()
        for item in self._items:
            yield item["id"]

    async def evaluate_item(self, item_id: str) -> ItemResult:
        """Evaluate a single OOLONG-Pairs item."""
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
            metrics["avg_f1"] = sum(all_f1_scores) / len(all_f1_scores)
            metrics["max_f1"] = max(all_f1_scores)
            metrics["min_f1"] = min(all_f1_scores)

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
