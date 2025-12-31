"""Base benchmark adapter interface."""
import hashlib
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Optional

from app.services.chutes_client import ChutesClient


@dataclass
class ItemResult:
    """Result for a single benchmark item."""

    item_id: str
    item_hash: Optional[str] = None
    prompt: Optional[str] = None
    response: Optional[str] = None
    expected: Optional[str] = None
    is_correct: Optional[bool] = None
    score: Optional[float] = None
    judge_output: Optional[dict[str, Any]] = None
    latency_ms: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    error: Optional[str] = None
    test_code: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None


@dataclass
class BenchmarkResult:
    """Aggregated result for a benchmark."""

    benchmark_name: str
    total_items: int
    completed_items: int
    correct_items: int
    score: float
    metrics: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class BenchmarkAdapter(ABC):
    """
    Base class for benchmark adapters.
    
    Each benchmark adapter must implement:
    - get_name(): Return the benchmark name identifier
    - get_display_name(): Return human-readable name
    - get_total_items(): Return total number of items in the benchmark
    - enumerate_items(): Yield item IDs
    - evaluate_item(): Evaluate a single item and return ItemResult
    
    Optional overrides:
    - supports_subset(): Whether subset sampling is supported
    - requires_setup(): Whether special setup is needed
    - get_setup_notes(): Notes about required setup
    - preload(): Called before evaluation starts
    - postprocess(): Called after all items are evaluated
    """

    def __init__(self, client: ChutesClient, model_slug: str):
        """Initialize adapter with Chutes client and model."""
        self.client = client
        self.model_slug = model_slug
        self._items_cache: Optional[list[str]] = None

    @abstractmethod
    def get_name(self) -> str:
        """Return the benchmark identifier (e.g., 'mmlu_pro')."""
        pass

    @abstractmethod
    def get_display_name(self) -> str:
        """Return human-readable name (e.g., 'MMLU-Pro')."""
        pass

    @abstractmethod
    async def get_total_items(self) -> int:
        """Return total number of items in the benchmark."""
        pass

    @abstractmethod
    async def enumerate_items(self) -> AsyncIterator[str]:
        """Yield all item IDs in the benchmark."""
        pass

    @abstractmethod
    async def evaluate_item(self, item_id: str) -> ItemResult:
        """
        Evaluate a single item.
        
        Args:
            item_id: The item identifier
            
        Returns:
            ItemResult with evaluation details
        """
        pass

    def supports_subset(self) -> bool:
        """Return True if subset sampling is supported."""
        return True

    def requires_setup(self) -> bool:
        """Return True if special setup is required."""
        return False

    def get_setup_notes(self) -> Optional[str]:
        """Return notes about required setup, if any."""
        return None

    async def preload(self) -> None:
        """Called before evaluation starts. Override to load data."""
        pass

    async def postprocess(self, results: list[ItemResult]) -> dict[str, Any]:
        """
        Called after all items are evaluated.
        
        Override to compute custom aggregate metrics.
        
        Returns:
            Additional metrics dict
        """
        return {}

    async def cleanup(self) -> None:
        """Cleanup any resources allocated by the adapter."""
        return None

    async def get_items_for_evaluation(self, subset_pct: int, seed: str) -> tuple[int, list[str]]:
        all_items: list[str] = []
        async for item_id in self.enumerate_items():
            all_items.append(item_id)
        total_items = len(all_items)
        return total_items, self.get_deterministic_subset(all_items, subset_pct, seed)

    def get_deterministic_subset(
        self,
        item_ids: list[str],
        subset_pct: int,
        seed: str,
    ) -> list[str]:
        """
        Get deterministic subset of items.
        
        Uses a stable seed derived from run_id + benchmark_name to ensure
        reproducibility across runs with the same configuration.
        
        Args:
            item_ids: All item IDs
            subset_pct: Percentage to sample (1, 5, 10, 25, 50, 100)
            seed: Seed string (typically run_id + benchmark_name)
            
        Returns:
            Deterministically sampled list of item IDs
        """
        if subset_pct >= 100:
            return item_ids

        # Create deterministic seed from string
        seed_int = int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16)
        rng = random.Random(seed_int)

        # Shuffle and take top K
        shuffled = item_ids.copy()
        rng.shuffle(shuffled)
        k = max(1, int(len(item_ids) * subset_pct / 100))
        return shuffled[:k]

    def compute_item_hash(self, item_data: Any) -> str:
        """Compute a stable hash for an item's data."""
        if isinstance(item_data, str):
            data_str = item_data
        else:
            import json
            data_str = json.dumps(item_data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]

    def format_empty_response_error(self, metadata: Optional[dict[str, Any]] = None) -> str:
        """Provide a consistent empty-response error with any available metadata."""
        if metadata:
            detail = metadata.get("response_error") or metadata.get("error")
            parts: list[str] = []
            if detail:
                parts.append(str(detail))
            response_source = metadata.get("response_source")
            finish_reason = metadata.get("finish_reason")
            attempts = metadata.get("response_attempts")
            if response_source:
                parts.append(f"source={response_source}")
            if finish_reason:
                parts.append(f"finish_reason={finish_reason}")
            if attempts:
                parts.append(f"attempts={attempts}")
            if parts:
                return " | ".join(parts)
        return "Model produced empty response"

    def format_truncation_error(
        self,
        metadata: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> Optional[str]:
        """Annotate an error message if the model response was truncated."""
        finish_reason = (metadata or {}).get("finish_reason")
        if finish_reason != "length":
            return error
        truncation = "Response truncated (finish_reason=length)"
        if not error:
            return truncation
        return f"{error} | {truncation}"

    def extract_choice_letter(self, response_text: Optional[str], valid_letters: str) -> str:
        """Extract a multiple-choice answer letter from the response."""
        if not response_text:
            return ""

        import re

        cleaned = re.sub(r"(?i)<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()
        if not cleaned:
            return ""

        cleaned = re.sub(r"\*\*([A-Za-z])\*\*", r"\1", cleaned)
        cleaned = re.sub(r"\*([A-Za-z])\*", r"\1", cleaned)
        cleaned = re.sub(r"`([A-Za-z])`", r"\1", cleaned)
        cleaned = re.sub(r"\\\\boxed\{\\s*([A-Za-z])\\s*\}", r"\1", cleaned)
        cleaned = re.sub(r"\\$\\s*([A-Za-z])\\s*\\$", r"\1", cleaned)

        upper_letters = valid_letters.upper()
        letter_set = f"[{re.escape(upper_letters)}]"

        # Prefer the last explicit "Answer: X" line if present.
        answer_matches = re.findall(
            rf"(?im)^\s*(?:answer|final answer)\s*[:\-]\s*\(?({letter_set})\)?\s*$",
            cleaned,
        )
        if answer_matches:
            return answer_matches[-1].upper()

        # Check lines from bottom to top for a standalone letter.
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        for line in reversed(lines):
            line_match = re.match(rf"^\(?({letter_set})\)?[\\.\)]?\s*$", line.upper())
            if line_match:
                return line_match.group(1).upper()

        # Allow the entire response to be a single letter.
        full_match = re.match(rf"^\s*\(?({letter_set})\)?[\\.\)]?\s*$", cleaned.upper())
        if full_match:
            return full_match.group(1).upper()

        return ""

    def extract_python_code(self, response: Optional[str]) -> str:
        """Extract Python code from model response robustly."""
        if not response:
            return ""
            
        import re
        
        # 1. Try to find a complete markdown block
        match = re.search(r"```(?:python|bash|diff)?\n(.*?)\n```", response, re.DOTALL)
        if match:
            return match.group(1).strip()
            
        # 2. Try to find an incomplete markdown block
        match = re.search(r"```(?:python|bash|diff)?\n(.*)", response, re.DOTALL)
        if match:
            return match.group(1).strip()
            
        # 3. If no markdown block, remove thinking block if present
        cleaned_response = re.sub(r"(?i)<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        
        # 4. If it still starts with <think> (unclosed), strip it
        if cleaned_response.lower().startswith("<think>"):
            cleaned_response = re.sub(r"(?i)^<think>", "", cleaned_response).strip()
            
        # 5. Final fallback
        return cleaned_response

    async def run_evaluation(
        self,
        subset_pct: int,
        seed: str,
        progress_callback: Optional[Callable[[int, int, Optional[ItemResult]], None]] = None,
    ) -> BenchmarkResult:
        """
        Run full evaluation with optional subset sampling.
        
        Args:
            subset_pct: Percentage of items to evaluate
            seed: Seed for deterministic sampling
            progress_callback: Optional callback(completed, total, last_result)
            
        Returns:
            BenchmarkResult with aggregate metrics
        """
        await self.preload()

        # Get all items
        all_items: list[str] = []
        async for item_id in self.enumerate_items():
            all_items.append(item_id)

        # Apply subset sampling
        items_to_evaluate = self.get_deterministic_subset(all_items, subset_pct, seed)

        results: list[ItemResult] = []
        correct = 0

        for i, item_id in enumerate(items_to_evaluate):
            result = await self.evaluate_item(item_id)
            results.append(result)

            if result.is_correct:
                correct += 1

            if progress_callback:
                progress_callback(i + 1, len(items_to_evaluate), result)

        # Compute metrics
        additional_metrics = await self.postprocess(results)
        score = correct / len(results) if results else 0.0

        return BenchmarkResult(
            benchmark_name=self.get_name(),
            total_items=len(all_items),
            completed_items=len(results),
            correct_items=correct,
            score=score,
            metrics={
                "accuracy": score,
                "sampled_pct": subset_pct,
                "sampled_count": len(items_to_evaluate),
                **additional_metrics,
            },
        )
