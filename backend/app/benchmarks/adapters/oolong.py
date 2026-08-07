"""OOLONG benchmark adapter.

Based on the OOLONG paper (arxiv:2511.02817), used in the RLM paper.
Tests long-context reasoning and aggregation capabilities requiring
semantic classification and aggregation across nearly all dataset entries.

Uses the oolongbench/oolong-synth dataset from HuggingFace.

This adapter loads the dataset once (cached on disk) and fetches items on demand,
avoiding large in-memory caches for 100% runs.
"""
import ast
import asyncio
import os
import re
import time
from datetime import date
from typing import Any, AsyncIterator, Optional

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.core.logging import get_logger

logger = get_logger(__name__)

# Total items in the oolong-synth test split (from dataset info)
OOLONG_SYNTH_TOTAL_ITEMS = 5200
OOLONG_SYNTH_REPO = "oolongbench/oolong-synth"
# Pin the corrected 2026-06-20 release. Loading mutable `main` makes an item ID
# name different text/answers after an upstream edit, which invalidates paired
# runs and repeats even though every local run still looks deterministic.
OOLONG_SYNTH_REVISION = "f0d59eaf0febf130664cfceb710436c8e3216b2b"
OOLONG_SYNTH_SPLIT = "test"
OOLONG_SYNTH_TEST_SHARDS = 41
# At the pinned revision, shards 0..33 contain 127 rows and shards 34..40
# contain 126. Keeping this mapping beside the immutable revision lets small
# explicit samples range-read only their Parquet row groups instead of
# downloading the full ~10 GB compressed test split before the first item.
OOLONG_SYNTH_LARGE_SHARD_COUNT = 34
OOLONG_SYNTH_LARGE_SHARD_ROWS = 127
OOLONG_SYNTH_SMALL_SHARD_ROWS = 126


_DATE_GOLD = re.compile(
    r"^\s*\[?\s*datetime\.date\(\s*(\d{4})\s*,\s*(\d{1,2})\s*,\s*(\d{1,2})\s*\)\s*\]?\s*$"
)


def _answer_type_name(answer_type: Any) -> str:
    """Return the enum member used by the published OOLONG dataset.

    The live dataset stores values such as ``ANSWER_TYPE.NUMERIC``. Older
    adapter tests used ``NUMERIC``. Accept both spellings, but never silently
    treat a numeric/date answer as a generic exact-match string.
    """
    return str(answer_type or "").rsplit(".", 1)[-1].upper()


def _test_shard_location(item_index: int) -> tuple[int, int]:
    """Map a global test row to its pinned Parquet shard and local row."""
    if item_index < 0 or item_index >= OOLONG_SYNTH_TOTAL_ITEMS:
        raise IndexError(item_index)
    first_region = OOLONG_SYNTH_LARGE_SHARD_COUNT * OOLONG_SYNTH_LARGE_SHARD_ROWS
    if item_index < first_region:
        return (
            item_index // OOLONG_SYNTH_LARGE_SHARD_ROWS,
            item_index % OOLONG_SYNTH_LARGE_SHARD_ROWS,
        )
    remainder = item_index - first_region
    return (
        OOLONG_SYNTH_LARGE_SHARD_COUNT
        + remainder // OOLONG_SYNTH_SMALL_SHARD_ROWS,
        remainder % OOLONG_SYNTH_SMALL_SHARD_ROWS,
    )


def _normalize_answer(value: Any) -> str:
    """Ground-truth answers arrive from the dataset as lists.

    `str([12])` is `"[12]"`, so an agent that correctly answers `12` was scored
    wrong: exact match compared `"12" != "[12]"`, and the numeric path threw on
    `float("[12]")` and returned 0.0. Every OOLONG number this harness has
    produced was depressed by it, and for NUMERIC answers essentially floored.

    Caught by the first `oolong_agentic` item: expected `[12]`, agent answered
    `12`, scored 0.0.
    """
    if isinstance(value, (list, tuple)):
        if not value:
            # An empty ground truth is a dataset problem, not an answer of "".
            # Keep it visible rather than turning it into something a blank
            # response would match.
            return "[]"
        if len(value) == 1:
            return str(value[0]).strip()
        return ", ".join(str(v).strip() for v in value)
    text = str(value).strip()
    date_match = _DATE_GOLD.fullmatch(text)
    if date_match:
        year, month, day = (int(part) for part in date_match.groups())
        return date(year, month, day).isoformat()
    # Cached items may already be the str() of a list.
    if len(text) >= 2 and text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text
        if isinstance(parsed, (list, tuple)):
            return _normalize_answer(parsed)
    return text


# Speaker/role labels a model prefixes an answer with when it has been reading a
# dialogue transcript. Formatting, not content.
_ROLE_PREFIX = re.compile(
    r"^\s*(?:user|assistant|system|speaker|answer|final answer|label|date|a|q)\s*[:\-]\s*",
    re.IGNORECASE,
)
# Wrapping decoration: quotes, backticks, markdown emphasis, brackets.
_WRAPPERS = [('"', '"'), ("'", "'"), ("`", "`"), ("**", "**"), ("*", "*"),
             ("(", ")"), ("[", "]"), ("{", "}")]


def _normalize_prediction(text: str) -> str:
    """Strip formatting from a prediction. Never change what it says.

    OOLONG scores by exact match, so `User: 88675` against a ground truth of
    `88675` scores 0 -- for BOTH arms. Enough of that and the benchmark is
    pinned at zero regardless of correctness, which is a floor effect, not a
    strict metric: a comparison where both arms score 0 on every item cannot
    show a harness difference in either direction, so the run is wasted.

    This removes only presentation: a leading speaker/role label, wrapping
    quotes / backticks / markdown emphasis, surrounding whitespace and trailing
    sentence punctuation.

    It deliberately does NOT do substring or fuzzy matching. A response of
    "Computers & Internet is less common than Family & Relationships" against a
    ground truth of "less common than" stays **wrong** -- recovering that would
    be grading, not formatting.
    """
    if not text:
        return ""
    out = text.strip()
    for _ in range(3):  # e.g. `**"19"**`
        before = out
        out = _ROLE_PREFIX.sub("", out).strip()
        for left, right in _WRAPPERS:
            if len(out) > len(left) + len(right) and out.startswith(left) and out.endswith(right):
                out = out[len(left):-len(right)].strip()
        out = out.rstrip(".!;,").strip()
        if out == before:
            break
    return out


def _compute_numeric_score(expected: str, predicted: str) -> float:
    """
    Compute score for numeric answers using 0.75^|y-ŷ| formula.

    From RLM paper: "Numerical answers as score(ŷ)=0.75^|y-ŷ|"
    """
    try:
        # Released OOLONG-synth parses these count answers with int(), so a
        # decimal-looking string such as "12.0" is not silently upgraded.
        expected_num = int(expected)
        predicted_num = int(predicted.strip())
        diff = abs(expected_num - predicted_num)
        return 0.75 ** diff
    except (ValueError, TypeError):
        return 0.0


def _is_exact_match(expected: str, predicted: str) -> bool:
    """Check the released scorer's stripped, case-sensitive exact match."""
    return expected.strip() == predicted.strip()


_COMPARISON_PHRASES = (
    "same frequency as",
    "same frequency",
    "more common than",
    "less common than",
    "more common",
    "less common",
)


def _comparison_relation(value: str) -> str:
    """Canonicalize the relation exactly as OOLONG's released parser does."""
    lowered = value.strip().lower()
    for phrase in _COMPARISON_PHRASES:
        if phrase in lowered:
            if phrase.startswith("more common"):
                return "more common"
            if phrase.startswith("less common"):
                return "less common"
            return "same frequency"
    return lowered


def _parse_date(value: str) -> Optional[date]:
    text = value.strip()
    gold_match = _DATE_GOLD.fullmatch(text)
    if gold_match:
        year, month, day = (int(part) for part in gold_match.groups())
        return date(year, month, day)
    try:
        from dateutil import parser as date_parser

        return date_parser.parse(text, fuzzy=False).date()
    except (TypeError, ValueError, OverflowError):
        return None


def score_answer(expected: str, predicted: str, answer_type: str) -> tuple[float, bool]:
    """Apply the released OOLONG-synth metric and deterministic answer parser.

    Numeric answers receive ``0.75 ** abs(gold - prediction)``. Dates are
    parsed before comparison, comparison answers are reduced to the named
    relation, and the remaining answer types use stripped, case-sensitive exact
    match.
    ``is_correct`` means exact credit; numeric near misses retain partial score
    but are not mislabeled as correct.
    """
    kind = _answer_type_name(answer_type)
    if kind == "NUMERIC":
        score = _compute_numeric_score(expected, predicted)
        return score, score == 1.0
    if kind == "DATE":
        expected_date = _parse_date(expected)
        predicted_date = _parse_date(predicted)
        correct = expected_date is not None and expected_date == predicted_date
        return (1.0 if correct else 0.0), correct
    if kind == "COMPARISON":
        correct = _comparison_relation(expected) == _comparison_relation(predicted)
        return (1.0 if correct else 0.0), correct
    correct = _is_exact_match(expected, predicted)
    return (1.0 if correct else 0.0), correct


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


@register_adapter("oolong")
class OolongAdapter(BenchmarkAdapter):
    """
    OOLONG benchmark adapter.

    Evaluates long-context reasoning and aggregation capabilities.
    Tasks require examining and transforming chunks of input semantically,
    then aggregating these chunks to form a final answer.

    Processing costs scale linearly with input length.

    Uses cached dataset access and on-demand item loading.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._dataset = None
        self._item_cache: dict[str, dict[str, Any]] = {}
        self._cache_limit = 128
        self._preloaded: bool = False
        self._target_item_ids: Optional[set[int]] = None
        self._use_streaming: bool = False
        self._stream_iter = None
        self._stream_index = -1
        self._stream_lock = asyncio.Lock()

    def get_name(self) -> str:
        return "oolong"

    def get_display_name(self) -> str:
        return "OOLONG"

    def supports_parallel_items(self) -> bool:
        # Allow parallel items; streaming fetches are serialized with a lock.
        return True

    async def get_total_items(self) -> int:
        """Get total items from dataset (known constant)."""
        return OOLONG_SYNTH_TOTAL_ITEMS

    async def preload(self) -> None:
        """Load the dataset once so items can be fetched by index."""
        if self._preloaded:
            return
        # The worker deliberately preloads before it calls
        # get_items_for_evaluation(). Recover explicit paired IDs from the run
        # config here as well, or a two-item run takes the full-dataset path.
        if not self._target_item_ids:
            config = getattr(self, "run_config", None) or {}
            merged: dict[str, Any] = {}
            for key in (self.get_name().rsplit("_", 1)[0], self.get_name()):
                candidate = config.get(key) or {}
                if isinstance(candidate, dict):
                    merged.update(candidate)
            requested = merged.get("item_ids")
            if requested:
                selected = {int(item_id) for item_id in requested}
                invalid = sorted(
                    item_id
                    for item_id in selected
                    if item_id < 0 or item_id >= OOLONG_SYNTH_TOTAL_ITEMS
                )
                if invalid:
                    raise ValueError(f"Unknown OOLONG item IDs: {invalid}")
                self._target_item_ids = selected
        if self._target_item_ids and len(self._target_item_ids) <= self._cache_limit:
            await asyncio.to_thread(self._preload_targeted_parquet_rows)
            self._preloaded = True
            return
        from datasets import load_dataset

        target_count = len(self._target_item_ids) if self._target_item_ids else None
        hf_token = os.environ.get("HF_TOKEN")
        if self._use_streaming:
            logger.info(
                "Loading OOLONG dataset (streaming)",
                target_count=target_count,
            )
            self._dataset = await asyncio.to_thread(
                load_dataset,
                OOLONG_SYNTH_REPO,
                split=OOLONG_SYNTH_SPLIT,
                token=hf_token,
                streaming=True,
                revision=OOLONG_SYNTH_REVISION,
            )
        else:
            logger.info(
                "Loading OOLONG dataset (non-streaming, cached by HuggingFace)",
                target_count=target_count,
            )
            self._dataset = await asyncio.to_thread(
                load_dataset,
                OOLONG_SYNTH_REPO,
                split=OOLONG_SYNTH_SPLIT,
                token=hf_token,
                keep_in_memory=False,
                revision=OOLONG_SYNTH_REVISION,
            )
        self._preloaded = True

    @staticmethod
    def _parquet_url(shard: int) -> str:
        return (
            f"https://huggingface.co/datasets/{OOLONG_SYNTH_REPO}/resolve/"
            f"{OOLONG_SYNTH_REVISION}/data/test-{shard:05d}-of-"
            f"{OOLONG_SYNTH_TEST_SHARDS:05d}.parquet"
        )

    def _preload_targeted_parquet_rows(self) -> None:
        """Range-read only row groups containing the explicitly selected rows.

        `datasets.load_dataset` materializes every shard even for two item IDs;
        the pinned test split is roughly 10 GB compressed. Parquet row groups
        retain the exact source bytes while keeping this small benchmark sample
        practical. No viewer API or mutable branch is involved.
        """
        import fsspec
        import pyarrow.parquet as pq

        targets_by_shard: dict[int, dict[int, int]] = {}
        for item_index in sorted(self._target_item_ids or set()):
            shard, local_row = _test_shard_location(item_index)
            targets_by_shard.setdefault(shard, {})[local_row] = item_index

        columns = [
            "context_window_text",
            "question",
            "answer",
            "answer_type",
            "task",
            "task_group",
            "context_len",
            "dataset",
            "num_labels",
            "context_window_id",
        ]
        filesystem = fsspec.filesystem("http", block_size=1024 * 1024)
        for shard, local_targets in targets_by_shard.items():
            url = self._parquet_url(shard)
            with filesystem.open(
                url,
                "rb",
                block_size=1024 * 1024,
                cache_type="readahead",
            ) as source:
                parquet = pq.ParquetFile(source)
                cursor = 0
                remaining = dict(local_targets)
                for row_group in range(parquet.num_row_groups):
                    row_count = parquet.metadata.row_group(row_group).num_rows
                    selected = {
                        local_row: item_index
                        for local_row, item_index in remaining.items()
                        if cursor <= local_row < cursor + row_count
                    }
                    if selected:
                        rows = parquet.read_row_group(
                            row_group, columns=columns
                        ).to_pylist()
                        for local_row, item_index in selected.items():
                            raw_item = rows[local_row - cursor]
                            self._item_cache[str(item_index)] = {
                                "id": str(item_index),
                                "context_window_text": raw_item["context_window_text"],
                                "question": raw_item["question"],
                                "answer": _normalize_answer(raw_item["answer"]),
                                "answer_type": raw_item.get("answer_type", ""),
                                "task": raw_item.get("task", ""),
                                "task_group": raw_item.get("task_group", ""),
                                "context_len": raw_item.get("context_len", 0),
                                "dataset": raw_item.get("dataset", ""),
                                "num_labels": raw_item.get("num_labels", 0),
                                "context_window_id": raw_item.get("context_window_id"),
                                "dataset_repo": OOLONG_SYNTH_REPO,
                                "dataset_revision": OOLONG_SYNTH_REVISION,
                                "dataset_split": OOLONG_SYNTH_SPLIT,
                                "dataset_transport": "pinned_parquet_range_read",
                                "dataset_shard": shard,
                                "dataset_shard_row": local_row,
                            }
                            remaining.pop(local_row)
                    cursor += row_count
                    if not remaining:
                        break
                if remaining:
                    missing = sorted(remaining.values())
                    raise RuntimeError(
                        f"Pinned OOLONG shard {shard} did not contain item(s) {missing}"
                    )

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
        Override to preselect item IDs for evaluation.

        We avoid heavy dataset work here so the worker can record sampled items
        before preloading the dataset on demand.
        """
        # Use the base selector so explicit item_ids work. The previous override
        # bypassed that path, silently replacing requested paired items with a
        # hash sample.
        total_items, items_to_evaluate = await super().get_items_for_evaluation(
            subset_pct, seed, subset_count
        )

        # Track target items for preloading and item access.
        self._target_item_ids = {int(item_id) for item_id in items_to_evaluate}
        self._item_cache.clear()
        self._use_streaming = False

        return total_items, items_to_evaluate

    async def _get_item(self, item_id: str) -> Optional[dict[str, Any]]:
        """Get item by ID from dataset."""
        cached = self._item_cache.get(item_id)
        if cached is not None:
            return cached
        if not self._preloaded:
            await self.preload()
        cached = self._item_cache.get(item_id)
        if cached is not None:
            return cached
        if self._dataset is None:
            return None

        try:
            idx = int(item_id)
        except (TypeError, ValueError):
            return None

        if self._use_streaming:
            async with self._stream_lock:
                cached = self._item_cache.get(item_id)
                if cached is not None:
                    return cached
                if self._stream_iter is None:
                    self._stream_iter = iter(self._dataset)
                    self._stream_index = -1
                while self._stream_index < idx:
                    try:
                        raw_item = next(self._stream_iter)
                    except StopIteration:
                        return None
                    self._stream_index += 1
                    item = {
                        "id": str(self._stream_index),
                        "context_window_text": raw_item["context_window_text"],
                        "question": raw_item["question"],
                        "answer": _normalize_answer(raw_item["answer"]),
                        "answer_type": raw_item.get("answer_type", ""),
                        "task": raw_item.get("task", ""),
                        "task_group": raw_item.get("task_group", ""),
                        "context_len": raw_item.get("context_len", 0),
                        "dataset": raw_item.get("dataset", ""),
                        "num_labels": raw_item.get("num_labels", 0),
                        "dataset_repo": OOLONG_SYNTH_REPO,
                        "dataset_revision": OOLONG_SYNTH_REVISION,
                        "dataset_split": OOLONG_SYNTH_SPLIT,
                    }
                    if self._cache_limit > 0:
                        if len(self._item_cache) >= self._cache_limit:
                            self._item_cache.clear()
                        self._item_cache[str(self._stream_index)] = item
                return self._item_cache.get(item_id)

        def _load_item() -> dict[str, Any]:
            item = self._dataset[idx]
            return {
                "id": str(idx),
                "context_window_text": item["context_window_text"],
                "question": item["question"],
                "answer": _normalize_answer(item["answer"]),
                "answer_type": item.get("answer_type", ""),
                "task": item.get("task", ""),
                "task_group": item.get("task_group", ""),
                "context_len": item.get("context_len", 0),
                "dataset": item.get("dataset", ""),
                "num_labels": item.get("num_labels", 0),
                "dataset_repo": OOLONG_SYNTH_REPO,
                "dataset_revision": OOLONG_SYNTH_REVISION,
                "dataset_split": OOLONG_SYNTH_SPLIT,
            }

        item = await asyncio.to_thread(_load_item)
        if item is None:
            return None
        if self._cache_limit > 0:
            if len(self._item_cache) >= self._cache_limit:
                self._item_cache.clear()
            self._item_cache[item_id] = item
        return item

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
            # Scored twice: on the raw extraction and on the formatting-
            # normalised one (see _normalize_prediction). The normalised score
            # is the reported one; both are kept on the item so a reviewer can
            # see exactly what the normalisation rule did rather than trust it.
            score_raw, correct_raw = score_answer(expected, extracted_answer, answer_type)
            normalized_answer = _normalize_prediction(extracted_answer)
            score, is_correct = score_answer(expected, normalized_answer, answer_type)

            item_metadata = {
                **metadata,
                "extracted_answer": extracted_answer,
                "normalized_answer": normalized_answer,
                "score_raw": score_raw,
                "is_correct_raw": correct_raw,
                "score_normalized": score,
                "normalization": "formatting_only",
                "task": item["task"],
                "task_group": item["task_group"],
                "answer_type": answer_type,
                "context_len": item["context_len"],
                "dataset_repo": item["dataset_repo"],
                "dataset_revision": item["dataset_revision"],
                "dataset_split": item["dataset_split"],
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
            if (result.metadata or {}).get("exclusion_reason"):
                continue
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
            if _answer_type_name(answer_type) == "NUMERIC":
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
