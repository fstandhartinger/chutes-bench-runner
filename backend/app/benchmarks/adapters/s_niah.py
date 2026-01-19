"""S-NIAH (Single Needle-in-a-Haystack) benchmark adapter.

Based on the RULER benchmark (Hsieh et al., 2024), used in the RLM paper.
Tests the ability to retrieve a single piece of information ("needle")
from long distractor texts ("haystack").
"""
import hashlib
import random
import time
from typing import Any, AsyncIterator, Optional

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.core.logging import get_logger

logger = get_logger(__name__)

# Paul Graham essay excerpts used as haystack filler (abbreviated for testing)
HAYSTACK_TEXTS = [
    "The way to get startup ideas is not to try to think of startup ideas. It's to look for problems, preferably problems you have yourself. The very best startup ideas tend to have three things in common: they're something the founders themselves want, that they themselves can build, and that few others realize are worth doing.",
    "What I tell founders is not to sweat the business model too much at first. The most important task at first is to build something people want. If you don't do that, it won't matter how clever your business model is.",
    "When you're operating on the principle of building something you yourself want, you're more likely to build something good. You know what you want better than anyone else.",
    "A lot of would-be founders believe that startups either take off or don't. You build something, make it available, and if you've made a better mousetrap, people beat a path to your door as promised. Or they don't, in which case the market must not exist.",
    "The most important thing a startup needs is customers, but the key to getting customers is building something they want.",
    "The best way to have startup ideas is to become the sort of person who has them. If you're at the leading edge of some rapidly changing field, you'll have a sense of how it could be improved.",
    "The hardest part of most startups is figuring out what to build. Once you know what to build, the work is very different.",
    "In the best case, the thing you're building grows into a platform. That's when things get really interesting.",
    "One of the biggest mistakes founders make is to focus on competitors when they should be focusing on users.",
    "The most important thing is not to have a formal business plan, but to understand what you're doing well enough that you can adapt.",
]


def _generate_haystack(seed: int, target_tokens: int) -> str:
    """Generate haystack text of approximately target_tokens length."""
    rng = random.Random(seed)
    texts = []
    current_tokens = 0

    while current_tokens < target_tokens:
        text = rng.choice(HAYSTACK_TEXTS)
        texts.append(text)
        # Rough estimate: 1 token ≈ 4 characters
        current_tokens += len(text) // 4

    return "\n\n".join(texts)


def _generate_needle_key(seed: int, key_type: str = "words") -> str:
    """Generate a needle key based on type."""
    rng = random.Random(seed)

    if key_type == "numbers":
        return str(rng.randint(1000000, 9999999))  # 7 digits
    elif key_type == "uuids":
        return "".join(rng.choices("0123456789abcdef", k=32))
    else:  # words
        adjectives = ["secret", "magic", "hidden", "special", "unique", "rare", "ancient", "golden"]
        nouns = ["code", "key", "phrase", "password", "token", "number", "identifier", "value"]
        return f"{rng.choice(adjectives)} {rng.choice(nouns)}"


def _generate_needle_value(seed: int, value_type: str = "words") -> str:
    """Generate a needle value based on type."""
    rng = random.Random(seed)

    if value_type == "numbers":
        return str(rng.randint(1000000, 9999999))  # 7 digits
    elif value_type == "uuids":
        return "".join(rng.choices("0123456789abcdef", k=32))
    else:  # words
        adjectives = ["blue", "red", "green", "yellow", "purple", "orange", "silver", "crystal"]
        nouns = ["dragon", "phoenix", "unicorn", "griffin", "serpent", "falcon", "lion", "tiger"]
        return f"{rng.choice(adjectives)} {rng.choice(nouns)}"


def _insert_needle_in_haystack(
    haystack: str,
    needle_key: str,
    needle_value: str,
    position: float,
    seed: int,
) -> str:
    """Insert needle at specified position (0.0 = start, 1.0 = end) in haystack."""
    needle_statement = f"The {needle_key} is: {needle_value}."

    paragraphs = haystack.split("\n\n")
    insert_idx = int(len(paragraphs) * position)
    insert_idx = max(0, min(insert_idx, len(paragraphs)))

    paragraphs.insert(insert_idx, needle_statement)
    return "\n\n".join(paragraphs)


@register_adapter("s_niah")
class SNIAHAdapter(BenchmarkAdapter):
    """
    S-NIAH (Single Needle-in-a-Haystack) benchmark adapter.

    Generates tasks that require finding a single "needle" (key-value pair)
    within a "haystack" of distractor text. Based on RULER benchmark.

    Supports configurable context lengths via context_size parameter.
    """

    # Default context sizes in tokens (powers of 2 as in RLM paper)
    DEFAULT_CONTEXT_SIZES = [8192, 16384, 32768, 65536, 131072, 262144]  # 8K to 256K

    # Number of tasks per context size
    TASKS_PER_SIZE = 10

    # Needle positions to test (spread across the context)
    NEEDLE_POSITIONS = [0.1, 0.3, 0.5, 0.7, 0.9]

    def __init__(
        self,
        *args: Any,
        context_sizes: Optional[list[int]] = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._context_sizes = context_sizes or self.DEFAULT_CONTEXT_SIZES
        self._items: list[dict[str, Any]] = []

    def get_name(self) -> str:
        return "s_niah"

    def get_display_name(self) -> str:
        return "S-NIAH (Needle-in-Haystack)"

    def supports_parallel_items(self) -> bool:
        return True

    async def get_total_items(self) -> int:
        """Total items = context_sizes * tasks_per_size."""
        if not self._items:
            await self.preload()
        return len(self._items)

    async def preload(self) -> None:
        """Generate all S-NIAH test items."""
        if self._items:
            return

        logger.info("Generating S-NIAH benchmark items", context_sizes=self._context_sizes)

        item_id = 0
        for context_size in self._context_sizes:
            for task_idx in range(self.TASKS_PER_SIZE):
                # Create deterministic seed from context_size and task_idx
                seed = int(hashlib.sha256(f"s_niah_{context_size}_{task_idx}".encode()).hexdigest()[:8], 16)

                # Generate needle key and value
                needle_key = _generate_needle_key(seed)
                needle_value = _generate_needle_value(seed + 1)

                # Select needle position
                position = self.NEEDLE_POSITIONS[task_idx % len(self.NEEDLE_POSITIONS)]

                # Generate haystack (leave room for needle and prompt overhead)
                haystack = _generate_haystack(seed + 2, target_tokens=context_size - 500)

                # Insert needle at position
                context = _insert_needle_in_haystack(
                    haystack, needle_key, needle_value, position, seed
                )

                self._items.append({
                    "id": str(item_id),
                    "context_size": context_size,
                    "task_idx": task_idx,
                    "needle_key": needle_key,
                    "needle_value": needle_value,
                    "needle_position": position,
                    "context": context,
                    "seed": seed,
                })
                item_id += 1

        logger.info(f"Generated {len(self._items)} S-NIAH items")

    async def enumerate_items(self) -> AsyncIterator[str]:
        """Yield all item IDs."""
        if not self._items:
            await self.preload()
        for item in self._items:
            yield item["id"]

    async def evaluate_item(self, item_id: str) -> ItemResult:
        """Evaluate a single S-NIAH item."""
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

        # Construct prompt
        prompt = (
            f"Some special information is hidden within the following text. "
            f"Make sure to memorize it. I will quiz you about it at the end.\n\n"
            f"{item['context']}\n\n"
            f"What is the {item['needle_key']}? "
            f"Only give me the answer and do not output any other words."
        )

        try:
            start_time = time.time()
            response_text, metadata = await self.client.get_completion_text(
                self.model_slug,
                prompt,
                max_tokens=256,
                min_output_tokens=0,
                temperature=0.0,
                timeout=120,
                response_attempts=2,
            )
            latency_ms = int((time.time() - start_time) * 1000)

            if not response_text or response_text is None:
                item_metadata = {
                    **metadata,
                    "context_size": item["context_size"],
                    "needle_position": item["needle_position"],
                }
                return ItemResult(
                    item_id=item_id,
                    item_hash=self.compute_item_hash(f"{item['needle_key']}:{item['needle_value']}"),
                    prompt=prompt,
                    response="",
                    expected=item["needle_value"],
                    is_correct=False,
                    score=0.0,
                    error=self.format_empty_response_error(metadata),
                    latency_ms=latency_ms,
                    input_tokens=metadata.get("usage", {}).get("prompt_tokens"),
                    output_tokens=metadata.get("usage", {}).get("completion_tokens"),
                    metadata=item_metadata,
                )

            # Exact match (case-insensitive, stripped) per benchmark spec
            response_clean = response_text.strip().lower()
            expected_clean = item["needle_value"].strip().lower()
            is_correct = response_clean == expected_clean

            item_metadata = {
                **metadata,
                "context_size": item["context_size"],
                "needle_position": item["needle_position"],
                "needle_key": item["needle_key"],
            }

            return ItemResult(
                item_id=item_id,
                item_hash=self.compute_item_hash(f"{item['needle_key']}:{item['needle_value']}"),
                prompt=prompt,
                response=response_text.strip(),
                expected=item["needle_value"],
                is_correct=is_correct,
                score=1.0 if is_correct else 0.0,
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
            logger.error("S-NIAH evaluation failed", item_id=item_id, error=error_detail)
            return ItemResult(
                item_id=item_id,
                prompt=prompt,
                response="",
                error=error_detail,
                metadata={
                    "context_size": item["context_size"],
                    "needle_position": item["needle_position"],
                },
            )

    async def postprocess(self, results: list[ItemResult]) -> dict[str, Any]:
        """Compute metrics broken down by context size and needle position."""
        metrics: dict[str, Any] = {}

        # Group by context size
        by_context_size: dict[int, list[ItemResult]] = {}
        by_position: dict[float, list[ItemResult]] = {}

        for result in results:
            if result.metadata:
                ctx_size = result.metadata.get("context_size")
                position = result.metadata.get("needle_position")

                if ctx_size is not None:
                    if ctx_size not in by_context_size:
                        by_context_size[ctx_size] = []
                    by_context_size[ctx_size].append(result)

                if position is not None:
                    if position not in by_position:
                        by_position[position] = []
                    by_position[position].append(result)

        # Accuracy by context size
        for ctx_size, ctx_results in sorted(by_context_size.items()):
            correct = sum(1 for r in ctx_results if r.is_correct)
            total = len(ctx_results)
            metrics[f"accuracy_{ctx_size // 1024}k"] = correct / total if total > 0 else 0.0

        # Accuracy by needle position
        for position, pos_results in sorted(by_position.items()):
            correct = sum(1 for r in pos_results if r.is_correct)
            total = len(pos_results)
            metrics[f"accuracy_pos_{int(position * 100)}pct"] = correct / total if total > 0 else 0.0

        return metrics
