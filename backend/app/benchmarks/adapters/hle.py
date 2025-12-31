"""Humanity's Last Exam (HLE) benchmark adapter."""
import os
import time
import re
from typing import Any, AsyncIterator, Optional

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.core.logging import get_logger

logger = get_logger(__name__)


@register_adapter("hle")
class HLEAdapter(BenchmarkAdapter):
    """
    Humanity's Last Exam benchmark adapter.
    
    A challenging benchmark of expert-level questions across many domains.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._items: list[dict[str, Any]] = []

    def get_name(self) -> str:
        return "hle"

    def get_display_name(self) -> str:
        return "Humanity's Last Exam"

    def requires_setup(self) -> bool:
        return False

    def get_setup_notes(self) -> Optional[str]:
        return None

    def _normalize_answer(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return " ".join(text.split())

    async def get_total_items(self) -> int:
        if not self._items:
            await self.preload()
        return len(self._items)

    async def preload(self) -> None:
        """Load HLE dataset."""
        if self._items:
            return

        try:
            from datasets import load_dataset

            logger.info("Loading Humanity's Last Exam dataset")
            hf_token = os.environ.get("HF_TOKEN")

            if hf_token:
                dataset = load_dataset("cais/hle", split="test", token=hf_token)
            else:
                dataset = load_dataset("cais/hle", split="test")

            self._items = []
            for i, item in enumerate(dataset):
                # Ensure all fields are strings, not None
                question = item.get("question") or ""
                answer = item.get("answer") or ""
                self._items.append({
                    "id": str(i),
                    "question": str(question) if question else "",
                    "answer": str(answer) if answer else "",
                    "subject": item.get("subject", ""),
                    "source": item.get("source", ""),
                })
            
            logger.info(f"Loaded {len(self._items)} HLE items")
        except Exception as e:
            detail = str(e)
            if not os.environ.get("HF_TOKEN"):
                detail = f"HF_TOKEN is required for HLE (gated dataset). {detail}"
            logger.error("Failed to load HLE", error=detail)
            self._items = []
            raise RuntimeError(detail) from e

    async def enumerate_items(self) -> AsyncIterator[str]:
        if not self._items:
            await self.preload()
        for item in self._items:
            yield item["id"]

    async def evaluate_item(self, item_id: str) -> ItemResult:
        """Evaluate a single HLE item."""
        if not self._items:
            await self.preload()

        item = next((i for i in self._items if i["id"] == item_id), None)
        if not item:
            return ItemResult(item_id=item_id, error=f"Item {item_id} not found")

        prompt = f"""Answer the following question. Provide a clear, concise answer.

Question: {item["question"]}

Answer:"""

        system_prompt = "You are an expert in all fields. Answer questions accurately and concisely."
        try:
            start_time = time.time()
            response_text, metadata = await self.client.get_completion_text(
                self.model_slug,
                prompt,
                system_prompt=system_prompt,
                max_tokens=4096,
                temperature=0.0,
            )
            latency_ms = int((time.time() - start_time) * 1000)

            # Robust handling of empty or None response
            if not response_text or response_text is None:
                item_metadata = {
                    **metadata,
                    "subject": item.get("subject"),
                    "system_prompt": system_prompt,
                }
                return ItemResult(
                    item_id=item_id,
                    item_hash=self.compute_item_hash(item["question"]),
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

            # HLE uses exact match or semantic similarity
            # Ensure answer is not None
            expected_raw = str(item.get("answer", "")).strip()
            response_raw = str(response_text).strip()
            expected = self._normalize_answer(expected_raw)
            response_clean = self._normalize_answer(response_raw)
            is_correct = expected == response_clean

            item_metadata = {
                **metadata,
                "subject": item.get("subject"),
                "system_prompt": system_prompt,
            }
            return ItemResult(
                item_id=item_id,
                item_hash=self.compute_item_hash(item["question"]),
                prompt=prompt,
                response=response_text.strip(),
                expected=expected_raw,
                is_correct=is_correct,
                score=1.0 if is_correct else 0.0,
                latency_ms=latency_ms,
                input_tokens=metadata.get("usage", {}).get("prompt_tokens"),
                output_tokens=metadata.get("usage", {}).get("completion_tokens"),
                metadata=item_metadata,
            )

        except Exception as e:
            logger.error("HLE evaluation failed", item_id=item_id, error=str(e))
            # Safely capture what we have
            res = locals().get("response_text", "")
            meta = locals().get("metadata") or {}
            item_metadata = {
                **meta,
                "subject": item.get("subject"),
                "system_prompt": system_prompt,
            }
            return ItemResult(
                item_id=item_id, 
                prompt=prompt, 
                response=res if res is not None else "", 
                error=str(e),
                metadata=item_metadata,
            )
