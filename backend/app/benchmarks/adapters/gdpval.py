"""GDPval-AA benchmark adapter."""
import os
import re
import time
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import pandas as pd
from docx import Document
from pypdf import PdfReader

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.benchmarks.utils import download_hf_file_async, load_dataset_with_retry
from app.core.config import get_settings
settings = get_settings()


GDPVAL_JUDGE_PROMPT = """You are judging whether a predicted answer is supported by the reference materials provided.
Use ONLY the reference materials to decide if the answer is correct. If the answer is partially correct or provides
too much detail, mark it as PARTIAL ANSWER. If the model refuses or says it cannot answer, mark NOT ATTEMPTED.

Question: {question}

Reference materials:
{reference_text}

Predicted answer: {predicted_answer}

Grade the predicted answer as one of:
A: CORRECT
B: INCORRECT
C: PARTIAL ANSWER
D: NOT ATTEMPTED

Return only the letter A, B, C, or D."""


@register_adapter("gdpval_aa")
class GDPvalAAAdapter(BenchmarkAdapter):
    """GDPval-AA benchmark adapter (LLM-judged with reference docs)."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._items: list[dict[str, Any]] = []
        self._reference_cache: dict[str, str] = {}

    def get_name(self) -> str:
        return "gdpval_aa"

    def get_display_name(self) -> str:
        return "GDPval-AA"

    def supports_parallel_items(self) -> bool:
        return True

    async def get_total_items(self) -> int:
        if not self._items:
            await self.preload()
        return len(self._items)

    async def preload(self) -> None:
        if self._items:
            return
        try:
            logger.info("Loading GDPval dataset")
            hf_token = os.environ.get("HF_TOKEN")
            dataset = await load_dataset_with_retry(
                "openai/gdpval",
                split="train",
                token=hf_token,
            )
            self._items = []
            for item in dataset:
                self._items.append(
                    {
                        "id": str(item.get("task_id")),
                        "task_id": str(item.get("task_id")),
                        "sector": str(item.get("sector") or ""),
                        "occupation": str(item.get("occupation") or ""),
                        "prompt": str(item.get("prompt") or ""),
                        "reference_files": list(item.get("reference_files") or []),
                    }
                )
            logger.info("Loaded %s GDPval items", len(self._items))
        except Exception as exc:
            logger.error("Failed to load GDPval", error=str(exc))
            self._items = []
            raise

    async def enumerate_items(self) -> AsyncIterator[str]:
        if not self._items:
            await self.preload()
        for item in self._items:
            yield item["id"]

    def _truncate(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + "\n...[truncated]"

    def _extract_text_from_file(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            return path.read_text(encoding="utf-8", errors="replace")
        if suffix in {".csv"}:
            df = pd.read_csv(path)
            return df.to_csv(index=False)
        if suffix in {".xlsx", ".xls", ".xlsm"}:
            data = pd.read_excel(path, sheet_name=None)
            parts = []
            for name, df in data.items():
                parts.append(f"Sheet: {name}\n{df.to_csv(index=False)}")
            return "\n\n".join(parts)
        if suffix == ".pdf":
            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        if suffix == ".docx":
            doc = Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs)
        return f"[Unsupported file type: {suffix}]"

    async def _load_reference_text(self, item: dict[str, Any]) -> str:
        reference_files = item.get("reference_files") or []
        if not reference_files:
            return "[No reference files provided]"

        limit = settings.gdpval_reference_char_limit
        chunks: list[str] = []
        total_chars = 0
        for filename in reference_files:
            cache_key = filename
            if cache_key in self._reference_cache:
                text = self._reference_cache[cache_key]
            else:
                try:
                    hf_token = os.environ.get("HF_TOKEN")
                    path = await download_hf_file_async(
                        repo_id="openai/gdpval",
                        filename=filename,
                        repo_type="dataset",
                        token=hf_token,
                        cache_subdir="gdpval",
                    )
                    text = self._extract_text_from_file(Path(path))
                except Exception as exc:
                    text = f"[Failed to load {filename}: {exc}]"
                self._reference_cache[cache_key] = text

            snippet = self._truncate(text, max(1000, limit - total_chars))
            chunks.append(f"Reference file: {filename}\n{snippet}")
            total_chars += len(snippet)
            if total_chars >= limit:
                break
        return "\n\n".join(chunks)

    def _parse_grade_letter(self, response: str) -> Optional[str]:
        if not response:
            return None
        cleaned = re.sub(r"(?i)<think>.*?</think>", "", response, flags=re.DOTALL).strip()
        if not cleaned:
            return None
        letters = re.findall(r"\b([ABCD])\b", cleaned.upper())
        if letters:
            return letters[-1]
        if "CORRECT" in cleaned.upper():
            return "A"
        if "INCORRECT" in cleaned.upper():
            return "B"
        if "PARTIAL" in cleaned.upper():
            return "C"
        if "NOT ATTEMPTED" in cleaned.upper() or "ABSTAIN" in cleaned.upper():
            return "D"
        return None

    async def _judge_answer(
        self,
        question: str,
        reference_text: str,
        response: str,
    ) -> tuple[Optional[str], dict[str, Any]]:
        judge_prompt = GDPVAL_JUDGE_PROMPT.format(
            question=question,
            reference_text=reference_text,
            predicted_answer=response,
        )
        start_time = time.time()
        judge_text, judge_meta = await self.judge_client.get_completion_text(
            settings.gdpval_judge_model,
            judge_prompt,
            temperature=0.0,
            max_tokens=256,
            min_output_tokens=0,
        )
        judge_latency_ms = int((time.time() - start_time) * 1000)
        letter = self._parse_grade_letter(judge_text or "")
        payload = {
            "grade_letter": letter,
            "raw_response": judge_text or "",
            "judge_latency_ms": judge_latency_ms,
            "judge_model": settings.gdpval_judge_model,
            "judge_metadata": judge_meta,
        }
        return letter, payload

    async def evaluate_item(self, item_id: str) -> ItemResult:
        if not self._items:
            await self.preload()

        item = next((i for i in self._items if i["id"] == item_id), None)
        if not item:
            return ItemResult(item_id=item_id, error=f"Item {item_id} not found")

        prompt = item.get("prompt", "")
        reference_text = await self._load_reference_text(item)
        full_prompt = (
            f"{prompt}\n\n"
            "Reference materials:\n"
            f"{reference_text}\n\n"
            "Answer:"
        )

        start_time = time.time()
        response_text, metadata = await self.client.get_completion_text(
            self.model_slug,
            full_prompt,
            temperature=0.0,
            max_tokens=4096,
            min_output_tokens=0,
        )
        latency_ms = int((time.time() - start_time) * 1000)

        if not response_text or not response_text.strip():
            return ItemResult(
                item_id=item_id,
                item_hash=self.compute_item_hash(item),
                prompt=full_prompt,
                response=response_text,
                expected=None,
                error=self.format_empty_response_error(metadata),
                latency_ms=latency_ms,
                metadata={
                    "sector": item.get("sector"),
                    "occupation": item.get("occupation"),
                    "task_id": item.get("task_id"),
                    "reference_files": item.get("reference_files"),
                    **metadata,
                },
            )

        grade_letter, judge_output = await self._judge_answer(prompt, reference_text, response_text)
        grade_label = {
            "A": "CORRECT",
            "B": "INCORRECT",
            "C": "PARTIAL ANSWER",
            "D": "NOT ATTEMPTED",
        }.get(grade_letter)
        if judge_output is not None:
            judge_output["grade"] = grade_label
            judge_output["grade_letter"] = grade_letter
            if grade_letter is None:
                judge_output["error"] = "Unrecognized judge response"

        score = None
        is_correct = None
        if grade_letter == "A":
            score = 1.0
            is_correct = True
        elif grade_letter == "B":
            score = 0.0
            is_correct = False
        elif grade_letter == "C":
            score = 0.5
        elif grade_letter == "D":
            score = 0.0

        return ItemResult(
            item_id=item_id,
            item_hash=self.compute_item_hash(item),
            prompt=full_prompt,
            response=response_text,
            expected=None,
            is_correct=is_correct,
            score=score,
            judge_output=judge_output,
            latency_ms=latency_ms,
            input_tokens=metadata.get("prompt_tokens") or metadata.get("usage", {}).get("prompt_tokens"),
            output_tokens=metadata.get("completion_tokens") or metadata.get("usage", {}).get("completion_tokens"),
            metadata={
                "sector": item.get("sector"),
                "occupation": item.get("occupation"),
                "task_id": item.get("task_id"),
                "reference_files": item.get("reference_files"),
                "grade": grade_label,
                "grade_letter": grade_letter,
                **metadata,
            },
        )

    async def postprocess(self, results: list[ItemResult]) -> dict[str, Any]:
        counts = {"correct": 0, "incorrect": 0, "partial": 0, "abstain": 0, "errors": 0}
        for result in results:
            grade = None
            if result.metadata:
                grade = result.metadata.get("grade")
            if not grade and result.judge_output:
                grade = result.judge_output.get("grade")
            if result.error:
                counts["errors"] += 1
            if grade == "CORRECT":
                counts["correct"] += 1
            elif grade == "INCORRECT":
                counts["incorrect"] += 1
            elif grade == "PARTIAL ANSWER":
                counts["partial"] += 1
            elif grade == "NOT ATTEMPTED":
                counts["abstain"] += 1
            elif result.error:
                counts["abstain"] += 1
            else:
                counts["incorrect"] += 1
        total = counts["correct"] + counts["incorrect"] + counts["partial"] + counts["abstain"]
        accuracy = counts["correct"] / total if total else 0.0
        return {
            "grade_correct": counts["correct"],
            "grade_incorrect": counts["incorrect"],
            "grade_partial": counts["partial"],
            "grade_abstain": counts["abstain"],
            "grade_errors": counts["errors"],
            "score_override": accuracy if total else 0.0,
        }
