"""AA-LCR (Artificial Analysis Long Context Reasoning) benchmark adapter."""
import os
import time
import zipfile
from typing import Any, AsyncIterator, Optional

from datasets import load_dataset

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.benchmarks.utils import download_hf_file
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


@register_adapter("aa_lcr")
class AALCRAdapter(BenchmarkAdapter):
    """
    AA-LCR benchmark adapter.

    Official dataset: ArtificialAnalysis/AA-LCR
    Scoring: LLM-based equality checker (Qwen3 235B A22B).
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._items: list[dict[str, Any]] = []
        self._zip_path: Optional[str] = None

    def get_name(self) -> str:
        return "aa_lcr"

    def get_display_name(self) -> str:
        return "AA-LCR"

    async def get_total_items(self) -> int:
        if not self._items:
            await self.preload()
        return len(self._items)

    def _ensure_zip(self) -> Optional[str]:
        if self._zip_path:
            return self._zip_path
        hf_token = os.environ.get("HF_TOKEN")
        try:
            zip_path = download_hf_file(
                repo_id="ArtificialAnalysis/AA-LCR",
                filename="extracted_text/AA-LCR_extracted-text.zip",
                repo_type="dataset",
                token=hf_token,
                cache_subdir="aa_lcr",
            )
            self._zip_path = str(zip_path)
            return self._zip_path
        except Exception as e:
            logger.error("Failed to download AA-LCR documents", error=str(e))
            return None

    async def preload(self) -> None:
        """Load AA-LCR dataset."""
        if self._items:
            return

        try:
            logger.info("Loading AA-LCR dataset")
            hf_token = os.environ.get("HF_TOKEN")
            dataset = load_dataset("ArtificialAnalysis/AA-LCR", split="test", token=hf_token)

            self._items = []
            for i, item in enumerate(dataset):
                filenames = [
                    f.strip()
                    for f in str(item.get("data_source_filenames", "")).split(";")
                    if f.strip()
                ]
                self._items.append(
                    {
                        "id": str(i),
                        "question": str(item.get("question") or ""),
                        "answer": str(item.get("answer") or ""),
                        "document_category": str(item.get("document_category") or ""),
                        "document_set_id": str(item.get("document_set_id") or ""),
                        "data_source_filenames": filenames,
                        "data_source_urls": str(item.get("data_source_urls") or ""),
                        "input_tokens": item.get("input_tokens"),
                    }
                )
            logger.info("Loaded %s AA-LCR items", len(self._items))
        except Exception as e:
            logger.error("Failed to load AA-LCR", error=str(e))
            self._items = []
            raise

    async def enumerate_items(self) -> AsyncIterator[str]:
        if not self._items:
            await self.preload()
        for item in self._items:
            yield item["id"]

    def _load_documents(self, item: dict[str, Any]) -> list[str]:
        zip_path = self._ensure_zip()
        if not zip_path:
            return []

        category = item.get("document_category") or ""
        set_id = item.get("document_set_id") or ""
        filenames = item.get("data_source_filenames") or []
        if not category or not set_id or not filenames:
            return []

        docs: list[str] = []
        base_dir = f"lcr/{category}/{set_id}/"
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                for filename in filenames:
                    path = base_dir + filename
                    try:
                        docs.append(zf.read(path).decode("utf-8", errors="replace"))
                    except KeyError:
                        logger.warning("AA-LCR document missing", path=path)
        except Exception as e:
            logger.error("Failed to read AA-LCR documents", error=str(e))
            return []
        return docs

    def _build_prompt(self, question: str, documents: list[str]) -> str:
        doc_sections = []
        for idx, doc in enumerate(documents, start=1):
            doc_sections.append(f"Document {idx}:\n{doc}\n")
        docs_text = "\n".join(doc_sections)
        return (
            "You are given a set of documents. Answer the question using only the "
            "information in the documents.\n\n"
            f"{docs_text}\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

    async def _judge_answer(self, question: str, official: str, candidate: str) -> tuple[bool, str]:
        judge_prompt = (
            "Assess whether the following CANDIDATE ANSWER is CORRECT or INCORRECT.\n"
            "For the CANDIDATE ANSWER to be correct, it must be consistent with the OFFICIAL ANSWER.\n\n"
            f"The question, for reference only: {question}\n"
            f"The OFFICIAL ANSWER: {official}\n"
            f"CANDIDATE ANSWER TO ASSESS: {candidate}\n\n"
            "Reply only with CORRECT or INCORRECT."
        )
        response, _ = await self.client.get_completion_text(
            settings.aa_lcr_judge_model,
            judge_prompt,
            temperature=0.0,
            max_tokens=32,
        )
        verdict = (response or "").strip().upper()
        if "CORRECT" in verdict and "INCORRECT" not in verdict:
            return True, response.strip()
        if "INCORRECT" in verdict and "CORRECT" not in verdict:
            return False, response.strip()
        return False, response.strip() or "Unrecognized judge response"

    async def evaluate_item(self, item_id: str) -> ItemResult:
        """Evaluate a single AA-LCR item."""
        if not self._items:
            await self.preload()

        item = next((i for i in self._items if i["id"] == item_id), None)
        if not item:
            return ItemResult(item_id=item_id, error=f"Item {item_id} not found")

        documents = self._load_documents(item)
        if not documents:
            return ItemResult(
                item_id=item_id,
                error="AA-LCR documents unavailable - ensure extracted_text bundle is accessible",
            )

        question = item.get("question", "")
        prompt = self._build_prompt(question, documents)

        system_prompt = "You are an expert researcher. Provide a concise, factual answer."
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

            if not response_text:
                item_metadata = {
                    **metadata,
                    "document_category": item.get("document_category"),
                    "document_set_id": item.get("document_set_id"),
                    "system_prompt": system_prompt,
                }
                return ItemResult(
                    item_id=item_id,
                    item_hash=self.compute_item_hash(question),
                    prompt=prompt,
                    response="",
                    expected=item.get("answer", ""),
                    is_correct=False,
                    score=0.0,
                    error=self.format_empty_response_error(metadata),
                    latency_ms=latency_ms,
                    input_tokens=metadata.get("usage", {}).get("prompt_tokens"),
                    output_tokens=metadata.get("usage", {}).get("completion_tokens"),
                    metadata=item_metadata,
                )

            is_correct, judge_response = await self._judge_answer(
                question,
                item.get("answer", ""),
                response_text,
            )
            item_metadata = {
                **metadata,
                "document_category": item.get("document_category"),
                "document_set_id": item.get("document_set_id"),
                "system_prompt": system_prompt,
                "judge_model": settings.aa_lcr_judge_model,
            }
            return ItemResult(
                item_id=item_id,
                item_hash=self.compute_item_hash(question),
                prompt=prompt,
                response=response_text.strip(),
                expected=item.get("answer", ""),
                is_correct=is_correct,
                score=1.0 if is_correct else 0.0,
                latency_ms=latency_ms,
                input_tokens=metadata.get("usage", {}).get("prompt_tokens"),
                output_tokens=metadata.get("usage", {}).get("completion_tokens"),
                judge_output={"judge_response": judge_response},
                metadata=item_metadata,
            )
        except Exception as e:
            logger.error("AA-LCR evaluation failed", item_id=item_id, error=str(e))
            meta = locals().get("metadata") or {}
            item_metadata = {
                **meta,
                "document_category": item.get("document_category"),
                "document_set_id": item.get("document_set_id"),
                "system_prompt": system_prompt,
            }
            return ItemResult(
                item_id=item_id,
                prompt=prompt,
                response=locals().get("response_text", "") or "",
                error=str(e),
                metadata=item_metadata,
            )
