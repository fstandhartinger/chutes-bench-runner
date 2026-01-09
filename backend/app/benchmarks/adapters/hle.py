"""Humanity's Last Exam (HLE) benchmark adapter."""
import json
import os
import re
import time
from typing import Any, AsyncIterator, Optional

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.benchmarks.utils import load_dataset_with_retry
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

HLE_SYSTEM_PROMPT = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your answer choice}\n"
    "Answer: {your chosen answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)

HLE_JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if there is no confidence score available.

Return a JSON object with keys: extracted_final_answer, reasoning, correct, confidence."""


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

    def supports_parallel_items(self) -> bool:
        return True

    def requires_setup(self) -> bool:
        return False

    def get_setup_notes(self) -> Optional[str]:
        return None

    def _normalize_answer(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return " ".join(text.split())

    def _extract_judge_payload(self, response_text: str) -> dict[str, Any]:
        if not response_text:
            return {}
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(response_text[start:end + 1])
            except json.JSONDecodeError:
                pass
        payload: dict[str, Any] = {}
        correct_match = re.search(r"correct\s*[:=]\s*(yes|no)", response_text, re.IGNORECASE)
        if correct_match:
            payload["correct"] = correct_match.group(1).lower()
        confidence_match = re.search(r"confidence\s*[:=]\s*(\d+)", response_text, re.IGNORECASE)
        if confidence_match:
            payload["confidence"] = int(confidence_match.group(1))
        extracted_match = re.search(
            r"extracted_final_answer\s*[:=]\s*(.+)", response_text, re.IGNORECASE
        )
        if extracted_match:
            payload["extracted_final_answer"] = extracted_match.group(1).strip()
        reasoning_match = re.search(r"reasoning\s*[:=]\s*(.+)", response_text, re.IGNORECASE)
        if reasoning_match:
            payload["reasoning"] = reasoning_match.group(1).strip()
        return payload

    async def _judge_answer(
        self,
        question: str,
        correct_answer: str,
        response: str,
    ) -> tuple[Optional[bool], dict[str, Any]]:
        judge_prompt = HLE_JUDGE_PROMPT.format(
            question=question,
            correct_answer=correct_answer,
            response=response,
        )
        start_time = time.time()
        judge_text, judge_meta = await self.client.get_completion_text(
            settings.hle_judge_model,
            judge_prompt,
            temperature=0.0,
            max_tokens=4096,
            min_output_tokens=0,
        )
        judge_latency_ms = int((time.time() - start_time) * 1000)
        payload = self._extract_judge_payload(judge_text or "")
        correct_value = payload.get("correct")
        is_correct = None
        if isinstance(correct_value, str):
            if correct_value.strip().lower() == "yes":
                is_correct = True
            elif correct_value.strip().lower() == "no":
                is_correct = False
        payload.setdefault("raw_response", judge_text or "")
        payload["judge_latency_ms"] = judge_latency_ms
        payload["judge_model"] = settings.hle_judge_model
        payload["judge_metadata"] = judge_meta
        return is_correct, payload

    async def get_total_items(self) -> int:
        if not self._items:
            await self.preload()
        return len(self._items)

    async def preload(self) -> None:
        """Load HLE dataset."""
        if self._items:
            return

        try:
            logger.info("Loading Humanity's Last Exam dataset")
            hf_token = os.environ.get("HF_TOKEN")

            if hf_token:
                dataset = await load_dataset_with_retry("cais/hle", split="test", token=hf_token)
            else:
                dataset = await load_dataset_with_retry("cais/hle", split="test")

            self._items = []
            for i, item in enumerate(dataset):
                # Ensure all fields are strings, not None
                question = item.get("question") or ""
                answer = item.get("answer") or ""
                self._items.append({
                    "id": str(i),
                    "question": str(question) if question else "",
                    "answer": str(answer) if answer else "",
                    "answer_type": item.get("answer_type"),
                    "image": item.get("image") or "",
                    "subject": item.get("subject", ""),
                    "source": item.get("source", ""),
                    "category": item.get("category", ""),
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

        question_text = item.get("question", "")
        image_payload = item.get("image") or ""
        user_content: list[dict[str, Any]] = [{"type": "text", "text": question_text}]
        if image_payload:
            user_content.append({"type": "image_url", "image_url": {"url": image_payload}})
        messages = [
            {"role": "system", "content": HLE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        try:
            start_time = time.time()
            response_text, metadata = await self.client.get_completion_messages(
                self.model_slug,
                messages,
                max_tokens=8192,
                min_output_tokens=0,
                temperature=0.0,
            )
            latency_ms = int((time.time() - start_time) * 1000)

            # Robust handling of empty or None response
            if not response_text or response_text is None:
                item_metadata = {
                    **metadata,
                    "subject": item.get("subject"),
                    "category": item.get("category"),
                    "answer_type": item.get("answer_type"),
                    "system_prompt": HLE_SYSTEM_PROMPT,
                    "has_image": bool(image_payload),
                }
                return ItemResult(
                    item_id=item_id,
                    item_hash=self.compute_item_hash(question_text),
                    prompt=question_text,
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

            expected_raw = str(item.get("answer", "")).strip()
            is_correct, judge_output = await self._judge_answer(
                question_text,
                expected_raw,
                str(response_text).strip(),
            )
            error = None
            if is_correct is None:
                error = "Unable to parse judge response"
            error = self.format_truncation_error(metadata, error)

            item_metadata = {
                **metadata,
                "subject": item.get("subject"),
                "category": item.get("category"),
                "answer_type": item.get("answer_type"),
                "system_prompt": HLE_SYSTEM_PROMPT,
                "has_image": bool(image_payload),
            }
            return ItemResult(
                item_id=item_id,
                item_hash=self.compute_item_hash(question_text),
                prompt=question_text,
                response=response_text.strip(),
                expected=expected_raw,
                is_correct=is_correct,
                score=1.0 if is_correct else 0.0 if is_correct is not None else 0.0,
                latency_ms=latency_ms,
                input_tokens=metadata.get("usage", {}).get("prompt_tokens"),
                output_tokens=metadata.get("usage", {}).get("completion_tokens"),
                error=error,
                judge_output=judge_output,
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
                "category": item.get("category"),
                "answer_type": item.get("answer_type"),
                "system_prompt": HLE_SYSTEM_PROMPT,
                "has_image": bool(item.get("image")),
            }
            return ItemResult(
                item_id=item_id, 
                prompt=question_text, 
                response=res if res is not None else "", 
                error=str(e),
                metadata=item_metadata,
            )
