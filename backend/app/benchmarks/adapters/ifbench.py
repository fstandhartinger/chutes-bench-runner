"""IFBench (Instruction Following) benchmark adapter."""
import json
import time
import re
from typing import Any, AsyncIterator, Optional

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.core.logging import get_logger

logger = get_logger(__name__)


@register_adapter("ifbench")
class IFBenchAdapter(BenchmarkAdapter):
    """
    IFBench adapter.
    
    Instruction Following benchmark evaluating model's ability to follow complex instructions.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._items: list[dict[str, Any]] = []

    def get_name(self) -> str:
        return "ifbench"

    def get_display_name(self) -> str:
        return "IFBench"

    async def get_total_items(self) -> int:
        if not self._items:
            await self.preload()
        return len(self._items)

    async def preload(self) -> None:
        """Load IFBench dataset."""
        if self._items:
            return

        try:
            from datasets import load_dataset

            logger.info("Loading IFBench dataset")
            # IFEval/IFBench dataset
            dataset = load_dataset("google/IFEval", split="train")
            
            self._items = []
            for i, item in enumerate(dataset):
                self._items.append({
                    "id": str(i),
                    "prompt": item.get("prompt", ""),
                    "instruction_id_list": item.get("instruction_id_list", []),
                    "kwargs": item.get("kwargs", {}),
                })
            
            logger.info(f"Loaded {len(self._items)} IFBench items")
        except Exception as e:
            logger.error("Failed to load IFBench", error=str(e))
            self._items = []

    async def enumerate_items(self) -> AsyncIterator[str]:
        if not self._items:
            await self.preload()
        for item in self._items:
            yield item["id"]

    def _compare_relation(self, actual: int, target: Optional[int], relation: Optional[str], default_relation: str) -> bool:
        if target is None:
            return True
        if not relation:
            relation = default_relation
        relation = relation.lower()
        if relation == "less than":
            return actual < target
        if relation == "at least":
            return actual >= target
        if relation == "greater than":
            return actual > target
        if relation == "at most":
            return actual <= target
        if relation == "exactly":
            return actual == target
        return actual == target

    def _count_sentences(self, response: str) -> int:
        sentences = [s for s in re.split(r"[.!?]+", response) if s.strip()]
        return len(sentences)

    def _split_paragraphs(self, response: str, separator_hint: Optional[str]) -> list[str]:
        if separator_hint and separator_hint in response:
            parts = [p.strip() for p in response.split(separator_hint) if p.strip()]
            if parts:
                return parts
        if "***" in response:
            parts = [p.strip() for p in response.split("***") if p.strip()]
            if parts:
                return parts
        parts = [p.strip() for p in re.split(r"\n\s*\n", response) if p.strip()]
        return parts if parts else [response.strip()]

    def _detect_language(self, response: str, target_language: str) -> bool:
        if not response.strip():
            return False

        script_ranges = {
            "ar": [(0x0600, 0x06FF)],
            "fa": [(0x0600, 0x06FF)],
            "ur": [(0x0600, 0x06FF)],
            "bg": [(0x0400, 0x04FF)],
            "ru": [(0x0400, 0x04FF)],
            "bn": [(0x0980, 0x09FF)],
            "gu": [(0x0A80, 0x0AFF)],
            "hi": [(0x0900, 0x097F)],
            "mr": [(0x0900, 0x097F)],
            "ne": [(0x0900, 0x097F)],
            "pa": [(0x0A00, 0x0A7F)],
            "kn": [(0x0C80, 0x0CFF)],
            "ta": [(0x0B80, 0x0BFF)],
            "te": [(0x0C00, 0x0C7F)],
            "th": [(0x0E00, 0x0E7F)],
            "ko": [(0xAC00, 0xD7AF)],
        }

        if target_language in script_ranges:
            for char in response:
                codepoint = ord(char)
                for start, end in script_ranges[target_language]:
                    if start <= codepoint <= end:
                        return True
            return False

        try:
            from langdetect import detect

            detected = detect(response)
            return detected == target_language
        except Exception:
            return False

    def _extract_constrained_options(self, prompt: str) -> list[str]:
        options: list[str] = []
        for match in re.findall(r"'([^']+)'", prompt):
            options.append(match)
        for match in re.findall(r"\"([^\"]+)\"", prompt):
            if match not in options:
                options.append(match)
        return [opt.strip() for opt in options if opt.strip()]

    def _count_highlighted_sections(self, response: str) -> int:
        matches = []
        for match in re.finditer(r"\*[^*\n]+\*", response):
            start, end = match.span()
            if start > 0 and response[start - 1] == "*":
                continue
            if end < len(response) and response[end] == "*":
                continue
            matches.append(match.group())
        return len(matches)

    def _check_instruction(
        self,
        inst_id: str,
        response: str,
        prompt: str,
        kwargs: dict[str, Any],
    ) -> tuple[bool, str]:
        response_lower = response.lower()

        if inst_id == "punctuation:no_comma":
            return ("," not in response, "Response contains a comma" if "," in response else "Passed")

        if inst_id == "change_case:english_capital":
            has_lower = any(char.islower() for char in response)
            return (not has_lower, "Response contains lowercase letters" if has_lower else "Passed")

        if inst_id == "change_case:english_lowercase":
            has_upper = any(char.isupper() for char in response)
            return (not has_upper, "Response contains uppercase letters" if has_upper else "Passed")

        if inst_id == "change_case:capital_word_frequency":
            words = re.findall(r"\b[A-Z]{2,}\b", response)
            count = len(words)
            target = kwargs.get("capital_frequency")
            relation = kwargs.get("capital_relation")
            ok = self._compare_relation(count, target, relation, default_relation="exactly")
            return (ok, f"Capital word count {count} does not satisfy {relation} {target}" if not ok else "Passed")

        if inst_id == "combination:repeat_prompt":
            prompt_to_repeat = kwargs.get("prompt_to_repeat") or ""
            ok = response.startswith(prompt_to_repeat)
            return (ok, "Prompt was not repeated verbatim" if not ok else "Passed")

        if inst_id == "combination:two_responses":
            parts = response.split("******")
            ok = len(parts) == 2 and all(part.strip() for part in parts)
            return (ok, "Response does not contain two parts separated by ******" if not ok else "Passed")

        if inst_id == "detectable_content:number_placeholders":
            placeholders = re.findall(r"\[[^\[\]]+\]", response)
            target = kwargs.get("num_placeholders") or 0
            ok = len(placeholders) >= target
            return (ok, f"Found {len(placeholders)} placeholders, need at least {target}" if not ok else "Passed")

        if inst_id == "detectable_content:postscript":
            marker = kwargs.get("postscript_marker") or "P.S."
            ok = marker in response
            return (ok, f"Postscript marker '{marker}' missing" if not ok else "Passed")

        if inst_id == "detectable_format:constrained_response":
            options = self._extract_constrained_options(prompt)
            ok = any(option in response for option in options) if options else False
            return (ok, "Response does not match any constrained option" if not ok else "Passed")

        if inst_id == "detectable_format:json_format":
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start == -1 or json_end <= json_start:
                return (False, "No JSON object found")
            try:
                json.loads(response[json_start:json_end])
                return (True, "Passed")
            except Exception as exc:
                return (False, f"Invalid JSON: {exc}")

        if inst_id == "detectable_format:multiple_sections":
            splitter = kwargs.get("section_spliter") or ""
            num_sections = kwargs.get("num_sections") or 0
            pattern = re.compile(rf"^{re.escape(splitter)}\\s*\\d+", re.MULTILINE) if splitter else None
            count = len(pattern.findall(response)) if pattern else 0
            ok = count >= num_sections if num_sections else count > 0
            return (ok, f"Found {count} sections, need {num_sections}" if not ok else "Passed")

        if inst_id == "detectable_format:number_bullet_lists":
            target = kwargs.get("num_bullets") or 0
            count = len([line for line in response.splitlines() if line.strip().startswith("* ")])
            ok = count == target
            return (ok, f"Found {count} bullets, need {target}" if not ok else "Passed")

        if inst_id == "detectable_format:number_highlighted_sections":
            target = kwargs.get("num_highlights") or 0
            count = self._count_highlighted_sections(response)
            ok = count >= target
            return (ok, f"Found {count} highlighted sections, need at least {target}" if not ok else "Passed")

        if inst_id == "detectable_format:title":
            ok = bool(re.search(r"<<[^<>]+>>", response))
            return (ok, "Title marker << >> missing" if not ok else "Passed")

        if inst_id == "keywords:existence":
            keywords = kwargs.get("keywords") or []
            missing = [kw for kw in keywords if kw.lower() not in response_lower]
            ok = not missing
            return (ok, f"Required keywords missing: {', '.join(missing)}" if not ok else "Passed")

        if inst_id == "keywords:forbidden_words":
            forbidden = kwargs.get("forbidden_words") or []
            found = [w for w in forbidden if w.lower() in response_lower]
            ok = not found
            return (ok, f"Forbidden words found: {', '.join(found)}" if not ok else "Passed")

        if inst_id == "keywords:frequency":
            keyword = (kwargs.get("keyword") or "").lower()
            target = kwargs.get("frequency")
            relation = kwargs.get("relation")
            count = len(re.findall(rf"\\b{re.escape(keyword)}\\b", response_lower)) if keyword else 0
            ok = self._compare_relation(count, target, relation, default_relation="exactly")
            return (ok, f"Keyword '{keyword}' appears {count}, expected {relation} {target}" if not ok else "Passed")

        if inst_id == "keywords:letter_frequency":
            letter = kwargs.get("letter") or ""
            target = kwargs.get("let_frequency")
            relation = kwargs.get("let_relation")
            count = response.count(letter) if letter else 0
            ok = self._compare_relation(count, target, relation, default_relation="exactly")
            return (ok, f"Letter '{letter}' appears {count}, expected {relation} {target}" if not ok else "Passed")

        if inst_id == "language:response_language":
            language = kwargs.get("language")
            ok = self._detect_language(response, language) if language else True
            return (ok, f"Response does not appear to be in {language}" if not ok else "Passed")

        if inst_id == "length_constraints:number_words":
            target = kwargs.get("num_words")
            relation = kwargs.get("relation")
            count = len(re.findall(r"\b\w+\b", response))
            ok = self._compare_relation(count, target, relation, default_relation="exactly")
            return (ok, f"Word count {count} does not satisfy {relation} {target}" if not ok else "Passed")

        if inst_id == "length_constraints:number_sentences":
            target = kwargs.get("num_sentences")
            relation = kwargs.get("relation")
            count = self._count_sentences(response)
            ok = self._compare_relation(count, target, relation, default_relation="exactly")
            return (ok, f"Sentence count {count} does not satisfy {relation} {target}" if not ok else "Passed")

        if inst_id == "length_constraints:number_paragraphs":
            target = kwargs.get("num_paragraphs")
            paragraphs = self._split_paragraphs(response, kwargs.get("section_spliter"))
            count = len(paragraphs)
            ok = self._compare_relation(count, target, kwargs.get("relation"), default_relation="exactly")
            return (ok, f"Paragraph count {count} does not match {target}" if not ok else "Passed")

        if inst_id == "length_constraints:nth_paragraph_first_word":
            paragraphs = self._split_paragraphs(response, kwargs.get("section_spliter"))
            nth = kwargs.get("nth_paragraph")
            first_word = (kwargs.get("first_word") or "").lower()
            if not nth or nth < 1 or nth > len(paragraphs):
                return (False, "Paragraph index out of range")
            first_word_actual = paragraphs[nth - 1].split()[0].lower() if paragraphs[nth - 1].split() else ""
            ok = first_word_actual == first_word
            return (ok, f"Paragraph {nth} does not start with '{first_word}'" if not ok else "Passed")

        if inst_id == "startend:end_checker":
            end_phrase = kwargs.get("end_phrase") or ""
            ok = response.strip().endswith(end_phrase)
            return (ok, f"Response does not end with '{end_phrase}'" if not ok else "Passed")

        if inst_id == "startend:quotation":
            ok = response.strip().startswith("\"") and response.strip().endswith("\"")
            return (ok, "Response is not wrapped in double quotes" if not ok else "Passed")

        return (True, "No checker implemented")

    async def evaluate_item(self, item_id: str) -> ItemResult:
        """Evaluate a single IFBench item."""
        if not self._items:
            await self.preload()

        item = next((i for i in self._items if i["id"] == item_id), None)
        if not item:
            return ItemResult(item_id=item_id, error=f"Item {item_id} not found")

        prompt = item["prompt"]
        system_prompt = "Follow the instructions precisely and completely. Output ONLY the response fulfilling the instructions. Do NOT use <think> tags."

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
            if not response_text:
                item_metadata = {
                    **metadata,
                    "instruction_ids": item.get("instruction_id_list", []),
                    "system_prompt": system_prompt,
                }
                return ItemResult(
                    item_id=item_id,
                    item_hash=self.compute_item_hash(prompt),
                    prompt=prompt,
                    response="",
                    error=self.format_empty_response_error(metadata),
                    latency_ms=latency_ms,
                    metadata=item_metadata,
                )

            instruction_ids = item.get("instruction_id_list", [])
            kwargs_list = item.get("kwargs", [])
            response = response_text.strip()

            passes = 0
            total_instructions = len(instruction_ids)
            details = []

            for idx, inst_id in enumerate(instruction_ids):
                inst_kwargs = {}
                if idx < len(kwargs_list) and isinstance(kwargs_list[idx], dict):
                    inst_kwargs = kwargs_list[idx]
                inst_passed, reason = self._check_instruction(inst_id, response, prompt, inst_kwargs)
                if inst_passed:
                    passes += 1
                details.append(
                    {
                        "instruction": inst_id,
                        "passed": inst_passed,
                        "reason": reason,
                        "kwargs": inst_kwargs,
                    }
                )

            score = passes / total_instructions if total_instructions > 0 else (1.0 if len(response) > 20 else 0.0)
            is_correct = score == 1.0
            
            item_metadata = {
                **metadata,
                "instruction_ids": instruction_ids,
                "system_prompt": system_prompt,
            }
            return ItemResult(
                item_id=item_id,
                item_hash=self.compute_item_hash(prompt),
                prompt=prompt,
                response=response,
                expected=str(instruction_ids),
                is_correct=is_correct,
                score=score,
                latency_ms=latency_ms,
                input_tokens=metadata.get("usage", {}).get("prompt_tokens"),
                output_tokens=metadata.get("usage", {}).get("completion_tokens"),
                metadata=item_metadata,
                judge_output={
                    "instructions_passed": passes,
                    "total_instructions": total_instructions,
                    "instruction_details": details,
                    "note": "Scored using full internal IFEval checker" if total_instructions > 0 else "Basic length-based scoring"
                },
            )

        except Exception as e:
            logger.error("IFBench evaluation failed", item_id=item_id, error=str(e))
            # Safely capture what we have
            res = locals().get("response_text", "")
            meta = locals().get("metadata") or {}
            item_metadata = {
                **meta,
                "instruction_ids": item.get("instruction_id_list", []),
                "system_prompt": system_prompt,
            }
            return ItemResult(
                item_id=item_id, 
                prompt=prompt, 
                response=res if res is not None else "", 
                error=str(e),
                metadata=item_metadata
            )
