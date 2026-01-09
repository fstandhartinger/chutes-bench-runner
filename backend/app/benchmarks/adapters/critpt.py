"""CritPt benchmark adapter."""
import time
from typing import Any, AsyncIterator, Optional

import httpx

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.benchmarks.utils import load_dataset_with_retry
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

CRITPT_SYSTEM_PROMPT = (
    "You are a physics research assistant specializing in solving complex, "
    "research-level problems using precise, step-by-step reasoning.\n\n"
    "Input\n\n"
    "Problems will be provided in Markdown format.\n\n"
    "Output (Markdown format)\n\n"
    "1. Step-by-Step Derivation - Show every non-trivial step in the solution. "
    "Justify steps using relevant physical laws, theorems, or mathematical identities.\n"
    "2. Mathematical Typesetting - Use LaTeX for all mathematics: `$...$` for inline expressions, "
    "`$$...$$` for display equations.\n"
    "3. Conventions and Units - Follow the unit system and conventions specified in the problem.\n"
    "4. Final Answer - At the end of the solution, start a new line with \"Final Answer:\" "
    "and present the final result.\n"
    "   For final answers involving values, follow the precision requirements specified in the problem.\n"
    "   If no precision is specified:\n"
    "   - If an exact value is possible, provide it (e.g., $\\sqrt(2)$, $\\pi/4$).\n"
    "   - If exact form is not feasible, retain at least 12 significant digits in the result.\n"
    "5. Parsing Structure - After obtaining the final answer, populate it into the code template provided at the end "
    "of the problem. This step is purely for formatting/display purposes. No additional reasoning or derivation should "
    "be performed. Do not import any modules or packages beyond what is provided in the template."
)


@register_adapter("critpt")
class CritPtAdapter(BenchmarkAdapter):
    """CritPt benchmark adapter."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._items: list[dict[str, Any]] = []

    def get_name(self) -> str:
        return "critpt"

    def get_display_name(self) -> str:
        return "CritPt"

    def supports_parallel_items(self) -> bool:
        return True

    def supports_subset(self) -> bool:
        return False

    def requires_setup(self) -> bool:
        return not settings.critpt_eval_url

    def get_setup_notes(self) -> Optional[str]:
        if settings.critpt_eval_url:
            return None
        return "CritPt evaluation requires CRITPT_EVAL_URL (and CRITPT_API_KEY if applicable)."

    async def get_total_items(self) -> int:
        if not self._items:
            await self.preload()
        return len(self._items)

    async def preload(self) -> None:
        if self._items:
            return
        try:
            logger.info("Loading CritPt dataset")
            dataset = await load_dataset_with_retry(
                "CritPt-Benchmark/CritPt",
                split="train",
            )
            self._items = []
            for item in dataset:
                self._items.append(
                    {
                        "id": str(item.get("problem_id")),
                        "problem_id": str(item.get("problem_id")),
                        "problem_type": str(item.get("problem_type") or ""),
                        "problem_description": str(item.get("problem_description") or ""),
                        "code_template": str(item.get("code_template") or ""),
                        "answer_code": str(item.get("answer_only_code") or ""),
                        "metadata_tag": str(item.get("metadata_tag") or ""),
                    }
                )
            logger.info("Loaded %s CritPt items", len(self._items))
        except Exception as exc:
            logger.error("Failed to load CritPt", error=str(exc))
            self._items = []
            raise

    async def enumerate_items(self) -> AsyncIterator[str]:
        if not self._items:
            await self.preload()
        for item in self._items:
            yield item["id"]

    async def get_items_for_evaluation(
        self,
        subset_pct: int,
        seed: str,
        subset_count: Optional[int] = None,
    ) -> tuple[int, list[str]]:
        """CritPt requires submitting the full batch in one evaluation."""
        if not self._items:
            await self.preload()
        all_items = [item["id"] for item in self._items]
        return len(all_items), all_items

    def _build_prompt(self, description: str, code_template: str) -> str:
        return (
            f"{description}\n\n"
            "Code template:\n"
            "```python\n"
            f"{code_template}\n"
            "```\n\n"
            "Fill the template with your final answer. Return the completed code."
        )

    async def evaluate_item(self, item_id: str) -> ItemResult:
        if not self._items:
            await self.preload()

        item = next((i for i in self._items if i["id"] == item_id), None)
        if not item:
            return ItemResult(item_id=item_id, error=f"Item {item_id} not found")

        prompt = self._build_prompt(item.get("problem_description", ""), item.get("code_template", ""))

        start_time = time.time()
        response_text, metadata = await self.client.get_completion_text(
            self.model_slug,
            prompt,
            system_prompt=CRITPT_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=8192,
            min_output_tokens=0,
        )
        latency_ms = int((time.time() - start_time) * 1000)

        if not response_text or not response_text.strip():
            return ItemResult(
                item_id=item_id,
                item_hash=self.compute_item_hash(item),
                prompt=prompt,
                response=response_text,
                expected=item.get("answer_code", ""),
                error=self.format_empty_response_error(metadata),
                latency_ms=latency_ms,
                metadata={
                    "problem_id": item.get("problem_id"),
                    "problem_type": item.get("problem_type"),
                    "metadata_tag": item.get("metadata_tag"),
                    "system_prompt": CRITPT_SYSTEM_PROMPT,
                    **metadata,
                },
            )

        code = self.extract_python_code(response_text)
        if not code:
            return ItemResult(
                item_id=item_id,
                item_hash=self.compute_item_hash(item),
                prompt=prompt,
                response=response_text,
                expected=item.get("answer_code", ""),
                error="No code block found in response",
                latency_ms=latency_ms,
                metadata={
                    "problem_id": item.get("problem_id"),
                    "problem_type": item.get("problem_type"),
                    "metadata_tag": item.get("metadata_tag"),
                    "system_prompt": CRITPT_SYSTEM_PROMPT,
                    **metadata,
                },
            )

        return ItemResult(
            item_id=item_id,
            item_hash=self.compute_item_hash(item),
            prompt=prompt,
            response=code,
            expected=item.get("answer_code", ""),
            latency_ms=latency_ms,
            input_tokens=metadata.get("prompt_tokens") or metadata.get("usage", {}).get("prompt_tokens"),
            output_tokens=metadata.get("completion_tokens") or metadata.get("usage", {}).get("completion_tokens"),
            metadata={
                "problem_id": item.get("problem_id"),
                "problem_type": item.get("problem_type"),
                "metadata_tag": item.get("metadata_tag"),
                "system_prompt": CRITPT_SYSTEM_PROMPT,
                "generation_config": {
                    "temperature": 0.0,
                    "max_tokens": metadata.get("max_tokens"),
                },
                **metadata,
            },
        )

    async def postprocess(self, results: list[ItemResult]) -> dict[str, Any]:
        eval_url = settings.critpt_eval_url.rstrip("/") if settings.critpt_eval_url else ""
        if not eval_url:
            return {
                "critpt_eval_error": "CRITPT_EVAL_URL is not configured",
                "score_override": None,
            }

        submissions = []
        for result in results:
            if not result.response:
                continue
            system_prompt = ""
            generation_config = {}
            if result.metadata:
                system_prompt = result.metadata.get("system_prompt", "")
                generation_config = result.metadata.get("generation_config", {})
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if result.prompt:
                messages.append({"role": "user", "content": result.prompt})
            submissions.append(
                {
                    "problem_id": result.item_id,
                    "generated_code": result.response,
                    "model": self.model_slug,
                    "generation_config": generation_config,
                    "messages": messages,
                }
            )

        if not submissions:
            return {
                "critpt_eval_error": "No valid submissions to evaluate",
                "score_override": None,
            }

        headers = {}
        if settings.critpt_api_key:
            headers["x-api-key"] = settings.critpt_api_key

        batch_metadata = {
            "benchmark": self.get_name(),
            "model": self.model_slug,
            "submission_count": len(submissions),
        }
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(7200.0, connect=10.0)) as client:
                response = await client.post(
                    eval_url,
                    json={"submissions": submissions, "batch_metadata": batch_metadata},
                    headers=headers,
                )
                response.raise_for_status()
                payload = response.json()
        except Exception as exc:
            logger.error("CritPt evaluation failed", error=str(exc))
            return {
                "critpt_eval_error": str(exc) or "CritPt evaluation failed",
                "score_override": None,
            }

        metrics = payload.get("metrics") or payload.get("summary")
        if not isinstance(metrics, dict):
            metrics = {
                key: value
                for key, value in payload.items()
                if isinstance(value, (int, float))
            }
        accuracy = None
        if isinstance(metrics, dict):
            accuracy = metrics.get("accuracy")
        return {
            "critpt_metrics": metrics or payload,
            "score_override": accuracy,
        }
