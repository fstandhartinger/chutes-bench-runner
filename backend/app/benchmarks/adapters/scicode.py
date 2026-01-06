"""SciCode benchmark adapter."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.benchmarks.scicode_utils import extract_function_name, get_function_from_code
from app.benchmarks.utils import (
    download_hf_file_async,
    download_http_file_async,
    load_dataset_with_retry,
)
from app.core.logging import get_logger
from app.services.sandy_service import SandyService

logger = get_logger(__name__)

SCICODE_REPO_BASE = "https://raw.githubusercontent.com/scicode-bench/SciCode/main/eval/data"


@register_adapter("scicode")
class SciCodeAdapter(BenchmarkAdapter):
    """
    SciCode benchmark adapter.

    Uses the official multi-step prompting and HDF5 numeric test data.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._items: list[dict[str, Any]] = []
        self._h5_path: Optional[Path] = None
        self._default_template: Optional[str] = None
        self._background_template: Optional[str] = None
        self._special_step_cache: dict[tuple[str, int], str] = {}
        self._sandbox_id: Optional[str] = None
        self.sandy = SandyService()

    def get_name(self) -> str:
        return "scicode"

    def get_display_name(self) -> str:
        return "SciCode"

    def requires_setup(self) -> bool:
        return False

    def get_setup_notes(self) -> Optional[str]:
        return None

    async def get_total_items(self) -> int:
        if not self._items:
            await self.preload()
        return len(self._items)

    async def preload(self) -> None:
        """Load SciCode dataset."""
        if self._items:
            return

        try:
            logger.info("Loading SciCode dataset")
            dataset = await load_dataset_with_retry(
                "SciCode1/SciCode",
                split="test",
                token=os.environ.get("HF_TOKEN"),
            )
            self._items = []
            for i, item in enumerate(dataset):
                self._items.append(
                    {
                        "id": str(i),
                        "problem_id": str(item.get("problem_id")),
                        "problem_name": item.get("problem_name", ""),
                        "required_dependencies": item.get("required_dependencies", ""),
                        "sub_steps": item.get("sub_steps", []),
                    }
                )
            logger.info("Loaded %s SciCode items", len(self._items))
        except Exception as e:
            logger.error("Failed to load SciCode", error=str(e))
            self._items = []
            raise

    async def enumerate_items(self) -> AsyncIterator[str]:
        if not self._items:
            await self.preload()
        for item in self._items:
            yield item["id"]

    async def _ensure_assets(self) -> None:
        if not self._h5_path:
            self._h5_path = await download_hf_file_async(
                repo_id="Srimadh/Scicode-test-data-h5",
                filename="test_data.h5",
                repo_type="dataset",
                token=os.environ.get("HF_TOKEN"),
                cache_subdir="scicode",
            )
        if not self._default_template:
            template_path = await download_http_file_async(
                f"{SCICODE_REPO_BASE}/multistep_template.txt",
                cache_subdir="scicode",
            )
            self._default_template = template_path.read_text(encoding="utf-8")
        if not self._background_template:
            background_path = await download_http_file_async(
                f"{SCICODE_REPO_BASE}/background_comment_template.txt",
                cache_subdir="scicode",
            )
            self._background_template = background_path.read_text(encoding="utf-8")
        if not self._special_step_cache:
            for filename, key in (
                ("13.6.txt", ("13", 5)),
                ("62.1.txt", ("62", 0)),
                ("76.3.txt", ("76", 2)),
            ):
                content_path = await download_http_file_async(
                    f"{SCICODE_REPO_BASE}/{filename}",
                    cache_subdir="scicode",
                )
                self._special_step_cache[key] = content_path.read_text(encoding="utf-8")

    async def _ensure_sandbox(self) -> Optional[str]:
        if self._sandbox_id:
            return self._sandbox_id

        sandbox_id = await self.sandy.create_sandbox()
        if not sandbox_id:
            return None
        self._sandbox_id = sandbox_id

        # Ensure scientific dependencies are available.
        result = await self.sandy.execute_command(
            sandbox_id,
            "python3 - <<'PY'\nimport h5py, numpy, scipy, sympy\nPY",
        )
        if result.get("exit_code") != 0:
            await self.sandy.execute_command(
                sandbox_id,
                "python3 -m pip install --no-cache-dir numpy scipy sympy h5py",
                timeout_ms=600000,
            )

        return sandbox_id

    async def _ensure_h5_in_sandbox(self, sandbox_id: str) -> str:
        target_path = "/workspace/scicode_test_data.h5"
        check = await self.sandy.execute_command(
            sandbox_id,
            f"test -f {target_path}",
        )
        if check.get("exit_code") == 0:
            return target_path

        hf_token = os.environ.get("HF_TOKEN")
        header = f"-H 'Authorization: Bearer {hf_token}'" if hf_token else ""
        url = "https://huggingface.co/datasets/Srimadh/Scicode-test-data-h5/resolve/main/test_data.h5"
        download_cmd = f"curl -L {header} '{url}' -o {target_path}"
        result = await self.sandy.execute_command(
            sandbox_id,
            download_cmd,
            timeout_ms=900000,
        )
        if result.get("exit_code") != 0:
            raise RuntimeError(result.get("stderr") or "Failed to download SciCode test data")
        return target_path

    def _get_prompt_template(self, with_background: bool) -> str:
        return self._background_template if with_background else self._default_template

    def _process_problem_steps(
        self,
        sub_steps: list[dict[str, Any]],
        num_steps: int,
        previous_llm_code: list[Optional[str]],
        with_background: bool,
    ) -> tuple[str, str, str]:
        output_lines: list[str] = []
        next_step: list[str] = []
        previous_code: list[str] = []
        for idx in range(num_steps - 1):
            step = sub_steps[idx]
            step_text = step.get("step_description_prompt", "")
            if with_background and step.get("step_background"):
                step_text = f"{step_text}\n{step.get('step_background')}"
            output_lines.append(step_text)
            output_lines.append(previous_llm_code[idx] or "")
            previous_code.append(previous_llm_code[idx] or "")
            output_lines.append("------")

        current = sub_steps[num_steps - 1]
        current_text = current.get("step_description_prompt", "")
        if with_background and current.get("step_background"):
            current_text = f"{current_text}\n{current.get('step_background')}"
        next_step.append(current_text)
        function_header = current.get("function_header", "")
        return_line = current.get("return_line", "")
        next_step.append(f"{function_header}\n\n{return_line}")

        output_str = "\n\n".join(output_lines[:-1]) if output_lines else ""
        next_step_str = "\n\n".join(next_step)
        previous_code_str = "\n".join(previous_code)
        return output_str, next_step_str, previous_code_str

    async def _run_scicode_tests(
        self,
        sandbox_id: str,
        problem_id: str,
        sub_steps: list[dict[str, Any]],
        step_code: dict[str, str],
        h5_path: str,
    ) -> tuple[bool, dict[str, Any]]:
        skipped_steps: set[int] = set()
        if problem_id == "13":
            skipped_steps.add(5)
        if problem_id == "62":
            skipped_steps.add(0)
        if problem_id == "76":
            skipped_steps.add(2)

        utils_path = Path(__file__).resolve().parents[1] / "scicode_utils.py"
        if utils_path.exists():
            await self.sandy.write_file(
                sandbox_id,
                "scicode_utils.py",
                utils_path.read_text(encoding="utf-8"),
            )

        total_steps = 0
        correct_steps = 0
        failures: list[dict[str, Any]] = []

        for idx, step in enumerate(sub_steps):
            if idx in skipped_steps:
                continue
            step_id = str(step.get("step_number"))
            tests = step.get("test_cases", [])
            code = step_code.get(step_id, "")
            if not code:
                failures.append(
                    {"step_id": step_id, "error": "Missing generated code for step"}
                )
                continue

            test_lines = [
                code,
                "",
                "from scicode_utils import process_hdf5_to_tuple",
                f"targets = process_hdf5_to_tuple('{step_id}', {len(tests)}, '{h5_path}')",
            ]
            for i, test_case in enumerate(tests):
                test_lines.append(f"target = targets[{i}]")
                test_lines.append("")
                test_lines.extend(test_case.splitlines())
            test_content = "\n".join(test_lines)
            filename = f"{step_id}.py"
            await self.sandy.write_file(sandbox_id, filename, test_content)

            result = await self.sandy.execute_command(
                sandbox_id,
                f"python3 {filename}",
                timeout_ms=1800000,
            )
            total_steps += 1
            if result.get("exit_code") == 0:
                correct_steps += 1
            else:
                failures.append(
                    {
                        "step_id": step_id,
                        "stderr": result.get("stderr"),
                        "stdout": result.get("stdout"),
                    }
                )

        problem_correct = correct_steps == total_steps and total_steps > 0
        return problem_correct, {
            "total_steps": total_steps,
            "correct_steps": correct_steps,
            "failures": failures,
        }

    async def evaluate_item(self, item_id: str) -> ItemResult:
        """Evaluate a single SciCode item using official step-by-step prompts."""
        if not self._items:
            await self.preload()

        item = next((i for i in self._items if i["id"] == item_id), None)
        if not item:
            return ItemResult(item_id=item_id, error=f"Item {item_id} not found")

        await self._ensure_assets()
        if not self._h5_path:
            return ItemResult(item_id=item_id, error="SciCode test data unavailable")

        problem_id = item.get("problem_id") or ""
        sub_steps = item.get("sub_steps") or []
        if not sub_steps:
            return ItemResult(item_id=item_id, error="No steps found for SciCode item")

        sandbox_id = await self._ensure_sandbox()
        if not sandbox_id:
            return ItemResult(item_id=item_id, error="Could not create sandbox")

        h5_path = await self._ensure_h5_in_sandbox(sandbox_id)

        with_background = False
        prompt_template = self._get_prompt_template(with_background)
        previous_llm_code: list[Optional[str]] = [None] * len(sub_steps)
        step_code: dict[str, str] = {}

        system_prompt = "Output ONLY the final Python code within a markdown code block. Do NOT provide explanations."

        try:
            start_time = time.time()
            for idx, step in enumerate(sub_steps):
                if problem_id == "13" and idx == 5:
                    previous_llm_code[idx] = self._special_step_cache.get(("13", 5))
                    continue
                if problem_id == "62" and idx == 0:
                    previous_llm_code[idx] = self._special_step_cache.get(("62", 0))
                    continue
                if problem_id == "76" and idx == 2:
                    previous_llm_code[idx] = self._special_step_cache.get(("76", 2))
                    continue

                for prev_idx in range(idx):
                    if previous_llm_code[prev_idx] is None:
                        prev_code = step_code.get(str(sub_steps[prev_idx].get("step_number")))
                        if prev_code:
                            try:
                                func_name = extract_function_name(
                                    sub_steps[prev_idx].get("function_header", "")
                                )
                                previous_llm_code[prev_idx] = get_function_from_code(prev_code, func_name)
                            except Exception:
                                previous_llm_code[prev_idx] = prev_code

                problem_steps_str, next_step_str, previous_code_str = self._process_problem_steps(
                    sub_steps=sub_steps,
                    num_steps=idx + 1,
                    previous_llm_code=previous_llm_code,
                    with_background=with_background,
                )
                dependencies = item.get("required_dependencies", "")
                prompt = prompt_template.format(
                    problem_steps_str=problem_steps_str,
                    next_step_str=next_step_str,
                    dependencies=dependencies,
                )
                response_text, metadata = await self.client.get_completion_text(
                    self.model_slug,
                    prompt,
                    system_prompt=system_prompt,
                    max_tokens=8192,
                    min_output_tokens=0,
                    temperature=0.0,
                )
                if not response_text:
                    return ItemResult(
                        item_id=item_id,
                        item_hash=self.compute_item_hash(problem_id),
                        prompt=prompt,
                        response="",
                        error=self.format_empty_response_error(metadata),
                    )

                extracted_code = self.extract_python_code(response_text)
                previous_llm_code[idx] = extracted_code
                full_code = f"{dependencies}\n{previous_code_str}\n{extracted_code}"
                step_code[str(step.get("step_number"))] = full_code

            latency_ms = int((time.time() - start_time) * 1000)

            problem_correct, test_details = await self._run_scicode_tests(
                sandbox_id,
                problem_id,
                sub_steps,
                step_code,
                h5_path,
            )

            return ItemResult(
                item_id=item_id,
                item_hash=self.compute_item_hash(problem_id),
                prompt="SciCode multi-step prompt",
                response="[Generated code for all steps]",
                expected="[All steps passed]",
                is_correct=problem_correct,
                score=1.0 if problem_correct else 0.0,
                latency_ms=latency_ms,
                judge_output=test_details,
                metadata={
                    "problem_id": problem_id,
                    "system_prompt": system_prompt,
                    "steps": len(sub_steps),
                },
            )
        except Exception as e:
            logger.error("SciCode evaluation failed", item_id=item_id, error=str(e))
            return ItemResult(
                item_id=item_id,
                error=str(e),
                metadata={"problem_id": problem_id},
            )

    async def cleanup(self) -> None:
        if self._sandbox_id:
            await self.sandy.terminate_sandbox(self._sandbox_id)
            self._sandbox_id = None
        return None
