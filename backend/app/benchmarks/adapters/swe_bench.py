"""SWE-Bench Pro benchmark adapter."""
from __future__ import annotations

import ast
import json
import os
import time
from typing import Any, AsyncIterator, Optional

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.benchmarks.utils import download_http_file, load_dataset_with_retry
from app.core.logging import get_logger
from app.services.sandy_service import SandyService

logger = get_logger(__name__)

SWE_BENCH_REPO = "https://raw.githubusercontent.com/scaleapi/SWE-bench_Pro-os/main"


@register_adapter("swe_bench_pro")
class SWEBenchProAdapter(BenchmarkAdapter):
    """
    SWE-Bench Pro adapter.

    Uses the official evaluation scripts with Docker Hub images.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._items: list[dict[str, Any]] = []
        self.sandy = SandyService()

    def get_name(self) -> str:
        return "swe_bench_pro"

    def get_display_name(self) -> str:
        return "SWE-Bench Pro"

    def requires_setup(self) -> bool:
        return False

    def get_setup_notes(self) -> Optional[str]:
        return None

    def supports_subset(self) -> bool:
        return True

    def get_item_timeout_seconds(self) -> Optional[int]:
        return 3600

    async def get_total_items(self) -> int:
        if not self._items:
            await self.preload()
        return len(self._items)

    async def preload(self) -> None:
        """Load SWE-Bench Pro dataset."""
        if self._items:
            return

        try:
            logger.info("Loading SWE-Bench Pro dataset")
            dataset = await load_dataset_with_retry(
                "ScaleAI/SWE-bench_Pro",
                split="test",
                token=os.environ.get("HF_TOKEN"),
            )
            self._items = []
            for i, item in enumerate(dataset):
                self._items.append(
                    {
                        "id": str(i),
                        "instance_id": str(item.get("instance_id", "")),
                        "repo": str(item.get("repo", "")),
                        "problem_statement": str(item.get("problem_statement", "")),
                        "before_repo_set_cmd": str(item.get("before_repo_set_cmd", "")),
                        "selected_test_files_to_run": str(item.get("selected_test_files_to_run", "")),
                        "base_commit": str(item.get("base_commit", "")),
                        "fail_to_pass": str(item.get("fail_to_pass", "")),
                        "pass_to_pass": str(item.get("pass_to_pass", "")),
                    }
                )
            logger.info("Loaded %s SWE-Bench Pro items", len(self._items))
        except Exception as e:
            logger.error("Failed to load SWE-Bench Pro", error=str(e))
            self._items = []
            raise

    async def enumerate_items(self) -> AsyncIterator[str]:
        if not self._items:
            await self.preload()
        for item in self._items:
            yield item["id"]

    def _download_run_script(self, instance_id: str, filename: str) -> str:
        path = download_http_file(
            f"{SWE_BENCH_REPO}/run_scripts/{instance_id}/{filename}",
            cache_subdir=f"swe_bench/{instance_id}",
            filename=filename,
        )
        return path.read_text(encoding="utf-8")

    def _download_dockerfile(self, folder: str, instance_id: str) -> str:
        path = download_http_file(
            f"{SWE_BENCH_REPO}/dockerfiles/{folder}/{instance_id}/Dockerfile",
            cache_subdir=f"swe_bench/{instance_id}",
            filename=f"{folder}.Dockerfile",
        )
        return path.read_text(encoding="utf-8")

    def _get_dockerhub_image_uri(self, uid: str, repo_name: str, dockerhub_username: str) -> str:
        repo_base, repo_name_only = repo_name.lower().split("/")
        hsh = uid.replace("instance_", "")
        if uid == "instance_element-hq__element-web-ec0f940ef0e8e3b61078f145f34dc40d1938e6c5-vnan":
            repo_name_only = "element-web"
        elif "element-hq" in repo_name.lower() and "element-web" in repo_name.lower():
            repo_name_only = "element"
            if hsh.endswith("-vnan"):
                hsh = hsh[:-5]
        elif hsh.endswith("-vnan"):
            hsh = hsh[:-5]
        tag = f"{repo_base}.{repo_name_only}-{hsh}"
        if len(tag) > 128:
            tag = tag[:128]
        return f"{dockerhub_username}/sweap-images:{tag}"

    def _create_entryscript(self, sample: dict[str, Any]) -> str:
        before_repo_set_cmd = sample.get("before_repo_set_cmd", "").strip()
        try:
            selected = ast.literal_eval(sample.get("selected_test_files_to_run", "[]"))
        except Exception:
            selected = []
        selected_test_files_to_run = ",".join(selected)
        base_commit = sample.get("base_commit", "")
        base_dockerfile = self._download_dockerfile("base_dockerfile", sample["instance_id"])
        instance_dockerfile = self._download_dockerfile("instance_dockerfile", sample["instance_id"])

        env_cmds: list[str] = []
        for dockerfile_content in (base_dockerfile, instance_dockerfile):
            for line in dockerfile_content.splitlines():
                line = line.strip()
                if line.startswith("ENV"):
                    env_cmds.append(line.replace("ENV", "export", 1))
        env_block = "\n".join(env_cmds)

        entry_script = f"""
{env_block}
# apply patch
cd /app
git reset --hard {base_commit}
git checkout {base_commit}
git apply -v /workspace/patch.diff
{before_repo_set_cmd}
# run test and save stdout and stderr to separate files
bash /workspace/run_script.sh {selected_test_files_to_run} > /workspace/stdout.log 2> /workspace/stderr.log
# run parsing script
python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json
""".strip()
        return entry_script + "\n"

    async def _read_file(self, sandbox_id: str, path: str) -> str:
        result = await self.sandy.execute_command(sandbox_id, f"cat {path}")
        return result.get("stdout", "")

    async def _get_env(self, sandbox_id: str, key: str) -> str:
        result = await self.sandy.execute_command(sandbox_id, f"printenv {key}")
        return result.get("stdout", "")

    async def evaluate_item(self, item_id: str) -> ItemResult:
        """Evaluate a single SWE-Bench Pro item."""
        if not self._items:
            await self.preload()

        item = next((i for i in self._items if i["id"] == item_id), None)
        if not item:
            return ItemResult(item_id=item_id, error=f"Item {item_id} not found")

        prompt = (
            "You are a software engineer fixing a bug in a GitHub repository. "
            "Provide a unified diff patch that fixes the issue.\n\n"
            f"Repository: {item.get('repo', 'unknown')}\n"
            f"Issue Description:\n{item.get('problem_statement')}\n\n"
            "Patch:\n```diff\n"
        )
        system_prompt = "Output ONLY the final git diff within a markdown code block. No explanations."

        try:
            start_time = time.time()
            response_text, metadata = await self.client.get_completion_text(
                self.model_slug,
                prompt,
                system_prompt=system_prompt,
                max_tokens=8192,
                temperature=0.0,
            )
            latency_ms = int((time.time() - start_time) * 1000)

            if not response_text:
                return ItemResult(
                    item_id=item_id,
                    item_hash=self.compute_item_hash(item.get("instance_id")),
                    prompt=prompt,
                    response="",
                    error=self.format_empty_response_error(metadata),
                    latency_ms=latency_ms,
                    metadata={"instance_id": item.get("instance_id"), "system_prompt": system_prompt},
                )

            patch = self.extract_python_code(response_text)
            sandbox_id = await self.sandy.create_sandbox()
            if not sandbox_id:
                return ItemResult(item_id=item_id, error="Could not create sandbox")

            try:
                entryscript = self._create_entryscript(item)
                await self.sandy.write_file(sandbox_id, "patch.diff", patch)
                await self.sandy.write_file(sandbox_id, "run_script.sh", self._download_run_script(item["instance_id"], "run_script.sh"))
                await self.sandy.write_file(sandbox_id, "parser.py", self._download_run_script(item["instance_id"], "parser.py"))
                await self.sandy.write_file(sandbox_id, "entryscript.sh", entryscript)

                dockerhub_username = "jefzda"
                image_uri = self._get_dockerhub_image_uri(
                    item["instance_id"], item.get("repo", ""), dockerhub_username
                )
                pull_result = await self.sandy.execute_command(
                    sandbox_id,
                    f"docker pull {image_uri}",
                    timeout_ms=900000,
                )
                if pull_result.get("exit_code") != 0:
                    return ItemResult(
                        item_id=item_id,
                        error=pull_result.get("stderr") or "Failed to pull Docker image",
                        metadata={"instance_id": item.get("instance_id")},
                    )

                host_volume = await self._get_env(sandbox_id, "SANDY_HOST_VOLUME")
                host_volume = host_volume.strip()
                if not host_volume:
                    return ItemResult(item_id=item_id, error="Sandy host volume path unavailable")

                run_result = await self.sandy.execute_command(
                    sandbox_id,
                    f"docker run --rm -v {host_volume}:/workspace --entrypoint /bin/bash {image_uri} -c \"bash /workspace/entryscript.sh\"",
                    timeout_ms=900000,
                )

                output_raw = await self._read_file(sandbox_id, "/workspace/output.json")
                try:
                    output = json.loads(output_raw) if output_raw else None
                except json.JSONDecodeError:
                    output = None

                stdout_log = await self._read_file(sandbox_id, "/workspace/stdout.log")
                stderr_log = await self._read_file(sandbox_id, "/workspace/stderr.log")

                passed_tests = set()
                if output and isinstance(output.get("tests"), list):
                    passed_tests = {
                        test.get("name")
                        for test in output.get("tests", [])
                        if test.get("status") == "PASSED"
                    }
                try:
                    f2p = set(ast.literal_eval(item.get("fail_to_pass", "[]")))
                except Exception:
                    f2p = set()
                try:
                    p2p = set(ast.literal_eval(item.get("pass_to_pass", "[]")))
                except Exception:
                    p2p = set()
                is_correct = (f2p | p2p) <= passed_tests

                error = None
                if run_result.get("exit_code") != 0:
                    error = run_result.get("stderr") or run_result.get("error")

                return ItemResult(
                    item_id=item_id,
                    item_hash=self.compute_item_hash(item.get("instance_id")),
                    prompt=prompt,
                    response=response_text.strip(),
                    expected="[SWE-bench Pro tests passed]",
                    is_correct=is_correct,
                    score=1.0 if is_correct else 0.0,
                    latency_ms=latency_ms,
                    input_tokens=metadata.get("usage", {}).get("prompt_tokens"),
                    output_tokens=metadata.get("usage", {}).get("completion_tokens"),
                    judge_output={
                        "output": output,
                        "stdout": stdout_log,
                        "stderr": stderr_log,
                        "exit_code": run_result.get("exit_code"),
                    },
                    error=error,
                    metadata={
                        "instance_id": item.get("instance_id"),
                        "repo": item.get("repo"),
                        "system_prompt": system_prompt,
                    },
                )
            finally:
                await self.sandy.terminate_sandbox(sandbox_id)

        except Exception as e:
            logger.error("SWE-Bench Pro evaluation failed", item_id=item_id, error=str(e))
            return ItemResult(
                item_id=item_id,
                prompt=prompt,
                response=locals().get("response_text", "") or "",
                error=str(e),
                metadata={"instance_id": item.get("instance_id")},
            )
