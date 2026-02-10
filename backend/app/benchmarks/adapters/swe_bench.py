"""SWE-Bench Pro benchmark adapter."""
from __future__ import annotations

import asyncio
import ast
import json
import os
import time
from typing import Any, AsyncIterator, Optional

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.benchmarks.utils import download_http_file, load_dataset_with_retry
from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.sandy_service import SandyService

logger = get_logger(__name__)

SWE_BENCH_REPO = "https://raw.githubusercontent.com/scaleapi/SWE-bench_Pro-os/main"

_DOCKER_RUN_OUTPUT_PREVIEW_CHARS = 4000


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
        self._sandbox_id: Optional[str] = None

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
                        "dockerhub_tag": str(item.get("dockerhub_tag", "")),
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

    async def _get_host_volume(self, sandbox_id: str) -> Optional[str]:
        result = await self.sandy.execute_command(sandbox_id, "printenv SANDY_HOST_VOLUME")
        host_volume = (result.get("stdout") or "").strip()
        if host_volume:
            return host_volume
        settings = get_settings()
        fallback_root = (settings.sandy_volume_root or "").rstrip("/")
        if fallback_root:
            return f"{fallback_root}/{sandbox_id}"
        return None

    def _build_agent_env(self) -> dict[str, str]:
        """Build env vars for Sandy agent runners.

        Sandy's Codex runner expects a `CHUTES_API_KEY` and will route through the
        Chutes responses proxy by default. Do not override the API base URL here.
        """
        api_key = self.client.get_api_key() or get_settings().chutes_api_key
        return {"CHUTES_API_KEY": api_key}

    async def _ensure_sandbox(self) -> Optional[str]:
        if self._sandbox_id:
            exists = await self.sandy.sandbox_exists(self._sandbox_id)
            if exists:
                return self._sandbox_id
            self._sandbox_id = None

        self._sandbox_id = await self.sandy.create_sandbox(enable_docker_socket=True)
        return self._sandbox_id

    async def cleanup(self) -> None:
        sandbox_id = self._sandbox_id
        self._sandbox_id = None
        if sandbox_id:
            await self.sandy.terminate_sandbox(sandbox_id)

    async def evaluate_item(self, item_id: str) -> ItemResult:
        """Evaluate a single SWE-Bench Pro item."""
        if not self._items:
            await self.preload()

        item = next((i for i in self._items if i["id"] == item_id), None)
        if not item:
            return ItemResult(item_id=item_id, error=f"Item {item_id} not found")

        repo = item.get("repo", "")
        base_commit = item.get("base_commit", "")
        instance_id = item.get("instance_id") or ""

        prompt = (
            "You are a software engineer fixing a bug in a GitHub repository. "
            "The repository is cloned at /workspace/repo and checked out to the base commit. "
            "Make the required code changes in the repo. Do not generate a patch yourself; "
            "the harness will create the patch after you finish.\n\n"
            f"Repository: {repo or 'unknown'}\n"
            f"Base Commit: {base_commit}\n"
            f"Issue Description:\n{item.get('problem_statement')}\n"
        )

        if not repo or not base_commit:
            return ItemResult(
                item_id=item_id,
                error="Missing repo or base commit",
                metadata={"instance_id": instance_id, "repo": repo},
            )

        start_time = time.time()
        agent_output = ""
        agent_summary: dict[str, Any] = {}
        agent_name = "codex"

        try:
            # SWE-Bench requires Docker socket access for running docker pull/run.
            # Keep a single sandbox for the full benchmark run to reduce upstream churn.
            sandbox_id = await self._ensure_sandbox()
            if not sandbox_id:
                sandbox_error = self.sandy.last_error or "Could not create sandbox"
                return ItemResult(
                    item_id=item_id,
                    error=sandbox_error,
                    metadata={"instance_id": instance_id, "repo": repo},
                )

            await self.sandy.execute_command(
                sandbox_id,
                "rm -rf /workspace/repo && rm -f /workspace/patch.diff /workspace/stdout.log /workspace/stderr.log /workspace/output.json || true",
            )

            # Ensure git is available
            git_check = await self.sandy.execute_command(sandbox_id, "git --version")
            if git_check.get("exit_code") != 0:
                await self.sandy.execute_command(
                    sandbox_id,
                    "apt-get update && apt-get install -y git",
                    timeout_ms=600000,
                )

            clone_cmd = f"rm -rf /workspace/repo && git clone https://github.com/{repo}.git /workspace/repo"
            clone_result: Optional[dict[str, Any]] = None
            for attempt in range(1, 4):
                clone_result = await self.sandy.execute_command(
                    sandbox_id,
                    clone_cmd,
                    timeout_ms=900000,
                )
                if clone_result.get("exit_code") == 0:
                    break
                if attempt < 3:
                    await asyncio.sleep(min(5 * attempt, 15))
            if not clone_result or clone_result.get("exit_code") != 0:
                return ItemResult(
                    item_id=item_id,
                    error=(clone_result or {}).get("stderr") or "Failed to clone repo",
                    metadata={"instance_id": instance_id, "repo": repo},
                )

            checkout_result = await self.sandy.execute_command(
                sandbox_id,
                f"cd /workspace/repo && git checkout {base_commit}",
            )
            if checkout_result.get("exit_code") != 0:
                return ItemResult(
                    item_id=item_id,
                    error=checkout_result.get("stderr") or "Failed to checkout base commit",
                    metadata={"instance_id": instance_id, "repo": repo},
                )

            agent_env_vars = self._build_agent_env()
            agent_result = await self.sandy.run_agent(
                sandbox_id,
                agent=agent_name,
                model=self.model_slug,
                prompt=prompt + "\nWork inside /workspace/repo.",
                max_duration=1800,
                raw_prompt=True,
                env_vars=agent_env_vars,
            )
            agent_summary = agent_result.get("summary") or {}
            agent_events = agent_result.get("events") or []
            agent_output = next(
                (event.get("text") for event in reversed(agent_events) if event.get("type") == "output"),
                "",
            )

            # Include untracked/new files (git diff alone will not).
            await self.sandy.execute_command(sandbox_id, "cd /workspace/repo && git add -A")
            patch_result = await self.sandy.execute_command(
                sandbox_id,
                f"cd /workspace/repo && git diff --cached --binary {base_commit} > /workspace/patch.diff && git reset",
            )
            if patch_result.get("exit_code") != 0:
                return ItemResult(
                    item_id=item_id,
                    error=patch_result.get("stderr") or "Failed to generate patch",
                    metadata={"instance_id": instance_id, "repo": repo},
                )

            patch = await self._read_file(sandbox_id, "/workspace/patch.diff")
            item_hash = self.compute_item_hash(instance_id)

            if not patch.strip():
                latency_ms = int((time.time() - start_time) * 1000)
                return ItemResult(
                    item_id=item_id,
                    item_hash=item_hash,
                    prompt=prompt,
                    response=agent_output,
                    is_correct=False,
                    score=0.0,
                    latency_ms=latency_ms,
                    judge_output={
                        "agent_summary": agent_summary,
                        "failure_reason": "Agent did not produce a patch",
                    },
                    metadata={
                        "instance_id": instance_id,
                        "repo": repo,
                        "agent": agent_name,
                        "agent_summary": agent_summary,
                    },
                )

            entryscript = self._create_entryscript(item)
            await self.sandy.write_file(sandbox_id, "patch.diff", patch)
            await self.sandy.write_file(
                sandbox_id,
                "run_script.sh",
                self._download_run_script(item["instance_id"], "run_script.sh"),
            )
            await self.sandy.write_file(
                sandbox_id,
                "parser.py",
                self._download_run_script(item["instance_id"], "parser.py"),
            )
            await self.sandy.write_file(sandbox_id, "entryscript.sh", entryscript)

            dockerhub_username = "jefzda"
            dockerhub_tag = (item.get("dockerhub_tag") or "").strip()
            if dockerhub_tag:
                image_uri = f"{dockerhub_username}/sweap-images:{dockerhub_tag}"
            else:
                image_uri = self._get_dockerhub_image_uri(
                    item["instance_id"], item.get("repo", ""), dockerhub_username
                )
            pull_result: Optional[dict[str, Any]] = None
            for attempt in range(1, 4):
                pull_result = await self.sandy.execute_command(
                    sandbox_id,
                    f"docker pull {image_uri}",
                    timeout_ms=900000,
                )
                if pull_result.get("exit_code") == 0:
                    break
                if attempt < 3:
                    await asyncio.sleep(min(10 * attempt, 30))
            if not pull_result or pull_result.get("exit_code") != 0:
                return ItemResult(
                    item_id=item_id,
                    error=(pull_result or {}).get("stderr") or "Failed to pull Docker image",
                    metadata={"instance_id": instance_id},
                )

            host_volume = await self._get_host_volume(sandbox_id)
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

            docker_stdout = run_result.get("stdout") or ""
            docker_stderr = run_result.get("stderr") or ""
            docker_combined_lower = (docker_stdout + "\n" + docker_stderr).lower()

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
            failure_reason = None
            if not output:
                # If the harness doesn't produce parseable output, we need to decide whether this is
                # infra (docker/sandbox) or a model failure (e.g., patch did not apply cleanly).
                if any(
                    needle in docker_combined_lower
                    for needle in (
                        "patch does not apply",
                        "error: patch failed",
                        "patch failed",
                    )
                ):
                    # Model produced an invalid/unapplicable patch.
                    failure_reason = "Patch failed to apply in harness"
                    error = None
                    is_correct = False
                else:
                    # Treat missing harness output as infra by default so we don't silently hide
                    # issues like docker daemon failures or missing evaluation scripts.
                    failure_reason = "Harness did not produce parseable output"
                    error = run_result.get("stderr") or run_result.get("error") or "Harness execution failed"
                    if isinstance(error, str) and len(error) > 800:
                        error = error[:800].rstrip() + "…"

            latency_ms = int((time.time() - start_time) * 1000)
            return ItemResult(
                item_id=item_id,
                item_hash=item_hash,
                prompt=prompt,
                response=agent_output,
                expected="[SWE-bench Pro tests passed]",
                is_correct=is_correct,
                score=1.0 if is_correct else 0.0,
                latency_ms=latency_ms,
                judge_output={
                    "output": output,
                    "stdout": stdout_log,
                    "stderr": stderr_log,
                    "exit_code": run_result.get("exit_code"),
                    "docker_stdout_preview": docker_stdout[:_DOCKER_RUN_OUTPUT_PREVIEW_CHARS],
                    "docker_stderr_preview": docker_stderr[:_DOCKER_RUN_OUTPUT_PREVIEW_CHARS],
                    "agent_summary": agent_summary,
                    "failure_reason": failure_reason,
                },
                error=error,
                metadata={
                    "instance_id": instance_id,
                    "repo": repo,
                    "agent": agent_name,
                    "agent_summary": agent_summary,
                },
            )

        except Exception as e:
            logger.error("SWE-Bench Pro evaluation failed", item_id=item_id, error=str(e))
            return ItemResult(
                item_id=item_id,
                prompt=prompt,
                response=agent_output or "",
                error=str(e),
                metadata={"instance_id": instance_id, "repo": repo},
            )
