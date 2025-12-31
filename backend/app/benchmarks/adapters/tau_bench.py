"""τ²-Bench Telecom benchmark adapter."""
from __future__ import annotations

import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, AsyncIterator, Optional

from app.benchmarks.base import BenchmarkAdapter, ItemResult
from app.benchmarks.registry import register_adapter
from app.benchmarks.utils import download_http_file
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

TAU2_REPO_ZIP = "https://github.com/sierra-research/tau2-bench/archive/refs/heads/main.zip"


@register_adapter("tau_bench_telecom")
class TauBenchTelecomAdapter(BenchmarkAdapter):
    """
    τ²-Bench Telecom adapter.

    Uses the official tau2 simulation framework for the telecom domain.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._items: list[dict[str, Any]] = []
        self._tau2_loaded = False
        self._tau2_repo: Optional[Path] = None

    def get_name(self) -> str:
        return "tau_bench_telecom"

    def get_display_name(self) -> str:
        return "τ²-Bench Telecom"

    def requires_setup(self) -> bool:
        return False

    def get_setup_notes(self) -> Optional[str]:
        return None

    def supports_subset(self) -> bool:
        return True

    async def get_total_items(self) -> int:
        if not self._items:
            await self.preload()
        return len(self._items)

    def _ensure_tau2_repo(self) -> None:
        if self._tau2_loaded:
            return
        settings = get_settings()
        repo_zip = download_http_file(TAU2_REPO_ZIP, cache_subdir="tau2", filename="tau2-bench.zip")
        repo_dir = repo_zip.with_suffix("")
        if not repo_dir.exists():
            with zipfile.ZipFile(repo_zip, "r") as zf:
                zf.extractall(repo_zip.parent)
        # The zip extracts to tau2-bench-main by default.
        extracted_dir = repo_zip.parent / "tau2-bench-main"
        if not extracted_dir.exists():
            extracted_dir = repo_dir
        self._tau2_repo = extracted_dir
        tau2_src = extracted_dir / "src"
        if str(tau2_src) not in sys.path:
            sys.path.insert(0, str(tau2_src))
        os.environ.setdefault("TAU2_DATA_DIR", str(extracted_dir / "data"))
        os.environ.setdefault("TAU2_LOG_LEVEL", "ERROR")
        os.environ.setdefault("OPENAI_API_BASE", settings.chutes_api_base_url)
        self._tau2_loaded = True

    async def preload(self) -> None:
        if self._items:
            return

        try:
            self._ensure_tau2_repo()
            from tau2.run import get_tasks

            tasks = get_tasks(task_set_name="telecom", task_split_name="base")
            self._items = []
            for task in tasks:
                self._items.append(
                    {
                        "id": str(task.id),
                        "task": task,
                    }
                )
            logger.info("Loaded %s τ²-Bench telecom tasks", len(self._items))
        except Exception as e:
            logger.error("Failed to load τ²-Bench telecom", error=str(e))
            self._items = []
            raise

    async def enumerate_items(self) -> AsyncIterator[str]:
        if not self._items:
            await self.preload()
        for item in self._items:
            yield item["id"]

    async def evaluate_item(self, item_id: str) -> ItemResult:
        """Evaluate a single τ²-Bench telecom task."""
        if not self._items:
            await self.preload()

        item = next((i for i in self._items if i["id"] == item_id), None)
        if not item:
            return ItemResult(item_id=item_id, error=f"Item {item_id} not found")

        self._ensure_tau2_repo()
        settings = get_settings()
        token = self.client.user_access_token or self.client.api_key
        if not token:
            return ItemResult(item_id=item_id, error="Missing Chutes API key for τ²-Bench evaluation")

        os.environ["OPENAI_API_KEY"] = token
        os.environ["OPENAI_API_BASE"] = settings.chutes_api_base_url

        agent_model = f"openai/{self.model_slug}"
        user_model = f"openai/{settings.tau2_user_model}" if settings.tau2_user_model else agent_model
        llm_args = {
            "temperature": 0.0,
            "api_key": token,
            "api_base": settings.chutes_api_base_url,
        }

        try:
            from tau2.run import run_tasks

            start_time = time.time()
            results = run_tasks(
                domain="telecom",
                tasks=[item["task"]],
                agent="llm_agent",
                user="user_simulator",
                llm_agent=agent_model,
                llm_args_agent=llm_args,
                llm_user=user_model,
                llm_args_user=llm_args,
                num_trials=1,
                max_steps=200,
                max_errors=10,
                save_to=None,
                console_display=False,
                max_concurrency=1,
                seed=300,
                log_level="ERROR",
            )
            latency_ms = int((time.time() - start_time) * 1000)

            simulation = results.simulations[0] if results.simulations else None
            reward_info = simulation.reward_info if simulation else None
            reward = reward_info.reward if reward_info else 0.0
            is_correct = reward >= 1.0 - 1e-6

            return ItemResult(
                item_id=item_id,
                item_hash=self.compute_item_hash(item_id),
                prompt="τ²-Bench telecom simulation",
                response="[Simulation completed]",
                expected="reward=1.0",
                is_correct=is_correct,
                score=reward,
                latency_ms=latency_ms,
                judge_output={
                    "reward_info": reward_info.model_dump() if reward_info else None,
                    "termination_reason": simulation.termination_reason.value if simulation else None,
                },
                metadata={
                    "agent_model": agent_model,
                    "user_model": user_model,
                },
            )
        except Exception as e:
            logger.error("τ²-Bench evaluation failed", item_id=item_id, error=str(e))
            return ItemResult(
                item_id=item_id,
                error=str(e),
                metadata={"task_id": item_id},
            )
