#!/usr/bin/env python3
"""
Autoscale benchmark workers running on the Sandy server.

Scales base workers (up to BASE_MAX_WORKERS) using the default compose project
and optionally scales extra workers (up to EXTRA_MAX_WORKERS) using a separate
compose project to avoid container name conflicts beyond worker-4.
"""
from __future__ import annotations

import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
import subprocess
import time
from typing import Any, Optional
from urllib.error import URLError, HTTPError
from urllib.request import Request, urlopen


def _int_env(key: str, default: int) -> int:
    raw = os.getenv(key)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _str_env(key: str, default: str) -> str:
    value = os.getenv(key)
    return value if value else default


def configure_logging(log_path: str, level: str) -> logging.Logger:
    logger = logging.getLogger("worker_autoscaler")
    logger.setLevel(level.upper())
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_handler = RotatingFileHandler(log_path, maxBytes=5_000_000, backupCount=3)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def fetch_runs_total(base_url: str, status: str, timeout: int, logger: logging.Logger) -> Optional[int]:
    url = f"{base_url}/api/runs?status={status}&limit=200"
    try:
        req = Request(url, headers={"User-Agent": "worker-autoscaler/1.0"})
        with urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        logger.warning("Failed to fetch %s runs: %s", status, exc)
        return None
    except Exception as exc:
        logger.warning("Unexpected error fetching %s runs: %s", status, exc)
        return None

    if isinstance(payload, dict):
        if isinstance(payload.get("total"), int):
            return int(payload["total"])
        runs = payload.get("runs") or []
        if isinstance(runs, list):
            return len(runs)
    return None


def run_compose(cmd: list[str], logger: logging.Logger, dry_run: bool, timeout: int) -> bool:
    logger.info("Compose command: %s", " ".join(cmd))
    if dry_run:
        return True
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning("Compose command timed out after %ss", timeout)
        return False
    if result.stdout:
        logger.info("Compose stdout: %s", result.stdout.strip())
    if result.stderr:
        logger.warning("Compose stderr: %s", result.stderr.strip())
    return result.returncode == 0


def scale_base(
    project: str,
    compose_file: str,
    env_file: str,
    workers: int,
    logger: logging.Logger,
    dry_run: bool,
    timeout: int,
) -> bool:
    cmd = [
        "docker-compose",
        "-p",
        project,
        "-f",
        compose_file,
        "--env-file",
        env_file,
        "up",
        "-d",
        "--no-build",
        "--scale",
        f"worker={workers}",
    ]
    return run_compose(cmd, logger, dry_run, timeout)


def scale_extra(
    project: str,
    compose_file: str,
    env_file: str,
    workers: int,
    logger: logging.Logger,
    dry_run: bool,
    timeout: int,
) -> bool:
    if workers <= 0:
        cmd = [
            "docker-compose",
            "-p",
            project,
            "-f",
            compose_file,
            "--env-file",
            env_file,
            "down",
            "--remove-orphans",
        ]
        return run_compose(cmd, logger, dry_run, timeout)

    cmd = [
        "docker-compose",
        "-p",
        project,
        "-f",
        compose_file,
        "--env-file",
        env_file,
        "up",
        "-d",
        "--no-build",
        "--scale",
        f"worker={workers}",
    ]
    return run_compose(cmd, logger, dry_run, timeout)


def compute_target_workers(
    running: int,
    queued: int,
    max_workers: int,
    min_workers: int,
    worker_max_concurrent: int,
) -> int:
    backlog = running + queued
    if backlog <= 0:
        return min_workers
    per_worker = max(worker_max_concurrent, 1)
    desired = math.ceil(backlog / per_worker)
    return max(min_workers, min(max_workers, desired))


def main() -> int:
    parser = argparse.ArgumentParser(description="Autoscale Sandy benchmark workers.")
    parser.add_argument("--once", action="store_true", help="Run one scaling iteration and exit.")
    args = parser.parse_args()

    backend_url = _str_env("BACKEND_URL", "https://chutes-bench-runner-api-v2.onrender.com").rstrip("/")
    compose_file = _str_env("COMPOSE_FILE", "/opt/chutes-bench-runner/docker-compose.worker.yml")
    env_file = _str_env("ENV_FILE", "/opt/chutes-bench-runner/.env.worker")
    base_max = _int_env("BASE_MAX_WORKERS", 4)
    extra_max = _int_env("EXTRA_MAX_WORKERS", 2)
    max_workers = _int_env("MAX_WORKERS", base_max + extra_max)
    min_workers = _int_env("MIN_WORKERS", 1)
    worker_max_concurrent = _int_env("WORKER_MAX_CONCURRENT", 2)
    poll_seconds = _int_env("SCALE_INTERVAL_SECONDS", 30)
    compose_timeout = _int_env("COMPOSE_TIMEOUT_SECONDS", 120)
    extra_project = _str_env("EXTRA_PROJECT", "chutes-bench-runner-extra")
    base_project = _str_env("BASE_PROJECT", "chutes-bench-runner")
    dry_run = _int_env("DRY_RUN", 0) == 1
    timeout = _int_env("API_TIMEOUT_SECONDS", 10)
    log_path = _str_env("LOG_PATH", "/var/log/chutes-bench-runner-autoscaler.log")
    log_level = _str_env("LOG_LEVEL", "INFO")

    logger = configure_logging(log_path, log_level)
    logger.info(
        "Autoscaler started backend=%s min=%s max=%s base_max=%s extra_max=%s",
        backend_url,
        min_workers,
        max_workers,
        base_max,
        extra_max,
    )

    last_target: Optional[int] = None

    while True:
        running = fetch_runs_total(backend_url, "running", timeout, logger)
        queued = fetch_runs_total(backend_url, "queued", timeout, logger)
        if running is None or queued is None:
            logger.warning("Skipping scale decision (running=%s queued=%s)", running, queued)
        else:
            target = compute_target_workers(
                running=running,
                queued=queued,
                max_workers=max_workers,
                min_workers=min_workers,
                worker_max_concurrent=worker_max_concurrent,
            )
            logger.info(
                "Queue status running=%s queued=%s target_workers=%s",
                running,
                queued,
                target,
            )
            if target != last_target:
                base_workers = min(base_max, target)
                extra_workers = max(0, target - base_max)
                ok_base = scale_base(
                    base_project,
                    compose_file,
                    env_file,
                    base_workers,
                    logger,
                    dry_run,
                    compose_timeout,
                )
                ok_extra = scale_extra(
                    extra_project,
                    compose_file,
                    env_file,
                    extra_workers,
                    logger,
                    dry_run,
                    compose_timeout,
                )
                if ok_base and ok_extra:
                    last_target = target
                    logger.info("Scaling applied base=%s extra=%s", base_workers, extra_workers)
                else:
                    logger.warning("Scaling failed base_ok=%s extra_ok=%s", ok_base, ok_extra)

        if args.once:
            return 0
        time.sleep(poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
