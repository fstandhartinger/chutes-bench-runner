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
import shutil
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


def _float_env(key: str, default: float) -> float:
    raw = os.getenv(key)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _int_from_env_file(path: str, key: str) -> Optional[int]:
    """Read an integer setting from a dotenv-style file."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                current_key, value = line.split("=", 1)
                if current_key.strip() != key:
                    continue
                value = value.strip().strip("\"'")
                try:
                    return int(value)
                except ValueError:
                    return None
    except FileNotFoundError:
        return None
    except Exception:
        return None
    return None


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


def resolve_compose_command(logger: logging.Logger) -> list[str]:
    """Resolve the compose CLI to use (docker compose preferred)."""
    configured = os.getenv("COMPOSE_COMMAND")
    if configured:
        return configured.split()

    docker_path = shutil.which("docker")
    if docker_path:
        try:
            result = subprocess.run(
                [docker_path, "compose", "version"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return [docker_path, "compose"]
        except subprocess.SubprocessError:
            pass

    compose_path = shutil.which("docker-compose")
    if compose_path:
        return [compose_path]

    logger.warning("Falling back to docker-compose default command.")
    return ["docker-compose"]


def get_memory_stats() -> tuple[Optional[float], Optional[float]]:
    """Return (memory_percent, available_gb) from /proc/meminfo."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            data = handle.read().splitlines()
        meminfo = {}
        for line in data:
            if ":" not in line:
                continue
            key, rest = line.split(":", 1)
            value = rest.strip().split()[0]
            meminfo[key] = int(value)
        total_kb = meminfo.get("MemTotal")
        avail_kb = meminfo.get("MemAvailable") or meminfo.get("MemFree")
        if not total_kb or not avail_kb:
            return None, None
        used_kb = total_kb - avail_kb
        percent = (used_kb / total_kb) * 100
        available_gb = avail_kb / (1024 * 1024)
        return percent, available_gb
    except Exception:
        return None, None


def get_disk_usage_percent(path: str) -> Optional[float]:
    """Return disk usage percent for the given path."""
    try:
        usage = shutil.disk_usage(path)
        if usage.total <= 0:
            return None
        return (usage.used / usage.total) * 100
    except Exception:
        return None


def get_cpu_load_percent() -> Optional[float]:
    """Return 1-minute load as a percentage of CPU core count."""
    try:
        load_1m = os.getloadavg()[0]
        cores = os.cpu_count() or 1
        if cores <= 0:
            cores = 1
        return (load_1m / cores) * 100
    except Exception:
        return None


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


def run_compose(
    cmd: list[str],
    logger: logging.Logger,
    dry_run: bool,
    timeout: int,
    cwd: Optional[str] = None,
) -> bool:
    logger.info("Compose command: %s", " ".join(cmd))
    if dry_run:
        return True
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired:
        logger.warning("Compose command timed out after %ss", timeout)
        return False
    if result.stdout:
        logger.info("Compose stdout: %s", result.stdout.strip())
    if result.stderr:
        logger.warning("Compose stderr: %s", result.stderr.strip())
    return result.returncode == 0


def get_worker_counts(
    base_project: str,
    extra_project: str,
    logger: logging.Logger,
    timeout: int,
) -> tuple[int, int]:
    """Return current running worker counts for base and extra projects."""
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.warning("Docker ps timed out after %ss", timeout)
        return 0, 0
    if result.returncode != 0:
        logger.warning("Docker ps failed: %s", (result.stderr or "").strip())
        return 0, 0

    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    # Docker Compose uses hyphens in container names (e.g., chutes-bench-runner-worker-1)
    base_prefix = f"{base_project}-worker-"
    extra_prefix = f"{extra_project}-worker-"
    base_count = sum(1 for name in names if name.startswith(base_prefix))
    extra_count = sum(1 for name in names if name.startswith(extra_prefix))

    # Debug logging to help diagnose count issues
    if base_count == 0 and extra_count == 0 and names:
        worker_names = [n for n in names if "worker" in n.lower()]
        if worker_names:
            logger.debug(
                "No workers matched prefixes %s/%s, but found worker-like names: %s",
                base_prefix, extra_prefix, worker_names[:5],
            )

    return base_count, extra_count


def scale_base(
    project: str,
    compose_file: str,
    env_file: str,
    workers: int,
    compose_command: list[str],
    logger: logging.Logger,
    dry_run: bool,
    timeout: int,
    cwd: Optional[str] = None,
) -> bool:
    cmd = [
        *compose_command,
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
    return run_compose(cmd, logger, dry_run, timeout, cwd=cwd)


def scale_extra(
    project: str,
    compose_file: str,
    env_file: str,
    workers: int,
    compose_command: list[str],
    logger: logging.Logger,
    dry_run: bool,
    timeout: int,
    cwd: Optional[str] = None,
) -> bool:
    if workers <= 0:
        cmd = [
            *compose_command,
            "-p",
            project,
            "-f",
            compose_file,
            "--env-file",
            env_file,
            "down",
            "--remove-orphans",
        ]
        return run_compose(cmd, logger, dry_run, timeout, cwd=cwd)

    cmd = [
        *compose_command,
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
    return run_compose(cmd, logger, dry_run, timeout, cwd=cwd)


def _read_text(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _write_text(path: str, content: str) -> bool:
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(content)
        return True
    except Exception:
        return False


def get_git_head(repo_path: str, logger: logging.Logger, timeout: int) -> Optional[str]:
    # Skip git operations entirely if the repo path is not a git repository
    git_dir = os.path.join(repo_path, ".git")
    if not os.path.exists(git_dir):
        return None

    try:
        result = subprocess.run(
            # systemd services sometimes run without HOME set, so global git config
            # (including safe.directory) may not be loaded. Pin safe.directory per-call.
            ["git", "-c", f"safe.directory={repo_path}", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        logger.warning("git rev-parse timed out after %ss", timeout)
        return None
    except Exception as exc:
        logger.warning("Failed to read git HEAD: %s", exc)
        return None

    if result.returncode != 0:
        logger.debug("git rev-parse failed: %s", (result.stderr or "").strip())
        return None

    head = (result.stdout or "").strip()
    return head or None


def maybe_rebuild_and_recreate_workers(
    repo_path: str,
    compose_file: str,
    env_file: str,
    base_project: str,
    extra_project: str,
    current_base: int,
    current_extra: int,
    compose_command: list[str],
    logger: logging.Logger,
    dry_run: bool,
    timeout: int,
) -> bool:
    """Ensure running workers match the current repo HEAD.

    The autoscaler historically ran `up ... --no-build` which means worker
    containers can keep running an old image indefinitely after `git pull`.
    This hook rebuilds the worker image once per git SHA and (optionally)
    recreates running containers so they pick up the rebuilt image.
    """
    if _int_env("AUTO_BUILD_ON_GIT_CHANGE", 1) != 1:
        return False

    head = get_git_head(repo_path, logger, timeout=timeout)
    if not head:
        return False

    state_file = _str_env("AUTO_BUILD_STATE_FILE", os.path.join(repo_path, ".autoscaler_last_built_sha"))
    last_built = (_read_text(state_file) or "").strip()
    if last_built == head:
        return False

    logger.info("Git HEAD update detected (last_built=%s head=%s); building worker image.", last_built or "-", head)
    build_cmd = [
        *compose_command,
        "-f",
        compose_file,
        "--env-file",
        env_file,
        "build",
        "worker",
    ]
    if not run_compose(build_cmd, logger, dry_run, timeout, cwd=repo_path):
        logger.warning("Worker image build failed; keeping existing workers.")
        return False

    if not _write_text(state_file, f"{head}\n"):
        logger.warning("Failed to write build state file %s; build may repeat on restart.", state_file)

    if _int_env("AUTO_RECREATE_ON_GIT_CHANGE", 1) != 1:
        return True

    # Recreate existing containers so they pick up the rebuilt image. We keep
    # the current scale (autoscaler may adjust it later in the same tick).
    if current_base > 0:
        recreate_base_cmd = [
            *compose_command,
            "-p",
            base_project,
            "-f",
            compose_file,
            "--env-file",
            env_file,
            "up",
            "-d",
            "--no-build",
            "--force-recreate",
            "--no-deps",
            "--scale",
            f"worker={current_base}",
        ]
        run_compose(recreate_base_cmd, logger, dry_run, timeout, cwd=repo_path)

    if current_extra > 0:
        recreate_extra_cmd = [
            *compose_command,
            "-p",
            extra_project,
            "-f",
            compose_file,
            "--env-file",
            env_file,
            "up",
            "-d",
            "--no-build",
            "--force-recreate",
            "--no-deps",
            "--scale",
            f"worker={current_extra}",
        ]
        run_compose(recreate_extra_cmd, logger, dry_run, timeout, cwd=repo_path)

    return True


def compute_target_workers(
    running: int,
    queued: int,
    max_workers: int,
    min_workers: int,
    worker_max_concurrent: int,
) -> int:
    """Compute target worker count from observed backlog.

    We must consider both queued and running jobs because reducing compose scale
    stops worker containers, which can interrupt active runs and leave them stuck
    in "running" until stale requeue recovers them.
    """
    per_worker = max(worker_max_concurrent, 1)
    queued_backlog = max(queued, 0)
    running_backlog = max(running, 0)
    backlog = queued_backlog + running_backlog
    if backlog <= 0:
        return min_workers
    desired = math.ceil(backlog / per_worker)
    return max(min_workers, min(max_workers, desired))


def main() -> int:
    parser = argparse.ArgumentParser(description="Autoscale Sandy benchmark workers.")
    parser.add_argument("--once", action="store_true", help="Run one scaling iteration and exit.")
    args = parser.parse_args()

    backend_url = _str_env("BACKEND_URL", "https://chutes-bench-runner-api-v2.onrender.com").rstrip("/")
    compose_file = _str_env("COMPOSE_FILE", "/opt/chutes-bench-runner/docker-compose.worker.yml")
    env_file = _str_env("ENV_FILE", "/opt/chutes-bench-runner/.env.worker")
    repo_path = _str_env("REPO_PATH", os.path.dirname(compose_file) or "/opt/chutes-bench-runner")
    base_max = _int_env("BASE_MAX_WORKERS", 4)
    extra_max = _int_env("EXTRA_MAX_WORKERS", 2)
    max_workers = _int_env("MAX_WORKERS", base_max + extra_max)
    min_workers = _int_env("MIN_WORKERS", 1)
    worker_max_concurrent = _int_env("WORKER_MAX_CONCURRENT", -1)
    if worker_max_concurrent <= 0:
        worker_max_concurrent = _int_from_env_file(env_file, "WORKER_MAX_CONCURRENT") or 2
    poll_seconds = _int_env("SCALE_INTERVAL_SECONDS", 30)
    compose_timeout = _int_env("COMPOSE_TIMEOUT_SECONDS", 120)
    memory_high = _float_env("MEMORY_HIGH_WATERMARK", 85.0)
    memory_emergency = _float_env("MEMORY_EMERGENCY_WATERMARK", 92.0)
    memory_scale_down = _int_env("MEMORY_SCALE_DOWN_STEP", 2)
    disk_check_path = _str_env("DISK_CHECK_PATH", "/")
    disk_high = _float_env("DISK_HIGH_WATERMARK", 88.0)
    disk_emergency = _float_env("DISK_EMERGENCY_WATERMARK", 93.0)
    cpu_high = _float_env("CPU_HIGH_WATERMARK", 90.0)
    extra_project = _str_env("EXTRA_PROJECT", "chutes-bench-runner-extra")
    base_project = _str_env("BASE_PROJECT", "chutes-bench-runner")
    dry_run = _int_env("DRY_RUN", 0) == 1
    timeout = _int_env("API_TIMEOUT_SECONDS", 10)
    log_path = _str_env("LOG_PATH", "/var/log/chutes-bench-runner-autoscaler.log")
    log_level = _str_env("LOG_LEVEL", "INFO")

    logger = configure_logging(log_path, log_level)
    compose_command = resolve_compose_command(logger)
    logger.info("Using compose command: %s", " ".join(compose_command))
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
        # Detect code updates and make sure running worker containers actually pick them up.
        current_base, current_extra = get_worker_counts(
            base_project,
            extra_project,
            logger,
            timeout=compose_timeout,
        )
        maybe_rebuild_and_recreate_workers(
            repo_path=repo_path,
            compose_file=compose_file,
            env_file=env_file,
            base_project=base_project,
            extra_project=extra_project,
            current_base=current_base,
            current_extra=current_extra,
            compose_command=compose_command,
            logger=logger,
            dry_run=dry_run,
            timeout=compose_timeout,
        )

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
            base_workers = min(base_max, target)
            extra_workers = max(0, target - base_max)
            current_base, current_extra = get_worker_counts(base_project, extra_project, logger, timeout=compose_timeout)
            current_total = current_base + current_extra
            mem_pct, mem_avail = get_memory_stats()
            disk_pct = get_disk_usage_percent(disk_check_path)
            cpu_load_pct = get_cpu_load_percent()
            logger.info(
                "Resource guard memory=%.1f%% available_gb=%.1f disk=%.1f%% cpu_load=%.1f%% current_workers=%s",
                mem_pct if mem_pct is not None else -1,
                mem_avail if mem_avail is not None else -1,
                disk_pct if disk_pct is not None else -1,
                cpu_load_pct if cpu_load_pct is not None else -1,
                current_total,
            )

            emergency_reasons: list[str] = []
            if mem_pct is not None and mem_pct >= memory_emergency:
                emergency_reasons.append(f"memory {mem_pct:.1f}% >= {memory_emergency:.1f}%")
            if disk_pct is not None and disk_pct >= disk_emergency:
                emergency_reasons.append(f"disk {disk_pct:.1f}% >= {disk_emergency:.1f}%")

            if emergency_reasons:
                target = max(min_workers, current_total - max(memory_scale_down, 1))
                logger.warning(
                    "Emergency guard triggered (%s), reducing target to %s",
                    "; ".join(emergency_reasons),
                    target,
                )
            else:
                high_reasons: list[str] = []
                if mem_pct is not None and mem_pct >= memory_high:
                    high_reasons.append(f"memory {mem_pct:.1f}% >= {memory_high:.1f}%")
                if disk_pct is not None and disk_pct >= disk_high:
                    high_reasons.append(f"disk {disk_pct:.1f}% >= {disk_high:.1f}%")
                if cpu_load_pct is not None and cpu_load_pct >= cpu_high:
                    high_reasons.append(f"cpu_load {cpu_load_pct:.1f}% >= {cpu_high:.1f}%")

                if high_reasons and target > current_total:
                    logger.warning(
                        "Resource high (%s), freezing scale-up at %s",
                        "; ".join(high_reasons),
                        current_total,
                    )
                    target = current_total

            target = max(min_workers, min(max_workers, target))
            base_workers = min(base_max, target)
            extra_workers = max(0, target - base_max)
            logger.info(
                "Queue status running=%s queued=%s target_workers=%s",
                running,
                queued,
                target,
            )
            if target != last_target or current_base != base_workers or current_extra != extra_workers:
                ok_base = scale_base(
                    base_project,
                    compose_file,
                    env_file,
                    base_workers,
                    compose_command,
                    logger,
                    dry_run,
                    compose_timeout,
                    cwd=repo_path,
                )
                ok_extra = scale_extra(
                    extra_project,
                    compose_file,
                    env_file,
                    extra_workers,
                    compose_command,
                    logger,
                    dry_run,
                    compose_timeout,
                    cwd=repo_path,
                )
                if ok_base and ok_extra:
                    last_target = target
                    logger.info(
                        "Scaling applied base=%s extra=%s (current base=%s extra=%s)",
                        base_workers,
                        extra_workers,
                        current_base,
                        current_extra,
                    )
                else:
                    logger.warning("Scaling failed base_ok=%s extra_ok=%s", ok_base, ok_extra)

        if args.once:
            return 0
        time.sleep(poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
