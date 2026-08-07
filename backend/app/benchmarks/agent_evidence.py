"""Durable, verified retention for Sandy CLI-agent evidence.

Sandy exec responses are size-capped.  A single base64 response can therefore
decode successfully while containing only a prefix of the requested file.  The
transfer below deliberately uses small chunks, checks every decoded chunk's
length, and verifies a SHA-256 that was computed inside the sandbox before the
archive is made visible as retained evidence.
"""

from __future__ import annotations

import ast
import asyncio
import base64
import binascii
import fcntl
import hashlib
import json
import os
import re
import shutil
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

EVIDENCE_SCHEMA_VERSION = 1
RLM_NATIVE_REPOSITORY_HELPERS = frozenset(
    {"read_file", "write_file", "apply_patch", "list_dir", "grep"}
)
MAX_RETAINED_NATIVE_HELPER_PATHS = 100
MAX_RETAINED_NATIVE_HELPER_PATH_LENGTH = 1_000

_PREPARE_EVIDENCE_SCRIPT = r"""
import hashlib
import io
import json
import os
import tarfile
import time

archive_path = "/tmp/chutes-bench-agent-evidence.tar.gz"
sources = [
    ("/root/.chutescoder/sessions", "rollouts/chutescoder/sessions"),
    ("/root/.codex/sessions", "rollouts/codex/sessions"),
    ("/workspace/.chutes/prime-agent-sessions", "rollouts/prime-agent/sessions"),
    ("/root/.chutescoder/config.toml", "config/chutescoder-config.toml"),
    ("/root/.chutescoder/model_catalog.json", "config/chutescoder-model-catalog.json"),
    ("/root/.codex/config.toml", "config/codex-config.toml"),
    ("/root/.codex/model_catalog.json", "config/codex-model-catalog.json"),
    ("/workspace/.chutes/prime-agent/models.json", "config/prime-agent-models.json"),
    ("/workspace/.chutes/agent_output.log", "agent/combined-stdout-stderr.log"),
    ("/workspace/.chutes/agent.done", "agent/exit-code.txt"),
    ("/workspace/.chutes/agent_launch_debug.sh", "agent/launch-debug.sh"),
]
present = [(source, target) for source, target in sources if os.path.exists(source)]
if not present:
    print(json.dumps({"error": "no rollout or agent output files found"}))
    raise SystemExit(3)

manifest = {
    "schema_version": 1,
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "sources": [target for _, target in present],
    "agent_output": "agent/combined-stdout-stderr.log",
}
manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()

with tarfile.open(archive_path, mode="w:gz", compresslevel=6) as bundle:
    info = tarfile.TarInfo("MANIFEST.json")
    info.size = len(manifest_bytes)
    info.mtime = int(time.time())
    bundle.addfile(info, io.BytesIO(manifest_bytes))
    for source, target in present:
        bundle.add(source, arcname=target, recursive=True)

digest = hashlib.sha256()
with open(archive_path, "rb") as evidence:
    for chunk in iter(lambda: evidence.read(1024 * 1024), b""):
        digest.update(chunk)
print(json.dumps({
    "path": archive_path,
    "size_bytes": os.path.getsize(archive_path),
    "sha256": digest.hexdigest(),
    "sources": manifest["sources"],
}))
"""


def _safe_component(value: str, *, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-.")
    if not cleaned:
        cleaned = fallback
    suffix = hashlib.sha256(value.encode()).hexdigest()[:10]
    return f"{cleaned[:80]}-{suffix}"


def _artifact_path(
    root: Path,
    *,
    run_id: str | None,
    benchmark_name: str,
    item_id: str,
    sandbox_id: str,
) -> Path:
    run_component = _safe_component(run_id or "unscoped", fallback="unscoped")
    benchmark_component = _safe_component(benchmark_name, fallback="benchmark")
    item_component = _safe_component(item_id, fallback="item")
    sandbox_component = _safe_component(sandbox_id, fallback="sandbox")
    return (
        root / run_component / benchmark_component / item_component / f"{sandbox_component}.tar.gz"
    )


def _all_storage_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != ".retention.lock" and not path.name.endswith(".jsonl")
    ]


def _append_prune_record(root: Path, path: Path, size: int, reason: str) -> None:
    record = {
        "pruned_at": datetime.now(UTC).isoformat(),
        "path": str(path),
        "size_bytes": size,
        "reason": reason,
    }
    with (root / "pruned.jsonl").open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record, sort_keys=True) + "\n")


def _unlink_for_pruning(root: Path, path: Path, reason: str) -> int:
    try:
        size = path.stat().st_size
        path.unlink()
        _append_prune_record(root, path, size, reason)
        return size
    except FileNotFoundError:
        return 0
    except OSError as exc:
        logger.warning(
            "Could not prune agent evidence file",
            path=str(path),
            reason=reason,
            error=str(exc),
        )
        return 0


def _reserve_local_file(
    final_path: Path,
    size_bytes: int,
    *,
    max_total_bytes: int,
    max_age_days: int,
    min_free_bytes: int,
) -> tuple[Path, BinaryIO]:
    """Prune under a cross-process lock, then reserve all incoming blocks."""
    root = final_path.parents[3]
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".retention.lock"
    lock_file = lock_path.open("a+b")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        now = time.time()
        cutoff = now - max_age_days * 86400
        files = _all_storage_files(root)

        for path in files:
            try:
                if path.stat().st_mtime < cutoff:
                    _unlink_for_pruning(root, path, "age_limit")
            except FileNotFoundError:
                continue

        files = _all_storage_files(root)
        total_bytes = sum(path.stat().st_size for path in files)
        free_bytes = shutil.disk_usage(root).free
        # A different worker container may currently be filling a reservation
        # in the same host directory. Count it against the cap, but never prune
        # a fresh partial out from under that transfer. Stale partials become
        # eligible after an hour (far beyond the 180-second transfer deadline).
        oldest_first = sorted(
            (
                path
                for path in files
                if not (path.name.endswith(".partial") and path.stat().st_mtime >= now - 3600)
            ),
            key=lambda path: path.stat().st_mtime,
        )
        while oldest_first and (
            total_bytes + size_bytes > max_total_bytes or free_bytes < min_free_bytes + size_bytes
        ):
            candidate = oldest_first.pop(0)
            removed = _unlink_for_pruning(root, candidate, "capacity_limit")
            total_bytes = max(0, total_bytes - removed)
            free_bytes = shutil.disk_usage(root).free

        if total_bytes + size_bytes > max_total_bytes:
            raise OSError(
                f"evidence store cap would be exceeded: "
                f"{total_bytes} + {size_bytes} > {max_total_bytes} bytes"
            )
        if free_bytes < min_free_bytes + size_bytes:
            raise OSError(
                f"insufficient free disk for evidence: free={free_bytes}, "
                f"required={min_free_bytes + size_bytes} bytes"
            )

        final_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path = final_path.with_suffix(final_path.suffix + ".partial")
        partial_file = partial_path.open("w+b")
        try:
            os.posix_fallocate(partial_file.fileno(), 0, size_bytes)
        except Exception:
            partial_file.close()
            partial_path.unlink(missing_ok=True)
            raise
        return partial_path, partial_file
    finally:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        lock_file.close()


async def _prepare_bundle(sandy: Any, sandbox_id: str) -> dict[str, Any]:
    encoded = base64.b64encode(_PREPARE_EVIDENCE_SCRIPT.encode()).decode("ascii")
    result = await sandy.execute_command(
        sandbox_id,
        f"echo {encoded} | base64 -d > /tmp/_prepare_agent_evidence.py "
        "&& nice -n 19 python3 /tmp/_prepare_agent_evidence.py",
        timeout_ms=120_000,
    )
    raw = ((result or {}).get("stdout") or "").strip()
    if not raw:
        raise RuntimeError(
            (result or {}).get("stderr")
            or (result or {}).get("error")
            or "evidence preparation produced no output"
        )
    try:
        metadata = json.loads(raw.splitlines()[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid evidence preparation response: {raw[:500]}") from exc
    if metadata.get("error"):
        raise RuntimeError(str(metadata["error"]))
    if (result or {}).get("exit_code") != 0:
        raise RuntimeError(f"evidence preparation exited {(result or {}).get('exit_code')}")
    return metadata


async def _read_chunk(
    sandy: Any,
    sandbox_id: str,
    sandbox_path: str,
    offset: int,
    count: int,
) -> tuple[int, bytes]:
    result = await sandy.execute_command(
        sandbox_id,
        f"dd if={sandbox_path} bs=1 skip={offset} count={count} 2>/dev/null | base64 -w0",
        timeout_ms=60_000,
    )
    if (result or {}).get("exit_code") != 0:
        raise RuntimeError(
            (result or {}).get("stderr")
            or (result or {}).get("error")
            or f"evidence chunk at offset {offset} failed"
        )
    encoded = ((result or {}).get("stdout") or "").strip()
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RuntimeError(f"invalid base64 evidence chunk at offset {offset}") from exc
    if len(decoded) != count:
        raise RuntimeError(
            f"truncated evidence chunk at offset {offset}: expected {count}, got {len(decoded)}"
        )
    return offset, decoded


async def _transfer_chunks(
    sandy: Any,
    sandbox_id: str,
    sandbox_path: str,
    destination: BinaryIO,
    size_bytes: int,
    *,
    chunk_bytes: int,
    concurrency: int,
) -> None:
    offsets = list(range(0, size_bytes, chunk_bytes))
    for start in range(0, len(offsets), concurrency):
        batch = offsets[start : start + concurrency]
        chunks = await asyncio.gather(
            *(
                _read_chunk(
                    sandy,
                    sandbox_id,
                    sandbox_path,
                    offset,
                    min(chunk_bytes, size_bytes - offset),
                )
                for offset in batch
            )
        )
        for offset, chunk in sorted(chunks):
            destination.seek(offset)
            destination.write(chunk)
    destination.flush()
    os.fsync(destination.fileno())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as retained:
        for chunk in iter(lambda: retained.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rollout_members(bundle: tarfile.TarFile) -> list[tarfile.TarInfo]:
    return sorted(
        (
            member
            for member in bundle.getmembers()
            if member.isfile()
            and "/sessions/" in f"/{member.name}"
            and member.name.endswith(".jsonl")
            and (
                Path(member.name).name.startswith("rollout-")
                or member.name.startswith("rollouts/prime-agent/sessions/")
            )
        ),
        key=lambda member: member.name,
    )


def read_rollout_metrics(path: Path) -> dict[str, Any]:
    """Count structured rollout events without matching event text."""
    rollout_line_count = 0
    compaction_events = 0
    compaction_events_by_type: dict[str, int] = {}
    tool_calls_by_name: dict[str, int] = {}
    rlm_native_helper_calls_by_name: dict[str, int] = {}
    rlm_native_helper_paths: set[str] = set()
    rlm_python_cells_with_subprocess = 0
    rlm_python_cells_with_docker = 0
    malformed_lines = 0
    rollouts: list[str] = []

    with tarfile.open(path, mode="r:gz") as bundle:
        for member in _rollout_members(bundle):
            extracted = bundle.extractfile(member)
            if extracted is None:
                continue
            rollouts.append(member.name)
            for raw_line in extracted:
                rollout_line_count += 1
                try:
                    event = json.loads(raw_line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    malformed_lines += 1
                    continue
                if not isinstance(event, dict):
                    malformed_lines += 1
                    continue
                payload = event.get("payload")
                if not isinstance(payload, dict):
                    continue
                payload_type = payload.get("type")
                if not isinstance(payload_type, str):
                    continue
                if "compact" in payload_type:
                    compaction_events += 1
                    compaction_events_by_type[payload_type] = (
                        compaction_events_by_type.get(payload_type, 0) + 1
                    )
                tool_name = payload.get("name")
                if payload_type.endswith("_call") and isinstance(tool_name, str):
                    tool_calls_by_name[tool_name] = tool_calls_by_name.get(tool_name, 0) + 1
                if payload_type == "function_call" and tool_name == "python":
                    arguments = payload.get("arguments")
                    if isinstance(arguments, str):
                        try:
                            arguments = json.loads(arguments)
                        except json.JSONDecodeError:
                            arguments = None
                    code = arguments.get("code") if isinstance(arguments, dict) else None
                    if not isinstance(code, str):
                        continue
                    if re.search(r"\bsubprocess\b", code):
                        rlm_python_cells_with_subprocess += 1
                    if re.search(r"\bdocker(?:\s+exec|\b)", code, flags=re.IGNORECASE):
                        rlm_python_cells_with_docker += 1
                    try:
                        tree = ast.parse(code)
                    except SyntaxError:
                        continue
                    for node in ast.walk(tree):
                        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
                            continue
                        helper = node.func.id
                        if helper not in RLM_NATIVE_REPOSITORY_HELPERS:
                            continue
                        rlm_native_helper_calls_by_name[helper] = (
                            rlm_native_helper_calls_by_name.get(helper, 0) + 1
                        )
                        path_node = node.args[0] if helper != "apply_patch" and node.args else None
                        if helper == "grep":
                            path_node = next(
                                (
                                    keyword.value
                                    for keyword in node.keywords
                                    if keyword.arg == "path"
                                ),
                                node.args[1] if len(node.args) > 1 else None,
                            )
                        if (
                            isinstance(path_node, ast.Constant)
                            and isinstance(path_node.value, str)
                            and len(path_node.value) <= MAX_RETAINED_NATIVE_HELPER_PATH_LENGTH
                            and len(rlm_native_helper_paths) < MAX_RETAINED_NATIVE_HELPER_PATHS
                        ):
                            rlm_native_helper_paths.add(path_node.value)

    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "complete": malformed_lines == 0,
        "malformed_lines": malformed_lines,
        "rollouts": rollouts,
        "rollout_line_count": rollout_line_count,
        "compaction_events": compaction_events,
        "compaction_events_by_type": dict(sorted(compaction_events_by_type.items())),
        "tool_calls_by_name": dict(sorted(tool_calls_by_name.items())),
        "rlm_native_helper_calls_by_name": dict(sorted(rlm_native_helper_calls_by_name.items())),
        "rlm_native_helper_paths": sorted(rlm_native_helper_paths),
        "rlm_python_cells_with_subprocess": rlm_python_cells_with_subprocess,
        "rlm_python_cells_with_docker": rlm_python_cells_with_docker,
    }


def read_token_usage_samples(path: Path) -> dict[str, Any]:
    """Read an ordered, source-attributed token-count series from a bundle."""
    samples: list[dict[str, Any]] = []
    rollouts: list[str] = []
    malformed_lines = 0
    with tarfile.open(path, mode="r:gz") as bundle:
        for member in _rollout_members(bundle):
            extracted = bundle.extractfile(member)
            if extracted is None:
                continue
            rollouts.append(member.name)
            rollout_sequence = 0
            prime_totals = {
                "input_tokens": 0,
                "cached_input_tokens": 0,
                "cache_write_input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }
            for line_number, raw_line in enumerate(extracted, start=1):
                try:
                    event = json.loads(raw_line)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    malformed_lines += 1
                    continue
                if not isinstance(event, dict):
                    malformed_lines += 1
                    continue
                if member.name.startswith("rollouts/prime-agent/sessions/"):
                    usage = None
                    event_kind = event.get("type")
                    if event_kind == "message":
                        message = event.get("message") or {}
                        if message.get("role") == "assistant":
                            usage = message.get("usage")
                    elif event_kind == "child_usage_attributed":
                        # The parent aggregate replaces an earlier message's
                        # usage. The childUsage field is the exact new delta.
                        usage = event.get("childUsage")
                    if not isinstance(usage, dict):
                        continue
                    non_cached = int(usage.get("input") or 0)
                    cache_read = int(usage.get("cacheRead") or 0)
                    cache_write = int(usage.get("cacheWrite") or 0)
                    output = int(usage.get("output") or 0)
                    last_usage = {
                        "input_tokens": non_cached + cache_read + cache_write,
                        "cached_input_tokens": cache_read,
                        "cache_write_input_tokens": cache_write,
                        "output_tokens": output,
                        "total_tokens": non_cached + cache_read + cache_write + output,
                    }
                    for key, value in last_usage.items():
                        prime_totals[key] += value
                    rollout_sequence += 1
                    samples.append(
                        {
                            "timestamp": event.get("timestamp"),
                            "rollout": member.name,
                            "rollout_sequence": rollout_sequence,
                            "line": line_number,
                            "event_type": event_kind,
                            "last_token_usage": last_usage,
                            "total_token_usage": dict(prime_totals),
                            "model_context_window": None,
                        }
                    )
                    continue
                payload = event.get("payload") or {}
                if payload.get("type") != "token_count":
                    continue
                info = payload.get("info")
                if not isinstance(info, dict):
                    continue
                rollout_sequence += 1
                samples.append(
                    {
                        "timestamp": event.get("timestamp"),
                        "rollout": member.name,
                        "rollout_sequence": rollout_sequence,
                        "line": line_number,
                        "last_token_usage": info.get("last_token_usage"),
                        "total_token_usage": info.get("total_token_usage"),
                        "model_context_window": info.get("model_context_window"),
                    }
                )

    samples.sort(
        key=lambda sample: (
            sample["timestamp"] is None,
            sample["timestamp"] or "",
            sample["rollout"],
            sample["rollout_sequence"],
        )
    )
    for sequence, sample in enumerate(samples, start=1):
        sample["sequence"] = sequence
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "complete": malformed_lines == 0,
        "malformed_lines": malformed_lines,
        "ordering": "event_timestamp_then_rollout_file_order",
        "events_seen": len(samples),
        "rollouts": rollouts,
        "samples": samples,
    }


async def retain_agent_evidence(
    sandy: Any,
    sandbox_id: str,
    *,
    run_id: str | None,
    benchmark_name: str,
    item_id: str,
    require_rollout: bool = False,
) -> dict[str, Any]:
    """Retain one item's evidence without ever raising into the score path."""
    settings = get_settings()
    final_path: Path | None = None
    partial_path: Path | None = None
    partial_file: BinaryIO | None = None
    try:
        async with asyncio.timeout(settings.agent_evidence_transfer_timeout_seconds):
            metadata = await _prepare_bundle(sandy, sandbox_id)
            size_bytes = int(metadata["size_bytes"])
            sandbox_sha256 = str(metadata["sha256"])
            sandbox_path = str(metadata["path"])
            if size_bytes <= 0:
                raise RuntimeError("sandbox evidence bundle is empty")
            if size_bytes > settings.agent_evidence_max_item_bytes:
                raise RuntimeError(
                    f"compressed evidence exceeds per-item cap: {size_bytes} > "
                    f"{settings.agent_evidence_max_item_bytes} bytes"
                )

            root = Path(settings.agent_evidence_dir)
            final_path = _artifact_path(
                root,
                run_id=run_id,
                benchmark_name=benchmark_name,
                item_id=item_id,
                sandbox_id=sandbox_id,
            )
            partial_path, partial_file = await asyncio.to_thread(
                _reserve_local_file,
                final_path,
                size_bytes,
                max_total_bytes=settings.agent_evidence_max_total_bytes,
                max_age_days=settings.agent_evidence_max_age_days,
                min_free_bytes=settings.agent_evidence_min_free_bytes,
            )
            await _transfer_chunks(
                sandy,
                sandbox_id,
                sandbox_path,
                partial_file,
                size_bytes,
                chunk_bytes=settings.agent_evidence_chunk_bytes,
                concurrency=max(1, settings.agent_evidence_chunk_concurrency),
            )
            partial_file.close()
            partial_file = None

            local_sha256 = await asyncio.to_thread(_sha256_file, partial_path)
            if local_sha256 != sandbox_sha256:
                corrupt_path = partial_path.with_suffix(partial_path.suffix + ".CORRUPT")
                os.replace(partial_path, corrupt_path)
                partial_path = None
                raise RuntimeError(
                    "evidence SHA-256 mismatch; incomplete bundle quarantined at "
                    f"{corrupt_path} (sandbox={sandbox_sha256}, local={local_sha256})"
                )

            parse_errors: list[str] = []
            try:
                token_usage_samples = await asyncio.to_thread(
                    read_token_usage_samples, partial_path
                )
            except Exception as exc:
                parse_errors.append(f"retained bundle token-series parse failed: {exc}")
                token_usage_samples = {
                    "schema_version": EVIDENCE_SCHEMA_VERSION,
                    "complete": False,
                    "error": parse_errors[-1],
                    "samples": [],
                }
            try:
                rollout_metrics = await asyncio.to_thread(read_rollout_metrics, partial_path)
            except Exception as exc:
                parse_errors.append(f"retained bundle rollout-metrics parse failed: {exc}")
                rollout_metrics = {
                    "schema_version": EVIDENCE_SCHEMA_VERSION,
                    "complete": False,
                    "error": parse_errors[-1],
                    "rollouts": [],
                    "rollout_line_count": None,
                    "compaction_events": None,
                    "compaction_events_by_type": {},
                    "tool_calls_by_name": {},
                    "rlm_native_helper_calls_by_name": {},
                    "rlm_native_helper_paths": [],
                    "rlm_python_cells_with_subprocess": None,
                    "rlm_python_cells_with_docker": None,
                }
            if require_rollout and not rollout_metrics.get("rollouts"):
                raise RuntimeError(
                    "evidence bundle contains no agent rollout JSONL; refusing "
                    "to retain a plausible but unauditable archive"
                )

            parse_error = "; ".join(parse_errors) or None
            os.replace(partial_path, final_path)
            partial_path = None
            return {
                "status": "retained",
                "path": str(final_path),
                "sha256": local_sha256,
                "size_bytes": size_bytes,
                "error": parse_error,
                "token_usage_samples": token_usage_samples,
                "rollout_metrics": rollout_metrics,
                "sandbox_sources": metadata.get("sources") or [],
                "retention_policy": {
                    "max_item_bytes": settings.agent_evidence_max_item_bytes,
                    "max_total_bytes": settings.agent_evidence_max_total_bytes,
                    "max_age_days": settings.agent_evidence_max_age_days,
                    "min_free_bytes": settings.agent_evidence_min_free_bytes,
                },
            }
    except Exception as exc:
        if partial_file is not None:
            partial_file.close()
        if partial_path is not None and partial_path.exists():
            partial_path.unlink(missing_ok=True)
        error = str(exc) or exc.__class__.__name__
        logger.error(
            "Agent evidence retention failed",
            sandbox_id=sandbox_id,
            item_id=item_id,
            error=error,
        )
        return {
            "status": "failed",
            "path": None,
            "sha256": None,
            "size_bytes": None,
            "error": error,
            "token_usage_samples": None,
            "rollout_metrics": None,
        }
