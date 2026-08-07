"""Immutable, outside-observed provenance for benchmark executions."""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import os
import re
import shlex
import socket
import subprocess
import tarfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import docker

from app.core.config import get_settings

PROVENANCE_SCHEMA = "bench-runner-provenance-v1"
GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
AGENT_COMMANDS = (
    "aider",
    "chutescoder",
    "claude",
    "codex",
    "droid",
    "openhands",
    "opencode",
    "prime-agent",
)
AGENT_COMMAND_BY_NAME = {
    "chutescoder-baseline": "chutescoder",
}


class ProvenanceError(RuntimeError):
    """Required execution provenance could not be proved."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    configured = (os.getenv("BENCH_RUNNER_GIT_SHA") or "").strip().lower()
    if configured:
        if not GIT_SHA_PATTERN.fullmatch(configured):
            raise ProvenanceError("BENCH_RUNNER_GIT_SHA is not a full 40-character Git SHA")
        return configured

    # Development/test fallback only. Production worker images do not contain
    # git, so an image built without the build arg fails closed here.
    repo_root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception as exc:
        raise ProvenanceError(
            "worker image has no embedded BENCH_RUNNER_GIT_SHA"
        ) from exc
    sha = result.stdout.strip().lower()
    if not GIT_SHA_PATTERN.fullmatch(sha):
        raise ProvenanceError("could not resolve a full bench-runner Git SHA")
    return sha


def _adapter_hashes() -> dict[str, str]:
    adapter_dir = Path(__file__).resolve().parents[1] / "benchmarks" / "adapters"
    files = sorted(adapter_dir.glob("*.py"))
    if not files:
        raise ProvenanceError(f"no adapter files found under {adapter_dir}")
    return {path.name: _sha256_file(path) for path in files}


def _archive_file_hash(container: Any, path: str) -> str:
    stream, _stat = container.get_archive(path)
    archive_bytes = b"".join(stream)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:*") as archive:
        regular = [member for member in archive.getmembers() if member.isfile()]
        if len(regular) != 1:
            raise ProvenanceError(f"expected one regular file at {path}, found {len(regular)}")
        extracted = archive.extractfile(regular[0])
        if extracted is None:
            raise ProvenanceError(f"could not read {path} from runtime container")
        return hashlib.sha256(extracted.read()).hexdigest()


def _resolve_binary(container: Any, command: str) -> str | None:
    quoted = shlex.quote(command)
    result = container.exec_run(
        ["sh", "-lc", f"p=$(command -v {quoted} 2>/dev/null) || exit 44; readlink -f \"$p\""],
        stdout=True,
        stderr=True,
    )
    if int(result.exit_code) == 44:
        return None
    if int(result.exit_code) != 0:
        raise ProvenanceError(f"failed to resolve agent binary {command}")
    path = (result.output or b"").decode("utf-8", errors="replace").strip()
    if not path.startswith("/"):
        raise ProvenanceError(f"agent binary {command} resolved to an invalid path: {path!r}")
    return path


@lru_cache(maxsize=8)
def _runtime_binaries(image_id: str) -> dict[str, dict[str, str] | None]:
    client = docker.from_env()
    container = client.containers.create(
        image_id,
        command=["sleep", "300"],
        network_disabled=True,
        labels={"chutes.bench.provenance-probe": "true"},
    )
    try:
        container.start()
        binaries: dict[str, dict[str, str] | None] = {}
        for command in AGENT_COMMANDS:
            path = _resolve_binary(container, command)
            binaries[command] = (
                {"path": path, "sha256": _archive_file_hash(container, path)}
                if path
                else None
            )
        return binaries
    finally:
        with contextlib.suppress(Exception):
            container.remove(force=True)


def _worker_image_id(client: Any) -> str:
    hostname = socket.gethostname()
    try:
        container = client.containers.get(hostname)
    except Exception as exc:
        raise ProvenanceError("could not inspect the running worker container") from exc
    return str(container.image.id)


def _collect_worker_provenance_sync() -> dict[str, Any]:
    client = docker.from_env()
    settings = get_settings()
    try:
        runtime_image = client.images.get(settings.sandy_runtime_image)
    except Exception as exc:
        raise ProvenanceError(
            f"could not resolve Sandy runtime image {settings.sandy_runtime_image!r}"
        ) from exc
    image_id = str(runtime_image.id)
    if not image_id.startswith("sha256:"):
        raise ProvenanceError(f"Sandy runtime did not resolve to a digest: {image_id!r}")

    git_sha = _git_sha()
    adapters = _adapter_hashes()
    adapter_set_sha256 = hashlib.sha256(
        "\n".join(f"{name}:{digest}" for name, digest in adapters.items()).encode("utf-8")
    ).hexdigest()
    code_version = hashlib.sha256(
        f"git:{git_sha}\nadapters:{adapter_set_sha256}\n".encode()
    ).hexdigest()
    return {
        "schema": PROVENANCE_SCHEMA,
        "bench_runner_git_sha": git_sha,
        "code_version": code_version,
        "worker_image_digest": _worker_image_id(client),
        "adapter_sha256": adapters,
        "adapter_set_sha256": adapter_set_sha256,
        "sandy_runtime_image_digest": image_id,
        "sandy_runtime_configured_tag": settings.sandy_runtime_image,
        "agent_binaries": _runtime_binaries(image_id),
    }


async def collect_worker_provenance() -> dict[str, Any]:
    """Snapshot actual worker/runtime artifacts before a run can produce results."""
    return await asyncio.to_thread(_collect_worker_provenance_sync)


def _sandbox_container(client: Any, sandbox_id: str):
    expected_name = f"sandy_{sandbox_id}"
    try:
        container = client.containers.get(expected_name)
    except Exception as exc:
        raise ProvenanceError(f"could not inspect Sandy sandbox {sandbox_id} from outside") from exc
    labels = (container.attrs.get("Config") or {}).get("Labels") or {}
    actual_id = labels.get("sandy.id")
    if actual_id and actual_id != sandbox_id:
        raise ProvenanceError(
            f"sandbox label mismatch: expected {sandbox_id}, observed {actual_id}"
        )
    return container


def capture_sandbox_agent_provenance(
    sandbox_id: str,
    agent_name: str,
    expected_run_provenance: dict[str, Any] | None,
) -> dict[str, Any]:
    """Inspect the exact sandbox image/binary from the worker, then bind it to the run."""
    client = docker.from_env()
    container = _sandbox_container(client, sandbox_id)
    image_digest = str(container.attrs.get("Image") or "")
    command = AGENT_COMMAND_BY_NAME.get(agent_name, agent_name)
    path = _resolve_binary(container, command)
    if not path:
        raise ProvenanceError(f"agent binary {command!r} is absent from the actual sandbox")
    binary_sha256 = _archive_file_hash(container, path)
    result = {
        "sandbox_id": sandbox_id,
        "sandy_runtime_image_digest": image_digest,
        "agent": agent_name,
        "agent_command": command,
        "agent_binary_path": path,
        "agent_binary_sha256": binary_sha256,
        "observed_from": "worker_docker_api",
    }

    expected = expected_run_provenance or {}
    expected_image = expected.get("sandy_runtime_image_digest")
    expected_binary = (expected.get("agent_binaries") or {}).get(command) or {}
    result["matches_run"] = (
        bool(expected_image)
        and image_digest == expected_image
        and binary_sha256 == expected_binary.get("sha256")
    )
    if not result["matches_run"]:
        raise ProvenanceError(
            "actual sandbox provenance differs from the run snapshot "
            f"(image={image_digest}, expected_image={expected_image}, "
            f"agent_sha256={binary_sha256}, expected_agent_sha256={expected_binary.get('sha256')})"
        )
    return result


def worker_image_id() -> str:
    """Return the immutable image id of this worker for trusted helper containers."""
    return _worker_image_id(docker.from_env())


def sandbox_container(sandbox_id: str):
    """Return a label-verified sandbox container from the outside Docker API."""
    return _sandbox_container(docker.from_env(), sandbox_id)
