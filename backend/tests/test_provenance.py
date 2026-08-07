"""Tests for immutable run and execution provenance."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy.dialects import postgresql

from app.models.benchmark import Benchmark
from app.models.model import Model
from app.models.run import BenchmarkRun, RunStatus
from app.services import provenance_service
from app.services.provenance_service import PROVENANCE_SCHEMA, ProvenanceError
from app.services.run_service import _run_provenance_lock_query, bind_run_provenance


def _provenance(git_sha: str = "a" * 40) -> dict:
    return {
        "schema": PROVENANCE_SCHEMA,
        "bench_runner_git_sha": git_sha,
        "code_version": "7" * 64,
        "worker_image_digest": "sha256:" + "1" * 64,
        "adapter_sha256": {"terminal_bench.py": "2" * 64},
        "adapter_set_sha256": "3" * 64,
        "sandy_runtime_image_digest": "sha256:" + "4" * 64,
        "sandy_runtime_configured_tag": "sandy-runtime:latest",
        "agent_binaries": {
            "codex": {"path": "/usr/bin/codex", "sha256": "5" * 64}
        },
    }


def test_worker_provenance_uses_runtime_digest_and_hashes_all_adapters(
    monkeypatch,
) -> None:
    runtime = SimpleNamespace(id="sha256:" + "4" * 64)
    client = SimpleNamespace(images=SimpleNamespace(get=lambda _tag: runtime))
    monkeypatch.setattr(provenance_service.docker, "from_env", lambda: client)
    monkeypatch.setattr(
        provenance_service,
        "get_settings",
        lambda: SimpleNamespace(sandy_runtime_image="sandy-runtime:latest"),
    )
    monkeypatch.setattr(provenance_service, "_git_sha", lambda: "a" * 40)
    monkeypatch.setattr(
        provenance_service,
        "_worker_image_id",
        lambda _client: "sha256:" + "1" * 64,
    )
    monkeypatch.setattr(
        provenance_service,
        "_runtime_binaries",
        lambda _image: {
            "codex": {"path": "/usr/bin/codex", "sha256": "5" * 64}
        },
    )

    result = provenance_service._collect_worker_provenance_sync()

    assert result["bench_runner_git_sha"] == "a" * 40
    assert result["sandy_runtime_image_digest"] == runtime.id
    assert result["worker_image_digest"].startswith("sha256:")
    assert result["agent_binaries"]["codex"]["sha256"] == "5" * 64
    assert "terminal_bench.py" in result["adapter_sha256"]
    assert "terminal_bench_gateway.py" in result["adapter_sha256"]
    assert len(result["adapter_set_sha256"]) == 64
    assert len(result["code_version"]) == 64


def test_invalid_embedded_git_sha_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv("BENCH_RUNNER_GIT_SHA", "main")

    with pytest.raises(ProvenanceError, match="full 40-character"):
        provenance_service._git_sha()


def test_provenance_lock_targets_only_benchmark_run_row() -> None:
    sql = str(
        _run_provenance_lock_query("run-id").compile(
            dialect=postgresql.dialect()
        )
    )

    assert "FOR UPDATE OF benchmark_runs" in sql


def test_actual_sandbox_agent_binary_must_match_run_snapshot(monkeypatch) -> None:
    sandbox = SimpleNamespace(
        attrs={
            "Image": "sha256:" + "4" * 64,
            "Config": {"Labels": {"sandy.id": "sandbox"}},
        }
    )
    client = SimpleNamespace()
    monkeypatch.setattr(provenance_service.docker, "from_env", lambda: client)
    monkeypatch.setattr(
        provenance_service,
        "_sandbox_container",
        lambda _client, _sandbox_id: sandbox,
    )
    monkeypatch.setattr(
        provenance_service,
        "_resolve_binary",
        lambda _container, _command: "/usr/bin/codex",
    )
    monkeypatch.setattr(
        provenance_service,
        "_archive_file_hash",
        lambda _container, _path: "5" * 64,
    )

    observed = provenance_service.capture_sandbox_agent_provenance(
        "sandbox", "codex", _provenance()
    )

    assert observed["matches_run"] is True
    assert observed["agent_binary_sha256"] == "5" * 64

    changed = _provenance()
    changed["agent_binaries"]["codex"]["sha256"] = "6" * 64
    with pytest.raises(ProvenanceError, match="differs from the run snapshot"):
        provenance_service.capture_sandbox_agent_provenance(
            "sandbox", "codex", changed
        )


@pytest.mark.asyncio
async def test_run_is_bound_once_to_full_provenance_snapshot(test_session) -> None:
    model = Model(
        slug="provenance-model",
        name="Provenance Model",
        provider="chutes",
        is_active=True,
    )
    benchmark = Benchmark(
        name="provenance-benchmark",
        display_name="Provenance Benchmark",
        adapter_class="ProvenanceAdapter",
        is_enabled=True,
        supports_subset=True,
    )
    test_session.add_all([model, benchmark])
    await test_session.flush()
    run = BenchmarkRun(
        model_id=model.id,
        model_slug=model.slug,
        provider="chutes",
        subset_pct=100,
        status=RunStatus.RUNNING.value,
    )
    test_session.add(run)
    await test_session.commit()

    provenance = _provenance()
    await bind_run_provenance(test_session, run.id, provenance)
    await test_session.refresh(run)

    assert run.provenance == provenance
    assert run.git_sha == "a" * 40
    assert run.code_version == "7" * 64

    changed = _provenance("b" * 40)
    with pytest.raises(RuntimeError, match="different code/runtime provenance"):
        await bind_run_provenance(test_session, run.id, changed)
