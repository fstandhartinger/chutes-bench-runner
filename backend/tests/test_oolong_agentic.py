import base64
import hashlib
from types import SimpleNamespace

import pytest

import app.benchmarks.adapters.oolong_agentic as oolong_agentic_module
from app.benchmarks.adapters.oolong_agentic import (
    EVIDENCE_RETENTION_EXCLUSION_REASON,
    NETWORK_PROBE_EXCLUSION_REASON,
    NETWORK_SEAL_BROKEN_EXCLUSION_REASON,
    OolongAgenticAdapter,
)
from app.benchmarks.base import ItemResult
from app.services.sandy_service import SandyService


class _CorpusSandy:
    def __init__(self):
        self.last_error = None
        self.encoded = ""

    async def write_file(self, sandbox_id, path, content):
        assert path == "corpus.b64"
        self.encoded = content
        return True

    async def execute_command(self, sandbox_id, command, timeout_ms=None):
        if command.startswith("base64 -d"):
            return {"exit_code": 0, "stdout": ""}
        raw = base64.b64decode(self.encoded)
        return {
            "exit_code": 0,
            "stdout": f"{len(raw)}\n{hashlib.sha256(raw).hexdigest()}\n",
        }


@pytest.mark.asyncio
async def test_corpus_upload_proves_bytes_and_sha256():
    adapter = OolongAgenticAdapter.__new__(OolongAgenticAdapter)
    adapter.sandy = _CorpusSandy()

    result = await adapter._write_corpus("sandbox", "long transcript\n" * 100)

    assert result["uploaded"] is True
    assert result["written_bytes"] == result["expected_bytes"]
    assert result["written_sha256"] == result["expected_sha256"]


class _SealSandy:
    def __init__(self, probe_stdout, probe_exit_code=0):
        self.probe_stdout = probe_stdout
        self.probe_exit_code = probe_exit_code

    async def execute_command(self, sandbox_id, command, timeout_ms=None):
        if "provider_addresses=$(getent" in command:
            return {"exit_code": 0, "stdout": ""}
        return {"exit_code": self.probe_exit_code, "stdout": self.probe_stdout}


class _OpenRouterClient:
    def get_api_base_url(self):
        return "https://openrouter.ai/api/v1"


class _RuntimeSandy:
    def __init__(self, exit_code=0, stdout=""):
        self.exit_code = exit_code
        self.stdout = stdout
        self.command = None

    async def execute_command(self, sandbox_id, command, timeout_ms=None):
        self.command = command
        return {
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": "missing dependency" if self.exit_code else "",
        }


class _RetentionSandy:
    def __init__(self):
        self.commands = []

    async def execute_command(self, sandbox_id, command, timeout_ms=None):
        self.commands.append(command)
        return {"exit_code": 0, "stdout": ""}


class _CreateResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"sandboxId": "agent-sandbox"}


class _CreateClient:
    def __init__(self):
        self.payload = None

    async def post(self, url, headers, json):
        self.payload = json
        return _CreateResponse()


@pytest.mark.asyncio
async def test_agent_sandbox_requests_agent_ready_pid_budget():
    sandy = SandyService.__new__(SandyService)
    sandy.api_key = "test"
    sandy.base_url = "https://sandy.test"
    sandy.docker_upstream = None
    sandy.last_error = None
    sandy.headers = {}
    sandy._client = _CreateClient()

    sandbox_id = await sandy.create_sandbox(requires_agent=True, timeout_minutes=21)

    assert sandbox_id == "agent-sandbox"
    assert sandy._client.payload["flavor"] == "agent-ready"
    assert sandy._client.payload["timeoutMinutes"] == 21


@pytest.mark.asyncio
async def test_rlm_runtime_preflight_imports_persistent_kernel_dependencies():
    adapter = OolongAgenticAdapter.__new__(OolongAgenticAdapter)
    adapter.sandy = _RuntimeSandy(
        stdout="/usr/local/bin/chutescoder\n9.7.0 0.4.0 0.1.0"
    )

    result = await adapter._probe_agent_runtime("sandbox", "chutescoder")

    assert result["ready"] is True
    assert "import IPython,dill,chutescoder_rlm" in adapter.sandy.command
    assert "PYTHONPATH=/opt/chutescoder/python" in adapter.sandy.command


@pytest.mark.asyncio
async def test_rlm_runtime_preflight_rejects_missing_ipython():
    adapter = OolongAgenticAdapter.__new__(OolongAgenticAdapter)
    adapter.sandy = _RuntimeSandy(exit_code=1)

    result = await adapter._probe_agent_runtime("sandbox", "chutescoder")

    assert result["ready"] is False
    assert result["exit_code"] == 1


@pytest.mark.asyncio
async def test_baseline_preflight_does_not_require_rlm_dependencies():
    adapter = OolongAgenticAdapter.__new__(OolongAgenticAdapter)
    adapter.sandy = _RuntimeSandy(stdout="/usr/local/bin/chutescoder")

    result = await adapter._probe_agent_runtime("sandbox", "chutescoder-baseline")

    assert result["ready"] is True
    assert "IPython" not in adapter.sandy.command


@pytest.mark.asyncio
async def test_timeout_path_mirrors_rollout_before_evidence_archive(monkeypatch):
    calls = []

    async def retain_rollout(sandy, sandbox_id, launch):
        calls.append(("rollout", sandbox_id, launch))

    async def retain_evidence(sandy, sandbox_id, **kwargs):
        calls.append(("evidence", sandbox_id, kwargs["item_id"]))
        return {
            "status": "retained",
            "path": "/evidence.tar.gz",
            "sha256": "a" * 64,
            "size_bytes": 10,
            "error": None,
            "token_usage_samples": {"samples": [{"sequence": 1}]},
        }

    monkeypatch.setattr(
        oolong_agentic_module, "retain_sandy_agent_rollout", retain_rollout
    )
    monkeypatch.setattr(oolong_agentic_module, "retain_agent_evidence", retain_evidence)
    adapter = OolongAgenticAdapter.__new__(OolongAgenticAdapter)
    adapter.sandy = _RetentionSandy()
    adapter.run_id = "run"
    adapter.get_name = lambda: "oolong_agentic"
    adapter._item_observability = {}
    state = adapter._new_item_observability("803")
    launch = SimpleNamespace(setup=object())
    state.update(agent_invoked=True, agent_launch=launch)

    await adapter._finish_evidence_retention("803", "sandbox")

    assert calls == [
        ("rollout", "sandbox", launch),
        ("evidence", "sandbox", "803"),
    ]
    assert state["rollout_retained"] is True
    assert state["evidence"]["token_usage_samples"]["samples"]
    assert state["evidence"]["rollout_retention_error"] is None


@pytest.mark.asyncio
async def test_network_seal_requires_hosts_entry_curl_and_failed_fetch():
    adapter = OolongAgenticAdapter.__new__(OolongAgenticAdapter)
    adapter.sandy = _SealSandy(
        "HOSTS=1\nCURL=yes\nSOURCE_FETCH=000\nPUBLIC_FETCH=000\nPROVIDER_FETCH=200"
    )
    adapter.client = _OpenRouterClient()

    sealed = await adapter._seal_network("sandbox")

    assert sealed["sealed"] is True
    assert sealed["probe_complete"] is True
    assert sealed["probe_exit_code"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stdout",
    [
        "HOSTS=0\nCURL=yes\nSOURCE_FETCH=000\nPUBLIC_FETCH=000\nPROVIDER_FETCH=200",
        "HOSTS=1\nCURL=no\nSOURCE_FETCH=NOCURL\nPUBLIC_FETCH=NOCURL\nPROVIDER_FETCH=NOCURL",
        "HOSTS=1\nCURL=yes\nSOURCE_FETCH=200\nPUBLIC_FETCH=000\nPROVIDER_FETCH=200",
        "HOSTS=1\nCURL=yes\nSOURCE_FETCH=000\nPUBLIC_FETCH=200\nPROVIDER_FETCH=200",
        "HOSTS=1\nCURL=yes\nSOURCE_FETCH=000\nPUBLIC_FETCH=000\nPROVIDER_FETCH=000",
    ],
)
async def test_network_seal_rejects_unproved_or_reachable_sources(stdout):
    adapter = OolongAgenticAdapter.__new__(OolongAgenticAdapter)
    adapter.sandy = _SealSandy(stdout)
    adapter.client = _OpenRouterClient()

    verdict = await adapter._probe_network_seal("sandbox")

    assert verdict["sealed"] is False


@pytest.mark.asyncio
async def test_network_probe_cannot_fork_is_infrastructure_not_broken_seal():
    adapter = OolongAgenticAdapter.__new__(OolongAgenticAdapter)
    adapter.sandy = _SealSandy("sh: 4: Cannot fork", probe_exit_code=2)
    adapter.client = _OpenRouterClient()

    verdict = await adapter._probe_network_seal("sandbox")

    assert verdict["sealed"] is False
    assert verdict["probe_complete"] is False
    assert (
        adapter._network_probe_exclusion_reason(
            verdict, NETWORK_SEAL_BROKEN_EXCLUSION_REASON
        )
        == NETWORK_PROBE_EXCLUSION_REASON
    )


def test_complete_failed_network_probe_remains_integrity_exclusion():
    verdict = {
        "probe_exit_code": 0,
        "probe_complete": True,
        "sealed": False,
    }

    assert (
        OolongAgenticAdapter._network_probe_exclusion_reason(
            verdict, NETWORK_SEAL_BROKEN_EXCLUSION_REASON
        )
        == NETWORK_SEAL_BROKEN_EXCLUSION_REASON
    )


def test_missing_evidence_excludes_item_without_changing_observed_score():
    adapter = OolongAgenticAdapter.__new__(OolongAgenticAdapter)
    adapter._item_observability = {
        "1253": {
            "agent_invoked": True,
            "evidence": {
                "status": "failed",
                "path": None,
                "sha256": None,
                "size_bytes": None,
                "error": "transfer failed",
                "token_usage_samples": None,
            },
        }
    }
    observed = ItemResult(item_id="1253", score=1.0, is_correct=True)

    result = adapter.attach_item_observability(observed)

    assert result.score == 1.0
    assert result.metadata["exclusion_reason"] == EVIDENCE_RETENTION_EXCLUSION_REASON
    assert result.agent_evidence_path is None


def test_worker_timeout_is_longer_than_agent_budget(monkeypatch):
    adapter = OolongAgenticAdapter.__new__(OolongAgenticAdapter)
    monkeypatch.setenv("OOLONG_AGENTIC_MAX_SECONDS", "37")

    assert adapter.get_item_timeout_seconds() == 337


def test_network_seal_keeps_only_provider_dns(monkeypatch):
    adapter = OolongAgenticAdapter.__new__(OolongAgenticAdapter)
    adapter.client = _OpenRouterClient()

    script = adapter._seal_script()

    assert "getent ahosts" in script
    assert "openrouter.ai" in script
    assert "nameserver 127.0.0.1" in script
    assert "datasets-server.huggingface.co" in script
