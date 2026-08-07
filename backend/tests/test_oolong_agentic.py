import base64
import hashlib

import pytest

from app.benchmarks.adapters.oolong_agentic import (
    EVIDENCE_RETENTION_EXCLUSION_REASON,
    OolongAgenticAdapter,
)
from app.benchmarks.base import ItemResult


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
    def __init__(self, probe_stdout):
        self.probe_stdout = probe_stdout

    async def execute_command(self, sandbox_id, command, timeout_ms=None):
        if "provider_addresses=$(getent" in command:
            return {"exit_code": 0, "stdout": ""}
        return {"exit_code": 0, "stdout": self.probe_stdout}


class _OpenRouterClient:
    def get_api_base_url(self):
        return "https://openrouter.ai/api/v1"


@pytest.mark.asyncio
async def test_network_seal_requires_hosts_entry_curl_and_failed_fetch():
    adapter = OolongAgenticAdapter.__new__(OolongAgenticAdapter)
    adapter.sandy = _SealSandy(
        "HOSTS=1\nCURL=yes\nSOURCE_FETCH=000\n"
        "PUBLIC_FETCH=000\nPROVIDER_FETCH=200"
    )
    adapter.client = _OpenRouterClient()

    sealed = await adapter._seal_network("sandbox")

    assert sealed["sealed"] is True


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
