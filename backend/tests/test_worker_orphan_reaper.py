"""Worker startup orphan-reaper tests."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.benchmarks.adapters import terminal_bench as terminal_bench_module
from app.worker import runner as runner_module
from app.worker.runner import TASK_SANDBOX_LABEL, BenchmarkWorker


class _FakeContainer:
    def __init__(self, sandbox_id: str, name: str):
        self.id = f"id-{name}"
        self.name = name
        self.removed = False
        self.attrs = {"Config": {"Labels": {TASK_SANDBOX_LABEL: sandbox_id}}}

    def remove(self, *, force: bool) -> None:
        assert force is True
        self.removed = True


class _FakeContainers:
    def __init__(self, containers: list[_FakeContainer]):
        self.containers = containers

    def list(self, *, all: bool, filters: dict[str, str]):
        assert all is True
        label_filter = filters["label"]
        key, separator, expected = label_filter.partition("=")
        matches = []
        for container in self.containers:
            if container.removed:
                continue
            labels = (container.attrs.get("Config") or {}).get("Labels") or {}
            if key not in labels:
                continue
            if separator and labels.get(key) != expected:
                continue
            matches.append(container)
        return matches


class _FakeImage:
    def __init__(self, tag: str):
        self.tags = [tag]


class _FakeImages:
    def __init__(self, images: list[_FakeImage]):
        self.images = images
        self.removed_tags: list[str] = []

    def list(self):
        return [image for image in self.images if image.tags]

    def remove(self, tag: str, *, force: bool) -> None:
        assert force is True
        self.removed_tags.append(tag)
        for image in self.images:
            if tag in image.tags:
                image.tags.remove(tag)


class _FakeNetwork:
    def __init__(self, sandbox_id: str, name: str):
        self.id = f"id-{name}"
        self.name = name
        self.removed = False
        self.attrs = {"Labels": {TASK_SANDBOX_LABEL: sandbox_id}}

    def remove(self) -> None:
        self.removed = True


class _FakeNetworks:
    def __init__(self, networks: list[_FakeNetwork]):
        self.networks = networks

    def list(self, *, filters: dict[str, str]):
        key, _, expected = filters["label"].partition("=")
        return [
            network
            for network in self.networks
            if not network.removed
            and (network.attrs.get("Labels") or {}).get(key) == expected
        ]


class _FakeSandy:
    def __init__(self, statuses: dict[str, bool | None], error: str | None = None):
        self.statuses = statuses
        self.last_error = error
        self.checked: list[str] = []

    async def sandbox_exists(self, sandbox_id: str) -> bool | None:
        self.checked.append(sandbox_id)
        return self.statuses[sandbox_id]


def _docker_client(
    containers: list[_FakeContainer],
    images: list[_FakeImage] | None = None,
    networks: list[_FakeNetwork] | None = None,
):
    return SimpleNamespace(
        containers=_FakeContainers(containers),
        images=_FakeImages(images or []),
        networks=_FakeNetworks(networks or []),
    )


def _install_fake_docker(monkeypatch, client) -> None:
    monkeypatch.setattr(runner_module.docker, "from_env", lambda: client)
    monkeypatch.setattr(terminal_bench_module.docker, "from_env", lambda: client)


def _summary(logger: MagicMock) -> dict:
    calls = [
        call
        for call in logger.info.call_args_list
        if call.args and call.args[0] == "Startup orphan reaper summary"
    ]
    assert len(calls) == 1
    return calls[0].kwargs


@pytest.mark.asyncio
async def test_startup_reaper_leaves_tracked_sandbox_resources_alone(monkeypatch) -> None:
    sandbox_id = "tracked123456789"
    container = _FakeContainer(sandbox_id, "tracked-task")
    image = _FakeImage(f"tbench_s{sandbox_id[:12]}_task:latest")
    network = _FakeNetwork(sandbox_id, "tracked-network")
    client = _docker_client([container], [image], [network])
    _install_fake_docker(monkeypatch, client)
    fake_logger = MagicMock()
    monkeypatch.setattr(runner_module, "logger", fake_logger)

    sandy = _FakeSandy({sandbox_id: True})
    await BenchmarkWorker().reap_orphaned_task_resources(sandy)

    assert sandy.checked == [sandbox_id]
    assert container.removed is False
    assert image.tags
    assert network.removed is False
    assert _summary(fake_logger) == {
        "scan_complete": True,
        "labelled_containers_seen": 1,
        "labelled_sandbox_ids_seen": 1,
        "invalid_labelled_containers_skipped": 0,
        "tracked_sandbox_ids_skipped": 1,
        "tracked_sandbox_ids": [sandbox_id],
        "unproven_sandbox_ids_skipped": 0,
        "unproven_sandbox_ids": [],
        "orphaned_sandbox_ids_reaped": 0,
        "reaped_sandbox_ids": [],
        "cleanup_failed_sandbox_ids": [],
    }


@pytest.mark.asyncio
async def test_startup_reaper_removes_resources_only_after_sandy_404(monkeypatch) -> None:
    sandbox_id = "orphan1234567890"
    container = _FakeContainer(sandbox_id, "orphan-task")
    image = _FakeImage(f"tbench_s{sandbox_id[:12]}_task:latest")
    network = _FakeNetwork(sandbox_id, "orphan-network")
    client = _docker_client([container], [image], [network])
    _install_fake_docker(monkeypatch, client)
    fake_logger = MagicMock()
    monkeypatch.setattr(runner_module, "logger", fake_logger)

    sandy = _FakeSandy({sandbox_id: False}, error="Sandbox not found")
    await BenchmarkWorker().reap_orphaned_task_resources(sandy)

    assert sandy.checked == [sandbox_id]
    assert container.removed is True
    assert image.tags == []
    assert network.removed is True
    summary = _summary(fake_logger)
    assert summary["orphaned_sandbox_ids_reaped"] == 1
    assert summary["reaped_sandbox_ids"] == [sandbox_id]
    assert summary["cleanup_failed_sandbox_ids"] == []


@pytest.mark.asyncio
async def test_startup_reaper_leaves_resources_on_sandy_api_error_and_logs(monkeypatch) -> None:
    sandbox_id = "uncertain1234567"
    container = _FakeContainer(sandbox_id, "uncertain-task")
    client = _docker_client([container])
    _install_fake_docker(monkeypatch, client)
    fake_logger = MagicMock()
    monkeypatch.setattr(runner_module, "logger", fake_logger)

    sandy = _FakeSandy({sandbox_id: None}, error="HTTP 503: unavailable")
    await BenchmarkWorker().reap_orphaned_task_resources(sandy)

    assert container.removed is False
    fake_logger.warning.assert_any_call(
        "Startup orphan reaper could not prove sandbox is untracked; "
        "leaving resources alone",
        sandbox_id=sandbox_id,
        reason="sandy_tracking_status_unknown",
        error="HTTP 503: unavailable",
    )
    summary = _summary(fake_logger)
    assert summary["unproven_sandbox_ids_skipped"] == 1
    assert summary["unproven_sandbox_ids"] == [sandbox_id]
    assert summary["orphaned_sandbox_ids_reaped"] == 0


@pytest.mark.asyncio
async def test_startup_reaper_does_not_count_incomplete_cleanup(monkeypatch) -> None:
    sandbox_id = "stuck12345678901"
    container = _FakeContainer(sandbox_id, "stuck-task")
    container.remove = MagicMock()
    client = _docker_client([container])
    _install_fake_docker(monkeypatch, client)
    fake_logger = MagicMock()
    monkeypatch.setattr(runner_module, "logger", fake_logger)

    sandy = _FakeSandy({sandbox_id: False}, error="Sandbox not found")
    await BenchmarkWorker().reap_orphaned_task_resources(sandy)

    summary = _summary(fake_logger)
    assert container.removed is False
    assert summary["orphaned_sandbox_ids_reaped"] == 0
    assert summary["reaped_sandbox_ids"] == []
    assert summary["cleanup_failed_sandbox_ids"] == [sandbox_id]
    fake_logger.warning.assert_any_call(
        "Startup orphan reaper could not verify resource cleanup",
        sandbox_id=sandbox_id,
        reason="owned_resource_cleanup_failed_or_incomplete",
    )


@pytest.mark.asyncio
async def test_startup_reaper_exception_does_not_prevent_worker_loop(monkeypatch) -> None:
    worker = BenchmarkWorker()
    events: list[str] = []

    async def failed_reaper() -> None:
        events.append("reaper")
        raise RuntimeError("docker socket unavailable")

    async def launch_runs() -> None:
        events.append("claim")
        worker.running = False

    monkeypatch.setattr(worker, "reap_orphaned_task_resources", failed_reaper)
    monkeypatch.setattr(worker, "requeue_stale_runs", AsyncMock())
    monkeypatch.setattr(worker, "touch_active_runs", AsyncMock())
    monkeypatch.setattr(worker, "touch_worker_heartbeat", AsyncMock())
    monkeypatch.setattr(worker, "reap_completed_runs", AsyncMock())
    monkeypatch.setattr(worker, "launch_runs", launch_runs)
    monkeypatch.setattr(runner_module.asyncio, "sleep", AsyncMock())
    fake_logger = MagicMock()
    monkeypatch.setattr(runner_module, "logger", fake_logger)

    await worker.start()

    assert events == ["reaper", "claim"]
    fake_logger.exception.assert_called_once_with(
        "Startup orphan reaper failed; worker startup will continue",
        error="docker socket unavailable",
    )
    fake_logger.warning.assert_called_once_with(
        "Startup orphan reaper summary",
        scan_complete=False,
        failure_reason="unexpected_reaper_failure",
        error="docker socket unavailable",
        labelled_containers_seen=None,
        labelled_sandbox_ids_seen=None,
        invalid_labelled_containers_skipped=None,
        tracked_sandbox_ids_skipped=None,
        tracked_sandbox_ids=None,
        unproven_sandbox_ids_skipped=None,
        unproven_sandbox_ids=None,
        orphaned_sandbox_ids_reaped=None,
        reaped_sandbox_ids=None,
        cleanup_failed_sandbox_ids=None,
    )
