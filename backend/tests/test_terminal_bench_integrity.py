"""Regression tests for the Terminal-Bench integrity boundary."""
from __future__ import annotations

import base64
import hashlib
import io
import tarfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.benchmarks.adapters import terminal_bench
from app.benchmarks.adapters.terminal_bench import TerminalBench21Adapter
from app.benchmarks.adapters.terminal_bench_gateway import (
    GatewayError,
    TaskGateway,
    build_agent_docker_wrapper,
)


def _partition() -> dict:
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as task:
        for name, content in {
            "instruction.md": b"public",
            "solution/solve.sh": b"reference",
            "tests/test.py": b"tests",
        }.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            task.addfile(member, io.BytesIO(content))
    return TerminalBench21Adapter._partition_answer_key(archive.getvalue())


@pytest.mark.asyncio
async def test_task_container_is_scanned_through_agent_and_outside_paths() -> None:
    adapter = TerminalBench21Adapter.__new__(TerminalBench21Adapter)
    partition = _partition()
    clean_output = "SOURCE_ARCHIVE=ABSENT\n"
    adapter.sandy = SimpleNamespace(
        execute_command=AsyncMock(
            return_value={"exit_code": 0, "stdout": clean_output}
        )
    )
    adapter._docker_exec_outside = AsyncMock(
        return_value={"exit_code": 0, "stdout": clean_output}
    )

    verdict = await adapter._verify_container_clean(
        "sandbox", "task-container", partition
    )

    assert verdict["clean"] is True
    agent_command = adapter.sandy.execute_command.await_args.args[1]
    assert "docker exec task-container" in agent_command
    assert "find / -type f" in agent_command
    outside_argv = adapter._docker_exec_outside.await_args.args[1]
    assert outside_argv[:2] == ["sh", "-c"]
    assert "find / -type f" in outside_argv[2]


@pytest.mark.asyncio
async def test_task_container_scan_fails_when_either_observer_finds_answer() -> None:
    adapter = TerminalBench21Adapter.__new__(TerminalBench21Adapter)
    partition = _partition()
    answer_hash = partition["reference_manifest"][0]["sha256"]
    adapter.sandy = SimpleNamespace(
        execute_command=AsyncMock(
            return_value={
                "exit_code": 0,
                "stdout": f"{answer_hash}  /solution/solve.sh\nSOURCE_ARCHIVE=ABSENT\n",
            }
        )
    )
    adapter._docker_exec_outside = AsyncMock(
        return_value={"exit_code": 0, "stdout": "SOURCE_ARCHIVE=ABSENT\n"}
    )

    verdict = await adapter._verify_container_clean(
        "sandbox", "task-container", partition
    )

    assert verdict["clean"] is False
    assert verdict["agent_path_matches"] == [
        f"{answer_hash}  /solution/solve.sh"
    ]


@pytest.mark.asyncio
async def test_restore_tests_checks_every_test_hash_and_never_reference_hash() -> None:
    adapter = TerminalBench21Adapter.__new__(TerminalBench21Adapter)
    partition = _partition()
    expected = partition["tests_manifest"][0]
    calls = [
        {"exit_code": 0},
        {
            "exit_code": 0,
            "stdout": f"{expected['sha256']}  /{expected['path']}\n",
        },
        {"exit_code": 0, "stdout": ""},
    ]
    adapter._docker_exec_outside = AsyncMock(side_effect=calls)
    adapter._put_archive_outside = AsyncMock(return_value={"exit_code": 0})

    verdict = await adapter._restore_tests("task-container", partition)

    assert verdict["restored"] is True
    assert verdict["reference_solution_restored"] is False
    assert verdict["observed_test_sha256"] == {
        f"/{expected['path']}": expected["sha256"]
    }
    transferred = adapter._put_archive_outside.await_args.args[2]
    assert hashlib.sha256(b"reference").hexdigest().encode() not in transferred
    with tarfile.open(fileobj=io.BytesIO(transferred)) as tests:
        assert "solution/solve.sh" not in tests.getnames()


@pytest.mark.asyncio
async def test_agent_position_docker_bypass_proof_requires_every_marker(
    monkeypatch,
) -> None:
    output = "\n".join(
        [
            "SOCKET=ABSENT",
            "CACHE_MOUNT=ABSENT",
            "CACHE_FILES=0",
            "RAW_DOCKER=BLOCKED",
            "SPAWN=BLOCKED",
            "OTHER_CONTAINER=BLOCKED",
            "TASK_PATH=WORKS",
        ]
    ).encode()
    sandbox = SimpleNamespace(
        exec_run=MagicMock(return_value=SimpleNamespace(exit_code=0, output=output))
    )
    monkeypatch.setattr(
        "app.benchmarks.adapters.terminal_bench.sandbox_container",
        lambda _sandbox_id: sandbox,
    )
    adapter = TerminalBench21Adapter.__new__(TerminalBench21Adapter)

    verdict = await adapter._verify_agent_docker_boundary("sandbox", "task")

    assert verdict["boundary_held"] is True
    command = sandbox.exec_run.call_args.args[0][2]
    assert "socket.AF_UNIX" in command
    assert "s.connect('/var/run/docker.sock')" in command
    assert "docker run --rm" in command
    assert "merge-diff-arc-agi-task/solution/solve.sh" in command
    assert "docker exec chutes-bench-runner-worker-1" in command
    assert "docker exec task true" in command


@pytest.mark.asyncio
async def test_agent_position_docker_bypass_proof_fails_if_spawn_succeeds(
    monkeypatch,
) -> None:
    output = "\n".join(
        [
            "SOCKET=ABSENT",
            "CACHE_MOUNT=ABSENT",
            "CACHE_FILES=0",
            "RAW_DOCKER=BLOCKED",
            "SPAWN=ESCAPED",
            "OTHER_CONTAINER=BLOCKED",
            "TASK_PATH=WORKS",
        ]
    ).encode()
    sandbox = SimpleNamespace(
        exec_run=MagicMock(return_value=SimpleNamespace(exit_code=0, output=output))
    )
    monkeypatch.setattr(
        "app.benchmarks.adapters.terminal_bench.sandbox_container",
        lambda _sandbox_id: sandbox,
    )
    adapter = TerminalBench21Adapter.__new__(TerminalBench21Adapter)

    verdict = await adapter._verify_agent_docker_boundary("sandbox", "task")

    assert verdict["boundary_held"] is False


@pytest.mark.asyncio
async def test_network_seal_cannot_pass_when_docker_boundary_fails() -> None:
    adapter = TerminalBench21Adapter.__new__(TerminalBench21Adapter)
    adapter._verify_agent_docker_boundary = AsyncMock(
        return_value={"boundary_held": False, "probe": "SPAWN=ESCAPED"}
    )
    adapter.sandy = SimpleNamespace(
        execute_command=AsyncMock(
            side_effect=[
                {"exit_code": 0},
                {"exit_code": 0},
                {"exit_code": 0, "stdout": "HOSTS=1\nFETCH=CURLFAIL\n"},
                {"exit_code": 0, "stdout": "HOSTS=1\nFETCH=CURLFAIL\n"},
            ]
        )
    )

    verdict = await adapter._seal_network("sandbox", "task")

    assert verdict["sandbox_blocked"] is True
    assert verdict["container_blocked"] is True
    assert verdict["sealed"] is False


def _gateway(container) -> TaskGateway:
    gateway = TaskGateway.__new__(TaskGateway)
    gateway.token = "token"
    gateway.container_id = "task-id"
    gateway.client = SimpleNamespace(
        containers=SimpleNamespace(get=MagicMock(return_value=container))
    )
    return gateway


def test_task_gateway_allows_only_exact_task_container_exec() -> None:
    result = SimpleNamespace(exit_code=0, output=(b"ok\n", b""))
    container = SimpleNamespace(
        id="task-id",
        name="task-name",
        attrs={"Mounts": []},
        exec_run=MagicMock(return_value=result),
    )
    gateway = _gateway(container)

    response = gateway.handle(
        {
            "token": "token",
            "container": "task-id",
            "operation": "exec",
            "argv": ["true"],
        }
    )

    assert response["exit_code"] == 0
    assert base64.b64decode(response["stdout_b64"]) == b"ok\n"
    with pytest.raises(GatewayError, match="operation denied"):
        gateway.handle(
            {
                "token": "token",
                "container": "task-id",
                "operation": "run",
            }
        )
    with pytest.raises(GatewayError, match="container denied"):
        gateway.handle(
            {
                "token": "token",
                "container": "worker-id",
                "operation": "exec",
                "argv": ["true"],
            }
        )


def test_task_gateway_refuses_task_with_docker_socket_mount() -> None:
    container = SimpleNamespace(
        id="task-id",
        name="task-name",
        attrs={
            "Mounts": [
                {
                    "Source": "/var/run/docker.sock",
                    "Destination": "/hidden/control.sock",
                }
            ]
        },
    )
    gateway = _gateway(container)

    with pytest.raises(GatewayError, match="Docker socket"):
        gateway.handle(
            {
                "token": "token",
                "container": "task-id",
                "operation": "exec",
                "argv": ["true"],
            }
        )


def test_agent_wrapper_has_no_container_creation_operation() -> None:
    wrapper = build_agent_docker_wrapper(
        endpoint="http://gateway/v1/docker",
        token="secret",
        container_id="task-id",
        container_name="task-name",
    )

    assert 'if operation != "exec"' in wrapper
    assert "the raw Docker socket is not mounted" in wrapper
    assert 'payload.update(token=CONFIG["token"], container=CONFIG["container_id"])' in wrapper


@pytest.mark.asyncio
async def test_gateway_helpers_ignore_worker_image_entrypoint(monkeypatch) -> None:
    adapter = TerminalBench21Adapter.__new__(TerminalBench21Adapter)
    adapter.sandy = SimpleNamespace(
        execute_command=AsyncMock(
            side_effect=[
                {"exit_code": 0},
                {"exit_code": 0, "stdout": "200\n"},
            ]
        )
    )
    sandbox = SimpleNamespace(id="sandbox-container")
    sandbox.attrs = {"Mounts": []}
    task = SimpleNamespace(
        id="task-container-id",
        attrs={
            "Config": {"Labels": {"chutes.bench.sandbox_id": "sandbox"}},
            "Mounts": [],
        },
    )
    gateway = SimpleNamespace(
        id="gateway-id",
        attrs={
            "NetworkSettings": {
                "Networks": {"bridge": {"IPAddress": "172.17.0.9"}}
            }
        },
        reload=MagicMock(),
        remove=MagicMock(),
    )
    containers = MagicMock()
    containers.get.side_effect = [
        task,
        terminal_bench.docker.errors.NotFound("absent"),
    ]
    containers.run.return_value = gateway
    monkeypatch.setattr(
        terminal_bench.docker,
        "from_env",
        lambda: SimpleNamespace(containers=containers),
    )
    monkeypatch.setattr(terminal_bench, "sandbox_container", lambda _id: sandbox)
    monkeypatch.setattr(terminal_bench, "worker_image_id", lambda: "worker-image")

    result = await adapter._start_task_gateway("sandbox", "task")

    assert result["raw_socket_absent_at_creation"] is True
    assert result["shared_cache_absent_at_creation"] is True
    assert containers.run.call_args_list[0].kwargs["entrypoint"] == ""


@pytest.mark.asyncio
async def test_gateway_refuses_sandbox_with_sensitive_host_mount(monkeypatch) -> None:
    adapter = TerminalBench21Adapter.__new__(TerminalBench21Adapter)
    sandbox = SimpleNamespace(
        attrs={
            "Mounts": [
                {
                    "Source": "/var/lib/sandy/cache",
                    "Destination": "/var/cache/sandy",
                }
            ]
        }
    )
    task = SimpleNamespace(id="task-container-id")
    containers = MagicMock()
    containers.get.return_value = task
    monkeypatch.setattr(
        terminal_bench.docker,
        "from_env",
        lambda: SimpleNamespace(containers=containers),
    )
    monkeypatch.setattr(terminal_bench, "sandbox_container", lambda _id: sandbox)

    with pytest.raises(
        RuntimeError,
        match="sandbox exposes the Docker socket or shared Sandy cache",
    ):
        await adapter._start_task_gateway("sandbox", "task")

    containers.run.assert_not_called()
