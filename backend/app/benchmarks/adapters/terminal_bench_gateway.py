"""Task-scoped Docker exec gateway for Terminal-Bench agents.

This process runs in a short-lived helper container outside the Sandy sandbox.
It owns the Docker socket; the agent never does.  The bearer token is not an
authority boundary by itself (the wrapper supplied to the agent necessarily
contains it).  The authority boundary is the server-side allow-list: one
immutable container id and a deliberately tiny operation set.
"""
from __future__ import annotations

import base64
import hmac
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import docker

MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_OUTPUT_BYTES = 32 * 1024 * 1024
SENSITIVE_HOST_MOUNT_ROOTS = tuple(
    os.path.realpath(path)
    for path in ("/var/run/docker.sock", "/var/lib/sandy/cache")
)


class GatewayError(RuntimeError):
    """A denied or malformed gateway request."""


def build_agent_docker_wrapper(
    *, endpoint: str, token: str, container_id: str, container_name: str
) -> str:
    """Return the Docker-compatible, task-scoped client installed for the agent."""
    config = json.dumps(
        {
            "endpoint": endpoint,
            "token": token,
            "container_id": container_id,
            "container_name": container_name,
        },
        separators=(",", ":"),
    )
    return f'''#!/usr/bin/env python3
import base64
import json
import sys
import urllib.error
import urllib.request

CONFIG = json.loads({config!r})

def denied(message):
    sys.stderr.write("DENIED: " + message + "\\n")
    raise SystemExit(126)

def request(payload):
    payload.update(token=CONFIG["token"], container=CONFIG["container_id"])
    req = urllib.request.Request(
        CONFIG["endpoint"],
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={{"Content-Type": "application/json"}},
    )
    try:
        with urllib.request.urlopen(req, timeout=86400) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read()
    except Exception as exc:
        sys.stderr.write("GATEWAY_ERROR: %s\\n" % exc)
        raise SystemExit(125)
    result = json.loads(body)
    stdout = base64.b64decode(result.get("stdout_b64") or "")
    stderr = base64.b64decode(result.get("stderr_b64") or "")
    if not stdout and result.get("stdout"):
        stdout = str(result["stdout"]).encode()
    if not stderr and result.get("stderr"):
        stderr = str(result["stderr"]).encode()
    sys.stdout.buffer.write(stdout)
    sys.stderr.buffer.write(stderr)
    raise SystemExit(int(result.get("exit_code", 125)))

args = sys.argv[1:]
if not args:
    denied("only task-scoped docker exec/ps/inspect/logs are available")
operation = args.pop(0)
if operation == "ps":
    request({{"operation": "ps"}})
if operation in ("inspect", "logs"):
    while args and args[0].startswith("-"):
        args.pop(0)
    if len(args) != 1 or args[0] not in (CONFIG["container_id"], CONFIG["container_name"]):
        denied("container is outside this task")
    request({{"operation": operation}})
if operation != "exec":
    denied("docker %s is unavailable; the raw Docker socket is not mounted" % operation)

workdir = None
user = None
environment = []
while args and args[0].startswith("-"):
    flag = args.pop(0)
    if flag in ("-i", "-t", "-it", "-ti"):
        continue
    if flag in ("-w", "--workdir", "-u", "--user", "-e", "--env"):
        if not args:
            denied("missing value for %s" % flag)
        value = args.pop(0)
        if flag in ("-w", "--workdir"):
            workdir = value
        elif flag in ("-u", "--user"):
            user = value
        else:
            environment.append(value)
        continue
    denied("unsupported exec flag %s" % flag)
if not args or args.pop(0) not in (CONFIG["container_id"], CONFIG["container_name"]):
    denied("container is outside this task")
if not args:
    denied("docker exec requires a command")
request({{
    "operation": "exec",
    "argv": args,
    "workdir": workdir,
    "user": user,
    "environment": environment,
}})
'''


def _required_env(name: str) -> str:
    value = (os.getenv(name) or "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


class TaskGateway:
    def __init__(self) -> None:
        self.token = _required_env("TB_GATEWAY_TOKEN")
        self.container_id = _required_env("TB_GATEWAY_CONTAINER_ID")
        self.client = docker.from_env()

    def _container(self):
        container = self.client.containers.get(self.container_id)
        if container.id != self.container_id:
            raise GatewayError("task container identity changed")
        for mount in container.attrs.get("Mounts") or []:
            source = os.path.realpath(str(mount.get("Source") or ""))
            if any(
                source == root or source.startswith(f"{root}/")
                for root in SENSITIVE_HOST_MOUNT_ROOTS
            ):
                raise GatewayError(
                    "task container exposes the Docker socket or shared Sandy cache"
                )
        return container

    def handle(self, payload: dict[str, Any]) -> dict[str, Any]:
        token = str(payload.get("token") or "")
        if not hmac.compare_digest(token, self.token):
            raise GatewayError("unauthorized")
        requested_container = str(payload.get("container") or "")
        if requested_container != self.container_id:
            raise GatewayError("container denied")

        operation = payload.get("operation")
        container = self._container()
        if operation == "ps":
            container.reload()
            return {
                "exit_code": 0,
                "stdout": f"{container.id[:12]}\t{container.name}\t{container.status}\n",
                "stderr": "",
            }
        if operation == "inspect":
            container.reload()
            safe = {
                "Id": container.id,
                "Name": container.name,
                "Image": container.attrs.get("Image"),
                "State": {"Status": (container.attrs.get("State") or {}).get("Status")},
            }
            return {"exit_code": 0, "stdout": json.dumps(safe) + "\n", "stderr": ""}
        if operation == "logs":
            output = container.logs(tail=200)
            return {
                "exit_code": 0,
                "stdout_b64": base64.b64encode(output[:MAX_OUTPUT_BYTES]).decode("ascii"),
                "stderr_b64": "",
            }
        if operation != "exec":
            raise GatewayError(f"operation denied: {operation}")

        argv = payload.get("argv")
        if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
            raise GatewayError("exec argv must be a non-empty string list")
        if sum(len(part.encode("utf-8")) for part in argv) > MAX_REQUEST_BYTES:
            raise GatewayError("exec argv is too large")
        environment = payload.get("environment") or []
        if not isinstance(environment, list) or not all(isinstance(x, str) for x in environment):
            raise GatewayError("environment must be a string list")
        workdir = payload.get("workdir")
        user = payload.get("user")
        result = container.exec_run(
            argv,
            stdout=True,
            stderr=True,
            demux=True,
            workdir=str(workdir) if workdir else None,
            environment=environment or None,
            user=str(user) if user else "",
        )
        stdout, stderr = result.output or (b"", b"")
        stdout = stdout or b""
        stderr = stderr or b""
        if len(stdout) + len(stderr) > MAX_OUTPUT_BYTES:
            raise GatewayError("exec output exceeded the gateway limit")
        return {
            "exit_code": int(result.exit_code),
            "stdout_b64": base64.b64encode(stdout).decode("ascii"),
            "stderr_b64": base64.b64encode(stderr).decode("ascii"),
        }


def main() -> None:
    gateway = TaskGateway()
    port = int(os.getenv("TB_GATEWAY_PORT") or "8765")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path != "/healthz":
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}\n')

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/docker":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length") or "0")
                if length <= 0 or length > MAX_REQUEST_BYTES:
                    raise GatewayError("invalid request size")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise GatewayError("request must be an object")
                response = gateway.handle(payload)
                status = 200
            except GatewayError as exc:
                response = {"exit_code": 126, "stdout": "", "stderr": f"DENIED: {exc}\n"}
                status = 403
            except Exception as exc:  # fail closed without leaking Docker details
                response = {"exit_code": 125, "stdout": "", "stderr": f"GATEWAY_ERROR: {exc}\n"}
                status = 500
            encoded = json.dumps(response, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, _format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
