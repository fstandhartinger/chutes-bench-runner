import asyncio
import json
import httpx
from typing import Any, Optional, Dict, List
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

class SandyService:
    """Service for interacting with the Sandy sandbox API."""

    def __init__(self):
        settings = get_settings()
        self.base_url = settings.sandy_base_url.rstrip("/")
        self.api_key = settings.sandy_api_key
        self.last_error: Optional[str] = None
        self.headers = {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            "Content-Type": "application/json"
        }

    async def create_sandbox(
        self,
        enable_docker_socket: bool = False,
        priority: int = 3,  # LOW priority for batch benchmark jobs
        preemptable: bool = True,  # Can be terminated under memory pressure
    ) -> Optional[str]:
        """Create a new sandbox and return its ID.

        Args:
            enable_docker_socket: If True, the sandbox will have access to the Docker socket.
                                  This is required for benchmarks that need to run Docker commands
                                  (e.g., Terminal-Bench, SWE-Bench). Disabled by default for security.
            priority: Sandbox priority level (0=CRITICAL, 1=HIGH, 2=NORMAL, 3=LOW).
                      Defaults to 3 (LOW) for batch benchmark jobs.
            preemptable: If True, sandbox can be terminated under memory pressure.
                         Defaults to True for benchmark sandboxes.
        """
        if not self.api_key:
            logger.error("Sandy API key is not configured")
            return None
        delay_seconds = 1
        last_error: Optional[str] = None
        self.last_error = None
        # Prepare payload with Docker socket option and priority settings
        payload: Dict[str, Any] = {
            "priority": priority,
            "preemptable": preemptable,
        }
        if enable_docker_socket:
            payload["enableDockerSocket"] = True
        timeout = httpx.Timeout(90.0, connect=10.0)
        for attempt in range(1, 4):
            async with httpx.AsyncClient(timeout=timeout) as client:
                try:
                    response = await client.post(
                        f"{self.base_url}/api/sandboxes",
                        headers=self.headers,
                        json=payload,
                    )
                    response.raise_for_status()
                    data = response.json()
                    sandbox_id = data.get("sandboxId")
                    if sandbox_id:
                        self.last_error = None
                        return sandbox_id
                    last_error = "Missing sandboxId in response"
                except httpx.HTTPStatusError as e:
                    detail = e.response.text or e.response.reason_phrase or str(e)
                    last_error = f"HTTP {e.response.status_code}: {detail}".strip()
                except Exception as e:
                    last_error = str(e) or e.__class__.__name__
                if last_error and len(last_error) > 500:
                    last_error = last_error[:500].rstrip() + "…"

            if attempt < 3:
                logger.warning(
                    "Retrying sandbox creation",
                    attempt=attempt,
                    error=last_error,
                )
                await asyncio.sleep(delay_seconds)
                delay_seconds = min(delay_seconds * 2, 10)

        if last_error:
            self.last_error = last_error
        logger.error("Failed to create sandbox", error=last_error)
        return None

    async def execute_command(
        self,
        sandbox_id: str,
        command: str,
        cwd: Optional[str] = None,
        env: Optional[dict[str, str]] = None,
        timeout_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute a command in the sandbox."""
        timeout_seconds = 60.0
        if timeout_ms is not None:
            timeout_seconds = max(timeout_seconds, timeout_ms / 1000 + 30)
        delay_seconds = 1
        last_error: Optional[str] = None
        self.last_error = None
        for attempt in range(1, 3):
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds, connect=10.0)) as client:
                try:
                    payload: dict[str, Any] = {"command": command}
                    if cwd:
                        payload["cwd"] = cwd
                    if env:
                        payload["env"] = env
                    if timeout_ms is not None:
                        payload["timeoutMs"] = timeout_ms
                    response = await client.post(
                        f"{self.base_url}/api/sandboxes/{sandbox_id}/exec",
                        headers=self.headers,
                        json=payload,
                    )
                    response.raise_for_status()
                    data = response.json()
                    self.last_error = None
                    return {
                        "success": True,
                        "stdout": data.get("stdout", ""),
                        "stderr": data.get("stderr", ""),
                        "exit_code": data.get("exitCode", 0),
                    }
                except httpx.HTTPStatusError as e:
                    error_detail = e.response.text or str(e)
                    last_error = error_detail
                    logger.error(
                        f"Failed to execute command in sandbox {sandbox_id}",
                        status_code=e.response.status_code,
                        error=error_detail,
                    )
                    if e.response.status_code not in {408, 429, 500, 502, 503, 504}:
                        break
                except Exception as e:
                    last_error = str(e) or e.__class__.__name__
                    logger.error(
                        f"Failed to execute command in sandbox {sandbox_id}",
                        error=last_error,
                    )
            if attempt < 2:
                logger.warning(
                    "Retrying sandbox exec",
                    attempt=attempt,
                    error=last_error,
                )
                await asyncio.sleep(delay_seconds)
                delay_seconds = min(delay_seconds * 2, 10)
        if last_error:
            self.last_error = last_error if len(last_error) <= 500 else f"{last_error[:500].rstrip()}…"
        return {"success": False, "error": last_error or "sandbox exec failed", "exit_code": -1}

    async def write_file(self, sandbox_id: str, path: str, content: str) -> bool:
        """Write a file to the sandbox."""
        delay_seconds = 1
        last_error: Optional[str] = None
        self.last_error = None
        for attempt in range(1, 4):
            async with httpx.AsyncClient(timeout=30.0) as client:
                try:
                    response = await client.post(
                        f"{self.base_url}/api/sandboxes/{sandbox_id}/files/write",
                        headers=self.headers,
                        json={"path": path, "content": content},
                    )
                    response.raise_for_status()
                    self.last_error = None
                    return True
                except httpx.HTTPStatusError as e:
                    detail = e.response.text or e.response.reason_phrase or str(e)
                    last_error = f"HTTP {e.response.status_code}: {detail}".strip()
                    if e.response.status_code not in {408, 429, 500, 502, 503, 504}:
                        break
                except Exception as e:
                    last_error = str(e) or e.__class__.__name__
            if attempt < 3:
                logger.warning(
                    "Retrying sandbox file write",
                    attempt=attempt,
                    error=last_error,
                )
                await asyncio.sleep(delay_seconds)
                delay_seconds = min(delay_seconds * 2, 10)
        if last_error:
            self.last_error = last_error
        logger.error(f"Failed to write file to sandbox {sandbox_id}", error=last_error)
        return False

    async def terminate_sandbox(self, sandbox_id: str) -> bool:
        """Terminate a sandbox."""
        delay_seconds = 1
        last_error: Optional[str] = None
        for attempt in range(1, 3):
            async with httpx.AsyncClient(timeout=30.0) as client:
                try:
                    response = await client.post(
                        f"{self.base_url}/api/sandboxes/{sandbox_id}/terminate",
                        headers=self.headers,
                    )
                    if response.status_code == 404:
                        return True
                    response.raise_for_status()
                    return True
                except httpx.HTTPStatusError as e:
                    detail = e.response.text or e.response.reason_phrase or str(e)
                    last_error = f"HTTP {e.response.status_code}: {detail}".strip()
                    if e.response.status_code not in {408, 429, 500, 502, 503, 504}:
                        break
                except Exception as e:
                    last_error = str(e) or e.__class__.__name__
            if attempt < 2:
                logger.warning(
                    "Retrying sandbox termination",
                    attempt=attempt,
                    error=last_error,
                )
                await asyncio.sleep(delay_seconds)
                delay_seconds = min(delay_seconds * 2, 5)
        logger.error(f"Failed to terminate sandbox {sandbox_id}", error=last_error)
        return False

    async def get_resources(self) -> Optional[Dict[str, Any]]:
        """Fetch current Sandy resource usage snapshot."""
        if not self.api_key:
            self.last_error = "Sandy API key is not configured"
            return None
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.base_url}/api/resources",
                    headers=self.headers,
                )
                response.raise_for_status()
                return response.json()
        except Exception as exc:
            self.last_error = str(exc) or exc.__class__.__name__
            return None

    async def get_metrics_timeseries(self, hours: int = 12) -> Optional[List[Dict[str, Any]]]:
        """Fetch Sandy telemetry metrics timeseries."""
        if not self.api_key:
            self.last_error = "Sandy API key is not configured"
            return None
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(
                    f"{self.base_url}/api/metrics/timeseries",
                    headers=self.headers,
                    params={"hours": hours},
                )
                response.raise_for_status()
                return response.json()
        except Exception as exc:
            self.last_error = str(exc) or exc.__class__.__name__
            return None

    async def get_sandbox_stats(self, sandbox_ids: Optional[list[str]] = None) -> Optional[List[Dict[str, Any]]]:
        """Fetch per-sandbox stats from Sandy."""
        if not self.api_key:
            self.last_error = "Sandy API key is not configured"
            return None
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                params: Dict[str, Any] = {}
                if sandbox_ids:
                    params["ids"] = ",".join(sandbox_ids)
                response = await client.get(
                    f"{self.base_url}/api/sandboxes/stats",
                    headers=self.headers,
                    params=params,
                )
                response.raise_for_status()
                return response.json()
        except Exception as exc:
            self.last_error = str(exc) or exc.__class__.__name__
            return None

    async def sandbox_exists(self, sandbox_id: str) -> Optional[bool]:
        """Check whether a sandbox exists (None when status is unknown)."""
        if not self.api_key:
            self.last_error = "Sandy API key is not configured"
            return None
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{self.base_url}/api/sandboxes/{sandbox_id}",
                    headers=self.headers,
                )
                if response.status_code == 404:
                    self.last_error = "Sandbox not found"
                    return False
                response.raise_for_status()
                self.last_error = None
                return True
        except Exception as exc:
            self.last_error = str(exc) or exc.__class__.__name__
            return None

    async def iter_agent_events(
        self,
        sandbox_id: str,
        agent: str,
        model: str,
        prompt: str,
        max_duration: int = 600,
    ):
        """Stream agent events from Sandy."""
        if not self.api_key:
            raise RuntimeError("Sandy API key is not configured")
        timeout = httpx.Timeout(None)
        payload = {
            "agent": agent,
            "model": model,
            "prompt": prompt,
            "maxDuration": max_duration,
        }
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/sandboxes/{sandbox_id}/agent/run",
                headers=self.headers,
                json=payload,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("data: "):
                        line = line[len("data: ") :].strip()
                    try:
                        yield json.loads(line)
                    except Exception:
                        continue

    async def run_agent(
        self,
        sandbox_id: str,
        agent: str,
        model: str,
        prompt: str,
        max_duration: int = 600,
    ) -> Dict[str, Any]:
        """Run an agent in the sandbox and return summary + collected events."""
        events: List[Dict[str, Any]] = []
        summary: Dict[str, Any] = {}
        async for event in self.iter_agent_events(
            sandbox_id=sandbox_id,
            agent=agent,
            model=model,
            prompt=prompt,
            max_duration=max_duration,
        ):
            if isinstance(event, dict):
                events.append(event)
                if event.get("type") == "complete":
                    summary = event
        return {"events": events, "summary": summary}

    async def run_python_code(self, code: str, timeout_ms: Optional[int] = None) -> Dict[str, Any]:
        """Convenience method to create a sandbox, run Python code, and terminate."""
        sandbox_id = await self.create_sandbox()
        if not sandbox_id:
            return {"success": False, "error": "Could not create sandbox"}

        try:
            # Write the code to a file
            await self.write_file(sandbox_id, "script.py", code)
            
            # Execute the code
            result = await self.execute_command(
                sandbox_id,
                "python3 script.py",
                timeout_ms=timeout_ms,
            )
            return result
        finally:
            await self.terminate_sandbox(sandbox_id)
