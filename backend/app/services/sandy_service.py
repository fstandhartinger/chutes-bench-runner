import asyncio
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
        self.headers = {
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
            "Content-Type": "application/json"
        }

    async def create_sandbox(self, enable_docker_socket: bool = False) -> Optional[str]:
        """Create a new sandbox and return its ID.

        Args:
            enable_docker_socket: If True, the sandbox will have access to the Docker socket.
                                  This is required for benchmarks that need to run Docker commands
                                  (e.g., Terminal-Bench, SWE-Bench). Disabled by default for security.
        """
        if not self.api_key:
            logger.error("Sandy API key is not configured")
            return None
        delay_seconds = 1
        last_error: Optional[str] = None
        # Prepare payload with Docker socket option
        payload = {}
        if enable_docker_socket:
            payload["enableDockerSocket"] = True
        for attempt in range(1, 4):
            async with httpx.AsyncClient(timeout=30.0) as client:
                try:
                    response = await client.post(
                        f"{self.base_url}/api/sandboxes",
                        headers=self.headers,
                        json=payload if payload else None,
                    )
                    response.raise_for_status()
                    data = response.json()
                    sandbox_id = data.get("sandboxId")
                    if sandbox_id:
                        return sandbox_id
                    last_error = "Missing sandboxId in response"
                except httpx.HTTPStatusError as e:
                    last_error = e.response.text or str(e)
                except Exception as e:
                    last_error = str(e) or e.__class__.__name__

            if attempt < 3:
                logger.warning(
                    "Retrying sandbox creation",
                    attempt=attempt,
                    error=last_error,
                )
                await asyncio.sleep(delay_seconds)
                delay_seconds = min(delay_seconds * 2, 10)

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
                return {
                    "success": True,
                    "stdout": data.get("stdout", ""),
                    "stderr": data.get("stderr", ""),
                    "exit_code": data.get("exitCode", 0)
                }
            except httpx.HTTPStatusError as e:
                error_detail = e.response.text or str(e)
                logger.error(
                    f"Failed to execute command in sandbox {sandbox_id}",
                    status_code=e.response.status_code,
                    error=error_detail,
                )
                return {"success": False, "error": error_detail, "exit_code": -1}
            except Exception as e:
                logger.error(
                    f"Failed to execute command in sandbox {sandbox_id}",
                    error=str(e) or e.__class__.__name__,
                )
                return {"success": False, "error": str(e), "exit_code": -1}

    async def write_file(self, sandbox_id: str, path: str, content: str) -> bool:
        """Write a file to the sandbox."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/sandboxes/{sandbox_id}/files/write",
                    headers=self.headers,
                    json={"path": path, "content": content}
                )
                response.raise_for_status()
                return True
            except Exception as e:
                logger.error(f"Failed to write file to sandbox {sandbox_id}", error=str(e))
                return False

    async def terminate_sandbox(self, sandbox_id: str) -> bool:
        """Terminate a sandbox."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            try:
                response = await client.post(
                    f"{self.base_url}/api/sandboxes/{sandbox_id}/terminate",
                    headers=self.headers
                )
                response.raise_for_status()
                return True
            except Exception as e:
                logger.error(f"Failed to terminate sandbox {sandbox_id}", error=str(e))
                return False

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
