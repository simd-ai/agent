# simd_agent/sandbox_client.py
"""Async HTTP client for interacting with the SIMD Sandbox execution service."""

import asyncio
import logging
from typing import Any

import httpx

from simd_agent.models import (
    SandboxArtifact,
    SandboxArtifactsResponse,
    SandboxState,
    SandboxStatus,
    SandboxSubmitResponse,
)
from simd_agent.settings import get_settings

logger = logging.getLogger(__name__)


class SandboxError(Exception):
    """Error interacting with sandbox."""
    pass


class SandboxClient:
    """Async HTTP client for the SIMD Sandbox service.
    
    Endpoints are centralized here for easy modification.
    """
    
    # Endpoint paths (centralized for easy changes)
    ENDPOINT_RUNS = "/v1/runs"
    ENDPOINT_RUN_STATUS = "/v1/runs/{run_id}/status"
    ENDPOINT_RUN_LOGS = "/v1/runs/{run_id}/logs"
    ENDPOINT_RUN_ARTIFACTS = "/v1/runs/{run_id}/artifacts"
    
    def __init__(
        self,
        base_url: str | None = None,
        timeout: int | None = None,
    ):
        """Initialize the sandbox client.
        
        Args:
            base_url: Override base URL (uses settings if not provided)
            timeout: Request timeout in seconds
        """
        settings = get_settings()
        self.base_url = (base_url or settings.sandbox_base_url).rstrip("/")
        self.timeout = timeout or settings.sandbox_timeout
        self.poll_interval = settings.sandbox_poll_interval
        
        self._client: httpx.AsyncClient | None = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                headers={"Accept": "application/json"},
            )
        return self._client
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
    
    async def submit_run(
        self,
        case_zip: bytes,
        run_script: str = "run.sh",
        metadata: dict[str, Any] | None = None,
    ) -> SandboxSubmitResponse:
        """Submit a case zip for execution.
        
        Args:
            case_zip: The case folder as a zip file bytes
            run_script: Script to execute (relative to case root)
            metadata: Optional metadata to attach to the run
            
        Returns:
            SandboxSubmitResponse with run_id
        """
        client = await self._get_client()
        
        files = {
            "file": ("case.zip", case_zip, "application/zip"),
        }
        data = {
            "run_script": run_script,
        }
        if metadata:
            data["metadata"] = str(metadata)
        
        try:
            response = await client.post(
                self.ENDPOINT_RUNS,
                files=files,
                data=data,
            )
            response.raise_for_status()
            result = response.json()
            return SandboxSubmitResponse(run_id=result["run_id"])
        except httpx.HTTPStatusError as e:
            logger.error(f"Sandbox submit failed: {e.response.status_code} - {e.response.text}")
            raise SandboxError(f"Failed to submit run: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Sandbox submit error: {e}")
            raise SandboxError(f"Failed to submit run: {e}")
    
    async def get_status(self, run_id: str) -> SandboxStatus:
        """Get the status of a sandbox run.
        
        Args:
            run_id: The sandbox run ID
            
        Returns:
            SandboxStatus with current state
        """
        client = await self._get_client()
        url = self.ENDPOINT_RUN_STATUS.format(run_id=run_id)
        
        try:
            response = await client.get(url)
            response.raise_for_status()
            result = response.json()
            
            return SandboxStatus(
                state=SandboxState(result["state"]),
                exit_code=result.get("exit_code"),
                started_at=result.get("started_at"),
                finished_at=result.get("finished_at"),
                duration_seconds=result.get("duration_seconds"),
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Sandbox status failed: {e.response.status_code}")
            raise SandboxError(f"Failed to get status: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Sandbox status error: {e}")
            raise SandboxError(f"Failed to get status: {e}")
    
    async def get_logs(
        self,
        run_id: str,
        tail: int | None = None,
    ) -> str:
        """Get execution logs for a sandbox run.
        
        Args:
            run_id: The sandbox run ID
            tail: Optional number of lines from the end
            
        Returns:
            Log text
        """
        client = await self._get_client()
        url = self.ENDPOINT_RUN_LOGS.format(run_id=run_id)
        params = {}
        if tail is not None:
            params["tail"] = tail
        
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            result = response.json()
            return result.get("text", "")
        except httpx.HTTPStatusError as e:
            logger.error(f"Sandbox logs failed: {e.response.status_code}")
            raise SandboxError(f"Failed to get logs: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Sandbox logs error: {e}")
            raise SandboxError(f"Failed to get logs: {e}")
    
    async def get_artifacts(self, run_id: str) -> SandboxArtifactsResponse:
        """Get artifacts produced by a sandbox run.
        
        Args:
            run_id: The sandbox run ID
            
        Returns:
            SandboxArtifactsResponse with list of artifacts
        """
        client = await self._get_client()
        url = self.ENDPOINT_RUN_ARTIFACTS.format(run_id=run_id)
        
        try:
            response = await client.get(url)
            response.raise_for_status()
            result = response.json()
            
            artifacts = [
                SandboxArtifact(
                    name=a["name"],
                    path=a["path"],
                    size_bytes=a.get("size_bytes", 0),
                    download_url=a.get("download_url"),
                )
                for a in result.get("artifacts", [])
            ]
            
            return SandboxArtifactsResponse(artifacts=artifacts)
        except httpx.HTTPStatusError as e:
            logger.error(f"Sandbox artifacts failed: {e.response.status_code}")
            raise SandboxError(f"Failed to get artifacts: {e.response.status_code}")
        except Exception as e:
            logger.error(f"Sandbox artifacts error: {e}")
            raise SandboxError(f"Failed to get artifacts: {e}")
    
    async def wait_for_completion(
        self,
        run_id: str,
        poll_interval: float | None = None,
        timeout: float | None = None,
        on_status: callable | None = None,
    ) -> SandboxStatus:
        """Wait for a sandbox run to complete.
        
        Args:
            run_id: The sandbox run ID
            poll_interval: Polling interval in seconds
            timeout: Maximum wait time in seconds
            on_status: Optional callback for status updates
            
        Returns:
            Final SandboxStatus
        """
        interval = poll_interval or self.poll_interval
        max_time = timeout or self.timeout
        elapsed = 0.0
        
        while elapsed < max_time:
            status = await self.get_status(run_id)
            
            if on_status:
                try:
                    await on_status(status)
                except Exception as e:
                    logger.warning(f"Status callback error: {e}")
            
            if status.state in (SandboxState.SUCCEEDED, SandboxState.FAILED):
                return status
            
            await asyncio.sleep(interval)
            elapsed += interval
        
        raise SandboxError(f"Timeout waiting for run {run_id} after {max_time}s")
    
    async def run_and_wait(
        self,
        case_zip: bytes,
        run_script: str = "run.sh",
        metadata: dict[str, Any] | None = None,
        on_status: callable | None = None,
    ) -> tuple[str, SandboxStatus, str]:
        """Submit a run and wait for completion.
        
        Args:
            case_zip: Case zip bytes
            run_script: Script to execute
            metadata: Optional metadata
            on_status: Optional status callback
            
        Returns:
            Tuple of (run_id, final_status, logs)
        """
        submit_result = await self.submit_run(case_zip, run_script, metadata)
        run_id = submit_result.run_id
        
        final_status = await self.wait_for_completion(run_id, on_status=on_status)
        logs = await self.get_logs(run_id)
        
        return run_id, final_status, logs
