# simd_agent/simulation_server_client.py
"""Async HTTP client for interacting with the SIMD Simulation Runner server.

The simulation server is an external FastAPI + OpenFOAM service that:
1. Receives a ZIP of OpenFOAM case files
2. Extracts and runs them
3. Streams events via SSE (Server-Sent Events)
4. Returns artifacts when done

Endpoints:
- POST /api/run           - Submit a case ZIP (full run)
- POST /api/run/test      - Submit for test run (1 iteration)
- GET  /api/run/{run_id}/status  - Current run status
- GET  /api/run/{run_id}/events  - SSE stream of real-time events
- GET  /api/run/{run_id}/artifacts - List result files
- GET  /health            - Health check
"""

import asyncio
import json
import logging
from enum import Enum
from typing import Any, AsyncIterator, Callable, Awaitable

import httpx

from pydantic import BaseModel, Field
from simd_agent.settings import get_settings

logger = logging.getLogger(__name__)


# --- Models ---

class SimRunStatus(str, Enum):
    """Simulation run status."""
    PENDING = "pending"
    EXTRACTING = "extracting"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class SimRunMode(str, Enum):
    """Simulation run mode."""
    TEST = "test"    # 1 iteration only - validates case
    FULL = "full"    # Full simulation run


class SimRunEvent(BaseModel):
    """Event from the simulation server SSE stream."""
    run_id: str
    seq: int
    ts: str
    type: str  # extract_started, run_started, run_progress, run_log, run_succeeded, run_failed, artifacts_ready
    level: str  # info, warn, error
    message: str
    payload: dict = Field(default_factory=dict)


class SimRunInfo(BaseModel):
    """Simulation run info."""
    run_id: str
    status: SimRunStatus
    mode: SimRunMode
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    exit_code: int | None = None
    error: str | None = None
    duration_seconds: float | None = None


class SimArtifact(BaseModel):
    """Simulation artifact."""
    name: str
    path: str
    size_bytes: int = 0
    download_url: str | None = None


class SimSubmitResponse(BaseModel):
    """Response from submitting a simulation run."""
    run_id: str
    status: str
    mode: str
    events_url: str
    status_url: str


class SimulationServerError(Exception):
    """Error interacting with simulation server."""
    pass


class SimulationServerClient:
    """Async HTTP client for the SIMD Simulation Runner server.
    
    This replaces the sandbox client for running OpenFOAM simulations.
    The simulation server uses SSE for real-time event streaming.
    """
    
    # Default simulation server URL (can be overridden in settings)
    DEFAULT_URL = "https://vernie-unpreservable-supermentally.ngrok-free.dev"
    
    def __init__(
        self,
        base_url: str | None = None,
        timeout: int = 300,
    ):
        """Initialize the simulation server client.
        
        Args:
            base_url: Override base URL (uses settings or default if not provided)
            timeout: Request timeout in seconds
        """
        settings = get_settings()
        # Use provided URL, or settings, or default
        self.base_url = (
            base_url or 
            getattr(settings, 'simulation_server_url', None) or 
            self.DEFAULT_URL
        ).rstrip("/")
        self.timeout = timeout
        
        self._client: httpx.AsyncClient | None = None
        logger.info(f"[SIM_SERVER] Initialized with base URL: {self.base_url}")
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                headers={
                    "Accept": "application/json",
                    # ngrok requires this header
                    "ngrok-skip-browser-warning": "true",
                },
            )
        return self._client
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
    
    async def health_check(self) -> dict[str, Any]:
        """Check if the simulation server is healthy.
        
        Returns:
            Health status dict with openfoam_available, active_runs, etc.
        """
        client = await self._get_client()
        
        try:
            response = await client.get("/health")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"[SIM_SERVER] Health check failed: {e.response.status_code}")
            raise SimulationServerError(f"Health check failed: {e.response.status_code}")
        except Exception as e:
            logger.error(f"[SIM_SERVER] Health check error: {e}")
            raise SimulationServerError(f"Health check failed: {e}")
    
    async def submit_run(
        self,
        case_zip: bytes,
        mode: SimRunMode = SimRunMode.FULL,
        run_id: str | None = None,
        callback_url: str | None = None,
        n_cores: int = 1,
    ) -> SimSubmitResponse:
        """Submit a case ZIP for execution.
        
        Args:
            case_zip: The OpenFOAM case folder as a zip file bytes
            mode: Run mode - TEST (1 iteration) or FULL
            run_id: Optional run ID (generated if not provided)
            callback_url: Optional URL to POST final status to
            n_cores: Number of MPI processes. 1 = serial (default). >1 = parallel.
            
        Returns:
            SimSubmitResponse with run_id and status URLs
        """
        client = await self._get_client()
        
        files = {
            "case_zip": ("case.zip", case_zip, "application/zip"),
        }
        data = {
            "mode": mode.value,
            "n_cores": str(n_cores),
        }
        if run_id:
            data["run_id"] = run_id
        if callback_url:
            data["callback_url"] = callback_url
        
        endpoint = "/api/run/test" if mode == SimRunMode.TEST else "/api/run"
        
        try:
            logger.info(f"[SIM_SERVER] Submitting run to {endpoint} (mode={mode.value})")
            response = await client.post(endpoint, files=files, data=data)
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"[SIM_SERVER] Run submitted: {result.get('run_id')}")
            return SimSubmitResponse(**result)
            
        except httpx.HTTPStatusError as e:
            error_text = e.response.text[:500] if e.response.text else str(e)
            logger.error(f"[SIM_SERVER] Submit failed: {e.response.status_code} - {error_text}")
            raise SimulationServerError(f"Failed to submit run: {e.response.status_code} - {error_text}")
        except Exception as e:
            logger.error(f"[SIM_SERVER] Submit error: {e}")
            raise SimulationServerError(f"Failed to submit run: {e}")
    
    async def submit_test_run(
        self,
        case_zip: bytes,
        run_id: str | None = None,
        n_cores: int = 1,
    ) -> SimSubmitResponse:
        """Shortcut for submitting a test run (1 iteration validation).
        
        Args:
            case_zip: The OpenFOAM case folder as a zip file bytes
            run_id: Optional run ID
            n_cores: Number of MPI processes (passed through to submit_run)
            
        Returns:
            SimSubmitResponse
        """
        return await self.submit_run(case_zip, mode=SimRunMode.TEST, run_id=run_id, n_cores=n_cores)
    
    async def get_status(self, run_id: str) -> SimRunInfo:
        """Get the current status of a simulation run.

        Args:
            run_id: The simulation run ID

        Returns:
            SimRunInfo with current status
        """
        client = await self._get_client()
        
        try:
            response = await client.get(f"/api/run/{run_id}/status")
            response.raise_for_status()
            result = response.json()
            return SimRunInfo(**result)
            
        except httpx.HTTPStatusError as e:
            logger.error(f"[SIM_SERVER] Status failed: {e.response.status_code}")
            raise SimulationServerError(f"Failed to get status: {e.response.status_code}")
        except Exception as e:
            logger.error(f"[SIM_SERVER] Status error: {e}")
            raise SimulationServerError(f"Failed to get status: {e}")
    
    async def stream_events(
        self,
        run_id: str,
        on_event: Callable[[SimRunEvent], Awaitable[None]] | None = None,
    ) -> AsyncIterator[SimRunEvent]:
        """Stream events from a simulation run via SSE.
        
        Args:
            run_id: The simulation run ID
            on_event: Optional callback for each event
            
        Yields:
            SimRunEvent objects as they arrive
        """
        client = await self._get_client()
        url = f"/api/run/{run_id}/events"
        
        try:
            logger.info(f"[SIM_SERVER] Starting SSE stream for run {run_id}")
            
            async with client.stream("GET", url, timeout=None) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    
                    # Parse SSE data line
                    data_str = line[5:].strip()  # Remove "data: " prefix
                    if not data_str:
                        continue
                    
                    try:
                        data = json.loads(data_str)
                        
                        # Check for stream end signal
                        if data.get("type") == "stream_end":
                            logger.info(f"[SIM_SERVER] SSE stream ended for run {run_id}")
                            break
                        
                        event = SimRunEvent(**data)
                        
                        if on_event:
                            await on_event(event)
                        
                        yield event
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"[SIM_SERVER] Failed to parse SSE event: {e}")
                        continue
                        
        except httpx.HTTPStatusError as e:
            logger.error(f"[SIM_SERVER] SSE stream failed: {e.response.status_code}")
            raise SimulationServerError(f"Failed to stream events: {e.response.status_code}")
        except Exception as e:
            logger.error(f"[SIM_SERVER] SSE stream error: {e}")
            raise SimulationServerError(f"Failed to stream events: {e}")
    
    async def get_artifacts(self, run_id: str) -> list[SimArtifact]:
        """Get artifacts produced by a simulation run.
        
        Args:
            run_id: The simulation run ID
            
        Returns:
            List of SimArtifact objects
        """
        client = await self._get_client()
        
        try:
            response = await client.get(f"/api/run/{run_id}/artifacts")
            response.raise_for_status()
            result = response.json()
            
            artifacts = [
                SimArtifact(**a)
                for a in result.get("artifacts", [])
            ]
            return artifacts
            
        except httpx.HTTPStatusError as e:
            logger.error(f"[SIM_SERVER] Artifacts failed: {e.response.status_code}")
            raise SimulationServerError(f"Failed to get artifacts: {e.response.status_code}")
        except Exception as e:
            logger.error(f"[SIM_SERVER] Artifacts error: {e}")
            raise SimulationServerError(f"Failed to get artifacts: {e}")
    
    async def download_artifact(
        self,
        run_id: str,
        file_path: str,
    ) -> bytes:
        """Download a specific artifact file.
        
        Args:
            run_id: The simulation run ID
            file_path: Path to the artifact within the case
            
        Returns:
            File contents as bytes
        """
        client = await self._get_client()
        
        try:
            response = await client.get(f"/api/run/{run_id}/artifacts/{file_path}")
            response.raise_for_status()
            return response.content
            
        except httpx.HTTPStatusError as e:
            logger.error(f"[SIM_SERVER] Artifact download failed: {e.response.status_code}")
            raise SimulationServerError(f"Failed to download artifact: {e.response.status_code}")
        except Exception as e:
            logger.error(f"[SIM_SERVER] Artifact download error: {e}")
            raise SimulationServerError(f"Failed to download artifact: {e}")
    
    async def get_vtk_results(self, run_id: str) -> dict:
        """Trigger VTK surface generation and return field metadata.

        Calls POST /api/run/{run_id}/vtk-results on the simulation server, which:
        - runs foamToVTK -latestTime
        - merges all boundary patches into one surface.vtp
        - pre-computes vector magnitudes (U_magnitude, etc.)

        Returns a dict matching the frontend contract:
            {
                "run_id": str,
                "vtp_url": str,   # relative URL on the sim server
                "fields": [
                    {"name": str, "num_components": int,
                     "range": [min, max], "location": "point",
                     "vector_source": str | None},
                    ...
                ]
            }
        """
        client = await self._get_client()
        try:
            response = await client.get(
                f"/api/run/{run_id}/vtk-results",
                timeout=180.0,  # foamToVTK can take a while
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"[SIM_SERVER] VTK results failed: {e.response.status_code} {e.response.text}")
            raise SimulationServerError(f"VTK generation failed: {e.response.status_code}")
        except Exception as e:
            logger.error(f"[SIM_SERVER] VTK results error: {e}")
            raise SimulationServerError(f"VTK results error: {e}")

    async def download_surface_vtp(self, run_id: str) -> bytes:
        """Download the merged surface VTP file for a completed run.

        The VTP must already exist (call get_vtk_results first).
        Returns the raw bytes of the VTP file.
        """
        client = await self._get_client()
        try:
            response = await client.get(
                f"/api/run/{run_id}/vtk/surface.vtp",
                timeout=120.0,
            )
            response.raise_for_status()
            return response.content
        except httpx.HTTPStatusError as e:
            raise SimulationServerError(f"VTP download failed: {e.response.status_code}")
        except Exception as e:
            raise SimulationServerError(f"VTP download error: {e}")

    async def get_precomputed_index(self, run_id: str) -> dict:
        """Fetch the precomputed VTP index from the simulation server.

        Calls GET /api/run/{run_id}/vtk-timesteps/index.json.

        The sim server builds this once after run_succeeded (triggered by
        runner.py's background precompute task).  If not yet ready it will
        trigger precompute on-demand and block until done.

        Returns a dict matching the simulation server spec:
            {
                "run_id": str,
                "total": int,
                "fields": [{"name": str, "num_components": int,
                             "range": [min, max], "location": "point",
                             "vector_source": str | null}, ...],
                "timesteps": [
                    {"time": 0.1, "filename": "t_0_1.vtp",
                     "url": "/api/run/{run_id}/vtk-timesteps/t_0_1.vtp"},
                    ...
                ]
            }
        """
        client = await self._get_client()
        try:
            response = await client.get(
                f"/api/run/{run_id}/vtk-timesteps/index.json",
                timeout=300.0,  # precompute can take time on first call
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"[SIM_SERVER] Precomputed index failed: {e.response.status_code} {e.response.text}")
            raise SimulationServerError(f"Precomputed index failed: {e.response.status_code}")
        except Exception as e:
            logger.error(f"[SIM_SERVER] Precomputed index error: {e}")
            raise SimulationServerError(f"Precomputed index error: {e}")

    async def download_precomputed_vtp(self, run_id: str, filename: str) -> bytes:
        """Download a single precomputed timestep VTP by filename.

        Calls GET /api/run/{run_id}/vtk-timesteps/{filename}.
        The filename comes from the index (e.g. "t_0_1.vtp").
        """
        client = await self._get_client()
        try:
            response = await client.get(
                f"/api/run/{run_id}/vtk-timesteps/{filename}",
                timeout=120.0,
            )
            response.raise_for_status()
            return response.content
        except httpx.HTTPStatusError as e:
            raise SimulationServerError(f"Precomputed VTP download failed: {e.response.status_code}")
        except Exception as e:
            raise SimulationServerError(f"Precomputed VTP download error: {e}")

    async def get_timesteps(self, run_id: str) -> dict:
        """Return the list of all simulation timesteps with their VTP URLs.

        Calls GET /api/run/{run_id}/timesteps on the simulation server.

        Returns a dict like:
            {
                "run_id": str,
                "timesteps": [
                    {"time": "0.0", "vtp_url": "/api/run/.../vtk-timestep/0.0/surface.vtp"},
                    ...
                ],
                "fields": [{"name": str, "num_components": int, "range": [...], ...}]
            }
        """
        client = await self._get_client()
        try:
            response = await client.get(
                f"/api/run/{run_id}/timesteps",
                timeout=60.0,
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"[SIM_SERVER] Timesteps failed: {e.response.status_code} {e.response.text}")
            raise SimulationServerError(f"Timesteps request failed: {e.response.status_code}")
        except Exception as e:
            logger.error(f"[SIM_SERVER] Timesteps error: {e}")
            raise SimulationServerError(f"Timesteps error: {e}")

    async def download_timestep_vtp(self, run_id: str, time_str: str) -> bytes:
        """Download the surface VTP for a specific simulation timestep.

        The VTP is built on-demand by the simulation server (cached after first call).
        Returns the raw bytes of the VTP file.
        """
        client = await self._get_client()
        try:
            response = await client.get(
                f"/api/run/{run_id}/vtk-timestep/{time_str}/surface.vtp",
                timeout=120.0,
            )
            response.raise_for_status()
            return response.content
        except httpx.HTTPStatusError as e:
            raise SimulationServerError(f"Timestep VTP download failed: {e.response.status_code}")
        except Exception as e:
            raise SimulationServerError(f"Timestep VTP download error: {e}")

    async def delete_run(self, run_id: str) -> bool:
        """Delete a completed run and its files.
        
        Args:
            run_id: The simulation run ID
            
        Returns:
            True if deleted successfully
        """
        client = await self._get_client()
        
        try:
            response = await client.delete(f"/api/run/{run_id}")
            response.raise_for_status()
            return True
            
        except httpx.HTTPStatusError as e:
            logger.error(f"[SIM_SERVER] Delete failed: {e.response.status_code}")
            return False
        except Exception as e:
            logger.error(f"[SIM_SERVER] Delete error: {e}")
            return False
    
    async def run_test_and_wait(
        self,
        case_zip: bytes,
        on_event: Callable[[SimRunEvent], Awaitable[None]] | None = None,
        run_id: str | None = None,
        n_cores: int = 1,
    ) -> tuple[str, SimRunInfo, list[SimRunEvent]]:
        """Submit a test run and wait for completion, streaming events.
        
        This is the primary method for the "Validate" button flow:
        1. Submit case for TEST mode (1 iteration)
        2. Stream all events
        3. Return final status and all events
        
        Args:
            case_zip: OpenFOAM case as ZIP bytes
            on_event: Optional callback for each event (for relaying to frontend)
            run_id: Optional run ID
            
        Returns:
            Tuple of (run_id, final_status, all_events)
        """
        # Submit test run
        submit_response = await self.submit_test_run(case_zip, run_id=run_id, n_cores=n_cores)
        sim_run_id = submit_response.run_id
        
        logger.info(f"[SIM_SERVER] Test run submitted: {sim_run_id}")
        
        # Collect all events
        all_events: list[SimRunEvent] = []
        
        # Stream events until completion
        async for event in self.stream_events(sim_run_id, on_event=on_event):
            all_events.append(event)
            
            # Check for terminal events
            if event.type in ("run_succeeded", "run_failed"):
                logger.info(f"[SIM_SERVER] Run completed with: {event.type}")
                # Continue to collect artifacts_ready event if any
                continue
            if event.type == "artifacts_ready":
                break
        
        # Get final status
        final_status = await self.get_status(sim_run_id)
        
        return sim_run_id, final_status, all_events
    
    async def run_full_and_stream(
        self,
        case_zip: bytes,
        on_event: Callable[[SimRunEvent], Awaitable[None]] | None = None,
        run_id: str | None = None,
        n_cores: int = 1,
    ) -> tuple[str, SimRunInfo, list[SimRunEvent]]:
        """Submit a full run and stream events until completion.
        
        Similar to run_test_and_wait but for full simulation runs.
        
        Args:
            case_zip: OpenFOAM case as ZIP bytes
            on_event: Callback for each event
            run_id: Optional run ID
            
        Returns:
            Tuple of (run_id, final_status, all_events)
        """
        # Submit full run
        submit_response = await self.submit_run(
            case_zip,
            mode=SimRunMode.FULL,
            run_id=run_id,
            n_cores=n_cores,
        )
        sim_run_id = submit_response.run_id
        
        logger.info(f"[SIM_SERVER] Full run submitted: {sim_run_id}")
        
        # Collect all events
        all_events: list[SimRunEvent] = []
        
        # Stream events until completion
        async for event in self.stream_events(sim_run_id, on_event=on_event):
            all_events.append(event)
            
            if event.type == "artifacts_ready":
                break
            if event.type == "run_failed":
                break
        
        # Get final status
        final_status = await self.get_status(sim_run_id)
        
        return sim_run_id, final_status, all_events
