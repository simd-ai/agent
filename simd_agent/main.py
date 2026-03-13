# simd_agent/main.py
"""FastAPI application with WebSocket endpoint for CFD workflow orchestration."""

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

# DB persistence disabled — frontend handles its own storage
# from simd_agent.db import close_db, init_db

# Try to import mesh module (may fail on some Python versions due to VTK compatibility)
MESH_ENABLED = False
try:
    from simd_agent.mesh import mesh_router, STORAGE_DIR
    MESH_ENABLED = True
except ImportError as e:
    logging.warning(f"Mesh module not available: {e}")
    mesh_router = None
    STORAGE_DIR = None
from simd_agent.run.event_bus import EventBus
from simd_agent.run.orchestration import Orchestrator
from simd_agent.run.simulation_server_client import SimulationServerClient, SimulationServerError
from simd_agent.models import (
    EventTypes,
    RunStatus,
    StartRequest,
)
from simd_agent.settings import get_settings
from simd_agent.store import EventStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting simd_agent service...")
    settings = get_settings()
    logger.info(f"Log level: {settings.log_level}")
    
    yield

    # Shutdown
    logger.info("Shutting down simd_agent service...")


# Create FastAPI app
app = FastAPI(
    title="SIMD Agent",
    description="CFD workflow orchestration service via WebSocket",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and mesh routes (if available)
if MESH_ENABLED and STORAGE_DIR:
    app.mount("/static", StaticFiles(directory=str(STORAGE_DIR)), name="static")
    app.include_router(mesh_router)
    logger.info("Mesh converter enabled at /api/mesh/convert")
else:
    logger.warning("Mesh converter disabled (VTK/PyVista not available)")


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "service": "simd_agent"}


@app.get("/")
async def root() -> dict[str, Any]:
    """Root endpoint with service info."""
    endpoints = {
        "websocket": "/ws/run",
        "chat": "/ws/chat",
        "precheck": "/api/precheck",
        "precheck_stream": "/ws/precheck",
        "runs": "/runs/{run_id}",
        "events": "/runs/{run_id}/events",
    }
    if MESH_ENABLED:
        endpoints["mesh_convert"] = "/api/mesh/convert"
    
    return {
        "service": "simd_agent",
        "version": "0.1.0",
        "mesh_enabled": MESH_ENABLED,
        "endpoints": endpoints,
    }


# --- Precheck Endpoint ---

@app.post("/api/precheck")
async def precheck(request: dict[str, Any]):
    """Analyze user prompt and extract simulation specifications.
    
    Uses LLM to parse natural language simulation descriptions and return
    structured configuration suggestions, boundary condition hints, and
    interpretation of the user's intent.
    
    Request (POST /api/precheck):
        {
            "prompt": "Simulate turbulent water flow through a pipe at 5 m/s",
            "has_mesh": true,
            "mesh_info": {
                "mesh_id": "abc-123",
                "file_name": "pipe.msh",
                "patches": [
                    { "name": "inlet", "type": "patch", "n_cells": 100 },
                    { "name": "outlet", "type": "patch", "n_cells": 100 },
                    { "name": "walls", "type": "wall", "n_cells": 5000 }
                ],
                "check_mesh": { "cells": 50000, "faces": 150000, "points": 55000 }
            }
        }
    
    Response:
        {
            "success": true,
            "confidence": 0.9,
            "message": "Detected turbulent pipe flow with water",
            "suggestedConfig": {
                "caseType": "internal_pipe_flow",
                "flowRegime": "turbulent",
                "timeScheme": "steady",
                "compressibility": "incompressible",
                "enableHeatTransfer": false,
                "gravity": false,
                "solver": { ... },
                "fluid": { ... },
                "turbulence": { ... },
                "boundaryConditions": { ... }
            },
            "boundaryHints": { ... },
            "interpretation": { ... },
            "confidenceScores": { ... }
        }
    """
    from simd_agent.precheck import get_precheck_service
    from simd_agent.precheck import (
        PrecheckRequest,
        PrecheckResponse,
        SuggestedConfig,
        SolverSettings,
        TurbulenceSettings,
        Interpretation,
        ConfidenceScores,
        FLUID_PRESETS,
    )
    
    try:
        # Parse and validate request
        precheck_request = PrecheckRequest(**request)

        # Human-friendly prompt validation (empty prompt without mesh, etc.)
        validation_error = precheck_request.validate_prompt()
        if validation_error:
            service = get_precheck_service()
            return service._create_friendly_error_response(validation_error).model_dump(by_alias=True)

        # Get precheck service and analyze
        service = get_precheck_service()
        response = await service.analyze(precheck_request)
        
        # Return with camelCase keys to match frontend expectations
        return response.model_dump(by_alias=True)
        
    except ValidationError as e:
        logger.warning(f"Invalid precheck request: {e}")
        # Return a minimal valid fallback response
        fallback = PrecheckResponse(
            success=False,
            confidence=0.0,
            message=f"Request validation failed: {e}",
            suggested_config=SuggestedConfig(
                case_type="general",
                flow_regime="turbulent",
                time_scheme="steady",
                compressibility="incompressible",
                enable_heat_transfer=False,
                gravity=False,
                solver=SolverSettings(),
                fluid=FLUID_PRESETS["air"],
                turbulence=TurbulenceSettings(model="kOmegaSST"),
                boundary_conditions={},
            ),
            interpretation=Interpretation(
                summary="Request validation failed",
                simulation_type="Unknown",
                key_physics=[],
                assumptions=[],
            ),
            confidence_scores=ConfidenceScores(
                overall=0.0,
                flow_regime=0.0,
                boundary_conditions=0.0,
                physics_settings=0.0,
            ),
            errors=[f"Invalid request: {e}"],
        )
        return fallback.model_dump(by_alias=True)
        
    except Exception as e:
        logger.exception(f"Precheck failed: {e}")
        # Return a minimal valid fallback response
        fallback = PrecheckResponse(
            success=False,
            confidence=0.0,
            message=f"Internal error: {e}",
            suggested_config=SuggestedConfig(
                case_type="general",
                flow_regime="turbulent",
                time_scheme="steady",
                compressibility="incompressible",
                enable_heat_transfer=False,
                gravity=False,
                solver=SolverSettings(),
                fluid=FLUID_PRESETS["air"],
                turbulence=TurbulenceSettings(model="kOmegaSST"),
                boundary_conditions={},
            ),
            interpretation=Interpretation(
                summary="Analysis failed due to internal error",
                simulation_type="Unknown",
                key_physics=[],
                assumptions=[],
            ),
            confidence_scores=ConfidenceScores(
                overall=0.0,
                flow_regime=0.0,
                boundary_conditions=0.0,
                physics_settings=0.0,
            ),
            errors=[f"Internal error: {e}"],
        )
        return fallback.model_dump(by_alias=True)


@app.websocket("/ws/precheck")
async def websocket_precheck(websocket: WebSocket):
    """WebSocket endpoint for streaming precheck analysis with thinking.

    Protocol
    --------
    1. Client connects.
    2. Client sends PrecheckRequest JSON as the first message.
    3. Server streams a sequence of typed event messages:

       {"type": "start"}
       {"type": "thought",        "text": "<incremental reasoning text>"}   # streamed live
       {"type": "spec_generating"}                                            # once
       {"type": "spec",           "data": {<PrecheckResponse camelCase>}}    # once
       {"type": "done"}

       On error:
       {"type": "error", "message": "<description>"}
       {"type": "done"}

    4. Server closes connection after "done".
    """
    from simd_agent.precheck import get_precheck_service
    from simd_agent.precheck import PrecheckRequest

    await websocket.accept()
    logger.info("[WS/precheck] New connection accepted")
    print("[WS/precheck] ✓ WebSocket accepted — waiting for request JSON...", flush=True)

    try:
        # ── Receive request (30 s timeout) ───────────────────────────────────
        try:
            data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
            print(f"[WS/precheck] ✓ Request received — prompt={str(data.get('prompt',''))[:80]!r}", flush=True)
        except asyncio.TimeoutError:
            print("[WS/precheck] ✗ Timeout: client never sent request JSON", flush=True)
            await websocket.send_json({"type": "error", "message": "Timeout waiting for request"})
            await websocket.send_json({"type": "done"})
            await websocket.close(code=1008)
            return

        try:
            request = PrecheckRequest(**data)
            print(f"[WS/precheck] ✓ PrecheckRequest parsed OK — has_mesh={request.has_mesh}", flush=True)
        except Exception as e:
            print(f"[WS/precheck] ✗ PrecheckRequest parse failed: {e}", flush=True)
            await websocket.send_json({"type": "error", "message": f"Invalid request: {e}"})
            await websocket.send_json({"type": "done"})
            await websocket.close(code=1003)
            return

        # ── Stream events from the service ───────────────────────────────────
        print("[WS/precheck] → Calling analyze_stream...", flush=True)
        service = get_precheck_service()
        async for event in service.analyze_stream(request):
            etype = event.get("type")
            if etype != "thought":  # skip noisy streaming text
                print(f"[WS/precheck]   event={etype}", flush=True)
            await websocket.send_json(event)
            if etype == "done":
                break
        print("[WS/precheck] ✓ Stream complete", flush=True)

    except WebSocketDisconnect:
        logger.info("[WS/precheck] Client disconnected")
    except Exception as e:
        logger.exception(f"[WS/precheck] Unhandled error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
            await websocket.send_json({"type": "done"})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


# --- Chat WebSocket Endpoint ---

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for the streaming CFD chat assistant.

    Protocol
    --------
    1. Client connects.
    2. Client sends a ChatRequest JSON per user turn.
    3. Server streams typed JSON events per turn:

       {"type": "token",       "delta": "…"}           # streamed text chunks
       {"type": "tool_start",  "tool": "…", "label": "…"}
       {"type": "tool_result", "tool": "…", "data": {…}}
       {"type": "artifact",    "kind": "…", "content": …}
       {"type": "done",        "suggested_actions": ["…"]}
       {"type": "error",       "message": "…"}

    4. Connection stays alive for multiple turns — client sends the next
       ChatRequest whenever the user types a new message.
    """
    from simd_agent.chat import ChatRequest, get_chat_service

    await websocket.accept()
    logger.info("[WS/chat] New connection accepted")

    try:
        service = get_chat_service()
    except Exception as exc:
        logger.error(f"[WS/chat] Failed to initialise ChatService: {exc}")
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.send_json({"type": "done"})
        await websocket.close(code=1011)
        return

    heartbeat_task = asyncio.create_task(
        _heartbeat_loop(websocket, get_settings().ws_heartbeat_interval)
    )

    try:
        while True:
            try:
                data = await websocket.receive_json()
            except WebSocketDisconnect:
                logger.info("[WS/chat] Client disconnected")
                break

            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            try:
                request = ChatRequest(**data)
            except Exception as exc:
                logger.warning(f"[WS/chat] Invalid request: {exc}")
                await websocket.send_json({"type": "error", "message": f"Invalid request: {exc}"})
                await websocket.send_json({"type": "done"})
                continue

            logger.info(
                f"[WS/chat] User turn: sim={request.simulation_id} "
                f"msg={request.message[:80]!r}"
            )

            try:
                async for event in service.handle_turn(request):
                    await websocket.send_json(event)
            except WebSocketDisconnect:
                logger.info("[WS/chat] Client disconnected during streaming")
                break
            except Exception as exc:
                logger.exception(f"[WS/chat] Error during turn: {exc}")
                try:
                    await websocket.send_json({"type": "error", "message": str(exc)})
                    await websocket.send_json({"type": "done"})
                except Exception:
                    break

    except Exception as exc:
        logger.exception(f"[WS/chat] Unhandled error: {exc}")
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        try:
            await websocket.close()
        except Exception:
            pass
        logger.info("[WS/chat] Connection closed")


@app.websocket("/ws/run")
async def websocket_run(websocket: WebSocket):
    """WebSocket endpoint for CFD workflow execution.
    
    Protocol:
    1. Client connects
    2. Client sends StartRequest JSON as first message
    3. Server streams AgentEvent JSON messages
    4. Server sends final event and closes connection
    """
    await websocket.accept()
    
    run_id = uuid4()
    store = EventStore()
    event_bus = None
    
    logger.info("=" * 60)
    logger.info(f"[WS] NEW CONNECTION - Run ID: {run_id}")
    logger.info("=" * 60)
    
    try:
        # Receive start request
        logger.info("[WS] Waiting for start request from client...")
        try:
            data = await asyncio.wait_for(
                websocket.receive_json(),
                timeout=30.0,  # 30 second timeout for initial message
            )
        except asyncio.TimeoutError:
            logger.error("[WS] Timeout waiting for start request")
            await websocket.send_json({
                "error": "Timeout waiting for start request",
                "type": "error",
            })
            await websocket.close(code=1008)
            return
        
        logger.info("[WS] Received start request:")
        logger.info("=" * 60)
        logger.info("[WS] FULL REQUEST PAYLOAD:")
        logger.info(f"[WS]   op: {data.get('op')}")
        logger.info(f"[WS]   provider: {data.get('provider')}")
        logger.info(f"[WS]   prompt_pack: {data.get('prompt_pack')}")
        logger.info(f"[WS]   user_requirements: {data.get('user_requirements', '')}")
        logger.info("[WS]   simulation_config:")
        sim_config = data.get('simulation_config', {})
        for key, value in sim_config.items():
            if isinstance(value, dict):
                logger.info(f"[WS]     {key}: {json.dumps(value, indent=6)}")
            else:
                logger.info(f"[WS]     {key}: {value}")
        constraints = data.get('constraints', {})
        if constraints:
            logger.info(f"[WS]   constraints: {json.dumps(constraints, indent=4)}")
        metadata = data.get('metadata', {})
        if metadata:
            logger.info(f"[WS]   metadata: {json.dumps(metadata, indent=4)}")
        logger.info("=" * 60)
        
        # Parse request
        try:
            request = StartRequest(**data)
            logger.info(f"[WS] StartRequest parsed successfully: op={request.op.value}")
        except ValidationError as e:
            logger.error(f"[WS] Invalid start request: {e}")
            await websocket.send_json({
                "error": f"Invalid start request: {e}",
                "type": "error",
            })
            await websocket.close(code=1003)
            return
        
        # Create event bus
        logger.info("[WS] Creating EventBus...")
        event_bus = EventBus(
            run_id=run_id,
            websocket=websocket,
            store=store,
            persist=True,
        )
        
        # Create run in database
        logger.info("[WS] Creating run in database...")
        try:
            await store.create_run(
                op=request.op,
                provider=request.provider,
                prompt_pack=request.prompt_pack,
                user_requirements=request.user_requirements,
                simulation_config=request.simulation_config,
                run_id=run_id,
                raw_config=data.get('simulation_config', {}),  # Save raw config too
            )
            logger.info(f"[WS] Run created in database: {run_id}")
        except Exception as e:
            logger.error(f"[WS] Failed to create run in database: {e}")
            # Continue anyway - event streaming still works
        
        # Update run status
        logger.info("[WS] Updating run status to RUNNING...")
        try:
            await store.update_run_status(run_id, RunStatus.RUNNING)
            logger.info("[WS] Run status updated to RUNNING")
        except Exception as e:
            logger.warning(f"[WS] Failed to update run status: {e}")
        
        # Create and run orchestrator
        logger.info("[WS] Creating Orchestrator...")
        orchestrator = Orchestrator(
            run_id=run_id,
            event_bus=event_bus,
            store=store,
            request=request,
        )
        
        # Start heartbeat task
        logger.info("[WS] Starting heartbeat task...")
        heartbeat_task = asyncio.create_task(
            _heartbeat_loop(websocket, get_settings().ws_heartbeat_interval)
        )
        
        try:
            # Execute the workflow
            logger.info("[WS] Starting orchestrator.run()...")
            logger.info("-" * 60)
            result = await orchestrator.run()
            logger.info("-" * 60)
            
            # Final event already sent by orchestrator
            logger.info(f"[WS] Run {run_id} completed with status: {result.status}")
            logger.info(f"[WS] Result summary: {result.summary if hasattr(result, 'summary') else 'N/A'}")
            
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for run {run_id}")
        
        # Update run status if possible
        try:
            await store.update_run_status(run_id, RunStatus.FAILED)
        except Exception:
            pass
    
    except Exception as e:
        logger.exception(f"WebSocket error for run {run_id}: {e}")
        
        # Try to send error event
        if event_bus:
            try:
                await event_bus.emit_error(
                    EventTypes.RUN_FAILED,
                    f"Internal error: {e}",
                )
                await event_bus.emit_final(
                    status="failed",
                    error=str(e),
                )
            except Exception:
                pass
        
        # Update database
        try:
            await store.finalize_run(
                run_id=run_id,
                status=RunStatus.FAILED,
                result={"error": str(e)},
            )
        except Exception:
            pass
    
    finally:
        # Close WebSocket
        try:
            await websocket.close()
        except Exception:
            pass


async def _heartbeat_loop(websocket: WebSocket, interval: int):
    """Send periodic pings to keep connection alive."""
    while True:
        try:
            await asyncio.sleep(interval)
            await websocket.send_json({"type": "ping"})
        except Exception:
            break


# Optional: Add a simple REST endpoint for run status lookup
@app.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    """Get run details by ID."""
    from uuid import UUID
    
    try:
        run_uuid = UUID(run_id)
    except ValueError:
        return {"error": "Invalid run ID format"}
    
    store = EventStore()
    run = await store.get_run(run_uuid)
    
    if run is None:
        return {"error": "Run not found"}
    
    return run.model_dump()


@app.get("/runs/{run_id}/events")
async def get_run_events(run_id: str) -> dict[str, Any]:
    """Get events for a run."""
    from uuid import UUID
    
    try:
        run_uuid = UUID(run_id)
    except ValueError:
        return {"error": "Invalid run ID format"}
    
    store = EventStore()
    events = await store.get_events(run_uuid)
    
    return {"events": [e.model_dump() for e in events]}


# ── Local VTK cache ──────────────────────────────────────────────────────────
# VTPs are downloaded from the simulation server ONCE per run and stored here.
# All subsequent frontend requests are served directly from local disk —
# no DB lookup, no HTTP round-trip to the sim server, no RAM buffering.

# Per-run download locks prevent concurrent coroutines from double-downloading.
_vtk_download_locks: dict[str, asyncio.Lock] = {}


def _vtk_local_dir(run_id: str) -> Path:
    """Return the local cache directory for a run's VTP files."""
    return Path(get_settings().vtk_cache_dir) / run_id


def _get_download_lock(run_id: str) -> asyncio.Lock:
    if run_id not in _vtk_download_locks:
        _vtk_download_locks[run_id] = asyncio.Lock()
    return _vtk_download_locks[run_id]


async def _resolve_sim_run_id(run_id: str) -> str:
    """Look up the simulation server run ID from the agent database.

    Raises HTTPException(404) if the run or its sim_run_id is not found.
    """
    try:
        run_uuid = UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run ID format")

    store = EventStore()
    run = await store.get_run(run_uuid)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    result_data: dict = {}
    if run.result:
        try:
            result_data = json.loads(run.result) if isinstance(run.result, str) else run.result
        except Exception:
            pass

    sim_run_id: str | None = result_data.get("sim_run_id")
    if not sim_run_id:
        raise HTTPException(
            status_code=404,
            detail="No simulation run ID stored — was the simulation successful?",
        )
    return sim_run_id


async def _ensure_vtk_cached(run_id: str) -> dict:
    """Download all VTP files from the sim server once and store locally.

    Workflow
    --------
    1. Resolve sim_run_id from the agent DB (one query, then never again).
    2. Fetch GET /api/run/{sim_run_id}/vtk-timesteps/index.json from sim server.
       The sim server triggers precompute on-demand if not already done.
    3. Download surface.vtp + every timestep VTP listed in the index.
    4. Write a local index.json with URLs rewritten to this agent's endpoints.

    All subsequent calls return the cached local index instantly.

    Returns the local index dict:
        {
            "run_id": str,
            "total": int,
            "fields": [...],
            "surface_vtp": "/api/runs/{run_id}/vtk/surface.vtp",
            "timesteps": [
                {"time": 0.1, "filename": "t_0_1.vtp",
                 "vtp_url": "/api/runs/{run_id}/vtk-timestep/0.1/surface.vtp"},
                ...
            ]
        }
    """
    local_dir = _vtk_local_dir(run_id)
    local_index_path = local_dir / "index.json"

    # Fast path: already cached
    if local_index_path.exists():
        return json.loads(local_index_path.read_text())

    async with _get_download_lock(run_id):
        # Double-check after acquiring lock (another coroutine may have filled it)
        if local_index_path.exists():
            return json.loads(local_index_path.read_text())

        sim_run_id = await _resolve_sim_run_id(run_id)
        settings = get_settings()

        ts_dir = local_dir / "timesteps"
        ts_dir.mkdir(parents=True, exist_ok=True)

        sim = SimulationServerClient(base_url=settings.simulation_server_url)
        try:
            # ── 1. Try precomputed index (newer sim servers) ──────────────────
            sim_index: dict | None = None
            logger.info(f"[VTK_CACHE] Fetching index for run {run_id} (sim={sim_run_id})")
            try:
                sim_index = await sim.get_precomputed_index(sim_run_id)
                if sim_index is not None:
                    raw_timesteps = sim_index.get("timesteps", [])
                    logger.info(
                        f"[VTK_CACHE] Sim server returned {len(raw_timesteps)} timesteps "
                        f"(as-received order): "
                        + ", ".join(
                            f"{ts.get('time')} → {ts.get('filename')}"
                            for ts in raw_timesteps
                        )
                    )
            except SimulationServerError as idx_err:
                logger.warning(
                    f"[VTK_CACHE] Precomputed index unavailable ({idx_err}), "
                    "falling back to vtk-results endpoint"
                )

            # ── 2. Download surface.vtp ───────────────────────────────────────
            logger.info(f"[VTK_CACHE] Downloading surface.vtp for {run_id}")
            if sim_index is None:
                logger.info(f"[VTK_CACHE] Using fallback /vtk-results endpoint (no precomputed index)")
                # Fallback: trigger foamToVTK on the sim server (old /vtk-results API)
                vtk_meta = await sim.get_vtk_results(sim_run_id)
                logger.info(f"[VTK_CACHE] Fallback vtk_meta keys: {list(vtk_meta.keys())} time={vtk_meta.get('time')}")
                surface_bytes = await sim.download_surface_vtp(sim_run_id)
                (local_dir / "surface.vtp").write_bytes(surface_bytes)
                # Use the real simulation time from the vtk-results response
                sim_time = float(vtk_meta.get("time") or 0.0)
                time_str = str(sim_time)
                ts_filename = f"t_{time_str.replace('.', '_')}.vtp"
                # Also write to timesteps/ so serve_timestep_vtp can find it by filename
                ts_dir.mkdir(parents=True, exist_ok=True)
                (ts_dir / ts_filename).write_bytes(surface_bytes)
                local_index = {
                    "run_id": run_id,
                    "sim_run_id": sim_run_id,
                    "total": 1,
                    "fields": vtk_meta.get("fields", []),
                    "surface_vtp": f"/api/runs/{run_id}/vtk/surface.vtp",
                    "timesteps": [
                        {
                            "time": sim_time,
                            "filename": ts_filename,
                            "vtp_url": f"/api/runs/{run_id}/vtk-timestep/{time_str}/surface.vtp",
                        }
                    ],
                }
                local_index_path.write_text(json.dumps(local_index))
                logger.info(f"[VTK_CACHE] Cached (fallback) surface.vtp for run {run_id} t={sim_time} → {local_dir}")
                return local_index

            surface_bytes = await sim.download_surface_vtp(sim_run_id)
            (local_dir / "surface.vtp").write_bytes(surface_bytes)

            # ── 3. Download each timestep VTP ─────────────────────────────────
            raw_ts_list = sim_index.get("timesteps", [])
            logger.info(
                f"[VTK_CACHE] About to download {len(raw_ts_list)} timesteps in received order: "
                + str([ts.get("time") for ts in raw_ts_list])
            )
            local_timesteps: list[dict] = []
            for i, ts in enumerate(raw_ts_list):
                sim_filename: str = ts["filename"]
                time_float = float(ts["time"])
                # Normalize filename using float repr so it always matches the URL
                # param that serve_timestep_vtp reconstructs (e.g. "2" → 2.0 → "t_2_0.vtp").
                # Without this, sim server filenames like "t_2.vtp" would be saved under
                # a name that the endpoint can never reconstruct from "2.0".
                normalized_filename = f"t_{str(time_float).replace('.', '_')}.vtp"
                logger.info(
                    f"[VTK_CACHE] [{i+1}/{len(raw_ts_list)}] "
                    f"Downloading sim={sim_filename!r} time={time_float} → local={normalized_filename!r}"
                )
                vtp_bytes = await sim.download_precomputed_vtp(sim_run_id, sim_filename)
                (ts_dir / normalized_filename).write_bytes(vtp_bytes)
                local_timesteps.append({
                    "time": time_float,
                    "filename": normalized_filename,
                    "vtp_url": f"/api/runs/{run_id}/vtk-timestep/{time_float}/surface.vtp",
                })
            logger.info(
                f"[VTK_CACHE] Saved local timesteps order: "
                + str([ts["time"] for ts in local_timesteps])
            )

            # ── 4. Write local index ──────────────────────────────────────────
            local_index = {
                "run_id": run_id,
                "sim_run_id": sim_run_id,
                "total": len(local_timesteps),
                "fields": sim_index.get("fields", []),
                "surface_vtp": f"/api/runs/{run_id}/vtk/surface.vtp",
                "timesteps": local_timesteps,
            }
            local_index_path.write_text(json.dumps(local_index))
            logger.info(
                f"[VTK_CACHE] Cached {len(local_timesteps)} VTPs for run {run_id} "
                f"→ {local_dir}"
            )
            return local_index

        except SimulationServerError as e:
            raise HTTPException(status_code=502, detail=f"Simulation server error: {e}")
        finally:
            await sim.close()


# ── VTK Results ──────────────────────────────────────────────────────────────

@app.get("/api/runs/{run_id}/vtk-results")
async def get_vtk_results(run_id: str, request: Request) -> dict[str, Any]:
    """Download all VTPs from sim server (once) and return field metadata.

    On first call this downloads every timestep VTP and the surface VTP to
    the local cache.  Subsequent calls return instantly from the local index.

    Response contract (matches frontend ResultViewer):
        {
            "run_id":          str,
            "time":            float,  # time of the FIRST timestep (not the latest)
            "vtp_url":         str,    # URL to the FIRST timestep VTP
            "fields":          [{...}],
            "total_timesteps": int,    # total number of available timesteps
        }

    The frontend should display the first timestep immediately.
    To advance through all timesteps the frontend calls GET /api/runs/{run_id}/playback
    (SSE stream) when the user clicks the playback button — that endpoint emits
    every frame in order.
    """
    local_index = await _ensure_vtk_cached(run_id)

    base = str(request.base_url).rstrip("/")

    # Sort ascending so index 0 is always the earliest frame.
    timesteps = sorted(local_index.get("timesteps", []), key=lambda t: float(t["time"]))
    first = timesteps[0] if timesteps else None

    first_time = float(first["time"]) if first else 0.0
    # Use the stored relative vtp_url so the URL path always matches the cached filename.
    # (surface.vtp always holds the *latest* frame downloaded from the sim server.)
    first_vtp_url = (
        f"{base}{first['vtp_url']}"
        if first
        else f"{base}/api/runs/{run_id}/vtk/surface.vtp"
    )

    return {
        "run_id": run_id,
        "time": first_time,
        "vtp_url": first_vtp_url,
        "fields": local_index.get("fields", []),
        "total_timesteps": len(timesteps),
    }


@app.get("/api/runs/{run_id}/vtk/surface.vtp")
async def serve_surface_vtp(run_id: str):
    """Serve the surface VTP from local cache — pure FileResponse, zero proxying."""
    local_path = _vtk_local_dir(run_id) / "surface.vtp"

    if not local_path.exists():
        # Trigger cache fill if this endpoint is hit before vtk-results
        await _ensure_vtk_cached(run_id)

    if not local_path.exists():
        raise HTTPException(status_code=404, detail="surface.vtp not in local cache")

    return FileResponse(
        str(local_path),
        media_type="application/xml",
        headers={
            "Content-Disposition": 'inline; filename="surface.vtp"',
            "Cache-Control": "public, max-age=86400",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ── Timesteps ────────────────────────────────────────────────────────────────

@app.get("/api/runs/{run_id}/timesteps")
async def get_timesteps(run_id: str, request: Request) -> dict[str, Any]:
    """Return sorted timestep list from local cache.

    Triggers the one-time VTP download if not already cached.

    Response:
        {
            "run_id": str,
            "total": int,
            "fields": [...],
            "timesteps": [{"time": 0.1, "vtp_url": "http://.../vtk-timestep/0.1/surface.vtp"}, ...]
        }
    """
    local_index = await _ensure_vtk_cached(run_id)

    base = str(request.base_url).rstrip("/")
    timesteps = [
        {
            "time": ts["time"],
            # Use the stored relative vtp_url so the path always matches the cached filename.
            "vtp_url": f"{base}{ts['vtp_url']}",
        }
        for ts in sorted(local_index.get("timesteps", []), key=lambda t: float(t["time"]))
    ]
    return {
        "run_id": run_id,
        "total": local_index.get("total", len(timesteps)),
        "fields": local_index.get("fields", []),
        "timesteps": timesteps,
    }


# ── Per-timestep VTP ─────────────────────────────────────────────────────────

@app.get("/api/runs/{run_id}/vtk-timestep/{time_str}/surface.vtp")
async def serve_timestep_vtp(run_id: str, time_str: str):
    """Serve a timestep VTP from local cache — pure FileResponse, zero proxying."""
    filename = f"t_{time_str.replace('.', '_')}.vtp"
    local_path = _vtk_local_dir(run_id) / "timesteps" / filename

    if not local_path.exists():
        # Trigger cache fill then retry
        await _ensure_vtk_cached(run_id)

    if not local_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"{filename} not in local cache — call /vtk-results first",
        )

    return FileResponse(
        str(local_path),
        media_type="application/xml",
        headers={
            "Content-Disposition": f'inline; filename="{filename}"',
            "Cache-Control": "public, max-age=86400",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ── Playback SSE ─────────────────────────────────────────────────────────────

@app.get("/api/runs/{run_id}/playback")
async def playback_sse(run_id: str, request: Request):
    """SSE playback stream built entirely from local cache.

    No sim-server calls, no RAM buffering, no pyvista.
    All VTPs are already on disk; this just emits URL pointers.

    Events:
        {"type": "playback_start",  "total_frames": N, "fields": [...]}
        {"type": "frame",           "frame_index": i, "time": 0.1,
                                    "vtp_url": "http://.../vtk-timestep/0.1/surface.vtp"}
        {"type": "playback_done"}
    """
    local_index = await _ensure_vtk_cached(run_id)
    base = str(request.base_url).rstrip("/")

    timesteps = sorted(local_index.get("timesteps", []), key=lambda t: float(t["time"]))
    fields = local_index.get("fields", [])

    async def _stream():
        yield f"data: {json.dumps({'type': 'playback_start', 'total_frames': len(timesteps), 'fields': fields})}\n\n"
        for i, ts in enumerate(timesteps):
            event = {
                "type": "frame",
                "frame_index": i,
                "total_frames": len(timesteps),
                "time": ts["time"],
                # Use the stored relative vtp_url — matches the normalized filename on disk.
                "vtp_url": f"{base}{ts['vtp_url']}",
            }
            yield f"data: {json.dumps(event)}\n\n"
        yield f"data: {json.dumps({'type': 'playback_done'})}\n\n"

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "simd_agent.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower(),
    )
