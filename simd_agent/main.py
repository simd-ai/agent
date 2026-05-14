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
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.exc import IntegrityError
from pydantic import ValidationError

# DB is initialized lazily in lifespan() below

# Try to import mesh module (may fail on some Python versions due to VTK compatibility)
MESH_ENABLED = False
try:
    from simd_agent.mesh import mesh_router
    MESH_ENABLED = True
except ImportError as e:
    logging.warning(f"Mesh module not available: {e}")
    mesh_router = None
from simd_agent.event_bus import EventBus
from simd_agent.orchestration import Orchestrator
from simd_agent.run.simulation_server_client import SimulationServerClient, SimulationServerError
from simd_agent.models import (
    RunStatus,
    StartRequest,
)
from simd_agent.settings import get_settings
from simd_agent.store import EventStore
from simd_agent.watch_bus import get_watch_bus
from simd_agent.services import user_service

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

    # Initialize database tables
    from simd_agent.db import init_db, close_db
    try:
        await init_db()
        logger.info("Database tables initialized")
    except Exception as e:
        logger.warning(f"Database init skipped (non-fatal): {e}")

    # Discover solver plugins
    from simd_agent.solvers import get_registry
    registry = get_registry()
    logger.info(f"Solver plugins loaded: {registry.names()}")

    yield

    # Shutdown
    logger.info("Shutting down simd_agent service...")
    from simd_agent.telemetry import get_telemetry
    get_telemetry().shutdown()
    try:
        await close_db()
    except Exception:
        pass


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

# ── Global FK-violation handler ──────────────────────────────────────────
# When a simulation is deleted, in-flight auto-save requests from the
# frontend may hit FK constraints. Return 404 instead of 500.
@app.exception_handler(IntegrityError)
async def integrity_error_handler(request: Request, exc: IntegrityError):
    if "ForeignKeyViolationError" in str(exc):
        logger.warning("[FK race] %s %s — resource was deleted", request.method, request.url.path)
        return JSONResponse(status_code=404, content={"detail": "Referenced resource was deleted"})
    raise exc


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Return a proper JSONResponse for unhandled errors so CORS headers are added."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )

# ── API routers (backend is system of record) ───────────────────────────
from simd_agent.api.users import router as users_router
from simd_agent.api.simulations import router as simulations_router
from simd_agent.api.runs import router as runs_router
from simd_agent.api.solvers import router as solvers_router
from simd_agent.api.meshes import router as meshes_router
from simd_agent.api.chat import router as chat_router
from simd_agent.api.precheck_lint import router as precheck_lint_router
from simd_agent.api.snapshot import router as snapshot_router
from simd_agent.api.reports import router as reports_router

app.include_router(users_router)
app.include_router(simulations_router)
app.include_router(runs_router)
app.include_router(solvers_router)
app.include_router(meshes_router)
app.include_router(chat_router)
app.include_router(precheck_lint_router)
app.include_router(snapshot_router)
app.include_router(reports_router)

# Mount mesh routes (if available) — mesh files are served from GCS
if MESH_ENABLED:
    app.include_router(mesh_router)
    logger.info("Mesh converter enabled at /api/mesh/convert (GCS storage)")
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
        "users": "/api/users",
        "simulations": "/api/simulations",
        "simulations_config": "/api/simulations/{id}/config",
        "simulations_form_state": "/api/simulations/{id}/form-state",
        "simulations_mesh": "/api/simulations/{id}/mesh",
        "simulations_patches": "/api/simulations/{id}/patches",
        "simulations_chat": "/api/simulations/{id}/chat",
        "simulations_precheck": "/api/simulations/{id}/precheck",
        "simulations_lint": "/api/simulations/{id}/lint",
        "simulations_snapshot": "/api/simulations/{id}/snapshot",
        "runs": "/api/runs",
        "runs_events": "/api/runs/{id}/events",
        "runs_progress": "/api/runs/{id}/progress",
        "runs_complete": "/api/runs/{id}/complete",
        "solvers": "/api/solvers",
    }
    if MESH_ENABLED:
        endpoints["mesh_convert"] = "/api/mesh/convert"

    return {
        "service": "simd_agent",
        "version": "0.3.0",
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
                # Send a user-friendly error message — raw tracebacks are noisy
                error_msg = (
                    "Something went wrong while processing your message. "
                    "Please try again. If the issue persists, try starting a new conversation."
                )
                try:
                    await websocket.send_json({"type": "error", "message": error_msg})
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


# Active orchestrators by run_id — allows the REST stop endpoint to signal
# a running orchestrator to stop gracefully.
_active_orchestrators: dict[UUID, Orchestrator] = {}


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
    cancelled = asyncio.Event()

    logger.info(f"[WS] New connection - run {run_id}")

    try:
        # Receive start request (30s timeout)
        try:
            data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning(f"[WS] Timeout waiting for start request - run {run_id}")
            await websocket.send_json({"type": "error", "error": "Timeout waiting for start request"})
            await websocket.close(code=1008)
            return

        # Parse request
        sim_config = data.get('simulation_config', {})
        if isinstance(sim_config, str):
            import json as _json
            try:
                sim_config = _json.loads(sim_config)
            except (ValueError, TypeError):
                sim_config = {}
            data['simulation_config'] = sim_config
        _solver_raw = sim_config.get('solver', {}) if isinstance(sim_config, dict) else {}
        solver_info = _solver_raw.get('solver', 'auto') if isinstance(_solver_raw, dict) else str(_solver_raw)
        logger.info(f"[WS] Request: op={data.get('op')}, solver={solver_info}")
        logger.info(f"[WS→BACKEND] simulation_config.solver = {_solver_raw}")
        logger.info(f"[WS→BACKEND] simulation_config.physics = {sim_config.get('physics', {}) if isinstance(sim_config, dict) else {}}")
        _mesh_raw = sim_config.get('mesh', {}) if isinstance(sim_config, dict) else {}
        _check_mesh_raw = _mesh_raw.get('check_mesh') or _mesh_raw.get('checkMesh') if isinstance(_mesh_raw, dict) else None
        logger.info(f"[WS→BACKEND] simulation_config.mesh.check_mesh = {_check_mesh_raw}")
        try:
            request = StartRequest(**data)
        except ValidationError as e:
            logger.warning(f"[WS] Invalid request - run {run_id}: {e}")
            await websocket.send_json({"type": "error", "error": f"Invalid start request: {e}"})
            await websocket.close(code=1003)
            return

        # Enforce run limit for free-tier users
        if request.op in ("CFD_CODEGEN_RUN", "CFD_RESUBMIT") and request.metadata.user_id:
            try:
                usage = await user_service.get_usage(UUID(request.metadata.user_id))
                if not usage.can_start_run:
                    from simd_agent.telemetry import get_telemetry, UsageLimitHit
                    get_telemetry().capture(
                        UsageLimitHit(limit_type="run", current_count=usage.run_count),
                        user_id=request.metadata.user_id,
                    )
                    await websocket.send_json({
                        "type": "error",
                        "error": f"Free plan allows up to {usage.limits.max_runs} simulation runs. "
                                 "Upgrade to Pro for unlimited runs.",
                        "code": "RUN_LIMIT_REACHED",
                    })
                    await websocket.close(code=1008)
                    return
            except Exception as e:
                logger.warning(f"[WS] Usage check failed (allowing run): {e}")

        # Create event bus
        event_bus = EventBus(
            run_id=run_id,
            websocket=websocket,
            store=store,
            persist=True,
        )

        # Create run in database (non-blocking - continue even if DB is down)
        try:
            await store.create_run(
                op=request.op,
                provider=request.provider,
                prompt_pack=request.prompt_pack,
                user_requirements=request.user_requirements,
                simulation_config=request.simulation_config,
                run_id=run_id,
                raw_config=data.get('simulation_config', {}),
                simulation_id=request.metadata.project_id,
            )
            await store.update_run_status(run_id, RunStatus.RUNNING)
        except Exception as e:
            logger.error(f"[WS] DB error (continuing): {e}")

        # Create orchestrator with cancellation support
        orchestrator = Orchestrator(
            run_id=run_id,
            event_bus=event_bus,
            store=store,
            request=request,
            cancelled=cancelled,
        )
        _active_orchestrators[run_id] = orchestrator

        # Start heartbeat
        heartbeat_task = asyncio.create_task(
            _heartbeat_loop(websocket, get_settings().ws_heartbeat_interval)
        )

        try:
            # Run with timeout from request constraints
            timeout = request.constraints.timeout_seconds
            result = await asyncio.wait_for(orchestrator.run(), timeout=timeout)
            logger.info(f"[WS] Run {run_id} completed: {result.status}")
        except asyncio.TimeoutError:
            logger.error(f"[WS] Run {run_id} timed out after {timeout}s")
            cancelled.set()
            if event_bus:
                try:
                    await event_bus.emit_run_failed(f"Run timed out after {timeout}s")
                    await event_bus.emit_final(status="failed", error=f"Timeout after {timeout}s")
                except Exception:
                    pass
            try:
                await store.finalize_run(run_id=run_id, status=RunStatus.FAILED, result={"error": f"Timeout after {timeout}s"})
            except Exception:
                pass
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass

    except WebSocketDisconnect:
        cancelled.set()
        logger.info(f"[WS] Client disconnected - run {run_id}. Marking as cancelled.")
        try:
            await store.finalize_run(
                run_id=run_id,
                status=RunStatus.CANCELLED,
                result={"error": "User cancelled (client disconnected)"},
            )
        except Exception:
            pass

    except Exception as e:
        cancelled.set()
        logger.exception(f"[WS] Unhandled error - run {run_id}: {e}")
        if event_bus:
            try:
                await event_bus.emit_run_failed(f"Internal error: {e}")
                await event_bus.emit_final(status="failed", error=str(e))
            except Exception:
                pass
        try:
            await store.finalize_run(run_id=run_id, status=RunStatus.FAILED, result={"error": str(e)})
        except Exception:
            pass

    finally:
        _active_orchestrators.pop(run_id, None)
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


# ── Run status (quick REST check used by the frontend on page load) ───────────

@app.get("/api/runs/{run_id}/status")
async def get_run_status(run_id: str) -> dict[str, Any]:
    """Quick status check — client calls this on page load to decide whether
    to open a /ws/watch connection for a still-running simulation."""
    try:
        run_uuid = UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run ID format")

    store = EventStore()
    run = await store.get_run(run_uuid)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    last_seq = await store.get_last_seq(run_uuid)

    return {
        "run_id": run_id,
        "status": run.status.value,
        "last_seq": last_seq,
        "progress": None,
    }


@app.post("/api/runs/{run_id}/cancel")
async def cancel_run(run_id: str) -> dict[str, Any]:
    """Explicitly cancel a run — marks it as CANCELLED in the DB.

    Called by the frontend when the user clicks "Cancel simulation".
    Works regardless of whether the run was started via /ws/run or
    reconnected via /ws/watch.
    """
    try:
        run_uuid = UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run ID format")

    store = EventStore()
    run = await store.get_run(run_uuid)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    if run.status in _TERMINAL_STATUSES:
        return {"run_id": run_id, "status": run.status.value, "already_terminal": True}

    # Signal the orchestrator so it stops the sim server process too
    orchestrator = _active_orchestrators.get(run_uuid)
    if orchestrator is not None:
        orchestrator._cancelled.set()
        logger.info(f"[CANCEL] Run {run_id} orchestrator signalled to cancel")

    await store.finalize_run(
        run_id=run_uuid,
        status=RunStatus.CANCELLED,
        result={"error": "User cancelled"},
    )
    logger.info(f"[CANCEL] Run {run_id} cancelled via REST endpoint")

    from simd_agent.telemetry import get_telemetry, RunCancelled
    get_telemetry().capture(RunCancelled())

    return {"run_id": run_id, "status": "cancelled"}


@app.post("/api/runs/{run_id}/stop")
async def stop_run(run_id: str) -> dict[str, Any]:
    """Gracefully stop a running simulation.

    Signals the active orchestrator to tell the sim server to stop the
    solver, reconstruct partial results, and return them as if the
    simulation completed.  The run is marked as STOPPED (not CANCELLED).
    """
    try:
        run_uuid = UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run ID format")

    store = EventStore()
    run = await store.get_run(run_uuid)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    if run.status in _TERMINAL_STATUSES:
        return {"run_id": run_id, "stopped": False, "reason": f"Run already {run.status.value}"}

    # Signal the orchestrator to stop the sim server run gracefully
    orchestrator = _active_orchestrators.get(run_uuid)
    if orchestrator is not None:
        orchestrator._stop_requested.set()
        logger.info(f"[STOP] Run {run_id} stop requested via REST endpoint")
        return {"run_id": run_id, "stopped": True}

    # No active orchestrator — fall back to marking stopped in DB
    await store.finalize_run(
        run_id=run_uuid,
        status=RunStatus.STOPPED,
        result={"info": "Stopped by user"},
    )
    logger.info(f"[STOP] Run {run_id} marked stopped (no active orchestrator)")
    return {"run_id": run_id, "stopped": True}


@app.post("/api/runs/{run_id}/continue")
async def continue_run(run_id: str) -> dict[str, Any]:
    """Continue a stopped simulation from the last checkpoint.

    Calls the sim server's continue endpoint (patches controlDict to
    ``startFrom latestTime`` and re-launches the solver), then streams
    events via the WatchBus so ``/ws/watch/{run_id}`` subscribers receive
    live updates.
    """
    try:
        run_uuid = UUID(run_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid run ID format")

    store = EventStore()
    run = await store.get_run(run_uuid)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    if run.status != RunStatus.STOPPED:
        raise HTTPException(
            status_code=409,
            detail=f"Run is {run.status.value}, only stopped runs can be continued",
        )

    sim_run_id = (run.result or {}).get("sim_run_id")
    if not sim_run_id:
        raise HTTPException(
            status_code=409,
            detail="No sim_run_id found — cannot continue this run",
        )

    # Reset run status to RUNNING
    await store.update_run_status(run_uuid, RunStatus.RUNNING)

    # Tell the sim server to continue
    sim_client = SimulationServerClient()
    try:
        await sim_client.continue_run(sim_run_id)
    except SimulationServerError as e:
        # Revert status on failure
        await store.finalize_run(run_id=run_uuid, status=RunStatus.STOPPED)
        raise HTTPException(status_code=502, detail=str(e))

    # Launch a background task to stream events from the continued sim run
    # into the DB + WatchBus, so /ws/watch subscribers receive live updates.
    asyncio.create_task(
        _stream_continued_run(run_uuid, sim_run_id, store)
    )

    logger.info(f"[CONTINUE] Run {run_id} resumed (sim_run_id={sim_run_id})")
    return {"run_id": run_id, "continued": True, "sim_run_id": sim_run_id}


async def _stream_continued_run(
    run_id: UUID,
    sim_run_id: str,
    store: EventStore,
) -> None:
    """Background task: stream SSE events from a continued sim run,
    persist them, and publish to the WatchBus."""
    from simd_agent.models import AgentEvent, EventLevel, EventTypes
    from simd_agent.run.simulation_server_client import SimRunEvent

    watch_bus = get_watch_bus()
    sim_client = SimulationServerClient()
    seq = 0

    # Get the current max seq from the DB so we continue numbering
    try:
        events = await store.get_events(run_id)
        if events:
            seq = max(e.seq for e in events) + 1
    except Exception:
        pass

    # Emit a sim_progress_reset so the frontend clears old residuals
    from datetime import datetime
    reset_event = AgentEvent(
        run_id=run_id, seq=seq, ts=datetime.utcnow(),
        level=EventLevel.INFO, type="sim_progress_reset",
        message="Continued simulation — clearing old residual data",
        payload={},
    )
    seq += 1
    try:
        await store.append_event(reset_event)
        watch_bus.publish_nowait(str(run_id), reset_event.to_ws_message())
    except Exception as e:
        logger.warning("[CONTINUE] Failed to emit reset event: %s", e)

    # Map sim server event types to agent event types (subset of orchestration.py)
    _EVENT_MAP = {
        "run_started": "sim_run_started",
        "run_log": "sim_run_log",
        "run_succeeded": "sim_run_succeeded",
        "run_failed": "sim_run_failed",
        "run_stopped": "sim_run_stopped",
        "artifacts_ready": "sim_artifacts_ready",
        "reconstruct_started": "sim_reconstruct_started",
        "reconstruct_complete": "sim_reconstruct_complete",
    }
    _SKIP = {"decompose_started", "decompose_complete", "decompose_failed",
             "decompose_log", "reconstruct_log", "run_log"}

    try:
        async for event in sim_client.stream_events(sim_run_id):
            if event.type in _SKIP:
                continue

            mapped = _EVENT_MAP.get(event.type, event.type)

            # Build sim_progress_batch for run_progress / run_progress_batch
            if event.type in ("run_progress", "run_progress_batch"):
                raw_items = (
                    event.payload.get("items", [])
                    if event.type == "run_progress_batch"
                    else [event.payload]
                )
                built = []
                for p in raw_items:
                    residuals_raw = p.get("residuals", {})
                    residuals = {}
                    for field, val in residuals_raw.items():
                        if isinstance(val, dict):
                            residuals[field] = {
                                "initial": float(val.get("initial", 0)),
                                "final": float(val.get("final", val.get("initial", 0))),
                                "iters": int(val.get("iters", 1)),
                            }
                        else:
                            residuals[field] = {"initial": float(val), "final": float(val), "iters": 1}

                    courant_raw = p.get("courant")
                    courant = (
                        {"mean": float(courant_raw["mean"]), "max": float(courant_raw["max"])}
                        if isinstance(courant_raw, dict) else None
                    )
                    cont_raw = p.get("continuity")
                    continuity = (
                        {"local": float(cont_raw["local"]), "global": float(cont_raw["global"]),
                         "cumulative": float(cont_raw["cumulative"])}
                        if isinstance(cont_raw, dict) else None
                    )
                    exec_raw = p.get("execution")
                    execution = (
                        {"stepSeconds": float(exec_raw.get("stepSeconds", exec_raw.get("step_seconds", 0))),
                         "clockSeconds": float(exec_raw.get("clockSeconds", exec_raw.get("clock_seconds", 0))),
                         "label": exec_raw.get("label", "")}
                        if isinstance(exec_raw, dict) else None
                    )
                    built.append({
                        "iteration": int(p.get("iteration", 0)),
                        "simTime": float(p.get("time", p.get("simTime", 0))),
                        "fields": list(p.get("fields", list(residuals.keys()))),
                        "residuals": residuals,
                        "courant": courant,
                        "continuity": continuity,
                        "execution": execution,
                    })

                if built:
                    ae = AgentEvent(
                        run_id=run_id, seq=seq, ts=datetime.utcnow(),
                        level=EventLevel.INFO, type="sim_progress_batch",
                        message=f"{len(built)} step(s)",
                        payload={"items": built},
                    )
                    seq += 1
                    watch_bus.publish_nowait(str(run_id), ae.to_ws_message())
                    # Don't persist progress events (too many)
                continue

            # Terminal: artifacts_ready means simulation finished
            if event.type == "artifacts_ready":
                break

            # Emit mapped event
            ae = AgentEvent(
                run_id=run_id, seq=seq, ts=datetime.utcnow(),
                level=EventLevel.INFO, type=mapped,
                message=event.message, payload=event.payload,
            )
            seq += 1
            try:
                await store.append_event(ae)
            except Exception:
                pass
            watch_bus.publish_nowait(str(run_id), ae.to_ws_message())

        # Check final status
        final_status = await sim_client.get_status(sim_run_id)

        if final_status.status.value in ("succeeded", "stopped"):
            final_label = final_status.status.value
            db_status = RunStatus.STOPPED if final_label == "stopped" else RunStatus.SUCCEEDED
            summary = (
                "Simulation stopped by user — partial results available"
                if final_label == "stopped"
                else "Continued simulation completed successfully"
            )
        else:
            db_status = RunStatus.FAILED
            summary = final_status.error or "Continued simulation failed"

        # Emit run_succeeded / run_stopped / run_failed
        run_event_type = {
            RunStatus.SUCCEEDED: "run_succeeded",
            RunStatus.STOPPED: "run_stopped",
            RunStatus.FAILED: "run_failed",
        }[db_status]

        ae = AgentEvent(
            run_id=run_id, seq=seq, ts=datetime.utcnow(),
            level=EventLevel.INFO, type=run_event_type,
            message=summary, payload={},
        )
        seq += 1
        try:
            await store.append_event(ae)
        except Exception:
            pass
        watch_bus.publish_nowait(str(run_id), ae.to_ws_message())

        # Emit final event
        final_ae = AgentEvent(
            run_id=run_id, seq=seq, ts=datetime.utcnow(),
            level=EventLevel.INFO, type="final",
            message=summary,
            payload={
                "run_id": str(run_id),
                "status": final_label if db_status != RunStatus.FAILED else "failed",
                "validated_config": None,
                "artifacts": [],
                "iterations": 0,
                "retries": 0,
                "summary": summary,
                "case_type": None,
                "solver": None,
                "error": None if db_status != RunStatus.FAILED else summary,
            },
        )
        seq += 1
        try:
            await store.append_event(final_ae)
        except Exception:
            pass
        watch_bus.publish_nowait(str(run_id), final_ae.to_ws_message())

        # Finalize in DB
        await store.finalize_run(
            run_id=run_id,
            status=db_status,
            result={
                "sim_run_id": sim_run_id,
                "summary": summary,
            },
        )
        logger.info("[CONTINUE] Run %s finalized as %s", run_id, db_status.value)

    except Exception as e:
        logger.error("[CONTINUE] Stream error for run %s: %s", run_id, e)
        # Mark as failed
        try:
            await store.finalize_run(
                run_id=run_id,
                status=RunStatus.FAILED,
                result={"error": str(e), "sim_run_id": sim_run_id},
            )
        except Exception:
            pass


# ── Reconnectable watch WebSocket ─────────────────────────────────────────────

_TERMINAL_STATUSES = {
    RunStatus.SUCCEEDED,
    RunStatus.FAILED,
    RunStatus.CANCELLED,
    RunStatus.STOPPED,
    RunStatus.NOT_CLEAR,
    RunStatus.CONFIG_INCOMPLETE,
}

_TERMINAL_EVENT_TYPES = frozenset(
    {"final", "run_succeeded", "run_failed", "simulation_not_clear"}
)


@app.websocket("/ws/watch/{run_id}")
async def websocket_watch(websocket: WebSocket, run_id: str, last_seq: int = 0):
    """Reconnectable observer for a running or already-completed simulation.

    Protocol
    --------
    1. Client connects (optionally with ?last_seq=N to skip already-seen events).
    2. Server replays all events with seq > last_seq from the DB (marked with
       "replayed": true so the client can skip animations).
    3. If the run is already in a terminal state, server sends a run_complete
       sentinel and closes.
    4. Otherwise the server forwards live events as they arrive (via WatchBus).
    5. When the run reaches a terminal event the server closes the connection.
    """
    await websocket.accept()

    try:
        run_uuid = UUID(run_id)
    except ValueError:
        await websocket.send_json({"type": "error", "message": "Invalid run ID"})
        await websocket.close(code=1003)
        return

    store = EventStore()

    # ── Phase 1: replay missed events ────────────────────────────────────────
    try:
        missed = await store.get_events_since(run_uuid, last_seq)
        for ev in missed:
            msg = {
                "run_id": str(ev.run_id),
                "seq": ev.seq,
                "ts": ev.ts.isoformat() if ev.ts else None,
                "level": ev.level.value,
                "type": ev.type,
                "message": ev.message,
                "payload": ev.payload,
                "replayed": True,
            }
            await websocket.send_json(msg)
    except WebSocketDisconnect:
        return
    except Exception as e:
        logger.error(f"[WS/watch] Replay error for {run_id}: {e}")

    # ── Phase 2: check terminal state before subscribing ─────────────────────
    run = await store.get_run(run_uuid)
    if run is None or run.status in _TERMINAL_STATUSES:
        try:
            status_val = run.status.value if run else "not_found"
            await websocket.send_json({"type": "run_complete", "status": status_val})
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass
        return

    # ── Phase 3: subscribe to live events ────────────────────────────────────
    bus = get_watch_bus()
    queue = await bus.subscribe(run_id)

    heartbeat_task = asyncio.create_task(
        _heartbeat_loop(websocket, get_settings().ws_heartbeat_interval)
    )

    try:
        while True:
            try:
                message = await asyncio.wait_for(queue.get(), timeout=60.0)
                await websocket.send_json(message)

                # Close after delivering the terminal event so the client has
                # time to process it before the connection disappears.
                if message.get("type") in _TERMINAL_EVENT_TYPES:
                    await asyncio.sleep(0.3)
                    break

            except asyncio.TimeoutError:
                # Periodic timeout — re-check if run finished while we waited
                run = await store.get_run(run_uuid)
                if run and run.status in _TERMINAL_STATUSES:
                    try:
                        await websocket.send_json(
                            {"type": "run_complete", "status": run.status.value}
                        )
                    except (WebSocketDisconnect, RuntimeError):
                        pass
                    break

    except WebSocketDisconnect:
        logger.info(f"[WS/watch] Client disconnected for run {run_id}")
    except RuntimeError:
        logger.info(f"[WS/watch] Client already disconnected for run {run_id}")
    except Exception as e:
        logger.exception(f"[WS/watch] Unhandled error for run {run_id}: {e}")
    finally:
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        await bus.unsubscribe(run_id, queue)
        try:
            await websocket.close()
        except Exception:
            pass


# ── Local VTK cache ──────────────────────────────────────────────────────────
# ── VTK Result Storage ────────────────────────────────────────────────────────
#
# VTPs are downloaded from the simulation server ONCE per run and stored in the
# configured object storage backend (local filesystem or GCS).
# All subsequent frontend requests are served from storage — no DB lookup,
# no HTTP round-trip to the sim server, no RAM buffering.

# Per-run download locks prevent concurrent coroutines from double-downloading.
_vtk_download_locks: dict[str, asyncio.Lock] = {}


def _vtk_key(run_id: str, filename: str) -> str:
    """Build the storage key for a VTK result file."""
    return f"results/{run_id}/{filename}"


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
    """Download all VTP files from the sim server once and store in object storage.

    Workflow
    --------
    1. Resolve sim_run_id from the agent DB (one query, then never again).
    2. Fetch GET /api/run/{sim_run_id}/vtk-timesteps/index.json from sim server.
       The sim server triggers precompute on-demand if not already done.
    3. Download surface.vtp + every timestep VTP listed in the index.
    4. Write an index.json with URLs rewritten to this agent's endpoints.

    All subsequent calls return the cached index instantly.

    Returns the index dict:
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
    from simd_agent.storage import get_storage
    storage = get_storage()

    index_key = _vtk_key(run_id, "index.json")

    # Fast path: already cached in storage
    index_data = await storage.download(index_key)
    if index_data is not None:
        return json.loads(index_data)

    async with _get_download_lock(run_id):
        # Double-check after acquiring lock (another coroutine may have filled it)
        index_data = await storage.download(index_key)
        if index_data is not None:
            return json.loads(index_data)

        sim_run_id = await _resolve_sim_run_id(run_id)
        settings = get_settings()

        sim = SimulationServerClient(base_url=settings.simulation_server_url)
        try:
            # ── 1. Try precomputed index (newer sim servers) ──────────────────
            sim_index: dict | None = None
            logger.info(f"[VTK] Fetching index for run {run_id} (sim={sim_run_id})")
            try:
                sim_index = await sim.get_precomputed_index(sim_run_id)
                if sim_index is not None:
                    raw_timesteps = sim_index.get("timesteps", [])
                    logger.info(
                        f"[VTK] Sim server returned {len(raw_timesteps)} timesteps "
                        f"(as-received order): "
                        + ", ".join(
                            f"{ts.get('time')} → {ts.get('filename')}"
                            for ts in raw_timesteps
                        )
                    )
            except SimulationServerError as idx_err:
                logger.warning(
                    f"[VTK] Precomputed index unavailable ({idx_err}), "
                    "falling back to vtk-results endpoint"
                )

            # ── 2. Download surface.vtp ───────────────────────────────────────
            logger.info(f"[VTK] Downloading surface.vtp for {run_id}")
            if sim_index is None:
                logger.info(f"[VTK] Using fallback /vtk-results endpoint (no precomputed index)")
                vtk_meta = await sim.get_vtk_results(sim_run_id)
                logger.info(f"[VTK] Fallback vtk_meta keys: {list(vtk_meta.keys())} time={vtk_meta.get('time')}")
                surface_bytes = await sim.download_surface_vtp(sim_run_id)
                await storage.upload(_vtk_key(run_id, "surface.vtp"), surface_bytes, "application/xml")
                sim_time = float(vtk_meta.get("time") or 0.0)
                time_str = str(sim_time)
                ts_filename = f"t_{time_str.replace('.', '_')}.vtp"
                await storage.upload(_vtk_key(run_id, f"timesteps/{ts_filename}"), surface_bytes, "application/xml")
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
                await storage.upload(index_key, json.dumps(local_index).encode())
                logger.info(f"[VTK] Cached (fallback) surface.vtp for run {run_id} t={sim_time}")
                return local_index

            # Try to download surface.vtp; sim servers using the precomputed index
            # may not expose /vtk/surface.vtp — fall back to the last timestep VTP.
            surface_bytes: bytes | None = None
            try:
                surface_bytes = await sim.download_surface_vtp(sim_run_id)
                await storage.upload(_vtk_key(run_id, "surface.vtp"), surface_bytes, "application/xml")
                logger.info(f"[VTK] surface.vtp downloaded for {run_id}")
            except SimulationServerError as surf_err:
                logger.warning(
                    f"[VTK] surface.vtp unavailable ({surf_err}); "
                    "will copy last timestep VTP as surface.vtp after download"
                )

            # ── 3. Download each timestep VTP (parallel) ────────────────────
            raw_ts_list = sim_index.get("timesteps", [])
            logger.info(
                f"[VTK] About to download {len(raw_ts_list)} timesteps in parallel: "
                + str([ts.get("time") for ts in raw_ts_list])
            )

            async def _download_one_ts(
                i: int, ts: dict
            ) -> tuple[int, float, str, bytes, list[dict]]:
                sim_filename: str = ts["filename"]
                time_float = float(ts["time"])
                step_fields: list[dict] = ts.get("fields", [])
                normalized_filename = f"t_{str(time_float).replace('.', '_')}.vtp"
                logger.info(
                    f"[VTK] [{i+1}/{len(raw_ts_list)}] "
                    f"Downloading sim={sim_filename!r} time={time_float}"
                )
                vtp_bytes = await sim.download_precomputed_vtp(
                    sim_run_id, sim_filename
                )
                await storage.upload(
                    _vtk_key(run_id, f"timesteps/{normalized_filename}"),
                    vtp_bytes, "application/xml",
                )
                return i, time_float, normalized_filename, vtp_bytes, step_fields

            ts_results = await asyncio.gather(
                *[_download_one_ts(i, ts) for i, ts in enumerate(raw_ts_list)]
            )
            # Sort by original index to preserve order
            ts_results_sorted = sorted(ts_results, key=lambda r: r[0])
            local_timesteps: list[dict] = []
            last_vtp_bytes: bytes | None = None
            for _, time_float, normalized_filename, vtp_bytes, step_fields in ts_results_sorted:
                last_vtp_bytes = vtp_bytes
                entry: dict = {
                    "time": time_float,
                    "filename": normalized_filename,
                    "vtp_url": f"/api/runs/{run_id}/vtk-timestep/{time_float}/surface.vtp",
                }
                if step_fields:
                    entry["fields"] = step_fields
                local_timesteps.append(entry)
            logger.info(
                f"[VTK] Saved {len(local_timesteps)} timesteps: "
                + str([ts["time"] for ts in local_timesteps])
            )

            # If surface.vtp was unavailable, use the last (highest-time) timestep
            if surface_bytes is None and local_timesteps and last_vtp_bytes:
                sorted_ts = sorted(local_timesteps, key=lambda t: float(t["time"]))
                last_ts = sorted_ts[-1]
                last_ts_data = await storage.download(
                    _vtk_key(run_id, f"timesteps/{last_ts['filename']}")
                )
                if last_ts_data:
                    await storage.upload(_vtk_key(run_id, "surface.vtp"), last_ts_data, "application/xml")
                    logger.info(f"[VTK] Used last timestep (t={last_ts['time']}) as surface.vtp")

            # ── 4. Write index ────────────────────────────────────────────────
            local_index = {
                "run_id": run_id,
                "sim_run_id": sim_run_id,
                "total": len(local_timesteps),
                "fields": sim_index.get("fields", []),
                "surface_vtp": f"/api/runs/{run_id}/vtk/surface.vtp",
                "timesteps": local_timesteps,
            }
            await storage.upload(index_key, json.dumps(local_index).encode())
            logger.info(f"[VTK] Cached {len(local_timesteps)} VTPs for run {run_id}")
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
    object storage.  Subsequent calls return instantly from the cached index.

    Response contract (matches frontend ResultViewer):
        {
            "run_id":          str,
            "time":            float,
            "vtp_url":         str,
            "fields":          [{...}],
            "total_timesteps": int,
        }
    """
    local_index = await _ensure_vtk_cached(run_id)

    base = str(request.base_url).rstrip("/")

    timesteps = sorted(local_index.get("timesteps", []), key=lambda t: float(t["time"]))
    last = timesteps[-1] if timesteps else None

    last_time = float(last["time"]) if last else 0.0
    last_vtp_url = (
        f"{base}{last['vtp_url']}"
        if last
        else f"{base}/api/runs/{run_id}/vtk/surface.vtp"
    )

    from simd_agent.telemetry import get_telemetry, ResultsViewed
    get_telemetry().capture(ResultsViewed())

    return {
        "run_id": run_id,
        "time": last_time,
        "vtp_url": last_vtp_url,
        "fields": local_index.get("fields", []),
        "total_timesteps": len(timesteps),
    }


@app.get("/api/runs/{run_id}/vtk/surface.vtp")
async def serve_surface_vtp(run_id: str):
    """Serve the surface VTP from object storage."""
    from simd_agent.storage import get_storage

    key = _vtk_key(run_id, "surface.vtp")
    data = await get_storage().download(key)

    if data is None:
        # Trigger cache fill if this endpoint is hit before vtk-results
        await _ensure_vtk_cached(run_id)
        data = await get_storage().download(key)

    if data is None:
        raise HTTPException(status_code=404, detail="surface.vtp not found")

    from fastapi.responses import Response
    return Response(
        content=data,
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
    """Return sorted timestep list from cache.

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
    timesteps = []
    for ts in sorted(local_index.get("timesteps", []), key=lambda t: float(t["time"])):
        entry = {
            "time": ts["time"],
            "vtp_url": f"{base}{ts['vtp_url']}",
        }
        if "fields" in ts:
            entry["fields"] = ts["fields"]
        timesteps.append(entry)
    return {
        "run_id": run_id,
        "total": local_index.get("total", len(timesteps)),
        "fields": local_index.get("fields", []),
        "timesteps": timesteps,
    }


# ── Per-timestep VTP ─────────────────────────────────────────────────────────

@app.get("/api/runs/{run_id}/vtk-timestep/{time_str}/surface.vtp")
async def serve_timestep_vtp(run_id: str, time_str: str):
    """Serve a timestep VTP from object storage."""
    from simd_agent.storage import get_storage

    filename = f"t_{time_str.replace('.', '_')}.vtp"
    key = _vtk_key(run_id, f"timesteps/{filename}")
    data = await get_storage().download(key)

    if data is None:
        # Trigger cache fill then retry
        await _ensure_vtk_cached(run_id)
        data = await get_storage().download(key)

    if data is None:
        raise HTTPException(
            status_code=404,
            detail=f"{filename} not found — call /vtk-results first",
        )

    from fastapi.responses import Response
    return Response(
        content=data,
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
    """SSE playback stream built entirely from cached index.

    No sim-server calls, no RAM buffering, no pyvista.
    All VTPs are already in storage; this just emits URL pointers.

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
                "vtp_url": f"{base}{ts['vtp_url']}",
            }
            if "fields" in ts:
                event["fields"] = ts["fields"]
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
