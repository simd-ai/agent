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

        # Enforce per-project run limit for free-tier users
        if request.op in ("CFD_CODEGEN_RUN", "CFD_RESUBMIT") and request.metadata.user_id:
            try:
                sim_uuid = (
                    UUID(request.metadata.project_id)
                    if request.metadata.project_id else None
                )
                usage = await user_service.get_usage(
                    UUID(request.metadata.user_id),
                    simulation_id=sim_uuid,
                )
                if not usage.can_start_run:
                    from simd_agent.telemetry import get_telemetry, UsageLimitHit
                    get_telemetry().capture(
                        UsageLimitHit(limit_type="run", current_count=usage.run_count),
                        user_id=request.metadata.user_id,
                    )
                    scope_note = "per project" if sim_uuid else "total"
                    await websocket.send_json({
                        "type": "error",
                        "error": f"Free plan allows up to {usage.limits.max_runs} simulation runs {scope_note}. "
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
        # Respect a terminal state already written by the /stop or /cancel
        # endpoint — they finalize the DB before the orchestrator's
        # OrchestrationError bubbles up here, and overwriting with FAILED
        # would strand the resume button (the /continue check expects STOPPED).
        existing_status: RunStatus | None = None
        try:
            existing = await store.get_run(run_id)
            existing_status = existing.status if existing else None
        except Exception:
            pass
        if existing_status in _TERMINAL_STATUSES:
            logger.info(
                f"[WS] Run {run_id} already finalized as {existing_status.value}; "
                f"not overwriting with FAILED"
            )
        else:
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


# ── Run summary — short JSON for the CLI (`simd ls`, end-of-run) ────────────
@app.get("/api/runs/{run_id}/summary")
async def get_run_summary(run_id: str, request: Request) -> dict[str, Any]:
    """One-shot summary of a run.

    Carries just enough for the CLI to render its final-stage block and for
    `simd ls` to show a table — without the full event firehose.  All fields
    are optional; any field the run hasn't reached yet returns ``None``.
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

    # Build the VTK URL only when the run actually has a sim_run_id — the
    # CLI knows that absence means "no VTK output exists" (same convention
    # as `/api/runs/{id}/vtk-results` returning 404).
    base = str(request.base_url).rstrip("/")
    sim_run_id = result_data.get("sim_run_id")
    vtk_url = (
        f"{base}/api/runs/{run_id}/vtk-results" if sim_run_id else None
    )

    # ``RunRow`` doesn't declare ``started_at`` / ``completed_at``; only
    # ``created_at`` is on the model.  Use getattr-with-fallback for both
    # so accessing an undeclared field doesn't AttributeError, then call
    # ``.isoformat()`` only if the value is a real datetime.
    _started = getattr(run, "started_at", None) or run.created_at
    _completed = getattr(run, "completed_at", None)
    started_at = _started.isoformat() if hasattr(_started, "isoformat") else _started
    completed_at = (
        _completed.isoformat() if hasattr(_completed, "isoformat") else _completed
    )

    return {
        "run_id":         run_id,
        "status":         run.status.value,
        "solver":         result_data.get("solver"),
        "op":             getattr(run, "op", None) or result_data.get("op"),
        "started_at":     started_at,
        "completed_at":   completed_at,
        "sim_run_id":     sim_run_id,
        "vtk_url":        vtk_url,
        "final_residuals": result_data.get("final_residuals"),
        "iterations":     result_data.get("iterations"),
        "error":          result_data.get("error"),
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
        # Emit an immediate ACK over the WebSocket so the frontend flips to
        # "Stopping…" right away.  Without this the user sees no feedback
        # until the sim server has finished reconstruction (10–30 s of
        # apparent silence while residuals keep streaming).
        sim_run_id = orchestrator._sim_run_id or (run.result or {}).get("sim_run_id")
        try:
            if orchestrator.event_bus is not None:
                await orchestrator.event_bus.emit_sim_stopping(
                    sim_run_id=sim_run_id, accepted=True,
                    reason="Stop received — terminating solver…",
                )
        except Exception as _ee:
            logger.warning(f"[STOP] Failed to emit sim_run_stopping ACK: {_ee}")

        # Drive the sim server directly — relying solely on the orchestrator
        # to pick up _stop_requested fails if it is outside the event-stream
        # loop (codegen, validation, between retries, reconstruction).
        if sim_run_id:
            sim_client = SimulationServerClient()
            try:
                await sim_client.stop_run(sim_run_id)
            except Exception as _ee:
                logger.warning(f"[STOP] sim_server.stop_run failed for {sim_run_id}: {_ee}")

        # Persist STOPPED so /continue can find the run in the correct state
        # even if the orchestrator never reaches a check point (pre-submit
        # stops are escalated to cancel by _check_cancelled, but the DB
        # still needs the STOPPED transition for the resume button to work).
        try:
            await store.update_run_status(run_uuid, RunStatus.STOPPED)
        except Exception as _ee:
            logger.warning(f"[STOP] Failed to persist STOPPED status: {_ee}")

        return {"run_id": run_id, "stopped": True, "sim_run_id": sim_run_id}

    # No active orchestrator on this worker.  This happens after a page
    # reload (the user is now on /ws/watch instead of /ws/run) or when a
    # different uvicorn worker holds the orchestrator.  Drive the sim
    # runner DIRECTLY using the persisted sim_run_id, then stream the
    # reconstruction events back to any /ws/watch subscribers so the
    # frontend sees the final results just like the happy path.
    sim_run_id = (run.result or {}).get("sim_run_id")
    watch_bus = get_watch_bus()

    if not sim_run_id:
        # Truly no idea what to do — mark the run stopped in DB and tell
        # the frontend.  Without sim_run_id we cannot reach the runner.
        await store.finalize_run(
            run_id=run_uuid,
            status=RunStatus.STOPPED,
            result={"info": "Stopped by user (no sim_run_id recorded)"},
        )
        # Push a synthetic run_stopped to any watchers so a stuck button
        # gets unstuck immediately.
        from datetime import datetime
        from simd_agent.models import AgentEvent, EventLevel
        seq = (await store.get_last_seq(run_uuid)) + 1
        ae = AgentEvent(
            run_id=run_uuid, seq=seq, ts=datetime.utcnow(),
            level=EventLevel.INFO, type="run_stopped",
            message="Stopped by user", payload={},
        )
        try:
            await store.append_event(ae)
        except Exception:
            pass
        watch_bus.publish_nowait(str(run_uuid), ae.to_ws_message())
        logger.info(f"[STOP] Run {run_id} marked stopped (no sim_run_id)")
        return {"run_id": run_id, "stopped": True}

    # Push an immediate ACK to watchers so the UI flips to "Stopping…"
    # regardless of how long the sim runner takes to reconstruct.
    from datetime import datetime
    from simd_agent.models import AgentEvent, EventLevel
    seq = (await store.get_last_seq(run_uuid)) + 1
    ack = AgentEvent(
        run_id=run_uuid, seq=seq, ts=datetime.utcnow(),
        level=EventLevel.INFO, type="sim_run_stopping",
        message="Stop received — terminating solver…",
        payload={"sim_run_id": sim_run_id, "accepted": True},
    )
    try:
        await store.append_event(ack)
    except Exception:
        pass
    watch_bus.publish_nowait(str(run_uuid), ack.to_ws_message())

    # Drive the sim runner.  Errors are non-fatal — even if the runner
    # never received our stop POST, the background streamer will still
    # finalize the run on its own once the solver process exits and the
    # runner emits artifacts_ready (sim runner side does reconstruction
    # autonomously based on its stopped_runs flag).
    sim_client = SimulationServerClient()
    try:
        await sim_client.stop_run(sim_run_id)
    except Exception as _ee:
        logger.warning(f"[STOP] sim_server.stop_run failed for {sim_run_id}: {_ee}")

    # Spawn a background task to relay reconstruct + artifacts_ready to
    # /ws/watch subscribers and finalize the run in the DB.  Uses the
    # same plumbing as continue (_stream_continued_run) — see main.py.
    asyncio.create_task(
        _stream_stop_to_completion(run_uuid, sim_run_id, store)
    )

    logger.info(f"[STOP] Run {run_id} stop driven externally (sim_run_id={sim_run_id})")
    return {"run_id": run_id, "stopped": True, "sim_run_id": sim_run_id}


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


async def _stream_stop_to_completion(
    run_id: UUID,
    sim_run_id: str,
    store: EventStore,
) -> None:
    """Background task: relay sim-runner reconstruction events to /ws/watch
    subscribers after a stop request was issued from a worker that does NOT
    own the orchestrator (page reload, multi-worker deployment).

    The sim runner kills the solver and ALWAYS proceeds to reconstruct +
    finalize on its own (see ``agent-simulation/app/runner.py:1163``), so
    even if this task fails halfway we still get partial results saved on
    disk.  The job of this task is purely to ferry the events back to the
    frontend so the UI does not look frozen on "Stopping…" forever.
    """
    from datetime import datetime
    from simd_agent.models import AgentEvent, EventLevel

    watch_bus = get_watch_bus()
    sim_client = SimulationServerClient()
    seq = (await store.get_last_seq(run_id)) + 1

    _MAP = {
        "run_started": "sim_run_started",
        "run_log": "sim_run_log",
        "run_stopping": "sim_run_stopping",
        "run_stopped": "sim_run_stopped",
        "run_succeeded": "sim_run_succeeded",
        "run_failed": "sim_run_failed",
        "artifacts_ready": "sim_artifacts_ready",
    }
    _SKIP = {"decompose_started", "decompose_complete", "decompose_failed",
             "decompose_log", "reconstruct_log", "run_log"}

    final_status = RunStatus.STOPPED
    summary = "Simulation stopped by user — partial results available"

    try:
        async for event in sim_client.stream_events(sim_run_id):
            if event.type in _SKIP:
                continue
            mapped = _MAP.get(event.type, event.type)
            ae = AgentEvent(
                run_id=run_id, seq=seq, ts=datetime.utcnow(),
                level=EventLevel.INFO, type=mapped,
                message=event.message, payload=event.payload,
            )
            seq += 1
            try:
                # Persist terminal-ish events so a fresh /ws/watch
                # reconnect can catch up; skip high-frequency progress.
                if event.type not in ("run_progress", "run_progress_batch"):
                    await store.append_event(ae)
            except Exception:
                pass
            watch_bus.publish_nowait(str(run_id), ae.to_ws_message())

            if event.type == "artifacts_ready":
                break
            if event.type == "run_failed":
                final_status = RunStatus.FAILED
                summary = event.message or "Simulation failed during stop"
                break

        # Final + terminal run_stopped if we didn't already see one
        run_event_type = (
            "run_stopped" if final_status == RunStatus.STOPPED else "run_failed"
        )
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

        final_ae = AgentEvent(
            run_id=run_id, seq=seq, ts=datetime.utcnow(),
            level=EventLevel.INFO, type="final",
            message=summary,
            payload={
                "run_id": str(run_id),
                "status": final_status.value,
                "validated_config": None,
                "artifacts": [],
                "iterations": 0,
                "retries": 0,
                "summary": summary,
                "case_type": None,
                "solver": None,
                "error": None,
            },
        )
        seq += 1
        try:
            await store.append_event(final_ae)
        except Exception:
            pass
        watch_bus.publish_nowait(str(run_id), final_ae.to_ws_message())

        await store.finalize_run(
            run_id=run_id,
            status=final_status,
            result={"sim_run_id": sim_run_id, "summary": summary},
        )
        logger.info("[STOP-RELAY] Run %s finalized as %s", run_id, final_status.value)

    except Exception as e:
        logger.error("[STOP-RELAY] Stream error for run %s: %s", run_id, e)
        # The sim runner reconstructs autonomously, so we still finalize as
        # STOPPED in the DB even if event relay failed — the user can
        # refresh and the run will be in the right state.
        try:
            await store.finalize_run(
                run_id=run_id,
                status=RunStatus.STOPPED,
                result={"sim_run_id": sim_run_id, "info": "Stop relay disconnected"},
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

# Per-run progress bus — the cache-fill pushes user-facing status updates here
# and the SSE endpoint /api/runs/{run_id}/vtk-progress drains them.  Each run
# has a single broadcast queue per subscriber so the frontend can attach late
# (e.g. on a page reload mid-fill) and still receive the current state.
_vtk_progress_subscribers: dict[str, list[asyncio.Queue]] = {}
_vtk_progress_last: dict[str, dict] = {}


def _vtk_progress_emit(run_id: str, payload: dict) -> None:
    """Broadcast a progress update for ``run_id`` to every SSE subscriber.

    Also stashes the latest payload so a subscriber that attaches AFTER the
    update was emitted immediately receives the current state — avoids a
    blank "Loading…" screen if the network call completes before the
    EventSource handshake.
    """
    _vtk_progress_last[run_id] = payload
    for q in _vtk_progress_subscribers.get(run_id, []):
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass


def _vtk_key(run_id: str, filename: str) -> str:
    """Build the storage key for a VTK result file."""
    return f"results/{run_id}/{filename}"


def _get_download_lock(run_id: str) -> asyncio.Lock:
    if run_id not in _vtk_download_locks:
        _vtk_download_locks[run_id] = asyncio.Lock()
    return _vtk_download_locks[run_id]


def _peek_vtp_sync(label: str, vtp_bytes: bytes) -> dict:
    """Parse VTP bytes and return a compact summary of what's inside.

    Used purely for diagnostics — the "white mesh" symptom in the UI
    means either the sim server returned a VTP with no field arrays,
    or our merge logic dropped them.  Calling this on the bytes after
    every download / merge tells us exactly where the data vanishes.
    """
    import tempfile
    import numpy as np
    import pyvista as pv

    if not vtp_bytes:
        return {"label": label, "empty": True}
    try:
        with tempfile.NamedTemporaryFile(suffix=".vtp", delete=False) as tf:
            tf.write(vtp_bytes)
            tf.flush()
            poly = pv.read(tf.name)
    except Exception as e:
        return {"label": label, "error": str(e), "bytes": len(vtp_bytes)}

    def _arr_summary(container) -> dict:
        out = {}
        for name in container.keys():
            arr = np.asarray(container[name])
            nc = int(arr.shape[1]) if arr.ndim > 1 else 1
            if arr.size == 0:
                out[name] = {"components": nc, "tuples": 0}
                continue
            if nc > 1:
                mag = np.linalg.norm(arr, axis=1)
                rng = [float(mag.min()), float(mag.max())]
            else:
                rng = [float(arr.min()), float(arr.max())]
            out[name] = {
                "components": nc,
                "tuples": int(arr.shape[0]),
                "range": rng,
                "sample": arr.flatten()[:8].tolist(),
            }
        return out

    return {
        "label": label,
        "bytes": len(vtp_bytes),
        "n_points": int(poly.n_points),
        "n_cells":  int(poly.n_cells),
        "point_arrays": _arr_summary(poly.point_data),
        "cell_arrays":  _arr_summary(poly.cell_data),
    }


def _merge_region_vtps_sync(
    region_vtps: list[tuple[str, bytes]],
) -> tuple[bytes, list[dict]]:
    """Combine per-region surface VTPs into a single VTP for the back-compat viewer.

    CHT cases ship one VTP per region per timestep; the sim server's
    ``/vtk/surface.vtp`` is single-region only.  Returning just the first
    region produces a flat slab in the UI (the empty-direction face of a
    2D case) with no field data — the "white rectangle" symptom.  Merging
    all regions yields the full geometry, with fluid fields (U, p, T, k,
    omega, nut) on the fluid regions and T on the solids.

    ``vtkAppendPolyData`` (via :func:`pyvista.PolyData.append_polydata`)
    zero-pads arrays missing from a region, so the merged dataset carries
    the union of all field names.  Field ranges are recomputed globally
    across the combined geometry so the colormap auto-scales sensibly.

    Returns:
        merged_vtp_bytes: combined VTP as raw XML bytes
        fields: per-field metadata in the shape the frontend expects
                ({name, num_components, range, location}).
    """
    import tempfile
    import numpy as np
    import pyvista as pv

    if not region_vtps:
        return b"", []

    with tempfile.TemporaryDirectory() as tmpdir:
        polys: list[pv.PolyData] = []
        for region_name, b in region_vtps:
            tmp = Path(tmpdir) / f"{region_name}.vtp"
            tmp.write_bytes(b)
            polys.append(pv.read(str(tmp)))

        if len(polys) == 1:
            merged = polys[0]
        else:
            # vtkAppendPolyData drops any array not present on every input,
            # so a fluid-only field (U, p, k, …) would vanish if we just
            # appended.  Pre-pad each region's PolyData with zero arrays
            # for the union of all array names, then append.
            def _components(arr) -> int:
                return int(arr.shape[1]) if arr.ndim > 1 else 1

            point_template: dict[str, tuple[int, np.dtype]] = {}
            cell_template:  dict[str, tuple[int, np.dtype]] = {}
            for poly in polys:
                for name in poly.point_data.keys():
                    arr = np.asarray(poly.point_data[name])
                    point_template.setdefault(name, (_components(arr), arr.dtype))
                for name in poly.cell_data.keys():
                    arr = np.asarray(poly.cell_data[name])
                    cell_template.setdefault(name, (_components(arr), arr.dtype))

            for poly in polys:
                for name, (nc, dtype) in point_template.items():
                    if name in poly.point_data:
                        continue
                    shape = (poly.n_points, nc) if nc > 1 else (poly.n_points,)
                    poly.point_data[name] = np.zeros(shape, dtype=dtype)
                for name, (nc, dtype) in cell_template.items():
                    if name in poly.cell_data:
                        continue
                    shape = (poly.n_cells, nc) if nc > 1 else (poly.n_cells,)
                    poly.cell_data[name] = np.zeros(shape, dtype=dtype)

            merged = polys[0].copy()
            for p in polys[1:]:
                merged = merged.append_polydata(p)

        # foamToVTK's extract_surface produces the same physical field on
        # both point_data and cell_data (e.g. ``T`` lives in both).  The
        # frontend keys its picker by name, so emitting both creates
        # duplicate keys (React warning) AND confuses the user with
        # apparent "twice the same property".  Dedupe here, preferring
        # point data — that's what vtk.js maps best for surface rendering.
        fields: list[dict] = []
        seen: set[str] = set()
        for location_name, container in (
            ("point", merged.point_data),
            ("cell",  merged.cell_data),
        ):
            for name in list(container.keys()):
                if name in seen:
                    continue
                # foamToVTK emits some bookkeeping arrays (vtkValidPointMask,
                # vtkGhostType, …) and the per-vector magnitude helper.  None
                # of these should appear in the field picker.
                if name.endswith("_magnitude") or name.startswith("vtk"):
                    continue
                arr = np.asarray(container[name])
                num_comp = int(arr.shape[1]) if arr.ndim > 1 else 1
                if arr.size == 0:
                    continue
                if num_comp > 1:
                    mag = np.linalg.norm(arr, axis=1)
                    rng = [float(mag.min()), float(mag.max())]
                else:
                    rng = [float(arr.min()), float(arr.max())]
                fields.append({
                    "name": name,
                    "num_components": num_comp,
                    "range": rng,
                    "location": location_name,
                })
                seen.add(name)

        out_path = Path(tmpdir) / "merged.vtp"
        merged.save(str(out_path))
        return out_path.read_bytes(), fields


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

        _vtk_progress_emit(run_id, {
            "phase": "fetch_index",
            "message": "Fetching simulation result index from the runner…",
        })

        sim = SimulationServerClient(base_url=settings.simulation_server_url)
        try:
            # ── 1. Try precomputed index (newer sim servers) ──────────────────
            sim_index: dict | None = None
            logger.info(f"[VTK] Fetching index for run {run_id} (sim={sim_run_id})")
            try:
                sim_index = await sim.get_precomputed_index(sim_run_id)
                if sim_index is not None:
                    raw_timesteps = sim_index.get("timesteps", [])
                    raw_regions   = sim_index.get("regions") or {}
                    # One-line summary; the full per-region breakdown
                    # goes to DEBUG.
                    logger.info(
                        "[VTK] sim-server index: fields=%s timesteps=%d regions=%s",
                        [f.get("name") for f in sim_index.get("fields", [])],
                        len(raw_timesteps),
                        list(raw_regions.keys()) or "NONE (single-region)",
                    )
                    for rname, rblock in raw_regions.items():
                        logger.debug(
                            "[VTK][DEBUG]   region %s: fields=%s timesteps=%s",
                            rname,
                            [f.get("name") for f in rblock.get("fields", [])],
                            [ts.get("time") for ts in rblock.get("timesteps", [])],
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
                logger.info(
                    "[VTK] fallback vtk_meta: keys=%s time=%s fields=%s",
                    list(vtk_meta.keys()), vtk_meta.get("time"),
                    [f.get("name") for f in vtk_meta.get("fields", [])],
                )
                if not vtk_meta.get("fields"):
                    logger.warning(
                        "[VTK] ⚠ sim server returned NO fields for run %s "
                        "— the white-mesh symptom comes from here. Check sim "
                        "server logs for foamToVTK / reconstruction issues.",
                        sim_run_id,
                    )
                    logger.debug("[VTK][DEBUG] Full vtk_meta: %s", vtk_meta)
                surface_bytes = await sim.download_surface_vtp(sim_run_id)
                vtp_peek = await asyncio.to_thread(_peek_vtp_sync, "fallback surface.vtp", surface_bytes)
                logger.info(
                    "[VTK] fallback surface.vtp: %d bytes, %d pts, %d cells, point=%s cell=%s",
                    vtp_peek.get("bytes", 0),
                    vtp_peek.get("n_points", 0), vtp_peek.get("n_cells", 0),
                    list((vtp_peek.get("point_arrays") or {}).keys()),
                    list((vtp_peek.get("cell_arrays")  or {}).keys()),
                )
                logger.debug("[VTK][DEBUG] fallback VTP full peek: %s", vtp_peek)
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

            # Detect multi-region BEFORE pulling surface.vtp — the sim
            # server's /vtk/surface.vtp is single-region only, so for CHT
            # cases we skip it and synthesise a merged surface from the
            # per-region last-timestep VTPs further down.
            sim_regions = sim_index.get("regions") or {}
            is_multi_region = bool(sim_regions)

            # Single-region: download the merged surface.vtp from the sim
            # server.  Some sim servers using the precomputed index don't
            # expose /vtk/surface.vtp — that's fine, the fallback at the
            # end of this function copies the last timestep into place.
            surface_bytes: bytes | None = None
            if not is_multi_region:
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
            #
            # Multi-region note: the sim server's index.json includes a
            # ``regions`` block when the case is CHT:
            #
            #     {
            #       "fields":    [...],     # back-compat = first region's
            #       "timesteps": [...],     # back-compat = first region's
            #       "regions": {
            #         "innerFluid": {"fields": [...], "timesteps": [...]},
            #         "outerFluid": {...},
            #         "wall":       {...}
            #       }
            #     }
            #
            # We download each region's VTPs under
            # ``timesteps/<region>/`` (preserved for future per-region UI)
            # AND merge them per timestep into a combined VTP at the
            # single-region path ``timesteps/<filename>.vtp``.  The
            # back-compat surface.vtp / timesteps endpoints then return
            # the merged dataset, which carries every region's geometry
            # and the union of all field arrays.

            async def _download_one_ts(
                i: int, total: int, ts: dict, region: str | None,
            ) -> tuple[int, float, str, bytes, list[dict]]:
                sim_filename: str = ts["filename"]
                time_float = float(ts["time"])
                step_fields: list[dict] = ts.get("fields", [])
                normalized_filename = f"t_{str(time_float).replace('.', '_')}.vtp"
                storage_subpath = (
                    f"timesteps/{region}/{normalized_filename}" if region
                    else f"timesteps/{normalized_filename}"
                )
                logger.info(
                    f"[VTK] [{i+1}/{total}] "
                    f"{'region=' + region + ' ' if region else ''}"
                    f"sim={sim_filename!r} time={time_float}"
                )
                vtp_bytes = await sim.download_precomputed_vtp(
                    sim_run_id, sim_filename, region=region,
                )
                # Peek into the downloaded bytes — the index claims certain
                # fields exist, but they only matter if they actually appear
                # in the VTP itself.  Full dump goes to DEBUG; INFO gets a
                # one-line summary so the terminal stays readable when 60+
                # VTPs land in one go (3 regions × 20 timesteps).
                vtp_peek = await asyncio.to_thread(
                    _peek_vtp_sync,
                    f"{region or 'single'}/t={time_float}",
                    vtp_bytes,
                )
                point_arrs = list((vtp_peek.get("point_arrays") or {}).keys())
                cell_arrs  = list((vtp_peek.get("cell_arrays")  or {}).keys())
                logger.info(
                    "[VTK] downloaded %s: %d bytes, %d pts, %d cells, point=%s cell=%s",
                    vtp_peek.get("label"), vtp_peek.get("bytes", 0),
                    vtp_peek.get("n_points", 0), vtp_peek.get("n_cells", 0),
                    point_arrs, cell_arrs,
                )
                logger.debug("[VTK][DEBUG] downloaded VTP full peek: %s", vtp_peek)
                await storage.upload(
                    _vtk_key(run_id, storage_subpath),
                    vtp_bytes, "application/xml",
                )
                return i, time_float, normalized_filename, vtp_bytes, step_fields

            local_regions: dict[str, dict] = {}
            last_vtp_bytes: bytes | None = None
            total_timesteps = 0

            if is_multi_region:
                # Run all region downloads concurrently — fan out across
                # all regions at once for the lowest possible end-to-end
                # latency on the post-run cache fill.
                tasks: list = []
                task_keys: list[tuple[str, int]] = []
                for region_name, region_block in sim_regions.items():
                    region_ts_list = region_block.get("timesteps", []) or []
                    for i, ts in enumerate(region_ts_list):
                        tasks.append(_download_one_ts(
                            i, len(region_ts_list), ts, region_name,
                        ))
                        task_keys.append((region_name, i))

                _vtk_progress_emit(run_id, {
                    "phase": "download",
                    "total": len(tasks),
                    "done":  0,
                    "message": f"Downloading {len(tasks)} VTPs from "
                               f"{len(sim_regions)} regions in parallel…",
                })
                results = await asyncio.gather(*tasks)
                _vtk_progress_emit(run_id, {
                    "phase": "download",
                    "total": len(tasks),
                    "done":  len(tasks),
                    "message": "Downloads complete — preparing merge",
                })

                # Group results back by region, preserving order
                grouped: dict[str, list] = {r: [] for r in sim_regions.keys()}
                for (region_name, _), r in zip(task_keys, results):
                    grouped[region_name].append(r)

                # Build the per-region index entries (preserved for any
                # future region-aware UI that wants to isolate one mesh).
                # Also group VTP bytes by time across regions for merging.
                by_time: dict[float, list[tuple[str, bytes]]] = {}
                for region_name, region_results in grouped.items():
                    region_results_sorted = sorted(
                        region_results, key=lambda r: r[0],
                    )
                    region_timesteps: list[dict] = []
                    for _, time_float, normalized_filename, vtp_bytes, step_fields in region_results_sorted:
                        entry: dict = {
                            "time": time_float,
                            "filename": normalized_filename,
                            "vtp_url": (
                                f"/api/runs/{run_id}/vtk-timestep/"
                                f"{time_float}/{region_name}/surface.vtp"
                            ),
                        }
                        if step_fields:
                            entry["fields"] = step_fields
                        region_timesteps.append(entry)
                        by_time.setdefault(time_float, []).append(
                            (region_name, vtp_bytes),
                        )
                    local_regions[region_name] = {
                        "fields":    sim_regions[region_name].get("fields", []),
                        "timesteps": region_timesteps,
                    }

                # ── Pipelined merge + upload ───────────────────────────────
                # Previously: ``await asyncio.gather(merges)`` followed by a
                # ``for ... await storage.upload`` loop.  The merges WERE
                # parallel but the upload loop serialised everything after,
                # adding ~1s per timestep (visible in logs as monotonic
                # 1.3-second gaps between "merged VTP for t=X" lines).
                # Now each timestep's merge + upload runs in its own task,
                # so 20 timesteps pipeline through the thread pool + GCS
                # connection pool concurrently.  Wall time drops from ~25 s
                # to a handful.
                times_sorted = sorted(by_time.keys())
                total = len(times_sorted)
                _vtk_progress_emit(run_id, {
                    "phase":   "merge",
                    "done":    0,
                    "total":   total,
                    "message": f"Merging {total} timesteps across "
                               f"{len(local_regions)} regions in parallel…",
                })

                done_count = 0
                done_lock = asyncio.Lock()

                async def _merge_and_upload(t: float) -> tuple[float, str, list[dict], bytes]:
                    """Merge one timestep across regions, then upload it.

                    Runs the CPU-bound pyvista work on a thread (releases the
                    GIL for VTK's C++ paths) and the upload as an async I/O
                    call.  Multiple instances of this coroutine race through
                    the event loop in parallel.
                    """
                    nonlocal done_count
                    merged_bytes, merged_fields = await asyncio.to_thread(
                        _merge_region_vtps_sync, by_time[t],
                    )
                    normalized_filename = f"t_{str(t).replace('.', '_')}.vtp"
                    await storage.upload(
                        _vtk_key(run_id, f"timesteps/{normalized_filename}"),
                        merged_bytes, "application/xml",
                    )
                    # Progress: bump counter under a lock so concurrent
                    # tasks don't race on the increment.
                    async with done_lock:
                        done_count += 1
                        _vtk_progress_emit(run_id, {
                            "phase":   "merge",
                            "done":    done_count,
                            "total":   total,
                            "message": f"Merged {done_count}/{total} timesteps",
                        })
                    return t, normalized_filename, merged_fields, merged_bytes

                # Fire all merges+uploads as parallel tasks.  asyncio.gather
                # preserves order so we don't need to sort again.
                pipeline_results = await asyncio.gather(*[
                    _merge_and_upload(t) for t in times_sorted
                ])

                local_timesteps = []
                last_merged_fields: list[dict] = []
                for t, normalized_filename, merged_fields, merged_bytes in pipeline_results:
                    local_timesteps.append({
                        "time": t,
                        "filename": normalized_filename,
                        "vtp_url": f"/api/runs/{run_id}/vtk-timestep/{t}/surface.vtp",
                        "fields": merged_fields,
                    })
                    last_vtp_bytes = merged_bytes
                    last_merged_fields = merged_fields

                top_level_fields = last_merged_fields
                total_timesteps = len(local_timesteps)
                logger.info(
                    f"[VTK] multi-region cache: "
                    + ", ".join(
                        f"{name}({len(b['timesteps'])}ts)"
                        for name, b in local_regions.items()
                    )
                    + f" → merged into {total_timesteps} combined timestep(s)"
                )
            else:
                # Single-region path — unchanged behaviour.
                raw_ts_list = sim_index.get("timesteps", [])
                logger.info(
                    f"[VTK] About to download {len(raw_ts_list)} timesteps in parallel: "
                    + str([ts.get("time") for ts in raw_ts_list])
                )
                ts_results = await asyncio.gather(*[
                    _download_one_ts(i, len(raw_ts_list), ts, None)
                    for i, ts in enumerate(raw_ts_list)
                ])
                ts_results_sorted = sorted(ts_results, key=lambda r: r[0])
                local_timesteps = []
                for _, time_float, normalized_filename, vtp_bytes, step_fields in ts_results_sorted:
                    last_vtp_bytes = vtp_bytes
                    entry = {
                        "time": time_float,
                        "filename": normalized_filename,
                        "vtp_url": f"/api/runs/{run_id}/vtk-timestep/{time_float}/surface.vtp",
                    }
                    if step_fields:
                        entry["fields"] = step_fields
                    local_timesteps.append(entry)
                top_level_fields = sim_index.get("fields", [])
                total_timesteps = len(local_timesteps)
                logger.info(
                    f"[VTK] Saved {len(local_timesteps)} timesteps: "
                    + str([ts["time"] for ts in local_timesteps])
                )

            # If surface.vtp was unavailable, use the last (highest-time) timestep.
            # Multi-region always lands here because we skipped the sim-server
            # surface.vtp download above — the merged VTP is the surface.
            if surface_bytes is None and local_timesteps and last_vtp_bytes:
                sorted_ts = sorted(local_timesteps, key=lambda t: float(t["time"]))
                last_ts = sorted_ts[-1]
                last_storage_subpath = f"timesteps/{last_ts['filename']}"
                last_ts_data = await storage.download(_vtk_key(run_id, last_storage_subpath))
                if last_ts_data:
                    await storage.upload(_vtk_key(run_id, "surface.vtp"), last_ts_data, "application/xml")
                    logger.info(f"[VTK] Used last timestep (t={last_ts['time']}) as surface.vtp")

            # ── 4. Write index ────────────────────────────────────────────────
            local_index: dict = {
                "run_id":      run_id,
                "sim_run_id":  sim_run_id,
                "total":       total_timesteps,
                "fields":      top_level_fields,
                "surface_vtp": f"/api/runs/{run_id}/vtk/surface.vtp",
                "timesteps":   local_timesteps,
            }
            if is_multi_region:
                local_index["regions"] = local_regions
            await storage.upload(index_key, json.dumps(local_index).encode())
            logger.info(
                f"[VTK] Cached {total_timesteps} VTP(s) for run {run_id}"
                f"{' across ' + str(len(local_regions)) + ' regions' if is_multi_region else ''}"
            )
            _vtk_progress_emit(run_id, {
                "phase":   "done",
                "total":   total_timesteps,
                "done":    total_timesteps,
                "message": f"Ready — {total_timesteps} timestep(s) cached",
            })
            return local_index

        except SimulationServerError as e:
            _vtk_progress_emit(run_id, {
                "phase":   "error",
                "message": f"Simulation server error: {e}",
            })
            raise HTTPException(status_code=502, detail=f"Simulation server error: {e}")
        finally:
            await sim.close()


# ── VTK Results ──────────────────────────────────────────────────────────────

@app.get("/api/runs/{run_id}/vtk-progress")
async def vtk_progress_stream(run_id: str, request: Request):
    """Server-Sent Events stream of cache-fill progress for ``run_id``.

    The frontend opens this alongside ``/vtk-results`` so the user sees
    "Downloading 60 VTPs…", "Merged 12/20 timesteps…", "Ready" instead of
    a blank loading box while the agent pipelines downloads + per-region
    merges + GCS uploads.  Each event has shape::

        {"phase": "download" | "merge" | "done" | "error",
         "done": int, "total": int, "message": str}

    Streaming protocol: SSE, one ``data:`` line per event, ``\\n\\n``
    delimited.  The client is responsible for closing on phase=done.
    """
    queue: asyncio.Queue = asyncio.Queue(maxsize=64)
    _vtk_progress_subscribers.setdefault(run_id, []).append(queue)

    # Replay the most recent event so the frontend immediately knows the
    # current phase — important for late subscribers (page reload mid-fill,
    # SSE handshake races the producer).
    if run_id in _vtk_progress_last:
        try:
            queue.put_nowait(_vtk_progress_last[run_id])
        except asyncio.QueueFull:
            pass

    async def _stream():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    # Heartbeat so intermediaries don't reap the connection.
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("phase") in ("done", "error"):
                    break
        finally:
            subs = _vtk_progress_subscribers.get(run_id, [])
            if queue in subs:
                subs.remove(queue)

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "Connection":        "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/runs/{run_id}/vtk-debug")
async def vtk_debug(run_id: str) -> dict[str, Any]:
    """Diagnostic dump of the cached VTK index + a peek into surface.vtp.

    Use this when the 3D viewer renders a white mesh: the response tells
    you whether the cached index has fields, whether the merged surface
    VTP actually carries the simulation arrays, or whether the run went
    through the fallback path (no precomputed index).
    """
    from simd_agent.storage import get_storage
    storage = get_storage()

    index_data = await storage.download(_vtk_key(run_id, "index.json"))
    index = json.loads(index_data) if index_data else None

    surface_bytes = await storage.download(_vtk_key(run_id, "surface.vtp"))
    surface_peek = (
        await asyncio.to_thread(_peek_vtp_sync, "surface.vtp", surface_bytes)
        if surface_bytes else None
    )

    return {
        "run_id": run_id,
        "index": index,
        "surface_vtp_peek": surface_peek,
    }


@app.post("/api/runs/{run_id}/vtk-clear-cache")
async def vtk_clear_cache(run_id: str) -> dict[str, Any]:
    """Delete the cached VTK index + surface so the next /vtk-results call
    re-downloads everything from the sim server.  Useful after fixing a
    bad cache (e.g. one created before the multi-region merge fix)."""
    from simd_agent.storage import get_storage
    storage = get_storage()
    # Delete the index — _ensure_vtk_cached re-runs when it's gone.
    deleted = []
    for sub in ("index.json", "surface.vtp"):
        try:
            await storage.delete(_vtk_key(run_id, sub))
            deleted.append(sub)
        except Exception as e:
            logger.warning(f"[VTK] clear-cache: could not delete {sub}: {e}")
    return {"run_id": run_id, "deleted": deleted}


def _dedupe_fields(fields: list[dict]) -> list[dict]:
    """Drop duplicate field entries that differ only by point/cell location.

    foamToVTK + extract_surface emit the same physical field on both
    ``point_data`` and ``cell_data``, so a raw field list contains
    ``[{name: T, location: point}, {name: T, location: cell}]``.  React
    keys collide and the picker shows duplicates.  Prefer the point
    entry — that's what vtk.js maps best for surface rendering.
    """
    seen: set[str] = set()
    out: list[dict] = []
    for f in fields:
        name = f.get("name")
        if not name or name in seen:
            continue
        out.append(f)
        seen.add(name)
    return out


def _absolute_region_block(base: str, region_block: dict) -> dict:
    """Return a copy of a region block with relative ``vtp_url``s rewritten
    to absolute URLs against the agent's base URL.  Same shape as the
    sim server's index but ready for the frontend to fetch."""
    out: dict[str, Any] = {
        "fields": _dedupe_fields(region_block.get("fields", []) or []),
        "timesteps": [],
    }
    for ts in region_block.get("timesteps", []):
        entry = dict(ts)
        entry["fields"] = _dedupe_fields(ts.get("fields", []) or [])
        if "vtp_url" in entry and not str(entry["vtp_url"]).startswith("http"):
            entry["vtp_url"] = f"{base}{entry['vtp_url']}"
        out["timesteps"].append(entry)
    # Last frame URL for the viewer's initial render.
    if out["timesteps"]:
        last = max(out["timesteps"], key=lambda t: float(t["time"]))
        out["last_vtp_url"] = last.get("vtp_url")
        out["last_time"] = float(last["time"])
    return out


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
            # multi-region only:
            "regions":         {"<name>": {fields, timesteps, last_vtp_url, last_time}, ...}
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

    response: dict[str, Any] = {
        "run_id": run_id,
        "time": last_time,
        "vtp_url": last_vtp_url,
        "fields": _dedupe_fields(local_index.get("fields", []) or []),
        "total_timesteps": len(timesteps),
    }

    raw_regions = local_index.get("regions")
    if raw_regions:
        response["regions"] = {
            name: _absolute_region_block(base, block)
            for name, block in raw_regions.items()
        }
    return response


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
        entry: dict = {
            "time": ts["time"],
            "vtp_url": f"{base}{ts['vtp_url']}",
        }
        if "fields" in ts:
            entry["fields"] = _dedupe_fields(ts["fields"] or [])
        timesteps.append(entry)
    response: dict[str, Any] = {
        "run_id": run_id,
        "total": local_index.get("total", len(timesteps)),
        "fields": _dedupe_fields(local_index.get("fields", []) or []),
        "timesteps": timesteps,
    }
    raw_regions = local_index.get("regions")
    if raw_regions:
        response["regions"] = {
            name: _absolute_region_block(base, block)
            for name, block in raw_regions.items()
        }
    return response


# ── Per-timestep VTP ─────────────────────────────────────────────────────────

@app.get("/api/runs/{run_id}/vtk-timestep/{time_str}/surface.vtp")
async def serve_timestep_vtp(run_id: str, time_str: str):
    """Serve a single-region timestep VTP from object storage.

    Backward-compat path: cached at ``timesteps/<filename>.vtp``.  For
    multi-region cases, callers should use the per-region endpoint
    :func:`serve_region_timestep_vtp` below — that's the URL emitted in
    the multi-region branch of ``_ensure_vtk_cache``.
    """
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


@app.get("/api/runs/{run_id}/vtk-timestep/{time_str}/{region}/surface.vtp")
async def serve_region_timestep_vtp(run_id: str, time_str: str, region: str):
    """Serve a per-region timestep VTP from object storage (CHT cases).

    Multi-region cases cache each region's VTPs under the storage prefix
    ``timesteps/<region>/`` so they never collide with other regions.
    The local index built by ``_ensure_vtk_cache`` emits URLs in this
    exact shape:

        /api/runs/{run_id}/vtk-timestep/{time}/{region}/surface.vtp
    """
    from simd_agent.storage import get_storage

    # Mirror the sim server's filename convention — replace '.' with '_'
    # so 1.5 → t_1_5.vtp, 200 → t_200.vtp.
    filename = f"t_{time_str.replace('.', '_')}.vtp"
    key = _vtk_key(run_id, f"timesteps/{region}/{filename}")
    data = await get_storage().download(key)

    if data is None:
        # Trigger cache fill then retry — covers the case where the
        # client polls the URL before the post-run download finishes.
        await _ensure_vtk_cached(run_id)
        data = await get_storage().download(key)

    if data is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"region={region!r} t={time_str!r} VTP not found — "
                "ensure /vtk-results has been called and the case is multi-region"
            ),
        )

    from fastapi.responses import Response
    return Response(
        content=data,
        media_type="application/xml",
        headers={
            "Content-Disposition": f'inline; filename="{region}_{filename}"',
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
