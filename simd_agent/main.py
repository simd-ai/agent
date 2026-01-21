# simd_agent/main.py
"""FastAPI application with WebSocket endpoint for CFD workflow orchestration."""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from simd_agent.db import close_db, init_db

# Try to import mesh module (may fail on some Python versions due to VTK compatibility)
MESH_ENABLED = False
try:
    from simd_agent.mesh import mesh_router, STORAGE_DIR
    MESH_ENABLED = True
except ImportError as e:
    logging.warning(f"Mesh module not available: {e}")
    mesh_router = None
    STORAGE_DIR = None
from simd_agent.event_bus import EventBus
from simd_agent.models import (
    AgentEvent,
    EventLevel,
    EventTypes,
    RunStatus,
    StartRequest,
)
from simd_agent.orchestration import Orchestrator
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
    
    try:
        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        # Continue anyway - DB might already exist
    
    yield
    
    # Shutdown
    logger.info("Shutting down simd_agent service...")
    await close_db()


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
        "precheck": "/api/precheck",
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
    
    Request:
        {
            "prompt": "Simulate turbulent flow through a pipe...",
            "mesh": { ... },  // Optional mesh info
            "previousConfig": { ... }  // Optional previous config
        }
    
    Response:
        {
            "success": true,
            "suggestedConfig": { ... },
            "boundaryHints": { ... },
            "interpretation": { ... },
            "confidence": { ... },
            ...
        }
    """
    from simd_agent.precheck import PrecheckRequest, get_precheck_service
    
    try:
        # Parse and validate request
        precheck_request = PrecheckRequest(**request)
        
        # Get precheck service and analyze
        service = get_precheck_service()
        response = await service.analyze(precheck_request)
        
        return response.model_dump()
        
    except ValidationError as e:
        logger.warning(f"Invalid precheck request: {e}")
        return {
            "success": False,
            "errors": [f"Invalid request: {e}"],
            "suggestedConfig": {
                "flowRegime": "turbulent",
                "timeScheme": "steady",
                "compressibility": "incompressible",
                "enableHeatTransfer": False,
                "maxIterations": 1000,
                "convergenceCriteria": 1e-6,
            },
            "interpretation": {
                "summary": "Request validation failed",
                "simulationType": "Unknown",
                "keyPhysics": [],
                "assumptions": [],
            },
            "confidence": {
                "overall": 0.0,
                "flowRegime": 0.0,
                "boundaryConditions": 0.0,
                "physicsSettings": 0.0,
            },
            "nextStep": 1,
            "shouldShowMeshViewer": False,
        }
    except Exception as e:
        logger.exception(f"Precheck failed: {e}")
        return {
            "success": False,
            "errors": [f"Internal error: {e}"],
            "suggestedConfig": {
                "flowRegime": "turbulent",
                "timeScheme": "steady",
                "compressibility": "incompressible",
                "enableHeatTransfer": False,
                "maxIterations": 1000,
                "convergenceCriteria": 1e-6,
            },
            "interpretation": {
                "summary": "Analysis failed due to internal error",
                "simulationType": "Unknown",
                "keyPhysics": [],
                "assumptions": [],
            },
            "confidence": {
                "overall": 0.0,
                "flowRegime": 0.0,
                "boundaryConditions": 0.0,
                "physicsSettings": 0.0,
            },
            "nextStep": 1,
            "shouldShowMeshViewer": False,
        }


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
    
    try:
        # Receive start request
        try:
            data = await asyncio.wait_for(
                websocket.receive_json(),
                timeout=30.0,  # 30 second timeout for initial message
            )
        except asyncio.TimeoutError:
            await websocket.send_json({
                "error": "Timeout waiting for start request",
                "type": "error",
            })
            await websocket.close(code=1008)
            return
        
        # Parse request
        try:
            request = StartRequest(**data)
        except ValidationError as e:
            await websocket.send_json({
                "error": f"Invalid start request: {e}",
                "type": "error",
            })
            await websocket.close(code=1003)
            return
        
        # Create event bus
        event_bus = EventBus(
            run_id=run_id,
            websocket=websocket,
            store=store,
            persist=True,
        )
        
        # Create run in database
        try:
            await store.create_run(
                op=request.op,
                provider=request.provider,
                prompt_pack=request.prompt_pack,
                user_requirements=request.user_requirements,
                simulation_config=request.simulation_config,
                run_id=run_id,
            )
        except Exception as e:
            logger.error(f"Failed to create run in database: {e}")
            # Continue anyway - event streaming still works
        
        # Update run status
        try:
            await store.update_run_status(run_id, RunStatus.RUNNING)
        except Exception as e:
            logger.warning(f"Failed to update run status: {e}")
        
        # Create and run orchestrator
        orchestrator = Orchestrator(
            run_id=run_id,
            event_bus=event_bus,
            store=store,
            request=request,
        )
        
        # Start heartbeat task
        heartbeat_task = asyncio.create_task(
            _heartbeat_loop(websocket, get_settings().ws_heartbeat_interval)
        )
        
        try:
            # Execute the workflow
            result = await orchestrator.run()
            
            # Final event already sent by orchestrator
            logger.info(f"Run {run_id} completed with status: {result.status}")
            
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
