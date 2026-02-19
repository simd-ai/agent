"""
API route definitions for Simulation Server.

UPDATED: Now supports external mesh files (.msh, .cas, etc.) that need conversion.
"""

import asyncio
import json
import shutil
import subprocess
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse, StreamingResponse

from app.config import RUNS_DIR, build_shell_command
from app.models import RunInfo, RunMode, RunStatus
from app.store import delete_run as store_delete_run
from app.store import emit_event, get_run, register_run, run_events, runs
from app.openfoam import collect_artifacts, detect_solver
from app.runner import run_simulation

router = APIRouter()


# ── Mesh Detection Helper ────────────────────────────────────

def detect_mesh_file(case_dir: Path) -> Tuple[Optional[Path], Optional[str]]:
    """
    Detect if there's an external mesh file that needs conversion.
    
    Returns (mesh_path, mesh_format) or (None, None) if no external mesh.
    """
    mesh_extensions = [".msh", ".cas", ".cgns", ".unv", ".neu"]
    
    for ext in mesh_extensions:
        matches = list(case_dir.glob(f"*{ext}"))
        if matches:
            return matches[0], ext.lstrip(".")
    
    return None, None


# ── Health ───────────────────────────────────────────────────

@router.get("/health")
async def health():
    """Health check — also verifies OpenFOAM is reachable."""
    foam_available = False
    foam_version: Optional[str] = None

    try:
        health_cmd = build_shell_command("simpleFoam -help 2>&1 | head -5")
        result = subprocess.run(
            health_cmd,
            shell=True,
            capture_output=True,
            timeout=10,
        )
        foam_available = result.returncode == 0
        if foam_available:
            output = result.stdout.decode()
            if "OpenFOAM" in output:
                foam_version = output.split("\n")[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return {
        "status": "healthy" if foam_available else "degraded",
        "openfoam_available": foam_available,
        "openfoam_version": foam_version,
        "runs_dir": str(RUNS_DIR),
        "active_runs": sum(
            1 for r in runs.values() if r.status == RunStatus.RUNNING
        ),
    }


# ── Submit run ───────────────────────────────────────────────

@router.post("/api/run")
async def submit_run(
    background_tasks: BackgroundTasks,
    case_zip: UploadFile = File(
        ..., description="ZIP file containing OpenFOAM case"
    ),
    mode: RunMode = Form(
        RunMode.FULL,
        description="'test' for 1 iteration, 'full' for complete run",
    ),
    run_id: Optional[str] = Form(
        None, description="Optional run ID (generated if omitted)"
    ),
    callback_url: Optional[str] = Form(
        None, description="URL to POST final status to"
    ),
):
    """Submit an OpenFOAM case for execution."""
    if not run_id:
        run_id = f"run-{uuid.uuid4().hex[:12]}"

    run_dir = RUNS_DIR / run_id
    case_dir = run_dir / "case"
    run_dir.mkdir(parents=True, exist_ok=True)

    run_events[run_id] = []

    zip_path = run_dir / "case.zip"
    content = await case_zip.read()
    zip_path.write_bytes(content)

    emit_event(
        run_id,
        "extract_started",
        "Extracting case files…",
        payload={"zip_size_bytes": len(content)},
    )

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(case_dir)

        subdirs = list(case_dir.iterdir())
        if len(subdirs) == 1 and subdirs[0].is_dir():
            nested = subdirs[0]
            for item in nested.iterdir():
                shutil.move(str(item), str(case_dir / item.name))
            nested.rmdir()

        # Validate case structure
        required = ["0", "constant", "system"]
        missing = [d for d in required if not (case_dir / d).exists()]
        if missing:
            raise ValueError(f"Invalid case: missing directories: {missing}")

        # ──────────────────────────────────────────────────────────────
        # KEY FIX: Accept external mesh files, not just polyMesh
        # ──────────────────────────────────────────────────────────────
        poly_mesh_exists = (case_dir / "constant" / "polyMesh").exists()
        block_mesh_exists = (case_dir / "system" / "blockMeshDict").exists()
        external_mesh, mesh_format = detect_mesh_file(case_dir)
        
        if not poly_mesh_exists and not block_mesh_exists and not external_mesh:
            raise ValueError(
                "Invalid case: no mesh source found. Need one of: "
                "constant/polyMesh, system/blockMeshDict, or external mesh file (.msh, .cas, etc.)"
            )
        
        mesh_source = (
            "polyMesh" if poly_mesh_exists else
            f"gmsh:{external_mesh.name}" if external_mesh else
            "blockMeshDict"
        )

        emit_event(
            run_id,
            "extract_complete",
            "Case files extracted",
            payload={
                "case_dir": str(case_dir),
                "solver": detect_solver(case_dir),
                "mesh_source": mesh_source,
            },
        )

    except zipfile.BadZipFile as exc:
        emit_event(run_id, "run_failed", f"Invalid ZIP file: {exc}", level="error")
        raise HTTPException(status_code=400, detail=f"Invalid ZIP file: {exc}")
    except Exception as exc:
        emit_event(run_id, "run_failed", f"Extraction failed: {exc}", level="error")
        raise HTTPException(status_code=400, detail=f"Invalid case ZIP: {exc}")

    run_info = RunInfo(
        run_id=run_id,
        status=RunStatus.PENDING,
        mode=mode,
        created_at=datetime.utcnow().isoformat() + "Z",
    )
    register_run(run_info)

    background_tasks.add_task(run_simulation, run_id, case_dir, mode)

    return {
        "run_id": run_id,
        "status": "pending",
        "mode": mode.value,
        "events_url": f"/api/run/{run_id}/events",
        "status_url": f"/api/run/{run_id}/status",
    }


@router.post("/api/run/test")
async def submit_test_run(
    background_tasks: BackgroundTasks,
    case_zip: UploadFile = File(...),
    run_id: Optional[str] = Form(None),
    callback_url: Optional[str] = Form(None),
):
    """Shortcut for test mode (1-iteration validation)."""
    return await submit_run(
        background_tasks=background_tasks,
        case_zip=case_zip,
        mode=RunMode.TEST,
        run_id=run_id,
        callback_url=callback_url,
    )


@router.get("/api/run/{run_id}/status")
async def get_run_status(run_id: str):
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run.model_dump()


@router.get("/api/run/{run_id}/events")
async def stream_events(run_id: str):
    """SSE stream for real-time run progress."""
    if run_id not in runs and run_id not in run_events:
        raise HTTPException(status_code=404, detail="Run not found")

    async def event_generator():
        last_seq = 0
        timeout_counter = 0.0
        max_timeout = 3600

        while timeout_counter < max_timeout:
            events = run_events.get(run_id, [])

            for event in events[last_seq:]:
                yield f"data: {event.model_dump_json()}\n\n"
                last_seq = event.seq
                timeout_counter = 0

            run = get_run(run_id)
            if run and run.status in (RunStatus.SUCCEEDED, RunStatus.FAILED):
                yield f"data: {json.dumps({'type': 'stream_end', 'run_id': run_id, 'status': run.status.value})}\n\n"
                break

            await asyncio.sleep(0.3)
            timeout_counter += 0.3

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/api/run/{run_id}/artifacts")
async def list_artifacts(run_id: str):
    if get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="Run not found")
    case_dir = RUNS_DIR / run_id / "case"
    return {"artifacts": collect_artifacts(run_id, case_dir)}


@router.get("/api/run/{run_id}/artifacts/{file_path:path}")
async def download_artifact(run_id: str, file_path: str):
    if get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="Run not found")
    full_path = RUNS_DIR / run_id / "case" / file_path
    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(full_path, filename=Path(file_path).name)


@router.delete("/api/run/{run_id}")
async def delete_run_endpoint(run_id: str):
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status == RunStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Cannot delete a running simulation")
    run_dir = RUNS_DIR / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    store_delete_run(run_id)
    return {"deleted": run_id}
