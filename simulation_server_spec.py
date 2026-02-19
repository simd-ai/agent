"""
Simulation Runner Server — FastAPI + OpenFOAM
=============================================

COMPLETE SPECIFICATION

This server:
1. Receives a ZIP of OpenFOAM case files from the main backend
2. Extracts them into a run directory
3. Runs OpenFOAM (test=1 iteration, or full run)
4. Streams events back via SSE (Server-Sent Events)
5. Returns artifacts (logs, results) when done

OpenFOAM is already installed on this machine.

To run:
    uvicorn simulation_server_spec:app --host 0.0.0.0 --port 9000 --reload
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from enum import Enum
from typing import Optional
import asyncio
import uuid
import os
import shutil
import zipfile
import subprocess
import json
import time
import re
from pathlib import Path
from datetime import datetime

app = FastAPI(title="SIMD Simulation Runner", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Configuration
# ============================================================

RUNS_DIR = Path(os.environ.get("RUNS_DIR", "/tmp/simd-runs"))
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# OpenFOAM environment (adjust for your installation)
OPENFOAM_ENV = os.environ.get("OPENFOAM_ENV", "/opt/openfoam/etc/bashrc")


# ============================================================
# Models
# ============================================================

class RunStatus(str, Enum):
    PENDING = "pending"
    EXTRACTING = "extracting"
    MESHING = "meshing"      # If blockMesh/snappyHexMesh needed
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class RunMode(str, Enum):
    TEST = "test"       # 1 iteration only — validates case compiles & runs
    FULL = "full"       # Full simulation run


class RunEvent(BaseModel):
    run_id: str
    seq: int
    ts: str
    type: str           # See event types below
    level: str          # "info" | "warn" | "error"
    message: str
    payload: dict = {}


class RunInfo(BaseModel):
    run_id: str
    status: RunStatus
    mode: RunMode
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    exit_code: Optional[int] = None
    error: Optional[str] = None
    duration_seconds: Optional[float] = None


# In-memory run tracking (use Redis/DB for production)
runs: dict[str, RunInfo] = {}
run_events: dict[str, list[RunEvent]] = {}


# ============================================================
# Event Types the server emits
# ============================================================
#
# "extract_started"   — ZIP extraction began
# "extract_complete"  — ZIP extracted, case directory ready
# "mesh_started"      — blockMesh/checkMesh started (if needed)
# "mesh_complete"     — Mesh generation done
# "run_started"       — OpenFOAM solver started
# "run_progress"      — Iteration/timestep progress (parsed from log)
# "run_log"           — Raw stdout/stderr chunk from solver
# "run_succeeded"     — Solver finished with exit code 0
# "run_failed"        — Solver finished with non-zero exit code
# "artifacts_ready"   — Post-processing done, results available
#
# For "run_progress" payload:
# {
#   "iteration": 42,
#   "time": 0.042,           # for transient
#   "residuals": {
#     "Ux": 1.2e-4,
#     "Uy": 3.1e-5,
#     "p": 2.8e-3,
#     "k": 1.1e-4,           # if turbulent
#     "omega": 5.2e-5        # if turbulent
#   },
#   "continuity_error": 1.5e-6
# }


# ============================================================
# Helpers
# ============================================================

def emit_event(
    run_id: str,
    event_type: str,
    message: str,
    level: str = "info",
    payload: dict = None,
) -> RunEvent:
    """Append an event to the run's event list."""
    if payload is None:
        payload = {}
    seq = len(run_events.get(run_id, [])) + 1
    event = RunEvent(
        run_id=run_id,
        seq=seq,
        ts=datetime.utcnow().isoformat() + "Z",
        type=event_type,
        level=level,
        message=message,
        payload=payload,
    )
    run_events.setdefault(run_id, []).append(event)
    return event


def parse_openfoam_log_line(line: str) -> Optional[dict]:
    """
    Parse a single line of OpenFOAM solver output.
    Returns a dict with iteration/residual info, or None if not parseable.

    Typical OpenFOAM log lines:
      smoothSolver:  Solving for Ux, Initial residual = 0.123, Final residual = 0.001, No Iterations 5
      GAMG:  Solving for p, Initial residual = 0.456, Final residual = 0.0001, No Iterations 12
      Time = 0.001
      ExecutionTime = 2.34 s
    """
    result = {}

    # Match "Solving for <field>" lines
    solving_match = re.search(
        r"Solving for (\w+),.*Final residual = ([\d.eE+-]+)",
        line
    )
    if solving_match:
        result["field"] = solving_match.group(1)
        try:
            result["residual"] = float(solving_match.group(2))
        except ValueError:
            pass

    # Match "Time = X" lines (transient)
    time_match = re.search(r"^Time = ([\d.eE+-]+)", line.strip())
    if time_match:
        try:
            result["time"] = float(time_match.group(1))
        except ValueError:
            pass

    # Match continuity error
    continuity_match = re.search(
        r"time step continuity errors.*sum local = ([\d.eE+-]+)",
        line
    )
    if continuity_match:
        try:
            result["continuity_error"] = float(continuity_match.group(1))
        except ValueError:
            pass

    return result if result else None


def detect_solver(case_dir: Path) -> str:
    """
    Detect which OpenFOAM solver to use from controlDict.
    Falls back to 'simpleFoam'.
    """
    control_dict = case_dir / "system" / "controlDict"
    if control_dict.exists():
        content = control_dict.read_text()
        for line in content.splitlines():
            if line.strip().startswith("application"):
                solver = line.split()[-1].rstrip(";").strip()
                return solver
    return "simpleFoam"


def detect_time_scheme(case_dir: Path) -> str:
    """Detect if the case is steady or transient."""
    control_dict = case_dir / "system" / "controlDict"
    if control_dict.exists():
        content = control_dict.read_text().lower()
        if "pimple" in content or "piso" in content:
            return "transient"
        # Check deltaT
        for line in content.splitlines():
            if "deltat" in line.lower():
                try:
                    dt = float(line.split()[-1].rstrip(";"))
                    if dt < 1:  # Likely transient
                        return "transient"
                except ValueError:
                    pass
    return "steady"


async def run_openfoam_command(
    run_id: str,
    cmd: list[str],
    case_dir: Path,
    event_prefix: str = "run",
) -> tuple[int, str]:
    """
    Run an OpenFOAM command and stream its output.
    
    Returns (exit_code, stderr_text)
    """
    # Source OpenFOAM environment and run command
    full_cmd = f"source {OPENFOAM_ENV} && " + " ".join(cmd)
    
    process = await asyncio.create_subprocess_shell(
        full_cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(case_dir),
    )

    iteration = 0
    residuals: dict[str, float] = {}
    current_time: float | None = None

    # Stream stdout line by line
    async for raw_line in process.stdout:
        line = raw_line.decode("utf-8", errors="replace").rstrip()

        # Emit raw log (limit frequency for large outputs)
        if iteration % 10 == 0 or "error" in line.lower() or "warning" in line.lower():
            emit_event(run_id, f"{event_prefix}_log", line, payload={"stream": "stdout"})

        # Parse for progress
        parsed = parse_openfoam_log_line(line)
        if parsed:
            if "field" in parsed and "residual" in parsed:
                residuals[parsed["field"]] = parsed["residual"]
            if "time" in parsed:
                current_time = parsed["time"]
                iteration += 1
                emit_event(
                    run_id,
                    f"{event_prefix}_progress",
                    f"Iteration {iteration}",
                    payload={
                        "iteration": iteration,
                        "time": current_time,
                        "residuals": dict(residuals),
                        "continuity_error": parsed.get("continuity_error"),
                    },
                )
                residuals.clear()

    # Wait for process to finish
    await process.wait()

    # Capture stderr
    stderr_data = await process.stderr.read()
    stderr_text = stderr_data.decode("utf-8", errors="replace") if stderr_data else ""
    
    if stderr_text:
        emit_event(
            run_id, f"{event_prefix}_log", stderr_text[:2000],
            level="warn", payload={"stream": "stderr"}
        )

    return process.returncode, stderr_text


def detect_mesh_file(case_dir: Path) -> tuple[Path | None, str | None]:
    """
    Detect if there's an external mesh file that needs conversion.
    
    Returns (mesh_path, mesh_format) or (None, None) if no external mesh.
    """
    # Common mesh file extensions to look for
    mesh_extensions = [".msh", ".cas", ".cgns", ".unv", ".neu"]
    
    for ext in mesh_extensions:
        matches = list(case_dir.glob(f"*{ext}"))
        if matches:
            return matches[0], ext.lstrip(".")
    
    return None, None


def get_mesh_converter(mesh_format: str) -> str:
    """Get the OpenFOAM mesh converter command for a given format."""
    converters = {
        "msh": "fluentMeshToFoam",
        "cas": "fluentMeshToFoam", 
        "cgns": "cgnsToFoam",
        "unv": "ideasUnvToFoam",
        "neu": "gambitToFoam",
    }
    return converters.get(mesh_format, "fluentMeshToFoam")


async def run_simulation(run_id: str, case_dir: Path, mode: RunMode):
    """
    Execute OpenFOAM simulation in the background.

    Flow:
    1. Check for external mesh file and convert if needed (mshToFoam, etc.)
    2. Run blockMesh if blockMeshDict exists and no polyMesh
    3. Run checkMesh
    4. Run the solver

    For TEST mode: override controlDict to run only 1 iteration.
    For FULL mode: run as-is.
    """
    try:
        runs[run_id].status = RunStatus.RUNNING
        runs[run_id].started_at = datetime.utcnow().isoformat() + "Z"
        start_time = time.time()

        # Step 1: Check for external mesh and convert if needed
        mesh_file, mesh_format = detect_mesh_file(case_dir)
        poly_mesh_dir = case_dir / "constant" / "polyMesh"
        
        if mesh_file and not poly_mesh_dir.exists():
            runs[run_id].status = RunStatus.MESHING
            emit_event(
                run_id, "mesh_conversion_started",
                f"Converting mesh: {mesh_file.name} ({mesh_format} format)",
                payload={"mesh_file": mesh_file.name, "format": mesh_format}
            )
            
            converter = get_mesh_converter(mesh_format)
            exit_code, stderr = await run_openfoam_command(
                run_id, [converter, mesh_file.name], case_dir, event_prefix="mesh"
            )
            
            if exit_code != 0:
                runs[run_id].status = RunStatus.FAILED
                runs[run_id].error = f"Mesh conversion failed: {stderr[:500]}"
                runs[run_id].exit_code = exit_code
                emit_event(
                    run_id, "mesh_conversion_failed",
                    f"Mesh conversion failed with exit code {exit_code}",
                    level="error",
                    payload={"exit_code": exit_code, "stderr": stderr[:2000]}
                )
                return
            
            emit_event(
                run_id, "mesh_conversion_complete",
                "Mesh conversion successful",
                payload={"converter": converter}
            )
        
        # Step 2: Run blockMesh if needed
        block_mesh_dict = case_dir / "system" / "blockMeshDict"
        if block_mesh_dict.exists() and not poly_mesh_dir.exists():
            runs[run_id].status = RunStatus.MESHING
            emit_event(run_id, "blockmesh_started", "Running blockMesh")
            
            exit_code, stderr = await run_openfoam_command(
                run_id, ["blockMesh"], case_dir, event_prefix="mesh"
            )
            
            if exit_code != 0:
                runs[run_id].status = RunStatus.FAILED
                runs[run_id].error = f"blockMesh failed: {stderr[:500]}"
                runs[run_id].exit_code = exit_code
                emit_event(
                    run_id, "blockmesh_failed",
                    f"blockMesh failed with exit code {exit_code}",
                    level="error"
                )
                return
            
            emit_event(run_id, "blockmesh_complete", "blockMesh completed successfully")
        
        # Verify mesh exists
        if not poly_mesh_dir.exists():
            runs[run_id].status = RunStatus.FAILED
            runs[run_id].error = "No mesh found: constant/polyMesh does not exist"
            emit_event(
                run_id, "run_failed",
                "No mesh found - need either external mesh file or blockMeshDict",
                level="error"
            )
            return
        
        # Step 3: Run checkMesh (optional, don't fail on warnings)
        emit_event(run_id, "checkmesh_started", "Running checkMesh")
        exit_code, _ = await run_openfoam_command(
            run_id, ["checkMesh"], case_dir, event_prefix="mesh"
        )
        emit_event(
            run_id, "checkmesh_complete",
            f"checkMesh completed (exit code: {exit_code})",
            level="warn" if exit_code != 0 else "info"
        )

        # Step 4: Run the solver
        runs[run_id].status = RunStatus.RUNNING
        solver = detect_solver(case_dir)

        # For TEST mode, patch controlDict to run 1 iteration
        if mode == RunMode.TEST:
            patch_control_dict_for_test(case_dir)

        emit_event(
            run_id, "run_started",
            f"Starting {solver} ({mode.value} mode)",
            payload={"solver": solver, "mode": mode.value}
        )

        # Run the solver
        exit_code, stderr_text = await run_openfoam_command(
            run_id, [solver], case_dir
        )
        duration = time.time() - start_time

        # Update run status
        runs[run_id].exit_code = exit_code
        runs[run_id].duration_seconds = round(duration, 2)
        runs[run_id].completed_at = datetime.utcnow().isoformat() + "Z"

        if exit_code == 0:
            runs[run_id].status = RunStatus.SUCCEEDED
            emit_event(
                run_id, "run_succeeded",
                f"Simulation completed in {duration:.1f}s",
                payload={
                    "exit_code": 0,
                    "duration_seconds": round(duration, 2),
                    "mode": mode.value,
                }
            )

            # Post-process: collect artifacts
            artifacts = collect_artifacts(run_id, case_dir)
            emit_event(
                run_id, "artifacts_ready",
                f"Results ready ({len(artifacts)} files)",
                payload={"artifacts": artifacts}
            )
        else:
            runs[run_id].status = RunStatus.FAILED
            runs[run_id].error = f"Solver exited with code {exit_code}"
            emit_event(
                run_id, "run_failed",
                f"Solver failed (exit code {exit_code})",
                level="error",
                payload={
                    "exit_code": exit_code,
                    "duration_seconds": round(duration, 2),
                    "stderr": stderr_text[:2000],
                }
            )

    except Exception as e:
        runs[run_id].status = RunStatus.FAILED
        runs[run_id].error = str(e)
        emit_event(run_id, "run_failed", f"Error: {str(e)}", level="error")


def patch_control_dict_for_test(case_dir: Path):
    """
    Modify controlDict to run only 1 iteration for test mode.
    For steady-state: set endTime=1, deltaT=1
    For transient: set endTime=deltaT (single timestep)
    """
    control_dict_path = case_dir / "system" / "controlDict"
    if not control_dict_path.exists():
        return

    content = control_dict_path.read_text()
    lines = content.splitlines()
    new_lines = []

    time_scheme = detect_time_scheme(case_dir)

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("endTime") and not stripped.startswith("//"):
            if time_scheme == "steady":
                new_lines.append("    endTime         1;")
            else:
                # For transient, run 1 timestep
                new_lines.append("    endTime         0.001;")
        elif stripped.startswith("writeInterval") and not stripped.startswith("//"):
            new_lines.append("    writeInterval   1;")
        else:
            new_lines.append(line)

    control_dict_path.write_text("\n".join(new_lines))


def collect_artifacts(run_id: str, case_dir: Path) -> list[dict]:
    """Collect output files as downloadable artifacts."""
    artifacts = []

    # Collect log files
    for log_file in case_dir.glob("log.*"):
        artifacts.append({
            "name": log_file.name,
            "path": str(log_file.relative_to(case_dir)),
            "size_bytes": log_file.stat().st_size,
            "type": "log",
            "download_url": f"/api/run/{run_id}/artifacts/{log_file.name}",
        })

    # Collect latest time directory results
    time_dirs = sorted(
        [d for d in case_dir.iterdir() 
         if d.is_dir() and d.name.replace(".", "").replace("-", "").isdigit()],
        key=lambda d: float(d.name) if d.name.replace(".", "").replace("-", "").isdigit() else 0,
    )
    if time_dirs:
        latest = time_dirs[-1]
        for f in latest.iterdir():
            if f.is_file():
                artifacts.append({
                    "name": f"{latest.name}/{f.name}",
                    "path": str(f.relative_to(case_dir)),
                    "size_bytes": f.stat().st_size,
                    "type": "field",
                    "download_url": f"/api/run/{run_id}/artifacts/{latest.name}/{f.name}",
                })

    # Collect postProcessing results
    pp_dir = case_dir / "postProcessing"
    if pp_dir.exists():
        for f in pp_dir.rglob("*"):
            if f.is_file():
                rel = f.relative_to(case_dir)
                artifacts.append({
                    "name": str(rel),
                    "path": str(rel),
                    "size_bytes": f.stat().st_size,
                    "type": "postprocess",
                    "download_url": f"/api/run/{run_id}/artifacts/{rel}",
                })

    return artifacts


# ============================================================
# Endpoints
# ============================================================

@app.get("/health")
async def health():
    """Health check — also verifies OpenFOAM is available."""
    # Check if simpleFoam is available
    foam_available = False
    foam_version = None
    
    try:
        result = subprocess.run(
            f"source {OPENFOAM_ENV} && simpleFoam -help 2>&1 | head -5",
            shell=True,
            capture_output=True,
            timeout=10,
        )
        foam_available = result.returncode == 0
        if foam_available:
            output = result.stdout.decode()
            # Try to extract version
            if "OpenFOAM" in output:
                foam_version = output.split("\n")[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return {
        "status": "healthy" if foam_available else "degraded",
        "openfoam_available": foam_available,
        "openfoam_version": foam_version,
        "runs_dir": str(RUNS_DIR),
        "active_runs": sum(1 for r in runs.values() if r.status == RunStatus.RUNNING),
    }


@app.post("/api/run")
async def submit_run(
    background_tasks: BackgroundTasks,
    case_zip: UploadFile = File(..., description="ZIP file containing OpenFOAM case"),
    mode: RunMode = Form(RunMode.FULL, description="'test' for 1 iteration, 'full' for complete run"),
    run_id: Optional[str] = Form(None, description="Optional run ID (generated if not provided)"),
    callback_url: Optional[str] = Form(None, description="URL to POST final status to"),
):
    """
    Submit an OpenFOAM case for execution.

    The ZIP must contain a valid OpenFOAM case structure:
    ```
    case.zip/
    ├── 0/                  # Initial conditions (U, p, T, k, omega, etc.)
    ├── constant/           # Mesh (polyMesh/) and physical properties
    │   ├── polyMesh/
    │   ├── transportProperties
    │   └── turbulenceProperties
    └── system/             # Solver settings
        ├── controlDict
        ├── fvSchemes
        └── fvSolution
    ```

    Returns run_id for tracking via SSE events.
    """
    # Generate run ID
    if not run_id:
        run_id = f"run-{uuid.uuid4().hex[:12]}"

    # Create run directory
    run_dir = RUNS_DIR / run_id
    case_dir = run_dir / "case"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Initialize event list
    run_events[run_id] = []

    # Save and extract ZIP
    zip_path = run_dir / "case.zip"
    content = await case_zip.read()
    zip_path.write_bytes(content)

    emit_event(run_id, "extract_started", "Extracting case files...",
               payload={"zip_size_bytes": len(content)})

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(case_dir)

        # Handle nested directory (if ZIP contains a single root folder)
        subdirs = list(case_dir.iterdir())
        if len(subdirs) == 1 and subdirs[0].is_dir():
            # Move contents up one level
            nested = subdirs[0]
            for item in nested.iterdir():
                shutil.move(str(item), str(case_dir / item.name))
            nested.rmdir()

        # Validate case structure
        required = ["0", "constant", "system"]
        missing = [d for d in required if not (case_dir / d).exists()]
        if missing:
            raise ValueError(f"Invalid case: missing directories: {missing}")

        # Check for mesh source - need ONE of:
        # 1. constant/polyMesh (already converted)
        # 2. External mesh file (.msh, .cas, .cgns, .unv, .neu)
        # 3. system/blockMeshDict (will generate mesh)
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
            f"external:{external_mesh.name}" if external_mesh else
            "blockMeshDict"
        )

        emit_event(
            run_id, "extract_complete", "Case files extracted",
            payload={
                "case_dir": str(case_dir),
                "solver": detect_solver(case_dir),
                "mesh_source": mesh_source,
            }
        )

    except zipfile.BadZipFile as e:
        emit_event(run_id, "run_failed", f"Invalid ZIP file: {e}", level="error")
        raise HTTPException(status_code=400, detail=f"Invalid ZIP file: {e}")
    except Exception as e:
        emit_event(run_id, "run_failed", f"Extraction failed: {e}", level="error")
        raise HTTPException(status_code=400, detail=f"Invalid case ZIP: {e}")

    # Register run
    runs[run_id] = RunInfo(
        run_id=run_id,
        status=RunStatus.PENDING,
        mode=mode,
        created_at=datetime.utcnow().isoformat() + "Z",
    )

    # Start simulation in background
    background_tasks.add_task(run_simulation, run_id, case_dir, mode)

    return {
        "run_id": run_id,
        "status": "pending",
        "mode": mode.value,
        "events_url": f"/api/run/{run_id}/events",
        "status_url": f"/api/run/{run_id}/status",
    }


@app.post("/api/run/test")
async def submit_test_run(
    background_tasks: BackgroundTasks,
    case_zip: UploadFile = File(...),
    run_id: Optional[str] = Form(None),
    callback_url: Optional[str] = Form(None),
):
    """Shortcut for test mode (1 iteration validation)."""
    return await submit_run(
        background_tasks=background_tasks,
        case_zip=case_zip,
        mode=RunMode.TEST,
        run_id=run_id,
        callback_url=callback_url,
    )


@app.get("/api/run/{run_id}/status")
async def get_run_status(run_id: str):
    """Get current status of a run."""
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="Run not found")
    return runs[run_id].model_dump()


@app.get("/api/run/{run_id}/events")
async def stream_events(run_id: str):
    """
    SSE (Server-Sent Events) stream for real-time run progress.

    The main backend connects here and relays events to the frontend via WebSocket.

    Event format (SSE):
    ```
    data: {"run_id":"run-abc","seq":1,"type":"run_started","message":"Starting simpleFoam","payload":{}}

    data: {"run_id":"run-abc","seq":2,"type":"run_progress","message":"Iteration 1","payload":{"iteration":1,"residuals":{"Ux":0.1,"p":0.5}}}

    data: {"run_id":"run-abc","seq":15,"type":"run_succeeded","message":"Completed in 12.3s","payload":{"exit_code":0}}
    ```
    """
    if run_id not in runs and run_id not in run_events:
        raise HTTPException(status_code=404, detail="Run not found")

    async def event_generator():
        last_seq = 0
        timeout_counter = 0
        max_timeout = 3600  # 1 hour max

        while timeout_counter < max_timeout:
            events = run_events.get(run_id, [])

            # Send any new events
            for event in events[last_seq:]:
                yield f"data: {event.model_dump_json()}\n\n"
                last_seq = event.seq
                timeout_counter = 0  # Reset timeout on activity

            # Check if run is terminal
            run = runs.get(run_id)
            if run and run.status in (RunStatus.SUCCEEDED, RunStatus.FAILED):
                # Send one final event then close
                yield f"data: {json.dumps({'type': 'stream_end', 'run_id': run_id, 'status': run.status.value})}\n\n"
                break

            await asyncio.sleep(0.3)  # Poll interval
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


@app.get("/api/run/{run_id}/artifacts/{file_path:path}")
async def download_artifact(run_id: str, file_path: str):
    """Download a specific artifact file from a completed run."""
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="Run not found")

    full_path = RUNS_DIR / run_id / "case" / file_path
    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")

    return FileResponse(full_path, filename=Path(file_path).name)


@app.get("/api/run/{run_id}/artifacts")
async def list_artifacts(run_id: str):
    """List all artifacts for a completed run."""
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="Run not found")

    case_dir = RUNS_DIR / run_id / "case"
    return {"artifacts": collect_artifacts(run_id, case_dir)}


@app.delete("/api/run/{run_id}")
async def delete_run(run_id: str):
    """Delete a completed run and its files."""
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="Run not found")

    run = runs[run_id]
    if run.status == RunStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Cannot delete running simulation")

    # Delete files
    run_dir = RUNS_DIR / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)

    # Remove from tracking
    del runs[run_id]
    if run_id in run_events:
        del run_events[run_id]

    return {"deleted": run_id}


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
