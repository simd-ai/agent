"""
API route definitions.

All endpoints are grouped under an APIRouter and mounted by the main app.

Supports Gmsh mesh files (.msh) that are converted via gmshToFoam.
"""

import asyncio
import json
import shutil
import subprocess
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.responses import FileResponse, StreamingResponse

from app.config import OPENFOAM_CACHED_ENV, RUNS_DIR
from app.models import RunInfo, RunMode, RunStatus
from app.store import delete_run as store_delete_run
from app.store import emit_event, get_run, register_run, run_events, runs
from app.openfoam import collect_artifacts, detect_solver
from app.runner import run_simulation

router = APIRouter()


# ── Mesh Detection Helper ────────────────────────────────────
# Imported from runner so both places use the same logic (supports all formats)
from app.runner import detect_mesh_file as _detect_mesh_file


# ── Health ───────────────────────────────────────────────────

@router.get("/health")
async def health():
    """Health check — also verifies OpenFOAM is reachable."""
    foam_available = False
    foam_version: Optional[str] = None

    try:
        result = subprocess.run(
            ["simpleFoam", "-help"],
            env=OPENFOAM_CACHED_ENV,
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
    n_cores: int = Form(
        1, description="Number of MPI processes. 1 = serial (default). >1 = parallel."
    ),
    callback_url: Optional[str] = Form(
        None, description="URL to POST final status to"
    ),
):
    """
    Submit an OpenFOAM case for execution.

    The ZIP must contain a valid OpenFOAM case structure with ONE of:
    - constant/polyMesh/ (pre-converted mesh)
    - A Gmsh mesh file (.msh) — will be converted via gmshToFoam
    - system/blockMeshDict (for blockMesh generation)

    Structure::

        case.zip/
        ├── 0/
        ├── constant/
        │   ├── transportProperties
        │   └── turbulenceProperties
        ├── system/
        │   ├── controlDict
        │   ├── fvSchemes
        │   └── fvSolution
        └── mesh.msh  (optional — converted via gmshToFoam)
    """
    # Generate run ID if not provided
    if not run_id:
        run_id = f"run-{uuid.uuid4().hex[:12]}"

    run_dir = RUNS_DIR / run_id
    case_dir = run_dir / "case"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Initialise event list early so SSE can connect immediately
    run_events[run_id] = []

    # Save and extract ZIP
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

        # Flatten if ZIP contains a single root folder
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

        # Check for mesh source — need ONE of:
        #   1. constant/polyMesh (already converted)
        #   2. Gmsh mesh file (.msh) — converted via gmshToFoam
        #   3. system/blockMeshDict (will generate mesh)
        poly_mesh_exists = (case_dir / "constant" / "polyMesh").exists()
        block_mesh_exists = (case_dir / "system" / "blockMeshDict").exists()
        external_mesh, mesh_format = _detect_mesh_file(case_dir)

        if not poly_mesh_exists and not block_mesh_exists and not external_mesh:
            raise ValueError(
                "Invalid case: no mesh source found. Need one of: "
                "constant/polyMesh, system/blockMeshDict, or a mesh file "
                "(.msh, .cas, .cgns, .unv, .neu)"
            )

        mesh_source = (
            "polyMesh" if poly_mesh_exists
            else f"{mesh_format}:{external_mesh.name}" if external_mesh  # type: ignore[union-attr]
            else "blockMeshDict"
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
        emit_event(
            run_id, "run_failed", f"Invalid ZIP file: {exc}", level="error"
        )
        raise HTTPException(status_code=400, detail=f"Invalid ZIP file: {exc}")
    except Exception as exc:
        emit_event(
            run_id, "run_failed", f"Extraction failed: {exc}", level="error"
        )
        raise HTTPException(
            status_code=400, detail=f"Invalid case ZIP: {exc}"
        )

    # Register run
    run_info = RunInfo(
        run_id=run_id,
        status=RunStatus.PENDING,
        mode=mode,
        created_at=datetime.utcnow().isoformat() + "Z",
    )
    register_run(run_info)

    print(f"[routes] submitting run  run_id={run_id}  mode={mode.value}  n_cores={n_cores}")

    # Launch simulation in the background
    background_tasks.add_task(run_simulation, run_id, case_dir, mode, n_cores)

    return {
        "run_id": run_id,
        "status": "pending",
        "mode": mode.value,
        "n_cores": n_cores,
        "events_url": f"/api/run/{run_id}/events",
        "status_url": f"/api/run/{run_id}/status",
    }


# ── Test-mode shortcut ──────────────────────────────────────

@router.post("/api/run/test")
async def submit_test_run(
    background_tasks: BackgroundTasks,
    case_zip: UploadFile = File(...),
    run_id: Optional[str] = Form(None),
    n_cores: int = Form(1),
    callback_url: Optional[str] = Form(None),
):
    """Shortcut for test mode (1-iteration validation)."""
    return await submit_run(
        background_tasks=background_tasks,
        case_zip=case_zip,
        mode=RunMode.TEST,
        run_id=run_id,
        n_cores=n_cores,
        callback_url=callback_url,
    )


# ── Run status ───────────────────────────────────────────────

@router.get("/api/run/{run_id}/status")
async def get_run_status(run_id: str):
    """Return the current status of a run."""
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run.model_dump()


# ── SSE event stream ────────────────────────────────────────

@router.get("/api/run/{run_id}/events")
async def stream_events(run_id: str):
    """
    SSE (Server-Sent Events) stream for real-time run progress.

    The main backend connects here and relays events to the frontend
    via WebSocket.
    """
    if run_id not in runs and run_id not in run_events:
        raise HTTPException(status_code=404, detail="Run not found")

    async def event_generator():
        last_seq = 0
        timeout_counter = 0.0
        max_timeout = 3600      # 1 hour hard limit
        heartbeat_interval = 15 # seconds between keepalive pings
        last_heartbeat = asyncio.get_event_loop().time()

        while timeout_counter < max_timeout:
            events = run_events.get(run_id, [])
            sent_any = False

            for event in events[last_seq:]:
                yield f"data: {event.model_dump_json()}\n\n"
                last_seq = event.seq
                timeout_counter = 0
                sent_any = True

            # Check if terminal
            run = get_run(run_id)
            if run and run.status in (RunStatus.SUCCEEDED, RunStatus.FAILED):
                yield (
                    f"data: {json.dumps({'type': 'stream_end', 'run_id': run_id, 'status': run.status.value})}\n\n"
                )
                break

            # Send a keepalive SSE comment if no real data has been sent recently.
            # SSE comment lines (": ...") are ignored by event handlers but keep the
            # TCP/SSL connection alive through proxies and load balancers.
            now = asyncio.get_event_loop().time()
            if not sent_any and (now - last_heartbeat) >= heartbeat_interval:
                yield ": heartbeat\n\n"
                last_heartbeat = now

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


# ── Artifacts ────────────────────────────────────────────────

@router.get("/api/run/{run_id}/artifacts")
async def list_artifacts(run_id: str):
    """List all artifacts for a completed run."""
    if get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="Run not found")

    case_dir = RUNS_DIR / run_id / "case"
    return {"artifacts": collect_artifacts(run_id, case_dir)}


@router.get("/api/run/{run_id}/artifacts/{file_path:path}")
async def download_artifact(run_id: str, file_path: str):
    """Download a specific artifact file."""
    if get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="Run not found")

    full_path = RUNS_DIR / run_id / "case" / file_path
    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(status_code=404, detail="Artifact not found")

    return FileResponse(full_path, filename=Path(file_path).name)


# ── VTK surface results ───────────────────────────────────────

def _collect_field_info(mesh, field_info: dict) -> None:
    """Scan point_data on a mesh, accumulate metadata, add *_magnitude arrays."""
    import numpy as np

    for arr_name in list(mesh.point_data.keys()):
        arr = mesh.point_data[arr_name]
        n_comp = arr.shape[1] if arr.ndim > 1 else 1

        if n_comp > 1:
            mag = np.linalg.norm(arr, axis=1)
            mag_name = f"{arr_name}_magnitude"
            mesh.point_data[mag_name] = mag
            field_info.setdefault(mag_name, {
                "name": mag_name,
                "num_components": 1,
                "range": [float(mag.min()), float(mag.max())],
                "location": "point",
                "vector_source": arr_name,
            })

        vals = arr if arr.ndim == 1 else arr.ravel()
        field_info.setdefault(arr_name, {
            "name": arr_name,
            "num_components": n_comp,
            "range": [float(vals.min()), float(vals.max())],
            "location": "point",
        })


# ── Sidecar JSON helpers ──────────────────────────────────────────────────────
# Each VTP gets a companion *.fields.json written at build time.
# Cache hits read the JSON instead of re-loading the mesh via pyvista.

def _sidecar_path(vtp_path: Path) -> Path:
    """Return path to the field-info sidecar JSON for a given VTP file."""
    return vtp_path.with_suffix(".fields.json")


def _write_field_sidecar(vtp_path: Path, fields: list[dict]) -> None:
    """Write field metadata next to a VTP so future cache hits skip mesh reads."""
    import json as _json
    _sidecar_path(vtp_path).write_text(_json.dumps(fields))


def _read_field_sidecar(vtp_path: Path) -> list[dict] | None:
    """Read sidecar JSON; returns None if missing or corrupt (triggers rebuild)."""
    import json as _json
    sc = _sidecar_path(vtp_path)
    if not sc.exists():
        return None
    try:
        return _json.loads(sc.read_text())
    except Exception:
        return None


def _build_surface_vtp(case_dir: Path, run_id: str) -> tuple[Path, list[dict]]:
    """Build (or return cached) a single surface VTP with all simulation fields.

    Strategy
    --------
    1. **pyvista OpenFOAMReader** — reads the full case (internal mesh + BCs).
       Cell data is mapped to points with cell_data_to_point_data() so every
       interior face carries field values.  For 2D cases (one cell thick in z)
       the full outer surface already IS the cross-section — no special slicing
       needed; frontAndBack faces are included automatically.

    2. **foamToVTK fallback** — if the OpenFOAM reader fails (e.g. missing
       reader plugin), fall back to running foamToVTK and merging the resulting
       files.  This gives boundary patches only, but is better than nothing.

    Returns (vtp_path, fields_metadata).
    Requires pyvista; raises ImportError if not installed.
    """
    import pyvista as pv

    vtk_dir = case_dir / "VTK"
    surface_vtp = vtk_dir / "surface.vtp"

    # ── Return cached result if available ─────────────────────────────────────
    if surface_vtp.exists():
        fields = _read_field_sidecar(surface_vtp)
        if fields is not None:
            return surface_vtp, fields
        # Sidecar missing (legacy file) — fall through to rebuild

    vtk_dir.mkdir(parents=True, exist_ok=True)

    # ── Strategy 1: pyvista OpenFOAMReader ───────────────────────────────────
    # This reads the internal mesh with real volumetric field data, which means:
    # - For 2D cases: the front/back faces carry actual field values (not "empty")
    # - For 3D cases: the full outer shell is correctly coloured
    try:
        reader = pv.OpenFOAMReader(str(case_dir))
        reader.set_active_time_point(reader.number_time_points - 1)  # latest time
        dataset = reader.read()

        # Prefer internal mesh; fall back to first available block
        internal = None
        if hasattr(dataset, "keys"):
            for key in ("internalMesh", "internal", "mesh"):
                if key in dataset.keys():
                    internal = dataset[key]
                    break
            if internal is None:
                internal = dataset[dataset.keys()[0]]
        else:
            internal = dataset

        # Map cell-centred data → point data so surface colours are smooth
        if internal.n_arrays > 0:
            internal = internal.cell_data_to_point_data()

        # Extract the outer surface — for a 2D (one-cell-thick) mesh this
        # includes the front and back faces with full field coverage.
        surface = internal.extract_surface()

        field_info = {}
        _collect_field_info(surface, field_info)

        if not field_info:
            raise ValueError("OpenFOAMReader returned no field arrays")

        result_fields = list(field_info.values())
        surface.save(str(surface_vtp))
        _write_field_sidecar(surface_vtp, result_fields)
        print(f"[VTK] Built via OpenFOAMReader: {len(field_info)} fields, {surface.n_points} pts")
        return surface_vtp, result_fields

    except Exception as foam_err:
        print(f"[VTK] OpenFOAMReader failed ({foam_err}), falling back to foamToVTK")

    # ── Strategy 2: foamToVTK fallback ───────────────────────────────────────
    subprocess.run(
        ["foamToVTK", "-latestTime", "-ascii"],
        cwd=case_dir,
        env=OPENFOAM_CACHED_ENV,
        capture_output=True,
        timeout=120,
        check=False,
    )

    vtp_files = sorted(vtk_dir.rglob("*.vtp")) if vtk_dir.exists() else []
    vtk_files = sorted(vtk_dir.rglob("*.vtk")) if vtk_dir.exists() else []
    all_mesh_files = vtp_files + vtk_files

    if not all_mesh_files:
        raise FileNotFoundError(f"foamToVTK produced no VTK files in {vtk_dir}")

    blocks = []
    field_info = {}

    for path in all_mesh_files:
        if path.name == "surface.vtp":
            continue
        try:
            mesh = pv.read(str(path))
        except Exception:
            continue
        # cell data → point data for volume meshes
        if mesh.n_arrays > 0:
            mesh = mesh.cell_data_to_point_data()
        _collect_field_info(mesh, field_info)
        blocks.append(mesh)

    if not blocks:
        raise ValueError("No readable VTK meshes found after foamToVTK")

    merged = pv.MultiBlock(blocks).combine()
    surface = merged.extract_surface()
    result_fields = list(field_info.values())
    surface.save(str(surface_vtp))
    _write_field_sidecar(surface_vtp, result_fields)

    return surface_vtp, result_fields


@router.get("/api/run/{run_id}/vtk-results")
async def get_vtk_results(run_id: str):
    """Generate (or return cached) a surface VTP with all simulation fields.

    Runs foamToVTK -latestTime, merges all boundary patches into one
    surface_fields.vtp, pre-computes vector magnitudes, and returns:
    {
        "run_id": "...",
        "vtp_url": "/api/run/.../vtk/surface.vtp",
        "fields": [
            {"name": "p",           "num_components": 1, "range": [-50, 0.5], "location": "point"},
            {"name": "U",           "num_components": 3, "range": [0, 5.2],   "location": "point"},
            {"name": "U_magnitude", "num_components": 1, "range": [0, 5.2],   "location": "point",
             "vector_source": "U"},
            ...
        ]
    }
    """
    if get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="Run not found")

    case_dir = RUNS_DIR / run_id / "case"
    if not case_dir.exists():
        raise HTTPException(status_code=404, detail="Case directory not found")

    loop = asyncio.get_event_loop()
    try:
        vtp_path, fields = await loop.run_in_executor(
            None, _build_surface_vtp, case_dir, run_id
        )
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="pyvista not installed on this server; cannot generate VTP surface",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"VTK generation failed: {exc}")

    return {
        "run_id": run_id,
        "vtp_url": f"/api/run/{run_id}/vtk/surface.vtp",
        "fields": fields,
    }


@router.get("/api/run/{run_id}/vtk/surface.vtp")
async def serve_surface_vtp(run_id: str):
    """Serve the merged surface VTP file for a completed run."""
    if get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="Run not found")

    vtp_path = RUNS_DIR / run_id / "case" / "VTK" / "surface.vtp"
    if not vtp_path.exists():
        raise HTTPException(
            status_code=404,
            detail="VTP not generated yet — call /vtk-results first",
        )

    return FileResponse(
        str(vtp_path),
        media_type="application/xml",
        filename="surface.vtp",
    )


# ── Simulation playback (multi-timestep) ─────────────────────

def _list_time_dirs(case_dir: Path) -> list[str]:
    """Return sorted list of OpenFOAM result time directory names (excluding 0)."""
    time_dirs = []
    for d in case_dir.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        try:
            t = float(name)
        except ValueError:
            continue
        if t > 0:
            time_dirs.append(name)
    # Sort numerically
    time_dirs.sort(key=lambda x: float(x))
    return time_dirs


def _build_timestep_vtp(case_dir: Path, run_id: str, time_str: str) -> tuple[Path, list[dict]]:
    """Build (or return cached) a surface VTP for a specific simulation timestep.

    Uses pyvista OpenFOAMReader, set to the requested time point.
    Returns (vtp_path, fields_metadata).
    """
    import pyvista as pv

    vtk_dir = case_dir / "VTK" / "timesteps"
    vtk_dir.mkdir(parents=True, exist_ok=True)
    surface_vtp = vtk_dir / f"t_{time_str.replace('.', '_')}.vtp"

    if surface_vtp.exists():
        fields = _read_field_sidecar(surface_vtp)
        if fields is not None:
            return surface_vtp, fields
        # Sidecar missing (legacy file) — fall through to rebuild

    reader = pv.OpenFOAMReader(str(case_dir))

    # Find the time point index closest to the requested time
    available_times = reader.time_values
    target = float(time_str)
    closest_idx = int(min(range(len(available_times)), key=lambda i: abs(available_times[i] - target)))
    reader.set_active_time_point(closest_idx)

    dataset = reader.read()

    internal = None
    if hasattr(dataset, "keys"):
        for key in ("internalMesh", "internal", "mesh"):
            if key in dataset.keys():
                internal = dataset[key]
                break
        if internal is None:
            internal = dataset[dataset.keys()[0]]
    else:
        internal = dataset

    if internal.n_arrays > 0:
        internal = internal.cell_data_to_point_data()

    surface = internal.extract_surface()
    field_info = {}
    _collect_field_info(surface, field_info)

    result_fields = list(field_info.values())
    surface.save(str(surface_vtp))
    _write_field_sidecar(surface_vtp, result_fields)
    return surface_vtp, result_fields


def _precompute_all_timesteps(case_dir: Path, run_id: str) -> dict:
    """Build every timestep VTP and write VTK/timesteps/index.json.

    Called once — either from runner.py after run_succeeded, or lazily on the
    first playback/timesteps request.  All subsequent requests are pure
    FileResponse with zero pyvista overhead.

    Returns the index dict:
        {
            "run_id": str,
            "total": int,
            "fields": [...],               # from last timestep
            "timesteps": [
                {"time": 0.1, "filename": "t_0_1.vtp",
                 "url": "/api/run/{run_id}/vtk-timesteps/t_0_1.vtp"},
                ...
            ]
        }
    """
    import json as _json

    time_dirs = _list_time_dirs(case_dir)
    vtk_dir = case_dir / "VTK" / "timesteps"
    vtk_dir.mkdir(parents=True, exist_ok=True)
    index_path = vtk_dir / "index.json"

    # Return cached index if complete
    if index_path.exists():
        try:
            existing = _json.loads(index_path.read_text())
            all_present = all(
                (vtk_dir / Path(ts["filename"]).name).exists()
                for ts in existing.get("timesteps", [])
            )
            if all_present and len(existing.get("timesteps", [])) == len(time_dirs):
                return existing
        except Exception:
            pass

    timesteps: list[dict] = []
    fields: list[dict] = []

    for time_str in time_dirs:
        try:
            vtp_path, ts_fields = _build_timestep_vtp(case_dir, run_id, time_str)
            filename = vtp_path.name
            timesteps.append({
                "time": float(time_str),
                "filename": filename,
                "url": f"/api/run/{run_id}/vtk-timesteps/{filename}",
            })
            if ts_fields:
                fields = ts_fields  # keep last non-empty set
        except Exception as exc:
            print(f"[VTK] precompute failed for t={time_str}: {exc}")

    index = {
        "run_id": run_id,
        "total": len(timesteps),
        "fields": fields,
        "timesteps": timesteps,
    }
    index_path.write_text(_json.dumps(index))
    print(f"[VTK] Precompute complete: {len(timesteps)} frames for run {run_id}")
    return index


@router.get("/api/run/{run_id}/timesteps")
async def list_timesteps(run_id: str):
    """List all simulation timesteps.  Triggers precompute if not done yet.

    Response:
        {
            "run_id": str,
            "total": int,
            "fields": [...],
            "timesteps": [{"time": 0.1, "vtp_url": "/api/run/.../vtk-timestep/0.1/surface.vtp"}, ...]
        }
    """
    case_dir = RUNS_DIR / run_id / "case"
    if not case_dir.exists():
        raise HTTPException(status_code=404, detail="Case directory not found")

    loop = asyncio.get_event_loop()
    try:
        index = await loop.run_in_executor(None, _precompute_all_timesteps, case_dir, run_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Precompute failed: {exc}")

    if not index.get("timesteps"):
        raise HTTPException(status_code=404, detail="No result timesteps found — simulation may not have run yet")

    # Reformat to legacy vtp_url shape for backwards compatibility
    timesteps = [
        {"time": ts["time"], "vtp_url": ts["url"]}
        for ts in index["timesteps"]
    ]
    return {"run_id": run_id, "total": index["total"], "fields": index["fields"], "timesteps": timesteps}


@router.get("/api/run/{run_id}/vtk-timesteps/index.json")
async def serve_timestep_index(run_id: str):
    """Return the precomputed index of all timestep VTPs with field metadata.

    The agent backend fetches this once and caches all referenced VTPs locally.
    No further queries to this server are needed after the agent has the index.

    Response (index.json):
        {
            "run_id": str,
            "total": int,
            "fields": [{"name": str, "num_components": int, "range": [min, max],
                        "location": "point", "vector_source": str | null}, ...],
            "timesteps": [
                {"time": 0.1, "filename": "t_0_1.vtp",
                 "url": "/api/run/{run_id}/vtk-timesteps/t_0_1.vtp"},
                ...
            ]
        }
    """
    case_dir = RUNS_DIR / run_id / "case"
    if not case_dir.exists():
        raise HTTPException(status_code=404, detail="Case directory not found")

    index_path = case_dir / "VTK" / "timesteps" / "index.json"

    if not index_path.exists():
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, _precompute_all_timesteps, case_dir, run_id)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Precompute failed: {exc}")

    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Index not available — no timesteps found")

    return FileResponse(str(index_path), media_type="application/json")


@router.get("/api/run/{run_id}/vtk-timesteps/{filename}")
async def serve_precomputed_vtp(run_id: str, filename: str):
    """Serve a precomputed timestep VTP by filename.

    All files are pre-built; this is a pure disk read — no pyvista involved.
    """
    if not filename.endswith(".vtp"):
        raise HTTPException(status_code=400, detail="Only .vtp files are served here")

    vtp_path = RUNS_DIR / run_id / "case" / "VTK" / "timesteps" / filename
    if not vtp_path.exists():
        raise HTTPException(status_code=404, detail=f"VTP not found: {filename} — trigger precompute first")

    return FileResponse(
        str(vtp_path),
        media_type="application/xml",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.get("/api/run/{run_id}/vtk-timestep/{time_str}/surface.vtp")
async def serve_timestep_vtp(run_id: str, time_str: str):
    """Serve the surface VTP for a specific timestep.

    Builds on-demand if the precompute job hasn't run yet (fallback path).
    Once precomputed, this is a fast FileResponse with no pyvista overhead.
    """
    case_dir = RUNS_DIR / run_id / "case"
    if not case_dir.exists():
        raise HTTPException(status_code=404, detail="Case directory not found")

    # Fast path: check if precomputed file already exists
    filename = f"t_{time_str.replace('.', '_')}.vtp"
    precomputed = case_dir / "VTK" / "timesteps" / filename
    if precomputed.exists():
        return FileResponse(
            str(precomputed),
            media_type="application/xml",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    # Slow path: build on-demand (only on first access before precompute finishes)
    try:
        loop = asyncio.get_event_loop()
        vtp_path, _ = await loop.run_in_executor(
            None, _build_timestep_vtp, case_dir, run_id, time_str
        )
    except ImportError:
        raise HTTPException(status_code=501, detail="pyvista not installed")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"VTP generation failed: {exc}")

    return FileResponse(
        str(vtp_path),
        media_type="application/xml",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.get("/api/run/{run_id}/playback")
async def playback_stream(run_id: str):
    """SSE stream of frame events driven entirely by the precomputed index.

    All VTPs are already on disk; this endpoint only emits URL pointers —
    no pyvista, no mesh reading, no blocking work per frame.

    Events:
        {"type": "playback_start",  "total_frames": N, "fields": [...]}
        {"type": "frame",           "frame_index": i, "time": 0.1,
                                    "vtp_url": "/api/run/.../vtk-timesteps/t_0_1.vtp"}
        {"type": "playback_done"}
        {"type": "error",           "detail": "..."}
    """
    case_dir = RUNS_DIR / run_id / "case"
    if not case_dir.exists():
        raise HTTPException(status_code=404, detail="Case directory not found")

    async def frame_generator():
        loop = asyncio.get_event_loop()
        try:
            index = await loop.run_in_executor(
                None, _precompute_all_timesteps, case_dir, run_id
            )
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"
            return

        timesteps = index.get("timesteps", [])
        fields = index.get("fields", [])

        yield f"data: {json.dumps({'type': 'playback_start', 'total_frames': len(timesteps), 'fields': fields})}\n\n"

        for i, ts in enumerate(timesteps):
            event = {
                "type": "frame",
                "frame_index": i,
                "total_frames": len(timesteps),
                "time": ts["time"],
                "vtp_url": ts["url"],
            }
            yield f"data: {json.dumps(event)}\n\n"

        yield f"data: {json.dumps({'type': 'playback_done'})}\n\n"

    return StreamingResponse(
        frame_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── Delete run ───────────────────────────────────────────────

@router.delete("/api/run/{run_id}")
async def delete_run_endpoint(run_id: str):
    """Delete a completed run and its files."""
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    if run.status == RunStatus.RUNNING:
        raise HTTPException(
            status_code=400, detail="Cannot delete a running simulation"
        )

    run_dir = RUNS_DIR / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)

    store_delete_run(run_id)
    return {"deleted": run_id}
