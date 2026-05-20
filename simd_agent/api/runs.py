# simd_agent/api/runs.py
"""Run, event, and simulation progress endpoints."""

import io
import logging
import math
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import Response, StreamingResponse

from simd_agent.api.auth import AuthenticatedUser, get_current_user, require_simulation_owner
from simd_agent.schemas.run import (
    ApplyRecommendationRequest,
    ApplyRecommendationResponse,
    EventOut,
    RunComplete,
    RunCreate,
    RunOut,
    RunUpdate,
    SimProgressBatch,
    SimProgressOut,
)
from simd_agent.services import run_service, mesh_repo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/runs", tags=["runs"])


# ── Runs ─────────────────────────────────────────────────────────────────


@router.post("", status_code=201)
async def create_run(
    body: RunCreate,
    user: AuthenticatedUser | None = Depends(get_current_user),
) -> RunOut:
    # Ownership check on the parent simulation
    if user and body.simulation_id:
        from simd_agent.services import simulation_service

        sim = await simulation_service.get(body.simulation_id)
        if not sim or sim.user_id != user.id:
            raise HTTPException(403, "You do not own the parent simulation")
    return await run_service.create(body)


@router.get("")
async def list_runs(
    simulation_id: UUID | None = None,
    status: str | None = None,
    limit: int = 50,
) -> list[RunOut]:
    return await run_service.list(simulation_id, status, limit)


@router.get("/{run_id}")
async def get_run(run_id: UUID) -> RunOut:
    run = await run_service.get(run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")
    return run


@router.get("/latest/{simulation_id}")
async def get_latest_run(
    simulation_id: UUID,
    _owner: AuthenticatedUser | None = Depends(require_simulation_owner),
) -> RunOut | None:
    run = await run_service.get_latest(simulation_id)
    if run:
        gf = run.generated_files
        fgm = run.file_generation_map
        gf_keys = list(gf.keys()) if gf else []
        fgm_keys = list(fgm.keys()) if fgm else []
        gf_chars = sum(len(v) for v in gf.values() if isinstance(v, str)) if gf else 0
        print(f"[LOAD] get_latest_run sim={simulation_id} run={run.id} status={run.status}")
        print(f"[LOAD]   generated_files: {len(gf_keys)} files, {gf_chars} chars, keys={gf_keys[:5]}{'...' if len(gf_keys) > 5 else ''}")
        print(f"[LOAD]   file_generation_map: {len(fgm_keys)} files, keys={fgm_keys[:5]}{'...' if len(fgm_keys) > 5 else ''}")
    else:
        print(f"[LOAD] get_latest_run sim={simulation_id} — no run found")
    return run


@router.patch("/{run_id}")
async def update_run(run_id: UUID, body: RunUpdate) -> RunOut:
    run = await run_service.update(run_id, body)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")
    return run


@router.post("/{run_id}/complete")
async def complete_run(run_id: UUID, body: RunComplete) -> RunOut:
    gf = body.generated_files
    fgm = body.file_generation_map
    gf_keys = list(gf.keys()) if gf else []
    fgm_keys = list(fgm.keys()) if fgm else []
    print(f"[COMPLETE] run={run_id} status={body.status}")
    print(f"[COMPLETE]   generated_files from frontend: {len(gf_keys)} files, keys={gf_keys[:5]}{'...' if len(gf_keys) > 5 else ''}")
    print(f"[COMPLETE]   file_generation_map from frontend: {len(fgm_keys)} files, keys={fgm_keys[:5]}{'...' if len(fgm_keys) > 5 else ''}")
    run = await run_service.complete(run_id, body)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")
    # Check what's in DB after save
    after_gf = run.generated_files
    after_fgm = run.file_generation_map
    print(f"[COMPLETE]   AFTER DB save — generated_files: {len(list(after_gf.keys())) if after_gf else 0} files, file_generation_map: {len(list(after_fgm.keys())) if after_fgm else 0} files")
    return run


# ── Apply Recommendation ───────────────────────────────────────────────


@router.post("/{run_id}/apply-recommendation")
async def apply_recommendation_endpoint(
    run_id: UUID,
    body: ApplyRecommendationRequest,
) -> ApplyRecommendationResponse:
    """Apply a convergence recommendation to a run's generated files.

    Modifies the specified OpenFOAM files (fvSolution, controlDict) based
    on the recommendation action and returns the modified file contents.
    The frontend can then display the changes and trigger a re-run.
    """
    run = await run_service.get(run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")

    if not run.generated_files:
        raise HTTPException(400, "No generated files available on this run")

    from simd_agent.run.apply_recommendation import apply_recommendation

    action = {"type": body.type, "changes": body.changes}
    modified = apply_recommendation(run.generated_files, action)

    # Find which files actually changed
    changed_keys = [
        k for k in modified
        if modified[k] != run.generated_files.get(k)
    ]

    if not changed_keys:
        raise HTTPException(
            422,
            "No files were modified — the recommendation could not be applied "
            "(the target settings may not exist in the generated files).",
        )

    # Save modified files back to the run
    await run_service.update(run_id, RunUpdate(
        generated_files=modified,
    ))

    logger.info(
        f"[APPLY] run={run_id} type={body.type} "
        f"changed={changed_keys}"
    )

    return ApplyRecommendationResponse(
        modified_files={k: modified[k] for k in changed_keys},
        changed_keys=changed_keys,
    )


# ── Export ───────────────────────────────────────────────────────────────


@router.get("/{run_id}/export")
async def export_case(run_id: UUID) -> StreamingResponse:
    """Export a complete OpenFOAM case as a ZIP archive.

    Packages generated files + original mesh + run.sh into a ready-to-run
    case directory that can be executed on any machine with OpenFOAM installed.
    """
    # 1. Fetch run
    run = await run_service.get(run_id)
    if not run:
        raise HTTPException(404, f"Run {run_id} not found")

    if not run.generated_files:
        raise HTTPException(400, "No generated files available for this run")

    # 2. Get mesh_id from mesh_info table (mesh_id = simulation_id)
    mesh_id: str | None = None
    mesh_filename: str | None = None
    if run.simulation_id:
        mesh_row = await mesh_repo.get_by_id(run.simulation_id)
        if mesh_row:
            mesh_id = mesh_row.get("mesh_id")
            mesh_filename = mesh_row.get("file_name")

    # 3. Retrieve mesh from storage (if available)
    mesh_bytes: bytes | None = None
    mesh_format: str | None = None
    if mesh_id:
        try:
            from simd_agent.run.mesh_retriever import async_get_mesh_file
            mesh_bytes, mesh_filename, mesh_format = await async_get_mesh_file(mesh_id)
        except Exception as e:
            logger.warning(f"[EXPORT] Could not retrieve mesh for run {run_id}: {e}")

    # 4. Package into ZIP — include run.sh + fix_mesh_setup.sh so the
    # downloaded case is runnable locally with `bash run.sh`.
    solver = run.solver or "simpleFoam"
    from simd_agent.run.packaging import package_case
    zip_bytes, file_list = package_case(
        files=run.generated_files,
        solver=solver,
        case_name="case",
        include_local_helpers=True,
        mesh_bytes=mesh_bytes,
        mesh_filename=mesh_filename,
        mesh_format=mesh_format,
    )

    logger.info(f"[EXPORT] run={run_id} solver={solver} files={len(file_list)} zip={len(zip_bytes)} bytes")

    from simd_agent.telemetry import get_telemetry, CaseExported
    get_telemetry().capture(CaseExported(solver=solver))

    # 5. Stream the ZIP back
    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="simd_case_{run_id}.zip"',
            "Content-Length": str(len(zip_bytes)),
        },
    )


# ── Events ───────────────────────────────────────────────────────────────


@router.get("/{run_id}/events")
async def list_events(run_id: UUID, after_seq: int | None = None) -> list[EventOut]:
    return await run_service.list_events(run_id, after_seq)


# ── Progress ─────────────────────────────────────────────────────────────


@router.get("/{run_id}/progress")
async def get_progress(run_id: UUID) -> list[SimProgressOut]:
    rows = await run_service.list_progress(run_id)
    print(f"[PROGRESS] GET /api/runs/{run_id}/progress → {len(rows)} rows")
    return rows


@router.post("/{run_id}/progress")
async def insert_progress(run_id: UUID, body: SimProgressBatch) -> dict[str, int]:
    count = await run_service.insert_progress(run_id, body)
    return {"inserted": count}


@router.delete("/{run_id}/progress", status_code=204)
async def delete_progress(run_id: UUID) -> None:
    await run_service.delete_progress(run_id)


# ── Convergence chart (matplotlib PNG, for the PDF report) ──────────────────


# Stable color palette — mirrors FIELD_COLORS in the frontend so reports
# match the in-app live chart at a glance.
_FIELD_COLORS = [
    "#F24604", "#3b82f6", "#10b981", "#f59e0b",
    "#8b5cf6", "#ec4899", "#06b6d4", "#84cc16",
    "#f43f5e", "#14b8a6",
]


@router.get(
    "/{run_id}/convergence-chart.png",
    responses={200: {"content": {"image/png": {}}}},
)
async def get_convergence_chart_png(run_id: UUID) -> Response:
    """Render the residual-vs-time chart as a print-friendly PNG for the PDF report.

    The frontend's Recharts component cannot be screenshot reliably offscreen
    (SVG paths come out empty), so the report fetches this server-rendered
    matplotlib image instead.  The in-app live chart is unaffected.
    """
    rows = await run_service.list_progress(run_id)
    if not rows:
        raise HTTPException(404, "No progress data for this run")

    # Collect every field name across the run (some appear later than iter 0)
    field_set: set[str] = set()
    for r in rows:
        if r.residuals:
            field_set.update(r.residuals.keys())
    if not field_set:
        raise HTTPException(404, "Progress rows contain no residuals")
    fields = sorted(field_set)

    # Build per-field (x=simTime, y=residual.final) series.  ``residuals`` is
    # ``{field: {"initial": ..., "final": ...}}`` from the solver runner.
    series: dict[str, tuple[list[float], list[float]]] = {f: ([], []) for f in fields}
    for r in rows:
        if r.sim_time is None or not r.residuals:
            continue
        for f in fields:
            raw = r.residuals.get(f)
            if isinstance(raw, dict):
                val = raw.get("final")
            else:
                val = raw
            try:
                vf = float(val)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(vf) or vf <= 0:
                continue
            series[f][0].append(r.sim_time)
            series[f][1].append(vf)

    # Drop fields with no usable points (avoids empty entries in the legend).
    fields = [f for f in fields if series[f][0]]
    if not fields:
        raise HTTPException(422, "No finite positive residuals to plot")

    # Render — Agg backend, lazy import so app boot doesn't pay for it.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 7), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for i, f in enumerate(fields):
        xs, ys = series[f]
        ax.plot(
            xs, ys,
            label=f,
            color=_FIELD_COLORS[i % len(_FIELD_COLORS)],
            linewidth=1.5,
        )

    # Log Y for residuals.  Log X only when the time axis spans > 2 decades
    # and is strictly positive — otherwise stick with linear so early-time
    # transient runs are not visually compressed.
    ax.set_yscale("log")
    all_x = [x for f in fields for x in series[f][0]]
    if all_x:
        x_min, x_max = min(all_x), max(all_x)
        if x_min > 0 and x_max / x_min > 100:
            ax.set_xscale("log")
            ax.set_xlabel("Simulation time (s) — log scale", fontsize=11, color="#334155")
        else:
            ax.set_xlabel("Simulation time (s)", fontsize=11, color="#334155")

    ax.set_ylabel("Residual (log)", fontsize=11, color="#334155")
    ax.set_title("Convergence — Residuals", fontsize=13, color="#16213e", pad=12, loc="left")

    ax.grid(True, which="major", linestyle="--", linewidth=0.5, color="#cbd5e1", alpha=0.8)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, color="#e2e8f0", alpha=0.6)
    ax.tick_params(colors="#475569", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
        spine.set_linewidth(0.8)

    ax.legend(
        loc="upper right",
        fontsize=9,
        frameon=True,
        facecolor="white",
        edgecolor="#e2e8f0",
        labelcolor="#1a1a2e",
        ncol=min(3, max(1, len(fields) // 3 + 1)),
    )

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)

    return Response(
        content=buf.getvalue(),
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=300"},
    )
