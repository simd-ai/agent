# simd_agent/chat/db.py
"""Database helpers and snapshot builder for the chat service."""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import text

from simd_agent.chat.models import ChatRequest, DataNeeds
from simd_agent.chat.tools import SimulationSnapshot
from simd_agent.db import get_session, portable_sql as _strip_pg_casts

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _json_safe(v: Any) -> Any:
    """Convert DB values to JSON-serialisable Python objects."""
    if isinstance(v, UUID):
        return str(v)
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

async def upsert_simulation_config(
    simulation_id: str,
    physics: dict[str, Any],
    solver: dict[str, Any],
    fluid: dict[str, Any],
    turbulence: dict[str, Any],
    derived: dict[str, Any] | None = None,
    regions: dict[str, Any] | list[Any] | None = None,
) -> None:
    """Write the normalised simulation config directly into the simulation_config table.

    Called by the orchestrator right after lint + solver selection so the chat
    service can read it immediately — without depending on the frontend to relay
    the simulation_config_ready event.

    turbulence must be empty ({}) for laminar flows.
    regions is None / empty for single-region cases; a multi-region (CHT) case
    carries either the backend's ``{"fluid": [...], "solid": [...]}`` shape
    or the frontend's flat ``[{name, kind, …}, …]``.
    """
    try:
        async with get_session() as session:
            await session.execute(
                text(_strip_pg_casts("""
                    INSERT INTO simulation_config
                        (simulation_id, cfd_physics, cfd_solver, cfd_fluid,
                         cfd_turbulence, cfd_derived, cfd_regions)
                    VALUES
                        (:sid, :physics::jsonb, :solver::jsonb, :fluid::jsonb,
                         :turbulence::jsonb, :derived::jsonb, :regions::jsonb)
                    ON CONFLICT (simulation_id) DO UPDATE SET
                        cfd_physics    = EXCLUDED.cfd_physics,
                        cfd_solver     = EXCLUDED.cfd_solver,
                        cfd_fluid      = EXCLUDED.cfd_fluid,
                        cfd_turbulence = EXCLUDED.cfd_turbulence,
                        cfd_derived    = EXCLUDED.cfd_derived,
                        cfd_regions    = EXCLUDED.cfd_regions
                """)),
                {
                    "sid":       simulation_id,
                    "physics":   json.dumps(physics),
                    "solver":    json.dumps(solver),
                    "fluid":     json.dumps(fluid),
                    "turbulence": json.dumps(turbulence),
                    "derived":   json.dumps(derived or {}),
                    # NULL (not "{}") when no regions — distinguishes
                    # "single-region case" from "user explicitly emptied
                    # the regions list".
                    "regions":   json.dumps(regions) if regions else None,
                },
            )
        logger.info(f"[chat/db] upsert_simulation_config OK — sim={simulation_id}")
    except Exception as exc:
        logger.warning(f"[chat/db] upsert_simulation_config failed: {exc}")


async def fetch_simulation_config(simulation_id: str) -> dict[str, Any]:
    """Fetch the full CFD config snapshot for a simulation."""
    try:
        async with get_session() as session:
            row = await session.execute(
                text("""
                    SELECT cfd_physics, cfd_solver, cfd_fluid, cfd_turbulence,
                           cfd_derived, cfd_regions
                    FROM simulation_config
                    WHERE simulation_id = :sid
                """),
                {"sid": simulation_id},
            )
            r = row.mappings().first()
            if not r:
                return {}
            return {
                "physics":    r["cfd_physics"]    or {},
                "solver":     r["cfd_solver"]     or {},
                "fluid":      r["cfd_fluid"]      or {},
                "turbulence": r["cfd_turbulence"] or {},
                "derived":    r["cfd_derived"]    or {},
                # ``regions`` is the only field that distinguishes its
                # missing case (None) from its empty case ([] / {}): a
                # multi-region case with no regions yet is meaningless,
                # while NULL ⇒ "single-region case, never set".
                "regions":    r["cfd_regions"],
            }
    except Exception as exc:
        logger.warning(f"[chat/db] fetch_simulation_config failed: {exc}")
        return {}


async def fetch_mesh_info(simulation_id: str) -> dict[str, Any]:
    """Fetch mesh info (patches, checkMesh stats) for a simulation."""
    try:
        async with get_session() as session:
            row = await session.execute(
                text("""
                    SELECT mesh_id, file_name, patches, check_mesh
                    FROM mesh_info
                    WHERE simulation_id = :sid
                """),
                {"sid": simulation_id},
            )
            r = row.mappings().first()
            if not r:
                return {}
            return {
                "mesh_id": r["mesh_id"],
                "file_name": r["file_name"],
                "patches": r["patches"] or [],
                "check_mesh": r["check_mesh"] or {},
            }
    except Exception as exc:
        logger.warning(f"[chat/db] fetch_mesh_info failed: {exc}")
        return {}


async def fetch_patch_configs(simulation_id: str) -> dict[str, Any]:
    """Fetch configured boundary conditions per patch."""
    try:
        async with get_session() as session:
            rows = await session.execute(
                text("""
                    SELECT patch_name, patch_class, patch_config, patch_info
                    FROM patch_configs
                    WHERE simulation_id = :sid
                """),
                {"sid": simulation_id},
            )
            result: dict[str, Any] = {}
            for r in rows.mappings().all():
                result[r["patch_name"]] = {
                    "class": r["patch_class"],
                    "config": r["patch_config"] or {},
                    "info": r["patch_info"] or {},
                }
            return result
    except Exception as exc:
        logger.warning(f"[chat/db] fetch_patch_configs failed: {exc}")
        return {}


async def fetch_latest_agent_run(simulation_id: str) -> dict[str, Any]:
    """Fetch the most recent agent_run for a simulation."""
    try:
        async with get_session() as session:
            row = await session.execute(
                text("""
                    SELECT id, status, label, type, lint_result,
                           generated_files, file_generation_map,
                           final_result, vtk_result, error_message,
                           result,
                           started_at, completed_at
                    FROM runs
                    WHERE simulation_id = :sid
                    ORDER BY started_at DESC
                    LIMIT 1
                """),
                {"sid": simulation_id},
            )
            r = row.mappings().first()
            if not r:
                return {}
            return {k: _json_safe(v) for k, v in dict(r).items()}
    except Exception as exc:
        logger.warning(f"[chat/db] fetch_latest_agent_run failed: {exc}")
        return {}


async def fetch_sim_progress(run_id: str, limit: int = 2000) -> list[dict[str, Any]]:
    """Fetch up to ``limit`` sim_progress rows for a run, ordered ascending.

    We fetch a large window (default 2000) so convergence assessment and chart
    data cover the full run, not just the tail.  The caller is responsible for
    downsampling before sending to the LLM context (see SimulationSnapshot.summary_dict).
    """
    try:
        async with get_session() as session:
            rows = await session.execute(
                text("""
                    SELECT iteration, sim_time, residuals, courant, continuity, execution, field_ranges, patch_values
                    FROM sim_progress
                    WHERE run_id = :rid
                    ORDER BY iteration ASC
                    LIMIT :lim
                """),
                {"rid": run_id, "lim": limit},
            )
            return [
                {k: _json_safe(v) for k, v in dict(r).items()}
                for r in rows.mappings().all()
            ]
    except Exception as exc:
        logger.warning(f"[chat/db] fetch_sim_progress failed: {exc}")
        return []


async def fetch_all_runs(simulation_id: str) -> list[dict[str, Any]]:
    """Fetch lightweight metadata for all runs of a simulation (newest first)."""
    try:
        async with get_session() as session:
            rows = await session.execute(
                text("""
                    SELECT id, status, label, type, solver,
                           started_at, completed_at, result
                    FROM runs
                    WHERE simulation_id = :sid
                    ORDER BY started_at ASC
                """),
                {"sid": simulation_id},
            )
            return [
                {k: _json_safe(v) for k, v in dict(r).items()}
                for r in rows.mappings().all()
            ]
    except Exception as exc:
        logger.warning(f"[chat/db] fetch_all_runs failed: {exc}")
        return []


def _normalize_progress_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalise a progress row from NDJSON (camelCase) to snake_case."""
    out: dict[str, Any] = {}
    out["iteration"] = row.get("iteration", 0)
    out["sim_time"] = row.get("sim_time") if row.get("sim_time") is not None else row.get("simTime")
    out["residuals"] = row.get("residuals")
    out["courant"] = row.get("courant")
    out["continuity"] = row.get("continuity")
    out["execution"] = row.get("execution")
    out["field_ranges"] = row.get("field_ranges") or row.get("fieldRanges")
    out["patch_values"] = row.get("patch_values") or row.get("patchValues")
    out["volume_integrals"] = row.get("volume_integrals") or row.get("volumeIntegrals")
    return {k: _json_safe(v) for k, v in out.items() if v is not None}


async def fetch_sim_progress_full(run_id: str, limit: int = 2000) -> list[dict[str, Any]]:
    """Fetch sim_progress using the progress store (NDJSON/GCS) first, Postgres fallback.

    Used by build_snapshot and tools that need field_ranges or cross-run data.
    """
    try:
        from simd_agent.progress_store import read_progress

        local = await read_progress(run_id)
        if local is not None:
            rows = [_normalize_progress_row(r) for r in local]
            return rows[:limit]
    except Exception as exc:
        logger.warning(f"[chat/db] progress store read failed for {run_id}: {exc}")

    # Fallback to Postgres (legacy runs or if progress store unavailable)
    return await fetch_sim_progress(run_id, limit=limit)


async def fetch_chat_history(simulation_id: str, limit: int = 20) -> list[dict[str, str]]:
    """Fetch recent persisted chat history from the DB."""
    try:
        async with get_session() as session:
            rows = await session.execute(
                text("""
                    SELECT role, content
                    FROM chat_messages
                    WHERE simulation_id = :sid
                    ORDER BY timestamp DESC
                    LIMIT :lim
                """),
                {"sid": simulation_id, "lim": limit},
            )
            results = [dict(r) for r in rows.mappings().all()]
            results.reverse()
            return results
    except Exception as exc:
        logger.warning(f"[chat/db] fetch_chat_history failed: {exc}")
        return []


async def persist_chat_message(
    simulation_id: str,
    role: str,
    content: str,
    suggested_actions: list[str] | None = None,
) -> None:
    """Persist a chat message to the database."""
    try:
        async with get_session() as session:
            await session.execute(
                text("""
                    INSERT INTO chat_messages (id, simulation_id, role, content, suggested_actions)
                    VALUES (:id, :sid, :role, :content, :sa)
                """),
                {
                    "id": uuid4(),
                    "sid": simulation_id,
                    "role": role,
                    "content": content,
                    "sa": json.dumps(suggested_actions) if suggested_actions else None,
                },
            )
    except Exception as exc:
        logger.warning(f"[chat/db] persist_chat_message failed: {exc}")


# ---------------------------------------------------------------------------
# Snapshot builder
# ---------------------------------------------------------------------------

async def build_snapshot(
    request: ChatRequest,
    data_needs: DataNeeds | None = None,
) -> SimulationSnapshot:
    """Assemble a SimulationSnapshot from the request context + Neon DB.

    When *data_needs* is provided, only the requested data sources are fetched.
    When ``None`` (fallback / legacy), everything is fetched as before.
    """
    ctx = request.context
    sim_id = request.simulation_id or ctx.simulation_id
    fetch_all = data_needs is None  # backward-compatible: fetch everything

    db_config: dict[str, Any] = {}
    db_mesh: dict[str, Any] = {}
    db_patches: dict[str, Any] = {}
    db_run: dict[str, Any] = {}
    db_progress: list[dict[str, Any]] = []
    db_all_runs: list[dict[str, Any]] = []

    if sim_id:
        # Always fetch config + latest run (cheap, needed for context)
        db_config = await fetch_simulation_config(sim_id)
        db_run = await fetch_latest_agent_run(sim_id)

        if fetch_all or data_needs.mesh_info:
            db_mesh = await fetch_mesh_info(sim_id)
        if fetch_all or data_needs.patches:
            db_patches = await fetch_patch_configs(sim_id)
        if fetch_all or data_needs.cross_run:
            db_all_runs = await fetch_all_runs(sim_id)

        need_progress = fetch_all or data_needs.sim_progress or data_needs.field_ranges
        if need_progress:
            run_id_for_progress = ctx.run_id or (db_run.get("id") if db_run else None)
            if run_id_for_progress:
                db_progress = await fetch_sim_progress_full(str(run_id_for_progress))

    generated_files: dict[str, str] = {}
    if ctx.generated_files:
        # Frontend sent files directly — use as-is (already path→content)
        generated_files = ctx.generated_files
    else:
        # Try file_generation_map first: {"0/U": {"path":..,"content":..,"status":..}, ...}
        # This is the primary store written by the agent after codegen.
        fgm = db_run.get("file_generation_map")
        if fgm and isinstance(fgm, dict):
            for key, entry in fgm.items():
                if isinstance(entry, dict):
                    content = entry.get("content", "")
                    path = entry.get("path", key)
                    if content:
                        generated_files[path] = content
                elif isinstance(entry, str):
                    # Fallback: map might be path→content directly
                    generated_files[key] = entry

        # If file_generation_map was empty, try generated_files as fallback
        if not generated_files:
            gf = db_run.get("generated_files")
            if isinstance(gf, dict):
                # Could be path→content or path→{path,content,...}
                for key, val in gf.items():
                    if isinstance(val, str):
                        generated_files[key] = val
                    elif isinstance(val, dict):
                        content = val.get("content", "")
                        path = val.get("path", key)
                        if content:
                            generated_files[path] = content
            elif isinstance(gf, list):
                for i, f in enumerate(gf):
                    if isinstance(f, dict) and f.get("content"):
                        generated_files[f.get("path", f"file_{i}")] = f["content"]

    # Multi-region per-region field metadata — fetched from the cached VTK
    # precompute index in object storage (``results/<run>/index.json``).
    # Populated by ``_ensure_vtk_cache`` in ``main.py`` after the
    # simulation completes; carries per-region fields with their min/max
    # range so chat tools can answer "max T in the wall" without falling
    # back to residual-trend surrogates.  Empty dict for single-region
    # runs and for runs whose VTK cache hasn't been built yet.
    vtk_index: dict[str, Any] = {}
    run_id_for_vtk = ctx.run_id or (str(db_run.get("id")) if db_run.get("id") else None)
    if run_id_for_vtk and (fetch_all or (data_needs and data_needs.vtk_result)):
        try:
            import json as _json
            from simd_agent.storage import get_storage
            _storage = get_storage()
            _idx_bytes = await _storage.download(f"results/{run_id_for_vtk}/index.json")
            if _idx_bytes:
                vtk_index = _json.loads(_idx_bytes) or {}
        except Exception:
            # Storage unavailable or index not yet written — fall through
            # to legacy ``vtk_result`` path.  Chat tools degrade gracefully.
            vtk_index = {}

    return SimulationSnapshot(
        run_id=ctx.run_id or (str(db_run["id"]) if db_run.get("id") else None),
        simulation_id=sim_id,
        mesh_id=ctx.mesh_id or db_mesh.get("mesh_id"),
        physics=ctx.physics or db_config.get("physics", {}),
        solver=ctx.solver or db_config.get("solver", {}),
        fluid=ctx.fluid or db_config.get("fluid", {}),
        turbulence=ctx.turbulence or db_config.get("turbulence", {}),
        patches=ctx.patches or db_patches,
        final_result=ctx.final_result or (db_run.get("final_result") if db_run else {}),
        vtk_result=ctx.vtk_result or (db_run.get("vtk_result") if db_run else {}),
        lint_result=ctx.lint_result or (db_run.get("lint_result") if db_run else {}),
        generated_files=generated_files,
        sim_progress=db_progress,
        simulation_config=db_config,
        agent_run=db_run,
        mesh_info=db_mesh.get("check_mesh") or db_mesh,
        all_runs=db_all_runs,
        vtk_index=vtk_index,
    )
