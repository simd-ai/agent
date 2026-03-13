# simd_agent/chat/db.py
"""Database helpers and snapshot builder for the chat service."""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import text

from simd_agent.chat.models import ChatRequest
from simd_agent.chat.tools import SimulationSnapshot
from simd_agent.db import get_session

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

async def fetch_simulation_config(simulation_id: str) -> dict[str, Any]:
    """Fetch cfd_physics, cfd_solver, cfd_fluid, cfd_turbulence from simulation_config."""
    try:
        async with get_session() as session:
            row = await session.execute(
                text("""
                    SELECT cfd_physics, cfd_solver, cfd_fluid, cfd_turbulence, cfd_derived
                    FROM simulation_config
                    WHERE simulation_id = :sid
                """),
                {"sid": simulation_id},
            )
            r = row.mappings().first()
            if not r:
                return {}
            return {
                "physics": r["cfd_physics"] or {},
                "solver": r["cfd_solver"] or {},
                "fluid": r["cfd_fluid"] or {},
                "turbulence": r["cfd_turbulence"] or {},
                "derived": r["cfd_derived"] or {},
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
                           started_at, completed_at
                    FROM agent_runs
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
                    SELECT iteration, sim_time, residuals, courant, continuity, execution
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

async def build_snapshot(request: ChatRequest) -> SimulationSnapshot:
    """Assemble a SimulationSnapshot from the request context + Neon DB."""
    ctx = request.context
    sim_id = request.simulation_id or ctx.simulation_id

    db_config: dict[str, Any] = {}
    db_mesh: dict[str, Any] = {}
    db_patches: dict[str, Any] = {}
    db_run: dict[str, Any] = {}
    db_progress: list[dict[str, Any]] = []

    if sim_id:
        db_config = await fetch_simulation_config(sim_id)
        db_mesh = await fetch_mesh_info(sim_id)
        db_patches = await fetch_patch_configs(sim_id)
        db_run = await fetch_latest_agent_run(sim_id)

        run_id_for_progress = ctx.run_id or (db_run.get("id") if db_run else None)
        if run_id_for_progress:
            db_progress = await fetch_sim_progress(str(run_id_for_progress))

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
    )
