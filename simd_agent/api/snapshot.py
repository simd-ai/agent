# simd_agent/api/snapshot.py
"""Snapshot endpoints — progressive loading in 3 tiers.

Tier 0 (primary):     simulation metadata only — <100 ms, unblocks UI skeleton
Tier 1 (priority):    tab-dependent data — what the active tab needs
Tier 2 (background):  everything else — loaded in background

Legacy:
  Step 1 (essentials):  simulation + chat + precheck
  Step 2 (background):  config + mesh + patches + lint + run
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from simd_agent.api.auth import AuthenticatedUser, get_current_user, require_simulation_owner
from simd_agent.schemas.simulation import (
    SnapshotOut,
    SnapshotPrimaryOut,
    SnapshotEssentialsOut,
    SnapshotBackgroundOut,
)
from simd_agent.services import snapshot_service

router = APIRouter(prefix="/api/simulations", tags=["snapshot"])


# ── Legacy: full snapshot in one call ──────────────────────────

@router.get("/{simulation_id}/snapshot")
async def get_snapshot(
    simulation_id: UUID,
    _owner: AuthenticatedUser | None = Depends(require_simulation_owner),
) -> SnapshotOut:
    result = await snapshot_service.load(simulation_id)
    if not result:
        raise HTTPException(404, f"Simulation {simulation_id} not found")
    return result


# ── Tier 0: primary — just the simulation row ─────────────────

@router.get("/{simulation_id}/snapshot/primary")
async def get_snapshot_primary(
    simulation_id: UUID,
    user: AuthenticatedUser | None = Depends(get_current_user),
) -> SnapshotPrimaryOut:
    result = await snapshot_service.load_primary(simulation_id)
    if not result:
        raise HTTPException(404, f"Simulation {simulation_id} not found")

    # Inline ownership check
    if user is not None:
        sim_user_id = result.simulation.get("user_id")
        if sim_user_id and str(sim_user_id) != str(user.id):
            raise HTTPException(403, "You do not own this simulation")

    return result


# ── Tier 1+2: essentials + background (existing) ──────────────

@router.get("/{simulation_id}/snapshot/essentials")
async def get_snapshot_essentials(
    simulation_id: UUID,
    user: AuthenticatedUser | None = Depends(get_current_user),
) -> SnapshotEssentialsOut:
    result = await snapshot_service.load_essentials(simulation_id)
    if not result:
        raise HTTPException(404, f"Simulation {simulation_id} not found")

    # Inline ownership check — reuses the sim row we already fetched
    if user is not None:
        sim_user_id = result.simulation.get("user_id")
        if sim_user_id and str(sim_user_id) != str(user.id):
            raise HTTPException(403, "You do not own this simulation")

    return result


@router.get("/{simulation_id}/snapshot/background")
async def get_snapshot_background(
    simulation_id: UUID,
    _user: AuthenticatedUser | None = Depends(get_current_user),
) -> SnapshotBackgroundOut:
    config_out, run_out = await snapshot_service.load_background(simulation_id)
    return SnapshotBackgroundOut(
        config=config_out.config,
        mesh=config_out.mesh,
        patches=config_out.patches,
        lint_report=config_out.lint_report,
        latest_run=run_out.latest_run,
    )
