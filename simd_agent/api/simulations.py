# simd_agent/api/simulations.py
"""Simulation CRUD + config + form-state endpoints."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException

from simd_agent.api.auth import AuthenticatedUser, get_current_user, require_simulation_owner
from simd_agent.schemas.simulation import (
    FormStateUpdate,
    SimulationConfigOut,
    SimulationConfigUpsert,
    SimulationCreate,
    SimulationOut,
    SimulationUpdate,
)
from simd_agent.services import simulation_service, user_service

router = APIRouter(prefix="/api/simulations", tags=["simulations"])


# ── Simulation CRUD ─────────────────────────────────────────────────────


@router.post("", status_code=201)
async def create_simulation(body: SimulationCreate) -> SimulationOut:
    # Enforce project limit for free-tier users
    try:
        usage = await user_service.get_usage(body.user_id)
    except ValueError:
        raise HTTPException(404, f"User {body.user_id} not found")
    if not usage.can_create_project:
        from simd_agent.telemetry import get_telemetry, UsageLimitHit
        get_telemetry().capture(
            UsageLimitHit(limit_type="project", current_count=usage.project_count),
            user_id=str(body.user_id),
        )
        raise HTTPException(
            403,
            f"Free plan allows up to {usage.limits.max_projects} active projects. "
            "Delete an existing project or upgrade to Pro for unlimited projects.",
        )

    sim = await simulation_service.create(body)

    from simd_agent.telemetry import get_telemetry, ProjectCreated
    get_telemetry().capture(ProjectCreated(), user_id=str(body.user_id))

    return sim


@router.get("")
async def list_simulations(
    user_id: UUID | None = None,
    user: AuthenticatedUser | None = Depends(get_current_user),
) -> list[SimulationOut]:
    # When authenticated, always scope to the current user
    effective_user_id = user.id if user else user_id
    return await simulation_service.list(effective_user_id)


@router.get("/{simulation_id}")
async def get_simulation(
    simulation_id: UUID,
    _owner: AuthenticatedUser | None = Depends(require_simulation_owner),
) -> SimulationOut:
    sim = await simulation_service.get(simulation_id)
    if not sim:
        raise HTTPException(404, f"Simulation {simulation_id} not found")
    return sim


@router.patch("/{simulation_id}")
async def update_simulation(
    simulation_id: UUID,
    body: SimulationUpdate,
    _owner: AuthenticatedUser | None = Depends(require_simulation_owner),
) -> SimulationOut:
    sim = await simulation_service.update(simulation_id, body)
    if not sim:
        raise HTTPException(404, f"Simulation {simulation_id} not found")
    return sim


@router.delete("/{simulation_id}", status_code=204)
async def delete_simulation(
    simulation_id: UUID,
    _owner: AuthenticatedUser | None = Depends(require_simulation_owner),
) -> None:
    if not await simulation_service.delete(simulation_id):
        raise HTTPException(404, f"Simulation {simulation_id} not found")

    from simd_agent.api.auth import invalidate_ownership_cache
    invalidate_ownership_cache(simulation_id)

    from simd_agent.telemetry import get_telemetry, ProjectDeleted
    get_telemetry().capture(
        ProjectDeleted(),
        user_id=str(_owner.id) if _owner else None,
    )


# ── Config ───────────────────────────────────────────────────────────────


@router.get("/{simulation_id}/config")
async def get_config(
    simulation_id: UUID,
    _owner: AuthenticatedUser | None = Depends(require_simulation_owner),
) -> SimulationConfigOut:
    cfg = await simulation_service.get_config(simulation_id)
    if not cfg:
        raise HTTPException(404, f"Config not found for simulation {simulation_id}")
    return cfg


@router.put("/{simulation_id}/config")
async def upsert_config(
    simulation_id: UUID,
    body: SimulationConfigUpsert,
    _owner: AuthenticatedUser | None = Depends(require_simulation_owner),
) -> SimulationConfigOut:
    return await simulation_service.upsert_config(simulation_id, body)


# ── Form State (combined save) ──────────────────────────────────────────


@router.put("/{simulation_id}/form-state")
async def save_form_state(
    simulation_id: UUID,
    body: FormStateUpdate,
    _owner: AuthenticatedUser | None = Depends(require_simulation_owner),
) -> dict[str, str]:
    await simulation_service.save_form_state(simulation_id, body)
    return {"status": "saved"}
