# simd_agent/api/meshes.py
"""Mesh info and patch config endpoints."""

from uuid import UUID

from fastapi import APIRouter, Depends

from simd_agent.api.auth import AuthenticatedUser, require_simulation_owner
from simd_agent.schemas.mesh import (
    MeshInfoOut,
    MeshInfoUpsert,
    PatchConfigOut,
    PatchConfigsBatchUpsert,
)
from simd_agent.services import mesh_repo, patch_repo

router = APIRouter(prefix="/api/simulations", tags=["meshes"])


# ── Mesh Info ────────────────────────────────────────────────────────────


@router.get("/{simulation_id}/mesh")
async def get_mesh_info(
    simulation_id: UUID,
    _owner: AuthenticatedUser | None = Depends(require_simulation_owner),
) -> MeshInfoOut | None:
    row = await mesh_repo.get_by_id(simulation_id)
    return MeshInfoOut(**row) if row else None


@router.put("/{simulation_id}/mesh")
async def upsert_mesh_info(
    simulation_id: UUID,
    body: MeshInfoUpsert,
    _owner: AuthenticatedUser | None = Depends(require_simulation_owner),
) -> MeshInfoOut:
    data = body.model_dump(exclude_none=True)
    data["simulation_id"] = simulation_id
    row = await mesh_repo.upsert(
        data=data,
        conflict_keys=["simulation_id"],
    )

    from simd_agent.telemetry import get_telemetry, MeshUploaded
    _patches = body.patches or []
    _cells = None
    if body.check_mesh and isinstance(body.check_mesh, dict):
        _cells = body.check_mesh.get("nCells") or body.check_mesh.get("n_cells")
    get_telemetry().capture(
        MeshUploaded(cell_count=_cells, patch_count=len(_patches)),
        user_id=str(_owner.id) if _owner else None,
    )

    return MeshInfoOut(**row)


# ── Patch Configs ────────────────────────────────────────────────────────


@router.get("/{simulation_id}/patches")
async def get_patch_configs(
    simulation_id: UUID,
    _owner: AuthenticatedUser | None = Depends(require_simulation_owner),
) -> list[PatchConfigOut]:
    rows = await patch_repo.list_for_simulation(simulation_id)
    return [PatchConfigOut(**row) for row in rows]


@router.put("/{simulation_id}/patches")
async def upsert_patch_configs(
    simulation_id: UUID,
    body: PatchConfigsBatchUpsert,
    _owner: AuthenticatedUser | None = Depends(require_simulation_owner),
) -> list[PatchConfigOut]:
    for patch in body.patches:
        await patch_repo.upsert_patch(simulation_id, patch.model_dump())

    rows = await patch_repo.list_for_simulation(simulation_id)
    return [PatchConfigOut(**row) for row in rows]
