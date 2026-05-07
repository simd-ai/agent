# simd_agent/api/precheck_lint.py
"""Precheck history and lint report endpoints."""

from uuid import UUID

from fastapi import APIRouter, Depends

from simd_agent.api.auth import AuthenticatedUser, require_simulation_owner
from simd_agent.schemas.precheck import (
    LintReportCreate,
    LintReportOut,
    PrecheckHistoryOut,
    PrecheckHistoryUpsert,
)
from simd_agent.services import lint_repo, precheck_repo

router = APIRouter(prefix="/api/simulations", tags=["precheck", "lint"])


# ── Precheck ─────────────────────────────────────────────────────────────


@router.get("/{simulation_id}/precheck")
async def get_precheck_history(
    simulation_id: UUID,
    _owner: AuthenticatedUser | None = Depends(require_simulation_owner),
) -> PrecheckHistoryOut | None:
    row = await precheck_repo.get_by_id(simulation_id)
    return PrecheckHistoryOut(**row) if row else None


@router.put("/{simulation_id}/precheck")
async def upsert_precheck_history(
    simulation_id: UUID,
    body: PrecheckHistoryUpsert,
    _owner: AuthenticatedUser | None = Depends(require_simulation_owner),
) -> PrecheckHistoryOut:
    data = body.model_dump(exclude_none=True)
    data["simulation_id"] = simulation_id
    row = await precheck_repo.upsert(
        data=data,
        conflict_keys=["simulation_id"],
    )
    return PrecheckHistoryOut(**row)


# ── Lint Reports ─────────────────────────────────────────────────────────


@router.get("/{simulation_id}/lint")
async def get_latest_lint_report(
    simulation_id: UUID,
    _owner: AuthenticatedUser | None = Depends(require_simulation_owner),
) -> LintReportOut | None:
    row = await lint_repo.get_latest(simulation_id)
    return LintReportOut(**row) if row else None


@router.get("/{simulation_id}/lint/all")
async def list_lint_reports(
    simulation_id: UUID,
    _owner: AuthenticatedUser | None = Depends(require_simulation_owner),
) -> list[LintReportOut]:
    rows = await lint_repo.list(
        filters={"simulation_id": simulation_id},
        order_by="created_at DESC",
    )
    return [LintReportOut(**row) for row in rows]


@router.post("/{simulation_id}/lint", status_code=201)
async def save_lint_report(
    simulation_id: UUID,
    body: LintReportCreate,
    _owner: AuthenticatedUser | None = Depends(require_simulation_owner),
) -> LintReportOut:
    data = body.model_dump(exclude_none=True)
    data["simulation_id"] = simulation_id
    row = await lint_repo.create(data)
    return LintReportOut(**row)
