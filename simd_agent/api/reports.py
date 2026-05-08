# simd_agent/api/reports.py
"""Simulation report upload, listing, and download."""

import logging
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from simd_agent.api.auth import AuthenticatedUser, get_current_user
from simd_agent.services import report_repo, simulation_service
from simd_agent.storage import get_storage

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/simulations", tags=["reports"])


def _storage_key(simulation_id: UUID, report_id: UUID) -> str:
    return f"reports/{simulation_id}/{report_id}.pdf"


@router.post("/{simulation_id}/reports", status_code=201)
async def upload_report(
    simulation_id: UUID,
    file: UploadFile = File(...),
    report_type: str = Form("standard"),
    run_id: str | None = Form(None),
    file_name: str = Form("simulation-report.pdf"),
    user: AuthenticatedUser | None = Depends(get_current_user),
):
    # Ownership check
    sim = await simulation_service.get(simulation_id)
    if not sim:
        raise HTTPException(404, "Simulation not found")
    if user and sim.get("user_id") and str(sim["user_id"]) != str(user.id):
        raise HTTPException(403, "Not your simulation")

    report_id = uuid4()
    key = _storage_key(simulation_id, report_id)

    data = await file.read()
    await get_storage().upload(key, data, content_type="application/pdf")

    row = await report_repo.create({
        "id": report_id,
        "simulation_id": simulation_id,
        "run_id": UUID(run_id) if run_id else None,
        "report_type": report_type,
        "file_name": file_name,
        "storage_key": key,
    })
    logger.info("[REPORT] Saved %s report %s for simulation %s", report_type, report_id, simulation_id)
    return row


@router.get("/{simulation_id}/reports")
async def list_reports(
    simulation_id: UUID,
    user: AuthenticatedUser | None = Depends(get_current_user),
):
    sim = await simulation_service.get(simulation_id)
    if not sim:
        raise HTTPException(404, "Simulation not found")
    if user and sim.get("user_id") and str(sim["user_id"]) != str(user.id):
        raise HTTPException(403, "Not your simulation")

    return await report_repo.list_for_simulation(simulation_id)


@router.get("/{simulation_id}/reports/{report_id}/download")
async def download_report(
    simulation_id: UUID,
    report_id: UUID,
    user: AuthenticatedUser | None = Depends(get_current_user),
):
    sim = await simulation_service.get(simulation_id)
    if not sim:
        raise HTTPException(404, "Simulation not found")
    if user and sim.get("user_id") and str(sim["user_id"]) != str(user.id):
        raise HTTPException(403, "Not your simulation")

    report = await report_repo.get_by_id(report_id)
    if not report or str(report["simulation_id"]) != str(simulation_id):
        raise HTTPException(404, "Report not found")

    data = await get_storage().download(report["storage_key"])
    if not data:
        raise HTTPException(404, "Report file not found in storage")

    return Response(
        content=data,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{report["file_name"]}"',
        },
    )
