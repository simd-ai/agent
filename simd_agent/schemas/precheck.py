# simd_agent/schemas/precheck.py
"""Precheck history and lint report request/response schemas."""

from typing import Any
from uuid import UUID

from pydantic import BaseModel


class PrecheckHistoryUpsert(BaseModel):
    submitted_prompt: str | None = None
    mesh_name: str | None = None
    mesh_cells: int | None = None
    steps: list[dict[str, Any]] | None = None
    review_thoughts: str | None = None
    review_items: list[dict[str, Any]] | None = None
    suggested_config: dict[str, Any] | None = None


class PrecheckHistoryOut(BaseModel):
    simulation_id: UUID
    submitted_prompt: str | None
    mesh_name: str | None
    mesh_cells: int | None
    steps: list[dict[str, Any]] | None
    review_thoughts: str | None
    review_items: list[dict[str, Any]] | None
    suggested_config: dict[str, Any] | None
    created_at: str


class LintReportCreate(BaseModel):
    is_valid: bool
    issues: list[dict[str, Any]] | None = None
    run_id: UUID | None = None


class LintReportOut(BaseModel):
    id: UUID
    simulation_id: UUID
    run_id: UUID | None
    is_valid: bool
    issues: list[dict[str, Any]] | None
    created_at: str
