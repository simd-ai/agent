# simd_agent/schemas/simulation.py
"""Simulation request/response schemas."""

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class SimulationCreate(BaseModel):
    user_id: UUID
    title: str = "Untitled Simulation"
    user_prompt: str | None = None
    selected_preset_id: str | None = None
    expert_mode: bool = False
    is_from_scratch_mode: bool = False


class SimulationUpdate(BaseModel):
    title: str | None = None
    active_step: int | None = None
    max_reached_step: int | None = None
    selected_preset_id: str | None = None
    user_prompt: str | None = None
    expert_mode: bool | None = None
    is_from_scratch_mode: bool | None = None
    active_tab: str | None = None


class SimulationOut(BaseModel):
    id: UUID
    user_id: UUID
    title: str
    active_step: int
    max_reached_step: int
    selected_preset_id: str | None
    user_prompt: str | None
    expert_mode: bool
    is_from_scratch_mode: bool
    active_tab: str
    created_at: str
    updated_at: str


class SimulationConfigUpsert(BaseModel):
    case_spec: dict[str, Any] | None = None
    cfd_physics: dict[str, Any] | None = None
    cfd_solver: dict[str, Any] | None = None
    cfd_fluid: dict[str, Any] | None = None
    cfd_turbulence: dict[str, Any] | None = None
    cfd_derived: dict[str, Any] | None = None


class SimulationConfigOut(BaseModel):
    simulation_id: UUID
    case_spec: dict[str, Any] | None = None
    cfd_physics: dict[str, Any] | None = None
    cfd_solver: dict[str, Any] | None = None
    cfd_fluid: dict[str, Any] | None = None
    cfd_turbulence: dict[str, Any] | None = None
    cfd_derived: dict[str, Any] | None = None


class FormStateUpdate(BaseModel):
    """Combined simulation metadata + config save (debounced from frontend)."""
    # Simulation fields
    active_step: int | None = None
    max_reached_step: int | None = None
    selected_preset_id: str | None = None
    user_prompt: str | None = None
    expert_mode: bool | None = None
    is_from_scratch_mode: bool | None = None
    active_tab: str | None = None
    # Config fields
    case_spec: dict[str, Any] | None = None
    cfd_physics: dict[str, Any] | None = None
    cfd_solver: dict[str, Any] | None = None
    cfd_fluid: dict[str, Any] | None = None
    cfd_turbulence: dict[str, Any] | None = None
    cfd_derived: dict[str, Any] | None = None


class SnapshotOut(BaseModel):
    """Complete simulation state for frontend hydration."""
    simulation: dict[str, Any]
    config: dict[str, Any] | None = None
    mesh: dict[str, Any] | None = None
    patches: list[dict[str, Any]] = Field(default_factory=list)
    precheck: dict[str, Any] | None = None
    lint_report: dict[str, Any] | None = None
    latest_run: dict[str, Any] | None = None
    chat: list[dict[str, Any]] = Field(default_factory=list)


# ── Progressive snapshot groups ─────────────────────────────────

class SnapshotPrimaryOut(BaseModel):
    """Tier 0: simulation metadata only — ultra-fast single-row query.

    Returns just the simulation row so the frontend can read activeTab and
    decide what to load next.  Unblocks the UI skeleton in <100 ms.
    """
    simulation: dict[str, Any]


class SnapshotEssentialsOut(BaseModel):
    """Group 1: simulation metadata + chat + precheck + mesh — shown immediately."""
    simulation: dict[str, Any]
    chat: list[dict[str, Any]] = Field(default_factory=list)
    precheck: dict[str, Any] | None = None
    mesh: dict[str, Any] | None = None


class SnapshotConfigOut(BaseModel):
    """Group 2: viewer data — loaded in background after essentials."""
    config: dict[str, Any] | None = None
    mesh: dict[str, Any] | None = None
    patches: list[dict[str, Any]] = Field(default_factory=list)
    lint_report: dict[str, Any] | None = None


class SnapshotRunOut(BaseModel):
    """Group 3: simulation results — loaded in background, potentially heavy."""
    latest_run: dict[str, Any] | None = None


class SnapshotBackgroundOut(BaseModel):
    """Groups 2+3 combined: all background data in one call."""
    config: dict[str, Any] | None = None
    mesh: dict[str, Any] | None = None
    patches: list[dict[str, Any]] = Field(default_factory=list)
    lint_report: dict[str, Any] | None = None
    latest_run: dict[str, Any] | None = None
