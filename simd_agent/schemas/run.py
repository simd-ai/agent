# simd_agent/schemas/run.py
"""Run, event, and simulation progress request/response schemas."""

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class RunCreate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: UUID
    simulation_id: UUID = Field(alias="simulationId")
    label: str | None = None
    type: str = "full"
    op: str = "CFD_CODEGEN_RUN"
    provider: str = "gemini"
    prompt_pack: str = Field(default="simd", alias="promptPack")
    user_requirements: str = Field(default="", alias="userRequirements")
    user_prompt_snapshot: str | None = Field(default=None, alias="userPromptSnapshot")


class RunUpdate(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    status: str | None = None
    lint_result: dict[str, Any] | None = Field(default=None, alias="lintResult")
    planning_result: dict[str, Any] | None = Field(default=None, alias="planningResult")
    generated_files: dict[str, Any] | None = Field(default=None, alias="generatedFiles")
    file_generation_map: dict[str, Any] | None = Field(default=None, alias="fileGenerationMap")
    final_result: dict[str, Any] | None = Field(default=None, alias="finalResult")
    vtk_result: dict[str, Any] | None = Field(default=None, alias="vtkResult")
    error_message: str | None = Field(default=None, alias="errorMessage")


class RunComplete(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    status: str  # succeeded | failed | cancelled
    final_result: dict[str, Any] | None = Field(default=None, alias="finalResult")
    vtk_result: dict[str, Any] | None = Field(default=None, alias="vtkResult")
    generated_files: dict[str, Any] | None = Field(default=None, alias="generatedFiles")
    file_generation_map: dict[str, Any] | None = Field(default=None, alias="fileGenerationMap")
    lint_result: dict[str, Any] | None = Field(default=None, alias="lintResult")
    planning_result: dict[str, Any] | None = Field(default=None, alias="planningResult")
    error_message: str | None = Field(default=None, alias="errorMessage")


class RunOut(BaseModel):
    id: UUID
    simulation_id: UUID | None
    label: str | None
    type: str
    status: str
    op: str | None
    provider: str | None
    solver: str | None
    attempts: int
    lint_result: dict[str, Any] | None = None
    planning_result: dict[str, Any] | None = None
    generated_files: dict[str, Any] | None = None
    file_generation_map: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    final_result: dict[str, Any] | None = None
    vtk_result: dict[str, Any] | None = None
    error_message: str | None = None
    user_prompt_snapshot: str | None = None
    started_at: str
    completed_at: str | None = None


class EventOut(BaseModel):
    id: UUID
    run_id: UUID
    seq: int
    ts: str
    level: str
    type: str
    message: str
    payload: dict[str, Any]


class SimProgressEntry(BaseModel):
    iteration: int
    sim_time: float | None = None
    fields: dict[str, Any] | None = None
    residuals: dict[str, Any] | None = None
    courant: dict[str, Any] | None = None
    continuity: dict[str, Any] | None = None
    execution: dict[str, Any] | None = None
    field_ranges: dict[str, Any] | None = None


class SimProgressBatch(BaseModel):
    entries: list[SimProgressEntry]


class ApplyRecommendationRequest(BaseModel):
    """Request body for applying a convergence recommendation."""
    type: str  # relaxation | time_step | more_iterations | mesh_refinement
    changes: dict[str, float] = Field(default_factory=dict)


class ApplyRecommendationResponse(BaseModel):
    """Response from applying a recommendation."""
    modified_files: dict[str, str]  # path → new content (only changed files)
    changed_keys: list[str]  # list of file paths that were modified


class SimProgressOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: UUID | None = None
    run_id: UUID | None = Field(None, alias="runId")
    iteration: int
    sim_time: float | None = Field(None, alias="simTime")
    fields: dict[str, Any] | list[str] | None = None
    residuals: dict[str, Any] | None = None
    courant: dict[str, Any] | None = None
    continuity: dict[str, Any] | None = None
    execution: dict[str, Any] | None = None
    field_ranges: dict[str, Any] | None = Field(None, alias="fieldRanges")
