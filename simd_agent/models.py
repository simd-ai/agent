# simd_agent/models.py
"""Pydantic models for WebSocket protocol and domain objects."""

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# --- Enums ---

class Operation(str, Enum):
    """Supported operations."""
    CFD_LINT = "CFD_LINT"
    CFD_CODEGEN_RUN = "CFD_CODEGEN_RUN"


class EventLevel(str, Enum):
    """Event severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


class RunStatus(str, Enum):
    """Overall run status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    NOT_CLEAR = "not_clear"


class FlowRegime(str, Enum):
    """CFD flow regime classification."""
    LAMINAR = "laminar"
    TRANSITIONAL = "transitional"
    TURBULENT = "turbulent"


class SandboxState(str, Enum):
    """Sandbox run states."""
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


# --- Client -> Server ---

class Constraints(BaseModel):
    """Constraints for the run."""
    max_retries: int = Field(default=3, ge=1, le=10)
    solver_preference: str | None = None
    mesh_preference: str | None = None
    timeout_seconds: int = Field(default=300, ge=30, le=3600)


class Metadata(BaseModel):
    """Optional metadata for tracking."""
    user_id: str | None = None
    project_id: str | None = None
    session_id: str | None = None
    tags: list[str] = Field(default_factory=list)


class StartRequest(BaseModel):
    """Initial WebSocket message from client to start a run."""
    op: Operation
    provider: str = Field(default="gemini3")
    prompt_pack: str = Field(default="simd")
    user_requirements: str = Field(
        ...,
        min_length=1,
        description="Natural language description of what the user wants to simulate",
    )
    simulation_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Partial or complete simulation configuration",
    )
    constraints: Constraints = Field(default_factory=Constraints)
    metadata: Metadata = Field(default_factory=Metadata)


# --- Server -> Client ---

class AgentEvent(BaseModel):
    """Event streamed from server to client via WebSocket."""
    run_id: UUID
    seq: int = Field(ge=0, description="Monotonically increasing sequence number")
    ts: datetime = Field(default_factory=datetime.utcnow)
    level: EventLevel = EventLevel.INFO
    type: str = Field(
        ...,
        description="Event type identifier",
    )
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)
    
    def to_ws_message(self) -> dict[str, Any]:
        """Serialize for WebSocket transmission."""
        return {
            "run_id": str(self.run_id),
            "seq": self.seq,
            "ts": self.ts.isoformat(),
            "level": self.level.value,
            "type": self.type,
            "message": self.message,
            "payload": self.payload,
        }


# --- Event Types (as constants for type safety) ---

class EventTypes:
    """Standard event type identifiers."""
    # Lifecycle
    RUN_STARTED = "run_started"
    RUN_SUCCEEDED = "run_succeeded"
    RUN_FAILED = "run_failed"
    SIMULATION_NOT_CLEAR = "simulation_not_clear"
    
    # Linting
    LINT_STARTED = "lint_started"
    LINT_RESULT = "lint_result"
    
    # Planning
    PLANNING_STARTED = "planning_started"
    PLANNING_COMPLETE = "planning_complete"
    SUBAGENT_STARTED = "subagent_started"
    SUBAGENT_UPDATE = "subagent_update"
    SUBAGENT_COMPLETE = "subagent_complete"
    
    # Codegen
    CODEGEN_STARTED = "codegen_started"
    CODEGEN_ITERATION = "codegen_iteration"
    CODEGEN_COMPLETE = "codegen_complete"
    
    # Sandbox
    SANDBOX_SUBMITTED = "sandbox_submitted"
    SANDBOX_STATUS = "sandbox_status"
    SANDBOX_LOGS = "sandbox_logs"
    SANDBOX_SUCCEEDED = "sandbox_succeeded"
    SANDBOX_FAILED = "sandbox_failed"
    
    # Self-healing
    ERROR_SUMMARY = "error_summary"
    RETRYING = "retrying"
    
    # Final
    FINAL = "final"


# --- Linting Models ---

class ApplyChange(BaseModel):
    """A recommended change to the simulation config."""
    path: str = Field(..., description="Dot-path or key to the config field")
    value: Any = Field(..., description="Recommended value")
    reason: str = Field(..., description="Why this change is recommended")
    severity: Literal["info", "warning", "error"] = "info"


class LintIssue(BaseModel):
    """A validation issue found during linting."""
    code: str = Field(..., description="Issue code (e.g., 'INVALID_UNITS')")
    path: str | None = Field(None, description="Config path where issue was found")
    message: str
    severity: Literal["warning", "error"]


class LintResult(BaseModel):
    """Result of CFD linting."""
    validated_config: dict[str, Any]
    apply_changes: list[ApplyChange] = Field(default_factory=list)
    issues: list[LintIssue] = Field(default_factory=list)
    detected_case_type: str | None = None
    detected_regime: FlowRegime | None = None
    selected_solver: str | None = None
    reynolds_number: float | None = None


# --- Planning Models ---

class WorkItem(BaseModel):
    """A unit of work for parallel sub-agents."""
    id: str
    task: str = Field(..., description="Task identifier (e.g., 'choose_solver')")
    description: str
    priority: int = Field(default=1, ge=1, le=10)
    dependencies: list[str] = Field(default_factory=list)


class SubAgentResult(BaseModel):
    """Result from a sub-agent task."""
    work_item_id: str
    task: str
    result: dict[str, Any]
    duration_ms: int


class PlanningResult(BaseModel):
    """Result of the planning phase."""
    work_items: list[WorkItem]
    case_type: str
    regime: FlowRegime | None = None
    solver: str
    turbulence_model: str | None = None
    mesh_strategy: str
    sub_results: list[SubAgentResult] = Field(default_factory=list)


# --- Sandbox Models ---

class SandboxSubmitResponse(BaseModel):
    """Response from sandbox run submission."""
    run_id: str


class SandboxStatus(BaseModel):
    """Status of a sandbox run."""
    state: SandboxState
    exit_code: int | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    duration_seconds: float | None = None


class SandboxArtifact(BaseModel):
    """An artifact produced by a sandbox run."""
    name: str
    path: str
    size_bytes: int
    download_url: str | None = None


class SandboxArtifactsResponse(BaseModel):
    """Response containing sandbox run artifacts."""
    artifacts: list[SandboxArtifact]


# --- Error Summary Models ---

class ErrorSummary(BaseModel):
    """Summary of sandbox execution error."""
    root_cause: str
    actionable_changes: list[dict[str, Any]] = Field(
        default_factory=list,
        description="List of changes to apply to fix the error",
    )
    affected_files: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


# --- Final Result Models ---

class FinalResult(BaseModel):
    """Final result payload for the 'final' event."""
    status: RunStatus
    validated_config: dict[str, Any] | None = None
    artifacts: list[SandboxArtifact] = Field(default_factory=list)
    iterations: int = 0
    retries: int = 0
    summary: str = ""
    case_type: str | None = None
    solver: str | None = None
    error: str | None = None


# --- Database Row Models ---

class RunRow(BaseModel):
    """Database row for a run."""
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    op: Operation
    status: RunStatus = RunStatus.PENDING
    provider: str
    prompt_pack: str
    user_requirements: str
    simulation_config: dict[str, Any]
    validated_config: dict[str, Any] | None = None
    attempts: int = 0
    result: dict[str, Any] | None = None


class EventRow(BaseModel):
    """Database row for an event."""
    id: UUID = Field(default_factory=uuid4)
    run_id: UUID
    seq: int
    ts: datetime
    level: EventLevel
    type: str
    message: str
    payload: dict[str, Any]
