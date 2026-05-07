# simd_agent/chat_models.py
"""Pydantic models for the /ws/chat endpoint."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Client → Server
# ---------------------------------------------------------------------------

class ChatHistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatContext(BaseModel):
    """Simulation context sent alongside each user turn."""
    run_id: str | None = None
    simulation_id: str | None = None
    mesh_id: str | None = None
    physics: dict[str, Any] | None = None
    solver: dict[str, Any] | None = None
    fluid: dict[str, Any] | None = None
    turbulence: dict[str, Any] | None = None
    patches: dict[str, Any] | None = None
    final_result: dict[str, Any] | None = None
    vtk_result: dict[str, Any] | None = None
    lint_result: dict[str, Any] | None = None
    generated_files: dict[str, str] | None = None


class ChatRequest(BaseModel):
    """A single user turn over the WebSocket.

    When ``mode`` is ``"precheck"``, the request originates from the ComposeView
    (simulation setup flow).  The chat service routes it to the setup
    conversation or analysis pipeline as appropriate.
    """
    message: str
    history: list[ChatHistoryMessage] = Field(default_factory=list)
    context: ChatContext = Field(default_factory=ChatContext)
    user_id: str | None = None
    simulation_id: str | None = None

    # ── Precheck / setup mode fields ──
    mode: Literal["chat", "precheck"] = "chat"
    has_mesh: bool = Field(default=False, alias="hasMesh")
    mesh_info: dict[str, Any] | None = Field(default=None, alias="meshInfo")
    confirm_analysis: bool = Field(default=False, alias="confirmAnalysis")
    conversation_summary: str | None = Field(default=None, alias="conversationSummary")
    simulation_context: dict[str, Any] | None = Field(default=None, alias="simulationContext")

    model_config = ConfigDict(populate_by_name=True)


# ---------------------------------------------------------------------------
# Server → Client (streamed JSON frames)
# ---------------------------------------------------------------------------

class ThinkingEvent(BaseModel):
    type: Literal["thinking"] = "thinking"
    delta: str


class TokenEvent(BaseModel):
    type: Literal["token"] = "token"
    delta: str


class ToolStartEvent(BaseModel):
    type: Literal["tool_start"] = "tool_start"
    tool: str
    label: str


class ToolResultEvent(BaseModel):
    type: Literal["tool_result"] = "tool_result"
    tool: str
    data: dict[str, Any]


class ArtifactEvent(BaseModel):
    type: Literal["artifact"] = "artifact"
    kind: Literal["markdown", "chart", "report", "report_request", "ready_to_analyze", "conversation_summary"]
    content: Any


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"
    suggested_actions: list[str] = Field(default_factory=list)


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str


# ---------------------------------------------------------------------------
# Query intent classification (output of QueryAnalyzer)
# ---------------------------------------------------------------------------

QueryCategory = Literal[
    "setup",          # simulation setup conversation (precheck mode)
    "data_plot",      # chart/graph of field values (pressure, temperature, etc.)
    "residuals",      # residual/convergence plots
    "data_query",     # questions about simulation results, values, statistics
    "file_inspect",   # view an OpenFOAM case file
    "cross_run",      # compare across multiple runs
    "report",         # generate a full report or PDF
    "troubleshoot",   # diagnose errors or failures
    "theory",         # general CFD/physics question (no data needed)
    "general",        # anything else
]


class ToolCallPlan(BaseModel):
    """A single tool call the analyzer recommends."""
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class DataNeeds(BaseModel):
    """Which data sources the snapshot builder should fetch."""
    sim_progress: bool = False
    field_ranges: bool = False
    vtk_result: bool = False
    generated_files: bool = False
    cross_run: bool = False
    patches: bool = False
    mesh_info: bool = False


class QueryIntent(BaseModel):
    """Structured output of the query analyzer."""
    category: str = "general"
    resolved_subject: str = ""
    tool_plan: list[ToolCallPlan] = Field(default_factory=list)
    data_needs: DataNeeds = Field(default_factory=DataNeeds)
    confidence: float = 0.5


# Union helper for serialisation
ChatEvent = ThinkingEvent | TokenEvent | ToolStartEvent | ToolResultEvent | ArtifactEvent | DoneEvent | ErrorEvent
