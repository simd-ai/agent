# simd_agent/chat_models.py
"""Pydantic models for the /ws/chat endpoint."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


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
    """A single user turn over the WebSocket."""
    message: str
    history: list[ChatHistoryMessage] = Field(default_factory=list)
    context: ChatContext = Field(default_factory=ChatContext)
    user_id: str | None = None
    simulation_id: str | None = None


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
    kind: Literal["markdown", "chart", "report", "report_request"]
    content: Any


class DoneEvent(BaseModel):
    type: Literal["done"] = "done"
    suggested_actions: list[str] = Field(default_factory=list)


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str


# Union helper for serialisation
ChatEvent = ThinkingEvent | TokenEvent | ToolStartEvent | ToolResultEvent | ArtifactEvent | DoneEvent | ErrorEvent
