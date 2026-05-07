# simd_agent/schemas/chat.py
"""Chat message request/response schemas."""

from typing import Any
from uuid import UUID

from pydantic import BaseModel


class ChatArtifactItem(BaseModel):
    kind: str  # chart | markdown | report | report_request
    content: str


class ChatMessageCreate(BaseModel):
    id: UUID | None = None
    role: str  # user | assistant
    content: str
    suggested_actions: list[str] | None = None
    artifacts: list[ChatArtifactItem] | None = None


class ChatMessageOut(BaseModel):
    id: UUID
    simulation_id: UUID
    role: str
    content: str
    suggested_actions: list[str] | None
    artifacts: list[ChatArtifactItem] | None = None
    timestamp: str


class ChatMessagesBatch(BaseModel):
    messages: list[ChatMessageCreate]
