# simd_agent/chat/__init__.py
"""Chat package — streaming CFD assistant over WebSocket."""

from simd_agent.chat.models import (
    ChatContext,
    ChatHistoryMessage,
    ChatRequest,
    DoneEvent,
    ErrorEvent,
    ThinkingEvent,
    TokenEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from simd_agent.chat.service import ChatService, get_chat_service
from simd_agent.chat.tools import SimulationSnapshot

__all__ = [
    "ChatContext",
    "ChatHistoryMessage",
    "ChatRequest",
    "ChatService",
    "DoneEvent",
    "ErrorEvent",
    "SimulationSnapshot",
    "ThinkingEvent",
    "TokenEvent",
    "ToolResultEvent",
    "ToolStartEvent",
    "get_chat_service",
]
