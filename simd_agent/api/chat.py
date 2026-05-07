# simd_agent/api/chat.py
"""Chat message endpoints."""

from uuid import UUID

from fastapi import APIRouter, Depends

from simd_agent.api.auth import AuthenticatedUser, require_simulation_owner
from simd_agent.schemas.chat import ChatMessageCreate, ChatMessageOut, ChatMessagesBatch
from simd_agent.services import chat_service

router = APIRouter(prefix="/api/simulations", tags=["chat"])


@router.get("/{simulation_id}/chat")
async def get_chat_messages(
    simulation_id: UUID,
    _owner: AuthenticatedUser | None = Depends(require_simulation_owner),
) -> list[ChatMessageOut]:
    return await chat_service.list(simulation_id)


@router.post("/{simulation_id}/chat", status_code=201)
async def save_chat_message(
    simulation_id: UUID,
    body: ChatMessageCreate,
    _owner: AuthenticatedUser | None = Depends(require_simulation_owner),
) -> ChatMessageOut:
    return await chat_service.save(simulation_id, body)


@router.put("/{simulation_id}/chat")
async def replace_chat_messages(
    simulation_id: UUID,
    body: ChatMessagesBatch,
    _owner: AuthenticatedUser | None = Depends(require_simulation_owner),
) -> list[ChatMessageOut]:
    return await chat_service.replace_all(simulation_id, body)
