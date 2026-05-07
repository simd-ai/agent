from __future__ import annotations

# simd_agent/services/chat_service.py
"""Chat message business logic."""

import logging
from uuid import UUID

from simd_agent.repositories.chat_repo import ChatRepository
from simd_agent.schemas.chat import ChatMessageCreate, ChatMessageOut, ChatMessagesBatch

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, repo: ChatRepository):
        self.repo = repo

    async def list(self, simulation_id: UUID) -> list[ChatMessageOut]:
        rows = await self.repo.list_for_simulation(simulation_id)
        return [ChatMessageOut(**row) for row in rows]

    async def save(self, simulation_id: UUID, body: ChatMessageCreate) -> ChatMessageOut:
        data = body.model_dump(exclude_none=True)
        data["simulation_id"] = simulation_id

        if body.id:
            # Idempotent insert
            try:
                row = await self.repo.create(data)
            except Exception:
                row = await self.repo.get_by_id(body.id)
        else:
            row = await self.repo.create(data)

        return ChatMessageOut(**row)

    async def replace_all(self, simulation_id: UUID, body: ChatMessagesBatch) -> list[ChatMessageOut]:
        logger.info("[ChatService] replace_all sim=%s incoming=%d messages, roles=%s",
                     simulation_id, len(body.messages),
                     [m.role for m in body.messages])

        messages = [
            {**msg.model_dump(exclude_none=True), "simulation_id": simulation_id}
            for msg in body.messages
        ]
        rows = await self.repo.sync_messages(simulation_id, messages)

        logger.info("[ChatService] replace_all sim=%s saved=%d rows", simulation_id, len(rows))
        return [ChatMessageOut(**row) for row in rows]
