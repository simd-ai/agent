from __future__ import annotations

# simd_agent/repositories/chat_repo.py
"""Chat message data access (many per simulation)."""

import json
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import text

from simd_agent.db import get_session
from simd_agent.repositories.base import PostgresRepository


class ChatRepository(PostgresRepository):
    table = "chat_messages"
    pk = "id"
    columns = ["id", "simulation_id", "role", "content", "suggested_actions", "artifacts", "timestamp::text"]
    json_columns = {"suggested_actions", "artifacts"}

    async def list_for_simulation(self, simulation_id: UUID) -> list[dict[str, Any]]:
        return await self.list(
            filters={"simulation_id": simulation_id},
            order_by="timestamp ASC",
        )

    async def delete_for_simulation(self, simulation_id: UUID) -> None:
        await self.execute_write(
            f"DELETE FROM {self.table} WHERE simulation_id = :id",
            {"id": simulation_id},
        )

    async def sync_messages(
        self,
        simulation_id: UUID,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Idempotent sync: delete stale rows + upsert current ones in a single transaction."""
        async with get_session() as session:
            msg_ids = [m["id"] for m in messages]

            # 1. Delete messages no longer in the list
            if msg_ids:
                await session.execute(
                    text(
                        f"DELETE FROM {self.table} "
                        f"WHERE simulation_id = :sim_id "
                        f"AND id != ALL(:ids)"
                    ),
                    {"sim_id": simulation_id, "ids": msg_ids},
                )
            else:
                await session.execute(
                    text(f"DELETE FROM {self.table} WHERE simulation_id = :sim_id"),
                    {"sim_id": simulation_id},
                )

            # 2. Upsert each message (stable ordering via timestamp offset)
            base_ts = datetime.now(timezone.utc)
            for i, msg in enumerate(messages):
                sa = msg.get("suggested_actions")
                sa_json = json.dumps(sa) if sa is not None and not isinstance(sa, str) else sa
                arts = msg.get("artifacts")
                arts_json = json.dumps(arts) if arts is not None and not isinstance(arts, str) else arts
                await session.execute(
                    text(
                        f"INSERT INTO {self.table} (id, simulation_id, role, content, suggested_actions, artifacts, timestamp) "
                        f"VALUES (:id, :sim_id, :role, :content, :sa, :arts, :ts) "
                        f"ON CONFLICT (id) DO UPDATE SET "
                        f"content = EXCLUDED.content, "
                        f"role = EXCLUDED.role, "
                        f"suggested_actions = EXCLUDED.suggested_actions, "
                        f"artifacts = EXCLUDED.artifacts"
                    ),
                    {
                        "id": msg["id"],
                        "sim_id": simulation_id,
                        "role": msg["role"],
                        "content": msg["content"],
                        "sa": sa_json,
                        "arts": arts_json,
                        "ts": base_ts + timedelta(milliseconds=i),
                    },
                )

            # 3. Return the final state
            result = await session.execute(
                text(f"SELECT {self._select_cols} FROM {self.table} WHERE simulation_id = :sim_id ORDER BY timestamp ASC"),
                {"sim_id": simulation_id},
            )
            return [dict(row) for row in result.mappings().all()]
