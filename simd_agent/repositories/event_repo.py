from __future__ import annotations

# simd_agent/repositories/event_repo.py
"""Event data access."""

from typing import Any
from uuid import UUID

from simd_agent.repositories.base import PostgresRepository


class EventRepository(PostgresRepository):
    table = "events"
    pk = "id"
    columns = ["id", "run_id", "seq", "ts::text", "level", "type", "message", "payload"]
    json_columns = {"payload"}

    async def list_for_run(self, run_id: UUID, after_seq: int | None = None) -> list[dict[str, Any]]:
        if after_seq is not None:
            return await self.execute_raw(
                f"SELECT {self._select_cols} FROM {self.table} "
                f"WHERE run_id = :run_id AND seq > :seq ORDER BY seq ASC",
                {"run_id": run_id, "seq": after_seq},
            )
        return await self.execute_raw(
            f"SELECT {self._select_cols} FROM {self.table} "
            f"WHERE run_id = :run_id ORDER BY seq ASC",
            {"run_id": run_id},
        )
