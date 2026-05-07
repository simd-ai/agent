# simd_agent/repositories/simulation_repo.py
"""Simulation data access."""

from typing import Any
from uuid import UUID

from simd_agent.repositories.base import PostgresRepository


class SimulationRepository(PostgresRepository):
    table = "simulations"
    pk = "id"
    columns = [
        "id", "user_id", "title", "active_step", "max_reached_step",
        "selected_preset_id", "user_prompt", "expert_mode",
        "is_from_scratch_mode", "active_tab",
        "created_at::text", "updated_at::text",
    ]

    async def update(self, id: UUID, data: dict[str, Any]) -> dict[str, Any] | None:
        """Override to auto-set updated_at."""
        if not data:
            return await self.get_by_id(id)

        serialized = self._serialize(data)
        set_clauses = ", ".join(f"{k} = :{k}" for k in serialized.keys())
        serialized["_pk"] = id

        return await self.execute_raw_one(
            f"UPDATE {self.table} SET {set_clauses}, updated_at = NOW() "
            f"WHERE {self.pk} = :_pk RETURNING {self._select_cols}",
            serialized,
        )
