from __future__ import annotations

# simd_agent/repositories/run_repo.py
"""Run data access."""

from typing import Any
from uuid import UUID

from simd_agent.repositories.base import PostgresRepository


class RunRepository(PostgresRepository):
    table = "runs"
    pk = "id"
    columns = [
        "id", "simulation_id", "label", "type", "status", "op", "provider",
        "prompt_pack", "user_requirements",
        "solver", "attempts", "lint_result", "planning_result",
        "generated_files", "file_generation_map", "final_result",
        "vtk_result", "error_message", "user_prompt_snapshot",
        "result",
        "started_at::text", "completed_at::text",
    ]
    json_columns = {
        "simulation_config", "validated_config", "lint_result",
        "planning_result", "generated_files", "file_generation_map",
        "final_result", "vtk_result", "result",
    }

    async def get_latest(self, simulation_id: UUID) -> dict[str, Any] | None:
        return await self.execute_raw_one(
            f"SELECT {self._select_cols} FROM {self.table} "
            f"WHERE simulation_id = :sim_id ORDER BY started_at DESC LIMIT 1",
            {"sim_id": simulation_id},
        )

    async def complete(self, id: UUID, data: dict[str, Any]) -> dict[str, Any] | None:
        """Mark a run as completed with final data."""
        serialized = self._serialize(data)
        set_parts = ["completed_at = NOW()"]
        for k in serialized:
            set_parts.append(f"{k} = :{k}")
        serialized["_pk"] = id

        return await self.execute_raw_one(
            f"UPDATE {self.table} SET {', '.join(set_parts)} "
            f"WHERE {self.pk} = :_pk RETURNING {self._select_cols}",
            serialized,
        )
