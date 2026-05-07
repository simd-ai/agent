from __future__ import annotations

# simd_agent/repositories/progress_repo.py
"""Simulation progress data access (many per run)."""

from typing import Any
from uuid import UUID

from simd_agent.repositories.base import PostgresRepository


class ProgressRepository(PostgresRepository):
    table = "sim_progress"
    pk = "id"
    columns = [
        "id", "run_id", "iteration", "sim_time",
        "fields", "residuals", "courant", "continuity", "execution",
        "field_ranges",
    ]
    json_columns = {"fields", "residuals", "courant", "continuity", "execution", "field_ranges"}

    async def list_for_run(self, run_id: UUID) -> list[dict[str, Any]]:
        return await self.list(
            filters={"run_id": run_id},
            order_by="iteration ASC",
        )

    async def delete_for_run(self, run_id: UUID) -> None:
        await self.execute_write(
            f"DELETE FROM {self.table} WHERE run_id = :id",
            {"id": run_id},
        )
