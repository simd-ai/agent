from __future__ import annotations

# simd_agent/repositories/lint_repo.py
"""Lint report data access (many per simulation)."""

from typing import Any
from uuid import UUID

from simd_agent.repositories.base import PostgresRepository


class LintRepository(PostgresRepository):
    table = "lint_reports"
    pk = "id"
    columns = ["id", "simulation_id", "run_id", "is_valid", "issues", "created_at::text"]
    json_columns = {"issues"}

    async def get_latest(self, simulation_id: UUID) -> dict[str, Any] | None:
        return await self.execute_raw_one(
            f"SELECT {self._select_cols} FROM {self.table} "
            f"WHERE simulation_id = :id ORDER BY created_at DESC LIMIT 1",
            {"id": simulation_id},
        )
