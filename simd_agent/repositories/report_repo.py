from __future__ import annotations

# simd_agent/repositories/report_repo.py
"""Simulation report data access (many per simulation)."""

from typing import Any
from uuid import UUID

from simd_agent.repositories.base import PostgresRepository


class ReportRepository(PostgresRepository):
    table = "simulation_reports"
    pk = "id"
    columns = [
        "id", "simulation_id", "run_id", "report_type",
        "file_name", "storage_key", "created_at::text",
    ]
    json_columns: set[str] = set()

    async def list_for_simulation(self, simulation_id: UUID) -> list[dict[str, Any]]:
        return await self.execute_raw(
            f"SELECT {self._select_cols} FROM {self.table} "
            f"WHERE simulation_id = :sid ORDER BY created_at DESC",
            {"sid": simulation_id},
        )
