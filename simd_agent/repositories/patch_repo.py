from __future__ import annotations

# simd_agent/repositories/patch_repo.py
"""Patch config data access (many per simulation)."""

from typing import Any
from uuid import UUID

from simd_agent.repositories.base import PostgresRepository


class PatchRepository(PostgresRepository):
    table = "patch_configs"
    pk = "id"
    columns = [
        "id", "simulation_id", "patch_name", "patch_class",
        "patch_config", "patch_info", "boundary_hint", "status",
    ]
    json_columns = {"patch_config", "patch_info", "boundary_hint"}

    async def list_for_simulation(self, simulation_id: UUID) -> list[dict[str, Any]]:
        return await self.list(
            filters={"simulation_id": simulation_id},
            order_by="patch_name",
        )

    async def upsert_patch(self, simulation_id: UUID, data: dict[str, Any]) -> dict[str, Any]:
        data["simulation_id"] = simulation_id
        return await self.upsert(
            data=data,
            conflict_keys=["simulation_id", "patch_name"],
            update_keys=["patch_class", "patch_config", "patch_info", "boundary_hint", "status"],
        )
