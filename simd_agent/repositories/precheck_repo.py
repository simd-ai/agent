# simd_agent/repositories/precheck_repo.py
"""Precheck history data access (1:1 with simulations)."""

from simd_agent.repositories.base import PostgresRepository


class PrecheckRepository(PostgresRepository):
    table = "precheck_history"
    pk = "simulation_id"
    columns = [
        "simulation_id", "submitted_prompt", "mesh_name", "mesh_cells",
        "steps", "review_thoughts", "review_items", "suggested_config",
        "created_at::text",
    ]
    json_columns = {"steps", "review_items", "suggested_config"}
