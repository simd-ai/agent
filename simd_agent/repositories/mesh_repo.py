# simd_agent/repositories/mesh_repo.py
"""Mesh info data access (1:1 with simulations)."""

from simd_agent.repositories.base import PostgresRepository


class MeshRepository(PostgresRepository):
    table = "mesh_info"
    pk = "simulation_id"
    columns = [
        "simulation_id", "mesh_id", "file_name", "uploaded_at::text",
        "patches", "viewer_artifacts", "check_mesh",
    ]
    json_columns = {"patches", "viewer_artifacts", "check_mesh"}
