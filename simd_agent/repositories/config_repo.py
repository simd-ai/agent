# simd_agent/repositories/config_repo.py
"""Simulation config data access (1:1 with simulations)."""

from simd_agent.repositories.base import PostgresRepository


class ConfigRepository(PostgresRepository):
    table = "simulation_config"
    pk = "simulation_id"
    columns = [
        "simulation_id", "case_spec", "cfd_physics", "cfd_solver",
        "cfd_fluid", "cfd_turbulence", "cfd_derived", "cfd_regions",
    ]
    json_columns = {
        "case_spec", "cfd_physics", "cfd_solver",
        "cfd_fluid", "cfd_turbulence", "cfd_derived", "cfd_regions",
    }
