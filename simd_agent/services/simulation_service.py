from __future__ import annotations

# simd_agent/services/simulation_service.py
"""Simulation business logic — manages simulations + their 1:1 config."""

from uuid import UUID

from simd_agent.repositories.simulation_repo import SimulationRepository
from simd_agent.repositories.config_repo import ConfigRepository
from simd_agent.schemas.simulation import (
    FormStateUpdate,
    SimulationConfigOut,
    SimulationConfigUpsert,
    SimulationCreate,
    SimulationOut,
    SimulationUpdate,
)

_CONFIG_KEYS = {"case_spec", "cfd_physics", "cfd_solver", "cfd_fluid", "cfd_turbulence", "cfd_derived"}


class SimulationService:
    def __init__(self, sim_repo: SimulationRepository, config_repo: ConfigRepository):
        self.sim_repo = sim_repo
        self.config_repo = config_repo

    async def create(self, body: SimulationCreate) -> SimulationOut:
        sim = await self.sim_repo.create(body.model_dump())
        # Auto-provision empty config row
        await self.config_repo.create({"simulation_id": sim["id"]})
        return SimulationOut(**sim)

    async def get(self, simulation_id: UUID) -> SimulationOut | None:
        row = await self.sim_repo.get_by_id(simulation_id)
        return SimulationOut(**row) if row else None

    async def list(self, user_id: UUID | None = None) -> list[SimulationOut]:
        filters = {"user_id": user_id} if user_id else None
        rows = await self.sim_repo.list(filters=filters, order_by="created_at DESC")
        return [SimulationOut(**row) for row in rows]

    async def update(self, simulation_id: UUID, body: SimulationUpdate) -> SimulationOut | None:
        data = body.model_dump(exclude_none=True)
        row = await self.sim_repo.update(simulation_id, data)
        return SimulationOut(**row) if row else None

    async def delete(self, simulation_id: UUID) -> bool:
        return await self.sim_repo.delete(simulation_id)

    # ── Config ───────────────────────────────────────────────────────

    async def get_config(self, simulation_id: UUID) -> SimulationConfigOut | None:
        row = await self.config_repo.get_by_id(simulation_id)
        return SimulationConfigOut(**row) if row else None

    async def upsert_config(self, simulation_id: UUID, body: SimulationConfigUpsert) -> SimulationConfigOut:
        data = {"simulation_id": simulation_id}
        for key in _CONFIG_KEYS:
            val = getattr(body, key)
            if val is not None:
                data[key] = val

        row = await self.config_repo.upsert(
            data=data,
            conflict_keys=["simulation_id"],
            update_keys=[k for k in _CONFIG_KEYS if k in data and k != "simulation_id"],
        )
        return SimulationConfigOut(**row)

    # ── Form State (combined save) ───────────────────────────────────

    async def save_form_state(self, simulation_id: UUID, body: FormStateUpdate) -> None:
        raw = body.model_dump(exclude_none=True)

        sim_fields = {k: v for k, v in raw.items() if k not in _CONFIG_KEYS}
        cfg_fields = {k: v for k, v in raw.items() if k in _CONFIG_KEYS}

        if sim_fields:
            await self.sim_repo.update(simulation_id, sim_fields)

        if cfg_fields:
            data = {"simulation_id": simulation_id, **cfg_fields}
            await self.config_repo.upsert(
                data=data,
                conflict_keys=["simulation_id"],
                update_keys=list(cfg_fields.keys()),
            )
