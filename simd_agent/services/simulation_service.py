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

# RAS turbulence models — anything in this set implies a turbulent simulation.
_RAS_MODELS = {"kepsilon", "komegasst", "komega", "spalartallmaras", "realizablekepsilon", "rngkepsilon"}


def _reconcile_physics_turbulence(
    physics: dict | None,
    turbulence: dict | None,
) -> tuple[dict | None, dict | None]:
    """Enforce the flow_regime ↔ turbulence.model invariant at the save boundary.

    Both columns are written together by ``save_form_state`` /
    ``upsert_config``.  Without a guard they can drift apart:

      - Frontend skipped applying precheck's laminar turbulence model →
        ``cfd_physics.flow_regime = laminar`` but
        ``cfd_turbulence.model   = kOmegaSST`` (Zustand default leaks
        through).
      - Or vice versa: a user toggles regime in the UI but the legacy
        ``cfdTurbulence.model`` is not updated.

    Invariant applied here:

      * ``flow_regime == "laminar"``  →  ``turbulence.model`` is forced
        to ``"laminar"`` and RAS-only derived fields (``k``, ``omega``,
        ``epsilon``, ``nut``, ``wall_functions``) are stripped.  The
        numeric inputs (``turbulence_intensity``, ``turbulence_length_scale``,
        ``hydraulic_diameter``) are kept so the user's last-known values
        survive a regime toggle and reappear when they flip back to
        turbulent.

    Why ``flow_regime`` wins (not the RAS model name):

      * A user toggling regime in the UI emits ``setCFDPhysics({flowRegime: …})``
        — an explicit choice.  The Zustand store does *not* simultaneously
        reset ``cfdTurbulence.model``, so without this guard a deliberate
        toggle to laminar would persist as ``{laminar, kOmegaSST}`` —
        exactly the drift state the user reported.
      * ``cfdTurbulence.model`` defaults to ``kOmegaSST`` and is often
        never explicitly chosen — it's the Zustand initial value carrying
        through.  Treating it as authoritative would silently override
        explicit laminar choices.
      * For the U-bend precheck bug specifically, the upstream fix is in
        the precheck (no auto-demote from bbox-D_h Re).  Once that lands,
        the precheck no longer writes ``flow_regime=laminar`` for a
        turbulent case, so the reconciler never sees the conflict.

    Returns the (possibly modified) dicts so the caller can write them.
    """
    if physics is None and turbulence is None:
        return physics, turbulence

    # Work on shallow copies so we never mutate the caller's payload.
    p = dict(physics) if physics is not None else None
    t = dict(turbulence) if turbulence is not None else None

    flow_regime = (p.get("flow_regime") if p else None)
    flow_regime = flow_regime.lower() if isinstance(flow_regime, str) else None

    model = (t.get("model") if t else None)
    model_norm = model.lower().replace("_", "") if isinstance(model, str) else None

    # flow_regime=laminar forces a laminar (or stripped) turbulence model.
    if flow_regime == "laminar" and t is not None:
        if model is not None and model_norm != "laminar":
            t["model"] = "laminar"
            # Wipe RAS-only derived fields so the chat agent / orchestrator
            # never reports a turbulence quantity for a laminar case.
            for k in ("k", "omega", "epsilon", "nut", "wall_functions"):
                t.pop(k, None)
        return p, t

    return p, t


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

        # Reconcile flow_regime ↔ turbulence.model before persisting.
        # Only meaningful when at least one of the two fields is in this update.
        if "cfd_physics" in data or "cfd_turbulence" in data:
            phys, turb = _reconcile_physics_turbulence(
                data.get("cfd_physics"),
                data.get("cfd_turbulence"),
            )
            if phys is not None:
                data["cfd_physics"] = phys
            if turb is not None:
                data["cfd_turbulence"] = turb

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
            # Reconcile flow_regime ↔ turbulence.model so the two JSONB
            # columns can never drift apart on the way to the DB.
            if "cfd_physics" in cfg_fields or "cfd_turbulence" in cfg_fields:
                phys, turb = _reconcile_physics_turbulence(
                    cfg_fields.get("cfd_physics"),
                    cfg_fields.get("cfd_turbulence"),
                )
                if phys is not None:
                    cfg_fields["cfd_physics"] = phys
                if turb is not None:
                    cfg_fields["cfd_turbulence"] = turb

            data = {"simulation_id": simulation_id, **cfg_fields}
            await self.config_repo.upsert(
                data=data,
                conflict_keys=["simulation_id"],
                update_keys=list(cfg_fields.keys()),
            )
