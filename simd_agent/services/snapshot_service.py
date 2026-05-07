from __future__ import annotations

# simd_agent/services/snapshot_service.py
"""Snapshot service — loads complete simulation state in one call."""

import asyncio
import logging
import time
from uuid import UUID

logger = logging.getLogger(__name__)

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from simd_agent.db import get_session
from simd_agent.repositories.simulation_repo import SimulationRepository
from simd_agent.repositories.config_repo import ConfigRepository
from simd_agent.repositories.mesh_repo import MeshRepository
from simd_agent.repositories.patch_repo import PatchRepository
from simd_agent.repositories.precheck_repo import PrecheckRepository
from simd_agent.repositories.lint_repo import LintRepository
from simd_agent.repositories.run_repo import RunRepository
from simd_agent.repositories.chat_repo import ChatRepository
from simd_agent.schemas.simulation import (
    SnapshotOut,
    SnapshotPrimaryOut,
    SnapshotEssentialsOut,
    SnapshotConfigOut,
    SnapshotRunOut,
)


class SnapshotService:
    def __init__(
        self,
        sim_repo: SimulationRepository,
        config_repo: ConfigRepository,
        mesh_repo: MeshRepository,
        patch_repo: PatchRepository,
        precheck_repo: PrecheckRepository,
        lint_repo: LintRepository,
        run_repo: RunRepository,
        chat_repo: ChatRepository | None = None,
    ):
        self.sim_repo = sim_repo
        self.config_repo = config_repo
        self.mesh_repo = mesh_repo
        self.patch_repo = patch_repo
        self.precheck_repo = precheck_repo
        self.lint_repo = lint_repo
        self.run_repo = run_repo
        self.chat_repo = chat_repo

    async def load(self, simulation_id: UUID) -> SnapshotOut | None:
        t0 = time.perf_counter()
        sim = await self.sim_repo.get_by_id(simulation_id)
        t_sim = time.perf_counter()
        if not sim:
            return None

        async def _timed(name: str, coro):
            s = time.perf_counter()
            result = await coro
            logger.info("[snapshot] %s: %.0fms", name, (time.perf_counter() - s) * 1000)
            return result

        async def _empty_list() -> list:
            return []

        config, mesh, patches, precheck, lint, latest_run, chat = await asyncio.gather(
            _timed("config", self.config_repo.get_by_id(simulation_id)),
            _timed("mesh", self.mesh_repo.get_by_id(simulation_id)),
            _timed("patches", self.patch_repo.list_for_simulation(simulation_id)),
            _timed("precheck", self.precheck_repo.get_by_id(simulation_id)),
            _timed("lint", self.lint_repo.get_latest(simulation_id)),
            _timed("latest_run", self.run_repo.get_latest(simulation_id)),
            _timed("chat", self.chat_repo.list_for_simulation(simulation_id) if self.chat_repo else _empty_list()),
        )

        t_end = time.perf_counter()
        logger.info("[snapshot] sim: %.0fms | parallel: %.0fms | total: %.0fms",
                    (t_sim - t0) * 1000, (t_end - t_sim) * 1000, (t_end - t0) * 1000)

        return SnapshotOut(
            simulation=sim,
            config=config,
            mesh=mesh,
            patches=patches,
            precheck=precheck,
            lint_report=lint,
            latest_run=latest_run,
            chat=chat,
        )

    # ── Progressive snapshot groups ────────────────────────────────

    async def load_primary(self, simulation_id: UUID) -> SnapshotPrimaryOut | None:
        """Tier 0: just the simulation row — single DB query, <100 ms."""
        t0 = time.perf_counter()
        sim = await self.sim_repo.get_by_id(simulation_id)
        logger.info("[snapshot/primary] sim: %.0fms", (time.perf_counter() - t0) * 1000)
        if not sim:
            return None
        return SnapshotPrimaryOut(simulation=sim)

    async def load_essentials(self, simulation_id: UUID) -> SnapshotEssentialsOut | None:
        """Group 1: simulation + chat + precheck + mesh — all parallel."""
        t0 = time.perf_counter()

        async def _timed(name: str, coro):
            s = time.perf_counter()
            result = await coro
            logger.info("[snapshot/essentials] %s: %.0fms", name, (time.perf_counter() - s) * 1000)
            return result

        async def _empty_list() -> list:
            return []

        sim, chat, precheck, mesh = await asyncio.gather(
            _timed("sim", self.sim_repo.get_by_id(simulation_id)),
            _timed("chat", self.chat_repo.list_for_simulation(simulation_id) if self.chat_repo else _empty_list()),
            _timed("precheck", self.precheck_repo.get_by_id(simulation_id)),
            _timed("mesh", self.mesh_repo.get_by_id(simulation_id)),
        )

        if not sim:
            return None

        logger.info("[snapshot/essentials] total: %.0fms", (time.perf_counter() - t0) * 1000)
        return SnapshotEssentialsOut(simulation=sim, chat=chat, precheck=precheck, mesh=mesh)

    async def load_background(self, simulation_id: UUID) -> tuple[SnapshotConfigOut, SnapshotRunOut]:
        """Groups 2+3 combined: config + mesh + patches + lint + run — all parallel."""
        t0 = time.perf_counter()

        async def _timed(name: str, coro):
            s = time.perf_counter()
            result = await coro
            logger.info("[snapshot/background] %s: %.0fms", name, (time.perf_counter() - s) * 1000)
            return result

        config, mesh, patches, lint, latest_run = await asyncio.gather(
            _timed("config", self.config_repo.get_by_id(simulation_id)),
            _timed("mesh", self.mesh_repo.get_by_id(simulation_id)),
            _timed("patches", self.patch_repo.list_for_simulation(simulation_id)),
            _timed("lint", self.lint_repo.get_latest(simulation_id)),
            _timed("run", self.run_repo.get_latest(simulation_id)),
        )

        logger.info("[snapshot/background] total: %.0fms", (time.perf_counter() - t0) * 1000)

        # Debug: patch data
        print(f"[SNAPSHOT] load_background sim={simulation_id} patches={len(patches)}")
        if patches:
            for i, p in enumerate(patches):
                pc = p.get("patch_config")
                pc_keys = list(pc.keys()) if isinstance(pc, dict) else None
                print(f"[SNAPSHOT]   patch[{i}] name={p.get('patch_name')} class={p.get('patch_class')} status={p.get('status')} config_keys={pc_keys}")
                if isinstance(pc, dict):
                    # Show first 200 chars of config
                    import json as _json
                    print(f"[SNAPSHOT]   patch[{i}] config={_json.dumps(pc)[:200]}")
        # Debug: config data
        if config:
            cfg_keys = list(config.keys())
            print(f"[SNAPSHOT] config keys={cfg_keys}")
            for k in ['cfd_physics', 'cfd_solver', 'cfd_fluid', 'cfd_turbulence']:
                v = config.get(k)
                if isinstance(v, dict):
                    import json as _json
                    print(f"[SNAPSHOT]   {k} = {_json.dumps(v)[:200]}")
        if latest_run:
            gf = latest_run.get("generated_files")
            fgm = latest_run.get("file_generation_map")
            gf_keys = list(gf.keys()) if isinstance(gf, dict) else []
            fgm_keys = list(fgm.keys()) if isinstance(fgm, dict) else []
            print(f"[SNAPSHOT] load_background sim={simulation_id} run={latest_run.get('id')} status={latest_run.get('status')}")
            print(f"[SNAPSHOT]   generated_files: {len(gf_keys)} files, keys={gf_keys[:5]}{'...' if len(gf_keys) > 5 else ''}")
            print(f"[SNAPSHOT]   file_generation_map: {len(fgm_keys)} files, keys={fgm_keys[:5]}{'...' if len(fgm_keys) > 5 else ''}")
        else:
            print(f"[SNAPSHOT] load_background sim={simulation_id} — no run found")
        return (
            SnapshotConfigOut(config=config, mesh=mesh, patches=patches, lint_report=lint),
            SnapshotRunOut(latest_run=latest_run),
        )
