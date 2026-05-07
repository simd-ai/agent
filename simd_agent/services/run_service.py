from __future__ import annotations

# simd_agent/services/run_service.py
"""Run business logic — manages runs, events, and simulation progress."""

from typing import Any
from uuid import UUID

from simd_agent.repositories.run_repo import RunRepository
from simd_agent.repositories.event_repo import EventRepository
from simd_agent.repositories.progress_repo import ProgressRepository
from simd_agent.schemas.run import (
    EventOut,
    RunComplete,
    RunCreate,
    RunOut,
    RunUpdate,
    SimProgressBatch,
    SimProgressOut,
)


class RunService:
    def __init__(
        self,
        run_repo: RunRepository,
        event_repo: EventRepository,
        progress_repo: ProgressRepository,
    ):
        self.run_repo = run_repo
        self.event_repo = event_repo
        self.progress_repo = progress_repo

    # ── Runs ─────────────────────────────────────────────────────────

    async def create(self, body: RunCreate) -> RunOut:
        data = body.model_dump()
        # Try insert, fall back to get on unique-violation conflict (idempotent)
        try:
            row = await self.run_repo.create(data)
        except Exception as exc:
            # Only swallow unique-violation conflicts; re-raise everything else
            exc_str = str(exc).lower()
            if "unique" in exc_str or "duplicate" in exc_str or "conflict" in exc_str:
                row = await self.run_repo.get_by_id(body.id)
            else:
                raise
        if row is None:
            raise ValueError(f"Failed to create or fetch run {body.id}")
        return RunOut(**row)

    async def get(self, run_id: UUID) -> RunOut | None:
        row = await self.run_repo.get_by_id(run_id)
        return RunOut(**row) if row else None

    async def list(
        self,
        simulation_id: UUID | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[RunOut]:
        filters: dict[str, Any] = {}
        if simulation_id:
            filters["simulation_id"] = simulation_id
        if status:
            filters["status"] = status

        rows = await self.run_repo.list(
            filters=filters or None,
            order_by="started_at DESC",
            limit=limit,
        )
        return [RunOut(**row) for row in rows]

    async def get_latest(self, simulation_id: UUID) -> RunOut | None:
        row = await self.run_repo.get_latest(simulation_id)
        return RunOut(**row) if row else None

    async def update(self, run_id: UUID, body: RunUpdate) -> RunOut | None:
        data = body.model_dump(exclude_none=True)
        row = await self.run_repo.update(run_id, data)
        return RunOut(**row) if row else None

    async def complete(self, run_id: UUID, body: RunComplete) -> RunOut | None:
        data = body.model_dump(exclude_none=True)
        print(f"[SVC] complete run={run_id} keys_being_written={list(data.keys())}")
        row = await self.run_repo.complete(run_id, data)
        return RunOut(**row) if row else None

    # ── Events ───────────────────────────────────────────────────────

    async def list_events(self, run_id: UUID, after_seq: int | None = None) -> list[EventOut]:
        rows = await self.event_repo.list_for_run(run_id, after_seq)
        return [EventOut(**row) for row in rows]

    # ── Progress ─────────────────────────────────────────────────────

    async def list_progress(self, run_id: UUID) -> list[SimProgressOut]:
        """Read progress from local NDJSON/zst file, falling back to DB for legacy data."""
        from simd_agent.progress_store import read_progress

        local = await read_progress(run_id)
        if local is not None:
            return [SimProgressOut(**row) for row in local]

        # Fallback: legacy runs stored progress in Postgres
        rows = await self.progress_repo.list_for_run(run_id)
        return [SimProgressOut(**row) for row in rows]

    async def insert_progress(self, run_id: UUID, body: SimProgressBatch) -> int:
        for entry in body.entries:
            data = entry.model_dump()
            data["run_id"] = run_id
            await self.progress_repo.create(data)
        return len(body.entries)

    async def delete_progress(self, run_id: UUID) -> None:
        from simd_agent.progress_store import delete_progress_files
        await delete_progress_files(run_id)
        await self.progress_repo.delete_for_run(run_id)
