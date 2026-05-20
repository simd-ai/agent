# simd_agent/store.py
"""Event store for persisting runs and events to the database."""

import json
from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from simd_agent.db import get_session
from simd_agent.models import (
    AgentEvent,
    EventLevel,
    EventRow,
    Operation,
    RunRow,
    RunStatus,
)


class EventStore:
    """Handles persistence of runs and events to Postgres."""
    
    def __init__(self, session: AsyncSession | None = None):
        """Initialize the event store.
        
        Args:
            session: Optional async session. If not provided, will create new sessions.
        """
        self._session = session
    
    async def create_run(
        self,
        op: Operation,
        provider: str,
        prompt_pack: str,
        user_requirements: str,
        simulation_config: dict[str, Any],
        run_id: UUID | None = None,
        raw_config: dict[str, Any] | None = None,
        simulation_id: UUID | str | None = None,
    ) -> UUID:
        """Create a new run record.
        
        Args:
            op: The operation type
            provider: LLM provider name
            prompt_pack: Prompt pack name
            user_requirements: User's requirements text
            simulation_config: Initial simulation configuration (may be normalized)
            run_id: Optional pre-generated run ID
            raw_config: Optional raw config as received from frontend (before normalization)
            
        Returns:
            The run ID
        """
        run_id = run_id or uuid4()
        
        # Store raw_config in result field temporarily if provided
        # This preserves the original payload for debugging
        result_data = None
        if raw_config:
            result_data = {"raw_config": raw_config}
        
        # Coerce simulation_id to UUID if provided as string
        sim_id: UUID | None = None
        if simulation_id is not None:
            sim_id = UUID(str(simulation_id)) if not isinstance(simulation_id, UUID) else simulation_id

        async with get_session() as session:
            await session.execute(
                text("""
                    INSERT INTO runs (id, op, provider, prompt_pack, user_requirements, simulation_config, status, result, simulation_id)
                    VALUES (:id, :op, :provider, :prompt_pack, :user_requirements, :simulation_config, :status, :result, :simulation_id)
                """),
                {
                    "id": run_id,
                    "op": op.value,
                    "provider": provider,
                    "prompt_pack": prompt_pack,
                    "user_requirements": user_requirements,
                    "simulation_config": json.dumps(simulation_config),
                    "status": RunStatus.PENDING.value,
                    "result": json.dumps(result_data) if result_data else None,
                    "simulation_id": sim_id,
                },
            )
        
        return run_id
    
    async def append_event(self, event: AgentEvent) -> UUID:
        """Append an event to the database.
        
        Args:
            event: The event to persist
            
        Returns:
            The event ID
        """
        event_id = uuid4()
        
        async with get_session() as session:
            await session.execute(
                text("""
                    INSERT INTO events (id, run_id, seq, ts, level, type, message, payload)
                    VALUES (:id, :run_id, :seq, :ts, :level, :type, :message, :payload)
                """),
                {
                    "id": event_id,
                    "run_id": event.run_id,
                    "seq": event.seq,
                    "ts": event.ts,
                    "level": event.level.value,
                    "type": event.type,
                    "message": event.message,
                    "payload": json.dumps(event.payload),
                },
            )
        
        return event_id
    
    async def update_generated_files(
        self,
        run_id: UUID,
        generated_files: dict[str, str],
    ) -> None:
        """Incrementally save generated_files for a run.

        Called after each codegen update or surgical fix so the DB always
        has the latest file contents, even if the run crashes mid-retry.
        """
        file_keys = list(generated_files.keys())
        total_chars = sum(len(v) for v in generated_files.values())
        print(f"[DB-SAVE] update_generated_files run={run_id} files={len(file_keys)} total_chars={total_chars} keys={file_keys}")
        async with get_session() as session:
            result = await session.execute(
                text("""
                    UPDATE runs SET generated_files = :generated_files
                    WHERE id = :id
                    RETURNING id
                """),
                {
                    "id": run_id,
                    "generated_files": json.dumps(generated_files),
                },
            )
            row = result.scalar()
            if row:
                print(f"[DB-SAVE] SUCCESS — updated run {run_id}")
            else:
                print(f"[DB-SAVE] WARNING — no row matched run_id={run_id}, nothing was updated!")

    async def update_run_status(
        self,
        run_id: UUID,
        status: RunStatus,
        attempts: int | None = None,
    ) -> None:
        """Update the status of a run.
        
        Args:
            run_id: The run ID
            status: New status
            attempts: Optional attempt count to update
        """
        async with get_session() as session:
            if attempts is not None:
                await session.execute(
                    text("""
                        UPDATE runs SET status = :status, attempts = :attempts
                        WHERE id = :id
                    """),
                    {"id": run_id, "status": status.value, "attempts": attempts},
                )
            else:
                await session.execute(
                    text("""
                        UPDATE runs SET status = :status WHERE id = :id
                    """),
                    {"id": run_id, "status": status.value},
                )
    
    async def set_sim_run_id(self, run_id: UUID, sim_run_id: str) -> None:
        """Persist the sim-server run id into ``runs.result`` mid-run.

        Without this, a page reload mid-simulation leaves a worker that did
        not own the orchestrator unable to drive the sim runner (the only
        place sim_run_id is otherwise available is the orchestrator's
        in-process state).  Stop / continue endpoints fall back to this
        column when ``_active_orchestrators`` doesn't have the run.

        The update is a JSONB merge — pre-existing keys in ``result`` are
        preserved so this does not clobber anything the orchestrator has
        already written.
        """
        from simd_agent.db import is_sqlite

        if is_sqlite():
            sql = """
                UPDATE runs
                SET result = json_patch(
                    COALESCE(result, '{}'),
                    json_object('sim_run_id', :sim_run_id)
                )
                WHERE id = :id
            """
        else:
            sql = """
                UPDATE runs
                SET result = COALESCE(result, '{}'::jsonb)
                             || jsonb_build_object('sim_run_id', :sim_run_id)
                WHERE id = :id
            """
        async with get_session() as session:
            await session.execute(
                text(sql),
                {"id": run_id, "sim_run_id": sim_run_id},
            )

    async def set_convergence(
        self, run_id: UUID, convergence: dict[str, Any],
    ) -> None:
        """Persist the latest convergence assessment into ``runs.result`` mid-run.

        Convergence is recomputed every ~25 sim_progress steps by the
        orchestrator and emitted via WebSocket as ``convergence_update``
        events.  Without this DB write, a page refresh mid-run would
        leave the frontend with empty OoM badges — the WS event was
        delivered but lost on reload, and the DB had no copy yet because
        ``finalize_run`` only runs at end-of-run.

        JSONB merge into the existing ``result`` field, so any other
        keys (``sim_run_id``, …) the orchestrator has already written
        are preserved.  Called on every assessment refresh; the
        frontend's ``restore-run.ts`` reads back from ``result.convergence``.
        """
        import json as _json
        from simd_agent.db import is_sqlite

        if is_sqlite():
            # On SQLite ``:conv`` is plain text; json() parses it to JSON.
            sql = """
                UPDATE runs
                SET result = json_patch(
                    COALESCE(result, '{}'),
                    json_object('convergence', json(:conv))
                )
                WHERE id = :id
            """
        else:
            sql = """
                UPDATE runs
                SET result = COALESCE(result, '{}'::jsonb)
                             || jsonb_build_object('convergence', CAST(:conv AS jsonb))
                WHERE id = :id
            """
        async with get_session() as session:
            await session.execute(
                text(sql),
                {"id": run_id, "conv": _json.dumps(convergence)},
            )

    async def finalize_run(
        self,
        run_id: UUID,
        status: RunStatus,
        validated_config: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        attempts: int = 0,
        normalized_config: dict[str, Any] | None = None,
        generated_files: dict[str, str] | None = None,
    ) -> None:
        """Finalize a run with final status and results.

        Args:
            run_id: The run ID
            status: Final status
            validated_config: Validated configuration (optional)
            result: Final result data (optional)
            attempts: Total number of attempts
            normalized_config: Normalized SimulationConfigV1 as dict (optional)
            generated_files: Map of file path → content (optional)
        """
        # Merge normalized_config into result if provided
        final_result = result or {}
        if normalized_config:
            final_result["normalized_config"] = normalized_config

        if generated_files:
            gf_keys = list(generated_files.keys())
            gf_chars = sum(len(v) for v in generated_files.values())
            print(f"[DB-SAVE] finalize_run run={run_id} status={status.value} files={len(gf_keys)} total_chars={gf_chars} keys={gf_keys}")
        else:
            print(f"[DB-SAVE] finalize_run run={run_id} status={status.value} generated_files=None (will clear!)")

        async with get_session() as session:
            await session.execute(
                text("""
                    UPDATE runs
                    SET status = :status,
                        validated_config = :validated_config,
                        result = :result,
                        attempts = :attempts,
                        generated_files = :generated_files
                    WHERE id = :id
                """),
                {
                    "id": run_id,
                    "status": status.value,
                    "validated_config": json.dumps(validated_config) if validated_config else None,
                    "result": json.dumps(final_result) if final_result else None,
                    "attempts": attempts,
                    "generated_files": json.dumps(generated_files) if generated_files else None,
                },
            )
            print(f"[DB-SAVE] finalize_run DONE for run {run_id}")
    
    async def get_run(self, run_id: UUID) -> RunRow | None:
        """Get a run by ID.
        
        Args:
            run_id: The run ID
            
        Returns:
            The run row or None if not found
        """
        async with get_session() as session:
            result = await session.execute(
                text("SELECT * FROM runs WHERE id = :id"),
                {"id": run_id},
            )
            row = result.mappings().first()
            if row is None:
                return None
            
            return RunRow(
                id=row["id"],
                created_at=row.get("created_at") or row.get("started_at"),
                op=Operation(row["op"]) if row.get("op") else Operation.CFD_CODEGEN_RUN,
                status=RunStatus(row["status"]),
                provider=row.get("provider") or "unknown",
                prompt_pack=row.get("prompt_pack") or "simd",
                user_requirements=row.get("user_requirements") or "",
                simulation_config=row.get("simulation_config") or {},
                validated_config=row.get("validated_config"),
                attempts=row.get("attempts") or 0,
                result=row.get("result"),
            )
    
    async def get_events(self, run_id: UUID) -> list[EventRow]:
        """Get all events for a run.

        Args:
            run_id: The run ID

        Returns:
            List of events ordered by sequence
        """
        async with get_session() as session:
            result = await session.execute(
                text("SELECT * FROM events WHERE run_id = :run_id ORDER BY seq"),
                {"run_id": run_id},
            )
            rows = result.mappings().all()

            return [
                EventRow(
                    id=row["id"],
                    run_id=row["run_id"],
                    seq=row["seq"],
                    ts=row["ts"],
                    level=EventLevel(row["level"]),
                    type=row["type"],
                    message=row["message"],
                    payload=row["payload"] or {},
                )
                for row in rows
            ]

    async def get_events_since(self, run_id: UUID, last_seq: int) -> list[EventRow]:
        """Get events for a run with seq > last_seq (for reconnect replay).

        Args:
            run_id: The run ID
            last_seq: Return only events with seq strictly greater than this value

        Returns:
            List of events ordered by sequence
        """
        async with get_session() as session:
            result = await session.execute(
                text(
                    "SELECT * FROM events WHERE run_id = :run_id AND seq > :seq ORDER BY seq"
                ),
                {"run_id": run_id, "seq": last_seq},
            )
            rows = result.mappings().all()

            return [
                EventRow(
                    id=row["id"],
                    run_id=row["run_id"],
                    seq=row["seq"],
                    ts=row["ts"],
                    level=EventLevel(row["level"]),
                    type=row["type"],
                    message=row["message"],
                    payload=row["payload"] or {},
                )
                for row in rows
            ]

    async def get_last_seq(self, run_id: UUID) -> int:
        """Return the highest event seq for a run (0 if no events yet)."""
        async with get_session() as session:
            result = await session.execute(
                text(
                    "SELECT COALESCE(MAX(seq), 0) AS last_seq FROM events WHERE run_id = :run_id"
                ),
                {"run_id": run_id},
            )
            row = result.mappings().first()
            return int(row["last_seq"]) if row else 0
