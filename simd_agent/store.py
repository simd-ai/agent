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
    ) -> UUID:
        """Create a new run record.
        
        Args:
            op: The operation type
            provider: LLM provider name
            prompt_pack: Prompt pack name
            user_requirements: User's requirements text
            simulation_config: Initial simulation configuration
            run_id: Optional pre-generated run ID
            
        Returns:
            The run ID
        """
        run_id = run_id or uuid4()
        
        async with get_session() as session:
            await session.execute(
                text("""
                    INSERT INTO runs (id, op, provider, prompt_pack, user_requirements, simulation_config, status)
                    VALUES (:id, :op, :provider, :prompt_pack, :user_requirements, :simulation_config, :status)
                """),
                {
                    "id": run_id,
                    "op": op.value,
                    "provider": provider,
                    "prompt_pack": prompt_pack,
                    "user_requirements": user_requirements,
                    "simulation_config": json.dumps(simulation_config),
                    "status": RunStatus.PENDING.value,
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
    
    async def finalize_run(
        self,
        run_id: UUID,
        status: RunStatus,
        validated_config: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        attempts: int = 0,
    ) -> None:
        """Finalize a run with final status and results.
        
        Args:
            run_id: The run ID
            status: Final status
            validated_config: Validated configuration (optional)
            result: Final result data (optional)
            attempts: Total number of attempts
        """
        async with get_session() as session:
            await session.execute(
                text("""
                    UPDATE runs 
                    SET status = :status, 
                        validated_config = :validated_config,
                        result = :result,
                        attempts = :attempts
                    WHERE id = :id
                """),
                {
                    "id": run_id,
                    "status": status.value,
                    "validated_config": json.dumps(validated_config) if validated_config else None,
                    "result": json.dumps(result) if result else None,
                    "attempts": attempts,
                },
            )
    
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
                created_at=row["created_at"],
                op=Operation(row["op"]),
                status=RunStatus(row["status"]),
                provider=row["provider"],
                prompt_pack=row["prompt_pack"],
                user_requirements=row["user_requirements"],
                simulation_config=row["simulation_config"] or {},
                validated_config=row["validated_config"],
                attempts=row["attempts"],
                result=row["result"],
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
