# simd_agent/event_bus.py
"""Event bus for emitting and streaming events via WebSocket."""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any, Callable, Coroutine
from uuid import UUID

from fastapi import WebSocket

from simd_agent.models import AgentEvent, EventLevel, EventTypes
from simd_agent.store import EventStore

logger = logging.getLogger(__name__)


class EventBus:
    """Manages event emission, persistence, and WebSocket streaming."""
    
    def __init__(
        self,
        run_id: UUID,
        websocket: WebSocket,
        store: EventStore | None = None,
        persist: bool = True,
    ):
        """Initialize the event bus.
        
        Args:
            run_id: The run ID for all events
            websocket: WebSocket connection for streaming
            store: Event store for persistence (optional)
            persist: Whether to persist events to database
        """
        self.run_id = run_id
        self.websocket = websocket
        self.store = store or EventStore()
        self.persist = persist
        self._seq = 0
        self._lock = asyncio.Lock()
        self._started_at = datetime.utcnow()
    
    async def emit(
        self,
        event_type: str,
        message: str,
        payload: dict[str, Any] | None = None,
        level: EventLevel = EventLevel.INFO,
    ) -> AgentEvent:
        """Emit an event: persist to DB and send via WebSocket.
        
        Args:
            event_type: Type of event (from EventTypes)
            message: Human-readable message
            payload: Additional event data
            level: Event severity level
            
        Returns:
            The created AgentEvent
        """
        async with self._lock:
            seq = self._seq
            self._seq += 1
        
        event = AgentEvent(
            run_id=self.run_id,
            seq=seq,
            ts=datetime.utcnow(),
            level=level,
            type=event_type,
            message=message,
            payload=payload or {},
        )
        
        # Persist to database
        if self.persist:
            try:
                await self.store.append_event(event)
            except Exception as e:
                logger.error(f"Failed to persist event: {e}")
        
        # Send via WebSocket
        try:
            await self.websocket.send_json(event.to_ws_message())
        except Exception as e:
            logger.error(f"Failed to send event via WebSocket: {e}")
        
        return event
    
    async def emit_debug(self, message: str, payload: dict[str, Any] | None = None) -> AgentEvent:
        """Emit a debug-level event."""
        return await self.emit(
            event_type="debug",
            message=message,
            payload=payload,
            level=EventLevel.DEBUG,
        )
    
    async def emit_info(
        self,
        event_type: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> AgentEvent:
        """Emit an info-level event."""
        return await self.emit(
            event_type=event_type,
            message=message,
            payload=payload,
            level=EventLevel.INFO,
        )
    
    async def emit_warn(
        self,
        event_type: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> AgentEvent:
        """Emit a warning-level event."""
        return await self.emit(
            event_type=event_type,
            message=message,
            payload=payload,
            level=EventLevel.WARN,
        )
    
    async def emit_error(
        self,
        event_type: str,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> AgentEvent:
        """Emit an error-level event."""
        return await self.emit(
            event_type=event_type,
            message=message,
            payload=payload,
            level=EventLevel.ERROR,
        )
    
    async def emit_run_started(self, op: str, provider: str) -> AgentEvent:
        """Emit run started event."""
        return await self.emit_info(
            EventTypes.RUN_STARTED,
            f"Run started: {op}",
            {"op": op, "provider": provider},
        )
    
    async def emit_lint_started(self) -> AgentEvent:
        """Emit lint started event."""
        return await self.emit_info(
            EventTypes.LINT_STARTED,
            "CFD linting started",
        )
    
    async def emit_lint_result(
        self,
        validated_config: dict[str, Any],
        apply_changes: list[dict[str, Any]],
        issues: list[dict[str, Any]],
        regime: str | None = None,
        solver: str | None = None,
        reynolds: float | None = None,
    ) -> AgentEvent:
        """Emit lint result event."""
        return await self.emit_info(
            EventTypes.LINT_RESULT,
            f"Linting complete: {len(issues)} issues, {len(apply_changes)} recommendations",
            {
                "validated_config": validated_config,
                "apply_changes": apply_changes,
                "issues": issues,
                "regime": regime,
                "solver": solver,
                "reynolds_number": reynolds,
            },
        )
    
    async def emit_planning_started(self) -> AgentEvent:
        """Emit planning started event."""
        return await self.emit_info(
            EventTypes.PLANNING_STARTED,
            "Planning phase started",
        )
    
    async def emit_planning_complete(
        self,
        work_items: list[dict[str, Any]],
        case_type: str,
        solver: str,
    ) -> AgentEvent:
        """Emit planning complete event."""
        return await self.emit_info(
            EventTypes.PLANNING_COMPLETE,
            f"Planning complete: {len(work_items)} work items, case={case_type}, solver={solver}",
            {
                "work_items": work_items,
                "case_type": case_type,
                "solver": solver,
            },
        )
    
    async def emit_subagent_started(self, work_item_id: str, task: str) -> AgentEvent:
        """Emit sub-agent started event."""
        return await self.emit_info(
            EventTypes.SUBAGENT_STARTED,
            f"Sub-agent started: {task}",
            {"work_item_id": work_item_id, "task": task},
        )
    
    async def emit_subagent_update(
        self,
        work_item_id: str,
        task: str,
        update: str,
    ) -> AgentEvent:
        """Emit sub-agent progress update."""
        return await self.emit_info(
            EventTypes.SUBAGENT_UPDATE,
            f"[{task}] {update}",
            {"work_item_id": work_item_id, "task": task, "update": update},
        )
    
    async def emit_subagent_complete(
        self,
        work_item_id: str,
        task: str,
        result: dict[str, Any],
        duration_ms: int,
    ) -> AgentEvent:
        """Emit sub-agent complete event."""
        return await self.emit_info(
            EventTypes.SUBAGENT_COMPLETE,
            f"Sub-agent complete: {task} ({duration_ms}ms)",
            {
                "work_item_id": work_item_id,
                "task": task,
                "result": result,
                "duration_ms": duration_ms,
            },
        )
    
    async def emit_codegen_started(self, iteration: int) -> AgentEvent:
        """Emit codegen started event."""
        return await self.emit_info(
            EventTypes.CODEGEN_STARTED,
            f"Code generation started (iteration {iteration})",
            {"iteration": iteration},
        )
    
    async def emit_codegen_iteration(
        self,
        iteration: int,
        files_generated: list[str],
    ) -> AgentEvent:
        """Emit codegen iteration event."""
        return await self.emit_info(
            EventTypes.CODEGEN_ITERATION,
            f"Generated {len(files_generated)} files",
            {"iteration": iteration, "files": files_generated},
        )
    
    async def emit_codegen_complete(
        self,
        iteration: int,
        case_zip_size: int | None = None,
    ) -> AgentEvent:
        """Emit codegen complete event."""
        return await self.emit_info(
            EventTypes.CODEGEN_COMPLETE,
            f"Code generation complete (iteration {iteration})",
            {"iteration": iteration, "case_zip_size": case_zip_size},
        )
    
    async def emit_sandbox_submitted(self, sandbox_run_id: str) -> AgentEvent:
        """Emit sandbox submitted event."""
        return await self.emit_info(
            EventTypes.SANDBOX_SUBMITTED,
            f"Submitted to sandbox: {sandbox_run_id}",
            {"sandbox_run_id": sandbox_run_id},
        )
    
    async def emit_sandbox_status(self, state: str, sandbox_run_id: str) -> AgentEvent:
        """Emit sandbox status update."""
        return await self.emit_info(
            EventTypes.SANDBOX_STATUS,
            f"Sandbox status: {state}",
            {"state": state, "sandbox_run_id": sandbox_run_id},
        )
    
    async def emit_sandbox_logs(
        self,
        sandbox_run_id: str,
        logs: str,
        truncated: bool = False,
    ) -> AgentEvent:
        """Emit sandbox logs event."""
        return await self.emit_info(
            EventTypes.SANDBOX_LOGS,
            "Sandbox execution logs",
            {
                "sandbox_run_id": sandbox_run_id,
                "logs": logs,
                "truncated": truncated,
            },
        )
    
    async def emit_sandbox_succeeded(
        self,
        sandbox_run_id: str,
        artifacts: list[dict[str, Any]],
    ) -> AgentEvent:
        """Emit sandbox succeeded event."""
        return await self.emit_info(
            EventTypes.SANDBOX_SUCCEEDED,
            f"Sandbox run succeeded with {len(artifacts)} artifacts",
            {"sandbox_run_id": sandbox_run_id, "artifacts": artifacts},
        )
    
    async def emit_sandbox_failed(
        self,
        sandbox_run_id: str,
        exit_code: int | None,
        logs_tail: str,
    ) -> AgentEvent:
        """Emit sandbox failed event."""
        return await self.emit_error(
            EventTypes.SANDBOX_FAILED,
            f"Sandbox run failed (exit_code={exit_code})",
            {
                "sandbox_run_id": sandbox_run_id,
                "exit_code": exit_code,
                "logs_tail": logs_tail,
            },
        )
    
    async def emit_error_summary(
        self,
        root_cause: str,
        actionable_changes: list[dict[str, Any]],
        affected_files: list[str],
    ) -> AgentEvent:
        """Emit error summary event."""
        return await self.emit_info(
            EventTypes.ERROR_SUMMARY,
            f"Error analysis: {root_cause}",
            {
                "root_cause": root_cause,
                "actionable_changes": actionable_changes,
                "affected_files": affected_files,
            },
        )
    
    async def emit_retrying(self, attempt: int, max_retries: int) -> AgentEvent:
        """Emit retrying event."""
        return await self.emit_warn(
            EventTypes.RETRYING,
            f"Retrying ({attempt}/{max_retries})",
            {"attempt": attempt, "max_retries": max_retries},
        )
    
    async def emit_simulation_not_clear(self, reason: str) -> AgentEvent:
        """Emit simulation not clear event."""
        return await self.emit_warn(
            EventTypes.SIMULATION_NOT_CLEAR,
            f"Simulation type unclear: {reason}",
            {"reason": reason},
        )
    
    async def emit_run_succeeded(
        self,
        summary: str,
        artifacts: list[dict[str, Any]] | None = None,
    ) -> AgentEvent:
        """Emit run succeeded event."""
        duration = (datetime.utcnow() - self._started_at).total_seconds()
        return await self.emit_info(
            EventTypes.RUN_SUCCEEDED,
            f"Run succeeded: {summary}",
            {"summary": summary, "artifacts": artifacts or [], "duration_seconds": duration},
        )
    
    async def emit_run_failed(self, error: str) -> AgentEvent:
        """Emit run failed event."""
        duration = (datetime.utcnow() - self._started_at).total_seconds()
        return await self.emit_error(
            EventTypes.RUN_FAILED,
            f"Run failed: {error}",
            {"error": error, "duration_seconds": duration},
        )
    
    async def emit_final(
        self,
        status: str,
        validated_config: dict[str, Any] | None = None,
        artifacts: list[dict[str, Any]] | None = None,
        iterations: int = 0,
        retries: int = 0,
        summary: str = "",
        case_type: str | None = None,
        solver: str | None = None,
        error: str | None = None,
    ) -> AgentEvent:
        """Emit final event with complete results."""
        duration = (datetime.utcnow() - self._started_at).total_seconds()
        return await self.emit(
            EventTypes.FINAL,
            f"Final: {status}",
            {
                "status": status,
                "validated_config": validated_config,
                "artifacts": artifacts or [],
                "iterations": iterations,
                "retries": retries,
                "summary": summary,
                "case_type": case_type,
                "solver": solver,
                "error": error,
                "duration_seconds": duration,
            },
            level=EventLevel.INFO if status == "succeeded" else EventLevel.ERROR,
        )
    
    @property
    def sequence(self) -> int:
        """Get current sequence number."""
        return self._seq
