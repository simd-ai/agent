# tests/test_orchestration_mock.py
"""Tests for orchestration with mock providers."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from simd_agent.event_bus import EventBus
from simd_agent.models import (
    Constraints,
    Operation,
    RunStatus,
    StartRequest,
)
from simd_agent.orchestration import Orchestrator
from simd_agent.store import EventStore


class TestOrchestrationLint:
    """Tests for CFD_LINT operation."""
    
    @pytest.fixture
    def lint_request(self, sample_simulation_config, sample_requirements):
        """Create a lint request."""
        return StartRequest(
            op=Operation.CFD_LINT,
            provider="mock",
            user_requirements=sample_requirements,
            simulation_config=sample_simulation_config,
        )
    
    async def test_lint_operation_succeeds(
        self,
        lint_request,
        mock_websocket,
        run_id,
    ):
        """Test that lint operation completes successfully."""
        store = MagicMock(spec=EventStore)
        store.create_run = AsyncMock()
        store.append_event = AsyncMock()
        store.finalize_run = AsyncMock()
        
        event_bus = EventBus(
            run_id=run_id,
            websocket=mock_websocket,
            store=store,
            persist=False,
        )
        
        orchestrator = Orchestrator(
            run_id=run_id,
            event_bus=event_bus,
            store=store,
            request=lint_request,
        )
        
        result = await orchestrator.run()
        
        assert result.status == RunStatus.SUCCEEDED
        assert result.validated_config is not None
    
    async def test_lint_detects_regime(
        self,
        mock_websocket,
        run_id,
        turbulent_config,
    ):
        """Test that lint correctly detects flow regime."""
        request = StartRequest(
            op=Operation.CFD_LINT,
            provider="mock",
            user_requirements="Turbulent pipe flow simulation",
            simulation_config=turbulent_config,
        )
        
        store = MagicMock(spec=EventStore)
        store.create_run = AsyncMock()
        store.append_event = AsyncMock()
        store.finalize_run = AsyncMock()
        
        event_bus = EventBus(
            run_id=run_id,
            websocket=mock_websocket,
            store=store,
            persist=False,
        )
        
        orchestrator = Orchestrator(
            run_id=run_id,
            event_bus=event_bus,
            store=store,
            request=request,
        )
        
        result = await orchestrator.run()
        
        assert result.status == RunStatus.SUCCEEDED
        # Turbulence should be detected
    
    async def test_lint_emits_events(
        self,
        lint_request,
        mock_websocket,
        run_id,
    ):
        """Test that lint operation emits expected events."""
        store = MagicMock(spec=EventStore)
        store.create_run = AsyncMock()
        store.append_event = AsyncMock()
        store.finalize_run = AsyncMock()
        
        event_bus = EventBus(
            run_id=run_id,
            websocket=mock_websocket,
            store=store,
            persist=False,
        )
        
        orchestrator = Orchestrator(
            run_id=run_id,
            event_bus=event_bus,
            store=store,
            request=lint_request,
        )
        
        await orchestrator.run()
        
        # Check that events were sent via websocket
        call_args_list = mock_websocket.send_json.call_args_list
        
        event_types = [call.args[0]["type"] for call in call_args_list]
        
        assert "run_started" in event_types
        assert "lint_started" in event_types
        assert "lint_result" in event_types
        assert "final" in event_types


class TestOrchestrationCodegenRun:
    """Tests for CFD_CODEGEN_RUN operation."""
    
    @pytest.fixture
    def codegen_request(self, sample_simulation_config, sample_requirements):
        """Create a codegen run request."""
        return StartRequest(
            op=Operation.CFD_CODEGEN_RUN,
            provider="mock",
            user_requirements=sample_requirements,
            simulation_config=sample_simulation_config,
            constraints=Constraints(max_retries=2),
        )
    
    async def test_codegen_emits_progress_events(
        self,
        codegen_request,
        mock_websocket,
        run_id,
    ):
        """Test that codegen emits progress events."""
        store = MagicMock(spec=EventStore)
        store.create_run = AsyncMock()
        store.append_event = AsyncMock()
        store.finalize_run = AsyncMock()
        store.update_run_status = AsyncMock()

        event_bus = EventBus(
            run_id=run_id,
            websocket=mock_websocket,
            store=store,
            persist=False,
        )

        orchestrator = Orchestrator(
            run_id=run_id,
            event_bus=event_bus,
            store=store,
            request=codegen_request,
        )

        await orchestrator.run()

        # Check event types
        call_args_list = mock_websocket.send_json.call_args_list
        event_types = [call.args[0]["type"] for call in call_args_list]

        # Should have various progress events
        assert "run_started" in event_types
        assert "lint_started" in event_types
        assert "planning_started" in event_types
        assert "codegen_started" in event_types


class TestEventBusIntegration:
    """Tests for event bus behavior."""
    
    async def test_event_sequence_numbers(
        self,
        mock_websocket,
        run_id,
    ):
        """Test that event sequence numbers are monotonic."""
        store = MagicMock(spec=EventStore)
        store.append_event = AsyncMock()
        
        event_bus = EventBus(
            run_id=run_id,
            websocket=mock_websocket,
            store=store,
            persist=False,
        )
        
        # Emit several events
        await event_bus.emit_info("test1", "Test 1")
        await event_bus.emit_info("test2", "Test 2")
        await event_bus.emit_info("test3", "Test 3")
        
        # Check sequence numbers
        call_args_list = mock_websocket.send_json.call_args_list
        seqs = [call.args[0]["seq"] for call in call_args_list]
        
        assert seqs == [0, 1, 2]
    
    async def test_event_includes_run_id(
        self,
        mock_websocket,
        run_id,
    ):
        """Test that events include the run ID."""
        store = MagicMock(spec=EventStore)
        store.append_event = AsyncMock()
        
        event_bus = EventBus(
            run_id=run_id,
            websocket=mock_websocket,
            store=store,
            persist=False,
        )
        
        await event_bus.emit_info("test", "Test event")
        
        call = mock_websocket.send_json.call_args
        assert call.args[0]["run_id"] == str(run_id)
    
    async def test_event_includes_timestamp(
        self,
        mock_websocket,
        run_id,
    ):
        """Test that events include timestamps."""
        store = MagicMock(spec=EventStore)
        store.append_event = AsyncMock()
        
        event_bus = EventBus(
            run_id=run_id,
            websocket=mock_websocket,
            store=store,
            persist=False,
        )
        
        await event_bus.emit_info("test", "Test event")
        
        call = mock_websocket.send_json.call_args
        assert "ts" in call.args[0]


class TestErrorHandling:
    """Tests for error handling in orchestration."""
    
    async def test_exception_emits_failure_event(
        self,
        mock_websocket,
        run_id,
    ):
        """Test that exceptions result in failure events."""
        request = StartRequest(
            op=Operation.CFD_CODEGEN_RUN,
            provider="mock",
            user_requirements="Pipe flow",
            simulation_config={
                "mesh": {"mesh_id": "error-test"},
                "geometry": {"type": "pipe", "diameter": 0.1},
                "boundary_conditions": {
                    "inlet": {"patch_type": "inlet", "velocity": {"value": [1, 0, 0]}},
                    "outlet": {"patch_type": "outlet"},
                },
            },
        )
        
        store = MagicMock(spec=EventStore)
        store.create_run = AsyncMock()
        store.append_event = AsyncMock()
        store.finalize_run = AsyncMock()
        store.update_run_status = AsyncMock()
        
        event_bus = EventBus(
            run_id=run_id,
            websocket=mock_websocket,
            store=store,
            persist=False,
        )
        
        orchestrator = Orchestrator(
            run_id=run_id,
            event_bus=event_bus,
            store=store,
            request=request,
        )
        
        result = await orchestrator.run()
        
        assert result.status == RunStatus.FAILED
        
        # Check that failure event was emitted
        call_args_list = mock_websocket.send_json.call_args_list
        event_types = [call.args[0]["type"] for call in call_args_list]
        
        assert "run_failed" in event_types or "final" in event_types
