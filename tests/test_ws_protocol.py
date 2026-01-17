# tests/test_ws_protocol.py
"""Tests for WebSocket protocol and message schemas."""

import pytest
from datetime import datetime
from uuid import uuid4

from simd_agent.models import (
    AgentEvent,
    ApplyChange,
    Constraints,
    EventLevel,
    EventTypes,
    FinalResult,
    FlowRegime,
    LintIssue,
    LintResult,
    Metadata,
    Operation,
    RunStatus,
    StartRequest,
    WorkItem,
)


class TestStartRequest:
    """Tests for StartRequest model."""
    
    def test_minimal_request(self):
        """Test creating a request with minimal fields."""
        request = StartRequest(
            op=Operation.CFD_LINT,
            user_requirements="Simulate a pipe flow",
        )
        
        assert request.op == Operation.CFD_LINT
        assert request.user_requirements == "Simulate a pipe flow"
        assert request.provider == "gemini3"  # default
        assert request.prompt_pack == "simd"  # default
        assert request.simulation_config == {}
        assert isinstance(request.constraints, Constraints)
        assert isinstance(request.metadata, Metadata)
    
    def test_full_request(self):
        """Test creating a request with all fields."""
        request = StartRequest(
            op=Operation.CFD_CODEGEN_RUN,
            provider="grok",
            prompt_pack="custom",
            user_requirements="Simulate turbulent pipe flow",
            simulation_config={"diameter": 0.1, "velocity": 10.0},
            constraints=Constraints(max_retries=5, solver_preference="simpleFoam"),
            metadata=Metadata(user_id="user-123", project_id="proj-456"),
        )
        
        assert request.op == Operation.CFD_CODEGEN_RUN
        assert request.provider == "grok"
        assert request.constraints.max_retries == 5
        assert request.metadata.user_id == "user-123"
    
    def test_empty_requirements_fails(self):
        """Test that empty requirements fail validation."""
        with pytest.raises(ValueError):
            StartRequest(
                op=Operation.CFD_LINT,
                user_requirements="",
            )


class TestAgentEvent:
    """Tests for AgentEvent model."""
    
    def test_event_creation(self):
        """Test creating an event."""
        run_id = uuid4()
        event = AgentEvent(
            run_id=run_id,
            seq=0,
            type=EventTypes.RUN_STARTED,
            message="Run started",
        )
        
        assert event.run_id == run_id
        assert event.seq == 0
        assert event.type == EventTypes.RUN_STARTED
        assert event.level == EventLevel.INFO  # default
        assert isinstance(event.ts, datetime)
        assert event.payload == {}
    
    def test_event_with_payload(self):
        """Test creating an event with payload."""
        event = AgentEvent(
            run_id=uuid4(),
            seq=5,
            type=EventTypes.LINT_RESULT,
            message="Linting complete",
            level=EventLevel.WARN,
            payload={"issues_count": 3, "changes_count": 2},
        )
        
        assert event.level == EventLevel.WARN
        assert event.payload["issues_count"] == 3
    
    def test_event_to_ws_message(self):
        """Test serialization for WebSocket."""
        run_id = uuid4()
        event = AgentEvent(
            run_id=run_id,
            seq=1,
            type=EventTypes.SANDBOX_SUBMITTED,
            message="Submitted to sandbox",
            payload={"sandbox_run_id": "sb-123"},
        )
        
        msg = event.to_ws_message()
        
        assert msg["run_id"] == str(run_id)
        assert msg["seq"] == 1
        assert msg["type"] == EventTypes.SANDBOX_SUBMITTED
        assert msg["message"] == "Submitted to sandbox"
        assert msg["payload"]["sandbox_run_id"] == "sb-123"
        assert "ts" in msg


class TestEventOrdering:
    """Tests for event sequence ordering."""
    
    def test_sequence_numbers(self):
        """Test that sequence numbers are properly ordered."""
        run_id = uuid4()
        events = [
            AgentEvent(run_id=run_id, seq=i, type=f"event_{i}", message=f"Event {i}")
            for i in range(10)
        ]
        
        for i, event in enumerate(events):
            assert event.seq == i
    
    def test_sequence_uniqueness(self):
        """Test that same sequence in different runs is allowed."""
        run1 = uuid4()
        run2 = uuid4()
        
        event1 = AgentEvent(run_id=run1, seq=0, type="test", message="Test 1")
        event2 = AgentEvent(run_id=run2, seq=0, type="test", message="Test 2")
        
        # Both should be valid with seq=0 in different runs
        assert event1.seq == event2.seq == 0
        assert event1.run_id != event2.run_id


class TestLintModels:
    """Tests for linting-related models."""
    
    def test_lint_result(self):
        """Test LintResult model."""
        result = LintResult(
            validated_config={"solver": "simpleFoam"},
            apply_changes=[
                ApplyChange(
                    path="turbulence_model",
                    value="laminar",
                    reason="Laminar flow detected",
                )
            ],
            issues=[
                LintIssue(
                    code="MISSING_INLET",
                    message="No inlet defined",
                    severity="warning",
                )
            ],
            detected_regime=FlowRegime.LAMINAR,
            reynolds_number=1500.0,
        )
        
        assert result.detected_regime == FlowRegime.LAMINAR
        assert result.reynolds_number == 1500.0
        assert len(result.apply_changes) == 1
        assert len(result.issues) == 1
    
    def test_apply_change_severities(self):
        """Test ApplyChange severity levels."""
        for severity in ["info", "warning", "error"]:
            change = ApplyChange(
                path="test",
                value="test",
                reason="test",
                severity=severity,
            )
            assert change.severity == severity


class TestFinalResult:
    """Tests for FinalResult model."""
    
    def test_successful_result(self):
        """Test successful final result."""
        result = FinalResult(
            status=RunStatus.SUCCEEDED,
            validated_config={"solver": "simpleFoam"},
            iterations=2,
            retries=1,
            summary="Case executed successfully",
            case_type="pipe_flow",
            solver="simpleFoam",
        )
        
        assert result.status == RunStatus.SUCCEEDED
        assert result.iterations == 2
        assert result.error is None
    
    def test_failed_result(self):
        """Test failed final result."""
        result = FinalResult(
            status=RunStatus.FAILED,
            iterations=3,
            retries=3,
            error="Max retries exceeded",
        )
        
        assert result.status == RunStatus.FAILED
        assert result.error == "Max retries exceeded"
    
    def test_not_clear_result(self):
        """Test simulation not clear result."""
        result = FinalResult(
            status=RunStatus.NOT_CLEAR,
            summary="Could not determine simulation type",
        )
        
        assert result.status == RunStatus.NOT_CLEAR


class TestConstraints:
    """Tests for Constraints model."""
    
    def test_default_constraints(self):
        """Test default constraint values."""
        constraints = Constraints()
        
        assert constraints.max_retries == 3
        assert constraints.solver_preference is None
        assert constraints.timeout_seconds == 300
    
    def test_custom_constraints(self):
        """Test custom constraint values."""
        constraints = Constraints(
            max_retries=5,
            solver_preference="pimpleFoam",
            timeout_seconds=600,
        )
        
        assert constraints.max_retries == 5
        assert constraints.solver_preference == "pimpleFoam"
    
    def test_constraint_bounds(self):
        """Test constraint value bounds."""
        # Max retries must be 1-10
        with pytest.raises(ValueError):
            Constraints(max_retries=0)
        
        with pytest.raises(ValueError):
            Constraints(max_retries=11)
        
        # Timeout must be 30-3600
        with pytest.raises(ValueError):
            Constraints(timeout_seconds=10)


class TestWorkItem:
    """Tests for WorkItem model."""
    
    def test_work_item_creation(self):
        """Test creating a work item."""
        item = WorkItem(
            id="choose_solver",
            task="choose_solver",
            description="Select appropriate solver",
        )
        
        assert item.id == "choose_solver"
        assert item.priority == 1  # default
        assert item.dependencies == []
    
    def test_work_item_with_dependencies(self):
        """Test work item with dependencies."""
        item = WorkItem(
            id="choose_turbulence",
            task="choose_turbulence",
            description="Select turbulence model",
            priority=2,
            dependencies=["choose_solver"],
        )
        
        assert item.dependencies == ["choose_solver"]
