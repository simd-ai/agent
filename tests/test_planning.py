# tests/test_planning.py
"""Tests for the planning module with parallel sub-agents."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from simd_agent.event_bus import EventBus
from simd_agent.models import FlowRegime, LintResult
from simd_agent.planning import (
    Planner,
    SharedContext,
    SubAgent,
    WORK_ITEMS,
)


class TestSharedContext:
    """Tests for the SharedContext class."""
    
    async def test_get_set(self):
        """Test basic get/set operations."""
        ctx = SharedContext()
        
        await ctx.set("key1", "value1")
        result = await ctx.get("key1")
        
        assert result == "value1"
    
    async def test_get_default(self):
        """Test get with default value."""
        ctx = SharedContext()
        
        result = await ctx.get("nonexistent", "default")
        
        assert result == "default"
    
    async def test_set_no_overwrite(self):
        """Test that set doesn't overwrite by default."""
        ctx = SharedContext({"key": "original"})
        
        result = await ctx.set("key", "new", overwrite=False)
        
        assert result is False
        assert await ctx.get("key") == "original"
    
    async def test_set_with_overwrite(self):
        """Test that set overwrites when requested."""
        ctx = SharedContext({"key": "original"})
        
        result = await ctx.set("key", "new", overwrite=True)
        
        assert result is True
        assert await ctx.get("key") == "new"
    
    async def test_merge(self):
        """Test merging data into context."""
        ctx = SharedContext({"a": 1})
        
        await ctx.merge({"b": 2, "c": 3})
        
        snapshot = await ctx.snapshot()
        assert snapshot == {"a": 1, "b": 2, "c": 3}
    
    async def test_merge_no_overwrite(self):
        """Test that merge doesn't overwrite by default."""
        ctx = SharedContext({"key": "original"})
        
        await ctx.merge({"key": "new"}, overwrite=False)
        
        assert await ctx.get("key") == "original"
    
    async def test_snapshot_is_copy(self):
        """Test that snapshot returns a copy."""
        ctx = SharedContext({"key": "value"})
        
        snapshot = await ctx.snapshot()
        snapshot["key"] = "modified"
        
        # Original should be unchanged
        assert await ctx.get("key") == "value"


class TestSubAgent:
    """Tests for SubAgent execution."""
    
    @pytest.fixture
    def mock_event_bus(self, mock_websocket, run_id):
        """Create a mock event bus."""
        store = MagicMock()
        store.append_event = AsyncMock()
        return EventBus(
            run_id=run_id,
            websocket=mock_websocket,
            store=store,
            persist=False,
        )
    
    async def test_choose_solver_laminar(self, mock_event_bus):
        """Test solver selection for laminar flow."""
        ctx = SharedContext({
            "regime": FlowRegime.LAMINAR,
            "case_type": "pipe_flow",
            "is_thermal": False,
            "is_transient": False,
        })
        
        work_item = WORK_ITEMS["choose_solver"]
        agent = SubAgent(work_item, mock_event_bus, ctx)
        
        result = await agent.run()
        
        assert result.task == "choose_solver"
        assert result.result["solver"] == "simpleFoam"
    
    async def test_choose_solver_thermal(self, mock_event_bus):
        """Test solver selection for thermal flow."""
        ctx = SharedContext({
            "regime": FlowRegime.TURBULENT,
            "case_type": "heat_transfer",
            "is_thermal": True,
            "is_transient": False,
        })
        
        work_item = WORK_ITEMS["choose_solver"]
        agent = SubAgent(work_item, mock_event_bus, ctx)
        
        result = await agent.run()
        
        assert "buoyant" in result.result["solver"].lower()
    
    async def test_choose_turbulence_laminar(self, mock_event_bus):
        """Test turbulence model for laminar flow."""
        ctx = SharedContext({
            "regime": FlowRegime.LAMINAR,
            "reynolds_number": 1500,
        })
        
        work_item = WORK_ITEMS["choose_turbulence"]
        agent = SubAgent(work_item, mock_event_bus, ctx)
        
        result = await agent.run()
        
        assert result.result["turbulence_model"] == "laminar"
    
    async def test_choose_turbulence_turbulent(self, mock_event_bus):
        """Test turbulence model for turbulent flow."""
        ctx = SharedContext({
            "regime": FlowRegime.TURBULENT,
            "reynolds_number": 100000,
        })
        
        work_item = WORK_ITEMS["choose_turbulence"]
        agent = SubAgent(work_item, mock_event_bus, ctx)
        
        result = await agent.run()
        
        assert result.result["turbulence_model"] in ["kEpsilon", "kOmegaSST"]
    
    async def test_design_mesh_high_re(self, mock_event_bus):
        """Test mesh design for high Reynolds number."""
        ctx = SharedContext({
            "reynolds_number": 500000,
            "geometry": {"diameter": 0.1},
        })
        
        work_item = WORK_ITEMS["design_mesh"]
        agent = SubAgent(work_item, mock_event_bus, ctx)
        
        result = await agent.run()
        
        assert result.result["mesh_strategy"] == "boundary_layer_refinement"
        assert result.result["mesh_base_cells"] >= 50
    
    async def test_subagent_emits_events(self, mock_event_bus, mock_websocket):
        """Test that sub-agent emits events."""
        ctx = SharedContext({
            "regime": FlowRegime.LAMINAR,
            "case_type": "pipe_flow",
        })
        
        work_item = WORK_ITEMS["choose_solver"]
        agent = SubAgent(work_item, mock_event_bus, ctx)
        
        await agent.run()
        
        # Check events were emitted
        call_args = mock_websocket.send_json.call_args_list
        event_types = [call.args[0]["type"] for call in call_args]
        
        assert "subagent_started" in event_types
        assert "subagent_complete" in event_types
    
    async def test_subagent_merges_result(self, mock_event_bus):
        """Test that sub-agent merges results into context."""
        ctx = SharedContext({
            "regime": FlowRegime.LAMINAR,
            "case_type": "pipe_flow",
        })
        
        work_item = WORK_ITEMS["choose_solver"]
        agent = SubAgent(work_item, mock_event_bus, ctx)
        
        await agent.run()
        
        # Result should be merged into context
        solver = await ctx.get("solver")
        assert solver == "simpleFoam"


class TestPlanner:
    """Tests for the Planner class."""
    
    @pytest.fixture
    def mock_event_bus(self, mock_websocket, run_id):
        store = MagicMock()
        store.append_event = AsyncMock()
        return EventBus(
            run_id=run_id,
            websocket=mock_websocket,
            store=store,
            persist=False,
        )
    
    @pytest.fixture
    def lint_result(self):
        """Create a sample lint result."""
        return LintResult(
            validated_config={"solver": "simpleFoam"},
            detected_case_type="pipe_flow",
            detected_regime=FlowRegime.TURBULENT,
            reynolds_number=100000,
        )
    
    async def test_plan_returns_result(self, mock_event_bus, lint_result):
        """Test that planning returns a result."""
        planner = Planner(mock_event_bus)
        
        result = await planner.plan(
            lint_result=lint_result,
            user_requirements="Pipe flow simulation",
            config={"geometry": {"diameter": 0.1}},
        )
        
        assert result is not None
        assert result.case_type == "pipe_flow"
        assert result.solver is not None
    
    async def test_plan_selects_work_items(self, mock_event_bus, lint_result):
        """Test that planning selects appropriate work items."""
        planner = Planner(mock_event_bus)
        
        result = await planner.plan(
            lint_result=lint_result,
            user_requirements="Pipe flow simulation",
            config={"geometry": {"diameter": 0.1}},
        )
        
        assert len(result.work_items) > 0
        work_ids = [w.id for w in result.work_items]
        assert "choose_solver" in work_ids
        assert "choose_turbulence" in work_ids
    
    async def test_plan_runs_subagents(self, mock_event_bus, lint_result):
        """Test that planning runs sub-agents."""
        planner = Planner(mock_event_bus)
        
        result = await planner.plan(
            lint_result=lint_result,
            user_requirements="Pipe flow simulation",
            config={"geometry": {"diameter": 0.1}},
        )
        
        # Should have sub-agent results
        assert len(result.sub_results) > 0
    
    async def test_plan_emits_events(self, mock_event_bus, lint_result, mock_websocket):
        """Test that planning emits appropriate events."""
        planner = Planner(mock_event_bus)
        
        await planner.plan(
            lint_result=lint_result,
            user_requirements="Pipe flow simulation",
            config={"geometry": {"diameter": 0.1}},
        )
        
        call_args = mock_websocket.send_json.call_args_list
        event_types = [call.args[0]["type"] for call in call_args]
        
        assert "planning_started" in event_types
        assert "planning_complete" in event_types


class TestDependencyLevels:
    """Tests for dependency level calculation."""
    
    @pytest.fixture
    def mock_event_bus(self, mock_websocket, run_id):
        store = MagicMock()
        store.append_event = AsyncMock()
        return EventBus(
            run_id=run_id,
            websocket=mock_websocket,
            store=store,
            persist=False,
        )
    
    def test_build_dependency_levels(self, mock_event_bus):
        """Test dependency level building."""
        planner = Planner(mock_event_bus)
        
        # Get the work items with dependencies
        work_items = [
            WORK_ITEMS["choose_solver"],
            WORK_ITEMS["choose_turbulence"],  # depends on choose_solver
            WORK_ITEMS["configure_numerics"],  # depends on choose_solver
        ]
        
        levels = planner._build_dependency_levels(work_items)
        
        # Should have at least 2 levels
        assert len(levels) >= 2
        
        # First level should have choose_solver (no dependencies)
        first_level_ids = [w.id for w in levels[0]]
        assert "choose_solver" in first_level_ids
    
    def test_parallel_execution_threshold(self, mock_event_bus):
        """Test that parallel execution is triggered for >= 3 items."""
        planner = Planner(mock_event_bus)
        
        # Threshold is 3
        assert planner.PARALLEL_THRESHOLD == 3
