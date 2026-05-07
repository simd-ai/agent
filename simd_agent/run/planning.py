# simd_agent/planning.py
"""Planner for CFD simulation setup with parallel sub-agents."""

import asyncio
import logging
import time
from typing import Any

from simd_agent.run.event_bus import EventBus
from simd_agent.models import (
    BoundaryType,
    FlowRegime,
    LintResult,
    PlanningResult,
    SimulationConfigV1,
    SubAgentResult,
    WorkItem,
)

logger = logging.getLogger(__name__)

# Work item definitions
WORK_ITEMS = {
    "choose_solver": WorkItem(
        id="choose_solver",
        task="choose_solver",
        description="Select the appropriate OpenFOAM solver based on physics",
        priority=1,
    ),
    "choose_turbulence": WorkItem(
        id="choose_turbulence",
        task="choose_turbulence",
        description="Select turbulence model based on flow regime",
        priority=2,
        dependencies=["choose_solver"],
    ),
    "design_mesh": WorkItem(
        id="design_mesh",
        task="design_mesh",
        description="Design mesh strategy and resolution",
        priority=3,
    ),
    "define_boundaries": WorkItem(
        id="define_boundaries",
        task="define_boundaries",
        description="Define boundary conditions for all patches",
        priority=4,
    ),
    "configure_numerics": WorkItem(
        id="configure_numerics",
        task="configure_numerics",
        description="Configure numerical schemes and solution settings",
        priority=5,
        dependencies=["choose_solver"],
    ),
    "setup_initial": WorkItem(
        id="setup_initial",
        task="setup_initial",
        description="Set up initial field conditions",
        priority=6,
        dependencies=["define_boundaries"],
    ),
}


class SharedContext:
    """Thread-safe shared context for parallel sub-agents.
    
    Uses asyncio.Lock to prevent race conditions when merging results.
    Keys are deterministically merged - later results don't overwrite existing keys
    unless explicitly allowed.
    """
    
    def __init__(self, initial: dict[str, Any] | None = None):
        """Initialize shared context.
        
        Args:
            initial: Initial context data
        """
        self._data: dict[str, Any] = dict(initial or {})
        self._lock = asyncio.Lock()
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get a value from context."""
        async with self._lock:
            return self._data.get(key, default)
    
    async def set(self, key: str, value: Any, overwrite: bool = False) -> bool:
        """Set a value in context.
        
        Args:
            key: The key to set
            value: The value
            overwrite: Whether to overwrite existing values
            
        Returns:
            True if value was set, False if key existed and overwrite=False
        """
        async with self._lock:
            if key in self._data and not overwrite:
                return False
            self._data[key] = value
            return True
    
    async def merge(self, data: dict[str, Any], overwrite: bool = False) -> None:
        """Merge data into context.
        
        Args:
            data: Dictionary to merge
            overwrite: Whether to overwrite existing keys
        """
        async with self._lock:
            for key, value in data.items():
                if key not in self._data or overwrite:
                    self._data[key] = value
    
    async def snapshot(self) -> dict[str, Any]:
        """Get a snapshot of the current context."""
        async with self._lock:
            return dict(self._data)


class SubAgent:
    """A sub-agent that performs a specific planning task."""
    
    def __init__(
        self,
        work_item: WorkItem,
        event_bus: EventBus,
        shared_context: SharedContext,
    ):
        """Initialize sub-agent.
        
        Args:
            work_item: The work item to process
            event_bus: Event bus for progress updates
            shared_context: Shared context for reading/writing results
        """
        self.work_item = work_item
        self.event_bus = event_bus
        self.shared_context = shared_context
    
    async def run(self) -> SubAgentResult:
        """Execute the sub-agent task."""
        start_time = time.monotonic()
        
        await self.event_bus.emit_subagent_started(
            self.work_item.id,
            self.work_item.task,
        )
        
        try:
            # Dispatch to specific task handler
            handler = getattr(self, f"_handle_{self.work_item.task}", None)
            if handler:
                result = await handler()
            else:
                result = await self._default_handler()
            
            # Merge result into shared context
            await self.shared_context.merge(result)
            
            duration_ms = int((time.monotonic() - start_time) * 1000)
            
            await self.event_bus.emit_subagent_complete(
                self.work_item.id,
                self.work_item.task,
                result,
                duration_ms,
            )
            
            return SubAgentResult(
                work_item_id=self.work_item.id,
                task=self.work_item.task,
                result=result,
                duration_ms=duration_ms,
            )
        except Exception as e:
            logger.error(f"Sub-agent {self.work_item.task} failed: {e}")
            raise
    
    async def _default_handler(self) -> dict[str, Any]:
        """Default handler for unknown tasks."""
        return {"task": self.work_item.task, "status": "completed"}
    
    async def _handle_choose_solver(self) -> dict[str, Any]:
        """Choose the appropriate solver."""
        ctx = await self.shared_context.snapshot()
        regime = ctx.get("regime")
        case_type = ctx.get("case_type")
        is_thermal = ctx.get("is_thermal", False)
        is_transient = ctx.get("is_transient", False)
        
        await self.event_bus.emit_subagent_update(
            self.work_item.id,
            self.work_item.task,
            f"Analyzing case: regime={regime}, type={case_type}",
        )
        
        # Solver selection logic
        if is_thermal:
            solver = "buoyantPimpleFoam" if is_transient else "buoyantSimpleFoam"
        elif is_transient:
            solver = "pimpleFoam"
        else:
            solver = "simpleFoam"
        
        return {"solver": solver, "solver_reason": f"Selected for {case_type or 'general'} flow"}
    
    async def _handle_choose_turbulence(self) -> dict[str, Any]:
        """Choose turbulence model."""
        ctx = await self.shared_context.snapshot()
        regime = ctx.get("regime")
        reynolds = ctx.get("reynolds_number")
        
        await self.event_bus.emit_subagent_update(
            self.work_item.id,
            self.work_item.task,
            f"Selecting model for Re={reynolds:.0f}" if reynolds else "Selecting turbulence model",
        )
        
        if regime == FlowRegime.LAMINAR or regime == "laminar":
            model = "laminar"
            reason = "Laminar flow, no turbulence modeling needed"
        elif regime == FlowRegime.TRANSITIONAL or regime == "transitional":
            model = "kOmegaSST"
            reason = "Transitional flow, k-omega SST for accuracy"
        else:
            model = "kEpsilon"
            reason = "Turbulent flow, k-epsilon for robustness"
        
        return {"turbulence_model": model, "turbulence_reason": reason}
    
    async def _handle_design_mesh(self) -> dict[str, Any]:
        """Design mesh strategy."""
        ctx = await self.shared_context.snapshot()
        geometry = ctx.get("geometry", {})
        reynolds = ctx.get("reynolds_number")
        
        await self.event_bus.emit_subagent_update(
            self.work_item.id,
            self.work_item.task,
            "Designing mesh resolution",
        )
        
        # Determine mesh strategy
        if reynolds and reynolds > 100000:
            strategy = "boundary_layer_refinement"
            base_cells = 50
        elif reynolds and reynolds > 10000:
            strategy = "moderate_refinement"
            base_cells = 30
        else:
            strategy = "uniform"
            base_cells = 20
        
        return {
            "mesh_strategy": strategy,
            "mesh_base_cells": base_cells,
            "mesh_grading": 1.2 if strategy != "uniform" else 1.0,
        }
    
    async def _handle_define_boundaries(self) -> dict[str, Any]:
        """Define boundary conditions using actual BC data from config."""
        ctx = await self.shared_context.snapshot()
        case_type = ctx.get("case_type")
        inlet = ctx.get("inlet", {})
        
        # Get boundary conditions from normalized config
        boundary_conditions = ctx.get("boundary_conditions", {})
        
        await self.event_bus.emit_subagent_update(
            self.work_item.id,
            self.work_item.task,
            f"Configuring {len(boundary_conditions)} boundary conditions",
        )
        
        # If we have explicit BCs, use them
        if boundary_conditions:
            boundaries = {}
            for patch_name, bc_data in boundary_conditions.items():
                bc_type = bc_data.get("type", "wall")
                
                if bc_type == "inlet" and "velocity" in bc_data:
                    vel_data = bc_data["velocity"]
                    boundaries[patch_name] = {
                        "type": vel_data.get("type", "fixedValue"),
                        "velocity": vel_data.get("value", [0, 0, 0]),
                    }
                elif bc_type == "outlet":
                    pres_data = bc_data.get("pressure", {})
                    boundaries[patch_name] = {
                        "type": pres_data.get("type", "zeroGradient"),
                        "pressure": pres_data.get("value", 0),
                    }
                elif bc_type == "wall":
                    boundaries[patch_name] = {
                        "type": "noSlip",
                    }
                elif bc_type == "symmetry":
                    boundaries[patch_name] = {
                        "type": "symmetry",
                    }
                elif bc_type == "empty":
                    boundaries[patch_name] = {
                        "type": "empty",
                    }
                else:
                    # Default to wall
                    boundaries[patch_name] = {
                        "type": "noSlip",
                    }
            
            return {"boundaries": boundaries, "boundary_conditions": boundary_conditions}
        
        # Fallback: Standard boundary setup
        boundaries = {
            "inlet": {
                "type": "fixedValue",
                "velocity": inlet.get("velocity") or inlet.get("magnitude") or [0, 0, 0],
            },
            "outlet": {
                "type": "zeroGradient",
            },
            "walls": {
                "type": "noSlip",
            },
        }
        
        return {"boundaries": boundaries}
    
    async def _handle_configure_numerics(self) -> dict[str, Any]:
        """Configure numerical schemes."""
        ctx = await self.shared_context.snapshot()
        solver = ctx.get("solver", "simpleFoam")
        
        await self.event_bus.emit_subagent_update(
            self.work_item.id,
            self.work_item.task,
            f"Configuring numerics for {solver}",
        )
        
        # Standard numerical settings
        numerics = {
            "ddtSchemes": "steadyState" if "simple" in solver.lower() else "Euler",
            "gradSchemes": "Gauss linear",
            "divSchemes": "Gauss linearUpwind",
            "laplacianSchemes": "Gauss linear corrected",
            "interpolationSchemes": "linear",
            "snGradSchemes": "corrected",
        }
        
        return {"numerics": numerics}
    
    async def _handle_setup_initial(self) -> dict[str, Any]:
        """Set up initial conditions."""
        ctx = await self.shared_context.snapshot()
        inlet = ctx.get("inlet", {})
        
        await self.event_bus.emit_subagent_update(
            self.work_item.id,
            self.work_item.task,
            "Setting initial field values",
        )
        
        velocity = inlet.get("velocity", [0, 0, 0])
        
        initial = {
            "U": {"internalField": f"uniform ({velocity[0]} {velocity[1]} {velocity[2]})" if isinstance(velocity, list) else f"uniform ({velocity} 0 0)"},
            "p": {"internalField": "uniform 0"},
        }
        
        return {"initial_conditions": initial}


class Planner:
    """Plans CFD simulation setup with parallel sub-agent execution."""
    
    # Threshold for parallel execution
    PARALLEL_THRESHOLD = 3
    
    def __init__(self, event_bus: EventBus):
        """Initialize the planner.
        
        Args:
            event_bus: Event bus for progress updates
        """
        self.event_bus = event_bus
    
    async def plan(
        self,
        lint_result: LintResult,
        user_requirements: str,
        config: dict[str, Any],
    ) -> PlanningResult:
        """Run the planning phase.
        
        Args:
            lint_result: Result from CFD linting
            user_requirements: User's requirements text
            config: Validated configuration
            
        Returns:
            PlanningResult with all planning decisions
        """
        await self.event_bus.emit_planning_started()
        
        # Extract boundary conditions from normalized config if available
        boundary_conditions = {}
        inlet_data = {}
        normalized_config: SimulationConfigV1 | None = lint_result.normalized_config
        
        if normalized_config:
            # Convert boundary conditions to codegen-friendly format
            for patch_name, bc in normalized_config.boundary_conditions.items():
                patch_type = bc.patch_type.value if isinstance(bc.patch_type, BoundaryType) else str(bc.patch_type)
                bc_data: dict[str, Any] = {"type": patch_type}
                
                if bc.velocity:
                    vel_vec = bc.velocity.get_velocity_vector()
                    bc_data["velocity"] = {
                        "type": bc.velocity.type,
                        "value": vel_vec or bc.velocity.value,
                        "magnitude": bc.velocity.get_magnitude(),
                    }
                    # Track inlet for shared context
                    if bc.is_inlet():
                        inlet_data = {
                            "velocity": vel_vec or bc.velocity.value,
                            "magnitude": bc.velocity.get_magnitude(),
                        }
                
                if bc.pressure:
                    bc_data["pressure"] = {
                        "type": bc.pressure.type,
                        "value": bc.pressure.value,
                    }
                
                if bc.temperature:
                    bc_data["temperature"] = {
                        "type": bc.temperature.type,
                        "value": bc.temperature.value,
                    }
                
                boundary_conditions[patch_name] = bc_data
        else:
            # Fallback to legacy format
            inlet_data = config.get("inlet", {})
            boundary_conditions = config.get("boundary_conditions", {})
        
        # Determine thermal and transient flags
        is_thermal = (
            "temperature" in user_requirements.lower() or 
            "heat" in user_requirements.lower() or
            (normalized_config and normalized_config.physics.heat_transfer)
        )
        is_transient = (
            "transient" in user_requirements.lower() or 
            "unsteady" in user_requirements.lower() or
            (normalized_config and str(normalized_config.physics.time_scheme.value) == "transient")
        )
        
        # Initialize shared context with linting results and BCs
        shared_context = SharedContext({
            "case_type": lint_result.detected_case_type,
            "regime": lint_result.detected_regime,
            "reynolds_number": lint_result.reynolds_number,
            "validated_config": lint_result.validated_config,
            "normalized_config": normalized_config.model_dump() if normalized_config else None,
            "user_requirements": user_requirements,
            "geometry": config.get("geometry", {}),
            "inlet": inlet_data,
            "boundary_conditions": boundary_conditions,  # Full BC data for codegen
            "is_thermal": is_thermal,
            "is_transient": is_transient,
            "fluid": normalized_config.fluid.model_dump() if normalized_config else {},
        })
        
        # Determine which work items to run
        work_items = self._select_work_items(lint_result)
        
        # Run sub-agents
        if len(work_items) >= self.PARALLEL_THRESHOLD:
            sub_results = await self._run_parallel(work_items, shared_context)
        else:
            sub_results = await self._run_sequential(work_items, shared_context)
        
        # Get final context
        final_ctx = await shared_context.snapshot()
        
        # Build planning result
        result = PlanningResult(
            work_items=work_items,
            case_type=lint_result.detected_case_type or "unknown",
            regime=lint_result.detected_regime,
            solver=final_ctx.get("solver", "simpleFoam"),
            turbulence_model=final_ctx.get("turbulence_model"),
            mesh_strategy=final_ctx.get("mesh_strategy", "uniform"),
            sub_results=sub_results,
        )
        
        await self.event_bus.emit_planning_complete(
            work_items=[w.model_dump() for w in work_items],
            case_type=result.case_type,
            solver=result.solver,
        )
        
        return result
    
    def _select_work_items(self, lint_result: LintResult) -> list[WorkItem]:
        """Select which work items to run based on linting result."""
        items = []
        
        # Always include core items
        items.append(WORK_ITEMS["choose_solver"])
        items.append(WORK_ITEMS["choose_turbulence"])
        items.append(WORK_ITEMS["design_mesh"])
        items.append(WORK_ITEMS["define_boundaries"])
        items.append(WORK_ITEMS["configure_numerics"])
        items.append(WORK_ITEMS["setup_initial"])
        
        # Sort by priority
        items.sort(key=lambda x: x.priority)
        
        return items
    
    async def _run_parallel(
        self,
        work_items: list[WorkItem],
        shared_context: SharedContext,
    ) -> list[SubAgentResult]:
        """Run sub-agents in parallel using asyncio.TaskGroup."""
        results = []
        
        # Group items by dependency level
        levels = self._build_dependency_levels(work_items)
        
        for level in levels:
            # Run items at this level in parallel
            async with asyncio.TaskGroup() as tg:
                tasks = []
                for item in level:
                    agent = SubAgent(item, self.event_bus, shared_context)
                    task = tg.create_task(agent.run())
                    tasks.append(task)
            
            # Collect results from this level
            for task in tasks:
                results.append(task.result())
        
        return results
    
    async def _run_sequential(
        self,
        work_items: list[WorkItem],
        shared_context: SharedContext,
    ) -> list[SubAgentResult]:
        """Run sub-agents sequentially."""
        results = []
        
        for item in work_items:
            agent = SubAgent(item, self.event_bus, shared_context)
            result = await agent.run()
            results.append(result)
        
        return results
    
    def _build_dependency_levels(
        self,
        work_items: list[WorkItem],
    ) -> list[list[WorkItem]]:
        """Build dependency levels for parallel execution.
        
        Items in the same level can run in parallel.
        """
        item_map = {item.id: item for item in work_items}
        levels: list[list[WorkItem]] = []
        remaining = set(item.id for item in work_items)
        completed = set()
        
        while remaining:
            # Find items whose dependencies are all completed
            current_level = []
            for item_id in list(remaining):
                item = item_map[item_id]
                deps = set(item.dependencies)
                if deps.issubset(completed):
                    current_level.append(item)
            
            if not current_level:
                # No progress - circular dependency or missing deps
                # Just add remaining items
                current_level = [item_map[item_id] for item_id in remaining]
                remaining.clear()
            else:
                for item in current_level:
                    remaining.discard(item.id)
                    completed.add(item.id)
            
            levels.append(current_level)
        
        return levels
