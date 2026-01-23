# simd_agent/orchestration.py
"""Main orchestration logic for CFD workflows."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any
from uuid import UUID

from simd_agent.error_summarizer import ErrorSummarizer
from simd_agent.event_bus import EventBus
from simd_agent.linting import CFDLinter
from simd_agent.models import (
    BoundaryType,
    FinalResult,
    LintResult,
    Operation,
    RunStatus,
    SandboxState,
    StartRequest,
)
from simd_agent.packaging import (
    extract_file_blocks,
    package_from_llm_output,
    merge_files,
)
from simd_agent.planning import Planner
from simd_agent.sandbox_client import SandboxClient, SandboxError
from simd_agent.settings import get_settings
from simd_agent.store import EventStore

logger = logging.getLogger(__name__)

# Path to prompt packs
PROMPTS_DIR = Path(__file__).parent / "prompts" / "packs"


class OrchestrationError(Exception):
    """Error during orchestration."""
    pass


class Orchestrator:
    """Orchestrates CFD workflows end-to-end.
    
    Supports two operations:
    - CFD_LINT: Validate/normalize config and return recommendations
    - CFD_CODEGEN_RUN: Generate OpenFOAM case and run with self-healing
    """
    
    def __init__(
        self,
        run_id: UUID,
        event_bus: EventBus,
        store: EventStore,
        request: StartRequest,
    ):
        """Initialize the orchestrator.
        
        Args:
            run_id: The run ID
            event_bus: Event bus for streaming events
            store: Event store for persistence
            request: The start request
        """
        self.run_id = run_id
        self.event_bus = event_bus
        self.store = store
        self.request = request
        self.settings = get_settings()
        
        # Components (lazily initialized)
        self._linter: CFDLinter | None = None
        self._planner: Planner | None = None
        self._sandbox: SandboxClient | None = None
        self._error_summarizer: ErrorSummarizer | None = None
        self._code_generator: Any = None
        
        # State
        self._iteration = 0
        self._retries = 0
        self._current_files: dict[str, str] = {}
        self._lint_result: LintResult | None = None
        self._previous_errors: list[dict[str, Any]] = []
    
    @property
    def linter(self) -> CFDLinter:
        """Get or create the CFD linter."""
        if self._linter is None:
            self._linter = CFDLinter(event_bus=self.event_bus)
        return self._linter
    
    @property
    def planner(self) -> Planner:
        """Get or create the planner."""
        if self._planner is None:
            self._planner = Planner(event_bus=self.event_bus)
        return self._planner
    
    @property
    def sandbox(self) -> SandboxClient:
        """Get or create the sandbox client."""
        if self._sandbox is None:
            self._sandbox = SandboxClient()
        return self._sandbox
    
    @property
    def error_summarizer(self) -> ErrorSummarizer:
        """Get or create the error summarizer."""
        if self._error_summarizer is None:
            self._error_summarizer = ErrorSummarizer(
                event_bus=self.event_bus,
                use_llm=True,
                code_generator=self._code_generator,
            )
        return self._error_summarizer
    
    async def _init_code_generator(self) -> None:
        """Initialize the code generator from codegen."""
        # Use internal mock generator for testing without codegen dependency
        if self.request.provider == "mock":
            logger.info("Using internal mock generator for provider='mock'")
            self._code_generator = MockCodeGenerator()
            return
        
        try:
            from codegen import CodeGenerator
            
            self._code_generator = CodeGenerator(
                provider=self.request.provider,
                prompt_pack=self.request.prompt_pack,
            )
        except ImportError:
            logger.warning("codegen not available, using mock generator")
            self._code_generator = MockCodeGenerator()
    
    async def run(self) -> FinalResult:
        """Execute the requested operation.
        
        Returns:
            FinalResult with operation outcome
        """
        logger.info("=" * 50)
        logger.info("[ORCHESTRATOR] Starting run...")
        logger.info(f"[ORCHESTRATOR]   Run ID: {self.run_id}")
        logger.info(f"[ORCHESTRATOR]   Operation: {self.request.op.value}")
        logger.info(f"[ORCHESTRATOR]   Provider: {self.request.provider}")
        logger.info(f"[ORCHESTRATOR]   Prompt Pack: {self.request.prompt_pack}")
        logger.info(f"[ORCHESTRATOR]   User requirements: {self.request.user_requirements}")
        logger.info(f"[ORCHESTRATOR]   Simulation config:")
        import json
        for key, value in self.request.simulation_config.items():
            logger.info(f"[ORCHESTRATOR]     {key}: {json.dumps(value) if isinstance(value, (dict, list)) else value}")
        logger.info(f"[ORCHESTRATOR]   Constraints: max_retries={self.request.constraints.max_retries}, timeout={self.request.constraints.timeout_seconds}s")
        logger.info("=" * 50)
        
        logger.info("[ORCHESTRATOR] Initializing code generator...")
        await self._init_code_generator()
        logger.info("[ORCHESTRATOR] Code generator initialized: %s", type(self._code_generator).__name__)
        
        logger.info("[ORCHESTRATOR] Emitting RUN_STARTED event...")
        await self.event_bus.emit_run_started(
            self.request.op.value,
            self.request.provider,
        )
        
        try:
            if self.request.op == Operation.CFD_LINT:
                logger.info("[ORCHESTRATOR] Running CFD_LINT operation...")
                return await self._run_lint()
            elif self.request.op == Operation.CFD_CODEGEN_RUN:
                logger.info("[ORCHESTRATOR] Running CFD_CODEGEN_RUN operation...")
                return await self._run_codegen_and_sandbox()
            else:
                raise OrchestrationError(f"Unknown operation: {self.request.op}")
        except Exception as e:
            logger.exception(f"[ORCHESTRATOR] Orchestration failed: {e}")
            await self.event_bus.emit_run_failed(str(e))
            return FinalResult(
                status=RunStatus.FAILED,
                error=str(e),
                iterations=self._iteration,
                retries=self._retries,
            )
        finally:
            if self._sandbox:
                await self._sandbox.close()
    
    async def _run_lint(self) -> FinalResult:
        """Run the CFD linting operation."""
        logger.info("[LINT] Starting CFD linting...")
        logger.info(f"[LINT] Input config keys: {list(self.request.simulation_config.keys())}")
        
        await self.event_bus.emit_lint_started()
        
        logger.info("[LINT] Calling linter.lint()...")
        lint_result = await self.linter.lint(
            self.request.simulation_config,
            self.request.user_requirements,
        )
        
        logger.info("[LINT] Linting complete:")
        logger.info(f"[LINT]   - Issues: {len(lint_result.issues)}")
        logger.info(f"[LINT]   - Apply changes: {len(lint_result.apply_changes)}")
        logger.info(f"[LINT]   - Detected regime: {lint_result.detected_regime}")
        logger.info(f"[LINT]   - Selected solver: {lint_result.selected_solver}")
        logger.info(f"[LINT]   - Reynolds number: {lint_result.reynolds_number}")
        logger.info(f"[LINT]   - Validated config keys: {list(lint_result.validated_config.keys())}")
        
        # Log full issues
        if lint_result.issues:
            logger.info("[LINT] === ISSUES ===")
            for i, issue in enumerate(lint_result.issues, 1):
                logger.info(f"[LINT]   Issue #{i}:")
                logger.info(f"[LINT]     - Code: {issue.code}")
                logger.info(f"[LINT]     - Severity: {issue.severity}")
                logger.info(f"[LINT]     - Path: {issue.path}")
                logger.info(f"[LINT]     - Message: {issue.message}")
        
        # Log full apply_changes (recommendations)
        if lint_result.apply_changes:
            logger.info("[LINT] === RECOMMENDATIONS ===")
            for i, change in enumerate(lint_result.apply_changes, 1):
                logger.info(f"[LINT]   Recommendation #{i}:")
                logger.info(f"[LINT]     - Path: {change.path}")
                logger.info(f"[LINT]     - Value: {change.value}")
                logger.info(f"[LINT]     - Severity: {change.severity}")
                logger.info(f"[LINT]     - Reason: {change.reason}")
        
        # Log validated config
        logger.info("[LINT] === VALIDATED CONFIG ===")
        for key, value in lint_result.validated_config.items():
            logger.info(f"[LINT]   {key}: {json.dumps(value) if isinstance(value, (dict, list)) else value}")
        
        logger.info("[LINT] Emitting LINT_RESULT event...")
        await self.event_bus.emit_lint_result(
            validated_config=lint_result.validated_config,
            apply_changes=[c.model_dump() for c in lint_result.apply_changes],
            issues=[i.model_dump() for i in lint_result.issues],
            regime=lint_result.detected_regime.value if lint_result.detected_regime else None,
            solver=lint_result.selected_solver,
            reynolds=lint_result.reynolds_number,
        )
        
        logger.info("[LINT] Finalizing run in database...")
        await self.store.finalize_run(
            run_id=self.run_id,
            status=RunStatus.SUCCEEDED,
            validated_config=lint_result.validated_config,
            result={"lint_result": lint_result.model_dump()},
        )
        
        logger.info("[LINT] Emitting FINAL event...")
        await self.event_bus.emit_final(
            status="succeeded",
            validated_config=lint_result.validated_config,
            summary=f"Linting complete: {len(lint_result.issues)} issues, {len(lint_result.apply_changes)} recommendations",
        )
        
        logger.info("[LINT] CFD linting operation completed successfully")
        return FinalResult(
            status=RunStatus.SUCCEEDED,
            validated_config=lint_result.validated_config,
            summary=f"Linting complete: {len(lint_result.issues)} issues",
        )
    
    async def _run_codegen_and_sandbox(self) -> FinalResult:
        """Run the full codegen + sandbox workflow with self-healing."""
        max_retries = self.request.constraints.max_retries
        
        # Step 1: Detect simulation type
        case_type = self._detect_simulation_type()
        if case_type is None:
            await self.event_bus.emit_simulation_not_clear(
                "Could not determine simulation type from requirements"
            )
            await self.event_bus.emit_final(
                status="not_clear",
                summary="Simulation type could not be determined",
            )
            return FinalResult(
                status=RunStatus.NOT_CLEAR,
                summary="Simulation type unclear",
            )
        
        # Step 2: Run CFD linting with operation context
        await self.event_bus.emit_lint_started()
        self._lint_result = await self.linter.lint(
            self.request.simulation_config,
            self.request.user_requirements,
            operation="CFD_CODEGEN_RUN",  # Stricter validation
        )
        
        await self.event_bus.emit_lint_result(
            validated_config=self._lint_result.validated_config,
            apply_changes=[c.model_dump() for c in self._lint_result.apply_changes],
            issues=[i.model_dump() for i in self._lint_result.issues],
            regime=self._lint_result.detected_regime.value if self._lint_result.detected_regime else None,
            solver=self._lint_result.selected_solver,
            reynolds=self._lint_result.reynolds_number,
            missing_fields=[m.model_dump() for m in self._lint_result.missing_fields],
            is_complete=self._lint_result.is_complete,
            detected_case_type=self._lint_result.detected_case_type,
        )
        
        # Step 2.5: ENFORCE COMPLETENESS - Stop if config is incomplete
        if not self._lint_result.is_complete:
            logger.warning(f"[CODEGEN] Config incomplete: {len(self._lint_result.missing_fields)} missing fields")
            for mf in self._lint_result.missing_fields:
                logger.warning(f"[CODEGEN]   - {mf.field}: {mf.description}")
            
            # Emit config_incomplete event
            await self.event_bus.emit_config_incomplete(
                missing_fields=[m.model_dump() for m in self._lint_result.missing_fields],
                suggestions=[
                    {"field": m.field, "value": m.suggested_value}
                    for m in self._lint_result.missing_fields if m.suggested_value
                ],
                can_lint=True,
                can_codegen=False,
            )
            
            # Finalize with CONFIG_INCOMPLETE status
            await self.store.finalize_run(
                run_id=self.run_id,
                status=RunStatus.CONFIG_INCOMPLETE,
                validated_config=self._lint_result.validated_config,
                result={
                    "missing_fields": [m.model_dump() for m in self._lint_result.missing_fields],
                    "lint_result": self._lint_result.model_dump(),
                },
            )
            
            missing_summary = ", ".join(m.field for m in self._lint_result.missing_fields[:3])
            if len(self._lint_result.missing_fields) > 3:
                missing_summary += f" (+{len(self._lint_result.missing_fields) - 3} more)"
            
            await self.event_bus.emit_final(
                status="config_incomplete",
                validated_config=self._lint_result.validated_config,
                summary=f"Config incomplete: missing {missing_summary}",
                error=f"Cannot run codegen: {len(self._lint_result.missing_fields)} required fields missing",
            )
            
            return FinalResult(
                status=RunStatus.CONFIG_INCOMPLETE,
                validated_config=self._lint_result.validated_config,
                summary=f"Config incomplete: {len(self._lint_result.missing_fields)} fields missing",
                error=f"Missing: {missing_summary}",
            )
        
        logger.info("[CODEGEN] Config is complete, proceeding with code generation")
        
        # Step 3: Planning phase
        planning_result = await self.planner.plan(
            lint_result=self._lint_result,
            user_requirements=self.request.user_requirements,
            config=self._lint_result.validated_config,
        )
        
        # Step 4: Self-healing loop
        while self._retries <= max_retries:
            self._iteration += 1
            
            try:
                # Generate code
                await self.event_bus.emit_codegen_started(self._iteration)
                
                code_result = await self._generate_code(planning_result)
                files = extract_file_blocks(code_result)
                
                if not files:
                    raise OrchestrationError("Code generation produced no files")
                
                # Merge with previous files if retrying
                if self._current_files and self._retries > 0:
                    files = merge_files(self._current_files, files)
                
                self._current_files = files
                
                await self.event_bus.emit_codegen_iteration(
                    self._iteration,
                    list(files.keys()),
                )
                
                # Package case
                solver = planning_result.solver or "simpleFoam"
                zip_bytes, file_list, warnings = package_from_llm_output(
                    code_result,
                    solver=solver,
                )
                
                await self.event_bus.emit_codegen_complete(
                    self._iteration,
                    len(zip_bytes),
                )
                
                # Submit to sandbox
                sandbox_result = await self._run_in_sandbox(zip_bytes)
                
                if sandbox_result["success"]:
                    # Success!
                    artifacts = sandbox_result.get("artifacts", [])
                    await self.event_bus.emit_run_succeeded(
                        f"Case executed successfully after {self._iteration} iteration(s)",
                        artifacts,
                    )
                    
                    await self.store.finalize_run(
                        run_id=self.run_id,
                        status=RunStatus.SUCCEEDED,
                        validated_config=self._lint_result.validated_config,
                        result={
                            "artifacts": artifacts,
                            "iterations": self._iteration,
                            "solver": solver,
                        },
                        attempts=self._iteration,
                    )
                    
                    await self.event_bus.emit_final(
                        status="succeeded",
                        validated_config=self._lint_result.validated_config,
                        artifacts=artifacts,
                        iterations=self._iteration,
                        retries=self._retries,
                        summary=f"Case executed successfully",
                        case_type=case_type,
                        solver=solver,
                    )
                    
                    return FinalResult(
                        status=RunStatus.SUCCEEDED,
                        validated_config=self._lint_result.validated_config,
                        artifacts=[],
                        iterations=self._iteration,
                        retries=self._retries,
                        case_type=case_type,
                        solver=solver,
                    )
                
                # Sandbox failed - check if we can retry
                if self._retries >= max_retries:
                    break
                
                self._retries += 1
                
                # Summarize error
                error_summary = await self.error_summarizer.summarize(
                    sandbox_result["logs"],
                    sandbox_result.get("exit_code"),
                    self._current_files,
                )
                
                await self.event_bus.emit_error_summary(
                    error_summary.root_cause,
                    error_summary.actionable_changes,
                    error_summary.affected_files,
                )
                
                # Store error for context in next iteration
                self._previous_errors.append({
                    "iteration": self._iteration,
                    "root_cause": error_summary.root_cause,
                    "changes": error_summary.actionable_changes,
                })
                
                await self.event_bus.emit_retrying(self._retries, max_retries)
                
            except SandboxError as e:
                logger.error(f"Sandbox error: {e}")
                if self._retries >= max_retries:
                    break
                self._retries += 1
                await self.event_bus.emit_retrying(self._retries, max_retries)
        
        # Max retries exceeded
        await self.event_bus.emit_run_failed(
            f"Failed after {max_retries} retries"
        )
        
        await self.store.finalize_run(
            run_id=self.run_id,
            status=RunStatus.FAILED,
            validated_config=self._lint_result.validated_config if self._lint_result else None,
            result={"error": "Max retries exceeded", "iterations": self._iteration},
            attempts=self._iteration,
        )
        
        await self.event_bus.emit_final(
            status="failed",
            validated_config=self._lint_result.validated_config if self._lint_result else None,
            iterations=self._iteration,
            retries=self._retries,
            summary=f"Failed after {max_retries} retries",
            error="Max retries exceeded",
        )
        
        return FinalResult(
            status=RunStatus.FAILED,
            iterations=self._iteration,
            retries=self._retries,
            error="Max retries exceeded",
        )
    
    def _detect_simulation_type(self) -> str | None:
        """Detect the simulation type from user requirements."""
        req = self.request.user_requirements.lower()
        config = self.request.simulation_config
        
        # Check explicit case_type
        if "case_type" in config:
            return config["case_type"]
        
        # Pattern matching
        patterns = {
            "pipe_flow": ["pipe", "duct", "channel", "tube", "internal flow"],
            "external_aero": ["external", "aerodynamic", "airfoil", "wing", "vehicle", "car", "cylinder external"],
            "heat_transfer": ["heat", "thermal", "temperature", "convection", "cooling", "heating"],
            "cavity": ["cavity", "lid-driven", "enclosed"],
            "mixing": ["mixing", "mixer", "impeller", "stirred"],
        }
        
        for case_type, keywords in patterns.items():
            for kw in keywords:
                if kw in req:
                    return case_type
        
        # If we have geometry hints
        geometry = config.get("geometry", {})
        if geometry.get("type") in ["pipe", "tube"]:
            return "pipe_flow"
        if geometry.get("type") in ["airfoil", "wing"]:
            return "external_aero"
        
        # Default to pipe_flow if we have diameter
        if "diameter" in geometry:
            return "pipe_flow"
        
        return None
    
    async def _generate_code(self, planning_result: Any) -> str:
        """Generate OpenFOAM case code using codegen."""
        try:
            from codegen import GenerationContext
            
            # Build enhanced requirements with planning context
            enhanced_requirements = self._build_requirements(planning_result)
            
            # Check if we're retrying after a failure
            if self._previous_errors and self._current_files:
                # Use codefix task for retries
                last_error = self._previous_errors[-1]
                previous_code = "\n\n".join(
                    f"```file:{path}\n{content}\n```"
                    for path, content in self._current_files.items()
                )
                
                context = GenerationContext(
                    task="codefix",
                    domain="openfoam",
                    requirements=enhanced_requirements,
                    previous_code=previous_code,
                    sandbox_error=last_error.get("root_cause", "Unknown error"),
                    sandbox_logs="\n".join(
                        change.get("description", "")
                        for change in last_error.get("changes", [])
                    ),
                )
            else:
                # Initial generation
                context = GenerationContext(
                    task="codegen",
                    domain="openfoam",
                    requirements=enhanced_requirements,
                )
            
            # codegen.generate() is synchronous, run in thread to avoid blocking
            result = await asyncio.to_thread(self._code_generator.generate, context)
            return result.final_text
        except ImportError:
            # Fallback to mock
            return self._mock_generate_code(planning_result)
    
    def _build_requirements(self, planning_result: Any) -> str:
        """Build enhanced requirements string with planning context and explicit BCs."""
        parts = [self.request.user_requirements]
        
        # Add validated config context
        if self._lint_result and self._lint_result.validated_config:
            parts.append(f"\n\nValidated configuration:\n```json\n{json.dumps(self._lint_result.validated_config, indent=2)}\n```")
        
        # Add planning decisions
        if planning_result.solver:
            parts.append(f"\n\nSolver: {planning_result.solver}")
        if planning_result.turbulence_model:
            parts.append(f"Turbulence model: {planning_result.turbulence_model}")
        if planning_result.mesh_strategy:
            parts.append(f"Mesh strategy: {planning_result.mesh_strategy}")
        if planning_result.case_type:
            parts.append(f"Case type: {planning_result.case_type}")
        
        # Add explicit boundary conditions section for codegen
        if self._lint_result and self._lint_result.normalized_config:
            bcs = self._lint_result.normalized_config.boundary_conditions
            if bcs:
                parts.append("\n\n## Boundary Conditions (MUST USE THESE):\n")
                for patch_name, bc in bcs.items():
                    patch_type = bc.patch_type.value if isinstance(bc.patch_type, BoundaryType) else str(bc.patch_type)
                    parts.append(f"\n### Patch: {patch_name} (type: {patch_type})")
                    
                    if bc.velocity:
                        vel_vec = bc.velocity.get_velocity_vector()
                        if vel_vec:
                            parts.append(f"\n- Velocity (U): type={bc.velocity.type}, value=({vel_vec[0]} {vel_vec[1]} {vel_vec[2]})")
                        elif bc.velocity.get_magnitude():
                            parts.append(f"\n- Velocity (U): type={bc.velocity.type}, magnitude={bc.velocity.get_magnitude()} m/s")
                    
                    if bc.pressure:
                        parts.append(f"\n- Pressure (p): type={bc.pressure.type}, value={bc.pressure.value}")
                    
                    if bc.temperature:
                        parts.append(f"\n- Temperature (T): type={bc.temperature.type}, value={bc.temperature.value} K")
                    
                    # Wall default
                    if bc.is_wall() and not bc.velocity:
                        parts.append(f"\n- Velocity (U): type=noSlip")
                    
                    # Outlet default
                    if bc.is_outlet() and not bc.pressure:
                        parts.append(f"\n- Pressure (p): type=fixedValue, value=0")
                
                # Add fluid properties
                fluid = self._lint_result.normalized_config.fluid
                parts.append(f"\n\n## Fluid Properties:")
                parts.append(f"\n- Fluid: {fluid.name}")
                parts.append(f"\n- Density (rho): {fluid.density} kg/m³")
                parts.append(f"\n- Kinematic viscosity (nu): {fluid.kinematic_viscosity} m²/s")
                
                # Add Reynolds number if calculated
                if self._lint_result.reynolds_number:
                    parts.append(f"\n- Reynolds number: {self._lint_result.reynolds_number:.0f}")
        
        return "".join(parts)
    
    def _mock_generate_code(self, planning_result: Any) -> str:
        """Generate mock OpenFOAM case for testing."""
        solver = planning_result.solver or "simpleFoam"
        
        return f'''```file:system/controlDict
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}}

application     {solver};
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         1000;
deltaT          1;
writeControl    timeStep;
writeInterval   100;
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
```

```file:system/fvSchemes
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}}

ddtSchemes
{{
    default         steadyState;
}}

gradSchemes
{{
    default         Gauss linear;
}}

divSchemes
{{
    default         none;
    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,k)      bounded Gauss upwind;
    div(phi,epsilon) bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}}

laplacianSchemes
{{
    default         Gauss linear corrected;
}}

interpolationSchemes
{{
    default         linear;
}}

snGradSchemes
{{
    default         corrected;
}}
```

```file:system/fvSolution
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}}

solvers
{{
    p
    {{
        solver          GAMG;
        tolerance       1e-06;
        relTol          0.1;
        smoother        GaussSeidel;
    }}

    U
    {{
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-05;
        relTol          0.1;
    }}
}}

SIMPLE
{{
    nNonOrthogonalCorrectors 0;
    consistent      yes;
    residualControl
    {{
        p               1e-4;
        U               1e-4;
    }}
}}

relaxationFactors
{{
    fields
    {{
        p               0.3;
    }}
    equations
    {{
        U               0.7;
    }}
}}
```

```file:system/blockMeshDict
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}

scale   1;

vertices
(
    (0 0 0)
    (1 0 0)
    (1 0.1 0)
    (0 0.1 0)
    (0 0 0.1)
    (1 0 0.1)
    (1 0.1 0.1)
    (0 0.1 0.1)
);

blocks
(
    hex (0 1 2 3 4 5 6 7) (20 10 1) simpleGrading (1 1 1)
);

boundary
(
    inlet
    {{
        type patch;
        faces
        (
            (0 4 7 3)
        );
    }}
    outlet
    {{
        type patch;
        faces
        (
            (1 2 6 5)
        );
    }}
    walls
    {{
        type wall;
        faces
        (
            (0 1 5 4)
            (3 7 6 2)
        );
    }}
    frontAndBack
    {{
        type empty;
        faces
        (
            (0 3 2 1)
            (4 5 6 7)
        );
    }}
);
```

```file:0/U
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}}

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{{
    inlet
    {{
        type            fixedValue;
        value           uniform (1 0 0);
    }}
    outlet
    {{
        type            zeroGradient;
    }}
    walls
    {{
        type            noSlip;
    }}
    frontAndBack
    {{
        type            empty;
    }}
}}
```

```file:0/p
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p;
}}

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{{
    inlet
    {{
        type            zeroGradient;
    }}
    outlet
    {{
        type            fixedValue;
        value           uniform 0;
    }}
    walls
    {{
        type            zeroGradient;
    }}
    frontAndBack
    {{
        type            empty;
    }}
}}
```

```file:constant/transportProperties
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      transportProperties;
}}

transportModel  Newtonian;
nu              [0 2 -1 0 0 0 0] 1e-06;
```
'''
    
    async def _run_in_sandbox(self, zip_bytes: bytes) -> dict[str, Any]:
        """Submit case to sandbox and wait for completion."""
        # Submit
        submit_result = await self.sandbox.submit_run(
            zip_bytes,
            run_script="run.sh",
            metadata={"run_id": str(self.run_id), "iteration": self._iteration},
        )
        sandbox_run_id = submit_result.run_id
        
        await self.event_bus.emit_sandbox_submitted(sandbox_run_id)
        
        # Wait for completion with status updates
        async def on_status(status):
            await self.event_bus.emit_sandbox_status(status.state.value, sandbox_run_id)
        
        try:
            final_status = await self.sandbox.wait_for_completion(
                sandbox_run_id,
                on_status=on_status,
            )
        except SandboxError as e:
            return {"success": False, "error": str(e), "logs": ""}
        
        # Get logs
        logs = await self.sandbox.get_logs(sandbox_run_id)
        truncated_logs = self.error_summarizer.truncate_logs(logs)
        
        await self.event_bus.emit_sandbox_logs(
            sandbox_run_id,
            truncated_logs,
            len(logs.split("\n")) > self.settings.max_log_lines_in_event,
        )
        
        if final_status.state == SandboxState.SUCCEEDED:
            # Get artifacts
            artifacts_resp = await self.sandbox.get_artifacts(sandbox_run_id)
            artifacts = [a.model_dump() for a in artifacts_resp.artifacts]
            
            await self.event_bus.emit_sandbox_succeeded(sandbox_run_id, artifacts)
            
            return {
                "success": True,
                "sandbox_run_id": sandbox_run_id,
                "artifacts": artifacts,
                "logs": logs,
            }
        else:
            await self.event_bus.emit_sandbox_failed(
                sandbox_run_id,
                final_status.exit_code,
                truncated_logs,
            )
            
            return {
                "success": False,
                "sandbox_run_id": sandbox_run_id,
                "exit_code": final_status.exit_code,
                "logs": logs,
            }


class MockCodeGenerator:
    """Mock code generator for testing without codegen."""
    
    def generate(self, context: Any) -> Any:
        """Generate mock result."""
        class MockResult:
            final_text = ""
            extracted_code_blocks = []
        
        result = MockResult()
        # Return minimal OpenFOAM case
        result.final_text = '''```file:system/controlDict
FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application simpleFoam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime 100;
deltaT 1;
writeControl timeStep;
writeInterval 10;
```
'''
        return result
