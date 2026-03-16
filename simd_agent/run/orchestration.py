# simd_agent/orchestration.py
"""Main orchestration logic for CFD workflows."""

import io
import json
import logging
import re
import zipfile
from pathlib import Path
from typing import Any
from uuid import UUID

from simd_agent.chat.db import upsert_simulation_config
from simd_agent.run.error_summarizer import ErrorSummarizer
from simd_agent.run.event_bus import EventBus
from simd_agent.run.linting import CFDLinter
from simd_agent.models import (
    BoundaryType,
    EventLevel,
    FinalResult,
    LintResult,
    Operation,
    RunStatus,
    StartRequest,
)
from simd_agent.run.packaging import (
    extract_file_blocks,
    package_from_llm_output,
    package_simulation_case,
    merge_files,
)
from simd_agent.run.planning import Planner
from simd_agent.run.genai_codegen import (
    GenAICodeGenerator,
    validate_generated_files,
    determine_solver,
)
from simd_agent.run.solver_selector import SolverSelector
from simd_agent.run.solver_docs import load_prompt_pack
from simd_agent.run.code_verifier import CodeVerifier
from simd_agent.run.simulation_server_client import (
    SimulationServerClient,
    SimulationServerError,
    SimRunMode,
    SimRunStatus,
    SimRunEvent,
)
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
        self._sim_server: SimulationServerClient | None = None
        self._error_summarizer: ErrorSummarizer | None = None
        self._code_generator: GenAICodeGenerator | None = None
        
        # State
        self._iteration = 0
        self._retries = 0
        self._current_files: dict[str, str] = {}
        self._lint_result: LintResult | None = None
        self._previous_errors: list[dict[str, Any]] = []
        self._selected_solver: str | None = None       # set by SolverSelector in phase 1
        self._solver_prompt_pack = None                 # set after solver selection
        # Incremental generation state
        self._accumulated_files: dict[str, str] = {}   # grows across retries
        self._missing_files_for_patch: list[str] = []  # set by verifier → drives patch prompt
        self._error_recovery_affected_files: list[str] = []  # set after sim failure → tells frontend which files will be fixed next iteration
    
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
    def sim_server(self) -> SimulationServerClient:
        """Get or create the simulation server client."""
        if self._sim_server is None:
            self._sim_server = SimulationServerClient()
        return self._sim_server
    
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
        """Initialize the code generator using Google GenAI."""
        try:
            # Use Google GenAI directly for code generation
            self._code_generator = GenAICodeGenerator(event_bus=self.event_bus)
            logger.info("[ORCHESTRATOR] Initialized GenAI code generator")
        except Exception as e:
            logger.error(f"[ORCHESTRATOR] Failed to initialize GenAI code generator: {e}")
            raise OrchestrationError(f"Failed to initialize code generator: {e}")
    
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
                return await self._run_codegen_and_simulate()
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
            if self._sim_server:
                await self._sim_server.close()
    
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
    
    async def _run_codegen_and_simulate(self) -> FinalResult:
        """Run the full code generation and simulation workflow with self-healing."""
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
        
        # ── Phase 1: LLM-assisted solver selection ───────────────────────
        await self.event_bus.emit(
            "solver_selection_started",
            message="Selecting the best solver based on simulation physics…",
            payload={"iteration": self._iteration},
        )
        try:
            selector = SolverSelector()
            solver = await selector.select(
                user_requirements=self.request.user_requirements,
                simulation_config=self.request.simulation_config,
                validated_config=self._lint_result.validated_config,
            )
        except Exception as _sel_exc:
            logger.warning(f"[CODEGEN] SolverSelector failed ({_sel_exc}), using heuristic fallback")
            solver = determine_solver(self._lint_result.validated_config)

        self._selected_solver = solver
        logger.info(f"[CODEGEN] Solver selected: {solver}")

        # Load the prompt pack for the selected solver so it can be emitted to the
        # frontend and stored for later API reads without re-hitting disk.
        self._solver_prompt_pack = load_prompt_pack(
            solver,
            self._lint_result.validated_config,
        )
        logger.info(
            f"[CODEGEN] Prompt pack loaded: files={self._solver_prompt_pack.prompt_files} "
            f"required_case_files={self._solver_prompt_pack.required_case_files}"
        )

        await self.event_bus.emit(
            "solver_selected",
            message=f"Solver selected: {solver}",
            payload={
                "solver": solver,
                "iteration": self._iteration,
                "heat_transfer": self._lint_result.validated_config.get("heat_transfer", False),
                "time_stepping":  self._lint_result.validated_config.get("time_stepping", "steady"),
                "compressibility": self._lint_result.validated_config.get("compressibility", "incompressible"),
                "prompt_files": self._solver_prompt_pack.prompt_files,
                "required_case_files": self._solver_prompt_pack.required_case_files,
            },
        )

        # ── Emit simulation_config_ready and persist directly to DB ──────────
        # 1. Emit the WS event so the frontend can sync its local state.
        # 2. Also write directly to the simulation_config Neon table so the
        #    chat service can read the config without depending on the frontend
        #    relaying the event.
        _vc = self._lint_result.validated_config or {}
        _nc = self._lint_result.normalized_config  # SimulationConfigV1 | None

        # Build a rich physics dict from the normalized config (the flat
        # validated_config has no nested "physics" key).
        if _nc:
            _physics_section = _nc.physics.model_dump()
        else:
            _physics_section = {
                "flow_regime":    _vc.get("flow_regime", "turbulent"),
                "time_scheme":    _vc.get("time_stepping", "steady"),
                "compressibility": _vc.get("compressibility", "incompressible"),
                "heat_transfer":  _vc.get("heat_transfer", False),
                "gravity":        _vc.get("gravity", False),
                "multiphase":     _vc.get("multiphase", False),
            }

        # Build solver section — inject the authoritative solver name.
        _raw_solver = _vc.get("solver", {})
        _solver_section = dict(_raw_solver) if isinstance(_raw_solver, dict) else {}
        _solver_section["solver"] = solver

        # Turbulence: empty for laminar flow so the chat agent never reports a
        # turbulence model for laminar simulations.
        _is_laminar = (
            _vc.get("flow_regime", "turbulent") == "laminar"
            or (_nc and _nc.physics.flow_regime and
                _nc.physics.flow_regime.value == "laminar")
        )
        _turbulence_section = {} if _is_laminar else _vc.get("turbulence", {})

        _fluid_section = _vc.get("fluid", {})

        sim_id = (
            self.request.simulation_config.get("simulation_id")
            or self.request.simulation_config.get("simulationId")
        )

        await self.event_bus.emit_simulation_config_ready(
            physics=_physics_section,
            solver=_solver_section,
            fluid=_fluid_section,
            turbulence=_turbulence_section,
            simulation_id=sim_id,
        )

        # Persist directly to the simulation_config table so the chat service
        # has the config available immediately (no frontend relay required).
        if sim_id:
            await upsert_simulation_config(
                simulation_id=sim_id,
                physics=_physics_section,
                solver=_solver_section,
                fluid=_fluid_section,
                turbulence=_turbulence_section,
            )

        # Get mesh_id from simulation config.
        # simulation_config["mesh"] can be a dict or a bare mesh-ID string.
        mesh_raw = self.request.simulation_config.get("mesh", {})
        mesh_config = mesh_raw if isinstance(mesh_raw, dict) else {}
        mesh_id = (
            mesh_config.get("mesh_id")
            or mesh_config.get("meshId")
            or (mesh_raw if isinstance(mesh_raw, str) else None)
        )
        
        # Step 4: Self-healing loop
        while self._retries <= max_retries:
            self._iteration += 1
            
            try:
                # Snapshot before reset so emit_codegen_iteration can report it
                _current_patch_files = list(self._missing_files_for_patch)
                _current_affected_files = list(self._error_recovery_affected_files)
                self._error_recovery_affected_files = []  # consumed — reset immediately

                # ── On retry: tell frontend to clear stale sim_progress data ──
                # A new simulation attempt is about to start; the residuals from the
                # failed run must not mix with the new run's data.
                if self._iteration > 1:
                    await self.event_bus.emit_sim_progress_reset(self._iteration)

                # ── Generate code ──
                await self.event_bus.emit_codegen_started(
                    self._iteration,
                    patching_files=_current_patch_files or None,
                    affected_files=_current_affected_files or None,
                )

                code_result = await self._generate_code(
                    planning_result,
                    missing_files=self._missing_files_for_patch or None,
                )
                new_files = extract_file_blocks(code_result)

                # Drop any internal surgical sentinels that leaked through
                # (should not happen after the _generate_code fix, but belt+suspenders)
                new_files = {k: v for k, v in new_files.items() if not k.startswith("__surgical:")}

                if not new_files and not self._accumulated_files:
                    raise OrchestrationError("Code generation produced no files")

                # Merge newly-generated files into the accumulator.
                # New files ALWAYS overwrite stale ones from previous iterations.
                if new_files:
                    self._accumulated_files.update(new_files)
                files = dict(self._accumulated_files)

                # Reset patch list — will be repopulated if verifier finds gaps again
                self._missing_files_for_patch = []

                # Legacy compat: keep _current_files in sync
                self._current_files = files
                
                # ── Codegen debug report — show endTime from generated controlDict ──
                _ctrl = files.get("system/controlDict", "")
                _end_time_match = re.search(r"endTime\s+([\d.eE+\-]+)", _ctrl)
                _start_from_match = re.search(r"startFrom\s+(\S+?)\s*;", _ctrl)
                _start_time_match = re.search(r"startTime\s+([\d.eE+\-]+)", _ctrl)
                _vconfig = self._lint_result.validated_config if self._lint_result else {}
                print("=" * 70)
                print("🕐 CONTROLDICT / ENDTIME REPORT")
                print(f"  Iteration              : {self._iteration}")
                print(f"  validated_config.max_iterations: {_vconfig.get('max_iterations', '<not set>')}")
                print(f"  validated_config.end_time      : {_vconfig.get('end_time', '<not set>')}")
                print(f"  LLM-generated endTime  : {_end_time_match.group(1) if _end_time_match else '⚠️  NOT FOUND in controlDict'}")
                print(f"  LLM-generated startFrom: {_start_from_match.group(1) if _start_from_match else '⚠️  NOT FOUND'}")
                print(f"  LLM-generated startTime: {_start_time_match.group(1) if _start_time_match else '⚠️  NOT FOUND'}")
                print("=" * 70)

                # ── Print all generated file contents for debugging ──
                print("\n" + "=" * 70)
                print("📄 GENERATED FILE CONTENTS")
                print("=" * 70)
                for _fpath in sorted(files.keys()):
                    _fcontent = files[_fpath]
                    print(f"\n{'─'*60}")
                    print(f"📁 {_fpath}  ({len(_fcontent)} chars)")
                    print(f"{'─'*60}")
                    # Print first 80 lines max to avoid flooding logs
                    _lines = _fcontent.splitlines()
                    for _ln in _lines:
                        print(f"  {_ln}")
                print("\n" + "=" * 70)

                # ── Post-generation validation ──
                logger.info(f"[CODEGEN] Validating {len(files)} generated files...")
                files, validation_issues = validate_generated_files(
                    files, solver, self._lint_result.validated_config
                )
                
                validation_errors = [i for i in validation_issues if i.severity == "error"]
                if validation_errors:
                    logger.warning(f"[CODEGEN] Validation found {len(validation_errors)} errors (auto-fixed where possible)")
                    for vi in validation_errors:
                        logger.warning(f"[CODEGEN]   {vi}")
                
                self._current_files = files

                # ── Super-model verification (quality gate) ──────────────────
                # Uses gemini_super_model to independently check consistency:
                # solver vs physics, patch coverage, endTime, field completeness.
                # Critical issues feed back into the self-healing loop.
                # Warnings are surfaced as events but don't block submission.
                await self.event_bus.emit(
                    "codegen_verification_started",
                    message=f"Verifying generated case (iteration {self._iteration})",
                    payload={"iteration": self._iteration, "file_count": len(files)},
                )
                try:
                    verifier = CodeVerifier()
                    verification = await verifier.verify(
                        files=files,
                        user_requirements=self.request.user_requirements,
                        validated_config=self._lint_result.validated_config,
                        solver=solver,
                    )
                except Exception as _ve:
                    logger.warning(f"[VERIFY] Verifier failed (non-fatal): {_ve}")
                    verification = None

                _ver_payload: dict[str, Any] = {
                    "iteration": self._iteration,
                    "passed": True,
                    "issues": [],
                    "summary": "Verification skipped (verifier unavailable)",
                }
                if verification:
                    _ver_payload = {
                        "iteration": self._iteration,
                        "passed": verification.passed,
                        "summary": verification.summary,
                        "issues": [
                            {
                                "severity": i.severity,
                                "category": i.category,
                                "message": i.message,
                                "fix_suggestion": i.fix_suggestion,
                            }
                            for i in verification.issues
                        ],
                    }
                    print("=" * 70)
                    print("🔍 CODE VERIFICATION REPORT")
                    print(f"  Iteration : {self._iteration}")
                    print(f"  Passed    : {verification.passed}")
                    print(f"  Summary   : {verification.summary}")
                    for _vi in verification.issues:
                        _icon = "🔴" if _vi.severity == "critical" else ("⚠️ " if _vi.severity == "warning" else "ℹ️ ")
                        print(f"  {_icon} [{_vi.category}] {_vi.message}")
                        if _vi.fix_suggestion:
                            print(f"      → {_vi.fix_suggestion}")
                    print("=" * 70)

                await self.event_bus.emit(
                    "codegen_verification_complete",
                    message=_ver_payload["summary"],
                    level=EventLevel.ERROR if not _ver_payload["passed"] else EventLevel.INFO,
                    payload=_ver_payload,
                )

                # If critical issues were found, raise so the self-healing loop
                # regenerates with the verification findings as error context.
                if verification and not verification.passed:
                    critical_issues = [i for i in verification.issues if i.severity == "critical"]
                    critical_msgs = "\n".join(
                        f"- [{i.category}] {i.message}"
                        + (f"\n  Fix: {i.fix_suggestion}" if i.fix_suggestion else "")
                        for i in critical_issues
                    )

                    # ── Collect missing files for incremental patch generation ──
                    # Categories that map to a specific missing file path:
                    _MISSING_FILE_CATEGORIES = {
                        "missing_system_dicts",
                        "missing_fields",
                        "missing_constant_dicts",
                        "missing_p_rgh",
                        "missing_gravity",
                    }
                    _missing: list[str] = []
                    _non_missing_critical: list = []
                    for ci in critical_issues:
                        if ci.category in _MISSING_FILE_CATEGORIES and ci.fix_suggestion:
                            # Extract the file path from the fix suggestion or message
                            # e.g. "Generate 'system/controlDict' with …" → system/controlDict
                            path_match = re.search(r"['\"`]([^'\"` ]+/[^'\"` ]+)['\"`]", ci.message)
                            if path_match:
                                _missing.append(path_match.group(1))
                            else:
                                _non_missing_critical.append(ci)
                        else:
                            _non_missing_critical.append(ci)

                    if _missing:
                        # Deduplicate while preserving order
                        seen: set[str] = set()
                        self._missing_files_for_patch = [
                            f for f in _missing if not (f in seen or seen.add(f))  # type: ignore[func-returns-value]
                        ]
                        logger.info(
                            f"[CODEGEN] Missing files identified — patch generation: "
                            f"{self._missing_files_for_patch}"
                        )

                    raise OrchestrationError(
                        f"Code verification found critical issues:\n{critical_msgs}"
                    )

                await self.event_bus.emit_codegen_iteration(
                    self._iteration,
                    list(files.keys()),
                    patching_files=_current_patch_files or None,
                    affected_files=_current_affected_files or None,
                )
                
                # ── Package case with mesh ──
                # Final safety: strip any internal sentinel keys before packaging
                files = {k: v for k, v in files.items() if not k.startswith("__surgical:")}
                if mesh_id:
                    logger.info(f"[CODEGEN] Packaging with mesh: {mesh_id}")
                    zip_bytes, file_list, warnings = package_simulation_case(
                        generated_files=files,
                        mesh_id=mesh_id,
                        solver=solver,
                    )
                else:
                    logger.info("[CODEGEN] No mesh_id provided, using generated mesh")
                    zip_bytes, file_list, warnings = package_from_llm_output(
                        code_result,
                        solver=solver,
                    )

                # ── ZIP debug report ─────────────────────────────────────────
                with zipfile.ZipFile(io.BytesIO(zip_bytes)) as _zf:
                    _zip_names = _zf.namelist()
                _has_msh    = any(n.endswith((".msh", ".cas", ".cgns", ".unv", ".neu")) for n in _zip_names)
                _has_poly   = any("polyMesh" in n for n in _zip_names)
                _has_block  = any(n.endswith("blockMeshDict") for n in _zip_names)
                _mesh_in_zip = _has_msh or _has_poly or _has_block

                _raw_mesh_val = self.request.simulation_config.get("mesh", "<not set>")
                print("=" * 70)
                print("📦 ZIP PACKAGE REPORT")
                print(f"  mesh_id extracted : {mesh_id!r}")
                print(f"  simulation_config['mesh'] raw value: {_raw_mesh_val!r}")
                print(f"  ZIP size          : {len(zip_bytes):,} bytes")
                print(f"  ZIP file count    : {len(_zip_names)}")
                print(f"  Files in ZIP      : {_zip_names}")
                print(f"  Has .msh/.cas/etc : {_has_msh}")
                print(f"  Has polyMesh/     : {_has_poly}")
                print(f"  Has blockMeshDict : {_has_block}")
                print(f"  ✅ Mesh present    : {_mesh_in_zip}")
                print("=" * 70)

                # ── Fail-fast if mesh_id was provided but mesh is missing ───
                # package_simulation_case silently drops the mesh when storage
                # lookup fails.  Catch this here so we don't waste all 3 retry
                # slots submitting a ZIP that the sim server will reject with 400.
                if mesh_id and not _mesh_in_zip:
                    raise OrchestrationError(
                        f"Mesh '{mesh_id}' not found in storage — the ZIP contains "
                        "no mesh source (no .msh file, no polyMesh/, no blockMeshDict). "
                        "Check that the mesh was uploaded successfully and the mesh_id "
                        "sent by the frontend matches the upload response's 'meshId'."
                    )

                await self.event_bus.emit_codegen_complete(
                    self._iteration,
                    len(zip_bytes),
                )
                
                # ── Submit to simulation server (TEST mode — 1 iteration) ──
                # TEST mode is used here for fast validation during the self-healing
                # loop; the server patches controlDict to endTime=1 so errors are
                # caught cheaply.  On success we immediately run the full simulation.
                sim_result = await self._run_on_sim_server(
                    zip_bytes,
                    mode=SimRunMode.TEST,
                    n_cores=1,   # always serial for test/dry-run
                )

                import json as _json
                print("\n" + "=" * 70)
                print(f"[SimServer] TEST run result (iteration {self._iteration})")
                print("=" * 70)
                print(_json.dumps(sim_result, indent=2, default=str))
                print("=" * 70 + "\n")

                if sim_result["success"]:
                    # ── TEST passed → automatically run the FULL simulation ──
                    # The generated controlDict already has the correct endTime
                    # (from validated_config.solver.max_iterations).  FULL mode
                    # lets the solver run to that endTime without any patching.
                    logger.info(
                        f"[CODEGEN] Test validation passed (iter {self._iteration}). "
                        "Starting full simulation run..."
                    )
                    await self.event_bus.emit(
                        "full_run_started",
                        message="Test validation passed — starting full simulation",
                        payload={"solver": solver, "iteration": self._iteration},
                    )

                    full_result = await self._run_on_sim_server(
                        zip_bytes,
                        mode=SimRunMode.FULL,
                        n_cores=12,
                    )

                    print("\n" + "=" * 70)
                    print(f"[SimServer] FULL run result (iteration {self._iteration})")
                    print("=" * 70)
                    print(_json.dumps(full_result, indent=2, default=str))
                    print("=" * 70 + "\n")

                    artifacts = full_result.get("artifacts", [])

                    if full_result["success"]:
                        await self.event_bus.emit_run_succeeded(
                            f"Simulation completed after {self._iteration} codegen iteration(s)",
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
                                # Store sim_run_id so the VTK endpoint can retrieve it
                                "sim_run_id": full_result.get("sim_run_id"),
                            },
                            attempts=self._iteration,
                        )

                        await self.event_bus.emit_final(
                            status="succeeded",
                            validated_config=self._lint_result.validated_config,
                            artifacts=artifacts,
                            iterations=self._iteration,
                            retries=self._retries,
                            summary="Simulation completed successfully",
                            case_type=case_type,
                            solver=solver,
                        )

                        return FinalResult(
                            status=RunStatus.SUCCEEDED,
                            validated_config=self._lint_result.validated_config,
                            artifacts=artifacts,
                            iterations=self._iteration,
                            retries=self._retries,
                            case_type=case_type,
                            solver=solver,
                        )
                    else:
                        # Full run failed even though test passed.
                        # Treat it like a normal simulation failure and let
                        # the self-healing loop try to fix it.
                        logger.warning(
                            "[CODEGEN] Full simulation failed after test passed: "
                            f"{full_result.get('error')}"
                        )
                        # Fall through to the failure-handling block below
                        sim_result = full_result
                
                # ── Simulation failed — prepare for retry ──
                sim_error = sim_result.get("error", "Unknown error")
                sim_logs = sim_result.get("logs", "")
                sim_exit_code = sim_result.get("exit_code")
                sim_stderr = sim_result.get("stderr", "")
                
                # ── RAW PRINT of the simulation error for visibility ──
                print("\n" + "=" * 70)
                print("🔴 SIMULATION SERVER ERROR (will attempt self-healing)")
                print("=" * 70)
                print(f"  Iteration: {self._iteration}")
                print(f"  Exit code: {sim_exit_code}")
                print(f"  Error: {sim_error}")
                if sim_stderr:
                    print(f"\n  --- STDERR / OpenFOAM Error ---")
                    for line in sim_stderr.split("\n")[-30:]:  # Last 30 lines
                        print(f"  {line}")
                elif sim_logs:
                    # If no stderr collected, show last part of logs
                    print(f"\n  --- Last log lines ---")
                    for line in sim_logs.split("\n")[-20:]:
                        print(f"  {line}")
                print("=" * 70)
                print(f"  Files sent: {list(self._current_files.keys())}")
                print("=" * 70 + "\n")
                
                if self._retries >= max_retries:
                    break
                
                self._retries += 1
                
                logger.warning(f"[CODEGEN] Simulation failed (attempt {self._retries}/{max_retries}): {sim_error}")
                
                # Store error for context in next LLM call.
                # The new simulation server emits solver output via the
                # run_failed payload's "stderr" key rather than run_log events,
                # so sim_logs may be empty.  Fall back to sim_stderr so the LLM
                # always gets the actual OpenFOAM error text.
                _err_entry = {
                    "iteration": self._iteration,
                    "source": "simulation_server",
                    "error": sim_error,
                    "details": sim_logs[-3000:] if sim_logs else sim_stderr[-3000:],
                    "stderr": sim_stderr[:5000] if sim_stderr else "",
                    "exit_code": sim_exit_code,
                }
                self._previous_errors.append(_err_entry)

                # ── LLM error diagnosis (replaces pattern-match-only summarizer) ──
                # Use the Gemini model to diagnose the error with full context:
                # all generated files + error output.  The diagnosis is fed back
                # into the next generation call as error context.
                try:
                    _diag = await self.error_summarizer.summarize(
                        logs=sim_stderr or sim_logs or sim_error,
                        exit_code=sim_exit_code,
                        current_files=self._current_files,
                    )
                    if _diag.root_cause and _diag.confidence > 0.3:
                        _err_entry["llm_diagnosis"] = _diag.root_cause
                        _err_entry["llm_fixes"] = [
                            f["description"] for f in (_diag.actionable_changes or [])
                        ]
                        _err_entry["llm_affected_files"] = _diag.affected_files or []
                        logger.info(
                            f"[DIAGNOSIS] root_cause={_diag.root_cause!r} "
                            f"confidence={_diag.confidence:.2f} "
                            f"affected={_diag.affected_files}"
                        )
                        await self.event_bus.emit_error_summary(
                            _diag.root_cause,
                            _diag.actionable_changes or [],
                            _diag.affected_files or [],
                        )
                except Exception as _de:
                    logger.warning(f"[DIAGNOSIS] Error summarizer failed (non-fatal): {_de}")

                # ── Surgical fixes — apply in-place without LLM regen ─────────
                # Detect error patterns that have known deterministic fixes.
                # Applied BEFORE deciding what files to regenerate.
                try:
                    from simd_agent.run.genai_codegen import GenAICodeGenerator as _GCG
                    _affected_hint = _GCG._identify_affected_files(
                        self._previous_errors,
                        self._current_files,
                        solver,
                        self._lint_result.validated_config,
                    )
                    _patched_files, _applied_fixes = _GCG.apply_surgical_fixes(
                        dict(self._accumulated_files),
                        _affected_hint,
                    )
                    if _applied_fixes:
                        _pre_patch = dict(self._accumulated_files)
                        self._accumulated_files.update(_patched_files)
                        for _fix in _applied_fixes:
                            logger.info(f"[SURGICAL] Applied: {_fix}")
                        await self.event_bus.emit(
                            "surgical_fix_applied",
                            message=f"Applied surgical fix: {_applied_fixes[0]}",
                            payload={"fixes": _applied_fixes},
                        )
                        # Emit file events for every file the surgical patch changed
                        # so the frontend reflects the updated content.
                        for _fp, _fc in _patched_files.items():
                            if _fc != _pre_patch.get(_fp):
                                await self.event_bus.emit_file_generated(
                                    _fp, _fc, self._iteration, len(_fc),
                                    mode="surgical",
                                )
                    # Store the real affected files (strip sentinels) so the next
                    # iteration's codegen_started can tell the frontend upfront
                    # which files will be regenerated.
                    self._error_recovery_affected_files = [
                        f for f in _affected_hint if not f.startswith("__surgical:")
                    ]
                except Exception as _sfe:
                    logger.warning(f"[SURGICAL] Surgical fix step failed (non-fatal): {_sfe}")

                await self.event_bus.emit_retrying(self._retries, max_retries)
                
            except SimulationServerError as e:
                print("\n" + "=" * 70)
                print("🔴 SIMULATION SERVER CONNECTION ERROR")
                print("=" * 70)
                print(f"  Iteration: {self._iteration}")
                print(f"  Error: {e}")
                print("=" * 70 + "\n")
                
                logger.error(f"Simulation server error: {e}")
                if self._retries >= max_retries:
                    break
                self._retries += 1
                
                self._previous_errors.append({
                    "iteration": self._iteration,
                    "source": "simulation_server_connection",
                    "error": str(e),
                    "details": str(e),
                })
                
                await self.event_bus.emit_retrying(self._retries, max_retries)
            
            except OrchestrationError as e:
                print("\n" + "=" * 70)
                print("🔴 ORCHESTRATION ERROR")
                print("=" * 70)
                print(f"  Iteration: {self._iteration}")
                print(f"  Error: {e}")
                print("=" * 70 + "\n")
                
                logger.error(f"Orchestration error: {e}")
                if self._retries >= max_retries:
                    break
                self._retries += 1
                
                self._previous_errors.append({
                    "iteration": self._iteration,
                    "source": "orchestrator",
                    "error": str(e),
                    "details": "",
                })
                
                await self.event_bus.emit_retrying(self._retries, max_retries)
        
        # Max retries exceeded
        last_error = self._previous_errors[-1]["error"] if self._previous_errors else "Unknown"
        await self.event_bus.emit_run_failed(
            f"Failed after {max_retries} retries. Last error: {last_error}"
        )
        
        await self.store.finalize_run(
            run_id=self.run_id,
            status=RunStatus.FAILED,
            validated_config=self._lint_result.validated_config if self._lint_result else None,
            result={
                "error": f"Max retries exceeded. Last: {last_error}",
                "iterations": self._iteration,
                "errors": self._previous_errors,
            },
            attempts=self._iteration,
        )
        
        await self.event_bus.emit_final(
            status="failed",
            validated_config=self._lint_result.validated_config if self._lint_result else None,
            iterations=self._iteration,
            retries=self._retries,
            summary=f"Failed after {max_retries} retries",
            error=last_error,
        )
        
        return FinalResult(
            status=RunStatus.FAILED,
            iterations=self._iteration,
            retries=self._retries,
            error=f"Max retries exceeded. Last: {last_error}",
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
    
    async def _generate_code(
        self,
        planning_result: Any,
        missing_files: list[str] | None = None,
    ) -> str:
        """Generate OpenFOAM case code using Google GenAI directly.

        Args:
            planning_result: Output from the planning phase.
            missing_files: If provided, use a patch prompt to generate ONLY these
                           files instead of the full codegen or fix prompt.
        """
        # Build enhanced requirements with planning context
        requirements = self._build_requirements(planning_result)

        # Get validated config
        validated_config = {}
        if self._lint_result and self._lint_result.validated_config:
            validated_config = self._lint_result.validated_config

        # Use the solver already selected by SolverSelector in phase 1.
        # Fall back to heuristic only if called before selection (shouldn't happen).
        solver = self._selected_solver or determine_solver(validated_config)
        case_type = planning_result.case_type or "pipe_flow"

        # ── Route to the right prompt ─────────────────────────────────────────
        # Priority: patch (missing files) > fix (sim errors) > full codegen
        previous_errors = None
        previous_files = self._accumulated_files if self._accumulated_files else None

        if not missing_files and self._previous_errors:
            previous_errors = self._previous_errors

        # Generate using GenAI
        result = await self._code_generator.generate(
            requirements=requirements,
            validated_config=validated_config,
            solver=solver,
            case_type=case_type,
            previous_errors=previous_errors,
            previous_files=previous_files,
            missing_files=missing_files,
            iteration=self._iteration,
        )

        return result
    
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
                    
                    # Empty/symmetry constraint patches — ALL fields must use this type
                    if patch_type == "empty":
                        parts.append(f"\n- **ALL fields** (U, p, T, k, omega, nut, epsilon): type=empty")
                        parts.append(f"\n  (This is a constraint patch — every field file MUST have `{patch_name} {{ type empty; }}`)")
                        continue
                    if patch_type == "symmetry":
                        parts.append(f"\n- **ALL fields**: type=symmetry")
                        continue
                    
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
    
    async def _run_on_sim_server(
        self,
        zip_bytes: bytes,
        mode: SimRunMode = SimRunMode.TEST,
        n_cores: int = 12,
    ) -> dict[str, Any]:
        """Submit case to simulation server and stream events.

        Events from the simulation server are relayed to the frontend via WebSocket.

        Args:
            zip_bytes: OpenFOAM case as ZIP bytes
            mode: TEST for 1 iteration validation, FULL for complete run

        Returns:
            Dict with success, sim_run_id, artifacts, logs, exit_code, etc.
        """
        logs_collected: list[str] = []
        stderr_collected: list[str] = []
        
        # Event callback to relay sim server events to frontend
        async def on_sim_event(event: SimRunEvent):
            logger.info(f"[SIM_SERVER] Event: {event.type} - {event.message}")
            
            # Collect logs — check both payload.line and message
            if event.type == "run_log":
                line = event.payload.get("line", event.message)
                logs_collected.append(line)
                # Collect stderr lines separately for error context
                if event.payload.get("stream") == "stderr" or event.level in ("error", "warn"):
                    stderr_collected.append(line)
                # Catch ALL OpenFOAM fatal errors (FOAM FATAL ERROR, FOAM FATAL IO ERROR, etc.)
                # NOTE: OpenFOAM writes errors and stack traces to STDOUT, not stderr.
                line_upper = line.upper()
                if any(pattern in line_upper for pattern in [
                    "FOAM FATAL", "FOAM EXITING", "CANNOT FIND FILE",
                    "NOT CONSTRAINT TYPE", "PATCH TYPE", "DIMENSIONS MISMATCH",
                    "UNKNOWN PATCHFIELD", "ENTRY NOT FOUND",
                    # FPE / crash signals
                    "FLOATING POINT", "SEGMENTATION FAULT", "SIGFPE", "SIGABRT",
                    "STACK TRACE", "[STACK TRACE]",
                    # Division by zero / bad fields
                    "DIVIDE", "OVERFLOW", "UNDERFLOW", "NAN", "INF",
                ]):
                    stderr_collected.append(line)
            
            # Also collect the message itself for non-log events with errors
            if event.level == "error" and event.type != "run_log":
                stderr_collected.append(f"[{event.type}] {event.message}")
            
            # Collect error info from failed events
            if event.type in ("run_failed", "mesh_conversion_failed", "blockmesh_failed", "checkmesh_failed"):
                stderr_text = event.payload.get("stderr", "")
                if stderr_text:
                    stderr_collected.append(stderr_text)
                # Always include stdout tail from run_failed — OpenFOAM writes
                # FATAL errors, FPE stack traces, and "Floating point exception"
                # to stdout, not stderr. runner.py now ships last 100 lines.
                stdout_text = event.payload.get("stdout", "")
                if stdout_text:
                    # Filter to include any line that looks like an error/trace
                    useful_lines = []
                    in_trace = False
                    for ln in stdout_text.splitlines():
                        lu = ln.upper()
                        if any(p in lu for p in [
                            "FOAM FATAL", "FOAM EXITING", "FOAM WARNING",
                            "FLOATING POINT", "SEGMENTATION", "STACK TRACE",
                            "[STACK TRACE]", "#1 ", "#2 ", "#3 ",
                            "DIVIDE", "OVERFLOW", "NAN", "INF",
                            "ERROR", "EXCEPTION", "CANNOT FIND",
                        ]):
                            in_trace = True
                        if in_trace:
                            useful_lines.append(ln)
                    if useful_lines:
                        stderr_collected.append("\n".join(useful_lines))
                    elif stdout_text.strip():
                        # No obvious error lines, but send last 50 lines anyway
                        stderr_collected.append("\n".join(stdout_text.splitlines()[-50:]))
            
            # Silently drop decompose/reconstruct events — frontend doesn't need them.
            # The simulation server handles MPI decomposition internally; we only
            # care about the actual solver progress and final result.
            _IGNORED_EVENT_TYPES = {
                "decompose_started", "decompose_complete", "decompose_failed",
                "decompose_log",
                "reconstruct_started", "reconstruct_complete", "reconstruct_failed",
                "reconstruct_log",
            }
            if event.type in _IGNORED_EVENT_TYPES:
                return

            # Skip verbose mesh_log events
            if event.type == "mesh_log" and not event.payload.get("important"):
                return

            # ── run_progress / run_progress_batch → sim_progress ────────────
            if event.type in ("run_progress", "run_progress_batch") and mode == SimRunMode.FULL:
                import json as _j
                raw_items: list[dict] = (
                    event.payload.get("items", [])
                    if event.type == "run_progress_batch"
                    else [event.payload]
                )

                def _build_sim_progress(p: dict) -> dict:
                    """Normalise one parsed time-step into the frontend contract."""
                    residuals_raw = p.get("residuals", {})

                    # Accept {field: scalar} (old) or {field: {initial,final,iters}} (new)
                    residuals: dict = {}
                    for field, val in residuals_raw.items():
                        if isinstance(val, dict):
                            residuals[field] = {
                                "initial": float(val.get("initial", 0)),
                                "final":   float(val.get("final", val.get("initial", 0))),
                                "iters":   int(val.get("iters", 1)),
                            }
                        else:
                            residuals[field] = {
                                "initial": float(val),
                                "final":   float(val),
                                "iters":   1,
                            }

                    courant_raw = p.get("courant")
                    courant = (
                        {"mean": float(courant_raw["mean"]), "max": float(courant_raw["max"])}
                        if isinstance(courant_raw, dict) else None
                    )

                    cont_raw = p.get("continuity")
                    continuity = (
                        {
                            "local":      float(cont_raw["local"]),
                            "global":     float(cont_raw["global"]),
                            "cumulative": float(cont_raw["cumulative"]),
                        }
                        if isinstance(cont_raw, dict) else None
                    )

                    exec_raw = p.get("execution")
                    execution = (
                        {
                            "stepSeconds":  float(exec_raw.get("stepSeconds", exec_raw.get("step_seconds", 0))),
                            "clockSeconds": float(exec_raw.get("clockSeconds", exec_raw.get("clock_seconds", 0))),
                            "label":        exec_raw.get("label", ""),
                        }
                        if isinstance(exec_raw, dict) else None
                    )

                    return {
                        "iteration":  int(p.get("iteration", 0)),
                        "simTime":    float(p.get("time", p.get("simTime", 0))),
                        "fields":     list(p.get("fields", list(residuals.keys()))),
                        "residuals":  residuals,
                        "courant":    courant,
                        "continuity": continuity,
                        "execution":  execution,
                    }

                built = [_build_sim_progress(p) for p in raw_items]
                await self.event_bus.emit_sim_event(
                    event_type="sim_progress_batch",
                    message=f"{len(built)} step(s)",
                    payload={"items": built},
                    level=event.level,
                )
                return

            # Map all other simulation server event types
            event_type_map = {
                "extract_started": "sim_extract_started",
                "extract_complete": "sim_extract_complete",
                "mesh_conversion_started": "mesh_conversion_started",
                "mesh_conversion_complete": "mesh_conversion_complete",
                "mesh_conversion_failed": "mesh_conversion_failed",
                "mesh_log": "mesh_log",
                "blockmesh_started": "blockmesh_started",
                "blockmesh_complete": "blockmesh_complete",
                "blockmesh_failed": "blockmesh_failed",
                "checkmesh_started": "checkmesh_started",
                "checkmesh_complete": "checkmesh_complete",
                "run_started": "sim_run_started",
                "run_log": "sim_run_log",
                "run_succeeded": "sim_run_succeeded",
                "run_failed": "sim_run_failed",
                "artifacts_ready": "sim_artifacts_ready",
                # Test-mode watchdog result (rc=-9 SIGKILL is expected, not an error)
                "run_test_passed": "sim_test_passed",
            }

            mapped_type = event_type_map.get(event.type, event.type)

            # Strip processor* files from artifacts_ready payload
            payload = event.payload
            if event.type == "artifacts_ready" and "artifacts" in payload:
                payload = {
                    **payload,
                    "artifacts": [
                        a for a in payload["artifacts"]
                        if not str(a.get("path", a.get("name", ""))).startswith("processor")
                    ],
                }

            await self.event_bus.emit_sim_event(
                event_type=mapped_type,
                message=event.message,
                payload=payload,
                level=event.level,
            )
        
        try:
            # Submit to simulation server
            submit_response = await self.sim_server.submit_run(
                zip_bytes,
                mode=mode,
                run_id=f"{self.run_id}-iter{self._iteration}",
                n_cores=n_cores,
            )
            sim_run_id = submit_response.run_id
            
            await self.event_bus.emit_sim_submitted(
                sim_run_id=sim_run_id,
                mode=mode.value,
                events_url=submit_response.events_url,
            )
            
            # Stream events and wait for completion
            all_events: list[SimRunEvent] = []
            test_passed_payload: dict | None = None   # set when run_test_passed arrives

            async for event in self.sim_server.stream_events(sim_run_id, on_event=on_sim_event):
                all_events.append(event)

                # ── Raw simulation data print ──────────────────────────────────
                if event.type in ("run_log", "run_progress", "run_progress_batch",
                                  "run_started", "run_succeeded", "run_failed",
                                  "run_test_passed"):
                    print("=" * 70)
                    print(f"[SIM RAW] type={event.type}  level={event.level}  seq={event.seq}")
                    print(f"[SIM RAW] message: {event.message}")
                    print("=" * 70)

                # ── Terminal conditions ────────────────────────────────────────
                if event.type == "run_test_passed":
                    # Watchdog confirmed ≥5 iterations; SIGKILL (rc=-9) is expected.
                    test_passed_payload = event.payload
                    break   # no artifacts_ready will follow — exit stream now

                if event.type == "artifacts_ready":
                    break   # full run completed normally

            logs = "\n".join(logs_collected)
            stderr = "\n".join(stderr_collected)

            # ── TEST mode: success is determined by run_test_passed, not rc ──
            if test_passed_payload is not None:
                iterations_seen = test_passed_payload.get("iterations_observed", 0)
                duration = test_passed_payload.get("duration_seconds", 0.0)
                print(f"[SIM] Test passed — {iterations_seen} iterations in {duration:.1f}s")
                return {
                    "success": True,
                    "test_passed": True,
                    "sim_run_id": sim_run_id,
                    "iterations_observed": iterations_seen,
                    "duration_seconds": duration,
                    "logs": logs,
                    "artifacts": [],
                }

            # ── FULL mode: check final status and collect artifacts ───────────
            final_status = await self.sim_server.get_status(sim_run_id)

            if final_status.status == SimRunStatus.SUCCEEDED:
                artifacts = await self.sim_server.get_artifacts(sim_run_id)
                artifacts_list = [
                    a.model_dump() for a in artifacts
                    if not (a.path or a.name or "").startswith("processor")
                ]
                print(f"[SIM] Succeeded — {len(artifacts_list)} artifacts")
                await self.event_bus.emit_sim_succeeded(
                    sim_run_id=sim_run_id,
                    duration_seconds=final_status.duration_seconds or 0.0,
                    artifacts=artifacts_list,
                )
                return {
                    "success": True,
                    "sim_run_id": sim_run_id,
                    "artifacts": artifacts_list,
                    "logs": logs,
                    "duration_seconds": final_status.duration_seconds,
                }
            else:
                error_msg = final_status.error or "Simulation failed"
                await self.event_bus.emit_sim_failed(
                    sim_run_id=sim_run_id,
                    error=error_msg,
                    exit_code=final_status.exit_code,
                )
                return {
                    "success": False,
                    "sim_run_id": sim_run_id,
                    "exit_code": final_status.exit_code,
                    "logs": logs,
                    "error": error_msg,
                    "stderr": stderr,
                }
                
        except SimulationServerError as e:
            logger.error(f"[SIM_SERVER] Error: {e}")
            await self.event_bus.emit_sim_failed(
                sim_run_id="unknown",
                error=str(e),
            )
            return {
                "success": False,
                "error": str(e),
                "logs": "\n".join(logs_collected),
                "stderr": "\n".join(stderr_collected),
            }



class OpenFOAMCaseGenerator:
    """Generate complete OpenFOAM case files from validated configuration.
    
    This generator creates proper OpenFOAM case structure without requiring
    the external codegen library. It uses the validated configuration from
    the linting/planning phases.
    """
    
    def generate(self, context: Any) -> Any:
        """Generate OpenFOAM case files.
        
        Args:
            context: GenerationContext with requirements and config
            
        Returns:
            Result object with final_text containing file blocks
        """
        class Result:
            final_text = ""
            extracted_code_blocks = []
        
        result = Result()
        
        # Extract configuration from context
        requirements = getattr(context, 'requirements', '')
        
        # Parse validated config from requirements (it's embedded as JSON)
        config = self._parse_config(requirements)
        
        # Generate all files
        files = []
        
        # Detect solver and settings
        solver = self._detect_solver(config)
        heat_transfer = config.get('physics', {}).get('heat_transfer', False)
        
        # System files
        files.append(self._generate_control_dict(config, solver))
        files.append(self._generate_fv_schemes(config, solver))
        files.append(self._generate_fv_solution(config, solver))
        
        # Constant files
        files.append(self._generate_transport_properties(config))
        files.append(self._generate_turbulence_properties(config))
        if heat_transfer:
            files.append(self._generate_thermophysical_properties(config))
        
        # Initial condition files (0 directory)
        files.append(self._generate_U(config))
        files.append(self._generate_p(config, solver))
        if heat_transfer:
            files.append(self._generate_T(config))
        
        # Turbulence fields
        turb_model = config.get('physics', {}).get('turbulence_model', 'kOmegaSST')
        if turb_model not in ['laminar', None]:
            if 'kOmega' in turb_model or 'SST' in turb_model:
                files.append(self._generate_k(config))
                files.append(self._generate_omega(config))
            elif 'kEpsilon' in turb_model:
                files.append(self._generate_k(config))
                files.append(self._generate_epsilon(config))
            files.append(self._generate_nut(config))
        
        result.final_text = "\n\n".join(files)
        return result
    
    def _parse_config(self, requirements: str) -> dict:
        """Extract validated config from requirements string."""
        import json
        import re
        
        # Try to find JSON block in requirements
        json_match = re.search(r'```json\s*(.*?)\s*```', requirements, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Return default config if not found
        return {
            'physics': {'heat_transfer': False, 'turbulence_model': 'kOmegaSST'},
            'solver': {'max_iterations': 1000},
            'boundary_conditions': {},
            'fluid': {'density': 1000, 'kinematic_viscosity': 1e-6},
        }
    
    def _detect_solver(self, config: dict) -> str:
        """Detect appropriate OpenFOAM solver (no buoyancy)."""
        physics = config.get('physics', {})
        time_scheme = physics.get('time_scheme', 'steady')
        
        # No buoyant solvers — simplify
        return 'simpleFoam' if time_scheme == 'steady' else 'pimpleFoam'
    
    def _foam_header(self, class_name: str, object_name: str, location: str = "") -> str:
        """Generate FoamFile header."""
        loc = f'    location    "{location}";\n' if location else ""
        return f'''FoamFile
{{
    version     2.0;
    format      ascii;
    class       {class_name};
{loc}    object      {object_name};
}}'''
    
    def _generate_control_dict(self, config: dict, solver: str) -> str:
        """Generate system/controlDict."""
        solver_config = config.get('solver', {})
        max_iter = solver_config.get('max_iterations', 1000)
        
        return f'''```file:system/controlDict
{self._foam_header("dictionary", "controlDict")}

application     {solver};
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {max_iter};
deltaT          1;
writeControl    timeStep;
writeInterval   1;
purgeWrite      0;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;

functions
{{
}}
```'''
    
    def _generate_fv_schemes(self, config: dict, solver: str) -> str:
        """Generate system/fvSchemes."""
        heat_transfer = config.get('physics', {}).get('heat_transfer', False)
        
        div_schemes = '''    div(phi,U)      bounded Gauss linearUpwind grad(U);
    div(phi,k)      bounded Gauss upwind;
    div(phi,omega)  bounded Gauss upwind;
    div(phi,epsilon) bounded Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;'''
        
        if heat_transfer:
            div_schemes += '''
    div(phi,h)      bounded Gauss upwind;
    div(phi,K)      bounded Gauss upwind;'''
        
        return f'''```file:system/fvSchemes
{self._foam_header("dictionary", "fvSchemes")}

ddtSchemes
{{
    default         steadyState;
}}

gradSchemes
{{
    default         Gauss linear;
    grad(U)         cellLimited Gauss linear 1;
}}

divSchemes
{{
    default         none;
{div_schemes}
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

wallDist
{{
    method meshWave;
}}
```'''
    
    def _generate_fv_solution(self, config: dict, solver: str) -> str:
        """Generate system/fvSolution."""
        solver_config = config.get('solver', {})
        convergence = solver_config.get('convergence_criteria', 1e-5)
        heat_transfer = config.get('physics', {}).get('heat_transfer', False)
        
        extra_solvers = ""
        extra_residuals = ""
        extra_relax = ""
        
        if heat_transfer:
            extra_solvers = '''
    h
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-8;
        relTol          0.01;
    }'''
            extra_residuals = f'''
        h               {convergence};'''
            extra_relax = '''
        h               0.7;'''
        
        return f'''```file:system/fvSolution
{self._foam_header("dictionary", "fvSolution")}

solvers
{{
    p
    {{
        solver          GAMG;
        tolerance       1e-8;
        relTol          0.01;
        smoother        GaussSeidel;
        nPreSweeps      0;
        nPostSweeps     2;
        cacheAgglomeration true;
        nCellsInCoarsestLevel 10;
        agglomerator    faceAreaPair;
        mergeLevels     1;
    }}

    "(U|k|omega|epsilon)"
    {{
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-8;
        relTol          0.01;
    }}{extra_solvers}
}}

SIMPLE
{{
    nNonOrthogonalCorrectors 0;
    consistent      yes;
    
    residualControl
    {{
        p               {convergence};
        U               {convergence};
        k               {convergence};
        omega           {convergence};
        epsilon         {convergence};{extra_residuals}
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
        k               0.7;
        omega           0.7;
        epsilon         0.7;{extra_relax}
    }}
}}
```'''
    
    def _generate_transport_properties(self, config: dict) -> str:
        """Generate constant/transportProperties."""
        fluid = config.get('fluid', {})
        nu = fluid.get('kinematic_viscosity', 1e-6)
        
        return f'''```file:constant/transportProperties
{self._foam_header("dictionary", "transportProperties")}

transportModel  Newtonian;
nu              [0 2 -1 0 0 0 0] {nu};
```'''
    
    def _generate_turbulence_properties(self, config: dict) -> str:
        """Generate constant/turbulenceProperties."""
        physics = config.get('physics', {})
        turb_model = physics.get('turbulence_model', 'kOmegaSST')
        
        if turb_model in ['laminar', None]:
            sim_type = "laminar"
            model_content = ""
        else:
            sim_type = "RAS"
            model_content = f'''
RAS
{{
    RASModel        {turb_model};
    turbulence      on;
    printCoeffs     on;
}}'''
        
        return f'''```file:constant/turbulenceProperties
{self._foam_header("dictionary", "turbulenceProperties")}

simulationType  {sim_type};{model_content}
```'''
    
    def _generate_thermophysical_properties(self, config: dict) -> str:
        """Generate constant/thermophysicalProperties for heat transfer."""
        from simd_agent.run.genai_codegen import _eos_for_liquid, _ico_poly_coeffs
        fluid = config.get('fluid', {})
        rho = fluid.get('density', 1000)
        Cp = fluid.get('specific_heat', 4182)
        k_thermal = fluid.get('thermal_conductivity', 0.6)
        mu = fluid.get('dynamic_viscosity', 1e-3)
        inlet_T = fluid.get('temperature')

        has_energy = config.get('enable_heat_transfer') or config.get('heat_transfer')
        eos = _eos_for_liquid(rho, inlet_T, bool(has_energy))

        if eos == "icoPolynomial":
            # icoPolynomial requires polynomial transport + hPolynomial thermo.
            # const+hConst+icoPolynomial → "Unknown fluidThermo type" FOAM error.
            a0, a1 = _ico_poly_coeffs(rho, inlet_T)
            kappa = Cp * mu / (k_thermal if k_thermal > 0 else Cp * mu / 0.7)
            return f'''```file:constant/thermophysicalProperties
{self._foam_header("dictionary", "thermophysicalProperties")}

thermoType
{{
    type            heRhoThermo;
    mixture         pureMixture;
    transport       polynomial;
    thermo          hPolynomial;
    equationOfState icoPolynomial;
    specie          specie;
    energy          sensibleEnthalpy;
}}

mixture
{{
    specie
    {{
        molWeight       28.9;
    }}
    thermodynamics
    {{
        Hf              0;
        Sf              0;
        CpCoeffs<8>     ({Cp} 0 0 0 0 0 0 0);
    }}
    transport
    {{
        muCoeffs<8>     ({mu} 0 0 0 0 0 0 0);
        kappaCoeffs<8>  ({k_thermal} 0 0 0 0 0 0 0);
    }}
    equationOfState
    {{
        rhoCoeffs<8>    ({a0:.3f} {a1:.4f} 0 0 0 0 0 0);
    }}
}}
```'''

        eos_block = f"    equationOfState\n    {{\n        rho             {rho};\n    }}"
        return f'''```file:constant/thermophysicalProperties
{self._foam_header("dictionary", "thermophysicalProperties")}

thermoType
{{
    type            heRhoThermo;
    mixture         pureMixture;
    transport       const;
    thermo          hConst;
    equationOfState {eos};
    specie          specie;
    energy          sensibleEnthalpy;
}}

mixture
{{
    specie
    {{
        molWeight       28.9;
    }}
{eos_block}
    thermodynamics
    {{
        Cp              {Cp};
        Hf              0;
    }}
    transport
    {{
        mu              {mu};
        Pr              {Cp * mu / k_thermal if k_thermal > 0 else 0.7};
    }}
}}
```'''
    
    def _generate_U(self, config: dict) -> str:
        """Generate 0/U velocity field."""
        bcs = config.get('boundary_conditions', {})
        
        bc_entries = []
        for patch_name, bc in bcs.items():
            vel = bc.get('velocity', {})
            vel_type = vel.get('type', 'zeroGradient')
            vel_value = vel.get('value')
            
            if vel_type == 'noSlip':
                bc_entries.append(f'''    {patch_name}
    {{
        type            noSlip;
    }}''')
            elif vel_type == 'fixedValue' and vel_value:
                if isinstance(vel_value, list) and len(vel_value) >= 3:
                    bc_entries.append(f'''    {patch_name}
    {{
        type            fixedValue;
        value           uniform ({vel_value[0]} {vel_value[1]} {vel_value[2]});
    }}''')
                else:
                    bc_entries.append(f'''    {patch_name}
    {{
        type            fixedValue;
        value           uniform ({vel_value} 0 0);
    }}''')
            else:
                bc_entries.append(f'''    {patch_name}
    {{
        type            zeroGradient;
    }}''')
        
        return f'''```file:0/U
{self._foam_header("volVectorField", "U", "0")}

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{{
{chr(10).join(bc_entries)}
}}
```'''
    
    def _generate_p(self, config: dict, solver: str) -> str:
        """Generate 0/p pressure field."""
        bcs = config.get('boundary_conditions', {})
        
        # Always use p (no buoyant solvers)
        field_name = "p"
        
        bc_entries = []
        for patch_name, bc in bcs.items():
            pressure = bc.get('pressure', {})
            p_type = pressure.get('type', 'zeroGradient')
            p_value = pressure.get('value', 0)
            
            if p_type == 'fixedValue':
                bc_entries.append(f'''    {patch_name}
    {{
        type            fixedValue;
        value           uniform {p_value};
    }}''')
            else:
                bc_entries.append(f'''    {patch_name}
    {{
        type            zeroGradient;
    }}''')
        
        return f'''```file:0/{field_name}
{self._foam_header("volScalarField", field_name, "0")}

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{{
{chr(10).join(bc_entries)}
}}
```'''
    
    def _generate_T(self, config: dict) -> str:
        """Generate 0/T temperature field."""
        bcs = config.get('boundary_conditions', {})
        fluid = config.get('fluid', {})
        T_ref = fluid.get('temperature', 300)
        
        bc_entries = []
        for patch_name, bc in bcs.items():
            temp = bc.get('temperature', {})
            T_type = temp.get('type', 'zeroGradient')
            T_value = temp.get('value', T_ref)
            
            if T_type == 'fixedValue':
                bc_entries.append(f'''    {patch_name}
    {{
        type            fixedValue;
        value           uniform {T_value};
    }}''')
            else:
                bc_entries.append(f'''    {patch_name}
    {{
        type            zeroGradient;
    }}''')
        
        return f'''```file:0/T
{self._foam_header("volScalarField", "T", "0")}

dimensions      [0 0 0 1 0 0 0];

internalField   uniform {T_ref};

boundaryField
{{
{chr(10).join(bc_entries)}
}}
```'''
    
    def _generate_k(self, config: dict) -> str:
        """Generate 0/k turbulent kinetic energy field."""
        bcs = config.get('boundary_conditions', {})
        
        bc_entries = []
        for patch_name, bc in bcs.items():
            k_bc = bc.get('k', {})
            k_type = k_bc.get('type', 'zeroGradient')
            k_value = k_bc.get('value', 0.1)
            
            if k_type == 'fixedValue':
                bc_entries.append(f'''    {patch_name}
    {{
        type            fixedValue;
        value           uniform {k_value};
    }}''')
            elif 'WallFunction' in k_type:
                bc_entries.append(f'''    {patch_name}
    {{
        type            {k_type};
        value           uniform {k_value};
    }}''')
            else:
                bc_entries.append(f'''    {patch_name}
    {{
        type            zeroGradient;
    }}''')
        
        return f'''```file:0/k
{self._foam_header("volScalarField", "k", "0")}

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform 0.1;

boundaryField
{{
{chr(10).join(bc_entries)}
}}
```'''
    
    def _generate_omega(self, config: dict) -> str:
        """Generate 0/omega specific dissipation rate field."""
        bcs = config.get('boundary_conditions', {})
        
        bc_entries = []
        for patch_name, bc in bcs.items():
            omega_bc = bc.get('omega', {})
            omega_type = omega_bc.get('type', 'zeroGradient')
            omega_value = omega_bc.get('value', 1)
            
            if omega_type == 'fixedValue':
                bc_entries.append(f'''    {patch_name}
    {{
        type            fixedValue;
        value           uniform {omega_value};
    }}''')
            elif 'WallFunction' in omega_type:
                bc_entries.append(f'''    {patch_name}
    {{
        type            {omega_type};
        value           uniform {omega_value};
    }}''')
            else:
                bc_entries.append(f'''    {patch_name}
    {{
        type            zeroGradient;
    }}''')
        
        return f'''```file:0/omega
{self._foam_header("volScalarField", "omega", "0")}

dimensions      [0 0 -1 0 0 0 0];

internalField   uniform 1;

boundaryField
{{
{chr(10).join(bc_entries)}
}}
```'''
    
    def _generate_epsilon(self, config: dict) -> str:
        """Generate 0/epsilon turbulent dissipation field."""
        bcs = config.get('boundary_conditions', {})
        
        bc_entries = []
        for patch_name, bc in bcs.items():
            eps_bc = bc.get('epsilon', {})
            eps_type = eps_bc.get('type', 'zeroGradient')
            eps_value = eps_bc.get('value', 0.01)
            
            if eps_type == 'fixedValue':
                bc_entries.append(f'''    {patch_name}
    {{
        type            fixedValue;
        value           uniform {eps_value};
    }}''')
            elif 'WallFunction' in eps_type:
                bc_entries.append(f'''    {patch_name}
    {{
        type            {eps_type};
        value           uniform {eps_value};
    }}''')
            else:
                bc_entries.append(f'''    {patch_name}
    {{
        type            zeroGradient;
    }}''')
        
        return f'''```file:0/epsilon
{self._foam_header("volScalarField", "epsilon", "0")}

dimensions      [0 2 -3 0 0 0 0];

internalField   uniform 0.01;

boundaryField
{{
{chr(10).join(bc_entries)}
}}
```'''
    
    def _generate_nut(self, config: dict) -> str:
        """Generate 0/nut turbulent viscosity field."""
        bcs = config.get('boundary_conditions', {})
        
        bc_entries = []
        for patch_name, bc in bcs.items():
            nut_bc = bc.get('nut', {})
            nut_type = nut_bc.get('type', 'calculated')
            nut_value = nut_bc.get('value', 0)
            
            if 'WallFunction' in nut_type:
                bc_entries.append(f'''    {patch_name}
    {{
        type            {nut_type};
        value           uniform {nut_value};
    }}''')
            else:
                bc_entries.append(f'''    {patch_name}
    {{
        type            calculated;
        value           uniform 0;
    }}''')
        
        return f'''```file:0/nut
{self._foam_header("volScalarField", "nut", "0")}

dimensions      [0 2 -1 0 0 0 0];

internalField   uniform 0;

boundaryField
{{
{chr(10).join(bc_entries)}
}}
```'''
