# simd_agent/linting.py
"""CFD configuration linting and validation service."""

import logging
import math
from typing import Any

from simd_agent.run.config_normalizer import (
    normalize_config,
    validate_config_completeness,
    get_config_summary,
)
from simd_agent.models import (
    ApplyChange,
    BoundaryType,
    FlowRegime,
    LintIssue,
    LintResult,
    MissingFieldInfo,
    SimulationConfigV1,
    TimeScheme,
)
from simd_agent.run.event_bus import EventBus

logger = logging.getLogger(__name__)


# Default values for common CFD parameters
DEFAULTS = {
    "solver": "simpleFoam",
    "turbulence_model": "kEpsilon",
    "mesh_resolution": "medium",
    "time_stepping": "steady",
    "write_interval": 100,
    "end_time": 1000,
    "viscosity": 1e-6,  # water at ~20°C
    "density": 1000.0,
}

# Supported solvers and their characteristics
SOLVERS = {
    "simpleFoam": {
        "type": "steady",
        "regime": ["laminar", "turbulent"],
        "description": "Steady-state incompressible solver",
    },
    "pimpleFoam": {
        "type": "transient",
        "regime": ["laminar", "turbulent"],
        "description": "Transient incompressible solver",
    },
    "pisoFoam": {
        "type": "transient",
        "regime": ["laminar", "turbulent"],
        "description": "Transient incompressible solver (PISO)",
    },
    "buoyantSimpleFoam": {
        "type": "steady",
        "regime": ["laminar", "turbulent"],
        "description": "Steady buoyant flow solver",
    },
    "buoyantPimpleFoam": {
        "type": "transient",
        "regime": ["laminar", "turbulent"],
        "description": "Transient buoyant flow solver",
    },
}

# Turbulence models
TURBULENCE_MODELS = {
    "laminar": {"regime": "laminar", "description": "No turbulence modeling"},
    "kEpsilon": {"regime": "turbulent", "description": "Standard k-epsilon"},
    "kOmegaSST": {"regime": "turbulent", "description": "k-omega SST"},
    "realizableKE": {"regime": "turbulent", "description": "Realizable k-epsilon"},
    "SpalartAllmaras": {"regime": "turbulent", "description": "Spalart-Allmaras one-equation"},
}

# Case type detection patterns (ordered by priority - more specific first)
CASE_PATTERNS = [
    ("heat_transfer", ["heated", "thermal", "temperature", "convection", "cooling", "heating", "heat transfer"]),
    ("cavity", ["cavity", "lid-driven", "lid driven", "enclosed"]),
    ("mixing", ["mixing", "mixer", "impeller", "stirred"]),
    ("porous", ["porous", "filter", "packed bed"]),
    ("external_aero", ["external flow", "aerodynamic", "airfoil", "aircraft", "vehicle", "bluff body", "around a"]),
    ("pipe_flow", ["pipe", "duct", "channel", "tube", "cylinder", "internal flow"]),
]


class CFDLinter:
    """CFD configuration linting and validation service.
    
    Provides:
    - Units sanity checks
    - Reynolds number calculation and regime detection
    - Solver/turbulence model selection
    - Mesh resolution guidance
    - Boundary condition coherence checks
    - Missing field detection
    """
    
    def __init__(self, event_bus: EventBus | None = None, use_llm: bool = False):
        """Initialize the linter.
        
        Args:
            event_bus: Optional event bus for progress updates
            use_llm: Whether to use LLM for ambiguous cases
        """
        self.event_bus = event_bus
        self.use_llm = use_llm
    
    async def lint(
        self,
        config: dict[str, Any],
        user_requirements: str = "",
        operation: str = "CFD_LINT",
    ) -> LintResult:
        """Lint and validate a CFD configuration.
        
        Args:
            config: The simulation configuration to validate (raw format)
            user_requirements: Optional user requirements text for context
            operation: The operation being performed (affects validation strictness)
            
        Returns:
            LintResult with validated config, changes, issues, and missing fields
        """
        issues: list[LintIssue] = []
        apply_changes: list[ApplyChange] = []
        missing_fields: list[MissingFieldInfo] = []
        
        # Step 1: Normalize the configuration
        logger.info("[LINT] Normalizing configuration...")
        normalized, detected_format, transformations = normalize_config(config)
        logger.info(f"[LINT] Detected format: {detected_format}")
        logger.info(f"[LINT] Transformations: {transformations}")
        
        # Emit config received event
        if self.event_bus:
            config_summary = get_config_summary(normalized)
            await self.event_bus.emit_config_received(
                config_keys=list(config.keys()),
                has_mesh=config_summary["has_mesh"],
                has_boundary_conditions=len(normalized.boundary_conditions) > 0,
                mesh_patches=config_summary["mesh_patches"],
                bc_patches=list(normalized.boundary_conditions.keys()),
            )
            
            await self.event_bus.emit_config_normalized(
                original_format=detected_format,
                normalized_keys=list(config_summary.keys()),
                transformations=transformations,
            )
        
        # Step 2: Validate completeness
        logger.info("[LINT] Validating completeness...")
        validation = validate_config_completeness(normalized, operation)
        missing_fields.extend(validation.missing_fields)
        
        if missing_fields and self.event_bus:
            await self.event_bus.emit_config_incomplete(
                missing_fields=[m.model_dump() for m in missing_fields],
                suggestions=[
                    {"field": m.field, "value": m.suggested_value}
                    for m in missing_fields if m.suggested_value
                ],
                can_lint=True,
                can_codegen=validation.is_complete,
            )
        
        # Step 3: Detect case type
        case_type = self._detect_case_type(normalized, user_requirements)
        logger.info(f"[LINT] Detected case type: {case_type}")
        
        # Step 4: Validate units and values
        unit_issues, unit_changes = self._validate_units(normalized)
        issues.extend(unit_issues)
        apply_changes.extend(unit_changes)
        
        # Step 5: Calculate Reynolds number
        reynolds = self._calculate_reynolds(normalized)
        logger.info(f"[LINT] Calculated Reynolds number: {reynolds}")
        
        # Step 6: Determine flow regime
        regime = self._determine_regime(reynolds, normalized)
        logger.info(f"[LINT] Determined regime: {regime}")
        
        # Step 7: Select solver and turbulence model
        solver_changes, solver, turb_model = self._select_solver_and_turbulence(
            normalized, regime, case_type
        )
        apply_changes.extend(solver_changes)
        
        # Update normalized config with selected values
        if solver:
            normalized.solver.type = solver
        if turb_model:
            normalized.physics.turbulence_model = turb_model
        if regime:
            normalized.physics.flow_regime = regime
        
        # Step 8: Validate boundary conditions against mesh
        bc_issues, bc_changes = self._validate_boundary_conditions(normalized, case_type)
        issues.extend(bc_issues)
        apply_changes.extend(bc_changes)
        
        # Step 9: Mesh resolution guidance
        mesh_changes = self._recommend_mesh_resolution(normalized, case_type, reynolds)
        apply_changes.extend(mesh_changes)
        
        # Step 10: Fill defaults for missing values
        default_changes = self._fill_defaults(normalized)
        apply_changes.extend(default_changes)
        
        # Build validated config dict (for backward compatibility)
        validated_config = self._build_validated_config(normalized, solver, turb_model)
        
        # Determine if config is complete for codegen
        is_complete = len(missing_fields) == 0
        
        # Log summary
        logger.info(f"[LINT] Lint complete:")
        logger.info(f"[LINT]   - Issues: {len(issues)}")
        logger.info(f"[LINT]   - Recommendations: {len(apply_changes)}")
        logger.info(f"[LINT]   - Missing fields: {len(missing_fields)}")
        logger.info(f"[LINT]   - Is complete: {is_complete}")
        
        return LintResult(
            validated_config=validated_config,
            normalized_config=normalized,
            apply_changes=apply_changes,
            issues=issues,
            missing_fields=missing_fields,
            detected_case_type=case_type,
            detected_regime=regime,
            selected_solver=solver or normalized.solver.type,
            reynolds_number=reynolds,
            is_complete=is_complete,
        )
    
    def _detect_case_type(
        self,
        config: SimulationConfigV1,
        user_requirements: str,
    ) -> str | None:
        """Detect the simulation case type from config and requirements."""
        # Check explicit case_type
        if config.case_type:
            return config.case_type
        
        # Search for patterns in user requirements (ordered by priority)
        req_lower = user_requirements.lower()
        
        for case_type, patterns in CASE_PATTERNS:
            for pattern in patterns:
                if pattern in req_lower:
                    return case_type
        
        # Check geometry type
        if config.geometry:
            geo_type = (config.geometry.type or "").lower()
            if geo_type in ["pipe", "tube", "cylinder"]:
                return "pipe_flow"
            if geo_type in ["airfoil", "wing", "vehicle"]:
                return "external_aero"
        
        return None
    
    def _validate_units(
        self,
        config: SimulationConfigV1,
    ) -> tuple[list[LintIssue], list[ApplyChange]]:
        """Validate and check units/values in the configuration."""
        issues = []
        changes = []
        
        # Check geometry dimensions
        if config.geometry:
            for dim_name in ["diameter", "length", "width", "height", "radius", "chord"]:
                val = getattr(config.geometry, dim_name, None)
                if val is not None and val <= 0:
                    issues.append(LintIssue(
                        code="INVALID_DIMENSION",
                        path=f"geometry.{dim_name}",
                        message=f"{dim_name} must be positive, got {val}",
                        severity="error",
                    ))
        
        # Check fluid properties
        if config.fluid.kinematic_viscosity <= 0:
            issues.append(LintIssue(
                code="INVALID_VISCOSITY",
                path="fluid.kinematic_viscosity",
                message=f"Kinematic viscosity must be positive, got {config.fluid.kinematic_viscosity}",
                severity="error",
            ))
        
        if config.fluid.density <= 0:
            issues.append(LintIssue(
                code="INVALID_DENSITY",
                path="fluid.density",
                message=f"Density must be positive, got {config.fluid.density}",
                severity="error",
            ))
        
        # Check velocity magnitude (compressibility warning)
        inlet_vel = config.get_inlet_velocity_magnitude()
        if inlet_vel and inlet_vel > 100:  # > ~Mach 0.3
            issues.append(LintIssue(
                code="HIGH_VELOCITY",
                path="boundary_conditions.inlet.velocity",
                message=f"Velocity {inlet_vel:.1f} m/s is high; compressibility effects may be significant (Mach > 0.3)",
                severity="warning",
            ))
        
        # Check temperature values (if heat transfer enabled)
        if config.physics.heat_transfer:
            for patch_name, bc in config.boundary_conditions.items():
                if bc.temperature and bc.temperature.value:
                    temp = bc.temperature.value
                    if temp < 0:
                        issues.append(LintIssue(
                            code="INVALID_TEMPERATURE",
                            path=f"boundary_conditions.{patch_name}.temperature",
                            message=f"Temperature must be positive (in Kelvin), got {temp}",
                            severity="error",
                        ))
                    elif temp < 200 or temp > 2000:
                        issues.append(LintIssue(
                            code="UNUSUAL_TEMPERATURE",
                            path=f"boundary_conditions.{patch_name}.temperature",
                            message=f"Temperature {temp} K seems unusual for typical CFD",
                            severity="warning",
                        ))
        
        return issues, changes
    
    def _calculate_reynolds(self, config: SimulationConfigV1) -> float | None:
        """Calculate Reynolds number from the normalized configuration."""
        # Get velocity magnitude
        velocity = config.get_inlet_velocity_magnitude()
        if velocity is None:
            logger.debug("[LINT] Cannot calculate Re: no inlet velocity")
            return None
        
        # Get characteristic length
        char_length = config.get_characteristic_length()
        if char_length is None:
            logger.debug("[LINT] Cannot calculate Re: no characteristic length")
            return None
        
        # Get kinematic viscosity
        nu = config.get_kinematic_viscosity()
        if nu <= 0:
            logger.debug("[LINT] Cannot calculate Re: invalid viscosity")
            return None
        
        # Re = (U * L) / nu
        re = (velocity * char_length) / nu
        logger.info(f"[LINT] Re = ({velocity} * {char_length}) / {nu} = {re:.0f}")
        
        return re
    
    def _determine_regime(
        self,
        reynolds: float | None,
        config: SimulationConfigV1,
    ) -> FlowRegime | None:
        """Determine flow regime from Reynolds number or config."""
        # Check if explicitly set in physics
        if config.physics.flow_regime:
            return config.physics.flow_regime
        
        # Use Reynolds number
        if reynolds is None:
            return None
        
        if reynolds < 2300:
            return FlowRegime.LAMINAR
        elif reynolds < 4000:
            return FlowRegime.TRANSITIONAL
        else:
            return FlowRegime.TURBULENT
    
    def _select_solver_and_turbulence(
        self,
        config: SimulationConfigV1,
        regime: FlowRegime | None,
        case_type: str | None,
    ) -> tuple[list[ApplyChange], str | None, str | None]:
        """Select appropriate solver and turbulence model."""
        changes = []
        
        current_solver = config.solver.type
        current_turb = config.physics.turbulence_model
        is_transient = config.physics.time_scheme == TimeScheme.TRANSIENT
        is_thermal = config.physics.heat_transfer or case_type == "heat_transfer"
        
        # Determine recommended solver
        # Buoyancy-driven (natural convection): heat transfer + single-phase + incompressible
        if is_thermal:
            if is_transient:
                recommended_solver = "buoyantPimpleFoam"
            else:
                recommended_solver = "buoyantSimpleFoam"
        elif is_transient:
            recommended_solver = "pimpleFoam"
        else:
            recommended_solver = "simpleFoam"
        
        # Determine recommended turbulence model
        if regime == FlowRegime.LAMINAR:
            recommended_turb = "laminar"
        elif regime == FlowRegime.TRANSITIONAL:
            recommended_turb = "kOmegaSST"  # Better for transitional
        else:
            recommended_turb = "kEpsilon"  # Robust default
        
        # Check if current solver is compatible
        solver = current_solver
        turb_model = current_turb
        
        if regime == FlowRegime.LAMINAR:
            if current_solver and current_solver in SOLVERS:
                solver_info = SOLVERS[current_solver]
                if "laminar" not in solver_info["regime"]:
                    changes.append(ApplyChange(
                        path="solver.type",
                        value=recommended_solver,
                        reason=f"Current solver {current_solver} not suitable for laminar flow",
                        severity="warning",
                    ))
                    solver = recommended_solver
            else:
                solver = recommended_solver
                changes.append(ApplyChange(
                    path="solver.type",
                    value=recommended_solver,
                    reason="Laminar flow detected, using appropriate solver",
                    severity="info",
                ))
            
            # Force laminar turbulence model
            if current_turb and current_turb != "laminar":
                changes.append(ApplyChange(
                    path="physics.turbulence_model",
                    value="laminar",
                    reason="Laminar flow; turbulence modeling not needed",
                    severity="info",
                ))
            turb_model = "laminar"
        
        elif regime == FlowRegime.TURBULENT:
            if not solver:
                solver = recommended_solver
                changes.append(ApplyChange(
                    path="solver.type",
                    value=recommended_solver,
                    reason="Turbulent flow, using robust solver",
                    severity="info",
                ))
            
            if current_turb == "laminar":
                changes.append(ApplyChange(
                    path="physics.turbulence_model",
                    value=recommended_turb,
                    reason="Turbulent flow detected; turbulence model required",
                    severity="warning",
                ))
                turb_model = recommended_turb
            elif not current_turb:
                turb_model = recommended_turb
                changes.append(ApplyChange(
                    path="physics.turbulence_model",
                    value=recommended_turb,
                    reason="Turbulent flow; k-epsilon is a robust default",
                    severity="info",
                ))
            else:
                turb_model = current_turb
        
        else:
            # Unknown regime - use current or defaults
            solver = solver or recommended_solver
            turb_model = turb_model or recommended_turb
        
        return changes, solver, turb_model
    
    def _validate_boundary_conditions(
        self,
        config: SimulationConfigV1,
        case_type: str | None,
    ) -> tuple[list[LintIssue], list[ApplyChange]]:
        """Validate boundary conditions against mesh patches."""
        issues = []
        changes = []
        
        # Count BC types
        inlets = []
        outlets = []
        walls = []
        
        for name, bc in config.boundary_conditions.items():
            if bc.is_inlet():
                inlets.append(name)
            elif bc.is_outlet():
                outlets.append(name)
            elif bc.is_wall():
                walls.append(name)
        
        # Check for required BCs
        if not inlets and case_type in ["pipe_flow", "external_aero", None]:
            issues.append(LintIssue(
                code="MISSING_INLET",
                path="boundary_conditions",
                message="No inlet boundary condition defined",
                severity="warning",
            ))
        
        if not outlets and case_type in ["pipe_flow", "external_aero", None]:
            issues.append(LintIssue(
                code="MISSING_OUTLET",
                path="boundary_conditions",
                message="No outlet boundary condition defined",
                severity="warning",
            ))
        
        # Check inlet completeness
        for inlet_name in inlets:
            bc = config.boundary_conditions[inlet_name]
            if not bc.velocity:
                issues.append(LintIssue(
                    code="INLET_MISSING_VELOCITY",
                    path=f"boundary_conditions.{inlet_name}",
                    message=f"Inlet '{inlet_name}' has no velocity specified",
                    severity="error",
                ))
            elif bc.velocity.get_magnitude() is None:
                issues.append(LintIssue(
                    code="INLET_INVALID_VELOCITY",
                    path=f"boundary_conditions.{inlet_name}.velocity",
                    message=f"Inlet '{inlet_name}' velocity has no valid magnitude",
                    severity="error",
                ))
        
        # Check outlet completeness
        for outlet_name in outlets:
            bc = config.boundary_conditions[outlet_name]
            # Outlet typically needs pressure or zeroGradient
            if not bc.pressure and bc.velocity:
                # Having velocity on outlet is unusual
                issues.append(LintIssue(
                    code="OUTLET_HAS_VELOCITY",
                    path=f"boundary_conditions.{outlet_name}",
                    message=f"Outlet '{outlet_name}' has velocity BC; typically use pressure BC",
                    severity="warning",
                ))
        
        # Check mesh patch coverage
        if config.mesh and config.mesh.patches:
            mesh_patches = {p.name for p in config.mesh.patches}
            bc_patches = set(config.boundary_conditions.keys())
            
            # Patches in mesh but not in BCs
            uncovered = mesh_patches - bc_patches
            for patch in uncovered:
                # Skip common auto-generated patches
                if patch.lower() in ["defaultfaces", "frontandback", "front", "back"]:
                    continue
                issues.append(LintIssue(
                    code="PATCH_NO_BC",
                    path=f"boundary_conditions.{patch}",
                    message=f"Mesh patch '{patch}' has no boundary condition defined",
                    severity="warning",
                ))
            
            # BCs for non-existent patches
            extra_bcs = bc_patches - mesh_patches
            for patch in extra_bcs:
                if config.mesh.patches:  # Only warn if we have mesh info
                    issues.append(LintIssue(
                        code="BC_NO_PATCH",
                        path=f"boundary_conditions.{patch}",
                        message=f"Boundary condition defined for non-existent patch '{patch}'",
                        severity="warning",
                    ))
        
        # Suggest default BCs for missing required patches
        if not inlets:
            changes.append(ApplyChange(
                path="boundary_conditions.inlet",
                value={
                    "patch_type": "inlet",
                    "velocity": {"type": "fixedValue", "value": [1, 0, 0]},
                    "pressure": {"type": "zeroGradient"},
                },
                reason="Add inlet boundary condition with default velocity",
                severity="warning",
            ))
        
        if not outlets:
            changes.append(ApplyChange(
                path="boundary_conditions.outlet",
                value={
                    "patch_type": "outlet",
                    "velocity": {"type": "zeroGradient"},
                    "pressure": {"type": "fixedValue", "value": 0},
                },
                reason="Add outlet boundary condition with zero pressure",
                severity="warning",
            ))
        
        return issues, changes
    
    def _recommend_mesh_resolution(
        self,
        config: SimulationConfigV1,
        case_type: str | None,
        reynolds: float | None,
    ) -> list[ApplyChange]:
        """Recommend mesh resolution based on case type and Reynolds number."""
        changes = []
        
        char_length = config.get_characteristic_length()
        if char_length is None:
            return changes
        
        # Recommendations based on regime
        if reynolds is not None:
            if reynolds < 2300:
                min_cells_across = 10
                message = "Laminar flow allows coarser mesh"
            elif reynolds < 10000:
                min_cells_across = 20
                message = "Moderate Re - medium resolution recommended"
            elif reynolds < 100000:
                min_cells_across = 30
                message = "High Re - finer mesh for boundary layers"
            else:
                min_cells_across = 50
                message = "Very high Re - fine mesh with wall functions recommended"
            
            changes.append(ApplyChange(
                path="mesh.cells_across_diameter",
                value=min_cells_across,
                reason=f"{message}; recommend at least {min_cells_across} cells across characteristic length",
                severity="info",
            ))
        
        return changes
    
    def _fill_defaults(self, config: SimulationConfigV1) -> list[ApplyChange]:
        """Fill in default values for missing configuration."""
        changes = []
        
        # Solver defaults
        if not config.solver.type or config.solver.type == "simpleFoam":
            pass  # Already default
        
        if config.solver.max_iterations == 1000:
            pass  # Already default
        
        if config.solver.write_interval == 100:
            pass  # Already default
        
        return changes
    
    def _build_validated_config(
        self,
        config: SimulationConfigV1,
        solver: str | None,
        turb_model: str | None,
    ) -> dict[str, Any]:
        """Build validated config dict for backward compatibility."""
        # For steady-state solvers the relevant iteration count is
        # solver.max_iterations; solver.end_time is a transient-only field
        # (defaults to None).  We expose both so the codegen layer can pick
        # whichever is set and fall back gracefully.
        _max_iter = config.solver.max_iterations  # e.g. 500 from frontend
        _end_time = config.solver.end_time or _max_iter or 1000

        _is_laminar = (
            config.physics.flow_regime == FlowRegime.LAMINAR
            or (turb_model or "").lower() == "laminar"
            or (config.physics.turbulence_model or "").lower() == "laminar"
        )

        validated = {
            "solver": solver or config.solver.type,
            # For laminar flow the model is "laminar" (no turbulence equations).
            # For turbulent flow fall back to kEpsilon if nothing is specified.
            "turbulence_model": "laminar" if _is_laminar else (
                turb_model or config.physics.turbulence_model or "kEpsilon"
            ),
            "time_stepping": config.physics.time_scheme.value,
            "write_interval": config.solver.write_interval,
            # max_iterations: the value the user explicitly set (or default 1000)
            "max_iterations": _max_iter,
            # end_time: for transient sims this is the physical end-time; for
            # steady it mirrors max_iterations so the LLM always has a value.
            "end_time": _end_time,
            # delta_t — propagated from solver.delta_t (frontend key: "delta_t" or "deltaT")
            "delta_t": config.solver.delta_t,
            "viscosity": config.fluid.kinematic_viscosity,
            "density": config.fluid.density,
            "mesh_resolution": "medium",
            # ── Physics flags for solver selection ────────────────────────────
            # These fields are consumed by SolverSelector and the codegen LLM.
            "heat_transfer":   config.physics.heat_transfer,
            "gravity":         getattr(config.physics, "gravity", False),
            "multiphase":      getattr(config.physics, "multiphase", False),
            "phases":          list(getattr(config.physics, "phases", []) or []),
            "compressibility": config.physics.compressibility.value,
            "flow_regime": (
                config.physics.flow_regime.value
                if config.physics.flow_regime
                else "turbulent"
            ),
            # ── Full fluid properties dict (consumed by build_case_spec) ──────
            # build_case_spec reads validated_config["fluid"] for nu, rho, mu,
            # cp, Pr.  Without this dict those values are null and thermophysical
            # properties cannot be computed correctly.
            "fluid": {
                "name":                 config.fluid.name,
                "density":              config.fluid.density,
                "kinematic_viscosity":  config.fluid.kinematic_viscosity,
                "nu":                   config.fluid.kinematic_viscosity,
                "dynamic_viscosity":    config.fluid.dynamic_viscosity,
                "mu":                   config.fluid.dynamic_viscosity,
                "rho":                  config.fluid.density,
                "specific_heat":        config.fluid.specific_heat,
                "cp":                   config.fluid.specific_heat,
                "Cp":                   config.fluid.specific_heat,
                "thermal_conductivity": config.fluid.thermal_conductivity,
                "thermal_diffusivity":  config.fluid.thermal_diffusivity,
                "prandtl_number":       config.fluid.prandtl_number,
                "prandtl":              config.fluid.prandtl_number,
                "Pr":                   config.fluid.prandtl_number,
                # Reference / bulk temperature (useful for internalField initial conditions)
                "temperature":          config.fluid.temperature,
                "T":                    config.fluid.temperature,
            },
        }
        
        # Add mesh info — include full mesh data with patch types for codegen
        if config.mesh:
            validated["mesh_id"] = config.mesh.mesh_id
            _mesh_dict: dict[str, Any] = {
                "mesh_id": config.mesh.mesh_id,
                "file_name": config.mesh.file_name,
                "patches": [
                    {"name": p.name, "type": p.type, "n_faces": p.n_faces}
                    for p in config.mesh.patches
                ],
            }
            # Include checkMesh quality metrics — used by solver plugins to
            # decide GAMG vs PBiCGStab and non-ortho correctors.
            if config.mesh.check_mesh:
                _mesh_dict["check_mesh"] = {
                    "max_non_orthogonality": config.mesh.check_mesh.max_non_orthogonality,
                    "avg_non_orthogonality": config.mesh.check_mesh.avg_non_orthogonality,
                    "max_skewness": config.mesh.check_mesh.max_skewness,
                    "max_aspect_ratio": config.mesh.check_mesh.max_aspect_ratio,
                    "n_severe_non_ortho": config.mesh.check_mesh.n_severe_non_ortho,
                    "mesh_ok": config.mesh.check_mesh.mesh_ok,
                }
                logger.info(
                    f"[LINTER] checkMesh included in validated_config: "
                    f"non_ortho={config.mesh.check_mesh.max_non_orthogonality}, "
                    f"skew={config.mesh.check_mesh.max_skewness}, "
                    f"aspect={config.mesh.check_mesh.max_aspect_ratio}"
                )
            else:
                logger.info("[LINTER] No checkMesh data on config.mesh → solver will use 'unknown' tier (PBiCGStab)")
            validated["mesh"] = _mesh_dict
        
        # Add turbulence config — carries pre-computed k/omega/epsilon/nut initial values
        # so the codegen layer never has to guess turbulence field values.
        # For laminar flow no turbulence model is active — skip the turbulence block
        # entirely so the chat agent and codegen layer do not receive misleading values.
        if _is_laminar:
            validated["turbulence"] = {}  # empty signals "no turbulence model"
        elif config.turbulence:
            tc = config.turbulence
            validated["turbulence"] = {
                "model":              tc.model,
                "intensity":          tc.intensity,
                "length_scale":       tc.length_scale,
                "hydraulic_diameter": tc.hydraulic_diameter,
                "k":                  tc.k,
                "omega":              tc.omega,
                "epsilon":            tc.epsilon,
                "nut":                tc.nut,
                "wall_functions":     tc.wall_functions,
            }
        else:
            # Even without an explicit turbulence block, expose the model name
            # from physics so build_case_spec has a single lookup path.
            turb_m = turb_model or config.physics.turbulence_model
            if turb_m:
                validated["turbulence"] = {"model": turb_m, "wall_functions": True}

        # Add geometry
        if config.geometry:
            validated["geometry"] = {
                "type": config.geometry.type,
                "diameter": config.geometry.diameter,
                "length": config.geometry.length,
            }
        
        # Add boundary conditions in legacy format
        for name, bc in config.boundary_conditions.items():
            if bc.is_inlet() and bc.velocity:
                validated["inlet"] = {
                    "velocity": bc.velocity.get_velocity_vector() or bc.velocity.value,
                }
            elif bc.is_outlet() and bc.pressure:
                validated["outlet"] = {
                    "pressure": bc.pressure.value,
                }
        
        # Add boundary_conditions in full format
        validated["boundary_conditions"] = {}
        for name, bc in config.boundary_conditions.items():
            bc_dict: dict[str, Any] = {
                "patch_type": bc.patch_type.value if isinstance(bc.patch_type, BoundaryType) else str(bc.patch_type),
            }
            if bc.velocity:
                vel_dict: dict[str, Any] = {
                    "type": bc.velocity.type,
                    # For flowRateInletVelocity the velocity value is a placeholder [0,0,0].
                    # The actual flow rate goes in massFlowRate/volumetricFlowRate below.
                    "value": [0.0, 0.0, 0.0] if bc.velocity.is_flow_rate_inlet()
                             else (bc.velocity.get_velocity_vector() or bc.velocity.value),
                }
                # Preserve flow rate keys + optional rho/rhoInlet/extrapolateProfile
                fr = bc.velocity.get_flow_rate()
                if fr is not None:
                    key_name, key_val = fr
                    vel_dict[key_name] = key_val
                    entries_dict: dict[str, Any] = {key_name: key_val}
                    if bc.velocity.rho_field is not None:
                        vel_dict["rho"] = bc.velocity.rho_field
                        entries_dict["rho"] = bc.velocity.rho_field
                    if bc.velocity.rho_inlet is not None:
                        vel_dict["rhoInlet"] = bc.velocity.rho_inlet
                        entries_dict["rhoInlet"] = bc.velocity.rho_inlet
                    if bc.velocity.extrapolate_profile is not None:
                        vel_dict["extrapolateProfile"] = bc.velocity.extrapolate_profile
                        entries_dict["extrapolateProfile"] = bc.velocity.extrapolate_profile
                    vel_dict["entries"] = entries_dict
                bc_dict["velocity"] = vel_dict
            if bc.pressure:
                bc_dict["pressure"] = {
                    "type": bc.pressure.type,
                    "value": bc.pressure.value,
                }
            if bc.temperature:
                bc_dict["temperature"] = {
                    "type": bc.temperature.type,
                    "value": bc.temperature.value,
                }
            # ── Turbulence BCs (top-level fields k/omega/epsilon/nut/alphat) ──
            # These are stored directly on BoundaryConditionV1 and must be
            # propagated verbatim so build_case_spec can inject them into 0/* prompts.
            for turb_field in ("k", "epsilon", "omega", "nut", "alphat", "nuTilda"):
                val = getattr(bc, turb_field, None)
                if val is not None:
                    bc_dict[turb_field] = val
            # Also propagate nested TurbulenceBCV1 (older path, if used)
            if bc.turbulence:
                for turb_field in ("k", "epsilon", "omega", "nut"):
                    val = getattr(bc.turbulence, turb_field, None)
                    if val is not None and turb_field not in bc_dict:
                        bc_dict[turb_field] = val
            validated["boundary_conditions"][name] = bc_dict

        return validated
