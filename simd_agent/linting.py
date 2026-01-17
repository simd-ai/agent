# simd_agent/linting.py
"""CFD configuration linting and validation service."""

import logging
import math
from typing import Any

from simd_agent.models import (
    ApplyChange,
    FlowRegime,
    LintIssue,
    LintResult,
)
from simd_agent.event_bus import EventBus

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
    "icoFoam": {
        "type": "transient",
        "regime": ["laminar"],
        "description": "Transient laminar incompressible solver",
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

# Case type detection patterns
CASE_PATTERNS = {
    "pipe_flow": ["pipe", "duct", "channel", "tube"],
    "external_aero": ["external", "aerodynamic", "airfoil", "wing", "vehicle", "car", "cylinder"],
    "heat_transfer": ["heat", "thermal", "temperature", "convection", "cooling", "heating"],
    "cavity": ["cavity", "lid-driven", "enclosed"],
    "mixing": ["mixing", "mixer", "impeller", "stirred"],
    "porous": ["porous", "filter", "packed bed"],
}


class CFDLinter:
    """CFD configuration linting and validation service.
    
    Provides:
    - Units sanity checks
    - Reynolds number calculation and regime detection
    - Solver/turbulence model selection
    - Mesh resolution guidance
    - Boundary condition coherence checks
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
    ) -> LintResult:
        """Lint and validate a CFD configuration.
        
        Args:
            config: The simulation configuration to validate
            user_requirements: Optional user requirements text for context
            
        Returns:
            LintResult with validated config, changes, and issues
        """
        issues: list[LintIssue] = []
        apply_changes: list[ApplyChange] = []
        validated = dict(config)  # Start with a copy
        
        # Step 1: Detect case type
        case_type = self._detect_case_type(config, user_requirements)
        
        # Step 2: Validate and normalize units
        unit_issues, unit_changes, validated = self._validate_units(validated)
        issues.extend(unit_issues)
        apply_changes.extend(unit_changes)
        
        # Step 3: Calculate Reynolds number (for applicable flows)
        reynolds = self._calculate_reynolds(validated)
        
        # Step 4: Determine flow regime
        regime = self._determine_regime(reynolds)
        
        # Step 5: Select solver and turbulence model
        solver_changes, validated = self._select_solver_and_turbulence(
            validated, regime, case_type
        )
        apply_changes.extend(solver_changes)
        
        # Step 6: Mesh resolution guidance
        mesh_changes = self._recommend_mesh_resolution(validated, case_type, reynolds)
        apply_changes.extend(mesh_changes)
        
        # Step 7: Boundary condition coherence
        bc_issues, bc_changes = self._validate_boundary_conditions(validated, case_type)
        issues.extend(bc_issues)
        apply_changes.extend(bc_changes)
        
        # Step 8: Fill defaults for missing values
        default_changes, validated = self._fill_defaults(validated)
        apply_changes.extend(default_changes)
        
        return LintResult(
            validated_config=validated,
            apply_changes=apply_changes,
            issues=issues,
            detected_case_type=case_type,
            detected_regime=regime,
            selected_solver=validated.get("solver"),
            reynolds_number=reynolds,
        )
    
    def _detect_case_type(
        self,
        config: dict[str, Any],
        user_requirements: str,
    ) -> str | None:
        """Detect the simulation case type from config and requirements."""
        # Check explicit case_type in config
        if "case_type" in config:
            return config["case_type"]
        
        # Search for patterns in user requirements
        req_lower = user_requirements.lower()
        
        for case_type, patterns in CASE_PATTERNS.items():
            for pattern in patterns:
                if pattern in req_lower:
                    return case_type
        
        # Check geometry hints
        geometry = config.get("geometry", {})
        geo_type = geometry.get("type", "").lower()
        
        if geo_type in ["pipe", "tube", "cylinder"]:
            return "pipe_flow"
        if geo_type in ["airfoil", "wing", "vehicle"]:
            return "external_aero"
        
        return None
    
    def _validate_units(
        self,
        config: dict[str, Any],
    ) -> tuple[list[LintIssue], list[ApplyChange], dict[str, Any]]:
        """Validate and normalize units in the configuration."""
        issues = []
        changes = []
        validated = dict(config)
        
        # Geometry validations
        geometry = validated.get("geometry", {})
        
        # Diameter/length must be positive
        for dim in ["diameter", "length", "width", "height", "radius"]:
            if dim in geometry:
                val = geometry[dim]
                if isinstance(val, (int, float)) and val <= 0:
                    issues.append(LintIssue(
                        code="INVALID_DIMENSION",
                        path=f"geometry.{dim}",
                        message=f"{dim} must be positive, got {val}",
                        severity="error",
                    ))
        
        # Viscosity must be positive
        viscosity = validated.get("viscosity") or validated.get("fluid", {}).get("viscosity")
        if viscosity is not None:
            if isinstance(viscosity, (int, float)) and viscosity <= 0:
                issues.append(LintIssue(
                    code="INVALID_VISCOSITY",
                    path="viscosity",
                    message=f"Kinematic viscosity must be positive, got {viscosity}",
                    severity="error",
                ))
        
        # Velocity checks
        velocity = validated.get("inlet", {}).get("velocity") or validated.get("velocity")
        if velocity is not None:
            # Velocity can be negative for direction, but magnitude should be reasonable
            if isinstance(velocity, (int, float)):
                if abs(velocity) > 1000:  # Supersonic check
                    issues.append(LintIssue(
                        code="HIGH_VELOCITY",
                        path="velocity",
                        message=f"Velocity {velocity} m/s may be supersonic; ensure incompressible assumption is valid",
                        severity="warning",
                    ))
            elif isinstance(velocity, (list, tuple)):
                # Vector velocity
                magnitude = math.sqrt(sum(v**2 for v in velocity))
                if magnitude > 1000:
                    issues.append(LintIssue(
                        code="HIGH_VELOCITY",
                        path="velocity",
                        message=f"Velocity magnitude {magnitude:.1f} m/s may be supersonic",
                        severity="warning",
                    ))
        
        # Density check
        density = validated.get("density") or validated.get("fluid", {}).get("density")
        if density is not None and isinstance(density, (int, float)) and density <= 0:
            issues.append(LintIssue(
                code="INVALID_DENSITY",
                path="density",
                message=f"Density must be positive, got {density}",
                severity="error",
            ))
        
        return issues, changes, validated
    
    def _calculate_reynolds(self, config: dict[str, Any]) -> float | None:
        """Calculate Reynolds number if sufficient parameters are available."""
        # Get characteristic velocity
        velocity = None
        inlet = config.get("inlet", {})
        if "velocity" in inlet:
            v = inlet["velocity"]
            if isinstance(v, (int, float)):
                velocity = abs(v)
            elif isinstance(v, (list, tuple)):
                velocity = math.sqrt(sum(x**2 for x in v))
        elif "velocity" in config:
            v = config["velocity"]
            if isinstance(v, (int, float)):
                velocity = abs(v)
            elif isinstance(v, (list, tuple)):
                velocity = math.sqrt(sum(x**2 for x in v))
        
        # Get characteristic length
        geometry = config.get("geometry", {})
        length = None
        
        # For pipe flow, use diameter
        if "diameter" in geometry:
            length = geometry["diameter"]
        elif "radius" in geometry:
            length = geometry["radius"] * 2
        # For external flows, use chord or length
        elif "chord" in geometry:
            length = geometry["chord"]
        elif "length" in geometry:
            length = geometry["length"]
        
        # Get viscosity
        viscosity = (
            config.get("viscosity")
            or config.get("fluid", {}).get("viscosity")
            or DEFAULTS["viscosity"]
        )
        
        if velocity is not None and length is not None and viscosity > 0:
            re = (velocity * length) / viscosity
            return re
        
        return None
    
    def _determine_regime(self, reynolds: float | None) -> FlowRegime | None:
        """Determine flow regime from Reynolds number."""
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
        config: dict[str, Any],
        regime: FlowRegime | None,
        case_type: str | None,
    ) -> tuple[list[ApplyChange], dict[str, Any]]:
        """Select appropriate solver and turbulence model."""
        changes = []
        validated = dict(config)
        
        current_solver = validated.get("solver")
        current_turb = validated.get("turbulence_model")
        
        # Determine if we need heat transfer
        is_thermal = case_type == "heat_transfer" or "temperature" in validated.get("boundary_conditions", {})
        
        # Determine solver based on regime and case type
        if regime == FlowRegime.LAMINAR:
            recommended_solver = "simpleFoam"
            recommended_turb = "laminar"
            
            if is_thermal:
                recommended_solver = "buoyantSimpleFoam"
            
            # Check if current solver is compatible
            if current_solver and current_solver in SOLVERS:
                solver_info = SOLVERS[current_solver]
                if "laminar" not in solver_info["regime"]:
                    changes.append(ApplyChange(
                        path="solver",
                        value=recommended_solver,
                        reason=f"Laminar flow (Re={validated.get('reynolds_number', '?'):.0f}); {current_solver} not suitable",
                        severity="warning",
                    ))
                    validated["solver"] = recommended_solver
            else:
                validated["solver"] = recommended_solver
                if not current_solver:
                    changes.append(ApplyChange(
                        path="solver",
                        value=recommended_solver,
                        reason=f"Laminar flow detected, using {recommended_solver}",
                        severity="info",
                    ))
            
            # Force laminar turbulence model
            if current_turb and current_turb != "laminar":
                changes.append(ApplyChange(
                    path="turbulence_model",
                    value="laminar",
                    reason="Laminar flow; turbulence modeling not needed",
                    severity="info",
                ))
            validated["turbulence_model"] = "laminar"
        
        elif regime == FlowRegime.TRANSITIONAL:
            # Transitional - recommend turbulent treatment with warning
            recommended_solver = "simpleFoam"
            recommended_turb = "kOmegaSST"  # Better for transitional
            
            if is_thermal:
                recommended_solver = "buoyantSimpleFoam"
            
            validated["solver"] = validated.get("solver") or recommended_solver
            validated["turbulence_model"] = validated.get("turbulence_model") or recommended_turb
            
            if not current_turb:
                changes.append(ApplyChange(
                    path="turbulence_model",
                    value=recommended_turb,
                    reason="Transitional flow; k-omega SST recommended for better accuracy",
                    severity="info",
                ))
        
        elif regime == FlowRegime.TURBULENT:
            recommended_solver = "simpleFoam"
            recommended_turb = "kEpsilon"
            
            if is_thermal:
                recommended_solver = "buoyantSimpleFoam"
            
            validated["solver"] = validated.get("solver") or recommended_solver
            
            # If user specified laminar turbulence for turbulent flow, warn
            if current_turb == "laminar":
                changes.append(ApplyChange(
                    path="turbulence_model",
                    value=recommended_turb,
                    reason=f"Turbulent flow detected; turbulence model required",
                    severity="warning",
                ))
                validated["turbulence_model"] = recommended_turb
            else:
                validated["turbulence_model"] = current_turb or recommended_turb
                if not current_turb:
                    changes.append(ApplyChange(
                        path="turbulence_model",
                        value=recommended_turb,
                        reason="Turbulent flow; k-epsilon is a robust default",
                        severity="info",
                    ))
        
        else:
            # Unknown regime - use defaults
            validated["solver"] = validated.get("solver") or "simpleFoam"
            validated["turbulence_model"] = validated.get("turbulence_model") or "kEpsilon"
        
        return changes, validated
    
    def _recommend_mesh_resolution(
        self,
        config: dict[str, Any],
        case_type: str | None,
        reynolds: float | None,
    ) -> list[ApplyChange]:
        """Recommend mesh resolution based on case type and Reynolds number."""
        changes = []
        geometry = config.get("geometry", {})
        
        # Get characteristic length for cell recommendations
        char_length = None
        if "diameter" in geometry:
            char_length = geometry["diameter"]
        elif "length" in geometry:
            char_length = geometry["length"]
        
        if char_length is None:
            return changes
        
        # Recommendations based on regime
        if reynolds is not None:
            if reynolds < 2300:
                # Laminar - can use coarser mesh
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
            
            current_mesh = config.get("mesh", {})
            current_cells = current_mesh.get("cells_across_diameter")
            
            if current_cells is None or current_cells < min_cells_across:
                changes.append(ApplyChange(
                    path="mesh.cells_across_diameter",
                    value=min_cells_across,
                    reason=f"{message}; recommend at least {min_cells_across} cells across diameter",
                    severity="info",
                ))
        
        return changes
    
    def _validate_boundary_conditions(
        self,
        config: dict[str, Any],
        case_type: str | None,
    ) -> tuple[list[LintIssue], list[ApplyChange]]:
        """Validate boundary condition coherence."""
        issues = []
        changes = []
        
        bcs = config.get("boundary_conditions", {})
        
        # Check for required boundaries
        has_inlet = any("inlet" in k.lower() for k in bcs.keys()) or "inlet" in config
        has_outlet = any("outlet" in k.lower() for k in bcs.keys()) or "outlet" in config
        has_wall = any("wall" in k.lower() for k in bcs.keys()) or "walls" in config
        
        if not has_inlet and case_type in ["pipe_flow", "external_aero"]:
            issues.append(LintIssue(
                code="MISSING_INLET",
                path="boundary_conditions",
                message="No inlet boundary condition defined",
                severity="warning",
            ))
        
        if not has_outlet and case_type in ["pipe_flow", "external_aero"]:
            issues.append(LintIssue(
                code="MISSING_OUTLET",
                path="boundary_conditions",
                message="No outlet boundary condition defined",
                severity="warning",
            ))
        
        if not has_wall:
            issues.append(LintIssue(
                code="MISSING_WALL",
                path="boundary_conditions",
                message="No wall boundary condition defined",
                severity="warning",
            ))
        
        return issues, changes
    
    def _fill_defaults(
        self,
        config: dict[str, Any],
    ) -> tuple[list[ApplyChange], dict[str, Any]]:
        """Fill in default values for missing configuration."""
        changes = []
        validated = dict(config)
        
        for key, default_value in DEFAULTS.items():
            if key not in validated:
                validated[key] = default_value
                changes.append(ApplyChange(
                    path=key,
                    value=default_value,
                    reason=f"Default value applied for {key}",
                    severity="info",
                ))
        
        return changes, validated
