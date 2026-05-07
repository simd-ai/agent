# simd_agent/solvers/buoyantPimpleFoam/solver.py
"""buoyantPimpleFoam solver plugin.

Transient, compressible, buoyancy-driven solver using PIMPLE.
Uses p_rgh and requires constant/g.
"""

from __future__ import annotations

import logging
from typing import Any

from simd_agent.solvers.base import (
    MatchResult,
    SolverPlugin,
    ValidationIssue,
    ValidationResult,
)

logger = logging.getLogger(__name__)


class BuoyantPimpleFoamSolver(SolverPlugin):
    """buoyantPimpleFoam — transient buoyant compressible RANS."""

    name = "buoyantPimpleFoam"
    algorithm = "PIMPLE"
    pressure_field = "p_rgh"
    is_transient = True
    is_compressible = True
    supports_energy = True
    needs_gravity = True
    is_multiphase = False

    def matches(self, config: dict[str, Any]) -> MatchResult:
        physics = config.get("physics", {}) or {}
        heat = bool(config.get("heat_transfer") or physics.get("heat_transfer"))
        transient = (config.get("time_stepping") or physics.get("time_scheme", "steady")) in ("transient", "unsteady")
        multiphase = bool(config.get("multiphase") or physics.get("multiphase"))
        buoyancy = bool(config.get("gravity") or physics.get("gravity"))

        if multiphase:
            return MatchResult(0.0, "buoyantPimpleFoam is single-phase only")
        if not transient:
            return MatchResult(0.2, "buoyantPimpleFoam is transient; buoyantSimpleFoam for steady")
        if buoyancy and heat:
            return MatchResult(0.95, "Transient buoyancy + heat — ideal for buoyantPimpleFoam")
        if buoyancy:
            return MatchResult(0.8, "Transient buoyancy-driven — buoyantPimpleFoam")
        return MatchResult(0.0, "No buoyancy — buoyantPimpleFoam not needed")

    def required_files(self, config: dict[str, Any]) -> list[str]:
        turb_model = self._get_turb_model_from_config(config)
        files = [
            "system/controlDict",
            # system/fvSchemes and system/fvSolution are generated
            # deterministically in validate(), not by the LLM.
            "system/fvOptions",
            "constant/thermophysicalProperties", "constant/turbulenceProperties",
            "constant/g",
            # 0/p is synthesised by _fix_pressure_field from 0/p_rgh, so the
            # LLM only needs to generate the solved field (0/p_rgh).
            "0/U", "0/p_rgh", "0/T",
        ]
        for f in self.turbulence_fields(turb_model):
            files.append(f"0/{f}")
        if turb_model not in ("laminar", "none", ""):
            files.append("0/alphat")
        return files

    def validate(self, files: dict[str, str], config: dict[str, Any]) -> ValidationResult:
        issues: list[ValidationIssue] = []
        fixed = dict(files)

        # Deterministic fvSolution + fvSchemes
        fixed["system/fvSolution"] = self._build_fv_solution(config)
        fixed["system/fvSchemes"] = self._build_fv_schemes(config)

        fixed = self._fix_controldict_solver(fixed, issues)
        fixed = self._fix_pressure_field(fixed, issues)
        fixed = self._fix_thermo_type_key(fixed, issues)
        fixed = self._ensure_gravity(fixed, issues)
        for ef in ("0/h", "0/e"):
            if ef in fixed:
                del fixed[ef]
                issues.append(ValidationIssue("warning", ef, f"Removed {ef}: thermo reads 0/T."))
        return ValidationResult(files=fixed, issues=issues)

