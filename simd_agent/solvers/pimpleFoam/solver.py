# simd_agent/solvers/pimpleFoam/solver.py
"""pimpleFoam solver plugin.

Transient, incompressible, isothermal RANS solver using the PIMPLE
algorithm (merged PISO-SIMPLE).  The transient counterpart of simpleFoam.
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


class PimpleFoamSolver(SolverPlugin):
    """pimpleFoam — transient incompressible isothermal RANS."""

    name = "pimpleFoam"
    algorithm = "PIMPLE"
    pressure_field = "p"
    is_transient = True
    is_compressible = False
    supports_energy = False
    needs_gravity = False
    is_multiphase = False

    def matches(self, config: dict[str, Any]) -> MatchResult:
        physics = config.get("physics", {}) or {}
        heat = bool(config.get("heat_transfer") or physics.get("heat_transfer"))
        compressible = (config.get("compressibility") or physics.get("compressibility", "incompressible")) == "compressible"
        transient = (config.get("time_stepping") or physics.get("time_scheme", "steady")) in ("transient", "unsteady")
        multiphase = bool(config.get("multiphase") or physics.get("multiphase"))

        if heat or compressible or multiphase:
            return MatchResult(0.0, "pimpleFoam cannot handle heat/compressible/multiphase")
        if not transient:
            return MatchResult(0.1, "pimpleFoam is transient; simpleFoam better for steady")
        return MatchResult(0.95, "Transient incompressible isothermal flow — ideal for pimpleFoam")

    def required_files(self, config: dict[str, Any]) -> list[str]:
        turb_model = self._get_turb_model_from_config(config)
        files = [
            "system/controlDict",
            # system/fvSchemes and system/fvSolution are generated
            # deterministically in validate(), not by the LLM.
            "constant/transportProperties", "constant/turbulenceProperties",
            "0/U", "0/p",
        ]
        for f in self.turbulence_fields(turb_model):
            files.append(f"0/{f}")
        return files

    def validate(self, files: dict[str, str], config: dict[str, Any]) -> ValidationResult:
        issues: list[ValidationIssue] = []
        fixed = dict(files)

        # Deterministic fvSolution + fvSchemes
        fixed["system/fvSolution"] = self._build_fv_solution(config)
        fixed["system/fvSchemes"] = self._build_fv_schemes(config)

        fixed = self._fix_controldict_solver(fixed, issues)
        fixed = self._fix_pressure_field(fixed, issues)
        fixed = self._fix_pressure_value(fixed, issues)
        fixed = self._remove_unneeded_thermo(fixed, issues)
        for ef in ["0/T", "0/h", "0/e"]:
            if ef in fixed:
                issues.append(ValidationIssue("warning", ef, f"pimpleFoam does not solve energy. Removing {ef}."))
                del fixed[ef]
        return ValidationResult(files=fixed, issues=issues)

