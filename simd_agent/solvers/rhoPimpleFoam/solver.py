# simd_agent/solvers/rhoPimpleFoam/solver.py
"""rhoPimpleFoam solver plugin.

Transient, compressible, single-phase solver using PIMPLE algorithm.
Solves the energy equation — used for transient heated/cryogenic flows.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from simd_agent.solvers.base import (
    MatchResult,
    SolverPlugin,
    ValidationIssue,
    ValidationResult,
)

logger = logging.getLogger(__name__)


class RhoPimpleFoamSolver(SolverPlugin):
    """rhoPimpleFoam — transient compressible energy RANS."""

    name = "rhoPimpleFoam"
    algorithm = "PIMPLE"
    pressure_field = "p"
    is_transient = True
    is_compressible = True
    supports_energy = True
    needs_gravity = False
    is_multiphase = False

    def matches(self, config: dict[str, Any]) -> MatchResult:
        physics = config.get("physics", {}) or {}
        heat = bool(config.get("heat_transfer") or physics.get("heat_transfer"))
        compressible = (config.get("compressibility") or physics.get("compressibility", "incompressible")) == "compressible"
        transient = (config.get("time_stepping") or physics.get("time_scheme", "steady")) in ("transient", "unsteady")
        multiphase = bool(config.get("multiphase") or physics.get("multiphase"))
        buoyancy = bool(config.get("gravity") or physics.get("gravity"))

        if multiphase:
            return MatchResult(0.0, "rhoPimpleFoam is single-phase only")
        if not transient:
            return MatchResult(0.2, "rhoPimpleFoam is transient; rhoSimpleFoam better for steady")
        if buoyancy:
            return MatchResult(0.3, "buoyantPimpleFoam better for buoyancy-driven transient flow")
        if compressible and heat:
            return MatchResult(0.95, "Transient compressible flow with heat — ideal for rhoPimpleFoam")
        if compressible:
            return MatchResult(0.85, "Transient compressible flow — rhoPimpleFoam is appropriate")
        if heat:
            return MatchResult(0.75, "Transient flow with heat — rhoPimpleFoam handles energy")
        return MatchResult(0.1, "No compressibility or heat — pimpleFoam would be simpler")

    def required_files(self, config: dict[str, Any]) -> list[str]:
        turb_model = self._get_turb_model_from_config(config)
        heat_transfer = bool(
            config.get("heat_transfer")
            or (config.get("physics", {}) or {}).get("heat_transfer")
        )
        files = [
            "system/controlDict",
            # system/fvSchemes and system/fvSolution are generated
            # deterministically in validate(), not by the LLM.
            "constant/thermophysicalProperties", "constant/turbulenceProperties",
            "0/U", "0/p", "0/T",
        ]
        if heat_transfer:
            files.append("system/fvOptions")
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

        # Fix alphat wall function namespace
        fixed = self._fix_alphat_wall(fixed, issues)
        # Remove energy field files
        for ef in ("0/h", "0/e"):
            if ef in fixed:
                del fixed[ef]
                issues.append(ValidationIssue("warning", ef, f"Removed {ef}: thermo reads 0/T."))

        # Enforce PIMPLE settings for isothermal case
        heat_transfer = bool(
            config.get("heat_transfer")
            or (config.get("physics", {}) or {}).get("heat_transfer")
        )
        if not heat_transfer:
            fixed = self._fix_isothermal_pimple(fixed, issues)

        return ValidationResult(files=fixed, issues=issues)


    def _fix_alphat_wall(self, files: dict[str, str], issues: list[ValidationIssue]) -> dict[str, str]:
        alphat = files.get("0/alphat", "")
        if alphat and "alphatWallFunction" in alphat and "compressible::alphatWallFunction" not in alphat:
            files["0/alphat"] = alphat.replace("alphatWallFunction", "compressible::alphatWallFunction")
            issues.append(ValidationIssue("warning", "0/alphat", "Fixed: alphatWallFunction -> compressible::alphatWallFunction."))
        return files

    def _fix_isothermal_pimple(self, files: dict[str, str], issues: list[ValidationIssue]) -> dict[str, str]:
        """For isothermal rhoPimpleFoam: PBiCGStab+DILU for p, nOuterCorrectors=2."""
        fvs = files.get("system/fvSolution", "")
        if not fvs:
            return files
        modified = False
        if "GAMG" in fvs and not bool(
            (files.get("system/fvSolution", "").count("GAMG") or 0)
            and "heat" in str(files.get("system/controlDict", "")).lower()
        ):
            fvs = re.sub(r"\bGAMG\b", "PBiCGStab", fvs)
            if "DILU" not in fvs:
                fvs = re.sub(r"preconditioner\s+\w+", "preconditioner  DILU", fvs)
            modified = True
            issues.append(ValidationIssue("warning", "system/fvSolution", "Isothermal: GAMG -> PBiCGStab+DILU for p."))
        if modified:
            files["system/fvSolution"] = fvs
        return files

