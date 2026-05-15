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
    ValidationIssue,
    ValidationResult,
)
from simd_agent.solvers.families import CompressibleMixin, TransientBase

logger = logging.getLogger(__name__)


class RhoPimpleFoamSolver(TransientBase, CompressibleMixin):
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
            "constant/thermophysicalProperties", "0/U", "0/p", "0/T",
        ]
        if heat_transfer:
            files.append("system/fvOptions")
        for f in self.turbulence_fields(turb_model):
            if f == "nut":
                continue  # 0/nut rendered deterministically (Phase 4)
            files.append(f"0/{f}")
        # 0/alphat is rendered deterministically (Phase 4).
        return files

    # ── Deterministic builders ────────────────────────────────────────────

    def _build_fv_solution(self, config: dict[str, Any]) -> str:
        """rhoPimpleFoam fvSolution — PIMPLE, compressible, energy.

        Profile-aware (gas vs cryogenic).  PIMPLE block adds nOuterCorrectors
        and nCorrectors; pressure has a pFinal block; equations have Final
        regex variants.
        """
        ctx = self._fv_context(config)
        eq_fields = self._equation_fields(ctx.turb_model)

        p_block, p_final = self._build_pressure_solver_block(ctx, is_simple=False)
        rho_block = self._build_rho_solver_block()
        eq_block, eq_final = self._build_equation_solver_block(eq_fields, is_simple=False)
        bounds_block = self._build_compressible_bounds(config, ctx)
        pimple_block = self._build_pimple_block(ctx, eq_fields, bounds_block)
        relax_block = self._build_relaxation_pimple(ctx)

        return (
            self._foam_file_header("fvSolution")
            + "solvers\n{\n"
            + p_block
            + p_final
            + rho_block
            + eq_block
            + eq_final
            + "}\n"
            + pimple_block
            + relax_block
            + self._foam_file_footer()
        )

    def _build_fv_schemes(self, config: dict[str, Any]) -> str:
        """rhoPimpleFoam fvSchemes — transient compressible (Euler ddt)."""
        ctx = self._fv_context(config)
        return (
            self._foam_file_header("fvSchemes")
            + self._build_ddt_block(ctx) + "\n"
            + self._build_grad_block(ctx) + "\n"
            + self._build_div_block(ctx) + "\n"
            + self._build_laplacian_block(ctx) + "\n"
            + self._build_interpolation_block() + "\n"
            + self._build_sngrad_block(ctx) + "\n"
            + self._build_flux_required_block()
            + ("\n" + self._build_wall_dist_block(ctx.turb_model)
               if ctx.turb_model != "laminar" else "")
            + self._foam_file_footer()
        )

    def validate(self, files: dict[str, str], config: dict[str, Any]) -> ValidationResult:
        issues: list[ValidationIssue] = []
        fixed = dict(files)

        # Deterministic fvSolution + fvSchemes
        # Deterministic files (LLM never generates these — Phase 4)

        fixed.update(self.render_deterministic_files(config))

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
        # Phase 2: the isothermal-rhoPimpleFoam GAMG→PBiCGStab switch is now
        # made up-front by resolve_pressure_solver_strategy() and rendered by
        # _build_pressure_solver_block().  The post-gen regex hack
        # (_fix_isothermal_pimple) is no longer needed.
        fixed = self._unify_inlet_turbulence(fixed, issues, config)

        # Hoisted BC robustness fixers (shared with rhoSimpleFoam) — match
        # the OF rhoPimpleFoam reference tutorials:
        #   * outlet U/T/k/ω/ε  → inletOutlet (handle backflow)
        #   * inlet k/ω/ε       → turbulentIntensity… / mixingLength…
        #     (derive from actual U rather than precheck-precomputed values)
        fixed = self._fix_outlet_backflow_bcs(fixed, issues, config)
        fixed = self._fix_inlet_turbulence_bc_types(fixed, issues, config)

        return ValidationResult(files=fixed, issues=issues)


    def _fix_alphat_wall(self, files: dict[str, str], issues: list[ValidationIssue]) -> dict[str, str]:
        alphat = files.get("0/alphat", "")
        if alphat and "alphatWallFunction" in alphat and "compressible::alphatWallFunction" not in alphat:
            files["0/alphat"] = alphat.replace("alphatWallFunction", "compressible::alphatWallFunction")
            issues.append(ValidationIssue("warning", "0/alphat", "Fixed: alphatWallFunction -> compressible::alphatWallFunction."))
        return files

    # _fix_isothermal_pimple was removed in Phase 2 — the GAMG→PBiCGStab
    # decision for isothermal rhoPimpleFoam now lives in
    # resolve_pressure_solver_strategy() and the renderer emits the correct
    # solver from the start.

