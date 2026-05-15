# simd_agent/solvers/heatTransfer/buoyantBoussinesqPimpleFoam/solver.py
"""buoyantBoussinesqPimpleFoam — transient incompressible Boussinesq.

Transient counterpart of ``buoyantBoussinesqSimpleFoam``.  Same Boussinesq
physics — constant density, buoyancy via ``-ρ₀·β·(T-T_ref)·g`` source,
energy equation in T — but the time loop is real (Euler / backward ddt,
PIMPLE outer correctors, ``<field>Final`` variants).

Mirrors the OpenFOAM ``tutorials/heatTransfer/buoyantBoussinesqPimpleFoam/hotRoom``
reference.
"""

from __future__ import annotations

import logging
from typing import Any

from simd_agent.solvers.base import (
    MatchResult,
    ValidationIssue,
    ValidationResult,
)
from simd_agent.solvers.families import (
    IncompressibleBoussinesqMixin,
    TransientBase,
)

logger = logging.getLogger(__name__)


class BuoyantBoussinesqPimpleFoamSolver(
    IncompressibleBoussinesqMixin, TransientBase
):
    """buoyantBoussinesqPimpleFoam — transient incompressible Boussinesq RANS.

    Mixin-first MRO so the Boussinesq overrides (``is_compressible=False``,
    ``energy_var="T"``, ``pressure_field="p_rgh"``, ``needs_gravity=True``)
    win over the ``SolverPlugin`` defaults.

    No ``rho`` (or ``rhoFinal``) solver block — incompressible Boussinesq
    has constant density.  ``TFinal``, ``UFinal``, ``p_rghFinal`` are
    emitted automatically by ``_build_pressure_solver_block`` and
    ``_build_equation_solver_block`` for the PIMPLE final outer iter.
    """

    name = "buoyantBoussinesqPimpleFoam"
    is_transient = True
    supports_energy = True
    is_multiphase = False

    # ── Matching ──────────────────────────────────────────────────────────

    def matches(self, config: dict[str, Any]) -> MatchResult:
        physics = config.get("physics", {}) or {}
        heat = bool(config.get("heat_transfer") or physics.get("heat_transfer"))
        compressible = (
            config.get("compressibility")
            or physics.get("compressibility", "incompressible")
        ) == "compressible"
        transient = (
            config.get("time_stepping")
            or physics.get("time_scheme", "steady")
        ) in ("transient", "unsteady")
        multiphase = bool(config.get("multiphase") or physics.get("multiphase"))
        buoyancy = bool(config.get("gravity") or physics.get("gravity"))

        if multiphase:
            return MatchResult(0.0, "buoyantBoussinesqPimpleFoam is single-phase only")
        if not transient:
            return MatchResult(
                0.2,
                "buoyantBoussinesqPimpleFoam is transient; "
                "buoyantBoussinesqSimpleFoam for steady",
            )
        if compressible:
            return MatchResult(
                0.2,
                "Compressible buoyancy → buoyantPimpleFoam; Boussinesq is incompressible.",
            )
        if buoyancy and heat:
            return MatchResult(
                0.95,
                "Transient incompressible buoyancy-driven flow with heat — "
                "ideal for buoyantBoussinesqPimpleFoam.",
            )
        if buoyancy:
            return MatchResult(
                0.6,
                "Transient incompressible buoyancy without heat — "
                "buoyantBoussinesqPimpleFoam works but may be overkill.",
            )
        return MatchResult(0.0, "No buoyancy — buoyantBoussinesqPimpleFoam not needed")

    # ── Required LLM-generated files ──────────────────────────────────────

    def required_files(self, config: dict[str, Any]) -> list[str]:
        turb_model = self._get_turb_model_from_config(config)
        files = [
            "system/controlDict",
            "constant/g",
            "0/U", "0/p_rgh", "0/T",
        ]
        for f in self.turbulence_fields(turb_model):
            if f == "nut":
                continue
            files.append(f"0/{f}")
        return files

    # ── Deterministic builders ────────────────────────────────────────────

    def _build_fv_solution(self, config: dict[str, Any]) -> str:
        """buoyantBoussinesqPimpleFoam fvSolution — PIMPLE, incompressible, p_rgh.

        Includes ``pFinal``, ``UFinal``, ``TFinal`` and the turbulence
        Final variants required by the PIMPLE final outer iteration.
        No ``rho`` block (constant density).
        """
        ctx = self._fv_context(config)
        eq_fields = self._equation_fields(ctx.turb_model)

        p_block, p_final_block = self._build_pressure_solver_block(ctx, is_simple=False)
        eq_block, eq_final_block = self._build_equation_solver_block(
            eq_fields, is_simple=False
        )
        # No compressible bounds — incompressible Boussinesq has constant ρ.
        bounds_block = ""
        pimple_block = self._build_pimple_block(ctx, eq_fields, bounds_block)
        relax_block = self._build_relaxation_pimple(ctx)

        return (
            self._foam_file_header("fvSolution")
            + "solvers\n{\n"
            + p_block
            + p_final_block
            + eq_block
            + eq_final_block
            + "}\n"
            + pimple_block
            + relax_block
            + self._foam_file_footer()
        )

    def _build_fv_schemes(self, config: dict[str, Any]) -> str:
        """buoyantBoussinesqPimpleFoam fvSchemes — transient, incompressible energy."""
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

    def _build_transport_properties(self, config: dict[str, Any]) -> str:
        """Render ``constant/transportProperties`` from fluid config + defaults."""
        rho, nu, beta, t_ref, Pr, Prt = self._extract_transport_inputs(config)
        return self.build_transport_properties(
            rho=rho, nu=nu, beta=beta, t_ref=t_ref, Pr=Pr, Prt=Prt,
        )

    def render_deterministic_files(self, config: dict[str, Any]) -> dict[str, str]:
        """Add ``constant/transportProperties`` to the deterministic file set."""
        files = super().render_deterministic_files(config)
        files["constant/transportProperties"] = self._build_transport_properties(config)
        return files

    # ── Validation ────────────────────────────────────────────────────────

    def validate(
        self, files: dict[str, str], config: dict[str, Any]
    ) -> ValidationResult:
        issues: list[ValidationIssue] = []
        fixed = dict(files)

        fixed.update(self.render_deterministic_files(config))

        # ``constant/thermophysicalProperties`` would confuse the solver —
        # Boussinesq uses transportProperties.
        if "constant/thermophysicalProperties" in fixed:
            del fixed["constant/thermophysicalProperties"]
            issues.append(
                ValidationIssue(
                    "warning",
                    "constant/thermophysicalProperties",
                    f"'{self.name}' uses transportProperties (Boussinesq); "
                    "removed thermophysicalProperties.",
                )
            )

        fixed = self._fix_controldict_solver(fixed, issues)
        fixed = self._fix_pressure_field(fixed, issues)
        fixed = self._ensure_gravity(fixed, issues)
        # residualControl in PIMPLE must be sub-dictionaries.
        fixed = self._fix_residual_control_format(fixed, issues)

        for ef in ("0/h", "0/e"):
            if ef in fixed:
                del fixed[ef]
                issues.append(
                    ValidationIssue(
                        "warning", ef,
                        f"Removed {ef}: Boussinesq energy variable is T.",
                    )
                )

        # Robustness fixers (outlet inletOutlet, inlet TI / mixing length).
        fixed = self._fix_outlet_backflow_bcs(fixed, issues, config)
        fixed = self._fix_inlet_turbulence_bc_types(fixed, issues, config)
        fixed = self._unify_inlet_turbulence(fixed, issues, config)

        return ValidationResult(files=fixed, issues=issues)
