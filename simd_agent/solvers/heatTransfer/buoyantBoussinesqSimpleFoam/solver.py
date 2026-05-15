# simd_agent/solvers/heatTransfer/buoyantBoussinesqSimpleFoam/solver.py
"""buoyantBoussinesqSimpleFoam — steady incompressible Boussinesq.

Models natural-convection / mixed-convection cases where the density
variation can be neglected everywhere except in the buoyancy term:

    ρ_eff = ρ₀ · (1 − β·(T − T_ref))

Uses ``p_rgh`` (modified pressure) and ``constant/g``.  Transport
properties (``ν``, ``β``, ``T_ref``, ``Pr``, ``Prt``) live in
``constant/transportProperties`` — no thermophysicalProperties needed
because the energy equation is a simple transport of T.

Mirrors the OpenFOAM ``tutorials/heatTransfer/buoyantBoussinesqSimpleFoam/hotRoom``
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
    SteadyBase,
)

logger = logging.getLogger(__name__)


class BuoyantBoussinesqSimpleFoamSolver(IncompressibleBoussinesqMixin, SteadyBase):
    """buoyantBoussinesqSimpleFoam — steady incompressible Boussinesq RANS.

    **Mixin order matters**: ``IncompressibleBoussinesqMixin`` is listed
    *before* ``SteadyBase`` so its class attributes (``is_compressible``,
    ``energy_var``, ``pressure_field``, ``needs_gravity``) override the
    ``SolverPlugin`` defaults via C3 MRO.  If the family base came first,
    those defaults would mask the mixin and the plugin would report
    ``pressure_field = "p"`` instead of ``"p_rgh"``.

    Plugin only adds the matching score, file manifest, and the
    deterministic fvSolution / fvSchemes recipes.
    """

    name = "buoyantBoussinesqSimpleFoam"
    is_transient = False
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
            return MatchResult(0.0, "buoyantBoussinesqSimpleFoam is single-phase only")
        if transient:
            return MatchResult(
                0.2,
                "buoyantBoussinesqSimpleFoam is steady; "
                "buoyantBoussinesqPimpleFoam for transient",
            )
        if compressible:
            return MatchResult(
                0.2,
                "Compressible buoyancy → buoyantSimpleFoam; Boussinesq is incompressible.",
            )
        if buoyancy and heat:
            return MatchResult(
                0.95,
                "Steady incompressible buoyancy-driven flow with heat — "
                "ideal for buoyantBoussinesqSimpleFoam.",
            )
        if buoyancy:
            return MatchResult(
                0.6,
                "Steady incompressible buoyancy without heat — "
                "buoyantBoussinesqSimpleFoam works but may be overkill.",
            )
        return MatchResult(0.0, "No buoyancy — buoyantBoussinesqSimpleFoam not needed")

    # ── Required LLM-generated files ──────────────────────────────────────

    def required_files(self, config: dict[str, Any]) -> list[str]:
        turb_model = self._get_turb_model_from_config(config)
        files = [
            "system/controlDict",
            # system/fvSchemes, system/fvSolution and constant/transportProperties
            # are rendered deterministically in validate(), not by the LLM.
            "constant/g",
            # Solved fields — LLM generates initial / BC values.
            "0/U", "0/p_rgh", "0/T",
        ]
        for f in self.turbulence_fields(turb_model):
            if f == "nut":
                continue  # 0/nut rendered deterministically
            files.append(f"0/{f}")
        return files

    # ── Deterministic builders ────────────────────────────────────────────

    def _build_fv_solution(self, config: dict[str, Any]) -> str:
        """buoyantBoussinesqSimpleFoam fvSolution — SIMPLE, incompressible, p_rgh.

        No ``rho`` block (incompressible).  Solvers: PCG+DIC for p_rgh;
        PBiCG+DILU for U / T / k / ε regex group; standard SIMPLE
        residualControl + tier-aware relaxation.
        """
        ctx = self._fv_context(config)
        eq_fields = self._equation_fields(ctx.turb_model)

        p_block, _ = self._build_pressure_solver_block(ctx, is_simple=True)
        eq_block, _ = self._build_equation_solver_block(eq_fields, is_simple=True)
        # No compressible bounds — incompressible Boussinesq has constant ρ.
        bounds_block = ""
        simple_block = self._build_simple_block(ctx, eq_fields, bounds_block)
        relax_block = self._build_relaxation_simple(ctx, eq_fields)

        return (
            self._foam_file_header("fvSolution")
            + "solvers\n{\n"
            + p_block
            + eq_block
            + "}\n"
            + simple_block
            + relax_block
            + self._foam_file_footer()
        )

    def _build_fv_schemes(self, config: dict[str, Any]) -> str:
        """buoyantBoussinesqSimpleFoam fvSchemes — steady, incompressible energy.

        No ``div(phid,p)``, no ``div(phi,K)`` (Boussinesq energy equation
        transports T directly with no kinetic-energy or pressure-work
        source).  Just U, T, turbulence-transport fields, and the viscous
        stress tensor in incompressible form.
        """
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
        """Render ``constant/transportProperties`` from fluid config + defaults.

        Mixed into the deterministic-files registry via ``render_deterministic_files``
        (overridden below).  Inputs come from ``config["fluid"]`` when
        present, otherwise the ``hotRoom`` defaults (air-like fluid).
        """
        rho, nu, beta, t_ref, Pr, Prt = self._extract_transport_inputs(config)
        return self.build_transport_properties(
            rho=rho, nu=nu, beta=beta, t_ref=t_ref, Pr=Pr, Prt=Prt,
        )

    def render_deterministic_files(self, config: dict[str, Any]) -> dict[str, str]:
        """Add ``constant/transportProperties`` to the deterministic file set.

        Boussinesq solvers use transportProperties instead of
        thermophysicalProperties — both replace any LLM-generated content.
        """
        files = super().render_deterministic_files(config)
        files["constant/transportProperties"] = self._build_transport_properties(config)
        return files

    # ── Validation ────────────────────────────────────────────────────────

    def validate(
        self, files: dict[str, str], config: dict[str, Any]
    ) -> ValidationResult:
        issues: list[ValidationIssue] = []
        fixed = dict(files)

        # Deterministic files (fvSolution, fvSchemes, transportProperties,
        # turbulenceProperties, 0/nut, 0/alphat) — LLM never generates these.
        fixed.update(self.render_deterministic_files(config))

        # ``constant/thermophysicalProperties`` would confuse the solver —
        # Boussinesq uses transportProperties.  Strip it if present.
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

        # Universal fixers.
        fixed = self._fix_controldict_solver(fixed, issues)
        fixed = self._fix_pressure_field(fixed, issues)
        fixed = self._ensure_gravity(fixed, issues)

        # No 0/h or 0/e — Boussinesq transports T directly.
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

        # Inlet turbulence quantities derived from precheck.
        fixed = self._unify_inlet_turbulence(fixed, issues, config)

        return ValidationResult(files=fixed, issues=issues)
