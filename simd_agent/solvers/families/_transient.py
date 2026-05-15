"""Transient (PIMPLE / PISO) family base.

Owns everything that's specific to transient solvers:

  * The ``PIMPLE { … }`` algorithm block with the OF reference keys
    (``consistent yes``, ``transonic no``, ``turbOnFinalIterOnly no``).
  * The transient relaxation pattern (``fields { "p.*" 0.9; "rho.*" 1; }``
    for compressible; per-field equation patterns matching the OF tutorial).
  * **The ``Final``-variant invariant** — every solved field needs a
    matching ``<field>Final`` entry because PIMPLE consults it on the
    final outer iteration of each time step.  Bug class
    (``Entry 'rhoFinal' not found``) **physically cannot leak into
    SteadyBase** because steady solvers don't have a final outer iter.

What it does NOT own (kept in ``SolverPlugin`` base):

  * BC fixers, foam-file headers, mesh-quality helpers, ``_fv_context``,
    grad/div/laplacian block builders (driven by ``regime_profile`` and
    paradigm-agnostic).
  * Identity attributes, abstract methods, prompt loaders.

Solvers inheriting:

  * ``PimpleFoamSolver`` (incompressible transient)
  * ``RhoPimpleFoamSolver(TransientBase, CompressibleMixin)``
  * ``BuoyantPimpleFoamSolver(TransientBase, BoussinesqMixin)``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from simd_agent.solvers.base import SolverPlugin

if TYPE_CHECKING:
    from simd_agent.solvers.contexts import FvBuildContext


class TransientBase(SolverPlugin):
    """Abstract base for PIMPLE-mode transient solvers.

    Sets ``algorithm = "PIMPLE"`` and ``is_transient = True`` as defaults.
    Owns the PIMPLE block + transient relaxation builders.
    """

    algorithm: str = "PIMPLE"
    is_transient: bool = True

    # ── PIMPLE algorithm block ────────────────────────────────────────────

    def _build_pimple_block(
        self,
        ctx: "FvBuildContext",
        eq_fields: list[str],
        bounds_block: str,
    ) -> str:
        """Build the ``PIMPLE { … }`` algorithm block — condition-aware.

        Three knobs are now driven by the resolved context instead of
        hard-coded defaults:

          * ``nOuterCorrectors`` scales with ΔT_BC + impulsive inlets:
              - ΔT < 50 K, no impulsive BCs     →  2  (standard transient)
              - ΔT 50–150 K, or impulsive BCs   → 10  (stiff coupling)
              - ΔT > 150 K (e.g. 320 K U-bend)  → 30  (very stiff)
              - Pressure ratio ≥ 3                → bump to ≥ 20
            Matches the OF rhoPimpleFoam ras tutorial which uses 50 for
            the angledDuct case with a heated cellZone.

          * ``consistent yes`` (SIMPLEC) is the default for smooth cases
            but DISABLED for impulsive startups — SIMPLEC's second
            corrector restarts at residual ~1 and can re-create the
            divergence we're trying to damp.

          * ``nNonOrthogonalCorrectors`` keeps its existing mesh-tier
            heuristic (bumped at high speed).

        The other tutorial keys (``transonic no``, ``turbOnFinalIterOnly
        no``) are unconditional — they prevent fork-to-fork drift.
        """
        n_non_ortho = ctx.n_non_ortho
        speed_tier = ctx.speed_tier
        pf = self.pressure_field

        if speed_tier == "high" and n_non_ortho < 2:
            n_non_ortho = 2

        # ── nOuterCorrectors: scale with stiffness signals ─────────────
        delta_t = ctx.delta_t_bc
        impulsive = ctx.has_impulsive_inlets
        high_dp = ctx.pressure_ratio >= 3.0
        if delta_t > 150 or impulsive and delta_t > 100:
            n_outer = 30
        elif delta_t > 50 or impulsive or high_dp:
            n_outer = 10
        else:
            n_outer = 2

        # ── consistent: disable SIMPLEC for impulsive / high-Δp startups ─
        consistent_yes = not (impulsive or high_dp or delta_t > 150)

        res_lines = (
            f"        {pf}   {{ tolerance 1e-4; relTol 0; }}\n"
            "        U   { tolerance 1e-4; relTol 0; }\n"
        )
        turb_res_fields = [
            f for f in eq_fields if f not in ("U", self.energy_var)
        ]
        for tf in turb_res_fields:
            res_lines += f"        {tf}   {{ tolerance 1e-3; relTol 0; }}\n"
        if self.supports_energy:
            res_lines += (
                f"        {self.energy_var}   "
                f"{{ tolerance 5e-3; relTol 0; }}\n"
            )

        return (
            f"\n{self.algorithm}\n"
            "{\n"
            f"    nOuterCorrectors    {n_outer};\n"
            "    nCorrectors         1;\n"
            f"    nNonOrthogonalCorrectors {n_non_ortho};\n"
            "    momentumPredictor   yes;\n"
            f"    consistent          {'yes' if consistent_yes else 'no'};\n"
            "    transonic           no;\n"
            "    turbOnFinalIterOnly no;\n"
            f"{bounds_block}"
            "\n"
            "    residualControl\n"
            "    {\n"
            f"{res_lines}"
            "    }\n"
            "}\n"
        )

    # ── Transient relaxationFactors block ─────────────────────────────────

    def _build_relaxation_pimple(self, ctx: "FvBuildContext",) -> str:
        """Build the ``relaxationFactors { … }`` block for a PIMPLE solver.

        Matches the OF rhoPimpleFoam ras tutorial pattern:

          fields    { "p.*"  0.9;  "rho.*"  1; }       # compressible only
          equations { "U.*"  0.9;  "h.*"  0.7;
                      "(k|epsilon|omega).*" 0.8; }

        Incompressible PIMPLE (pimpleFoam) omits the ``fields`` block.
        Cryogenic + high-speed cases tighten relaxation to keep the outer
        iteration stable.

        The ``"<field>.*"`` regex covers both the base solver entry and
        the ``Final`` variant used on the last outer iteration.
        """
        profile = ctx.profile
        speed_tier = ctx.speed_tier

        if profile == "cryogenic":
            u_relax, h_relax, turb_relax = 0.5, 0.3, 0.5
            p_relax = 0.7
        elif speed_tier == "high":
            u_relax, h_relax, turb_relax = 0.3, 0.3, 0.3
            p_relax = 0.7
        elif speed_tier == "moderate":
            u_relax, h_relax, turb_relax = 0.7, 0.5, 0.7
            p_relax = 0.9
        else:
            u_relax, h_relax, turb_relax = 0.9, 0.7, 0.8
            p_relax = 0.9

        eq_lines = f'        "U.*"           {u_relax};\n'
        if self.supports_energy:
            eq_lines += (
                f'        "{self.energy_var}.*"           {h_relax};\n'
            )
        eq_lines += (
            f'        "(k|epsilon|omega).*" {turb_relax};\n'
        )

        if self.is_compressible:
            fields_block = (
                "    fields\n"
                "    {\n"
                f'        "{self.pressure_field}.*"           {p_relax};\n'
                '        "rho.*"         1;\n'
                "    }\n"
            )
        else:
            fields_block = ""

        return (
            "\nrelaxationFactors\n"
            "{\n"
            f"{fields_block}"
            "    equations\n"
            "    {\n"
            f"{eq_lines}"
            "    }\n"
            "}\n"
        )
