"""Steady (SIMPLE) family base.

Owns everything that's specific to steady-state solvers:

  * The ``SIMPLE { … }`` algorithm block (non-ortho correctors, optional
    SIMPLEC, residualControl as plain scalars).
  * The steady relaxation pattern (``rho 0.05`` for compressible
    SIMPLE; tier-aware U/p/turb/h damping).
  * ``ddt`` defaults to ``steadyState``.
  * **No ``Final`` variants** — SIMPLE has no concept of an outer-loop
    final iteration.

What it does NOT own (kept in ``SolverPlugin`` base):

  * BC fixers, foam-file headers, mesh-quality helpers, ``_fv_context``,
    grad/div/laplacian block builders (driven by ``regime_profile`` and
    paradigm-agnostic).
  * Identity attributes, abstract methods, prompt loaders.

Solvers inheriting:

  * ``SimpleFoamSolver`` (incompressible steady)
  * ``RhoSimpleFoamSolver(SteadyBase, CompressibleMixin)``
  * ``BuoyantSimpleFoamSolver(SteadyBase, BoussinesqMixin)``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from simd_agent.solvers.base import SolverPlugin

if TYPE_CHECKING:
    from simd_agent.solvers.contexts import FvBuildContext


class SteadyBase(SolverPlugin):
    """Abstract base for SIMPLE-mode steady-state solvers.

    Sets ``algorithm = "SIMPLE"`` and ``is_transient = False`` as defaults
    so plugins don't have to repeat them.  Owns the SIMPLE block + steady
    relaxation builders.
    """

    algorithm: str = "SIMPLE"
    is_transient: bool = False

    # ── SIMPLE algorithm block ────────────────────────────────────────────

    def _build_simple_block(
        self,
        ctx: "FvBuildContext",
        eq_fields: list[str],
        bounds_block: str,
    ) -> str:
        """Build the ``SIMPLE { … }`` algorithm block.

        Encodes: non-ortho correctors, SIMPLEC switch, compressible bounds
        (passed in as ``bounds_block``), pRef, and residualControl as plain
        scalars (the SIMPLE format — PIMPLE uses sub-dicts instead).
        """
        n_non_ortho = ctx.n_non_ortho
        use_simplec = ctx.use_simplec
        tier = ctx.tier
        profile = ctx.profile
        speed_tier = ctx.speed_tier
        pf = self.pressure_field

        if speed_tier == "high" and n_non_ortho < 2:
            n_non_ortho = 2

        simplec_line = ""
        if (
            profile == "gas"
            and use_simplec
            and tier != "unknown"
            and speed_tier != "high"
        ):
            simplec_line = "    consistent      yes;\n"

        # Pressure tolerance — per-solver (rhoSimpleFoam: 1e-3, others: 1e-4).
        p_tol = self.pressure_residual_tol
        p_tol_str = (
            f"{p_tol:.0e}".replace("e-0", "e-").replace("e+0", "e+")
        )
        res_lines = (
            f"        {pf:<16}{p_tol_str};\n"
            f"        U               1e-4;\n"
        )
        turb_res_fields = [
            f for f in eq_fields if f not in ("U", self.energy_var)
        ]
        if turb_res_fields:
            if len(turb_res_fields) == 1:
                res_lines += f"        {turb_res_fields[0]:<16}1e-3;\n"
            else:
                turb_regex = f'"({"|".join(turb_res_fields)})"'
                res_lines += f"        {turb_regex:<16}1e-3;\n"
        if self.supports_energy:
            res_lines += f"        {self.energy_var:<16}1e-3;\n"

        return (
            f"\n{self.algorithm}\n"
            "{\n"
            f"    nNonOrthogonalCorrectors {n_non_ortho};\n"
            f"{simplec_line}"
            f"{bounds_block}"
            "    pRefCell        0;\n"
            "    pRefValue       0;\n"
            "\n"
            "    residualControl\n"
            "    {\n"
            f"{res_lines}"
            "    }\n"
            "}\n"
        )

    # ── Steady relaxationFactors block ────────────────────────────────────

    def _build_relaxation_simple(
        self,
        ctx: "FvBuildContext",
        eq_fields: list[str],
    ) -> str:
        """Build the ``relaxationFactors { … }`` block for a SIMPLE solver.

        Profile-aware: cryogenic forces conservative h=0.05; gas uses
        velocity-tier-aware textbook values.  ``rho 0.05`` is added to the
        fields block for compressible SIMPLE (matches OF rhoSimpleFoam
        tutorial — 95% density damping prevents the pressure-correction →
        density-update → continuity-error feedback loop).
        """
        profile = ctx.profile
        speed_tier = ctx.speed_tier
        pf = self.pressure_field

        if profile == "cryogenic":
            u_relax, p_relax, turb_relax, h_relax = 0.5, 0.3, 0.5, 0.05
        elif speed_tier == "high":
            u_relax, p_relax, turb_relax, h_relax = 0.3, 0.2, 0.3, 0.3
        elif speed_tier == "moderate":
            u_relax, p_relax, turb_relax, h_relax = 0.5, 0.3, 0.5, 0.5
        else:
            u_relax, p_relax, turb_relax, h_relax = 0.7, 0.3, 0.7, 0.5

        relax_eq_lines = f"        U               {u_relax};\n"
        for f in eq_fields:
            if f == "U":
                continue
            if f == self.energy_var:
                relax_eq_lines += (
                    f"        {self.energy_var:<16}{h_relax};\n"
                )
            else:
                relax_eq_lines += f"        {f:<16}{turb_relax};\n"

        # Density under-relaxation — compressible SIMPLE only.
        rho_relax_line = ""
        if self.is_compressible:
            rho_relax_line = "        rho             0.05;\n"

        return (
            "\nrelaxationFactors\n"
            "{\n"
            "    fields\n"
            "    {\n"
            f"        {pf:<16}{p_relax};\n"
            f"{rho_relax_line}"
            "    }\n"
            "    equations\n"
            "    {\n"
            f"{relax_eq_lines}"
            "    }\n"
            "}\n"
        )
