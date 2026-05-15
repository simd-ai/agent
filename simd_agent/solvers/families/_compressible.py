"""Compressible mixin — for solvers that solve a ρ transport equation.

Composed via MRO with one of the families (Steady or Transient).  Adds
methods specific to compressible physics:

  * ``_build_rho_solver_block``     — emits the ``rho`` solver entry.
                                      Adds a ``rhoFinal`` variant when the
                                      composed plugin is on a transient
                                      algorithm (PIMPLE/PISO).  This is the
                                      fix for the historical
                                      ``Entry 'rhoFinal' not found`` IO
                                      error on rhoPimpleFoam.
  * ``_build_compressible_bounds``  — renders rhoMin / rhoMax / pMin /
                                      pMax / transonic from the resolved
                                      ``CompressibleBounds`` strategy.

This mixin does NOT inherit ``SolverPlugin`` directly — it relies on the
composed solver to inherit ``SolverPlugin`` via its family base.  That
keeps the MRO linear: e.g.
``RhoSimpleFoamSolver(SteadyBase, CompressibleMixin)`` →
``RhoSimpleFoamSolver → SteadyBase → CompressibleMixin → SolverPlugin``.

Composed solvers:

  * ``RhoSimpleFoamSolver(SteadyBase, CompressibleMixin)``
  * ``RhoPimpleFoamSolver(TransientBase, CompressibleMixin)``
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from simd_agent.solvers.contexts import FvBuildContext


class CompressibleMixin:
    """Methods shared by every compressible solver (rho*-Foam family).

    The mixin assumes the composed class has the standard ``SolverPlugin``
    surface (``self.is_compressible``, ``self.algorithm``,
    ``self.energy_var``, …).  No abstract methods of its own.
    """

    # ── ρ solver block (with PIMPLE Final variant) ───────────────────────

    def _build_rho_solver_block(self) -> str:
        """Compressible solvers need a ``rho`` solver entry.

        PIMPLE / PISO additionally need a ``rhoFinal`` entry — on the
        final outer iteration of each time step the algorithm looks up
        ``<field>Final`` for every solved field.  Missing entry triggers
        ``FOAM FATAL IO ERROR: Entry 'rhoFinal' not found``.  SIMPLE has
        no final-iteration concept, so the Final variant is omitted there
        — but it's tested by ``TestRhoSimpleFoamHasNoFinals`` to ensure
        we never accidentally emit one on a steady case.
        """
        # ``self.is_compressible`` / ``self.algorithm`` come from the
        # composed plugin's class attributes.  The mixin doesn't declare
        # them — relies on duck typing through MRO.
        if not getattr(self, "is_compressible", False):  # noqa: SLF001
            return ""

        base = (
            "\n    rho\n"
            "    {\n"
            "        solver          PCG;\n"
            "        preconditioner  DIC;\n"
            "        tolerance       1e-06;\n"
            "        relTol          0;\n"
            "    }\n"
        )
        algo = getattr(self, "algorithm", "SIMPLE")
        if algo in ("PIMPLE", "PISO"):
            base += (
                "\n    rhoFinal\n"
                "    {\n"
                "        $rho;\n"
                "        relTol          0;\n"
                "    }\n"
            )
        return base

    # ── rhoMin / rhoMax / pMin / pMax / transonic ────────────────────────

    def _build_compressible_bounds(
        self,
        config: dict[str, Any],
        ctx: "FvBuildContext",
    ) -> str:
        """Render the rho/p bounds + transonic block for the SIMPLE/PIMPLE body.

        Delegates to ``resolve_compressible_bounds`` so every per-field
        decision (rho_min from real ρ, pMax sized to inlet stagnation,
        transonic detection from Mach) lives in one typed resolver.
        """
        if not getattr(self, "is_compressible", False):
            return ""
        from simd_agent.run.case_spec import resolve_compressible_bounds

        profile = ctx.profile
        vel_mag = ctx.vel_mag

        fluid = config.get("fluid") or {}
        rho_cfg: float | None = None
        if isinstance(fluid, dict):
            for k in ("density", "rho"):
                v = fluid.get(k)
                if v is None:
                    continue
                try:
                    rho_cfg = float(v)
                    break
                except (TypeError, ValueError):
                    pass

        bc_temps: list[float] = []
        inlet_t: float | None = None
        for pbc in (config.get("boundary_conditions") or {}).values():
            if not isinstance(pbc, dict):
                continue
            t_entry = pbc.get("temperature") or pbc.get("T")
            t_val = (
                t_entry.get("value") or t_entry.get("uniform")
                if isinstance(t_entry, dict)
                else t_entry
            )
            try:
                tv = float(t_val)
            except (TypeError, ValueError):
                continue
            bc_temps.append(tv)
            if inlet_t is None:
                inlet_t = tv

        # Mach from speed of sound (gas only).
        if profile == "cryogenic":
            mach = 0.0
        else:
            t_for_a = inlet_t if inlet_t and inlet_t > 0 else 300.0
            a_sound = (1.4 * 287.0 * t_for_a) ** 0.5
            mach = (vel_mag / a_sound) if a_sound > 0 else 0.0

        # Operating + inlet pressure.
        op_p = 101325.0
        inlet_p: float | None = None
        bcs = config.get("boundary_conditions") or {}
        try:
            outlet_p_entry = bcs.get("outlet", {}).get("pressure", {})
            if isinstance(outlet_p_entry, dict):
                pv = outlet_p_entry.get("value") or outlet_p_entry.get("uniform")
                if pv is not None:
                    op_p = float(pv)
        except (TypeError, ValueError, AttributeError):
            pass
        for _name, _pbc in bcs.items():
            if _name == "outlet" or not isinstance(_pbc, dict):
                continue
            p_entry = _pbc.get("pressure") or _pbc.get("p")
            p_val = (
                p_entry.get("value") or p_entry.get("uniform")
                if isinstance(p_entry, dict) else p_entry
            )
            try:
                pv = float(p_val)
            except (TypeError, ValueError):
                continue
            if pv > 0 and (inlet_p is None or pv > inlet_p):
                inlet_p = pv

        bounds = resolve_compressible_bounds(
            is_compressible=True,
            profile=profile,
            rho=rho_cfg,
            bc_temps=sorted(set(bc_temps)),
            eos_t_ceiling=None,
            op_p=op_p,
            mach=mach,
            inlet_p=inlet_p,
        )

        lines = ""
        if bounds.rho_min is not None and bounds.rho_max is not None:
            lines += (
                f"    rhoMin          {bounds.rho_min:.3g};\n"
                f"    rhoMax          {bounds.rho_max:.3g};\n"
            )
        if bounds.p_min is not None and bounds.p_max is not None:
            lines += (
                f"    pMin            {bounds.p_min:.6g};\n"
                f"    pMax            {bounds.p_max:.6g};\n"
            )
        if bounds.transonic:
            lines += "    transonic       yes;\n"
        return lines
