# tests/test_residual_and_energy_solver.py
"""Tests for the loosened pressure residual + dedicated energy solver block.

Both fixes track the OpenFOAM
``compressible/rhoSimpleFoam/angledDuctExplicitFixedCoeff`` reference:

  * ``residualControl.p = 1e-3`` for rhoSimpleFoam (the OF tutorial uses
    ``1e-2``; we keep ``1e-4`` for incompressible solvers where the
    pressure equation is much better behaved).

  * Energy (``e`` / ``h``) gets its own ``PBiCG + DILU`` block rather
    than being bucketed with U / k / ω under the smoothSolver regex.
    ``Solving for h, Initial residual = 1, No Iterations 1-2`` in the
    failure log was a direct symptom of the bucketed under-convergence.
"""

from __future__ import annotations

from simd_agent.solvers.contexts import FvBuildContext
from simd_agent.solvers.compressible.rhoSimpleFoam.solver import RhoSimpleFoamSolver
from simd_agent.solvers.compressible.rhoPimpleFoam.solver import RhoPimpleFoamSolver
from simd_agent.solvers.incompressible.simpleFoam.solver import SimpleFoamSolver


def _ctx(**overrides) -> FvBuildContext:
    defaults: dict = dict(
        tier="good", non_ortho=20.0, use_simplec=False, n_non_ortho=1,
        vel_mag=5.0, speed_tier="low", bc_temps=(280.0, 500.0),
        bc_pressures=(101325.0, 1.435e6),
        profile="gas", heat_transfer_active=True, turb_model="kOmegaSST",
        mesh_quality={},
    )
    defaults.update(overrides)
    return FvBuildContext(**defaults)


# ── Pressure residual tolerance ────────────────────────────────────────────


class TestPressureResidualTolerance:
    def test_rhoSimpleFoam_uses_1e_minus_3(self):
        """rhoSimpleFoam relaxes ``p`` residual to ``1e-3``."""
        plugin = RhoSimpleFoamSolver()
        ctx = _ctx()
        eq_fields = plugin._equation_fields("kOmegaSST")
        out = plugin._build_simple_block(ctx, eq_fields, "")
        # residualControl block should have p 1e-3 (not 1e-4).
        res_block = out.split("residualControl")[1]
        assert "p               1e-3;" in res_block
        assert "p               1e-4;" not in res_block

    def test_simpleFoam_keeps_1e_minus_4(self):
        """Incompressible simpleFoam keeps the tighter ``1e-4`` default."""
        plugin = SimpleFoamSolver()
        ctx = _ctx(profile="gas")
        eq_fields = plugin._equation_fields("kOmegaSST")
        out = plugin._build_simple_block(ctx, eq_fields, "")
        res_block = out.split("residualControl")[1]
        assert "p               1e-4;" in res_block

    def test_other_fields_untouched(self):
        """U / turb / energy tolerances unchanged — only ``p`` was loosened."""
        plugin = RhoSimpleFoamSolver()
        ctx = _ctx()
        eq_fields = plugin._equation_fields("kOmegaSST")
        out = plugin._build_simple_block(ctx, eq_fields, "")
        res_block = out.split("residualControl")[1]
        assert "U               1e-4;" in res_block
        assert '"(k|omega)"     1e-3;' in res_block
        assert "e               1e-3;" in res_block


# ── Dedicated energy solver block (PBiCG + DILU) ────────────────────────────


class TestDedicatedEnergyBlock:
    def test_rhoSimpleFoam_e_has_own_block_with_PBiCG_DILU(self):
        plugin = RhoSimpleFoamSolver()
        eq_fields = plugin._equation_fields("kOmegaSST")
        eq_block, eq_final = plugin._build_equation_solver_block(
            eq_fields, is_simple=True
        )
        # ``e`` has its own block.
        assert "\n    e\n" in eq_block
        # That block uses PBiCG + DILU.
        e_section = eq_block.split("\n    e\n")[1].split("    }\n")[0]
        assert "solver          PBiCG;" in e_section
        assert "preconditioner  DILU;" in e_section
        # SIMPLE has no Final variant.
        assert eq_final == ""

    def test_e_not_in_smoothSolver_regex(self):
        """The regex group for smoothSolver must NOT include ``e`` as a field."""
        plugin = RhoSimpleFoamSolver()
        eq_fields = plugin._equation_fields("kOmegaSST")
        eq_block, _ = plugin._build_equation_solver_block(
            eq_fields, is_simple=True
        )
        # The smoothSolver regex is exactly ``"(U|k|omega)"`` — and
        # crucially *not* ``"(U|k|omega|e)"`` which is what we used to
        # emit.  Substring containment is unreliable here (``omega``
        # contains ``e``), so assert the exact regex.
        assert '"(U|k|omega)"' in eq_block
        assert '"(U|k|omega|e)"' not in eq_block
        assert '"(U|k|omega|h)"' not in eq_block

    def test_rhoPimpleFoam_h_has_own_block_and_Final_variant(self):
        """PIMPLE generates both ``h`` and ``hFinal`` blocks."""
        plugin = RhoPimpleFoamSolver()
        eq_fields = plugin._equation_fields("kOmegaSST")
        eq_block, eq_final = plugin._build_equation_solver_block(
            eq_fields, is_simple=False
        )
        assert "\n    h\n" in eq_block
        assert "solver          PBiCG;" in eq_block.split("\n    h\n")[1]
        # Final block exists and contains hFinal.
        assert "\n    hFinal\n" in eq_final
        assert "solver          PBiCG;" in eq_final.split("\n    hFinal\n")[1]

    def test_simpleFoam_has_no_energy_block(self):
        """Incompressible simpleFoam (supports_energy=False) — no energy split."""
        plugin = SimpleFoamSolver()
        eq_fields = plugin._equation_fields("kOmegaSST")
        eq_block, _ = plugin._build_equation_solver_block(
            eq_fields, is_simple=True
        )
        # No standalone h or e block; no PBiCG entry.
        assert "\n    h\n" not in eq_block
        assert "\n    e\n" not in eq_block
        assert "PBiCG" not in eq_block

    def test_single_field_no_regex_quotes(self):
        """If only U remains after extracting energy, the block uses ``U`` (no quotes)."""
        plugin = RhoSimpleFoamSolver()
        # Force laminar so only ``U`` and ``e`` are present.
        eq_block, _ = plugin._build_equation_solver_block(
            ["U", "e"], is_simple=True
        )
        # ``U`` heads the smoothSolver block (no surrounding quotes).
        assert "\n    U\n" in eq_block
        # ``e`` heads its own PBiCG block.
        assert "\n    e\n" in eq_block
