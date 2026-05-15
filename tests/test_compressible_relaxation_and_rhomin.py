# tests/test_compressible_relaxation_and_rhomin.py
"""Tests for the OpenFOAM-tutorial-aligned ρ relaxation + rho_min floor.

Two structural fixes from the
``OpenFOAM-2.2.x/tutorials/compressible/rhoSimpleFoam/angledDuctExplicitFixedCoeff``
reference:

  * ``relaxationFactors.fields { p 0.3; rho 0.05; }`` — 95 % damping of
    the density update.  Without it the density bounces freely between
    SIMPLE iterations, driving the continuity-error → pressure-correction
    → density-update feedback loop unstable.

  * ``rhoMin = 0.5 × ρ_inlet`` — a tight lower clamp (matches the
    tutorial's ``rhoMin 0.5`` for ρ ≈ 1.2 kg/m³).  Caps the resolver-side
    floor at ``max(0.1, 0.5·ρ_inlet)`` so non-physical negative densities
    are caught early.
"""

from __future__ import annotations

from simd_agent.run.case_spec.resolvers import resolve_compressible_bounds
from simd_agent.solvers.contexts import FvBuildContext
from simd_agent.solvers.compressible.rhoSimpleFoam.solver import RhoSimpleFoamSolver
from simd_agent.solvers.incompressible.simpleFoam.solver import SimpleFoamSolver
from simd_agent.solvers.heatTransfer.buoyantSimpleFoam.solver import BuoyantSimpleFoamSolver


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


# ── rho relaxation ─────────────────────────────────────────────────────────


class TestRhoRelaxationForCompressibleSimple:
    def test_rhoSimpleFoam_emits_rho_005_in_fields(self):
        """rhoSimpleFoam — gas profile — must damp ρ to 5 %."""
        plugin = RhoSimpleFoamSolver()
        ctx = _ctx(profile="gas")
        eq_fields = plugin._equation_fields("kOmegaSST")
        out = plugin._build_relaxation_simple(ctx, eq_fields)
        assert "rho             0.05;" in out
        # And inside the fields block, not the equations block.
        fields_block = out.split("fields")[1].split("equations")[0]
        assert "rho             0.05;" in fields_block

    def test_rhoSimpleFoam_cryogenic_also_damps_rho(self):
        """Cryogenic icoPolynomial ρ(T) is even more sensitive — keep 0.05."""
        plugin = RhoSimpleFoamSolver()
        ctx = _ctx(profile="cryogenic")
        eq_fields = plugin._equation_fields("kOmegaSST")
        out = plugin._build_relaxation_simple(ctx, eq_fields)
        assert "rho             0.05;" in out

    def test_buoyantSimpleFoam_does_not_damp_rho(self):
        """Boussinesq buoyantSimpleFoam derives ρ from T analytically — no rho relaxation needed."""
        plugin = BuoyantSimpleFoamSolver()
        if not plugin.is_compressible:  # buoyantSimpleFoam is incompressible Boussinesq
            ctx = _ctx(profile="gas")
            eq_fields = plugin._equation_fields("kOmegaSST")
            out = plugin._build_relaxation_simple(ctx, eq_fields)
            assert "rho" not in out

    def test_simpleFoam_does_not_emit_rho(self):
        """Incompressible simpleFoam has no ρ field — must not appear."""
        plugin = SimpleFoamSolver()
        ctx = _ctx(profile="gas")
        eq_fields = plugin._equation_fields("kOmegaSST")
        out = plugin._build_relaxation_simple(ctx, eq_fields)
        assert "rho" not in out


# ── rho_min floor ──────────────────────────────────────────────────────────


class TestRhoMinFloor:
    def test_atmospheric_rho_min_matches_OF_shape(self):
        """Atmospheric air (ρ ≈ 1.18) → rho_min ≈ 0.6 (~0.5 × ρ)."""
        b = resolve_compressible_bounds(
            is_compressible=True, profile="gas", rho=None,
            bc_temps=[300.0], eos_t_ceiling=None,
            op_p=101325.0, mach=0.1, inlet_p=None,
        )
        # ρ = 1.013e5 / (287 · 300) ≈ 1.177 → rho_min ≈ 0.589
        assert 0.55 <= b.rho_min <= 0.65
        # rho_max still has the 10 floor for atmospheric.
        assert b.rho_max == 10.0

    def test_compressor_inlet_rho_min_scales_with_density(self):
        """1.435 MPa, 280 K → ρ ≈ 17.85 → rho_min ≈ 8.9."""
        b = resolve_compressible_bounds(
            is_compressible=True, profile="gas", rho=None,
            bc_temps=[280.0, 500.0], eos_t_ceiling=None,
            op_p=101325.0, mach=0.1, inlet_p=1.435e6,
        )
        # rho_min = 0.5 × 17.85 = 8.93
        assert 8.5 <= b.rho_min <= 9.5
        assert 26.0 <= b.rho_max <= 30.0
        assert b.rho_min < b.rho_max

    def test_vacuum_chamber_rho_min_stays_at_floor(self):
        """At 100 Pa, ρ ≈ 1.2e-3 → 0.5·ρ ≈ 6e-4 — below safety floor."""
        b = resolve_compressible_bounds(
            is_compressible=True, profile="gas", rho=None,
            bc_temps=[300.0], eos_t_ceiling=None,
            op_p=100.0, mach=0.0, inlet_p=None,
        )
        # Floor at 0.1 — never go below.
        assert b.rho_min == 0.1

    def test_strict_less_than_invariant_holds_for_extremes(self):
        """rho_min must always be strictly less than rho_max."""
        # Extreme high pressure — both bounds get hit by ceilings.
        b = resolve_compressible_bounds(
            is_compressible=True, profile="gas", rho=None,
            bc_temps=[1000.0], eos_t_ceiling=None,
            op_p=101325.0, mach=0.0, inlet_p=2.0e8,
        )
        assert b.rho_min is not None
        assert b.rho_max is not None
        assert b.rho_min < b.rho_max

    def test_cryogenic_branch_unchanged(self):
        """Cryogenic ρ_inlet known → still uses [0.5·ρ, 1.5·ρ]."""
        b = resolve_compressible_bounds(
            is_compressible=True, profile="cryogenic", rho=808.0,
            bc_temps=[77.0, 200.0], eos_t_ceiling=248.9,
            op_p=101325.0, mach=0.0, inlet_p=None,
        )
        assert b.rho_min == 808.0 * 0.5
        assert b.rho_max == 808.0 * 1.5
