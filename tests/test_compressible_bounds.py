# tests/test_compressible_bounds.py
"""Tests for rhoSimpleFoam compressible bounds (pMin/pMax, ρ_max, div(phi,U)).

These three bounds are what stops the catastrophic startup cascade on
compressible gas cases with a high pressure ratio:

  * pressureControl unbounded  →  p goes to ±1e+42  →  GAMG::scale SIGFPE
  * rhoMax = 10 (hard-coded)   →  clamps 30–60 % of inlet cells at moderate
                                  compressor pressures (air @ 1.4 MPa, 280 K
                                  has ρ ≈ 18 kg/m³)
  * linearUpwindV for div(phi,U) on a 14:1 startup  →  acoustic overshoot
                                                      → +1e+25 m/s velocity
"""

from simd_agent.run.case_spec.resolvers import resolve_compressible_bounds
from simd_agent.solvers.contexts import FvBuildContext


# ── ρ_max from real inlet density (gas) ──────────────────────────────────────


class TestRhoMaxFromInletDensity:
    def test_air_at_atmospheric_pressure_keeps_floor_of_10(self):
        # Air @ 1 atm, 300 K → ρ ≈ 1.18 kg/m³.  1.5× safety = 1.77, but the
        # floor at 10 preserves the loose-safety-net behaviour for normal
        # atmospheric cases.
        b = resolve_compressible_bounds(
            is_compressible=True, profile="gas", rho=None,
            bc_temps=[300.0], eos_t_ceiling=None,
            op_p=101325.0, mach=0.1, inlet_p=101325.0,
        )
        assert b.rho_max == 10.0

    def test_compressor_inlet_at_14_bar_uses_real_density(self):
        # The exact case the user hit: air @ 1.435 MPa, T_cold = 280 K.
        # ρ = 1.435e6 / (287 × 280) = 17.85 kg/m³.  1.5× = 26.78.
        b = resolve_compressible_bounds(
            is_compressible=True, profile="gas", rho=None,
            bc_temps=[280.0, 500.0], eos_t_ceiling=None,
            op_p=101325.0, mach=0.1, inlet_p=1.435e6,
        )
        # Old behaviour: rho_max = 10 → clamped 30-60% of inlet cells.
        # New behaviour: ≥ 26 so the inlet density is never clamped.
        assert b.rho_max >= 26.0
        assert b.rho_max <= 30.0

    def test_extreme_pressure_capped_at_200(self):
        # Rocket-chamber-scale pressure shouldn't produce ρ_max in the
        # thousands — we cap at 200 to keep the bound sensible.
        b = resolve_compressible_bounds(
            is_compressible=True, profile="gas", rho=None,
            bc_temps=[1000.0], eos_t_ceiling=None,
            op_p=101325.0, mach=0.3, inlet_p=2.0e8,
        )
        assert b.rho_max == 200.0

    def test_uses_coldest_bc_temp(self):
        # Multiple BC temps — the resolver must pick the *coldest* because
        # ρ = p/(R·T) is largest at the coldest temperature.
        b = resolve_compressible_bounds(
            is_compressible=True, profile="gas", rho=None,
            bc_temps=[1000.0, 280.0, 500.0], eos_t_ceiling=None,
            op_p=101325.0, mach=0.1, inlet_p=1.435e6,
        )
        # 1.435e6 / (287 × 280) × 1.5 ≈ 26.7
        assert 26.0 <= b.rho_max <= 28.0

    def test_no_bc_temps_uses_standard_temperature(self):
        # No BC temps known → fall back to 288.15 K (ISA standard).
        b = resolve_compressible_bounds(
            is_compressible=True, profile="gas", rho=None,
            bc_temps=None, eos_t_ceiling=None,
            op_p=101325.0, mach=0.1, inlet_p=1.435e6,
        )
        # 1.435e6 / (287 × 288.15) × 1.5 ≈ 25.99 → still well above 10
        assert b.rho_max > 20.0


# ── pMin / pMax presence ──────────────────────────────────────────────────────


class TestPressureBoundsPresent:
    def test_gas_compressible_always_has_p_bounds(self):
        b = resolve_compressible_bounds(
            is_compressible=True, profile="gas", rho=None,
            bc_temps=[300.0], eos_t_ceiling=None,
            op_p=101325.0, mach=0.1, inlet_p=None,
        )
        assert b.p_min is not None
        assert b.p_max is not None
        assert b.p_min > 0
        assert b.p_max > b.p_min

    def test_p_max_scales_with_high_inlet_pressure(self):
        # When inlet pressure is much higher than operating, pMax must
        # accommodate it — otherwise the inlet BC value crashes against
        # the clamp.
        b_low = resolve_compressible_bounds(
            is_compressible=True, profile="gas", rho=None,
            bc_temps=[300.0], eos_t_ceiling=None,
            op_p=101325.0, mach=0.1, inlet_p=None,
        )
        b_high = resolve_compressible_bounds(
            is_compressible=True, profile="gas", rho=None,
            bc_temps=[300.0], eos_t_ceiling=None,
            op_p=101325.0, mach=0.1, inlet_p=1.435e6,
        )
        # High-inlet case pMax must be at least 1.5× the inlet, comfortably
        # accommodating the 1.435 MPa boundary value.
        assert b_high.p_max >= 1.435e6 * 1.5 * 0.99
        # And it must be higher than the low-inlet case.
        assert b_high.p_max > b_low.p_max

    def test_incompressible_has_no_bounds(self):
        b = resolve_compressible_bounds(
            is_compressible=False, profile="gas", rho=None,
            bc_temps=None, eos_t_ceiling=None,
            op_p=101325.0, mach=0.0, inlet_p=None,
        )
        assert b.rho_max is None
        assert b.p_max is None
        assert b.transonic is False


# ── pressure_ratio property + div(phi,U) decision ────────────────────────────


def _ctx(bc_pressures: tuple[float, ...] = (), **overrides) -> FvBuildContext:
    """Build a minimal FvBuildContext for div-block tests."""
    defaults: dict = dict(
        tier="good", non_ortho=20.0, use_simplec=False, n_non_ortho=1,
        vel_mag=5.0, speed_tier="low", bc_temps=(280.0, 500.0),
        bc_pressures=bc_pressures,
        profile="gas", heat_transfer_active=True, turb_model="kOmegaSST",
        mesh_quality={},
    )
    defaults.update(overrides)
    return FvBuildContext(**defaults)


class TestPressureRatioProperty:
    def test_single_pressure_is_unity(self):
        assert _ctx(bc_pressures=(101325.0,)).pressure_ratio == 1.0

    def test_empty_is_unity(self):
        assert _ctx(bc_pressures=()).pressure_ratio == 1.0

    def test_high_ratio_compressor(self):
        # 1.435 MPa inlet against 101325 Pa outlet → 14.17:1.
        ctx = _ctx(bc_pressures=(101325.0, 1.435e6))
        assert 14.0 < ctx.pressure_ratio < 15.0


class TestDivPhiUStartupSafety:
    """The pressure-ratio guard in _build_div_block."""

    def test_high_dp_pimple_forces_upwind(self):
        # Compressor case on rhoPimpleFoam (PIMPLE): 14:1 pressure ratio
        # → upwind kicks in via the pressure-ratio guard.  rhoSimpleFoam
        # (SIMPLE) uses upwind unconditionally now (see
        # test_rhosimplefoam_of_reference.py), so this guard's role has
        # narrowed to the PIMPLE/PISO branch only.
        from simd_agent.solvers.rhoPimpleFoam.solver import RhoPimpleFoamSolver
        plugin = RhoPimpleFoamSolver()
        ctx = _ctx(bc_pressures=(101325.0, 1.435e6), speed_tier="low")
        out = plugin._build_div_block(ctx)
        assert "div(phi,U)      bounded Gauss upwind;" in out
        assert "linearUpwindV" not in out.split("div(phi,U)")[1].split(";")[0]

    def test_low_dp_pimple_keeps_linearUpwindV(self):
        # Atmospheric outlet against atmospheric inlet → ratio ≈ 1.
        # linearUpwindV remains the accuracy-preferred choice for
        # PIMPLE-mode compressible — the Δt absorbs any startup overshoot.
        from simd_agent.solvers.rhoPimpleFoam.solver import RhoPimpleFoamSolver
        plugin = RhoPimpleFoamSolver()
        ctx = _ctx(bc_pressures=(101325.0, 1.2e5), speed_tier="low")
        out = plugin._build_div_block(ctx)
        assert "linearUpwindV grad(U)" in out

    def test_high_speed_pimple_still_upwind_regardless_of_dp(self):
        # The pre-existing high-speed guard still forces upwind for PIMPLE.
        from simd_agent.solvers.rhoPimpleFoam.solver import RhoPimpleFoamSolver
        plugin = RhoPimpleFoamSolver()
        ctx = _ctx(bc_pressures=(101325.0,), speed_tier="high")
        out = plugin._build_div_block(ctx)
        assert "div(phi,U)      bounded Gauss upwind;" in out
