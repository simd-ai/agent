"""Tests for the Phase 2 physics resolvers in case_spec.resolvers.

Each resolver replaces one or more regex post-validators that used to live
in genai_codegen.py.  Tests verify the resolver returns the correct
strategy for every physics combination it used to handle.
"""

import pytest

from simd_agent.run.case_spec import (
    resolve_compressible_bounds,
    resolve_div_phi_h_scheme,
    resolve_fv_options_max,
    resolve_pressure_solver_strategy,
)


# ── resolve_pressure_solver_strategy — replaces Check 7c + Check 7e ────────


def test_pressure_resolver_default_GAMG_with_PCG_DIC_coarsest() -> None:
    """The standard case: compressible energy solver on a good mesh."""
    s = resolve_pressure_solver_strategy(
        solver_name="rhoSimpleFoam",
        is_compressible=True,
        mesh_tier="moderate",
        heat_transfer_active=True,
    )
    assert s.top_level == "GAMG"
    assert s.smoother_or_precond == "GaussSeidel"
    # Matches the OpenFOAM rhoSimpleFoam tutorial coarsest cap of 20 cells.
    assert s.n_coarsest_cells == 20
    assert s.coarsest is not None
    assert s.coarsest.solver == "PCG"
    assert s.coarsest.preconditioner == "DIC"


def test_pressure_resolver_rhoPimpleFoam_isothermal_skips_GAMG() -> None:
    """rhoPimpleFoam + no heat → PBiCGStab+DILU (the rule from Check 7e)."""
    s = resolve_pressure_solver_strategy(
        solver_name="rhoPimpleFoam",
        is_compressible=True,
        mesh_tier="moderate",
        heat_transfer_active=False,
    )
    assert s.top_level == "PBiCGStab"
    assert s.smoother_or_precond == "DILU"
    assert s.coarsest is None


def test_pressure_resolver_rhoPimpleFoam_with_heat_keeps_GAMG() -> None:
    """rhoPimpleFoam + heat is the normal compressible-energy path."""
    s = resolve_pressure_solver_strategy(
        solver_name="rhoPimpleFoam",
        is_compressible=True,
        mesh_tier="good",
        heat_transfer_active=True,
    )
    assert s.top_level == "GAMG"
    assert s.coarsest is not None and s.coarsest.preconditioner == "DIC"


def test_pressure_resolver_poor_mesh_compressible_falls_back() -> None:
    """Poor mesh → direct solver (no GAMG agglomeration risk)."""
    s = resolve_pressure_solver_strategy(
        solver_name="rhoSimpleFoam",
        is_compressible=True,
        mesh_tier="poor",
        heat_transfer_active=True,
    )
    assert s.top_level == "PBiCGStab"
    assert s.smoother_or_precond == "DILU"
    assert s.coarsest is None


def test_pressure_resolver_poor_mesh_incompressible_falls_back() -> None:
    s = resolve_pressure_solver_strategy(
        solver_name="simpleFoam",
        is_compressible=False,
        mesh_tier="poor",
        heat_transfer_active=False,
    )
    assert s.top_level == "PCG"
    assert s.smoother_or_precond == "DIC"


def test_pressure_resolver_coarsest_never_DILU() -> None:
    """Property-style check across all reasonable inputs."""
    for solver in ("simpleFoam", "rhoSimpleFoam", "rhoPimpleFoam", "buoyantSimpleFoam"):
        for tier in ("good", "moderate", "poor", "unknown"):
            for heat in (True, False):
                for compr in (True, False):
                    s = resolve_pressure_solver_strategy(
                        solver_name=solver,
                        is_compressible=compr,
                        mesh_tier=tier,
                        heat_transfer_active=heat,
                    )
                    if s.coarsest is not None:
                        assert s.coarsest.preconditioner != "DILU", (
                            f"DILU at coarsest for {solver}/{tier}/heat={heat}/compr={compr}"
                        )


# ── resolve_compressible_bounds — replaces Check 3c2 ────────────────────────


def test_bounds_gas_loose_window() -> None:
    """Gas bounds size from the *coldest* BC temperature and the inlet pressure.

    At 300 K and 1 atm: ρ = p / (R·T) ≈ 1.177 kg/m³.
    Per the OpenFOAM rhoSimpleFoam reference tutorial shape:
      rho_min = 0.5·ρ ≈ 0.59  (with floor at 0.1)
      rho_max = max(10, 1.5·ρ) → 10 here.
    """
    b = resolve_compressible_bounds(
        is_compressible=True,
        profile="gas",
        rho=1.2,
        bc_temps=[300, 500],
        eos_t_ceiling=None,
        op_p=101325.0,
        mach=0.02,
    )
    # 0.5 × 1.177 ≈ 0.589 (well above the 0.1 floor).
    assert b.rho_min is not None
    assert 0.55 <= b.rho_min <= 0.65
    assert b.rho_max == 10.0
    assert b.transonic is False
    assert b.p_min is not None and b.p_min > 0


def test_bounds_gas_transonic_above_mach_05() -> None:
    b = resolve_compressible_bounds(
        is_compressible=True, profile="gas", rho=1.2,
        bc_temps=[300], eos_t_ceiling=None, op_p=101325.0, mach=0.6,
    )
    assert b.transonic is True


def test_bounds_cryogenic_envelopes_inlet_rho() -> None:
    b = resolve_compressible_bounds(
        is_compressible=True, profile="cryogenic", rho=808.0,
        bc_temps=[77, 200], eos_t_ceiling=248.9, op_p=400000.0, mach=0.0,
    )
    assert b.rho_min == pytest.approx(404.0)
    assert b.rho_max == pytest.approx(1212.0)
    assert b.transonic is False


def test_bounds_incompressible_all_none() -> None:
    b = resolve_compressible_bounds(
        is_compressible=False, profile="gas", rho=None,
        bc_temps=None, eos_t_ceiling=None, op_p=101325.0, mach=0.0,
    )
    assert b.rho_min is None and b.rho_max is None
    assert b.transonic is False


# ── resolve_fv_options_max — replaces Check 3c2 fvOptions clamp ────────────


def test_fv_options_max_cryogenic_uses_90pct_ceiling() -> None:
    m = resolve_fv_options_max(
        profile="cryogenic", bc_temps=[77, 200], eos_t_ceiling=248.9, t_floor=38.5,
    )
    assert m == pytest.approx(248.9 * 0.9)


def test_fv_options_max_gas_uses_15x_bc_capped_at_3000() -> None:
    m = resolve_fv_options_max(
        profile="gas", bc_temps=[300, 500, 600], eos_t_ceiling=None,
    )
    # max(BC) * 1.5 = 900, capped at 3000
    assert m == pytest.approx(900.0)


def test_fv_options_max_gas_caps_at_3000() -> None:
    m = resolve_fv_options_max(
        profile="gas", bc_temps=[2500.0], eos_t_ceiling=None,
    )
    assert m == 3000.0


def test_fv_options_max_no_bc_uses_default() -> None:
    m = resolve_fv_options_max(profile="gas", bc_temps=[], eos_t_ceiling=None)
    # Default BC max = 500 → 500*1.5 = 750
    assert m == pytest.approx(750.0)


# ── resolve_div_phi_h_scheme — replaces Check 7d ───────────────────────────


def test_div_phi_h_upwind_for_large_dT() -> None:
    """ΔT > 100 K — must be upwind to avoid enthalpy overshoot → clamp loop."""
    s = resolve_div_phi_h_scheme(
        is_compressible_energy=True, bc_temps=[280, 600],
    )
    assert "upwind" in s and "linearUpwind" not in s


def test_div_phi_h_default_compressible_energy() -> None:
    s = resolve_div_phi_h_scheme(
        is_compressible_energy=True, bc_temps=[300, 310],
    )
    assert "upwind" in s


def test_div_phi_h_noop_for_incompressible() -> None:
    s = resolve_div_phi_h_scheme(
        is_compressible_energy=False, bc_temps=None,
    )
    assert "upwind" in s
