"""Tests for rhoSimpleFoam dual-profile (gas vs cryogenic) configuration.

Verifies that the deterministic builders emit the right numerics for each
profile, that the profile is correctly detected from the validated config,
and that the rhoSimpleFoam plugin's validators only fire when relevant.
"""

import pytest

from simd_agent.run.case_spec import (
    _select_thermo_profile,
    _thermo_profile_from_config,
    _density_bounds_for_profile,
    _estimate_inlet_mach,
    build_case_spec,
)
from simd_agent.solvers.rhoSimpleFoam.solver import RhoSimpleFoamSolver


# ── Profile selection ─────────────────────────────────────────────────────────


def test_profile_air_room_temp_is_gas() -> None:
    assert _select_thermo_profile("air", 300.0, 1.2, has_heat_transfer=True) == "gas"


def test_profile_ln2_by_name_is_cryogenic() -> None:
    assert (
        _select_thermo_profile("liquid nitrogen", 77.0, 808.0, has_heat_transfer=True)
        == "cryogenic"
    )


def test_profile_lh2_short_name_is_cryogenic() -> None:
    assert (
        _select_thermo_profile("LH2", 20.3, 71.0, has_heat_transfer=True) == "cryogenic"
    )


def test_profile_cold_inlet_triggers_cryogenic_even_unnamed() -> None:
    # No fluid name but inlet < 200 K → cryogenic
    assert _select_thermo_profile(None, 100.0, None, False) == "cryogenic"


def test_profile_water_with_heat_is_cryogenic_via_density() -> None:
    # Liquid (ρ>200) with heat transfer → cryogenic profile (EOS clamps needed)
    assert _select_thermo_profile("water", 300.0, 1000.0, True) == "cryogenic"


def test_profile_air_no_heat_is_gas() -> None:
    assert _select_thermo_profile("air", 300.0, 1.2, False) == "gas"


def test_profile_default_when_nothing_known() -> None:
    assert _select_thermo_profile(None, None, None, False) == "gas"


# ── Config-dict wrapper ───────────────────────────────────────────────────────


def test_thermo_profile_from_config_gas() -> None:
    cfg = {
        "fluid": {"name": "air", "density": 1.2},
        "boundary_conditions": {
            "inlet": {"temperature": {"type": "fixedValue", "value": 300.0}},
        },
        "physics": {"heat_transfer": True},
    }
    assert _thermo_profile_from_config(cfg) == "gas"


def test_thermo_profile_from_config_cryogenic() -> None:
    cfg = {
        "fluid": {"name": "liquid nitrogen", "density": 808.0},
        "boundary_conditions": {
            "inlet": {"temperature": {"type": "fixedValue", "value": 77.0}},
        },
        "physics": {"heat_transfer": True},
    }
    assert _thermo_profile_from_config(cfg) == "cryogenic"


# ── Density bounds ────────────────────────────────────────────────────────────


def test_density_bounds_gas_loose() -> None:
    rmin, rmax = _density_bounds_for_profile("gas", 1.2, None, None)
    assert rmin == 0.1 and rmax == 10.0


def test_density_bounds_cryogenic_envelope() -> None:
    rmin, rmax = _density_bounds_for_profile("cryogenic", 808.0, 248.9, [77.0, 400.0])
    assert rmin == pytest.approx(404.0)
    assert rmax == pytest.approx(1212.0)


def test_density_bounds_no_rho_returns_none_for_cryogenic() -> None:
    assert _density_bounds_for_profile("cryogenic", None, None, None) == (None, None)


# ── Mach estimate ─────────────────────────────────────────────────────────────


def test_mach_subsonic_air() -> None:
    m = _estimate_inlet_mach("gas", [50.0, 0.0, 0.0], 300.0)
    # |U|=50, a≈347 → M≈0.14
    assert 0.10 < m < 0.20


def test_mach_cryogenic_is_zero() -> None:
    assert _estimate_inlet_mach("cryogenic", [10.0, 0.0, 0.0], 77.0) == 0.0


def test_mach_no_velocity_zero() -> None:
    assert _estimate_inlet_mach("gas", [], 300.0) == 0.0


# ── Builder integration: gas profile fvSolution ───────────────────────────────


def _air_config() -> dict:
    return {
        "fluid": {"name": "air", "density": 1.2, "cp": 1005, "mu": 1.8e-5, "Pr": 0.71},
        "physics": {
            "compressibility": "compressible",
            "heat_transfer": True,
            "time_scheme": "steady",
            "turbulence_model": "kOmegaSST",
            "flow_regime": "turbulent",
        },
        "boundary_conditions": {
            "inlet": {
                "patch_type": "inlet",
                "velocity": {"type": "fixedValue", "value": [10.0, 0.0, 0.0]},
                "temperature": {"type": "fixedValue", "value": 300.0},
                "pressure": {"type": "zeroGradient"},
            },
            "outlet": {
                "patch_type": "outlet",
                "pressure": {"type": "fixedValue", "value": 101325.0},
            },
            "walls": {
                "patch_type": "wall",
                "temperature": {"type": "fixedValue", "value": 400.0},
            },
        },
        "mesh": {
            "patches": [
                {"name": "inlet", "type": "patch"},
                {"name": "outlet", "type": "patch"},
                {"name": "walls", "type": "wall"},
            ],
        },
    }


def _ln2_config() -> dict:
    cfg = _air_config()
    cfg["fluid"] = {"name": "liquid nitrogen", "density": 808.0, "cp": 2042, "mu": 1.58e-4, "Pr": 1.0}
    cfg["boundary_conditions"]["inlet"]["temperature"]["value"] = 77.0
    cfg["boundary_conditions"]["walls"]["temperature"]["value"] = 200.0
    return cfg


def test_fvsolution_gas_has_rhoMin_rhoMax() -> None:
    plugin = RhoSimpleFoamSolver()
    fvs = plugin._build_fv_solution(_air_config())
    assert "rhoMin" in fvs and "rhoMax" in fvs
    # Gas relaxation: e=0.5, U=0.7 (rhoSimpleFoam uses ``e``, the
    # OF-tutorial-aligned ``sensibleInternalEnergy`` variable).
    assert "e               0.5" in fvs
    assert "U               0.7" in fvs


def test_fvsolution_cryogenic_relaxation_e_low() -> None:
    plugin = RhoSimpleFoamSolver()
    fvs = plugin._build_fv_solution(_ln2_config())
    # Cryogenic relaxation also targets the rhoSimpleFoam energy var ``e``:
    # e=0.05, U=0.5.
    assert "e               0.05" in fvs
    assert "U               0.5" in fvs
    # No SIMPLEC in cryogenic
    assert "consistent      yes" not in fvs


def test_fvschemes_gas_uses_bounded_upwind_for_div_phi_U() -> None:
    """SIMPLE-mode compressible always uses plain ``upwind`` for div(phi,U).

    Matches the OpenFOAM reference rhoSimpleFoam tutorial choice
    (``compressible/rhoSimpleFoam/angledDuctExplicitFixedCoeff``).  The
    accuracy-preferred ``linearUpwindV grad(U)`` is reserved for PIMPLE
    (rhoPimpleFoam) where the Δt absorbs the acoustic overshoot.
    """
    plugin = RhoSimpleFoamSolver()
    fvs = plugin._build_fv_schemes(_air_config())
    assert "div(phi,U)      bounded Gauss upwind" in fvs
    # cellLimited on grad(U) is independent of the div(phi,U) choice — kept
    # for non-orthogonal meshes.
    assert "cellLimited Gauss linear 1" in fvs


def test_fvschemes_cryogenic_uses_upwind_only() -> None:
    plugin = RhoSimpleFoamSolver()
    fvs = plugin._build_fv_schemes(_ln2_config())
    # Cryogenic forces div(phi,U) upwind, no cellLimited
    assert "div(phi,U)      bounded Gauss upwind" in fvs
    assert "cellLimited" not in fvs


# ── CaseSpec carries profile fields ──────────────────────────────────────────


def test_case_spec_includes_profile_gas() -> None:
    spec = build_case_spec("rhoSimpleFoam", _air_config())
    assert spec.thermo_profile == "gas"
    assert spec.rho_min == 0.1
    assert spec.rho_max == 10.0
    assert spec.transonic is False  # 10 m/s air → M≈0.03


def test_case_spec_includes_profile_cryogenic() -> None:
    spec = build_case_spec("rhoSimpleFoam", _ln2_config())
    assert spec.thermo_profile == "cryogenic"
    assert spec.rho_min is not None and spec.rho_min > 0
    assert spec.rho_max is not None and spec.rho_max > spec.rho_min
    assert spec.transonic is False  # cryogenic: liquid sound speed dominates


# ── Regression: coarsestLevelCorr preconditioner must be valid for symmetric matrix


def _extract_coarsest_block(fvs: str) -> str:
    """Return the coarsestLevelCorr block body as a single string."""
    import re
    m = re.search(r"coarsestLevelCorr\s*\{([^}]*)\}", fvs, re.DOTALL)
    assert m, "coarsestLevelCorr block not present in fvSolution"
    return m.group(1)


def test_coarsest_never_uses_DILU() -> None:
    """OpenFOAM 2406 rejects DILU as a coarsestLevelCorr preconditioner.

    The agglomerated coarsest GAMG matrix is symmetric — preconditioner
    must come from {DIC, FDIC, GAMG, diagonal, distributedDIC, none}.
    Using DILU triggers: 'Unknown symmetric matrix preconditioner type DILU'.
    """
    plugin = RhoSimpleFoamSolver()
    fvs = plugin._build_fv_solution(_air_config())
    body = _extract_coarsest_block(fvs)
    assert "DILU" not in body, (
        "coarsestLevelCorr contains DILU; OpenFOAM will reject it at runtime"
    )


def test_compressible_coarsest_uses_PCG_DIC() -> None:
    """At the coarsest agglomerated level, the matrix is symmetric for any
    solver — use PCG+DIC for fast (3–5 iter) convergence.  'preconditioner
    none' would converge in ~30 iter, ~5× slower."""
    plugin = RhoSimpleFoamSolver()
    fvs = plugin._build_fv_solution(_air_config())
    body = _extract_coarsest_block(fvs)
    assert "PCG" in body
    assert "DIC" in body
    # The historical "preconditioner none" workaround is no longer used —
    # nCoarsestCells 500 prevents the over-agglomeration that originally
    # motivated it.
    assert "preconditioner  none" not in body


def test_incompressible_coarsest_uses_PCG_DIC() -> None:
    from simd_agent.solvers.simpleFoam.solver import SimpleFoamSolver
    plugin = SimpleFoamSolver()
    cfg = _air_config()
    cfg["physics"]["compressibility"] = "incompressible"
    cfg["physics"]["heat_transfer"] = False
    fvs = plugin._build_fv_solution(cfg)
    body = _extract_coarsest_block(fvs)
    assert "PCG" in body
    assert "DIC" in body and "DILU" not in body


# ── Inlet turbulence unification ─────────────────────────────────────────────


def _two_inlet_air_config() -> dict:
    """Air config with two inlets at different velocities — like the U-bend test."""
    return {
        "fluid": {"name": "air", "density": 1.2, "cp": 1005, "mu": 1.81e-5, "Pr": 0.713},
        "physics": {
            "compressibility": "compressible",
            "heat_transfer": True,
            "time_scheme": "steady",
            "turbulence_model": "kOmegaSST",
            "flow_regime": "turbulent",
        },
        "boundary_conditions": {
            "inlet_main": {
                "patch_type": "inlet",
                "velocity": {"type": "fixedValue", "value": [4.0, 0.0, 0.0]},
                "temperature": {"type": "fixedValue", "value": 500.0},
            },
            "inlet_small": {
                "patch_type": "inlet",
                "velocity": {"type": "fixedValue", "value": [1.0, 0.0, 0.0]},
                "temperature": {"type": "fixedValue", "value": 280.0},
            },
            "outlet": {
                "patch_type": "outlet",
                "pressure": {"type": "fixedValue", "value": 101325.0},
            },
            "walls": {
                "patch_type": "wall",
                "temperature": {"type": "fixedValue", "value": 600.0},
            },
        },
        "turbulence_initial_values": {"k": 0.06, "omega": 44.7},
    }


def test_inlet_k_preserved_when_implied_TI_is_in_reasonable_band() -> None:
    """When per-patch TI is not explicitly set but the LLM's k values imply a
    reasonable TI (between 0.1% and 30%), the validator respects them.

    Rationale: different TI per inlet is a legitimate CFD scenario (turbulent
    jet vs. settled coflow).  Without an explicit signal from the user, we
    trust the LLM's choice as long as it's physically plausible.
    """
    plugin = RhoSimpleFoamSolver()
    cfg = _two_inlet_air_config()
    cfg["mesh"] = {
        "check_mesh": {"bounding_box": {"min": [0, 0, 0], "max": [0.175, 0.198, 0.08]}}
    }
    # inlet_main: U=4, k=0.06 → implied TI=5%
    # inlet_small: U=1, k=0.00015 → implied TI=1% (in [0.1%, 30%] band)
    plausible_k = """\
FoamFile { object k; }
internalField uniform 0.06;
boundaryField
{
    inlet_main { type fixedValue; value uniform 0.06; }
    inlet_small { type fixedValue; value uniform 0.00015; }
    walls { type kqRWallFunction; value uniform 0.06; }
}
"""
    issues: list = []
    fixed = plugin._unify_inlet_turbulence({"0/k": plausible_k}, issues, cfg)
    body = fixed["0/k"]
    # Both values preserved
    import re as _re
    k_main = float(_re.search(
        r"inlet_main\s*\{[^}]*value\s+uniform\s+([\d.eE+\-]+)", body, _re.DOTALL
    ).group(1))
    k_small = float(_re.search(
        r"inlet_small\s*\{[^}]*value\s+uniform\s+([\d.eE+\-]+)", body, _re.DOTALL
    ).group(1))
    assert abs(k_main - 0.06) / 0.06 < 0.05
    assert abs(k_small - 0.00015) / 0.00015 < 0.10


def test_inlet_k_clamped_when_implied_TI_is_absurd() -> None:
    """If the LLM produced k that implies an absurd TI (e.g. >30% or <0.1%),
    the validator falls back to the global default (5%) and recomputes."""
    plugin = RhoSimpleFoamSolver()
    cfg = _two_inlet_air_config()
    cfg["mesh"] = {
        "check_mesh": {"bounding_box": {"min": [0, 0, 0], "max": [0.175, 0.198, 0.08]}}
    }
    # inlet_small k=1.0, U=1 → implied TI = √(2/3) / 1 = 0.816 (81.6%) — absurd
    absurd_k = """\
FoamFile { object k; }
internalField uniform 0.06;
boundaryField
{
    inlet_main { type fixedValue; value uniform 0.06; }
    inlet_small { type fixedValue; value uniform 1.0; }
    walls { type kqRWallFunction; value uniform 0.06; }
}
"""
    issues: list = []
    fixed = plugin._unify_inlet_turbulence({"0/k": absurd_k}, issues, cfg)
    body = fixed["0/k"]
    import re as _re
    k_small = float(_re.search(
        r"inlet_small\s*\{[^}]*value\s+uniform\s+([\d.eE+\-]+)", body, _re.DOTALL
    ).group(1))
    # Fallback to TI=5%: 1.5·(1·0.05)² = 0.00375
    assert abs(k_small - 0.00375) / 0.00375 < 0.05, (
        f"k_small = {k_small}, expected ≈ 0.00375 after fallback"
    )


def test_inlet_omega_recomputed_per_inlet_with_shared_L() -> None:
    """omega differs per inlet because k differs; L is shared."""
    plugin = RhoSimpleFoamSolver()
    cfg = _two_inlet_air_config()
    cfg["mesh"] = {
        "check_mesh": {"bounding_box": {"min": [0, 0, 0], "max": [0.175, 0.198, 0.08]}}
    }
    inconsistent_omega = """\
FoamFile { object omega; }
internalField uniform 44.7;
boundaryField
{
    inlet_main { type fixedValue; value uniform 44.7; }
    inlet_small { type fixedValue; value uniform 2; }
    walls { type omegaWallFunction; value uniform 44.7; }
}
"""
    issues: list = []
    fixed = plugin._unify_inlet_turbulence({"0/omega": inconsistent_omega}, issues, cfg)
    body = fixed["0/omega"]
    import re as _re, math
    # Verify the physical relation ω_main / ω_small = √(k_main/k_small) = √(U_main/U_small)² = 4
    om_main = float(_re.search(
        r"inlet_main\s*\{[^}]*value\s+uniform\s+([\d.eE+\-]+)", body, _re.DOTALL
    ).group(1))
    om_small = float(_re.search(
        r"inlet_small\s*\{[^}]*value\s+uniform\s+([\d.eE+\-]+)", body, _re.DOTALL
    ).group(1))
    assert om_small < om_main, "ω_small should be smaller (lower velocity)"
    ratio = om_main / om_small
    # Physical ratio: ω ∝ √k ∝ U → ratio should be U_main/U_small = 4
    assert abs(ratio - 4.0) / 4.0 < 0.05, f"ω ratio = {ratio}, expected ≈ 4"


def test_inlet_turbulence_honors_user_specified_per_patch_TI() -> None:
    """When the BC specifies turbulenceIntensity per patch, the validator uses it.

    Scenario: a turbulent jet (TI=10%) mixing with a settled coflow (TI=1%).
    The two inlets should NOT share a TI — each gets its own k from
    k = 1.5·(U·I_patch)².
    """
    plugin = RhoSimpleFoamSolver()
    cfg = _two_inlet_air_config()
    cfg["mesh"] = {
        "check_mesh": {"bounding_box": {"min": [0, 0, 0], "max": [0.175, 0.198, 0.08]}}
    }
    # Override per-patch TI:
    #   inlet_main is a turbulent jet — TI=10%
    #   inlet_small is settled coflow — TI=1%
    cfg["boundary_conditions"]["inlet_main"]["turbulenceIntensity"] = 0.10
    cfg["boundary_conditions"]["inlet_small"]["turbulenceIntensity"] = 0.01

    bad_k = """\
FoamFile { object k; }
internalField uniform 0.06;
boundaryField
{
    inlet_main { type fixedValue; value uniform 0.5; }
    inlet_small { type fixedValue; value uniform 0.5; }
    walls { type kqRWallFunction; value uniform 0.06; }
}
"""
    issues: list = []
    fixed = plugin._unify_inlet_turbulence({"0/k": bad_k}, issues, cfg)
    body = fixed["0/k"]

    import re as _re
    # Expected: k_main = 1.5·(4·0.10)² = 0.24
    k_main = float(_re.search(
        r"inlet_main\s*\{[^}]*value\s+uniform\s+([\d.eE+\-]+)", body, _re.DOTALL
    ).group(1))
    assert abs(k_main - 0.24) / 0.24 < 0.05, f"k_main = {k_main}, expected ≈ 0.24 (TI=10%)"

    # Expected: k_small = 1.5·(1·0.01)² = 0.00015
    k_small = float(_re.search(
        r"inlet_small\s*\{[^}]*value\s+uniform\s+([\d.eE+\-]+)", body, _re.DOTALL
    ).group(1))
    assert abs(k_small - 0.00015) / 0.00015 < 0.10, (
        f"k_small = {k_small}, expected ≈ 0.00015 (TI=1%)"
    )

    # Sanity: with these per-patch TIs, k_main / k_small = (U·TI)²_main / (U·TI)²_small
    # = (4·0.1)² / (1·0.01)² = 0.16 / 0.0001 = 1600
    assert abs(k_main / k_small - 1600.0) / 1600.0 < 0.10


def test_validator_reverse_engineers_TI_when_BC_lacks_it() -> None:
    """If a BC has no explicit TI but the LLM already produced sensible per-inlet
    k values, the validator should reverse-engineer the per-patch TI rather than
    defaulting to 5% and overwriting the LLM's intent.
    """
    plugin = RhoSimpleFoamSolver()
    cfg = _two_inlet_air_config()
    cfg["mesh"] = {
        "check_mesh": {"bounding_box": {"min": [0, 0, 0], "max": [0.175, 0.198, 0.08]}}
    }
    # No turbulenceIntensity on either BC — LLM picked TI=8% for both, k matches
    # k_main = 1.5·(4·0.08)² = 0.1536; k_small = 1.5·(1·0.08)² = 0.0096
    consistent_at_8pct = """\
FoamFile { object k; }
internalField uniform 0.1536;
boundaryField
{
    inlet_main { type fixedValue; value uniform 0.1536; }
    inlet_small { type fixedValue; value uniform 0.0096; }
    walls { type kqRWallFunction; value uniform 0.1536; }
}
"""
    issues: list = []
    fixed = plugin._unify_inlet_turbulence({"0/k": consistent_at_8pct}, issues, cfg)
    # Values should be preserved (the implied per-patch TI of 8% is reasonable)
    assert "0.1536" in fixed["0/k"]
    assert "0.0096" in fixed["0/k"]


def test_heuristic_picks_rhoSimpleFoam_for_hot_air_no_gravity() -> None:
    """Regression — the original symptom was simpleFoam being chosen for
    a hot-air heated-wall case because the user said 'without gravity'.

    Decision should be: heat transfer active + no gravity → rhoSimpleFoam
    (forced convection — simpleFoam can't solve energy).
    """
    from simd_agent.run.solver_selector import _heuristic_fallback
    cfg = {
        "physics": {
            "compressibility": "incompressible",  # precheck didn't catch ΔT-driven density variation
            "heat_transfer": False,                # not explicitly flagged either
            "time_scheme": "steady",
            "gravity": False,                       # user said "without gravity"
            "flow_regime": "turbulent",
        },
        "boundary_conditions": {
            "inlet_main": {
                "patch_type": "inlet",
                "temperature": {"type": "fixedValue", "value": 500.0},
            },
            "inlet_small": {
                "patch_type": "inlet",
                "temperature": {"type": "fixedValue", "value": 280.0},
            },
            "outlet": {
                "patch_type": "outlet",
                "pressure": {"type": "fixedValue", "value": 101325.0},
            },
            "walls": {
                "patch_type": "wall",
                "temperature": {"type": "fixedValue", "value": 600.0},
            },
        },
    }
    assert _heuristic_fallback(cfg) == "rhoSimpleFoam"


def test_heuristic_picks_buoyantSimpleFoam_when_gravity_on() -> None:
    """Same physics but with gravity → buoyant solver."""
    from simd_agent.run.solver_selector import _heuristic_fallback
    cfg = {
        "physics": {
            "compressibility": "incompressible",
            "heat_transfer": True,
            "time_scheme": "steady",
            "gravity": True,
            "flow_regime": "turbulent",
        },
        "boundary_conditions": {
            "inlet": {
                "patch_type": "inlet",
                "temperature": {"type": "fixedValue", "value": 300.0},
            },
            "walls": {
                "patch_type": "wall",
                "temperature": {"type": "fixedValue", "value": 600.0},
            },
        },
    }
    assert _heuristic_fallback(cfg) == "buoyantSimpleFoam"


def test_heuristic_picks_simpleFoam_for_isothermal_low_mach() -> None:
    """Truly isothermal (no BC temperature gradient) → simpleFoam."""
    from simd_agent.run.solver_selector import _heuristic_fallback
    cfg = {
        "physics": {
            "compressibility": "incompressible",
            "heat_transfer": False,
            "time_scheme": "steady",
            "gravity": False,
            "flow_regime": "turbulent",
        },
        "boundary_conditions": {
            "inlet": {
                "patch_type": "inlet",
                "velocity": {"type": "fixedValue", "value": [1.0, 0.0, 0.0]},
            },
            "outlet": {
                "patch_type": "outlet",
                "pressure": {"type": "fixedValue", "value": 0.0},
            },
            "walls": {"patch_type": "wall"},
        },
    }
    assert _heuristic_fallback(cfg) == "simpleFoam"


def test_heuristic_detects_heat_from_BC_temperature_spread() -> None:
    """Even if heat_transfer flag is missing, BC temperature spread → heat=True."""
    from simd_agent.run.solver_selector import _extract_flags
    cfg = {
        "physics": {"compressibility": "incompressible"},
        "boundary_conditions": {
            "inlet": {
                "patch_type": "inlet",
                "temperature": {"type": "fixedValue", "value": 280.0},
            },
            "walls": {
                "patch_type": "wall",
                "temperature": {"type": "fixedValue", "value": 600.0},
            },
        },
    }
    flags = _extract_flags(cfg)
    assert flags["heat"] is True


def test_fvoptions_max_clamp_does_not_crash_re_sub() -> None:
    """Regression — Check 3c2 clamping fvOptions.max must not crash.

    Earlier the re.sub call passed (pattern, string) without a replacement,
    triggering 'sub() missing 1 required positional argument: string'.
    """
    from simd_agent.run.genai_codegen import validate_generated_files
    files = {
        "system/fvOptions": (
            "FoamFile { object fvOptions; }\n"
            "temperatureLimiter\n"
            "{\n"
            "    type            limitTemperature;\n"
            "    selectionMode   all;\n"
            "    min             140.0;\n"
            "    max             100000;   // LLM placeholder\n"
            "}\n"
        ),
        # Minimal companion files so the validator doesn't trip on other checks
        "system/controlDict": "application buoyantSimpleFoam;\n",
        "0/T": "internalField uniform 280;\nboundaryField{ walls{type fixedValue; value uniform 600;}}\n",
    }
    cfg = _two_inlet_air_config()
    # No icoPolynomial path — perfectGas (gas profile) — should hit the
    # gas-profile branch (no EOS ceiling).
    fixed, issues = validate_generated_files(files, "buoyantSimpleFoam", cfg)
    # No crash. fvOptions max should be sensible now (≤ 3000 K, not 100000).
    import re as _re
    m = _re.search(r"\bmax\s+(\d+)\s*;", fixed["system/fvOptions"])
    assert m, "max line missing after clamp"
    new_max = int(m.group(1))
    assert new_max <= 3000, f"max={new_max} not clamped (expected ≤ 3000)"


def test_inlet_turbulence_noop_when_already_consistent_per_inlet() -> None:
    """If both inlets already use the same TI, no change is made."""
    plugin = RhoSimpleFoamSolver()
    cfg = _two_inlet_air_config()
    cfg["mesh"] = {
        "check_mesh": {"bounding_box": {"min": [0, 0, 0], "max": [0.175, 0.198, 0.08]}}
    }
    # inlet_main U=4 → k=0.06; inlet_small U=1 → k=0.00375 (both TI=5%)
    consistent = """\
FoamFile { object k; }
internalField uniform 0.06;
boundaryField
{
    inlet_main { type fixedValue; value uniform 0.06; }
    inlet_small { type fixedValue; value uniform 0.00375; }
    walls { type kqRWallFunction; value uniform 0.06; }
}
"""
    issues: list = []
    fixed = plugin._unify_inlet_turbulence({"0/k": consistent}, issues, cfg)
    assert not any("Recomputed inlet k" in i.message for i in issues)
    assert fixed["0/k"] == consistent
