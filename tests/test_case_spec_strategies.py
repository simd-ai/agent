"""Tests for the Phase 1 typed strategy sub-models in case_spec.strategies.

These verify that invalid CFD combinations raise ValidationError at
construction time — bug classes that disappear by construction.
"""

import math
import pytest
from pydantic import ValidationError

from simd_agent.run.case_spec import (
    CaseSpec,
    CoarsestLevelCorr,
    CompressibleBounds,
    FluidThermo,
    InletTurbulence,
    PressureSolverStrategy,
    build_case_spec,
)


# ── FluidThermo ──────────────────────────────────────────────────────────────


def test_fluid_thermo_perfectGas_const_hConst_air() -> None:
    """Standard rhoSimpleFoam gas recipe — must accept."""
    t = FluidThermo(
        package="hePsiThermo",
        eos="perfectGas",
        transport="const",
        thermo="hConst",
        energy="sensibleEnthalpy",
    )
    assert t.eos == "perfectGas"


def test_fluid_thermo_icoPolynomial_requires_polynomial_transport() -> None:
    """icoPolynomial + const transport throws 'Unknown fluidThermo type' in OpenFOAM —
    we make the combination unconstructable."""
    with pytest.raises(ValidationError, match="icoPolynomial requires transport='polynomial'"):
        FluidThermo(
            package="heRhoThermo",
            eos="icoPolynomial",
            transport="const",          # ← invalid
            thermo="hConst",
            energy="sensibleEnthalpy",
        )


def test_fluid_thermo_icoPolynomial_requires_hPolynomial_thermo() -> None:
    with pytest.raises(ValidationError, match="thermo='hPolynomial' or 'ePolynomial'"):
        FluidThermo(
            package="heRhoThermo",
            eos="icoPolynomial",
            transport="polynomial",
            thermo="hConst",            # ← invalid
            energy="sensibleEnthalpy",
        )


def test_fluid_thermo_valid_cryogenic_recipe() -> None:
    """LN2 / LH2 / LOX cryogenic recipe — must accept."""
    t = FluidThermo(
        package="heRhoThermo",
        eos="icoPolynomial",
        transport="polynomial",
        thermo="hPolynomial",
        energy="sensibleEnthalpy",
    )
    assert t.transport == "polynomial"
    assert t.thermo == "hPolynomial"


def test_fluid_thermo_is_frozen() -> None:
    t = FluidThermo(
        package="hePsiThermo",
        eos="perfectGas",
        transport="const",
        thermo="hConst",
        energy="sensibleEnthalpy",
    )
    with pytest.raises(ValidationError):
        t.eos = "icoPolynomial"  # type: ignore[misc]


# ── CoarsestLevelCorr ────────────────────────────────────────────────────────


def test_coarsest_default_is_PCG_DIC() -> None:
    """The default — fast and OpenFOAM-valid."""
    c = CoarsestLevelCorr()
    assert c.solver == "PCG"
    assert c.preconditioner == "DIC"


def test_coarsest_rejects_DILU() -> None:
    """DILU is the historical bug — must not be representable here."""
    with pytest.raises(ValidationError):
        CoarsestLevelCorr(preconditioner="DILU")  # type: ignore[arg-type]


def test_coarsest_accepts_none_preconditioner() -> None:
    """The defensive fallback — valid on any matrix."""
    c = CoarsestLevelCorr(solver="PBiCGStab", preconditioner="none")
    assert c.preconditioner == "none"


def test_coarsest_rejects_unknown_solver() -> None:
    with pytest.raises(ValidationError):
        CoarsestLevelCorr(solver="NotARealSolver")  # type: ignore[arg-type]


# ── PressureSolverStrategy ───────────────────────────────────────────────────


def test_pressure_strategy_GAMG_requires_coarsest() -> None:
    with pytest.raises(ValidationError, match="GAMG requires a coarsestLevelCorr"):
        PressureSolverStrategy(top_level="GAMG", coarsest=None)


def test_pressure_strategy_GAMG_with_default_coarsest_ok() -> None:
    s = PressureSolverStrategy(
        top_level="GAMG",
        smoother_or_precond="GaussSeidel",
        coarsest=CoarsestLevelCorr(),
    )
    assert s.coarsest is not None
    assert s.coarsest.solver == "PCG"


def test_pressure_strategy_PCG_rejects_coarsest() -> None:
    with pytest.raises(ValidationError, match="coarsestLevelCorr is only valid when top_level='GAMG'"):
        PressureSolverStrategy(
            top_level="PCG",
            smoother_or_precond="DIC",
            coarsest=CoarsestLevelCorr(),
        )


def test_pressure_strategy_direct_path_PBiCGStab_DILU() -> None:
    """Top-level direct solver on an asymmetric (compressible) matrix — valid."""
    s = PressureSolverStrategy(
        top_level="PBiCGStab",
        smoother_or_precond="DILU",
    )
    assert s.coarsest is None


# ── CompressibleBounds ───────────────────────────────────────────────────────


def test_bounds_min_lt_max() -> None:
    b = CompressibleBounds(rho_min=0.1, rho_max=10.0)
    assert b.rho_min == 0.1
    assert b.rho_max == 10.0


def test_bounds_reject_inverted_rho() -> None:
    with pytest.raises(ValidationError, match="rho_min .* must be strictly less than rho_max"):
        CompressibleBounds(rho_min=5.0, rho_max=1.0)


def test_bounds_reject_inverted_p() -> None:
    with pytest.raises(ValidationError, match="p_min .* must be strictly less than p_max"):
        CompressibleBounds(p_min=2e5, p_max=1e5)


def test_bounds_all_none_is_valid() -> None:
    """An empty/no-op bounds object is allowed — used by incompressible paths."""
    b = CompressibleBounds()
    assert b.rho_min is None
    assert b.transonic is False


# ── InletTurbulence ──────────────────────────────────────────────────────────


def test_inlet_turbulence_k_omega_epsilon_derived() -> None:
    """k = 1.5·(U·I)²; ω = √k / (Cμ^0.25·L); ε = Cμ^0.75 · k^1.5 / L."""
    inlet = InletTurbulence(
        patch_name="inlet_main",
        velocity_mag=4.0,
        intensity=0.05,
        length_scale=0.07 * 0.04,  # L = 0.07 · D_h
    )
    # k = 1.5 · (4 · 0.05)² = 0.06
    assert inlet.k == pytest.approx(0.06)
    # ω and ε derived from k and L — check they're > 0 and sensible
    assert inlet.omega > 0.0
    assert inlet.epsilon > 0.0
    # ω formula: √0.06 / (0.09^0.25 · 0.0028) ≈ 159
    expected_omega = math.sqrt(0.06) / (0.09 ** 0.25 * 0.0028)
    assert inlet.omega == pytest.approx(expected_omega, rel=1e-6)


def test_inlet_turbulence_intensity_bounds() -> None:
    # Below 0.1% — rejected
    with pytest.raises(ValidationError):
        InletTurbulence(patch_name="x", velocity_mag=1, intensity=0.0001, length_scale=0.01)
    # Above 30% — rejected
    with pytest.raises(ValidationError):
        InletTurbulence(patch_name="x", velocity_mag=1, intensity=0.5, length_scale=0.01)


def test_inlet_turbulence_zero_velocity_rejected() -> None:
    with pytest.raises(ValidationError):
        InletTurbulence(patch_name="x", velocity_mag=0.0, intensity=0.05, length_scale=0.01)


# ── CaseSpec carries the strategy slots (Phase 1: still defaulted) ──────────


def _minimal_air_cfg() -> dict:
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
            "inlet": {
                "patch_type": "inlet",
                "velocity": {"type": "fixedValue", "value": [5.0, 0.0, 0.0]},
                "temperature": {"type": "fixedValue", "value": 300.0},
                "pressure": {"type": "zeroGradient"},
            },
            "outlet": {
                "patch_type": "outlet",
                "pressure": {"type": "fixedValue", "value": 101325.0},
            },
            "walls": {
                "patch_type": "wall",
                "temperature": {"type": "fixedValue", "value": 500.0},
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


def test_case_spec_has_strategy_slots_defaulted_to_none() -> None:
    """Phase 1 contract: the slots exist on CaseSpec but are None / []."""
    spec = build_case_spec("rhoSimpleFoam", _minimal_air_cfg())
    assert spec.thermo_strategy is None
    assert spec.pressure_solver_strategy is None
    assert spec.compressible_bounds_strategy is None
    assert spec.inlet_turbulence_strategy == []


def test_case_spec_accepts_strategy_assignment() -> None:
    """A future resolver can fill the slots — CaseSpec doesn't reject them."""
    spec = build_case_spec("rhoSimpleFoam", _minimal_air_cfg())
    spec.thermo_strategy = FluidThermo(
        package="hePsiThermo",
        eos="perfectGas",
        transport="const",
        thermo="hConst",
        energy="sensibleEnthalpy",
    )
    spec.pressure_solver_strategy = PressureSolverStrategy(
        top_level="GAMG",
        smoother_or_precond="GaussSeidel",
        coarsest=CoarsestLevelCorr(),
    )
    spec.compressible_bounds_strategy = CompressibleBounds(rho_min=0.1, rho_max=10.0)
    spec.inlet_turbulence_strategy = [
        InletTurbulence(
            patch_name="inlet",
            velocity_mag=5.0,
            intensity=0.05,
            length_scale=0.005,
        ),
    ]
    # Roundtrip OK
    assert spec.thermo_strategy.eos == "perfectGas"
    assert spec.pressure_solver_strategy.coarsest.preconditioner == "DIC"
    assert spec.compressible_bounds_strategy.rho_max == 10.0
    # k = 1.5·(5·0.05)² = 0.09375
    assert spec.inlet_turbulence_strategy[0].k == pytest.approx(0.09375)


def test_backward_compat_imports() -> None:
    """All public names previously exposed by case_spec.py are still importable
    from the package root for byte-for-byte backward compatibility."""
    from simd_agent.run.case_spec import (
        CaseSpec as _C,
        build_case_spec as _b,
        _select_thermo_profile,
        _thermo_profile_from_config,
        _density_bounds_for_profile,
        _estimate_inlet_mach,
        _mesh_quality_decisions,
        _SOLVER_PROPS,
        _TURB_FIELDS,
    )
    assert _C is CaseSpec
    assert _b is build_case_spec
    assert "rhoSimpleFoam" in _SOLVER_PROPS
    assert "kOmegaSST" in _TURB_FIELDS
    assert _select_thermo_profile("air", 300, 1.2, False) == "gas"
    assert _select_thermo_profile("LN2", 77, 808, True) == "cryogenic"
    assert _mesh_quality_decisions(None)["mesh_quality_tier"] == "unknown"
    assert _density_bounds_for_profile("gas", 1.2, None, None) == (0.1, 10.0)
    assert _estimate_inlet_mach("cryogenic", [1.0, 0, 0], 77) == 0.0
    assert _thermo_profile_from_config({
        "fluid": {"name": "air"},
        "boundary_conditions": {},
        "physics": {},
    }) == "gas"
