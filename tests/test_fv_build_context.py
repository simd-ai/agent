"""Tests for the typed FvBuildContext — Phase 3 contract."""

import pytest

from simd_agent.solvers.contexts import FvBuildContext
from simd_agent.solvers.compressible.rhoSimpleFoam.solver import RhoSimpleFoamSolver


# ── Construction & immutability ──────────────────────────────────────────────


def _ctx(**overrides) -> FvBuildContext:
    defaults = dict(
        tier="moderate",
        non_ortho=30.0,
        use_simplec=False,
        n_non_ortho=1,
        vel_mag=4.0,
        speed_tier="low",
        bc_temps=(280.0, 500.0, 600.0),
        profile="gas",
        heat_transfer_active=True,
        turb_model="kOmegaSST",
    )
    defaults.update(overrides)
    return FvBuildContext(**defaults)


def test_context_is_frozen() -> None:
    """FvBuildContext is frozen — mutating a renderer's input is impossible."""
    ctx = _ctx()
    with pytest.raises(Exception):
        ctx.tier = "good"  # type: ignore[misc]


def test_context_delta_t_bc_property() -> None:
    ctx = _ctx(bc_temps=(280.0, 500.0, 600.0))
    assert ctx.delta_t_bc == pytest.approx(320.0)


def test_context_delta_t_bc_single_bc_is_zero() -> None:
    ctx = _ctx(bc_temps=(300.0,))
    assert ctx.delta_t_bc == 0.0


def test_context_is_laminar_property() -> None:
    assert _ctx(turb_model="laminar").is_laminar is True
    assert _ctx(turb_model="kOmegaSST").is_laminar is False


# ── _fv_context returns the typed object ─────────────────────────────────────


def _air_config() -> dict:
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
                "velocity": {"type": "fixedValue", "value": [4.0, 0.0, 0.0]},
                "temperature": {"type": "fixedValue", "value": 500.0},
                "pressure": {"type": "zeroGradient"},
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
        "mesh": {
            "patches": [
                {"name": "inlet", "type": "patch"},
                {"name": "outlet", "type": "patch"},
                {"name": "walls", "type": "wall"},
            ],
        },
    }


def test_fv_context_returns_typed_object() -> None:
    plugin = RhoSimpleFoamSolver()
    ctx = plugin._fv_context(_air_config())
    assert isinstance(ctx, FvBuildContext)
    # Attribute access (not subscript)
    assert ctx.profile == "gas"
    assert ctx.heat_transfer_active is True
    assert ctx.turb_model == "kOmegaSST"
    assert ctx.bc_temps == (500.0, 600.0)
    assert ctx.tier in ("good", "moderate", "poor", "unknown")


def test_fv_context_no_dict_indexing() -> None:
    """Subscripting the context raises — Phase 3 ensures attribute access only."""
    plugin = RhoSimpleFoamSolver()
    ctx = plugin._fv_context(_air_config())
    with pytest.raises(TypeError):
        _ = ctx["tier"]  # type: ignore[index]


def test_fv_context_renders_via_helpers() -> None:
    """End-to-end: plugin builders consume the typed context and emit text."""
    plugin = RhoSimpleFoamSolver()
    fvs = plugin._build_fv_solution(_air_config())
    fvc = plugin._build_fv_schemes(_air_config())
    assert "solvers" in fvs and "SIMPLE" in fvs
    assert "divSchemes" in fvc and "gradSchemes" in fvc
