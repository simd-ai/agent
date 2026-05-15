"""End-to-end tests for the turbulence-model chain across all 6 solvers.

These verify the regression that demoted `rhoSimpleFoam` to `simulationType
laminar` (and caused the SIGFPE cascade) cannot reappear: the precheck-shaped
config carries the turbulence model into the CaseSpec, the renderer reads
it from the resolver, and laminar is never the silent fallback.
"""

import pytest

from simd_agent.run.case_spec import (
    TurbulenceSpec,
    build_case_spec,
    resolve_turbulence_spec,
)
from simd_agent.run.normalizer import _normalize_physics
from simd_agent.solvers.heatTransfer.buoyantPimpleFoam.solver import BuoyantPimpleFoamSolver
from simd_agent.solvers.heatTransfer.buoyantSimpleFoam.solver import BuoyantSimpleFoamSolver
from simd_agent.solvers.incompressible.pimpleFoam.solver import PimpleFoamSolver
from simd_agent.solvers.compressible.rhoPimpleFoam.solver import RhoPimpleFoamSolver
from simd_agent.solvers.compressible.rhoSimpleFoam.solver import RhoSimpleFoamSolver
from simd_agent.solvers.incompressible.simpleFoam.solver import SimpleFoamSolver

ALL_PLUGINS = [
    SimpleFoamSolver,
    PimpleFoamSolver,
    RhoSimpleFoamSolver,
    RhoPimpleFoamSolver,
    BuoyantSimpleFoamSolver,
    BuoyantPimpleFoamSolver,
]


# ── TurbulenceSpec invariants ────────────────────────────────────────────


def test_turbulence_spec_rejects_mixed_laminar() -> None:
    """flow_regime=turbulent + model=laminar — invalid by construction."""
    with pytest.raises(ValueError, match="either both are laminar"):
        TurbulenceSpec(
            flow_regime="turbulent",
            model="laminar",
            simulation_type="RAS",
        )


def test_turbulence_spec_rejects_laminar_with_RAS_sim_type() -> None:
    with pytest.raises(ValueError, match="simulation_type must match"):
        TurbulenceSpec(
            flow_regime="laminar",
            model="laminar",
            simulation_type="RAS",
        )


def test_turbulence_spec_LES_requires_LES_model() -> None:
    with pytest.raises(ValueError, match="LES-family"):
        TurbulenceSpec(
            flow_regime="turbulent",
            model="kOmegaSST",  # not LES
            simulation_type="LES",
        )


def test_turbulence_spec_valid_RAS_kOmegaSST() -> None:
    spec = TurbulenceSpec(
        flow_regime="turbulent",
        model="kOmegaSST",
        simulation_type="RAS",
    )
    assert spec.model == "kOmegaSST"
    assert spec.wall_functions is True


def test_turbulence_spec_valid_laminar() -> None:
    spec = TurbulenceSpec(
        flow_regime="laminar",
        model="laminar",
        simulation_type="laminar",
    )
    assert spec.flow_regime == "laminar"


# ── resolve_turbulence_spec — precheck shape ─────────────────────────────


def _precheck_shaped_config(model: str = "kOmegaSST") -> dict:
    """Mirrors the precheck output: turbulence is a sub-object, not flat."""
    return {
        "physics": {
            "flow_regime": "turbulent",
            "compressibility": "compressible",
            "heat_transfer": True,
        },
        "turbulence": {  # ← precheck nests it here
            "model": model,
            "wall_functions": True,
        },
        "boundary_conditions": {},
        "mesh": {"patches": []},
    }


@pytest.mark.parametrize("plugin_cls", ALL_PLUGINS)
def test_precheck_turbulence_model_survives_resolver(plugin_cls: type) -> None:
    """The precheck-shaped config carries kOmegaSST into the resolver.

    Before the fix, the resolver only checked ``physics.turbulence_model``
    and silently fell back to laminar — the SIGFPE root cause.
    """
    plugin = plugin_cls()
    cfg = _precheck_shaped_config("kOmegaSST")
    spec = resolve_turbulence_spec(plugin, cfg)
    assert spec.flow_regime == "turbulent"
    assert spec.model == "kOmegaSST"
    assert spec.simulation_type == "RAS"


@pytest.mark.parametrize("plugin_cls", ALL_PLUGINS)
def test_resolver_falls_back_to_plugin_default(plugin_cls: type) -> None:
    """When no model is set anywhere, the resolver uses the plugin's default
    (kOmegaSST for all 6 today) — not a silent laminar."""
    plugin = plugin_cls()
    cfg = {
        "physics": {"flow_regime": "turbulent"},
        "turbulence": {},
        "boundary_conditions": {},
        "mesh": {"patches": []},
    }
    spec = resolve_turbulence_spec(plugin, cfg)
    assert spec.model == plugin.default_turbulence_model
    assert spec.model != "laminar"


@pytest.mark.parametrize("plugin_cls", ALL_PLUGINS)
def test_resolver_respects_explicit_laminar(plugin_cls: type) -> None:
    """A user explicitly asking for laminar gets laminar — even on a
    compressible energy solver (it's their call)."""
    plugin = plugin_cls()
    cfg = {
        "physics": {"flow_regime": "laminar"},
        "turbulence": {"model": "kOmegaSST"},  # ignored — flow_regime wins
        "boundary_conditions": {},
        "mesh": {"patches": []},
    }
    spec = resolve_turbulence_spec(plugin, cfg)
    assert spec.flow_regime == "laminar"
    assert spec.model == "laminar"
    assert spec.simulation_type == "laminar"


def test_resolver_rejects_invalid_model() -> None:
    """A model not in the plugin's valid set raises — surfaces at build_case_spec
    time, not after an OpenFOAM run failure."""
    plugin = RhoSimpleFoamSolver()
    cfg = {
        "physics": {"flow_regime": "turbulent", "turbulence_model": "DESModel"},
        "boundary_conditions": {},
        "mesh": {"patches": []},
    }
    with pytest.raises(ValueError, match="does not support turbulence model"):
        resolve_turbulence_spec(plugin, cfg)


# ── End-to-end: precheck shape → CaseSpec → renderer ─────────────────────


def _air_compressible_cfg_precheck_shaped() -> dict:
    """Matches the actual precheck output that caused the SIGFPE."""
    return {
        "fluid": {"name": "air", "density": 1.225, "cp": 1006, "mu": 1.81e-5, "Pr": 0.713},
        "physics": {
            "flow_regime": "turbulent",
            "time_scheme": "steady",
            "compressibility": "compressible",
            "heat_transfer": True,
        },
        "turbulence": {     # ← precheck shape
            "model": "kOmegaSST",
            "wall_functions": True,
        },
        "boundary_conditions": {
            "inlet_main": {
                "patch_type": "inlet",
                "velocity": {"type": "fixedValue", "value": [5.0, 0.0, 0.0]},
                "temperature": {"type": "fixedValue", "value": 500.0},
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
                {"name": "inlet_main", "type": "patch"},
                {"name": "outlet", "type": "patch"},
                {"name": "walls", "type": "wall"},
            ],
        },
    }


@pytest.mark.parametrize("solver_name", [
    "rhoSimpleFoam", "rhoPimpleFoam",
    "buoyantSimpleFoam", "buoyantPimpleFoam",
])
def test_case_spec_carries_kOmegaSST_from_precheck_shape(solver_name: str) -> None:
    """Regression — precheck shape must produce simulationType RAS, not laminar."""
    cfg = _air_compressible_cfg_precheck_shaped()
    spec = build_case_spec(solver_name, cfg)
    assert spec.turbulence_spec is not None
    assert spec.turbulence_spec.simulation_type == "RAS"
    assert spec.turbulence_spec.model == "kOmegaSST"
    # And the legacy scalars stay consistent
    assert spec.turbulence_model == "kOmegaSST"
    assert spec.sim_type == "RAS"


def test_rhoSimpleFoam_renderer_emits_RAS_not_laminar() -> None:
    """The renderer (which produces the actual .of file) must output RAS."""
    plugin = RhoSimpleFoamSolver()
    cfg = _air_compressible_cfg_precheck_shaped()
    rendered = plugin._build_turbulence_properties(cfg)
    assert "simulationType  RAS" in rendered
    assert "RASModel        kOmegaSST" in rendered
    assert "laminar" not in rendered or "simulationType  laminar" not in rendered


# ── Normalizer reads precheck shape correctly ────────────────────────────


def test_normalizer_reads_turbulence_model_from_precheck_shape() -> None:
    cfg = {
        "physics": {"flow_regime": "turbulent"},
        "turbulence": {"model": "kOmegaSST"},
    }
    out = _normalize_physics(cfg)
    assert out.turbulence_model == "kOmegaSST"


def test_normalizer_reads_RASModel_too() -> None:
    """Some configs put it under turbulence.RASModel — accept that too."""
    cfg = {
        "physics": {"flow_regime": "turbulent"},
        "turbulence": {"RASModel": "kEpsilon"},
    }
    out = _normalize_physics(cfg)
    assert out.turbulence_model == "kEpsilon"


def test_normalizer_returns_none_when_no_model_anywhere() -> None:
    """Don't fabricate a model in the normalizer — let the resolver use the
    plugin default later in the pipeline."""
    cfg = {"physics": {"flow_regime": "turbulent"}}
    out = _normalize_physics(cfg)
    assert out.turbulence_model is None  # resolver will apply plugin default


# ── Per-plugin turbulence declarations ───────────────────────────────────


@pytest.mark.parametrize("plugin_cls", ALL_PLUGINS)
def test_every_plugin_declares_a_default_model(plugin_cls: type) -> None:
    plugin = plugin_cls()
    assert plugin.default_turbulence_model
    assert plugin.default_turbulence_model in plugin.valid_turbulence_models


@pytest.mark.parametrize("plugin_cls", ALL_PLUGINS)
def test_every_plugin_accepts_laminar(plugin_cls: type) -> None:
    plugin = plugin_cls()
    assert "laminar" in plugin.valid_turbulence_models
