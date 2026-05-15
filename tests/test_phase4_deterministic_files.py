"""Phase 4 tests — files moved off the LLM codegen path.

For each file migrated, verify:
  1. The plugin's ``required_files()`` no longer lists the file
     (so the LLM loop will skip it).
  2. ``render_deterministic_files(config)`` returns it with content that
     matches the OpenFOAM tutorial conventions.
  3. ``case_spec.required_*_files`` is filtered consistently (so the
     orchestrator doesn't ask the LLM for it).
"""

import pytest

from simd_agent.run.case_spec import build_case_spec
from simd_agent.solvers.compressible.rhoSimpleFoam.solver import RhoSimpleFoamSolver
from simd_agent.solvers.incompressible.simpleFoam.solver import SimpleFoamSolver
from simd_agent.solvers.compressible.rhoPimpleFoam.solver import RhoPimpleFoamSolver


# ── Fixtures ──────────────────────────────────────────────────────────────


def _air_compressible_config() -> dict:
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


# ── constant/turbulenceProperties ─────────────────────────────────────────


def test_turbulence_properties_rendered_RAS_kOmegaSST() -> None:
    plugin = RhoSimpleFoamSolver()
    text = plugin._build_turbulence_properties(_air_compressible_config())
    assert "simulationType  RAS" in text
    assert "RASModel        kOmegaSST" in text
    assert "turbulence      on" in text
    assert "printCoeffs     on" in text


def test_turbulence_properties_laminar() -> None:
    plugin = SimpleFoamSolver()
    cfg = _air_compressible_config()
    cfg["physics"]["flow_regime"] = "laminar"
    cfg["physics"]["turbulence_model"] = "laminar"
    cfg["physics"]["compressibility"] = "incompressible"
    cfg["physics"]["heat_transfer"] = False
    text = plugin._build_turbulence_properties(cfg)
    assert "simulationType  laminar" in text
    assert "RAS" not in text


def test_turbulence_properties_not_in_required_files() -> None:
    """LLM loop must NOT be asked to generate turbulenceProperties."""
    plugin = RhoSimpleFoamSolver()
    assert "constant/turbulenceProperties" not in plugin.required_files(_air_compressible_config())


def test_case_spec_required_const_excludes_deterministic_turbulenceProperties() -> None:
    """case_spec also filters — the orchestrator iterates this list."""
    spec = build_case_spec("rhoSimpleFoam", _air_compressible_config())
    assert "constant/turbulenceProperties" not in spec.required_constant_files


# ── 0/nut ─────────────────────────────────────────────────────────────────


def test_nut_uses_nutkWallFunction_at_wall() -> None:
    plugin = RhoSimpleFoamSolver()
    text = plugin._build_nut(_air_compressible_config())
    assert "internalField   uniform 0" in text
    # Wall has the wall function
    assert "nutkWallFunction" in text
    # Inlet / outlet have calculated
    assert "calculated" in text
    # Dimensions
    assert "[0 2 -1 0 0 0 0]" in text


def test_nut_not_in_required_files() -> None:
    plugin = RhoSimpleFoamSolver()
    assert "0/nut" not in plugin.required_files(_air_compressible_config())


def test_nut_in_deterministic_files() -> None:
    plugin = RhoSimpleFoamSolver()
    files = plugin.render_deterministic_files(_air_compressible_config())
    assert "0/nut" in files
    assert "nutkWallFunction" in files["0/nut"]


# ── 0/alphat ──────────────────────────────────────────────────────────────


def test_alphat_uses_compressible_namespace() -> None:
    plugin = RhoSimpleFoamSolver()
    text = plugin._build_alphat(_air_compressible_config())
    # The namespace-qualified form is OpenFOAM 2406-correct
    assert "compressible::alphatWallFunction" in text
    assert "Prt             0.85" in text
    # Dimensions
    assert "[1 -1 -1 0 0 0 0]" in text


def test_alphat_only_for_compressible_energy() -> None:
    plugin = SimpleFoamSolver()
    cfg = _air_compressible_config()
    cfg["physics"]["compressibility"] = "incompressible"
    cfg["physics"]["heat_transfer"] = False
    files = plugin.render_deterministic_files(cfg)
    # No alphat for incompressible solvers
    assert "0/alphat" not in files


def test_alphat_not_in_required_files_for_compressible_energy() -> None:
    plugin = RhoSimpleFoamSolver()
    cfg = _air_compressible_config()
    assert "0/alphat" not in plugin.required_files(cfg)


# ── render_deterministic_files() integration ──────────────────────────────


def test_deterministic_files_complete_set() -> None:
    """Compressible energy turbulent solver: 5 deterministic files."""
    plugin = RhoSimpleFoamSolver()
    files = plugin.render_deterministic_files(_air_compressible_config())
    assert set(files.keys()) == {
        "system/fvSolution",
        "system/fvSchemes",
        "constant/turbulenceProperties",
        "0/nut",
        "0/alphat",
    }


def test_deterministic_files_incompressible_subset() -> None:
    """Incompressible turbulent solver: 4 files (no 0/alphat)."""
    plugin = SimpleFoamSolver()
    cfg = _air_compressible_config()
    cfg["physics"]["compressibility"] = "incompressible"
    cfg["physics"]["heat_transfer"] = False
    files = plugin.render_deterministic_files(cfg)
    assert "0/nut" in files
    assert "0/alphat" not in files
    assert "constant/turbulenceProperties" in files


def test_no_overlap_between_required_and_deterministic() -> None:
    """Phase 4 invariant: required_files() and render_deterministic_files()
    MUST be disjoint — otherwise the LLM and the renderer would both produce
    the same file and the merge order would matter.
    """
    for plugin_cls in (RhoSimpleFoamSolver, RhoPimpleFoamSolver, SimpleFoamSolver):
        plugin = plugin_cls()
        cfg = _air_compressible_config()
        # Adjust config for the right solver type
        if plugin_cls is SimpleFoamSolver:
            cfg["physics"]["compressibility"] = "incompressible"
            cfg["physics"]["heat_transfer"] = False
        elif plugin_cls is RhoPimpleFoamSolver:
            cfg["physics"]["time_scheme"] = "transient"

        required = set(plugin.required_files(cfg))
        deterministic = set(plugin.render_deterministic_files(cfg).keys())
        overlap = required & deterministic
        assert overlap == set(), f"{plugin_cls.__name__}: overlap {overlap}"
