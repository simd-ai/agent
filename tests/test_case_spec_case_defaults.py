# tests/test_case_spec_case_defaults.py
"""Tests for the :mod:`build_case_spec` ↔ ``case_defaults`` bridge.

These pin the strangler-fig contract: every direct-config lookup in
:func:`build_case_spec` keeps working unchanged, AND a freshly-enriched
config (``case_defaults`` block present) lets the canonical values flow
in for fields the direct lookup couldn't resolve.

Pair each "legacy works" assertion with an "enriched also works" twin so
the migration's two halves are visible in the same file.
"""

from __future__ import annotations

import pytest

from simd_agent.run.case_spec import build_case_spec


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────


def _bare_cfg() -> dict:
    """Mesh + solver only — no per-patch BCs, no fluid props.

    Models the "generic prompt, wizard set bulk T + inlet U globally"
    scenario that case_defaults is meant to bridge.
    """
    return {
        "solver":  {"solver": "simpleFoam"},
        "physics": {"compressibility": "incompressible"},
        "mesh":    {"patches": [
            {"name": "inlet"}, {"name": "outlet"}, {"name": "wall"},
        ]},
        "fluid":   {"name": "air"},
        "boundary_conditions": {},
    }


def _full_case_defaults() -> dict:
    """A realistic case_defaults block as the enrichment pipeline produces it."""
    return {
        "inlet_velocity":           (2.5, 0.0, 0.0),
        "inlet_temperature":        290.0,
        "inlet_pressure":           150000.0,
        "ambient_pressure":         101325.0,
        "bulk_temperature":         290.0,
        "bulk_density":             998.0,
        "bulk_kinematic_viscosity": 1.0e-6,
        "bulk_dynamic_viscosity":   1.0e-3,
        "bulk_prandtl":             7.0,
        "turbulence_intensity":     0.05,
        "wall_temperatures":        {"hot_wall": 350.0, "cold_wall": 290.0},
    }


# ────────────────────────────────────────────────────────────────────────────
# Legacy direct-config path still works (regression safety net)
# ────────────────────────────────────────────────────────────────────────────


def test_legacy_config_without_case_defaults_unchanged():
    """No ``case_defaults`` block → behaviour identical to pre-migration."""
    spec = build_case_spec("simpleFoam", _bare_cfg())
    assert spec.inlet_velocity    == [0.0, 0.0, 0.0]
    assert spec.inlet_temperature is None
    assert spec.wall_temperature  is None
    assert spec.operating_pressure == 101325.0  # last-resort default
    assert spec.rho is None
    assert spec.nu  is None
    assert spec.mu  is None
    assert spec.prandtl is None


def test_direct_inlet_bc_still_wins_over_case_defaults():
    """A patch-level inlet velocity overrides whatever case_defaults says."""
    cfg = _bare_cfg()
    cfg["boundary_conditions"]["inlet"] = {
        "velocity": {"type": "fixedValue", "value": [9.9, 0.0, 0.0]},
    }
    cfg["case_defaults"] = _full_case_defaults()
    spec = build_case_spec("simpleFoam", cfg)
    assert spec.inlet_velocity == [9.9, 0.0, 0.0]


def test_direct_fluid_density_still_wins_over_case_defaults():
    """Explicit ``fluid.density`` overrides ``case_defaults.bulk_density``."""
    cfg = _bare_cfg()
    cfg["fluid"]["density"] = 1.225
    cfg["case_defaults"] = _full_case_defaults()
    spec = build_case_spec("simpleFoam", cfg)
    assert spec.rho == 1.225  # not the case_defaults 998.0


# ────────────────────────────────────────────────────────────────────────────
# case_defaults fills the gaps the direct path leaves
# ────────────────────────────────────────────────────────────────────────────


def test_inlet_velocity_filled_from_case_defaults():
    cfg = _bare_cfg()
    cfg["case_defaults"] = _full_case_defaults()
    spec = build_case_spec("simpleFoam", cfg)
    assert spec.inlet_velocity == [2.5, 0.0, 0.0]


def test_inlet_temperature_filled_from_case_defaults():
    cfg = _bare_cfg()
    cfg["case_defaults"] = _full_case_defaults()
    spec = build_case_spec("simpleFoam", cfg)
    assert spec.inlet_temperature == 290.0


def test_wall_temperature_collapses_dict_to_max():
    """Multi-wall ``wall_temperatures`` dict → CaseSpec uses the hot side."""
    cfg = _bare_cfg()
    cfg["case_defaults"] = _full_case_defaults()
    spec = build_case_spec("simpleFoam", cfg)
    assert spec.wall_temperature == 350.0


def test_operating_pressure_filled_from_ambient():
    cfg = _bare_cfg()
    cd = _full_case_defaults()
    cfg["case_defaults"] = cd
    spec = build_case_spec("simpleFoam", cfg)
    assert spec.operating_pressure == cd["ambient_pressure"]


def test_bulk_fluid_properties_filled_from_case_defaults():
    cfg = _bare_cfg()
    cfg["case_defaults"] = _full_case_defaults()
    spec = build_case_spec("simpleFoam", cfg)
    assert spec.rho     == 998.0
    assert spec.nu      == 1.0e-6
    assert spec.mu      == 1.0e-3
    assert spec.prandtl == 7.0


def test_nu_still_derived_from_mu_over_rho_when_only_case_defaults_provides_them():
    """The existing ν = μ/ρ derivation kicks in even when both come from case_defaults."""
    cfg = _bare_cfg()
    cfg["case_defaults"] = {
        # No ``bulk_kinematic_viscosity`` — only μ + ρ available
        "bulk_dynamic_viscosity": 1.8e-5,
        "bulk_density":           1.2,
    }
    spec = build_case_spec("simpleFoam", cfg)
    # case_defaults.bulk_kinematic_viscosity itself is missing here, so the
    # case-defaults lookup gets None; the legacy μ/ρ derivation closes the
    # gap using values just-resolved from case_defaults.
    assert spec.mu  == 1.8e-5
    assert spec.rho == 1.2
    assert spec.nu  == pytest.approx(1.5e-5, rel=1e-9)


def test_empty_wall_temperatures_dict_does_not_set_wall_T():
    """Defensive: an empty ``wall_temperatures`` dict means 'no wall T'."""
    cfg = _bare_cfg()
    cfg["case_defaults"] = {**_full_case_defaults(), "wall_temperatures": {}}
    spec = build_case_spec("simpleFoam", cfg)
    assert spec.wall_temperature is None


def test_case_defaults_with_only_some_fields_partial_fill():
    """Partial case_defaults — only the present keys fill, others stay None / default."""
    cfg = _bare_cfg()
    cfg["case_defaults"] = {
        "inlet_temperature": 77.0,  # only T
        # ... nothing else
    }
    spec = build_case_spec("simpleFoam", cfg)
    assert spec.inlet_temperature == 77.0
    assert spec.rho is None
    assert spec.inlet_velocity == [0.0, 0.0, 0.0]
