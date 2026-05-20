# tests/test_enrichment_case_defaults.py
"""Unit tests for the case-defaults resolver.

Each test exercises one signal-resolution path so a future contributor
adding a new priority rule can see exactly which input drives which
output.  No LLM, no I/O — these are pure-function tests on a single
config dict.
"""

from __future__ import annotations

import pytest

from simd_agent.run.enrichment import EnrichmentContext
from simd_agent.run.enrichment.case_defaults import apply


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _run(config: dict) -> dict:
    """Apply the step and return the resolved ``case_defaults`` block."""
    ctx = EnrichmentContext(config=config, user_requirements="")
    import asyncio
    asyncio.run(apply(ctx))
    return config["case_defaults"]


# ────────────────────────────────────────────────────────────────────────────
# Inlet velocity resolution
# ────────────────────────────────────────────────────────────────────────────


def test_inlet_velocity_from_legacy_block():
    out = _run({"inlet": {"velocity": [1.5, 0.0, 0.0]}})
    assert out["inlet_velocity"] == (1.5, 0.0, 0.0)


def test_inlet_velocity_from_scalar_legacy_block_treated_as_plus_x():
    out = _run({"inlet": {"velocity": 0.5}})
    assert out["inlet_velocity"] == (0.5, 0.0, 0.0)


def test_inlet_velocity_from_per_patch_bc_when_legacy_missing():
    out = _run({
        "boundary_conditions": {
            "inlet1": {"velocity": {"type": "fixedValue", "value": [2.0, 0.0, 0.0]}},
        },
    })
    assert out["inlet_velocity"] == (2.0, 0.0, 0.0)


def test_inlet_velocity_zero_in_legacy_falls_through_to_patch_lookup():
    out = _run({
        "inlet": {"velocity": [0.0, 0.0, 0.0]},  # placeholder zero — ignore
        "boundary_conditions": {
            "inlet1": {"velocity": {"type": "fixedValue", "value": [3.0, 0.0, 0.0]}},
        },
    })
    assert out["inlet_velocity"] == (3.0, 0.0, 0.0)


def test_inlet_velocity_none_when_no_signal():
    out = _run({"fluid": {"density": 1.2}})
    assert out["inlet_velocity"] is None


# ────────────────────────────────────────────────────────────────────────────
# Inlet temperature resolution
# ────────────────────────────────────────────────────────────────────────────


def test_inlet_temperature_from_inlet_patch_bc():
    out = _run({
        "boundary_conditions": {
            "innerFluid_inlet": {
                "temperature": {"type": "fixedValue", "value": 77.0},
            },
        },
    })
    assert out["inlet_temperature"] == 77.0


def test_inlet_temperature_falls_back_to_bulk_when_no_inlet_bc():
    out = _run({"fluid": {"temperature": 290.0}})
    assert out["inlet_temperature"] == 290.0
    assert out["bulk_temperature"] == 290.0


def test_inlet_temperature_inlet_bc_wins_over_bulk():
    out = _run({
        "fluid": {"temperature": 290.0},
        "boundary_conditions": {
            "inlet1": {
                "patch_class": "inlet",
                "temperature": {"value": 77.0},
            },
        },
    })
    assert out["inlet_temperature"] == 77.0
    assert out["bulk_temperature"] == 290.0


def test_non_inlet_patch_temperature_is_ignored():
    """A wall patch with a temperature value must not leak into ``inlet_temperature``."""
    out = _run({
        "boundary_conditions": {
            "wall_top": {
                "patch_class": "wall",
                "temperature": {"value": 400.0},
            },
        },
        "fluid": {"temperature": 300.0},
    })
    assert out["inlet_temperature"] == 300.0  # bulk fallback, not the wall


# ────────────────────────────────────────────────────────────────────────────
# Inlet / ambient pressure
# ────────────────────────────────────────────────────────────────────────────


def test_ambient_pressure_from_outlet_block():
    out = _run({"outlet": {"pressure": 101325.0}})
    assert out["ambient_pressure"] == 101325.0
    # inlet_pressure falls back to ambient when no inlet BC carries one
    assert out["inlet_pressure"] == 101325.0


def test_inlet_pressure_from_inlet_patch_overrides_ambient():
    out = _run({
        "outlet": {"pressure": 101325.0},
        "boundary_conditions": {
            "inlet1": {
                "patch_class": "inlet",
                "pressure": {"value": 200000.0},
            },
        },
    })
    assert out["inlet_pressure"] == 200000.0
    assert out["ambient_pressure"] == 101325.0


# ────────────────────────────────────────────────────────────────────────────
# Bulk fluid properties
# ────────────────────────────────────────────────────────────────────────────


def test_bulk_density_from_density_key():
    out = _run({"fluid": {"density": 998.0}})
    assert out["bulk_density"] == 998.0


def test_bulk_density_from_rho_alias():
    out = _run({"fluid": {"rho": 1.225}})
    assert out["bulk_density"] == 1.225


def test_bulk_density_none_when_zero_or_missing():
    """Zero / missing means 'no real signal' — never a programming error."""
    assert _run({"fluid": {"density": 0.0}})["bulk_density"] is None
    assert _run({"fluid": {}})["bulk_density"] is None
    assert _run({})["bulk_density"] is None


def test_bulk_kinematic_viscosity_direct():
    out = _run({"fluid": {"kinematic_viscosity": 1.0e-6}})
    assert out["bulk_kinematic_viscosity"] == 1.0e-6


def test_bulk_kinematic_viscosity_derived_from_mu_over_rho():
    """When only μ + ρ are given, ν is derived as μ/ρ."""
    out = _run({"fluid": {"dynamic_viscosity": 1.8e-5, "density": 1.2}})
    assert out["bulk_kinematic_viscosity"] == pytest.approx(1.5e-5, rel=1e-9)


def test_bulk_kinematic_viscosity_direct_wins_over_derivation():
    """If both ν and (μ, ρ) are given, ν is reported as-is (no override)."""
    out = _run({
        "fluid": {
            "kinematic_viscosity": 9.99e-9,  # the user-typed value wins
            "dynamic_viscosity": 1.8e-5,
            "density": 1.2,
        },
    })
    assert out["bulk_kinematic_viscosity"] == 9.99e-9


def test_bulk_dynamic_viscosity_direct():
    out = _run({"fluid": {"mu": 1.0e-3}})
    assert out["bulk_dynamic_viscosity"] == 1.0e-3


def test_bulk_dynamic_viscosity_derived_from_nu_times_rho():
    """When only ν + ρ are given, μ is derived as ν·ρ."""
    out = _run({"fluid": {"nu": 1.5e-5, "density": 1.2}})
    assert out["bulk_dynamic_viscosity"] == pytest.approx(1.8e-5, rel=1e-9)


def test_bulk_viscosity_none_when_only_one_of_mu_or_rho():
    """Derivation requires BOTH μ and ρ — neither alone is enough."""
    assert _run({"fluid": {"dynamic_viscosity": 1.8e-5}})["bulk_kinematic_viscosity"] is None
    assert _run({"fluid": {"density": 1.2}})["bulk_kinematic_viscosity"] is None


def test_bulk_prandtl_from_each_alias():
    for alias in ("prandtl_number", "prandtl", "Pr"):
        out = _run({"fluid": {alias: 0.71}})
        assert out["bulk_prandtl"] == 0.71, f"alias {alias!r} not honoured"


# ────────────────────────────────────────────────────────────────────────────
# Turbulence intensity
# ────────────────────────────────────────────────────────────────────────────


def test_turbulence_intensity_from_turbulence_block():
    out = _run({"turbulence": {"intensity": 0.05}})
    assert out["turbulence_intensity"] == 0.05


def test_turbulence_intensity_none_when_block_missing_or_empty():
    assert _run({})["turbulence_intensity"] is None
    assert _run({"turbulence": {}})["turbulence_intensity"] is None
    assert _run({"turbulence": {"intensity": 0.0}})["turbulence_intensity"] is None


# ────────────────────────────────────────────────────────────────────────────
# Wall temperatures
# ────────────────────────────────────────────────────────────────────────────


def test_wall_temperatures_collects_all_wall_patches_with_T():
    out = _run({
        "boundary_conditions": {
            "hot_wall":  {"patch_class": "wall", "temperature": {"value": 400.0}},
            "cold_wall": {"patch_class": "wall", "temperature": {"value": 290.0}},
        },
    })
    assert out["wall_temperatures"] == {"hot_wall": 400.0, "cold_wall": 290.0}


def test_wall_temperatures_skips_walls_without_explicit_T():
    """Adiabatic walls (no temperature BC) don't appear in the map."""
    out = _run({
        "boundary_conditions": {
            "hot_wall":      {"patch_class": "wall", "temperature": {"value": 400.0}},
            "adiabatic_top": {"patch_class": "wall"},
        },
    })
    assert out["wall_temperatures"] == {"hot_wall": 400.0}


def test_wall_temperatures_excludes_cht_coupled_interfaces():
    """``*_to_*`` patches are owned by the multi-region renderer."""
    out = _run({
        "boundary_conditions": {
            "wall_left_end":      {"patch_class": "wall", "temperature": {"value": 400.0}},
            "wall_to_innerFluid": {"patch_class": "wall", "temperature": {"value": 999.0}},
        },
    })
    assert out["wall_temperatures"] == {"wall_left_end": 400.0}


def test_wall_temperatures_falls_back_to_name_when_role_missing():
    """If no patch_class set, ``wall``/``_wall`` suffix or ``wall_`` prefix is used."""
    out = _run({
        "boundary_conditions": {
            "hot_wall":      {"temperature": {"value": 400.0}},  # name suffix
            "wall_left_end": {"temperature": {"value": 350.0}},  # name prefix
            "inlet1":        {"temperature": {"value": 999.0}},  # ignored
        },
    })
    assert out["wall_temperatures"] == {"hot_wall": 400.0, "wall_left_end": 350.0}


def test_wall_temperatures_empty_dict_when_no_walls():
    out = _run({"boundary_conditions": {
        "inlet":  {"patch_class": "inlet", "temperature": {"value": 300.0}},
        "outlet": {"patch_class": "outlet"},
    }})
    assert out["wall_temperatures"] == {}


# ────────────────────────────────────────────────────────────────────────────
# Idempotence + diagnostics
# ────────────────────────────────────────────────────────────────────────────


def test_apply_is_idempotent():
    cfg = {"fluid": {"temperature": 300.0}, "inlet": {"velocity": [1.0, 0.0, 0.0]}}
    a = _run(cfg)
    b = _run(cfg)
    assert a == b


@pytest.mark.asyncio
async def test_apply_emits_info_issue():
    ctx = EnrichmentContext(config={"fluid": {"temperature": 290.0}}, user_requirements="")
    await apply(ctx)
    assert len(ctx.issues) == 1
    issue = ctx.issues[0]
    assert issue.severity == "info"
    assert issue.step == "case_defaults"
    assert issue.code == "RESOLVED"
    assert isinstance(issue.payload, dict)
    assert issue.payload["bulk_temperature"] == 290.0
