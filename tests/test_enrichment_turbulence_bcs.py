# tests/test_enrichment_turbulence_bcs.py
"""Unit tests for the turbulence-BC propagation step.

Mirrors the structure of :mod:`test_enrichment_inlet_bcs`: a CHT-shaped
fixture plus one test per behavioural axis.  Both modules use the same
``_patch_lookup`` helpers, so when the inlet-finding logic evolves
these tests catch both sides of the change.
"""

from __future__ import annotations

import asyncio

import pytest

from simd_agent.run.enrichment import EnrichmentContext
from simd_agent.run.enrichment.turbulence_bcs import apply


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────


def _cht_config(turbulence: dict | None = None) -> dict:
    return {
        "regions": {
            "fluid": [
                {"name": "innerFluid", "kind": "fluid", "interfaces": ["wall"]},
                {"name": "outerFluid", "kind": "fluid", "interfaces": ["wall"]},
            ],
            "solid": [
                {"name": "wall", "kind": "solid",
                 "interfaces": ["innerFluid", "outerFluid"]},
            ],
        },
        "mesh": {"patches": [
            {"name": "innerFluid_inlet"},  {"name": "innerFluid_outlet"},
            {"name": "outerFluid_inlet"},  {"name": "outerFluid_outlet"},
            {"name": "wall_left_end"},     {"name": "wall_right_end"},
            {"name": "innerFluid_to_wall"}, {"name": "outerFluid_to_wall"},
        ]},
        "boundary_conditions": {},
        "turbulence": turbulence if turbulence is not None else {
            "model":   "kOmegaSST",
            "k":       0.00375,
            "omega":   10.6,
            "epsilon": None,
            "nut":     1.0e-5,
        },
    }


def _run(config: dict) -> None:
    ctx = EnrichmentContext(config=config, user_requirements="")
    asyncio.run(apply(ctx))


# ────────────────────────────────────────────────────────────────────────────
# Primary contract
# ────────────────────────────────────────────────────────────────────────────


def test_propagates_positive_turbulence_fields_to_every_fluid_inlet():
    cfg = _cht_config()
    _run(cfg)
    for inlet in ("innerFluid_inlet", "outerFluid_inlet"):
        bc = cfg["boundary_conditions"][inlet]
        assert bc["k"]["value"]   == 0.00375
        assert bc["omega"]["value"] == 10.6
        assert bc["nut"]["value"] == 1.0e-5


def test_skips_fields_whose_case_level_value_is_none():
    """``validated["turbulence"]["epsilon"] is None`` → never written."""
    cfg = _cht_config()
    _run(cfg)
    for inlet in ("innerFluid_inlet", "outerFluid_inlet"):
        assert "epsilon" not in cfg["boundary_conditions"][inlet]


def test_skips_fields_whose_case_level_value_is_zero_or_negative():
    """Non-positive values are treated as 'no signal'."""
    cfg = _cht_config(turbulence={"model": "kEpsilon", "k": 0.0, "epsilon": -1.0})
    _run(cfg)
    assert "k" not in cfg["boundary_conditions"].get("innerFluid_inlet", {})
    assert "epsilon" not in cfg["boundary_conditions"].get("innerFluid_inlet", {})


def test_existing_patch_value_is_preserved():
    cfg = _cht_config()
    cfg["boundary_conditions"]["innerFluid_inlet"] = {
        "k": {"type": "fixedValue", "value": 0.001},  # user-set
    }
    _run(cfg)
    # k preserved
    assert cfg["boundary_conditions"]["innerFluid_inlet"]["k"]["value"] == 0.001
    # other fields filled in from case-level values
    assert cfg["boundary_conditions"]["innerFluid_inlet"]["omega"]["value"] == 10.6


def test_existing_scalar_patch_value_is_preserved():
    """Older configs store turbulence as a bare scalar, not a dict — honour both."""
    cfg = _cht_config()
    cfg["boundary_conditions"]["innerFluid_inlet"] = {"k": 0.42}  # bare scalar
    _run(cfg)
    assert cfg["boundary_conditions"]["innerFluid_inlet"]["k"] == 0.42


def test_only_inlet_patches_get_updated():
    cfg = _cht_config()
    _run(cfg)
    for non_inlet in (
        "innerFluid_outlet", "outerFluid_outlet",
        "wall_left_end", "innerFluid_to_wall",
    ):
        assert non_inlet not in cfg["boundary_conditions"] or \
               not any(f in cfg["boundary_conditions"][non_inlet] for f in ("k", "omega", "epsilon", "nut"))


def test_creates_bc_entry_for_inlet_with_no_prior_bc():
    cfg = _cht_config()
    # innerFluid_inlet missing entirely from boundary_conditions — should be created.
    assert "innerFluid_inlet" not in cfg["boundary_conditions"]
    _run(cfg)
    bc = cfg["boundary_conditions"]["innerFluid_inlet"]
    assert bc["k"]["value"]   == 0.00375
    assert bc["omega"]["value"] == 10.6


# ────────────────────────────────────────────────────────────────────────────
# No-op cases
# ────────────────────────────────────────────────────────────────────────────


def test_no_op_when_laminar():
    """Linter writes ``turbulence = {}`` for laminar flow — step short-circuits."""
    cfg = _cht_config(turbulence={})
    _run(cfg)
    assert cfg["boundary_conditions"] == {}


def test_no_op_when_no_fluid_regions():
    """Single-region cases (no regions block) — step short-circuits."""
    cfg = _cht_config()
    cfg.pop("regions")
    _run(cfg)
    assert cfg["boundary_conditions"] == {}


def test_no_op_when_no_case_level_turbulence_values():
    """Model selected but values not yet pre-computed."""
    cfg = _cht_config(turbulence={"model": "kOmegaSST"})
    _run(cfg)
    assert cfg["boundary_conditions"] == {}


def test_handles_missing_boundary_conditions_dict():
    """Step creates ``boundary_conditions`` on demand."""
    cfg = _cht_config()
    cfg.pop("boundary_conditions")
    _run(cfg)
    assert cfg["boundary_conditions"]["innerFluid_inlet"]["k"]["value"] == 0.00375


def test_patch_role_via_patch_class_alias():
    """Inlet selection works when the role comes via ``patchClass`` instead of name."""
    cfg = _cht_config()
    # Rename inlet → in (no _inlet suffix), declare role.
    for p in cfg["mesh"]["patches"]:
        if p["name"] == "innerFluid_inlet":
            p["name"] = "innerFluid_in"
    cfg["boundary_conditions"]["innerFluid_in"] = {"patchClass": "inlet"}
    _run(cfg)
    assert cfg["boundary_conditions"]["innerFluid_in"]["k"]["value"] == 0.00375
