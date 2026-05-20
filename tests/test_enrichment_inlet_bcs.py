# tests/test_enrichment_inlet_bcs.py
"""Unit tests for the inlet-BC propagation step.

Migrated from the legacy ``test_inlet_bc_propagator`` — the function
itself moved to :mod:`simd_agent.run.enrichment.inlet_bcs` but the
behaviour contract is unchanged: copy ``region.T_init`` / ``U_init``
onto each region's inlet patches, never override values that look
non-default, and stay a no-op when there's nothing to propagate.
"""

from __future__ import annotations

import asyncio

import pytest

from simd_agent.run.enrichment import EnrichmentContext
from simd_agent.run.enrichment.inlet_bcs import apply


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────


def _cht_regascold_config() -> dict:
    """Counter-flow regasifier shape.

    - innerFluid (LN2):   inlet at 77 K, +x flow 0.05 m/s
    - outerFluid (water): inlet at 290 K, -x flow 0.10 m/s
    - wall (stainless):   adiabatic ends
    - boundary_conditions seeded with the precheck's 300 K / zero placeholders
    """
    return {
        "regions": {
            "fluid": [
                {"name": "innerFluid", "kind": "fluid",
                 "T_init": 77.0,  "U_init": (0.05, 0.0, 0.0),
                 "interfaces": ["wall"]},
                {"name": "outerFluid", "kind": "fluid",
                 "T_init": 290.0, "U_init": (-0.10, 0.0, 0.0),
                 "interfaces": ["wall"]},
            ],
            "solid": [
                {"name": "wall", "kind": "solid",
                 "interfaces": ["innerFluid", "outerFluid"]},
            ],
        },
        "mesh": {"patches": [
            {"name": "innerFluid_inlet",  "type": "patch"},
            {"name": "innerFluid_outlet", "type": "patch"},
            {"name": "outerFluid_inlet",  "type": "patch"},
            {"name": "outerFluid_outlet", "type": "patch"},
            {"name": "innerFluid_to_wall", "type": "mappedWall"},
            {"name": "outerFluid_to_wall", "type": "mappedWall"},
            {"name": "wall_left_end",     "type": "wall"},
            {"name": "wall_right_end",    "type": "wall"},
        ]},
        "boundary_conditions": {
            "innerFluid_inlet": {
                "T": {"type": "fixedValue", "value": 300.0},          # placeholder
                "U": {"type": "fixedValue", "value": [0.0, 0.0, 0.0]},  # placeholder
            },
            "outerFluid_inlet": {
                "T": {"type": "fixedValue", "value": 300.0},
                "U": {"type": "fixedValue", "value": [0.0, 0.0, 0.0]},
            },
        },
    }


def _run(config: dict) -> None:
    ctx = EnrichmentContext(config=config, user_requirements="")
    asyncio.run(apply(ctx))


# ────────────────────────────────────────────────────────────────────────────
# Tests — primary contract
# ────────────────────────────────────────────────────────────────────────────


def test_overrides_placeholder_T_with_region_T_init():
    cfg = _cht_regascold_config()
    _run(cfg)
    assert cfg["boundary_conditions"]["innerFluid_inlet"]["T"]["value"] == 77.0
    assert cfg["boundary_conditions"]["outerFluid_inlet"]["T"]["value"] == 290.0


def test_overrides_zero_U_with_region_U_init_keeps_sign():
    cfg = _cht_regascold_config()
    _run(cfg)
    assert cfg["boundary_conditions"]["innerFluid_inlet"]["U"]["value"] == [0.05, 0.0, 0.0]
    assert cfg["boundary_conditions"]["outerFluid_inlet"]["U"]["value"] == [-0.10, 0.0, 0.0]


def test_does_not_touch_pressure():
    """The step is explicit about T / U only; p / p_rgh stay where they are."""
    cfg = _cht_regascold_config()
    cfg["boundary_conditions"]["innerFluid_inlet"]["p"] = {"value": 101325.0}
    _run(cfg)
    assert cfg["boundary_conditions"]["innerFluid_inlet"]["p"] == {"value": 101325.0}


def test_explicit_patch_T_is_preserved():
    cfg = _cht_regascold_config()
    cfg["boundary_conditions"]["innerFluid_inlet"]["T"] = {"value": 100.0}
    _run(cfg)
    assert cfg["boundary_conditions"]["innerFluid_inlet"]["T"]["value"] == 100.0


def test_explicit_patch_U_is_preserved():
    cfg = _cht_regascold_config()
    cfg["boundary_conditions"]["innerFluid_inlet"]["U"] = {"value": [0.3, 0.0, 0.0]}
    _run(cfg)
    assert cfg["boundary_conditions"]["innerFluid_inlet"]["U"]["value"] == [0.3, 0.0, 0.0]


def test_only_inlet_patches_get_updated():
    """Outlet, wall, interface patches are not touched."""
    cfg = _cht_regascold_config()
    _run(cfg)
    # No BCs were written for non-inlet patches
    for non_inlet in ("innerFluid_outlet", "outerFluid_outlet",
                      "wall_left_end", "innerFluid_to_wall"):
        bc = cfg["boundary_conditions"].get(non_inlet, {})
        assert "T" not in bc and "U" not in bc


def test_patch_role_via_patch_class_alias():
    """Roles can come in via ``patchClass`` / ``patch_class`` / ``patch_type``."""
    cfg = _cht_regascold_config()
    # rename innerFluid_inlet → innerFluid_in (no _inlet suffix), declare role
    bc = cfg["boundary_conditions"].pop("innerFluid_inlet")
    bc["patchClass"] = "inlet"
    cfg["boundary_conditions"]["innerFluid_in"] = bc
    for p in cfg["mesh"]["patches"]:
        if p["name"] == "innerFluid_inlet":
            p["name"] = "innerFluid_in"
    _run(cfg)
    assert cfg["boundary_conditions"]["innerFluid_in"]["T"]["value"] == 77.0


def test_creates_bc_entry_for_inlet_with_no_prior_bc():
    """An inlet patch with no existing BC still gets a populated entry."""
    cfg = _cht_regascold_config()
    cfg["boundary_conditions"].pop("innerFluid_inlet")
    _run(cfg)
    bc = cfg["boundary_conditions"]["innerFluid_inlet"]
    assert bc["T"]["value"] == 77.0
    assert bc["U"]["value"] == [0.05, 0.0, 0.0]


# ────────────────────────────────────────────────────────────────────────────
# No-op cases
# ────────────────────────────────────────────────────────────────────────────


def test_no_op_single_region():
    cfg = {"mesh": {"patches": [{"name": "inlet"}]}, "boundary_conditions": {}}
    _run(cfg)
    assert cfg["boundary_conditions"] == {}


def test_no_op_when_extractor_did_not_populate_T_or_U():
    cfg = _cht_regascold_config()
    for r in cfg["regions"]["fluid"]:
        r["T_init"] = None
        r["U_init"] = None
    before = {k: dict(v) for k, v in cfg["boundary_conditions"].items()}
    _run(cfg)
    assert cfg["boundary_conditions"] == before


def test_no_op_when_region_U_init_is_zero_vector():
    cfg = _cht_regascold_config()
    cfg["regions"]["fluid"][0]["U_init"] = (0.0, 0.0, 0.0)
    _run(cfg)
    assert cfg["boundary_conditions"]["innerFluid_inlet"]["T"]["value"] == 77.0
    assert cfg["boundary_conditions"]["innerFluid_inlet"]["U"]["value"] == [0.0, 0.0, 0.0]


def test_handles_missing_boundary_conditions_dict():
    cfg = _cht_regascold_config()
    cfg.pop("boundary_conditions")
    _run(cfg)
    inlet = cfg.get("boundary_conditions", {}).get("innerFluid_inlet")
    assert inlet and inlet["T"]["value"] == 77.0
