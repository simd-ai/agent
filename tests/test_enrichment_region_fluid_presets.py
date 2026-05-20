# tests/test_enrichment_region_fluid_presets.py
"""Unit tests for the per-region fluid preset inference step.

The step replaces the auto-detector's heuristic ``"air"`` default with a
real cryogen / water / oil preset when the region's T_init + the case's
bulk density + the case's fluid name agree on what the fluid is.
"""

from __future__ import annotations

import asyncio

from simd_agent.run.enrichment import EnrichmentContext
from simd_agent.run.enrichment.region_fluid_presets import apply


def _cht_cfg(
    *,
    fluid_name: str = "",
    bulk_density: float | None = None,
    inlet_temperature: float | None = None,
    inner_T: float | None = 77.0,
    outer_T: float | None = 290.0,
    inner_preset: str = "air",
    outer_preset: str = "air",
) -> dict:
    return {
        "fluid": {"name": fluid_name},
        "case_defaults": {
            "bulk_density":       bulk_density,
            "inlet_temperature":  inlet_temperature,
        },
        "regions": {
            "fluid": [
                {"name": "innerFluid", "kind": "fluid", "fluid_preset": inner_preset,
                 "T_init": inner_T, "interfaces": ["wall"]},
                {"name": "outerFluid", "kind": "fluid", "fluid_preset": outer_preset,
                 "T_init": outer_T, "interfaces": ["wall"]},
            ],
            "solid": [
                {"name": "wall", "kind": "solid", "solid_preset": "stainless",
                 "interfaces": ["innerFluid", "outerFluid"]},
            ],
        },
    }


def _run(cfg: dict) -> None:
    ctx = EnrichmentContext(config=cfg, user_requirements="")
    asyncio.run(apply(ctx))


def test_ln2_case_overrides_air_on_matching_region():
    """LN2 + room-T outer region → ln2 + water (regasifier topology).

    Per-region pass: innerFluid (T=77, case fluid name says LN2) → ln2.
    Regasifier topology pass: exactly one cryogenic region + a room-T
    region (outerFluid at 290 K) → outerFluid flips from air to water,
    because air would carry ~1000× too little enthalpy to make the
    heat-exchange visualisation meaningful, and water is the realistic
    process fluid for a regasifier.
    """
    cfg = _cht_cfg(
        fluid_name="Liquid Nitrogen (LN2)",
        bulk_density=808.0,
        inlet_temperature=77.0,
        inner_T=77.0,
        outer_T=290.0,
    )
    _run(cfg)
    inner = cfg["regions"]["fluid"][0]
    outer = cfg["regions"]["fluid"][1]
    assert inner["fluid_preset"] == "ln2"
    assert outer["fluid_preset"] == "water"


def test_regasifier_rule_requires_exactly_one_cryogen():
    """If two cryogenic regions exist (e.g. LN2 + LH2 case), the
    regasifier rule must NOT fire — the topology isn't a regasifier."""
    cfg = _cht_cfg(
        fluid_name="Liquid Nitrogen (LN2)",
        bulk_density=808.0,
        inlet_temperature=77.0,
        inner_T=77.0,
        outer_T=20.0,  # second cryo
    )
    _run(cfg)
    # Both regions become cryogens via per-region inference; neither
    # gets flipped to water.
    assert cfg["regions"]["fluid"][0]["fluid_preset"] == "ln2"
    assert cfg["regions"]["fluid"][1]["fluid_preset"] == "lh2"


def test_lh2_keyword_recognised():
    cfg = _cht_cfg(
        fluid_name="liquid hydrogen",
        bulk_density=70.85,
        inlet_temperature=20.0,
        inner_T=20.0,
        outer_T=290.0,
    )
    _run(cfg)
    assert cfg["regions"]["fluid"][0]["fluid_preset"] == "lh2"


def test_cryogenic_T_only_infers_preset_when_name_is_silent():
    """No fluid name + region T_init = 77 K → infer ln2 from temperature alone."""
    cfg = _cht_cfg(
        fluid_name="",
        bulk_density=None,
        inlet_temperature=None,
        inner_T=77.0,
        outer_T=290.0,
    )
    _run(cfg)
    assert cfg["regions"]["fluid"][0]["fluid_preset"] == "ln2"


def test_explicit_preset_is_preserved():
    """A non-air preset set upstream (by extractor / UI) must never be overwritten."""
    cfg = _cht_cfg(
        fluid_name="Liquid Nitrogen (LN2)",
        bulk_density=808.0,
        inlet_temperature=77.0,
        inner_T=77.0,
        outer_T=290.0,
        inner_preset="lox",   # user / extractor said LOX
    )
    _run(cfg)
    # Heuristic must NOT downgrade the explicit LOX choice
    assert cfg["regions"]["fluid"][0]["fluid_preset"] == "lox"


def test_pure_air_case_left_alone():
    """No cryo signals at all → both regions stay 'air'."""
    cfg = _cht_cfg(
        fluid_name="",
        bulk_density=1.2,
        inlet_temperature=293.0,
        inner_T=293.0,
        outer_T=293.0,
    )
    _run(cfg)
    for r in cfg["regions"]["fluid"]:
        assert r["fluid_preset"] == "air"


def test_water_inferred_from_room_temp_high_density():
    """Room-T region + bulk_density ≈ 998 → water."""
    cfg = _cht_cfg(
        fluid_name="water",
        bulk_density=998.0,
        inlet_temperature=295.0,
        inner_T=295.0,
        outer_T=295.0,
    )
    _run(cfg)
    for r in cfg["regions"]["fluid"]:
        assert r["fluid_preset"] == "water"


def test_no_fluid_regions_is_noop():
    """Single-region cases (no regions block) must not crash."""
    cfg = {"fluid": {"name": "Liquid Nitrogen (LN2)"}}
    _run(cfg)  # should not raise
    # And nothing should be added
    assert "regions" not in cfg
