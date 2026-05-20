# tests/test_enrichment_pipeline.py
"""Integration tests for the enrichment pipeline.

End-to-end exercises across the full default step list, including
RegionExtractor stubbing so the LLM step doesn't reach for the
network during CI.  These tests pin the contract between steps —
edit them when you change a step's I/O.
"""

from __future__ import annotations

import asyncio

import pytest

from simd_agent.run.enrichment import (
    DEFAULT_STEPS,
    EnrichmentContext,
    EnrichmentIssue,
    enrich_validated_config,
)
from simd_agent.run.enrichment import region_details


# ────────────────────────────────────────────────────────────────────────────
# RegionExtractor stub — no network, predictable outputs
# ────────────────────────────────────────────────────────────────────────────


class _StubRegionExtractor:
    """Drop-in for RegionExtractor that echoes the input regions list.

    By default it returns whatever was passed in (no-op LLM).  Override
    ``self.refined`` to simulate an LLM that filled in fields.
    """

    refined: list[dict] | None = None

    async def extract(self, user_requirements, regions):
        if self.refined is not None:
            return self.refined
        return regions


@pytest.fixture(autouse=True)
def _stub_region_extractor(monkeypatch):
    monkeypatch.setattr(region_details, "RegionExtractor", _StubRegionExtractor)


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────


def _generic_cht_config() -> dict:
    """The user's regression scenario: CHT mesh, generic prompt, no per-region
    signals, but a case-level bulk T + inlet velocity from the wizard."""
    return {
        "mesh": {"patches": [
            {"name": "innerFluid_inlet"}, {"name": "innerFluid_outlet"},
            {"name": "innerFluid_symmetry"}, {"name": "innerFluid_to_wall"},
            {"name": "outerFluid_inlet"}, {"name": "outerFluid_outlet"},
            {"name": "outerFluid_top"}, {"name": "outerFluid_to_wall"},
            {"name": "wall_left_end"},   {"name": "wall_right_end"},
            {"name": "wall_to_innerFluid"}, {"name": "wall_to_outerFluid"},
        ]},
        "boundary_conditions": {},
        "fluid":  {"temperature": 350.0},
        "inlet":  {"velocity": 0.5},
        "outlet": {"pressure": 101325.0},
    }


def _single_region_config() -> dict:
    """No region prefixes in mesh patches → topology stays single-region."""
    return {
        "mesh": {"patches": [
            {"name": "inlet"}, {"name": "outlet"}, {"name": "wall"},
        ]},
        "boundary_conditions": {},
        "fluid": {"temperature": 300.0},
        "inlet": {"velocity": 1.5},
    }


# ────────────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────────────


def test_end_to_end_cht_user_regression():
    """The exact regression scenario: case-level T + U flow into every inlet."""
    cfg = _generic_cht_config()
    issues = asyncio.run(enrich_validated_config(
        config=cfg,
        user_requirements="laminar steady CFD with heat transfer",
    ))

    # case_defaults populated
    assert cfg["case_defaults"]["inlet_velocity"]   == (0.5, 0.0, 0.0)
    assert cfg["case_defaults"]["inlet_temperature"] == 350.0
    assert cfg["case_defaults"]["ambient_pressure"]  == 101325.0

    # Topology auto-detected
    assert cfg["regions"]["fluid"] and cfg["regions"]["solid"]

    # Per-region inits backfilled
    for r in cfg["regions"]["fluid"]:
        assert r["T_init"] == 350.0
        assert r["U_init"] == (0.5, 0.0, 0.0)
        assert r["p_init"] == 101325.0

    # Per-patch BCs propagated for inlets
    assert cfg["boundary_conditions"]["innerFluid_inlet"]["T"]["value"] == 350.0
    assert cfg["boundary_conditions"]["innerFluid_inlet"]["U"]["value"] == [0.5, 0.0, 0.0]
    assert cfg["boundary_conditions"]["outerFluid_inlet"]["T"]["value"] == 350.0

    # No errors — pipeline ran to completion
    assert not any(i.severity == "error" for i in issues)


def test_single_region_skips_region_steps():
    """No regions key after pipeline → multi-region steps are no-ops."""
    cfg = _single_region_config()
    asyncio.run(enrich_validated_config(config=cfg, user_requirements=""))
    # case_defaults still resolved
    assert cfg["case_defaults"]["inlet_velocity"] == (1.5, 0.0, 0.0)
    # regions block never appears
    assert "regions" not in cfg


def test_extractor_values_win_over_case_defaults():
    """Real prompt-derived inits are not overwritten by the case-level fallback."""
    cfg = _generic_cht_config()

    # Stub the extractor to fill innerFluid with explicit LN2 inputs
    class _Refined(_StubRegionExtractor):
        refined = [
            {"name": "innerFluid", "kind": "fluid", "fluid_preset": "ln2",
             "T_init": 77.0, "U_init": (0.05, 0.0, 0.0), "interfaces": ["wall"]},
            {"name": "outerFluid", "kind": "fluid", "fluid_preset": "air",
             "interfaces": ["wall"]},   # no T/U → case-level fallback applies
            {"name": "wall", "kind": "solid", "solid_preset": "stainless",
             "interfaces": ["innerFluid", "outerFluid"]},
        ]
    import simd_agent.run.enrichment.region_details as rd
    original = rd.RegionExtractor
    rd.RegionExtractor = _Refined
    try:
        asyncio.run(enrich_validated_config(
            config=cfg, user_requirements="innerFluid is LN2 at 77 K",
        ))
    finally:
        rd.RegionExtractor = original

    inner = next(r for r in cfg["regions"]["fluid"] if r["name"] == "innerFluid")
    outer = next(r for r in cfg["regions"]["fluid"] if r["name"] == "outerFluid")
    assert inner["T_init"] == 77.0                    # extractor value preserved
    assert inner["U_init"] == (0.05, 0.0, 0.0)
    assert outer["T_init"] == 350.0                   # case-level fallback
    assert outer["U_init"] == (0.5, 0.0, 0.0)


def test_extractor_failure_is_a_warning_not_an_error():
    """LLM/network failure logs a warning issue but the pipeline still completes."""
    cfg = _generic_cht_config()

    class _Failing(_StubRegionExtractor):
        async def extract(self, user_requirements, regions):
            raise RuntimeError("LLM unavailable")
    import simd_agent.run.enrichment.region_details as rd
    original = rd.RegionExtractor
    rd.RegionExtractor = _Failing
    try:
        issues = asyncio.run(enrich_validated_config(
            config=cfg, user_requirements="something",
        ))
    finally:
        rd.RegionExtractor = original

    extractor_warnings = [
        i for i in issues
        if i.severity == "warning" and i.step == "region_details"
    ]
    assert extractor_warnings, "extractor failure should surface as a warning"
    # The rest of the pipeline still ran — case-level fallback applied
    assert cfg["regions"]["fluid"][0]["T_init"] == 350.0
    assert cfg["regions"]["fluid"][0]["U_init"] == (0.5, 0.0, 0.0)


def test_custom_step_list_can_replace_defaults():
    """The pipeline accepts a custom step list (for tests / custom flows)."""
    cfg = {"fluid": {"temperature": 290.0}}
    from simd_agent.run.enrichment import case_defaults
    issues = asyncio.run(enrich_validated_config(
        config=cfg,
        user_requirements="",
        steps=(case_defaults.apply,),  # only the first step
    ))
    assert cfg["case_defaults"]["bulk_temperature"] == 290.0
    assert "regions" not in cfg
    assert len(issues) == 1


def test_pipeline_halts_on_error_severity():
    """A custom step that adds an error issue stops the pipeline immediately."""
    seen: list[str] = []

    async def step_a(ctx):
        seen.append("a")
        ctx.add_error("step_a", code="BOOM", message="forced halt")

    async def step_b(ctx):
        seen.append("b")  # must never run

    issues = asyncio.run(enrich_validated_config(
        config={}, user_requirements="",
        steps=(step_a, step_b),
    ))
    assert seen == ["a"]
    assert any(i.severity == "error" and i.code == "BOOM" for i in issues)


def test_default_step_order_is_stable():
    """Locking the published step order so accidental reorderings get caught.

    Add new steps to the tuple, but only where their dependencies allow.
    """
    names = tuple(s.__module__.rsplit(".", 1)[-1] for s in DEFAULT_STEPS)
    assert names == (
        "case_defaults",
        "region_topology",
        "region_details",
        "region_inits",
        "region_fluid_presets",
        "inlet_bcs",
        "turbulence_bcs",
        "wall_bcs",
        "topology_lint",
    )
