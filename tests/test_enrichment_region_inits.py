# tests/test_enrichment_region_inits.py
"""Unit tests for the per-region init backfill.

The step reads ``case_defaults`` (set by the earlier step in the
pipeline) and fills in any region whose extractor-supplied init is
``None``.  These tests stage ``case_defaults`` directly so each rule
is exercised in isolation, independent of the upstream resolver.
"""

from __future__ import annotations

import asyncio

import pytest

from simd_agent.run.enrichment import EnrichmentContext
from simd_agent.run.enrichment.region_inits import apply


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────


def _two_fluid_one_solid(*, T_inits=(None, None), U_inits=(None, None), p_inits=(None, None)) -> dict:
    """CHT topology with knobs for whether each fluid region already has inits."""
    return {
        "regions": {
            "fluid": [
                {"name": "innerFluid", "kind": "fluid", "T_init": T_inits[0],
                 "U_init": U_inits[0], "p_init": p_inits[0], "interfaces": ["wall"]},
                {"name": "outerFluid", "kind": "fluid", "T_init": T_inits[1],
                 "U_init": U_inits[1], "p_init": p_inits[1], "interfaces": ["wall"]},
            ],
            "solid": [
                {"name": "wall", "kind": "solid",
                 "interfaces": ["innerFluid", "outerFluid"]},
            ],
        },
    }


def _seed_case_defaults(config: dict, **overrides) -> None:
    base = {
        "inlet_velocity":    None,
        "inlet_temperature": None,
        "inlet_pressure":    None,
        "ambient_pressure":  None,
        "bulk_temperature":  None,
    }
    base.update(overrides)
    config["case_defaults"] = base


def _run(config: dict) -> EnrichmentContext:
    ctx = EnrichmentContext(config=config, user_requirements="")
    asyncio.run(apply(ctx))
    return ctx


# ────────────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────────────


def test_seeds_missing_inits_from_case_defaults():
    cfg = _two_fluid_one_solid()
    _seed_case_defaults(
        cfg,
        inlet_velocity=(0.5, 0.0, 0.0),
        inlet_temperature=350.0,
        inlet_pressure=101325.0,
    )
    _run(cfg)
    for r in cfg["regions"]["fluid"]:
        assert r["T_init"] == 350.0
        assert r["U_init"] == (0.5, 0.0, 0.0)
        assert r["p_init"] == 101325.0


def test_existing_region_inits_never_overridden():
    cfg = _two_fluid_one_solid(
        T_inits=(77.0, None),                # innerFluid already has T from extractor
        U_inits=((0.05, 0.0, 0.0), None),
    )
    _seed_case_defaults(
        cfg,
        inlet_velocity=(0.5, 0.0, 0.0),
        inlet_temperature=350.0,
    )
    _run(cfg)
    # innerFluid keeps its extractor-supplied values
    assert cfg["regions"]["fluid"][0]["T_init"] == 77.0
    assert cfg["regions"]["fluid"][0]["U_init"] == (0.05, 0.0, 0.0)
    # outerFluid takes the case-level fallback
    assert cfg["regions"]["fluid"][1]["T_init"] == 350.0
    assert cfg["regions"]["fluid"][1]["U_init"] == (0.5, 0.0, 0.0)


def test_temperature_uses_bulk_when_no_inlet_temperature():
    cfg = _two_fluid_one_solid()
    _seed_case_defaults(cfg, bulk_temperature=290.0)
    _run(cfg)
    for r in cfg["regions"]["fluid"]:
        assert r["T_init"] == 290.0


def test_pressure_uses_ambient_when_no_inlet_pressure():
    cfg = _two_fluid_one_solid()
    _seed_case_defaults(cfg, ambient_pressure=101325.0)
    _run(cfg)
    for r in cfg["regions"]["fluid"]:
        assert r["p_init"] == 101325.0


def test_noop_when_no_signals():
    cfg = _two_fluid_one_solid()
    _seed_case_defaults(cfg)  # all None
    ctx = _run(cfg)
    for r in cfg["regions"]["fluid"]:
        assert r["T_init"] is None
        assert r["U_init"] is None
        assert r["p_init"] is None
    # No info issue emitted when nothing changed.
    assert all(i.code != "BACKFILLED" for i in ctx.issues)


def test_noop_for_single_region_case():
    """No ``regions`` block → return immediately, no error."""
    cfg = {"case_defaults": {"inlet_temperature": 350.0,
                             "inlet_velocity": (0.5, 0.0, 0.0)}}
    _run(cfg)  # must not raise


def test_emits_info_issue_when_any_region_changed():
    cfg = _two_fluid_one_solid()
    _seed_case_defaults(cfg, inlet_temperature=350.0)
    ctx = _run(cfg)
    backfill_issues = [i for i in ctx.issues if i.code == "BACKFILLED"]
    assert len(backfill_issues) == 1
    assert backfill_issues[0].payload is not None
    assert len(backfill_issues[0].payload["changes"]) == 2
