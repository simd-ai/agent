# tests/test_enrichment_wall_bcs.py
"""Unit tests for the wall-temperature BC normaliser.

The step bridges the lowercase ``temperature`` key the linter writes
(via :class:`BoundaryConditionV1.temperature`) to the uppercase ``T``
key the multi-region BC renderer reads.  Source of truth is
``case_defaults["wall_temperatures"]`` — itself resolved earlier in
the pipeline from the same lowercase user inputs.
"""

from __future__ import annotations

import asyncio

import pytest

from simd_agent.run.enrichment import EnrichmentContext
from simd_agent.run.enrichment.wall_bcs import apply


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _run(config: dict) -> EnrichmentContext:
    ctx = EnrichmentContext(config=config, user_requirements="")
    asyncio.run(apply(ctx))
    return ctx


# ────────────────────────────────────────────────────────────────────────────
# Primary contract
# ────────────────────────────────────────────────────────────────────────────


def test_writes_canonical_T_for_each_wall_in_case_defaults():
    cfg = {
        "case_defaults": {
            "wall_temperatures": {"hot_wall": 400.0, "cold_wall": 290.0},
        },
        "boundary_conditions": {},
    }
    _run(cfg)
    assert cfg["boundary_conditions"]["hot_wall"]["T"] == {
        "type": "fixedValue", "value": 400.0,
    }
    assert cfg["boundary_conditions"]["cold_wall"]["T"] == {
        "type": "fixedValue", "value": 290.0,
    }


def test_creates_bc_entry_for_wall_with_no_prior_bc():
    cfg = {
        "case_defaults": {"wall_temperatures": {"hot_wall": 400.0}},
        # No boundary_conditions block at all
    }
    _run(cfg)
    assert cfg["boundary_conditions"]["hot_wall"]["T"]["value"] == 400.0


def test_preserves_existing_canonical_T():
    cfg = {
        "case_defaults": {"wall_temperatures": {"hot_wall": 400.0}},
        "boundary_conditions": {
            "hot_wall": {"T": {"type": "fixedValue", "value": 350.0}},
        },
    }
    _run(cfg)
    # User-set canonical T preserved exactly
    assert cfg["boundary_conditions"]["hot_wall"]["T"]["value"] == 350.0


def test_preserves_existing_bare_scalar_T():
    """Older configs store T as a bare scalar — that still counts as user intent."""
    cfg = {
        "case_defaults": {"wall_temperatures": {"hot_wall": 400.0}},
        "boundary_conditions": {"hot_wall": {"T": 350.0}},
    }
    _run(cfg)
    assert cfg["boundary_conditions"]["hot_wall"]["T"] == 350.0


def test_overwrites_when_existing_T_value_is_zero_or_missing():
    """A patch with placeholder/zero ``T`` is treated as no real value."""
    cfg = {
        "case_defaults": {"wall_temperatures": {"hot_wall": 400.0}},
        "boundary_conditions": {
            "hot_wall": {"T": {"type": "fixedValue", "value": 0.0}},
        },
    }
    _run(cfg)
    assert cfg["boundary_conditions"]["hot_wall"]["T"]["value"] == 400.0


def test_preserves_unrelated_fields_on_the_patch():
    """Step only writes ``T`` — other fields (U, k, …) on the patch are untouched."""
    cfg = {
        "case_defaults": {"wall_temperatures": {"hot_wall": 400.0}},
        "boundary_conditions": {
            "hot_wall": {
                "U": {"type": "noSlip"},
                "k": {"type": "kqRWallFunction", "value": 0.001},
            },
        },
    }
    _run(cfg)
    assert cfg["boundary_conditions"]["hot_wall"]["T"]["value"] == 400.0
    assert cfg["boundary_conditions"]["hot_wall"]["U"] == {"type": "noSlip"}
    assert cfg["boundary_conditions"]["hot_wall"]["k"]["value"] == 0.001


# ────────────────────────────────────────────────────────────────────────────
# Defensive cases
# ────────────────────────────────────────────────────────────────────────────


def test_noop_when_case_defaults_missing():
    cfg: dict = {"boundary_conditions": {}}
    _run(cfg)
    assert cfg["boundary_conditions"] == {}


def test_noop_when_wall_temperatures_empty():
    cfg = {"case_defaults": {"wall_temperatures": {}}, "boundary_conditions": {}}
    _run(cfg)
    assert cfg["boundary_conditions"] == {}


def test_skips_non_positive_temperatures():
    """Defensive: negative / zero / non-numeric values are ignored, not written."""
    cfg = {
        "case_defaults": {"wall_temperatures": {
            "good": 400.0,
            "zero": 0.0,
            "negative": -10.0,
            "garbage": "hot",
        }},
        "boundary_conditions": {},
    }
    _run(cfg)
    assert "good" in cfg["boundary_conditions"]
    assert "zero" not in cfg["boundary_conditions"]
    assert "negative" not in cfg["boundary_conditions"]
    assert "garbage" not in cfg["boundary_conditions"]


def test_handles_missing_boundary_conditions_dict():
    """Step creates ``boundary_conditions`` on demand."""
    cfg = {"case_defaults": {"wall_temperatures": {"hot_wall": 400.0}}}
    _run(cfg)
    assert cfg["boundary_conditions"]["hot_wall"]["T"]["value"] == 400.0


# ────────────────────────────────────────────────────────────────────────────
# Integration: the lowercase → uppercase bridge end-to-end
# ────────────────────────────────────────────────────────────────────────────


def test_end_to_end_temperature_to_T_bridge():
    """Closes the linter ``temperature`` → renderer ``T`` shape gap.

    Simulates what happens when the user sets a wall temperature in the
    UI: the linter writes ``boundary_conditions[<wall>]["temperature"]``;
    the case_defaults step picks it up; this step copies it onto the
    canonical ``"T"`` key that the multi-region renderer + LLM filler
    actually read.
    """
    from simd_agent.run.enrichment.case_defaults import apply as resolve_defaults

    cfg = {
        "boundary_conditions": {
            "outer_wall": {
                "patch_class": "wall",
                "temperature": {"type": "fixedValue", "value": 350.0},
            },
        },
    }
    # case_defaults resolves wall_temperatures from the lowercase user input
    asyncio.run(resolve_defaults(EnrichmentContext(config=cfg, user_requirements="")))
    assert cfg["case_defaults"]["wall_temperatures"] == {"outer_wall": 350.0}
    # ``T`` (uppercase) does not exist yet
    assert "T" not in cfg["boundary_conditions"]["outer_wall"]

    # wall_bcs writes the canonical T BC
    _run(cfg)
    assert cfg["boundary_conditions"]["outer_wall"]["T"] == {
        "type": "fixedValue", "value": 350.0,
    }
