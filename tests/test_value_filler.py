# tests/test_value_filler.py
"""Unit tests for the value filler — covers both single- and multi-region paths.

Network-touching code (the actual Gemini call) is stubbed via
``monkeypatch`` so these tests are fast and offline.  Coverage:

  * Path routing: ``0/<field>`` → single-region, ``0/<region>/<field>`` →
    multi-region, anything else → no-op.
  * Single-region: filler reads per-patch BCs + ``case_defaults`` and the
    LLM-rewritten body lands in the output.
  * Multi-region: structural sanity check rejects a response that drops
    a patch entry; the deterministic template is preserved.
  * Failure handling: LLM raises → original kept, no exception propagates.
  * Byte-identical: response matches template → silently kept as-is.
  * Markdown fences: ``` blocks are stripped before structural check.
"""

from __future__ import annotations

import asyncio

import pytest

from simd_agent.run.value_filler import filler as filler_mod
from simd_agent.run.value_filler import fill_field_values
from simd_agent.run.value_filler.contexts import build_for_path
from simd_agent.run.value_filler.validation import (
    extract_file_body,
    looks_structurally_sound,
)


# ────────────────────────────────────────────────────────────────────────────
# Fixtures — deterministic templates
# ────────────────────────────────────────────────────────────────────────────


_SINGLE_T = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0";
    object      T;
}
dimensions      [0 0 0 1 0 0 0];
internalField   uniform 300;
boundaryField
{
    inlet
    {
        type            fixedValue;
        value           uniform 300;
    }
    outlet
    {
        type            inletOutlet;
        inletValue      uniform 300;
        value           uniform 300;
    }
    wall
    {
        type            zeroGradient;
    }
}
"""


_MULTI_T_INNER = """\
FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    location    "0/innerFluid";
    object      T;
}
dimensions      [0 0 0 1 0 0 0];
internalField   uniform 300;
boundaryField
{
    innerFluid_inlet
    {
        type            fixedValue;
        value           uniform 300;
    }
    innerFluid_outlet
    {
        type            inletOutlet;
        inletValue      uniform 300;
        value           uniform 300;
    }
    innerFluid_to_wall
    {
        type            compressible::turbulentTemperatureCoupledBaffleMixed;
        Tnbr            T;
        kappaMethod     fluidThermo;
        value           uniform 300;
    }
}
"""


def _single_region_config() -> dict:
    return {
        "mesh": {"patches": [
            {"name": "inlet"}, {"name": "outlet"}, {"name": "wall"},
        ]},
        "fluid": {"name": "water"},
        "case_defaults": {
            "inlet_velocity":    (1.5, 0.0, 0.0),
            "inlet_temperature": 290.0,
            "inlet_pressure":    101325.0,
            "bulk_temperature":  290.0,
            "bulk_density":      998.0,
        },
        "boundary_conditions": {},
    }


def _cht_config() -> dict:
    return {
        "regions": {
            "fluid": [
                {"name": "innerFluid", "kind": "fluid",
                 "fluid_preset": "ln2", "T_init": 77.0,
                 "U_init": (0.05, 0.0, 0.0), "p_init": 101325.0,
                 "interfaces": ["wall"]},
            ],
            "solid": [],
        },
        "mesh": {"patches": [
            {"name": "innerFluid_inlet",   "type": "patch"},
            {"name": "innerFluid_outlet",  "type": "patch"},
            {"name": "innerFluid_to_wall", "type": "mappedWall"},
        ]},
        "case_defaults": {"inlet_temperature": 77.0},
        "boundary_conditions": {},
    }


# ────────────────────────────────────────────────────────────────────────────
# Stub provider plumbing
# ────────────────────────────────────────────────────────────────────────────


class _StubResponse:
    def __init__(self, text: str):
        self.candidates = [
            type("C", (), {
                "content": type("Co", (), {
                    "parts": [type("P", (), {"text": text})()],
                })(),
            })()
        ]


class _StubClient:
    """Drop-in for ``provider.client`` — matches the path inside the
    contents string to a recorded reply and echoes it back."""

    def __init__(self, replies):
        self._replies = replies
        self.calls: list[str] = []
        self.aio = type("Aio", (), {"models": self})()  # client.aio.models.generate_content(...)

    async def generate_content(self, *, model, contents):
        self.calls.append(contents)
        if isinstance(self._replies, Exception):
            raise self._replies
        for path, body in self._replies.items():
            if path in contents:
                return _StubResponse(body)
        raise RuntimeError(f"_StubClient got unexpected prompt: {contents[:200]!r}")


class _StubProvider:
    def __init__(self, client):
        self.client = client
        self.models = {"default": "gemini-stub", "super": "gemini-stub"}


def _install_stub(monkeypatch, replies):
    client = _StubClient(replies)
    monkeypatch.setattr(filler_mod, "get_provider", lambda: _StubProvider(client))
    return client


# ────────────────────────────────────────────────────────────────────────────
# Path routing — auto-classifies single vs multi vs non-target
# ────────────────────────────────────────────────────────────────────────────


def test_path_router_single_region_T():
    ctx = build_for_path("0/T", _single_region_config())
    assert ctx is not None
    assert ctx["mode"] == "single"
    assert ctx["field"] == "T"


def test_path_router_multi_region_T():
    ctx = build_for_path("0/innerFluid/T", _cht_config())
    assert ctx is not None
    assert ctx["mode"] == "multi"
    assert ctx["name"] == "innerFluid"
    assert ctx["field"] == "T"


@pytest.mark.parametrize("path", [
    "system/controlDict",
    "constant/transportProperties",
    "0/k",            # turbulence field — outside the target set
    "0/innerFluid/k", # same, multi-region path
    "0/some/nested/path",
])
def test_path_router_skips_non_targets(path):
    assert build_for_path(path, _single_region_config()) is None


def test_path_router_returns_none_when_region_unknown():
    """Multi-region path referencing a region that isn't in ``regions`` → None."""
    ctx = build_for_path("0/ghostRegion/T", _cht_config())
    assert ctx is None


# ────────────────────────────────────────────────────────────────────────────
# Single-region: end-to-end
# ────────────────────────────────────────────────────────────────────────────


def test_single_region_rewrite_lands_in_output(monkeypatch):
    rewritten = _SINGLE_T.replace("300", "290")
    _install_stub(monkeypatch, {"0/T": rewritten})

    files = {
        "0/T":                _SINGLE_T,
        "system/controlDict": "// dont touch me",
    }
    out = asyncio.run(fill_field_values(files, _single_region_config(), "water case"))
    assert "uniform 290" in out["0/T"]
    assert out["system/controlDict"] == "// dont touch me"


def test_single_region_byte_identical_response_is_kept_as_template(monkeypatch):
    """Byte-identical = LLM agreed with template → no-op safety net.

    Uses a rstrip-ed template because :func:`extract_file_body` always
    strips trailing whitespace from the LLM response.  With a no-trailing-
    whitespace template the round-trip is exact and the byte-identical
    rejection path fires — the filler returns ``None`` and the input
    files dict is preserved.
    """
    template = _SINGLE_T.rstrip()
    _install_stub(monkeypatch, {"0/T": template})
    files = {"0/T": template}
    out = asyncio.run(fill_field_values(files, _single_region_config(), "prompt"))
    assert out["0/T"] == template


def test_single_region_surfaces_per_patch_value_into_ctx():
    """Per-patch T values flow into the per-file ctx → into the prompt table."""
    cfg = _single_region_config()
    cfg["boundary_conditions"]["inlet"] = {
        "patch_class": "inlet",
        "T": {"type": "fixedValue", "value": 77.0},
    }
    ctx = build_for_path("0/T", cfg)
    inlet = next(p for p in ctx["patches"] if p["name"] == "inlet")
    assert inlet["T_value"] == 77.0
    assert inlet["T_type"]  == "fixedValue"


# ────────────────────────────────────────────────────────────────────────────
# Multi-region: end-to-end + structural sanity check
# ────────────────────────────────────────────────────────────────────────────


def test_multi_region_rewrite_lands_in_output(monkeypatch):
    rewritten = _MULTI_T_INNER.replace("300", "77")
    _install_stub(monkeypatch, {"0/innerFluid/T": rewritten})

    files = {"0/innerFluid/T": _MULTI_T_INNER}
    out = asyncio.run(fill_field_values(files, _cht_config(), "innerFluid at 77 K"))
    # The filler strips trailing whitespace from the LLM response
    assert "uniform 77" in out["0/innerFluid/T"]


def test_multi_region_structurally_unsound_response_falls_back(monkeypatch):
    """LLM drops a patch → sanity check rejects it, template kept."""
    bad = """\
FoamFile { object T; }
dimensions [0 0 0 1 0 0 0];
internalField uniform 77;
boundaryField
{
    innerFluid_inlet
    {
        type            fixedValue;
        value           uniform 77;
    }
}
"""
    _install_stub(monkeypatch, {"0/innerFluid/T": bad})
    files = {"0/innerFluid/T": _MULTI_T_INNER}
    out = asyncio.run(fill_field_values(files, _cht_config(), "prompt"))
    assert out["0/innerFluid/T"] == _MULTI_T_INNER  # template preserved


# ────────────────────────────────────────────────────────────────────────────
# Failure handling + markdown stripping
# ────────────────────────────────────────────────────────────────────────────


def test_llm_call_failure_keeps_template(monkeypatch):
    _install_stub(monkeypatch, RuntimeError("API down"))
    files = {"0/innerFluid/T": _MULTI_T_INNER}
    out = asyncio.run(fill_field_values(files, _cht_config(), "prompt"))
    assert out["0/innerFluid/T"] == _MULTI_T_INNER


def test_strips_markdown_fences(monkeypatch):
    raw = "```cpp\n" + _MULTI_T_INNER.replace("300", "77") + "\n```"
    _install_stub(monkeypatch, {"0/innerFluid/T": raw})
    files = {"0/innerFluid/T": _MULTI_T_INNER}
    out = asyncio.run(fill_field_values(files, _cht_config(), "prompt"))
    assert "```" not in out["0/innerFluid/T"]
    assert "uniform 77" in out["0/innerFluid/T"]


# ────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ────────────────────────────────────────────────────────────────────────────


def test_sanity_check_passes_when_all_patches_present():
    assert looks_structurally_sound(_MULTI_T_INNER, _MULTI_T_INNER)


def test_sanity_check_fails_when_a_patch_is_missing():
    truncated = _MULTI_T_INNER.replace(
        "    innerFluid_outlet\n    {\n        "
        "type            inletOutlet;\n        "
        "inletValue      uniform 300;\n        "
        "value           uniform 300;\n    }\n",
        "",
    )
    assert looks_structurally_sound(truncated, _MULTI_T_INNER) is False


def test_sanity_check_fails_without_foamfile_header():
    assert looks_structurally_sound("garbage that is not openfoam", _MULTI_T_INNER) is False


def test_extract_file_body_strips_single_leading_and_trailing_fences():
    resp = _StubResponse("```\nFoamFile {}\n```")
    assert extract_file_body(resp) == "FoamFile {}"
