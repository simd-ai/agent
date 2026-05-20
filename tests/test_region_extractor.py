# tests/test_region_extractor.py
"""Unit tests for the multi-region detail extractor.

The LLM call itself is integration-tested via the live API in
``test_integration_real.py`` — these tests cover everything else:
the merge logic, the tool schema, the system-prompt formatting, and
the no-op behaviour on empty / single-region input.
"""

from __future__ import annotations

import pytest

from simd_agent.run.region_extractor import (
    RegionExtractor,
    _FLUID_PRESETS,
    _SOLID_PRESETS,
)


# Heuristic-detected regions for the Regascold cyl_cht_2d mesh: two fluid
# regions (air / air defaulted from the auto-detector) + one stainless wall.
_HEURISTIC = [
    {"name": "innerFluid", "kind": "fluid",
     "fluid_preset": "air",       "interfaces": ["wall"]},
    {"name": "wall",       "kind": "solid",
     "solid_preset": "stainless", "interfaces": ["innerFluid", "outerFluid"]},
    {"name": "outerFluid", "kind": "fluid",
     "fluid_preset": "air",       "interfaces": ["wall"]},
]


class TestMerge:
    """LLM details overlay onto the heuristic-built region list."""

    def test_fluid_preset_overridden_by_llm(self):
        llm = [
            {"name": "innerFluid", "kind": "fluid", "fluid_preset": "ln2"},
            {"name": "outerFluid", "kind": "fluid", "fluid_preset": "water"},
        ]
        merged = RegionExtractor._merge(_HEURISTIC, llm)
        by_name = {r["name"]: r for r in merged}
        assert by_name["innerFluid"]["fluid_preset"] == "ln2"
        assert by_name["outerFluid"]["fluid_preset"] == "water"
        # Solid kept its heuristic value (LLM didn't speak about it)
        assert by_name["wall"]["solid_preset"] == "stainless"

    def test_inlet_velocity_packed_as_3_tuple(self):
        llm = [
            {"name": "innerFluid", "kind": "fluid", "inlet_velocity": 0.05},
            {"name": "outerFluid", "kind": "fluid", "inlet_velocity": -0.10},
        ]
        merged = RegionExtractor._merge(_HEURISTIC, llm)
        by_name = {r["name"]: r for r in merged}
        # Counter-flow: signs preserved into (Ux, Uy, Uz)
        assert by_name["innerFluid"]["U_init"] == (0.05, 0.0, 0.0)
        assert by_name["outerFluid"]["U_init"] == (-0.10, 0.0, 0.0)

    def test_temperature_and_pressure_passthrough(self):
        llm = [
            {"name": "innerFluid", "kind": "fluid",
             "inlet_temperature": 77.0, "inlet_pressure": 101325},
            {"name": "outerFluid", "kind": "fluid",
             "inlet_temperature": 290.0},
        ]
        merged = RegionExtractor._merge(_HEURISTIC, llm)
        by_name = {r["name"]: r for r in merged}
        assert by_name["innerFluid"]["T_init"] == 77.0
        assert by_name["innerFluid"]["p_init"] == 101325.0
        assert by_name["outerFluid"]["T_init"] == 290.0
        # No p_init given for outer → key omitted, not set to None
        assert "p_init" not in by_name["outerFluid"]

    def test_turbulence_model_passthrough(self):
        llm = [
            {"name": "innerFluid", "kind": "fluid", "turbulence_model": "laminar"},
            {"name": "outerFluid", "kind": "fluid", "turbulence_model": "kOmegaSST"},
        ]
        merged = RegionExtractor._merge(_HEURISTIC, llm)
        by_name = {r["name"]: r for r in merged}
        assert by_name["innerFluid"]["turbulence_model"] == "laminar"
        assert by_name["outerFluid"]["turbulence_model"] == "kOmegaSST"

    def test_llm_disagreement_on_kind_wins(self):
        # If the LLM says the wall is actually a fluid (e.g. user prompt
        # described 3 fluids), trust the prompt.
        weird_heuristic = [
            {"name": "rA", "kind": "solid", "solid_preset": "steel"},
            {"name": "rB", "kind": "fluid", "fluid_preset": "air"},
        ]
        llm = [
            {"name": "rA", "kind": "fluid", "fluid_preset": "water"},
        ]
        merged = RegionExtractor._merge(weird_heuristic, llm)
        by_name = {r["name"]: r for r in merged}
        assert by_name["rA"]["kind"] == "fluid"
        assert by_name["rA"]["fluid_preset"] == "water"
        # solid_preset dropped because LLM reclassified to fluid
        assert "solid_preset" not in by_name["rA"]

    def test_interfaces_preserved(self):
        llm = [{"name": "innerFluid", "kind": "fluid", "fluid_preset": "ln2"}]
        merged = RegionExtractor._merge(_HEURISTIC, llm)
        by_name = {r["name"]: r for r in merged}
        # Topology comes from the heuristic detector — must survive merge.
        assert by_name["innerFluid"]["interfaces"] == ["wall"]
        assert by_name["wall"]["interfaces"] == ["innerFluid", "outerFluid"]

    def test_extra_llm_regions_are_dropped(self):
        # If the LLM hallucinates a region the mesh doesn't have, ignore it.
        llm = [
            {"name": "innerFluid", "kind": "fluid", "fluid_preset": "ln2"},
            {"name": "ghost",      "kind": "fluid", "fluid_preset": "air"},
        ]
        merged = RegionExtractor._merge(_HEURISTIC, llm)
        names = [r["name"] for r in merged]
        assert names == ["innerFluid", "wall", "outerFluid"]


class TestToolSchema:
    """The forced-tool schema enumerates only registered presets."""

    def test_tool_constructs_with_real_provider_types(self):
        rx = RegionExtractor()
        tool = rx._build_tool(rx._provider.types, ["a", "b", "c"])
        fd = tool.function_declarations[0]
        assert fd.name == "report_region_details"

        entry = fd.parameters.properties["regions"].items
        props = entry.properties
        # Names enum reflects what we passed in
        assert list(props["name"].enum) == ["a", "b", "c"]
        # Presets enums come from the MultiRegionBase tables
        assert set(props["fluid_preset"].enum) == set(_FLUID_PRESETS)
        assert set(props["solid_preset"].enum) == set(_SOLID_PRESETS)
        # Required fields are minimal — extras are optional / nullable
        assert set(entry.required) == {"name", "kind"}

    def test_system_prompt_lists_regions(self):
        sp = RegionExtractor._system_prompt(_HEURISTIC)
        assert "innerFluid" in sp
        assert "wall" in sp
        assert "outerFluid" in sp
        # Reminds the model about the topology
        assert "interfaces=wall" in sp


class TestExtractNoOpPaths:
    """The extractor is a no-op when the input doesn't warrant a call."""

    @pytest.mark.asyncio
    async def test_empty_prompt_returns_input(self):
        out = await RegionExtractor().extract("", _HEURISTIC)
        assert out is _HEURISTIC

    @pytest.mark.asyncio
    async def test_whitespace_prompt_returns_input(self):
        out = await RegionExtractor().extract("   \n  \t  ", _HEURISTIC)
        assert out is _HEURISTIC

    @pytest.mark.asyncio
    async def test_empty_regions_returns_input(self):
        out = await RegionExtractor().extract("a prompt", [])
        assert out == []

    @pytest.mark.asyncio
    async def test_single_region_is_passthrough(self):
        # Single-region cases never reach the multi-region plugin, but
        # if someone calls the extractor with one region the API call
        # would be wasted — short-circuit and return the input.
        one = [{"name": "fluid", "kind": "fluid", "fluid_preset": "air"}]
        out = await RegionExtractor().extract("a prompt", one)
        assert out is one
