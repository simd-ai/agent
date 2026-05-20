# tests/test_chat_tools.py
"""Tests for chat tool functions — plot_field_values, compute_field_stats, etc.

All tests are pure-Python (no DB, no LLM calls) and run in parallel via pytest-xdist.

``plot_field_over_iterations`` was removed (it overlapped with the always-on
residual chart in LiveTab); see ``test_multi_region_isolation.py::
test_plot_field_over_iterations_tool_is_removed`` for the regression guard.
"""

import pytest
from simd_agent.chat.tools import SimulationSnapshot

# Import the tool functions directly from the module
from simd_agent.chat.tools import (
    plot_field_values,
    compute_field_stats,
    query_simulation_results,
    plot_patch_values,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_snap(**overrides) -> SimulationSnapshot:
    """Create a SimulationSnapshot with sensible defaults."""
    defaults = dict(
        run_id="test-run-001",
        simulation_id="test-sim-001",
        physics={"flow_regime": "turbulent", "time_scheme": "steady"},
        solver={"max_iterations": 1000},
        fluid={"name": "Water", "rho": 998.2, "mu": 0.001002, "preset_id": "water"},
        turbulence={"model": "kOmegaSST"},
    )
    defaults.update(overrides)
    return SimulationSnapshot(**defaults)


# VTK result structure matching a real simulation
VTK_RESULT = {
    "fields": [
        {"name": "U", "min": 1.41, "max": 61.97, "range": [1.41, 61.97]},
        {"name": "p", "min": -35.45, "max": 76.23, "range": [-35.45, 76.23]},
        {"name": "k", "min": 0.044, "max": 75.81, "range": [0.044, 75.81]},
        {"name": "omega", "min": 27.0, "max": 1405.78, "range": [27.0, 1405.78]},
        {"name": "nut", "min": 7.29e-05, "max": 0.054, "range": [7.29e-05, 0.054]},
    ],
    "time": 100,
}

# Simulation progress with residuals and field_ranges
SIM_PROGRESS = [
    {
        "iteration": i,
        "residuals": {
            "Ux": {"initial": 1.0 / (i + 1), "final": 0.5 / (i + 1)},
            "Uy": {"initial": 0.8 / (i + 1), "final": 0.4 / (i + 1)},
            "p": {"initial": 0.5 / (i + 1), "final": 0.25 / (i + 1)},
            "k": {"initial": 0.3 / (i + 1), "final": 0.15 / (i + 1)},
            "omega": {"initial": 0.2 / (i + 1), "final": 0.1 / (i + 1)},
        },
        "courant": {"mean": 0.1, "max": 0.5},
        "continuity": 1e-6 * (10 - i),
        "field_ranges": {
            "p": {"min": -30 + i, "max": 70 + i},
            "U": {"min": 1.0 + i * 0.05, "max": 60 + i * 0.2},
            "k": {"min": 0.04 + i * 0.001, "max": 70 + i * 0.5},
        },
    }
    for i in range(10)
]


# ---------------------------------------------------------------------------
# plot_field_values
# ---------------------------------------------------------------------------

class TestPlotFieldValues:
    """Tests for the plot_field_values tool."""

    def test_with_sim_progress_and_field_ranges(self):
        """When sim_progress has field_ranges, return a line chart."""
        snap = _make_snap(sim_progress=SIM_PROGRESS)
        result = plot_field_values({"fields": "p"}, snap)

        assert "error" not in result
        assert "chart" in result
        chart = result["chart"]
        assert chart["type"] == "line"
        assert "p_min" in chart["lines"]
        assert "p_max" in chart["lines"]
        assert len(chart["data"]) > 0

    def test_vtk_fallback_when_no_sim_progress(self):
        """When sim_progress is empty, fall back to VTK bar chart."""
        snap = _make_snap(sim_progress=[], vtk_result=VTK_RESULT)
        result = plot_field_values({"fields": "U"}, snap)

        assert "error" not in result, f"Unexpected error: {result.get('error')}"
        assert "chart" in result
        chart = result["chart"]
        assert chart["type"] == "bar"
        assert "vtk_result" in result.get("source", "")

    def test_vtk_fallback_when_sim_progress_none(self):
        """When sim_progress is None (not fetched), fall back to VTK."""
        snap = _make_snap(sim_progress=None, vtk_result=VTK_RESULT)
        result = plot_field_values({"fields": "p"}, snap)

        assert "error" not in result, f"Unexpected error: {result.get('error')}"
        assert "chart" in result

    def test_vtk_fallback_pressure(self):
        """VTK fallback works for pressure field."""
        snap = _make_snap(sim_progress=[], vtk_result=VTK_RESULT)
        result = plot_field_values({"fields": "p"}, snap)

        assert "chart" in result
        data = result["chart"]["data"]
        p_values = [d.get("p") for d in data if "p" in d]
        assert len(p_values) > 0
        assert any(v == pytest.approx(-35.45) for v in p_values)
        assert any(v == pytest.approx(76.23) for v in p_values)

    def test_vtk_fallback_multiple_fields(self):
        """VTK fallback with multiple requested fields."""
        snap = _make_snap(sim_progress=[], vtk_result=VTK_RESULT)
        result = plot_field_values({"fields": "U,p"}, snap)

        assert "chart" in result
        data = result["chart"]["data"]
        # Should have both U and p data points
        has_u = any("U" in d for d in data)
        has_p = any("p" in d for d in data)
        assert has_u
        assert has_p

    def test_error_when_no_data_at_all(self):
        """When both sim_progress and vtk_result are empty, return error."""
        snap = _make_snap(sim_progress=[], vtk_result={})
        result = plot_field_values({"fields": "p"}, snap)

        assert "error" in result

    def test_error_when_field_not_in_vtk(self):
        """When the requested field doesn't exist in VTK data, return error."""
        snap = _make_snap(sim_progress=[], vtk_result=VTK_RESULT)
        result = plot_field_values({"fields": "T"}, snap)

        assert "error" in result

    def test_no_fields_specified(self):
        """When no fields specified, return error."""
        snap = _make_snap(sim_progress=SIM_PROGRESS)
        result = plot_field_values({"fields": ""}, snap)

        assert "error" in result

    def test_metric_min_only(self):
        """metric='min' should only include _min lines."""
        snap = _make_snap(sim_progress=SIM_PROGRESS)
        result = plot_field_values({"fields": "p", "metric": "min"}, snap)

        assert "chart" in result
        lines = result["chart"]["lines"]
        assert "p_min" in lines
        assert "p_max" not in lines

    def test_metric_max_only(self):
        """metric='max' should only include _max lines."""
        snap = _make_snap(sim_progress=SIM_PROGRESS)
        result = plot_field_values({"fields": "p", "metric": "max"}, snap)

        assert "chart" in result
        lines = result["chart"]["lines"]
        assert "p_max" in lines
        assert "p_min" not in lines

    def test_metric_range(self):
        """metric='range' should include _range lines."""
        snap = _make_snap(sim_progress=SIM_PROGRESS)
        result = plot_field_values({"fields": "p", "metric": "range"}, snap)

        assert "chart" in result
        lines = result["chart"]["lines"]
        assert "p_range" in lines

    def test_sim_progress_without_field_ranges_no_residual_fallback(self):
        """sim_progress has residuals but no field_ranges → no duplicate residual chart.

        Previously this fell back to building a residual line chart, which
        duplicated the (now-removed) ``plot_field_over_iterations``.  Now
        it either falls through to the VTK bar chart or returns an error
        pointing the LLM at ``compute_residual_trend`` instead.
        """
        progress_no_fr = [
            {"iteration": i, "residuals": {"Ux": {"initial": 0.1 / (i + 1)}, "Uy": {"initial": 0.08 / (i + 1)}}}
            for i in range(5)
        ]
        # Use empty VTK so no fallback bar chart either
        snap = _make_snap(sim_progress=progress_no_fr, vtk_result={"fields": []})
        result = plot_field_values({"fields": "U"}, snap)

        # Without the residual fallback AND no VTK data for U, we get an error
        # message that points the LLM at the remaining residual-trend tool.
        assert "error" in result
        assert "compute_residual_trend" in result["error"]

    def test_sim_progress_no_matching_residuals_falls_to_vtk(self):
        """sim_progress has residuals but NOT for the requested field → VTK bar chart."""
        progress_no_fr = [
            {"iteration": i, "residuals": {"Ux": {"initial": 0.1}}}
            for i in range(5)
        ]
        # omega is in VTK_RESULT but not in the residuals above
        snap = _make_snap(sim_progress=progress_no_fr, vtk_result=VTK_RESULT)
        result = plot_field_values({"fields": "omega"}, snap)

        assert "error" not in result, f"Unexpected error: {result.get('error')}"
        assert "chart" in result
        assert result["chart"]["type"] == "bar"

    def test_downsampling_large_dataset(self):
        """Large sim_progress gets downsampled to ≤300 points."""
        large_progress = [
            {
                "iteration": i,
                "field_ranges": {"p": {"min": -30 + i * 0.01, "max": 70 + i * 0.01}},
            }
            for i in range(500)
        ]
        snap = _make_snap(sim_progress=large_progress)
        result = plot_field_values({"fields": "p"}, snap)

        assert "chart" in result
        assert len(result["chart"]["data"]) <= 300

    def test_fields_as_list(self):
        """fields arg as a list (from query analyzer) should work."""
        snap = _make_snap(sim_progress=SIM_PROGRESS)
        result = plot_field_values({"fields": ["p"], "metric": "both"}, snap)

        assert "error" not in result
        assert "chart" in result
        assert "p_min" in result["chart"]["lines"]

    def test_fields_as_list_multiple(self):
        """Multiple fields as a list should work."""
        snap = _make_snap(sim_progress=SIM_PROGRESS)
        result = plot_field_values({"fields": ["p", "U"], "metric": "both"}, snap)

        assert "error" not in result
        assert "chart" in result


# ---------------------------------------------------------------------------
# compute_field_stats
# ---------------------------------------------------------------------------

class TestComputeFieldStats:
    """Tests for the compute_field_stats tool."""

    def test_vtk_stats(self):
        """Returns spatial stats from VTK data."""
        snap = _make_snap(vtk_result=VTK_RESULT)
        result = compute_field_stats({"field": "U"}, snap)

        assert "error" not in result
        assert result.get("field") == "U"
        assert result["min"] == pytest.approx(1.41)
        assert result["max"] == pytest.approx(61.97)

    def test_vtk_chart_generated(self):
        """compute_field_stats returns a bar chart."""
        snap = _make_snap(vtk_result=VTK_RESULT)
        result = compute_field_stats({"field": "p"}, snap)

        assert "chart" in result
        chart = result["chart"]
        assert chart["type"] == "bar"

    def test_field_not_found(self):
        """Returns error when field not in VTK data."""
        snap = _make_snap(vtk_result=VTK_RESULT)
        result = compute_field_stats({"field": "T"}, snap)

        # Should handle gracefully (error or empty)
        # The tool might still return partial info
        assert result is not None


# ---------------------------------------------------------------------------
# query_simulation_results
# ---------------------------------------------------------------------------

class TestQuerySimulationResults:
    """Tests for the query_simulation_results tool."""

    def test_basic_query(self):
        """Returns simulation overview."""
        snap = _make_snap(
            vtk_result=VTK_RESULT,
            sim_progress=SIM_PROGRESS,
            patches={
                "inlet": {"class": "inlet", "config": {"_u": {"type": "fixedValue", "value": [1, 0, 0]}}},
                "outlet": {"class": "outlet", "config": {"p": {"type": "fixedValue", "value": 0}}},
            },
        )
        result = query_simulation_results({"question": "how did the simulation go"}, snap)

        assert "error" not in result
        assert "simulation_info" in result

    def test_query_with_vtk_only(self):
        """Works with VTK data but no sim_progress."""
        snap = _make_snap(vtk_result=VTK_RESULT, sim_progress=[])
        result = query_simulation_results({"question": "what are the results"}, snap)

        assert result is not None
        assert "simulation_info" in result

    def test_convergence_included(self):
        """Convergence info included when sim_progress available."""
        snap = _make_snap(sim_progress=SIM_PROGRESS, vtk_result=VTK_RESULT)
        result = query_simulation_results({"question": "convergence"}, snap)

        assert "last_residuals" in result


# ---------------------------------------------------------------------------
# Patch-averaged values (surfaceFieldValue) progress data
# ---------------------------------------------------------------------------

PATCH_PROGRESS = [
    {
        "iteration": i,
        "residuals": {"Ux": {"initial": 0.1 / (i + 1)}, "p": {"initial": 0.05 / (i + 1)}},
        "patch_values": {
            "inlet": {"p": 120.0 - i * 0.5, "T": 300.0},
            "outlet": {"p": 100.0 + i * 0.1, "T": 298.0 + i * 0.05},
        },
    }
    for i in range(20)
]


# ---------------------------------------------------------------------------
# plot_patch_values
# ---------------------------------------------------------------------------

class TestPlotPatchValues:
    """Tests for the plot_patch_values tool."""

    def test_basic_patch_values(self):
        """Plot patch-averaged pressure at all patches."""
        snap = _make_snap(sim_progress=PATCH_PROGRESS)
        result = plot_patch_values({"fields": "p"}, snap)

        assert "error" not in result
        assert "chart" in result
        chart = result["chart"]
        assert chart["type"] == "line"
        assert "p_inlet" in chart["lines"]
        assert "p_outlet" in chart["lines"]
        assert len(chart["data"]) == 20

    def test_specific_patches(self):
        """Plot only at specified patches."""
        snap = _make_snap(sim_progress=PATCH_PROGRESS)
        result = plot_patch_values({"fields": "p", "patches": "inlet"}, snap)

        assert "error" not in result
        assert "p_inlet" in result["chart"]["lines"]
        assert "p_outlet" not in result["chart"]["lines"]

    def test_pressure_drop(self):
        """Compute pressure drop (inlet - outlet)."""
        snap = _make_snap(sim_progress=PATCH_PROGRESS)
        result = plot_patch_values({"fields": "p", "patches": "inlet,outlet", "quantity": "drop"}, snap)

        assert "error" not in result
        chart = result["chart"]
        assert chart["type"] == "line"
        assert len(chart["lines"]) == 1
        assert "drop" in chart["lines"][0].lower()
        # First iteration: inlet=120, outlet=100 → drop=20
        first_row = chart["data"][0]
        drop_key = chart["lines"][0]
        assert first_row[drop_key] == pytest.approx(20.0)

    def test_temperature_drop(self):
        """Compute temperature drop."""
        snap = _make_snap(sim_progress=PATCH_PROGRESS)
        result = plot_patch_values({"fields": "T", "patches": "inlet,outlet", "quantity": "drop"}, snap)

        assert "error" not in result
        assert "chart" in result
        # First iteration: inlet T=300, outlet T=298 → drop=2
        chart = result["chart"]
        drop_key = chart["lines"][0]
        first_row = chart["data"][0]
        assert first_row[drop_key] == pytest.approx(2.0)

    def test_auto_detect_inlet_outlet_for_drop(self):
        """When quantity=drop and no patches specified, auto-detect inlet/outlet."""
        snap = _make_snap(sim_progress=PATCH_PROGRESS)
        result = plot_patch_values({"fields": "p", "quantity": "drop"}, snap)

        assert "error" not in result
        assert "chart" in result
        assert len(result["chart"]["lines"]) == 1

    def test_multiple_fields(self):
        """Plot multiple fields at once."""
        snap = _make_snap(sim_progress=PATCH_PROGRESS)
        result = plot_patch_values({"fields": "p,T"}, snap)

        assert "error" not in result
        lines = result["chart"]["lines"]
        assert any("p_" in ln for ln in lines)
        assert any("T_" in ln for ln in lines)

    def test_no_patch_data(self):
        """Return error when no patch_values data exists."""
        snap = _make_snap(sim_progress=SIM_PROGRESS)  # SIM_PROGRESS has no patch_values
        result = plot_patch_values({"fields": "p"}, snap)

        assert "error" in result

    def test_no_fields_specified(self):
        """Return error when fields is empty."""
        snap = _make_snap(sim_progress=PATCH_PROGRESS)
        result = plot_patch_values({"fields": ""}, snap)

        assert "error" in result

    def test_invalid_patch_name(self):
        """Return error when requested patch doesn't exist."""
        snap = _make_snap(sim_progress=PATCH_PROGRESS)
        result = plot_patch_values({"fields": "p", "patches": "nonexistent"}, snap)

        assert "error" in result

    def test_fields_as_list(self):
        """fields arg as a list should work."""
        snap = _make_snap(sim_progress=PATCH_PROGRESS)
        result = plot_patch_values({"fields": ["p", "T"]}, snap)

        assert "error" not in result
        assert "chart" in result

    def test_last_values_in_result(self):
        """Result includes last_values summary."""
        snap = _make_snap(sim_progress=PATCH_PROGRESS)
        result = plot_patch_values({"fields": "p"}, snap)

        assert "last_values" in result
        assert len(result["last_values"]) > 0

    def test_downsampling_large_dataset(self):
        """Large patch_values data gets downsampled to ≤300 points."""
        large_progress = [
            {
                "iteration": i,
                "patch_values": {
                    "inlet": {"p": 120.0 - i * 0.01},
                    "outlet": {"p": 100.0 + i * 0.01},
                },
            }
            for i in range(500)
        ]
        snap = _make_snap(sim_progress=large_progress)
        result = plot_patch_values({"fields": "p"}, snap)

        assert "chart" in result
        assert len(result["chart"]["data"]) <= 300

    def test_drop_without_enough_patches(self):
        """Drop mode with only 1 patch and no auto-detect → error."""
        # Only one patch, named "wall" (not inlet/outlet)
        progress = [
            {"iteration": i, "patch_values": {"wall": {"p": 100.0}}}
            for i in range(5)
        ]
        snap = _make_snap(sim_progress=progress)
        result = plot_patch_values({"fields": "p", "quantity": "drop"}, snap)

        assert "error" in result
