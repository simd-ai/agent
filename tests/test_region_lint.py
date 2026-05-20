# tests/test_region_lint.py
"""Tests for the multi-region consistency lint (A4)."""

from __future__ import annotations

import pytest

from simd_agent.run.region_lint import lint_regions


def _healthy_cht_config() -> dict:
    """A well-formed CHT config matching the cyl_cht_2d test mesh."""
    return {
        "mesh": {
            "cell_zones": ["innerFluid", "wall", "outerFluid"],
            "patches": [
                {"name": "innerFluid_inlet",     "type": "patch"},
                {"name": "innerFluid_outlet",    "type": "patch"},
                {"name": "innerFluid_symmetry",  "type": "symmetry"},
                {"name": "wall_left_end",        "type": "patch"},
                {"name": "wall_right_end",       "type": "patch"},
                {"name": "outerFluid_inlet",     "type": "patch"},
                {"name": "outerFluid_outlet",    "type": "patch"},
                {"name": "outerFluid_top",       "type": "patch"},
            ],
        },
        "boundary_conditions": {
            "innerFluid_inlet":  {"patchClass": "inlet"},
            "outerFluid_inlet":  {"patchClass": "inlet"},
        },
        "regions": {
            "fluid": [
                {"name": "innerFluid", "interfaces": ["wall"]},
                {"name": "outerFluid", "interfaces": ["wall"]},
            ],
            "solid": [
                {"name": "wall", "interfaces": ["innerFluid", "outerFluid"]},
            ],
        },
    }


def _codes(issues):
    return [i["code"] for i in issues]


class TestNoOpPaths:
    def test_no_regions_returns_empty(self):
        # Single-region cases never reach this code; should be a clean no-op.
        assert lint_regions({}) == []
        assert lint_regions({"regions": {}}) == []
        assert lint_regions({"regions": {"fluid": [], "solid": []}}) == []

    def test_healthy_cht_has_no_issues(self):
        assert lint_regions(_healthy_cht_config()) == []


class TestCellzoneMismatch:
    def test_zone_without_region_is_error(self):
        cfg = _healthy_cht_config()
        cfg["mesh"]["cell_zones"].append("ghost")
        codes = _codes(lint_regions(cfg))
        assert "cellzone_without_region" in codes

    def test_region_without_zone_is_warning(self):
        cfg = _healthy_cht_config()
        cfg["regions"]["fluid"].append({
            "name": "phantom", "interfaces": ["wall"],
        })
        # phantom also needs to be on wall's interfaces or it triggers
        # other warnings — keep one-sided to isolate the cellZone check
        codes = _codes(lint_regions(cfg))
        assert "region_without_cellzone" in codes

    def test_cellzones_empty_skips_check(self):
        # When the importer didn't extract cellZones, the check is skipped —
        # we don't want false alarms on older meshes / formats.
        cfg = _healthy_cht_config()
        cfg["mesh"]["cell_zones"] = []
        assert "cellzone_without_region" not in _codes(lint_regions(cfg))
        assert "region_without_cellzone" not in _codes(lint_regions(cfg))


class TestRegionPatchOwnership:
    def test_region_with_zero_owned_patches_warns(self):
        cfg = _healthy_cht_config()
        cfg["regions"]["fluid"].append({
            "name": "lonelyFluid",
            "interfaces": ["wall"],
        })
        # Add cellZone so the cellzone-mismatch check is happy
        cfg["mesh"]["cell_zones"].append("lonelyFluid")
        # And the wall must back-reference it
        cfg["regions"]["solid"][0]["interfaces"].append("lonelyFluid")
        codes = _codes(lint_regions(cfg))
        assert "region_without_patches" in codes


class TestFluidInlet:
    def test_fluid_with_no_inlet_warns(self):
        cfg = _healthy_cht_config()
        # Drop the inner-fluid inlet patch
        cfg["mesh"]["patches"] = [
            p for p in cfg["mesh"]["patches"]
            if p["name"] != "innerFluid_inlet"
        ]
        cfg["boundary_conditions"].pop("innerFluid_inlet", None)
        codes = _codes(lint_regions(cfg))
        assert "fluid_region_no_inlet" in codes


class TestSolidInterfaces:
    def test_solid_with_no_fluid_interface_errors(self):
        cfg = _healthy_cht_config()
        cfg["regions"]["solid"][0]["interfaces"] = []
        codes = _codes(lint_regions(cfg))
        assert "solid_region_no_fluid_interface" in codes

    def test_solid_with_only_phantom_interface_errors(self):
        cfg = _healthy_cht_config()
        cfg["regions"]["solid"][0]["interfaces"] = ["ghost"]
        issues = lint_regions(cfg)
        codes = _codes(issues)
        assert "solid_region_no_fluid_interface" in codes
        # And the dangling-interface check should fire too
        assert "interface_dangling" in codes


class TestInterfaceReciprocity:
    def test_one_sided_interface_warns(self):
        cfg = _healthy_cht_config()
        # Remove innerFluid from the wall's interfaces — wall still has outerFluid
        # so it's not orphaned, but the innerFluid ⇆ wall link is one-sided.
        cfg["regions"]["solid"][0]["interfaces"] = ["outerFluid"]
        codes = _codes(lint_regions(cfg))
        assert "interface_one_sided" in codes
        # Solid still has at least one fluid neighbour
        assert "solid_region_no_fluid_interface" not in codes

    def test_dangling_interface_errors(self):
        cfg = _healthy_cht_config()
        cfg["regions"]["fluid"][0]["interfaces"].append("typoName")
        codes = _codes(lint_regions(cfg))
        assert "interface_dangling" in codes


class TestIssueShape:
    def test_issues_carry_severity_code_message(self):
        cfg = _healthy_cht_config()
        cfg["regions"]["solid"][0]["interfaces"] = []
        issues = lint_regions(cfg)
        for i in issues:
            assert "severity" in i
            assert i["severity"] in ("error", "warning", "info")
            assert "code" in i and isinstance(i["code"], str)
            assert "message" in i and isinstance(i["message"], str)
