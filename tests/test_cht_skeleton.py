# tests/test_cht_skeleton.py
"""Tests for the chtMultiRegion{Simple,}Foam Phase 1 skeleton.

These tests lock the architectural contract — RegionSpec invariants,
MultiRegionBase identity, regionProperties + per-region thermo
rendering — so the Phase 2 implementation (per-region fvSchemes /
fvSolution / mapped BCs / orchestrator integration) can land without
silently breaking the foundation.

Scope explicitly NOT covered here (it's the Phase 2 work):
  * Per-region fvSchemes / fvSolution rendering
  * Mapped fluid-solid coupled-BC generation
  * changeDictionaryDict per region
  * Orchestrator handling of tree-structured file manifests
  * packaging.py multi-region zip layout
"""

from __future__ import annotations

import pytest

from simd_agent.run.case_spec import CaseRegions, RegionSpec
from simd_agent.solvers.families import MultiRegionBase, SteadyBase, TransientBase
from simd_agent.solvers.heatTransfer.chtMultiRegionFoam.solver import (
    ChtMultiRegionFoamSolver,
)
from simd_agent.solvers.heatTransfer.chtMultiRegionSimpleFoam.solver import (
    ChtMultiRegionSimpleFoamSolver,
)


# ── RegionSpec invariants ───────────────────────────────────────────────────


class TestRegionSpec:
    def test_valid_fluid_region(self):
        r = RegionSpec(name="topAir", kind="fluid", thermo_profile="gas")
        assert r.name == "topAir"
        assert r.kind == "fluid"
        # Defaults — air-like fluid.
        assert r.Cp == 1006.0
        assert r.mu == 1.8e-5
        assert r.Pr == 0.7

    def test_valid_solid_region(self):
        r = RegionSpec(
            name="heater", kind="solid", thermo_profile="solid",
            rho_solid=8000.0, kappa_solid=80.0, Cp_solid=450.0,
        )
        assert r.kind == "solid"
        assert r.turbulence_model is None

    def test_solid_cannot_have_turbulence_model(self):
        with pytest.raises(ValueError, match="cannot declare a turbulence"):
            RegionSpec(
                name="heater",
                kind="solid",
                thermo_profile="solid",
                turbulence_model="kEpsilon",
            )

    def test_solid_requires_solid_thermo_profile(self):
        with pytest.raises(ValueError, match="thermo_profile='solid'"):
            RegionSpec(name="heater", kind="solid", thermo_profile="gas")

    def test_fluid_cannot_use_solid_thermo_profile(self):
        with pytest.raises(ValueError, match="cannot use"):
            RegionSpec(name="topAir", kind="fluid", thermo_profile="solid")

    def test_region_name_must_be_valid_identifier(self):
        with pytest.raises(ValueError):
            RegionSpec(name="123bad", kind="fluid", thermo_profile="gas")
        with pytest.raises(ValueError):
            RegionSpec(name="has space", kind="fluid", thermo_profile="gas")


# ── CaseRegions invariants ──────────────────────────────────────────────────


class TestCaseRegions:
    def test_valid_two_region_case(self):
        fluid = RegionSpec(name="topAir", kind="fluid", thermo_profile="gas")
        solid = RegionSpec(name="heater", kind="solid", thermo_profile="solid")
        regions = CaseRegions(fluid_regions=[fluid], solid_regions=[solid])
        assert regions.region_names == ["topAir", "heater"]
        # Fluids come first in all_regions.
        assert regions.all_regions[0].kind == "fluid"

    def test_no_fluid_regions_is_invalid(self):
        solid = RegionSpec(name="heater", kind="solid", thermo_profile="solid")
        with pytest.raises(ValueError, match="at least one fluid"):
            CaseRegions(fluid_regions=[], solid_regions=[solid])

    def test_duplicate_region_names_rejected(self):
        fluid = RegionSpec(name="reg", kind="fluid", thermo_profile="gas")
        solid = RegionSpec(name="reg", kind="solid", thermo_profile="solid")
        with pytest.raises(ValueError, match="Duplicate region name"):
            CaseRegions(fluid_regions=[fluid], solid_regions=[solid])


# ── MultiRegionBase identity ────────────────────────────────────────────────


class TestMultiRegionBase:
    def test_simple_variant_mro(self):
        cls = ChtMultiRegionSimpleFoamSolver
        mro = [c.__name__ for c in cls.__mro__]
        # Mixin must come before SteadyBase → SolverPlugin so its
        # overrides win on class-attribute lookup.
        assert mro.index("MultiRegionBase") < mro.index("SolverPlugin")
        assert issubclass(cls, MultiRegionBase)
        assert issubclass(cls, SteadyBase)
        assert not issubclass(cls, TransientBase)

    def test_pimple_variant_mro(self):
        cls = ChtMultiRegionFoamSolver
        mro = [c.__name__ for c in cls.__mro__]
        assert mro.index("MultiRegionBase") < mro.index("SolverPlugin")
        assert issubclass(cls, MultiRegionBase)
        assert issubclass(cls, TransientBase)
        assert not issubclass(cls, SteadyBase)

    def test_identity_attributes(self):
        for cls in (ChtMultiRegionSimpleFoamSolver, ChtMultiRegionFoamSolver):
            p = cls()
            assert p.is_multi_region is True
            assert p.is_compressible is True
            assert p.supports_energy is True
            assert p.needs_gravity is True
            assert p.pressure_field == "p_rgh"
            assert p.energy_var == "h"


# ── Required-files manifest ─────────────────────────────────────────────────


_CFG_TWO_REGIONS = {
    "physics": {
        "compressibility": "compressible",
        "heat_transfer": True,
        "gravity": True,
    },
    "regions": {
        "fluid": [
            {"name": "topAir", "thermo_profile": "gas",
             "turbulence_model": "kEpsilon"},
        ],
        "solid": [
            {"name": "heater", "rho_solid": 8000.0,
             "kappa_solid": 80.0, "Cp_solid": 450.0},
        ],
    },
}


class TestRequiredFilesTree:
    """``required_files()`` is the LLM-targeted manifest (Phase 3: only the
    files the LLM is asked to generate — ``system/controlDict``).  The full
    case tree is asserted against :meth:`all_case_files` instead, which
    catalogues every file the case ships with (LLM + deterministic).
    """

    def test_required_files_is_llm_only(self):
        plugin = ChtMultiRegionSimpleFoamSolver()
        files = plugin.required_files(_CFG_TWO_REGIONS)
        # The LLM should generate exactly the case-level controlDict and
        # nothing under constant/<region>/ or 0/<region>/.
        assert files == ["system/controlDict"]

    def test_all_case_files_includes_top_level(self):
        plugin = ChtMultiRegionSimpleFoamSolver()
        files = plugin.all_case_files(_CFG_TWO_REGIONS)
        assert "system/controlDict" in files
        assert "constant/regionProperties" in files

    def test_all_case_files_includes_per_region_thermo(self):
        plugin = ChtMultiRegionSimpleFoamSolver()
        files = plugin.all_case_files(_CFG_TWO_REGIONS)
        assert "constant/topAir/thermophysicalProperties" in files
        assert "constant/heater/thermophysicalProperties" in files

    def test_fluid_region_has_turbulence_and_g(self):
        plugin = ChtMultiRegionSimpleFoamSolver()
        files = plugin.all_case_files(_CFG_TWO_REGIONS)
        assert "constant/topAir/turbulenceProperties" in files
        assert "constant/topAir/g" in files

    def test_solid_region_has_no_turbulence_or_g(self):
        plugin = ChtMultiRegionSimpleFoamSolver()
        files = plugin.all_case_files(_CFG_TWO_REGIONS)
        assert "constant/heater/turbulenceProperties" not in files
        assert "constant/heater/g" not in files

    def test_solid_region_only_solves_T(self):
        """Solid regions have 0/<region>/T only — no U / p / k / ε."""
        plugin = ChtMultiRegionSimpleFoamSolver()
        files = plugin.all_case_files(_CFG_TWO_REGIONS)
        assert "0/heater/T" in files
        assert "0/heater/U" not in files
        assert "0/heater/p" not in files

    def test_fluid_region_solves_full_NS_plus_energy_plus_turb(self):
        plugin = ChtMultiRegionSimpleFoamSolver()
        files = plugin.all_case_files(_CFG_TWO_REGIONS)
        for field in ("T", "U", "p", "p_rgh", "k", "epsilon"):
            assert f"0/topAir/{field}" in files, f"missing 0/topAir/{field}"


# ── regionProperties + per-region thermo content ───────────────────────────


class TestRenderedFiles:
    def test_regionProperties_lists_fluids_and_solids(self):
        plugin = ChtMultiRegionSimpleFoamSolver()
        det = plugin.render_deterministic_files(_CFG_TWO_REGIONS)
        rp = det["constant/regionProperties"]
        assert "fluid       (topAir)" in rp
        assert "solid       (heater)" in rp

    def test_fluid_thermo_uses_heRhoThermo(self):
        plugin = ChtMultiRegionSimpleFoamSolver()
        det = plugin.render_deterministic_files(_CFG_TWO_REGIONS)
        thermo = det["constant/topAir/thermophysicalProperties"]
        assert "heRhoThermo" in thermo
        assert "perfectGas" in thermo
        assert "sensibleEnthalpy" in thermo
        # No solid keys.
        assert "heSolidThermo" not in thermo
        assert "rhoConst" not in thermo

    def test_solid_thermo_uses_heSolidThermo(self):
        plugin = ChtMultiRegionSimpleFoamSolver()
        det = plugin.render_deterministic_files(_CFG_TWO_REGIONS)
        thermo = det["constant/heater/thermophysicalProperties"]
        assert "heSolidThermo" in thermo
        assert "rhoConst" in thermo
        assert "constIso" in thermo
        # Solid-region thermo values from config.
        assert "kappa       80" in thermo
        assert "Cp          450" in thermo
        assert "rho         8000" in thermo

    def test_default_region_when_config_missing(self):
        """No regions config → one fluid + zero solids (degenerate but valid)."""
        plugin = ChtMultiRegionSimpleFoamSolver()
        # CaseRegions invariant rejects no-fluid; extract_regions provides
        # a default single fluid so the contract holds.
        det = plugin.render_deterministic_files({})
        assert "constant/regionProperties" in det
        assert "fluid       (fluid)" in det["constant/regionProperties"]


# ── Matching ────────────────────────────────────────────────────────────────


class TestMatching:
    def test_simple_variant_scores_steady_multiregion_high(self):
        plugin = ChtMultiRegionSimpleFoamSolver()
        m = plugin.matches(_CFG_TWO_REGIONS)
        assert m.score >= 0.9

    def test_simple_variant_rejects_single_region(self):
        """Single-region heat case → buoyantSimpleFoam, not CHT."""
        plugin = ChtMultiRegionSimpleFoamSolver()
        m = plugin.matches({
            "physics": {"heat_transfer": True, "gravity": True},
            # No regions block at all.
        })
        assert m.score == 0.0

    def test_pimple_variant_rejects_steady(self):
        plugin = ChtMultiRegionFoamSolver()
        m = plugin.matches({
            **_CFG_TWO_REGIONS,
            "physics": {**_CFG_TWO_REGIONS["physics"], "time_scheme": "steady"},
        })
        assert m.score < 0.5

    def test_pimple_variant_scores_transient_multiregion_high(self):
        plugin = ChtMultiRegionFoamSolver()
        m = plugin.matches({
            **_CFG_TWO_REGIONS,
            "physics": {**_CFG_TWO_REGIONS["physics"], "time_scheme": "transient"},
        })
        assert m.score >= 0.9


# ── Registry auto-discovery ─────────────────────────────────────────────────


class TestRegistry:
    def test_both_cht_solvers_registered(self):
        from simd_agent.solvers import get_registry
        names = get_registry().names()
        assert "chtMultiRegionSimpleFoam" in names
        assert "chtMultiRegionFoam" in names

    def test_cht_solvers_classified_as_p_rgh_and_gravity_and_energy(self):
        from simd_agent.solvers import get_registry
        registry = get_registry()
        for name in ("chtMultiRegionSimpleFoam", "chtMultiRegionFoam"):
            assert name in registry.p_rgh_solvers(), name
            assert name in registry.gravity_solvers(), name
            assert name in registry.energy_solvers(), name
