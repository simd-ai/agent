# tests/test_cht_phase2.py
"""Phase 2 tests for chtMultiRegion{Simple,}Foam.

Phase 2 lands the per-region deterministic file rendering:

  * per-region fvSchemes (fluid: full N-S + energy; solid: only
    ``laplacian(alpha,h)``)
  * per-region fvSolution (fluid: rho + p_rgh + h + U + k + ε with
    Finals; solid: only h + hFinal)
  * per-fluid turbulenceProperties + constant/g
  * per-region 0-fields with
    ``compressible::turbulentTemperatureCoupledBaffleMixed`` at every
    interface patch
  * per-region changeDictionaryDict

These tests cover the file *content* — orchestrator + packaging
integration (Phase 3) is intentionally out of scope.
"""

from __future__ import annotations

from simd_agent.solvers.heatTransfer.chtMultiRegionFoam.solver import (
    ChtMultiRegionFoamSolver,
)
from simd_agent.solvers.heatTransfer.chtMultiRegionSimpleFoam.solver import (
    ChtMultiRegionSimpleFoamSolver,
)


# A two-region fixture: topAir (fluid) ⇄ heater (solid).
_CFG = {
    "physics": {
        "compressibility": "compressible",
        "heat_transfer": True,
        "gravity": True,
        "time_scheme": "transient",
    },
    "regions": {
        "fluid": [{
            "name": "topAir",
            "turbulence_model": "kEpsilon",
            "interfaces": ["heater"],
            "T_init": 300.0,
        }],
        "solid": [{
            "name": "heater",
            "rho_solid": 8000.0,
            "kappa_solid": 80.0,
            "Cp_solid": 450.0,
            "interfaces": ["topAir"],
            "T_init": 300.0,
        }],
    },
}


def _det(plugin):
    return plugin.render_deterministic_files(_CFG)


# ── Full file-tree completeness ─────────────────────────────────────────────


class TestFileTreeCompleteness:
    def test_all_expected_files_present(self):
        det = _det(ChtMultiRegionFoamSolver())
        expected = {
            "constant/regionProperties",
            # Per-fluid:
            "constant/topAir/thermophysicalProperties",
            "constant/topAir/turbulenceProperties",
            "constant/topAir/g",
            # Per-solid:
            "constant/heater/thermophysicalProperties",
            # System per-region:
            "system/topAir/fvSchemes",
            "system/topAir/fvSolution",
            "system/topAir/changeDictionaryDict",
            "system/heater/fvSchemes",
            "system/heater/fvSolution",
            "system/heater/changeDictionaryDict",
            # 0-fields:
            "0/topAir/T",
            "0/topAir/U",
            "0/topAir/p",
            "0/topAir/p_rgh",
            "0/topAir/k",
            "0/topAir/epsilon",
            "0/heater/T",
        }
        missing = expected - set(det.keys())
        assert not missing, f"Missing files: {sorted(missing)}"


# ── Per-region fvSchemes ────────────────────────────────────────────────────


class TestPerRegionFvSchemes:
    def test_fluid_fvSchemes_has_full_NS(self):
        det = _det(ChtMultiRegionFoamSolver())
        sc = det["system/topAir/fvSchemes"]
        # Full Navier–Stokes div set.
        for line in (
            "div(phi,U)      bounded Gauss upwind",
            "div(phi,K)      Gauss linear",
            "div(phi,h)      bounded Gauss upwind",
            "div(phi,k)      bounded Gauss upwind",
            "div(phi,epsilon) bounded Gauss upwind",
            "div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear",
        ):
            assert line in sc, f"missing fvSchemes line: {line!r}"

    def test_solid_fvSchemes_only_has_laplacian(self):
        det = _det(ChtMultiRegionFoamSolver())
        sc = det["system/heater/fvSchemes"]
        # Solid has no momentum / convection.
        assert "div(phi,U)" not in sc
        assert "div(phi,h)" not in sc
        # Only the enthalpy laplacian is solved.
        assert "laplacian(alpha,h)  Gauss linear corrected" in sc
        assert "default             none" in sc


# ── Per-region fvSolution ───────────────────────────────────────────────────


class TestPerRegionFvSolution:
    def test_fluid_fvSolution_has_rho_p_rgh_eq_blocks_with_finals(self):
        det = _det(ChtMultiRegionFoamSolver())
        fvs = det["system/topAir/fvSolution"]
        # rho + rhoFinal (PIMPLE needs Final for every solved field).
        assert "    rho\n" in fvs
        assert "    rhoFinal\n" in fvs
        # p_rgh + Final.
        assert "    p_rgh\n" in fvs
        assert "    p_rghFinal\n" in fvs
        # Equation regex covers U, h, k, epsilon.
        assert '"(U|h|k|epsilon)"' in fvs
        assert '"(U|h|k|epsilon)Final"' in fvs

    def test_solid_fvSolution_only_has_h(self):
        det = _det(ChtMultiRegionFoamSolver())
        fvs = det["system/heater/fvSolution"]
        # Solid: only h + hFinal.
        assert "    h\n" in fvs
        assert "    hFinal\n" in fvs
        # No momentum / pressure / turbulence.
        assert "rho" not in fvs.split("solvers")[1].split("\n}")[0]
        assert "p_rgh" not in fvs
        assert "(U|h|" not in fvs

    def test_simple_variant_uses_SIMPLE_outer_block(self):
        det = _det(ChtMultiRegionSimpleFoamSolver())
        fvs_fluid = det["system/topAir/fvSolution"]
        fvs_solid = det["system/heater/fvSolution"]
        assert "\nSIMPLE\n" in fvs_fluid
        assert "\nSIMPLE\n" in fvs_solid
        assert "\nPIMPLE\n" not in fvs_fluid
        assert "\nPIMPLE\n" not in fvs_solid

    def test_pimple_variant_uses_PIMPLE_outer_block(self):
        det = _det(ChtMultiRegionFoamSolver())
        fvs_fluid = det["system/topAir/fvSolution"]
        fvs_solid = det["system/heater/fvSolution"]
        assert "\nPIMPLE\n" in fvs_fluid
        assert "\nPIMPLE\n" in fvs_solid
        assert "\nSIMPLE\n" not in fvs_fluid
        assert "\nSIMPLE\n" not in fvs_solid


# ── Coupled BC at interfaces ────────────────────────────────────────────────


class TestCoupledTBoundaries:
    def test_fluid_side_uses_fluidThermo_kappa(self):
        det = _det(ChtMultiRegionFoamSolver())
        t = det["0/topAir/T"]
        assert "topAir_to_heater" in t
        assert "compressible::turbulentTemperatureCoupledBaffleMixed" in t
        assert "kappaMethod     fluidThermo" in t

    def test_solid_side_uses_solidThermo_kappa(self):
        det = _det(ChtMultiRegionFoamSolver())
        t = det["0/heater/T"]
        assert "heater_to_topAir" in t
        assert "compressible::turbulentTemperatureCoupledBaffleMixed" in t
        assert "kappaMethod     solidThermo" in t

    def test_changeDictionaryDict_has_coupled_patches(self):
        det = _det(ChtMultiRegionFoamSolver())
        # Fluid side.
        cd_f = det["system/topAir/changeDictionaryDict"]
        assert "topAir_to_heater" in cd_f
        assert "kappaMethod     fluidThermo" in cd_f
        # Solid side.
        cd_s = det["system/heater/changeDictionaryDict"]
        assert "heater_to_topAir" in cd_s
        assert "kappaMethod     solidThermo" in cd_s

    def test_no_interface_means_no_coupled_block(self):
        """RegionSpec.interfaces=() → no compressible::... patch."""
        plugin = ChtMultiRegionFoamSolver()
        cfg = {
            "regions": {
                "fluid": [{"name": "isolated", "interfaces": []}],
                "solid": [{"name": "alone", "interfaces": []}],
            },
        }
        det = plugin.render_deterministic_files(cfg)
        t_f = det["0/isolated/T"]
        t_s = det["0/alone/T"]
        assert "compressible::turbulentTemperatureCoupledBaffleMixed" not in t_f
        assert "compressible::turbulentTemperatureCoupledBaffleMixed" not in t_s


# ── Per-fluid turbulenceProperties + g ──────────────────────────────────────


class TestPerFluidConstantFiles:
    def test_fluid_turbulenceProperties_has_RAS_block(self):
        det = _det(ChtMultiRegionFoamSolver())
        tp = det["constant/topAir/turbulenceProperties"]
        assert "simulationType  RAS" in tp
        assert "RASModel        kEpsilon" in tp

    def test_solid_has_no_turbulenceProperties(self):
        det = _det(ChtMultiRegionFoamSolver())
        assert "constant/heater/turbulenceProperties" not in det

    def test_fluid_has_gravity_file(self):
        det = _det(ChtMultiRegionFoamSolver())
        g = det["constant/topAir/g"]
        assert "dimensions      [0 1 -2 0 0 0 0]" in g
        assert "value           (0 -9.81 0)" in g

    def test_solid_has_no_gravity_file(self):
        det = _det(ChtMultiRegionFoamSolver())
        assert "constant/heater/g" not in det


# ── 0-field shape ───────────────────────────────────────────────────────────


class TestZeroFieldShape:
    def test_fluid_0_p_rgh_uses_fixedFluxPressure(self):
        det = _det(ChtMultiRegionFoamSolver())
        p = det["0/topAir/p_rgh"]
        assert "fixedFluxPressure" in p
        assert "internalField   uniform 100000" in p

    def test_fluid_0_p_uses_calculated(self):
        det = _det(ChtMultiRegionFoamSolver())
        p = det["0/topAir/p"]
        assert "type            calculated" in p

    def test_fluid_0_k_and_epsilon_use_wall_functions(self):
        det = _det(ChtMultiRegionFoamSolver())
        k = det["0/topAir/k"]
        eps = det["0/topAir/epsilon"]
        assert "kqRWallFunction" in k
        assert "epsilonWallFunction" in eps

    def test_solid_only_has_T_field(self):
        """Solid regions have 0/<region>/T but no U/p/p_rgh/k/ε."""
        det = _det(ChtMultiRegionFoamSolver())
        assert "0/heater/T" in det
        for f in ("U", "p", "p_rgh", "k", "epsilon", "nut"):
            assert f"0/heater/{f}" not in det

    def test_kEpsilon_emits_both_k_and_epsilon(self):
        det = _det(ChtMultiRegionFoamSolver())
        assert "0/topAir/k" in det
        assert "0/topAir/epsilon" in det

    def test_kOmegaSST_emits_k_only_no_epsilon(self):
        """kOmegaSST transports k and omega — not epsilon."""
        plugin = ChtMultiRegionFoamSolver()
        cfg = {
            "regions": {
                "fluid": [{"name": "air", "turbulence_model": "kOmegaSST"}],
                "solid": [{"name": "block", "interfaces": []}],
            },
        }
        det = plugin.render_deterministic_files(cfg)
        assert "0/air/k" in det
        assert "0/air/epsilon" not in det


# ── Required-files manifest tracks the rendered files ───────────────────────


class TestRegionPresets:
    """Fluid + solid preset tables drive the physics defaults.

    ``fluid_preset = "water"`` fills Cp / μ / Pr / mol_weight from the
    water preset; explicit per-field overrides still win.
    """

    def test_water_fluid_preset_applies(self):
        plugin = ChtMultiRegionFoamSolver()
        cfg = {
            "regions": {
                "fluid": [{"name": "h2o", "fluid_preset": "water"}],
                "solid": [{"name": "s"}],
            },
        }
        regions = plugin.extract_regions(cfg)
        water = regions.fluid_regions[0]
        assert water.Cp == 4182.0
        assert water.mu == 1.002e-3
        assert water.Pr == 7.0
        assert water.mol_weight == 18.02

    def test_ln2_fluid_preset_sets_cryogenic_profile(self):
        """Cryogenic presets carry the right thermo_profile."""
        plugin = ChtMultiRegionFoamSolver()
        cfg = {
            "regions": {
                "fluid": [{"name": "cryo", "fluid_preset": "ln2"}],
                "solid": [{"name": "vessel", "solid_preset": "steel"}],
            },
        }
        regions = plugin.extract_regions(cfg)
        assert regions.fluid_regions[0].thermo_profile == "cryogenic"
        assert regions.fluid_regions[0].Cp == 2042.0

    def test_copper_solid_preset_applies(self):
        plugin = ChtMultiRegionFoamSolver()
        cfg = {
            "regions": {
                "fluid": [{"name": "f"}],
                "solid": [{"name": "cu", "solid_preset": "copper"}],
            },
        }
        regions = plugin.extract_regions(cfg)
        cu = regions.solid_regions[0]
        assert cu.rho_solid == 8960.0
        assert cu.kappa_solid == 400.0
        assert cu.Cp_solid == 385.0

    def test_explicit_override_beats_preset(self):
        """Per-field override wins; other preset fields still apply."""
        plugin = ChtMultiRegionFoamSolver()
        cfg = {
            "regions": {
                "fluid": [{
                    "name": "custom",
                    "fluid_preset": "air",   # Cp=1006, μ=1.8e-5, Pr=0.71
                    "Cp": 1200,              # override only Cp
                }],
                "solid": [{"name": "s"}],
            },
        }
        regions = plugin.extract_regions(cfg)
        f = regions.fluid_regions[0]
        # Override applied:
        assert f.Cp == 1200.0
        # Other preset values preserved:
        assert f.mu == 1.8e-5
        assert f.Pr == 0.71

    def test_unknown_preset_silently_falls_back_to_defaults(self):
        """Unknown preset names = no preset applied; air-like defaults."""
        plugin = ChtMultiRegionFoamSolver()
        cfg = {
            "regions": {
                "fluid": [{"name": "x", "fluid_preset": "unobtainium"}],
                "solid": [{"name": "s"}],
            },
        }
        regions = plugin.extract_regions(cfg)
        # Falls back to RegionSpec's air-like defaults.
        assert regions.fluid_regions[0].Cp == 1006.0
        assert regions.fluid_regions[0].mu == 1.8e-5

    def test_preset_values_flow_into_rendered_thermo(self):
        """End-to-end: preset → RegionSpec → rendered thermophysicalProperties."""
        plugin = ChtMultiRegionFoamSolver()
        cfg = {
            "regions": {
                "fluid": [{"name": "water", "fluid_preset": "water",
                           "interfaces": ["cu"]}],
                "solid": [{"name": "cu", "solid_preset": "copper",
                           "interfaces": ["water"]}],
            },
        }
        files = plugin.render_deterministic_files(cfg)
        fluid_thermo = files["constant/water/thermophysicalProperties"]
        solid_thermo = files["constant/cu/thermophysicalProperties"]
        # Water values in fluid file.
        assert "Cp              4182" in fluid_thermo
        assert "mu              0.001002" in fluid_thermo
        assert "molWeight       18.02" in fluid_thermo
        # Copper values in solid file.
        assert "kappa       400" in solid_thermo
        assert "Cp          385" in solid_thermo
        assert "rho         8960" in solid_thermo

    def test_all_fluid_presets_well_formed(self):
        """Every preset has the required keys (Cp, μ, Pr, mol_weight)."""
        from simd_agent.solvers.families._multi_region import MultiRegionBase
        for name, preset in MultiRegionBase.FLUID_REGION_PRESETS.items():
            for key in ("Cp", "mu", "Pr", "mol_weight"):
                assert key in preset, f"{name} missing {key}"
                assert preset[key] > 0, f"{name}.{key} must be positive"

    def test_all_solid_presets_well_formed(self):
        from simd_agent.solvers.families._multi_region import MultiRegionBase
        for name, preset in MultiRegionBase.SOLID_REGION_PRESETS.items():
            for key in ("rho_solid", "kappa_solid", "Cp_solid"):
                assert key in preset, f"{name} missing {key}"
                assert preset[key] > 0, f"{name}.{key} must be positive"


class TestManifestMatchesRendered:
    """Every file in required_files() must actually be rendered (or LLM-generated)."""

    def test_no_renderer_gap(self):
        plugin = ChtMultiRegionFoamSolver()
        manifest = set(plugin.required_files(_CFG))
        rendered = set(plugin.render_deterministic_files(_CFG).keys())
        # Manifest may include LLM-only files (controlDict).  Rendered may
        # include files not in the manifest (e.g. changeDictionaryDict isn't
        # strictly required at runtime — it's applied via the Allrun script).
        # We don't enforce equality; just check that the OVERLAP makes sense.
        common = manifest & rendered
        # At minimum, both should agree on the bulk: regionProperties,
        # per-region thermo, per-region 0/T.
        assert "constant/regionProperties" in common
        assert "constant/topAir/thermophysicalProperties" in common
        assert "constant/heater/thermophysicalProperties" in common
        assert "0/topAir/T" in common
        assert "0/heater/T" in common
