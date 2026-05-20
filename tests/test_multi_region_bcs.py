# tests/test_multi_region_bcs.py
"""Tests for the BC bridge: per-region 0/<region>/<field> rendering reads
``config["boundary_conditions"]`` and emits per-patch blocks, instead of
the old catch-all ``".*"`` wall default."""

from __future__ import annotations

import pytest

from simd_agent.run.case_spec.strategies import RegionSpec
from simd_agent.solvers.families import _multi_region_bcs as bcs


# Mesh patches for a cyl_cht_2d-style case + reasonable BC specs.
_MESH_PATCHES = [
    {"name": "innerFluid_inlet",     "type": "patch"},
    {"name": "innerFluid_outlet",    "type": "patch"},
    {"name": "innerFluid_symmetry",  "type": "symmetry"},
    {"name": "wall_left_end",        "type": "patch"},
    {"name": "wall_right_end",       "type": "patch"},
    {"name": "outerFluid_inlet",     "type": "patch"},
    {"name": "outerFluid_outlet",    "type": "patch"},
    {"name": "outerFluid_top",       "type": "patch"},
    {"name": "front",                "type": "empty"},
    {"name": "back",                 "type": "empty"},
]


def _cfg(extra_bcs: dict | None = None) -> dict:
    """Build a minimal validated_config for the renderer tests."""
    return {
        "mesh": {"patches": _MESH_PATCHES},
        "boundary_conditions": extra_bcs or {},
    }


# Region specs that mirror what the orchestrator produces after region
# auto-detection + the per-region extractor.
def _inner_fluid_region(**kw) -> RegionSpec:
    return RegionSpec(
        name="innerFluid", kind="fluid",
        thermo_profile="cryogenic",
        interfaces=("wall",),
        fluid_preset="ln2",
        Cp=2042.0, mol_weight=28.01, mu=1.58e-4, Pr=2.3,
        T_init=77.0, p_init=101325.0, U_init=(0.05, 0.0, 0.0),
        turbulence_model="laminar",
        **kw,
    )


def _wall_region(**kw) -> RegionSpec:
    return RegionSpec(
        name="wall", kind="solid",
        thermo_profile="solid",
        interfaces=("innerFluid", "outerFluid"),
        solid_preset="stainless",
        rho_solid=7900.0, kappa_solid=16.2, Cp_solid=500.0,
        T_init=200.0,
        **kw,
    )


def _outer_fluid_region(**kw) -> RegionSpec:
    return RegionSpec(
        name="outerFluid", kind="fluid",
        thermo_profile="gas",
        interfaces=("wall",),
        fluid_preset="water",
        Cp=4182.0, mol_weight=18.02, mu=1.002e-3, Pr=7.0,
        T_init=290.0, p_init=101325.0, U_init=(-0.10, 0.0, 0.0),
        turbulence_model="laminar",
        **kw,
    )


class TestPatchOwnership:
    """``region_patches`` filters mesh patches by region prefix + shared constraints."""

    def test_inner_fluid_owns_only_its_prefix_plus_shared(self):
        owned = bcs.region_patches("innerFluid", _cfg())
        assert owned[:3] == [
            "innerFluid_inlet", "innerFluid_outlet", "innerFluid_symmetry",
        ]
        # front/back are constraint-typed, shared across all regions
        assert "front" in owned and "back" in owned
        # Patches belonging to other regions are NOT included
        for foreign in ("wall_left_end", "outerFluid_inlet", "outerFluid_top"):
            assert foreign not in owned

    def test_wall_solid_includes_no_inlets(self):
        owned = bcs.region_patches("wall", _cfg())
        for inlet in ("innerFluid_inlet", "outerFluid_inlet"):
            assert inlet not in owned
        assert "wall_left_end" in owned
        assert "wall_right_end" in owned


class TestPatchRoleInference:
    """``patch_role`` authority order: BC patchClass → mesh type → suffix."""

    def test_bc_patchclass_wins(self):
        cfg = _cfg({"innerFluid_inlet": {"patchClass": "outlet"}})
        # The patchClass override beats the *_inlet suffix.
        assert bcs.patch_role("innerFluid_inlet", cfg) == "outlet"

    def test_mesh_type_drives_constraint_roles(self):
        # No BC entry — mesh type ``symmetry`` and ``empty`` should be honoured.
        assert bcs.patch_role("innerFluid_symmetry", _cfg()) == "symmetry"
        assert bcs.patch_role("front", _cfg()) == "empty"
        assert bcs.patch_role("back", _cfg()) == "empty"

    def test_name_suffix_heuristic(self):
        assert bcs.patch_role("innerFluid_inlet",  _cfg()) == "inlet"
        assert bcs.patch_role("innerFluid_outlet", _cfg()) == "outlet"
        assert bcs.patch_role("wall_left_end",     _cfg()) == "wall"
        assert bcs.patch_role("wall_right_end",    _cfg()) == "wall"
        assert bcs.patch_role("outerFluid_top",    _cfg()) == "wall"

    def test_unknown_patch_falls_back_to_wall(self):
        assert bcs.patch_role("someOddPatch", _cfg()) == "wall"


class TestUVelocityRendering:
    """``build_region_0_U`` emits per-patch BCs with the right inlet vector."""

    def test_inlet_uses_explicit_bc_vector(self):
        cfg = _cfg({
            "innerFluid_inlet": {
                "patchClass": "inlet",
                "U": {"type": "fixedValue", "value": [0.05, 0.0, 0.0]},
            },
        })
        out = bcs.build_region_0_U(_inner_fluid_region(), cfg)
        assert "innerFluid_inlet" in out
        # fixedValue with the explicit vector — counter-flow signs preserved
        assert "fixedValue" in out
        assert "(0.05 0 0)" in out

    def test_inlet_falls_back_to_region_U_init(self):
        # No BC spec — renderer pulls from RegionSpec.U_init
        out = bcs.build_region_0_U(_outer_fluid_region(), _cfg())
        assert "outerFluid_inlet" in out
        assert "(-0.1 0 0)" in out

    def test_outlet_uses_inlet_outlet(self):
        out = bcs.build_region_0_U(_inner_fluid_region(), _cfg())
        assert "innerFluid_outlet" in out
        # inletOutlet is the right BC for an outflow
        assert "inletOutlet" in out

    def test_walls_get_no_slip(self):
        # outerFluid_top is a wall via the suffix heuristic
        out = bcs.build_region_0_U(_outer_fluid_region(), _cfg())
        # Find the outerFluid_top block and assert it has noSlip
        idx = out.find("outerFluid_top")
        assert idx >= 0
        # Next ~100 chars contain the block body
        assert "noSlip" in out[idx:idx + 200]

    def test_coupled_interface_is_no_slip(self):
        out = bcs.build_region_0_U(_inner_fluid_region(), _cfg())
        # auto-created coupled patch — momentum doesn't cross
        assert "innerFluid_to_wall" in out
        idx = out.find("innerFluid_to_wall")
        assert "noSlip" in out[idx:idx + 200]


class TestTemperatureRendering:
    """``build_region_0_T`` emits the coupled CHT BC at every interface."""

    def test_fluid_inlet_uses_explicit_temperature(self):
        cfg = _cfg({
            "innerFluid_inlet": {
                "patchClass": "inlet",
                "T": {"type": "fixedValue", "value": 77.0},
            },
        })
        out = bcs.build_region_0_T(_inner_fluid_region(), cfg)
        idx = out.find("innerFluid_inlet")
        assert idx >= 0
        assert "fixedValue" in out[idx:idx + 200]
        assert "uniform 77" in out[idx:idx + 200]

    def test_fluid_outlet_uses_inlet_outlet(self):
        out = bcs.build_region_0_T(_inner_fluid_region(), _cfg())
        idx = out.find("innerFluid_outlet")
        assert "inletOutlet" in out[idx:idx + 200]

    def test_coupled_chT_at_every_interface(self):
        out = bcs.build_region_0_T(_inner_fluid_region(), _cfg())
        # The coupled BC must appear for innerFluid_to_wall
        assert "innerFluid_to_wall" in out
        assert (
            "compressible::turbulentTemperatureCoupledBaffleMixed"
            in out
        )
        # Fluid side uses fluidThermo for the conjugate κ
        assert "kappaMethod     fluidThermo" in out

    def test_solid_side_uses_solidThermo(self):
        out = bcs.build_region_0_T(_wall_region(), _cfg())
        # Solid region has TWO coupled interfaces (innerFluid, outerFluid)
        assert "wall_to_innerFluid" in out
        assert "wall_to_outerFluid" in out
        assert "kappaMethod     solidThermo" in out

    def test_solid_non_coupled_patches_default_adiabatic(self):
        out = bcs.build_region_0_T(_wall_region(), _cfg())
        idx = out.find("wall_left_end")
        # Solids without an explicit T fall to zeroGradient (adiabatic)
        assert "zeroGradient" in out[idx:idx + 200]


class TestPressureRendering:
    """``build_region_0_p_rgh`` handles outlets via fixedValue, walls via fixedFluxPressure."""

    def test_outlet_uses_explicit_pressure_value(self):
        cfg = _cfg({
            "innerFluid_outlet": {
                "patchClass": "outlet",
                "p_rgh": {"type": "fixedValue", "value": 101325},
            },
        })
        out = bcs.build_region_0_p_rgh(_inner_fluid_region(), cfg)
        idx = out.find("innerFluid_outlet")
        assert "fixedValue" in out[idx:idx + 200]
        assert "101325" in out[idx:idx + 200]

    def test_walls_use_fixed_flux_pressure(self):
        out = bcs.build_region_0_p_rgh(_inner_fluid_region(), _cfg())
        idx = out.find("innerFluid_to_wall")
        assert "fixedFluxPressure" in out[idx:idx + 200]


class TestRoundTripThroughValidator:
    """End-to-end: the new renderer + universal constraint-patch fixer
    produce a case where every constraint-type patch BC matches its mesh
    type — i.e. the FOAM IO error that crashed the original Regascold run
    cannot recur."""

    def test_symmetry_and_empty_get_fixed(self):
        from simd_agent.solvers import get_registry
        cfg = _cfg()
        # Three regions auto-detected from prefix
        cfg["regions"] = {
            "fluid": [
                {"name": "innerFluid", "fluid_preset": "ln2",
                 "interfaces": ["wall"]},
                {"name": "outerFluid", "fluid_preset": "water",
                 "interfaces": ["wall"]},
            ],
            "solid": [
                {"name": "wall", "solid_preset": "stainless",
                 "interfaces": ["innerFluid", "outerFluid"]},
            ],
        }
        plug = get_registry().get("chtMultiRegionSimpleFoam")

        # Pretend the LLM emitted only the controlDict — validate_full
        # then merges in the deterministic tree and runs the universal
        # constraint-patch BC fixer on top.
        files = {
            "system/controlDict":
            "application chtMultiRegionSimpleFoam;\n"
            "startTime 0; endTime 100; deltaT 1; writeInterval 10;\n",
        }
        result = plug.validate_full(files, cfg)
        out = result.files

        # 0/innerFluid/U: innerFluid_symmetry must end up with type symmetry,
        # NOT the noSlip that the renderer initially writes.
        u_inner = out["0/innerFluid/U"]
        sym_block = u_inner[u_inner.find("innerFluid_symmetry"):]
        sym_block = sym_block[: sym_block.find("}") + 1]
        assert "type            symmetry;" in sym_block

        # front/back must be empty in every field file (universal fix)
        for path in (
            "0/innerFluid/U", "0/innerFluid/T", "0/wall/T",
            "0/outerFluid/U", "0/outerFluid/T",
        ):
            content = out[path]
            for cp in ("front", "back"):
                idx = content.find(cp)
                if idx < 0:
                    continue
                block = content[idx: idx + 200]
                assert "type            empty;" in block, (
                    f"{path}:{cp} did not get type=empty"
                )
