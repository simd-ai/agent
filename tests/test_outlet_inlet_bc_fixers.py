# tests/test_outlet_inlet_bc_fixers.py
"""Tests for the outlet/inlet boundary-condition fixers (#7 + #8).

Both fixers track the OpenFOAM rhoSimpleFoam reference tutorial
(``compressible/rhoSimpleFoam/angledDuctExplicitFixedCoeff``).

  #7 — Outlet U / T / k / ω / ε must use ``inletOutlet``, not
       ``zeroGradient`` — so a momentarily-reversed outflow falls
       back to ``inletValue`` instead of pulling garbage upstream.

  #8 — Inlet k / ω / ε must derive from the *actual* inlet velocity
       at runtime (``turbulentIntensityKineticEnergyInlet``,
       ``turbulentMixingLengthFrequencyInlet``, …) — not from a
       precheck-precomputed ``fixedValue`` number that drifts from
       the real mass-flow-derived U.
"""

from __future__ import annotations

from simd_agent.solvers.base import ValidationIssue
from simd_agent.solvers.compressible.rhoSimpleFoam.solver import RhoSimpleFoamSolver


# ── _rewrite_patch_body helper ──────────────────────────────────────────────


class TestRewritePatchBody:
    def test_basic_rewrite(self):
        content = """
inlet_main
{
    type    fixedValue;
    value   uniform 0.5;
}
outlet
{
    type    zeroGradient;
}
""".strip()
        out = RhoSimpleFoamSolver._rewrite_patch_body(
            content, "outlet", "type            inletOutlet;\nvalue           $internalField;"
        )
        assert out is not None
        assert "inletOutlet" in out
        # inlet_main left alone
        assert "uniform 0.5" in out

    def test_missing_patch_returns_none(self):
        content = "inlet { type fixedValue; }"
        out = RhoSimpleFoamSolver._rewrite_patch_body(
            content, "nonexistent", "type zeroGradient;"
        )
        assert out is None


# ── Outlet backflow fixer (#7) ──────────────────────────────────────────────


_OUTLET_CONFIG = {
    "boundary_conditions": {
        "inlet": {"patch_class": "inlet"},
        "outlet": {"patch_class": "outlet"},
        "walls": {"patch_class": "wall"},
    },
    "turbulence": {"hydraulic_diameter": 0.02},
}


def _0U_with_outlet_zeroGradient() -> str:
    return """FoamFile { class volVectorField; object U; }

internalField   uniform (0 0 0);

boundaryField
{
    inlet
    {
        type            flowRateInletVelocity;
        massFlowRate    constant 0.012;
        value           uniform (0 0 0);
    }
    outlet
    {
        type            zeroGradient;
    }
    walls
    {
        type            noSlip;
    }
}
"""


def _0T_with_outlet_zeroGradient() -> str:
    return """FoamFile { class volScalarField; object T; }
internalField uniform 300;
boundaryField
{
    inlet
    {
        type    fixedValue;
        value   uniform 300;
    }
    outlet
    {
        type    zeroGradient;
    }
    walls
    {
        type    fixedValue;
        value   uniform 600;
    }
}
"""


class TestOutletBackflowFixer:
    def test_outlet_U_zeroGradient_becomes_inletOutlet(self):
        plugin = RhoSimpleFoamSolver()
        files = {"0/U": _0U_with_outlet_zeroGradient()}
        issues: list[ValidationIssue] = []
        out = plugin._fix_outlet_backflow_bcs(files, issues, _OUTLET_CONFIG)
        assert "inletOutlet" in out["0/U"]
        # Inlet block must NOT have been touched.
        assert "flowRateInletVelocity" in out["0/U"]
        # An issue was emitted.
        assert any(
            "Outlet 'outlet' on U" in (i.message or "") for i in issues
        )

    def test_outlet_T_zeroGradient_becomes_inletOutlet(self):
        plugin = RhoSimpleFoamSolver()
        files = {"0/T": _0T_with_outlet_zeroGradient()}
        issues: list[ValidationIssue] = []
        out = plugin._fix_outlet_backflow_bcs(files, issues, _OUTLET_CONFIG)
        # outlet block: inletOutlet now.
        outlet_block = out["0/T"].split("outlet")[1].split("walls")[0]
        assert "inletOutlet" in outlet_block
        assert "zeroGradient" not in outlet_block
        # walls block: still fixedValue 600
        assert "uniform 600" in out["0/T"]

    def test_outlet_already_inletOutlet_is_noop(self):
        plugin = RhoSimpleFoamSolver()
        already = """FoamFile { class volVectorField; }
boundaryField
{
    outlet
    {
        type            inletOutlet;
        inletValue      uniform (0 0 0);
        value           uniform (0 0 0);
    }
}
"""
        files = {"0/U": already}
        issues: list[ValidationIssue] = []
        out = plugin._fix_outlet_backflow_bcs(files, issues, _OUTLET_CONFIG)
        assert out["0/U"] == already
        assert issues == []

    def test_no_outlet_in_config_is_noop(self):
        plugin = RhoSimpleFoamSolver()
        cfg = {"boundary_conditions": {"inlet": {"patch_class": "inlet"}}}
        files = {"0/U": _0U_with_outlet_zeroGradient()}
        issues: list[ValidationIssue] = []
        out = plugin._fix_outlet_backflow_bcs(files, issues, cfg)
        assert "zeroGradient" in out["0/U"]
        assert issues == []


# ── Inlet turbulence BC type fixer (#8) ─────────────────────────────────────


_INLET_CONFIG = {
    "boundary_conditions": {
        "inlet_main": {"patch_class": "inlet"},
        "inlet_small": {"patch_class": "inlet"},
        "outlet": {"patch_class": "outlet"},
    },
    "turbulence": {
        "hydraulic_diameter": 0.02,
        "turbulence_intensity": 5.0,
    },
}


def _0k_with_inlet_fixedValue() -> str:
    return """FoamFile { class volScalarField; object k; }
internalField uniform 0.0002;
boundaryField
{
    inlet_main
    {
        type            fixedValue;
        value           uniform 0.0002;
    }
    inlet_small
    {
        type            fixedValue;
        value           uniform 0.0002;
    }
    outlet
    {
        type            zeroGradient;
    }
    walls
    {
        type            kqRWallFunction;
        value           uniform 0.0002;
    }
}
"""


def _0omega_with_inlet_fixedValue() -> str:
    return """FoamFile { class volScalarField; object omega; }
internalField uniform 100;
boundaryField
{
    inlet_main
    {
        type            fixedValue;
        value           uniform 100;
    }
    inlet_small
    {
        type            fixedValue;
        value           uniform 44.7;
    }
    outlet
    {
        type            zeroGradient;
    }
    walls
    {
        type            omegaWallFunction;
        value           uniform 100;
    }
}
"""


class TestInletTurbulenceBCTypeFixer:
    def test_inlet_k_fixedValue_becomes_turbulentIntensity(self):
        plugin = RhoSimpleFoamSolver()
        files = {"0/k": _0k_with_inlet_fixedValue()}
        issues: list[ValidationIssue] = []
        out = plugin._fix_inlet_turbulence_bc_types(files, issues, _INLET_CONFIG)
        # Both inlets converted.
        assert out["0/k"].count("turbulentIntensityKineticEnergyInlet") == 2
        assert "intensity       0.0500;" in out["0/k"]
        # Outlet untouched (zeroGradient kept — the outlet fixer handles that).
        outlet_block = out["0/k"].split("outlet")[1].split("walls")[0]
        assert "zeroGradient" in outlet_block
        # Walls untouched.
        assert "kqRWallFunction" in out["0/k"]

    def test_inlet_omega_fixedValue_becomes_mixingLengthFrequency(self):
        plugin = RhoSimpleFoamSolver()
        files = {"0/omega": _0omega_with_inlet_fixedValue()}
        issues: list[ValidationIssue] = []
        out = plugin._fix_inlet_turbulence_bc_types(files, issues, _INLET_CONFIG)
        # Both inlets converted to turbulentMixingLengthFrequencyInlet.
        assert out["0/omega"].count("turbulentMixingLengthFrequencyInlet") == 2
        # mixingLength = 0.07 · 0.02 = 0.0014 m.
        assert "mixingLength    0.0014;" in out["0/omega"]
        # The fallback 'value' line is preserved per inlet — different
        # values survive the rewrite (100 for inlet_main, 44.7 for inlet_small).
        assert "uniform 100" in out["0/omega"]
        assert "uniform 44.7" in out["0/omega"]
        # Walls untouched.
        assert "omegaWallFunction" in out["0/omega"]

    def test_default_mixing_length_when_d_h_missing(self):
        plugin = RhoSimpleFoamSolver()
        files = {"0/omega": _0omega_with_inlet_fixedValue()}
        cfg = {
            "boundary_conditions": {"inlet_main": {"patch_class": "inlet"}},
            "turbulence": {},  # No D_h!
        }
        issues: list[ValidationIssue] = []
        out = plugin._fix_inlet_turbulence_bc_types(files, issues, cfg)
        # Falls back to 1 cm.
        assert "mixingLength    0.0100;" in out["0/omega"]

    def test_no_inlet_in_config_is_noop(self):
        plugin = RhoSimpleFoamSolver()
        cfg = {"boundary_conditions": {"outlet": {"patch_class": "outlet"}}}
        files = {"0/k": _0k_with_inlet_fixedValue()}
        issues: list[ValidationIssue] = []
        out = plugin._fix_inlet_turbulence_bc_types(files, issues, cfg)
        assert out["0/k"] == _0k_with_inlet_fixedValue()
        assert issues == []

    def test_already_turbulent_intensity_bc_is_noop(self):
        plugin = RhoSimpleFoamSolver()
        already = """FoamFile { class volScalarField; }
boundaryField
{
    inlet_main
    {
        type            turbulentIntensityKineticEnergyInlet;
        intensity       0.05;
        value           uniform 0.0002;
    }
}
"""
        files = {"0/k": already}
        cfg = {"boundary_conditions": {"inlet_main": {"patch_class": "inlet"}}}
        issues: list[ValidationIssue] = []
        out = plugin._fix_inlet_turbulence_bc_types(files, issues, cfg)
        assert out["0/k"] == already
        assert issues == []

    def test_epsilon_uses_dissipation_rate_inlet(self):
        plugin = RhoSimpleFoamSolver()
        epsilon = """FoamFile { class volScalarField; object epsilon; }
boundaryField
{
    inlet_main
    {
        type            fixedValue;
        value           uniform 1.0;
    }
}
"""
        files = {"0/epsilon": epsilon}
        cfg = {
            "boundary_conditions": {"inlet_main": {"patch_class": "inlet"}},
            "turbulence": {"hydraulic_diameter": 0.05},
        }
        issues: list[ValidationIssue] = []
        out = plugin._fix_inlet_turbulence_bc_types(files, issues, cfg)
        assert "turbulentMixingLengthDissipationRateInlet" in out["0/epsilon"]
        # 0.07 · 0.05 = 0.0035 m
        assert "mixingLength    0.0035;" in out["0/epsilon"]
