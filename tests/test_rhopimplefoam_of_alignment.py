# tests/test_rhopimplefoam_of_alignment.py
"""Tests for rhoPimpleFoam OF-tutorial alignment.

Tracks the OpenFOAM 4.x ``compressible/rhoPimpleFoam/ras/angledDuct``
tutorial.  Three changes covered here:

  1. ``_fix_outlet_backflow_bcs`` + ``_fix_inlet_turbulence_bc_types``
     hoisted to ``SolverPlugin`` base so rhoPimpleFoam inherits the same
     robust BCs as rhoSimpleFoam.

  2. PIMPLE block adds ``consistent yes``, ``transonic no`` and
     ``turbOnFinalIterOnly no`` — explicit settings that prevent
     fork-to-fork drift.

  3. ``relaxationFactors`` carries the OF tutorial's full structure:
     ``fields { "p.*" 0.9; "rho.*" 1; }`` and
     ``equations { "U.*" 0.9; "h.*" 0.7; "(k|epsilon|omega).*" 0.8; }``.
"""

from __future__ import annotations

from simd_agent.solvers.base import SolverPlugin
from simd_agent.solvers.compressible.rhoPimpleFoam.solver import RhoPimpleFoamSolver
from simd_agent.solvers.compressible.rhoSimpleFoam.solver import RhoSimpleFoamSolver


# ── BC fixers are inherited, not duplicated ─────────────────────────────────


class TestBCFixersOnBase:
    def test_methods_live_on_SolverPlugin(self):
        """Hoisted methods are defined on the abstract base."""
        for name in (
            "_rewrite_patch_body",
            "_classify_patches",
            "_fix_outlet_backflow_bcs",
            "_fix_inlet_turbulence_bc_types",
        ):
            assert name in SolverPlugin.__dict__, (
                f"{name} should be defined on SolverPlugin, not a plugin subclass"
            )

    def test_rhoSimpleFoam_inherits_via_base(self):
        """rhoSimpleFoam does NOT redefine the methods (no override)."""
        for name in (
            "_rewrite_patch_body",
            "_classify_patches",
            "_fix_outlet_backflow_bcs",
            "_fix_inlet_turbulence_bc_types",
        ):
            assert name not in RhoSimpleFoamSolver.__dict__, (
                f"{name} should be inherited from SolverPlugin, not re-defined "
                "on RhoSimpleFoamSolver"
            )

    def test_rhoPimpleFoam_inherits_via_base(self):
        """rhoPimpleFoam picks up the BC fixers via inheritance."""
        plugin = RhoPimpleFoamSolver()
        assert hasattr(plugin, "_fix_outlet_backflow_bcs")
        assert hasattr(plugin, "_fix_inlet_turbulence_bc_types")


# ── End-to-end: rhoPimpleFoam BC fixers actually fire ───────────────────────


_RHOPIMPLE_CFG = {
    "physics": {
        "compressibility": "compressible",
        "heat_transfer": True,
        "time_scheme": "transient",
        "turbulence_model": "kEpsilon",
    },
    "fluid": {"rho": 1.18, "mu": 1.81e-5, "Cp": 1006, "k": 0.026, "temperature": 300},
    "boundary_conditions": {
        "inlet": {
            "patch_class": "inlet",
            "pressure": {"value": 1.0e5},
            "temperature": {"value": 293},
        },
        "outlet": {
            "patch_class": "outlet",
            "pressure": {"value": 1.0e5},
            "temperature": {"value": 293},
        },
        "walls": {"patch_class": "wall"},
    },
    "turbulence": {"hydraulic_diameter": 0.02, "turbulence_intensity": 5.0},
    "mesh": {},
}


def _0k_with_outlet_zeroGradient_and_inlet_fixedValue() -> str:
    return """FoamFile { class volScalarField; object k; }
internalField uniform 1;
boundaryField
{
    inlet
    {
        type            fixedValue;
        value           uniform 1;
    }
    outlet
    {
        type            zeroGradient;
    }
    walls
    {
        type            kqRWallFunction;
        value           uniform 1;
    }
}
"""


class TestRhoPimpleFoamValidateRunsBCFixers:
    def test_outlet_zeroGradient_becomes_inletOutlet(self):
        plugin = RhoPimpleFoamSolver()
        files = {"0/k": _0k_with_outlet_zeroGradient_and_inlet_fixedValue()}
        result = plugin.validate(files, _RHOPIMPLE_CFG)
        out_k = result.files["0/k"]
        # Outlet block converted.
        outlet_block = out_k.split("outlet")[1].split("walls")[0]
        assert "inletOutlet" in outlet_block
        assert "zeroGradient" not in outlet_block

    def test_inlet_fixedValue_becomes_turbulentIntensity(self):
        plugin = RhoPimpleFoamSolver()
        files = {"0/k": _0k_with_outlet_zeroGradient_and_inlet_fixedValue()}
        result = plugin.validate(files, _RHOPIMPLE_CFG)
        out_k = result.files["0/k"]
        assert "turbulentIntensityKineticEnergyInlet" in out_k
        assert "intensity       0.0500;" in out_k


# ── PIMPLE block enrichment ─────────────────────────────────────────────────


class TestPimpleBlockKeys:
    def test_consistent_transonic_turbOnFinalIterOnly_present(self):
        plugin = RhoPimpleFoamSolver()
        fvs = plugin.render_deterministic_files(_RHOPIMPLE_CFG)["system/fvSolution"]
        pimple_block = fvs.split("PIMPLE")[1].split("relaxationFactors")[0]
        assert "consistent          yes;" in pimple_block
        assert "transonic           no;" in pimple_block
        assert "turbOnFinalIterOnly no;" in pimple_block


# ── relaxationFactors structure ─────────────────────────────────────────────


class TestRelaxationFactorsStructure:
    def test_compressible_has_fields_block_with_p_and_rho(self):
        plugin = RhoPimpleFoamSolver()
        fvs = plugin.render_deterministic_files(_RHOPIMPLE_CFG)["system/fvSolution"]
        relax = fvs.split("relaxationFactors")[1]
        # fields block with the OF-tutorial patterns.
        assert "fields" in relax
        fields_block = relax.split("fields")[1].split("equations")[0]
        assert '"p.*"' in fields_block
        assert '"rho.*"' in fields_block
        assert "0.9" in fields_block
        assert "1" in fields_block  # rho

    def test_compressible_equations_use_OF_patterns(self):
        plugin = RhoPimpleFoamSolver()
        fvs = plugin.render_deterministic_files(_RHOPIMPLE_CFG)["system/fvSolution"]
        relax = fvs.split("relaxationFactors")[1]
        eq_block = relax.split("equations")[1]
        # OF-tutorial relaxation patterns.
        assert '"U.*"' in eq_block
        assert '"h.*"' in eq_block
        assert '"(k|epsilon|omega).*"' in eq_block
        # OF values: U 0.9, h 0.7, turb 0.8.
        assert "0.9" in eq_block
        assert "0.7" in eq_block
        assert "0.8" in eq_block

    def test_incompressible_pimple_has_no_fields_block(self):
        """Incompressible pimpleFoam — fields block is omitted (no rho)."""
        from simd_agent.solvers.incompressible.pimpleFoam.solver import PimpleFoamSolver
        plugin = PimpleFoamSolver()
        cfg = {
            "physics": {
                "compressibility": "incompressible",
                "time_scheme": "transient",
                "turbulence_model": "kOmegaSST",
            },
            "fluid": {"rho": 1000, "mu": 1e-3, "Cp": 4182, "k": 0.6, "temperature": 300},
            "boundary_conditions": {
                "inlet": {"patch_class": "inlet"},
                "outlet": {"patch_class": "outlet"},
                "walls": {"patch_class": "wall"},
            },
            "mesh": {},
        }
        fvs = plugin.render_deterministic_files(cfg)["system/fvSolution"]
        relax = fvs.split("relaxationFactors")[1]
        # No ``fields`` sub-block for incompressible PIMPLE.
        assert "fields" not in relax.split("equations")[0]
