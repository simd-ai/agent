# tests/test_buoyant_boussinesq_solvers.py
"""Tests for buoyantBoussinesq{Simple,Pimple}Foam — incompressible Boussinesq.

Tracks the OpenFOAM 4.x ``tutorials/heatTransfer/buoyantBoussinesq*``
reference: constant density, buoyancy via ``-ρ₀·β·(T−T_ref)·g`` source,
T transport equation (not h or e), transportProperties (not
thermophysicalProperties), p_rgh + constant/g.

Key structural invariants pinned here:

  * Mixin-first MRO: the IncompressibleBoussinesqMixin overrides
    ``SolverPlugin`` defaults (would otherwise mask them).
  * ``div_block`` emits ``div(phi,T)`` and **does NOT** emit
    ``div(phi,K)`` / ``div(phi,Ekp)`` (Boussinesq energy has no kinetic-
    energy convection term).
  * ``div_block`` emits **no** ``div(phi*,p)`` line (incompressible).
  * Viscous term uses the incompressible form ``div((nuEff*dev2…))``.
  * PIMPLE variant emits ``TFinal``, ``p_rghFinal``, etc. — no ``rhoFinal``
    because there's no rho field.
  * ``constant/transportProperties`` is rendered deterministically
    (``constant/thermophysicalProperties`` removed if the LLM emits one).
"""

from __future__ import annotations

import re

from simd_agent.solvers.families import (
    IncompressibleBoussinesqMixin,
    SteadyBase,
    TransientBase,
)
from simd_agent.solvers.heatTransfer.buoyantBoussinesqSimpleFoam.solver import (
    BuoyantBoussinesqSimpleFoamSolver,
)
from simd_agent.solvers.heatTransfer.buoyantBoussinesqPimpleFoam.solver import (
    BuoyantBoussinesqPimpleFoamSolver,
)


_BASE_CFG = {
    "physics": {
        "compressibility": "incompressible",
        "heat_transfer": True,
        "gravity": True,
        "turbulence_model": "kEpsilon",
        "flow_regime": "turbulent",
    },
    "fluid": {
        "rho": 1.0,
        "mu": 1.5e-5,
        "beta": 3e-3,
        "temperature": 300,
    },
    "boundary_conditions": {
        "inlet": {"patch_class": "inlet", "temperature": {"value": 293}},
        "outlet": {"patch_class": "outlet", "pressure": {"value": 0}},
        "walls": {"patch_class": "wall", "temperature": {"value": 310}},
    },
    "mesh": {},
}


# ── Identity + MRO ──────────────────────────────────────────────────────────


class TestBoussinesqIdentity:
    def test_steady_attributes(self):
        p = BuoyantBoussinesqSimpleFoamSolver()
        assert p.name == "buoyantBoussinesqSimpleFoam"
        assert p.algorithm == "SIMPLE"
        assert p.is_transient is False
        assert p.is_compressible is False
        assert p.energy_var == "T"
        assert p.pressure_field == "p_rgh"
        assert p.needs_gravity is True
        assert p.supports_energy is True
        assert p.is_multiphase is False

    def test_transient_attributes(self):
        p = BuoyantBoussinesqPimpleFoamSolver()
        assert p.name == "buoyantBoussinesqPimpleFoam"
        assert p.algorithm == "PIMPLE"
        assert p.is_transient is True
        assert p.is_compressible is False
        assert p.energy_var == "T"
        assert p.pressure_field == "p_rgh"
        assert p.needs_gravity is True

    def test_mixin_first_mro_for_steady(self):
        """``IncompressibleBoussinesqMixin`` must come before ``SolverPlugin``."""
        mro = [c.__name__ for c in BuoyantBoussinesqSimpleFoamSolver.__mro__]
        assert mro.index("IncompressibleBoussinesqMixin") < mro.index("SolverPlugin")
        assert mro.index("BoussinesqMixin") < mro.index("SolverPlugin")

    def test_mixin_first_mro_for_transient(self):
        mro = [c.__name__ for c in BuoyantBoussinesqPimpleFoamSolver.__mro__]
        assert mro.index("IncompressibleBoussinesqMixin") < mro.index("SolverPlugin")

    def test_inherits_family_bases(self):
        assert issubclass(BuoyantBoussinesqSimpleFoamSolver, SteadyBase)
        assert issubclass(BuoyantBoussinesqSimpleFoamSolver, IncompressibleBoussinesqMixin)
        assert issubclass(BuoyantBoussinesqPimpleFoamSolver, TransientBase)
        assert issubclass(BuoyantBoussinesqPimpleFoamSolver, IncompressibleBoussinesqMixin)


# ── divSchemes — Boussinesq has no K/Ekp and no div(phi*,p) ────────────────


class TestDivSchemes:
    def test_steady_div_block_emits_div_phi_T(self):
        plugin = BuoyantBoussinesqSimpleFoamSolver()
        fvs = plugin.render_deterministic_files(_BASE_CFG)["system/fvSchemes"]
        div = fvs.split("divSchemes")[1].split("laplacianSchemes")[0]
        assert "div(phi,U)" in div
        assert "div(phi,T)" in div
        assert "div(phi,k)" in div
        assert "div(phi,epsilon)" in div

    def test_steady_div_block_omits_K_and_Ekp(self):
        """Boussinesq energy equation has no kinetic-energy convection term."""
        plugin = BuoyantBoussinesqSimpleFoamSolver()
        fvs = plugin.render_deterministic_files(_BASE_CFG)["system/fvSchemes"]
        div = fvs.split("divSchemes")[1].split("laplacianSchemes")[0]
        assert "div(phi,K)" not in div
        assert "div(phi,Ekp)" not in div

    def test_steady_div_block_omits_pressure_flux(self):
        """Incompressible: no div(phid,p) or div(phiv,p)."""
        plugin = BuoyantBoussinesqSimpleFoamSolver()
        fvs = plugin.render_deterministic_files(_BASE_CFG)["system/fvSchemes"]
        div = fvs.split("divSchemes")[1].split("laplacianSchemes")[0]
        assert "div(phid,p)" not in div
        assert "div(phiv,p)" not in div

    def test_steady_viscous_term_is_incompressible_form(self):
        plugin = BuoyantBoussinesqSimpleFoamSolver()
        fvs = plugin.render_deterministic_files(_BASE_CFG)["system/fvSchemes"]
        # Incompressible: div((nuEff*dev2(T(grad(U))))) — no rho factor.
        assert "div((nuEff*dev2(T(grad(U)))))" in fvs
        assert "rho*nuEff" not in fvs

    def test_transient_inherits_same_div_structure(self):
        plugin = BuoyantBoussinesqPimpleFoamSolver()
        fvs = plugin.render_deterministic_files(_BASE_CFG)["system/fvSchemes"]
        div = fvs.split("divSchemes")[1].split("laplacianSchemes")[0]
        assert "div(phi,T)" in div
        assert "div(phi,K)" not in div
        assert "div(phid,p)" not in div


# ── PIMPLE Final coverage — no rhoFinal ────────────────────────────────────


def _top_level_solver_entries(fvs: str) -> list[str]:
    solvers_block = fvs.split("solvers")[1].split("\n}")[0]
    return re.findall(
        r'^\s{4}(\w+|"[^"]+")\n\s{4}\{', solvers_block, re.MULTILINE
    )


class TestPimpleFinalCoverage:
    def test_pimple_has_p_rgh_T_U_finals(self):
        plugin = BuoyantBoussinesqPimpleFoamSolver()
        fvs = plugin.render_deterministic_files(_BASE_CFG)["system/fvSolution"]
        entries = _top_level_solver_entries(fvs)
        assert "p_rgh" in entries
        assert "p_rghFinal" in entries
        # T has its own block (PBiCG+DILU) because supports_energy=True.
        assert "T" in entries
        assert "TFinal" in entries

    def test_pimple_has_NO_rho_or_rhoFinal(self):
        """Incompressible Boussinesq — no rho field at all."""
        plugin = BuoyantBoussinesqPimpleFoamSolver()
        fvs = plugin.render_deterministic_files(_BASE_CFG)["system/fvSolution"]
        entries = _top_level_solver_entries(fvs)
        assert "rho" not in entries
        assert "rhoFinal" not in entries

    def test_simple_has_no_finals(self):
        """SIMPLE has no Final variants."""
        plugin = BuoyantBoussinesqSimpleFoamSolver()
        fvs = plugin.render_deterministic_files(_BASE_CFG)["system/fvSolution"]
        entries = _top_level_solver_entries(fvs)
        assert not any(e.endswith(("Final", 'Final"')) for e in entries)


# ── transportProperties — deterministic; thermophysicalProperties removed ──


class TestTransportProperties:
    def test_renders_transportProperties_with_fluid_values(self):
        plugin = BuoyantBoussinesqSimpleFoamSolver()
        files = plugin.render_deterministic_files(_BASE_CFG)
        tp = files["constant/transportProperties"]
        # Object header.
        assert "object      transportProperties;" in tp
        assert "transportModel  Newtonian;" in tp
        # nu from mu/rho = 1.5e-5 / 1.0 = 1.5e-5.
        assert "nu              [0 2 -1 0 0 0 0] 1.5e-05;" in tp
        # beta from fluid.beta = 3e-3.
        assert "beta            [0 0 0 -1 0 0 0] 0.003;" in tp
        # TRef from fluid.temperature = 300.
        assert "TRef            [0 0 0 1 0 0 0] 300;" in tp

    def test_validate_strips_thermophysicalProperties(self):
        plugin = BuoyantBoussinesqSimpleFoamSolver()
        files = {"constant/thermophysicalProperties": "thermoType { … }"}
        result = plugin.validate(files, _BASE_CFG)
        assert "constant/thermophysicalProperties" not in result.files
        # And there's a warning about the removal.
        assert any(
            "thermophysicalProperties" in i.file for i in result.issues
        )

    def test_default_values_when_fluid_missing(self):
        """No fluid block → hotRoom defaults."""
        plugin = BuoyantBoussinesqSimpleFoamSolver()
        cfg = {**_BASE_CFG, "fluid": {}}
        tp = plugin.render_deterministic_files(cfg)["constant/transportProperties"]
        # Default ν = 1e-5 (from the OF hotRoom tutorial).
        assert "1e-05" in tp
        # Default β = 3e-3.
        assert "0.003" in tp


# ── Matching ────────────────────────────────────────────────────────────────


class TestMatching:
    def test_steady_buoyant_incompressible_with_heat_scores_high(self):
        plugin = BuoyantBoussinesqSimpleFoamSolver()
        m = plugin.matches({
            "physics": {
                "compressibility": "incompressible",
                "heat_transfer": True,
                "gravity": True,
                "time_scheme": "steady",
            },
        })
        assert m.score >= 0.9
        assert m.matches

    def test_steady_solver_rejects_transient_config(self):
        plugin = BuoyantBoussinesqSimpleFoamSolver()
        m = plugin.matches({
            "physics": {
                "compressibility": "incompressible",
                "heat_transfer": True,
                "gravity": True,
                "time_scheme": "transient",
            },
        })
        assert m.score < 0.5

    def test_transient_solver_rejects_steady_config(self):
        plugin = BuoyantBoussinesqPimpleFoamSolver()
        m = plugin.matches({
            "physics": {
                "compressibility": "incompressible",
                "heat_transfer": True,
                "gravity": True,
                "time_scheme": "steady",
            },
        })
        assert m.score < 0.5

    def test_compressible_config_rejects_Boussinesq(self):
        """Compressible buoyancy → buoyantSimpleFoam, not the Boussinesq variant."""
        plugin = BuoyantBoussinesqSimpleFoamSolver()
        m = plugin.matches({
            "physics": {
                "compressibility": "compressible",
                "heat_transfer": True,
                "gravity": True,
                "time_scheme": "steady",
            },
        })
        assert m.score < 0.5


# ── Registry auto-discovery ─────────────────────────────────────────────────


class TestRegistry:
    def test_both_solvers_registered(self):
        from simd_agent.solvers import get_registry
        registry = get_registry()
        names = registry.names()
        assert "buoyantBoussinesqSimpleFoam" in names
        assert "buoyantBoussinesqPimpleFoam" in names

    def test_both_classified_as_p_rgh_and_gravity(self):
        from simd_agent.solvers import get_registry
        registry = get_registry()
        p_rgh_solvers = registry.p_rgh_solvers()
        gravity_solvers = registry.gravity_solvers()
        assert "buoyantBoussinesqSimpleFoam" in p_rgh_solvers
        assert "buoyantBoussinesqPimpleFoam" in p_rgh_solvers
        assert "buoyantBoussinesqSimpleFoam" in gravity_solvers
        assert "buoyantBoussinesqPimpleFoam" in gravity_solvers
