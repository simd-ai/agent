# tests/test_turbulence_regime_profile.py
"""Tests for the per-regime scheme bundle (TurbulenceRegimeProfile + resolver).

Tracks the three OpenFOAM 4.x rhoPimpleFoam reference tutorials:

  * compressible/rhoPimpleFoam/laminar/helmholtzResonance
  * compressible/rhoPimpleFoam/ras/angledDuct
  * compressible/rhoPimpleFoam/les/pitzDaily

plus the rhoSimpleFoam/ras tutorial for SIMPLE-mode steady.  The resolver
encodes every per-regime scheme choice; the renderer reads attribute
access against the resolved profile.
"""

from __future__ import annotations

import pytest

from simd_agent.run.case_spec import (
    TurbulenceRegimeProfile,
    resolve_regime_profile,
)
from simd_agent.solvers.rhoPimpleFoam.solver import RhoPimpleFoamSolver
from simd_agent.solvers.rhoSimpleFoam.solver import RhoSimpleFoamSolver


# ── Pydantic invariants ─────────────────────────────────────────────────────


class TestProfileInvariants:
    def test_laminar_must_not_declare_div_phi_turb(self):
        with pytest.raises(ValueError, match="must not declare div_phi_turb"):
            TurbulenceRegimeProfile(
                simulation_type="laminar",
                ddt_scheme="Euler",
                div_phi_U="Gauss limitedLinearV 1",
                div_phi_energy="Gauss limitedLinear 1",
                div_phi_K="Gauss limitedLinear 1",
                div_phi_p="Gauss limitedLinear 1",
                div_phi_turb="Gauss upwind",  # ← invalid for laminar
                pressure_flux="phiv",
                turbulence_properties_block="simulationType laminar;\n",
            )

    def test_ras_requires_div_phi_turb(self):
        with pytest.raises(ValueError, match="requires.*div_phi_turb"):
            TurbulenceRegimeProfile(
                simulation_type="RAS",
                ddt_scheme="Euler",
                div_phi_U="Gauss upwind",
                div_phi_energy="Gauss upwind",
                div_phi_K="Gauss linear",
                div_phi_p="Gauss upwind",
                div_phi_turb=None,  # ← invalid for RAS
                pressure_flux="phid",
                turbulence_properties_block="simulationType RAS;\nRAS{}\n",
            )

    def test_les_requires_div_phi_turb(self):
        with pytest.raises(ValueError, match="requires.*div_phi_turb"):
            TurbulenceRegimeProfile(
                simulation_type="LES",
                ddt_scheme="backward",
                div_phi_U="Gauss LUST grad(U)",
                div_phi_energy="Gauss LUST grad(h)",
                div_phi_K="Gauss linear",
                div_phi_p="Gauss linear",
                div_phi_turb=None,  # ← invalid for LES
                pressure_flux="phiv",
                turbulence_properties_block="simulationType LES;\nLES{}\n",
            )


# ── Resolver vs OF tutorial values ──────────────────────────────────────────


class TestLaminarResolver:
    """OF: tutorials/compressible/rhoPimpleFoam/laminar/helmholtzResonance"""

    def test_matches_OF_laminar_choices(self):
        p = resolve_regime_profile(
            simulation_type="laminar",
            turb_model="laminar",
            algorithm="PIMPLE",
            is_compressible=True,
            energy_var="e",
        )
        assert p.ddt_scheme == "Euler"
        assert p.div_phi_U == "Gauss limitedLinearV 1"
        assert p.div_phi_energy == "Gauss limitedLinear 1"
        assert p.div_phi_K == "Gauss limitedLinear 1"
        assert p.div_phi_p == "Gauss limitedLinear 1"
        assert p.div_phi_turb is None
        assert p.pressure_flux == "phiv"
        assert "simulationType  laminar;" in p.turbulence_properties_block


class TestRASResolver:
    """OF: tutorials/compressible/rhoPimpleFoam/ras/angledDuct"""

    def test_pimple_matches_OF_ras_choices(self):
        p = resolve_regime_profile(
            simulation_type="RAS",
            turb_model="kEpsilon",
            algorithm="PIMPLE",
            is_compressible=True,
            energy_var="h",
        )
        assert p.ddt_scheme == "Euler"
        # OF tutorial uses plain ``Gauss upwind`` for div(phi,U) in PIMPLE.
        # We emit ``bounded Gauss upwind`` — same scheme, just adds the
        # ``bounded`` keyword which clamps the corrected scheme to be
        # bounded.  Both accepted by OpenFOAM.
        assert "Gauss upwind" in p.div_phi_U
        assert "Gauss upwind" in p.div_phi_energy
        # KEY: div(phi,K) is Gauss linear in PIMPLE (NOT bounded upwind).
        assert p.div_phi_K == "Gauss linear"
        assert p.pressure_flux == "phid"  # compressibility-coupled
        # Turbulence-properties block has RASModel kEpsilon.
        assert "simulationType  RAS;" in p.turbulence_properties_block
        assert "RASModel        kEpsilon;" in p.turbulence_properties_block

    def test_simple_uses_steadyState_ddt(self):
        """rhoSimpleFoam (SIMPLE-mode RAS) — ddt is steadyState, not Euler."""
        p = resolve_regime_profile(
            simulation_type="RAS",
            turb_model="kOmegaSST",
            algorithm="SIMPLE",
            is_compressible=True,
            energy_var="e",
        )
        assert p.ddt_scheme == "steadyState"
        # SIMPLE keeps the upwind safety net for div(phi,K).
        assert p.div_phi_K == "bounded Gauss upwind"


class TestLESResolver:
    """OF: tutorials/compressible/rhoPimpleFoam/les/pitzDaily"""

    def test_matches_OF_les_choices(self):
        p = resolve_regime_profile(
            simulation_type="LES",
            turb_model="kEqn",
            algorithm="PIMPLE",
            is_compressible=True,
            energy_var="h",
        )
        # LES uses backward time-stepping for second-order accuracy.
        assert p.ddt_scheme == "backward"
        assert p.div_phi_U == "Gauss LUST grad(U)"
        assert p.div_phi_energy == "Gauss LUST grad(h)"
        assert p.div_phi_K == "Gauss linear"
        assert p.div_phi_p == "Gauss linear"
        # LES is low-Mach by construction → kinematic pressure flux.
        assert p.pressure_flux == "phiv"
        # Turbulence properties carry the full LES sub-dict.
        block = p.turbulence_properties_block
        assert "simulationType  LES;" in block
        assert "LESModel        kEqn;" in block
        assert "delta           cubeRootVol;" in block
        assert "cubeRootVolCoeffs" in block

    def test_les_uses_energy_var_in_div_phi_energy(self):
        """The energy var (h or e) flows through to the LUST grad(…) name."""
        for energy_var in ("h", "e"):
            p = resolve_regime_profile(
                simulation_type="LES",
                turb_model="dynamicKEqn",
                algorithm="PIMPLE",
                is_compressible=True,
                energy_var=energy_var,
            )
            assert p.div_phi_energy == f"Gauss LUST grad({energy_var})"


# ── End-to-end render: rhoPimpleFoam over three regimes ─────────────────────


_BASE_CFG = {
    "physics": {
        "compressibility": "compressible",
        "heat_transfer": True,
        "time_scheme": "transient",
    },
    "fluid": {"rho": 1.18, "mu": 1.81e-5, "Cp": 1006, "k": 0.026, "temperature": 300},
    "boundary_conditions": {
        "inlet": {"patch_class": "inlet", "pressure": {"value": 1.0e5}, "temperature": {"value": 293}},
        "outlet": {"patch_class": "outlet", "pressure": {"value": 1.0e5}, "temperature": {"value": 293}},
        "walls": {"patch_class": "wall"},
    },
    "mesh": {},
}


def _cfg(model: str, flow_regime: str | None = None, sim_type: str | None = None):
    cfg = {
        **_BASE_CFG,
        "physics": {**_BASE_CFG["physics"], "turbulence_model": model},
    }
    if flow_regime:
        cfg["physics"]["flow_regime"] = flow_regime
    if sim_type:
        cfg["physics"]["simulation_type"] = sim_type
    return cfg


class TestRhoPimpleFoamEndToEnd:
    """Whole-file render check: every regime produces the right divSchemes."""

    def test_laminar_renders_laminar_schemes(self):
        plugin = RhoPimpleFoamSolver()
        files = plugin.render_deterministic_files(
            _cfg("laminar", flow_regime="laminar")
        )
        sc = files["system/fvSchemes"]
        assert "Gauss limitedLinearV 1" in sc      # div(phi,U)
        assert "div(phiv,p)" in sc                 # kinematic flux
        assert "div(phid,p)" not in sc             # not compressible flux
        # No transported-turbulence div lines for laminar.
        assert "div(phi,k)" not in sc
        assert "div(phi,epsilon)" not in sc
        # turbulenceProperties is just the one line.
        tp = files["constant/turbulenceProperties"]
        assert "simulationType  laminar;" in tp
        assert "RAS" not in tp
        assert "LES" not in tp

    def test_ras_renders_ras_schemes(self):
        plugin = RhoPimpleFoamSolver()
        files = plugin.render_deterministic_files(_cfg("kEpsilon"))
        sc = files["system/fvSchemes"]
        # RAS uses bounded Gauss upwind for U and h.
        assert "div(phi,U)      bounded Gauss upwind" in sc
        assert "div(phi,h)      bounded Gauss upwind" in sc
        # PIMPLE-mode div(phi,K) is Gauss linear (NOT bounded upwind).
        assert "div(phi,K)      Gauss linear" in sc
        # Compressible flux for pressure.
        assert "div(phid,p)" in sc
        # Transported k + epsilon present.
        assert "div(phi,k)" in sc
        assert "div(phi,epsilon)" in sc
        # turbulenceProperties has the RAS block.
        tp = files["constant/turbulenceProperties"]
        assert "RASModel        kEpsilon;" in tp

    def test_les_renders_les_schemes(self):
        plugin = RhoPimpleFoamSolver()
        files = plugin.render_deterministic_files(
            _cfg("LES", flow_regime="turbulent", sim_type="LES")
        )
        sc = files["system/fvSchemes"]
        # LES uses LUST for U and h.
        assert "Gauss LUST grad(U)" in sc
        assert "Gauss LUST grad(h)" in sc
        assert "div(phi,K)      Gauss linear" in sc
        # LES uses kinematic flux (low-Mach pressure equation).
        assert "div(phiv,p)" in sc
        # ddt is backward for time-accurate LES.
        assert "default         backward;" in sc
        # turbulenceProperties has the LES sub-dict.
        tp = files["constant/turbulenceProperties"]
        assert "simulationType  LES;" in tp
        assert "cubeRootVolCoeffs" in tp


class TestRhoSimpleFoamStillWorks:
    """rhoSimpleFoam (SIMPLE-mode RAS) — regression check."""

    def test_steady_ddt_and_phid_flux(self):
        plugin = RhoSimpleFoamSolver()
        cfg = _cfg("kOmegaSST")
        cfg["physics"]["time_scheme"] = "steady"
        files = plugin.render_deterministic_files(cfg)
        sc = files["system/fvSchemes"]
        # SIMPLE → steadyState ddt.
        assert "default         steadyState;" in sc
        # Compressible RAS still uses phid for pressure.
        assert "div(phid,p)" in sc
        # rhoSimpleFoam uses e (sensibleInternalEnergy) — energy var is ``e``.
        assert "div(phi,e)" in sc
        # And the corresponding Ekp kinetic-energy term.
        assert "div(phi,Ekp)" in sc
