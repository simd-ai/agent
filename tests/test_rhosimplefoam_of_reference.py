# tests/test_rhosimplefoam_of_reference.py
"""Tests that rhoSimpleFoam output matches the OpenFOAM reference tutorial.

Reference:
``OpenFOAM-2.2.x/tutorials/compressible/rhoSimpleFoam/angledDuctExplicitFixedCoeff``

Two structural choices from that tutorial drive these tests:

  1. ``div(phi,U) = bounded Gauss upwind`` — UNCONDITIONALLY for
     SIMPLE-mode compressible.  The steady solver has no time
     derivative to absorb the acoustic overshoot ``linearUpwindV``
     produces at startup.

  2. ``energy sensibleInternalEnergy`` (field name ``e``) — not
     enthalpy.  Internal energy avoids the ∂p/∂t pressure-work source
     term in the energy equation; that source term is the dominant
     startup transient on steady compressible cases.
"""

from __future__ import annotations

from simd_agent.solvers.contexts import FvBuildContext
from simd_agent.solvers.rhoSimpleFoam.solver import RhoSimpleFoamSolver
from simd_agent.solvers.rhoPimpleFoam.solver import RhoPimpleFoamSolver


def _ctx(**overrides) -> FvBuildContext:
    defaults: dict = dict(
        tier="good", non_ortho=20.0, use_simplec=False, n_non_ortho=1,
        vel_mag=5.0, speed_tier="low", bc_temps=(280.0, 500.0),
        bc_pressures=(101325.0, 101325.0),
        profile="gas", heat_transfer_active=True, turb_model="kOmegaSST",
        mesh_quality={},
    )
    defaults.update(overrides)
    return FvBuildContext(**defaults)


# ── div(phi,U) — SIMPLE compressible is upwind unconditionally ─────────────


class TestDivPhiUForSimpleCompressible:
    def test_low_dp_low_speed_still_forces_upwind(self):
        """OF tutorial uses ``upwind`` even at quiescent conditions.

        Pre-fix: our generator emitted ``linearUpwindV grad(U)`` when
        Δp/p ≈ 1 and speed_tier was low/moderate — accuracy-preferred
        but unstable at startup.  Post-fix: always upwind for SIMPLE.
        """
        plugin = RhoSimpleFoamSolver()
        ctx = _ctx(speed_tier="low", bc_pressures=(101325.0, 1.2e5))
        out = plugin._build_div_block(ctx)
        assert "div(phi,U)      bounded Gauss upwind;" in out
        assert "linearUpwindV" not in out

    def test_moderate_speed_still_upwind(self):
        plugin = RhoSimpleFoamSolver()
        ctx = _ctx(speed_tier="moderate", bc_pressures=(101325.0, 1.2e5))
        out = plugin._build_div_block(ctx)
        assert "div(phi,U)      bounded Gauss upwind;" in out
        assert "linearUpwindV" not in out

    def test_high_dp_still_upwind(self):
        # The pressure-ratio guard is now redundant for SIMPLE — kept
        # alive in the PIMPLE branch for transient cases.
        plugin = RhoSimpleFoamSolver()
        ctx = _ctx(speed_tier="low", bc_pressures=(101325.0, 1.435e6))
        out = plugin._build_div_block(ctx)
        assert "div(phi,U)      bounded Gauss upwind;" in out


class TestDivPhiUForPimpleCompressibleStillUsesLinearUpwindV:
    """PIMPLE keeps the accuracy-preferred scheme — Δt absorbs the overshoot."""

    def test_low_speed_low_dp_uses_linearUpwindV(self):
        plugin = RhoPimpleFoamSolver()
        ctx = _ctx(speed_tier="low", bc_pressures=(101325.0, 1.2e5))
        out = plugin._build_div_block(ctx)
        assert "linearUpwindV grad(U)" in out

    def test_pimple_high_dp_falls_back_to_upwind(self):
        plugin = RhoPimpleFoamSolver()
        ctx = _ctx(speed_tier="low", bc_pressures=(101325.0, 1.435e6))
        out = plugin._build_div_block(ctx)
        assert "div(phi,U)      bounded Gauss upwind;" in out
        # linearUpwindV must not appear for the U scheme line
        u_line = next(
            ln for ln in out.splitlines()
            if ln.lstrip().startswith("div(phi,U)")
        )
        assert "linearUpwindV" not in u_line


# ── energy variable: rhoSimpleFoam uses ``e`` (sensibleInternalEnergy) ─────


class TestEnergyVarRendering:
    def test_rhoSimpleFoam_declares_e(self):
        plugin = RhoSimpleFoamSolver()
        assert plugin.energy_var == "e"

    def test_rhoPimpleFoam_still_declares_h(self):
        # rhoPimpleFoam keeps enthalpy — its transient loop handles
        # ∂p/∂t well, and enthalpy is the conventional rhoPimpleFoam
        # variable on most modern OpenFOAM forks.
        plugin = RhoPimpleFoamSolver()
        assert plugin.energy_var == "h"

    def test_div_block_emits_div_phi_e_for_rhoSimpleFoam(self):
        plugin = RhoSimpleFoamSolver()
        ctx = _ctx()
        out = plugin._build_div_block(ctx)
        assert "div(phi,e)" in out
        # And NOT div(phi,h) — they must not coexist.
        assert "div(phi,h)" not in out

    def test_div_block_emits_div_phi_h_for_rhoPimpleFoam(self):
        plugin = RhoPimpleFoamSolver()
        ctx = _ctx()
        out = plugin._build_div_block(ctx)
        assert "div(phi,h)" in out
        assert "div(phi,e)" not in out

    def test_equation_fields_includes_e_for_rhoSimpleFoam(self):
        plugin = RhoSimpleFoamSolver()
        fields = plugin._equation_fields("kOmegaSST")
        assert "e" in fields
        assert "h" not in fields

    def test_residualControl_uses_e_for_rhoSimpleFoam(self):
        plugin = RhoSimpleFoamSolver()
        ctx = _ctx()
        out = plugin._build_simple_block(ctx, plugin._equation_fields("kOmegaSST"), "")
        # residualControl block contains ``e`` (not ``h``).
        assert "e               1e-3;" in out
        assert "h               1e-3;" not in out


class TestKineticEnergyConvectionTerm:
    """``div(phi,Ekp)`` vs ``div(phi,K)`` depending on the energy form.

    sensibleEnthalpy (h) carries the pressure-work term inside h → only K
    is convected.  sensibleInternalEnergy (e) excludes p/ρ → Ekp = K + p/ρ
    is the convected term and the divScheme must be ``div(phi,Ekp)``.
    Missing this with ``default none`` is a fatal IO error.
    """

    def test_rhoSimpleFoam_with_e_emits_div_phi_Ekp(self):
        plugin = RhoSimpleFoamSolver()
        ctx = _ctx()
        out = plugin._build_div_block(ctx)
        assert "div(phi,Ekp)    bounded Gauss upwind;" in out
        # And NOT div(phi,K) — they are mutually exclusive.
        assert "div(phi,K)" not in out

    def test_rhoPimpleFoam_with_h_emits_div_phi_K(self):
        plugin = RhoPimpleFoamSolver()
        ctx = _ctx()
        out = plugin._build_div_block(ctx)
        assert "div(phi,K)      bounded Gauss upwind;" in out
        # And NOT div(phi,Ekp).
        assert "div(phi,Ekp)" not in out


# ── thermophysicalProperties auto-correction ───────────────────────────────


class TestEnergyFormFixer:
    def test_sensibleEnthalpy_is_rewritten_to_sensibleInternalEnergy(self):
        plugin = RhoSimpleFoamSolver()
        tp_path = "constant/thermophysicalProperties"
        thermo = """
thermoType
{
    type            hePsiThermo;
    mixture         pureMixture;
    transport       const;
    thermo          hConst;
    equationOfState perfectGas;
    specie          specie;
    energy          sensibleEnthalpy;
}
""".strip()
        files = {tp_path: thermo}
        issues: list = []
        out = plugin._fix_energy_form(files, issues)
        assert "sensibleInternalEnergy" in out[tp_path]
        assert "sensibleEnthalpy" not in out[tp_path]
        assert any(
            "sensibleInternalEnergy" in (i.message or "") for i in issues
        )

    def test_already_correct_thermo_is_noop(self):
        plugin = RhoSimpleFoamSolver()
        tp_path = "constant/thermophysicalProperties"
        thermo = "energy          sensibleInternalEnergy;"
        files = {tp_path: thermo}
        issues: list = []
        out = plugin._fix_energy_form(files, issues)
        assert out[tp_path] == thermo
        assert issues == []

    def test_missing_thermo_file_is_noop(self):
        plugin = RhoSimpleFoamSolver()
        files: dict[str, str] = {}
        issues: list = []
        out = plugin._fix_energy_form(files, issues)
        assert out == {}
        assert issues == []
