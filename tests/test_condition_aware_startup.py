# tests/test_condition_aware_startup.py
"""Tests for the condition-aware startup fixers.

Three interlocking pieces (see the user's rhoPimpleFoam U-bend case
that crashed with ``deltaT = 1e-65`` for the motivation):

  1. ``FvBuildContext.has_impulsive_inlets`` + ``bulk_velocity`` — derived
     from ``boundary_conditions`` at _fv_context time.
  2. ``bc_fixers.fix_initial_velocity_field`` — seeds 0/U.internalField
     from U_bulk when impulsive inlets are present, so iteration 1
     doesn't have to accelerate the fluid from 0 → ~100 m/s in one Δt.
  3. ``legacy_fixers.fix_controldict_time_stepping`` — clamps maxCo /
     maxDeltaT / deltaT for impulsive cases.
  4. ``TransientBase._build_pimple_block`` — nOuterCorrectors + consistent
     are now derived from ΔT_BC, pressure_ratio, and has_impulsive_inlets.

Bug class that disappears: ``deltaT → 1e-65`` floating-point underflow
on mass-flow-inlet startups.
"""

from __future__ import annotations

from simd_agent.solvers import bc_fixers, legacy_fixers
from simd_agent.solvers.base import ValidationIssue
from simd_agent.solvers.contexts import FvBuildContext
from simd_agent.solvers.compressible.rhoPimpleFoam.solver import (
    RhoPimpleFoamSolver,
)


# ── FvBuildContext: impulsive detection + U_bulk ───────────────────────────


def _user_ubend_cfg() -> dict:
    """The exact case shape that crashed with deltaT=1e-65."""
    return {
        "physics": {
            "compressibility": "compressible", "heat_transfer": True,
            "time_scheme": "transient", "turbulence_model": "kOmegaSST",
        },
        "fluid": {"rho": 1.18, "mu": 1.81e-5, "Cp": 1006, "k": 0.026,
                  "temperature": 300},
        "boundary_conditions": {
            "inlet_main": {
                "patch_class": "inlet",
                "U": {"type": "flowRateInletVelocity", "massFlowRate": 0.012},
                "temperature": {"value": 500},
            },
            "inlet_small": {
                "patch_class": "inlet",
                "U": {"type": "flowRateInletVelocity", "massFlowRate": 0.001},
                "temperature": {"value": 280},
            },
            "outlet": {"patch_class": "outlet",
                       "pressure": {"value": 101325},
                       "temperature": {"value": 500}},
            "walls": {"patch_class": "wall", "temperature": {"value": 600}},
        },
        "mesh": {},
    }


class TestContextImpulsiveDetection:
    def test_impulsive_flag_set_on_user_ubend(self):
        ctx = RhoPimpleFoamSolver()._fv_context(_user_ubend_cfg())
        assert ctx.has_impulsive_inlets is True
        # U_bulk should be order ~95 m/s for this case
        # (0.012 kg/s at 280 K → ρ≈1.26 → U=0.012/(1.26·1e-4)≈95).
        assert 50.0 < ctx.bulk_velocity < 250.0

    def test_no_impulsive_on_velocity_BC(self):
        cfg = _user_ubend_cfg()
        cfg["boundary_conditions"]["inlet_main"]["U"] = {
            "type": "fixedValue", "value": [10.0, 0, 0],
        }
        cfg["boundary_conditions"]["inlet_small"]["U"] = {
            "type": "fixedValue", "value": [2.0, 0, 0],
        }
        ctx = RhoPimpleFoamSolver()._fv_context(cfg)
        assert ctx.has_impulsive_inlets is False
        # bulk_velocity still extracted from fixedValue magnitude.
        assert ctx.bulk_velocity >= 10.0


# ── 0/U.internalField seeding ──────────────────────────────────────────────


class TestVelocityICSeeding:
    def test_seeds_zero_initial_field(self):
        files = {
            "0/U": (
                "FoamFile { object U; }\n"
                "dimensions [0 1 -1 0 0 0 0];\n"
                "internalField   uniform (0 0 0);\n"
                "boundaryField { }\n"
            ),
        }
        issues: list[ValidationIssue] = []
        out = bc_fixers.fix_initial_velocity_field(
            files, issues,
            has_impulsive_inlets=True, bulk_velocity=95.0,
        )
        assert "uniform (0 0 0)" not in out["0/U"]
        # Seeded at 0.5 × U_bulk = 47.5 m/s.
        assert "uniform (47.5 0 0)" in out["0/U"]
        assert any("Seeded internalField" in i.message for i in issues)

    def test_skips_when_not_impulsive(self):
        files = {"0/U": "internalField uniform (0 0 0);\n"}
        issues: list[ValidationIssue] = []
        out = bc_fixers.fix_initial_velocity_field(
            files, issues,
            has_impulsive_inlets=False, bulk_velocity=95.0,
        )
        assert out["0/U"] == files["0/U"]
        assert issues == []

    def test_skips_when_field_already_seeded(self):
        """Don't overwrite a hand-tuned non-zero initial field."""
        files = {"0/U": "internalField uniform (10 5 0);\n"}
        issues: list[ValidationIssue] = []
        out = bc_fixers.fix_initial_velocity_field(
            files, issues,
            has_impulsive_inlets=True, bulk_velocity=95.0,
        )
        assert out["0/U"] == files["0/U"]
        assert issues == []

    def test_no_op_when_U_file_missing(self):
        out = bc_fixers.fix_initial_velocity_field(
            {}, [], has_impulsive_inlets=True, bulk_velocity=95.0,
        )
        assert out == {}


# ── controlDict time-stepping clamp ────────────────────────────────────────


class TestControlDictClamp:
    def _cd(self, maxCo: float, maxDeltaT: float, deltaT: float) -> str:
        return (
            "FoamFile { object controlDict; }\n"
            "application rhoPimpleFoam;\n"
            f"deltaT          {deltaT:g};\n"
            "endTime 0.1;\n"
            "adjustTimeStep yes;\n"
            f"maxCo           {maxCo:g};\n"
            f"maxDeltaT       {maxDeltaT:g};\n"
        )

    def test_clamps_aggressive_maxCo(self):
        files = {"system/controlDict": self._cd(2.0, 3.3e-3, 1e-4)}
        issues: list[ValidationIssue] = []
        out = legacy_fixers.fix_controldict_time_stepping(
            files, issues,
            is_transient=True, has_impulsive_inlets=True, bulk_velocity=95.0,
        )
        assert "maxCo           0.3" in out["system/controlDict"]
        assert "maxDeltaT       0.001" in out["system/controlDict"]
        # deltaT scaled down to 0.1·h/U with h=5e-3, U=95 → 5.26e-6
        assert "deltaT          5" in out["system/controlDict"]

    def test_skips_steady(self):
        files = {"system/controlDict": self._cd(2.0, 3.3e-3, 1e-4)}
        out = legacy_fixers.fix_controldict_time_stepping(
            files, [],
            is_transient=False, has_impulsive_inlets=True, bulk_velocity=95.0,
        )
        # Untouched.
        assert "maxCo           2" in out["system/controlDict"]

    def test_skips_when_not_impulsive(self):
        files = {"system/controlDict": self._cd(2.0, 3.3e-3, 1e-4)}
        out = legacy_fixers.fix_controldict_time_stepping(
            files, [],
            is_transient=True, has_impulsive_inlets=False, bulk_velocity=95.0,
        )
        assert "maxCo           2" in out["system/controlDict"]

    def test_never_loosens_already_tight_values(self):
        """If LLM already picked maxCo=0.1, don't bump it to 0.3."""
        files = {"system/controlDict": self._cd(0.1, 1e-4, 1e-6)}
        out = legacy_fixers.fix_controldict_time_stepping(
            files, [],
            is_transient=True, has_impulsive_inlets=True, bulk_velocity=95.0,
        )
        assert "maxCo           0.1" in out["system/controlDict"]
        assert "maxDeltaT       0.0001" in out["system/controlDict"]


# ── PIMPLE block: nOuterCorrectors + consistent driven by stiffness ────────


def _fake_ctx(**overrides) -> FvBuildContext:
    defaults: dict = dict(
        tier="good", non_ortho=20.0, use_simplec=False, n_non_ortho=1,
        vel_mag=5.0, speed_tier="low", bc_temps=(300.0, 320.0),
        bc_pressures=(101325.0,),
        has_impulsive_inlets=False, bulk_velocity=5.0,
        profile="gas", heat_transfer_active=True, turb_model="kOmegaSST",
        mesh_quality={},
    )
    defaults.update(overrides)
    return FvBuildContext(**defaults)


class TestPimpleBlockStiffnessAware:
    def test_smooth_case_keeps_2_outer_correctors_and_simplec(self):
        plugin = RhoPimpleFoamSolver()
        ctx = _fake_ctx(bc_temps=(300.0, 320.0), has_impulsive_inlets=False)
        out = plugin._build_pimple_block(ctx, ["U", "h", "k", "omega"], "")
        assert "nOuterCorrectors    2" in out
        assert "consistent          yes" in out

    def test_high_dt_bc_bumps_to_10_correctors(self):
        plugin = RhoPimpleFoamSolver()
        ctx = _fake_ctx(bc_temps=(280.0, 380.0))   # ΔT = 100 K
        out = plugin._build_pimple_block(ctx, ["U", "h"], "")
        assert "nOuterCorrectors    10" in out

    def test_user_ubend_gets_30_correctors_and_consistent_no(self):
        """User's case: ΔT_BC = 320 K, impulsive inlets → very stiff."""
        plugin = RhoPimpleFoamSolver()
        ctx = _fake_ctx(
            bc_temps=(280.0, 500.0, 600.0),
            has_impulsive_inlets=True,
            bulk_velocity=95.0,
        )
        out = plugin._build_pimple_block(ctx, ["U", "h", "k", "omega"], "")
        assert "nOuterCorrectors    30" in out
        assert "consistent          no" in out

    def test_high_pressure_ratio_alone_bumps_correctors(self):
        plugin = RhoPimpleFoamSolver()
        # ΔT small, but p_inlet:p_outlet ratio = 14:1 (compressor case).
        ctx = _fake_ctx(
            bc_temps=(300.0, 320.0),
            bc_pressures=(101325.0, 1.435e6),
        )
        out = plugin._build_pimple_block(ctx, ["U", "h"], "")
        assert "nOuterCorrectors    10" in out
        assert "consistent          no" in out


# ── End-to-end: validate() on the user's case produces all three fixes ────


class TestEndToEnd:
    def test_user_ubend_case_gets_all_three_fixes(self):
        plugin = RhoPimpleFoamSolver()
        cfg = _user_ubend_cfg()
        seed = {
            "0/U": (
                "FoamFile { class volVectorField; object U; }\n"
                "dimensions [0 1 -1 0 0 0 0];\n"
                "internalField   uniform (0 0 0);\n"
                "boundaryField { }\n"
            ),
            "system/controlDict": (
                "FoamFile { object controlDict; }\n"
                "application rhoPimpleFoam;\n"
                "deltaT 0.0001;\n"
                "endTime 0.1;\n"
                "adjustTimeStep yes;\n"
                "maxCo 2.0;\n"
                "maxDeltaT 0.0033;\n"
            ),
        }
        result = plugin.validate(seed, cfg)

        # 1. 0/U.internalField seeded.
        assert "uniform (0 0 0)" not in result.files["0/U"]
        assert "uniform (47" in result.files["0/U"]  # ≈ 0.5 × 95.

        # 2. controlDict tightened.
        cd = result.files["system/controlDict"]
        assert "maxCo           0.3" in cd
        assert "maxDeltaT       0.001" in cd

        # 3. PIMPLE block has nOuterCorrectors=30 + consistent=no.
        fvs = result.files["system/fvSolution"]
        assert "nOuterCorrectors    30" in fvs
        assert "consistent          no" in fvs
