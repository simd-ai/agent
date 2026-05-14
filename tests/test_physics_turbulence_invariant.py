# tests/test_physics_turbulence_invariant.py
"""Tests for the flow_regime ↔ turbulence.model reconciler.

The reconciler lives in ``simd_agent/services/simulation_service.py`` and
runs at the save boundary (``save_form_state`` / ``upsert_config``).  It
guarantees that the two JSONB columns ``cfd_physics`` and
``cfd_turbulence`` can never drift apart in the DB.
"""

from simd_agent.services.simulation_service import _reconcile_physics_turbulence


class TestLaminarRegimeStripsRasModel:
    """flow_regime=laminar forces turbulence.model to be laminar (or absent)."""

    def test_strips_stale_kOmegaSST_when_regime_is_laminar(self):
        phys = {"flow_regime": "laminar"}
        turb = {"model": "kOmegaSST", "turbulence_intensity": 5, "k": 0.1, "omega": 100}

        p, t = _reconcile_physics_turbulence(phys, turb)

        # flow_regime unchanged (was already laminar).
        assert p == {"flow_regime": "laminar"}
        # Model forced to laminar; RAS-only derived fields wiped.
        assert t["model"] == "laminar"
        assert "k" not in t
        assert "omega" not in t
        # Numeric properties survive — keep the user's last-known values.
        assert t["turbulence_intensity"] == 5

    def test_strips_kEpsilon_when_regime_is_laminar(self):
        phys = {"flow_regime": "laminar"}
        turb = {"model": "kEpsilon", "epsilon": 0.01, "wall_functions": True}

        _, t = _reconcile_physics_turbulence(phys, turb)

        assert t["model"] == "laminar"
        assert "epsilon" not in t
        assert "wall_functions" not in t

    def test_already_laminar_is_a_no_op(self):
        phys = {"flow_regime": "laminar"}
        turb = {"model": "laminar", "turbulence_intensity": None}

        p, t = _reconcile_physics_turbulence(phys, turb)

        assert p == {"flow_regime": "laminar"}
        assert t == {"model": "laminar", "turbulence_intensity": None}

    def test_laminar_with_no_model_field_is_a_no_op(self):
        phys = {"flow_regime": "laminar"}
        turb = {"turbulence_intensity": 5}

        p, t = _reconcile_physics_turbulence(phys, turb)

        # No model field to strip — nothing changes.
        assert p == {"flow_regime": "laminar"}
        assert t == {"turbulence_intensity": 5}


class TestFlowRegimeWinsOverStaleModel:
    """flow_regime is authoritative; a stale RAS model is stripped to laminar.

    Rationale: a user toggling flow_regime in the UI is an *explicit*
    choice.  The Zustand store does not simultaneously reset
    cfdTurbulence.model, so without this guard the deliberate toggle
    would persist as {laminar, kOmegaSST}.  We treat flow_regime as the
    stronger signal and reconcile the model down.
    """

    def test_kOmegaSST_stripped_when_user_toggled_to_laminar(self):
        # The exact DB state from the U-bend case the user reported:
        # cfd_physics.flow_regime = laminar  AND  cfd_turbulence.model = kOmegaSST
        phys = {"flow_regime": "laminar"}
        turb = {"model": "kOmegaSST", "turbulence_intensity": 5}

        p, t = _reconcile_physics_turbulence(phys, turb)

        assert p == {"flow_regime": "laminar"}
        assert t["model"] == "laminar"
        # Numeric input survives — the user's last value reappears when
        # they flip back to turbulent.
        assert t["turbulence_intensity"] == 5

    def test_kEpsilon_stripped_when_regime_is_laminar(self):
        phys = {"flow_regime": "laminar", "time_scheme": "steady"}
        turb = {"model": "kEpsilon"}

        p, t = _reconcile_physics_turbulence(phys, turb)

        assert p == {"flow_regime": "laminar", "time_scheme": "steady"}
        assert t["model"] == "laminar"

    def test_spalart_stripped_when_regime_is_laminar(self):
        phys = {"flow_regime": "laminar"}
        turb = {"model": "spalartAllmaras"}

        p, t = _reconcile_physics_turbulence(phys, turb)

        assert p == {"flow_regime": "laminar"}
        assert t["model"] == "laminar"


class TestNoDrift:
    """Consistent inputs pass through unchanged."""

    def test_turbulent_with_kOmegaSST_unchanged(self):
        phys = {"flow_regime": "turbulent"}
        turb = {"model": "kOmegaSST", "k": 0.05, "omega": 50}

        p, t = _reconcile_physics_turbulence(phys, turb)

        assert p == {"flow_regime": "turbulent"}
        assert t == {"model": "kOmegaSST", "k": 0.05, "omega": 50}

    def test_laminar_with_laminar_unchanged(self):
        phys = {"flow_regime": "laminar"}
        turb = {"model": "laminar"}

        p, t = _reconcile_physics_turbulence(phys, turb)

        assert p == {"flow_regime": "laminar"}
        assert t == {"model": "laminar"}


class TestEdgeCases:
    """Corner cases — missing fields, None inputs, non-string regime."""

    def test_both_none_returns_none(self):
        p, t = _reconcile_physics_turbulence(None, None)
        assert p is None
        assert t is None

    def test_physics_only_no_turbulence(self):
        phys = {"flow_regime": "laminar"}
        p, t = _reconcile_physics_turbulence(phys, None)
        assert p == {"flow_regime": "laminar"}
        assert t is None

    def test_turbulence_only_no_physics(self):
        # Without flow_regime we can't enforce either side of the invariant.
        # The dicts pass through untouched.
        turb = {"model": "kOmegaSST"}
        p, t = _reconcile_physics_turbulence(None, turb)
        assert p is None
        assert t == {"model": "kOmegaSST"}

    def test_caller_dict_not_mutated(self):
        # The reconciler must not mutate the caller's payload.
        phys = {"flow_regime": "laminar"}
        turb = {"model": "kOmegaSST"}

        _reconcile_physics_turbulence(phys, turb)

        assert phys == {"flow_regime": "laminar"}
        assert turb == {"model": "kOmegaSST"}

    def test_case_insensitive_regime(self):
        phys = {"flow_regime": "LAMINAR"}
        turb = {"model": "kOmegaSST"}

        _, t = _reconcile_physics_turbulence(phys, turb)

        # Upper-cased "LAMINAR" still recognised — strip the RAS model.
        assert t["model"] == "laminar"

    def test_case_insensitive_model(self):
        # Already-laminar model in any case is a no-op.
        phys = {"flow_regime": "laminar"}
        turb = {"model": "LAMINAR"}

        p, t = _reconcile_physics_turbulence(phys, turb)

        assert p == {"flow_regime": "laminar"}
        # Already "laminar" (case-insensitive) — left as-is.
        assert t == {"model": "LAMINAR"}
