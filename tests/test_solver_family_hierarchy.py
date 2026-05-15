# tests/test_solver_family_hierarchy.py
"""Lock the solver-family inheritance architecture.

Each solver inherits from a family base + optional mixins so that
paradigm-specific logic (PIMPLE Finals, SIMPLE relaxation, ρ block,
buoyancy gravity check) is **physically isolated** to the family or
mixin that owns it.  Bug class that disappears:

    "Edit to all-transient-solvers-need-Finals accidentally leaks into
     a steady solver via shared base.py code."

These tests pin the inheritance shape so a future refactor that
flattens the hierarchy is caught immediately.
"""

from __future__ import annotations

from simd_agent.solvers.base import SolverPlugin
from simd_agent.solvers.families import (
    BoussinesqMixin,
    CompressibleMixin,
    SteadyBase,
    TransientBase,
)
from simd_agent.solvers.simpleFoam.solver import SimpleFoamSolver
from simd_agent.solvers.pimpleFoam.solver import PimpleFoamSolver
from simd_agent.solvers.rhoSimpleFoam.solver import RhoSimpleFoamSolver
from simd_agent.solvers.rhoPimpleFoam.solver import RhoPimpleFoamSolver
from simd_agent.solvers.buoyantSimpleFoam.solver import BuoyantSimpleFoamSolver
from simd_agent.solvers.buoyantPimpleFoam.solver import BuoyantPimpleFoamSolver


# ── Each solver lands in the expected family ────────────────────────────────


class TestSteadyFamily:
    def test_simpleFoam_is_steady(self):
        assert issubclass(SimpleFoamSolver, SteadyBase)
        assert not issubclass(SimpleFoamSolver, TransientBase)

    def test_rhoSimpleFoam_is_steady_and_compressible(self):
        assert issubclass(RhoSimpleFoamSolver, SteadyBase)
        assert issubclass(RhoSimpleFoamSolver, CompressibleMixin)
        assert not issubclass(RhoSimpleFoamSolver, TransientBase)

    def test_buoyantSimpleFoam_is_steady_compressible_boussinesq(self):
        assert issubclass(BuoyantSimpleFoamSolver, SteadyBase)
        assert issubclass(BuoyantSimpleFoamSolver, CompressibleMixin)
        assert issubclass(BuoyantSimpleFoamSolver, BoussinesqMixin)
        assert not issubclass(BuoyantSimpleFoamSolver, TransientBase)


class TestTransientFamily:
    def test_pimpleFoam_is_transient(self):
        assert issubclass(PimpleFoamSolver, TransientBase)
        assert not issubclass(PimpleFoamSolver, SteadyBase)

    def test_rhoPimpleFoam_is_transient_and_compressible(self):
        assert issubclass(RhoPimpleFoamSolver, TransientBase)
        assert issubclass(RhoPimpleFoamSolver, CompressibleMixin)
        assert not issubclass(RhoPimpleFoamSolver, SteadyBase)

    def test_buoyantPimpleFoam_is_transient_compressible_boussinesq(self):
        assert issubclass(BuoyantPimpleFoamSolver, TransientBase)
        assert issubclass(BuoyantPimpleFoamSolver, CompressibleMixin)
        assert issubclass(BuoyantPimpleFoamSolver, BoussinesqMixin)
        assert not issubclass(BuoyantPimpleFoamSolver, SteadyBase)


# ── Methods land on the right class (not on universal base) ────────────────


class TestMethodOwnership:
    def test_simple_block_lives_on_SteadyBase(self):
        assert "_build_simple_block" in SteadyBase.__dict__
        assert "_build_simple_block" not in TransientBase.__dict__
        assert "_build_simple_block" not in SolverPlugin.__dict__

    def test_pimple_block_lives_on_TransientBase(self):
        assert "_build_pimple_block" in TransientBase.__dict__
        assert "_build_pimple_block" not in SteadyBase.__dict__
        assert "_build_pimple_block" not in SolverPlugin.__dict__

    def test_rho_solver_block_lives_on_CompressibleMixin(self):
        assert "_build_rho_solver_block" in CompressibleMixin.__dict__
        assert "_build_rho_solver_block" not in SolverPlugin.__dict__

    def test_compressible_bounds_lives_on_CompressibleMixin(self):
        assert "_build_compressible_bounds" in CompressibleMixin.__dict__
        assert "_build_compressible_bounds" not in SolverPlugin.__dict__

    def test_ensure_gravity_lives_on_BoussinesqMixin(self):
        assert "_ensure_gravity" in BoussinesqMixin.__dict__
        assert "_ensure_gravity" not in SolverPlugin.__dict__


# ── Behaviour: each plugin still exposes the methods it needs ───────────────


class TestPluginMethodAccess:
    """Verify the inheritance produces the right method dispatch."""

    def test_steady_solvers_have_simple_block(self):
        for plugin_cls in (SimpleFoamSolver, RhoSimpleFoamSolver, BuoyantSimpleFoamSolver):
            assert hasattr(plugin_cls(), "_build_simple_block"), plugin_cls

    def test_steady_solvers_do_NOT_have_pimple_block(self):
        for plugin_cls in (SimpleFoamSolver, RhoSimpleFoamSolver, BuoyantSimpleFoamSolver):
            assert not hasattr(plugin_cls(), "_build_pimple_block"), plugin_cls

    def test_transient_solvers_have_pimple_block(self):
        for plugin_cls in (PimpleFoamSolver, RhoPimpleFoamSolver, BuoyantPimpleFoamSolver):
            assert hasattr(plugin_cls(), "_build_pimple_block"), plugin_cls

    def test_transient_solvers_do_NOT_have_simple_block(self):
        for plugin_cls in (PimpleFoamSolver, RhoPimpleFoamSolver, BuoyantPimpleFoamSolver):
            assert not hasattr(plugin_cls(), "_build_simple_block"), plugin_cls

    def test_compressible_solvers_have_rho_block(self):
        for plugin_cls in (
            RhoSimpleFoamSolver, RhoPimpleFoamSolver,
            BuoyantSimpleFoamSolver, BuoyantPimpleFoamSolver,
        ):
            assert hasattr(plugin_cls(), "_build_rho_solver_block"), plugin_cls

    def test_incompressible_solvers_do_NOT_have_rho_block(self):
        for plugin_cls in (SimpleFoamSolver, PimpleFoamSolver):
            assert not hasattr(plugin_cls(), "_build_rho_solver_block"), plugin_cls

    def test_boussinesq_solvers_have_ensure_gravity(self):
        for plugin_cls in (BuoyantSimpleFoamSolver, BuoyantPimpleFoamSolver):
            assert hasattr(plugin_cls(), "_ensure_gravity"), plugin_cls

    def test_non_boussinesq_solvers_do_NOT_have_ensure_gravity(self):
        for plugin_cls in (SimpleFoamSolver, PimpleFoamSolver,
                           RhoSimpleFoamSolver, RhoPimpleFoamSolver):
            assert not hasattr(plugin_cls(), "_ensure_gravity"), plugin_cls
