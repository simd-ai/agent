# tests/test_gamg_coarsest_cells.py
"""GAMG ``nCoarsestCells`` cap matches the OpenFOAM reference tutorial.

The reference ``compressible/rhoSimpleFoam/angledDuctExplicitFixedCoeff``
caps the coarsest level at 20 cells.  Our previous default of 500 was a
defensive choice for huge production meshes that paid a real
convergence-rate cost on every-day case sizes.

OF's library default is 10; 20 is a conservative middle that:
  * stays at OF-tutorial size for fast small-matrix coarse solves
  * leaves a safety margin against over-agglomeration on tet meshes
  * matches the value the OF maintainers have actually tested
"""

from __future__ import annotations

from simd_agent.run.case_spec.resolvers import resolve_pressure_solver_strategy
from simd_agent.run.case_spec.strategies import PressureSolverStrategy
from simd_agent.solvers.contexts import FvBuildContext
from simd_agent.solvers.compressible.rhoSimpleFoam.solver import RhoSimpleFoamSolver


def _ctx(**overrides) -> FvBuildContext:
    defaults: dict = dict(
        tier="good", non_ortho=20.0, use_simplec=False, n_non_ortho=1,
        vel_mag=5.0, speed_tier="low", bc_temps=(280.0, 500.0),
        bc_pressures=(101325.0, 1.435e6),
        profile="gas", heat_transfer_active=True, turb_model="kOmegaSST",
        mesh_quality={},
    )
    defaults.update(overrides)
    return FvBuildContext(**defaults)


class TestResolverDefault:
    def test_rhoSimpleFoam_default_is_20(self):
        s = resolve_pressure_solver_strategy(
            solver_name="rhoSimpleFoam",
            is_compressible=True,
            mesh_tier="good",
            heat_transfer_active=True,
        )
        assert s.n_coarsest_cells == 20

    def test_pydantic_default_is_20(self):
        """Default of the typed strategy itself."""
        from simd_agent.run.case_spec.strategies import CoarsestLevelCorr
        s = PressureSolverStrategy(
            top_level="GAMG",
            smoother_or_precond="GaussSeidel",
            coarsest=CoarsestLevelCorr(),
        )
        assert s.n_coarsest_cells == 20


class TestEndToEndRender:
    def test_rendered_fvSolution_uses_20(self):
        """End-to-end on a compressor-style config."""
        plugin = RhoSimpleFoamSolver()
        cfg = {
            "physics": {
                "compressibility": "compressible",
                "heat_transfer": True,
                "time_scheme": "steady",
                "turbulence_model": "kOmegaSST",
            },
            "fluid": {"rho": 1.18, "mu": 1.81e-5, "Cp": 1006, "k": 0.026},
            "boundary_conditions": {
                "inlet": {
                    "patch_class": "inlet",
                    "pressure": {"value": 1.435e6},
                    "temperature": {"value": 500},
                },
                "outlet": {
                    "patch_class": "outlet",
                    "pressure": {"value": 101325},
                    "temperature": {"value": 400},
                },
                "walls": {"patch_class": "wall", "temperature": {"value": 600}},
            },
            "mesh": {},
        }
        fvs = plugin.render_deterministic_files(cfg)["system/fvSolution"]
        # The p block must carry the 20-cell cap (and NOT the old 500).
        p_block = fvs.split("p\n")[1].split("rho\n")[0]
        assert "nCoarsestCells  20;" in p_block
        assert "nCoarsestCells  500;" not in p_block
