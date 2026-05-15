"""buoyantBoussinesqSimpleFoam — steady, incompressible Boussinesq."""

from simd_agent.solvers.heatTransfer.buoyantBoussinesqSimpleFoam.solver import (
    BuoyantBoussinesqSimpleFoamSolver,
)

solver_plugin = BuoyantBoussinesqSimpleFoamSolver()
Solver = BuoyantBoussinesqSimpleFoamSolver
