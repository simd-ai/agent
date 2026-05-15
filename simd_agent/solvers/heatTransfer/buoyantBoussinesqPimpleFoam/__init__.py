"""buoyantBoussinesqPimpleFoam — transient, incompressible Boussinesq."""

from simd_agent.solvers.heatTransfer.buoyantBoussinesqPimpleFoam.solver import (
    BuoyantBoussinesqPimpleFoamSolver,
)

solver_plugin = BuoyantBoussinesqPimpleFoamSolver()
Solver = BuoyantBoussinesqPimpleFoamSolver
