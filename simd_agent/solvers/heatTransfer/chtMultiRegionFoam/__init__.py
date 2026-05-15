"""chtMultiRegionFoam — transient multi-region conjugate heat transfer.

**Phase 1 — architectural skeleton.**  See the
``chtMultiRegionSimpleFoam`` package for the shared rationale.  This
plugin only differs in the algorithm (PIMPLE / transient) and matching
score for unsteady configs.
"""

from simd_agent.solvers.heatTransfer.chtMultiRegionFoam.solver import (
    ChtMultiRegionFoamSolver,
)

solver_plugin = ChtMultiRegionFoamSolver()
Solver = ChtMultiRegionFoamSolver
