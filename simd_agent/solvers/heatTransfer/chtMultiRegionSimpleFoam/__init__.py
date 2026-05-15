"""chtMultiRegionSimpleFoam — steady multi-region conjugate heat transfer.

**Phase 1 — architectural skeleton.**  The plugin registers with the
correct identity (multi-region, SIMPLE, compressible, p_rgh, gravity)
and emits the bare-minimum file tree (regionProperties +
per-region thermophysicalProperties).  Per-region fvSchemes /
fvSolution, mapped fluid-solid boundaries, and changeDictionaryDict
land in Phase 2.
"""

from simd_agent.solvers.heatTransfer.chtMultiRegionSimpleFoam.solver import (
    ChtMultiRegionSimpleFoamSolver,
)

solver_plugin = ChtMultiRegionSimpleFoamSolver()
Solver = ChtMultiRegionSimpleFoamSolver
