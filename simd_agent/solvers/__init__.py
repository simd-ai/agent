# simd_agent/solvers — Solver plugin system
#
# Each solver lives in its own sub-package (e.g. solvers/simpleFoam/)
# and exposes a SolverPlugin subclass.  The registry discovers them
# automatically at startup — no core code changes needed to add a solver.

from simd_agent.solvers.base import SolverPlugin, MatchResult, ValidationResult, ValidationIssue
from simd_agent.solvers.registry import SolverRegistry, get_registry

__all__ = [
    "SolverPlugin",
    "MatchResult",
    "ValidationResult",
    "ValidationIssue",
    "SolverRegistry",
    "get_registry",
]
