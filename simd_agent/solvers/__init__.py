# simd_agent/solvers — Solver plugin system
#
# Each solver lives in its own sub-package (e.g. solvers/simpleFoam/)
# and exposes a SolverPlugin subclass.  The registry discovers them
# automatically at startup — no core code changes needed to add a solver.

from simd_agent.solvers.base import SolverPlugin, MatchResult, ValidationResult, ValidationIssue
from simd_agent.solvers.registry import SolverRegistry, get_registry


def is_multi_region_solver(solver: str | None) -> bool:
    """Return True iff the named solver's plugin declares ``is_multi_region=True``.

    Central source of truth for "is this a CHT case?" used by every
    code path that needs to choose between the single-region and
    multi-region subsystems.  The plugin's ``is_multi_region`` class
    attribute is authoritative — adding a new CHT plugin to the
    registry automatically routes correctly without touching any caller.

    Returns False on any registry lookup failure (unknown solver name,
    registry not yet built, …) — the single-region path is the safe
    default for the current plugin roster.
    """
    if not solver:
        return False
    try:
        plugin = get_registry().get(solver)
        return bool(getattr(plugin, "is_multi_region", False))
    except Exception:
        return False


__all__ = [
    "SolverPlugin",
    "MatchResult",
    "ValidationResult",
    "ValidationIssue",
    "SolverRegistry",
    "get_registry",
    "is_multi_region_solver",
]
