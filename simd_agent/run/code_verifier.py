# simd_agent/run/code_verifier.py
"""Post-generation code verifier — thin dispatcher.

Routes to the single-region or multi-region verifier based on the
selected solver's ``is_multi_region`` plugin attribute.  The actual
checks live in dedicated subsystem modules:

  * :mod:`simd_agent.run.single_region.verifier` — flat case-tree
    rule-based checks (~270 lines of fast, deterministic invariants).
  * :mod:`simd_agent.run.multi_region.verifier` — CHT verifier (no-op
    today; the deterministic per-region renderer in
    :class:`MultiRegionBase` is authoritative).

Why a dispatcher and not just `if`-branches in the call site?  The
verifier is exposed as a single public entry point — :class:`CodeVerifier` —
that the orchestrator wires through its self-healing loop.  Keeping the
selection logic here means new solver families (multi-fluid VOF,
overset, particle-laden flows) can each ship their own verifier module
without the orchestrator caring.

The data models :class:`VerificationIssue` / :class:`VerificationResult`
live here because both subsystem modules import them, and pulling them
into a third "types" module would be churn for two consumers.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Data models ───────────────────────────────────────────────────────────

class VerificationIssue(BaseModel):
    severity: Literal["critical", "warning", "info"]
    category: str = Field(
        description=(
            "Short category tag, e.g. 'solver_mismatch', 'missing_field', "
            "'bc_inconsistency', 'turbulence_mismatch', "
            "'heat_transfer_solver', 'patch_coverage'"
        )
    )
    message: str
    fix_suggestion: str | None = None


class VerificationResult(BaseModel):
    passed: bool = Field(
        description="True when there are no critical issues — safe to proceed."
    )
    issues: list[VerificationIssue] = Field(default_factory=list)
    summary: str


# ── Dispatcher ────────────────────────────────────────────────────────────

class CodeVerifier:
    """Dispatch between single-region and multi-region verifiers.

    The dispatcher consults the solver plugin's ``is_multi_region`` flag
    (single source of truth — adding a new CHT plugin to the registry
    automatically routes here without touching this module).  On any
    registry lookup failure the dispatcher falls back to the
    single-region path, which is the safe default for the current
    plugin roster.
    """

    def __init__(self) -> None:
        pass  # Verifiers themselves hold no state.

    async def verify(
        self,
        files: dict[str, str],
        user_requirements: str,
        validated_config: dict[str, Any],
        solver: str,
    ) -> VerificationResult:
        # Lazy imports — the subsystem modules import VerificationIssue /
        # VerificationResult from this file, so we can't top-import them
        # without creating a cycle.
        if self._is_multi_region_solver(solver):
            from simd_agent.run.multi_region.verifier import verify as _mr_verify
            return await _mr_verify(files, user_requirements, validated_config, solver)
        from simd_agent.run.single_region.verifier import verify as _sr_verify
        return await _sr_verify(files, user_requirements, validated_config, solver)

    @staticmethod
    def _is_multi_region_solver(solver: str) -> bool:
        """Return True when the named solver plugin sets ``is_multi_region``."""
        try:
            from simd_agent.solvers import get_registry
            plugin = get_registry().get(solver)
            return bool(getattr(plugin, "is_multi_region", False))
        except Exception:
            return False
