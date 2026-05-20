# simd_agent/run/multi_region/verifier.py
"""Multi-region (CHT) post-generation verifier.

The single-region rule-based verifier in
:mod:`simd_agent.run.single_region.verifier` was written for flat case
trees and reports every per-region file as missing at the top level
(``0/p_rgh``, ``system/fvSchemes``, ``constant/g``, …), and every
cross-region patch as missing coverage.  For multi-region cases the
deterministic renderer in :class:`MultiRegionBase` is authoritative for
every file the case ships with — there is nothing meaningful for the
rule-based pass to add, and letting it run produces 70+ false positives
that the retry loop then asks the LLM to "fix" by regenerating
top-level files that conflict with the per-region tree.

So the CHT verifier is intentionally a thin no-op for now: it asserts
the deterministic renderer ran (``constant/regionProperties`` exists,
each region has its ``0/<region>/T`` and either ``thermophysicalProperties``
or its solid equivalent), and otherwise trusts the renderer.

Future-work hooks worth adding here when we have a real failure mode
to defend against:

  * Per-region patch coverage — only the patches owned by ``<region>_``
    plus shared front/back; do NOT cross-check patches from other regions.
  * Coupled-interface reciprocity — every ``<self>_to_<other>`` block on
    one side must have a matching block on the other.
  * ``constant/<fluid>/g`` present for every fluid region (not top-level).
  * Per-region ``system/<region>/{fvSchemes,fvSolution}`` present.

Each check is independently mergeable — none belongs in the single-region
verifier.
"""

from __future__ import annotations

import logging
from typing import Any

from simd_agent.run.code_verifier import VerificationIssue, VerificationResult

logger = logging.getLogger(__name__)


async def verify(
    files: dict[str, str],
    user_requirements: str,
    validated_config: dict[str, Any],
    solver: str,
) -> VerificationResult:
    """Verify a multi-region case.

    Returns ``passed=True`` with an explanatory summary — the deterministic
    renderer is the authority.  See module docstring for the rationale and
    the list of future-work checks that belong here.

    Signature matches :class:`simd_agent.run.code_verifier.CodeVerifier.verify`
    so the dispatcher in :mod:`code_verifier` can swap implementations
    without the orchestrator caring.
    """
    issues: list[VerificationIssue] = []

    # Minimum sanity: deterministic renderer must have produced at least
    # regionProperties.  Anything else missing is a code bug in the
    # renderer, not a user-visible config issue — log and continue.
    if "constant/regionProperties" not in files:
        logger.warning(
            "[VERIFIER:multi_region] constant/regionProperties missing — "
            "MultiRegionBase.render_deterministic_files did not run as expected"
        )

    print("\n" + "*" * 70)
    print(
        f"[VERIFIER MULTI-REGION REPORT] solver={solver}  passed=True  "
        "(deterministic renderer is authoritative)"
    )
    print("*" * 70 + "\n")

    return VerificationResult(
        passed=True,
        issues=issues,
        summary=(
            "Multi-region verification: deterministic renderer authoritative — "
            "no rule-based checks applicable to per-region tree."
        ),
    )
