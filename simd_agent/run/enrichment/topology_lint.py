# simd_agent/run/enrichment/topology_lint.py
"""Topology sanity check — dangling interfaces, orphan solids, missing inlets.

Thin wrapper around :func:`simd_agent.run.multi_region.lint_regions`
that re-emits its issues as :class:`EnrichmentIssue` objects so they
flow through the pipeline's standard diagnostic channel.

This is the only step that can produce ``error``-severity issues; the
pipeline halts as soon as the first one shows up.
"""

from __future__ import annotations

import logging
from typing import Any, cast

from simd_agent.run.enrichment.context import EnrichmentContext, Severity
from simd_agent.run.multi_region import lint_regions

logger = logging.getLogger(__name__)

_STEP = "topology_lint"


async def apply(ctx: EnrichmentContext) -> None:
    config = ctx.config
    if not config.get("regions"):
        return

    issues: list[dict[str, Any]] = lint_regions(config)
    for issue in issues:
        sev = cast(Severity, issue.get("severity", "warning"))
        code = str(issue.get("code", "UNKNOWN"))
        message = str(issue.get("message", ""))
        payload = issue.get("payload")
        logger.log(
            logging.ERROR if sev == "error" else logging.WARNING,
            "[ENRICH:%s] [%s] %s", _STEP, code, message,
        )
        ctx.add_issue(
            severity=sev,
            step=_STEP,
            code=code,
            message=message,
            payload=payload if isinstance(payload, dict) else None,
        )
