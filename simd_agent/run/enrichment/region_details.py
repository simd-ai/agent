# simd_agent/run/enrichment/region_details.py
"""LLM-driven per-region detail extraction.

Wraps :class:`simd_agent.run.multi_region.RegionExtractor` so that
its (LLM-bound) refinement of per-region presets and inlet conditions
participates in the enrichment pipeline like every other step.

Failure mode
------------
The extractor is the only LLM step in the enrichment pipeline.  Its
network call can fail for many uninteresting reasons (rate limits,
transient DNS, model overload).  Failures are non-fatal here — we
emit a ``warning`` issue and return without mutating the config.
Downstream steps (``region_inits``, ``inlet_bcs``) then operate on
whatever defaults the topology-detection step already produced.
"""

from __future__ import annotations

import logging
from typing import Any

from simd_agent.run.enrichment.context import EnrichmentContext
from simd_agent.run.multi_region import RegionExtractor

logger = logging.getLogger(__name__)

_STEP = "region_details"


async def apply(ctx: EnrichmentContext) -> None:
    config = ctx.config
    if not config.get("regions"):
        return  # Single-region cases skip the extractor entirely.

    flat = _flatten_regions(config["regions"])
    if len(flat) < 2:
        # The extractor requires ≥ 2 regions (it's CHT-shaped); skip
        # the LLM call for trivial topologies.
        return

    try:
        refined = await RegionExtractor().extract(ctx.user_requirements, flat)
    except Exception as exc:
        # Match the orchestrator's original "warn and continue" policy:
        # the case is still runnable with the heuristic defaults; we
        # just lose the prompt-derived refinements.
        ctx.add_warning(
            _STEP,
            code="EXTRACTOR_FAILED",
            message=f"{type(exc).__name__}: {exc}",
        )
        logger.warning("[ENRICH:%s] extractor raised: %s", _STEP, exc)
        return

    config["regions"] = _split_by_kind(refined)

    fluids_summary = [
        (r["name"], r.get("fluid_preset"), r.get("T_init"), r.get("U_init"))
        for r in config["regions"]["fluid"]
    ]
    solids_summary = [
        (r["name"], r.get("solid_preset")) for r in config["regions"]["solid"]
    ]
    logger.info(
        "[ENRICH:%s] fluids=%s solids=%s",
        _STEP, fluids_summary, solids_summary,
    )
    ctx.add_info(
        _STEP,
        code="REFINED",
        message=f"fluids={fluids_summary} solids={solids_summary}",
    )


# ────────────────────────────────────────────────────────────────────────────
# Shape helpers
# ────────────────────────────────────────────────────────────────────────────


def _flatten_regions(regions: dict[str, Any]) -> list[dict[str, Any]]:
    """Merge fluid+solid into a single list, stamping ``kind`` on each.

    The extractor takes a flat list (it doesn't care about the
    fluid/solid grouping at the dict level — its tool call schema
    asks for ``kind`` per region).
    """
    flat: list[dict[str, Any]] = []
    for r in regions.get("fluid", []) or []:
        flat.append({**r, "kind": "fluid"})
    for r in regions.get("solid", []) or []:
        flat.append({**r, "kind": "solid"})
    return flat


def _split_by_kind(flat: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Inverse of :func:`_flatten_regions`."""
    return {
        "fluid": [r for r in flat if r.get("kind") == "fluid"],
        "solid": [r for r in flat if r.get("kind") == "solid"],
    }
