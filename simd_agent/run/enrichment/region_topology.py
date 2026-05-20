# simd_agent/run/enrichment/region_topology.py
"""Auto-detect multi-region topology from the mesh.

Thin wrapper around :func:`simd_agent.run.multi_region.detect_regions_from_mesh`
so it composes uniformly with the other enrichment steps and surfaces a
diagnostic when it produces a non-trivial result.

Single-region cases and cases where the precheck / UI already wrote a
``regions`` block are no-ops.
"""

from __future__ import annotations

import logging
from typing import Any

from simd_agent.run.enrichment._patch_lookup import (
    inlet_patches_for_region,
    mesh_patch_names,
)
from simd_agent.run.enrichment.context import EnrichmentContext
from simd_agent.run.multi_region import detect_regions_from_mesh

logger = logging.getLogger(__name__)

_STEP = "region_topology"


async def apply(ctx: EnrichmentContext) -> None:
    config = ctx.config
    if not config.get("regions"):
        # No regions block yet — try to auto-detect from the mesh.
        auto = detect_regions_from_mesh(config)
        if auto is None:
            return  # Single-region case — leave ``regions`` absent.
        config["regions"] = auto
        summary = {
            "fluids": [r["name"] for r in auto.get("fluid", [])],
            "solids": [r["name"] for r in auto.get("solid", [])],
        }
        logger.info("[ENRICH:%s] %s", _STEP, summary)
        ctx.add_info(
            _STEP,
            code="DETECTED",
            message=f"Auto-detected regions: {summary}",
            payload=summary,
        )

    # Stamp flow_kind on every fluid region.  Closed-cavity regions
    # (sealed bath, electronic-cooling enclosure, differentially-heated
    # cavity) have no inlet patch — their internalField is the only
    # source of the initial state and the inlet_bcs propagator must
    # skip them.
    _stamp_flow_kind(config)


def _stamp_flow_kind(config: dict[str, Any]) -> None:
    """Mark each fluid region as ``"through"`` or ``"closed"`` based on patches.

    Rule: if any patch matching ``<region>_*`` looks like an inlet
    (``patch_class == "inlet"`` or name ends with ``_inlet``), the
    region is ``"through"``.  Otherwise it's ``"closed"`` — a sealed
    cavity driven by walls + coupling alone.

    Idempotent.  Existing ``flow_kind`` values on regions are preserved
    so that an upstream UI / precheck override always wins.
    """
    fluid_regions = (config.get("regions") or {}).get("fluid") or []
    if not fluid_regions:
        return
    patch_names = mesh_patch_names(config)
    bcs = config.get("boundary_conditions") or {}
    stamped: list[tuple[str, str]] = []
    for region in fluid_regions:
        if region.get("flow_kind"):
            continue  # respect existing value
        name = region.get("name")
        if not isinstance(name, str) or not name:
            continue
        inlets = inlet_patches_for_region(name, patch_names, bcs)
        kind = "through" if inlets else "closed"
        region["flow_kind"] = kind
        stamped.append((name, kind))
    if stamped:
        logger.info("[ENRICH:%s] flow_kind: %s", _STEP, stamped)
