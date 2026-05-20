# simd_agent/run/enrichment/inlet_bcs.py
"""Propagate per-region ``T_init`` / ``U_init`` onto each region's inlet patches.

Why
---
The deterministic multi-region BC renderer in
:mod:`simd_agent.solvers.families._multi_region_bcs` reads each inlet's
``T`` and ``U`` from the patch's own entry in
``config["boundary_conditions"]`` first, falling back to
:class:`RegionSpec.T_init` / ``U_init`` only when the patch carries no
explicit value.  When the precheck pipeline stamped placeholder 300 K /
zero vectors on every inlet (its boundary planner is region-blind),
those placeholders win against the now-correct region inits unless we
overwrite them here.

This step is intentionally conservative: existing patch values that
look *non-default* are treated as "the user said so, leave them alone"
and never overwritten.
"""

from __future__ import annotations

import logging
from typing import Any

from simd_agent.run.enrichment._patch_lookup import (
    inlet_patches_for_region,
    mesh_patch_names,
)
from simd_agent.run.enrichment.context import EnrichmentContext

logger = logging.getLogger(__name__)

_STEP = "inlet_bcs"

# Values the precheck's region-blind planner stamps when it found
# nothing specific in the prompt — used to decide overwrite-safe vs
# leave-alone.  Anything else is treated as an intentional user input.
_PLACEHOLDER_T: float = 300.0
_PLACEHOLDER_U: tuple[float, float, float] = (0.0, 0.0, 0.0)


# ────────────────────────────────────────────────────────────────────────────
# Step entry point
# ────────────────────────────────────────────────────────────────────────────


async def apply(ctx: EnrichmentContext) -> None:
    config = ctx.config
    fluid_regions = (config.get("regions") or {}).get("fluid") or []
    if not fluid_regions:
        return

    patch_names = mesh_patch_names(config)
    if not patch_names:
        return

    bcs: dict[str, dict[str, Any]] = config.setdefault("boundary_conditions", {})

    logger.info(
        "[ENRICH:%s] propagating from regions: %s",
        _STEP,
        [
            (r.get("name"), r.get("T_init"), r.get("U_init"), r.get("p_init"))
            for r in fluid_regions
        ],
    )

    for region in fluid_regions:
        name = region.get("name")
        if not isinstance(name, str) or not name:
            continue
        # Closed-cavity regions (sealed bath, electronic enclosure, …)
        # have no inlet patch by construction.  Skip them explicitly so
        # we never try to write T/U on a non-existent inlet — and so
        # the log makes the closed shape visible.
        if region.get("flow_kind") == "closed":
            logger.info("[ENRICH:%s] %s: flow_kind=closed, skipping", _STEP, name)
            continue
        for patch_name in inlet_patches_for_region(name, patch_names, bcs):
            patch_bc = bcs.setdefault(patch_name, {})
            _maybe_apply_T(patch_bc, patch_name, region.get("T_init"))
            _maybe_apply_U(patch_bc, patch_name, region.get("U_init"))


# ────────────────────────────────────────────────────────────────────────────
# Per-field apply rules
# ────────────────────────────────────────────────────────────────────────────


def _maybe_apply_T(patch_bc: dict[str, Any], patch_name: str, T_init: Any) -> None:
    if not isinstance(T_init, (int, float)):
        return
    spec = patch_bc.get("T") or {}
    if not _is_placeholder_T(spec):
        return  # patch already carries an intentional, non-default T
    patch_bc["T"] = {"type": "fixedValue", "value": float(T_init)}
    logger.info("[ENRICH:%s] %s: T = %.2f K", _STEP, patch_name, float(T_init))


def _maybe_apply_U(patch_bc: dict[str, Any], patch_name: str, U_init: Any) -> None:
    if not _is_vec3(U_init):
        return
    vec = (float(U_init[0]), float(U_init[1]), float(U_init[2]))
    if vec == _PLACEHOLDER_U:
        return  # extractor produced no real U either — nothing to propagate
    spec = patch_bc.get("U") or {}
    if not _is_placeholder_U(spec):
        return
    patch_bc["U"] = {"type": "fixedValue", "value": list(vec)}
    logger.info("[ENRICH:%s] %s: U = (%g, %g, %g) m/s", _STEP, patch_name, *vec)


# ────────────────────────────────────────────────────────────────────────────
# Placeholder detection
# ────────────────────────────────────────────────────────────────────────────


def _is_placeholder_T(spec: dict[str, Any]) -> bool:
    if not spec:
        return True
    val = spec.get("value")
    if not isinstance(val, (int, float)):
        return True
    return float(val) == _PLACEHOLDER_T


def _is_placeholder_U(spec: dict[str, Any]) -> bool:
    if not spec:
        return True
    val = spec.get("value")
    if isinstance(val, (int, float)):
        return float(val) == 0.0
    if _is_vec3(val):
        return tuple(float(c) for c in val[:3]) == _PLACEHOLDER_U
    return True


def _is_vec3(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) >= 3
        and all(isinstance(v, (int, float)) for v in value[:3])
    )
