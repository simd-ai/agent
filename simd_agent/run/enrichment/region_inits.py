# simd_agent/run/enrichment/region_inits.py
"""Backfill missing per-region ``T_init`` / ``U_init`` / ``p_init``.

Priority order — strongest signal first
---------------------------------------

1. The :class:`RegionExtractor` LLM may have already filled these from
   the user's prompt.  Those values are never overwritten.

2. The region's own **inlet patch BC** in
   ``config["boundary_conditions"][<region>_inlet]`` — this is the
   per-region source of truth.  Mandatory for multi-region cases
   where each fluid region has its own inlet at a different state
   (e.g. innerFluid inlet at 77 K, outerFluid inlet at 290 K).

3. The **global** ``case_defaults`` block — used only as a last-resort
   fallback for single-region cases (where there's exactly one inlet
   and case_defaults was derived from it).  In multi-region cases
   this would seed *every* fluid region with the same value, which is
   exactly the bug this rewrite fixes — so it's now scoped strictly
   to single-region configs.

4. ``flow_kind = "closed"`` regions (sealed cavity, no inlet patch)
   get no backfill from inlet sources.  Their ``T_init`` / ``U_init``
   / ``p_init`` must come from upstream user input (precheck / wizard)
   or stay ``None``; the value_filler then uses role defaults.
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

_STEP = "region_inits"

# Values the precheck's region-blind planner stamps when it found
# nothing specific in the prompt.  Mirror of the same constants in
# inlet_bcs.py — kept local to keep the modules decoupled.
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
    bcs = config.get("boundary_conditions") or {}
    defaults = config.get("case_defaults") or {}

    # ── Pass 1: collect per-region inlet BC readings ─────────────
    # Read each region's inlet BC BEFORE deciding whether to apply the
    # global fallback.  This lets us distinguish two scenarios that look
    # similar from the per-region side but have very different right
    # answers:
    #
    #   (a) NO region has its own inlet BC values — the user supplied
    #       only a case-wide bulk temperature in the wizard.  Best
    #       guess: broadcast that one value to every region.
    #
    #   (b) AT LEAST ONE region has its own values — case_defaults was
    #       likely derived from THAT region's inlet and would be wrong
    #       for the others.  Don't broadcast; leave the silent regions
    #       at None so the LLM falls back to role defaults.
    per_region_reads: dict[str, tuple] = {}
    any_region_has_bc = False
    for region in fluid_regions:
        name = region.get("name")
        if not isinstance(name, str) or not name:
            continue
        T_bc, U_bc, p_bc = _read_region_inlet_bc(name, patch_names, bcs)
        per_region_reads[name] = (T_bc, U_bc, p_bc)
        if any(v is not None for v in (T_bc, U_bc, p_bc)):
            any_region_has_bc = True

    # ── Pass 2: pick global fallbacks (only safe when nobody has BCs) ──
    fallback_T = fallback_U = fallback_p = None
    if not any_region_has_bc:
        fallback_T = defaults.get("inlet_temperature") or defaults.get("bulk_temperature")
        fallback_U = defaults.get("inlet_velocity")
        fallback_p = defaults.get("inlet_pressure") or defaults.get("ambient_pressure")

    # ── Pass 3: apply ────────────────────────────────────────────
    changed: list[dict[str, Any]] = []
    for region in fluid_regions:
        name = region.get("name")
        if not isinstance(name, str) or not name:
            continue
        before = (region.get("T_init"), region.get("U_init"), region.get("p_init"))
        T_from_bc, U_from_bc, p_from_bc = per_region_reads.get(name, (None, None, None))

        if region.get("T_init") is None:
            if T_from_bc is not None:
                region["T_init"] = T_from_bc
            elif fallback_T is not None:
                region["T_init"] = fallback_T
        if region.get("U_init") is None:
            if U_from_bc is not None:
                region["U_init"] = U_from_bc
            elif fallback_U is not None:
                region["U_init"] = fallback_U
        if region.get("p_init") is None:
            if p_from_bc is not None:
                region["p_init"] = p_from_bc
            elif fallback_p is not None:
                region["p_init"] = fallback_p

        after = (region.get("T_init"), region.get("U_init"), region.get("p_init"))
        if before != after:
            changed.append({
                "region": name,
                "flow_kind": region.get("flow_kind"),
                "before": before,
                "after": after,
            })

    if changed:
        logger.info("[ENRICH:%s] backfilled %d region(s): %s", _STEP, len(changed), changed)
        ctx.add_info(
            _STEP,
            code="BACKFILLED",
            message=f"Backfilled {len(changed)} region(s)",
            payload={"changes": changed},
        )


# ────────────────────────────────────────────────────────────────────────────
# Per-region inlet BC reader
# ────────────────────────────────────────────────────────────────────────────


def _read_region_inlet_bc(
    region_name: str,
    patch_names: list[str],
    bcs: dict[str, Any],
) -> tuple[float | None, tuple[float, float, float] | None, float | None]:
    """Extract T / U / p from the region's inlet patch BCs.

    A multi-region case may have multiple inlet patches per region
    (rare but legal).  We pick the first one with a meaningful T (or
    U / p) — meaningful meaning "not the precheck's region-blind
    placeholder".  The precheck's placeholder values (T=300, U=0) are
    rejected so they don't pollute a region whose real state is
    still unknown.
    """
    T: float | None = None
    U: tuple[float, float, float] | None = None
    p: float | None = None
    for patch_name in inlet_patches_for_region(region_name, patch_names, bcs):
        patch_bc = bcs.get(patch_name) or {}
        if not isinstance(patch_bc, dict):
            continue
        if T is None:
            T = _real_T(patch_bc)
        if U is None:
            U = _real_U(patch_bc)
        if p is None:
            p = _real_p(patch_bc)
        if T is not None and U is not None and p is not None:
            break
    return T, U, p


def _real_T(patch_bc: dict[str, Any]) -> float | None:
    """Return the patch's T value if it's a meaningful, non-placeholder reading.

    Both the lowercase ``temperature`` (precheck / wizard shape) and
    uppercase ``T`` (renderer / filler shape) are accepted — the
    enrichment pipeline normalises these later, but at this point
    either may already be present.
    """
    for key in ("T", "temperature"):
        spec = patch_bc.get(key)
        if not isinstance(spec, dict):
            continue
        v = spec.get("value")
        if isinstance(v, (int, float)) and v > 0 and float(v) != _PLACEHOLDER_T:
            return float(v)
    return None


def _real_U(patch_bc: dict[str, Any]) -> tuple[float, float, float] | None:
    """Return the patch's U value if it's a meaningful, non-zero vector."""
    for key in ("U", "velocity"):
        spec = patch_bc.get(key)
        if not isinstance(spec, dict):
            continue
        v = spec.get("value")
        vec = _coerce_vec3(v)
        if vec is not None and vec != _PLACEHOLDER_U:
            return vec
    return None


def _real_p(patch_bc: dict[str, Any]) -> float | None:
    """Return the patch's p value if it's a meaningful, positive number."""
    for key in ("p", "pressure"):
        spec = patch_bc.get(key)
        if not isinstance(spec, dict):
            continue
        v = spec.get("value")
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    return None


def _coerce_vec3(value: Any) -> tuple[float, float, float] | None:
    """Coerce a scalar / list / tuple to a 3-component velocity vector."""
    if isinstance(value, (int, float)):
        return (float(value), 0.0, 0.0)
    if (
        isinstance(value, (list, tuple))
        and len(value) >= 3
        and all(isinstance(v, (int, float)) for v in value[:3])
    ):
        return (float(value[0]), float(value[1]), float(value[2]))
    return None
