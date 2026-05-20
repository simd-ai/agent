# simd_agent/run/enrichment/turbulence_bcs.py
"""Propagate case-level turbulence values onto per-region inlet patches.

Mirror of :mod:`inlet_bcs` for the k / ω / ε / ν_t fields:

  * Source of truth — ``validated["turbulence"]``, populated by the
    linter from the user's Step-1 wizard inputs (model + intensity +
    length scale → pre-computed k/ω/ε via the standard correlations).
  * Target — every fluid region's inlet patches, keyed under
    ``boundary_conditions[<patch>][<field>]`` as the same
    ``{type, value}`` dict shape the LLM 0/* generator emits.

Skipped automatically when:
  * The case is laminar (``validated["turbulence"] == {}``).
  * No fluid regions are present (single-region cases — those flow
    turbulence values straight into ``0/k`` etc. via the prompt pack).
  * Every field on the case-level turbulence block is ``None`` (model
    selected but values not pre-computed yet).

Placeholder rule: a patch's existing turbulence field wins if it
carries a positive numeric value.  Missing field / zero / ``None``
counts as "no real signal" — overwrite-safe.
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

_STEP = "turbulence_bcs"

# Fields the step manages.  Listed in OpenFOAM-conventional order so the
# log output is stable + greppable.  Adding a new field (e.g. ``alphat``,
# ``nuTilda``) is one tuple entry — placeholder/apply logic generalises.
_FIELDS: tuple[str, ...] = ("k", "omega", "epsilon", "nut")


# ────────────────────────────────────────────────────────────────────────────
# Step entry point
# ────────────────────────────────────────────────────────────────────────────


async def apply(ctx: EnrichmentContext) -> None:
    config = ctx.config
    fluid_regions = (config.get("regions") or {}).get("fluid") or []
    if not fluid_regions:
        return

    values = _case_turbulence_values(config)
    if not values:
        return  # laminar, or no pre-computed turbulence values to share.

    patch_names = mesh_patch_names(config)
    if not patch_names:
        return

    bcs: dict[str, dict[str, Any]] = config.setdefault("boundary_conditions", {})

    logger.info(
        "[ENRICH:%s] propagating turbulence onto fluid inlets: %s",
        _STEP, values,
    )

    for region in fluid_regions:
        name = region.get("name")
        if not isinstance(name, str) or not name:
            continue
        for patch_name in inlet_patches_for_region(name, patch_names, bcs):
            patch_bc = bcs.setdefault(patch_name, {})
            for field, value in values.items():
                _maybe_apply(patch_bc, patch_name, field, value)


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _case_turbulence_values(config: dict[str, Any]) -> dict[str, float]:
    """Pull positive numeric k / ω / ε / ν_t from ``validated["turbulence"]``."""
    turb = config.get("turbulence")
    if not isinstance(turb, dict):
        return {}
    out: dict[str, float] = {}
    for field in _FIELDS:
        v = turb.get(field)
        if isinstance(v, (int, float)) and v > 0:
            out[field] = float(v)
    return out


def _maybe_apply(
    patch_bc: dict[str, Any],
    patch_name: str,
    field: str,
    value: float,
) -> None:
    """Write ``field`` if the patch doesn't already carry a real value."""
    if _is_existing_value(patch_bc.get(field)):
        return  # patch carries a user-set value — leave it alone
    patch_bc[field] = {"type": "fixedValue", "value": value}
    logger.info("[ENRICH:%s] %s: %s = %g", _STEP, patch_name, field, value)


def _is_existing_value(spec: Any) -> bool:
    """True if ``spec`` looks like a real, user-set turbulence value.

    Accepts either the structured ``{type, value}`` dict (the canonical
    BC shape) or a bare scalar (older configs).  Zero / missing /
    non-numeric are treated as "no signal" — overwrite-safe.
    """
    if isinstance(spec, dict):
        v = spec.get("value")
        return isinstance(v, (int, float)) and v > 0
    return isinstance(spec, (int, float)) and spec > 0
