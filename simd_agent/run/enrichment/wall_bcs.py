# simd_agent/run/enrichment/wall_bcs.py
"""Normalise user-set wall temperatures to the canonical BC shape.

Why this exists
---------------
The linter writes per-patch wall temperatures under the lowercase
``boundary_conditions[<patch>]["temperature"]`` key (mirroring the
``BoundaryConditionV1.temperature`` Pydantic field) but the multi-region
BC renderer in :mod:`simd_agent.solvers.families._multi_region_bcs`
reads the uppercase ``"T"`` key (OpenFOAM field-name convention).
Without bridging the two, a user-typed wall temperature is silently
dropped on CHT cases and the wall renders as adiabatic — the user's
input goes nowhere.

This step reads :data:`case_defaults["wall_temperatures"]` — already
resolved by :mod:`case_defaults`, with CHT-coupled interfaces
(``*_to_*``) deliberately excluded — and writes each entry into the
patch's canonical ``"T": {"type": "fixedValue", "value": <T>}`` BC.
Existing real values on the patch are never overwritten.

Solver-agnostic by design.  Runs for single-region cases too, where the
LLM 0/T generator + plugin validators benefit from the same canonical
shape.
"""

from __future__ import annotations

import logging
from typing import Any

from simd_agent.run.enrichment.context import EnrichmentContext

logger = logging.getLogger(__name__)

_STEP = "wall_bcs"


# ────────────────────────────────────────────────────────────────────────────
# Step entry point
# ────────────────────────────────────────────────────────────────────────────


async def apply(ctx: EnrichmentContext) -> None:
    config = ctx.config
    cd = config.get("case_defaults") or {}
    wall_temps = cd.get("wall_temperatures") or {}
    if not isinstance(wall_temps, dict) or not wall_temps:
        return  # No wall T resolved by case_defaults — nothing to write.

    bcs: dict[str, dict[str, Any]] = config.setdefault("boundary_conditions", {})

    logger.info(
        "[ENRICH:%s] propagating wall temperatures: %s",
        _STEP, wall_temps,
    )

    for patch_name, temp in wall_temps.items():
        if not isinstance(patch_name, str) or not patch_name:
            continue
        if not isinstance(temp, (int, float)) or temp <= 0:
            continue
        patch_bc = bcs.setdefault(patch_name, {})
        if _has_real_value(patch_bc.get("T")):
            continue  # User already set the canonical T BC; respect it.
        patch_bc["T"] = {"type": "fixedValue", "value": float(temp)}
        logger.info("[ENRICH:%s] %s: T = %.2f K", _STEP, patch_name, float(temp))


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────


def _has_real_value(spec: Any) -> bool:
    """True if ``spec`` already carries a real, user-set value.

    Accepts both the canonical ``{type, value}`` dict and the older
    bare-scalar shape that some configs still use.  Zero / missing /
    non-numeric counts as "no signal" — overwrite-safe.
    """
    if isinstance(spec, dict):
        v = spec.get("value")
        return isinstance(v, (int, float)) and v > 0
    return isinstance(spec, (int, float)) and spec > 0
