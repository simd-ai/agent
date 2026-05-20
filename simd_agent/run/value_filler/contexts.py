# simd_agent/run/value_filler/contexts.py
"""Per-file context builders for the value filler.

Each builder produces a context dict that the prompt builder consumes.
There are two kinds today:

  * Single-region — target files at ``0/<T|U|p|p_rgh>``.  Patches come
    straight from ``config["mesh"]["patches"]`` with their roles
    resolved by :func:`patch_role`.  No region prefix.
  * Multi-region — target files at ``0/<region>/<T|U|p|p_rgh>``.
    Patches are restricted to those the region owns (``<region>_*``).

Adding a third kind in the future (per-cellZone source terms,
per-functionObject probes, …) is a matter of dropping a new builder
function in here and registering it in :func:`build_for_path`.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from simd_agent.solvers.families._multi_region_bcs import patch_role


# Fields the filler targets.  Restricting both regexes to this set
# keeps ``0/system`` / ``0/constant`` / region cellZones from ever
# being misclassified as a filler target.
_TARGET_FIELDS: tuple[str, ...] = ("T", "U", "p", "p_rgh")
_FIELD_ALT: str = "|".join(re.escape(f) for f in _TARGET_FIELDS)

_SINGLE_REGION_RE = re.compile(rf"^0/(?P<field>{_FIELD_ALT})$")
_MULTI_REGION_RE  = re.compile(
    rf"^0/(?P<region>[^/]+)/(?P<field>{_FIELD_ALT})$"
)


# ────────────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────────────


def build_for_path(path: str, config: dict[str, Any]) -> dict[str, Any] | None:
    """Return the per-file context for ``path``, or ``None`` if not a target.

    Auto-routes by path shape — see module docstring.  Returns ``None``
    for any non-target file (``constant/*``, ``system/*``,
    ``0/<unknown_field>``, …) so the caller can skip them.
    """
    m = _MULTI_REGION_RE.match(path)
    if m:
        return _build_multi_region(m.group("region"), m.group("field"), config)
    m = _SINGLE_REGION_RE.match(path)
    if m:
        return _build_single_region(m.group("field"), config)
    return None


# ────────────────────────────────────────────────────────────────────────────
# Single-region context
# ────────────────────────────────────────────────────────────────────────────


def _build_single_region(field: str, config: dict[str, Any]) -> dict[str, Any]:
    patch_names = _mesh_patch_names(config)
    fluid = config.get("fluid") or {}
    return {
        "mode":           "single",
        "field":          field,
        "patches":        _patches_with_values(patch_names, field, config),
        "case_defaults":  config.get("case_defaults") or {},
        "fluid_name":     (fluid.get("name") or "").strip() or None,
    }


# ────────────────────────────────────────────────────────────────────────────
# Multi-region context
# ────────────────────────────────────────────────────────────────────────────


def _build_multi_region(
    region_name: str, field: str, config: dict[str, Any],
) -> dict[str, Any] | None:
    """Return per-region context, or ``None`` if the region isn't defined.

    Returning ``None`` (rather than an empty dict) lets the caller
    distinguish "config doesn't describe this region" from "context is
    valid but every value is None" — the filler skips the former
    entirely and silently keeps the deterministic template.
    """
    regions = (config.get("regions") or {})
    region = _find_region(regions, region_name)
    if region is None:
        return None

    patch_names = _mesh_patch_names(config)
    region_patch_names = [p for p in patch_names if p.startswith(f"{region_name}_")]

    return {
        "mode":          "multi",
        "field":         field,
        "name":          region_name,
        "kind":          region["kind"],
        "flow_kind":     region.get("flow_kind"),  # "through" | "closed" | None
        "fluid_preset":  region.get("fluid_preset"),
        "solid_preset":  region.get("solid_preset"),
        "T_init":        region.get("T_init"),
        "U_init":        region.get("U_init"),
        "p_init":        region.get("p_init"),
        "patches":       _patches_with_values(region_patch_names, field, config),
        "interfaces":    list(region.get("interfaces") or ()),
        "case_defaults": config.get("case_defaults") or {},
    }


def _find_region(
    regions: dict[str, Any], name: str,
) -> dict[str, Any] | None:
    """Locate ``name`` in either the fluid or solid list, returning it
    with a ``kind`` key stamped on."""
    for kind, lst in (("fluid", regions.get("fluid") or []),
                      ("solid", regions.get("solid") or [])):
        for r in lst:
            if r.get("name") == name:
                return {**r, "kind": kind}
    return None


# ────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ────────────────────────────────────────────────────────────────────────────


def _mesh_patch_names(config: dict[str, Any]) -> list[str]:
    """All patch names from ``config["mesh"]["patches"]`` as strings."""
    raw = (config.get("mesh") or {}).get("patches") or []
    out: list[str] = []
    for p in raw:
        n = p["name"] if isinstance(p, dict) else getattr(p, "name", None)
        if isinstance(n, str) and n:
            out.append(n)
    return out


def _patches_with_values(
    patch_names: Iterable[str],
    field: str,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Patches with their roles + any explicit per-field BC values.

    Surfacing the per-patch values from ``config["boundary_conditions"]``
    means the prompt's "AUTHORITATIVE PER-PATCH VALUES" table carries
    the values without the LLM having to re-derive them from a
    possibly-summarised user prompt.
    """
    bcs = (config.get("boundary_conditions") or {})
    field_key = "p_rgh" if field == "p_rgh" else field

    out: list[dict[str, Any]] = []
    for name in patch_names:
        entry: dict[str, Any] = {"name": name, "role": patch_role(name, config)}
        patch_bc = bcs.get(name) or {}
        if isinstance(patch_bc, dict):
            spec = patch_bc.get(field_key)
            if isinstance(spec, dict):
                val = spec.get("value")
                if val is not None:
                    entry[f"{field_key}_value"] = val
                    entry[f"{field_key}_type"]  = spec.get("type")
        out.append(entry)
    return out
