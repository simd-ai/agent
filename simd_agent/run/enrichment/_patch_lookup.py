# simd_agent/run/enrichment/_patch_lookup.py
"""Shared mesh-patch lookup helpers used by per-field BC propagators.

Both :mod:`inlet_bcs` and :mod:`turbulence_bcs` need the same "give me
the inlet patches owned by this region" logic — and any future
propagator (wall BCs, outlet BCs, …) will too.  Centralising it here
keeps each propagator focused on what it writes, not on how to find
the patches.

Naming convention: ``_`` prefix on the module signals package-private.
Callers within :mod:`simd_agent.run.enrichment` import directly; nothing
outside the package should depend on this module.
"""

from __future__ import annotations

from typing import Any


def mesh_patch_names(config: dict[str, Any]) -> list[str]:
    """All patch names from ``config["mesh"]["patches"]`` as strings."""
    raw = (config.get("mesh") or {}).get("patches") or []
    out: list[str] = []
    for p in raw:
        name = p["name"] if isinstance(p, dict) else getattr(p, "name", None)
        if isinstance(name, str) and name:
            out.append(name)
    return out


def inlet_patches_for_region(
    region_name: str,
    patch_names: list[str],
    bcs: dict[str, dict[str, Any]],
) -> list[str]:
    """Patches owned by ``region_name`` (by ``<name>_`` prefix) that look inlet-ish.

    Both the explicit ``patch_class == "inlet"`` and the ``_inlet`` name
    suffix are accepted — the precheck pipeline writes the role under
    different keys depending on which path produced the patch.
    """
    prefix = f"{region_name}_"
    return [
        p for p in patch_names
        if p.startswith(prefix) and is_inlet(p, bcs.get(p) or {})
    ]


def is_inlet(patch_name: str, patch_bc: dict[str, Any]) -> bool:
    """True if the patch is an inlet via stored role or ``_inlet`` suffix."""
    role = (
        patch_bc.get("patch_class")
        or patch_bc.get("patchClass")
        or patch_bc.get("patch_type")
    )
    if isinstance(role, str) and role.lower() == "inlet":
        return True
    return patch_name.endswith("_inlet")
