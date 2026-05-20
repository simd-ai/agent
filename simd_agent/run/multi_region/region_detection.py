# simd_agent/run/multi_region/region_detection.py
"""Region detection helpers for multi-region (CHT) cases.

These helpers translate raw mesh metadata (cellZones + patches) into the
``config["regions"]`` shape that :class:`MultiRegionBase` consumes.  The
detection is intentionally *plumbing only* — no LLM calls, no validation;
it just answers "is this a CHT mesh, and if so what are the regions
called?" so the orchestrator can branch.

Two detection paths:

  1. :func:`detect_regions_from_cell_zones` — authoritative, uses the
     gmsh Physical Volume names extracted by :mod:`simd_agent.mesh.cell_zones`.
  2. :func:`detect_regions_from_patch_prefixes` — heuristic fallback for
     meshes whose importer didn't populate ``cell_zones``; groups patches
     by the substring before their first underscore.

Pair both with the single-shot dispatcher :func:`detect_regions_from_mesh`.

Lives under ``run/multi_region/`` so the single-region pipeline can be
imported / tested in isolation without dragging in CHT-specific code.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any


# ── Naming heuristics ────────────────────────────────────────────────────

# Tokens that mark a region-name as a solid material.  Matches both the
# mesh naming convention we use in the test generators (innerFluid_*,
# wall_*, outerFluid_*) and common industrial CHT cases (heater, shell,
# tube, casing, …).
_SOLID_REGION_HINTS: tuple[str, ...] = (
    "wall", "solid", "metal", "steel", "tube", "pipe",
    "shell", "heater", "casing", "housing",
)


def is_solid_name(name: str) -> bool:
    """Return True when a region-name hint looks like a solid material."""
    lo = name.lower()
    return any(hint in lo for hint in _SOLID_REGION_HINTS)


def fluid_preset_for(name: str) -> str:
    """Map a region name to a :class:`MultiRegionBase` fluid preset.

    Used by :func:`detect_regions_from_mesh` to seed sensible RegionSpec
    defaults from name keywords alone; the :class:`RegionExtractor` LLM
    pass overrides these with values from the user's prompt.  Defaults
    to ``"air"``.
    """
    lo = name.lower()
    if any(k in lo for k in ("ln2", "nitrogen")):
        return "ln2"
    if any(k in lo for k in ("lh2", "hydrogen")):
        return "lh2"
    if any(k in lo for k in ("lox", "oxygen")):
        return "lox"
    if any(k in lo for k in ("lng", "methane")):
        return "lng"
    # Liquid helium-4 (LHe, ~4 K) and gas helium (~293 K) have densities
    # three orders of magnitude apart — keep them as separate presets so
    # the SIMPLE rhoMin/rhoMax bounds in chtMultiRegionSimpleFoam target
    # the right operating point.
    if "lhe" in lo or "liquid helium" in lo or "liquidhelium" in lo:
        return "lhe"
    if "helium" in lo:
        return "helium"
    if "water" in lo or "h2o" in lo:
        return "water"
    if "oil" in lo:
        return "oil"
    return "air"


def solid_preset_for(name: str) -> str:
    """Map a region name to a :class:`MultiRegionBase` solid preset.

    Generic ``wall`` / ``pipe`` / ``tube`` prefixes default to **stainless**
    (316L, κ = 16.2 W/m·K) rather than mild carbon steel (κ = 80 W/m·K) —
    much more representative for the heat-exchanger / cryogenic use
    cases we ship.
    """
    lo = name.lower()
    if any(k in lo for k in ("steel", "stainless", "316", "wall", "pipe", "tube")):
        return "stainless"
    if "copper" in lo:
        return "copper"
    if any(k in lo for k in ("alum", "aluminium", "aluminum")):
        return "aluminum"
    if "concrete" in lo:
        return "concrete"
    if "glass" in lo:
        return "glass"
    return "steel"


# ── Region-tree assembly ────────────────────────────────────────────────

def build_region_tree(
    fluid_names: list[str], solid_names: list[str],
) -> dict[str, list[dict]] | None:
    """Assemble a ``config["regions"]`` dict from classified region names.

    Returns ``None`` when the topology isn't CHT-shaped (no fluids OR no
    solids) — the orchestrator falls through to the single-region path
    when this happens.  Interfaces are inferred as "every fluid touches
    every solid" (accurate for stacked / concentric geometries; complex
    topologies can override by populating ``config["regions"]`` directly).
    """
    if not fluid_names or not solid_names:
        return None
    return {
        "fluid": [
            {
                "name": n,
                "fluid_preset": fluid_preset_for(n),
                "interfaces": list(solid_names),
            }
            for n in fluid_names
        ],
        "solid": [
            {
                "name": n,
                "solid_preset": solid_preset_for(n),
                "interfaces": list(fluid_names),
            }
            for n in solid_names
        ],
    }


# ── Detection entry points ──────────────────────────────────────────────

def detect_regions_from_cell_zones(
    cell_zones: list[str],
) -> dict[str, list[dict]] | None:
    """Authoritative path: build the regions dict from mesh cellZones.

    cellZones are the gmsh ``Physical Volume`` groups (or ``cellZones``
    entries in an existing OpenFOAM polyMesh) populated by
    :mod:`simd_agent.mesh.cell_zones`.  When present they're the ground
    truth for region identity — independent of how the user named their
    boundary patches.
    """
    names = sorted({z for z in (cell_zones or []) if isinstance(z, str) and z})
    if len(names) < 2:
        return None
    fluids: list[str] = []
    solids: list[str] = []
    for n in names:
        (solids if is_solid_name(n) else fluids).append(n)
    return build_region_tree(fluids, solids)


def detect_regions_from_patch_prefixes(
    validated_config: dict[str, Any],
) -> dict[str, list[dict]] | None:
    """Heuristic fallback: group mesh patch names by their ``<prefix>_``.

    Used when ``mesh.cell_zones`` is unavailable (older meshes, or
    formats the importer can't introspect).  Patches whose mesh type is
    ``empty`` / ``wedge`` (front/back z faces in 2D cases) are skipped
    because they belong to every region.  Names without an underscore
    have no region prefix and are also skipped.
    """
    mesh = validated_config.get("mesh") or {}
    if not isinstance(mesh, dict):
        return None
    patches = mesh.get("patches") or []
    if not patches:
        return None

    by_prefix: dict[str, list[str]] = defaultdict(list)
    for p in patches:
        if isinstance(p, dict):
            name, ptype = p.get("name", ""), (p.get("type") or "").lower()
        else:
            name = getattr(p, "name", "") or ""
            ptype = (getattr(p, "type", "") or "").lower()
        if not name or "_" not in name:
            continue
        if ptype in ("empty", "wedge"):
            continue
        by_prefix[name.split("_", 1)[0]].append(name)

    if len(by_prefix) < 2:
        return None

    fluids: list[str] = []
    solids: list[str] = []
    for prefix in sorted(by_prefix):
        (solids if is_solid_name(prefix) else fluids).append(prefix)
    return build_region_tree(fluids, solids)


def detect_regions_from_mesh(
    validated_config: dict[str, Any],
) -> dict[str, list[dict]] | None:
    """Detect CHT regions from the validated mesh metadata.

    Authority order:

      1. ``mesh.cell_zones`` (or ``cellZones``) — gmsh Physical Volume
         names.  Authoritative: a region exists iff its cellZone exists
         in the mesh, independent of patch naming.
      2. Patch-name prefix heuristic.

    Returns a ``config["regions"]`` shaped dict, or ``None`` when the
    topology isn't CHT-shaped.  Single-region cases get ``None`` so the
    rest of the pipeline keeps its existing single-region behaviour
    untouched.
    """
    mesh = validated_config.get("mesh") or {}
    if isinstance(mesh, dict):
        # Accept both snake_case and camelCase from upstream payloads.
        zones = mesh.get("cell_zones") or mesh.get("cellZones") or []
        if isinstance(zones, list) and zones:
            from_zones = detect_regions_from_cell_zones(zones)
            if from_zones is not None:
                return from_zones
    return detect_regions_from_patch_prefixes(validated_config)


# ── Solver dispatch ─────────────────────────────────────────────────────

def force_cht_solver_if_multi_region(
    validated_config: dict[str, Any],
) -> str | None:
    """Return the canonical chtMultiRegion solver for a multi-region case.

    Multi-region (CHT) is detected by the presence of BOTH at least one
    fluid region AND at least one solid region in
    ``validated_config["regions"]``.  Single-fluid / single-solid /
    no-regions cases return ``None`` so the caller falls through to the
    LLM-driven :class:`SolverSelector`.

    For CHT cases the solver is uniquely determined by the time scheme:

      * steady    → ``chtMultiRegionSimpleFoam``
      * transient → ``chtMultiRegionFoam``

    No other registered solver can consume a multi-region case (they
    don't read ``constant/regionProperties``), so this is a hard
    physical constraint, not a heuristic.  Live physics edits work
    naturally — flipping the time scheme picks the matching variant
    on the next run with no LLM call needed.
    """
    regions = validated_config.get("regions") or {}
    if not isinstance(regions, dict):
        return None
    fluids = regions.get("fluid") or []
    solids = regions.get("solid") or []
    if not (isinstance(fluids, list) and isinstance(solids, list)):
        return None
    if not (fluids and solids):
        return None

    physics = validated_config.get("physics") or {}
    time_scheme = (
        validated_config.get("time_stepping")
        or validated_config.get("time_scheme")
        or (physics.get("time_scheme") if isinstance(physics, dict) else None)
        or "steady"
    )
    is_transient = str(time_scheme).lower() in ("transient", "unsteady")
    return "chtMultiRegionFoam" if is_transient else "chtMultiRegionSimpleFoam"
