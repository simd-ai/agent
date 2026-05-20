# simd_agent/run/region_lint.py
"""Consistency lint for multi-region (CHT) configs.

Companion to :mod:`simd_agent.run.linting` (which validates fluid/solver/
turbulence/mesh consistency for single-region cases) — this module owns
the region-shaped invariants only.  Runs in the orchestrator after region
detection (:func:`_detect_regions_from_mesh`) and per-region extraction
(:class:`RegionExtractor`), so it sees the fully-populated
``config["regions"]`` rather than the raw normalised config.

The check we care about for "make CHT correct for any user case":

  * ``cellZones`` ⇆ ``regions`` parity — every cellZone in the mesh has
    a matching region; every region has a backing cellZone (when the
    mesh importer populated the list).
  * Every fluid region has at least one inlet patch (warn for natural-
    convection cases that legitimately have none).
  * Every solid region has at least one fluid interface (otherwise CHT
    has no heat-transfer partner — error).
  * Every interface is reciprocal: ``A.interfaces ∋ B  ⇒  B.interfaces ∋ A``.
  * Every region has ≥ 1 boundary patch (a region with zero patches
    after splitMeshRegions is unreachable).

Free functions, no class — same style as :mod:`bc_fixers` and
:mod:`region_extractor`.  Issues are returned as plain dicts so they can
travel through the event bus without coupling to a specific Pydantic model.
"""

from __future__ import annotations

from typing import Any, Literal


RegionIssueSeverity = Literal["error", "warning", "info"]


def _issue(
    severity: RegionIssueSeverity, code: str, message: str, **payload: Any,
) -> dict[str, Any]:
    """Build a single region-lint issue dict."""
    out: dict[str, Any] = {
        "severity": severity,
        "code": code,
        "message": message,
    }
    if payload:
        out["payload"] = payload
    return out


def _region_patches(region_name: str, patches: list[dict[str, Any]]) -> list[str]:
    """Patches owned by a region — prefix match on ``<region>_``.

    Mirrors :func:`_multi_region_bcs.region_patches` but takes a raw patch
    list rather than a full config dict, so this module stays decoupled
    from the renderer.
    """
    prefix = f"{region_name}_"
    out: list[str] = []
    for p in patches:
        name = (
            p.get("name") if isinstance(p, dict)
            else getattr(p, "name", None)
        )
        if isinstance(name, str) and name.startswith(prefix):
            out.append(name)
    return out


def _patch_role(patch_name: str, bcs: dict[str, Any]) -> str:
    """Return the BC role for one patch (``inlet`` / ``outlet`` / ``wall`` / …).

    Mirrors the role-inference in :mod:`_multi_region_bcs.patch_role` but
    only needs the BC-side data: lint doesn't try to read mesh patch
    types here (those are handled by the universal constraint-BC fix).
    """
    spec = bcs.get(patch_name)
    if isinstance(spec, dict):
        pc = (
            spec.get("patchClass")
            or spec.get("patch_class")
            or spec.get("patch_type")
        )
        if isinstance(pc, str) and pc:
            return pc.lower()
    low = patch_name.lower()
    for suffix, role in (
        ("_inlet", "inlet"),
        ("_outlet", "outlet"),
        ("_symmetry", "symmetry"),
    ):
        if low.endswith(suffix):
            return role
    return "wall"


def lint_regions(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Run every region-shaped consistency check and return a flat issue list.

    Empty list when no issues are found — including the trivial case
    where ``config["regions"]`` is absent (single-region case).  Callers
    should branch on the severity field; the orchestrator emits warnings
    via the event bus and aborts on errors.
    """
    regions = config.get("regions") or {}
    fluids: list[dict[str, Any]] = list(regions.get("fluid") or [])
    solids: list[dict[str, Any]] = list(regions.get("solid") or [])
    if not fluids and not solids:
        return []  # single-region — not our concern

    mesh = config.get("mesh") or {}
    patches: list[dict[str, Any]] = (
        list(mesh.get("patches") or []) if isinstance(mesh, dict) else []
    )
    cell_zones: list[str] = []
    if isinstance(mesh, dict):
        cell_zones = list(mesh.get("cell_zones") or mesh.get("cellZones") or [])
    bcs: dict[str, Any] = config.get("boundary_conditions") or {}

    issues: list[dict[str, Any]] = []
    issues += _check_cellzones_match_regions(cell_zones, fluids, solids)
    issues += _check_each_region_has_patches(fluids + solids, patches)
    issues += _check_each_fluid_has_inlet(fluids, patches, bcs)
    issues += _check_each_solid_has_interface(solids, fluids)
    issues += _check_interfaces_reciprocal(fluids + solids)
    return issues


# ── Individual checks ─────────────────────────────────────────────────────


def _check_cellzones_match_regions(
    cell_zones: list[str],
    fluids: list[dict[str, Any]],
    solids: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """cellZones from the mesh must align with the regions dict.

    Only fires when the mesh importer populated cellZones — if the field
    is empty we have no ground truth and the check is skipped.
    """
    if not cell_zones:
        return []
    region_names = {r.get("name") for r in fluids + solids if r.get("name")}
    zone_names = {z for z in cell_zones if isinstance(z, str) and z}
    issues: list[dict[str, Any]] = []
    missing_regions = zone_names - region_names
    extra_regions = region_names - zone_names
    if missing_regions:
        issues.append(_issue(
            "error", "cellzone_without_region",
            f"Mesh has cellZones with no matching region in config: "
            f"{sorted(missing_regions)}",
            zones=sorted(missing_regions),
        ))
    if extra_regions:
        issues.append(_issue(
            "warning", "region_without_cellzone",
            f"config['regions'] references regions absent from the mesh "
            f"cellZones: {sorted(extra_regions)}",
            regions=sorted(extra_regions),
        ))
    return issues


def _check_each_region_has_patches(
    regions: list[dict[str, Any]],
    patches: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """A region with zero owned patches is unreachable post-splitMeshRegions."""
    issues: list[dict[str, Any]] = []
    for r in regions:
        name = r.get("name")
        if not name:
            continue
        owned = _region_patches(name, patches)
        if not owned:
            issues.append(_issue(
                "warning", "region_without_patches",
                f"Region {name!r} has no owned mesh patches "
                f"(expected patch names with prefix {name!r}_).  "
                "splitMeshRegions will still create the cellZone but the "
                "region will have only the auto-generated coupled interfaces.",
                region=name,
            ))
    return issues


def _check_each_fluid_has_inlet(
    fluids: list[dict[str, Any]],
    patches: list[dict[str, Any]],
    bcs: dict[str, Any],
) -> list[dict[str, Any]]:
    """Forced-flow fluid regions need an inlet; natural convection doesn't.

    We emit a warning (not error) — natural-convection enclosures with
    a fixed-T wall + cold wall are perfectly valid CHT cases.  The user
    sees the warning and confirms intent.
    """
    issues: list[dict[str, Any]] = []
    for r in fluids:
        name = r.get("name")
        if not name:
            continue
        owned = _region_patches(name, patches)
        if not any(_patch_role(p, bcs) == "inlet" for p in owned):
            issues.append(_issue(
                "warning", "fluid_region_no_inlet",
                f"Fluid region {name!r} has no inlet patch.  This is OK "
                "for natural-convection / closed-cavity cases; otherwise "
                "the fluid will be quiescent.",
                region=name, owned_patches=owned,
            ))
    return issues


def _check_each_solid_has_interface(
    solids: list[dict[str, Any]],
    fluids: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """A solid with no fluid interface can't participate in CHT — error."""
    fluid_names = {r.get("name") for r in fluids if r.get("name")}
    issues: list[dict[str, Any]] = []
    for s in solids:
        name = s.get("name")
        if not name:
            continue
        ifaces = list(s.get("interfaces") or [])
        if not any(i in fluid_names for i in ifaces):
            issues.append(_issue(
                "error", "solid_region_no_fluid_interface",
                f"Solid region {name!r} has no interface to any fluid "
                f"region.  CHT needs at least one fluid neighbour for the "
                f"coupled temperature BC to do any work; check that the "
                f"region's 'interfaces' list names a real fluid region.",
                region=name, interfaces=ifaces,
            ))
    return issues


def _check_interfaces_reciprocal(
    regions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """``A.interfaces ∋ B  ⇒  B.interfaces ∋ A`` — coupled BCs need both sides."""
    by_name = {r.get("name"): r for r in regions if r.get("name")}
    issues: list[dict[str, Any]] = []
    for a_name, a in by_name.items():
        for b_name in a.get("interfaces") or ():
            b = by_name.get(b_name)
            if b is None:
                # The interface points to a region the config doesn't
                # know about — a typo or the region was dropped.
                issues.append(_issue(
                    "error", "interface_dangling",
                    f"Region {a_name!r} declares an interface with {b_name!r} "
                    "but no such region exists in the config.",
                    region=a_name, missing_neighbour=b_name,
                ))
                continue
            if a_name not in (b.get("interfaces") or ()):
                issues.append(_issue(
                    "warning", "interface_one_sided",
                    f"Interface {a_name!r} ⇆ {b_name!r} is declared on "
                    f"{a_name} but missing from {b_name}'s interfaces list — "
                    "the coupled BC pair will be one-sided.",
                    region=a_name, neighbour=b_name,
                ))
    return issues
