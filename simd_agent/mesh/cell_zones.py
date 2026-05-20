# simd_agent/mesh/cell_zones.py
"""Extract ``cellZones`` from mesh files.

A cellZone is a named group of 3D cells — gmsh calls these "Physical
Volume" groups, OpenFOAM serialises them under ``constant/polyMesh/cellZones``,
and ``splitMeshRegions -cellZones -overwrite`` uses them to peel a unified
mesh into the per-region polyMesh trees that chtMultiRegion{Simple,}Foam
expects.

This module is the agent-side authority for "does this mesh describe
multiple regions, and what are they called?".  It feeds two consumers:

  * :func:`simd_agent.run.orchestration._detect_regions_from_mesh`, which
    prefers the cellZone list over the patch-name prefix heuristic when
    it's available — robust against meshes with arbitrary patch names.
  * :class:`simd_agent.models.MeshInfoV1` — the data carrier that round-
    trips through the precheck pipeline and is persisted with the
    simulation config.

Implementation: we read the .msh via ``meshio`` (already a hard dep, used
elsewhere in :mod:`simd_agent.mesh.converters`) and inspect
``mesh.field_data`` for entries whose dimension equals the cell
dimensionality.  ``field_data`` maps Physical Group names → ``(tag, dim)``.
Dim 3 = volume, dim 2 = surface, dim 1 = curve.  For 2D-extruded cases
where the volume cells are still hexahedra (dim 3 in gmsh terms after
``Extrude``), we still get one entry per physical volume name.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_cell_zones_from_msh(msh_path: Path | str) -> list[str]:
    """Return the list of cellZone names (3D Physical Groups) in a .msh file.

    Returns an empty list when:
      * the file does not exist;
      * meshio cannot read it;
      * the mesh has no Physical Volume groups (single-region cases or
        meshes whose volumes were never tagged).

    Never raises — callers always get a list, possibly empty.  The
    function is intentionally lightweight (parses with meshio, no full
    PyVista round-trip) so it can be called cheaply during precheck.
    """
    p = Path(msh_path)
    if not p.exists() or not p.is_file():
        return []

    try:
        import meshio
    except ImportError:  # pragma: no cover — meshio is a hard dep elsewhere
        logger.warning("[cell_zones] meshio not installed; skipping extraction")
        return []

    try:
        mesh = meshio.read(str(p))
    except Exception as exc:
        logger.warning(
            "[cell_zones] meshio failed to read %s (%s); returning []",
            p, exc,
        )
        return []

    field_data = getattr(mesh, "field_data", None) or {}
    # field_data: {name: (tag, dim)} where dim==3 means volume.
    names: list[str] = []
    seen: set[str] = set()
    for name, info in field_data.items():
        try:
            _tag, dim = int(info[0]), int(info[1])
        except (TypeError, ValueError, IndexError):
            continue
        if dim != 3:
            continue
        if name in seen:
            continue
        seen.add(name)
        names.append(str(name))

    # Stable order — caller may want deterministic iteration.
    names.sort()
    return names
