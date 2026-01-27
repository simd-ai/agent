"""Mesh format converters."""

import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import meshio
import numpy as np
import pyvista as pv
from fastapi import HTTPException

from .utils import (
    extract_surface_from_any,
    poly_from_triangles,
    read_with_pyvista,
    safe_name,
)

# Lazy import gmsh to avoid import errors if not installed
_gmsh = None


def _get_gmsh():
    """Lazy-load gmsh module."""
    global _gmsh
    if _gmsh is None:
        try:
            import gmsh
            _gmsh = gmsh
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="gmsh is not installed. Install with: pip install gmsh",
            )
    return _gmsh


def convert_gmsh_msh(msh_path: Path) -> Tuple[pv.PolyData, Dict[str, pv.PolyData]]:
    """
    Convert Gmsh .msh file to surface PolyData with optional patch splitting.

    Gmsh .msh often contains explicit boundary triangles with physical groups.
    We build patch PolyData from those triangles.

    Args:
        msh_path: Path to the .msh file

    Returns:
        Tuple of (merged_surface, patches_dict)
    """
    mesh = meshio.read(str(msh_path))

    pts = np.asarray(mesh.points)
    if pts.shape[1] == 2:
        pts = np.c_[pts, np.zeros((pts.shape[0], 1))]

    # meshio structure:
    # mesh.cells is list of CellBlock(type, data)
    # mesh.cell_data is dict: {name: [array_per_cellblock]}
    gmsh_phys_lists = None
    if "gmsh:physical" in mesh.cell_data:
        gmsh_phys_lists = mesh.cell_data["gmsh:physical"]

    tri_block_indices: List[int] = []
    for i, cb in enumerate(mesh.cells):
        if cb.type in ("triangle", "triangle6"):
            tri_block_indices.append(i)

    patches: Dict[str, pv.PolyData] = {}

    if len(tri_block_indices) > 0:
        # Map physical tag -> name from field_data if present
        phys_tag_to_name: Dict[int, str] = {}
        if mesh.field_data:
            for name, (tag, dim) in mesh.field_data.items():
                phys_tag_to_name[int(tag)] = str(name)

        all_tris = []

        for idx in tri_block_indices:
            cb = mesh.cells[idx]
            tris = np.asarray(cb.data, dtype=np.int64)
            all_tris.append(tris)

            if gmsh_phys_lists is not None and idx < len(gmsh_phys_lists):
                tags = np.asarray(gmsh_phys_lists[idx], dtype=np.int64)
                for tag in np.unique(tags):
                    name = phys_tag_to_name.get(int(tag), f"phys_{int(tag)}")
                    name = safe_name(name)
                    sub_tris = tris[tags == tag]
                    if sub_tris.size == 0:
                        continue
                    patches[name] = poly_from_triangles(pts, sub_tris)

        merged = poly_from_triangles(pts, np.vstack(all_tris))
        merged = merged.clean()

        for k in list(patches.keys()):
            patches[k] = patches[k].clean()

        return merged, patches

    # Fallback: write mesh to VTU and extract surface
    tmp_vtu = msh_path.with_suffix(".vtu")
    meshio.write(str(tmp_vtu), mesh)
    grid = read_with_pyvista(tmp_vtu)
    surface = extract_surface_from_any(grid).clean()
    return surface, {}


def convert_vtk_formats(file_path: Path) -> pv.PolyData:
    """
    Convert VTU/VTK/VTP/STL/OBJ files to surface PolyData.

    Args:
        file_path: Path to the input file

    Returns:
        Surface PolyData
    """
    ds = read_with_pyvista(file_path)
    return extract_surface_from_any(ds).clean()


def convert_openfoam_zip(case_zip: Path, work_dir: Path) -> pv.PolyData:
    """
    Convert OpenFOAM case zip to surface PolyData.

    Requires foamToVTK in PATH (OpenFOAM installed).

    Args:
        case_zip: Path to the OpenFOAM case zip file
        work_dir: Working directory for extraction

    Returns:
        Surface PolyData

    Raises:
        HTTPException: If conversion fails
    """
    case_dir = work_dir / "case"
    case_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(case_zip, "r") as z:
        z.extractall(case_dir)

    # Find OpenFOAM case root (has system/controlDict)
    candidates = list(case_dir.rglob("system/controlDict"))
    if not candidates:
        raise HTTPException(
            status_code=400,
            detail="Zip does not look like an OpenFOAM case (missing system/controlDict).",
        )

    case_root = candidates[0].parent.parent

    # Check for foamToVTK
    if shutil.which("foamToVTK") is None:
        raise HTTPException(
            status_code=400,
            detail="foamToVTK not found in PATH. Install OpenFOAM on the server.",
        )

    # Run foamToVTK
    cmd = ["foamToVTK", "-case", str(case_root), "-latestTime"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise HTTPException(
            status_code=400,
            detail=f"foamToVTK failed:\n{proc.stderr[:2000]}",
        )

    vtk_dir = case_root / "VTK"
    if not vtk_dir.exists():
        raise HTTPException(
            status_code=400,
            detail="foamToVTK ran but did not produce a VTK/ directory.",
        )

    # Find produced VTK files
    produced = (
        list(vtk_dir.rglob("*.vtu"))
        + list(vtk_dir.rglob("*.vtk"))
        + list(vtk_dir.rglob("*.vtp"))
    )
    if not produced:
        raise HTTPException(
            status_code=400,
            detail="No VTK/VTU/VTP files found after foamToVTK.",
        )

    # Choose largest file as best guess
    produced.sort(key=lambda p: p.stat().st_size, reverse=True)
    ds = read_with_pyvista(produced[0])
    return extract_surface_from_any(ds).clean()


def convert_step_file(
    step_path: Path,
    mesh_size: Optional[float] = None,
    mesh_size_min: Optional[float] = None,
    mesh_size_max: Optional[float] = None,
) -> Tuple[pv.PolyData, Dict[str, pv.PolyData]]:
    """
    Convert a STEP/STP CAD file to surface PolyData with optional patch splitting.

    Uses gmsh to import the STEP file via OpenCASCADE and generate a surface mesh.
    Physical groups from the CAD model (if any) become patches.

    Args:
        step_path: Path to the .step or .stp file
        mesh_size: Target mesh element size (auto-computed if not provided)
        mesh_size_min: Minimum mesh element size
        mesh_size_max: Maximum mesh element size

    Returns:
        Tuple of (merged_surface, patches_dict)

    Raises:
        HTTPException: If conversion fails
    """
    gmsh = _get_gmsh()

    # Generate a temporary .msh output path
    msh_path = step_path.with_suffix(".msh")

    try:
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)  # Suppress terminal output

        # Import STEP file using OpenCASCADE kernel
        gmsh.model.add("step_import")
        
        try:
            gmsh.model.occ.importShapes(str(step_path))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to import STEP file: {str(e)}",
            )

        # Synchronize to make entities available
        gmsh.model.occ.synchronize()

        # Get bounding box for auto mesh size computation
        if mesh_size is None:
            try:
                xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
                diag = ((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2) ** 0.5
                mesh_size = diag / 50  # Default: ~50 elements across diagonal
                if mesh_size_min is None:
                    mesh_size_min = mesh_size / 10
                if mesh_size_max is None:
                    mesh_size_max = mesh_size * 2
            except Exception:
                mesh_size = 1.0
                mesh_size_min = 0.1
                mesh_size_max = 10.0

        # Set mesh size options
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size_min or mesh_size / 10)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size_max or mesh_size * 2)
        gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
        gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 6)

        # Get all surfaces and create physical groups for patches
        surfaces = gmsh.model.getEntities(dim=2)
        if not surfaces:
            raise HTTPException(
                status_code=400,
                detail="No surfaces found in STEP file.",
            )

        # Create physical groups for each surface (patch)
        for i, (dim, tag) in enumerate(surfaces, start=1):
            name = f"Surface_{tag}"
            # Try to get the entity name from STEP file
            try:
                entity_name = gmsh.model.getEntityName(dim, tag)
                if entity_name:
                    name = safe_name(entity_name)
            except Exception:
                pass
            gmsh.model.addPhysicalGroup(dim, [tag], i, name=name)

        # Generate 2D surface mesh
        try:
            gmsh.model.mesh.generate(2)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Mesh generation failed: {str(e)}",
            )

        # Check if mesh was generated
        node_tags, _, _ = gmsh.model.mesh.getNodes()
        if len(node_tags) == 0:
            raise HTTPException(
                status_code=400,
                detail="Mesh generation produced no nodes.",
            )

        # Write to MSH format
        gmsh.write(str(msh_path))

    finally:
        gmsh.finalize()

    # Use existing MSH converter to process the generated mesh
    try:
        surface, patches = convert_gmsh_msh(msh_path)
    finally:
        # Cleanup temporary MSH file
        if msh_path.exists():
            msh_path.unlink()

    return surface, patches
