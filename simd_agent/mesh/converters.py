"""Mesh format converters."""

import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

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
