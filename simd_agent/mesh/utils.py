"""Utility functions for mesh processing."""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pyvista as pv


def safe_name(s: str) -> str:
    """Sanitize a string for use as a filename."""
    s = s.strip()
    s = re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s[:120] if s else "patch"


def write_polydata_vtp(poly: pv.PolyData, out_path: Path) -> None:
    """Write a PolyData to a VTP file, ensuring it's triangulated."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    poly = poly.triangulate()
    poly.save(str(out_path))


def poly_from_triangles(points: np.ndarray, tri_cells: np.ndarray) -> pv.PolyData:
    """
    Create a PolyData from points and triangle indices.

    Args:
        points: (N, 3) array of vertex coordinates
        tri_cells: (M, 3) array of triangle vertex indices

    Returns:
        PyVista PolyData object
    """
    if tri_cells.size == 0:
        return pv.PolyData()

    # VTK face format: [3, i0, i1, i2, 3, i0, i1, i2, ...]
    faces = np.hstack(
        [np.full((tri_cells.shape[0], 1), 3, dtype=np.int64), tri_cells]
    ).ravel()
    return pv.PolyData(points, faces)


def extract_surface_from_any(dataset: pv.DataSet) -> pv.PolyData:
    """Extract surface from any VTK dataset."""
    if isinstance(dataset, pv.PolyData):
        return dataset
    return dataset.extract_surface()


def choose_patch_array(poly: pv.PolyData) -> Optional[str]:
    """
    Try to find a cell-data array that looks like a patch/region id.

    Returns:
        Name of the array if found, None otherwise
    """
    candidates = [
        "patch",
        "Patch",
        "PatchId",
        "patchId",
        "BoundaryId",
        "RegionId",
        "region",
        "gmsh:physical",
        "PhysicalGroup",
        "CellEntityIds",
        "EntityIds",
    ]
    cell_keys = list(poly.cell_data.keys())

    for name in candidates:
        if name in cell_keys:
            return name

    # Fallback heuristic: few unique integers
    for k in cell_keys:
        arr = poly.cell_data[k]
        if arr is None:
            continue
        a = np.asarray(arr)
        if a.ndim == 1 and np.issubdtype(a.dtype, np.integer):
            u = np.unique(a)
            if 1 < len(u) <= 200:
                return k
    return None


def split_poly_by_cell_array(
    poly: pv.PolyData, array_name: str
) -> Dict[str, pv.PolyData]:
    """Split a PolyData into multiple based on cell array values."""
    arr = np.asarray(poly.cell_data[array_name])
    out: Dict[str, pv.PolyData] = {}
    for val in np.unique(arr):
        mask = arr == val
        sub = poly.extract_cells(mask)
        out[str(val)] = extract_surface_from_any(sub)
    return out


def read_with_pyvista(path: Path) -> pv.DataSet:
    """Read any VTK-compatible file with PyVista."""
    return pv.read(str(path))


def get_array_info(poly: pv.PolyData) -> Dict[str, Any]:
    """
    Get information about all data arrays in a PolyData.
    
    Returns a dictionary with:
    - cellDataArrays: list of cell data array info
    - pointDataArrays: list of point data array info
    - detectedPatchArray: the auto-detected patch array name (if any)
    """
    cell_arrays: List[Dict[str, Any]] = []
    point_arrays: List[Dict[str, Any]] = []
    
    # Process cell data arrays
    for key in poly.cell_data.keys():
        arr = poly.cell_data[key]
        if arr is None:
            continue
        
        a = np.asarray(arr)
        arr_info: Dict[str, Any] = {
            "name": key,
            "dtype": str(a.dtype),
            "shape": list(a.shape),
            "numComponents": a.shape[1] if a.ndim > 1 else 1,
        }
        
        # For integer arrays, this could be used for patch splitting
        if np.issubdtype(a.dtype, np.integer) and a.ndim == 1:
            unique = np.unique(a)
            arr_info["uniqueValues"] = unique.tolist() if len(unique) <= 50 else None
            arr_info["numUniqueValues"] = len(unique)
            arr_info["canSplitPatches"] = 1 < len(unique) <= 200
        else:
            arr_info["canSplitPatches"] = False
        
        # For numeric arrays, get the range
        if np.issubdtype(a.dtype, np.number) and a.size > 0:
            arr_info["range"] = [float(np.nanmin(a)), float(np.nanmax(a))]
        
        cell_arrays.append(arr_info)
    
    # Process point data arrays
    for key in poly.point_data.keys():
        arr = poly.point_data[key]
        if arr is None:
            continue
        
        a = np.asarray(arr)
        arr_info = {
            "name": key,
            "dtype": str(a.dtype),
            "shape": list(a.shape),
            "numComponents": a.shape[1] if a.ndim > 1 else 1,
        }
        
        if np.issubdtype(a.dtype, np.number) and a.size > 0:
            arr_info["range"] = [float(np.nanmin(a)), float(np.nanmax(a))]
        
        point_arrays.append(arr_info)
    
    # Get the auto-detected patch array
    detected = choose_patch_array(poly)
    
    return {
        "cellDataArrays": cell_arrays,
        "pointDataArrays": point_arrays,
        "detectedPatchArray": detected,
    }
