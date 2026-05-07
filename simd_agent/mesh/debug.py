"""Debug utilities for inspecting mesh data."""

import numpy as np
import pyvista as pv
from pathlib import Path
from typing import Any, Dict


def inspect_polydata(poly: pv.PolyData) -> Dict[str, Any]:
    """
    Inspect a PolyData object and return detailed information about its data arrays.
    
    Returns:
        Dictionary with cell_data, point_data, and field_data information
    """
    info = {
        "geometry": {
            "n_points": int(poly.n_points),
            "n_cells": int(poly.n_cells),
            "bounds": poly.bounds,
        },
        "cell_data": {},
        "point_data": {},
        "field_data": {},
    }
    
    # Inspect cell data arrays
    for key in poly.cell_data.keys():
        arr = poly.cell_data[key]
        if arr is None:
            continue
            
        a = np.asarray(arr)
        cell_info = {
            "dtype": str(a.dtype),
            "shape": a.shape,
            "range": [float(a.min()), float(a.max())] if a.size > 0 else None,
        }
        
        # For integer arrays, show unique values (potential patch IDs)
        if np.issubdtype(a.dtype, np.integer) and a.ndim == 1:
            unique = np.unique(a)
            if len(unique) <= 50:  # Only show if reasonable number
                cell_info["unique_values"] = unique.tolist()
                cell_info["value_counts"] = {
                    int(val): int(np.sum(a == val)) for val in unique
                }
        
        info["cell_data"][key] = cell_info
    
    # Inspect point data arrays
    for key in poly.point_data.keys():
        arr = poly.point_data[key]
        if arr is None:
            continue
            
        a = np.asarray(arr)
        point_info = {
            "dtype": str(a.dtype),
            "shape": a.shape,
            "range": [float(a.min()), float(a.max())] if a.size > 0 else None,
        }
        
        info["point_data"][key] = point_info
    
    # Inspect field data
    for key in poly.field_data.keys():
        arr = poly.field_data[key]
        if arr is None:
            continue
            
        a = np.asarray(arr)
        field_info = {
            "dtype": str(a.dtype),
            "shape": a.shape,
        }
        
        if a.size <= 20:  # Show small arrays
            field_info["values"] = a.tolist()
        
        info["field_data"][key] = field_info
    
    return info


def inspect_dataset(dataset: pv.DataSet) -> Dict[str, Any]:
    """
    Inspect any VTK dataset and return detailed information.
    """
    info = {
        "type": type(dataset).__name__,
        "geometry": {
            "n_points": int(dataset.n_points),
            "n_cells": int(dataset.n_cells),
            "bounds": dataset.bounds,
        },
        "cell_data": {},
        "point_data": {},
        "field_data": {},
    }
    
    # Inspect cell data arrays
    for key in dataset.cell_data.keys():
        arr = dataset.cell_data[key]
        if arr is None:
            continue
            
        a = np.asarray(arr)
        cell_info = {
            "dtype": str(a.dtype),
            "shape": a.shape,
            "range": [float(a.min()), float(a.max())] if a.size > 0 else None,
        }
        
        if np.issubdtype(a.dtype, np.integer) and a.ndim == 1:
            unique = np.unique(a)
            if len(unique) <= 50:
                cell_info["unique_values"] = unique.tolist()
                cell_info["value_counts"] = {
                    int(val): int(np.sum(a == val)) for val in unique
                }
        
        info["cell_data"][key] = cell_info
    
    # Inspect point data arrays
    for key in dataset.point_data.keys():
        arr = dataset.point_data[key]
        if arr is None:
            continue
            
        a = np.asarray(arr)
        point_info = {
            "dtype": str(a.dtype),
            "shape": a.shape,
            "range": [float(a.min()), float(a.max())] if a.size > 0 else None,
        }
        
        info["point_data"][key] = point_info
    
    # Inspect field data
    for key in dataset.field_data.keys():
        arr = dataset.field_data[key]
        if arr is None:
            continue
            
        a = np.asarray(arr)
        field_info = {
            "dtype": str(a.dtype),
            "shape": a.shape,
        }
        
        if a.size <= 20:
            field_info["values"] = a.tolist()
        
        info["field_data"][key] = field_info
    
    return info


def print_mesh_info(file_path: Path) -> Dict[str, Any]:
    """
    Load a mesh file and print detailed information about its data arrays.
    
    Args:
        file_path: Path to mesh file (.vtu, .vtp, .vtk, .stl, .obj, .msh, etc.)
    
    Returns:
        Dictionary with inspection results
    """
    import meshio
    
    ext = file_path.suffix.lower()
    
    if ext == ".msh":
        # Gmsh format - use meshio
        mesh = meshio.read(str(file_path))
        
        info = {
            "file": str(file_path),
            "format": "gmsh",
            "points": {
                "count": mesh.points.shape[0],
                "dimensions": mesh.points.shape[1],
            },
            "cells": {},
            "cell_data": {},
            "point_data": {},
            "field_data": {},
        }
        
        # Cell blocks
        for i, cell_block in enumerate(mesh.cells):
            cell_type = cell_block.type
            cell_count = len(cell_block.data)
            info["cells"][f"{i}_{cell_type}"] = {
                "type": cell_type,
                "count": cell_count,
            }
        
        # Cell data
        for key, arrays in mesh.cell_data.items():
            info["cell_data"][key] = {}
            for i, arr in enumerate(arrays):
                if arr is None:
                    continue
                a = np.asarray(arr)
                arr_info = {
                    "shape": a.shape,
                    "dtype": str(a.dtype),
                    "range": [float(a.min()), float(a.max())] if a.size > 0 else None,
                }
                
                if np.issubdtype(a.dtype, np.integer) and a.ndim == 1:
                    unique = np.unique(a)
                    if len(unique) <= 50:
                        arr_info["unique_values"] = unique.tolist()
                        arr_info["value_counts"] = {
                            int(val): int(np.sum(a == val)) for val in unique
                        }
                
                info["cell_data"][key][f"block_{i}"] = arr_info
        
        # Point data
        for key, arr in mesh.point_data.items():
            if arr is None:
                continue
            a = np.asarray(arr)
            info["point_data"][key] = {
                "shape": a.shape,
                "dtype": str(a.dtype),
                "range": [float(a.min()), float(a.max())] if a.size > 0 else None,
            }
        
        # Field data (physical group names in Gmsh)
        if mesh.field_data:
            for name, (tag, dim) in mesh.field_data.items():
                info["field_data"][name] = {
                    "tag": int(tag),
                    "dimension": int(dim),
                }
        
        return info
    
    else:
        # VTK formats - use PyVista
        dataset = pv.read(str(file_path))
        
        info = {
            "file": str(file_path),
            "format": ext,
        }
        
        info.update(inspect_dataset(dataset))
        
        return info
