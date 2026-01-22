"""API route handlers."""

import shutil
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pyvista as pv
from fastapi import APIRouter, File, HTTPException, UploadFile

logger = logging.getLogger(__name__)

from .config import PUBLIC_BASE_URL, STORAGE_DIR, TMP_DIR
from .converters import (
    convert_gmsh_msh,
    convert_openfoam_zip,
    convert_vtk_formats,
)
from .utils import (
    choose_patch_array,
    safe_name,
    split_poly_by_cell_array,
    write_polydata_vtp,
    get_array_info,
)
from .debug import print_mesh_info

router = APIRouter()

router = APIRouter(prefix="/api/mesh", tags=["mesh"])

@router.post("/convert")
async def convert(file: UploadFile = File(...)):
    """
    Convert a mesh file to VTP format.

    Accepts:
      - .msh (Gmsh)
      - .vtu, .vtk, .vtp, .stl, .obj (VTK formats)
      - .zip (OpenFOAM case)

    Returns:
        JSON with mesh ID, VTP URLs, and metadata
    """
    mesh_id = uuid4().hex
    out_dir = STORAGE_DIR / mesh_id
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = TMP_DIR / mesh_id
    tmp_dir.mkdir(parents=True, exist_ok=True)

    in_name = file.filename or "upload"
    in_ext = Path(in_name).suffix.lower()
    in_path = tmp_dir / f"input{in_ext if in_ext else ''}"

    # Save upload to disk
    with open(in_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    surface: Optional[pv.PolyData] = None
    patches: Dict[str, pv.PolyData] = {}
    array_info: Dict[str, Any] = {}

    try:
        if in_ext == ".msh":
            surface, patches = convert_gmsh_msh(in_path)

        elif in_ext in [".vtu", ".vtk", ".vtp", ".stl", ".obj"]:
            surface = convert_vtk_formats(in_path)

            # Get ALL available arrays - this is sent to frontend
            array_info = get_array_info(surface)
            
            logger.info(f"Available cell data arrays: {[a['name'] for a in array_info['cellDataArrays']]}")
            logger.info(f"Available point data arrays: {[a['name'] for a in array_info['pointDataArrays']]}")
            logger.info(f"Detected patch array: {array_info['detectedPatchArray']}")

            # Try to split into patches if a useful array exists
            patch_arr = array_info.get("detectedPatchArray")
            
            if patch_arr:
                split = split_poly_by_cell_array(surface, patch_arr)
                logger.info(f"Split into {len(split)} patches using array '{patch_arr}'")
                logger.info(f"Patch values: {list(split.keys())}")
                
                if 1 < len(split) <= 200:
                    # Use raw values as keys - these are the actual values from the array
                    patches = {str(k): v.clean() for k, v in split.items()}
                    logger.info(f"Patch names (raw values): {list(patches.keys())}")
                else:
                    logger.warning(f"Split produced {len(split)} patches (outside 2-200 range), ignoring")

        elif in_ext == ".zip":
            surface = convert_openfoam_zip(in_path, tmp_dir)

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file extension: {in_ext}",
            )

        if surface is None or surface.n_points == 0:
            raise HTTPException(
                status_code=400,
                detail="Conversion produced an empty surface.",
            )

        # Get array info if not already done (for non-VTK formats)
        if not array_info:
            array_info = get_array_info(surface)

        # Write merged surface
        surface_path = out_dir / "surface.vtp"
        write_polydata_vtp(surface, surface_path)

        # Write patch files
        patch_urls: Dict[str, str] = {}
        patch_info: List[Dict[str, Any]] = []
        patch_array_name = array_info.get("detectedPatchArray") if array_info else None
        
        if patches:
            patches_dir = out_dir / "patches"
            for value, poly in patches.items():
                if poly is None or poly.n_points == 0:
                    continue
                # Use the raw value for filename (sanitized)
                safe = safe_name(str(value))
                p_path = patches_dir / f"{safe}.vtp"
                write_polydata_vtp(poly, p_path)
                url = f"{PUBLIC_BASE_URL}/static/{mesh_id}/patches/{p_path.name}"
                patch_urls[safe] = url
                patch_info.append({
                    "id": safe,                    # Safe filename version
                    "value": value,                # Raw array value (e.g., "1", "2")
                    "name": value,                 # Display name (same as value for now)
                    "url": url,
                    "nPoints": int(poly.n_points),
                    "nCells": int(poly.n_cells),
                })

        return {
            "meshId": mesh_id,
            "surfaceVtpUrl": f"{PUBLIC_BASE_URL}/static/{mesh_id}/surface.vtp",
            "patchVtpUrls": patch_urls,
            "patches": patch_info,
            "patchArrayName": patch_array_name,    # Which array was used to split
            "availableArrays": array_info,
            "meta": {
                "nPoints": int(surface.n_points),
                "nCells": int(surface.n_cells),
            },
        }

    finally:
        # Cleanup temp directory
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.post("/debug/inspect")
async def inspect_mesh(file: UploadFile = File(...)):
    """
    Debug endpoint: Inspect a mesh file and return all data arrays.

    Useful for debugging patch detection and understanding mesh structure.

    Accepts:
      - .msh (Gmsh)
      - .vtu, .vtk, .vtp, .stl, .obj (VTK formats)

    Returns:
        JSON with detailed information about all data arrays in the mesh
    """
    tmp_id = uuid4().hex
    tmp_dir = TMP_DIR / tmp_id
    tmp_dir.mkdir(parents=True, exist_ok=True)

    in_name = file.filename or "upload"
    in_ext = Path(in_name).suffix.lower()
    in_path = tmp_dir / f"input{in_ext if in_ext else ''}"

    # Save upload to disk
    with open(in_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    try:
        if in_ext not in [".msh", ".vtu", ".vtk", ".vtp", ".stl", ".obj"]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file extension for inspection: {in_ext}",
            )

        # Run inspection
        info = print_mesh_info(in_path)

        # Also add patch detection info for VTK formats
        if in_ext in [".vtu", ".vtk", ".vtp", ".stl", ".obj"]:
            surface = convert_vtk_formats(in_path)
            detected_patch_array = choose_patch_array(surface)

            info["patch_detection"] = {
                "detected_array": detected_patch_array,
                "would_split": False,
            }

            if detected_patch_array:
                split = split_poly_by_cell_array(surface, detected_patch_array)
                info["patch_detection"]["would_split"] = 1 < len(split) <= 200
                info["patch_detection"]["num_patches"] = len(split)
                info["patch_detection"]["patch_names"] = list(split.keys())

        return info

    finally:
        # Cleanup
        shutil.rmtree(tmp_dir, ignore_errors=True)
