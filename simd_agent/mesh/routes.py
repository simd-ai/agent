"""API routes for mesh conversion."""

import logging
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from .config import PUBLIC_BASE_URL, STORAGE_DIR, TMP_DIR
from .converters import convert_gmsh_msh, convert_openfoam_zip, convert_vtk_formats
from .utils import (
    choose_patch_array,
    get_array_info,
    split_poly_by_cell_array,
    write_polydata_vtp,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/mesh", tags=["mesh"])


# --- Response Models ---

class PatchInfo(BaseModel):
    """Information about a mesh patch."""
    name: str
    type: str
    nCells: int
    vtpUrl: str


class CheckMeshResult(BaseModel):
    """Mesh statistics from checkMesh-like analysis."""
    cells: int
    faces: int
    points: int
    boundingBox: Optional[Dict[str, List[float]]] = None
    characteristicLength: Optional[float] = None


class ConvertResponse(BaseModel):
    """Response from mesh conversion endpoint."""
    meshId: str
    format: str
    mergedVtpUrl: str
    patches: List[PatchInfo]
    checkMesh: CheckMeshResult
    arrayInfo: Optional[Dict[str, Any]] = None


# --- Routes ---

@router.post("/convert", response_model=ConvertResponse)
async def convert_mesh(
    file: UploadFile = File(...),
    split_array: Optional[str] = Form(None),
) -> ConvertResponse:
    """
    Convert a mesh file to VTP format for vtk.js visualization.
    
    Supported formats:
    - .msh (Gmsh)
    - .vtu, .vtk, .vtp (VTK formats)
    - .stl, .obj (Surface meshes)
    - .zip (OpenFOAM case - requires foamToVTK)
    
    Args:
        file: The mesh file to convert
        split_array: Optional name of cell array to split patches by
        
    Returns:
        ConvertResponse with URLs to converted VTP files
    """
    mesh_id = str(uuid.uuid4())
    
    # Create working directories
    work_dir = TMP_DIR / mesh_id
    out_dir = STORAGE_DIR / mesh_id
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Save uploaded file
        filename = file.filename or "mesh"
        ext = Path(filename).suffix.lower()
        input_path = work_dir / filename
        
        with open(input_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"Processing {filename} ({len(content)} bytes)")
        
        # Convert based on format
        patches: Dict[str, Any] = {}
        
        if ext == ".msh":
            merged, patches = convert_gmsh_msh(input_path)
        elif ext == ".zip":
            merged = convert_openfoam_zip(input_path, work_dir)
        elif ext in (".vtu", ".vtk", ".vtp", ".stl", ".obj"):
            merged = convert_vtk_formats(input_path)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {ext}. "
                       f"Supported: .msh, .vtu, .vtk, .vtp, .stl, .obj, .zip",
            )
        
        # Get array info for the merged mesh
        array_info = get_array_info(merged)
        
        # If no patches from format-specific conversion, try splitting by array
        if not patches:
            # Use provided split_array or auto-detect
            arr_name = split_array or choose_patch_array(merged)
            if arr_name and arr_name in merged.cell_data:
                patches = split_poly_by_cell_array(merged, arr_name)
                logger.info(f"Split mesh by '{arr_name}' into {len(patches)} patches")
        
        # Write merged VTP
        merged_path = out_dir / "merged.vtp"
        write_polydata_vtp(merged, merged_path)
        merged_url = f"{PUBLIC_BASE_URL}/static/{mesh_id}/merged.vtp"
        
        # Write patch VTPs
        patch_infos: List[PatchInfo] = []
        for patch_name, patch_poly in patches.items():
            patch_file = out_dir / f"patch_{patch_name}.vtp"
            write_polydata_vtp(patch_poly, patch_file)
            
            patch_infos.append(PatchInfo(
                name=patch_name,
                type="patch",  # Could be inferred from name
                nCells=int(patch_poly.n_cells),
                vtpUrl=f"{PUBLIC_BASE_URL}/static/{mesh_id}/patch_{patch_name}.vtp",
            ))
        
        # Compute checkMesh-like statistics
        bounds = merged.bounds
        bbox = {
            "min": [bounds[0], bounds[2], bounds[4]],
            "max": [bounds[1], bounds[3], bounds[5]],
        }
        
        # Characteristic length (diagonal of bounding box)
        import numpy as np
        diag = np.sqrt(
            (bounds[1] - bounds[0])**2 +
            (bounds[3] - bounds[2])**2 +
            (bounds[5] - bounds[4])**2
        )
        
        check_mesh = CheckMeshResult(
            cells=int(merged.n_cells),
            faces=int(merged.n_cells),  # For surface mesh, faces ≈ cells
            points=int(merged.n_points),
            boundingBox=bbox,
            characteristicLength=float(diag),
        )
        
        return ConvertResponse(
            meshId=mesh_id,
            format=ext.lstrip("."),
            mergedVtpUrl=merged_url,
            patches=patch_infos,
            checkMesh=check_mesh,
            arrayInfo=array_info,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Mesh conversion failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Mesh conversion failed: {str(e)}",
        )
    finally:
        # Cleanup temp directory
        if work_dir.exists():
            shutil.rmtree(work_dir, ignore_errors=True)


@router.get("/{mesh_id}")
async def get_mesh_info(mesh_id: str) -> Dict[str, Any]:
    """Get information about a previously converted mesh."""
    out_dir = STORAGE_DIR / mesh_id
    
    if not out_dir.exists():
        raise HTTPException(status_code=404, detail="Mesh not found")
    
    # List all VTP files
    vtp_files = list(out_dir.glob("*.vtp"))
    
    return {
        "meshId": mesh_id,
        "files": [
            {
                "name": f.name,
                "url": f"{PUBLIC_BASE_URL}/static/{mesh_id}/{f.name}",
                "size": f.stat().st_size,
            }
            for f in vtp_files
        ],
    }


@router.delete("/{mesh_id}")
async def delete_mesh(mesh_id: str) -> Dict[str, str]:
    """Delete a converted mesh and its files."""
    out_dir = STORAGE_DIR / mesh_id
    
    if not out_dir.exists():
        raise HTTPException(status_code=404, detail="Mesh not found")
    
    shutil.rmtree(out_dir, ignore_errors=True)
    
    return {"status": "deleted", "meshId": mesh_id}
