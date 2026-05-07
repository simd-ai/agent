"""API route handlers for mesh conversion and serving."""

import shutil
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyvista as pv
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

logger = logging.getLogger(__name__)

from .config import TMP_DIR
from .converters import (
    convert_gmsh_msh,
    convert_openfoam_zip,
    convert_step_file,
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

router = APIRouter(prefix="/api/mesh", tags=["mesh"])

# ── Storage key helpers ───────────────────────────────────────────────────

def _mesh_key(simulation_id: str, filename: str) -> str:
    """Build the storage key: meshes/{simulation_id}/{filename}"""
    return f"meshes/{simulation_id}/{filename}"


@router.post("/convert")
async def convert(
    request: Request,
    file: UploadFile = File(...),
    simulation_id: str = Form(...),
):
    """
    Convert a mesh file to VTP format and upload to object storage.

    The simulation_id is used as the storage key — mesh files live at
    meshes/{simulation_id}/.

    Accepts:
      - .msh (Gmsh)
      - .vtu, .vtk, .vtp, .stl, .obj (VTK formats)
      - .step, .stp (STEP CAD files)
      - .zip (OpenFOAM case)

    Returns:
        JSON with mesh ID (= simulation_id), proxy URLs, and metadata
    """
    from simd_agent.storage import get_storage
    storage = get_storage()

    mesh_id = simulation_id

    tmp_dir = TMP_DIR / mesh_id
    tmp_dir.mkdir(parents=True, exist_ok=True)

    in_name = file.filename or "upload"
    in_ext = Path(in_name).suffix.lower()
    in_path = tmp_dir / f"input{in_ext if in_ext else ''}"

    # Save upload to temp disk
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

        elif in_ext in [".step", ".stp"]:
            surface, patches = convert_step_file(in_path)
            array_info = get_array_info(surface)

        elif in_ext in [".vtu", ".vtk", ".vtp", ".obj"]:
            surface = convert_vtk_formats(in_path)
            array_info = get_array_info(surface)

            logger.info(f"Available cell data arrays: {[a['name'] for a in array_info['cellDataArrays']]}")
            logger.info(f"Available point data arrays: {[a['name'] for a in array_info['pointDataArrays']]}")
            logger.info(f"Detected patch array: {array_info['detectedPatchArray']}")

            patch_arr = array_info.get("detectedPatchArray")

            if patch_arr:
                split = split_poly_by_cell_array(surface, patch_arr)
                logger.info(f"Split into {len(split)} patches using array '{patch_arr}'")
                logger.info(f"Patch values: {list(split.keys())}")

                if 1 < len(split) <= 200:
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

        # ── Upload original mesh ────────────────────────────────────
        original_bytes = in_path.read_bytes()
        original_key = _mesh_key(mesh_id, f"original{in_ext}")
        await storage.upload(original_key, original_bytes)
        logger.info(f"Uploaded original mesh: {original_key}")

        # ── Write + upload surface VTP ──────────────────────────────
        surface_local = tmp_dir / "surface.vtp"
        write_polydata_vtp(surface, surface_local)
        surface_bytes = surface_local.read_bytes()
        surface_key = _mesh_key(mesh_id, "surface.vtp")
        await storage.upload(surface_key, surface_bytes, content_type="application/xml")

        # Build proxy URLs (never expire, unlike signed URLs)
        base = str(request.base_url).rstrip("/")
        surface_proxy_url = f"{base}/api/mesh/{mesh_id}/vtp/surface.vtp"

        # ── Write + upload patch VTPs ───────────────────────────────
        patch_urls: Dict[str, str] = {}
        patch_info: List[Dict[str, Any]] = []
        patch_array_name = array_info.get("detectedPatchArray") if array_info else None

        if patches:
            for value, poly in patches.items():
                if poly is None or poly.n_points == 0:
                    continue
                safe = safe_name(str(value))
                p_local = tmp_dir / f"patch_{safe}.vtp"
                write_polydata_vtp(poly, p_local)
                p_bytes = p_local.read_bytes()
                p_key = _mesh_key(mesh_id, f"patches/{safe}.vtp")
                await storage.upload(p_key, p_bytes, content_type="application/xml")
                p_proxy_url = f"{base}/api/mesh/{mesh_id}/vtp/patches/{safe}.vtp"
                patch_urls[safe] = p_proxy_url
                patch_info.append({
                    "id": safe,
                    "value": value,
                    "name": value,
                    "url": p_proxy_url,
                    "nPoints": int(poly.n_points),
                    "nCells": int(poly.n_cells),
                })

        return {
            "meshId": mesh_id,
            "surfaceVtpUrl": surface_proxy_url,
            "patchVtpUrls": patch_urls,
            "patches": patch_info,
            "patchArrayName": patch_array_name,
            "availableArrays": array_info,
            "meta": {
                "nPoints": int(surface.n_points),
                "nCells": int(surface.n_cells),
            },
            "originalMesh": {
                "fileName": in_name,
                "format": in_ext.lstrip("."),
                "storagePath": original_key,
            },
        }

    finally:
        # Cleanup temp directory
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.post("/{mesh_id}/check")
async def check_mesh_quality(mesh_id: str):
    """Run OpenFOAM checkMesh on the stored mesh via the simulation server.

    Downloads the original mesh from GCS, sends it to the sim server's
    ``POST /api/mesh/check`` endpoint which runs ``gmshToFoam`` +
    ``checkMesh -allGeometry``, then returns parsed quality metrics.

    Returns:
        Structured JSON with max_non_orthogonality, avg_non_orthogonality,
        max_skewness, max_aspect_ratio, n_severe_non_ortho, mesh_ok.
    """
    from simd_agent.storage import get_storage
    from simd_agent.run.simulation_server_client import (
        SimulationServerClient,
        SimulationServerError,
    )

    storage = get_storage()

    # Find the original mesh file in storage
    mesh_data = None
    found_ext = ""
    for ext in (".msh", ".zip"):
        key = f"meshes/{mesh_id}/original{ext}"
        mesh_data = await storage.download(key)
        if mesh_data:
            found_ext = ext
            break

    if not mesh_data:
        raise HTTPException(status_code=404, detail="Original mesh not found in storage")

    client = SimulationServerClient()
    try:
        result = await client.check_mesh(mesh_data, f"mesh{found_ext}")
        return result
    except SimulationServerError as e:
        detail = str(e)
        # Pick a more specific status code based on the error
        if "Cannot reach" in detail or "connection" in detail.lower():
            status = 503  # Service Unavailable
        elif "timed out" in detail.lower():
            status = 504  # Gateway Timeout
        else:
            status = 502  # Bad Gateway (upstream error)
        logger.error(f"[MESH] checkMesh failed: {detail}")
        raise HTTPException(status_code=status, detail=detail)
    finally:
        await client.close()


@router.get("/{mesh_id}/vtp/{file_path:path}")
async def serve_vtp(mesh_id: str, file_path: str):
    """Proxy VTP files from object storage — returns bytes with correct content type.

    URLs never expire. The backend fetches from storage and streams to the
    browser with appropriate caching headers.

    Examples:
        GET /api/mesh/{mesh_id}/vtp/surface.vtp
        GET /api/mesh/{mesh_id}/vtp/patches/inlet.vtp
    """
    from simd_agent.storage import get_storage

    key = _mesh_key(mesh_id, file_path)
    data = await get_storage().download(key)
    if data is None:
        raise HTTPException(status_code=404, detail=f"VTP file not found: {file_path}")

    from fastapi.responses import Response
    return Response(
        content=data,
        media_type="application/xml",
        headers={
            "Content-Disposition": f'inline; filename="{Path(file_path).name}"',
            "Cache-Control": "public, max-age=86400",
            "Access-Control-Allow-Origin": "*",
        },
    )


@router.get("/{mesh_id}/original")
async def get_original_mesh(mesh_id: str):
    """
    Get the original mesh file for a given mesh ID (= simulation_id).
    """
    from simd_agent.storage import get_storage
    storage = get_storage()

    prefix = _mesh_key(mesh_id, "original")
    blobs = await storage.list_keys(prefix)

    if not blobs:
        raise HTTPException(
            status_code=404,
            detail=f"Original mesh file not found for {mesh_id}.",
        )

    key = blobs[0]
    data = await storage.download(key)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Failed to download mesh: {key}")

    filename = Path(key).name
    from fastapi.responses import Response
    return Response(
        content=data,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{mesh_id}/info")
async def get_mesh_info(mesh_id: str):
    """Get information about a stored mesh."""
    from simd_agent.storage import get_storage
    storage = get_storage()

    prefix = f"meshes/{mesh_id}/"
    blobs = await storage.list_keys(prefix)

    if not blobs:
        raise HTTPException(status_code=404, detail=f"Mesh not found: {mesh_id}")

    original_blobs = [b for b in blobs if "/original" in b]
    vtp_blobs = [b for b in blobs if b.endswith(".vtp") and "/patches/" not in b]
    patch_blobs = [b for b in blobs if "/patches/" in b and b.endswith(".vtp")]

    original_info = None
    if original_blobs:
        name = Path(original_blobs[0]).name
        original_info = {
            "fileName": name,
            "format": Path(name).suffix.lstrip("."),
        }

    return {
        "meshId": mesh_id,
        "originalMesh": original_info,
        "surfaceVtp": "surface.vtp" if vtp_blobs else None,
        "patchCount": len(patch_blobs),
        "storagePaths": blobs,
    }


@router.post("/debug/inspect")
async def inspect_mesh(file: UploadFile = File(...)):
    """Debug endpoint: Inspect a mesh file and return all data arrays."""
    from uuid import uuid4
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
        if in_ext not in [".msh", ".vtu", ".vtk", ".vtp", ".stl", ".obj", ".step", ".stp"]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file extension for inspection: {in_ext}",
            )

        info = print_mesh_info(in_path)

        detected_patch_array = None
        if in_ext in [".vtu", ".vtk", ".vtp", ".stl", ".obj"]:
            surface = convert_vtk_formats(in_path)
            detected_patch_array = choose_patch_array(surface)
        elif in_ext in [".step", ".stp"]:
            surface, patches = convert_step_file(in_path)
            detected_patch_array = choose_patch_array(surface)
            info["step_conversion"] = {
                "num_patches": len(patches),
                "patch_names": list(patches.keys()),
            }

        if in_ext in [".vtu", ".vtk", ".vtp", ".stl", ".obj", ".step", ".stp"]:
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
        shutil.rmtree(tmp_dir, ignore_errors=True)
