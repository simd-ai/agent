# simd_agent/run/mesh_retriever.py
"""Mesh file retrieval from object storage for simulation packaging.

Mesh files are stored at:
    meshes/{mesh_id}/original.{ext}
    meshes/{mesh_id}/surface.vtp
    meshes/{mesh_id}/patches/*.vtp

The mesh_id is the simulation_id.
"""

import asyncio
import logging
from pathlib import Path
from typing import Tuple

from simd_agent.mesh.config import TMP_DIR

logger = logging.getLogger(__name__)


class MeshNotFoundError(Exception):
    """Raised when mesh file cannot be found."""
    pass


def _mesh_key(mesh_id: str, filename: str) -> str:
    return f"meshes/{mesh_id}/{filename}"


def _run_async(coro):
    """Run an async coroutine from sync code, handling nested event loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside an async context — use a thread to avoid deadlock
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)


async def async_get_mesh_file(mesh_id: str) -> Tuple[bytes, str, str]:
    """Retrieve the original mesh file from object storage (async).

    Args:
        mesh_id: The mesh identifier (= simulation_id)

    Returns:
        Tuple of (file_bytes, original_filename, mesh_format)

    Raises:
        MeshNotFoundError: If mesh or original file not found
    """
    from simd_agent.storage import get_storage
    storage = get_storage()

    prefix = _mesh_key(mesh_id, "original")
    blobs = await storage.list_keys(prefix)

    if not blobs:
        raise MeshNotFoundError(f"Original mesh file not found for {mesh_id}")

    key = blobs[0]
    data = await storage.download(key)
    if data is None:
        raise MeshNotFoundError(f"Failed to download mesh: {key}")

    filename = Path(key).name
    mesh_format = Path(filename).suffix.lstrip(".")

    logger.info(f"[MESH] Retrieved mesh: {key} ({mesh_format} format, {len(data)} bytes)")

    return data, filename, mesh_format


def get_mesh_file(mesh_id: str) -> Tuple[bytes, str, str]:
    """Retrieve the original mesh file from object storage (sync wrapper).

    Delegates to async_get_mesh_file, handling the case where we're called
    from inside a running event loop (e.g. from packaging.py called by
    the async orchestrator).
    """
    return _run_async(async_get_mesh_file(mesh_id))


def get_mesh_path(mesh_id: str) -> Path:
    """Download the original mesh file to a temp path and return it.

    Args:
        mesh_id: The mesh identifier (= simulation_id)

    Returns:
        Path to the downloaded mesh file (in temp dir)

    Raises:
        MeshNotFoundError: If mesh not found
    """
    data, filename, _ = get_mesh_file(mesh_id)

    tmp_dir = TMP_DIR / f"mesh_{mesh_id}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    local_path = tmp_dir / filename
    local_path.write_bytes(data)

    return local_path


def detect_mesh_converter(mesh_format: str) -> str:
    """Determine which OpenFOAM mesh converter to use.

    Args:
        mesh_format: File extension (without dot) - e.g., "msh", "stl"

    Returns:
        OpenFOAM converter command name
    """
    converters = {
        "msh": "fluentMeshToFoam",
        "cas": "fluentMeshToFoam",
        "ccm": "ccmToFoam",
        "cgns": "cgnsToFoam",
        "vtk": "foamToVTK -case",
        "vtu": "foamToVTK -case",
        "foam": None,
    }

    return converters.get(mesh_format.lower(), "fluentMeshToFoam")


def get_mesh_conversion_commands(mesh_format: str, mesh_filename: str) -> list[str]:
    """Get the sequence of commands to convert mesh to OpenFOAM format.

    Args:
        mesh_format: File extension (without dot)
        mesh_filename: Name of the mesh file in the case directory

    Returns:
        List of shell commands to run for mesh conversion
    """
    commands = []

    fmt = mesh_format.lower()

    if fmt in ("msh",):
        commands = [
            f"fluentMeshToFoam {mesh_filename}",
        ]

    elif fmt in ("cas",):
        commands = [
            f"fluentMeshToFoam {mesh_filename}",
        ]

    elif fmt in ("cgns",):
        commands = [
            f"cgnsToFoam {mesh_filename}",
        ]

    elif fmt in ("foam", "openfoam"):
        commands = []

    else:
        commands = [
            f"fluentMeshToFoam {mesh_filename}",
        ]

    return commands
