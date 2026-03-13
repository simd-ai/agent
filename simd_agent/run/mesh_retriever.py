# simd_agent/mesh_retriever.py
"""Mesh file retrieval for simulation packaging.

This module provides functionality to retrieve original mesh files from storage
for inclusion in OpenFOAM simulation case packages.
"""

import logging
from pathlib import Path
from typing import Tuple

from simd_agent.mesh.config import STORAGE_DIR

logger = logging.getLogger(__name__)


class MeshNotFoundError(Exception):
    """Raised when mesh file cannot be found."""
    pass


def get_mesh_file(mesh_id: str) -> Tuple[bytes, str, str]:
    """Retrieve the original mesh file from storage.
    
    Args:
        mesh_id: The mesh identifier (from /api/mesh/convert response)
        
    Returns:
        Tuple of (file_bytes, original_filename, mesh_format)
        
    Raises:
        MeshNotFoundError: If mesh or original file not found
    """
    mesh_dir = STORAGE_DIR / mesh_id
    
    if not mesh_dir.exists():
        raise MeshNotFoundError(f"Mesh directory not found: {mesh_id}")
    
    # Find original mesh file
    original_files = list(mesh_dir.glob("original.*"))
    
    if not original_files:
        raise MeshNotFoundError(
            f"Original mesh file not found for {mesh_id}. "
            "This mesh may have been uploaded before original file storage was enabled."
        )
    
    original_path = original_files[0]
    mesh_format = original_path.suffix.lstrip(".")
    
    logger.info(f"[MESH] Retrieved mesh file: {original_path} ({mesh_format} format)")
    
    return original_path.read_bytes(), original_path.name, mesh_format


def get_mesh_path(mesh_id: str) -> Path:
    """Get the path to the original mesh file.
    
    Args:
        mesh_id: The mesh identifier
        
    Returns:
        Path to the original mesh file
        
    Raises:
        MeshNotFoundError: If mesh not found
    """
    mesh_dir = STORAGE_DIR / mesh_id
    
    if not mesh_dir.exists():
        raise MeshNotFoundError(f"Mesh directory not found: {mesh_id}")
    
    original_files = list(mesh_dir.glob("original.*"))
    
    if not original_files:
        raise MeshNotFoundError(f"Original mesh file not found for {mesh_id}")
    
    return original_files[0]


def detect_mesh_converter(mesh_format: str) -> str:
    """Determine which OpenFOAM mesh converter to use.
    
    Args:
        mesh_format: File extension (without dot) - e.g., "msh", "stl"
        
    Returns:
        OpenFOAM converter command name
    """
    # Mapping of file formats to OpenFOAM converters
    converters = {
        # Gmsh/Fluent mesh formats
        "msh": "fluentMeshToFoam",  # Fluent mesh format (Gmsh exports this)
        
        # CFD formats
        "cas": "fluentMeshToFoam",  # Fluent case file
        "ccm": "ccmToFoam",          # Star-CCM+ format
        "cgns": "cgnsToFoam",        # CGNS format
        
        # Native Gmsh (requires gmshToFoam which may need separate install)
        # Some versions of Gmsh export in Fluent format with .msh extension
        
        # STL (for snappyHexMesh workflow - different process)
        "stl": "surfaceFeatureExtract",  # Part of snappyHexMesh workflow
        
        # VTK formats (need conversion)
        "vtk": "foamToVTK -case",  # Actually reverse direction - may need custom handling
        "vtu": "foamToVTK -case",
        
        # OpenFOAM native (no conversion needed)
        "foam": None,  # Already OpenFOAM format
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
        # Fluent/Gmsh mesh - most common case
        # fluentMeshToFoam expects the mesh file in the case root
        commands = [
            f"fluentMeshToFoam {mesh_filename}",
        ]
        
    elif fmt in ("cas",):
        # Fluent case file
        commands = [
            f"fluentMeshToFoam {mesh_filename}",
        ]
        
    elif fmt in ("cgns",):
        # CGNS format
        commands = [
            f"cgnsToFoam {mesh_filename}",
        ]
        
    elif fmt in ("stl",):
        # STL requires snappyHexMesh workflow
        # This is more complex - requires blockMesh first, then snappyHexMesh
        commands = [
            "blockMesh",  # Create background mesh
            "surfaceFeatureExtract",  # Extract features from STL
            "snappyHexMesh -overwrite",  # Generate mesh around STL geometry
        ]
        
    elif fmt in ("foam", "openfoam"):
        # Already OpenFOAM format - no conversion needed
        commands = []
        
    else:
        # Default: try fluentMeshToFoam
        commands = [
            f"fluentMeshToFoam {mesh_filename}",
        ]
    
    return commands
