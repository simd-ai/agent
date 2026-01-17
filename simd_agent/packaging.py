# simd_agent/packaging.py
"""Package generated OpenFOAM case files into a zip archive."""

import io
import logging
import re
import zipfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Pattern to match file blocks in LLM output
# Expected format:
# ```file:path/to/file
# <content>
# ```
FILE_BLOCK_PATTERN = re.compile(
    r"```file:([^\n]+)\n(.*?)```",
    re.DOTALL,
)

# Alternative pattern for triple backtick with language hint
# ```openfoam:path/to/file
ALT_FILE_BLOCK_PATTERN = re.compile(
    r"```(?:openfoam|foam|sh|bash|text)?:([^\n]+)\n(.*?)```",
    re.DOTALL,
)

# Standard run.sh template for OpenFOAM cases
RUN_SCRIPT_TEMPLATE = """#!/bin/bash
# Generated run script for OpenFOAM case
set -e

# Source OpenFOAM environment (adjust path as needed)
if [ -f /opt/openfoam/etc/bashrc ]; then
    source /opt/openfoam/etc/bashrc
elif [ -f /usr/lib/openfoam/openfoam*/etc/bashrc ]; then
    source /usr/lib/openfoam/openfoam*/etc/bashrc
fi

echo "=== Starting OpenFOAM case execution ==="
echo "Working directory: $(pwd)"

# Run blockMesh if blockMeshDict exists
if [ -f system/blockMeshDict ]; then
    echo "=== Running blockMesh ==="
    blockMesh
else
    echo "No blockMeshDict found, skipping blockMesh"
fi

# Run checkMesh
if command -v checkMesh &> /dev/null; then
    echo "=== Running checkMesh ==="
    checkMesh || echo "checkMesh reported issues, continuing..."
fi

# Run the solver
SOLVER="{solver}"
echo "=== Running solver: $SOLVER ==="

# For minimal first-run validation, limit iterations
if [ -n "$SIMD_QUICK_RUN" ]; then
    echo "Quick run mode: limiting to 10 iterations"
    # Modify controlDict for quick run
    sed -i.bak 's/endTime.*/endTime 10;/' system/controlDict 2>/dev/null || true
fi

$SOLVER

echo "=== Case execution complete ==="
exit 0
"""


class PackagingError(Exception):
    """Error during case packaging."""
    pass


def extract_file_blocks(llm_output: str) -> dict[str, str]:
    """Extract file blocks from LLM output.
    
    Expects format:
    ```file:relative/path/to/file
    file content here
    ```
    
    Args:
        llm_output: Raw LLM output text
        
    Returns:
        Dictionary mapping file paths to content
    """
    files = {}
    
    # Try primary pattern first
    matches = FILE_BLOCK_PATTERN.findall(llm_output)
    for path, content in matches:
        path = path.strip()
        content = content.rstrip("\n")
        files[path] = content
    
    # Try alternative pattern
    alt_matches = ALT_FILE_BLOCK_PATTERN.findall(llm_output)
    for path, content in alt_matches:
        path = path.strip()
        content = content.rstrip("\n")
        if path not in files:  # Don't overwrite
            files[path] = content
    
    return files


def validate_openfoam_structure(files: dict[str, str]) -> list[str]:
    """Validate that the files form a valid OpenFOAM case structure.
    
    Args:
        files: Dictionary of file paths to content
        
    Returns:
        List of validation warnings (empty if valid)
    """
    warnings = []
    
    # Required directories
    required_files = {
        "system/controlDict": "Missing system/controlDict",
        "system/fvSchemes": "Missing system/fvSchemes", 
        "system/fvSolution": "Missing system/fvSolution",
    }
    
    for required, message in required_files.items():
        if not any(p == required or p.endswith(f"/{required}") for p in files.keys()):
            warnings.append(message)
    
    # Check for initial conditions (0 directory)
    has_initial = any(
        p.startswith("0/") or p.startswith("0.orig/") or "/0/" in p
        for p in files.keys()
    )
    if not has_initial:
        warnings.append("No initial conditions (0/ directory) found")
    
    # Check for mesh or blockMeshDict
    has_mesh = any(
        "blockMeshDict" in p or "polyMesh" in p
        for p in files.keys()
    )
    if not has_mesh:
        warnings.append("No mesh definition (blockMeshDict or polyMesh) found")
    
    return warnings


def generate_run_script(
    solver: str = "simpleFoam",
    quick_run: bool = True,
) -> str:
    """Generate the run.sh script.
    
    Args:
        solver: The OpenFOAM solver to use
        quick_run: Whether to enable quick run mode
        
    Returns:
        Run script content
    """
    script = RUN_SCRIPT_TEMPLATE.format(solver=solver)
    return script


def package_case(
    files: dict[str, str],
    solver: str = "simpleFoam",
    case_name: str = "case",
    include_run_script: bool = True,
) -> tuple[bytes, list[str]]:
    """Package OpenFOAM case files into a zip archive.
    
    Args:
        files: Dictionary mapping relative paths to file content
        solver: Solver name for run script
        case_name: Root folder name in the zip
        include_run_script: Whether to include run.sh
        
    Returns:
        Tuple of (zip_bytes, file_list)
    """
    # Validate structure
    warnings = validate_openfoam_structure(files)
    for w in warnings:
        logger.warning(f"Case structure warning: {w}")
    
    # Create zip in memory
    zip_buffer = io.BytesIO()
    file_list = []
    
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add case files
        for path, content in files.items():
            # Normalize path
            path = path.lstrip("/")
            if not path.startswith(case_name):
                full_path = f"{case_name}/{path}"
            else:
                full_path = path
            
            zf.writestr(full_path, content)
            file_list.append(full_path)
        
        # Add run script if requested
        if include_run_script:
            run_script = generate_run_script(solver=solver)
            run_path = f"{case_name}/run.sh"
            zf.writestr(run_path, run_script)
            file_list.append(run_path)
    
    zip_buffer.seek(0)
    return zip_buffer.read(), file_list


def package_from_llm_output(
    llm_output: str,
    solver: str = "simpleFoam",
    case_name: str = "case",
) -> tuple[bytes, list[str], list[str]]:
    """Package an OpenFOAM case from raw LLM output.
    
    Args:
        llm_output: Raw LLM output containing file blocks
        solver: Solver name for run script
        case_name: Root folder name in the zip
        
    Returns:
        Tuple of (zip_bytes, file_list, warnings)
    """
    # Extract files from LLM output
    files = extract_file_blocks(llm_output)
    
    if not files:
        raise PackagingError("No file blocks found in LLM output")
    
    # Validate
    warnings = validate_openfoam_structure(files)
    
    # Package
    zip_bytes, file_list = package_case(
        files=files,
        solver=solver,
        case_name=case_name,
        include_run_script=True,
    )
    
    return zip_bytes, file_list, warnings


def extract_solver_from_controldict(content: str) -> str | None:
    """Try to extract solver name from controlDict.
    
    Args:
        content: controlDict file content
        
    Returns:
        Solver name or None
    """
    # Look for application keyword
    match = re.search(r"application\s+(\w+)\s*;", content)
    if match:
        return match.group(1)
    return None


def merge_files(
    base_files: dict[str, str],
    patch_files: dict[str, str],
) -> dict[str, str]:
    """Merge patch files into base files.
    
    Patch files override base files with the same path.
    
    Args:
        base_files: Original file set
        patch_files: Files to merge/override
        
    Returns:
        Merged file dictionary
    """
    result = dict(base_files)
    result.update(patch_files)
    return result
