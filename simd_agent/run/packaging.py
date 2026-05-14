# simd_agent/packaging.py
"""Package generated OpenFOAM case files into a zip archive."""

import io
import logging
import re
import zipfile

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

# Standard run.sh template for OpenFOAM cases (with mesh conversion)
RUN_SCRIPT_TEMPLATE = """#!/bin/bash
# Generated run script for OpenFOAM case
set -e

# Source OpenFOAM environment
if [ -f /opt/openfoam/etc/bashrc ]; then
    source /opt/openfoam/etc/bashrc
elif [ -f /usr/lib/openfoam/openfoam*/etc/bashrc ]; then
    source /usr/lib/openfoam/openfoam*/etc/bashrc
fi

echo "=== Starting OpenFOAM case execution ==="
echo "Working directory: $(pwd)"

# Step 1: Mesh conversion (if mesh file provided)
{mesh_conversion_commands}

# Step 2: Run checkMesh
if command -v checkMesh &> /dev/null && [ -d constant/polyMesh ]; then
    echo "=== Running checkMesh ==="
    checkMesh || echo "checkMesh reported issues, continuing..."
fi

# Step 3: Run the solver
SOLVER="{solver}"
echo "=== Running solver: $SOLVER ==="
$SOLVER

echo "=== Case execution complete ==="
exit 0
"""

# Mesh conversion commands template
MESH_CONVERSION_TEMPLATE = """
# Convert external mesh to OpenFOAM format
if [ -f "{mesh_filename}" ]; then
    echo "=== Converting mesh: {mesh_filename} ==="
    {conversion_command}
    
    # Verify mesh was created
    if [ ! -d constant/polyMesh ]; then
        echo "ERROR: Mesh conversion failed - constant/polyMesh not created"
        exit 1
    fi
    echo "=== Mesh conversion successful ==="
    
    # Post-mesh-conversion fixes
    echo "=== Running post-mesh-conversion fixes ==="
    if [ -f fix_mesh_setup.sh ]; then
        bash fix_mesh_setup.sh
    fi
else
    echo "No external mesh file found, expecting blockMeshDict or existing polyMesh"
fi
"""

# ────────────────────────────────────────────────────────────
# Post-mesh-conversion fix script
# ────────────────────────────────────────────────────────────
# This script runs AFTER gmshToFoam / fluentMeshToFoam and fixes:
# 1. constant/polyMesh/boundary: ensures "wall" patches have type=wall & inGroups(wall)
# 2. Reads actual boundary patches and ensures all 0/* files have matching entries
#    (adds frontAndBack with type empty if present in mesh but missing from 0/* files)
# 3. system/fvSchemes: adds wallDist { method meshWave; } if missing
# 4. Removes any invented "front_and_back" (underscore) references from 0/* files
FIX_MESH_SETUP_SCRIPT = r"""#!/bin/bash
# fix_mesh_setup.sh — Post-mesh-conversion fixes for OpenFOAM cases
# This script is auto-generated and runs after mesh conversion (gmshToFoam, etc.)
set -e

BOUNDARY_FILE="constant/polyMesh/boundary"
FVSCHEMES_FILE="system/fvSchemes"

echo "--- Post-mesh-conversion fix script ---"

# ─── ALL FIXES via single Python script for reliability ───
if [ -f "$BOUNDARY_FILE" ]; then
    python3 - "$BOUNDARY_FILE" "$FVSCHEMES_FILE" << 'PYEOF'
import sys
import re
import os
import glob

boundary_file = sys.argv[1]
fvschemes_file = sys.argv[2]

# ================================================================
# Fix 1: Fix wall patch type in boundary file
# ================================================================
print("  [Fix 1] Checking boundary file for wall patch type...")

with open(boundary_file, 'r') as f:
    boundary_content = f.read()

original_boundary = boundary_content

# Fix: change type from 'patch' to 'wall' for the patch named 'wall'
wall_patch_pattern = re.compile(
    r'(\s+wall\s*\n\s*\{[^}]*?type\s+)patch(\s*;)',
    re.MULTILINE | re.DOTALL,
)
boundary_content = wall_patch_pattern.sub(r'\1wall\2', boundary_content)

# Fix physicalType too if present
wall_phystype_pattern = re.compile(
    r'(\s+wall\s*\n\s*\{[^}]*?physicalType\s+)patch(\s*;)',
    re.MULTILINE | re.DOTALL,
)
boundary_content = wall_phystype_pattern.sub(r'\1wall\2', boundary_content)

# Ensure inGroups contains 'wall' in the wall patch block
wall_block_pattern = re.compile(
    r'(wall\s*\n\s*\{[^}]*?)inGroups\s+\d+\([^)]*\)',
    re.MULTILINE | re.DOTALL,
)
def fix_wall_ingroups(match):
    prefix = match.group(1)
    return prefix + 'inGroups        1(wall)'
boundary_content = wall_block_pattern.sub(fix_wall_ingroups, boundary_content)

# If wall patch has type wall but no inGroups, add it
wall_type_no_ingroups = re.compile(
    r'(\s+wall\s*\n\s*\{\s*\n\s*type\s+wall\s*;\s*\n)(\s*(?:physicalType|nFaces))',
    re.MULTILINE,
)
match = wall_type_no_ingroups.search(boundary_content)
if match:
    # Check if inGroups is already in the wall block
    wall_start = boundary_content.find('\n    wall\n')
    if wall_start >= 0:
        wall_block_end = boundary_content.find('}', wall_start)
        wall_block = boundary_content[wall_start:wall_block_end]
        if 'inGroups' not in wall_block:
            boundary_content = wall_type_no_ingroups.sub(
                r'\1        inGroups        1(wall);\n\2',
                boundary_content,
                count=1,
            )

if boundary_content != original_boundary:
    with open(boundary_file, 'w') as f:
        f.write(boundary_content)
    print(f"    Fixed wall patch type in {boundary_file}")
else:
    print(f"    Wall patch type already correct")

# ================================================================
# Fix 2: Read boundary patches and sync with 0/* field files
# ================================================================
print("  [Fix 2] Reading boundary patches and syncing with 0/* files...")

# Parse patch names and types from boundary file
patch_pattern = re.compile(
    r'^\s{4}(\w+)\s*\n\s*\{[^}]*?type\s+(\w+)\s*;',
    re.MULTILINE,
)
boundary_patches = {}
for match in patch_pattern.finditer(boundary_content):
    name = match.group(1)
    ptype = match.group(2)
    boundary_patches[name] = ptype

print(f"    Boundary patches found: {boundary_patches}")

# For each 0/* field file, ensure all boundary patches are present
field_files = glob.glob('0/*')
for field_file in field_files:
    if not os.path.isfile(field_file):
        continue
    
    with open(field_file, 'r') as f:
        content = f.read()
    
    if 'boundaryField' not in content:
        continue
    
    original_content = content
    field_name = os.path.basename(field_file)
    
    # Remove invented front_and_back (underscore) patches
    fab_pattern = re.compile(
        r'\n\s{4}front_and_back\s*\n\s*\{[^}]*\}\n?',
        re.MULTILINE,
    )
    content = fab_pattern.sub('\n', content)
    
    # Check which boundary patches are missing from this file
    for patch_name, patch_type in boundary_patches.items():
        # Check if this patch is already in the file
        patch_check = re.search(
            rf'^\s{{4}}{re.escape(patch_name)}\s*$',
            content,
            re.MULTILINE,
        )
        if patch_check:
            continue
        
        # Patch is missing — add it before the closing } of boundaryField
        print(f"    Adding missing patch '{patch_name}' (type={patch_type}) to {field_file}")
        
        # Determine the BC entry based on the mesh patch type
        def make_bc(pname, bc_type, extra=''):
            lines = '    ' + pname + '\n    {\n        type            ' + bc_type + ';'
            if extra:
                lines += '\n' + extra
            lines += '\n    }'
            return lines
        
        if patch_type == 'empty':
            bc_block = make_bc(patch_name, 'empty')
        elif patch_type in ('symmetry', 'symmetryPlane'):
            bc_block = make_bc(patch_name, 'symmetry')
        elif patch_type == 'wall':
            # Wall BCs depend on the field
            wall_bcs = {
                'U': ('noSlip', ''),
                'p': ('zeroGradient', ''),
                'nut': ('nutkWallFunction', '        value           uniform 0;'),
                'k': ('kqRWallFunction', '        value           uniform 0.001;'),
                'omega': ('omegaWallFunction', '        value           uniform 1;'),
                'epsilon': ('epsilonWallFunction', '        value           uniform 0.001;'),
                'T': ('zeroGradient', ''),
            }
            bc_info = wall_bcs.get(field_name, ('zeroGradient', ''))
            bc_block = make_bc(patch_name, bc_info[0], bc_info[1])
        else:
            bc_block = make_bc(patch_name, 'zeroGradient')
        
        # Insert before the last closing brace of boundaryField
        # Find the last } in the file (closing boundaryField)
        last_brace = content.rfind('}')
        if last_brace > 0:
            content = content[:last_brace] + bc_block + '\n' + content[last_brace:]
    
    if content != original_content:
        with open(field_file, 'w') as f:
            f.write(content)
        print(f"    Updated {field_file}")

# ================================================================
# Fix 3: Add wallDist to fvSchemes if missing
# ================================================================
print("  [Fix 3] Checking fvSchemes for wallDist...")

if os.path.isfile(fvschemes_file):
    with open(fvschemes_file, 'r') as f:
        schemes_content = f.read()
    
    if 'wallDist' not in schemes_content:
        schemes_content = schemes_content.rstrip() + '\n\nwallDist\n{\n    method meshWave;\n}\n'
        with open(fvschemes_file, 'w') as f:
            f.write(schemes_content)
        print("    Added wallDist { method meshWave; } to fvSchemes")
    else:
        print("    wallDist already present in fvSchemes")
else:
    print(f"    WARNING: {fvschemes_file} not found")

print("--- Post-mesh-conversion fixes complete ---")
PYEOF

else
    echo "  No boundary file found, skipping boundary fixes"
    
    # Still do wallDist fix even without boundary file
    if [ -f "$FVSCHEMES_FILE" ]; then
        if ! grep -q "wallDist" "$FVSCHEMES_FILE"; then
            echo "  Adding wallDist block to $FVSCHEMES_FILE..."
            cat >> "$FVSCHEMES_FILE" << 'WALLDIST'

wallDist
{
    method meshWave;
}
WALLDIST
            echo "  wallDist block added"
        fi
    fi
fi
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

    # Note: mesh presence is NOT checked here because all cases use an external
    # mesh converted by gmshToFoam.  blockMeshDict is intentionally never generated.

    return warnings


def generate_run_script(
    solver: str = "simpleFoam",
    mesh_filename: str | None = None,
    mesh_format: str | None = None,
) -> str:
    """Generate the run.sh script.
    
    Args:
        solver: The OpenFOAM solver to use
        mesh_filename: Name of the mesh file (if external mesh provided)
        mesh_format: Format of the mesh file (msh, stl, etc.)
        
    Returns:
        Run script content
    """
    # Generate mesh conversion commands if mesh provided
    mesh_commands = ""
    if mesh_filename and mesh_format:
        conversion_cmd = _get_conversion_command(mesh_format, mesh_filename)
        if conversion_cmd:
            mesh_commands = MESH_CONVERSION_TEMPLATE.format(
                mesh_filename=mesh_filename,
                conversion_command=conversion_cmd,
            )
    
    script = RUN_SCRIPT_TEMPLATE.format(
        solver=solver,
        mesh_conversion_commands=mesh_commands,
    )
    return script


def _get_conversion_command(mesh_format: str, mesh_filename: str) -> str:
    """Get the mesh conversion command for a given format.
    
    Args:
        mesh_format: File format (without dot)
        mesh_filename: Name of the mesh file
        
    Returns:
        Shell command for mesh conversion
    """
    fmt = mesh_format.lower()
    
    if fmt in ("msh",):
        # Gmsh mesh — gmshToFoam handles both Gmsh 2.x and 4.x format
        return f"gmshToFoam {mesh_filename}"
    elif fmt in ("cas",):
        # Fluent case
        return f"fluentMeshToFoam {mesh_filename}"
    elif fmt in ("cgns",):
        return f"cgnsToFoam {mesh_filename}"
    elif fmt in ("stl",):
        # STL requires snappyHexMesh workflow - more complex
        return "blockMesh && surfaceFeatureExtract && snappyHexMesh -overwrite"
    elif fmt in ("unv",):
        return f"ideasUnvToFoam {mesh_filename}"
    elif fmt in ("neu",):
        # Gambit neutral file
        return f"gambitToFoam {mesh_filename}"
    else:
        # Default to fluent converter for unknown formats
        return f"fluentMeshToFoam {mesh_filename}"


def package_case(
    files: dict[str, str],
    solver: str = "simpleFoam",
    case_name: str = "case",
    include_local_helpers: bool = False,
    mesh_bytes: bytes | None = None,
    mesh_filename: str | None = None,
    mesh_format: str | None = None,
) -> tuple[bytes, list[str]]:
    """Package OpenFOAM case files into a zip archive.

    Args:
        files: Dictionary mapping relative paths to file content
        solver: Solver name for run script
        case_name: Root folder name in the zip
        include_local_helpers: Include run.sh + fix_mesh_setup.sh for
            local execution by an end user. The simulation runner server
            never executes these (it orchestrates mesh conversion, MPI
            decomposition, the solver, and reconstruction itself in
            agent-simulation/app/runner.py), so the default is False.
            Set True only for the /api/runs/{id}/export download path.
        mesh_bytes: Optional mesh file bytes to include
        mesh_filename: Original mesh filename
        mesh_format: Mesh format (msh, stl, etc.)

    Returns:
        Tuple of (zip_bytes, file_list)
    """
    # Validate structure (skip mesh check if external mesh provided)
    warnings = validate_openfoam_structure(files)
    if mesh_bytes:
        # Remove mesh warning if we have external mesh
        warnings = [w for w in warnings if "mesh" not in w.lower()]
    for w in warnings:
        logger.warning(f"Case structure warning: {w}")
    
    # Create zip in memory
    zip_buffer = io.BytesIO()
    file_list = []
    
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add mesh file if provided
        if mesh_bytes and mesh_filename:
            mesh_path = f"{case_name}/{mesh_filename}"
            zf.writestr(mesh_path, mesh_bytes)
            file_list.append(mesh_path)
            logger.info(f"Added mesh file to package: {mesh_path}")
        
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
        
        if include_local_helpers:
            run_script = generate_run_script(
                solver=solver,
                mesh_filename=mesh_filename,
                mesh_format=mesh_format,
            )
            run_path = f"{case_name}/run.sh"
            zf.writestr(run_path, run_script)
            file_list.append(run_path)

            fix_script_path = f"{case_name}/fix_mesh_setup.sh"
            zf.writestr(fix_script_path, FIX_MESH_SETUP_SCRIPT)
            file_list.append(fix_script_path)
    
    zip_buffer.seek(0)
    return zip_buffer.read(), file_list


def package_from_llm_output(
    llm_output: str,
    solver: str = "simpleFoam",
    case_name: str = "case",
    mesh_bytes: bytes | None = None,
    mesh_filename: str | None = None,
    mesh_format: str | None = None,
) -> tuple[bytes, list[str], list[str]]:
    """Package an OpenFOAM case from raw LLM output.
    
    Args:
        llm_output: Raw LLM output containing file blocks
        solver: Solver name for run script
        case_name: Root folder name in the zip
        mesh_bytes: Optional mesh file bytes to include
        mesh_filename: Original mesh filename
        mesh_format: Mesh format (msh, stl, etc.)
        
    Returns:
        Tuple of (zip_bytes, file_list, warnings)
    """
    # Extract files from LLM output
    files = extract_file_blocks(llm_output)
    
    if not files:
        raise PackagingError("No file blocks found in LLM output")
    
    # Validate
    warnings = validate_openfoam_structure(files)
    if mesh_bytes:
        # Remove mesh warning if external mesh provided
        warnings = [w for w in warnings if "mesh" not in w.lower()]
    
    # Package — slim ZIP for the simulation runner server (it ignores
    # run.sh / fix_mesh_setup.sh and orchestrates execution itself).
    zip_bytes, file_list = package_case(
        files=files,
        solver=solver,
        case_name=case_name,
        mesh_bytes=mesh_bytes,
        mesh_filename=mesh_filename,
        mesh_format=mesh_format,
    )

    return zip_bytes, file_list, warnings


def package_simulation_case(
    generated_files: dict[str, str],
    mesh_id: str,
    solver: str = "simpleFoam",
    case_name: str = "case",
) -> tuple[bytes, list[str], list[str]]:
    """Package a complete simulation case with mesh from storage.

    Main entry point for the simulation-runner-server submission path:
    1. Generated OpenFOAM files (0/, system/, constant/)
    2. Mesh file retrieved from storage

    The sim server orchestrates mesh conversion, MPI decomposition, the
    solver, and reconstruction itself, so no run.sh / fix_mesh_setup.sh
    is included.

    Args:
        generated_files: Dictionary of generated OpenFOAM files
        mesh_id: Mesh ID from /api/mesh/convert
        solver: OpenFOAM solver to use
        case_name: Root folder name in zip

    Returns:
        Tuple of (zip_bytes, file_list, warnings)
    """
    from simd_agent.run.mesh_retriever import get_mesh_file, MeshNotFoundError

    try:
        mesh_bytes, mesh_filename, mesh_format = get_mesh_file(mesh_id)
        logger.info(f"Retrieved mesh: {mesh_filename} ({mesh_format} format, {len(mesh_bytes)} bytes)")
    except MeshNotFoundError as e:
        logger.error(f"Failed to retrieve mesh: {e}")
        mesh_bytes = None
        mesh_filename = None
        mesh_format = None

    warnings = validate_openfoam_structure(generated_files)
    if mesh_bytes:
        warnings = [w for w in warnings if "mesh" not in w.lower()]

    zip_bytes, file_list = package_case(
        files=generated_files,
        solver=solver,
        case_name=case_name,
        mesh_bytes=mesh_bytes,
        mesh_filename=mesh_filename,
        mesh_format=mesh_format,
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
