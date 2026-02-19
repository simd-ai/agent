# simd_agent/genai_codegen.py
"""OpenFOAM code generator using Google GenAI directly.

Generates complete OpenFOAM case files via the Google GenAI API.
Includes post-generation validation to catch inconsistencies before
sending to the simulation server.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from simd_agent.settings import get_settings

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts" / "packs" / "simd"

# Solvers we support (no buoyancy for now)
ALLOWED_SOLVERS = {"simpleFoam", "pimpleFoam"}
BUOYANT_SOLVERS = {"buoyantSimpleFoam", "buoyantPimpleFoam"}


# ────────────────────────────────────────────────────────────
# Post-generation validation
# ────────────────────────────────────────────────────────────

class ValidationIssue:
    """An issue found during post-generation validation."""
    def __init__(self, severity: str, file: str, message: str, fix: str | None = None):
        self.severity = severity  # "error" | "warning"
        self.file = file
        self.message = message
        self.fix = fix

    def __repr__(self):
        return f"[{self.severity}] {self.file}: {self.message}"


def validate_generated_files(
    files: dict[str, str],
    solver: str,
    config: dict[str, Any],
) -> tuple[dict[str, str], list[ValidationIssue]]:
    """Validate and auto-fix generated OpenFOAM files for consistency.
    
    Checks:
    1. controlDict application matches expected solver
    2. If solver is simpleFoam/pimpleFoam, ensure 0/p exists (not p_rgh)
    3. All patch names appear in every 0/* field file
    4. No buoyant-only files if solver is not buoyant
    5. Turbulence fields match selected model
    
    Returns:
        Tuple of (possibly-fixed files dict, list of issues found)
    """
    issues: list[ValidationIssue] = []
    fixed_files = dict(files)
    
    # Get expected patch names from config
    bcs = config.get("boundary_conditions", {})
    expected_patches = set(bcs.keys())
    
    # Get physics
    physics = config.get("physics", {})
    heat_transfer = physics.get("heat_transfer", False)
    turb_model = physics.get("turbulence_model", "kOmegaSST")
    
    # ── Check 1: controlDict solver ──
    control_dict = fixed_files.get("system/controlDict", "")
    if control_dict:
        app_match = re.search(r'application\s+(\w+)\s*;', control_dict)
        if app_match:
            declared_solver = app_match.group(1)
            if declared_solver in BUOYANT_SOLVERS:
                # AUTO-FIX: Replace buoyant solver with non-buoyant equivalent
                replacement = "simpleFoam" if "Simple" in declared_solver else "pimpleFoam"
                issues.append(ValidationIssue(
                    "error", "system/controlDict",
                    f"Buoyant solver '{declared_solver}' not supported, replacing with '{replacement}'",
                    fix=f"application {replacement};"
                ))
                fixed_files["system/controlDict"] = control_dict.replace(
                    f"application     {declared_solver};",
                    f"application     {replacement};"
                ).replace(
                    f"application {declared_solver};",
                    f"application {replacement};"
                )
                solver = replacement
            elif declared_solver != solver:
                # Solver mismatch with what planner chose
                if declared_solver in ALLOWED_SOLVERS:
                    solver = declared_solver  # Trust the LLM if it's a valid solver
                else:
                    issues.append(ValidationIssue(
                        "error", "system/controlDict",
                        f"Unknown solver '{declared_solver}', replacing with '{solver}'",
                    ))
                    fixed_files["system/controlDict"] = re.sub(
                        r'application\s+\w+\s*;',
                        f'application     {solver};',
                        control_dict
                    )
    
    # ── Check 2: p vs p_rgh ──
    if solver in ALLOWED_SOLVERS:
        # These solvers need 0/p, NOT 0/p_rgh
        if "0/p_rgh" in fixed_files and "0/p" not in fixed_files:
            issues.append(ValidationIssue(
                "error", "0/p_rgh",
                f"'{solver}' requires 0/p, not 0/p_rgh. Renaming.",
                fix="Renamed 0/p_rgh → 0/p"
            ))
            content = fixed_files.pop("0/p_rgh")
            # Fix the object name inside the file
            content = content.replace("object      p_rgh;", "object      p;")
            content = content.replace("object p_rgh;", "object p;")
            fixed_files["0/p"] = content
        
        if "0/p_rgh" in fixed_files and "0/p" in fixed_files:
            # Both exist — remove p_rgh
            issues.append(ValidationIssue(
                "warning", "0/p_rgh",
                "Both 0/p and 0/p_rgh exist. Removing 0/p_rgh.",
            ))
            del fixed_files["0/p_rgh"]
    
    # ── Check 3: Remove buoyant-only files ──
    if solver in ALLOWED_SOLVERS:
        buoyant_files = ["constant/thermophysicalProperties", "constant/g"]
        for bf in buoyant_files:
            if bf in fixed_files:
                issues.append(ValidationIssue(
                    "warning", bf,
                    f"'{bf}' not needed for {solver}. Removing.",
                ))
                del fixed_files[bf]
    
    # ── Check 3b: Remove invented "front_and_back" patches ──
    # The LLM sometimes invents a "front_and_back" (with underscores) patch
    # that doesn't exist in the mesh. Only "frontAndBack" (camelCase) is real.
    invented_patch_pattern = re.compile(
        r'\n\s{4}front_and_back\s*\n\s*\{\s*\n(?:\s+\w+[^}]*\n)*?\s*\}\n?',
        re.MULTILINE,
    )
    for file_path, content in list(fixed_files.items()):
        if not file_path.startswith("0/"):
            continue
        if "front_and_back" in content:
            new_content = invented_patch_pattern.sub('\n', content)
            if new_content != content:
                issues.append(ValidationIssue(
                    "warning", file_path,
                    "Removed invented 'front_and_back' patch (only 'frontAndBack' exists in mesh).",
                    fix="Removed front_and_back block"
                ))
                fixed_files[file_path] = new_content
    
    # ── Extract mesh patch info (used by Check 3c and 4b) ──
    mesh_info = config.get("mesh", {})
    if isinstance(mesh_info, str):
        mesh_info = {}
    mesh_patches_list = mesh_info.get("patches", []) if isinstance(mesh_info, dict) else []
    
    # ── Check 3c: Auto-add frontAndBack (empty) to 0/* files for 2D meshes ──
    # If mesh has a frontAndBack patch of type empty, ensure all 0/* files include it.
    # This is critical for 2D simulations from Gmsh.
    mesh_has_front_and_back = False
    if mesh_patches_list:
        for mp in mesh_patches_list:
            mp_name = mp.get("name", "") if isinstance(mp, dict) else getattr(mp, "name", "")
            mp_type = mp.get("type", "") if isinstance(mp, dict) else getattr(mp, "type", "")
            if mp_name.lower() == "frontandback" and mp_type == "empty":
                mesh_has_front_and_back = True
                break
    
    if mesh_has_front_and_back:
        front_and_back_block = "\n    frontAndBack\n    {\n        type            empty;\n    }\n"
        for file_path, content in list(fixed_files.items()):
            if not file_path.startswith("0/"):
                continue
            if "boundaryField" not in content:
                continue
            if "frontAndBack" in content:
                continue  # Already present
            
            # Insert before the closing } of boundaryField
            last_brace = content.rfind("}")
            if last_brace > 0:
                fixed_files[file_path] = content[:last_brace] + front_and_back_block + content[last_brace:]
                issues.append(ValidationIssue(
                    "warning", file_path,
                    "Added missing 'frontAndBack' (empty) patch for 2D mesh.",
                    fix="Added frontAndBack { type empty; }"
                ))
    
    # ── Check 4: Patch names in 0/* files ──
    if expected_patches:
        for file_path, content in list(fixed_files.items()):
            if not file_path.startswith("0/"):
                continue
            
            # Check if file has boundaryField
            if "boundaryField" not in content:
                continue
            
            # Extract patch names from the file
            file_patches = set(re.findall(r'^\s{4}(\w+)\s*$', content, re.MULTILINE))
            
            missing = expected_patches - file_patches
            if missing:
                issues.append(ValidationIssue(
                    "warning", file_path,
                    f"Missing patches: {missing}. These must be defined or OpenFOAM will crash.",
                ))
    
    # ── Check 4b: Constraint type vs mesh patch type ──
    # 'empty' and 'symmetry' BCs can only be used if the mesh patch is actually that type.
    # If the mesh has a patch of type 'patch' or 'wall', using 'empty' or 'symmetry' will crash.
    if mesh_patches_list:
        # Build a map of patch_name -> mesh_type
        mesh_patch_types = {}
        for mp in mesh_patches_list:
            if isinstance(mp, dict):
                mesh_patch_types[mp.get("name", "")] = mp.get("type", "patch")
            elif hasattr(mp, "name"):
                mesh_patch_types[mp.name] = mp.type
        
        # Check all 0/* files for constraint type mismatches
        constraint_types = {"empty", "symmetry", "wedge", "cyclic", "processor"}
        for file_path, content in list(fixed_files.items()):
            if not file_path.startswith("0/"):
                continue
            if "boundaryField" not in content:
                continue
            
            for patch_name, mesh_type in mesh_patch_types.items():
                # Find the BC type for this patch in this file
                # Pattern: patch_name\n    {\n        type    <type>;
                bc_match = re.search(
                    rf'{re.escape(patch_name)}\s*\n\s*\{{\s*\n\s*type\s+(\w+)\s*;',
                    content
                )
                if not bc_match:
                    continue
                
                bc_type = bc_match.group(1)
                
                # If BC uses a constraint type but mesh patch is NOT that type → error
                if bc_type in constraint_types and mesh_type != bc_type:
                    # AUTO-FIX: Replace constraint type with a safe default
                    if bc_type == "empty" and mesh_type in ("patch", "wall"):
                        # Replace 'empty' with 'zeroGradient' (safe default)
                        safe_type = "zeroGradient"
                        issues.append(ValidationIssue(
                            "error", file_path,
                            f"Patch '{patch_name}' has mesh type '{mesh_type}' but BC type 'empty'. "
                            f"'empty' requires mesh type 'empty'. Replacing with '{safe_type}'.",
                            fix=f"type {safe_type};"
                        ))
                        # Do the replacement
                        old_block = bc_match.group(0)
                        new_block = old_block.replace(f"type            {bc_type};", f"type            {safe_type};")
                        new_block = new_block.replace(f"type {bc_type};", f"type {safe_type};")
                        fixed_files[file_path] = content.replace(old_block, new_block)
                        content = fixed_files[file_path]  # Update for further checks
                    
                    elif bc_type == "symmetry" and mesh_type not in ("symmetry", "symmetryPlane"):
                        safe_type = "zeroGradient"
                        issues.append(ValidationIssue(
                            "error", file_path,
                            f"Patch '{patch_name}' has mesh type '{mesh_type}' but BC type 'symmetry'. "
                            f"Replacing with '{safe_type}'.",
                            fix=f"type {safe_type};"
                        ))
                        old_block = bc_match.group(0)
                        new_block = old_block.replace(f"type            {bc_type};", f"type            {safe_type};")
                        new_block = new_block.replace(f"type {bc_type};", f"type {safe_type};")
                        fixed_files[file_path] = content.replace(old_block, new_block)
                        content = fixed_files[file_path]
    
    # ── Check 5: Required field files ──
    required_files = ["system/controlDict", "system/fvSchemes", "system/fvSolution",
                      "0/U", "0/p", "constant/transportProperties"]
    
    if turb_model and turb_model not in ["laminar", "none", None]:
        required_files.append("constant/turbulenceProperties")
        required_files.append("0/nut")
        if "kOmega" in turb_model or "SST" in turb_model:
            required_files.extend(["0/k", "0/omega"])
        elif "kEpsilon" in turb_model:
            required_files.extend(["0/k", "0/epsilon"])
    
    if heat_transfer:
        required_files.append("0/T")
    
    for rf in required_files:
        if rf not in fixed_files:
            issues.append(ValidationIssue(
                "error", rf,
                f"Required file '{rf}' is missing from generated output.",
            ))
    
    # ── Check 6: fvSolution solver algorithm ──
    fv_solution = fixed_files.get("system/fvSolution", "")
    if fv_solution and solver == "simpleFoam":
        if "SIMPLE" not in fv_solution and "SIMPLE\n" not in fv_solution:
            issues.append(ValidationIssue(
                "warning", "system/fvSolution",
                "simpleFoam requires a SIMPLE block in fvSolution.",
            ))
    elif fv_solution and solver == "pimpleFoam":
        if "PIMPLE" not in fv_solution:
            issues.append(ValidationIssue(
                "warning", "system/fvSolution",
                "pimpleFoam requires a PIMPLE block in fvSolution.",
            ))
    
    # ── Check 7: wallDist in fvSchemes (required for kOmegaSST, kEpsilon, etc.) ──
    fv_schemes = fixed_files.get("system/fvSchemes", "")
    if fv_schemes and turb_model and turb_model not in ["laminar", "none", None]:
        if "wallDist" not in fv_schemes:
            issues.append(ValidationIssue(
                "warning", "system/fvSchemes",
                f"Missing 'wallDist' block required by turbulence model '{turb_model}'. Adding.",
                fix="Added wallDist { method meshWave; }"
            ))
            # Append wallDist block before the end of the file
            wall_dist_block = "\nwallDist\n{\n    method meshWave;\n}\n"
            fixed_files["system/fvSchemes"] = fv_schemes.rstrip() + "\n" + wall_dist_block
    
    # Log results
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]
    if issues:
        logger.info(f"[VALIDATE] {len(errors)} errors, {len(warnings)} warnings found")
        for issue in issues:
            logger.info(f"[VALIDATE]   {issue}")
    else:
        logger.info("[VALIDATE] All checks passed")
    
    return fixed_files, issues


def determine_solver(config: dict[str, Any]) -> str:
    """Determine the correct solver from validated config.
    
    Rules (simplified — no buoyancy):
    - Steady → simpleFoam
    - Transient → pimpleFoam
    """
    physics = config.get("physics", {})
    time_scheme = physics.get("time_scheme", "steady")
    
    if time_scheme == "transient":
        return "pimpleFoam"
    return "simpleFoam"


# ────────────────────────────────────────────────────────────
# Code Generator
# ────────────────────────────────────────────────────────────

class GenAICodeGenerator:
    """Generate OpenFOAM case files using Google GenAI.
    
    Replaces the codegen library with direct Google GenAI API calls.
    Includes post-generation validation to catch inconsistencies.
    """
    
    def __init__(self, model: str | None = None):
        settings = get_settings()
        self.model = model or settings.gemini_model
        
        api_key = (
            settings.gemini_api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        if not api_key:
            raise ValueError("GEMINI_API_KEY not configured in settings or environment")
        
        self.client = genai.Client(api_key=api_key)
        
        self._codegen_prompt = self._load_prompt("codegen.md")
        self._codefix_prompt = self._load_prompt("codefix.md")
    
    def _load_prompt(self, filename: str) -> str:
        prompt_path = PROMPTS_DIR / filename
        if prompt_path.exists():
            return prompt_path.read_text()
        logger.warning(f"Prompt file not found: {prompt_path}")
        return ""
    
    async def generate(
        self,
        requirements: str,
        validated_config: dict[str, Any],
        solver: str = "simpleFoam",
        case_type: str = "pipe_flow",
        previous_errors: list[dict] | None = None,
        previous_files: dict[str, str] | None = None,
    ) -> str:
        """Generate OpenFOAM case files.
        
        Returns:
            String containing file blocks in ```file:path format
        """
        # Force solver to non-buoyant
        if solver in BUOYANT_SOLVERS:
            solver = "simpleFoam" if "Simple" in solver else "pimpleFoam"
            logger.info(f"[GENAI] Forced solver to {solver} (no buoyancy)")
        
        if previous_errors:
            prompt = self._build_fix_prompt(
                requirements=requirements,
                validated_config=validated_config,
                solver=solver,
                previous_errors=previous_errors,
                previous_files=previous_files or {},
            )
        else:
            prompt = self._build_codegen_prompt(
                requirements=requirements,
                validated_config=validated_config,
                solver=solver,
                case_type=case_type,
            )
        
        logger.info(f"[GENAI] Generating code with model {self.model}, solver={solver}")
        
        response = await self._call_genai(prompt)
        logger.info(f"[GENAI] Generated response: {len(response)} chars")
        return response
    
    def _build_codegen_prompt(
        self,
        requirements: str,
        validated_config: dict[str, Any],
        solver: str,
        case_type: str,
    ) -> str:
        system_prompt = self._codegen_prompt
        
        # Build mesh patch type info for the LLM
        mesh_patch_info = ""
        mesh_data = validated_config.get("mesh", {})
        # Handle mesh being a string (just mesh_id) or a dict
        if isinstance(mesh_data, str):
            mesh_data = {}
        mesh_patches = mesh_data.get("patches", []) if isinstance(mesh_data, dict) else []
        if mesh_patches:
            lines = []
            for mp in mesh_patches:
                if isinstance(mp, dict):
                    name = mp.get("name", "?")
                    mtype = mp.get("type", "patch")
                else:
                    name = getattr(mp, "name", "?")
                    mtype = getattr(mp, "type", "patch")
                lines.append(f"  - `{name}`: mesh type = `{mtype}`")
            mesh_patch_info = "\n".join(lines)
        
        constraint_warning = ""
        if mesh_patch_info:
            constraint_warning = f"""
## Mesh Patch Types (CRITICAL — constraint type mismatch = crash)
{mesh_patch_info}

**RULE**: You can ONLY use `type empty;` if the mesh patch type is `empty`.
You can ONLY use `type symmetry;` if the mesh patch type is `symmetry` or `symmetryPlane`.
If the mesh patch type is `patch`, use `zeroGradient` or `fixedValue` — NEVER `empty` or `symmetry`.
If the mesh patch type is `wall`, use `noSlip` for U, `zeroGradient` for p, wall functions for turbulence.
"""
        
        # Detect 2D simulation: if mesh has frontAndBack (empty type) or is a 2D Gmsh mesh
        # For 2D sims, we MUST include frontAndBack in all field files
        is_2d = False
        has_front_and_back_in_mesh = False
        for mp in mesh_patches:
            name = mp.get("name", "") if isinstance(mp, dict) else getattr(mp, "name", "")
            mtype = mp.get("type", "") if isinstance(mp, dict) else getattr(mp, "type", "")
            if name.lower() in ("frontandback", "front_and_back") and mtype == "empty":
                has_front_and_back_in_mesh = True
                is_2d = True
        
        # For 2D Gmsh meshes, gmshToFoam will create frontAndBack as empty
        # We should always include it since most Gmsh meshes are 2D
        two_d_warning = ""
        if is_2d or has_front_and_back_in_mesh:
            two_d_warning = """
## 2D Simulation — frontAndBack (CRITICAL)
This is a 2D simulation. After mesh conversion, there will be a `frontAndBack` patch of type `empty`.
You **MUST** include `frontAndBack` with `type empty;` in ALL `0/*` field files (U, p, T, k, omega, nut, epsilon).
"""
        else:
            # Even if not explicitly 2D, Gmsh meshes often produce frontAndBack
            two_d_warning = """
## 2D Mesh Note
This mesh is from Gmsh. After `gmshToFoam` conversion, the mesh will likely have a `frontAndBack` patch of type `empty`.
Include `frontAndBack` with `type empty;` in ALL `0/*` field files to be safe.
A post-conversion fix script will also add any missing patches, but it's best to include them upfront.
"""
        constraint_warning += two_d_warning
        
        user_message = f"""## Task
Generate a complete OpenFOAM case for the following simulation.

## User Requirements
{requirements}

## Validated Configuration
```json
{json.dumps(validated_config, indent=2, default=str)}
```

## Selected Solver: {solver}
## Case Type: {case_type}
{constraint_warning}
## CRITICAL RULES
1. The `application` in controlDict MUST be `{solver}`
2. You MUST generate `0/p` (NOT `0/p_rgh`) — {solver} reads `0/p`
3. Every patch in boundary_conditions MUST appear in EVERY 0/* field file
4. Do NOT generate blockMeshDict — we use an external mesh
5. Do NOT generate thermophysicalProperties or constant/g
6. Output files using ```file:path/to/file format
7. NEVER use `type empty;` unless the mesh patch type is `empty`
8. NEVER use `type symmetry;` unless the mesh patch type is `symmetry` or `symmetryPlane`
9. Do NOT invent patch names like `front_and_back` — only use patches from the config
10. If using a turbulence model (kOmegaSST, kEpsilon, etc.), fvSchemes MUST include: wallDist {{ method meshWave; }}

Generate the OpenFOAM case files now:"""
        
        return f"{system_prompt}\n\n{user_message}"
    
    def _build_fix_prompt(
        self,
        requirements: str,
        validated_config: dict[str, Any],
        solver: str,
        previous_errors: list[dict],
        previous_files: dict[str, str],
    ) -> str:
        system_prompt = self._codefix_prompt
        
        files_str = "\n\n".join([
            f"```file:{path}\n{content}\n```"
            for path, content in previous_files.items()
        ])
        
        errors_str = "\n".join([
            f"- [{e.get('source', 'unknown')}] {e.get('error', 'Unknown error')}"
            + (f"\n  Details: {e.get('details', '')}" if e.get('details') else "")
            + (f"\n  Stderr: {e.get('stderr', '')[:2000]}" if e.get('stderr') else "")
            for e in previous_errors
        ])
        
        user_message = f"""## Task
Fix the OpenFOAM case that failed during execution on the simulation server.

## Original Requirements
{requirements}

## Validated Configuration
```json
{json.dumps(validated_config, indent=2, default=str)}
```

## Selected Solver: {solver}

## Previous Files (that failed)
{files_str}

## Errors from Simulation Server
{errors_str}

## CRITICAL FIX RULES
1. The `application` in controlDict MUST be `{solver}`
2. `{solver}` reads `0/p` — if you see "cannot find file 0/p", generate `0/p` NOT `0/p_rgh`
3. Every patch MUST appear in EVERY 0/* field file
4. Do NOT use buoyantSimpleFoam or buoyantPimpleFoam
5. Check that ALL field files referenced by the solver exist
6. If error says "not constraint type 'empty'" — the mesh patch is type 'patch', NOT 'empty'. Use `zeroGradient` instead of `empty`.
7. If error says "not constraint type 'symmetry'" — the mesh patch is type 'patch', NOT 'symmetry'. Use `zeroGradient` instead.
8. ONLY use `type empty;` if the mesh declares the patch as type `empty`
9. ONLY use `type symmetry;` if the mesh declares the patch as type `symmetry` or `symmetryPlane`
10. If error mentions "nutkWallFunction" or "Patch type for patch wall must be wall" — this means the mesh `wall` patch has type `patch` instead of `wall`. A fix script handles this, but ensure your wall BCs are correct (noSlip for U, wall functions for nut/k/omega).
11. If error mentions "wallDist" — add `wallDist {{ method meshWave; }}` to fvSchemes.
12. Do NOT invent patch names like `front_and_back`. Only use patches from the config.
13. For 2D meshes: include `frontAndBack` with `type empty;` in ALL 0/* files.
14. fvSchemes MUST include `wallDist {{ method meshWave; }}` for turbulent models.

Generate ALL corrected files now:"""
        
        return f"{system_prompt}\n\n{user_message}"
    
    async def _call_genai(self, prompt: str) -> str:
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.15,
                    max_output_tokens=16000,
                    stop_sequences=["## End of Case"],
                ),
            )
            return response.text or ""
        except Exception as e:
            logger.error(f"[GENAI] API call failed: {e}")
            raise


# ────────────────────────────────────────────────────────────
# File extraction
# ────────────────────────────────────────────────────────────

def extract_file_blocks(text: str) -> dict[str, str]:
    """Extract file blocks from LLM output.
    
    Parses text with ```file:path/to/file blocks.
    """
    files = {}
    
    pattern = r'```file:([^\n]+)\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    
    for path, content in matches:
        path = path.strip()
        content = content.strip()
        if path and content:
            files[path] = content
    
    logger.info(f"[EXTRACT] Extracted {len(files)} files from LLM output")
    return files
