# simd_agent/genai_codegen.py
"""OpenFOAM code generator using Google GenAI directly.

Generates complete OpenFOAM case files via the Google GenAI API.
Includes post-generation validation to catch inconsistencies before
sending to the simulation server.
"""

import asyncio
import copy
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from simd_agent.settings import get_settings
from simd_agent.case_spec import CaseSpec, build_case_spec

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts" / "packs" / "simd"
SOLVERS_DIR = PROMPTS_DIR / "solvers"

# Re-export canonical sets from solver_selector so callers can import from here
from simd_agent.solver_selector import (  # noqa: E402 — intentional late import
    ALLOWED_SOLVERS,
    P_SOLVERS,
    P_RGH_SOLVERS,
    ENERGY_SOLVERS,
    GRAVITY_SOLVERS,
    THERMO_SOLVERS,
)

# Legacy alias kept for backward-compat (buoyant solvers are no longer in the list)
BUOYANT_SOLVERS: set[str] = set()  # cleared — not used anymore


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
    
    # Normalize: replace any "front_and_back" (snake_case artifact) with "frontAndBack"
    if "front_and_back" in expected_patches:
        expected_patches.discard("front_and_back")
        expected_patches.add("frontAndBack")
    
    # Get physics — validated_config exposes these at top level now
    heat_transfer = config.get("heat_transfer", False)
    turb_model    = config.get("turbulence_model", "kOmegaSST")
    # Legacy fallback: physics sub-dict (older format)
    _physics = config.get("physics", {})
    if not heat_transfer:
        heat_transfer = _physics.get("heat_transfer", False)
    if not turb_model or turb_model == "kOmegaSST":
        turb_model = _physics.get("turbulence_model", turb_model) or turb_model
    
    # ── Check 1: controlDict solver — ALWAYS enforce the selected solver ──
    # The selected solver is the authoritative source of truth (set by the
    # deterministic SolverSelector).  The LLM sometimes writes a sibling solver
    # (e.g. "pimpleFoam" instead of "rhoPimpleFoam") which then causes the
    # downstream required-files logic to silently remove files that are actually
    # needed.  We NEVER trust the LLM's application declaration.
    control_dict = fixed_files.get("system/controlDict", "")
    if control_dict:
        app_match = re.search(r'application\s+(\w+)\s*;', control_dict)
        if app_match:
            declared_solver = app_match.group(1)
            if declared_solver != solver:
                issues.append(ValidationIssue(
                    "warning", "system/controlDict",
                    f"LLM wrote 'application {declared_solver}' but selected solver is "
                    f"'{solver}'. Correcting.",
                    fix=f"application     {solver};"
                ))
                fixed_files["system/controlDict"] = re.sub(
                    r'application\s+\w+\s*;',
                    f'application     {solver};',
                    control_dict
                )
    
    # ── Check 2: p vs p_rgh ──
    if solver in P_SOLVERS:
        # These solvers need 0/p, NOT 0/p_rgh
        if "0/p_rgh" in fixed_files and "0/p" not in fixed_files:
            issues.append(ValidationIssue(
                "error", "0/p_rgh",
                f"'{solver}' requires 0/p, not 0/p_rgh. Renaming.",
                fix="Renamed 0/p_rgh → 0/p"
            ))
            content = fixed_files.pop("0/p_rgh")
            content = content.replace("object      p_rgh;", "object      p;")
            content = content.replace("object p_rgh;", "object p;")
            fixed_files["0/p"] = content
        if "0/p_rgh" in fixed_files and "0/p" in fixed_files:
            issues.append(ValidationIssue("warning", "0/p_rgh", "Both 0/p and 0/p_rgh exist. Removing 0/p_rgh."))
            del fixed_files["0/p_rgh"]

    elif solver in P_RGH_SOLVERS:
        # These solvers need 0/p_rgh, NOT 0/p
        if "0/p" in fixed_files and "0/p_rgh" not in fixed_files:
            issues.append(ValidationIssue(
                "error", "0/p",
                f"'{solver}' requires 0/p_rgh, not 0/p. Renaming.",
                fix="Renamed 0/p → 0/p_rgh"
            ))
            content = fixed_files.pop("0/p")
            content = content.replace("object      p;", "object      p_rgh;")
            content = content.replace("object p;", "object p_rgh;")
            fixed_files["0/p_rgh"] = content
        if "0/p" in fixed_files and "0/p_rgh" in fixed_files:
            issues.append(ValidationIssue("warning", "0/p", "Both 0/p and 0/p_rgh exist. Removing 0/p."))
            del fixed_files["0/p"]

    # ── Check 3: Remove unneeded thermo/gravity files for simple solvers ──
    if solver in P_SOLVERS and solver not in THERMO_SOLVERS:
        # Isothermal incompressible — no thermophysicalProperties, no g
        for extra in ["constant/thermophysicalProperties", "constant/g"]:
            if extra in fixed_files:
                issues.append(ValidationIssue("warning", extra, f"'{extra}' not needed for {solver}. Removing."))
                del fixed_files[extra]

    # ── Check 3b: VOF solvers MUST have constant/g — add stub if missing ──
    if solver in GRAVITY_SOLVERS and "constant/g" not in fixed_files:
        issues.append(ValidationIssue(
            "error", "constant/g",
            f"'{solver}' requires constant/g. Adding default (0 -9.81 0).",
            fix="Added constant/g"
        ))
        fixed_files["constant/g"] = (
            "FoamFile\n{\n    version 2.0;\n    format ascii;\n"
            "    class uniformDimensionedVectorField;\n    object g;\n}\n"
            "dimensions [0 1 -2 0 0 0 0];\nvalue (0 -9.81 0);\n"
        )
    
    # ── Check 3a-rho: Inject missing rho solver block for compressible solvers ──
    # rhoPimpleFoam and rhoSimpleFoam solve rhoEqn internally.  OpenFOAM reads
    # solver settings for `rho` from fvSolution/solvers at startup — if the entry
    # is absent it immediately exits with:
    #   "Entry 'rho' not found in dictionary system/fvSolution/solvers"
    # Note: the FIELD FILE 0/rho should NOT be generated (it's NO_READ).
    # The solver entry in fvSolution IS required.
    _RHO_EQN_SOLVERS = {"rhoPimpleFoam", "rhoSimpleFoam"}
    _RHO_BLOCK = (
        "\n    // Auto-injected: rhoEqn solver (required by compressible solvers)\n"
        "    rho\n"
        "    {\n"
        "        solver      diagonal;\n"
        "        tolerance   1e-12;\n"
        "        relTol      0;\n"
        "    }\n"
        "    rhoFinal { $rho; relTol 0; }\n"
    )
    if solver in _RHO_EQN_SOLVERS and "system/fvSolution" in fixed_files:
        _fvs = fixed_files["system/fvSolution"]
        # Check if rho solver entry already present (as standalone or inside regex group)
        _has_rho_entry = bool(
            re.search(r'\brho\s*\{', _fvs)
            or re.search(r'"[^"]*\brho\b[^"]*"\s*\{', _fvs)  # regex group containing rho
        )
        if not _has_rho_entry:
            # Inject right after the pFinal block (or after the opening 'solvers {')
            _fvs_fixed = re.sub(
                r'(solvers\s*\{)',
                r'\1' + _RHO_BLOCK,
                _fvs,
                count=1,
            )
            if _fvs_fixed != _fvs:
                fixed_files["system/fvSolution"] = _fvs_fixed
                issues.append(ValidationIssue(
                    "warning", "system/fvSolution",
                    f"Auto-injected 'rho {{ solver diagonal; }}' block into fvSolution/solvers. "
                    f"'{solver}' solves rhoEqn and requires this entry at runtime.",
                    fix="Injected rho / rhoFinal solver blocks"
                ))
                logger.info(
                    f"[VALIDATE] Auto-injected rho solver block into fvSolution for {solver}"
                )

    # ── Check 3b-thermo: Fix thermoType key 'thermodynamics' → 'thermo' ───────
    # OpenFOAM 2406 requires the key to be 'thermo' inside thermoType{}.
    # The key 'thermodynamics' is valid ONLY as a sub-dict name inside mixture{}.
    # The LLM often confuses these two and writes "thermodynamics  hConst;"
    # which causes: "Entry 'thermo' not found in dictionary thermoType"
    _thermo_path = "constant/thermophysicalProperties"
    if _thermo_path in fixed_files:
        _tp_content = fixed_files[_thermo_path]
        # Match "thermodynamics  <model>;" where model is a single word (hConst, janaf, etc.)
        # but NOT "thermodynamics  {" (which is the legitimate sub-dict in mixture)
        _fixed_tp = re.sub(
            r'\bthermodynamics(\s+)(hConst|eConst|janaf|hTabular|eTabular|hPolynomial|ePolynomial|hIcoTabular|eIcoTabular)\s*;',
            r'thermo\1\2;',
            _tp_content,
        )
        if _fixed_tp != _tp_content:
            issues.append(ValidationIssue(
                "warning", _thermo_path,
                "Auto-fixed: replaced 'thermodynamics <model>' with 'thermo <model>' in thermoType block "
                "(OpenFOAM 2406 requires 'thermo' as the key inside thermoType{}).",
                fix="thermodynamics → thermo in thermoType block"
            ))
            fixed_files[_thermo_path] = _fixed_tp
            logger.info(
                f"[VALIDATE] Auto-fixed thermoType key: 'thermodynamics' → 'thermo' in {_thermo_path}"
            )

    # ── Check 3c-fvOptions: Auto-inject constant/fvOptions for cryogenic cases ──
    # When the fluid is cryogenic (LN2, LH2, LOX, T < 200 K) and the solver is
    # compressible (rho*), the temperature field can diverge to negative values
    # during the first ~100 iterations due to numerical overshoot on cold/warm
    # boundary interfaces.  A limitTemperature fvOption acts as a safety net.
    _COMPRESSIBLE_RHO_SOLVERS = {"rhoPimpleFoam", "rhoSimpleFoam"}
    if solver in _COMPRESSIBLE_RHO_SOLVERS and _is_cryogenic_fluid(config):
        if "constant/fvOptions" not in fixed_files:
            _fvoptions_content = (
                "FoamFile\n"
                "{\n"
                "    version     2.0;\n"
                "    format      ascii;\n"
                "    class       dictionary;\n"
                "    location    \"constant\";\n"
                "    object      fvOptions;\n"
                "}\n"
                "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
                "temperatureLimiter\n"
                "{\n"
                "    type            limitTemperature;\n"
                "    active          yes;\n\n"
                "    selectionMode   all;\n\n"
                "    min             1;       // absolute floor [K] — blocks T going to 0 or negative\n"
                "    max             100000;  // effectively unlimited ceiling\n"
                "}\n\n"
                "// ************************************************************************* //\n"
            )
            fixed_files["constant/fvOptions"] = _fvoptions_content
            issues.append(ValidationIssue(
                "warning", "constant/fvOptions",
                "Auto-injected constant/fvOptions with limitTemperature for cryogenic fluid. "
                "This prevents 'Negative Temperature' divergence during startup iterations.",
                fix="Added constant/fvOptions { limitTemperature { min 1; max 100000; } }"
            ))
            logger.info(
                f"[VALIDATE] Auto-injected constant/fvOptions (limitTemperature) for "
                f"cryogenic fluid with solver {solver}"
            )

    # ── Check 3d-energy: 0/h and 0/e are never generated — nothing to fix ─────
    # rhoPimpleFoam / rhoSimpleFoam rely on 0/T; the thermo package initialises
    # h/e from T at startup.  If the LLM generated 0/h by mistake, remove it.
    for _ef in ("h", "e"):
        _fpath = f"0/{_ef}"
        if _fpath in fixed_files:
            del fixed_files[_fpath]
            issues.append(ValidationIssue(
                "warning", _fpath,
                f"Removed {_fpath}: energy fields must NOT be provided as field files. "
                "The thermo package initialises h/e from 0/T at startup. "
                "Providing 0/h causes 'Negative initial temperature T0' crashes.",
                fix=f"Deleted {_fpath}; ensure 0/T is present with temperature BCs in Kelvin."
            ))

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
    # If mesh has a frontAndBack patch, ensure all 0/* files include it with type empty.
    # frontAndBack is ALWAYS empty in 2D OpenFOAM simulations — regardless of what
    # the mesh config says the type is (it might incorrectly say "patch").
    mesh_has_front_and_back = False
    if mesh_patches_list:
        for mp in mesh_patches_list:
            mp_name = mp.get("name", "") if isinstance(mp, dict) else getattr(mp, "name", "")
            name_lower = mp_name.lower().replace("_", "")
            if name_lower in ("frontandback", "frontback"):
                mesh_has_front_and_back = True
                break
    
    # Also check boundary_conditions keys for frontAndBack
    if not mesh_has_front_and_back:
        bcs = config.get("boundary_conditions", {})
        for bc_name in bcs:
            name_lower = bc_name.lower().replace("_", "")
            if name_lower in ("frontandback", "frontback"):
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
                # Block already present — ensure its type is `empty` (not symmetry / patch / etc.)
                # Pattern: frontAndBack { ... type <something_not_empty>; ... }
                _fab_wrong_type = re.search(
                    r'(frontAndBack\s*\{[^}]*?type\s+)(?!empty\b)(\w+)(\s*;[^}]*?\})',
                    content,
                    re.DOTALL,
                )
                if _fab_wrong_type:
                    wrong_type = _fab_wrong_type.group(2)
                    new_content = re.sub(
                        r'(frontAndBack\s*\{[^}]*?type\s+)(?!empty\b)(\w+)(\s*;)',
                        r'\1empty\3',
                        content,
                        flags=re.DOTALL,
                    )
                    if new_content != content:
                        fixed_files[file_path] = new_content
                        issues.append(ValidationIssue(
                            "warning", file_path,
                            f"Auto-fixed: frontAndBack type '{wrong_type}' → 'empty' "
                            f"(frontAndBack is always an empty patch in 2D meshes).",
                            fix="frontAndBack type → empty"
                        ))
                continue  # Already present (possibly just fixed type)

            # Missing entirely — insert before the closing } of boundaryField
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
    # EXCEPTION: frontAndBack is ALWAYS empty (2D constraint) — override any incorrect mesh type.
    if mesh_patches_list:
        # Build a map of patch_name -> mesh_type
        mesh_patch_types = {}
        for mp in mesh_patches_list:
            if isinstance(mp, dict):
                mp_name = mp.get("name", "")
                mp_type = mp.get("type", "patch")
            elif hasattr(mp, "name"):
                mp_name = mp.name
                mp_type = mp.type
            else:
                continue
            
            # Force well-known constraint patches to their correct type
            name_lower = mp_name.lower().replace("_", "")
            if name_lower in ("frontandback", "frontback", "defaultfaces"):
                mp_type = "empty"
            
            mesh_patch_types[mp_name] = mp_type
        
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
    
    # ── Check 5: Required field files (per-solver-family) ──
    required_files = ["system/controlDict", "system/fvSchemes", "system/fvSolution", "0/U"]

    # Pressure field
    if solver in P_RGH_SOLVERS:
        required_files.append("0/p_rgh")
    else:
        required_files.append("0/p")

    # Transport / thermo
    if solver in THERMO_SOLVERS:
        required_files.append("constant/thermophysicalProperties")
    elif solver not in P_RGH_SOLVERS:
        # Simple single-phase incompressible need transportProperties
        required_files.append("constant/transportProperties")

    # Gravity
    if solver in GRAVITY_SOLVERS:
        required_files.append("constant/g")

    # Energy / temperature
    if solver in ENERGY_SOLVERS or heat_transfer:
        required_files.append("0/T")

    # Turbulence
    if turb_model and turb_model not in ["laminar", "none", None]:
        required_files.append("constant/turbulenceProperties")
        required_files.append("0/nut")
        if "kOmega" in turb_model or "SST" in turb_model:
            required_files.extend(["0/k", "0/omega"])
        elif "kEpsilon" in turb_model:
            required_files.extend(["0/k", "0/epsilon"])

    # fvOptions — cryogenic + compressible solvers
    _COMPRESSIBLE_RHO_SOLVERS_V = {"rhoPimpleFoam", "rhoSimpleFoam"}
    if solver in _COMPRESSIBLE_RHO_SOLVERS_V and _is_cryogenic_fluid(config):
        required_files.append("constant/fvOptions")

    for rf in required_files:
        if rf not in fixed_files:
            issues.append(ValidationIssue(
                "error", rf,
                f"Required file '{rf}' is missing from generated output.",
            ))
    
    # ── Check 6: fvSolution solver algorithm ──
    fv_solution = fixed_files.get("system/fvSolution", "")
    if fv_solution:
        if solver in ("simpleFoam", "rhoSimpleFoam") and "SIMPLE" not in fv_solution:
            issues.append(ValidationIssue("warning", "system/fvSolution", f"{solver} requires a SIMPLE block in fvSolution."))
        elif solver in ("pimpleFoam", "rhoPimpleFoam") and "PIMPLE" not in fv_solution:
            issues.append(ValidationIssue("warning", "system/fvSolution", f"{solver} requires a PIMPLE block in fvSolution."))
        elif solver == "icoFoam" and "PISO" not in fv_solution:
            issues.append(ValidationIssue("warning", "system/fvSolution", "icoFoam requires a PISO block in fvSolution."))
        elif solver in P_RGH_SOLVERS and "PIMPLE" not in fv_solution:
            issues.append(ValidationIssue("warning", "system/fvSolution", f"{solver} (VOF) requires a PIMPLE block in fvSolution."))
    
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


def _is_cryogenic_fluid(config: dict[str, Any]) -> bool:
    """Return True when the fluid is cryogenic (LN2, LH2, LOX, etc.).

    Detection heuristics (any match → True):
      • fluid.name contains one of the cryogenic keywords
      • fluid.temperature < 200 K  (cryogenic range)
    """
    fluid = config.get("fluid", {})
    if not isinstance(fluid, dict):
        fluid = {}
    name = (fluid.get("name") or config.get("fluid_name") or "").lower()
    _CRYO_KEYWORDS = (
        "ln2", "lh2", "lox", "ln₂", "lh₂",
        "liquid nitrogen", "liquid hydrogen", "liquid oxygen",
        "nitrogen", "hydrogen", "cryogen",
    )
    if any(kw in name for kw in _CRYO_KEYWORDS):
        return True
    temp = fluid.get("temperature") or config.get("temperature")
    if temp is not None:
        try:
            if float(temp) < 200.0:
                return True
        except (TypeError, ValueError):
            pass
    return False


def build_required_files_list(solver: str, config: dict[str, Any]) -> list[str]:
    """Return the exact list of files the LLM must generate for this solver/config.

    This is the single source of truth used both in:
      • the codegen prompt (so the LLM knows what to produce)
      • the patch prompt (so we can ask for only missing files)
      • validate_generated_files (cross-check after generation)
    """
    heat_transfer = (
        config.get("heat_transfer")
        or config.get("enable_heat_transfer")
        or config.get("physics", {}).get("heat_transfer")
        or False
    )
    turb_model = (
        config.get("turbulence_model")
        or config.get("physics", {}).get("turbulence_model", "")
        or ""
    )

    required: list[str] = [
        "system/controlDict",
        "system/fvSchemes",
        "system/fvSolution",
        "0/U",
    ]

    # Pressure field
    required.append("0/p_rgh" if solver in P_RGH_SOLVERS else "0/p")

    # Thermophysical or transport
    if solver in THERMO_SOLVERS:
        required.append("constant/thermophysicalProperties")
    else:
        required.append("constant/transportProperties")

    # Gravity (VOF)
    if solver in GRAVITY_SOLVERS:
        required.append("constant/g")

    # Energy / temperature
    if solver in ENERGY_SOLVERS or heat_transfer:
        required.append("0/T")

    # Turbulence (always generate turbulenceProperties; fields only when model active)
    required.append("constant/turbulenceProperties")
    if turb_model and turb_model not in ("laminar", "none"):
        required += ["0/k", "0/nut"]
        if "kOmega" in turb_model or "SST" in turb_model:
            required.append("0/omega")
        elif "kEpsilon" in turb_model or "Epsilon" in turb_model:
            required.append("0/epsilon")
        if solver in ENERGY_SOLVERS:
            required.append("0/alphat")

    # fvOptions — required for cryogenic/liquid simulations with compressible solvers
    # limitTemperature prevents Negative Temperature divergence during the first
    # ~100 iterations when cold initial conditions coexist with warm boundaries.
    _COMPRESSIBLE_SOLVERS = {"rhoPimpleFoam", "rhoSimpleFoam"}
    if solver in _COMPRESSIBLE_SOLVERS and _is_cryogenic_fluid(config):
        required.append("constant/fvOptions")

    return required


def determine_solver(config: dict[str, Any]) -> str:
    """Deterministic fallback solver selection from validated_config.

    Prefer SolverSelector (LLM-assisted) over this function.
    This is a pure-logic fallback used when the selector is unavailable.
    """
    from simd_agent.solver_selector import _heuristic_fallback
    return _heuristic_fallback(config)


# ────────────────────────────────────────────────────────────
# Per-file generation hints (drives focused single-file prompts)
# ────────────────────────────────────────────────────────────

_FILE_HINTS: dict[str, str] = {
    "system/controlDict": (
        "Generate `system/controlDict`.\n"
        "Required entries: application (must equal the selected solver), "
        "startFrom startTime, startTime 0, stopAt endTime, endTime (from config), "
        "deltaT, writeControl runTime, writeInterval, writeFormat ascii, "
        "runTimeModifiable true.\n"
        "Steady solvers (simpleFoam, rhoSimpleFoam): deltaT 1.\n"
        "Transient solvers: use a physically reasonable deltaT (e.g. 0.001-0.01 s)."
    ),
    "system/fvSchemes": (
        "Generate `system/fvSchemes`.\n"
        "Use the numerical schemes from the Solver Instructions.\n"
        "Must include: ddtSchemes, gradSchemes, divSchemes (with correct viscous term), "
        "laplacianSchemes, interpolationSchemes, snGradSchemes.\n"
        "Include `wallDist { method meshWave; }` when a turbulence model is active."
    ),
    "system/fvSolution": (
        "Generate `system/fvSolution`.\n"
        "Must include:\n"
        "  • solvers block — p/pFinal, U, turbulence fields, h/T if energy solver\n"
        "  • algorithm block: SIMPLE (simpleFoam/rhoSimpleFoam), "
        "PIMPLE (pimpleFoam/rhoPimpleFoam/VOF), or PISO (icoFoam)\n"
        "  • relaxationFactors\n"
        "Use the template from the Solver Instructions."
    ),
    "0/U": (
        "Generate `0/U` (velocity field).\n"
        "Dimensions: [0 1 -1 0 0 0 0]\n"
        "Use the exact patch names and velocity values from the config:\n"
        "  inlet   → fixedValue, value = inlet velocity vector\n"
        "  outlet  → zeroGradient\n"
        "  wall    → noSlip\n"
        "  frontAndBack (2D mesh) → empty"
    ),
    "0/p": (
        "Generate `0/p` (pressure field).\n"
        "Incompressible (simpleFoam, pimpleFoam, icoFoam): "
        "dimensions [0 2 -2 0 0 0 0], values in m²/s²\n"
        "Compressible (rhoSimpleFoam, rhoPimpleFoam): "
        "dimensions [1 -1 -2 0 0 0 0], values in Pa\n"
        "Use pressure values (operating, inlet, outlet) from the config.\n"
        "  inlet  → zeroGradient\n"
        "  outlet → fixedValue (atmospheric or configured pressure)\n"
        "  wall   → zeroGradient\n"
        "  frontAndBack (2D) → empty"
    ),
    "0/p_rgh": (
        "Generate `0/p_rgh` (modified pressure = p − ρgh).\n"
        "Dimensions: [1 -1 -2 0 0 0 0]\n"
        "Used by VOF solvers (interFoam, compressibleInterFoam, etc.).\n"
        "  walls/inlets → fixedFluxPressure\n"
        "  outlet       → totalPressure or fixedValue\n"
        "  frontAndBack → empty"
    ),
    "0/T": (
        "Generate `0/T` (temperature field).\n"
        "Dimensions: [0 0 0 1 0 0 0], values in Kelvin.\n"
        "Use temperature values from the config.\n"
        "  inlet  → fixedValue (inlet temperature)\n"
        "  outlet → zeroGradient\n"
        "  wall   → fixedValue (wall temperature if configured) or zeroGradient (adiabatic)\n"
        "  frontAndBack → empty"
    ),
    "0/k": (
        "Generate `0/k` (turbulent kinetic energy).\n"
        "Dimensions: [0 2 -2 0 0 0 0]\n"
        "Estimate: k = 1.5 × (I × |U|)²   where I ≈ 0.05 (5 % turbulence intensity)\n"
        "  inlet  → fixedValue (computed k)\n"
        "  wall   → kqRWallFunction\n"
        "  outlet → zeroGradient\n"
        "  frontAndBack → empty"
    ),
    "0/omega": (
        "Generate `0/omega` (specific dissipation rate — kOmegaSST).\n"
        "Dimensions: [0 0 -1 0 0 0 0]\n"
        "Estimate: ω = k^0.5 / (Cμ^0.25 × ℓ)   Cμ = 0.09, ℓ ≈ 0.07 × hydraulic_diameter\n"
        "  inlet  → fixedValue\n"
        "  wall   → omegaWallFunction\n"
        "  outlet → zeroGradient\n"
        "  frontAndBack → empty"
    ),
    "0/epsilon": (
        "Generate `0/epsilon` (turbulent dissipation — kEpsilon).\n"
        "Dimensions: [0 2 -3 0 0 0 0]\n"
        "Estimate: ε = Cμ^0.75 × k^1.5 / ℓ   ℓ ≈ 0.07 × hydraulic_diameter\n"
        "  inlet  → fixedValue\n"
        "  wall   → epsilonWallFunction\n"
        "  outlet → zeroGradient\n"
        "  frontAndBack → empty"
    ),
    "0/nut": (
        "Generate `0/nut` (turbulent kinematic viscosity).\n"
        "Dimensions: [0 2 -1 0 0 0 0]\n"
        "internalField: uniform 0\n"
        "  inlet/outlet → calculated (value uniform 0)\n"
        "  wall         → nutkWallFunction (value uniform 0)\n"
        "  frontAndBack → empty"
    ),
    "0/alphat": (
        "Generate `0/alphat` (turbulent thermal diffusivity).\n"
        "Dimensions: [1 -1 -1 0 0 0 0]\n"
        "internalField: uniform 0\n"
        "  inlet/outlet → calculated (value uniform 0)\n"
        "  wall         → compressible::alphatWallFunction (Prt 0.85)\n"
        "  frontAndBack → empty"
    ),
    "constant/transportProperties": (
        "Generate `constant/transportProperties`.\n"
        "Format:\n"
        "  transportModel  Newtonian;\n"
        "  nu              [0 2 -1 0 0 0 0] <kinematic_viscosity>;\n"
        "Compute nu = mu / rho if only dynamic viscosity and density are given."
    ),
    "constant/turbulenceProperties": (
        "Generate `constant/turbulenceProperties`.\n"
        "kOmegaSST or similar RANS:\n"
        "  simulationType  RAS;\n"
        "  RAS { RASModel kOmegaSST; turbulence on; printCoeffs on; }\n"
        "Laminar:\n"
        "  simulationType  laminar;\n"
        "Use the turbulence model from the config."
    ),
    "constant/thermophysicalProperties": (
        "Generate `constant/thermophysicalProperties`.\n"
        "For incompressible liquid (LN2, water, oil): heRhoThermo + rhoConst\n"
        "For ideal gas: hePsiThermo + perfectGas\n"
        "The mixture block must have: specie (nMoles, molWeight), "
        "thermodynamics (Cp, Hf), transport (mu, Pr), "
        "equationOfState (rho for rhoConst — omit for perfectGas).\n"
        "Use physical properties from the config. "
        "IMPORTANT: section name is 'thermodynamics' (not 'thermo')."
    ),
    "constant/g": (
        "Generate `constant/g` (gravitational acceleration).\n"
        "class: uniformDimensionedVectorField\n"
        "dimensions: [0 1 -2 0 0 0 0]\n"
        "value: (0 -9.81 0)"
    ),
}


# ────────────────────────────────────────────────────────────
# Code Generator
# ────────────────────────────────────────────────────────────

class GenAICodeGenerator:
    """Generate OpenFOAM case files using Google GenAI.
    
    Replaces the codegen library with direct Google GenAI API calls.
    Includes post-generation validation to catch inconsistencies.
    """
    
    def __init__(self, event_bus=None):
        settings = get_settings()
        self.model = settings.gemini_super_model
        self.event_bus = event_bus  # Optional[EventBus] — injected by orchestrator

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

    def _load_solver_prompt(self, solver: str) -> str:
        """Load the solver-specific instruction file from prompts/packs/simd/solvers/."""
        solver_path = SOLVERS_DIR / f"{solver}.md"
        if solver_path.exists():
            return solver_path.read_text()
        logger.warning(f"[GENAI] No solver-specific prompt for '{solver}' at {solver_path}")
        return ""
    
    # ── Parallel (per-file) code generation ──────────────────────────────────

    async def generate_parallel(
        self,
        requirements: str,
        validated_config: dict[str, Any],
        solver: str,
        case_type: str = "pipe_flow",
        files_to_generate: list[str] | None = None,
        error_context: dict[str, Any] | None = None,
        case_spec: CaseSpec | None = None,
        iteration: int = 1,
        codegen_mode: str = "full",
    ) -> str:
        """Generate OpenFOAM files in parallel — one focused LLM call per file.

        Each file gets its own concurrent API call with a focused single-file
        prompt, eliminating the truncation / missing-file problems of generating
        the full case in one shot.  This same method is used for both the initial
        generation and for error-recovery (just pass error_context in the latter).

        Args:
            files_to_generate: Explicit list of file paths to generate.
                               None → generate all required files for the solver.
            error_context: Optional dict with keys:
                  "errors"         – list[dict] of previous_errors from orchestrator
                  "previous_files" – dict[str, str] of previously generated files

        Returns:
            Standard ```file:path\\n<content>\\n``` block string, compatible with
            extract_file_blocks().
        """
        if solver not in ALLOWED_SOLVERS:
            logger.warning(f"[PARALLEL] Solver '{solver}' not in ALLOWED_SOLVERS — proceeding anyway")

        # ── Build CaseSpec (single source of truth) ───────────────────────────
        if case_spec is None:
            case_spec = build_case_spec(solver, validated_config)

        if files_to_generate is None:
            files_to_generate = (
                case_spec.required_0_fields
                + case_spec.required_constant_files
                + case_spec.required_system_files
            )

        mode = "ERROR-RECOVERY" if error_context else "INITIAL"
        logger.info(
            f"[PARALLEL] {mode}: solver={solver}  files={len(files_to_generate)}"
        )
        print(f"\n{'='*70}")
        print(f"[PARALLEL CODEGEN] {mode}  solver={solver}  files={len(files_to_generate)}")
        for f in files_to_generate:
            print(f"  → {f}")
        print(f"{'='*70}\n")

        # ── Build shared context (slim — no full solver .md for field files) ──
        shared_ctx = self._build_shared_context(
            requirements, validated_config, solver, case_type, error_context
        )

        # Semaphore: max 6 concurrent API calls to avoid rate-limiting
        sem = asyncio.Semaphore(6)

        async def _one(file_path: str) -> tuple[str, str | None]:
            async with sem:
                # Notify frontend that this file is being generated
                if self.event_bus:
                    try:
                        await self.event_bus.emit_file_generating(
                            file_path, iteration, mode=codegen_mode
                        )
                    except Exception:
                        pass
                try:
                    content = await self._generate_single_file(
                        file_path, solver, validated_config, shared_ctx,
                        case_spec=case_spec,
                        error_context=error_context,
                    )
                    logger.info(f"[PARALLEL] ✅  {file_path}  ({len(content)} chars)")
                    print(f"\n{'─'*60}")
                    print(f"📄 GENERATED: {file_path}  ({len(content)} chars)")
                    print(f"{'─'*60}")
                    print(content)
                    print(f"{'─'*60}\n")
                    # Notify frontend that this file is ready (with its content)
                    if self.event_bus:
                        try:
                            await self.event_bus.emit_file_generated(
                                file_path, content, iteration, len(content),
                                mode=codegen_mode,
                            )
                        except Exception:
                            pass
                    return file_path, content
                except Exception as exc:
                    logger.warning(f"[PARALLEL] ❌  {file_path}  — {exc}")
                    print(f"\n{'─'*60}")
                    print(f"❌ FAILED:    {file_path}  — {exc}")
                    print(f"{'─'*60}\n")
                    return file_path, None

        # First pass — all files concurrently
        results = await asyncio.gather(*(_one(f) for f in files_to_generate))

        generated: dict[str, str] = {}
        failed: list[str] = []
        for path, content in results:
            if content:
                generated[path] = content
            else:
                failed.append(path)

        # Second pass — retry failed files sequentially (gentler on rate limits)
        if failed:
            logger.warning(f"[PARALLEL] Retrying {len(failed)} failed file(s): {failed}")
            for path in failed:
                try:
                    content = await self._generate_single_file(
                        path, solver, validated_config, shared_ctx,
                        case_spec=case_spec,
                        error_context=error_context,
                    )
                    if content:
                        generated[path] = content
                        logger.info(f"[PARALLEL] ✅ retry  {path}")
                except Exception as exc:
                    logger.error(f"[PARALLEL] ❌ retry  {path}: {exc}")

        logger.info(
            f"[PARALLEL] Done: {len(generated)}/{len(files_to_generate)} files generated"
        )
        print(f"\n{'='*70}")
        print(f"[PARALLEL CODEGEN] Complete: {len(generated)}/{len(files_to_generate)} files")
        for path in files_to_generate:
            icon = "✅" if path in generated else "❌ MISSING"
            print(f"  {icon}  {path}")
        print(f"{'='*70}\n")

        # Return as standard ```file:path block string
        return "\n\n".join(
            f"```file:{path}\n{content}\n```"
            for path, content in generated.items()
        )

    def _build_shared_context(
        self,
        requirements: str,
        validated_config: dict[str, Any],
        solver: str,
        case_type: str,
        error_context: dict[str, Any] | None = None,
    ) -> str:
        """Build the context string shared by all single-file generation calls.

        Contains: solver instructions + codegen rules + full config.
        When error_context is provided also includes a summary of what failed.
        """
        solver_instructions = self._load_solver_prompt(solver)
        system_prompt = self._codegen_prompt
        if solver_instructions:
            system_prompt = (
                f"{system_prompt}\n\n"
                f"---\n\n## Solver Instructions — {solver}\n\n{solver_instructions}"
            )

        # Strip irrelevant turbulence fields from boundary conditions
        turb_model = (
            validated_config.get("turbulence_model")
            or validated_config.get("physics", {}).get("turbulence_model", "")
            or ""
        )
        config_for_llm = copy.deepcopy(validated_config)
        bcs = config_for_llm.get("boundary_conditions", {})
        if isinstance(bcs, dict):
            is_komega  = "kOmega" in turb_model or "SST" in turb_model
            is_kepsilon = "kEpsilon" in turb_model or "Epsilon" in turb_model
            for patch_bc in bcs.values():
                if not isinstance(patch_bc, dict):
                    continue
                turb = patch_bc.get("turbulence") if isinstance(patch_bc.get("turbulence"), dict) else patch_bc
                if is_komega:
                    turb.pop("epsilon", None)
                elif is_kepsilon:
                    turb.pop("omega", None)

        # Compute endTime (steady = iteration count, transient = physical seconds)
        solver_cfg = config_for_llm.get("solver", {})
        if not isinstance(solver_cfg, dict):
            solver_cfg = {}
        _is_steady = solver in {"simpleFoam", "rhoSimpleFoam"}
        if _is_steady:
            end_time = (
                config_for_llm.get("max_iterations")
                or solver_cfg.get("max_iterations")
                or solver_cfg.get("maxIterations")
                or 1000
            )
        else:
            end_time = (
                config_for_llm.get("end_time")
                or solver_cfg.get("endTime")
                or solver_cfg.get("end_time")
                or config_for_llm.get("max_iterations")
                or 1000
            )

        # Build optional error summary section
        error_section = ""
        if error_context:
            errors = error_context.get("errors", [])
            if errors:
                err_lines = []
                for e in errors:
                    src = e.get("source", "unknown")
                    msg = e.get("error", "")
                    stderr = e.get("stderr", "") or e.get("details", "")
                    err_lines.append(f"  [{src}] {msg}")
                    if stderr:
                        # Include last 20 lines of stderr for context
                        tail = "\n".join(stderr.splitlines()[-20:])
                        err_lines.append(f"  OpenFOAM output:\n{tail}")
                error_section = (
                    "\n\n---\n\n"
                    "## ⚠️  Previous Run Failed — Error Context\n\n"
                    "The previous simulation attempt failed with these errors.  "
                    "Each file below must be corrected to fix the issue.\n\n"
                    + "\n".join(err_lines)
                )

        return (
            f"{system_prompt}\n\n"
            f"---\n\n"
            f"## Simulation Context\n\n"
            f"### User Requirements\n{requirements}\n\n"
            f"### Validated Configuration\n"
            f"```json\n{json.dumps(config_for_llm, indent=2, default=str)}\n```\n\n"
            f"### Selected Solver: `{solver}`\n"
            f"### Case Type: `{case_type}`\n"
            f"### endTime: `{end_time}`\n"
            f"{error_section}\n"
        )

    async def _generate_single_file(
        self,
        file_path: str,
        solver: str,
        validated_config: dict[str, Any],
        shared_context: str,
        case_spec: CaseSpec | None = None,
        error_context: dict[str, Any] | None = None,
    ) -> str:
        """Generate a single OpenFOAM file with a tightly focused LLM call.

        Uses CaseSpec for a SHORT, focused prompt so the model has ample output
        budget.  Falls back to shared_context if CaseSpec is not available.
        Detects truncated output and retries up to once.
        """
        prompt = self._build_single_file_prompt(
            file_path=file_path,
            solver=solver,
            case_spec=case_spec,
            shared_context=shared_context,
            error_context=error_context,
        )

        # ── Print complete prompt for 0/* files so BC values are visible in logs ──
        # print(
        #     f"\n{'═' * 70}\n"
        #     f"🔍 PROMPT → {file_path}  ({len(prompt)} chars)\n"
        #     f"{'═' * 70}\n"
        #     f"{prompt}\n"
        #     f"{'═' * 70}\n"
        # )

        for attempt in (1, 2):
            response = await self._call_genai_single_file(prompt)
            content = self._extract_single_file_content(response, file_path)
            if content and not _is_truncated(file_path, content):
                return content
            if content and attempt == 1:
                # Truncated — retry with explicit continuation hint
                logger.warning(
                    f"[PARALLEL] '{file_path}' appears truncated ({len(content)} chars) "
                    f"— retrying"
                )
                prompt = (
                    f"{prompt}\n\n"
                    f"⚠️  The previous generation was TRUNCATED.  "
                    f"Generate the COMPLETE file again.  "
                    f"Make sure to include ALL boundary condition patches and close all braces."
                )
            elif not content:
                logger.warning(f"[PARALLEL] '{file_path}' extraction failed (attempt {attempt})")

        # Return whatever we have even if truncated (verifier will catch it)
        if content:
            return content

        raise RuntimeError(
            f"[PARALLEL] Could not extract content for '{file_path}' "
            f"(response {len(response)} chars)"
        )

    def _build_single_file_prompt(
        self,
        file_path: str,
        solver: str,
        case_spec: CaseSpec | None,
        shared_context: str,
        error_context: dict[str, Any] | None = None,
    ) -> str:
        """Build a SHORT, focused prompt for a single file.

        When CaseSpec is available, the prompt contains ONLY the information
        relevant to this specific file — not the full solver .md or config JSON.
        This dramatically reduces input tokens, leaving more room for output.
        """
        hint = _FILE_HINTS.get(
            file_path,
            f"Generate the file `{file_path}` appropriate for the `{solver}` solver.",
        )

        # ── CaseSpec-based context (compact, relevant only) ───────────────────
        if case_spec:
            spec_dict = case_spec.as_prompt_dict()
            import json as _json
            spec_str = _json.dumps(spec_dict, indent=2, default=str)
            context_section = (
                f"## CaseSpec (single source of truth — use ONLY these values)\n"
                f"```json\n{spec_str}\n```\n"
            )
            # For system files and constant files that need solver templates,
            # include a brief solver template note rather than the full .md
            if file_path.startswith("system/") or file_path.startswith("constant/"):
                solver_note_brief = self._brief_solver_note(file_path, case_spec)
                context_section += f"\n{solver_note_brief}"
            # For 0/* field files: inject mandatory per-patch BC table so the LLM
            # cannot ignore the user-specified values (temperature, velocity, k, omega…)
            elif file_path.startswith("0/"):
                bc_table = self._format_field_bc_table(file_path, case_spec)
                if bc_table:
                    context_section += (
                        f"\n## ⚠️  MANDATORY boundary conditions for `{file_path.split('/', 1)[1]}`\n"
                        f"These values come directly from the user's requirements. "
                        f"You MUST use them exactly — do NOT substitute defaults:\n\n"
                        f"{bc_table}\n"
                    )
        else:
            context_section = shared_context

        # ── Pressure dimension note ───────────────────────────────────────────
        p_note = ""
        if file_path in ("0/p", "0/p_rgh") and case_spec:
            if case_spec.compressible:
                p_note = "\n⚠️  COMPRESSIBLE: dimensions [1 -1 -2 0 0 0 0], values in Pa."
            else:
                p_note = "\n⚠️  INCOMPRESSIBLE: dimensions [0 2 -2 0 0 0 0], values in m²/s²."

        energy_note = ""  # 0/h and 0/e are never generated; thermo initialises from 0/T

        # ── Error context (previous file + error) ────────────────────────────
        error_section = ""
        if error_context:
            prev_files = error_context.get("previous_files", {})
            if file_path in prev_files:
                prev = prev_files[file_path]
                if len(prev) > 1500:
                    prev = prev[:1500] + "\n... (truncated)"
                error_section = (
                    f"\n### ⚠️  Previous version (BROKEN — fix it)\n```\n{prev}\n```\n"
                )

        task = "FIX this file" if error_context else "Generate EXACTLY this one file"

        return (
            f"# {task}: `{file_path}`\n\n"
            f"{context_section}\n\n"
            f"## File instructions\n{hint}{p_note}{energy_note}{error_section}\n\n"
            f"## Rules\n"
            f"1. Output ONLY `{file_path}` — no other files\n"
            f"2. Patch names MUST match `patch_names` in CaseSpec exactly\n"
            f"3. Physical values MUST come from CaseSpec/BC table — do NOT invent defaults\n"
            f"4. `application` in controlDict MUST be `{solver}`\n"
            f"5. endTime in controlDict MUST be an integer (e.g. 88 not 88.0); deltaT may be float\n"
            f"6. Generate the COMPLETE file — include ALL patches in boundaryField\n"
            f"7. No leading spaces on top-level dict keys\n"
            f"8. Wrap output EXACTLY as:\n"
            f"   ```file:{file_path}\n"
            f"   FoamFile\n"
            f"   {{\n"
            f"       ...\n"
            f"   }}\n"
            f"   <rest of content>\n"
            f"   ```\n"
            f"   (first line inside block MUST be `FoamFile`, never `{file_path}`)\n\n"
            f"Generate complete `{file_path}` now:"
        )

    def _format_field_bc_table(self, file_path: str, cs: "CaseSpec") -> str:
        """Build a human-readable per-patch BC table for a single 0/* field.

        For energy fields (0/h, 0/e): uses CaseSpec.energy_bc_values (J/kg)
        computed deterministically from T * Cp — never passes raw temperature
        values into an energy-field prompt.

        For all other fields: maps to the matching key in each patch's
        boundary_conditions entry.
        """
        field_name = file_path.split("/", 1)[1]  # "0/T" → "T"

        # 0/h and 0/e are never generated — thermo initialises from 0/T.
        if field_name in ("h", "e"):
            return ""  # should not be called, but guard just in case

        # ── Standard fields (U, p, T, k, omega, …) ───────────────────────────
        _FIELD_KEY_MAP: dict[str, list[str]] = {
            "U":      ["velocity", "U"],
            "p":      ["pressure", "p"],
            "p_rgh":  ["pressure", "p_rgh"],
            "T":      ["temperature", "T"],
            "k":      ["k"],
            "omega":  ["omega"],
            "epsilon":["epsilon"],
            "nut":    ["nut"],
            "alphat": ["alphat"],
        }
        keys_to_try = _FIELD_KEY_MAP.get(field_name, [field_name])

        bcs = cs.boundary_conditions or {}
        if not bcs:
            return ""

        lines = []
        for patch in cs.patch_names:
            ptype = cs.patch_type_by_name.get(patch, "patch")

            # Constraint patches use their geometric type unconditionally
            if ptype == "empty":
                lines.append(f"  {patch:20s}: type empty;")
                continue
            if ptype == "symmetry":
                lines.append(f"  {patch:20s}: type symmetry;")
                continue

            patch_bc = bcs.get(patch, {})
            field_entry: Any = None
            for k in keys_to_try:
                if k in patch_bc:
                    field_entry = patch_bc[k]
                    break

            if field_entry is None:
                # For turbulence fields: try CaseSpec.turbulence_initial_values as fallback
                tiv = cs.turbulence_initial_values or {}
                fallback_val = tiv.get(field_name)
                if fallback_val is not None:
                    lines.append(
                        f"  {patch:20s}: (no BC specified — "
                        f"suggest fixedValue {fallback_val} at inlet, zeroGradient elsewhere)"
                    )
                else:
                    lines.append(f"  {patch:20s}: (not specified — use zeroGradient as fallback)")
                continue

            if isinstance(field_entry, dict):
                bc_type = field_entry.get("type", "zeroGradient")
                bc_val  = field_entry.get("value")
                if bc_val is not None:
                    # Format vectors as "(x y z)", scalars as plain numbers
                    if isinstance(bc_val, (list, tuple)):
                        val_str = "(" + " ".join(str(v) for v in bc_val) + ")"
                    else:
                        val_str = str(bc_val)
                    lines.append(f"  {patch:20s}: type {bc_type};  value uniform {val_str};")
                else:
                    lines.append(f"  {patch:20s}: type {bc_type};")
            elif isinstance(field_entry, str):
                lines.append(f"  {patch:20s}: type {field_entry};")
            else:
                lines.append(f"  {patch:20s}: {field_entry}")

        # Append internalField guidance using turbulence_initial_values
        tiv = cs.turbulence_initial_values or {}
        if field_name in tiv:
            lines.append(
                f"\n# internalField: uniform {tiv[field_name]}  "
                f"(pre-computed from turbulence config)"
            )

        return "\n".join(lines) if lines else ""

    def _brief_solver_note(self, file_path: str, cs: CaseSpec) -> str:
        """Return a concise template note for system/constant files."""
        if file_path == "system/fvSchemes":
            return (
                f"## fvSchemes guidance\n"
                f"- ddtSchemes: `steadyState` if steady, `Euler` if transient\n"
                f"- divSchemes: include div(phi,U), div(phi,{cs.energy_field or 'h'}) if energy, "
                f"div(phi,k), div(phi,{('omega' if 'omega' in cs.turbulence_fields else 'epsilon') if cs.turbulence_fields else 'omega'})\n"
                f"- wallDist: include if turbulence_model != laminar\n"
                f"- viscous stress: `div((({('rho*' if cs.compressible else '')}nuEff)*dev2(T(grad(U)))))  Gauss linear;`\n"
            )
        if file_path == "system/fvSolution":
            rho_note = ""
            if cs.solver in {"rhoPimpleFoam", "rhoSimpleFoam"}:
                rho_note = (
                    "\n⚠️  CRITICAL: Include explicit `rho` solver entry (even though 0/rho is NOT generated):\n"
                    "  rho      { solver diagonal; tolerance 1e-12; relTol 0; }\n"
                    "  rhoFinal { $rho; relTol 0; }\n"
                    "Missing it causes: \"Entry 'rho' not found in dictionary system/fvSolution/solvers\""
                )
            energy_str = f", {cs.energy_field}" if cs.energy_field else ""
            turb_str = f", {', '.join(cs.turbulence_fields)}" if cs.turbulence_fields else ""
            return (
                f"## fvSolution guidance\n"
                f"- Algorithm block: {cs.algorithm} {{}}\n"
                f"- Solver entries: p/pFinal{', rho/rhoFinal' if cs.solver in {'rhoPimpleFoam', 'rhoSimpleFoam'} else ''}, U{energy_str}{turb_str}\n"
                f"- relaxationFactors: fields {{p 0.3;{' rho 0.05;' if cs.solver in {'rhoPimpleFoam', 'rhoSimpleFoam'} else ''}}} equations {{U 0.7;}}\n"
                + rho_note
            )
        if file_path == "system/controlDict":
            # endTime must be an integer (no decimal point) for both steady and transient
            end_time_fmt = int(cs.end_time) if cs.end_time == int(cs.end_time) else cs.end_time
            return (
                f"## controlDict guidance\n"
                f"- application: {cs.solver}\n"
                f"- endTime: {end_time_fmt}   ← integer, NO decimal point (write {end_time_fmt} not {float(cs.end_time)})\n"
                f"- deltaT: {cs.delta_t}   ← may be float\n"
                f"- startFrom startTime; startTime 0;\n"
            )
        if file_path == "constant/turbulenceProperties":
            return (
                f"## turbulenceProperties guidance\n"
                f"- simulationType: {cs.sim_type};\n"
                f"- {cs.sim_type} {{ {cs.sim_type}Model {cs.turbulence_model}; turbulence on; printCoeffs on; }}\n"
            )
        if file_path == "constant/thermophysicalProperties":
            eos = "rhoConst" if cs.rho is not None else "perfectGas"
            return (
                f"## thermophysicalProperties — EXACT required structure\n\n"
                f"⚠️  CRITICAL KEY NAMES — wrong names cause FOAM FATAL IO ERROR:\n"
                f"- Inside `thermoType {{}}`: the key is `thermo` (NOT `thermodynamics`)\n"
                f"- Inside `mixture {{}}`: the sub-dict is `thermodynamics` (NOT `thermo`)\n\n"
                f"```\n"
                f"thermoType\n"
                f"{{\n"
                f"    type            heRhoThermo;\n"
                f"    mixture         pureMixture;\n"
                f"    transport       const;\n"
                f"    thermo          hConst;          // ← MUST be 'thermo', NOT 'thermodynamics'\n"
                f"    equationOfState {eos};\n"
                f"    specie          specie;\n"
                f"    energy          sensibleEnthalpy;\n"
                f"}}\n\n"
                f"mixture\n"
                f"{{\n"
                f"    specie\n"
                f"    {{\n"
                f"        nMoles      1;\n"
                f"        molWeight   {cs.rho and 28.97 or 28.97:.4g};\n"
                f"    }}\n"
                f"    thermodynamics              // ← MUST be 'thermodynamics' (sub-dict in mixture)\n"
                f"    {{\n"
                f"        Cp          {cs.cp or 1000};\n"
                f"        Hf          0;\n"
                f"    }}\n"
                f"    transport\n"
                f"    {{\n"
                f"        mu          {cs.mu or 1.8e-5};\n"
                f"        Pr          {cs.prandtl or 0.713};\n"
                f"    }}\n"
                + (f"    equationOfState {{ rho {cs.rho}; }}\n" if cs.rho is not None else "")
                + f"}}\n"
                f"```\n"
                f"Use the physical values from CaseSpec (Cp={cs.cp or 1000}, mu={cs.mu or 1.8e-5}, Pr={cs.prandtl or 0.713}).\n"
            )
        return ""

    def _extract_single_file_content(self, response: str, file_path: str) -> str | None:
        """Extract file content from LLM response, trying multiple strategies."""
        # Strategy 1: file:path wrapper (preferred)
        blocks = extract_file_blocks(response)
        if file_path in blocks and blocks[file_path].strip():
            return _strip_file_header(blocks[file_path].strip())
        if len(blocks) == 1:
            content = _strip_file_header(next(iter(blocks.values())).strip())
            if content:
                return content
        # Strategy 2: strip all backtick fences and return raw content
        raw = re.sub(
            r"```(?:file:|openfoam:|foam:|sh:|bash:|text:)?[^\n]*\n?", "", response
        ).strip()
        raw = re.sub(r"^file:\S+\n", "", raw).strip()
        raw = _strip_file_header(raw)
        if raw and len(raw) > 30:
            return raw
        return None

    async def _call_genai_single_file(self, prompt: str) -> str:
        """LLM call for single-file generation.

        4096 output tokens is ample for any single OpenFOAM file while keeping
        the per-call cost low.  Temperature 0.1 for deterministic output.
        """
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=8192,
                ),
            )
            return response.text or ""
        except Exception as exc:
            logger.error(f"[GENAI_SINGLE] API call failed: {exc}")
            raise

    @staticmethod
    def _identify_affected_files(
        previous_errors: list[dict],
        previous_files: dict[str, str],
        solver: str,
        validated_config: dict[str, Any],
    ) -> list[str]:
        """Determine which files need to be regenerated based on sim errors.

        Maps OpenFOAM error patterns to the files most likely at fault.
        Falls back to regenerating all required files if the error is unclear.
        """
        all_errors_text = " ".join(
            (e.get("error", "") + " " + (e.get("stderr", "") or e.get("details", "")))
            for e in previous_errors
        ).lower()

        affected: set[str] = set()

        # ── cannot find file ──────────────────────────────────────────────
        import re as _re
        for m in _re.finditer(r'cannot find file[^"]*"([^"]+)"', all_errors_text, _re.IGNORECASE):
            raw_path = m.group(1).strip()
            # Strip case/ prefix that the sim server may add
            for prefix in ("case/", "/case/", "/tmp/simd-runs/"):
                if raw_path.startswith(prefix) or "/" + prefix in raw_path:
                    raw_path = raw_path.split("case/", 1)[-1]
                    break
            affected.add(raw_path)

        # ── boundary / patch errors → all 0/* field files ────────────────
        BC_PATTERNS = [
            "cannot find patchfield",
            "fvpatchfield",
            "boundary condition",
            "patch.*not defined",
            "not constraint type",
            "inlet.*not defined",
            "outlet.*not defined",
        ]
        if any(_re.search(p, all_errors_text) for p in BC_PATTERNS):
            affected.update(f for f in previous_files if f.startswith("0/"))
            if not affected:
                # No previous 0/* files; regenerate all
                affected.update(build_required_files_list(solver, validated_config))

        # ── scheme / numerical errors ─────────────────────────────────────
        SCHEME_PATTERNS = [
            "unknown discretisation scheme",
            "walldist",
            "interpolationscheme",
            "divscheme",
            "laplacianscheme",
        ]
        if any(_re.search(p, all_errors_text) for p in SCHEME_PATTERNS):
            affected.add("system/fvSchemes")

        # ── solver / convergence errors ───────────────────────────────────
        SOLVER_PATTERNS = [
            "maximum number of iterations",
            "floating point exception",
            "singular matrix",
            "excess of.*under-relaxation",
        ]
        if any(_re.search(p, all_errors_text) for p in SOLVER_PATTERNS):
            affected.update(["system/fvSolution", "system/fvSchemes", "system/controlDict"])

        # ── thermophysical / transport errors ─────────────────────────────
        THERMO_PATTERNS = [
            "thermodynamics",
            "thermophysical",
            "thermomodel",
            "entry.*not found in dictionary.*mixture",
        ]
        if any(_re.search(p, all_errors_text) for p in THERMO_PATTERNS):
            affected.add("constant/thermophysicalProperties")

        # ── turbulence errors ─────────────────────────────────────────────
        TURB_PATTERNS = ["turbulenceproperties", "rasmodel", "turbulence.*on"]
        if any(_re.search(p, all_errors_text) for p in TURB_PATTERNS):
            affected.add("constant/turbulenceProperties")

        # If nothing was identified, regenerate all required files
        if not affected:
            logger.warning(
                "[PARALLEL] Could not identify affected files from error — "
                "regenerating all required files"
            )
            return build_required_files_list(solver, validated_config)

        logger.info(f"[PARALLEL] Affected files identified from error: {sorted(affected)}")
        return sorted(affected)

    # ── Main generate() entry point ───────────────────────────────────────────

    async def generate(
        self,
        requirements: str,
        validated_config: dict[str, Any],
        solver: str = "simpleFoam",
        case_type: str = "pipe_flow",
        previous_errors: list[dict] | None = None,
        previous_files: dict[str, str] | None = None,
        missing_files: list[str] | None = None,
        iteration: int = 1,
    ) -> str:
        """Generate OpenFOAM case files.

        Routing (all paths use the same parallel per-file engine)
        ──────────────────────────────────────────────────────────
        • neither                  → parallel ALL required files  (initial run)
        • missing_files provided   → parallel ONLY those files    (patch run)
        • previous_errors provided → identify affected files from errors,
                                     then parallel those files with error context
                                     (error-recovery run)

        Returns:
            String containing file blocks in ```file:path format
        """
        if solver not in ALLOWED_SOLVERS:
            logger.warning(f"[GENAI] Solver '{solver}' is not in ALLOWED_SOLVERS — proceeding anyway")

        if missing_files:
            # ── Patch: parallel generation of only the missing files ───────
            logger.info(
                f"[GENAI] PATCH parallel generation: "
                f"solver={solver}  missing={missing_files}"
            )
            return await self.generate_parallel(
                requirements=requirements,
                validated_config=validated_config,
                solver=solver,
                case_type=case_type,
                files_to_generate=missing_files,
                iteration=iteration,
                codegen_mode="patch",
            )

        elif previous_errors:
            # ── Error-recovery: identify affected files → parallel fix ─────
            affected = self._identify_affected_files(
                previous_errors=previous_errors,
                previous_files=previous_files or {},
                solver=solver,
                validated_config=validated_config,
            )
            error_ctx = {
                "errors": previous_errors,
                "previous_files": previous_files or {},
            }
            logger.info(
                f"[GENAI] ERROR-RECOVERY parallel generation: "
                f"solver={solver}  affected={affected}"
            )
            return await self.generate_parallel(
                requirements=requirements,
                validated_config=validated_config,
                solver=solver,
                case_type=case_type,
                files_to_generate=affected,
                error_context=error_ctx,
                iteration=iteration,
                codegen_mode="fix",
            )

        else:
            # ── Initial: parallel ALL required files ───────────────────────
            logger.info(f"[GENAI] INITIAL parallel generation: solver={solver}")
            return await self.generate_parallel(
                requirements=requirements,
                validated_config=validated_config,
                solver=solver,
                case_type=case_type,
                iteration=iteration,
                codegen_mode="full",
            )
    
    def _build_codegen_prompt(
        self,
        requirements: str,
        validated_config: dict[str, Any],
        solver: str,
        case_type: str,
    ) -> str:
        # ── Base prompt + solver-specific instructions ──────────────────
        solver_instructions = self._load_solver_prompt(solver)
        system_prompt = self._codegen_prompt
        if solver_instructions:
            system_prompt = (
                f"{system_prompt}\n\n"
                f"---\n\n## Solver Instructions — {solver}\n\n{solver_instructions}"
            )

        # ── Strip irrelevant turbulence fields from boundary conditions ──
        # The precheck service computes BOTH epsilon and omega values and puts
        # them in every patch's BC even when only one is used by the solver.
        # Sending the unused field to the LLM causes it to generate a spurious
        # 0/epsilon (for kOmegaSST) or 0/omega (for kEpsilon) file.
        turb_model = (
            validated_config.get("physics", {}).get("turbulence_model") or ""
        )
        config_for_llm = copy.deepcopy(validated_config)
        bcs = config_for_llm.get("boundary_conditions", {})
        if isinstance(bcs, dict):
            is_komega = "kOmega" in turb_model or "SST" in turb_model
            is_kepsilon = "kEpsilon" in turb_model or "Epsilon" in turb_model
            for patch_bc in bcs.values():
                if not isinstance(patch_bc, dict):
                    continue
                turb = patch_bc.get("turbulence") if isinstance(patch_bc.get("turbulence"), dict) else patch_bc
                if is_komega:
                    # kOmegaSST uses k + omega; strip epsilon
                    turb.pop("epsilon", None)
                elif is_kepsilon:
                    # kEpsilon uses k + epsilon; strip omega
                    turb.pop("omega", None)

        # ── Extract endTime for explicit injection into controlDict rule ──
        # Steady solvers (simpleFoam, rhoSimpleFoam):
        #   endTime = iteration count → use max_iterations
        # Transient solvers (pimpleFoam, rhoPimpleFoam, inter* …):
        #   endTime = physical seconds → use end_time
        solver_raw = config_for_llm.get("solver", {})
        solver_cfg = solver_raw if isinstance(solver_raw, dict) else {}

        _is_steady = solver in {"simpleFoam", "rhoSimpleFoam"}

        if _is_steady:
            max_iterations = (
                config_for_llm.get("max_iterations")        # linting output (preferred for steady)
                or solver_cfg.get("max_iterations")
                or solver_cfg.get("maxIterations")
                or solver_cfg.get("endTime")
                or solver_cfg.get("end_time")
                or config_for_llm.get("end_time")
                or 1000
            )
        else:
            max_iterations = (
                config_for_llm.get("end_time")              # physical seconds (preferred for transient)
                or solver_cfg.get("endTime")
                or solver_cfg.get("end_time")
                or config_for_llm.get("max_iterations")
                or solver_cfg.get("max_iterations")
                or solver_cfg.get("maxIterations")
                or 1000
            )

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
        
        # Pressure field rule — depends on solver family
        _pressure_rule = (
            f"You MUST generate `0/p_rgh` (NOT `0/p`) — {solver} reads `p_rgh`"
            if solver in P_RGH_SOLVERS
            else f"You MUST generate `0/p` (NOT `0/p_rgh`) — {solver} reads `p`"
        )
        # Thermo rule
        _thermo_rule = (
            "You MUST generate `constant/thermophysicalProperties` (energy equation is active)"
            if solver in THERMO_SOLVERS
            else "Do NOT generate `constant/thermophysicalProperties` (not needed for this solver)"
        )
        # g file rule
        _g_rule = (
            "You MUST generate `constant/g` (VOF solver — always required)"
            if solver in GRAVITY_SOLVERS
            else "Do NOT generate `constant/g` (not needed for this solver)"
        )
        # T file rule
        _t_rule = (
            "You MUST generate `0/T` (energy equation is active)"
            if (solver in ENERGY_SOLVERS or config_for_llm.get("heat_transfer"))
            else "Do NOT generate `0/T` unless heat_transfer is explicitly true"
        )

        # Build the required files checklist for this specific solver + config
        _required_files = build_required_files_list(solver, config_for_llm)
        _required_files_str = "\n".join(f"  - `{f}`" for f in _required_files)

        # fvOptions rule — injected as an explicit critical rule when cryogenic
        _fvoptions_rule = ""
        if "constant/fvOptions" in _required_files:
            _fvoptions_rule = (
                "\n14. **`constant/fvOptions` is REQUIRED** — this is a cryogenic fluid. "
                "Generate `constant/fvOptions` with a `limitTemperature` block "
                "(`min 1; max 100000; selectionMode all;`) to prevent 'Negative Temperature' "
                "divergence during startup. It MUST appear in your generated file list."
            )

        user_message = f"""## Task
Generate a complete OpenFOAM case for the following simulation.

## User Requirements
{requirements}

## Validated Configuration
```json
{json.dumps(config_for_llm, indent=2, default=str)}
```

## Selected Solver: {solver}
## Case Type: {case_type}
{constraint_warning}
## ⚠️  REQUIRED FILES CHECKLIST — you MUST generate ALL of these files
{_required_files_str}

Do NOT skip any file from this list.  Missing files will cause the simulation to fail.

## CRITICAL RULES
1. The `application` in controlDict MUST be `{solver}` — do NOT change it
2. {_pressure_rule}
3. Every patch in boundary_conditions MUST appear in EVERY 0/* field file
4. Do NOT generate blockMeshDict — we use an external mesh
5. {_thermo_rule}
6. {_g_rule}
7. {_t_rule}
8. Output files using ```file:path/to/file format
9. NEVER use `type empty;` unless the mesh patch type is `empty`
10. NEVER use `type symmetry;` unless the mesh patch type is `symmetry` or `symmetryPlane`
11. Do NOT invent patch names like `front_and_back` — only use patches from the config
12. If using a turbulence model, fvSchemes MUST include: wallDist {{ method meshWave; }}
13. controlDict `endTime` MUST be exactly `{max_iterations}`{_fvoptions_rule}

Follow the **Solver Instructions** section above for required files, fvSchemes, and fvSolution templates.

Generate ALL files from the checklist above now:"""
        
        return f"{system_prompt}\n\n{user_message}"
    
    def _build_patch_prompt(
        self,
        requirements: str,
        validated_config: dict[str, Any],
        solver: str,
        missing_files: list[str],
        existing_files: dict[str, str],
    ) -> str:
        """Build a prompt asking the LLM to generate ONLY the missing files.

        Used on retry when some files were already generated correctly and only
        a subset is missing.  Passing the existing files as context avoids
        contradictions (e.g. the LLM can see the patch names from 0/U).
        """
        solver_instructions = self._load_solver_prompt(solver)
        system_prompt = self._codegen_prompt
        if solver_instructions:
            system_prompt = (
                f"{system_prompt}\n\n"
                f"---\n\n## Solver Instructions — {solver}\n\n{solver_instructions}"
            )

        missing_str = "\n".join(f"  - `{f}`" for f in missing_files)
        existing_str = "\n".join(f"  - `{f}`" for f in sorted(existing_files))

        # Include existing files as read-only context so the LLM can see patches/dimensions
        existing_content_str = ""
        if existing_files:
            snippets = []
            for path, content in existing_files.items():
                # Truncate large files to keep prompt concise
                preview = content if len(content) <= 600 else content[:600] + "\n... (truncated)"
                snippets.append(f"```file:{path}\n{preview}\n```")
            existing_content_str = "\n\n".join(snippets)

        user_message = f"""## Task
Some required OpenFOAM files were missing from the previous generation attempt.
Generate ONLY the files listed under **Missing Files** below.
Do NOT regenerate the files already listed under **Existing Files**.

## Missing Files — generate ALL of these
{missing_str}

## Existing Files — DO NOT regenerate (shown for context only)
{existing_str}

{existing_content_str}

## User Requirements
{requirements}

## Validated Configuration
```json
{json.dumps(validated_config, indent=2, default=str)}
```

## Selected Solver: {solver}

## CRITICAL RULES
1. Output ONLY the missing files listed above using ```file:path format
2. application in controlDict = `{solver}`
3. Every patch from the existing files MUST appear in every new 0/* field file
4. Match dimensions/units from the existing files

Generate ONLY the missing files now:"""

        return f"{system_prompt}\n\n{user_message}"

    def _build_fix_prompt(
        self,
        requirements: str,
        validated_config: dict[str, Any],
        solver: str,
        previous_errors: list[dict],
        previous_files: dict[str, str],
    ) -> str:
        """Build a prompt to fix a case that failed on the simulation server.

        Includes the solver-specific instructions and required-files checklist
        so the LLM knows exactly what files the solver needs — same as the
        initial codegen prompt.
        """
        # ── Load solver-specific instructions ────────────────────────────────
        solver_instructions = self._load_solver_prompt(solver)
        system_prompt = self._codefix_prompt
        if solver_instructions:
            system_prompt = (
                f"{system_prompt}\n\n"
                f"---\n\n## Solver Instructions — {solver}\n\n{solver_instructions}"
            )

        # ── Solver-specific rules (same as codegen prompt) ────────────────────
        _required_files = build_required_files_list(solver, validated_config)
        _required_files_str = "\n".join(f"  - `{f}`" for f in _required_files)

        _pressure_rule = (
            f"You MUST generate `0/p_rgh` (NOT `0/p`) — {solver} reads `p_rgh`"
            if solver in P_RGH_SOLVERS
            else f"You MUST generate `0/p` (NOT `0/p_rgh`) — {solver} reads `p`"
        )
        _thermo_rule = (
            "You MUST generate `constant/thermophysicalProperties` (energy equation active)"
            if solver in THERMO_SOLVERS
            else "Do NOT generate `constant/thermophysicalProperties` (not needed for this solver)"
        )
        _g_rule = (
            "You MUST generate `constant/g` (VOF solver — always required)"
            if solver in GRAVITY_SOLVERS
            else "Do NOT generate `constant/g` (not needed for this solver)"
        )
        heat_transfer = (
            validated_config.get("heat_transfer")
            or validated_config.get("physics", {}).get("heat_transfer")
            or False
        )
        _t_rule = (
            "You MUST generate `0/T` (energy equation is active)"
            if (solver in ENERGY_SOLVERS or heat_transfer)
            else "Do NOT generate `0/T` unless heat_transfer is explicitly true"
        )

        # ── Format previous files and errors ─────────────────────────────────
        files_str = "\n\n".join(
            f"```file:{path}\n{content}\n```"
            for path, content in previous_files.items()
        )
        errors_str = "\n".join(
            f"- [{e.get('source', 'unknown')}] {e.get('error', 'Unknown error')}"
            + (f"\n  Details: {e.get('details', '')}" if e.get('details') else "")
            + (f"\n  Stderr:\n{e.get('stderr', '')}" if e.get('stderr') else "")
            for e in previous_errors
        )

        user_message = f"""## Task
Fix the OpenFOAM case that failed during execution on the simulation server.
Use the **Validated Configuration** below for ALL physical values (velocity, pressure,
temperature, viscosity, density, etc.) — do NOT invent or change values.

## Original Requirements
{requirements}

## Validated Configuration
```json
{json.dumps(validated_config, indent=2, default=str)}
```

## Selected Solver: {solver}

## ⚠️  REQUIRED FILES CHECKLIST — the fixed case MUST contain ALL of these
{_required_files_str}

## Previous Files (that failed — fix or regenerate as needed)
{files_str}

## Errors from Simulation Server
{errors_str}

## CRITICAL FIX RULES
1. The `application` in controlDict MUST be `{solver}` — do NOT change it
2. {_pressure_rule}
3. {_thermo_rule}
4. {_g_rule}
5. {_t_rule}
6. Every patch MUST appear in EVERY `0/*` field file — use EXACT names from config
7. Do NOT generate `system/blockMeshDict` — external mesh is used
8. ONLY use `type empty;` if the mesh patch type is `empty`
9. ONLY use `type symmetry;` if the mesh patch type is `symmetry` or `symmetryPlane`
10. If error says "not constraint type" — replace `empty`/`symmetry` with `zeroGradient`
11. If error mentions "wallDist" — add `wallDist {{ method meshWave; }}` to `system/fvSchemes`
12. Do NOT invent patch names — only use patches from the Validated Configuration
13. For 2D meshes: include `frontAndBack` with `type empty;` in ALL `0/*` files
14. fvSchemes MUST include `wallDist {{ method meshWave; }}` when using turbulence
15. Use physical values from the Validated Configuration — do NOT invent values

Follow the **Solver Instructions** section above for correct file templates.

Generate ALL corrected files now (all files from the checklist above):"""

        return f"{system_prompt}\n\n{user_message}"
    
    async def _call_genai(self, prompt: str) -> str:
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.15,
                    max_output_tokens=32000,
                    stop_sequences=["## End of Case"],
                ),
            )
            return response.text or ""
        except Exception as e:
            logger.error(f"[GENAI] API call failed: {e}")
            raise


# ────────────────────────────────────────────────────────────
# File extraction helpers
# ────────────────────────────────────────────────────────────

def _is_truncated(file_path: str, content: str) -> bool:
    """Detect obviously truncated OpenFOAM file content.

    A file is considered truncated if it starts correctly but doesn't close
    all its braces, or if a field file is missing its boundaryField section.
    """
    if not content or len(content) < 20:
        return True
    stripped = content.strip()
    # All OpenFOAM dict/field files must end with }
    if not stripped.endswith("}") and not stripped.endswith("// ************************************************************************* //"):
        return True
    # Field files (0/*) MUST contain boundaryField
    if file_path.startswith("0/") and "boundaryField" not in content:
        return True
    # Check brace balance
    opens = content.count("{")
    closes = content.count("}")
    if opens > 0 and abs(opens - closes) > 1:
        return True
    return False


def _strip_file_header(content: str) -> str:
    """Remove any stray ``file:path`` marker line that leaked into file content.

    The LLM sometimes echoes the ``file:path`` delimiter as the first line of
    the actual content.  OpenFOAM will choke on it, so strip it defensively.
    """
    # Strip a leading "```file:<path>" or "file:<path>" line
    content = re.sub(r"^```file:\S+\n?", "", content)
    content = re.sub(r"^file:\S+\n?", "", content)
    return content.strip()


def extract_file_blocks(text: str) -> dict[str, str]:
    """Extract ```file:path blocks from LLM output.

    Applies `_strip_file_header` to every extracted content block so that a
    stray ``file:path`` echo line never reaches the generated files.
    """
    files: dict[str, str] = {}

    pattern = r'```file:([^\n]+)\n(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)

    for path, content in matches:
        path = path.strip()
        content = _strip_file_header(content)
        if path and content:
            files[path] = content

    logger.info(f"[EXTRACT] Extracted {len(files)} files from LLM output")
    return files
