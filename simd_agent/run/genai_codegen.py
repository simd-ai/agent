# simd_agent/genai_codegen.py
"""OpenFOAM code generator using the LLM provider registry.

Generates complete OpenFOAM case files via the configured LLM provider.
Includes post-generation validation to catch inconsistencies before
sending to the simulation server.
"""

import asyncio
import copy
import json
import logging
import re
from pathlib import Path
from typing import Any

from simd_agent.llm import get_provider
from simd_agent.run.case_spec import CaseSpec, build_case_spec

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent / "prompts" / "packs" / "simd"
FLUIDS_DIR  = PROMPTS_DIR / "fluids"

# Maps fluid name fragments → fluid pack filename (without .md)
_FLUID_PACK_MAP: list[tuple[list[str], str]] = [
    (["liquid nitrogen", "ln2", " nitrogen"],   "liquidNitrogen"),
    (["liquid hydrogen", "lh2", " hydrogen"],   "liquidHydrogen"),
    (["liquid oxygen",   "lox", " oxygen"],     "liquidOxygen"),
    (["liquid helium",   "lhe", " helium"],     "liquidHelium"),
    (["liquid methane",  "lng", " methane"],    "liquidMethane"),
    (["liquid argon",    "lar", " argon"],      "liquidArgon"),
]


def _load_fluid_pack(fluid_name: str) -> str:
    """Load the cryogenic fluid-specific prompt pack, if one exists.

    Matches fluid name against known cryogenic fluids and returns the
    markdown content of the matching pack file.  Returns empty string when
    no match is found (non-cryogenic or unknown fluid).
    """
    if not fluid_name:
        return ""
    lower = fluid_name.lower()
    for keywords, pack_name in _FLUID_PACK_MAP:
        if any(kw in lower for kw in keywords):
            pack_path = FLUIDS_DIR / f"{pack_name}.md"
            if pack_path.exists():
                return pack_path.read_text(encoding="utf-8")
            break  # matched but file missing — no fallback
    return ""


# Re-export canonical sets from solver_selector so callers can import from here
from simd_agent.run.solver_selector import (  # noqa: E402 — intentional late import
    ALLOWED_SOLVERS,
    P_SOLVERS,
    P_RGH_SOLVERS,
    ENERGY_SOLVERS,
    GRAVITY_SOLVERS,
    THERMO_SOLVERS,
)



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

    **Single-region only.**  Multi-region (CHT) cases have a flat ``0/<field>``
    file layout that does not exist — fields live under ``0/<region>/`` and
    every single-region check here would either misfire or quietly damage
    the per-region tree.  When this function is called for a multi-region
    solver it returns the input untouched with a loud log line so the bug
    is visible upstream.  See :mod:`simd_agent.run.multi_region` for the
    multi-region equivalents.

    Returns:
        Tuple of (possibly-fixed files dict, list of issues found)
    """
    # Defensive depth — single-region checks must NEVER run against a
    # multi-region case.  base.py:validate_full also gates this call from
    # the caller side, but a second self-defending guard here means an
    # unguarded import-and-call (e.g. tests, future refactors) is safe.
    from simd_agent.solvers import is_multi_region_solver
    if is_multi_region_solver(solver):
        logger.warning(
            "[VALIDATE] validate_generated_files called for multi-region "
            "solver %r — skipped (single-region validator does not "
            "understand the per-region case tree)",
            solver,
        )
        return dict(files), []

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
    turb_model    = config.get("turbulence_model", "") or ""
    # Legacy fallback: physics sub-dict (older format)
    _physics = config.get("physics", {})
    if not heat_transfer:
        heat_transfer = _physics.get("heat_transfer", False)
    if not turb_model:
        turb_model = _physics.get("turbulence_model", "") or ""
    # flow_regime overrides turbulence_model — laminar always wins
    flow_regime = config.get("flow_regime") or _physics.get("flow_regime", "")
    if flow_regime == "laminar" or not turb_model:
        turb_model = "laminar"
    
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
                # Loud log so a solver-swap attempt by the LLM is obvious
                # in production output.  This is a defensive correction:
                # the selected solver is set ONCE at run start by the
                # orchestrator and must never be changed during retries.
                logger.warning(
                    "[VALIDATE] ⚠ LLM attempted solver swap in controlDict: "
                    "'application %s;' → corrected to 'application %s;' "
                    "(orchestrator-locked solver)",
                    declared_solver, solver,
                )
                print(
                    f"\n{'='*70}\n"
                    f"[VALIDATE] ⚠ SOLVER-SWAP ATTEMPT BLOCKED\n"
                    f"  LLM wrote: application {declared_solver};\n"
                    f"  Corrected to: application {solver};\n"
                    f"  The selected solver is locked for the duration of this run.\n"
                    f"{'='*70}\n"
                )
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
        # buoyantSimpleFoam / buoyantPimpleFoam: need BOTH 0/p_rgh (solved) and 0/p (calculated).
        # p_rgh is the primary solved pressure; p is derived as p = p_rgh + rho*(g·x).
        if "0/p_rgh" in fixed_files and "0/p" not in fixed_files:
            content = fixed_files["0/p_rgh"].replace("object      p_rgh;", "object      p;").replace("object p_rgh;", "object p;")
            fixed_files["0/p"] = content
            issues.append(ValidationIssue("warning", "0/p", f"'{solver}' needs both 0/p_rgh and 0/p. Synthesised 0/p from 0/p_rgh."))
        elif "0/p" in fixed_files and "0/p_rgh" not in fixed_files:
            content = fixed_files["0/p"].replace("object      p;", "object      p_rgh;").replace("object p;", "object p_rgh;")
            fixed_files["0/p_rgh"] = content
            issues.append(ValidationIssue("warning", "0/p_rgh", f"'{solver}' needs both 0/p_rgh and 0/p. Synthesised 0/p_rgh from 0/p."))
        # If both exist — correct, leave as-is

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
    # Apply to base file AND all per-phase thermo files (thermophysicalProperties.*)
    _thermo_path = "constant/thermophysicalProperties"
    _thermo_file_paths = [
        k for k in fixed_files
        if k == _thermo_path or k.startswith("constant/thermophysicalProperties.")
    ]
    for _tp_path in _thermo_file_paths:
        _tp_content = fixed_files[_tp_path]
        # Match "thermodynamics  <model>;" where model is a single word (hConst, janaf, etc.)
        # but NOT "thermodynamics  {" (which is the legitimate sub-dict in mixture)
        _fixed_tp = re.sub(
            r'\bthermodynamics(\s+)(hConst|eConst|janaf|hTabular|eTabular|hPolynomial|ePolynomial|hIcoTabular|eIcoTabular)\s*;',
            r'thermo\1\2;',
            _tp_content,
        )
        if _fixed_tp != _tp_content:
            issues.append(ValidationIssue(
                "warning", _tp_path,
                "Auto-fixed: replaced 'thermodynamics <model>' with 'thermo <model>' in thermoType block "
                "(OpenFOAM 2406 requires 'thermo' as the key inside thermoType{}).",
                fix="thermodynamics → thermo in thermoType block"
            ))
            fixed_files[_tp_path] = _fixed_tp
            logger.info(
                f"[VALIDATE] Auto-fixed thermoType key: 'thermodynamics' → 'thermo' in {_tp_path}"
            )

    # ── Check 3b2-eos: rhoConst → icoPolynomial for liquid with heat transfer ──
    # When heat_transfer=True and fluid density > 200 kg/m³ (liquid), rhoConst is
    # physically wrong — density never changes with temperature, breaking energy coupling.
    # icoPolynomial must be used so ρ(T) = a0 + a1·T captures the real density variation.
    if _thermo_path in fixed_files:
        _tp2 = fixed_files[_thermo_path]
        if "rhoConst" in _tp2:
            _physics_cfg2 = config.get("physics") or {}
            _has_energy2 = bool(_physics_cfg2.get("heat_transfer") or _physics_cfg2.get("energy"))
            _fluid2 = config.get("fluid") or {}
            try:
                _rho2 = float(_fluid2.get("density") or _fluid2.get("rho") or 0)
            except (TypeError, ValueError):
                _rho2 = 0.0
            _eos_needed = _eos_for_liquid(_rho2, None, _has_energy2) if _rho2 > 0 else "rhoConst"
            if _eos_needed == "icoPolynomial":
                # Compute inlet temperature from BCs
                _bcs2 = config.get("boundary_conditions") or {}
                _inlet_t2: float | None = None
                for _pbc2 in _bcs2.values():
                    if isinstance(_pbc2, dict):
                        _t_bc = (_pbc2.get("temperature") or {})
                        if isinstance(_t_bc, dict) and _t_bc.get("type") == "fixedValue":
                            _v = _t_bc.get("value")
                            try:
                                _inlet_t2 = float(_v)
                                break
                            except (TypeError, ValueError):
                                pass
                _a0_fix, _a1_fix = _ico_poly_coeffs(_rho2, _inlet_t2)
                _mu2 = float(_fluid2.get("dynamic_viscosity") or _fluid2.get("mu") or 1e-3)
                _cp2 = float(_fluid2.get("specific_heat") or _fluid2.get("Cp") or 1000.0)
                _kappa2 = float(_fluid2.get("thermal_conductivity") or _fluid2.get("k") or 0.026)
                _mw2 = float(_fluid2.get("molar_mass") or _fluid2.get("molWeight") or 28.0)

                # Build a replacement thermophysicalProperties using icoPolynomial
                _tp2_fixed = (
                    "FoamFile\n{\n"
                    "    version     2.0;\n    format      ascii;\n"
                    "    class       dictionary;\n    location    \"constant\";\n"
                    "    object      thermophysicalProperties;\n}\n"
                    "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
                    "thermoType\n{\n"
                    "    type            heRhoThermo;\n"
                    "    mixture         pureMixture;\n"
                    "    transport       polynomial;\n"
                    "    thermo          hPolynomial;\n"
                    "    equationOfState icoPolynomial;\n"
                    "    specie          specie;\n"
                    "    energy          sensibleEnthalpy;\n"
                    "}\n\n"
                    "mixture\n{\n"
                    f"    specie\n    {{\n        molWeight   {_mw2:.3f};\n    }}\n"
                    "    thermodynamics\n    {\n"
                    "        Hf              0;\n        Sf              0;\n"
                    f"        CpCoeffs<8>     ({_cp2} 0 0 0 0 0 0 0);\n"
                    "    }\n"
                    "    transport\n    {\n"
                    f"        muCoeffs<8>     ({_mu2} 0 0 0 0 0 0 0);\n"
                    f"        kappaCoeffs<8>  ({_kappa2} 0 0 0 0 0 0 0);\n"
                    "    }\n"
                    "    equationOfState\n    {\n"
                    f"        rhoCoeffs<8>    ({_a0_fix:.4f} {_a1_fix:.4f} 0 0 0 0 0 0);\n"
                    "    }\n"
                    "}\n\n"
                    "// ************************************************************************* //\n"
                )
                fixed_files[_thermo_path] = _tp2_fixed
                issues.append(ValidationIssue(
                    "warning", _thermo_path,
                    f"Auto-fixed: replaced rhoConst (fixed ρ={_rho2} kg/m³) with icoPolynomial "
                    f"(ρ(T) = {_a0_fix:.1f} + {_a1_fix:.2f}·T) — heat transfer is active, "
                    "density must vary with temperature for correct energy coupling.",
                    fix="rhoConst → icoPolynomial"
                ))
                logger.info(
                    f"[VALIDATE] Auto-fixed rhoConst → icoPolynomial "
                    f"(a0={_a0_fix:.2f}, a1={_a1_fix:.2f}, rho_ref={_rho2} at T={_inlet_t2} K)"
                )

    # NOTE: Check 3d-turbProps (auto-regen of constant/turbulenceProperties)
    # was deleted in Phase 4.  The file is now rendered up-front by
    # ``SolverPlugin._build_turbulence_properties()`` via
    # ``render_deterministic_files()``; the LLM never sees it.

    # ── Check 3e-laminarFields: Strip turbulence fields when flow is laminar ──
    # If CaseSpec correctly set turb_model=laminar but the LLM still generated 0/k,
    # 0/omega, 0/nut, 0/epsilon, or 0/nuTilda, remove them — they are not read by
    # the solver and will generate spurious warnings or confuse post-processing.
    _TURB_ONLY_FIELDS = {"0/k", "0/omega", "0/nut", "0/epsilon", "0/nuTilda", "0/mut", "0/alphat"}
    if turb_model in ("laminar", "none", ""):
        for _tf in list(fixed_files.keys()):
            if _tf in _TURB_ONLY_FIELDS:
                del fixed_files[_tf]
                issues.append(ValidationIssue(
                    "warning", _tf,
                    f"Removed '{_tf}': flow is laminar — turbulence fields must not be present.",
                    fix=f"Deleted {_tf} (laminar flow)"
                ))
                logger.info(f"[VALIDATE] Removed {_tf}: laminar flow, turbulence field not needed")

    # ── Check 3f-flowRateInletVelocity: Fix rho/massFlowRate syntax in 0/U ──────
    # Per OF 2306/2406 docs (flowRateInletVelocity):
    #   `rho`     — optional word: name of density field (default "rho").
    #              If "none", flow rate is treated as volumetric.
    #   `rhoInlet`— optional scalar: startup fallback when density field not in DB.
    #
    # For incompressible solvers (pimpleFoam, simpleFoam) — NO rho field:
    #   • `rho 880;`  → FOAM FATAL IO ERROR: "expected word, found double 880"
    #   • `rho rho;` (without rhoInlet) → FOAM crash: rho field not found, no fallback
    #   • `massFlowRate + rhoInlet` → VALID per docs (fallback scalar; no `rho` word needed)
    #   • `volumetricFlowRate` → PREFERRED (cleanest — no density keyword at all)
    #
    # Strategy: convert massFlowRate → volumetricFlowRate when rho is available (cleanest).
    # If no rho in config, strip rho <word> and leave massFlowRate + rhoInlet (also valid).
    # The `constant` qualifier is REQUIRED in OF 2406 for Function1<scalar> values.
    _INCOMPRESSIBLE_SOLVERS = {"pimpleFoam", "simpleFoam"}
    _COMPRESSIBLE_RHO_SOLVERS = {"rhoPimpleFoam", "rhoSimpleFoam", "buoyantPimpleFoam", "buoyantSimpleFoam"}
    _u_content = fixed_files.get("0/U", "")
    if _u_content and "flowRateInletVelocity" in _u_content:
        _u_fixed = _u_content

        if solver in _INCOMPRESSIBLE_SOLVERS:
            # Remove any `rho <word>;` line (rho rho; rho none; rho rhoField; etc.)
            _u_fixed = re.sub(r'\n\s*rho\s+\w+\s*;', '', _u_fixed)

            # If massFlowRate is present, convert to volumetricFlowRate using config rho
            if "massFlowRate" in _u_fixed:
                _fluid_fr = config.get("fluid", {}) or {}
                _rho_fr = None
                try:
                    _rho_fr = float(
                        _fluid_fr.get("rho") or _fluid_fr.get("density") or 0
                    ) or None
                except (TypeError, ValueError):
                    pass
                if _rho_fr and _rho_fr > 0:
                    _mdot_match = re.search(
                        r'\bmassFlowRate\s+(?:constant\s+)?([\d.eE+\-]+)\s*;', _u_fixed
                    )
                    if _mdot_match:
                        try:
                            _mdot_val = float(_mdot_match.group(1))
                            _q_val = _mdot_val / _rho_fr
                            _u_fixed = re.sub(
                                r'\bmassFlowRate\s+(?:constant\s+)?[\d.eE+\-]+\s*;',
                                f'massFlowRate    constant {_q_val:.8g};',
                                _u_fixed,
                            )
                            # Rename keyword
                            _u_fixed = _u_fixed.replace(
                                f'massFlowRate    constant {_q_val:.8g};',
                                f'volumetricFlowRate constant {_q_val:.8g};',
                            )
                            # Also remove rhoInlet if present (not needed for volumetric)
                            _u_fixed = re.sub(r'\n\s*rhoInlet\s+[\d.eE+\-]+\s*;', '', _u_fixed)
                            issues.append(ValidationIssue(
                                "warning", "0/U",
                                f"Auto-converted massFlowRate → volumetricFlowRate={_q_val:.6g} m³/s "
                                f"(= {_mdot_val} kg/s ÷ {_rho_fr} kg/m³). "
                                f"'{solver}' has no rho field — massFlowRate is not valid.",
                                fix="massFlowRate → volumetricFlowRate"
                            ))
                            logger.info(
                                f"[VALIDATE] 0/U: massFlowRate {_mdot_val} → "
                                f"volumetricFlowRate {_q_val:.6g} for {solver}"
                            )
                        except (ValueError, TypeError):
                            pass

        # For ALL solvers: ensure Function1 `constant` qualifier is present
        # OF 2406 flowRateInletVelocity uses Function1<scalar>; bare numbers may fail.
        for _fr_kw in ("massFlowRate", "volumetricFlowRate"):
            # Match `keyword <number>;` but NOT `keyword constant <number>;`
            _u_fixed = re.sub(
                rf'\b({_fr_kw})\s+(?!constant\b)([\d.eE+\-]+)\s*;',
                lambda m: f'{m.group(1)}    constant {m.group(2)};',
                _u_fixed,
            )

        if _u_fixed != _u_content:
            fixed_files["0/U"] = _u_fixed

    # ── Check 3h-deltaT: Enforce deltaT and adjustTimeStep for transient solvers ──
    # The LLM often picks deltaT values orders of magnitude too small (e.g. 0.0001 instead
    # of 0.001), multiplying runtime by 10x with no physical benefit.  For transient solvers
    # we also inject `adjustTimeStep yes; maxCo 0.5;` so OpenFOAM can push deltaT as high
    # as the Courant limit allows — this gives maximum speed without instability.
    _ctrl = fixed_files.get("system/controlDict", "")
    if _ctrl:
        # Build the CaseSpec from config to get canonical deltaT and transient flag
        _cs_solver_props = {
            "simpleFoam": False, "rhoSimpleFoam": False,
            "buoyantSimpleFoam": False,
            "pimpleFoam": True, "rhoPimpleFoam": True,
            "buoyantPimpleFoam": True,
        }
        _is_transient = _cs_solver_props.get(solver, True)
        _canonical_dt = float(config.get("delta_t") or 0.001)

        # Compute function object write interval for transient solvers.
        # Uses runTime control at 2x the file writeInterval so the convergence
        # chart gets ~50 data points — enough for trends, not enough to choke
        # the frontend with 19K+ SVG points.
        if _is_transient:
            _solver_cfg = config.get("solver", {}) or {}
            _cfg_end_time = float(
                config.get("end_time")
                or _solver_cfg.get("end_time")
                or _solver_cfg.get("endTime")
                or 10.0
            )
            _target_snaps = max(30, min(100, int(_cfg_end_time * 10)))
            _file_write_int = _cfg_end_time / _target_snaps
            _func_write_int = _file_write_int * 2
        else:
            _func_write_int = None  # steady: keep timeStep/1

        # Correct deltaT if LLM went below the canonical value
        _dt_match = re.search(r'\bdeltaT\s+([\d.eE+\-]+)\s*;', _ctrl)
        if _dt_match:
            try:
                _llm_dt = float(_dt_match.group(1))
                if _llm_dt < _canonical_dt * 0.99:  # 1% tolerance
                    _ctrl = re.sub(
                        r'\bdeltaT\s+[\d.eE+\-]+\s*;',
                        f'deltaT          {_canonical_dt};',
                        _ctrl,
                    )
                    issues.append(ValidationIssue(
                        "warning", "system/controlDict",
                        f"Auto-corrected deltaT: LLM wrote {_llm_dt} but canonical value is "
                        f"{_canonical_dt}. Using {_canonical_dt} to avoid unnecessarily slow simulation.",
                        fix=f"deltaT {_llm_dt} → {_canonical_dt}"
                    ))
                    logger.info(
                        f"[VALIDATE] Corrected deltaT {_llm_dt} → {_canonical_dt} in controlDict"
                    )
            except (ValueError, TypeError):
                pass

        # Inject adjustTimeStep + maxCo for transient solvers if missing
        if _is_transient and "adjustTimeStep" not in _ctrl:
            # PIMPLE handles higher Courant numbers via outer corrector loops
            _fallback_maxco = 2.0 if solver in ("pimpleFoam", "rhoPimpleFoam", "buoyantPimpleFoam") else 0.5
            # Insert after the deltaT line
            _ctrl = re.sub(
                r'(deltaT\s+[\d.eE+\-]+\s*;)',
                rf'\1\n\nadjustTimeStep  yes;\nmaxCo           {_fallback_maxco};',
                _ctrl,
                count=1,
            )
            # Switch writeControl to adjustableRunTime if it's using runTime
            _ctrl = re.sub(
                r'writeControl\s+runTime\s*;',
                'writeControl    adjustableRunTime;',
                _ctrl,
            )
            issues.append(ValidationIssue(
                "warning", "system/controlDict",
                f"Auto-injected 'adjustTimeStep yes; maxCo {_fallback_maxco};' — lets OpenFOAM auto-scale "
                "deltaT up to the Courant limit so the simulation runs as fast as physics allows.",
                fix=f"Added adjustTimeStep + maxCo {_fallback_maxco}"
            ))
            logger.info(f"[VALIDATE] Injected adjustTimeStep yes; maxCo {_fallback_maxco} into controlDict")

        # Determine solver classification flags (used by fieldMinMax, surfaceFieldValue, volFieldValue)
        _is_buoyant = solver in ("buoyantSimpleFoam", "buoyantPimpleFoam")
        _is_multiphase = solver in (
            "interFoam", "interIsoFoam",
            "compressibleInterFoam", "compressibleInterIsoFoam",
            "compressibleMultiphaseInterFoam",
        )

        # ── Inject fieldMinMax function object for convergence monitoring ──
        # Outputs min/max of solved fields each iteration → parsed from run_log
        # events in the orchestrator and forwarded to the frontend as field_ranges.
        if "fieldMinMax" not in _ctrl:
            _minmax_fields = ["U"]
            # Pressure field depends on solver
            if _is_buoyant:
                _minmax_fields.extend(["p_rgh", "p"])
            elif _is_multiphase:
                _minmax_fields.extend(["p_rgh", "p"])
            else:
                _minmax_fields.append("p")
            # Temperature for any energy solver
            if heat_transfer or _is_buoyant:
                _minmax_fields.append("T")
            # Turbulence fields
            if turb_model and turb_model != "laminar":
                if "kOmega" in turb_model or turb_model == "kOmegaSST":
                    _minmax_fields.extend(["k", "omega"])
                elif "kEpsilon" in turb_model:
                    _minmax_fields.extend(["k", "epsilon"])
            # Phase fraction for multiphase
            if _is_multiphase:
                _minmax_fields.append("alpha.water")
            _fields_str = " ".join(_minmax_fields)

            if _func_write_int is not None:
                _fmm_wctrl = "runTime"
                _fmm_wint = f"{_func_write_int:.6g}"
            else:
                _fmm_wctrl = "timeStep"
                _fmm_wint = "1"
            _fmm_block = f"""
    fieldMinMax
    {{
        type            fieldMinMax;
        libs            (fieldFunctionObjects);
        fields          ({_fields_str});
        writeControl    {_fmm_wctrl};
        writeInterval   {_fmm_wint};
        log             true;
    }}"""
            if "functions" in _ctrl:
                # Merge into existing functions block
                _ctrl = re.sub(
                    r'(functions\s*\{)',
                    r'\1' + _fmm_block,
                    _ctrl,
                    count=1,
                )
            else:
                _ctrl += f"""

functions
{{{_fmm_block}
}}
"""
            logger.info("[VALIDATE] Injected fieldMinMax function object into controlDict")

        # ── Inject surfaceFieldValue function objects for patch-averaged values ──
        # Tracks per-patch averages each iteration → parsed from run_log in
        # the orchestrator as patch_values (pressure drop, temperature drop, etc.)
        # One function object per patch for clean log output.
        if "surfaceFieldValue" not in _ctrl:
            _sfv_patches: list[str] = []
            for _sfv_pname, _sfv_pbc in bcs.items():
                if not isinstance(_sfv_pbc, dict):
                    continue
                _sfv_pt = _sfv_pbc.get("patch_type", "")
                if _sfv_pt in ("inlet", "pressure_inlet", "mass_flow_inlet",
                               "outlet", "pressure_outlet"):
                    _sfv_patches.append(_sfv_pname)
                elif _sfv_pname.lower() in ("inlet", "outlet"):
                    _sfv_patches.append(_sfv_pname)

            if _sfv_patches:
                _sfv_fields = ["p"]
                if heat_transfer or _is_buoyant:
                    _sfv_fields.append("T")
                _sfv_fields_str = " ".join(_sfv_fields)

                # Build one function object per patch — OF2406 requires:
                # type, libs, writeFields, regionType, name (=patch name),
                # fields, operation, writeControl, writeInterval, log
                if _func_write_int is not None:
                    _sfv_wctrl = "runTime"
                    _sfv_wint = f"{_func_write_int:.6g}"
                else:
                    _sfv_wctrl = "timeStep"
                    _sfv_wint = "1"
                _sfv_blocks = ""
                for _sfv_pn in _sfv_patches:
                    _sfv_blocks += f"""
    patchAvg_{_sfv_pn}
    {{
        type            surfaceFieldValue;
        libs            (fieldFunctionObjects);
        regionType      patch;
        name            {_sfv_pn};
        fields          ({_sfv_fields_str});
        operation       areaAverage;
        writeFields     false;
        writeControl    {_sfv_wctrl};
        writeInterval   {_sfv_wint};
        log             true;
    }}"""

                if _sfv_blocks:
                    if "functions" in _ctrl:
                        _ctrl = re.sub(
                            r'(functions\s*\{)',
                            r'\1' + _sfv_blocks,
                            _ctrl,
                            count=1,
                        )
                    else:
                        _ctrl += f"""

functions
{{{_sfv_blocks}
}}
"""
                    logger.info(
                        "[VALIDATE] Injected surfaceFieldValue (areaAverage) for patches: %s, fields: %s",
                        _sfv_patches, _sfv_fields,
                    )

        # ── Inject volFieldValue function objects for domain-wide volume integrals ──
        # Outputs volume-averaged or volume-integrated values each iteration →
        # parsed from run_log events in the orchestrator and forwarded to the
        # frontend as volume_integrals (domain-wide pressure avg, temperature avg,
        # liquid volume for multiphase).
        if "volFieldValue" not in _ctrl:
            _vol_fields: list[tuple[str, str]] = []  # (field, operation)

            # Volume-averaged pressure (all solvers)
            if _is_buoyant:
                _vol_fields.append(("p_rgh", "volAverage"))
                _vol_fields.append(("p", "volAverage"))
            elif _is_multiphase:
                _vol_fields.append(("p_rgh", "volAverage"))
            else:
                _vol_fields.append(("p", "volAverage"))

            # Volume-averaged temperature (energy solvers)
            if heat_transfer or _is_buoyant:
                _vol_fields.append(("T", "volAverage"))

            # Liquid volume fraction integral (multiphase — gives liquid volume in m³)
            if _is_multiphase:
                _vol_fields.append(("alpha.water", "volIntegrate"))

            # Build one function object per (field, operation) pair
            if _func_write_int is not None:
                _vfv_wctrl = "runTime"
                _vfv_wint = f"{_func_write_int:.6g}"
            else:
                _vfv_wctrl = "timeStep"
                _vfv_wint = "1"
            _vfv_blocks = ""
            for _vf_field, _vf_op in _vol_fields:
                _safe_name = _vf_field.replace(".", "_")
                _vfv_blocks += f"""
    vol_{_vf_op}_{_safe_name}
    {{
        type            volFieldValue;
        libs            (fieldFunctionObjects);
        fields          ({_vf_field});
        operation       {_vf_op};
        regionType      all;
        writeFields     false;
        writeControl    {_vfv_wctrl};
        writeInterval   {_vfv_wint};
        log             true;
    }}"""

            if _vfv_blocks:
                if "functions" in _ctrl:
                    _ctrl = re.sub(
                        r'(functions\s*\{)',
                        r'\1' + _vfv_blocks,
                        _ctrl,
                        count=1,
                    )
                else:
                    _ctrl += f"""

functions
{{{_vfv_blocks}
}}
"""
                logger.info(
                    "[VALIDATE] Injected volFieldValue for: %s",
                    [(f, o) for f, o in _vol_fields],
                )

        fixed_files["system/controlDict"] = _ctrl

    # ── Check 3c-fvOptions: Auto-inject system/fvOptions for compressible energy solvers ──
    # Temperature can diverge to negative values during early iterations due to numerical
    # overshoot on boundary interfaces.  A limitTemperature fvOption acts as a safety net
    # for all compressible single-phase energy solvers.
    _FVOPTIONS_ENERGY_SOLVERS = {
        "rhoPimpleFoam", "rhoSimpleFoam",
        "buoyantSimpleFoam", "buoyantPimpleFoam",
    }
    _eos_ceiling: float | None = None
    _t_floor: float = 1.0
    if solver in _FVOPTIONS_ENERGY_SOLVERS:
        # Compute BC temperature floor and EOS ceiling — used by both the fallback
        # injection (when LLM omitted fvOptions) and the clamp check (when LLM generated
        # it but chose a max above the icoPolynomial zero-density point).
        _bcs = config.get("boundary_conditions", {}) or {}
        _t_vals: list[float] = []
        for _patch_bc in _bcs.values():
            if not isinstance(_patch_bc, dict):
                continue
            for _t_key in ("temperature", "T"):
                _t_raw = _patch_bc.get(_t_key)
                if isinstance(_t_raw, dict):
                    _tv = _t_raw.get("value") or _t_raw.get("uniform")
                elif isinstance(_t_raw, (int, float)):
                    _tv = float(_t_raw)
                else:
                    _tv = None
                if _tv is not None:
                    try:
                        _t_vals.append(float(_tv))
                    except (TypeError, ValueError):
                        pass
        _min_bc_t = min(_t_vals) if _t_vals else 200.0
        _t_floor = max(1.0, _min_bc_t * 0.5)
        _fluid_cfg = config.get("fluid") or {}
        _cfg_rho = _fluid_cfg.get("rho") or _fluid_cfg.get("density")
        if _cfg_rho is not None:
            try:
                _cfg_rho_f = float(_cfg_rho)
                _physics_cfg = config.get("physics") or {}
                _has_energy_cfg = bool(_physics_cfg.get("heat_transfer") or _physics_cfg.get("energy"))
                if _eos_for_liquid(_cfg_rho_f, _min_bc_t, _has_energy_cfg) == "icoPolynomial":
                    _a0_fv, _a1_fv = _ico_poly_coeffs(_cfg_rho_f, _min_bc_t)
                    if abs(_a1_fv) > 1e-10:
                        _eos_ceiling = abs(_a0_fv) / abs(_a1_fv)
            except (TypeError, ValueError):
                pass

        if "system/fvOptions" not in fixed_files:
            # Fallback injection — only when wall is genuinely hotter than inlet.
            # Isothermal (wall_T ≈ inlet_T) and adiabatic (no wall T) do not need
            # fvOptions: temperature never drifts above inlet T → EOS ceiling is safe.
            # Only inject when wall_T > inlet_T + 10 K (actual heat exchange).
            _fv_bcs = config.get("boundary_conditions") or {}
            _fv_inlet_t: float | None = None
            _fv_wall_temps: list[float] = []
            for _fv_pname, _fv_pbc in _fv_bcs.items():
                if not isinstance(_fv_pbc, dict):
                    continue
                _fv_pt = _fv_pbc.get("patch_type", "")
                _fv_t_entry = _fv_pbc.get("temperature") or _fv_pbc.get("T")
                _fv_tv = (_fv_t_entry.get("value") if isinstance(_fv_t_entry, dict) else _fv_t_entry)
                if _fv_tv is None:
                    continue
                try:
                    _fv_tv_f = float(_fv_tv)
                except (TypeError, ValueError):
                    continue
                if _fv_pt in ("inlet", "pressure_inlet", "mass_flow_inlet") or _fv_pname == "inlet":
                    _fv_inlet_t = _fv_tv_f
                elif _fv_pname not in ("outlet",):
                    _fv_wall_temps.append(_fv_tv_f)

            _fv_wall_heats = False
            if _fv_wall_temps:
                _fv_max_wall = max(_fv_wall_temps)
                if _fv_inlet_t is not None:
                    _fv_wall_heats = _fv_max_wall > _fv_inlet_t + 10.0
                else:
                    _fv_wall_heats = _fv_max_wall > 200.0  # conservative for unknown inlet T

            if _fv_wall_heats:
                _t_ceil = (_eos_ceiling * 0.9) if _eos_ceiling else 100000.0
                _t_ceil = max(_t_floor * 3.0, _t_ceil)
                _fvoptions_content = (
                    "FoamFile\n"
                    "{\n"
                    "    version     2.0;\n"
                    "    format      ascii;\n"
                    "    class       dictionary;\n"
                    "    location    \"system\";\n"
                    "    object      fvOptions;\n"
                    "}\n"
                    "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
                    "temperatureLimiter\n"
                    "{\n"
                    "    type            limitTemperature;\n"
                    "    active          yes;\n\n"
                    "    selectionMode   all;\n\n"
                    f"    min             {_t_floor:.1f};   // 50% of coldest BC temperature [K]\n"
                    f"    max             {_t_ceil:.0f};  // 90% of EOS ceiling = same as wall BC clamp in 0/T\n"
                    "}\n\n"
                    "// ************************************************************************* //\n"
                )
                fixed_files["system/fvOptions"] = _fvoptions_content
                issues.append(ValidationIssue(
                    "warning", "system/fvOptions",
                    "Auto-injected system/fvOptions with limitTemperature: wall is hotter than "
                    f"inlet ({_fv_max_wall:.0f} K vs {_fv_inlet_t or '?'} K inlet). "
                    "This prevents 'Negative Temperature' divergence during startup iterations.",
                    fix=f"Added system/fvOptions {{ limitTemperature {{ min {_t_floor:.1f}; max {_t_ceil:.0f}; }} }}"
                ))
                logger.info(
                    f"[VALIDATE] Auto-injected system/fvOptions for {solver}: "
                    f"wall_T={_fv_max_wall:.0f}K > inlet_T={_fv_inlet_t or '?'}K"
                )

    # ── Check 3c2 (Phase 2): clamp LLM's fvOptions.max to resolver's choice ──
    # The resolver `resolve_fv_options_max` knows the right value (gas:
    # min(3000, max(BC_T)·1.5); cryogenic: 0.9 × EOS ceiling).  This validator
    # is now a thin guard that only fires when the LLM picked a value that
    # would either crash the solver (cryogenic ≥ ceiling) or is clearly
    # nonphysical (gas > 1.5× the resolver's value).  Replaces the 80-line
    # branch that used to live here.
    if solver in _FVOPTIONS_ENERGY_SOLVERS and "system/fvOptions" in fixed_files:
        from simd_agent.run.case_spec import resolve_fv_options_max
        _fvo = fixed_files["system/fvOptions"]
        import re as _re_fvo
        _max_match = _re_fvo.search(r'\bmax\s+([\d.eE+\-]+)\s*;', _fvo)
        if _max_match:
            try:
                _llm_max = float(_max_match.group(1))
                _bcs_fvo = config.get("boundary_conditions") or {}
                _bc_t_vals: list[float] = []
                for _pbc in _bcs_fvo.values():
                    if not isinstance(_pbc, dict):
                        continue
                    _t_e = _pbc.get("temperature") or _pbc.get("T")
                    _t_v = (
                        _t_e.get("value") or _t_e.get("uniform")
                        if isinstance(_t_e, dict) else _t_e
                    )
                    try:
                        _bc_t_vals.append(float(_t_v))
                    except (TypeError, ValueError):
                        pass

                _profile = "cryogenic" if _eos_ceiling is not None else "gas"
                _resolved_max = resolve_fv_options_max(
                    profile=_profile,
                    bc_temps=_bc_t_vals,
                    eos_t_ceiling=_eos_ceiling,
                    t_floor=_t_floor,
                )

                _needs_clamp = (
                    (_eos_ceiling is not None and _llm_max >= _eos_ceiling)
                    or _llm_max > _resolved_max * 1.5
                )
                if _needs_clamp:
                    fixed_files["system/fvOptions"] = _re_fvo.sub(
                        r'\bmax\s+[\d.eE+\-]+\s*;',
                        f'max             {_resolved_max:.0f};',
                        _fvo,
                        count=1,
                    )
                    issues.append(ValidationIssue(
                        "warning", "system/fvOptions",
                        f"fvOptions max={_llm_max:.0f} K clamped to {_resolved_max:.0f} K "
                        f"(profile={_profile}, resolved by resolve_fv_options_max).",
                        fix=f"Changed max from {_llm_max:.0f} to {_resolved_max:.0f} K"
                    ))
            except (ValueError, AttributeError):
                pass

    # ── Check 3c3: Clamp fixedValue temperatures in 0/T to EOS ceiling ──────────
    # limitTemperature (fvOptions) only applies to internal cell values.
    # Boundary face values are set by the BC directly — a wall fixedValue at 400K
    # still gives ρ(400K) = negative at that face even when fvOptions max=200K.
    # The turbulence model accesses boundary T when computing alphat/nut → divide
    # by negative ρ → SIGFPE.  Clamp any fixedValue temperature above the EOS
    # ceiling in 0/T itself so the face value is always physically valid.
    if solver in _COMPRESSIBLE_RHO_SOLVERS and _eos_ceiling is not None and "0/T" in fixed_files:
        import re as _re_T
        _t_content = fixed_files["0/T"]
        _safe_wall_t = _eos_ceiling * 0.9

        def _clamp_fixedvalue_T(m: "re.Match") -> str:  # type: ignore[name-defined]
            try:
                val = float(m.group(1))
            except ValueError:
                return m.group(0)
            if val > _eos_ceiling:
                return m.group(0).replace(
                    m.group(1),
                    f"{_safe_wall_t:.1f}",
                )
            return m.group(0)

        _t_fixed = _re_T.sub(
            r'value\s+uniform\s+([\d.eE+\-]+)\s*;',
            _clamp_fixedvalue_T,
            _t_content,
        )
        if _t_fixed != _t_content:
            fixed_files["0/T"] = _t_fixed
            issues.append(ValidationIssue(
                "warning", "0/T",
                f"Auto-clamped fixedValue temperature(s) in 0/T from above EOS ceiling "
                f"({_eos_ceiling:.1f} K, where icoPolynomial ρ→0) to {_safe_wall_t:.1f} K. "
                "limitTemperature in fvOptions does NOT protect boundary face values — "
                "a fixedValue above the EOS ceiling gives ρ < 0 at wall faces → SIGFPE "
                "in compressibleTurbulenceModels on first turbulence evaluation.",
                fix=f"Clamped fixedValue T > {_eos_ceiling:.1f} K → {_safe_wall_t:.1f} K in 0/T"
            ))
            logger.warning(
                f"[VALIDATE] 0/T fixedValue temperatures clamped to {_safe_wall_t:.1f} K "
                f"(icoPolynomial EOS ceiling={_eos_ceiling:.1f} K). "
                "Wall T above EOS ceiling → negative ρ at boundary faces → SIGFPE."
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

    # ── Check 3d2: Strip deprecated nMoles from all thermophysical files ───────
    # OpenFOAM moved to a mass basis — nMoles is no longer read or needed.
    # The LLM frequently still generates it; strip it silently to keep files clean.
    _nMoles_re = re.compile(r'^\s*nMoles\s+\S+\s*;\s*\n?', re.MULTILINE)
    for _tf in list(fixed_files):
        if not _tf.startswith("constant/thermophysicalProperties"):
            continue
        _tc = fixed_files[_tf]
        _tc_fixed = _nMoles_re.sub('', _tc)
        if _tc_fixed != _tc:
            fixed_files[_tf] = _tc_fixed
            logger.debug(f"[VALIDATE] Stripped deprecated nMoles from {_tf}")

    # NOTE: Check 3e (alphatWallFunction → compressible::alphatWallFunction)
    # was deleted in Phase 4.  ``0/alphat`` is now rendered from scratch by
    # ``SolverPlugin._build_alphat()`` which emits the namespace-qualified
    # name from the start — the LLM never produces this file.

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

    # ── Geometric 2D detection ──
    # Detect empty patches by geometry rather than name:
    #   1. If bounding box has one dimension < 5% of the max → mesh is 2D
    #   2. Patches with n_faces ≥ 80% of nCells are the flat-face empty patches
    # This handles meshes where the user named patches in any language.
    _geom_detected_empty_patches: set[str] = set()
    _check_mesh = (mesh_info.get("check_mesh") or mesh_info.get("checkMesh")) if isinstance(mesh_info, dict) else None
    if _check_mesh and isinstance(_check_mesh, dict) and mesh_patches_list:
        _bb = _check_mesh.get("bounding_box") or _check_mesh.get("boundingBox")
        _n_cells = _check_mesh.get("cells", 0)
        if _bb and isinstance(_bb, dict) and _n_cells > 0:
            _bb_min = _bb.get("min", [0, 0, 0])
            _bb_max = _bb.get("max", [0, 0, 0])
            if len(_bb_min) == 3 and len(_bb_max) == 3:
                _dims = [abs(_bb_max[i] - _bb_min[i]) for i in range(3)]
                _max_dim = max(_dims) if max(_dims) > 0 else 1.0
                _is_2d = any(d / _max_dim < 0.05 for d in _dims)
                if _is_2d:
                    for mp in mesh_patches_list:
                        _mp_name = mp.get("name", "") if isinstance(mp, dict) else getattr(mp, "name", "")
                        _mp_faces = (mp.get("n_faces", 0) if isinstance(mp, dict)
                                     else getattr(mp, "n_faces", 0))
                        if _mp_faces >= _n_cells * 0.8:
                            _geom_detected_empty_patches.add(_mp_name)
                    if _geom_detected_empty_patches:
                        logger.info(
                            "[VALIDATE] Geometric 2D detection: empty patches = %s "
                            "(bbox dims=%s, nCells=%d)",
                            _geom_detected_empty_patches, _dims, _n_cells,
                        )

    # Well-known empty patch names (heuristic fallback when geometry isn't available)
    _KNOWN_EMPTY_NAMES = {"frontandback", "frontback", "defaultfaces", "front", "back", "side", "empty"}

    # ── Check 3c: Auto-add empty patches to 0/* files for 2D meshes ──
    # Detects empty patches by geometry (face count in 2D mesh) AND by well-known names.
    # This ensures patches named in any language are still handled correctly.
    _empty_patch_names: set[str] = set(_geom_detected_empty_patches)
    if mesh_patches_list:
        for mp in mesh_patches_list:
            mp_name = mp.get("name", "") if isinstance(mp, dict) else getattr(mp, "name", "")
            name_lower = mp_name.lower().replace("_", "")
            if name_lower in _KNOWN_EMPTY_NAMES:
                _empty_patch_names.add(mp_name)

    # Also check boundary_conditions keys
    bcs = config.get("boundary_conditions", {})
    for bc_name in bcs:
        name_lower = bc_name.lower().replace("_", "")
        if name_lower in ("frontandback", "frontback"):
            _empty_patch_names.add(bc_name)
    
    for _ep_name in _empty_patch_names:
        _ep_block = f"\n    {_ep_name}\n    {{\n        type            empty;\n    }}\n"
        for file_path, content in list(fixed_files.items()):
            if not file_path.startswith("0/"):
                continue
            if "boundaryField" not in content:
                continue

            if _ep_name in content:
                # Block already present — ensure its type is `empty`
                _ep_wrong_type = re.search(
                    rf'({re.escape(_ep_name)}\s*\{{[^}}]*?type\s+)(?!empty\b)(\w+)(\s*;[^}}]*?\}})',
                    content,
                    re.DOTALL,
                )
                if _ep_wrong_type:
                    wrong_type = _ep_wrong_type.group(2)
                    new_content = re.sub(
                        rf'({re.escape(_ep_name)}\s*\{{[^}}]*?type\s+)(?!empty\b)(\w+)(\s*;)',
                        r'\1empty\3',
                        content,
                        flags=re.DOTALL,
                    )
                    if new_content != content:
                        fixed_files[file_path] = new_content
                        content = new_content
                        issues.append(ValidationIssue(
                            "warning", file_path,
                            f"Auto-fixed: '{_ep_name}' type '{wrong_type}' → 'empty' "
                            f"(2D constraint patch).",
                            fix=f"{_ep_name} type → empty"
                        ))
                continue  # Already present (possibly just fixed type)

            # Missing entirely — insert before the closing } of boundaryField
            last_brace = content.rfind("}")
            if last_brace > 0:
                fixed_files[file_path] = content[:last_brace] + _ep_block + content[last_brace:]
                issues.append(ValidationIssue(
                    "warning", file_path,
                    f"Added missing '{_ep_name}' (empty) patch for 2D mesh.",
                    fix=f"Added {_ep_name} {{ type empty; }}"
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
    # EXCEPTION: Patches detected as empty (by geometry or by well-known names) are forced
    # to type 'empty'.  The sim server's fix_boundary_types() applies the same correction
    # after mesh conversion, so the agent must match to avoid a validator/sim-server mismatch.
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

            # Force empty patches detected by geometry (2D face-count heuristic)
            if mp_name in _geom_detected_empty_patches:
                mp_type = "empty"
            else:
                # Fallback: force well-known constraint patch names
                name_lower = mp_name.lower().replace("_", "")
                if name_lower in _KNOWN_EMPTY_NAMES:
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
    # Pre-strip fvOptions when the wall does NOT create actual heat exchange.
    # fvOptions limitTemperature is needed ONLY when the wall is genuinely HOTTER than
    # the inlet fluid — i.e. wall_T > inlet_T + 10 K.
    #
    # Two cases that do NOT need fvOptions (and should be stripped):
    #   1. Adiabatic walls: no fixedValue wall temperature at all.
    #   2. Isothermal walls: wall_T ≈ inlet_T — the wall keeps fluid at operating
    #      temperature with no net heat flux (e.g. LN2 pipe at 77K wall+inlet).
    #      Temperature will never drift above inlet T → EOS ceiling is safe.
    #
    # This strip runs AFTER Check 3c to catch both LLM-generated and auto-injected
    # fvOptions, including files accumulated across self-healing iterations.
    _strip_bcs = config.get("boundary_conditions") or {}
    _strip_inlet_t: float | None = None
    _strip_wall_temps: list[float] = []
    for _strip_pname, _strip_pbc in _strip_bcs.items():
        if not isinstance(_strip_pbc, dict):
            continue
        _strip_pt = _strip_pbc.get("patch_type", "")
        _strip_t_entry = _strip_pbc.get("temperature") or _strip_pbc.get("T")
        if isinstance(_strip_t_entry, dict):
            _strip_tv = _strip_t_entry.get("value") if _strip_t_entry.get("type") == "fixedValue" else None
        elif isinstance(_strip_t_entry, (int, float)):
            _strip_tv = float(_strip_t_entry)
        else:
            _strip_tv = None
        if _strip_tv is None:
            continue
        try:
            _strip_tv_f = float(_strip_tv)
        except (TypeError, ValueError):
            continue
        if _strip_pt in ("inlet", "pressure_inlet", "mass_flow_inlet") or _strip_pname == "inlet":
            _strip_inlet_t = _strip_tv_f
        elif _strip_pname not in ("outlet",):
            _strip_wall_temps.append(_strip_tv_f)

    # Also try top-level config keys for inlet_T (older format)
    if _strip_inlet_t is None:
        for _tk in ("inlet_temperature",):
            if config.get(_tk) is not None:
                try:
                    _strip_inlet_t = float(config[_tk])
                except (TypeError, ValueError):
                    pass

    _strip_wall_heats_fluid = False
    if _strip_wall_temps:
        _strip_max_wall = max(_strip_wall_temps)
        if _strip_inlet_t is not None:
            _strip_wall_heats_fluid = _strip_max_wall > _strip_inlet_t + 10.0
        else:
            # No inlet T — conservative: only strip if wall looks cryogenic or ambient
            _strip_wall_heats_fluid = _strip_max_wall > 200.0

    if not _strip_wall_heats_fluid and "system/fvOptions" in fixed_files:
        del fixed_files["system/fvOptions"]
        _strip_reason = (
            f"wall_T={max(_strip_wall_temps):.0f} K ≤ inlet_T={_strip_inlet_t:.0f} K + 10 K (isothermal — no heat flux)"
            if _strip_wall_temps and _strip_inlet_t is not None
            else "no wall fixedValue temperature BC (adiabatic)"
        )
        issues.append(ValidationIssue(
            "warning", "system/fvOptions",
            f"Removed system/fvOptions: {_strip_reason}. "
            "limitTemperature only needed when wall heats fluid above EOS ceiling.",
        ))
        logger.info(f"[VALIDATE] Stripped system/fvOptions: {_strip_reason}")

    required_files = ["system/controlDict", "system/fvSchemes", "system/fvSolution", "0/U"]

    # Pressure field
    if solver in P_RGH_SOLVERS:
        required_files.append("0/p_rgh")
    else:
        required_files.append("0/p")

    # Transport / thermo
    # Boussinesq variants are an exception inside THERMO_SOLVERS: they solve
    # the energy equation BUT use constant/transportProperties (β, T_ref, ν)
    # instead of constant/thermophysicalProperties — density is constant ρ₀
    # in the Boussinesq approximation, no full EOS is needed.
    _is_boussinesq = "Boussinesq" in solver
    if _is_boussinesq:
        required_files.append("constant/transportProperties")
    elif solver in THERMO_SOLVERS:
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

    # fvOptions — only when wall is hotter than inlet (actual heat exchange).
    # Reuse the same wall-heats-fluid determination from the pre-strip above.
    # _strip_wall_heats_fluid was computed from the same config just above.
    _RHO_SINGLE_PHASE_FVOPTIONS_V = {
        "rhoPimpleFoam", "rhoSimpleFoam",
        "buoyantSimpleFoam", "buoyantPimpleFoam",
    }
    if solver in _RHO_SINGLE_PHASE_FVOPTIONS_V and _strip_wall_heats_fluid:
        required_files.append("system/fvOptions")

    for rf in required_files:
        if rf not in fixed_files:
            issues.append(ValidationIssue(
                "error", rf,
                f"Required file '{rf}' is missing from generated output.",
            ))
    
    # ── Check 6: fvSolution solver algorithm ──
    # Suffix-driven: *SimpleFoam → SIMPLE block, *PimpleFoam → PIMPLE,
    # chtMultiRegionFoam (transient) → PIMPLE.  Covers every variant we
    # ship (rho*, buoyant*, buoyantBoussinesq*, chtMultiRegion*) without
    # listing them by name.
    fv_solution = fixed_files.get("system/fvSolution", "")
    if fv_solution:
        needs_simple = solver.endswith("SimpleFoam")
        needs_pimple = (
            solver.endswith("PimpleFoam")
            or solver == "chtMultiRegionFoam"
        )
        if needs_simple and "SIMPLE" not in fv_solution:
            issues.append(ValidationIssue(
                "warning", "system/fvSolution",
                f"{solver} requires a SIMPLE block in fvSolution.",
            ))
        elif needs_pimple and "PIMPLE" not in fv_solution:
            issues.append(ValidationIssue(
                "warning", "system/fvSolution",
                f"{solver} requires a PIMPLE block in fvSolution.",
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

    # ── Check 7b: compressible energy solvers require div(phi,K) and div(phid,p) ──
    # With `default none`, missing these terms causes a fatal "cannot find" error.
    # With `default bounded Gauss upwind` the default covers them, but we inject them
    # explicitly anyway so the intent is clear and crash-proof regardless of default choice.
    _COMPRESSIBLE_ENERGY_SOLVERS = {
        "rhoSimpleFoam", "rhoPimpleFoam",
        "buoyantSimpleFoam", "buoyantPimpleFoam",
    }
    # Subset that requires div(phid,p): rho* solvers only.
    # buoyant solvers use p_rgh as primary — div(phid,p) is NOT present/needed.
    _PHID_P_REQUIRED_SOLVERS = {
        "rhoSimpleFoam", "rhoPimpleFoam",
    }
    fv_schemes = fixed_files.get("system/fvSchemes", "")
    if fv_schemes and solver in _COMPRESSIBLE_ENERGY_SOLVERS and "divSchemes" in fv_schemes:
        missing_terms: list[tuple[str, str]] = []
        if "div(phi,K)" not in fv_schemes:
            missing_terms.append(("div(phi,K)", "bounded Gauss upwind"))
        if solver in _PHID_P_REQUIRED_SOLVERS and "div(phid,p)" not in fv_schemes:
            missing_terms.append(("div(phid,p)", "Gauss limitedLinear 1"))

        if missing_terms:
            term_names = [t[0] for t in missing_terms]
            issues.append(ValidationIssue(
                "warning", "system/fvSchemes",
                f"Compressible energy solver '{solver}' missing required div terms: {term_names}. Adding.",
                fix=f"Injected {term_names} into divSchemes"
            ))
            # Inject missing terms right after the divSchemes opening brace
            inject = "".join(
                f"\n    {term:<44} {scheme};" for term, scheme in missing_terms
            )
            fv_schemes = re.sub(
                r"(divSchemes\s*\{)",
                r"\1" + inject,
                fv_schemes,
                count=1,
                flags=re.DOTALL,
            )
            fixed_files["system/fvSchemes"] = fv_schemes

    # NOTE: Check 7c (GAMG coarsest hardening), Check 7d (div(phi,h) upwind
    # for large ΔT) and Check 7e (rhoPimpleFoam GAMG→PBiCGStab) were deleted
    # in Phase 2.  Their physics decisions now live in case_spec.resolvers
    # (resolve_pressure_solver_strategy, resolve_div_phi_h_scheme); the
    # renderer produces correct fvSolution / fvSchemes from the start.

    # ── Check 8: flowRateInletVelocity MUST have massFlowRate/volumetricFlowRate ──
    # OpenFOAM 2406 fatal: "Please supply either 'volumetricFlowRate' or 'massFlowRate'"
    # The LLM sometimes generates only a `value` entry and forgets the required key.
    u_file = fixed_files.get("0/U", "")
    if u_file and "flowRateInletVelocity" in u_file:
        # Find every patch block that uses flowRateInletVelocity
        # Pattern: captures the whole patch block (up to the matching closing brace)
        import re as _re

        # Detect compressible solver — rho field is available at runtime
        _is_compressible_solver = solver in _COMPRESSIBLE_ENERGY_SOLVERS

        # Pull rho and fluid density from config if available
        _fluid_rho: float | None = None
        _fluid_rho = (
            config.get("fluid_rho")
            or config.get("fluid", {}).get("rho")
            or config.get("fluid", {}).get("density")
        )
        if _fluid_rho:
            _fluid_rho = float(_fluid_rho)

        # Recover the actual flow rate value from validated_config BCs
        _bc_flow_rate: float | None = None
        _flow_rate_key = "massFlowRate"
        _bcs_cfg = config.get("boundary_conditions", {}) or {}
        for _pname, _pbc in _bcs_cfg.items():
            _u_bc = _pbc.get("velocity") or _pbc.get("U") or {}
            if isinstance(_u_bc, dict) and _u_bc.get("type") == "flowRateInletVelocity":
                # Source 1: explicit entries dict
                _entries = _u_bc.get("entries") or {}
                for _fk in ("massFlowRate", "volumetricFlowRate"):
                    _rv = _entries.get(_fk)
                    if _rv is not None and float(_rv) != 0:
                        _bc_flow_rate = float(_rv)
                        _flow_rate_key = _fk
                        break
                # Source 2: top-level key
                if _bc_flow_rate is None:
                    for _fk in ("massFlowRate", "volumetricFlowRate"):
                        _rv = _u_bc.get(_fk)
                        if _rv is not None and float(_rv) != 0:
                            _bc_flow_rate = float(_rv)
                            _flow_rate_key = _fk
                            break
                # Source 3: scalar value
                if _bc_flow_rate is None:
                    _raw = _u_bc.get("value")
                    if isinstance(_raw, (int, float)) and _raw != 0:
                        _bc_flow_rate = float(_raw)
                    # Source 4: misinterpreted as vector [flow_rate, 0, 0]
                    elif isinstance(_raw, (list, tuple)) and len(_raw) >= 1:
                        _first = _raw[0]
                        if isinstance(_first, (int, float)) and _first != 0:
                            _bc_flow_rate = float(_first)
                break

        # Find all patch blocks that contain flowRateInletVelocity but lack
        # Recover rho/rhoInlet from config only if the user explicitly specified them
        _user_rho_field: str | None = None
        _user_rho_inlet: float | None = None
        for _pname, _pbc in _bcs_cfg.items():
            _u_bc = _pbc.get("velocity") or _pbc.get("U") or {}
            if isinstance(_u_bc, dict) and _u_bc.get("type") == "flowRateInletVelocity":
                _user_rho_field = _u_bc.get("rho") or None
                _ri = _u_bc.get("rhoInlet")
                if _ri is not None:
                    try:
                        _user_rho_inlet = float(_ri)
                    except Exception:
                        pass
                break

        # massFlowRate AND volumetricFlowRate
        def _fix_flow_rate_patch(m: "_re.Match[str]") -> str:
            block = m.group(0)
            if "massFlowRate" in block or "volumetricFlowRate" in block or "meanVelocity" in block:
                return block  # already has a flow rate key — don't touch

            _mfr = _bc_flow_rate if _bc_flow_rate is not None else 0.0

            # Only inject rho/rhoInlet if the user's config had them
            _rho_line = ""
            if _user_rho_field is not None:
                _rho_line = f"        rho             {_user_rho_field};\n"
            if _user_rho_inlet is not None:
                _rho_line += f"        rhoInlet        {_user_rho_inlet};\n"

            _mfr_line = f"        {_flow_rate_key}    {_mfr};\n"

            # Replace the value entry with a safe vector placeholder
            block = _re.sub(
                r'value\s+uniform\s+\([^)]*\)\s*;',
                "value           uniform (0 0 0);",
                block,
            )
            block = _re.sub(
                r'value\s+uniform\s+[\d.eE+\-]+\s*;',
                "value           uniform (0 0 0);",
                block,
            )

            # Inject massFlowRate + rho before the closing brace of the patch block
            block = _re.sub(
                r'(\s*\}\s*)$',
                "\n" + _mfr_line + _rho_line + r'\1',
                block,
                count=1,
            )
            return block

        # Match each named patch block that contains flowRateInletVelocity
        _fixed_u = _re.sub(
            r'(\w+\s*\{[^{}]*flowRateInletVelocity[^{}]*\})',
            _fix_flow_rate_patch,
            u_file,
            flags=_re.DOTALL,
        )
        if _fixed_u != u_file:
            issues.append(ValidationIssue(
                "error", "0/U",
                "flowRateInletVelocity inlet is missing 'massFlowRate'/'volumetricFlowRate'. "
                "Auto-injected required entries. "
                "(OpenFOAM fatal: 'Please supply either volumetricFlowRate or massFlowRate')",
                fix="Injected massFlowRate + rho/rhoInlet into flowRateInletVelocity patch block",
            ))
            fixed_files["0/U"] = _fixed_u
            logger.info("[VALIDATE] Auto-fixed: injected massFlowRate into flowRateInletVelocity patch(es) in 0/U")

    # ── Check 8b: massFlowRate / volumetricFlowRate must NOT be zero ──
    # Zero flow rate = no flow enters the domain = singular momentum matrix = SIGFPE crash.
    u_file = fixed_files.get("0/U", "")
    if u_file and "flowRateInletVelocity" in u_file:
        import re as _re
        _zero_flow = _re.search(
            r'(massFlowRate|volumetricFlowRate)\s+(0+(\.0*)?)\s*;',
            u_file,
        )
        if _zero_flow:
            _key_found = _zero_flow.group(1)
            # Try to recover the actual flow rate from validated_config
            _bcs_cfg = config.get("boundary_conditions", {}) or {}
            _recovered_val: float | None = None
            for _pname, _pbc in _bcs_cfg.items():
                _u_bc = _pbc.get("velocity") or _pbc.get("U") or {}
                if isinstance(_u_bc, dict):
                    # Source 1: explicit entries dict (best source)
                    _entries = _u_bc.get("entries") or {}
                    for _fk in ("massFlowRate", "volumetricFlowRate"):
                        _rv = _entries.get(_fk)
                        if _rv is not None and float(_rv) != 0:
                            _recovered_val = float(_rv)
                            _key_found = _fk
                            break
                    # Source 2: top-level massFlowRate/volumetricFlowRate key
                    if _recovered_val is None:
                        for _fk in ("massFlowRate", "volumetricFlowRate"):
                            _rv = _u_bc.get(_fk)
                            if _rv is not None and float(_rv) != 0:
                                _recovered_val = float(_rv)
                                _key_found = _fk
                                break
                    # Source 3: scalar value (flow rate stored as generic value)
                    if _recovered_val is None:
                        _raw = _u_bc.get("value")
                        if isinstance(_raw, (int, float)) and _raw != 0:
                            _recovered_val = float(_raw)
                        # Source 4: value was misinterpreted as velocity vector [flow_rate, 0, 0]
                        elif isinstance(_raw, (list, tuple)) and len(_raw) >= 1:
                            _first = _raw[0]
                            if isinstance(_first, (int, float)) and _first != 0:
                                _recovered_val = float(_first)
                    break
            if _recovered_val is not None:
                u_file = u_file.replace(
                    _zero_flow.group(0),
                    f"{_key_found}    {_recovered_val};",
                )
                fixed_files["0/U"] = u_file
                issues.append(ValidationIssue(
                    "error", "0/U",
                    f"flowRateInletVelocity had {_key_found} = 0 (guaranteed divergence). "
                    f"Recovered actual value {_recovered_val} from validated_config.",
                    fix=f"Set {_key_found} to {_recovered_val}",
                ))
                logger.info(f"[VALIDATE] Auto-fixed: {_key_found} 0 → {_recovered_val} in 0/U")
            else:
                issues.append(ValidationIssue(
                    "error", "0/U",
                    f"flowRateInletVelocity has {_key_found} = 0 — this will cause "
                    "divergence (SIGFPE). Could not recover the actual value from config.",
                ))
                logger.error(f"[VALIDATE] CRITICAL: {_key_found} is 0 in 0/U and no fallback found")

    # ── Check IC/BC physical consistency ─────────────────────────────────────
    # Validate that the initial conditions and boundary conditions are physically
    # coherent BEFORE sending to the sim server.  Catches common LLM errors that
    # cause immediate divergence but no readable error message from OpenFOAM.
    import re as _ic_re

    # Check 1: 0/T internalField within BC temperature range
    # For compressible energy solvers: if internalField T exceeds the icoPolynomial EOS
    # ceiling (ρ→0), the simulation will produce negative density on step 1 → SIGFPE.
    # Common cause: LLM defaults internalField to 300K (room temperature) instead of the
    # inlet temperature (e.g. 77K for LN2), giving ρ = 1167.9 − 4.7×300 = −242 kg/m³.
    # Even with fvOptions limitTemperature, the internalField is written BEFORE the fvOptions
    # limiter runs — on iteration 0 the density field is already negative → crash.
    # Auto-correct internalField to inlet_temperature when it would violate the EOS ceiling.
    _t_file = fixed_files.get("0/T", "")
    if _t_file:
        _int_m = _ic_re.search(r'internalField\s+uniform\s+([\d.eE+\-]+)', _t_file)
        if _int_m:
            _T_int = float(_int_m.group(1))
            # Collect all fixedValue T BCs (from the 0/T file itself)
            _bc_fixed_temps = [float(m) for m in _ic_re.findall(
                r'type\s+fixedValue.*?value\s+uniform\s+([\d.eE+\-]+)', _t_file, _ic_re.DOTALL
            )]

            # Extract inlet temperature from config — needed for auto-correction target
            _cfg_bcs = config.get("boundary_conditions") or {}
            _inlet_T_cfg: float | None = None
            for _pname, _pbc_d in _cfg_bcs.items():
                if not isinstance(_pbc_d, dict):
                    continue
                _pt = _pbc_d.get("patch_type", "")
                if _pt not in ("inlet", "pressure_inlet", "mass_flow_inlet") and _pname not in ("inlet",):
                    continue
                _t_entry = _pbc_d.get("temperature") or _pbc_d.get("T")
                if isinstance(_t_entry, dict):
                    _tv = _t_entry.get("value") or _t_entry.get("uniform")
                elif isinstance(_t_entry, (int, float)):
                    _tv = float(_t_entry)
                else:
                    _tv = None
                if _tv is not None:
                    try:
                        _inlet_T_cfg = float(_tv)
                        break
                    except (TypeError, ValueError):
                        pass
            # Also accept top-level keys in config (older format)
            if _inlet_T_cfg is None:
                for _k in ("inlet_temperature", "temperature"):
                    _v = config.get("fluid", {}).get("temperature") if _k == "temperature" else config.get(_k)
                    if _v is not None:
                        try:
                            _inlet_T_cfg = float(_v)
                            break
                        except (TypeError, ValueError):
                            pass

            # ── Auto-correct: internalField above EOS ceiling → crash on step 0 ──
            # When ρ(T_internal) < 0 (T_internal > EOS ceiling), the pressure equation
            # is ill-posed from the first iteration — fvOptions never gets a chance to
            # clamp because the density field is already negative when rho.correctBoundaryConditions()
            # runs.  Correct internalField to inlet_T (safest physical value).
            if (
                _eos_ceiling is not None
                and solver in _COMPRESSIBLE_RHO_SOLVERS
                and _T_int > _eos_ceiling * 0.95
            ):
                # Pick best correction target: prefer inlet_T from config, else smallest BC temp
                _T_corr = _inlet_T_cfg
                if _T_corr is None and _bc_fixed_temps:
                    _T_corr = min(_bc_fixed_temps)
                if _T_corr is None:
                    _T_corr = _t_floor  # last resort: use the fvOptions floor
                fixed_files["0/T"] = _ic_re.sub(
                    r'(internalField\s+uniform\s+)[\d.eE+\-]+',
                    rf'\g<1>{_T_corr:.4g}',
                    _t_file,
                )
                _t_file = fixed_files["0/T"]  # update for downstream checks
                issues.append(ValidationIssue(
                    "warning", "0/T",
                    f"internalField T={_T_int} K exceeds icoPolynomial EOS ceiling "
                    f"({_eos_ceiling:.1f} K where ρ→0). "
                    f"At {_T_int} K, ρ would be negative → SIGFPE on iteration 0. "
                    f"Auto-corrected internalField → {_T_corr:.4g} K (inlet temperature).",
                    fix=f"internalField uniform {_T_corr:.4g};"
                ))
                logger.warning(
                    f"[VALIDATE] 0/T internalField {_T_int} K > EOS ceiling {_eos_ceiling:.1f} K "
                    f"(ρ→0). Corrected to inlet_T={_T_corr:.4g} K to prevent negative density on step 0."
                )

            if _bc_fixed_temps:
                _T_bc_min = min(_bc_fixed_temps)
                _T_bc_max = max(_bc_fixed_temps)
                # Re-read internalField after potential auto-correction above
                _int_m2 = _ic_re.search(r'internalField\s+uniform\s+([\d.eE+\-]+)', _t_file)
                _T_int_current = float(_int_m2.group(1)) if _int_m2 else _T_int
                if _T_int_current < 0.5 * _T_bc_min or _T_int_current > 2.0 * _T_bc_max:
                    issues.append(ValidationIssue(
                        "warning", "0/T",
                        f"internalField T={_T_int_current} K is far outside BC range "
                        f"[{_T_bc_min}, {_T_bc_max}] K. This causes large initial "
                        "energy residuals and may diverge immediately.",
                    ))
            # Check T > 0 (absolute zero crash)
            if _T_int <= 0:
                issues.append(ValidationIssue(
                    "error", "0/T",
                    f"internalField T={_T_int} K — non-positive temperature will crash thermo.",
                ))

    # Check 2: 0/k and 0/omega must be positive
    for _turb_field in ("k", "omega", "epsilon"):
        _tf_content = fixed_files.get(f"0/{_turb_field}", "")
        if not _tf_content:
            continue
        _tf_m = _ic_re.search(r'internalField\s+uniform\s+([\d.eE+\-]+)', _tf_content)
        if _tf_m and float(_tf_m.group(1)) <= 0:
            issues.append(ValidationIssue(
                "error", f"0/{_turb_field}",
                f"internalField {_turb_field}={_tf_m.group(1)} — turbulence fields must be positive.",
            ))

    # ── Check 2b: k must be physically plausible for the flow ─────────────────
    # k > 5e-3 is wrong for low-speed subsonic pipe flow (cryogenic or otherwise).
    # At U=10 m/s with I=5%: k = 1.5*(0.05*10)^2 = 0.0375  (extreme upper bound)
    # At U=1 m/s  with I=5%: k ≈ 3.75e-3
    # At U=0.2 m/s (cryogenic):  k ≈ 1.5*(0.05*0.2)^2 ≈ 1.5e-4
    # LLM often defaults to k=0.1 or k=0.01 — both wrong for the velocities here.
    _k_content = fixed_files.get("0/k", "")
    if _k_content:
        _k_int_m = _ic_re.search(r'internalField\s+uniform\s+([\d.eE+\-]+)', _k_content)
        if _k_int_m:
            _k_val = float(_k_int_m.group(1))
            _K_WARN_THRESHOLD = 5e-3  # k=0.005 corresponds to U≈0.82 m/s; anything above is suspicious
            if _k_val > _K_WARN_THRESHOLD:
                # Try to find a pre-computed reference value from the turbulence config
                _turb_cfg_ref = (config.get("turbulence") or {})
                _k_ref = None
                for _kk in ("k",):
                    _vv = _turb_cfg_ref.get(_kk)
                    if _vv and float(_vv) > 0:
                        _k_ref = float(_vv)
                        break
                _k_corrected = _k_ref if (_k_ref and _k_ref < _K_WARN_THRESHOLD) else 2e-4
                # Auto-correct internalField and all uniform value lines in the file
                _k_fixed = _ic_re.sub(
                    r'internalField\s+uniform\s+[\d.eE+\-]+',
                    f'internalField   uniform {_k_corrected}',
                    _k_content, count=1
                )
                # Replace all bare `value uniform <large_k>` occurrences
                _k_fixed = _ic_re.sub(
                    r'(value\s+uniform\s+)[\d.eE+\-]+',
                    lambda _m: _m.group(1) + str(_k_corrected),
                    _k_fixed
                )
                fixed_files["0/k"] = _k_fixed
                issues.append(ValidationIssue(
                    "warning", "0/k",
                    f"internalField k={_k_val} is unreasonably large for subsonic pipe flow "
                    f"(threshold {_K_WARN_THRESHOLD}). Auto-corrected to {_k_corrected}. "
                    "Formula: k = 1.5*(I*U)^2, I=0.05. Check mass flow rate and density.",
                    fix=f"Auto-corrected internalField and fixedValue BCs from {_k_val} to {_k_corrected}",
                ))

    # ── Check 2c: turbulence field values must match pre-computed reference ──
    # The precheck/frontend pre-computes k, omega, epsilon, nut from the user's
    # velocity, turbulence intensity, and hydraulic diameter.  The LLM is
    # instructed to use these values, but sometimes ignores them and writes
    # wrong values for internalField and inlet fixedValue (e.g. k=0.0002
    # instead of the pre-computed 3.375 for U=30 m/s at 5% TI).
    # This check enforces the pre-computed values whenever they exist.
    _turb_ref = config.get("turbulence") or {}
    for _tf_name, _tf_key in [("k", "k"), ("omega", "omega"), ("epsilon", "epsilon")]:
        _ref_val = None
        _rv = _turb_ref.get(_tf_key)
        if _rv is not None:
            try:
                _ref_val = float(_rv)
            except (TypeError, ValueError):
                pass
        if _ref_val is None or _ref_val <= 0:
            continue

        _tf_content = fixed_files.get(f"0/{_tf_name}", "")
        if not _tf_content:
            continue

        _tf_int_m = _ic_re.search(
            r'internalField\s+uniform\s+([\d.eE+\-]+)', _tf_content
        )
        if not _tf_int_m:
            continue

        _tf_gen_val = float(_tf_int_m.group(1))

        # Allow 50% tolerance for LLM rounding; outside that, auto-correct
        if _tf_gen_val <= 0 or _tf_gen_val < 0.5 * _ref_val or _tf_gen_val > 2.0 * _ref_val:
            # Replace internalField
            _tf_fixed = _ic_re.sub(
                r'internalField\s+uniform\s+[\d.eE+\-]+',
                f'internalField   uniform {_ref_val:.6g}',
                _tf_content, count=1,
            )
            # Replace all bare `value uniform <wrong>` to ensure consistency
            # across inlet, outlet.inletValue, etc.
            _tf_fixed = _ic_re.sub(
                r'(value\s+uniform\s+)[\d.eE+\-]+',
                lambda _m: _m.group(1) + f'{_ref_val:.6g}',
                _tf_fixed,
            )
            fixed_files[f"0/{_tf_name}"] = _tf_fixed
            issues.append(ValidationIssue(
                "warning", f"0/{_tf_name}",
                f"LLM generated {_tf_name}={_tf_gen_val:.6g} but pre-computed "
                f"reference is {_ref_val:.6g} (from turbulence config). "
                f"Auto-corrected internalField and all uniform value entries.",
                fix=f"Auto-corrected {_tf_name} from {_tf_gen_val:.6g} to {_ref_val:.6g}",
            ))
            logger.warning(
                f"[VALIDATE] Check 2c: 0/{_tf_name} internalField {_tf_gen_val:.6g} "
                f"→ {_ref_val:.6g} (pre-computed reference). "
                f"Ratio: {_tf_gen_val/_ref_val:.2f}x."
            )

    # ── Check 2d: OF 2406 requires `value` on certain inlet BCs ──────────────
    # turbulentIntensityKineticEnergyInlet, turbulentMixingLengthFrequencyInlet,
    # turbulentMixingLengthDissipationRateInlet (and a few siblings) MUST carry
    # an explicit `value uniform <X>;` entry under OpenFOAM 2406's stricter
    # reader — without it the solver dies on iteration 0 with
    #   "Required entry 'value' : missing for patch <inlet> in dictionary
    #   '0/<field>/boundaryField/<inlet>'"
    # We patch each patch block in 0/k, 0/omega, 0/epsilon: if it uses one of
    # these inlet types AND has no `value` line, append `value uniform $internalField;`
    # immediately before the closing brace.
    _VALUE_REQUIRING_TYPES = (
        "turbulentIntensityKineticEnergyInlet",
        "turbulentMixingLengthFrequencyInlet",
        "turbulentMixingLengthDissipationRateInlet",
    )
    # Block matcher: a patch entry is `<name> { ... }`.  We allow nested
    # braces (none expected at this level) by matching the shortest balanced
    # block via a non-greedy outer + greedy inner trick.
    _PATCH_BLOCK_RE = _ic_re.compile(
        r'(\b\w+\s*\{)([^{}]*)(\})',
        _ic_re.DOTALL,
    )
    for _tf_name in ("k", "omega", "epsilon"):
        _tf_path = f"0/{_tf_name}"
        _tf_content = fixed_files.get(_tf_path, "")
        if not _tf_content:
            continue
        _patched_any = False
        _patched_blocks: list[str] = []

        def _maybe_inject_value(m):
            nonlocal _patched_any
            head, body, tail = m.group(1), m.group(2), m.group(3)
            # Only consider blocks that DECLARE one of the value-requiring types.
            type_m = _ic_re.search(r'\btype\s+([A-Za-z]+)\s*;', body)
            if not type_m or type_m.group(1) not in _VALUE_REQUIRING_TYPES:
                return m.group(0)
            # Already has a `value` line — nothing to do.
            if _ic_re.search(r'\bvalue\b\s', body):
                return m.group(0)
            # Inject `value uniform $internalField;` right before the closing brace,
            # preserving indentation from the line before the brace if we can find it.
            indent_m = _ic_re.search(r'(?m)^(\s*)\S', body.splitlines()[-1] if body.strip() else "")
            indent = indent_m.group(1) if indent_m else "        "
            new_body = body.rstrip() + f"\n{indent}value           uniform $internalField;\n    "
            _patched_any = True
            # Grab patch name from head for diagnostics
            name_m = _ic_re.match(r'\s*(\w+)', head)
            _patched_blocks.append(name_m.group(1) if name_m else "?")
            return head + new_body + tail

        _new_content = _PATCH_BLOCK_RE.sub(_maybe_inject_value, _tf_content)
        if _patched_any:
            fixed_files[_tf_path] = _new_content
            issues.append(ValidationIssue(
                "warning", _tf_path,
                f"OF 2406 requires `value` entry on inlet BC types "
                f"({', '.join(_VALUE_REQUIRING_TYPES)}). Auto-injected "
                f"`value uniform $internalField;` for patch(es): "
                f"{', '.join(_patched_blocks)}.",
                fix=f"Injected missing value entries in 0/{_tf_name} ({len(_patched_blocks)} patch(es))",
            ))
            logger.warning(
                "[VALIDATE] Check 2d: 0/%s — injected missing `value` "
                "entries for value-requiring inlet types on patch(es): %s",
                _tf_name, _patched_blocks,
            )

    # Check 3: for icoPolynomial EOS, warn if any T BC exceeds eos_t_ceiling
    if _eos_ceiling is not None:
        for _field_name, _content in fixed_files.items():
            if not _field_name.startswith("0/"):
                continue
            for _val_str in _ic_re.findall(r'value\s+uniform\s+([\d.eE+\-]+)', _content):
                try:
                    _v = float(_val_str)
                except ValueError:
                    continue
                # Only flag T values (plausible range 1-5000 K)
                if 1.0 < _v < 5000.0 and _v >= _eos_ceiling:
                    issues.append(ValidationIssue(
                        "warning", _field_name,
                        f"Value {_v} K meets or exceeds icoPolynomial EOS ceiling "
                        f"({_eos_ceiling:.1f} K where ρ→0). "
                        "Cells reaching this temperature will have ρ≤0 → SIGFPE. "
                        "The fvOptions limitTemperature max should be below this value.",
                    ))
                    break  # one warning per file is enough

    # Check 4: 0/p internalField must match outlet fixedValue (prevents GAMG SIGFPE at iter 1)
    _p_file = fixed_files.get("0/p", "")
    if _p_file:
        _p_int_m = _ic_re.search(r'internalField\s+uniform\s+([\d.eE+\-]+)', _p_file)
        if _p_int_m:
            _p_int = float(_p_int_m.group(1))
            # Find outlet fixedValue pressure
            _outlet_block_m = _ic_re.search(
                r'outlet\s*\{[^}]*type\s+fixedValue[^}]*value\s+uniform\s+([\d.eE+\-]+)',
                _p_file, _ic_re.DOTALL | _ic_re.IGNORECASE
            )
            if _outlet_block_m:
                _p_outlet = float(_outlet_block_m.group(1))
                if abs(_p_int - _p_outlet) / max(abs(_p_outlet), 1.0) > 0.01:
                    # Auto-fix: set internalField to match outlet
                    fixed_files["0/p"] = _ic_re.sub(
                        r'(internalField\s+uniform\s+)[\d.eE+\-]+',
                        rf'\g<1>{_p_outlet}',
                        _p_file,
                        count=1,
                    )
                    issues.append(ValidationIssue(
                        "warning", "0/p",
                        f"internalField p={_p_int} Pa does not match outlet fixedValue "
                        f"p={_p_outlet} Pa. Mismatch causes GAMG SIGFPE at iteration 1. "
                        f"Auto-fixed: set internalField to {_p_outlet}.",
                        fix=f"internalField uniform {_p_outlet}"
                    ))

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


def _eos_for_liquid(rho: float, inlet_temp: float | None, has_energy: bool) -> str:
    """Choose the appropriate equation of state for a liquid.

    Rules:
      - rhoConst: isothermal liquid with no significant temperature variation.
      - icoPolynomial: liquid with active heat transfer OR cryogenic temperature.
        Density of real liquids (especially cryogenics) is strongly T-dependent —
        rhoConst introduces large errors when T varies significantly.
    """
    is_cryo = inlet_temp is not None and inlet_temp < 200.0
    # Cryogenic liquids (LH2 rho≈71, LHe rho≈125, LN2 rho≈808) always need
    # icoPolynomial — even if rho is low.  The density check must come AFTER
    # the cryo/energy check to avoid misclassifying light cryogenic liquids.
    if is_cryo or has_energy:
        if rho > 30.0:
            return "icoPolynomial"
        # rho <= 30 with energy: likely a gas — perfectGas would be better,
        # but caller asked for liquid EOS, so rhoConst is the safe fallback.
        return "rhoConst"
    if rho <= 30.0:
        # Very low density, no energy, not cryogenic → likely gas
        return "rhoConst"
    return "rhoConst"


def _ico_poly_coeffs(rho: float, inlet_temp: float | None) -> tuple[float, float]:
    """Compute linear icoPolynomial coefficients a0, a1 such that ρ(T) = a0 + a1*T.

    Uses a typical liquid thermal expansion slope for the temperature range:
      T < 8 K   (LHe range):   dρ/dT ≈ -5.0 kg/m³/K
      T < 35 K  (LH2 range):   dρ/dT ≈ -0.7 kg/m³/K
      T < 100 K (LN2/LOX):     dρ/dT ≈ -4.7 kg/m³/K
      otherwise (water/oils):  dρ/dT ≈ -0.5 kg/m³/K
    """
    T_ref = inlet_temp if inlet_temp is not None else 300.0
    if T_ref < 8.0:
        a1 = -5.0   # liquid helium (He-4): ~125 kg/m³ at 4.2 K
    elif T_ref < 35.0:
        a1 = -0.7
    elif T_ref < 100.0:
        a1 = -4.7
    else:
        a1 = -0.5
    a0 = rho - a1 * T_ref
    return a0, a1


def build_required_files_list(solver: str, config: dict[str, Any]) -> list[str]:
    """Return the exact list of files the LLM must generate for this solver/config.

    Delegates to the solver plugin's ``required_files(config)`` when a plugin
    is registered (the plugin is the single source of truth post-refactor).
    Falls back to the legacy inline logic for unknown solvers, so callers can
    still ask for files without a plugin being installed.

    **Multi-region cases never reach the legacy fallback.**  The plugin
    contract pins the LLM-owned manifest to ``["system/controlDict"]`` for
    every CHT solver; every other case file is rendered deterministically
    by :class:`MultiRegionBase`.  A defensive early-return below guarantees
    this even if a future refactor accidentally bypasses the plugin lookup.
    """
    from simd_agent.solvers import get_registry, is_multi_region_solver

    if is_multi_region_solver(solver):
        try:
            _plugin = get_registry().get(solver)
            if _plugin is not None:
                return _plugin.required_files(config)
        except Exception:
            pass
        # Plugin lookup failed but we know it's multi-region — pin to the
        # one file the LLM legitimately owns in any CHT case.
        return ["system/controlDict"]

    try:
        _plugin = get_registry().get(solver)
    except Exception:
        _plugin = None
    if _plugin is not None:
        return _plugin.required_files(config)

    # ── Legacy fallback (kept verbatim for solvers without a plugin) ──────
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

    # fvOptions (limitTemperature) — for energy solvers when wall is HOTTER than inlet
    # (actual heat exchange that could push T above EOS ceiling).
    # Isothermal (wall_T ≈ inlet_T) and adiabatic (no wall T) do not need fvOptions.
    _RHO_SINGLE_PHASE_FVOPTIONS = {
        "rhoPimpleFoam", "rhoSimpleFoam",
        "buoyantSimpleFoam", "buoyantPimpleFoam",
    }
    _brl_inlet_t: float | None = None
    _brl_wall_temps: list[float] = []
    for _brl_pname, _brl_pbc in (config.get("boundary_conditions") or {}).items():
        if not isinstance(_brl_pbc, dict):
            continue
        _brl_pt = _brl_pbc.get("patch_type", "")
        _brl_t = _brl_pbc.get("temperature") or _brl_pbc.get("T")
        if isinstance(_brl_t, dict):
            _brl_tv = _brl_t.get("value") if _brl_t.get("type") == "fixedValue" else None
        elif isinstance(_brl_t, (int, float)):
            _brl_tv = float(_brl_t)
        else:
            _brl_tv = None
        if _brl_tv is None:
            continue
        try:
            _brl_tv_f = float(_brl_tv)
        except (TypeError, ValueError):
            continue
        if _brl_pt in ("inlet", "pressure_inlet", "mass_flow_inlet") or _brl_pname == "inlet":
            _brl_inlet_t = _brl_tv_f
        elif _brl_pname not in ("outlet",):
            _brl_wall_temps.append(_brl_tv_f)

    _brl_wall_heats = False
    if _brl_wall_temps:
        _brl_max_wall = max(_brl_wall_temps)
        if _brl_inlet_t is not None:
            _brl_wall_heats = _brl_max_wall > _brl_inlet_t + 10.0
        else:
            _brl_wall_heats = _brl_max_wall > 200.0  # conservative when no inlet_T

    if solver in _RHO_SINGLE_PHASE_FVOPTIONS and _brl_wall_heats:
        required.append("system/fvOptions")

    return required


def determine_solver(config: dict[str, Any]) -> str:
    """Deterministic fallback solver selection from validated_config.

    Prefer SolverSelector (LLM-assisted) over this function.
    This is a pure-logic fallback used when the selector is unavailable.

    Multi-region (CHT) is handled BEFORE this function runs — see
    :func:`simd_agent.run.multi_region.force_cht_solver_if_multi_region`,
    which the orchestrator calls first and returns the canonical
    ``chtMultiRegionSimpleFoam`` / ``chtMultiRegionFoam`` without an
    LLM call.  If a multi-region ``config["regions"]`` somehow reaches
    this single-region heuristic, we short-circuit to the CHT solver so
    we never run single-region physics rules on CHT topology.
    """
    from simd_agent.run.multi_region import force_cht_solver_if_multi_region
    _cht = force_cht_solver_if_multi_region(config)
    if _cht is not None:
        return _cht

    from simd_agent.run.solver_selector import _heuristic_fallback
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
        "  • algorithm block: SIMPLE (simpleFoam/rhoSimpleFoam/buoyantSimpleFoam), "
        "or PIMPLE (pimpleFoam/rhoPimpleFoam/buoyantPimpleFoam)\n"
        "  • relaxationFactors\n"
        "Use the template from the Solver Instructions."
    ),
    "0/U": (
        "Generate `0/U` (velocity field).\n"
        "Dimensions: [0 1 -1 0 0 0 0]\n"
        "If a mass flow rate is specified at the inlet, use flowRateInletVelocity:\n"
        "  - EXACTLY ONE of massFlowRate or volumetricFlowRate must be present.\n"
        "  - value uniform (0 0 0) is a placeholder — never put the flow rate here.\n"
        "  - rho/rhoInlet are OPTIONAL — only include them if they appear in the BC table.\n"
        "    Do NOT add rho/rhoInlet unless the user specified them.\n"
        "  - Do not include rho/rhoInlet for volumetric flow inlets.\n"
        "If a fixed velocity is specified: fixedValue with value uniform (<vx> <vy> <vz>).\n"
        "  inlet  → flowRateInletVelocity (mass/vol flow) or fixedValue (velocity)\n"
        "  outlet → zeroGradient\n"
        "  wall   → noSlip\n"
        "  frontAndBack → empty"
    ),
    "0/p": (
        "Generate `0/p` (pressure field).\n"
        "Incompressible (simpleFoam, pimpleFoam): "
        "dimensions [0 2 -2 0 0 0 0], values in m²/s². "
        "internalField uniform 0. outlet fixedValue uniform 0. "
        "Do NOT use 101325 or operating_pressure — only gradients matter.\n"
        "Compressible (rhoSimpleFoam, rhoPimpleFoam): "
        "dimensions [1 -1 -2 0 0 0 0], values in Pa. "
        "Use operating_pressure from config.\n"
        "  inlet  → zeroGradient\n"
        "  outlet → fixedValue\n"
        "  wall   → zeroGradient\n"
        "  frontAndBack (2D) → empty"
    ),
    "0/p_rgh": (
        "Generate `0/p_rgh` (modified pressure = p - rho*g*h).\n"
        "Dimensions: [1 -1 -2 0 0 0 0]\n"
        "Used by buoyant solvers (buoyantSimpleFoam, buoyantPimpleFoam).\n"
        "internalField: use `initial_domain_pressure` from the case spec.\n"
        "If initial_domain_pressure is None, fall back to operating_pressure.\n"
        "For fixed-flux boundaries with gravity:\n"
        "  inlet  → fixedFluxPressure\n"
        "  wall   → fixedFluxPressure\n"
        "For a specified static-pressure outlet:\n"
        "  outlet → fixedValue (configured outlet pressure)\n"
        "  frontAndBack → empty"
    ),
    "0/T": (
        "Generate `0/T` (temperature field).\n"
        "Dimensions: [0 0 0 1 0 0 0], values in Kelvin.\n"
        "Always generate an explicit positive internalField.\n"
        "CRITICAL: For compressible energy solvers (rhoSimpleFoam, rhoPimpleFoam, buoyantSimpleFoam, buoyantPimpleFoam):\n"
        "  internalField MUST equal `inlet_temperature` from the case spec (NOT 300 K).\n"
        "  Reason: icoPolynomial EOS → ρ(T) = a0 + a1·T. If you write 300K for LN2 (inlet=77K),\n"
        "  ρ = 1167.9 − 4.7×300 = −242 kg/m³ (negative) → SIGFPE on iteration 0 before any limiter runs.\n"
        "Use `initial_domain_temperature` from the case spec as the internalField value when present.\n"
        "If initial_domain_temperature is None, use `inlet_temperature`. NEVER default to 300 K.\n"
        "  inlet  → fixedValue (inlet temperature)\n"
        "  outlet → zeroGradient\n"
        "  wall   → fixedValue (wall temperature if configured) or zeroGradient (adiabatic)\n"
        "  frontAndBack → empty"
    ),
    "0/alpha": (
        "Generate the liquid phase-fraction field `0/alpha.<liquidPhase>` (or `0/alpha.<liquidPhase>.orig` for setFields cases).\n"
        "Dimensions: [0 0 0 0 0 0 0].\n"
        "alpha=1 means the cell is fully occupied by the liquid phase; alpha=0 means fully gas/vapour.\n\n"
        "The `initial_phase_layout` field in the case spec determines the internalField:\n"
        "  uniform_gas          → internalField uniform 0  (domain starts as vapour)\n"
        "  uniform_liquid       → internalField uniform 1  (domain starts as liquid)\n"
        "  liquid_region_in_gas → internalField uniform 0  (this is the .orig template; setFields fills the region)\n"
        "  gas_region_in_liquid → internalField uniform 1  (this is the .orig template; setFields clears the region)\n\n"
        "Boundary conditions:\n"
        "  inlet         → fixedValue uniform 1 (liquid injected) or fixedValue uniform 0 (vapour injected)\n"
        "  outlet        → zeroGradient\n"
        "  wall          → zeroGradient\n"
        "  frontAndBack  → empty"
    ),
    "0/k": (
        "Generate `0/k` (turbulent kinetic energy).\n"
        "Dimensions: [0 2 -2 0 0 0 0]\n"
        "IMPORTANT: Do NOT use k=0.1 as a default — that is ~500× too large for low-speed pipe flow.\n"
        "Estimate inlet U from mass flow rate (m_dot / (rho × A_inlet)) then:\n"
        "  k = 1.5 × (I × U)²   where I ≈ 0.05 (5 % turbulence intensity)\n"
        "For low-speed cryogenic pipe flow (U < 0.5 m/s), k is typically 1e-5 to 1e-3.\n"
        "  inlet  → fixedValue (computed k)\n"
        "  wall   → kqRWallFunction\n"
        "  outlet → zeroGradient\n"
        "  frontAndBack → empty"
    ),
    "0/omega": (
        "Generate `0/omega` (specific dissipation rate — kOmegaSST).\n"
        "Dimensions: [0 0 -1 0 0 0 0]\n"
        "Estimate: ω = k^0.5 / (Cμ^0.25 × ℓ)   Cμ = 0.09, ℓ ≈ 0.07 × hydraulic_diameter\n"
        "k and omega MUST be consistent — nut = k/omega; a tiny k with omega=100 gives nut≈0 (wrong).\n"
        "For low-speed cryogenic pipe flow, omega is typically 10–30 s⁻¹.\n"
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
        "Generate `constant/turbulenceProperties` using the turbulence model from the config (CaseSpec.sim_type / CaseSpec.turbulence_model).\n"
        "LAMINAR (sim_type=laminar): write ONLY 'simulationType  laminar;' — no model sub-dict.\n"
        "RANS (sim_type=RAS): 'simulationType  RAS;' + 'RAS { RASModel <model>; turbulence on; printCoeffs on; }'\n"
        "NEVER default to kOmegaSST when the config says laminar."
    ),
    "constant/thermophysicalProperties": (
        "Generate `constant/thermophysicalProperties`.\n\n"
        "## For rhoSimpleFoam / rhoPimpleFoam (single-phase compressible)\n"
        "EOS SELECTION — pick the FIRST rule that matches:\n"
        "  A. Cryogenic liquid (T_inlet < 200 K) OR has_heat_transfer=true:\n"
        "       → heRhoThermo + icoPolynomial + polynomial transport\n"
        "       (applies even for low-density cryogenics like LH2≈71 kg/m³, LHe≈125 kg/m³)\n"
        "       mixture/thermodynamics: CpCoeffs<8> (<Cp> 0 0 0 0 0 0 0), Hf=0, Sf=0\n"
        "       mixture/transport: muCoeffs<8> (<mu> ...), kappaCoeffs<8> (<kappa> ...)\n"
        "       mixture/equationOfState: rhoCoeffs<8> (<a0> <a1> 0 0 0 0 0 0)\n"
        "         where a1 = dρ/dT (LN2/LOX: -4.7; LH2: -0.7; LHe: -5.0; water/oil: -0.5)\n"
        "         and a0 = rho_inlet - a1 * T_inlet\n"
        "  B. rho <= 30 kg/m³ (gas/vapour): hePsiThermo + perfectGas + const/sutherland\n"
        "  C. rho > 30 AND isothermal (no heat transfer, T_inlet >= 200 K): heRhoThermo + rhoConst + const transport\n\n"
        "CRITICAL: when has_heat_transfer=true, ALWAYS use icoPolynomial (not rhoConst).\n"
        "Inside thermoType{}: keyword is 'thermo' (not 'thermodynamics').\n"
        "Inside mixture{}: sub-dict is 'thermodynamics' (not 'thermo')."
    ),
    "constant/g": (
        "Generate `constant/g` (gravitational acceleration).\n"
        "class: uniformDimensionedVectorField\n"
        "dimensions: [0 1 -2 0 0 0 0]\n"
        "value: (0 -9.81 0)"
    ),
    "system/setFieldsDict": (
        "Generate `system/setFieldsDict` for OpenFOAM's `setFields` utility.\n\n"
        "This file is needed when the initial phase distribution is non-uniform "
        "(initial_phase_layout = 'liquid_region_in_gas' or 'gas_region_in_liquid').\n"
        "The companion file `0/alpha.<liquidPhase>.orig` contains a uniform-0 template;\n"
        "setFields overwrites the alpha field inside the specified region.\n\n"
        "Standard structure:\n"
        "  defaultFieldValues ( volScalarFieldValue alpha.<liquidPhase> 0 );\n"
        "  regions\n"
        "  (\n"
        "    boxToCell\n"
        "    {\n"
        "        box (<xmin> <ymin> <zmin>) (<xmax> <ymax> <zmax>);\n"
        "        fieldValues ( volScalarFieldValue alpha.<liquidPhase> 1 );\n"
        "    }\n"
        "  );\n\n"
        "Use the mesh geometry from the config to define a sensible liquid region.\n"
        "If mesh geometry is unknown, use a small region near the inlet.\n"
        "For a gas_region_in_liquid layout: defaultFieldValues sets alpha=1 and the\n"
        "boxToCell region sets alpha=0 (the gas bubble location)."
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
        self._provider = get_provider()
        self.client = self._provider.client
        self.model = self._provider.models.get("super", self._provider.models["default"])
        self.event_bus = event_bus  # Optional[EventBus] — injected by orchestrator
        
        self._codegen_prompt = self._load_prompt("codegen.md")
        self._codefix_prompt = self._load_prompt("codefix.md")
    
    def _load_prompt(self, filename: str) -> str:
        prompt_path = PROMPTS_DIR / filename
        if prompt_path.exists():
            return prompt_path.read_text()
        logger.warning(f"Prompt file not found: {prompt_path}")
        return ""

    def _load_solver_prompt(self, solver: str) -> str:
        """Load the solver-specific system prompt from the plugin package."""
        from simd_agent.solvers import get_registry
        plugin = get_registry().get(solver)
        if plugin is None:
            logger.warning(f"[GENAI] No solver plugin registered for '{solver}'")
            return ""
        return plugin.system_prompt()
    
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

        Contains: solver identity/global rules + codegen rules + full config.
        Per-file docs are injected in _build_single_file_prompt, NOT here.
        When error_context is provided also includes a summary of what failed.
        """
        from simd_agent.solvers import get_registry
        _plugin = get_registry().get(solver)
        solver_instructions = _plugin.system_prompt() if _plugin else ""
        system_prompt = self._codegen_prompt
        if solver_instructions:
            system_prompt = (
                f"{system_prompt}\n\n"
                f"---\n\n## Solver Identity & Global Rules — {solver}\n\n{solver_instructions}"
            )

        # Inject fluid-specific pack when a known cryogenic fluid is configured.
        # This provides exact EOS coefficients, phase names, and safe temperature limits
        # so the LLM doesn't have to guess from raw density/viscosity values.
        _fluid_cfg = validated_config.get("fluid") or {}
        _fluid_name = (
            _fluid_cfg.get("name")
            or validated_config.get("fluid_name")
            or ""
        )
        _fluid_pack = _load_fluid_pack(_fluid_name)
        if _fluid_pack:
            system_prompt = (
                f"{system_prompt}\n\n"
                f"---\n\n## Fluid Properties Pack — {_fluid_name}\n\n{_fluid_pack}"
            )
            logger.info(f"[CODEGEN] Loaded fluid pack for '{_fluid_name}'")

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
        _is_steady = solver in {"simpleFoam", "rhoSimpleFoam", "buoyantSimpleFoam"}
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
                err_lines = [
                    "## ⚠️  Iteration History — All Previous Errors and Fix Attempts\n",
                    "You are regenerating files after one or more failed simulation attempts.",
                    "Each entry below is a SEPARATE iteration — read ALL of them before generating.",
                    "Do NOT repeat the same mistake that was diagnosed in a previous iteration.\n",
                ]
                for e in errors:
                    iteration = e.get("iteration", "?")
                    src = e.get("source", "unknown")
                    msg = e.get("error", "")
                    stderr = e.get("stderr", "") or e.get("details", "")
                    diagnosis = e.get("llm_diagnosis", "")
                    fixes = e.get("llm_fixes") or []

                    err_lines.append(f"### Iteration {iteration} — FAILED [{src}]")
                    if msg:
                        err_lines.append(f"**Error**: {msg}")
                    if stderr:
                        tail = "\n".join(stderr.splitlines()[-20:])
                        err_lines.append(f"**OpenFOAM output** (last 20 lines):\n```\n{tail}\n```")
                    if diagnosis:
                        err_lines.append(f"**LLM Root-Cause Diagnosis**: {diagnosis}")
                    if fixes:
                        err_lines.append("**Fixes attempted in the next iteration**:")
                        for fix in fixes:
                            err_lines.append(f"  - {fix}")
                    err_lines.append("")

                error_section = "\n\n---\n\n" + "\n".join(err_lines)

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
            # controlDict: strip any LLM-generated functions block — the
            # validator always injects the correct one.  This prevents
            # truncation from an overly long output.
            if content and file_path == "system/controlDict":
                content = _strip_functions_block(content)
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
        # Exact lookup first; for alpha.* and alpha.*.orig fall back to the "0/alpha" key
        hint = _FILE_HINTS.get(file_path)
        if hint is None and (file_path.startswith("0/alpha.") or ".orig" in file_path):
            # Strip .orig suffix for hint lookup (0/alpha.liquidNitrogen.orig → "0/alpha")
            _fp_base = file_path.removesuffix(".orig")
            if _fp_base.startswith("0/alpha."):
                hint = _FILE_HINTS.get("0/alpha")
        if hint is None:
            hint = f"Generate the file `{file_path}` appropriate for the `{solver}` solver."

        # ── CaseSpec-based context (compact, file-specific) ──────────────────
        if case_spec:
            spec_dict = case_spec.as_prompt_dict_for_file(file_path)
            import json as _json
            spec_str = _json.dumps(spec_dict, indent=2, default=str)
            context_section = (
                f"## CaseSpec (single source of truth — use ONLY these values)\n"
                f"```json\n{spec_str}\n```\n"
            )

            # ── Per-file solver doc (targeted instructions for this one file) ──
            from simd_agent.solvers import get_registry
            from simd_agent.solvers.base import SolverPlugin
            _plugin = get_registry().get(solver)
            _file_doc_rel = SolverPlugin._file_doc_relpath(file_path)
            _file_doc = _plugin.prompt_for_file(file_path) if _plugin is not None else ""
            if _file_doc:
                context_section += (
                    f"\n\n## File-specific instructions ({_file_doc_rel})\n"
                    f"{_file_doc}\n"
                )

            # Fallback: _brief_solver_note() only when no per-file doc exists
            if not _file_doc and (file_path.startswith("system/") or file_path.startswith("constant/")):
                solver_note_brief = self._brief_solver_note(file_path, case_spec)
                context_section += f"\n{solver_note_brief}"

            # For 0/* field files: always inject mandatory per-patch BC table
            if file_path.startswith("0/"):
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
                p_note = "\nCOMPRESSIBLE: dimensions [1 -1 -2 0 0 0 0], values in Pa."
            else:
                p_note = (
                    "\nINCOMPRESSIBLE: dimensions [0 2 -2 0 0 0 0], values in m²/s²."
                    "\nUse internalField uniform 0 and outlet fixedValue uniform 0."
                    "\nDo NOT use 101325 or operating_pressure — those are Pa (absolute), not kinematic gauge."
                )

        energy_note = ""  # 0/h and 0/e are never generated; thermo initialises from 0/T

        # ── Error context: iteration history + previous broken file ────────────
        error_section = ""
        if error_context:
            errors = error_context.get("errors", [])

            # Build iteration history summary (capped at 10k chars total)
            if errors:
                history_lines = [
                    "\n### ⚠️  Previous Iteration Errors — Read Before Regenerating\n",
                    "Do NOT repeat mistakes diagnosed in earlier iterations.\n",
                ]
                for e in errors:
                    iteration = e.get("iteration", "?")
                    src = e.get("source", "unknown")
                    msg = e.get("error", "")
                    stderr = e.get("stderr", "") or e.get("details", "")
                    diagnosis = e.get("llm_diagnosis", "")
                    fixes = e.get("llm_fixes") or []

                    history_lines.append(f"**Iteration {iteration} [{src}]**: {msg}")
                    if stderr:
                        # Include last 10 lines of stderr per iteration
                        tail = "\n".join(stderr.splitlines()[-10:])
                        history_lines.append(f"```\n{tail}\n```")
                    if diagnosis:
                        history_lines.append(f"Diagnosis: {diagnosis}")
                    if fixes:
                        history_lines.append("Fixes attempted: " + "; ".join(fixes[:3]))
                    history_lines.append("")

                history_text = "\n".join(history_lines)
                # Cap at 10k characters to avoid context bloat
                if len(history_text) > 10000:
                    history_text = history_text[:10000] + "\n... (truncated — see earlier iterations)\n"
                error_section += history_text

            # Show previous version of THIS specific file — full content
            # so the LLM has complete context.  Strip the validator-injected
            # functions block from controlDict: it's always re-injected after
            # generation, and reproducing it wastes output tokens / causes
            # truncation that loops the retry cycle.
            prev_files = error_context.get("previous_files", {})
            if file_path in prev_files:
                prev = prev_files[file_path]
                if file_path == "system/controlDict":
                    prev = _strip_functions_block(prev)
                error_section += (
                    f"\n### Previous version of `{file_path}` (BROKEN — fix it)\n```\n{prev}\n```\n"
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
            f"4. `application` in controlDict MUST be EXACTLY `{solver}` — "
            f"this solver is LOCKED for this run, never substitute another\n"
            f"   solver name (rhoSimpleFoam, buoyantSimpleFoam, etc.) even if "
            f"the error log seems to suggest a different solver would fix it.\n"
            f"   Fixes belong in BCs, schemes, relaxation, fvOptions — never\n"
            f"   in the `application` field.\n"
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
            f"   (first line inside block MUST be `FoamFile`, never `{file_path}`)\n"
            + (f"9. Do NOT include a `functions` block — function objects "
               f"(fieldMinMax, surfaceFieldValue, volFieldValue) are auto-injected "
               f"by the validator after generation. End the file after the last "
               f"solver setting.\n" if file_path == "system/controlDict" else "")
            + f"\nGenerate complete `{file_path}` now:"
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

            def _fmt_val(v: Any) -> str:
                if isinstance(v, (list, tuple)):
                    return "(" + " ".join(str(x) for x in v) + ")"
                return str(v)

            if isinstance(field_entry, dict):
                bc_type = field_entry.get("type", "zeroGradient")

                entries: dict[str, Any] = {}
                if isinstance(field_entry.get("entries"), dict):
                    entries.update(field_entry["entries"])
                for kk, vv in field_entry.items():
                    if kk in ("type", "entries"):
                        continue
                    entries.setdefault(kk, vv)

                if bc_type == "flowRateInletVelocity":
                    # ── Ensure the flow rate key is present and non-zero ──────
                    _fr_keys = ("massFlowRate", "volumetricFlowRate", "meanVelocity")
                    _flow_rate_key = "massFlowRate"
                    has_required = any(
                        k in entries and entries[k] is not None and entries[k] != 0
                        for k in _fr_keys
                    )
                    if not has_required:
                        # Remove any zero/None placeholders
                        for _k in _fr_keys:
                            entries.pop(_k, None)

                        # Recovery chain — 4 sources in priority order
                        _recovered: float | None = None
                        raw_val = entries.pop("value", None)
                        # Source 1: scalar value IS the flow rate
                        if isinstance(raw_val, (int, float)) and raw_val != 0:
                            _recovered = float(raw_val)
                        # Source 2: misinterpreted as velocity vector [flow_rate, 0, 0]
                        elif isinstance(raw_val, (list, tuple)) and len(raw_val) >= 1:
                            first = raw_val[0]
                            if isinstance(first, (int, float)) and first != 0:
                                _recovered = float(first)

                        if _recovered is not None:
                            entries[_flow_rate_key] = _recovered
                        else:
                            entries[_flow_rate_key] = 0
                            logger.warning(
                                "[BC_TABLE] flowRateInletVelocity inlet has no flow rate value — "
                                "massFlowRate set to 0. The simulation WILL diverge."
                            )

                    # ── rho / rhoInlet — only if the user specified them ──────
                    # Do NOT auto-inject: these are user-owned entries.
                    # The LLM prompt (via _FILE_HINTS and solver .md) instructs the
                    # model to add rho/rhoInlet when appropriate.
                    # The only cleanup: remove rho/rhoInlet that were carried over from
                    # a volumetricFlowRate inlet (they don't belong there).
                    _has_vfr = "volumetricFlowRate" in entries and entries["volumetricFlowRate"] not in (None, 0)
                    if _has_vfr:
                        entries.pop("rho", None)
                        entries.pop("rhoInlet", None)

                    # ── Optional: extrapolateProfile ─────────────────────────
                    # Keep it if the user specified it; omit by default (plug flow = safer for codegen)
                    if entries.get("extrapolateProfile") is False:
                        entries.pop("extrapolateProfile")

                    # ── value is always a vector placeholder (NOT the flow rate) ──
                    entries["value"] = [0, 0, 0]

                # ── Emit the OpenFOAM dictionary snippet ─────────────────────
                # Ordering: type first, then flow-rate key, then rho/rhoInlet,
                # then extrapolateProfile (if set), then value last.
                _ORDERED_KEYS = [
                    "massFlowRate", "volumetricFlowRate", "meanVelocity",
                    "rho", "rhoInlet", "extrapolateProfile",
                ]
                if bc_type == "flowRateInletVelocity":
                    ordered_entries: dict[str, Any] = {}
                    for _ok in _ORDERED_KEYS:
                        if _ok in entries:
                            ordered_entries[_ok] = entries[_ok]
                    for _ok, _ov in entries.items():
                        if _ok not in ordered_entries and _ok != "value":
                            ordered_entries[_ok] = _ov
                    if "value" in entries:
                        ordered_entries["value"] = entries["value"]
                    entries = ordered_entries

                snippet = [f"  {patch} {{", f"      type            {bc_type};"]
                for kk, vv in entries.items():
                    if kk == "value":
                        snippet.append(f"      value           uniform {_fmt_val(vv)};")
                    elif isinstance(vv, bool):
                        snippet.append(f"      {kk:16s}{'yes' if vv else 'no'};")
                    else:
                        snippet.append(f"      {kk:16s}{_fmt_val(vv)};")
                snippet.append("  }")
                lines.append("\n".join(snippet))
                continue

            elif isinstance(field_entry, str):
                lines.append(f"  {patch:20s}: type {field_entry};")
            else:
                lines.append(f"  {patch:20s}: {field_entry}")

        if field_name == "U":
            lines.append("\n# internalField: do NOT put massFlowRate here.")
            lines.append("# internalField: use a safe initial guess (uniform (0 0 0)) if unknown.")
        elif field_name in ("p", "p_rgh"):
            if cs.compressible:
                # Compressible solvers use absolute pressure in Pa
                lines.append("\n# internalField: start near operating pressure to avoid blow-up")
                lines.append(f"# internalField: uniform {cs.operating_pressure}")
            else:
                # Incompressible solvers use kinematic gauge pressure (m²/s²) — value is 0
                lines.append("\n# internalField: uniform 0  (incompressible = gauge kinematic pressure)")
                lines.append("# DO NOT use 101325 or any absolute pressure value for incompressible solvers")
        else:
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
            ddt = "Euler" if cs.transient else "steadyState"
            efield = cs.energy_field or "h"
            rho_prefix = "rho*" if cs.compressible else ""
            visc_term = f"div((({rho_prefix}nuEff)*dev2(T(grad(U)))))"

            # Compressible energy solvers require K and phid,p explicitly
            compressible_lines = ""
            if cs.energy == "he":
                compressible_lines = (
                    f"    div(phi,{efield})                          bounded Gauss upwind;\n"
                    f"    div(phi,K)                                bounded Gauss upwind;\n"
                    f"    div(phid,p)                               Gauss limitedLinear 1;\n"
                )

            # Turbulence scalars (only the ones that actually exist as fields)
            turb_scalars = [f for f in ("k", "omega", "epsilon") if f in cs.turbulence_fields]
            turb_lines = "".join(
                f"    div(phi,{f})                                bounded Gauss limitedLinear 1;\n"
                for f in turb_scalars
            )

            wall_dist = "wallDist { method meshWave; }\n" if cs.turbulence_model not in ("laminar", "none") else ""

            return (
                "## fvSchemes — COPY THIS EXACT SHAPE\n"
                f"ddtSchemes      {{ default {ddt}; }}\n"
                "gradSchemes     { default Gauss linear; }\n"
                "divSchemes\n"
                "{\n"
                "    default                                   bounded Gauss upwind;\n"
                "    div(phi,U)                                bounded Gauss linearUpwind grad(U);\n"
                f"{compressible_lines}"
                f"{turb_lines}"
                f"    {visc_term}  Gauss linear;\n"
                "}\n"
                "laplacianSchemes     { default Gauss linear corrected; }\n"
                "interpolationSchemes { default linear; }\n"
                "snGradSchemes        { default corrected; }\n"
                f"{wall_dist}"
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
            if cs.transient:
                n_steps = int(cs.end_time / cs.delta_t)
                adjust_note = (
                    f"\n- adjustTimeStep: yes   ← REQUIRED for transient solvers; lets OpenFOAM auto-scale "
                    f"deltaT up to maxCo={cs.max_co} so the simulation runs as fast as physics allows\n"
                    f"- maxCo: {cs.max_co}\n"
                    f"- DO NOT set deltaT smaller than {cs.delta_t}. "
                    f"With deltaT={cs.delta_t} and endTime={end_time_fmt} that is already {n_steps} steps — "
                    f"making it smaller (e.g. 0.0001) would multiply the runtime by 10x with no benefit.\n"
                    f"- writeControl: adjustableRunTime  (pair with adjustTimeStep)\n"
                    f"- writeInterval: {cs.write_interval:.6g}  (write ~{int(cs.end_time / cs.write_interval)} snapshots)\n"
                    f"- maxDeltaT: {cs.max_delta_t:.6g}\n"
                )
            else:
                wi = int(cs.end_time / 20) if cs.end_time > 0 else 100
                wi = max(wi, 1)
                adjust_note = f"- writeInterval: {wi}  (write ~20 snapshots)\n"
            return (
                f"## controlDict guidance\n"
                f"- application: {cs.solver}\n"
                f"- endTime: {end_time_fmt}   ← integer, NO decimal point (write {end_time_fmt} not {float(cs.end_time)})\n"
                f"- deltaT: {cs.delta_t}   ← may be float; NEVER go below this value\n"
                f"- startFrom startTime; startTime 0;\n"
                + adjust_note
            )
        if file_path == "constant/turbulenceProperties":
            if cs.sim_type == "laminar":
                return (
                    "## turbulenceProperties guidance\n"
                    "Flow is LAMINAR. Write ONLY:\n"
                    "  simulationType  laminar;\n"
                    "DO NOT add any RAS, LES, or model sub-dict — laminar flow needs no turbulence model block.\n"
                )
            return (
                f"## turbulenceProperties guidance\n"
                f"- simulationType: {cs.sim_type};\n"
                f"- {cs.sim_type} {{ RASModel {cs.turbulence_model}; turbulence on; printCoeffs on; }}\n"
            )
        if file_path == "system/fvOptions":
            t_min = cs.fv_options_t_min or max(1.0, (cs.inlet_temperature or 200.0) * 0.5)
            ceiling = cs.fv_options_eos_t_ceiling
            bc_temps = cs.fv_options_bc_temps or []
            # Fallback max: 90% of EOS ceiling (consistent with validator Check 3c3)
            t_max_fallback = ceiling * 0.9 if ceiling else 100000.0
            ceiling_note = (
                f"\n⚠️  EOS ceiling: ρ(T) = a0 + a1·T → ρ = 0 at T ≈ {ceiling:.1f} K.\n"
                f"   max MUST be below {ceiling:.1f} K or density goes negative → SIGFPE.\n"
                if ceiling else ""
            )
            bc_note = f"BC temperatures in case: {bc_temps} K\n" if bc_temps else ""
            return (
                f"## system/fvOptions — temperature limiter (REQUIRED)\n\n"
                f"This file is REQUIRED for all compressible energy solvers.\n"
                f"It prevents negative-temperature and negative-density divergence.\n\n"
                f"Known temperatures:\n"
                f"  {bc_note}"
                f"  min (floor) = {t_min:.1f} K  (50% of coldest BC — numerical floor)\n"
                f"{ceiling_note}"
                f"\nChoose max = the highest physically meaningful temperature for this fluid.\n"
                f"If unsure, use {t_max_fallback:.0f} K (conservative safe default).\n\n"
                f"```\n"
                f"FoamFile\n"
                f"{{\n"
                f"    version     2.0;\n"
                f"    format      ascii;\n"
                f"    class       dictionary;\n"
                f"    location    \"system\";\n"
                f"    object      fvOptions;\n"
                f"}}\n"
                f"// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
                f"temperatureLimiter\n"
                f"{{\n"
                f"    type            limitTemperature;\n"
                f"    active          yes;\n\n"
                f"    selectionMode   all;\n\n"
                f"    min             {t_min:.1f};   // 50% of coldest BC temperature [K]\n"
                f"    max             {t_max_fallback:.0f};  // highest physically meaningful T [K]\n"
                f"}}\n\n"
                f"// ************************************************************************* //\n"
                f"```\n"
            )
        if file_path == "constant/thermophysicalProperties":
            if cs.rho is not None:
                eos = _eos_for_liquid(cs.rho, cs.inlet_temperature, cs.energy == "he")
                thermo_type = "heRhoThermo"
                if eos == "icoPolynomial":
                    # icoPolynomial REQUIRES transport=polynomial + thermo=hPolynomial.
                    # Valid chain: heRhoThermo<pureMixture<polynomial<hPolynomial<icoPolynomial>>>>
                    # Using const+hConst+icoPolynomial → "Unknown fluidThermo type" FOAM error.
                    a0, a1 = _ico_poly_coeffs(cs.rho, cs.inlet_temperature)
                    cp_val = cs.cp or 1000.0
                    mu_val = cs.mu or 1.8e-5
                    pr_val = cs.prandtl or 0.713
                    kappa_val = mu_val * cp_val / pr_val
                    eos_note = (
                        f"⚠️  EOS = icoPolynomial → MUST use transport=polynomial, thermo=hPolynomial.\n"
                        f"    (const+hConst+icoPolynomial is NOT a valid OpenFOAM type combination.)\n"
                        f"    ρ(T) = {a0:.3f} + ({a1:.4f})*T  [kg/m³, T in K]\n"
                        f"    At T_inlet={cs.inlet_temperature or 'unknown'}K → ρ ≈ {cs.rho:.1f} kg/m³ ✓\n"
                    )
                    return (
                        f"## thermophysicalProperties — EXACT required structure\n\n"
                        + eos_note
                        + f"```\n"
                        f"thermoType\n"
                        f"{{\n"
                        f"    type            heRhoThermo;\n"
                        f"    mixture         pureMixture;\n"
                        f"    transport       polynomial;      // MUST be polynomial when using icoPolynomial EOS\n"
                        f"    thermo          hPolynomial;     // MUST be hPolynomial when transport=polynomial\n"
                        f"    equationOfState icoPolynomial;\n"
                        f"    specie          specie;\n"
                        f"    energy          sensibleEnthalpy;\n"
                        f"}}\n\n"
                        f"mixture\n"
                        f"{{\n"
                        f"    specie\n"
                        f"    {{\n"
                        f"        molWeight   28.97;\n"
                        f"    }}\n"
                        f"    thermodynamics    // hPolynomial: use CpCoeffs<8> + Hf + Sf (NOT plain Cp)\n"
                        f"    {{\n"
                        f"        Hf              0;\n"
                        f"        Sf              0;\n"
                        f"        CpCoeffs<8>     ({cp_val:.4g} 0 0 0 0 0 0 0);\n"
                        f"    }}\n"
                        f"    transport         // polynomial: use muCoeffs + kappaCoeffs (NOT mu/Pr)\n"
                        f"    {{\n"
                        f"        muCoeffs<8>     ({mu_val:.4g} 0 0 0 0 0 0 0);\n"
                        f"        kappaCoeffs<8>  ({kappa_val:.4g} 0 0 0 0 0 0 0);\n"
                        f"    }}\n"
                        f"    equationOfState\n"
                        f"    {{\n"
                        f"        rhoCoeffs<8>    ({a0:.3f} {a1:.4f} 0 0 0 0 0 0);\n"
                        f"    }}\n"
                        f"}}\n"
                        f"```\n"
                        f"Physical values: Cp={cp_val}, mu={mu_val}, Pr={pr_val}, kappa=mu*Cp/Pr={kappa_val:.5g}\n"
                    )
                else:
                    eos_block = f"    equationOfState {{ rho {cs.rho}; }}\n"
                    eos_note = ""
            else:
                eos = "perfectGas"
                thermo_type = "hePsiThermo"
                eos_block = ""
                eos_note = ""
            return (
                f"## thermophysicalProperties — EXACT required structure\n\n"
                + eos_note
                + f"⚠️  CRITICAL KEY NAMES — wrong names cause FOAM FATAL IO ERROR:\n"
                f"- Inside `thermoType {{}}`: the key is `thermo` (NOT `thermodynamics`)\n"
                f"- Inside `mixture {{}}`: the sub-dict is `thermodynamics` (NOT `thermo`)\n\n"
                f"```\n"
                f"thermoType\n"
                f"{{\n"
                f"    type            {thermo_type};\n"
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
                f"        molWeight   28.97;\n"
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
                + eos_block
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
                config=self._provider.types.GenerateContentConfig(
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
        ai_editable_files: list[str] | None = None,
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

        # ── LLM diagnosis: trust affected_files from error_summarizer ────────
        # The error_summarizer LLM often identifies the real root-cause file
        # more accurately than regex pattern matching.  Include its suggestions
        # directly so they get regenerated with the full error context.
        for e in previous_errors:
            for llm_file in e.get("llm_affected_files") or []:
                if isinstance(llm_file, str) and llm_file:
                    affected.add(llm_file)

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
            "unknown div scheme",
            "walldist",
            "interpolationscheme",
            "divscheme",
            "laplacianscheme",
        ]
        if any(_re.search(p, all_errors_text) for p in SCHEME_PATTERNS):
            affected.add("system/fvSchemes")

        # ── fvSolution / solver entry errors ─────────────────────────────
        FVSOLUTION_PATTERNS = [
            r"entry\s+'?solver'?\s+not found in dictionary.*fvsolution",
            r"entry\s+'?solver'?\s+not found in dictionary.*solvers",
            r"not found in dictionary.*alpha\.",
        ]
        if any(_re.search(p, all_errors_text) for p in FVSOLUTION_PATTERNS):
            affected.add("system/fvSolution")

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

        # ── liquidProperties selector errors → liquid phase file ONLY ─────
        # When stack trace shows thermophysicalPropertiesSelector<liquidProperties>,
        # the failure is in the liquid phase thermo file (native N2/H2/O2/He model),
        # NOT in the vapour file (perfectGas). Target the liquid file only.
        LIQUID_PROPS_PATTERNS = [
            "liquidproperties",
            "thermophysicalpropertiesselector.*liquid",
            "liquid.*selector",
        ]
        if any(_re.search(p, all_errors_text) for p in LIQUID_PROPS_PATTERNS):
            # Add all per-phase liquid thermo files (not the base file)
            for f in previous_files:
                if f.startswith("constant/thermophysicalProperties.") and f != "constant/thermophysicalProperties":
                    # Heuristic: liquid phase name typically doesn't end in 'Vapour' or 'Gas'
                    base = f.split("constant/thermophysicalProperties.", 1)[-1].lower()
                    if not any(v in base for v in ("vapour", "vapor", "gas")):
                        affected.add(f)
            # Also flag the base file in case phases list is wrong
            affected.add("constant/thermophysicalProperties")

        # ── turbulence errors ─────────────────────────────────────────────
        TURB_PATTERNS = ["turbulenceproperties", "rasmodel", "turbulence.*on"]
        if any(_re.search(p, all_errors_text) for p in TURB_PATTERNS):
            affected.add("constant/turbulenceProperties")

        # ── GAMG / DIC SIGFPE — surgical flag ─────────────────────────────
        # DICPreconditioner::calcReciprocalD crashes when the pressure matrix has
        # zero/negative diagonal.  This happens when the flow diverges and the
        # coarsest GAMG level becomes ill-conditioned.
        # Flag so the retry loop can apply a surgical fix instead of full regen.
        #
        # IMPORTANT: only match the LATEST error, not all accumulated ones.
        # If GAMG appeared in iteration 1 but was already patched, we must not
        # re-trigger this sentinel in iteration 5 when the real error is something
        # else (e.g. a syntax error).  Matching all_errors_text would cause
        # fvSolution to be permanently locked out of LLM regeneration.
        GAMG_DIC_PATTERNS = [
            "dicpreconditioner",
            "calcreciprocald",
            "gamgsolver.*solvecoarsestlevel",
            "gamgsolver.*vcycle",
            "gamgsolver.*scale",
        ]
        _latest_error_text = ""
        if previous_errors:
            _le = previous_errors[-1]
            _latest_error_text = (
                _le.get("error", "") + " " + (_le.get("stderr", "") or _le.get("details", ""))
            ).lower()
        if any(_re.search(p, _latest_error_text) for p in GAMG_DIC_PATTERNS):
            affected.add("system/fvSolution")
            # Set a sentinel so _apply_surgical_fixes knows the exact action
            affected.add("__surgical:switch_p_solver_gamg_to_pbicgstab__")

        # If nothing was identified, regenerate all required files
        if not affected:
            logger.warning(
                "[PARALLEL] Could not identify affected files from error — "
                "regenerating all required files"
            )
            return build_required_files_list(solver, validated_config)

        # ── Filter out files the plugin generates deterministically ──────────
        # Some plugins (e.g. simpleFoam) build certain files deterministically
        # in validate() — they are NOT in plugin.required_files().  Regenerating
        # them via LLM is wasted compute since validate() will discard the result.
        #
        # HOWEVER: if a deterministic file is the ONLY affected file (the error
        # points exclusively at it), filtering it out leaves only generic files
        # like controlDict — the self-healing loop can never fix the real issue.
        # In that case, we must still regenerate OTHER files that could influence
        # the solver behavior, rather than leaving the loop stuck.
        try:
            from simd_agent.solvers import get_registry as _gr
            _plug = _gr().get(solver)
        except Exception:
            _plug = None
        if _plug is not None:
            _plugin_files = set(_plug.required_files(validated_config))
            # User-unlocked deterministic files are sent to the LLM like any
            # other affected file — the orchestrator restores LLM output after
            # validate_full() to preserve the regeneration.
            _ai_editable_set = set(ai_editable_files or ())
            _deterministic = {
                f for f in affected
                if not f.startswith("__surgical:")
                and f not in _plugin_files
                and f not in _ai_editable_set
            }
            if _deterministic:
                logger.info(
                    f"[PARALLEL] Skipping LLM regen for deterministic files: "
                    f"{sorted(_deterministic)}"
                )
                # Check if removing deterministic files would leave us with
                # ONLY system/controlDict or nothing meaningful — that means
                # the real problem IS in the deterministic builder and
                # regenerating controlDict alone won't help.
                _remaining = affected - _deterministic - {
                    f for f in affected if f.startswith("__surgical:")
                }
                if _remaining <= {"system/controlDict"} or not _remaining:
                    logger.warning(
                        f"[PARALLEL] Deterministic files {sorted(_deterministic)} "
                        f"are the primary error targets but are built by the "
                        f"plugin validator, not the LLM.  Regenerating ALL "
                        f"LLM-owned files to give the validator fresh input."
                    )
                    # Regenerate all plugin-required files so the validator
                    # gets a fresh set to rebuild deterministic files from.
                    affected = set(build_required_files_list(solver, validated_config))
                else:
                    affected -= _deterministic

        logger.info(f"[PARALLEL] Affected files identified from error: {sorted(affected)}")
        return sorted(affected)

    @staticmethod
    def apply_surgical_fixes(
        files: dict[str, str],
        affected_files: list[str],
    ) -> tuple[dict[str, str], list[str]]:
        """Apply targeted in-place fixes to existing files based on error patterns.

        Returns the (possibly modified) files dict and a list of applied fix descriptions.
        Surgical fixes avoid full LLM regeneration when the root cause is a known
        numerical setting (e.g. wrong pressure solver choice after divergence).
        """
        applied: list[str] = []
        import re as _sre

        # ── Surgical fix: switch GAMG → PBiCGStab for pressure ────────────────
        # Triggered by DICPreconditioner::calcReciprocalD SIGFPE.
        # When the coarsest GAMG level has a non-positive-definite matrix (caused by
        # divergence), even adding coarsestLevelCorr cannot help.  The safest fallback
        # is to replace the GAMG p solver entirely with PBiCGStab+DILU, which does not
        # depend on matrix positive-definiteness.
        if "__surgical:switch_p_solver_gamg_to_pbicgstab__" in affected_files:
            fvs = files.get("system/fvSolution", "")
            if fvs and "GAMG" in fvs:
                # Replace the entire p solver block (from "p" opening to matching "}")
                _new_p_block = (
                    "p\n"
                    "    {\n"
                    "        solver          PBiCGStab;\n"
                    "        preconditioner  DILU;\n"
                    "        tolerance       1e-6;\n"
                    "        relTol          0.1;\n"
                    "    }"
                )
                # Match the p solver block: p { ... } (simple non-nested match)
                _patched = _sre.sub(
                    r'p\s*\{[^}]*solver\s+GAMG[^}]*\}',
                    _new_p_block,
                    fvs,
                    count=1,
                    flags=_sre.DOTALL,
                )
                if _patched != fvs:
                    # Also update pFinal to match
                    _patched = _sre.sub(
                        r'pFinal\s*\{[^}]*\}',
                        "pFinal\n    {\n        $p;\n        relTol          0;\n    }",
                        _patched,
                        count=1,
                        flags=_sre.DOTALL,
                    )
                    files["system/fvSolution"] = _patched
                    applied.append(
                        "Switched 0/p solver from GAMG to PBiCGStab+DILU "
                        "(DICPreconditioner SIGFPE — coarsest GAMG level ill-conditioned)"
                    )
                    logger.info("[SURGICAL] Switched p solver: GAMG → PBiCGStab+DILU")

        return files, applied

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
        ai_editable_files: list[str] | None = None,
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
                ai_editable_files=ai_editable_files or [],
            )
            # Cap error history sent to the LLM to the most recent iterations.
            # Older errors add context noise and inflate the prompt; regex-based
            # _identify_affected_files still sees the full list above.
            _MAX_LLM_ERROR_HISTORY = 3
            error_ctx = {
                "errors": previous_errors[-_MAX_LLM_ERROR_HISTORY:],
                "previous_files": previous_files or {},
            }
            # Strip internal-only surgical sentinels — they are never real files
            # to generate; they are handled by apply_surgical_fixes() in the
            # orchestrator before this call.
            # Also exclude the files *owned* by each sentinel: those were already
            # patched in-place by the orchestrator, and regenerating them from
            # scratch would overwrite the surgical fix.
            _SURGICAL_OWNED: dict[str, list[str]] = {
                "__surgical:switch_p_solver_gamg_to_pbicgstab__": ["system/fvSolution"],
            }
            _surgical_skip: set[str] = set()
            for _sent in affected:
                if _sent.startswith("__surgical:"):
                    _surgical_skip.update(_SURGICAL_OWNED.get(_sent, []))
            if _surgical_skip:
                logger.info(
                    f"[SURGICAL] Skipping LLM regen for surgically-patched files: {sorted(_surgical_skip)}"
                )
            affected_real = [
                f for f in affected
                if not f.startswith("__surgical:") and f not in _surgical_skip
            ]
            logger.info(
                f"[GENAI] ERROR-RECOVERY parallel generation: "
                f"solver={solver}  affected={affected_real}"
            )
            return await self.generate_parallel(
                requirements=requirements,
                validated_config=validated_config,
                solver=solver,
                case_type=case_type,
                files_to_generate=affected_real,
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

        _is_steady = solver in {"simpleFoam", "rhoSimpleFoam", "buoyantSimpleFoam"}

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

        # fvOptions rule — injected as an explicit critical rule for compressible solvers
        _fvoptions_rule = ""
        if "system/fvOptions" in _required_files:
            _fvoptions_rule = (
                "\n14. **`system/fvOptions` is REQUIRED** — this is a compressible energy solver. "
                "Generate `system/fvOptions` with a `limitTemperature` block "
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
                config=self._provider.types.GenerateContentConfig(
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
# controlDict functions-block stripping
# ────────────────────────────────────────────────────────────

def _strip_functions_block(content: str) -> str:
    """Strip the ``functions { ... }`` block from controlDict content.

    Function objects (fieldMinMax, surfaceFieldValue, volFieldValue) are always
    re-injected by the validator after generation.  Including them in the
    error-recovery context causes the LLM to reproduce the large block, which
    frequently gets truncated mid-way — leaving unbalanced braces that the
    simulation server rejects, triggering an infinite retry loop.

    Uses brace-depth counting to handle arbitrarily nested sub-dicts.
    """
    match = re.search(r'\n\s*functions\s*\{', content)
    if not match:
        return content
    start = match.start()
    depth = 0
    i = match.end() - 1  # position of the opening '{'
    while i < len(content):
        if content[i] == '{':
            depth += 1
        elif content[i] == '}':
            depth -= 1
            if depth == 0:
                # Found the matching closing brace
                return content[:start].rstrip() + "\n"
        i += 1
    # functions block is unclosed (already truncated) — strip it all
    return content[:start].rstrip() + "\n"


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
