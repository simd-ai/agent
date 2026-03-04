# simd_agent/code_verifier.py
"""Post-generation code verification using rule-based checks only.

After the flash codegen model produces OpenFOAM files, the verifier runs a
set of fast, deterministic checks to catch structural mistakes before the ZIP
is sent to the simulation server.  No LLM is used here — solver selection is
the only step that uses the super model.

The verifier does NOT rewrite files.  It only reports issues.  Critical issues
(e.g. wrong solver for the selected physics) are surfaced as OrchestrationErrors
so the self-healing loop regenerates with explicit guidance.  Warnings are
emitted as events and logged, but do not block submission.
"""

import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from simd_agent.solver_selector import (
    ALLOWED_SOLVERS,
    ENERGY_SOLVERS,
    GRAVITY_SOLVERS,
    P_RGH_SOLVERS,
    P_SOLVERS,
    THERMO_SOLVERS,
)

logger = logging.getLogger(__name__)


# ── Data models ───────────────────────────────────────────────────────────────

class VerificationIssue(BaseModel):
    severity: Literal["critical", "warning", "info"]
    category: str = Field(
        description=(
            "Short category tag, e.g. 'solver_mismatch', 'missing_field', "
            "'bc_inconsistency', 'turbulence_mismatch', "
            "'heat_transfer_solver', 'patch_coverage'"
        )
    )
    message: str
    fix_suggestion: str | None = None


class VerificationResult(BaseModel):
    passed: bool = Field(
        description="True when there are no critical issues — safe to proceed."
    )
    issues: list[VerificationIssue] = Field(default_factory=list)
    summary: str


# ── Solver property sets (derived from solver_selector canonical sets) ─────────

# Solvers that solve an energy/temperature equation
_HEAT_TRANSFER_SOLVERS = ENERGY_SOLVERS  # rhoSimpleFoam, rhoPimpleFoam, compressible* VOF

# Steady-state solvers (endTime = iteration count, not physical seconds)
_STEADY_SOLVERS = {"simpleFoam", "rhoSimpleFoam"}

# Transient solvers (endTime = physical seconds)
_TRANSIENT_SOLVERS = ALLOWED_SOLVERS - _STEADY_SOLVERS


# ── Verifier ──────────────────────────────────────────────────────────────────

class CodeVerifier:
    """Verify generated OpenFOAM files for correctness and consistency.

    Uses fast rule-based checks only — no LLM involved.  The super model is
    used upstream in SolverSelector, not here.
    """

    def __init__(self) -> None:
        pass  # No LLM client needed

    # ── Public API ────────────────────────────────────────────────────────────

    async def verify(
        self,
        files: dict[str, str],
        user_requirements: str,
        validated_config: dict[str, Any],
        solver: str,
    ) -> VerificationResult:
        """Run rule-based verification checks and return a consolidated result."""

        issues = self._rule_based_checks(files, validated_config, solver)

        critical = [i for i in issues if i.severity == "critical"]
        warnings  = [i for i in issues if i.severity == "warning"]

        passed = len(critical) == 0
        parts = []
        if critical:
            parts.append(f"{len(critical)} critical issue(s)")
        if warnings:
            parts.append(f"{len(warnings)} warning(s)")
        summary = (
            "Verification passed — no critical issues."
            if passed
            else "Verification FAILED: " + ", ".join(parts) + "."
        )

        result = VerificationResult(passed=passed, issues=issues, summary=summary)

        # ── Print verification output with *** banner ──────────────────────
        print("\n" + "*" * 70)
        print(f"[VERIFIER RULE-BASED REPORT] solver={solver}  passed={passed}")
        print("*" * 70)
        if not issues:
            print("  ✅ No issues found.")
        for issue in issues:
            icon = "🔴" if issue.severity == "critical" else "⚠️ " if issue.severity == "warning" else "ℹ️ "
            print(f"  {icon} [{issue.category}] {issue.message}")
            if issue.fix_suggestion:
                print(f"      → {issue.fix_suggestion}")
        print("*" * 70 + "\n")

        return result

    # ── Rule-based checks (free, instant) ─────────────────────────────────────

    def _rule_based_checks(
        self,
        files: dict[str, str],
        validated_config: dict[str, Any],
        solver: str,
    ) -> list[VerificationIssue]:
        issues: list[VerificationIssue] = []

        # ── 1. Solver vs heat transfer ─────────────────────────────────────
        heat_transfer = (
            validated_config.get("heat_transfer")
            or validated_config.get("enable_heat_transfer")
            or validated_config.get("physics", {}).get("heat_transfer")
            or validated_config.get("physics", {}).get("enable_heat_transfer")
        )
        if heat_transfer and solver not in _HEAT_TRANSFER_SOLVERS:
            # Pick the correct replacement from ALLOWED_SOLVERS
            is_transient = validated_config.get("time_stepping", "steady") == "transient"
            suggested = "rhoPimpleFoam" if is_transient else "rhoSimpleFoam"
            issues.append(
                VerificationIssue(
                    severity="critical",
                    category="heat_transfer_solver",
                    message=(
                        f"Heat transfer is enabled but solver '{solver}' does not solve "
                        "the energy equation.  Temperature (T) will not be computed."
                    ),
                    fix_suggestion=(
                        f"Use '{suggested}' — it solves the energy equation and is in "
                        "the allowed solver list."
                    ),
                )
            )

        # ── 2. Solver vs time scheme ───────────────────────────────────────
        time_stepping = validated_config.get("time_stepping", "steady")
        if time_stepping == "transient" and solver in _STEADY_SOLVERS:
            suggested = "rhoPimpleFoam" if heat_transfer else "pimpleFoam"
            issues.append(
                VerificationIssue(
                    severity="critical",
                    category="solver_time_mismatch",
                    message=(
                        f"Time scheme is 'transient' but '{solver}' is a steady-state "
                        "solver."
                    ),
                    fix_suggestion=f"Use '{suggested}' for transient flow.",
                )
            )

        # ── 3. Pressure field: p vs p_rgh ─────────────────────────────────
        if solver in P_RGH_SOLVERS and "0/p_rgh" not in files:
            issues.append(
                VerificationIssue(
                    severity="critical",
                    category="missing_p_rgh",
                    message=f"'{solver}' requires '0/p_rgh' but it is missing.",
                    fix_suggestion="Generate '0/p_rgh' instead of '0/p'.",
                )
            )
        if solver in P_SOLVERS and "0/p" not in files and "0/p_rgh" in files:
            issues.append(
                VerificationIssue(
                    severity="critical",
                    category="wrong_pressure_field",
                    message=f"'{solver}' requires '0/p' but only '0/p_rgh' was generated.",
                    fix_suggestion="Rename '0/p_rgh' to '0/p' and fix dimensions to [0 2 -2 0 0 0 0].",
                )
            )

        # ── 4. Required system files ───────────────────────────────────────
        for sf in ("system/controlDict", "system/fvSchemes", "system/fvSolution"):
            if sf not in files:
                issues.append(
                    VerificationIssue(
                        severity="critical",
                        category="missing_system_dicts",
                        message=f"Required file '{sf}' is missing.",
                        fix_suggestion=f"Generate '{sf}' with settings appropriate for {solver}.",
                    )
                )

        # ── 5. Required constant files ─────────────────────────────────────
        if solver in THERMO_SOLVERS and "constant/thermophysicalProperties" not in files:
            issues.append(
                VerificationIssue(
                    severity="critical",
                    category="missing_constant_dicts",
                    message=f"'{solver}' requires 'constant/thermophysicalProperties' but it is missing.",
                    fix_suggestion="Generate 'constant/thermophysicalProperties' with thermoType, mixture, etc.",
                )
            )
        if solver in GRAVITY_SOLVERS and "constant/g" not in files:
            issues.append(
                VerificationIssue(
                    severity="critical",
                    category="missing_gravity",
                    message=f"'{solver}' (VOF) always requires 'constant/g' but it is missing.",
                    fix_suggestion="Generate 'constant/g' with dimensions [0 1 -2 0 0 0 0].",
                )
            )
        # turbulenceProperties is required by ALL solvers — OpenFOAM crashes at
        # startup if it is absent, regardless of the turbulence model chosen.
        if "constant/turbulenceProperties" not in files:
            issues.append(
                VerificationIssue(
                    severity="critical",
                    category="missing_constant_dicts",
                    message="'constant/turbulenceProperties' is missing.",
                    fix_suggestion=(
                        "Generate 'constant/turbulenceProperties' with "
                        "'simulationType RAS;' (or LES/laminar) and the model block."
                    ),
                )
            )

        # ── 5b. rho solver entry for compressible single-phase solvers ────────
        # rhoPimpleFoam and rhoSimpleFoam call rhoEqn.solve() which looks up
        # fvSolution/solvers/rho.  Missing this entry causes immediate crash.
        _RHO_EQN_SOLVERS_V = {"rhoPimpleFoam", "rhoSimpleFoam"}
        if solver in _RHO_EQN_SOLVERS_V:
            _fvs_content = files.get("system/fvSolution", "")
            _has_rho = bool(
                re.search(r'\brho\s*\{', _fvs_content)
                or re.search(r'"[^"]*\brho\b[^"]*"\s*\{', _fvs_content)
            )
            if _fvs_content and not _has_rho:
                issues.append(VerificationIssue(
                    severity="critical",
                    category="missing_rho_solver",
                    message=(
                        f"'{solver}' solves rhoEqn but fvSolution/solvers has no 'rho' entry. "
                        "This causes: \"Entry 'rho' not found in dictionary system/fvSolution/solvers\"."
                    ),
                    fix_suggestion=(
                        "Add to fvSolution/solvers: "
                        "rho { solver diagonal; tolerance 1e-12; relTol 0; }  "
                        "rhoFinal { $rho; relTol 0; }"
                    ),
                ))

        # ── 5d. thermophysicalProperties — thermoType key validation ──────────
        # In OpenFOAM 2406 the thermoType dict MUST use the key 'thermo', not
        # 'thermodynamics'.  The LLM confuses it with the mixture sub-dict name.
        # This check fires as a warning so we don't block submission; the
        # deterministic auto-fix in validate_generated_files already corrects it.
        _tp = files.get("constant/thermophysicalProperties", "")
        if _tp and solver in THERMO_SOLVERS:
            # 'thermodynamics <word>;' inside thermoType — wrong key
            if re.search(
                r'\bthermodynamics\s+(hConst|eConst|janaf|hTabular|eTabular|hPolynomial|ePolynomial)\s*;',
                _tp,
            ):
                issues.append(VerificationIssue(
                    severity="critical",
                    category="thermoType_key_wrong",
                    message=(
                        "constant/thermophysicalProperties uses 'thermodynamics <model>' "
                        "inside thermoType{} — OpenFOAM 2406 requires 'thermo <model>' "
                        "(the 'thermodynamics' keyword is only valid as a sub-dict "
                        "inside mixture{})."
                    ),
                    fix_suggestion=(
                        "Replace 'thermodynamics  hConst;' (or janaf, eConst, …) "
                        "with 'thermo  hConst;' inside the thermoType{} block."
                    ),
                ))

        # ── 6. Required field files ────────────────────────────────────────
        required_fields = ["0/U"]
        required_fields.append("0/p_rgh" if solver in P_RGH_SOLVERS else "0/p")
        if solver in ENERGY_SOLVERS or heat_transfer:
            required_fields.append("0/T")

        turb_model = (
            validated_config.get("turbulence_model")
            or validated_config.get("physics", {}).get("turbulence_model", "")
            or ""
        )
        if turb_model and turb_model not in ("laminar", "none"):
            required_fields += ["0/k", "0/nut"]
            if "kOmega" in turb_model or "SST" in turb_model:
                required_fields.append("0/omega")
            elif "kEpsilon" in turb_model or "Epsilon" in turb_model:
                required_fields.append("0/epsilon")
            if solver in ENERGY_SOLVERS:
                required_fields.append("0/alphat")

        for rf in required_fields:
            if rf not in files:
                issues.append(
                    VerificationIssue(
                        severity="critical",
                        category="missing_fields",
                        message=f"Required field '{rf}' is missing.",
                        fix_suggestion=f"Generate '{rf}' with boundary conditions for all patches.",
                    )
                )

        # ── 7. Patch coverage in 0/* files ────────────────────────────────
        patch_names: list[str] = []
        bcs = validated_config.get("boundary_conditions", {})
        if isinstance(bcs, dict):
            patch_names = list(bcs.keys())

        for path, content in files.items():
            if not path.startswith("0/") or "boundaryField" not in content:
                continue
            for patch in patch_names:
                if patch not in content:
                    issues.append(
                        VerificationIssue(
                            severity="critical",
                            category="patch_coverage",
                            message=f"Patch '{patch}' is missing from '{path}'.",
                            fix_suggestion=f"Add a boundary condition entry for '{patch}' in '{path}'.",
                        )
                    )

            # ── 9. Algorithm block ────────────────────────────────────────
            fv_sol = files.get("system/fvSolution", "")
            if fv_sol:
                if solver in ("simpleFoam", "rhoSimpleFoam") and "SIMPLE" not in fv_sol:
                    issues.append(VerificationIssue(
                        severity="warning", category="missing_algorithm_block",
                        message=f"{solver} requires a SIMPLE block in fvSolution.",
                        fix_suggestion="Add 'SIMPLE { ... }' to system/fvSolution.",
                    ))
                elif solver in ("pimpleFoam", "rhoPimpleFoam") and "PIMPLE" not in fv_sol:
                    issues.append(VerificationIssue(
                        severity="warning", category="missing_algorithm_block",
                        message=f"{solver} requires a PIMPLE block in fvSolution.",
                        fix_suggestion="Add 'PIMPLE { ... }' to system/fvSolution.",
                    ))
                elif solver == "icoFoam" and "PISO" not in fv_sol:
                    issues.append(VerificationIssue(
                        severity="warning", category="missing_algorithm_block",
                        message="icoFoam requires a PISO block in fvSolution.",
                        fix_suggestion="Add 'PISO { ... }' to system/fvSolution.",
                    ))
                elif solver in P_RGH_SOLVERS and "PIMPLE" not in fv_sol:
                    issues.append(VerificationIssue(
                        severity="warning", category="missing_algorithm_block",
                        message=f"{solver} (VOF) requires a PIMPLE block in fvSolution.",
                        fix_suggestion="Add 'PIMPLE { ... }' to system/fvSolution.",
                    ))

        return issues
