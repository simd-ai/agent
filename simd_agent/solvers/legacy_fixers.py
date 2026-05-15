"""Legacy regex-based fixers for LLM-emitted OpenFOAM files.

These functions correct common LLM output mistakes against known
OpenFOAM 2406 pitfalls (wrong dimensions, wrong key names, missing
clamps, etc.).  Most pre-date the Phase-4 deterministic-rendering
architecture and are now defensive-only — the deterministic renderer
already emits the right thing in most cases, but the fixers remain as
a safety net for any path that still goes through the LLM.

Hoisted out of ``SolverPlugin`` into a standalone module because:

  * They don't customize per subclass — same regex logic applies whether
    the plugin is rhoSimpleFoam, pimpleFoam, or any other.
  * They use at most 2 ``self.*`` attributes; making them free functions
    with explicit keyword params documents the inputs.
  * Discoverability — when looking for "the GAMG coarsest fix", you can
    open one file rather than scrolling through 2 500 LOC of base.py.

Convention: every function takes ``files`` and ``issues`` as positional
args (mutated in place), accepts plugin-specific values as keyword args.
Returns the (same) ``files`` dict for ergonomic chaining.

Used via thin wrapper methods on ``SolverPlugin`` so existing plugin
``validate()`` calls (``self._fix_X(files, issues)``) keep working
unchanged.  New code should call the free functions directly.
"""

from __future__ import annotations

import re
from typing import Any

from simd_agent.solvers.base import ValidationIssue


# ────────────────────────────────────────────────────────────────────────────
# Brace balance — runs first, every other fixer depends on balanced syntax
# ────────────────────────────────────────────────────────────────────────────


def balance_braces(content: str) -> str:
    """Balance curly braces in an OpenFOAM dictionary file.

    Handles the two most common LLM brace errors:

      1. Double-close — two consecutive ``}``-only lines at the same
         indentation with only blank lines between them (Pass 1).
      2. Missing ``}`` at EOF — appended before the footer comment.

    Depth-based fallback (Pass 2) catches any remaining imbalance.
    """
    # --- Count braces outside comments ---
    stripped = re.sub(r"//[^\n]*", "", content)
    stripped = re.sub(r"/\*.*?\*/", "", stripped, flags=re.DOTALL)
    opens = stripped.count("{")
    closes = stripped.count("}")

    if opens == closes:
        return content

    if closes > opens:
        excess = closes - opens
        lines = content.split("\n")

        # Pass 1: indentation-based double-close detection
        to_remove: set[int] = set()
        removed = 0
        in_block_comment = False
        prev_brace: tuple[int, int] | None = None

        for i, line in enumerate(lines):
            if in_block_comment:
                if "*/" in line:
                    in_block_comment = False
                continue

            analysis = re.sub(r"//.*$", "", line)
            analysis = re.sub(r"/\*.*?\*/", "", analysis)
            if "/*" in analysis:
                analysis = analysis[: analysis.find("/*")]
                in_block_comment = True

            stripped_part = analysis.strip()

            if stripped_part == "":
                continue

            if stripped_part == "}":
                indent = len(line) - len(line.lstrip())
                if prev_brace is not None:
                    _, prev_indent = prev_brace
                    if indent == prev_indent and removed < excess:
                        to_remove.add(i)
                        removed += 1
                        continue
                prev_brace = (i, indent)
            else:
                prev_brace = None

        if to_remove:
            lines = [
                l for idx, l in enumerate(lines) if idx not in to_remove
            ]

        # Pass 2: depth-based fallback
        remaining = excess - removed
        if remaining > 0:
            result: list[str] = []
            depth = 0
            in_block_comment = False
            removed2 = 0

            for line in lines:
                analysis = line
                if in_block_comment:
                    end_idx = analysis.find("*/")
                    if end_idx >= 0:
                        analysis = analysis[end_idx + 2 :]
                        in_block_comment = False
                    else:
                        result.append(line)
                        continue

                analysis = re.sub(r"//.*$", "", analysis)
                analysis = re.sub(r"/\*.*?\*/", "", analysis)
                if "/*" in analysis:
                    analysis = analysis[: analysis.find("/*")]
                    in_block_comment = True

                line_opens = analysis.count("{")
                line_closes = analysis.count("}")
                new_depth = depth + line_opens - line_closes

                if (
                    new_depth < 0
                    and removed2 < remaining
                    and analysis.strip() == "}"
                ):
                    removed2 += 1
                    depth = depth + line_opens
                    continue

                depth = new_depth
                result.append(line)

            return "\n".join(result)

        return "\n".join(lines)

    # Missing closing braces — append before the standard footer comment
    missing = opens - closes
    closing = "}\n" * missing
    footer_re = re.compile(r"\n(// \*{10,}.*?)\s*$", re.DOTALL)
    m = footer_re.search(content)
    if m:
        return content[: m.start()] + "\n" + closing + content[m.start() :]
    return content.rstrip() + "\n" + closing


def fix_brace_balance(
    files: dict[str, str],
    issues: list[ValidationIssue],
) -> dict[str, str]:
    """Apply ``balance_braces`` to every file and emit warnings."""
    for fpath in list(files.keys()):
        content = files[fpath]
        fixed = balance_braces(content)
        if fixed != content:
            files[fpath] = fixed
            issues.append(
                ValidationIssue(
                    "warning",
                    fpath,
                    "Fixed mismatched curly braces — LLM generated "
                    "unbalanced dictionary syntax.",
                    fix="Balanced { } in OpenFOAM dictionary",
                )
            )
    return files


# ────────────────────────────────────────────────────────────────────────────
# controlDict / pressure / thermo
# ────────────────────────────────────────────────────────────────────────────


def fix_controldict_solver(
    files: dict[str, str],
    issues: list[ValidationIssue],
    *,
    solver_name: str,
) -> dict[str, str]:
    """Ensure ``controlDict`` declares ``application <solver_name>``."""
    control_dict = files.get("system/controlDict", "")
    if not control_dict:
        return files
    app_match = re.search(r"application\s+(\w+)\s*;", control_dict)
    if app_match and app_match.group(1) != solver_name:
        issues.append(
            ValidationIssue(
                "warning",
                "system/controlDict",
                f"LLM wrote 'application {app_match.group(1)}' but solver is "
                f"'{solver_name}'. Correcting.",
                fix=f"application     {solver_name};",
            )
        )
        files["system/controlDict"] = re.sub(
            r"application\s+\w+\s*;",
            f"application     {solver_name};",
            control_dict,
        )
    return files


def fix_pressure_field(
    files: dict[str, str],
    issues: list[ValidationIssue],
    *,
    solver_name: str,
    pressure_field: str,
) -> dict[str, str]:
    """Ensure the correct pressure field (``p`` vs ``p_rgh``) is present."""
    if pressure_field == "p":
        if "0/p_rgh" in files and "0/p" not in files:
            issues.append(
                ValidationIssue(
                    "error",
                    "0/p_rgh",
                    f"'{solver_name}' requires 0/p, not 0/p_rgh. Renaming.",
                    fix="Renamed 0/p_rgh -> 0/p",
                )
            )
            content = files.pop("0/p_rgh")
            content = content.replace("object      p_rgh;", "object      p;")
            content = content.replace("object p_rgh;", "object p;")
            files["0/p"] = content
        if "0/p_rgh" in files and "0/p" in files:
            issues.append(
                ValidationIssue(
                    "warning",
                    "0/p_rgh",
                    "Both 0/p and 0/p_rgh exist. Removing 0/p_rgh.",
                )
            )
            del files["0/p_rgh"]

    elif pressure_field == "p_rgh":
        if "0/p_rgh" in files and "0/p" not in files:
            content = (
                files["0/p_rgh"]
                .replace("object      p_rgh;", "object      p;")
                .replace("object p_rgh;", "object p;")
            )
            files["0/p"] = content
            issues.append(
                ValidationIssue(
                    "warning",
                    "0/p",
                    f"'{solver_name}' needs both 0/p_rgh and 0/p. Synthesised 0/p.",
                )
            )
        elif "0/p" in files and "0/p_rgh" not in files:
            content = (
                files["0/p"]
                .replace("object      p;", "object      p_rgh;")
                .replace("object p;", "object p_rgh;")
            )
            files["0/p_rgh"] = content
            issues.append(
                ValidationIssue(
                    "warning",
                    "0/p_rgh",
                    f"'{solver_name}' needs both 0/p_rgh and 0/p. Synthesised 0/p_rgh.",
                )
            )
    return files


def fix_pressure_value(
    files: dict[str, str],
    issues: list[ValidationIssue],
    *,
    solver_name: str,
    is_compressible: bool,
) -> dict[str, str]:
    """Fix absolute pressure values in 0/p for incompressible solvers.

    Incompressible solvers use kinematic gauge pressure with dimensions
    ``[0 2 -2 0 0 0 0]`` and reference value 0.  The LLM frequently writes
    ``internalField uniform 101325`` (absolute Pa), which is nonsensical
    for kinematic pressure and causes SIGFPE in GAMG.
    """
    if is_compressible:
        return files

    p_content = files.get("0/p", "")
    if not p_content:
        return files

    changed = False

    if re.search(r"dimensions\s+\[\s*1\s+-1\s+-2\s+0\s+0\s+0\s+0\s*\]", p_content):
        p_content = re.sub(
            r"dimensions\s+\[\s*1\s+-1\s+-2\s+0\s+0\s+0\s+0\s*\]",
            "dimensions      [0 2 -2 0 0 0 0]",
            p_content,
        )
        changed = True
        issues.append(
            ValidationIssue(
                "warning",
                "0/p",
                f"'{solver_name}' is incompressible — pressure dimensions "
                f"must be [0 2 -2 0 0 0 0] (kinematic, m²/s²), "
                f"not [1 -1 -2 0 0 0 0] (Pa). Corrected.",
                fix="dimensions [0 2 -2 0 0 0 0];",
            )
        )

    m_int = re.search(
        r"(internalField\s+uniform\s+)([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
        p_content,
    )
    if m_int:
        try:
            p_val = float(m_int.group(2))
            if abs(p_val) > 1000:
                p_content = (
                    p_content[: m_int.start()]
                    + f"{m_int.group(1)}0"
                    + p_content[m_int.end() :]
                )
                changed = True
                issues.append(
                    ValidationIssue(
                        "warning",
                        "0/p",
                        f"'{solver_name}' is incompressible — internalField "
                        f"p={p_val} is absolute Pa, but kinematic gauge "
                        f"pressure should be 0. Corrected to prevent SIGFPE.",
                        fix="internalField uniform 0;",
                    )
                )
        except ValueError:
            pass

    for m_patch in re.finditer(r"(\w+)\s*\{([^}]*)\}", p_content, re.DOTALL):
        block = m_patch.group(2)
        if "fixedValue" not in block:
            continue
        m_val = re.search(
            r"(value\s+uniform\s+)([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
            block,
        )
        if not m_val:
            continue
        try:
            fv = float(m_val.group(2))
        except ValueError:
            continue
        if abs(fv) > 1000:
            abs_start = m_patch.start(2) + m_val.start()
            abs_end = m_patch.start(2) + m_val.end()
            p_content = (
                p_content[:abs_start]
                + f"{m_val.group(1)}0"
                + p_content[abs_end:]
            )
            changed = True
            issues.append(
                ValidationIssue(
                    "warning",
                    "0/p",
                    f"Outlet fixedValue p={fv} is absolute Pa — "
                    f"corrected to 0 for incompressible kinematic pressure.",
                    fix="value uniform 0;",
                )
            )

    if changed:
        files["0/p"] = p_content
    return files


def remove_unneeded_thermo(
    files: dict[str, str],
    issues: list[ValidationIssue],
    *,
    solver_name: str,
    supports_energy: bool,
) -> dict[str, str]:
    """Remove thermophysicalProperties and constant/g for non-energy solvers."""
    if supports_energy:
        return files
    for extra in ("constant/thermophysicalProperties", "constant/g"):
        if extra in files:
            issues.append(
                ValidationIssue(
                    "warning",
                    extra,
                    f"'{extra}' not needed for {solver_name}. Removing.",
                )
            )
            del files[extra]
    return files


def fix_thermo_type_key(
    files: dict[str, str],
    issues: list[ValidationIssue],
) -> dict[str, str]:
    """Fix ``thermodynamics`` → ``thermo`` inside thermoType blocks.

    OpenFOAM 2406 requires ``thermo`` as the key inside ``thermoType{}``.
    ``thermodynamics`` is only valid inside ``mixture{}``.
    """
    thermo_paths = [
        k
        for k in files
        if k == "constant/thermophysicalProperties"
        or k.startswith("constant/thermophysicalProperties.")
    ]
    for tp_path in thermo_paths:
        content = files[tp_path]
        fixed = re.sub(
            r"\bthermodynamics(\s+)(hConst|eConst|janaf|hTabular|eTabular|hPolynomial|ePolynomial|hIcoTabular|eIcoTabular)\s*;",
            r"thermo\1\2;",
            content,
        )
        if fixed != content:
            issues.append(
                ValidationIssue(
                    "warning",
                    tp_path,
                    "Auto-fixed: 'thermodynamics' -> 'thermo' in thermoType block.",
                    fix="thermodynamics -> thermo in thermoType block",
                )
            )
            files[tp_path] = fixed
    return files


# ────────────────────────────────────────────────────────────────────────────
# fvSchemes / fvSolution
# ────────────────────────────────────────────────────────────────────────────


def fix_fv_schemes_non_ortho(
    files: dict[str, str],
    issues: list[ValidationIssue],
    config: dict[str, Any],
) -> dict[str, str]:
    """Harden laplacian/snGrad schemes for non-orthogonal meshes.

    On meshes with non-orthogonality > 40°, ``Gauss linear corrected``
    creates an ill-conditioned pressure Laplacian that causes SIGFPE in
    GAMGSolver::scale.  Switch to ``limited corrected <factor>`` which
    blends corrected and uncorrected based on mesh quality.
    """
    fvs = files.get("system/fvSchemes", "")
    if not fvs:
        return files

    mesh = config.get("mesh", {}) or {}
    check_mesh = mesh.get("check_mesh") or mesh.get("checkMesh")
    from simd_agent.run.case_spec import _mesh_quality_decisions

    mq = _mesh_quality_decisions(check_mesh)
    non_ortho = mq.get("mesh_max_non_orthogonality") or 0.0
    tier = mq["mesh_quality_tier"]

    if tier == "good" and non_ortho < 40:
        return files

    if non_ortho >= 65 or tier == "poor":
        factor = "0.33"
    else:
        factor = "0.5"

    changed = False

    if re.search(
        r"laplacianSchemes[^}]*default\s+Gauss\s+linear\s+corrected\s*;",
        fvs, re.DOTALL,
    ):
        fvs = re.sub(
            r"(laplacianSchemes[^}]*default\s+)Gauss\s+linear\s+corrected(\s*;)",
            rf"\1Gauss linear limited corrected {factor}\2",
            fvs,
        )
        changed = True

    if re.search(
        r"snGradSchemes[^}]*default\s+corrected\s*;", fvs, re.DOTALL
    ):
        fvs = re.sub(
            r"(snGradSchemes[^}]*default\s+)corrected(\s*;)",
            rf"\1limited corrected {factor}\2",
            fvs,
        )
        changed = True

    if changed:
        files["system/fvSchemes"] = fvs
        issues.append(
            ValidationIssue(
                "warning",
                "system/fvSchemes",
                f"Hardened laplacian/snGrad → 'limited corrected {factor}' "
                f"(mesh tier='{tier}', non-ortho={non_ortho:.1f}°). "
                f"Pure 'corrected' causes SIGFPE on non-orthogonal meshes.",
                fix=f"limited corrected {factor}",
            )
        )

    return files


def fix_relaxation_factors(
    files: dict[str, str],
    issues: list[ValidationIssue],
    *,
    algorithm: str,
    pressure_field: str,
    max_equation_relaxation: float = 0.8,
) -> dict[str, str]:
    """Enforce safe relaxation factors in fvSolution.

    Two checks (SIMPLE algorithm only):
      1. Pressure field relaxation must exist in ``fields {}`` block.
      2. Equation relaxation values above ``max_equation_relaxation`` are
         clamped to 0.7 (industry-standard conservative default).
    """
    fv = files.get("system/fvSolution", "")
    if not fv:
        return files

    changed = False
    pf = pressure_field

    if algorithm == "SIMPLE":
        has_pressure_relax = bool(
            re.search(
                rf"fields\s*\{{[^}}]*\b{re.escape(pf)}\s+[\d.]+",
                fv,
                re.DOTALL,
            )
        )
        if not has_pressure_relax:
            m = re.search(r"(relaxationFactors\s*\{)", fv)
            if m:
                fv = (
                    fv[: m.end()]
                    + f"\n    fields      {{ {pf} 0.3; }}"
                    + fv[m.end() :]
                )
                changed = True
                issues.append(
                    ValidationIssue(
                        "warning",
                        "system/fvSolution",
                        f"Missing pressure field relaxation. "
                        f"Added: fields {{ {pf} 0.3; }}",
                        fix=f"fields {{ {pf} 0.3; }}",
                    )
                )

    eq_match = re.search(r"equations\s*\{", fv)
    if eq_match:
        start = eq_match.end()
        depth = 1
        pos = start
        while pos < len(fv) and depth > 0:
            if fv[pos] == "{":
                depth += 1
            elif fv[pos] == "}":
                depth -= 1
            pos += 1
        eq_end = pos
        eq_inner = fv[start : eq_end - 1]

        clamped_names: list[str] = []

        def _clamp(m: re.Match) -> str:
            name = m.group(1)
            val_str = m.group(3)
            try:
                val = float(val_str)
            except ValueError:
                return m.group(0)
            if val > max_equation_relaxation:
                clamped_names.append(f"{name}={val_str}")
                return f"{m.group(1)}{m.group(2)}0.7;"
            return m.group(0)

        new_eq = re.sub(
            r'((?:"[^"]*"|\w+))(\s+)([\d.]+)\s*;',
            _clamp,
            eq_inner,
        )
        if new_eq != eq_inner:
            fv = fv[:start] + new_eq + fv[eq_end - 1 :]
            changed = True
            issues.append(
                ValidationIssue(
                    "warning",
                    "system/fvSolution",
                    f"Equation relaxation too aggressive "
                    f"({', '.join(clamped_names)}). Clamped to 0.7.",
                    fix="Clamped equation relaxation to 0.7",
                )
            )

    if changed:
        files["system/fvSolution"] = fv
    return files


def fix_non_orthogonal_correctors(
    files: dict[str, str],
    issues: list[ValidationIssue],
    *,
    minimum: int = 1,
) -> dict[str, str]:
    """Ensure ``nNonOrthogonalCorrectors >= minimum`` in fvSolution."""
    fv = files.get("system/fvSolution", "")
    if not fv:
        return files

    m = re.search(r"nNonOrthogonalCorrectors\s+(\d+)\s*;", fv)
    if m:
        val = int(m.group(1))
        if val < minimum:
            fv = (
                fv[: m.start()]
                + f"nNonOrthogonalCorrectors {minimum};"
                + fv[m.end() :]
            )
            files["system/fvSolution"] = fv
            issues.append(
                ValidationIssue(
                    "warning",
                    "system/fvSolution",
                    f"nNonOrthogonalCorrectors {val} too low for "
                    f"general meshes. Set to {minimum}.",
                    fix=f"nNonOrthogonalCorrectors {minimum};",
                )
            )
    return files


def fix_gamg_coarsest_level(
    files: dict[str, str],
    issues: list[ValidationIssue],
    *,
    pressure_field: str,
) -> dict[str, str]:
    """Harden GAMG pressure solver against tet-mesh SIGFPE.

    Two-part fix on the pressure GAMG block:
      1. ``nCoarsestCells 20`` — matches the OF rhoSimpleFoam reference.
      2. ``coarsestLevelCorr`` with ``PBiCGStab + preconditioner none`` —
         pure Krylov, no diagonal inverse → immune to SIGFPE.
    """
    fv = files.get("system/fvSolution", "")
    if not fv:
        return files

    pf = pressure_field

    p_block_re = re.compile(
        rf"{re.escape(pf)}\s*\{{([^{{}}]*(?:\{{[^{{}}]*\}}[^{{}}]*)*)\}}",
        re.DOTALL,
    )
    m = p_block_re.search(fv)
    if not m:
        return files

    block_inner = m.group(1)

    if not re.search(r"solver\s+GAMG\s*;", block_inner):
        return files

    changed = False

    if "nCoarsestCells" not in block_inner:
        fv = re.sub(
            r"(solver\s+GAMG\s*;)",
            r"\1\n        nCoarsestCells  20;",
            fv,
            count=1,
        )
        changed = True
        issues.append(
            ValidationIssue(
                "warning",
                "system/fvSolution",
                "GAMG: added nCoarsestCells 20 to match the OF "
                "rhoSimpleFoam reference and keep coarse solves small.",
                fix="nCoarsestCells 20;",
            )
        )
        m = p_block_re.search(fv)
        if not m:
            files["system/fvSolution"] = fv
            return files
        block_inner = m.group(1)

    if "coarsestLevelCorr" in block_inner:
        if "symGaussSeidel" in block_inner or "smoothSolver" in block_inner:
            old_corr_re = re.compile(
                r"coarsestLevelCorr\s*\{[^}]*\}",
                re.DOTALL,
            )
            new_corr = (
                "coarsestLevelCorr\n"
                "        {\n"
                "            solver          PBiCGStab;\n"
                "            preconditioner  none;\n"
                "            tolerance       1e-9;\n"
                "            relTol          0;\n"
                "        }"
            )
            new_fv = old_corr_re.sub(new_corr, fv, count=1)
            if new_fv != fv:
                fv = new_fv
                changed = True
                issues.append(
                    ValidationIssue(
                        "warning",
                        "system/fvSolution",
                        "GAMG coarsestLevelCorr: replaced "
                        "smoothSolver+symGaussSeidel with "
                        "PBiCGStab+none (no diagonal inverse "
                        "→ immune to tet-mesh SIGFPE).",
                        fix="coarsestLevelCorr { solver PBiCGStab; "
                        "preconditioner none; }",
                    )
                )
    else:
        coarsest = (
            "\n        coarsestLevelCorr\n"
            "        {\n"
            "            solver          PBiCGStab;\n"
            "            preconditioner  none;\n"
            "            tolerance       1e-9;\n"
            "            relTol          0;\n"
            "        }"
        )
        insert_pos = m.start(1) + len(block_inner)
        fv = fv[:insert_pos] + coarsest + "\n    " + fv[insert_pos:]
        changed = True
        issues.append(
            ValidationIssue(
                "warning",
                "system/fvSolution",
                "GAMG: injected coarsestLevelCorr with "
                "PBiCGStab+none — prevents SIGFPE on tet meshes "
                "(no diagonal inverse at coarsest level).",
                fix="coarsestLevelCorr { solver PBiCGStab; "
                "preconditioner none; }",
            )
        )

    if changed:
        files["system/fvSolution"] = fv
    return files


def fix_residual_control_format(
    files: dict[str, str],
    issues: list[ValidationIssue],
    *,
    algorithm: str,
) -> dict[str, str]:
    """Convert plain-scalar residualControl entries to sub-dictionaries.

    PIMPLE requires each entry as ``p { tolerance 1e-4; relTol 0; }``;
    plain scalars crash with "Residual data for p must be specified as a
    dictionary".  SIMPLE accepts both formats, so this is a no-op there.
    """
    fv = files.get("system/fvSolution", "")
    if not fv or "residualControl" not in fv:
        return files
    if algorithm == "SIMPLE":
        return files

    _rc_re = re.compile(r'(residualControl\s*\{)(.*?)(\})', re.DOTALL)

    def _fix_rc_block(m: re.Match) -> str:
        header = m.group(1)
        body = m.group(2)
        closing = m.group(3)

        _scalar_re = re.compile(
            r'^(\s+)'
            r'("?\(?[\w|.*]+\)?"?)'
            r'\s+'
            r'([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)'
            r'\s*;',
            re.MULTILINE,
        )

        fixed_body = body
        found_any = False
        for sm in reversed(list(_scalar_re.finditer(body))):
            indent = sm.group(1)
            field_name = sm.group(2)
            value = sm.group(3)
            replacement = (
                f"{indent}{field_name} {{ tolerance {value}; relTol 0; }}"
            )
            fixed_body = (
                fixed_body[:sm.start()] + replacement + fixed_body[sm.end():]
            )
            found_any = True

        if found_any:
            issues.append(
                ValidationIssue(
                    "warning",
                    "system/fvSolution",
                    "residualControl entries were plain scalars — "
                    "converted to sub-dictionaries (OF2406 requirement).",
                    fix="{ tolerance X; relTol 0; }",
                )
            )

        return header + fixed_body + closing

    new_fv = _rc_re.sub(_fix_rc_block, fv)
    if new_fv != fv:
        files["system/fvSolution"] = new_fv
    return files
