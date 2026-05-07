# simd_agent/solvers/base.py
"""Abstract base class for OpenFOAM solver plugins.

Every solver that SIMD Agent can generate code for is a subclass of
``SolverPlugin``.  The plugin encapsulates:

  - **Identity**: name, algorithm, pressure field, capabilities
  - **Matching**: how well this solver fits a given simulation config
  - **File manifest**: which OpenFOAM case files to generate
  - **Prompts**: solver-specific LLM instructions for each file
  - **Validation**: post-generation checks and auto-fixes

To add a new solver, create ``simd_agent/solvers/<name>/__init__.py``
that exports a subclass.  The registry discovers it automatically.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Result types ─────────────────────────────────────────────────────────────


@dataclass
class MatchResult:
    """How well a solver matches a simulation config."""

    score: float  # 0.0 (no match) → 1.0 (perfect)
    reason: str = ""
    warnings: list[str] = field(default_factory=list)

    @property
    def matches(self) -> bool:
        return self.score > 0.0


@dataclass
class ValidationIssue:
    """A single issue found during post-generation validation."""

    severity: str  # "error" | "warning"
    file: str
    message: str
    fix: str | None = None

    def __repr__(self) -> str:
        return f"[{self.severity}] {self.file}: {self.message}"


@dataclass
class ValidationResult:
    """Result of solver-specific validation."""

    files: dict[str, str]  # possibly-fixed files
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)


# ── Turbulence field lookup ──────────────────────────────────────────────────

TURBULENCE_FIELDS: dict[str, list[str]] = {
    "laminar": [],
    "none": [],
    "kOmegaSST": ["k", "omega", "nut"],
    "kEpsilon": ["k", "epsilon", "nut"],
    "SpalartAllmaras": ["nuTilda", "nut"],
    "kOmega": ["k", "omega", "nut"],
}


# ── Abstract base ────────────────────────────────────────────────────────────


class SolverPlugin(ABC):
    """Base class for all OpenFOAM solver plugins.

    Subclasses MUST set the class-level attributes and implement all
    abstract methods.  Non-abstract methods provide sensible defaults
    that can be overridden when a solver has special needs.
    """

    # ── Identity (set these as class attributes in subclasses) ────────────

    name: str = ""  # e.g. "simpleFoam"
    algorithm: str = ""  # "SIMPLE" | "PIMPLE" | "PISO"
    pressure_field: str = "p"  # "p" | "p_rgh"
    is_transient: bool = False
    is_compressible: bool = False
    supports_energy: bool = False
    needs_gravity: bool = False
    is_multiphase: bool = False

    # ── Prompt directory (auto-set from subclass location) ────────────────

    @property
    def prompts_dir(self) -> Path:
        """Directory containing this solver's prompt .md files.

        By default: ``<solver_package>/prompts/``
        Override if your prompts live elsewhere.
        """
        import inspect

        cls_file = inspect.getfile(type(self))
        return Path(cls_file).parent / "prompts"

    # ── Abstract methods (MUST implement) ─────────────────────────────────

    @abstractmethod
    def matches(self, config: dict[str, Any]) -> MatchResult:
        """Return a confidence score for how well this solver fits *config*.

        The registry calls ``matches()`` on every registered solver and
        picks the highest score.  The LLM-based SolverSelector is used
        as a tiebreaker or when scores are ambiguous.

        Args:
            config: The validated simulation config dict.

        Returns:
            MatchResult with score 0.0–1.0.
        """
        ...

    @abstractmethod
    def required_files(self, config: dict[str, Any]) -> list[str]:
        """Return the list of OpenFOAM case file paths to generate.

        Example: ``["system/controlDict", "system/fvSchemes", "0/U", "0/p"]``

        Args:
            config: The validated simulation config dict.
        """
        ...

    @abstractmethod
    def validate(
        self, files: dict[str, str], config: dict[str, Any]
    ) -> ValidationResult:
        """Run solver-specific validation and auto-fixes on generated files.

        Called after the LLM generates all files but before packaging.
        May modify file contents (e.g. fix wrong solver name in controlDict).

        Args:
            files: Generated file contents keyed by path.
            config: The validated simulation config dict.

        Returns:
            ValidationResult with possibly-fixed files and issues list.
        """
        ...

    # ── Full validation orchestrator ──────────────────────────────────────

    def validate_full(
        self, files: dict[str, str], config: dict[str, Any]
    ) -> ValidationResult:
        """Single entry point for post-generation validation.

        Runs the plugin's own ``validate()`` (which applies reusable
        base-class helpers + any solver-specific fixes), then delegates to
        the shared monolithic validator for the battle-tested SIGFPE/crash
        checks that have not yet been migrated into plugin helpers.

        This is the only method the orchestrator should call.  Subclasses
        override ``validate()``; this method composes them with the shared
        validation pipeline.
        """
        # 0. Fix brace balance FIRST — other validators parse blocks with
        #    regex that assumes balanced braces.  An extra '}' from the LLM
        #    will break every downstream fixer.
        pre_issues: list[ValidationIssue] = []
        files = self._fix_brace_balance(files, pre_issues)

        # 1. Plugin-specific validation (uses base class helpers).
        plugin_result = self.validate(files, config)
        fixed = plugin_result.files
        issues = list(pre_issues) + list(plugin_result.issues)

        # 2. Shared legacy validator — lazy import to avoid circular deps.
        try:
            from simd_agent.run.genai_codegen import (
                validate_generated_files as _legacy_validate,
                ValidationIssue as _LegacyIssue,
            )
        except ImportError:
            _legacy_validate = None
            _LegacyIssue = None

        if _legacy_validate is not None:
            fixed, legacy_issues = _legacy_validate(fixed, self.name, config)
            for li in legacy_issues:
                issues.append(
                    ValidationIssue(
                        severity=getattr(li, "severity", "warning"),
                        file=getattr(li, "file", ""),
                        message=getattr(li, "message", str(li)),
                        fix=getattr(li, "fix", None),
                    )
                )

        return ValidationResult(files=fixed, issues=issues)

    # ── Prompt loading (override for custom prompt strategies) ────────────

    def system_prompt(self) -> str:
        """Load the solver identity prompt (_solver.md or solver.md).

        This is included in the shared context for every LLM call.
        """
        solver_md = self.prompts_dir / "_solver.md"
        if solver_md.exists():
            return solver_md.read_text(encoding="utf-8")
        # Fallback: single monolithic file
        legacy = self.prompts_dir.parent / f"{self.name}.md"
        return legacy.read_text(encoding="utf-8") if legacy.exists() else ""

    def prompt_for_file(self, file_path: str) -> str:
        """Load the solver-specific prompt doc for a single case file.

        Maps OpenFOAM paths to prompt doc paths:
          - ``system/fvSchemes`` → ``prompts/system/fvSchemes.md``
          - ``0/U``             → ``prompts/fields/U.md``
          - ``constant/g``      → ``prompts/constant/g.md``

        Returns empty string if no solver-specific doc exists.
        """
        doc_relpath = self._file_doc_relpath(file_path)
        doc_path = self.prompts_dir / doc_relpath
        if doc_path.exists():
            return doc_path.read_text(encoding="utf-8")
        return ""

    # ── Turbulence fields ─────────────────────────────────────────────────

    def turbulence_fields(self, turb_model: str) -> list[str]:
        """Return 0/ field names for a given turbulence model.

        Override if your solver has non-standard turbulence field requirements.
        """
        return TURBULENCE_FIELDS.get(turb_model, [])

    # ── Common validation helpers (use in your validate()) ────────────────

    def _fix_brace_balance(
        self,
        files: dict[str, str],
        issues: list[ValidationIssue],
    ) -> dict[str, str]:
        """Fix mismatched curly braces in OpenFOAM dictionary files.

        LLMs sometimes generate an extra ``}`` after a sub-dictionary,
        prematurely closing the parent block.  OpenFOAM reports::

            FOAM FATAL IO ERROR: Unexpected '}' while reading dictionary entry

        This validator removes lone ``}`` lines that push brace depth
        negative, and appends missing ``}`` at EOF when depth stays positive.
        Runs in ``validate_full()`` *before* all other validators since they
        depend on balanced syntax for regex parsing.
        """
        for fpath in list(files.keys()):
            content = files[fpath]
            fixed = self._balance_braces(content)
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

    @staticmethod
    def _balance_braces(content: str) -> str:
        """Balance curly braces in an OpenFOAM dictionary file.

        Handles the two most common LLM brace errors:

        1. **Double-close** — two consecutive ``}``-only lines at the same
           indentation with only blank lines between them.  The second one
           is the extra brace that prematurely closes the parent block.
           Detected by indentation analysis (Pass 1).

        2. **Missing ``}``** at EOF — appends before the footer comment.

        A depth-based fallback (Pass 2) catches any remaining imbalance
        that the indentation heuristic missed.
        """
        # --- Count braces outside comments ---
        stripped = re.sub(r"//[^\n]*", "", content)
        stripped = re.sub(r"/\*.*?\*/", "", stripped, flags=re.DOTALL)
        opens = stripped.count("{")
        closes = stripped.count("}")

        if opens == closes:
            return content  # already balanced

        if closes > opens:
            excess = closes - opens
            lines = content.split("\n")

            # ── Pass 1: indentation-based double-close detection ─────────
            # Two consecutive '}'-only lines at the SAME indent level (with
            # only blank lines between) is the classic LLM double-close.
            # In valid OpenFOAM dicts, consecutive '}' lines always have
            # DECREASING indent (closing nested blocks outward).
            to_remove: set[int] = set()
            removed = 0
            in_block_comment = False
            prev_brace: tuple[int, int] | None = None  # (line_idx, indent)

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
                    continue  # blank line — don't reset prev_brace

                if stripped_part == "}":
                    indent = len(line) - len(line.lstrip())
                    if prev_brace is not None:
                        _, prev_indent = prev_brace
                        if indent == prev_indent and removed < excess:
                            to_remove.add(i)
                            removed += 1
                            continue  # keep prev_brace for chained triples
                    prev_brace = (i, indent)
                else:
                    prev_brace = None

            if to_remove:
                lines = [
                    l for idx, l in enumerate(lines) if idx not in to_remove
                ]

            # ── Pass 2: depth-based fallback for remaining excess ────────
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

    def _fix_controldict_solver(
        self, files: dict[str, str], issues: list[ValidationIssue]
    ) -> dict[str, str]:
        """Ensure controlDict declares the correct solver application."""
        control_dict = files.get("system/controlDict", "")
        if not control_dict:
            return files
        app_match = re.search(r"application\s+(\w+)\s*;", control_dict)
        if app_match and app_match.group(1) != self.name:
            issues.append(
                ValidationIssue(
                    "warning",
                    "system/controlDict",
                    f"LLM wrote 'application {app_match.group(1)}' but solver is "
                    f"'{self.name}'. Correcting.",
                    fix=f"application     {self.name};",
                )
            )
            files["system/controlDict"] = re.sub(
                r"application\s+\w+\s*;",
                f"application     {self.name};",
                control_dict,
            )
        return files

    def _fix_pressure_field(
        self, files: dict[str, str], issues: list[ValidationIssue]
    ) -> dict[str, str]:
        """Ensure the correct pressure field (p vs p_rgh) is present."""
        if self.pressure_field == "p":
            # Solvers that use p should not have p_rgh
            if "0/p_rgh" in files and "0/p" not in files:
                issues.append(
                    ValidationIssue(
                        "error",
                        "0/p_rgh",
                        f"'{self.name}' requires 0/p, not 0/p_rgh. Renaming.",
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

        elif self.pressure_field == "p_rgh":
            # Buoyant solvers need BOTH p_rgh (solved) and p (calculated)
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
                        f"'{self.name}' needs both 0/p_rgh and 0/p. Synthesised 0/p.",
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
                        f"'{self.name}' needs both 0/p_rgh and 0/p. Synthesised 0/p_rgh.",
                    )
                )
        return files

    def _fix_pressure_value(
        self, files: dict[str, str], issues: list[ValidationIssue]
    ) -> dict[str, str]:
        """Fix absolute pressure values in 0/p for incompressible solvers.

        Incompressible solvers (simpleFoam, pimpleFoam) use kinematic gauge
        pressure with dimensions [0 2 -2 0 0 0 0] and values in m²/s².
        The reference pressure is 0 — only gradients matter.

        The LLM frequently writes ``internalField uniform 101325`` (absolute
        Pa), which is nonsensical for kinematic pressure and causes SIGFPE
        because GAMG sees a huge uniform field with near-zero gradients,
        creating ill-conditioned coarse levels.

        This validator:
        - Fixes internalField to 0 when value > 1000 (clearly absolute Pa)
        - Fixes outlet fixedValue to 0 when value > 1000
        - Also fixes wrong dimensions [1 -1 -2 0 0 0 0] → [0 2 -2 0 0 0 0]
        """
        if self.is_compressible:
            return files  # compressible solvers use absolute Pa — leave alone

        p_content = files.get("0/p", "")
        if not p_content:
            return files

        changed = False

        # Fix dimensions: Pa [1 -1 -2 0 0 0 0] → kinematic [0 2 -2 0 0 0 0]
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
                    f"'{self.name}' is incompressible — pressure dimensions "
                    f"must be [0 2 -2 0 0 0 0] (kinematic, m²/s²), "
                    f"not [1 -1 -2 0 0 0 0] (Pa). Corrected.",
                    fix="dimensions [0 2 -2 0 0 0 0];",
                )
            )

        # Fix internalField: absolute value → 0
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
                            f"'{self.name}' is incompressible — internalField "
                            f"p={p_val} is absolute Pa, but kinematic gauge "
                            f"pressure should be 0. Corrected to prevent SIGFPE.",
                            fix="internalField uniform 0;",
                        )
                    )
            except ValueError:
                pass

        # Fix outlet fixedValue: absolute value → 0
        # Find all fixedValue patches and fix values > 1000
        for m_patch in re.finditer(
            r"(\w+)\s*\{([^}]*)\}", p_content, re.DOTALL
        ):
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
                # Replace value in the original content
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

    def _remove_unneeded_thermo(
        self, files: dict[str, str], issues: list[ValidationIssue]
    ) -> dict[str, str]:
        """Remove thermophysicalProperties and g for non-energy solvers."""
        if not self.supports_energy:
            for extra in ["constant/thermophysicalProperties", "constant/g"]:
                if extra in files:
                    issues.append(
                        ValidationIssue(
                            "warning",
                            extra,
                            f"'{extra}' not needed for {self.name}. Removing.",
                        )
                    )
                    del files[extra]
        return files

    # ── Velocity classification ──────────────────────────────────────────

    @staticmethod
    def _extract_velocity_magnitude(config: dict[str, Any]) -> float:
        """Extract maximum inlet velocity magnitude from config (m/s).

        Scans all inlet patches and returns the highest velocity magnitude
        found.  Used by the deterministic builders to adapt numerical
        schemes and relaxation factors to the flow speed.
        """
        bcs = config.get("boundary_conditions", {}) or {}
        max_mag = 0.0

        for name, bc in bcs.items():
            if not isinstance(bc, dict):
                continue
            if "inlet" not in name.lower():
                continue

            raw = bc.get("velocity") or bc.get("U")
            mag = 0.0

            if isinstance(raw, dict):
                # Structured BC: {"type": ..., "value": [...]}
                v = raw.get("value")
                if isinstance(v, (list, tuple)) and len(v) >= 3:
                    mag = (float(v[0]) ** 2 + float(v[1]) ** 2 + float(v[2]) ** 2) ** 0.5
                else:
                    m = raw.get("magnitude") or raw.get("meanVelocity")
                    if m is not None:
                        try:
                            mag = abs(float(m))
                        except (TypeError, ValueError):
                            pass
            elif isinstance(raw, (list, tuple)) and len(raw) >= 3:
                mag = (float(raw[0]) ** 2 + float(raw[1]) ** 2 + float(raw[2]) ** 2) ** 0.5
            elif isinstance(raw, (int, float)):
                mag = abs(float(raw))

            # Also check velocity_magnitude / speed / U_mag keys
            if mag == 0.0:
                for key in ("velocity_magnitude", "speed", "U_mag"):
                    m_raw = bc.get(key)
                    if m_raw is not None:
                        if isinstance(m_raw, dict):
                            m_raw = m_raw.get("value", m_raw)
                        try:
                            mag = abs(float(m_raw))
                        except (TypeError, ValueError):
                            pass
                        if mag > 0:
                            break

            if mag > max_mag:
                max_mag = mag

        return max_mag

    @staticmethod
    def _flow_speed_tier(vel_mag: float) -> str:
        """Classify flow speed for numerical scheme selection.

        Thresholds are chosen to be conservative for liquid flows (water,
        cryogenics) where high cell Peclet numbers cause linearUpwind to
        overshoot at separation points and sharp gradients.

        Returns:
            ``"low"``      — |U| < 15 m/s:  linearUpwind safe, standard relaxation
            ``"moderate"``  — 15 ≤ |U| < 50: linearUpwind safe, tighter relaxation
            ``"high"``      — |U| ≥ 50:      upwind required, conservative relaxation
        """
        if vel_mag < 15.0:
            return "low"
        elif vel_mag < 50.0:
            return "moderate"
        return "high"

    # ── Deterministic file builders ─────────────────────────────────────

    @staticmethod
    def _get_turb_model_from_config(config: dict[str, Any]) -> str:
        """Extract the turbulence model name from a simulation config dict.

        Falls back to ``"kOmegaSST"`` when unspecified and turbulent.
        """
        physics = config.get("physics", {}) or {}
        flow_regime = (
            config.get("flow_regime")
            or physics.get("flow_regime", "turbulent")
        )
        if flow_regime == "laminar":
            return "laminar"
        return (
            config.get("turbulence_model")
            or physics.get("turbulence_model")
            or "kOmegaSST"
        )

    def _build_fv_schemes(self, config: dict[str, Any]) -> str:
        """Build ``system/fvSchemes`` deterministically from config.

        Produces a complete, balanced fvSchemes file whose content depends
        only on solver identity (compressible, transient, energy, …),
        turbulence model, and mesh quality tier.  No LLM involved.
        """
        turb_model = self._get_turb_model_from_config(config)

        # ddtSchemes
        ddt = "Euler" if self.is_transient else "steadyState"

        # Mesh quality → laplacian / snGrad blending
        mesh = (config.get("mesh", {}) or {})
        check_mesh = mesh.get("check_mesh") or mesh.get("checkMesh")
        from simd_agent.run.case_spec import _mesh_quality_decisions

        mq = _mesh_quality_decisions(check_mesh)
        non_ortho = mq.get("mesh_max_non_orthogonality") or 0.0
        tier = mq["mesh_quality_tier"]

        if tier == "good" and non_ortho < 40:
            lap_scheme = "Gauss linear corrected"
            sn_scheme = "corrected"
        elif non_ortho >= 65 or tier == "poor":
            lap_scheme = "Gauss linear limited corrected 0.33"
            sn_scheme = "limited corrected 0.33"
        else:
            lap_scheme = "Gauss linear limited corrected 0.5"
            sn_scheme = "limited corrected 0.5"

        # ── Velocity-aware scheme selection ─────────────────────────────
        vel_mag = self._extract_velocity_magnitude(config)
        speed_tier = self._flow_speed_tier(vel_mag)
        logger.info(
            f"[FV_SCHEMES] {self.name}: inlet velocity={vel_mag:.1f} m/s → "
            f"speed_tier='{speed_tier}'"
        )

        # divSchemes — built from solver capabilities + velocity tier
        div_lines: list[str] = []
        div_lines.append("    default         none;")

        if self.is_compressible:
            # Compressible: always upwind for stability
            div_lines.append("    div(phi,U)      bounded Gauss upwind;")
            if self.supports_energy:
                div_lines.append(
                    "    div(phi,h)      bounded Gauss upwind;"
                )
                div_lines.append(
                    "    div(phi,K)      bounded Gauss upwind;"
                )
            # div(phid,p) — pressure-dilatation term, only for rho* solvers.
            # buoyantSimpleFoam/buoyantPimpleFoam do NOT have this term;
            # including it causes: "cannot find scheme div(phid,p)"
            if not self.needs_gravity:
                div_lines.append("    div(phid,p)     Gauss upwind;")
        else:
            # Incompressible: velocity-dependent scheme
            # High velocity (≥50 m/s): upwind — linearUpwind overshoots at
            # separation points and sharp gradients, causing runaway divergence
            # (pressure oscillation 1e+16 → SIGFPE in 5-10 iterations).
            if speed_tier == "high":
                div_lines.append(
                    "    div(phi,U)      bounded Gauss upwind;"
                )
            else:
                div_lines.append(
                    "    div(phi,U)      bounded Gauss linearUpwind grad(U);"
                )

        # Turbulence div schemes — also velocity-aware
        turb_fields = (
            self.turbulence_fields(turb_model)
            if turb_model != "laminar"
            else []
        )
        # Only transported fields (not nut/alphat)
        transported = [
            f for f in turb_fields if f in ("k", "omega", "epsilon", "nuTilda")
        ]
        if transported:
            div_lines.append("")
            # High velocity: use upwind for turbulence too — limitedLinear can
            # allow negative k/omega at high Pe, causing bounding messages and
            # eventually SIGFPE in the turbulence model.
            turb_scheme = (
                "bounded Gauss upwind" if speed_tier == "high"
                else "bounded Gauss limitedLinear 1"
            )
            for f in transported:
                div_lines.append(
                    f"    div(phi,{f})    {turb_scheme};"
                )

        # Viscous stress tensor
        if self.is_compressible:
            div_lines.append("")
            div_lines.append(
                "    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;"
            )
        else:
            div_lines.append("")
            div_lines.append(
                "    div((nuEff*dev2(T(grad(U))))) Gauss linear;"
            )

        div_block = "\n".join(div_lines)

        # wallDist — only for turbulent flows
        wall_dist = ""
        if turb_model != "laminar":
            wall_dist = (
                "\nwallDist\n"
                "{\n"
                "    method          meshWave;\n"
                "}\n"
            )

        # fluxRequired — needed by the pressure-velocity coupling algorithm
        pf = self.pressure_field  # "p" or "p_rgh"
        flux_required = (
            "\nfluxRequired\n"
            "{\n"
            "    default         no;\n"
            f"    {pf};\n"
            "}\n"
        )

        return (
            "FoamFile\n"
            "{\n"
            "    version     2.0;\n"
            "    format      ascii;\n"
            "    class       dictionary;\n"
            "    object      fvSchemes;\n"
            "}\n"
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
            "\n"
            "ddtSchemes\n"
            "{\n"
            f"    default         {ddt};\n"
            "}\n"
            "\n"
            "gradSchemes\n"
            "{\n"
            "    default         Gauss linear;\n"
            "}\n"
            "\n"
            "divSchemes\n"
            "{\n"
            f"{div_block}\n"
            "}\n"
            "\n"
            "laplacianSchemes\n"
            "{\n"
            f"    default         {lap_scheme};\n"
            "}\n"
            "\n"
            "interpolationSchemes\n"
            "{\n"
            "    default         linear;\n"
            "}\n"
            "\n"
            "snGradSchemes\n"
            "{\n"
            f"    default         {sn_scheme};\n"
            "}\n"
            f"{flux_required}"
            f"{wall_dist}"
            "\n"
            "// ************************************************************************* //\n"
        )

    def _build_fv_solution(self, config: dict[str, Any]) -> str:
        """Build ``system/fvSolution`` deterministically from config.

        Generic builder used by solvers that do not override this method.
        ``simpleFoam`` provides its own specialised version.
        """
        turb_model = self._get_turb_model_from_config(config)
        mesh = (config.get("mesh", {}) or {})
        check_mesh = mesh.get("check_mesh") or mesh.get("checkMesh")
        from simd_agent.run.case_spec import _mesh_quality_decisions

        logger.info(f"[FV_SOLUTION] {self.name}: config['mesh'] keys = {list(mesh.keys()) if isinstance(mesh, dict) else 'N/A'}")
        logger.info(f"[FV_SOLUTION] {self.name}: check_mesh = {check_mesh}")

        mq = _mesh_quality_decisions(check_mesh)
        n_non_ortho = mq["n_non_ortho_correctors"]
        use_simplec = mq["use_simplec"]
        tier = mq["mesh_quality_tier"]
        non_ortho = mq.get("mesh_max_non_orthogonality") or 0.0

        is_simple = self.algorithm == "SIMPLE"
        pf = self.pressure_field  # "p" or "p_rgh"

        # ── Velocity-aware relaxation ────────────────────────────────────
        vel_mag = self._extract_velocity_magnitude(config)
        speed_tier = self._flow_speed_tier(vel_mag)
        logger.info(
            f"[FV_SOLUTION] {self.name}: inlet velocity={vel_mag:.1f} m/s → "
            f"speed_tier='{speed_tier}'"
        )

        # ── Pressure solver ──────────────────────────────────────────────
        use_gamg = tier == "good" or (tier == "moderate" and non_ortho < 50)
        logger.info(f"[FV_SOLUTION] {self.name}: tier='{tier}', non_ortho={non_ortho}° → pressure solver = {'GAMG' if use_gamg else 'PBiCGStab'}")
        if use_gamg:
            p_block = (
                f"    {pf}\n"
                "    {\n"
                "        solver          GAMG;\n"
                "        smoother        GaussSeidel;\n"
                "        nCoarsestCells  500;\n"
                "        tolerance       1e-06;\n"
                f"        relTol          {'0.1' if is_simple else '0.01'};\n"
                "        coarsestLevelCorr\n"
                "        {\n"
                "            solver          PBiCGStab;\n"
                "            preconditioner  DIC;\n"
                "            tolerance       1e-9;\n"
                "            relTol          0;\n"
                "        }\n"
                "    }\n"
            )
        else:
            p_block = (
                f"    {pf}\n"
                "    {\n"
                "        solver          PBiCGStab;\n"
                "        preconditioner  DIC;\n"
                "        tolerance       1e-06;\n"
                f"        relTol          {'0.1' if is_simple else '0.01'};\n"
                "    }\n"
            )

        # PIMPLE: pFinal block
        p_final = ""
        if not is_simple:
            p_final = (
                f"\n    {pf}Final\n"
                "    {\n"
                f"        ${pf};\n"
                "        relTol          0;\n"
                "    }\n"
            )

        # ── Equation solvers ─────────────────────────────────────────────
        eq_fields = ["U"]
        if turb_model in ("kOmegaSST", "kOmega"):
            eq_fields += ["k", "omega"]
        elif turb_model == "kEpsilon":
            eq_fields += ["k", "epsilon"]
        elif turb_model == "SpalartAllmaras":
            eq_fields += ["nuTilda"]

        if self.supports_energy:
            eq_fields.append("h")

        if len(eq_fields) == 1:
            eq_regex = eq_fields[0]
        else:
            eq_regex = f'"({"|".join(eq_fields)})"'

        eq_block = (
            f"\n    {eq_regex}\n"
            "    {\n"
            "        solver          smoothSolver;\n"
            "        smoother        symGaussSeidel;\n"
            "        tolerance       1e-05;\n"
            "        relTol          0.1;\n"
            "    }\n"
        )

        # PIMPLE: equation Final blocks — repeat settings explicitly
        eq_final = ""
        if not is_simple:
            if eq_regex.startswith('"'):
                inner = eq_regex[1:-1]  # e.g. (U|k|omega)
                final_regex = f'"{inner}Final"'
            else:
                final_regex = f"{eq_regex}Final"
            eq_final = (
                f"\n    {final_regex}\n"
                "    {\n"
                "        solver          smoothSolver;\n"
                "        smoother        symGaussSeidel;\n"
                "        tolerance       1e-06;\n"
                "        relTol          0;\n"
                "    }\n"
            )

        # Compressible: rho solver
        rho_block = ""
        if self.is_compressible:
            rho_block = (
                "\n    rho\n"
                "    {\n"
                "        solver          PCG;\n"
                "        preconditioner  DIC;\n"
                "        tolerance       1e-06;\n"
                "        relTol          0;\n"
                "    }\n"
            )

        # ── nNonOrthogonalCorrectors — bump for high velocity ────────────
        # High velocity creates large pressure source terms; extra
        # correctors improve accuracy on general meshes.
        if speed_tier == "high" and n_non_ortho < 2:
            n_non_ortho = 2

        # ── Algorithm block ──────────────────────────────────────────────
        if is_simple:
            # Disable SIMPLEC at high velocity — H1 correction amplifies
            # pressure oscillations that are already large.
            simplec_line = ""
            if use_simplec and tier != "unknown" and speed_tier != "high":
                simplec_line = "    consistent      yes;\n"

            # Residual control — plain scalars for SIMPLE
            res_lines = (
                f"        {pf:<16}1e-4;\n"
                f"        U               1e-4;\n"
            )
            turb_res_fields = [
                f for f in eq_fields if f not in ("U", "h")
            ]
            if turb_res_fields:
                if len(turb_res_fields) == 1:
                    res_lines += (
                        f"        {turb_res_fields[0]:<16}1e-3;\n"
                    )
                else:
                    turb_regex = f'"({"|".join(turb_res_fields)})"'
                    res_lines += (
                        f"        {turb_regex:<16}1e-3;\n"
                    )

            algo_block = (
                f"\n{self.algorithm}\n"
                "{\n"
                f"    nNonOrthogonalCorrectors {n_non_ortho};\n"
                f"{simplec_line}"
                "    pRefCell        0;\n"
                "    pRefValue       0;\n"
                "\n"
                "    residualControl\n"
                "    {\n"
                f"{res_lines}"
                "    }\n"
                "}\n"
            )

            # Relaxation factors for SIMPLE — velocity-aware
            # High velocity needs tighter damping to prevent pressure
            # oscillation cascade (p: 1e+16 → SIGFPE in <10 iterations).
            if speed_tier == "high":
                u_relax, p_relax, turb_relax, h_relax = 0.3, 0.2, 0.3, 0.3
            elif speed_tier == "moderate":
                u_relax, p_relax, turb_relax, h_relax = 0.5, 0.3, 0.5, 0.5
            else:
                u_relax, p_relax, turb_relax, h_relax = 0.7, 0.3, 0.5, 0.5

            relax_eq_lines = f"        U               {u_relax};\n"
            for f in eq_fields:
                if f == "U":
                    continue
                if f == "h":
                    relax_eq_lines += f"        h               {h_relax};\n"
                else:
                    relax_eq_lines += f"        {f:<16}{turb_relax};\n"

            relax_block = (
                "\nrelaxationFactors\n"
                "{\n"
                "    fields\n"
                "    {\n"
                f"        {pf:<16}{p_relax};\n"
                "    }\n"
                "    equations\n"
                "    {\n"
                f"{relax_eq_lines}"
                "    }\n"
                "}\n"
            )
        else:
            # PIMPLE algorithm block
            res_lines = (
                f"        {pf}   {{ tolerance 1e-4; relTol 0; }}\n"
                "        U   { tolerance 1e-4; relTol 0; }\n"
            )
            if self.supports_energy:
                res_lines += (
                    "        h   { tolerance 5e-3; relTol 0; }\n"
                )

            algo_block = (
                f"\n{self.algorithm}\n"
                "{\n"
                "    nOuterCorrectors    2;\n"
                "    nCorrectors         2;\n"
                f"    nNonOrthogonalCorrectors {n_non_ortho};\n"
                "    momentumPredictor   yes;\n"
                "\n"
                "    residualControl\n"
                "    {\n"
                f"{res_lines}"
                "    }\n"
                "}\n"
            )

            # Relaxation for PIMPLE with nOuterCorrectors > 1
            # Velocity-aware: tighter under-relaxation at high speed
            if speed_tier == "high":
                pimple_u_relax, pimple_catch_all = 0.3, 0.3
            elif speed_tier == "moderate":
                pimple_u_relax, pimple_catch_all = 0.5, 0.5
            else:
                pimple_u_relax, pimple_catch_all = 0.7, 0.7

            relax_block = (
                "\nrelaxationFactors\n"
                "{\n"
                f'    equations {{ U {pimple_u_relax}; ".*" {pimple_catch_all}; }}\n'
                "}\n"
            )

        return (
            "FoamFile\n"
            "{\n"
            "    version     2.0;\n"
            "    format      ascii;\n"
            "    class       dictionary;\n"
            "    object      fvSolution;\n"
            "}\n"
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
            "\n"
            "solvers\n"
            "{\n"
            f"{p_block}"
            f"{p_final}"
            f"{rho_block}"
            f"{eq_block}"
            f"{eq_final}"
            "}\n"
            f"{algo_block}"
            f"{relax_block}"
            "\n"
            "// ************************************************************************* //\n"
        )

    def _fix_fv_schemes_non_ortho(
        self,
        files: dict[str, str],
        issues: list[ValidationIssue],
        config: dict[str, Any],
    ) -> dict[str, str]:
        """Harden laplacian/snGrad schemes for non-orthogonal meshes.

        On meshes with non-orthogonality > 40°, ``Gauss linear corrected``
        creates an ill-conditioned pressure Laplacian that causes SIGFPE in
        GAMGSolver::scale (or PBiCGStab divergence).  Switch to
        ``limited corrected <factor>`` which blends corrected and uncorrected
        based on mesh quality — stable on poor meshes while preserving
        accuracy on good cells.

        When no checkMesh data is available (unknown tier), be conservative
        and apply ``limited corrected 0.5`` — it never hurts accuracy on
        good meshes, but prevents the crash on bad ones.
        """
        fvs = files.get("system/fvSchemes", "")
        if not fvs:
            return files

        # Extract mesh quality — delegate to case_spec helper
        mesh = (config.get("mesh", {}) or {})
        check_mesh = mesh.get("check_mesh") or mesh.get("checkMesh")
        from simd_agent.run.case_spec import _mesh_quality_decisions

        mq = _mesh_quality_decisions(check_mesh)
        non_ortho = mq.get("mesh_max_non_orthogonality") or 0.0
        tier = mq["mesh_quality_tier"]

        # Good meshes with low non-ortho — no intervention needed
        if tier == "good" and non_ortho < 40:
            return files

        # Determine the blending factor
        if non_ortho >= 65 or tier == "poor":
            factor = "0.33"
        else:
            # moderate, unknown, or non_ortho >= 40
            factor = "0.5"

        changed = False

        # Fix laplacianSchemes: "Gauss linear corrected" → "Gauss linear limited corrected <f>"
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

        # Fix snGradSchemes: "corrected" → "limited corrected <f>"
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

    def _ensure_gravity(
        self, files: dict[str, str], issues: list[ValidationIssue]
    ) -> dict[str, str]:
        """Ensure constant/g exists for solvers that need it."""
        if self.needs_gravity and "constant/g" not in files:
            issues.append(
                ValidationIssue(
                    "error",
                    "constant/g",
                    f"'{self.name}' requires constant/g. Adding default.",
                    fix="Added constant/g",
                )
            )
            files["constant/g"] = (
                "FoamFile\n{\n    version 2.0;\n    format ascii;\n"
                "    class uniformDimensionedVectorField;\n    object g;\n}\n"
                "dimensions [0 1 -2 0 0 0 0];\nvalue (0 -9.81 0);\n"
            )
        return files

    def _fix_thermo_type_key(
        self, files: dict[str, str], issues: list[ValidationIssue]
    ) -> dict[str, str]:
        """Fix 'thermodynamics' → 'thermo' inside thermoType blocks.

        OpenFOAM 2406 requires 'thermo' as the key inside thermoType{}.
        The LLM often writes 'thermodynamics' which is only valid inside mixture{}.
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

    def _fix_relaxation_factors(
        self,
        files: dict[str, str],
        issues: list[ValidationIssue],
        max_equation_relaxation: float = 0.8,
    ) -> dict[str, str]:
        """Enforce safe relaxation factors in fvSolution.

        Two checks (SIMPLE algorithm only):
        1. Pressure field relaxation must exist in ``fields {}`` block.
        2. Equation relaxation values above *max_equation_relaxation* are
           clamped to 0.7 (the industry-standard conservative default).

        This is a safety net — even when the prompt template is correct,
        the LLM may hallucinate aggressive values.  Commercial solvers like
        Ansys Fluent enforce similar guardrails.
        """
        fv = files.get("system/fvSolution", "")
        if not fv:
            return files

        changed = False
        pf = self.pressure_field  # "p" or "p_rgh"

        # --- 1. Ensure fields { <pf> 0.3; } exists for SIMPLE solvers ---
        if self.algorithm == "SIMPLE":
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

        # --- 2. Clamp equation relaxation > max → 0.7 ---
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
            eq_end = pos  # position after closing brace
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

    def _fix_non_orthogonal_correctors(
        self,
        files: dict[str, str],
        issues: list[ValidationIssue],
        minimum: int = 1,
    ) -> dict[str, str]:
        """Ensure nNonOrthogonalCorrectors >= *minimum* in fvSolution.

        With 0 correctors, any mesh non-orthogonality degrades pressure
        accuracy.  At least 1 corrector is needed for general meshes;
        compressible energy solvers often need 2.
        """
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

    def _fix_gamg_coarsest_level(
        self,
        files: dict[str, str],
        issues: list[ValidationIssue],
    ) -> dict[str, str]:
        """Harden GAMG pressure solver against tet-mesh SIGFPE.

        On unstructured tet meshes, GAMG agglomerates down to a tiny
        coarsest level where the matrix diagonal can have zeros.  Every
        smoother and preconditioner that computes D^{-1} (DIC, DILU,
        GaussSeidel, symGaussSeidel) will SIGFPE.

        Two-part fix applied to the pressure GAMG block:

        1. ``nCoarsestCells 500`` — prevents over-agglomeration so the
           coarsest matrix stays well-conditioned (default is 10).
        2. ``coarsestLevelCorr`` with ``PBiCGStab; preconditioner none``
           — pure Krylov iteration with no diagonal inverse, so even a
           degenerate coarsest-level matrix cannot cause SIGFPE.

        Also patches existing coarsestLevelCorr blocks that use
        ``smoothSolver + symGaussSeidel`` (our earlier fix that still
        divides by diagonal and therefore still crashes).
        """
        fv = files.get("system/fvSolution", "")
        if not fv:
            return files

        pf = self.pressure_field  # "p" or "p_rgh"

        # Match the pressure solver block (supports one level of nested
        # sub-blocks like coarsestLevelCorr {}).  [^{}] in the non-nested
        # segments ensures we don't accidentally consume a nested '{'.
        p_block_re = re.compile(
            rf"{re.escape(pf)}\s*\{{([^{{}}]*(?:\{{[^{{}}]*\}}[^{{}}]*)*)\}}",
            re.DOTALL,
        )
        m = p_block_re.search(fv)
        if not m:
            return files

        block_inner = m.group(1)

        # Only applies to GAMG
        if not re.search(r"solver\s+GAMG\s*;", block_inner):
            return files

        changed = False

        # --- 1. Ensure nCoarsestCells is set ---
        if "nCoarsestCells" not in block_inner:
            # Insert right after "solver GAMG;" for clean formatting
            fv = re.sub(
                r"(solver\s+GAMG\s*;)",
                r"\1\n        nCoarsestCells  500;",
                fv,
                count=1,
            )
            changed = True
            issues.append(
                ValidationIssue(
                    "warning",
                    "system/fvSolution",
                    "GAMG: added nCoarsestCells 500 to prevent "
                    "over-agglomeration on tet meshes.",
                    fix="nCoarsestCells 500;",
                )
            )
            # Re-match after insertion
            m = p_block_re.search(fv)
            if not m:
                files["system/fvSolution"] = fv
                return files
            block_inner = m.group(1)

        # --- 2. Fix or inject coarsestLevelCorr ---
        if "coarsestLevelCorr" in block_inner:
            # Patch existing block: replace smoother-based solvers with
            # PBiCGStab + no preconditioner (no diagonal inverse).
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
            # Inject new coarsestLevelCorr block
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

    def _fix_residual_control_format(
        self,
        files: dict[str, str],
        issues: list[ValidationIssue],
    ) -> dict[str, str]:
        """Convert plain-scalar residualControl entries to sub-dictionaries.

        PIMPLE's ``pimpleControl`` calls ``solutionControl::read(false)``
        which requires each residualControl entry to be a sub-dictionary:
        ``p { tolerance 1e-4; relTol 0; }``
        Plain scalars like ``p 1e-4;`` cause a fatal crash:
          "Residual data for p must be specified as a dictionary"

        SIMPLE's ``simpleControl`` calls ``solutionControl::read(true)``
        (absTolOnly) so plain scalars are valid — but sub-dictionaries
        also work, so we convert unconditionally for safety.
        """
        fv = files.get("system/fvSolution", "")
        if not fv or "residualControl" not in fv:
            return files
        # SIMPLE-based solvers accept both formats — only PIMPLE crashes on scalars.
        # For SIMPLE, skip this fix since scalar format is canonical.
        if self.algorithm == "SIMPLE":
            return files

        # Find the residualControl block(s) — there may be more than one
        # (SIMPLE + PIMPLE in some configs, though rare)
        # Pattern: residualControl { ... }
        _rc_re = re.compile(
            r'(residualControl\s*\{)(.*?)(\})',
            re.DOTALL,
        )

        def _fix_rc_block(m: re.Match) -> str:
            header = m.group(1)
            body = m.group(2)
            closing = m.group(3)

            # Check if any entry is already a sub-dict (has { })
            # If all entries are already dicts, skip
            # Pattern for scalar entry: field_name  scalar_value;
            # where field_name can be quoted regex like "(k|omega)"
            _scalar_re = re.compile(
                r'^(\s+)'                           # leading whitespace
                r'("?\(?[\w|.*]+\)?"?)'             # field name (may be quoted regex)
                r'\s+'
                r'([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)'  # numeric value
                r'\s*;',
                re.MULTILINE,
            )

            fixed_body = body
            found_any = False
            for sm in reversed(list(_scalar_re.finditer(body))):
                indent = sm.group(1)
                field_name = sm.group(2)
                value = sm.group(3)
                replacement = f"{indent}{field_name} {{ tolerance {value}; relTol 0; }}"
                fixed_body = fixed_body[:sm.start()] + replacement + fixed_body[sm.end():]
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

    # ── Internal helpers ──────────────────────────────────────────────────

    @staticmethod
    def _file_doc_relpath(file_path: str) -> str:
        """Map an OpenFOAM case file path to its prompt doc relative path.

        system/fvSchemes → system/fvSchemes.md
        0/U              → fields/U.md
        constant/g       → constant/g.md
        """
        if file_path.startswith("system/"):
            return f"system/{file_path.split('/', 1)[1]}.md"
        if file_path.startswith("constant/"):
            rel = file_path.split("/", 1)[1]
            if rel.startswith("thermophysicalProperties."):
                return "constant/thermophysicalProperties.md"
            return f"constant/{rel}.md"
        if file_path.startswith("0/"):
            rel = file_path.split("/", 1)[1]
            if rel.startswith("alpha."):
                return "fields/alpha.md"
            return f"fields/{rel}.md"
        return f"{file_path}.md"
