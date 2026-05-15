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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from simd_agent.solvers.contexts import FvBuildContext

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

    # Energy variable name as it appears in fvSchemes (``div(phi,h)`` vs
    # ``div(phi,e)``), fvSolution (the equation-solver regex group and
    # ``residualControl``), and the boundary-field file.  Must match the
    # ``energy`` choice in ``constant/thermophysicalProperties``:
    #   * ``"h"`` ↔ ``energy sensibleEnthalpy`` (transports enthalpy)
    #   * ``"e"`` ↔ ``energy sensibleInternalEnergy`` (transports internal energy)
    #
    # rhoSimpleFoam overrides this to ``"e"`` to match the OpenFOAM reference
    # tutorials — internal energy avoids the pressure-work source term and is
    # the preferred form for steady compressible.  Plugins that don't carry
    # an energy equation leave this as ``"h"`` (unused).
    energy_var: str = "h"

    # Pressure residual tolerance used in ``residualControl``.  Compressible
    # SIMPLE solvers (rhoSimpleFoam) loosen this to ``1e-3`` — pressure
    # rarely reaches ``1e-4`` in a finite number of outer iterations on a
    # compressible case, so the tighter value just made the run go to
    # ``endTime`` after the flow had already steadied.  The OF reference
    # tutorial uses ``1e-2``; ``1e-3`` is our middle ground.
    pressure_residual_tol: float = 1e-4

    # ── Turbulence requirements (overridable per plugin) ──────────────────
    # The plugin declares which RAS / LES models it supports and what to use
    # when the user / planner failed to pick one.  ``resolve_turbulence_spec``
    # consumes these to populate ``CaseSpec.turbulence_spec`` deterministically
    # — laminar is no longer the silent fallback when a turbulent solver is
    # selected (the failure mode that demoted rhoSimpleFoam to laminar and
    # caused the SIGFPE cascade).
    default_turbulence_model: str = "kOmegaSST"
    valid_turbulence_models: frozenset[str] = frozenset({
        "laminar",
        "kOmegaSST", "kOmega", "kEpsilon",
        "SpalartAllmaras",
        "LES",
    })

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

    def _get_turb_model_from_config(self, config: dict[str, Any]) -> str:
        """Extract the turbulence model name from a simulation config dict.

        Delegates to ``resolve_turbulence_spec`` so every helper that asks
        "what's the turbulence model for this case?" gets the same answer
        the renderer will use.  Falls back to this plugin's
        ``default_turbulence_model`` instead of a static "kOmegaSST".
        """
        from simd_agent.run.case_spec import resolve_turbulence_spec
        return resolve_turbulence_spec(self, config).model

    # ════════════════════════════════════════════════════════════════════════
    # Composable building blocks for fvSolution and fvSchemes
    # ════════════════════════════════════════════════════════════════════════
    # Each solver plugin assembles its own ``_build_fv_solution`` and
    # ``_build_fv_schemes`` by composing these helpers in the order that fits
    # its physics.  The helpers below are deliberately small, parameterised,
    # and free of solver-name branching — they read solver identity from
    # ``self`` attributes (algorithm, is_compressible, supports_energy, …)
    # and per-case context from arguments.
    #
    # The monolithic ``_build_fv_solution`` / ``_build_fv_schemes`` defined
    # later in this file remain as a legacy fallback; new plugins should
    # override both methods using these helpers and never call the monolith.

    # ── FoamFile boilerplate ──────────────────────────────────────────────

    @staticmethod
    def _foam_file_header(object_name: str) -> str:
        """Return the canonical FoamFile header for ``object_name``."""
        return (
            "FoamFile\n"
            "{\n"
            "    version     2.0;\n"
            "    format      ascii;\n"
            "    class       dictionary;\n"
            f"    object      {object_name};\n"
            "}\n"
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
            "\n"
        )

    @staticmethod
    def _foam_file_footer() -> str:
        """Return the canonical trailing comment used by every case file."""
        return "\n// ************************************************************************* //\n"

    # ── Per-build context ─────────────────────────────────────────────────

    def _fv_context(self, config: dict[str, Any]) -> "FvBuildContext":
        """Compute the shared per-build context for fvSolution / fvSchemes.

        Returns a frozen ``FvBuildContext`` — every renderer helper consumes
        this typed object via attribute access (``ctx.tier``, ``ctx.profile``
        etc.) instead of the old dict-of-Any indexing.  Phase 3 contract.
        """
        from simd_agent.run.case_spec import (
            _mesh_quality_decisions,
            _thermo_profile_from_config,
            resolve_regime_profile,
            resolve_turbulence_spec,
        )
        from simd_agent.solvers.contexts import FvBuildContext

        mesh = config.get("mesh", {}) or {}
        check_mesh = mesh.get("check_mesh") or mesh.get("checkMesh")
        mq = _mesh_quality_decisions(check_mesh)
        vel_mag = self._extract_velocity_magnitude(config)
        speed_tier = self._flow_speed_tier(vel_mag)
        profile = (
            _thermo_profile_from_config(config) if self.is_compressible else "gas"
        )
        turb_model = self._get_turb_model_from_config(config)
        phys = config.get("physics") or {}
        # Heat-transfer detection — explicit config flag only; rhoPimpleFoam
        # can run isothermal even though it supports energy.
        heat_transfer_active = bool(
            config.get("heat_transfer")
            or phys.get("heat_transfer")
            or phys.get("energy")
        )
        # BC temperatures — feed into the div(phi,h) scheme resolver.
        # BC pressures — feed into the pressure-ratio decision for div(phi,U).
        bc_temps: list[float] = []
        bc_pressures: list[float] = []
        for pbc in (config.get("boundary_conditions") or {}).values():
            if not isinstance(pbc, dict):
                continue
            t_entry = pbc.get("temperature") or pbc.get("T")
            t_val = (
                t_entry.get("value") or t_entry.get("uniform")
                if isinstance(t_entry, dict)
                else t_entry
            )
            try:
                bc_temps.append(float(t_val))
            except (TypeError, ValueError):
                pass
            p_entry = pbc.get("pressure") or pbc.get("p")
            p_val = (
                p_entry.get("value") or p_entry.get("uniform")
                if isinstance(p_entry, dict)
                else p_entry
            )
            try:
                pv = float(p_val)
                if pv > 0:
                    bc_pressures.append(pv)
            except (TypeError, ValueError):
                pass

        # Resolve the per-regime scheme bundle (laminar / RAS / LES) once.
        # The renderer helpers read attribute access against this object
        # instead of nested if/else over a string regime tag.  Algorithm
        # comes from the plugin (SIMPLE / PIMPLE / PISO); energy_var comes
        # from the plugin's class attribute.
        try:
            _turb_spec = resolve_turbulence_spec(self, config)
            sim_type = _turb_spec.simulation_type
            model_for_block = _turb_spec.model
        except Exception:
            # Defensive — if the resolver can't run (corrupt config), fall
            # back to RAS+kOmegaSST so the renderer still emits a working
            # case.  The actual fault will surface elsewhere as a lint issue.
            sim_type = "RAS"
            model_for_block = turb_model or "kOmegaSST"

        from typing import Literal as _Literal, cast as _cast
        _algo = _cast(
            _Literal["SIMPLE", "PIMPLE", "PISO"],
            (self.algorithm if self.algorithm in ("SIMPLE", "PIMPLE", "PISO")
             else "SIMPLE"),
        )
        regime_profile = resolve_regime_profile(
            simulation_type=sim_type,
            turb_model=model_for_block,
            algorithm=_algo,
            is_compressible=self.is_compressible,
            energy_var=self.energy_var,
        )

        return FvBuildContext(
            tier=mq["mesh_quality_tier"],
            non_ortho=mq.get("mesh_max_non_orthogonality") or 0.0,
            use_simplec=mq["use_simplec"],
            n_non_ortho=mq["n_non_ortho_correctors"],
            vel_mag=vel_mag,
            speed_tier=speed_tier,
            bc_temps=tuple(sorted(set(bc_temps))),
            bc_pressures=tuple(sorted(set(bc_pressures))),
            profile=profile,
            heat_transfer_active=heat_transfer_active,
            turb_model=turb_model,
            mesh_quality=mq,
            regime_profile=regime_profile,
        )

    # ── Equation field lookup ─────────────────────────────────────────────

    def _equation_fields(self, turb_model: str) -> list[str]:
        """Return the list of equation fields for the equation-solver regex.

        Always includes ``U``, the active turbulence transported fields, and
        ``h`` when the solver supports energy.  Order is preserved to keep
        the regex group readable.
        """
        eq_fields: list[str] = ["U"]
        if turb_model in ("kOmegaSST", "kOmega"):
            eq_fields += ["k", "omega"]
        elif turb_model == "kEpsilon":
            eq_fields += ["k", "epsilon"]
        elif turb_model == "SpalartAllmaras":
            eq_fields += ["nuTilda"]
        if self.supports_energy:
            eq_fields.append(self.energy_var)
        return eq_fields

    # ── Pressure solver block ─────────────────────────────────────────────

    def _build_pressure_solver_block(
        self,
        ctx: "FvBuildContext",
        is_simple: bool | None = None,
    ) -> tuple[str, str]:
        """Build the ``p`` (or ``p_rgh``) solver block and optional ``pFinal``.

        Returns ``(p_block, p_final_block)``.  ``p_final_block`` is empty
        for SIMPLE.

        Phase 2: the (GAMG vs PBiCGStab, coarsestLevelCorr settings,
        rhoPimpleFoam-isothermal special-case) decisions live entirely in
        ``resolve_pressure_solver_strategy``.  This helper is now a pure
        renderer that consumes the resolved strategy.  Replaces what used
        to be Check 7c (GAMG hardening) and Check 7e (isothermal rho*).
        """
        from simd_agent.run.case_spec import resolve_pressure_solver_strategy

        if is_simple is None:
            is_simple = self.algorithm == "SIMPLE"
        pf = self.pressure_field

        # Read heat-transfer flag from ctx (extracted from config by _fv_context).
        # Falls back to True for compressible-energy solvers if not set.
        heat = ctx.heat_transfer_active

        strategy = resolve_pressure_solver_strategy(
            solver_name=self.name,
            is_compressible=self.is_compressible,
            mesh_tier=ctx.tier,
            heat_transfer_active=heat,
        )

        rel_tol_str = f"{strategy.rel_tol:g}"

        if strategy.top_level == "GAMG":
            assert strategy.coarsest is not None  # enforced by Pydantic
            cl = strategy.coarsest
            p_block = (
                f"    {pf}\n"
                "    {\n"
                "        solver          GAMG;\n"
                f"        smoother        {strategy.smoother_or_precond};\n"
                f"        nCoarsestCells  {strategy.n_coarsest_cells};\n"
                f"        tolerance       {strategy.tolerance:g};\n"
                f"        relTol          {rel_tol_str};\n"
                "        coarsestLevelCorr\n"
                "        {\n"
                f"            solver          {cl.solver};\n"
                f"            preconditioner  {cl.preconditioner};\n"
                f"            tolerance       {cl.tolerance:g};\n"
                f"            relTol          {cl.rel_tol:g};\n"
                "        }\n"
                "    }\n"
            )
        else:
            # Direct Krylov path (PBiCGStab or PCG) — no coarsestLevelCorr.
            p_block = (
                f"    {pf}\n"
                "    {\n"
                f"        solver          {strategy.top_level};\n"
                f"        preconditioner  {strategy.smoother_or_precond};\n"
                f"        tolerance       {strategy.tolerance:g};\n"
                f"        relTol          {rel_tol_str};\n"
                "    }\n"
            )

        p_final_block = ""
        if not is_simple:
            p_final_block = (
                f"\n    {pf}Final\n"
                "    {\n"
                f"        ${pf};\n"
                "        relTol          0;\n"
                "    }\n"
            )
        return p_block, p_final_block

    # ── Density (rho) solver block ────────────────────────────────────────

    def _build_rho_solver_block(self) -> str:
        """Compressible solvers need a ``rho`` solver entry; empty otherwise."""
        if not self.is_compressible:
            return ""
        return (
            "\n    rho\n"
            "    {\n"
            "        solver          PCG;\n"
            "        preconditioner  DIC;\n"
            "        tolerance       1e-06;\n"
            "        relTol          0;\n"
            "    }\n"
        )

    # ── Equation solver block (regex over U, turb fields, h) ──────────────

    def _build_equation_solver_block(
        self,
        eq_fields: list[str],
        is_simple: bool | None = None,
    ) -> tuple[str, str]:
        """Build the equation regex solver block and its PIMPLE Final variant.

        Returns ``(eq_block, eq_final_block)`` — Final is empty for SIMPLE.

        Energy split: when the solver carries an energy equation, the
        energy variable (``e`` or ``h``) is broken out of the smoothSolver
        regex into its own ``PBiCG + DILU`` block.  This matches the
        OpenFOAM rhoSimpleFoam reference tutorial
        (``compressible/rhoSimpleFoam/angledDuctExplicitFixedCoeff``):
        smoothSolver tends to under-converge ``h``/``e`` in a single
        outer SIMPLE iteration (``Solving for h, Initial residual = 1,
        No Iterations 1-2``), leaving the energy field perpetually behind
        the pressure–velocity coupling and amplifying continuity errors.
        ``PBiCG`` with ``DILU`` preconditioning is the OF-tutorial choice
        for scalar transport equations on asymmetric matrices.
        """
        if is_simple is None:
            is_simple = self.algorithm == "SIMPLE"

        # Split: regex group covers everything except the energy variable.
        # When supports_energy is true the energy var gets its own block.
        if self.supports_energy and self.energy_var in eq_fields:
            non_energy = [f for f in eq_fields if f != self.energy_var]
        else:
            non_energy = list(eq_fields)

        if len(non_energy) == 1:
            eq_regex = non_energy[0]
        elif len(non_energy) >= 2:
            eq_regex = f'"({"|".join(non_energy)})"'
        else:
            eq_regex = ""  # only energy in eq_fields (rare / degenerate)

        eq_block = ""
        if eq_regex:
            eq_block = (
                f"\n    {eq_regex}\n"
                "    {\n"
                "        solver          smoothSolver;\n"
                "        smoother        symGaussSeidel;\n"
                "        tolerance       1e-05;\n"
                "        relTol          0.1;\n"
                "    }\n"
            )

        # Dedicated energy block — PBiCG + DILU.
        if self.supports_energy and self.energy_var in eq_fields:
            eq_block += (
                f"\n    {self.energy_var}\n"
                "    {\n"
                "        solver          PBiCG;\n"
                "        preconditioner  DILU;\n"
                "        tolerance       1e-06;\n"
                "        relTol          0.1;\n"
                "    }\n"
            )

        eq_final_block = ""
        if not is_simple:
            if eq_regex:
                if eq_regex.startswith('"'):
                    inner = eq_regex[1:-1]
                    final_regex = f'"{inner}Final"'
                else:
                    final_regex = f"{eq_regex}Final"
                eq_final_block = (
                    f"\n    {final_regex}\n"
                    "    {\n"
                    "        solver          smoothSolver;\n"
                    "        smoother        symGaussSeidel;\n"
                    "        tolerance       1e-06;\n"
                    "        relTol          0;\n"
                    "    }\n"
                )
            if self.supports_energy and self.energy_var in eq_fields:
                eq_final_block += (
                    f"\n    {self.energy_var}Final\n"
                    "    {\n"
                    "        solver          PBiCG;\n"
                    "        preconditioner  DILU;\n"
                    "        tolerance       1e-06;\n"
                    "        relTol          0;\n"
                    "    }\n"
                )
        return eq_block, eq_final_block

    # ── Compressible bounds (rhoMin / rhoMax / transonic) ─────────────────

    def _build_compressible_bounds(
        self,
        config: dict[str, Any],
        ctx: "FvBuildContext",
    ) -> str:
        """Render the rhoMin / rhoMax / transonic bounds block.

        Phase 2: delegates to ``resolve_compressible_bounds`` which returns
        a typed ``CompressibleBounds``.  This helper extracts the inputs
        (ρ, BC temperatures, operating pressure, Mach) from the config and
        renders the resolved strategy as OpenFOAM dict text.  Replaces the
        in-place arithmetic that lived here before.
        """
        if not self.is_compressible:
            return ""
        from simd_agent.run.case_spec import resolve_compressible_bounds

        profile = ctx.profile
        vel_mag = ctx.vel_mag

        fluid = config.get("fluid") or {}
        rho_cfg: float | None = None
        if isinstance(fluid, dict):
            for k in ("density", "rho"):
                v = fluid.get(k)
                if v is None:
                    continue
                try:
                    rho_cfg = float(v)
                    break
                except (TypeError, ValueError):
                    pass

        bc_temps: list[float] = []
        inlet_t: float | None = None
        for pbc in (config.get("boundary_conditions") or {}).values():
            if not isinstance(pbc, dict):
                continue
            t_entry = pbc.get("temperature") or pbc.get("T")
            t_val = (
                t_entry.get("value") or t_entry.get("uniform")
                if isinstance(t_entry, dict)
                else t_entry
            )
            try:
                tv = float(t_val)
            except (TypeError, ValueError):
                continue
            bc_temps.append(tv)
            if inlet_t is None:
                inlet_t = tv

        # Inlet Mach for the transonic decision (gas only).
        if profile == "cryogenic":
            mach = 0.0
        else:
            t_for_a = inlet_t if inlet_t and inlet_t > 0 else 300.0
            a_sound = (1.4 * 287.0 * t_for_a) ** 0.5
            mach = (vel_mag / a_sound) if a_sound > 0 else 0.0

        # Operating pressure — pull from outlet BC if present.
        # Inlet pressure — pull the highest pressure across all BCs; used
        # to size rho_max (ρ ≈ p / (R·T)) and pMax for the pressure clamp.
        op_p = 101325.0
        inlet_p: float | None = None
        bcs = config.get("boundary_conditions") or {}
        try:
            outlet_p_entry = (
                bcs.get("outlet", {})
                .get("pressure", {})
            )
            if isinstance(outlet_p_entry, dict):
                pv = outlet_p_entry.get("value") or outlet_p_entry.get("uniform")
                if pv is not None:
                    op_p = float(pv)
        except (TypeError, ValueError, AttributeError):
            pass
        for _name, _pbc in bcs.items():
            if _name == "outlet" or not isinstance(_pbc, dict):
                continue
            p_entry = _pbc.get("pressure") or _pbc.get("p")
            p_val = (
                p_entry.get("value") or p_entry.get("uniform")
                if isinstance(p_entry, dict) else p_entry
            )
            try:
                pv = float(p_val)
            except (TypeError, ValueError):
                continue
            if pv > 0 and (inlet_p is None or pv > inlet_p):
                inlet_p = pv

        bounds = resolve_compressible_bounds(
            is_compressible=True,
            profile=profile,
            rho=rho_cfg,
            bc_temps=sorted(set(bc_temps)),
            eos_t_ceiling=None,
            op_p=op_p,
            mach=mach,
            inlet_p=inlet_p,
        )

        lines = ""
        if bounds.rho_min is not None and bounds.rho_max is not None:
            lines += (
                f"    rhoMin          {bounds.rho_min:.3g};\n"
                f"    rhoMax          {bounds.rho_max:.3g};\n"
            )
        # pMin / pMax — without these, ``pressureControl`` is effectively
        # unbounded and a divergent rhoSimpleFoam can push p to ±1e+42
        # before the SIGFPE finally fires inside ``GAMGSolver::scale``.
        # The resolver always returns them for compressible cases, so this
        # branch is just for safety in unit tests.
        if bounds.p_min is not None and bounds.p_max is not None:
            lines += (
                f"    pMin            {bounds.p_min:.6g};\n"
                f"    pMax            {bounds.p_max:.6g};\n"
            )
        if bounds.transonic:
            lines += "    transonic       yes;\n"
        return lines

    # ── SIMPLE algorithm block ────────────────────────────────────────────

    def _build_simple_block(
        self,
        ctx: "FvBuildContext",
        eq_fields: list[str],
        bounds_block: str,
    ) -> str:
        """Build the ``SIMPLE { … }`` algorithm block.

        Encodes: non-ortho correctors, SIMPLEC switch, compressible bounds,
        pRef, and residualControl.  Pure assembly — no plugin-specific logic.
        """
        n_non_ortho = ctx.n_non_ortho
        use_simplec = ctx.use_simplec
        tier = ctx.tier
        profile = ctx.profile
        speed_tier = ctx.speed_tier
        pf = self.pressure_field

        # Bump correctors at high speed
        if speed_tier == "high" and n_non_ortho < 2:
            n_non_ortho = 2

        simplec_line = ""
        if (
            profile == "gas"
            and use_simplec
            and tier != "unknown"
            and speed_tier != "high"
        ):
            simplec_line = "    consistent      yes;\n"

        # residualControl — plain scalars for SIMPLE
        # Pressure tolerance — per-solver (rhoSimpleFoam: 1e-3, others: 1e-4).
        # Format with enough precision for both: 1e-04 / 1e-03 print as "1e-04"
        # and "1e-03" which OpenFOAM accepts identically to "1e-4" / "1e-3".
        p_tol = self.pressure_residual_tol
        p_tol_str = f"{p_tol:.0e}".replace("e-0", "e-").replace("e+0", "e+")
        res_lines = (
            f"        {pf:<16}{p_tol_str};\n"
            f"        U               1e-4;\n"
        )
        turb_res_fields = [
            f for f in eq_fields if f not in ("U", self.energy_var)
        ]
        if turb_res_fields:
            if len(turb_res_fields) == 1:
                res_lines += f"        {turb_res_fields[0]:<16}1e-3;\n"
            else:
                turb_regex = f'"({"|".join(turb_res_fields)})"'
                res_lines += f"        {turb_regex:<16}1e-3;\n"
        if self.supports_energy:
            res_lines += f"        {self.energy_var:<16}1e-3;\n"

        return (
            f"\n{self.algorithm}\n"
            "{\n"
            f"    nNonOrthogonalCorrectors {n_non_ortho};\n"
            f"{simplec_line}"
            f"{bounds_block}"
            "    pRefCell        0;\n"
            "    pRefValue       0;\n"
            "\n"
            "    residualControl\n"
            "    {\n"
            f"{res_lines}"
            "    }\n"
            "}\n"
        )

    # ── PIMPLE algorithm block ────────────────────────────────────────────

    def _build_pimple_block(
        self,
        ctx: "FvBuildContext",
        eq_fields: list[str],
        bounds_block: str,
    ) -> str:
        """Build the ``PIMPLE { … }`` algorithm block."""
        n_non_ortho = ctx.n_non_ortho
        speed_tier = ctx.speed_tier
        pf = self.pressure_field

        if speed_tier == "high" and n_non_ortho < 2:
            n_non_ortho = 2

        res_lines = (
            f"        {pf}   {{ tolerance 1e-4; relTol 0; }}\n"
            "        U   { tolerance 1e-4; relTol 0; }\n"
        )
        turb_res_fields = [
            f for f in eq_fields if f not in ("U", self.energy_var)
        ]
        for tf in turb_res_fields:
            res_lines += f"        {tf}   {{ tolerance 1e-3; relTol 0; }}\n"
        if self.supports_energy:
            res_lines += (
                f"        {self.energy_var}   "
                f"{{ tolerance 5e-3; relTol 0; }}\n"
            )

        return (
            f"\n{self.algorithm}\n"
            "{\n"
            "    nOuterCorrectors    2;\n"
            "    nCorrectors         2;\n"
            f"    nNonOrthogonalCorrectors {n_non_ortho};\n"
            "    momentumPredictor   yes;\n"
            f"{bounds_block}"
            "\n"
            "    residualControl\n"
            "    {\n"
            f"{res_lines}"
            "    }\n"
            "}\n"
        )

    # ── Relaxation blocks ─────────────────────────────────────────────────

    def _build_relaxation_simple(
        self,
        ctx: "FvBuildContext",
        eq_fields: list[str],
    ) -> str:
        """Build the ``relaxationFactors { … }`` block for a SIMPLE solver.

        Profile-aware: cryogenic forces conservative h=0.05, gas uses
        velocity-tier-aware textbook values.
        """
        profile = ctx.profile
        speed_tier = ctx.speed_tier
        pf = self.pressure_field

        if profile == "cryogenic":
            u_relax, p_relax, turb_relax, h_relax = 0.5, 0.3, 0.5, 0.05
        elif speed_tier == "high":
            u_relax, p_relax, turb_relax, h_relax = 0.3, 0.2, 0.3, 0.3
        elif speed_tier == "moderate":
            u_relax, p_relax, turb_relax, h_relax = 0.5, 0.3, 0.5, 0.5
        else:
            u_relax, p_relax, turb_relax, h_relax = 0.7, 0.3, 0.7, 0.5

        relax_eq_lines = f"        U               {u_relax};\n"
        for f in eq_fields:
            if f == "U":
                continue
            if f == self.energy_var:
                relax_eq_lines += (
                    f"        {self.energy_var:<16}{h_relax};\n"
                )
            else:
                relax_eq_lines += f"        {f:<16}{turb_relax};\n"

        # Density under-relaxation — compressible SIMPLE solvers only.
        # The OpenFOAM rhoSimpleFoam reference tutorials damp ρ by 95 %
        # (``rho 0.05``).  Without it the density jumps freely between
        # iterations, amplifying the pressure-correction → density-update
        # → continuity-error loop that drove the user's case to ±1e+42 Pa
        # before SIGFPE.  No downside: ρ converges to the same fixed point
        # either way, just smoother.  Incompressible / Boussinesq solvers
        # (simpleFoam, buoyantSimpleFoam) don't need this — their ρ is
        # either constant or derived analytically from T.
        rho_relax_line = ""
        if self.is_compressible:
            rho_relax_line = "        rho             0.05;\n"

        return (
            "\nrelaxationFactors\n"
            "{\n"
            "    fields\n"
            "    {\n"
            f"        {pf:<16}{p_relax};\n"
            f"{rho_relax_line}"
            "    }\n"
            "    equations\n"
            "    {\n"
            f"{relax_eq_lines}"
            "    }\n"
            "}\n"
        )

    def _build_relaxation_pimple(self, ctx: "FvBuildContext",) -> str:
        """Build the ``relaxationFactors { … }`` block for a PIMPLE solver."""
        profile = ctx.profile
        speed_tier = ctx.speed_tier
        if profile == "cryogenic":
            u_relax, catch_all = 0.5, 0.5
        elif speed_tier == "high":
            u_relax, catch_all = 0.3, 0.3
        elif speed_tier == "moderate":
            u_relax, catch_all = 0.5, 0.5
        else:
            u_relax, catch_all = 0.7, 0.7
        return (
            "\nrelaxationFactors\n"
            "{\n"
            f'    equations {{ U {u_relax}; ".*" {catch_all}; }}\n'
            "}\n"
        )

    # ── fvSchemes section helpers ─────────────────────────────────────────

    def _build_ddt_block(
        self,
        ctx: "FvBuildContext | None" = None,
    ) -> str:
        """ddtSchemes — read from the resolved regime_profile when available.

        LES needs ``backward`` (2nd-order time accuracy) instead of
        ``Euler``; SIMPLE-mode steady solvers stay on ``steadyState``.
        The regime resolver encodes those choices.  Falls back to the
        plugin's algorithm-driven default when ctx is not provided
        (legacy callers / tests).
        """
        if ctx is not None and ctx.regime_profile is not None:
            ddt = ctx.regime_profile.ddt_scheme
        else:
            ddt = "Euler" if self.is_transient else "steadyState"
        return (
            "ddtSchemes\n"
            "{\n"
            f"    default         {ddt};\n"
            "}\n"
        )

    def _build_grad_block(self, ctx: "FvBuildContext",) -> str:
        """gradSchemes — cellLimited grad(U) for compressible gas only."""
        if self.is_compressible and ctx.profile == "gas":
            grad_u_line = "    grad(U)         cellLimited Gauss linear 1;\n"
        else:
            grad_u_line = ""
        return (
            "gradSchemes\n"
            "{\n"
            "    default         Gauss linear;\n"
            f"{grad_u_line}"
            "}\n"
        )

    def _build_div_block(self, ctx: "FvBuildContext",) -> str:
        """divSchemes — driven by the resolved regime_profile.

        Every per-regime scheme choice (laminar / RAS / LES) is encoded in
        ``ctx.regime_profile`` via ``resolve_regime_profile``.  This renderer
        is now a pure assembly step over those values.

        Legacy fallback: when ctx.regime_profile is None (test callers that
        build FvBuildContext directly), the previous algorithm-aware /
        speed-aware literals are used.  Production code always goes through
        ``_fv_context`` which now constructs the profile.
        """
        from simd_agent.run.case_spec import resolve_div_phi_h_scheme

        speed_tier = ctx.speed_tier
        profile = ctx.profile
        turb_model = ctx.turb_model
        rp = ctx.regime_profile  # may be None (legacy path)

        lines: list[str] = ["    default         none;"]

        if rp is not None:
            # ── Profile-driven path (Phase 5 — typed regime resolver) ──
            lines.append(f"    div(phi,U)      {rp.div_phi_U};")
            if self.supports_energy:
                lines.append(
                    f"    div(phi,{self.energy_var})      {rp.div_phi_energy};"
                )
                # Kinetic-energy convection term — name depends on the
                # energy variable (Ekp for sensibleInternalEnergy, K
                # otherwise).  Scheme comes from the regime profile.
                ke_name = "Ekp" if self.energy_var == "e" else "K"
                lines.append(
                    f"    div(phi,{ke_name})      {rp.div_phi_K};"
                )
            # Pressure-work term — uses the flux name dictated by the regime
            # (phid for compressible RAS, phiv for laminar / LES / low-Mach).
            # rho* solvers only — buoyant p_rgh solvers don't have this term.
            if self.is_compressible and not self.needs_gravity:
                lines.append(
                    f"    div({rp.pressure_flux},p)     {rp.div_phi_p};"
                )
            # Transported turbulence fields — None for laminar.
            if rp.div_phi_turb is not None:
                turb_fields = self.turbulence_fields(turb_model)
                transported = [
                    f for f in turb_fields
                    if f in ("k", "omega", "epsilon", "nuTilda")
                ]
                if transported:
                    lines.append("")
                    for f in transported:
                        lines.append(f"    div(phi,{f})    {rp.div_phi_turb};")
        else:
            # ── Legacy literal path (kept for tests that build the ctx
            # directly without a regime_profile — primarily unit tests for
            # the renderer helpers).  Mirrors the pre-Phase-5 behaviour.
            _bc_temps = list(ctx.bc_temps)
            if self.is_compressible:
                if self.algorithm == "SIMPLE":
                    lines.append("    div(phi,U)      bounded Gauss upwind;")
                else:
                    _high_dp = ctx.pressure_ratio >= 3.0
                    if (
                        profile == "gas"
                        and speed_tier in ("low", "moderate")
                        and not _high_dp
                    ):
                        lines.append("    div(phi,U)      bounded Gauss linearUpwindV grad(U);")
                    else:
                        lines.append("    div(phi,U)      bounded Gauss upwind;")
                if self.supports_energy:
                    _h_scheme = resolve_div_phi_h_scheme(
                        is_compressible_energy=True,
                        bc_temps=_bc_temps if _bc_temps else None,
                    )
                    lines.append(
                        f"    div(phi,{self.energy_var})      {_h_scheme};"
                    )
                    if self.energy_var == "e":
                        lines.append("    div(phi,Ekp)    bounded Gauss upwind;")
                    else:
                        lines.append("    div(phi,K)      bounded Gauss upwind;")
                if not self.needs_gravity:
                    lines.append("    div(phid,p)     Gauss upwind;")
            else:
                if speed_tier == "high":
                    lines.append("    div(phi,U)      bounded Gauss upwind;")
                else:
                    lines.append("    div(phi,U)      bounded Gauss linearUpwind grad(U);")

            turb_fields = (
                self.turbulence_fields(turb_model)
                if turb_model != "laminar"
                else []
            )
            transported = [
                f for f in turb_fields
                if f in ("k", "omega", "epsilon", "nuTilda")
            ]
            if transported:
                lines.append("")
                turb_scheme = (
                    "bounded Gauss upwind" if speed_tier == "high"
                    else "bounded Gauss limitedLinear 1"
                )
                for f in transported:
                    lines.append(f"    div(phi,{f})    {turb_scheme};")

        # Viscous stress tensor — same form in all regimes.
        lines.append("")
        if self.is_compressible:
            lines.append("    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;")
        else:
            lines.append("    div((nuEff*dev2(T(grad(U))))) Gauss linear;")

        block_body = "\n".join(lines)
        return (
            "divSchemes\n"
            "{\n"
            f"{block_body}\n"
            "}\n"
        )

    def _build_laplacian_block(self, ctx: "FvBuildContext",) -> str:
        scheme = self._mesh_blended_scheme(ctx, kind="laplacian")
        return (
            "laplacianSchemes\n"
            "{\n"
            f"    default         {scheme};\n"
            "}\n"
        )

    def _build_sngrad_block(self, ctx: "FvBuildContext",) -> str:
        scheme = self._mesh_blended_scheme(ctx, kind="sngrad")
        return (
            "snGradSchemes\n"
            "{\n"
            f"    default         {scheme};\n"
            "}\n"
        )

    @staticmethod
    def _mesh_blended_scheme(ctx: "FvBuildContext", kind: str) -> str:
        """Pick laplacian / snGrad scheme from mesh tier + non-orthogonality."""
        tier = ctx.tier
        non_ortho = ctx.non_ortho
        if tier == "good" and non_ortho < 40:
            return "Gauss linear corrected" if kind == "laplacian" else "corrected"
        if non_ortho >= 65 or tier == "poor":
            return (
                "Gauss linear limited corrected 0.33"
                if kind == "laplacian" else "limited corrected 0.33"
            )
        return (
            "Gauss linear limited corrected 0.5"
            if kind == "laplacian" else "limited corrected 0.5"
        )

    @staticmethod
    def _build_interpolation_block() -> str:
        return (
            "interpolationSchemes\n"
            "{\n"
            "    default         linear;\n"
            "}\n"
        )

    def _build_flux_required_block(self) -> str:
        return (
            "fluxRequired\n"
            "{\n"
            "    default         no;\n"
            f"    {self.pressure_field};\n"
            "}\n"
        )

    @staticmethod
    def _build_wall_dist_block(turb_model: str) -> str:
        if turb_model == "laminar":
            return ""
        return (
            "wallDist\n"
            "{\n"
            "    method          meshWave;\n"
            "}\n"
        )

    # ── Whole-file deterministic renderers (Phase 4) ─────────────────────

    def _build_turbulence_properties(self, config: dict[str, Any]) -> str:
        """Render ``constant/turbulenceProperties`` from scratch.

        Uses the resolved ``TurbulenceRegimeProfile`` (which carries the
        full simulationType + model sub-dict text — including the LES
        ``delta`` / ``cubeRootVolCoeffs`` block — directly).  Falls back
        to the legacy in-line composition when the profile can't be
        built (e.g. corrupt config) so the renderer never raises.
        """
        from typing import Literal as _Literal, cast as _cast
        from simd_agent.run.case_spec import (
            resolve_regime_profile,
            resolve_turbulence_spec,
        )

        spec = resolve_turbulence_spec(self, config)
        try:
            _algo = _cast(
                _Literal["SIMPLE", "PIMPLE", "PISO"],
                (self.algorithm
                 if self.algorithm in ("SIMPLE", "PIMPLE", "PISO")
                 else "SIMPLE"),
            )
            rp = resolve_regime_profile(
                simulation_type=spec.simulation_type,
                turb_model=spec.model,
                algorithm=_algo,
                is_compressible=self.is_compressible,
                energy_var=self.energy_var,
            )
            body = rp.turbulence_properties_block
        except Exception:
            # Legacy fallback — identical to pre-Phase-5 behaviour.
            if spec.simulation_type == "laminar":
                body = "simulationType  laminar;\n"
            elif spec.simulation_type == "LES":
                body = (
                    "simulationType  LES;\n\n"
                    "LES\n{\n"
                    f"    LESModel        {spec.model};\n"
                    "    turbulence      on;\n"
                    "    printCoeffs     on;\n"
                    "}\n"
                )
            else:
                body = (
                    "simulationType  RAS;\n\n"
                    "RAS\n{\n"
                    f"    RASModel        {spec.model};\n"
                    "    turbulence      on;\n"
                    "    printCoeffs     on;\n"
                    "}\n"
                )
        return (
            self._foam_file_header("turbulenceProperties")
            + body
            + self._foam_file_footer()
        )

    # ── 0/nut and 0/alphat — patch-role dispatch (Phase 4) ───────────────

    def _build_nut(self, config: dict[str, Any]) -> str:
        """Render ``0/nut`` from scratch.

        Pure patch-role dispatch:
          * inlet / outlet  → ``calculated`` with ``value uniform 0``
          * wall            → ``nutkWallFunction`` (high-y+ default)
          * symmetry        → ``symmetry``
          * empty / wedge   → ``empty`` / ``wedge``

        Solver-agnostic: the file is identical for every RANS solver that
        carries ``nut``.  Replaces a chunk of LLM time + the patch-coverage
        validator's nut path.
        """
        patches = self._patch_role_pairs(config)
        bcs: list[str] = []
        for name, role in patches:
            if role == "wall":
                bcs.append(
                    f"    {name}\n"
                    "    {\n"
                    "        type            nutkWallFunction;\n"
                    "        value           uniform 0;\n"
                    "    }"
                )
            elif role == "empty":
                bcs.append(f"    {name}\n    {{\n        type            empty;\n    }}")
            elif role == "wedge":
                bcs.append(f"    {name}\n    {{\n        type            wedge;\n    }}")
            elif role == "symmetry":
                bcs.append(f"    {name}\n    {{\n        type            symmetry;\n    }}")
            else:
                bcs.append(
                    f"    {name}\n"
                    "    {\n"
                    "        type            calculated;\n"
                    "        value           uniform 0;\n"
                    "    }"
                )
        return (
            self._foam_file_header_field("volScalarField", "nut", "[0 2 -1 0 0 0 0]")
            + "internalField   uniform 0;\n\n"
            + "boundaryField\n{\n"
            + "\n".join(bcs)
            + "\n}\n"
            + self._foam_file_footer()
        )

    def _build_alphat(self, config: dict[str, Any]) -> str:
        """Render ``0/alphat`` from scratch.

        Compressible-energy turbulent flows only.  Like ``0/nut`` but with
        ``compressible::alphatWallFunction`` (the namespace-qualified form
        OF 2406 requires) plus ``Prt 0.85`` at walls.  Replaces Check 3e
        (alphatWallFunction namespace fix).
        """
        patches = self._patch_role_pairs(config)
        bcs: list[str] = []
        for name, role in patches:
            if role == "wall":
                bcs.append(
                    f"    {name}\n"
                    "    {\n"
                    "        type            compressible::alphatWallFunction;\n"
                    "        Prt             0.85;\n"
                    "        value           uniform 0;\n"
                    "    }"
                )
            elif role == "empty":
                bcs.append(f"    {name}\n    {{\n        type            empty;\n    }}")
            elif role == "wedge":
                bcs.append(f"    {name}\n    {{\n        type            wedge;\n    }}")
            elif role == "symmetry":
                bcs.append(f"    {name}\n    {{\n        type            symmetry;\n    }}")
            else:
                bcs.append(
                    f"    {name}\n"
                    "    {\n"
                    "        type            calculated;\n"
                    "        value           uniform 0;\n"
                    "    }"
                )
        return (
            self._foam_file_header_field("volScalarField", "alphat", "[1 -1 -1 0 0 0 0]")
            + "internalField   uniform 0;\n\n"
            + "boundaryField\n{\n"
            + "\n".join(bcs)
            + "\n}\n"
            + self._foam_file_footer()
        )

    # ── Per-field FoamFile header (volScalarField / volVectorField) ──────

    @staticmethod
    def _foam_file_header_field(class_name: str, object_name: str, dimensions: str) -> str:
        """Return the FoamFile header for a 0/ field file."""
        return (
            "FoamFile\n"
            "{\n"
            "    version     2.0;\n"
            "    format      ascii;\n"
            f"    class       {class_name};\n"
            f"    object      {object_name};\n"
            "}\n"
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n"
            "\n"
            f"dimensions      {dimensions};\n\n"
        )

    def _patch_role_pairs(self, config: dict[str, Any]) -> list[tuple[str, str]]:
        """Return (patch_name, role) pairs from the validated config.

        Role classification (matches CaseSpec.patch_type_by_name):
          * "wall" / "empty" / "wedge" / "symmetry" — verbatim
          * "inlet" / "outlet" / "patch" — fall into the default branch
        """
        bcs = config.get("boundary_conditions") or {}
        mesh = config.get("mesh", {}) or {}
        patches_meta = mesh.get("patches", []) if isinstance(mesh, dict) else []
        # Mesh-side roles are authoritative for empty / wedge / symmetry walls;
        # BC-side patch_class tells us inlet / outlet / wall otherwise.
        type_by_name: dict[str, str] = {}
        for mp in patches_meta:
            if isinstance(mp, dict):
                n, t = mp.get("name"), mp.get("type", "patch")
            else:
                n, t = getattr(mp, "name", None), getattr(mp, "type", "patch")
            if not n:
                continue
            type_by_name[n] = t

        pairs: list[tuple[str, str]] = []
        for name, pbc in bcs.items():
            mesh_type = type_by_name.get(name)
            if mesh_type in ("empty", "wedge", "symmetry"):
                pairs.append((name, mesh_type))
                continue
            if mesh_type == "wall":
                pairs.append((name, "wall"))
                continue
            # Fall back to BC patch_type/class
            pt = ""
            if isinstance(pbc, dict):
                pt = (pbc.get("patch_type") or pbc.get("patch_class") or "").lower()
            if pt == "wall":
                pairs.append((name, "wall"))
            elif pt in ("symmetry", "empty", "wedge"):
                pairs.append((name, pt))
            else:
                pairs.append((name, "patch"))
        return pairs

    # ── Deterministic-files registry (Phase 4 architecture) ──────────────

    def render_deterministic_files(self, config: dict[str, Any]) -> dict[str, str]:
        """Return files this plugin renders from scratch — LLM never sees them.

        Phase 4 mechanism: any file in the returned dict is generated by the
        plugin's own renderer (Python only, no LLM call) and merged into the
        final file set.  The plugin's ``required_files()`` must exclude every
        key in this dict so the codegen loop doesn't waste an LLM call on it.

        Default covers the universally-deterministic files:
          * ``system/fvSolution`` — built from typed strategy (Phase 2/3)
          * ``system/fvSchemes``  — same
          * ``constant/turbulenceProperties`` — pure simulationType / model
          * ``0/nut``    — pure patch-role dispatch (when turbulent)
          * ``0/alphat`` — pure patch-role dispatch (when compressible energy)

        Plugins with additional deterministic files (e.g. ``constant/g`` for
        buoyant solvers) extend the dict in their override.
        """
        turb_model = self._get_turb_model_from_config(config)
        is_turbulent = turb_model not in ("laminar", "none", "")

        files: dict[str, str] = {
            "system/fvSolution": self._build_fv_solution(config),
            "system/fvSchemes": self._build_fv_schemes(config),
            "constant/turbulenceProperties": self._build_turbulence_properties(config),
        }
        if is_turbulent:
            files["0/nut"] = self._build_nut(config)
            # alphat only exists for compressible energy solvers.
            if self.is_compressible and self.supports_energy:
                files["0/alphat"] = self._build_alphat(config)
        return files

    # ── End of composable helpers ─────────────────────────────────────────

    def _build_fv_solution(self, config: dict[str, Any]) -> str:
        """Each plugin assembles its own fvSolution from the helpers above.

        Default raises — implementing per-solver is mandatory because the
        composition (which blocks, in which order, with what bounds) is
        different for every solver.  See e.g. ``rhoSimpleFoam/solver.py``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _build_fv_solution() using "
            "the composable helpers in SolverPlugin (e.g. _foam_file_header, "
            "_build_pressure_solver_block, _build_simple_block / _build_pimple_block, "
            "_build_relaxation_simple / _build_relaxation_pimple)."
        )

    def _build_fv_schemes(self, config: dict[str, Any]) -> str:
        """Each plugin assembles its own fvSchemes from the helpers above.

        Default raises — implementing per-solver is mandatory.  See e.g.
        ``rhoSimpleFoam/solver.py``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _build_fv_schemes() using "
            "the composable helpers in SolverPlugin (_build_ddt_block, "
            "_build_grad_block, _build_div_block, _build_laplacian_block, "
            "_build_sngrad_block, _build_flux_required_block, ...)."
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

    def _unify_inlet_turbulence(
        self,
        files: dict[str, str],
        issues: list[ValidationIssue],
        config: dict[str, Any],
    ) -> dict[str, str]:
        """Re-derive each turbulent inlet's k / ω / ε using a SHARED TI and L.

        Physics — what should be the same vs. different across inlets:

        * SAME everywhere (flow-wide properties):
            - Turbulence intensity I (e.g. 5% for internal flow)
            - Turbulence length scale L = 0.07 · D_h (geometry-derived)
        * DIFFERENT per inlet (because each inlet has its own U):
            - k  = 1.5 · (U · I)²
            - ω  = √k / (Cμ^0.25 · L)
            - ε  = Cμ^0.75 · k^1.5 / L

        Two inlets at different speeds therefore *should* have different
        k, ω, ε values — what the LLM gets wrong is using *different I*
        per inlet (e.g. 5% for inlet_main but 1% for inlet_small).  This
        validator recomputes every inlet's value from a single I + L so the
        TI is consistent while preserving the velocity-dependent physics.

        ``internalField`` and wall-function ``value`` entries are left
        alone — they're representative scalars, not per-inlet conditions.
        """
        from math import sqrt

        # Turbulence intensity — flow-wide property.  5% is the textbook
        # default for internal flows (pipes / ducts).  Configurable later
        # via CaseSpec if a use case demands a different value.
        TI = 0.05
        Cmu = 0.09
        Cmu_quarter = Cmu ** 0.25
        Cmu_three_quarter = Cmu ** 0.75

        # Length scale L = 0.07 · D_h.  Derive D_h from mesh bbox; fall
        # back to a CaseSpec-precomputed value if available.
        D_h: float | None = None
        mesh = config.get("mesh", {}) or {}
        check_mesh = mesh.get("check_mesh") or mesh.get("checkMesh")
        if check_mesh:
            bbox = (
                check_mesh.get("bounding_box")
                if isinstance(check_mesh, dict)
                else getattr(check_mesh, "bounding_box", None)
            )
            if bbox:
                try:
                    bb_min = bbox.get("min") or bbox.get("Min")
                    bb_max = bbox.get("max") or bbox.get("Max")
                    extents = sorted(
                        abs(float(bb_max[i]) - float(bb_min[i])) for i in range(3)
                    )
                    a, b = extents[0], extents[1]
                    if a > 0 and b > 0:
                        D_h = 2.0 * a * b / (a + b)
                except (KeyError, IndexError, TypeError, ValueError):
                    pass
        # Reverse-engineer L from precomputed k/ω if D_h still unknown
        L: float | None = None
        if D_h is not None and D_h > 0:
            L = 0.07 * D_h
        else:
            tiv = config.get("turbulence_initial_values") or {}
            try:
                k0 = float(tiv.get("k"))
                w0 = float(tiv.get("omega"))
                if k0 > 0 and w0 > 0:
                    L = sqrt(k0) / (Cmu_quarter * w0)
            except (TypeError, ValueError):
                pass
        if L is None or L <= 0:
            L = 0.007  # final fallback (≈ 7% of a 0.1 m duct)

        # Walk every inlet, compute per-inlet target values.
        # Each inlet can have its OWN turbulence intensity (e.g. a turbulent
        # jet at 10% mixing with a settled coflow at 1%).  Lookup order:
        #   1. pbc["turbulenceIntensity"] / pbc["turbulence_intensity"] — user-specified
        #   2. Reverse-engineered from the existing 0/k value (LLM may have used the
        #      intent the user gave even if the metadata wasn't carried through)
        #   3. Global default TI (5% for internal flow)
        bcs = config.get("boundary_conditions") or {}
        inlet_targets: dict[str, dict[str, float]] = {}  # patch → {k, omega, epsilon}
        inlet_TIs: dict[str, float] = {}  # for diagnostics

        # If the LLM-generated 0/k exists, build a map of patch → existing k
        # so we can recover the LLM's TI choice when the BC lacks one.
        existing_k_values: dict[str, float] = {}
        k_file = files.get("0/k")
        if k_file:
            for _pn in bcs:
                m = re.search(
                    rf"\b{re.escape(_pn)}\s*\{{[^}}]*?value\s+uniform\s+([\d.eE+\-]+)",
                    k_file,
                    re.DOTALL,
                )
                if m:
                    try:
                        existing_k_values[_pn] = float(m.group(1))
                    except ValueError:
                        pass

        for pname, pbc in bcs.items():
            if not isinstance(pbc, dict):
                continue
            pt = (pbc.get("patch_type") or "").lower()
            if pt not in ("inlet", "pressure_inlet", "mass_flow_inlet") and "inlet" not in pname.lower():
                continue
            U_i = self._patch_velocity_magnitude(pbc)
            if U_i is None or U_i <= 0:
                continue

            # 1. User-specified per-patch TI
            TI_i = pbc.get("turbulenceIntensity")
            if TI_i is None:
                TI_i = pbc.get("turbulence_intensity")
            try:
                TI_i = float(TI_i) if TI_i is not None else None
            except (TypeError, ValueError):
                TI_i = None
            # Planner sometimes returns a percent rather than a fraction
            if TI_i is not None and TI_i > 1.0:
                TI_i = TI_i / 100.0

            # 2. Reverse-engineer from existing k if no explicit TI was set
            if TI_i is None and pname in existing_k_values:
                k_existing = existing_k_values[pname]
                if k_existing > 0:
                    # k = 1.5 · (U·I)²  →  I = √(2k/3) / U
                    TI_implied = (2.0 * k_existing / 3.0) ** 0.5 / U_i
                    # Accept only physically reasonable TIs (0.1% to 30%)
                    if 0.001 <= TI_implied <= 0.30:
                        TI_i = TI_implied

            # 3. Global default
            if TI_i is None:
                TI_i = TI

            k_i = 1.5 * (U_i * TI_i) ** 2
            omega_i = sqrt(k_i) / (Cmu_quarter * L)
            epsilon_i = Cmu_three_quarter * (k_i ** 1.5) / L
            inlet_targets[pname] = {
                "k": k_i,
                "omega": omega_i,
                "epsilon": epsilon_i,
            }
            inlet_TIs[pname] = TI_i

        if not inlet_targets:
            return files

        # Rewrite each inlet patch block in 0/k, 0/omega, 0/epsilon
        for field_name in ("k", "omega", "epsilon"):
            fpath = f"0/{field_name}"
            content = files.get(fpath)
            if not content:
                continue
            new_content = content
            changed: list[tuple[str, float, float]] = []  # (patch, old, new)
            for patch, targets in inlet_targets.items():
                target = targets[field_name]
                # Find the inlet block and its current value
                pm = re.search(
                    rf"\b{re.escape(patch)}\s*\{{([^}}]*?)value\s+uniform\s+([\d.eE+\-]+)(\s*;)",
                    new_content,
                    re.DOTALL,
                )
                if not pm:
                    continue
                try:
                    current = float(pm.group(2))
                except ValueError:
                    continue
                # Tolerance: 5% — TI estimation has inherent uncertainty
                if abs(current - target) / max(target, 1e-12) < 0.05:
                    continue
                new_content = re.sub(
                    rf"(\b{re.escape(patch)}\s*\{{[^}}]*?value\s+uniform\s+)[\d.eE+\-]+(\s*;)",
                    rf"\g<1>{target:.6g}\g<2>",
                    new_content,
                    count=1,
                    flags=re.DOTALL,
                )
                changed.append((patch, current, target))

            if changed:
                files[fpath] = new_content
                diff_str = ", ".join(
                    f"{p}: {old:.4g}→{new:.4g}" for p, old, new in changed
                )
                ti_str = ", ".join(
                    f"{p} TI={inlet_TIs[p]:.1%}" for p, _, _ in changed
                ) if inlet_TIs else f"default {TI:.1%}"
                issues.append(
                    ValidationIssue(
                        "warning",
                        fpath,
                        f"Recomputed inlet {field_name} from each patch's own (U, TI) "
                        f"with shared L={L:.4g} m.  Per-patch TI: {ti_str}.  "
                        f"Values differ across inlets when velocities or TIs differ — "
                        f"that's correct physics.  Changed: {diff_str}.",
                        fix=f"{field_name} per inlet from U·I·L formula",
                    )
                )
        return files

    @staticmethod
    def _patch_velocity_magnitude(pbc: dict[str, Any]) -> float | None:
        """Extract |U| from a single patch BC dict (used by turbulence rederivation)."""
        u = pbc.get("velocity") or pbc.get("U")
        if isinstance(u, dict):
            v = u.get("value")
            if isinstance(v, (list, tuple)) and len(v) >= 3:
                try:
                    return (float(v[0]) ** 2 + float(v[1]) ** 2 + float(v[2]) ** 2) ** 0.5
                except (TypeError, ValueError):
                    return None
            for k in ("magnitude", "meanVelocity", "massFlowRate", "volumetricFlowRate"):
                m = u.get(k)
                if m is not None:
                    try:
                        return abs(float(m))
                    except (TypeError, ValueError):
                        pass
        elif isinstance(u, (list, tuple)) and len(u) >= 3:
            try:
                return (float(u[0]) ** 2 + float(u[1]) ** 2 + float(u[2]) ** 2) ** 0.5
            except (TypeError, ValueError):
                return None
        return None

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

        1. ``nCoarsestCells 20`` — matches the OpenFOAM
           ``rhoSimpleFoam/angledDuctExplicitFixedCoeff`` reference.
           OF's default is 10; 20 is a conservative middle that keeps
           coarse solves small while leaving a safety margin against
           the over-agglomeration SIGFPE on tet meshes.
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
