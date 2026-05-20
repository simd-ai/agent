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
    # Multi-region (CHT) flag.  Default ``False`` covers every single-
    # region plugin (simpleFoam, pimpleFoam, rhoSimpleFoam, …); only
    # :class:`MultiRegionBase` subclasses (cht* solvers) override this to
    # ``True``.  ``validate_full`` reads it to decide whether to invoke
    # the flat single-region validator after plugin-specific validate().
    is_multi_region: bool = False

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

        # 2. Shared single-region validator — only runs for single-region
        # plugins.  Multi-region (CHT) cases ship a flat-incompatible
        # per-region tree (``0/<region>/<field>``, ``system/<region>/...``,
        # ``constant/<region>/...``); every check in the single-region
        # validator is written for the flat single-region layout and would
        # either no-op noisily or silently damage the per-region files
        # (e.g. stamp a top-level ``0/T`` "missing", rewrite per-region
        # boundary lists, …).  The deterministic renderer in
        # :class:`MultiRegionBase` is authoritative for those files instead.
        if not self.is_multi_region:
            try:
                from simd_agent.run.single_region import (
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

        # 3. Universal constraint-patch BC fix — runs LAST so it overrides
        #    anything the LLM or the legacy validator left behind on
        #    symmetry / empty / wedge patches.  Touches single-region and
        #    multi-region cases identically; only patches whose polyMesh
        #    type is a constraint type are rewritten.
        fixed = self._fix_constraint_patch_bcs(fixed, config, issues)

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
        """Fix mismatched curly braces (thin wrapper, see ``legacy_fixers``)."""
        from simd_agent.solvers import legacy_fixers
        return legacy_fixers.fix_brace_balance(files, issues)

    def _fix_constraint_patch_bcs(
        self, files: dict[str, str], config: dict[str, Any],
        issues: list[ValidationIssue],
    ) -> dict[str, str]:
        """Force ``symmetry`` / ``empty`` / ``wedge`` BCs to match the
        mesh-side patch type (thin wrapper, see ``bc_fixers``)."""
        from simd_agent.solvers import bc_fixers
        return bc_fixers.fix_constraint_patch_bcs(files, issues, config)

    @staticmethod
    def _balance_braces(content: str) -> str:
        """Balance curly braces (thin wrapper, see ``legacy_fixers``)."""
        from simd_agent.solvers import legacy_fixers
        return legacy_fixers.balance_braces(content)

    def _fix_controldict_solver(
        self, files: dict[str, str], issues: list[ValidationIssue]
    ) -> dict[str, str]:
        """Ensure controlDict declares the correct solver (thin wrapper)."""
        from simd_agent.solvers import legacy_fixers
        return legacy_fixers.fix_controldict_solver(
            files, issues, solver_name=self.name
        )

    def _fix_pressure_field(
        self, files: dict[str, str], issues: list[ValidationIssue]
    ) -> dict[str, str]:
        """Ensure the correct pressure field is present (thin wrapper)."""
        from simd_agent.solvers import legacy_fixers
        return legacy_fixers.fix_pressure_field(
            files, issues,
            solver_name=self.name,
            pressure_field=self.pressure_field,
        )

    def _fix_pressure_value(
        self, files: dict[str, str], issues: list[ValidationIssue]
    ) -> dict[str, str]:
        """Fix absolute pressure in 0/p for incompressible (thin wrapper)."""
        from simd_agent.solvers import legacy_fixers
        return legacy_fixers.fix_pressure_value(
            files, issues,
            solver_name=self.name,
            is_compressible=self.is_compressible,
        )

    def _remove_unneeded_thermo(
        self, files: dict[str, str], issues: list[ValidationIssue]
    ) -> dict[str, str]:
        """Remove thermo files for non-energy solvers (thin wrapper)."""
        from simd_agent.solvers import legacy_fixers
        return legacy_fixers.remove_unneeded_thermo(
            files, issues,
            solver_name=self.name,
            supports_energy=self.supports_energy,
        )

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

        # Detect impulsive (mass-flow) inlets — these need special startup
        # handling (seeded U field, tighter maxCo, more PIMPLE iters).
        # Compute the bulk velocity estimate while we're walking the BCs.
        has_impulsive_inlets, bulk_velocity = self._extract_impulsive_state(
            config, bc_pressures, bc_temps
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
            has_impulsive_inlets=has_impulsive_inlets,
            bulk_velocity=bulk_velocity,
            profile=profile,
            heat_transfer_active=heat_transfer_active,
            turb_model=turb_model,
            mesh_quality=mq,
            regime_profile=regime_profile,
        )

    @staticmethod
    def _extract_impulsive_state(
        config: dict[str, Any],
        bc_pressures: list[float],
        bc_temps: list[float],
    ) -> tuple[bool, float]:
        """Return ``(has_impulsive_inlets, bulk_velocity)``.

        An "impulsive inlet" is one using ``flowRateInletVelocity`` —
        these force U from mdot at the patch even when the internal
        field is (0,0,0), which produces a giant pressure spike at
        iteration 1 unless the internal field is pre-seeded.

        ``bulk_velocity`` is the maximum among:
          * U = mdot / (ρ · A) for each mass-flow inlet (assumed
            cross-section A ≈ 1e-4 m² because we don't have mesh patch
            areas at codegen time — the resulting magnitude is right
            within a factor of 2-5, which is enough to remove the
            iteration-1 shock),
          * explicit U magnitudes from ``fixedValue`` velocity inlets.
        """
        bcs = config.get("boundary_conditions") or {}
        if not isinstance(bcs, dict):
            return False, 0.0

        # Reference density: gas at coldest BC temp + highest BC pressure.
        # This over-estimates ρ on warm air → U_bulk is slightly conservative.
        p_high = max((p for p in bc_pressures if p > 0), default=101325.0)
        t_cold = min((t for t in bc_temps if t > 0), default=288.15)
        rho_ref = max(0.1, p_high / (287.0 * max(t_cold, 1.0)))

        # Heuristic patch cross-section.  Without mesh data the best we
        # can do is a single representative area; the resulting U_bulk is
        # an over-estimate for small inlets and an under-estimate for
        # large ones — but the goal here is to KILL the impulsive shock,
        # not to match the final converged velocity.
        A_inlet_default = 1e-4  # m²

        has_impulsive = False
        u_bulk = 0.0

        for _name, pbc in bcs.items():
            if not isinstance(pbc, dict):
                continue
            u_entry = pbc.get("U") or pbc.get("velocity") or {}
            if not isinstance(u_entry, dict):
                continue
            u_type = (u_entry.get("type") or "").strip()
            if u_type == "flowRateInletVelocity":
                has_impulsive = True
                mdot = u_entry.get("massFlowRate") or u_entry.get("volumetricFlowRate")
                try:
                    mdot_val = float(mdot) if mdot is not None else 0.0
                except (TypeError, ValueError):
                    mdot_val = 0.0
                if u_type == "flowRateInletVelocity" and u_entry.get("volumetricFlowRate") is not None:
                    # Volumetric: U = Q / A.
                    u_est = abs(mdot_val) / A_inlet_default
                else:
                    # Mass flow: U = mdot / (ρ · A).
                    u_est = abs(mdot_val) / (rho_ref * A_inlet_default)
                u_bulk = max(u_bulk, u_est)
            elif u_type == "fixedValue":
                v = u_entry.get("value")
                if isinstance(v, (list, tuple)) and len(v) >= 3:
                    try:
                        mag = sum(float(c) ** 2 for c in v[:3]) ** 0.5
                        u_bulk = max(u_bulk, mag)
                    except (TypeError, ValueError):
                        pass

        return has_impulsive, u_bulk

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
        """Build ``p`` solver block + optional ``pFinal`` (thin wrapper, see ``blocks``)."""
        from simd_agent.solvers import blocks
        return blocks.pressure_solver_block(self, ctx, is_simple=is_simple)

    # ── Equation solver block (regex over U, turb fields, h) ──────────────

    def _build_equation_solver_block(
        self,
        eq_fields: list[str],
        is_simple: bool | None = None,
    ) -> tuple[str, str]:
        """Build equation solver block + Final variant (thin wrapper)."""
        from simd_agent.solvers import blocks
        return blocks.equation_solver_block(self, eq_fields, is_simple=is_simple)

    # ── fvSchemes section helpers ─────────────────────────────────────────

    def _build_ddt_block(
        self,
        ctx: "FvBuildContext | None" = None,
    ) -> str:
        """ddtSchemes (thin wrapper, see ``blocks``)."""
        from simd_agent.solvers import blocks
        return blocks.ddt_block(self, ctx)

    def _build_grad_block(self, ctx: "FvBuildContext",) -> str:
        """gradSchemes (thin wrapper, see ``blocks``)."""
        from simd_agent.solvers import blocks
        return blocks.grad_block(self, ctx)

    def _build_div_block(self, ctx: "FvBuildContext",) -> str:
        """divSchemes (thin wrapper, see ``blocks``)."""
        from simd_agent.solvers import blocks
        return blocks.div_block(self, ctx)

    def _build_laplacian_block(self, ctx: "FvBuildContext",) -> str:
        """laplacianSchemes (thin wrapper, see ``blocks``)."""
        from simd_agent.solvers import blocks
        return blocks.laplacian_block(self, ctx)

    def _build_sngrad_block(self, ctx: "FvBuildContext",) -> str:
        """snGradSchemes (thin wrapper, see ``blocks``)."""
        from simd_agent.solvers import blocks
        return blocks.sngrad_block(self, ctx)

    @staticmethod
    def _mesh_blended_scheme(ctx: "FvBuildContext", kind: str) -> str:
        """Pick mesh-aware scheme (thin wrapper, see ``blocks``)."""
        from simd_agent.solvers import blocks
        return blocks.mesh_blended_scheme(ctx, kind)

    @staticmethod
    def _build_interpolation_block() -> str:
        """interpolationSchemes (thin wrapper, see ``blocks``)."""
        from simd_agent.solvers import blocks
        return blocks.interpolation_block()

    def _build_flux_required_block(self) -> str:
        """fluxRequired (thin wrapper, see ``blocks``)."""
        from simd_agent.solvers import blocks
        return blocks.flux_required_block(self)

    @staticmethod
    def _build_wall_dist_block(turb_model: str) -> str:
        """wallDist (thin wrapper, see ``blocks``)."""
        from simd_agent.solvers import blocks
        return blocks.wall_dist_block(turb_model)

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
        """Harden laplacian/snGrad for non-orthogonal meshes (thin wrapper)."""
        from simd_agent.solvers import legacy_fixers
        return legacy_fixers.fix_fv_schemes_non_ortho(files, issues, config)

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
        """Fix 'thermodynamics' → 'thermo' key (thin wrapper)."""
        from simd_agent.solvers import legacy_fixers
        return legacy_fixers.fix_thermo_type_key(files, issues)

    def _fix_relaxation_factors(
        self,
        files: dict[str, str],
        issues: list[ValidationIssue],
        max_equation_relaxation: float = 0.8,
    ) -> dict[str, str]:
        """Enforce safe relaxation factors (thin wrapper)."""
        from simd_agent.solvers import legacy_fixers
        return legacy_fixers.fix_relaxation_factors(
            files, issues,
            algorithm=self.algorithm,
            pressure_field=self.pressure_field,
            max_equation_relaxation=max_equation_relaxation,
        )

    def _fix_non_orthogonal_correctors(
        self,
        files: dict[str, str],
        issues: list[ValidationIssue],
        minimum: int = 1,
    ) -> dict[str, str]:
        """Ensure nNonOrthogonalCorrectors >= minimum (thin wrapper)."""
        from simd_agent.solvers import legacy_fixers
        return legacy_fixers.fix_non_orthogonal_correctors(
            files, issues, minimum=minimum,
        )

    def _fix_gamg_coarsest_level(
        self,
        files: dict[str, str],
        issues: list[ValidationIssue],
    ) -> dict[str, str]:
        """Harden GAMG pressure solver (thin wrapper)."""
        from simd_agent.solvers import legacy_fixers
        return legacy_fixers.fix_gamg_coarsest_level(
            files, issues, pressure_field=self.pressure_field,
        )

    # ── Outlet / inlet BC robustness (universal, paradigm-agnostic) ─────────
    #
    # Applied across every solver that has a 0/ directory.  Hoisted into
    # SolverPlugin (rather than a family base) because the same OF tutorial
    # pattern — outlet ``inletOutlet`` + inlet ``turbulentIntensity…`` BCs —
    # applies regardless of whether the algorithm is SIMPLE or PIMPLE.

    @staticmethod
    def _rewrite_patch_body(
        content: str, patch_name: str, new_body: str
    ) -> str | None:
        """Replace one patch block's body (thin wrapper, see ``bc_fixers``)."""
        from simd_agent.solvers import bc_fixers
        return bc_fixers.rewrite_patch_body(content, patch_name, new_body)

    @staticmethod
    def _classify_patches(
        config: dict[str, Any],
    ) -> tuple[list[str], list[str]]:
        """Return ``(outlets, inlets)`` (thin wrapper, see ``bc_fixers``)."""
        from simd_agent.solvers import bc_fixers
        return bc_fixers.classify_patches(config)

    def _fix_outlet_backflow_bcs(
        self,
        files: dict[str, str],
        issues: list[ValidationIssue],
        config: dict[str, Any],
    ) -> dict[str, str]:
        """Outlet zeroGradient → inletOutlet (thin wrapper)."""
        from simd_agent.solvers import bc_fixers
        return bc_fixers.fix_outlet_backflow_bcs(files, issues, config)

    def _fix_inlet_turbulence_bc_types(
        self,
        files: dict[str, str],
        issues: list[ValidationIssue],
        config: dict[str, Any],
    ) -> dict[str, str]:
        """Inlet k/ω/ε runtime-derived BCs (thin wrapper)."""
        from simd_agent.solvers import bc_fixers
        return bc_fixers.fix_inlet_turbulence_bc_types(files, issues, config)

    def _fix_residual_control_format(
        self,
        files: dict[str, str],
        issues: list[ValidationIssue],
    ) -> dict[str, str]:
        """Convert plain-scalar residualControl to sub-dicts (thin wrapper)."""
        from simd_agent.solvers import legacy_fixers
        return legacy_fixers.fix_residual_control_format(
            files, issues, algorithm=self.algorithm,
        )

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
