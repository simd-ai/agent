# simd_agent/solvers/rhoSimpleFoam/solver.py
"""rhoSimpleFoam solver plugin.

Steady-state, compressible, single-phase solver using SIMPLE algorithm.
Solves the energy equation (0/T) and requires thermophysicalProperties.
Used for forced convection with significant density variations (e.g.
heated gas flows, cryogenic pipe flows).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from simd_agent.solvers.base import (
    MatchResult,
    ValidationIssue,
    ValidationResult,
)
from simd_agent.solvers.families import CompressibleMixin, SteadyBase

logger = logging.getLogger(__name__)


class RhoSimpleFoamSolver(SteadyBase, CompressibleMixin):
    """rhoSimpleFoam — steady compressible energy RANS."""

    name = "rhoSimpleFoam"
    algorithm = "SIMPLE"
    pressure_field = "p"
    is_transient = False
    is_compressible = True
    supports_energy = True
    needs_gravity = False
    is_multiphase = False

    # Match the OpenFOAM reference rhoSimpleFoam tutorials (e.g.
    # ``compressible/rhoSimpleFoam/angledDuctExplicitFixedCoeff``) which
    # transport **internal energy** (``e``), not enthalpy.  The
    # corresponding ``thermoType.energy`` is ``sensibleInternalEnergy``
    # and the divScheme is ``div(phi,e)``.  Avoiding the pressure-work
    # source term ``∂p/∂t`` in the energy equation removes a major
    # startup transient on steady compressible cases.
    energy_var = "e"

    # Loosen the pressure residual to match the OF reference tutorial
    # (which uses ``p 1e-2``).  ``1e-4`` was practically never reached on
    # compressible cases and the run kept iterating long after the flow
    # had stabilised — wasting compute and forcing manual termination.
    # ``1e-3`` is our middle-ground value (10× tighter than OF, 10× looser
    # than our previous incompressible-style default).
    pressure_residual_tol = 1e-3

    # ── Matching ──────────────────────────────────────────────────────────

    def matches(self, config: dict[str, Any]) -> MatchResult:
        physics = config.get("physics", {}) or {}

        heat = bool(
            config.get("heat_transfer")
            or physics.get("heat_transfer")
        )
        compressible = (
            config.get("compressibility") or physics.get("compressibility", "incompressible")
        ) == "compressible"
        transient = (
            config.get("time_stepping") or physics.get("time_scheme", "steady")
        ) in ("transient", "unsteady")
        multiphase = bool(config.get("multiphase") or physics.get("multiphase"))
        buoyancy = bool(config.get("gravity") or physics.get("gravity"))

        # Disqualifiers
        if multiphase:
            return MatchResult(0.0, "rhoSimpleFoam is single-phase only")
        if transient:
            return MatchResult(
                0.1,
                "rhoSimpleFoam is steady-state; rhoPimpleFoam for transient compressible",
                warnings=["Consider rhoPimpleFoam for transient flow"],
            )
        if buoyancy:
            return MatchResult(
                0.3,
                "rhoSimpleFoam can do forced convection but buoyantSimpleFoam is better for natural convection",
                warnings=["Consider buoyantSimpleFoam for buoyancy-driven flow"],
            )

        # Needs compressibility or heat transfer to justify over simpleFoam
        if compressible and heat:
            return MatchResult(0.95, "Steady compressible flow with heat transfer — ideal for rhoSimpleFoam")
        if compressible:
            return MatchResult(0.85, "Steady compressible flow — rhoSimpleFoam is appropriate")
        if heat:
            return MatchResult(0.8, "Steady flow with heat transfer — rhoSimpleFoam handles energy equation")

        # Incompressible, no heat — simpleFoam is better
        return MatchResult(0.1, "No compressibility or heat — simpleFoam would be simpler")

    # ── Required files ────────────────────────────────────────────────────

    def required_files(self, config: dict[str, Any]) -> list[str]:
        turb_model = self._get_turb_model_from_config(config)
        heat_transfer = self._has_heat_transfer(config)

        files = [
            "system/controlDict",
            # system/fvSchemes and system/fvSolution are generated
            # deterministically in validate(), not by the LLM.
            "constant/thermophysicalProperties",
            "0/U",
            "0/p",
            "0/T",
        ]

        # fvOptions only when heat transfer is active
        if heat_transfer:
            files.append("system/fvOptions")

        # Turbulence fields
        for f in self.turbulence_fields(turb_model):
            if f == "nut":
                continue  # 0/nut rendered deterministically (Phase 4)
            files.append(f"0/{f}")

        # alphat for compressible turbulent flow
        # 0/alphat is rendered deterministically (Phase 4).

        return files

    # ── Deterministic builders (own recipe; composes base helpers) ───────

    def _build_fv_solution(self, config: dict[str, Any]) -> str:
        """rhoSimpleFoam fvSolution — SIMPLE, compressible, energy.

        Profile-aware (gas vs cryogenic).  Reads like an OpenFOAM tutorial:
        header → solvers{p, pFinal, rho, U|k|omega|h} → SIMPLE → relaxation.
        """
        ctx = self._fv_context(config)
        eq_fields = self._equation_fields(ctx.turb_model)

        p_block, _ = self._build_pressure_solver_block(ctx, is_simple=True)
        rho_block = self._build_rho_solver_block()
        eq_block, _ = self._build_equation_solver_block(eq_fields, is_simple=True)
        bounds_block = self._build_compressible_bounds(config, ctx)
        simple_block = self._build_simple_block(ctx, eq_fields, bounds_block)
        relax_block = self._build_relaxation_simple(ctx, eq_fields)

        return (
            self._foam_file_header("fvSolution")
            + "solvers\n{\n"
            + p_block
            + rho_block
            + eq_block
            + "}\n"
            + simple_block
            + relax_block
            + self._foam_file_footer()
        )

    def _build_fv_schemes(self, config: dict[str, Any]) -> str:
        """rhoSimpleFoam fvSchemes — steady, compressible divSchemes."""
        ctx = self._fv_context(config)
        return (
            self._foam_file_header("fvSchemes")
            + self._build_ddt_block(ctx) + "\n"
            + self._build_grad_block(ctx) + "\n"
            + self._build_div_block(ctx) + "\n"
            + self._build_laplacian_block(ctx) + "\n"
            + self._build_interpolation_block() + "\n"
            + self._build_sngrad_block(ctx) + "\n"
            + self._build_flux_required_block()
            + ("\n" + self._build_wall_dist_block(ctx.turb_model)
               if ctx.turb_model != "laminar" else "")
            + self._foam_file_footer()
        )

    # ── Validation ────────────────────────────────────────────────────────

    def validate(
        self, files: dict[str, str], config: dict[str, Any]
    ) -> ValidationResult:
        issues: list[ValidationIssue] = []
        fixed = dict(files)

        # Deterministic fvSolution + fvSchemes
        # Deterministic files (LLM never generates these — Phase 4)

        fixed.update(self.render_deterministic_files(config))

        # Thermo profile — drives which fixers run
        from simd_agent.run.case_spec import _thermo_profile_from_config
        profile = _thermo_profile_from_config(config)
        logger.info(f"[VALIDATE] rhoSimpleFoam: profile='{profile}'")

        # Common checks
        fixed = self._fix_controldict_solver(fixed, issues)
        fixed = self._fix_pressure_field(fixed, issues)
        fixed = self._fix_thermo_type_key(fixed, issues)
        fixed = self._fix_energy_form(fixed, issues)

        # rhoSimpleFoam-specific overrides on deterministic fvSolution
        # SIMPLEC and h relaxation only enforced under cryogenic profile.
        if profile == "cryogenic":
            fixed = self._fix_simple_consistent(fixed, issues)
            fixed = self._fix_relaxation(fixed, issues)
            fixed = self._fix_non_ortho_correctors(fixed, issues)
        fixed = self._fix_steady_end_time(fixed, issues, config)
        fixed = self._fix_pressure_internal_field(fixed, issues, config)
        fixed = self._fix_temperature_internal_field(fixed, issues, config)
        fixed = self._fix_alphat_wall_function(fixed, issues)
        fixed = self._remove_energy_fields(fixed, issues)
        fixed = self._unify_inlet_turbulence(fixed, issues, config)
        fixed = self._fix_outlet_backflow_bcs(fixed, issues, config)
        fixed = self._fix_inlet_turbulence_bc_types(fixed, issues, config)
        fixed = self._check_patch_coverage(fixed, issues, config)

        return ValidationResult(files=fixed, issues=issues)

    # ── rhoSimpleFoam-specific validators ─────────────────────────────────

    def _fix_simple_consistent(
        self, files: dict[str, str], issues: list[ValidationIssue]
    ) -> dict[str, str]:
        """Ensure SIMPLE uses standard mode (consistent no).

        consistent yes causes the second corrector to restart at residual ~1.0,
        over-correcting velocity and recreating divergence.
        """
        fvs = files.get("system/fvSolution", "")
        if not fvs:
            return files

        if re.search(r"\bconsistent\s+yes\s*;", fvs):
            fvs = re.sub(r"\bconsistent\s+yes\s*;", "consistent       no;", fvs)
            files["system/fvSolution"] = fvs
            issues.append(
                ValidationIssue(
                    "warning",
                    "system/fvSolution",
                    "Changed 'consistent yes' -> 'consistent no' (SIMPLEC causes divergence).",
                )
            )
        return files

    def _fix_relaxation(
        self, files: dict[str, str], issues: list[ValidationIssue]
    ) -> dict[str, str]:
        """Enforce safe relaxation factors for rhoSimpleFoam.

        h relaxation MUST be 0.05 — with T < 298.15K (h < 0), any non-zero
        div(phi) produces large artificial cooling.
        """
        fvs = files.get("system/fvSolution", "")
        if not fvs:
            return files

        # Check h relaxation — must be 0.05
        h_match = re.search(r"\bh\s+([\d.]+)\s*;", fvs)
        if h_match:
            try:
                h_val = float(h_match.group(1))
                if h_val > 0.1:
                    fvs = re.sub(r"\bh\s+[\d.]+\s*;", "h               0.05;", fvs)
                    files["system/fvSolution"] = fvs
                    issues.append(
                        ValidationIssue(
                            "warning",
                            "system/fvSolution",
                            f"h relaxation changed from {h_val} -> 0.05 (prevents T crash).",
                        )
                    )
            except ValueError:
                pass
        return files

    def _fix_non_ortho_correctors(
        self, files: dict[str, str], issues: list[ValidationIssue]
    ) -> dict[str, str]:
        """nNonOrthogonalCorrectors MUST be 2 for rhoSimpleFoam.

        Fewer correctors leave large div(phi) -> h * div(phi) artificial cooling.
        """
        fvs = files.get("system/fvSolution", "")
        if not fvs:
            return files

        match = re.search(r"nNonOrthogonalCorrectors\s+(\d+)\s*;", fvs)
        if match:
            val = int(match.group(1))
            if val < 2:
                fvs = re.sub(
                    r"nNonOrthogonalCorrectors\s+\d+\s*;",
                    "nNonOrthogonalCorrectors 2;",
                    fvs,
                )
                files["system/fvSolution"] = fvs
                issues.append(
                    ValidationIssue(
                        "warning",
                        "system/fvSolution",
                        f"nNonOrthogonalCorrectors changed from {val} -> 2 (required for rhoSimpleFoam).",
                    )
                )
        return files

    def _fix_steady_end_time(
        self,
        files: dict[str, str],
        issues: list[ValidationIssue],
        config: dict[str, Any],
    ) -> dict[str, str]:
        """Ensure endTime is integer for steady solver."""
        cd = files.get("system/controlDict", "")
        if not cd:
            return files
        match = re.search(r"endTime\s+([\d.eE+\-]+)\s*;", cd)
        if match:
            val = match.group(1)
            try:
                fval = float(val)
                ival = int(fval)
                if fval != ival or "." in val:
                    files["system/controlDict"] = re.sub(
                        r"endTime\s+[\d.eE+\-]+\s*;",
                        f"endTime     {ival};",
                        cd,
                    )
                    issues.append(
                        ValidationIssue(
                            "warning",
                            "system/controlDict",
                            f"endTime fixed to integer {ival} for steady solver.",
                        )
                    )
            except ValueError:
                pass
        return files

    def _fix_pressure_internal_field(
        self,
        files: dict[str, str],
        issues: list[ValidationIssue],
        config: dict[str, Any],
    ) -> dict[str, str]:
        """0/p internalField MUST equal outlet fixedValue pressure.

        Mismatch causes SIGFPE in GAMGSolver::scale on iteration 1.
        """
        p_content = files.get("0/p", "")
        if not p_content:
            return files

        bcs = config.get("boundary_conditions", {})
        outlet_p = None
        for pname, pbc in bcs.items():
            if not isinstance(pbc, dict):
                continue
            pt = pbc.get("patch_type", "")
            if pt in ("outlet", "pressure_outlet") or pname == "outlet":
                p_entry = pbc.get("pressure") or pbc.get("p")
                if isinstance(p_entry, dict):
                    outlet_p = p_entry.get("value") or p_entry.get("uniform")
                elif isinstance(p_entry, (int, float)):
                    outlet_p = p_entry
                break

        if outlet_p is not None:
            try:
                outlet_p_val = float(outlet_p)
                internal_match = re.search(
                    r"internalField\s+uniform\s+([\d.eE+\-]+)\s*;", p_content
                )
                if internal_match:
                    internal_val = float(internal_match.group(1))
                    if abs(internal_val - outlet_p_val) > 1.0:
                        p_content = re.sub(
                            r"internalField\s+uniform\s+[\d.eE+\-]+\s*;",
                            f"internalField   uniform {outlet_p_val};",
                            p_content,
                        )
                        files["0/p"] = p_content
                        issues.append(
                            ValidationIssue(
                                "warning",
                                "0/p",
                                f"internalField changed from {internal_val} to {outlet_p_val} "
                                f"(must match outlet pressure to avoid SIGFPE).",
                            )
                        )
            except (TypeError, ValueError):
                pass
        return files

    def _fix_temperature_internal_field(
        self,
        files: dict[str, str],
        issues: list[ValidationIssue],
        config: dict[str, Any],
    ) -> dict[str, str]:
        """0/T internalField MUST equal inlet temperature.

        At room temp for LN2 (inlet=77K): rho = 1167.9 - 4.7*300 = -242 kg/m3
        -> SIGFPE on iteration 0.
        """
        t_content = files.get("0/T", "")
        if not t_content:
            return files

        bcs = config.get("boundary_conditions", {})
        inlet_t = None
        for pname, pbc in bcs.items():
            if not isinstance(pbc, dict):
                continue
            pt = pbc.get("patch_type", "")
            if pt in ("inlet", "pressure_inlet", "mass_flow_inlet") or pname == "inlet":
                t_entry = pbc.get("temperature") or pbc.get("T")
                if isinstance(t_entry, dict):
                    inlet_t = t_entry.get("value") or t_entry.get("uniform")
                elif isinstance(t_entry, (int, float)):
                    inlet_t = t_entry
                break

        if inlet_t is not None:
            try:
                inlet_t_val = float(inlet_t)
                internal_match = re.search(
                    r"internalField\s+uniform\s+([\d.eE+\-]+)\s*;", t_content
                )
                if internal_match:
                    internal_val = float(internal_match.group(1))
                    # If T is wildly different from inlet (e.g. 300K vs 77K)
                    if abs(internal_val - inlet_t_val) > 50.0:
                        t_content = re.sub(
                            r"internalField\s+uniform\s+[\d.eE+\-]+\s*;",
                            f"internalField   uniform {inlet_t_val};",
                            t_content,
                        )
                        files["0/T"] = t_content
                        issues.append(
                            ValidationIssue(
                                "warning",
                                "0/T",
                                f"internalField changed from {internal_val}K to {inlet_t_val}K "
                                f"(must match inlet to avoid negative rho -> SIGFPE).",
                            )
                        )
            except (TypeError, ValueError):
                pass
        return files

    def _fix_alphat_wall_function(
        self, files: dict[str, str], issues: list[ValidationIssue]
    ) -> dict[str, str]:
        """Fix alphatWallFunction -> compressible::alphatWallFunction.

        OpenFOAM 2406 ESI requires the namespace-qualified BC type.
        """
        alphat = files.get("0/alphat", "")
        if not alphat:
            return files

        if "alphatWallFunction" in alphat and "compressible::alphatWallFunction" not in alphat:
            fixed = alphat.replace(
                "alphatWallFunction", "compressible::alphatWallFunction"
            )
            files["0/alphat"] = fixed
            issues.append(
                ValidationIssue(
                    "warning",
                    "0/alphat",
                    "Fixed: alphatWallFunction -> compressible::alphatWallFunction (OF 2406).",
                )
            )
        return files


    def _fix_energy_form(
        self, files: dict[str, str], issues: list[ValidationIssue]
    ) -> dict[str, str]:
        """Force ``energy sensibleInternalEnergy`` in thermophysicalProperties.

        rhoSimpleFoam uses ``energy_var = "e"`` to match the OpenFOAM
        reference tutorials.  The deterministic fvSchemes / fvSolution
        renderers already emit ``div(phi,e)`` and ``e`` in the
        residualControl + relaxation blocks; if the LLM-emitted thermo
        file still declares ``sensibleEnthalpy`` (the rhoSimpleFoam
        prompt template used to default to this), OpenFOAM will
        complain that the energy field name and the thermoType disagree.

        Auto-correct the thermo dict in-place — no LLM round-trip needed.
        """
        tp_path = "constant/thermophysicalProperties"
        tp = files.get(tp_path, "")
        if not tp:
            return files

        if "sensibleEnthalpy" in tp:
            new_tp = tp.replace("sensibleEnthalpy", "sensibleInternalEnergy")
            files[tp_path] = new_tp
            issues.append(
                ValidationIssue(
                    "warning",
                    tp_path,
                    "Changed 'energy sensibleEnthalpy' → "
                    "'energy sensibleInternalEnergy' to match the "
                    "rhoSimpleFoam reference tutorials (and the "
                    "deterministic div(phi,e) / residualControl e).",
                )
            )
        # If the file uses ``thermo hConst`` paired with sensibleInternalEnergy,
        # nothing else needs adjustment — ``hConst`` is valid for both energy
        # forms (Cp ≈ Cv to within R for an ideal gas).  We leave Cp values
        # alone; OpenFOAM converts internally.
        return files

    def _remove_energy_fields(
        self, files: dict[str, str], issues: list[ValidationIssue]
    ) -> dict[str, str]:
        """Remove 0/h and 0/e — thermo reads 0/T."""
        for ef in ("0/h", "0/e"):
            if ef in files:
                del files[ef]
                issues.append(
                    ValidationIssue(
                        "warning",
                        ef,
                        f"Removed {ef}: thermo initialises h/e from 0/T at startup.",
                    )
                )
        return files


    def _check_patch_coverage(
        self,
        files: dict[str, str],
        issues: list[ValidationIssue],
        config: dict[str, Any],
    ) -> dict[str, str]:
        """Warn if any 0/ file is missing patch entries."""
        bcs = config.get("boundary_conditions", {})
        expected = set(bcs.keys())
        if not expected:
            return files

        for fpath, content in list(files.items()):
            if not fpath.startswith("0/"):
                continue
            for patch_name in expected:
                if patch_name not in content:
                    issues.append(
                        ValidationIssue(
                            "warning",
                            fpath,
                            f"Patch '{patch_name}' not found in {fpath}.",
                        )
                    )
        return files

    # ── Helpers ────────────────────────────────────────────────────────────


    @staticmethod
    def _has_heat_transfer(config: dict[str, Any]) -> bool:
        physics = config.get("physics", {}) or {}
        return bool(
            config.get("heat_transfer")
            or physics.get("heat_transfer")
        )
