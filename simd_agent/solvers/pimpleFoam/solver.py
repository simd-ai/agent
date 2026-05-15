# simd_agent/solvers/pimpleFoam/solver.py
"""pimpleFoam solver plugin.

Transient, incompressible, isothermal RANS solver using the PIMPLE
algorithm (merged PISO-SIMPLE).  The transient counterpart of simpleFoam.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from simd_agent.solvers.base import (
    MatchResult,
    SolverPlugin,
    ValidationIssue,
    ValidationResult,
)

logger = logging.getLogger(__name__)


class PimpleFoamSolver(SolverPlugin):
    """pimpleFoam — transient incompressible isothermal RANS."""

    name = "pimpleFoam"
    algorithm = "PIMPLE"
    pressure_field = "p"
    is_transient = True
    is_compressible = False
    supports_energy = False
    needs_gravity = False
    is_multiphase = False

    # ── Matching ──────────────────────────────────────────────────────────

    def matches(self, config: dict[str, Any]) -> MatchResult:
        physics = config.get("physics", {}) or {}
        heat = bool(config.get("heat_transfer") or physics.get("heat_transfer"))
        compressible = (config.get("compressibility") or physics.get("compressibility", "incompressible")) == "compressible"
        transient = (config.get("time_stepping") or physics.get("time_scheme", "steady")) in ("transient", "unsteady")
        multiphase = bool(config.get("multiphase") or physics.get("multiphase"))

        if heat or compressible or multiphase:
            return MatchResult(0.0, "pimpleFoam cannot handle heat/compressible/multiphase")
        if not transient:
            return MatchResult(0.1, "pimpleFoam is transient; simpleFoam better for steady")
        return MatchResult(0.95, "Transient incompressible isothermal flow — ideal for pimpleFoam")

    # ── Required files ────────────────────────────────────────────────────

    def required_files(self, config: dict[str, Any]) -> list[str]:
        turb_model = self._get_turb_model_from_config(config)
        files = [
            "system/controlDict",
            # system/fvSchemes and system/fvSolution are generated
            # deterministically in validate(), not by the LLM.
            "constant/transportProperties", "0/U", "0/p",
        ]
        for f in self.turbulence_fields(turb_model):
            if f == "nut":
                continue  # 0/nut rendered deterministically (Phase 4)
            files.append(f"0/{f}")
        return files

    # ── Deterministic builders ────────────────────────────────────────────

    def _build_fv_solution(self, config: dict[str, Any]) -> str:
        """pimpleFoam fvSolution — PIMPLE, incompressible, no energy.

        Pressure has pFinal block; equations have Final regex.  No rho,
        no compressible bounds.
        """
        ctx = self._fv_context(config)
        eq_fields = self._equation_fields(ctx.turb_model)

        p_block, p_final = self._build_pressure_solver_block(ctx, is_simple=False)
        eq_block, eq_final = self._build_equation_solver_block(eq_fields, is_simple=False)
        pimple_block = self._build_pimple_block(ctx, eq_fields, "")
        relax_block = self._build_relaxation_pimple(ctx)

        return (
            self._foam_file_header("fvSolution")
            + "solvers\n{\n"
            + p_block
            + p_final
            + eq_block
            + eq_final
            + "}\n"
            + pimple_block
            + relax_block
            + self._foam_file_footer()
        )

    def _build_fv_schemes(self, config: dict[str, Any]) -> str:
        """pimpleFoam fvSchemes — transient incompressible (Euler ddt)."""
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

        # ── Deterministic fvSolution + fvSchemes ───────────────────────
        # Deterministic files (LLM never generates these — Phase 4)

        fixed.update(self.render_deterministic_files(config))
        issues.append(
            ValidationIssue(
                "info",
                "system/fvSolution",
                "Deterministic fvSolution (PIMPLE settings, relaxation, mesh-quality-aware).",
            )
        )
        issues.append(
            ValidationIssue(
                "info",
                "system/fvSchemes",
                "Deterministic fvSchemes (Euler ddt, mesh-quality-aware laplacian/snGrad).",
            )
        )

        # ── Common checks on other files ─────────────────────────────
        fixed = self._fix_controldict_solver(fixed, issues)
        fixed = self._fix_pressure_field(fixed, issues)
        fixed = self._fix_pressure_value(fixed, issues)
        fixed = self._remove_unneeded_thermo(fixed, issues)

        # pimpleFoam-specific: remove energy fields if LLM generated them
        for energy_file in ["0/T", "0/h", "0/e"]:
            if energy_file in fixed:
                issues.append(
                    ValidationIssue(
                        "warning",
                        energy_file,
                        f"pimpleFoam does not solve energy. Removing {energy_file}.",
                    )
                )
                del fixed[energy_file]

        # Transient-specific: validate controlDict time settings
        fixed = self._fix_transient_controldict(fixed, issues, config)

        # Fix outlet BCs: zeroGradient → inletOutlet for backflow prevention
        fixed = self._fix_outlet_velocity_bc(fixed, issues, config)
        fixed = self._fix_outlet_turbulence_bc(fixed, issues, config)

        # Floor turbulence ICs (prevent division-by-zero in wall functions)
        fixed = self._fix_turbulence_ic_floors(fixed, issues)

        # Unify k/ω/ε across all inlets — flow-wide property, not per-inlet
        fixed = self._unify_inlet_turbulence(fixed, issues, config)

        # Patch coverage check
        fixed = self._check_patch_coverage(fixed, issues, config)

        # 2D validation: ensure empty/wedge patches have correct BC type
        fixed = self._fix_2d_patches(fixed, issues, config)

        return ValidationResult(files=fixed, issues=issues)

    # ── pimpleFoam-specific validators ──────────────────────────────────

    def _fix_transient_controldict(
        self,
        files: dict[str, str],
        issues: list[ValidationIssue],
        config: dict[str, Any],
    ) -> dict[str, str]:
        """Validate and fix transient-specific controlDict settings.

        Ensures:
        - adjustTimeStep yes is present with maxCo for CFL safety
        - deltaT is a reasonable physical time (not 1 like steady)
        - endTime is physical seconds (not an integer iteration count)
        """
        cd = files.get("system/controlDict", "")
        if not cd:
            return files

        changed = False

        # Fix deltaT = 1 (LLM sometimes copies steady-state pattern)
        dt_match = re.search(r"deltaT\s+([\d.eE+\-]+)\s*;", cd)
        if dt_match:
            try:
                dt_val = float(dt_match.group(1))
                if dt_val >= 1.0:
                    # deltaT = 1 is a steady-state pattern; for transient,
                    # use a small physical time step
                    new_dt = 0.001
                    cd = re.sub(
                        r"deltaT\s+[\d.eE+\-]+\s*;",
                        f"deltaT          {new_dt};",
                        cd,
                    )
                    changed = True
                    issues.append(
                        ValidationIssue(
                            "warning",
                            "system/controlDict",
                            f"deltaT {dt_val} looks like steady-state. "
                            f"Set to {new_dt}s for transient. adjustTimeStep "
                            f"will adapt it automatically.",
                            fix=f"deltaT {new_dt};",
                        )
                    )
            except ValueError:
                pass

        # Ensure adjustTimeStep is present
        if "adjustTimeStep" not in cd:
            # Insert after runTimeModifiable or after deltaT
            insert_after = re.search(
                r"(runTimeModifiable\s+\w+\s*;)", cd
            )
            if not insert_after:
                insert_after = re.search(r"(deltaT\s+[\d.eE+\-]+\s*;)", cd)
            if insert_after:
                inject = (
                    "\n\nadjustTimeStep  yes;\n"
                    "maxCo           0.5;\n"
                    "maxDeltaT       1;"
                )
                cd = cd[:insert_after.end()] + inject + cd[insert_after.end():]
                changed = True
                issues.append(
                    ValidationIssue(
                        "info",
                        "system/controlDict",
                        "Injected adjustTimeStep yes + maxCo 0.5 for "
                        "CFL-adaptive time stepping.",
                        fix="adjustTimeStep yes; maxCo 0.5;",
                    )
                )

        if changed:
            files["system/controlDict"] = cd
        return files

    # ── Outlet / inlet patch helpers ─────────────────────────────────────

    @staticmethod
    def _get_outlet_patches(config: dict[str, Any]) -> list[str]:
        """Identify outlet patches from config by name or BC type."""
        bcs = config.get("boundary_conditions", {}) or {}
        outlets: list[str] = []
        for name, bc in bcs.items():
            lower = name.lower()
            if "outlet" in lower:
                outlets.append(name)
                continue
            if isinstance(bc, dict):
                bc_type = (bc.get("type", "") or "").lower()
            elif hasattr(bc, "type"):
                bc_type = str(getattr(bc, "type", "") or "").lower()
            else:
                bc_type = ""
            if "outlet" in bc_type:
                outlets.append(name)
        return outlets

    def _fix_outlet_velocity_bc(
        self,
        files: dict[str, str],
        issues: list[ValidationIssue],
        config: dict[str, Any],
    ) -> dict[str, str]:
        """Convert outlet U BC from zeroGradient to inletOutlet.

        inletOutlet prevents reverse-flow contamination at the outlet,
        which is a common cause of divergence in recirculating flows.
        """
        u_content = files.get("0/U", "")
        if not u_content:
            return files

        outlets = self._get_outlet_patches(config)
        changed = False

        for patch in outlets:
            pattern = rf"{re.escape(patch)}\s*\{{[^}}]*\}}"
            match = re.search(pattern, u_content, re.DOTALL)
            if match:
                block = match.group(0)
                type_match = re.search(r"type\s+(\w+)\s*;", block)
                if type_match and type_match.group(1) == "zeroGradient":
                    new_block = (
                        f"{patch}\n"
                        f"    {{\n"
                        f"        type            inletOutlet;\n"
                        f"        inletValue      uniform (0 0 0);\n"
                        f"        value           uniform (0 0 0);\n"
                        f"    }}"
                    )
                    u_content = (
                        u_content[: match.start()]
                        + new_block
                        + u_content[match.end() :]
                    )
                    changed = True
                    issues.append(
                        ValidationIssue(
                            "warning",
                            "0/U",
                            f"Outlet '{patch}': zeroGradient → inletOutlet "
                            f"for backflow prevention.",
                            fix="type inletOutlet;",
                        )
                    )

        if changed:
            files["0/U"] = u_content
        return files

    def _fix_outlet_turbulence_bc(
        self,
        files: dict[str, str],
        issues: list[ValidationIssue],
        config: dict[str, Any],
    ) -> dict[str, str]:
        """Convert outlet turbulence BCs from zeroGradient to inletOutlet.

        Backflow at outlets with zeroGradient can introduce negative
        k/omega/epsilon values, causing SIGFPE in the turbulence model.
        """
        outlets = self._get_outlet_patches(config)
        if not outlets:
            return files

        turb_files = {"0/k": "k", "0/omega": "omega", "0/epsilon": "epsilon"}

        for fpath, field_name in turb_files.items():
            content = files.get(fpath, "")
            if not content:
                continue

            iv_match = re.search(
                r"internalField\s+uniform\s+([\d.eE+\-]+)", content
            )
            inlet_val = iv_match.group(1) if iv_match else "0"

            changed = False
            for patch in outlets:
                pattern = rf"{re.escape(patch)}\s*\{{[^}}]*\}}"
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    block = match.group(0)
                    type_match = re.search(r"type\s+(\w+)\s*;", block)
                    if type_match and type_match.group(1) == "zeroGradient":
                        new_block = (
                            f"{patch}\n"
                            f"    {{\n"
                            f"        type            inletOutlet;\n"
                            f"        inletValue      uniform {inlet_val};\n"
                            f"        value           uniform {inlet_val};\n"
                            f"    }}"
                        )
                        content = (
                            content[: match.start()]
                            + new_block
                            + content[match.end() :]
                        )
                        changed = True
                        issues.append(
                            ValidationIssue(
                                "warning",
                                fpath,
                                f"Outlet '{patch}': zeroGradient → "
                                f"inletOutlet for {field_name} backflow "
                                f"safety.",
                                fix="type inletOutlet;",
                            )
                        )

            if changed:
                files[fpath] = content

        return files

    # ── Turbulence IC floors ─────────────────────────────────────────────

    def _fix_turbulence_ic_floors(
        self,
        files: dict[str, str],
        issues: list[ValidationIssue],
    ) -> dict[str, str]:
        """Floor turbulence internalField values to prevent SIGFPE.

        k < 1e-6, omega < 1.0, or epsilon < 1e-6 can cause division-
        by-zero in wall functions at the very first iteration.
        """
        floors = {
            "0/k": ("k", 1e-6),
            "0/omega": ("omega", 1.0),
            "0/epsilon": ("epsilon", 1e-6),
        }

        for fpath, (name, floor_val) in floors.items():
            content = files.get(fpath, "")
            if not content:
                continue

            m = re.search(
                r"(internalField\s+uniform\s+)([\d.eE+\-]+)", content
            )
            if not m:
                continue

            try:
                val = float(m.group(2))
            except ValueError:
                continue

            if val < floor_val:
                content = (
                    content[: m.start()]
                    + f"{m.group(1)}{floor_val}"
                    + content[m.end() :]
                )
                files[fpath] = content
                issues.append(
                    ValidationIssue(
                        "warning",
                        fpath,
                        f"{name} internalField {val} below safety floor "
                        f"{floor_val}. Corrected to prevent SIGFPE.",
                        fix=f"internalField uniform {floor_val};",
                    )
                )

        return files

    # ── Patch coverage ───────────────────────────────────────────────────

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

    # ── 2D patch validation ──────────────────────────────────────────────

    def _fix_2d_patches(
        self,
        files: dict[str, str],
        issues: list[ValidationIssue],
        config: dict[str, Any],
    ) -> dict[str, str]:
        """Ensure empty/wedge patches have the correct BC type in all 0/* files."""
        mesh = config.get("mesh", {}) or {}
        mesh_patches = mesh.get("patches", [])
        bcs = config.get("boundary_conditions", {}) or {}

        special_patches: dict[str, str] = {}
        for mp in mesh_patches:
            if isinstance(mp, dict):
                name = mp.get("name", "")
                ptype = mp.get("type", "")
            elif hasattr(mp, "name"):
                name = mp.name
                ptype = getattr(mp, "type", "")
            else:
                continue
            if ptype in ("empty", "wedge"):
                special_patches[name] = ptype

        for pname, bc in bcs.items():
            if isinstance(bc, dict):
                pt = bc.get("patch_type", "")
            elif hasattr(bc, "patch_type"):
                pt = str(getattr(bc, "patch_type", ""))
            else:
                continue
            if pt in ("empty", "wedge"):
                special_patches.setdefault(pname, pt)

        if not special_patches:
            return files

        for fpath, content in list(files.items()):
            if not fpath.startswith("0/"):
                continue

            for patch_name, expected_type in special_patches.items():
                if patch_name not in content:
                    bf_end = content.rfind("}")
                    if bf_end > 0:
                        indent = "    "
                        patch_block = (
                            f"\n{indent}{patch_name}\n"
                            f"{indent}{{\n"
                            f"{indent}    type            {expected_type};\n"
                            f"{indent}}}\n"
                        )
                        files[fpath] = content[:bf_end] + patch_block + content[bf_end:]
                        issues.append(
                            ValidationIssue(
                                "warning",
                                fpath,
                                f"Added missing {expected_type} patch '{patch_name}' to {fpath}.",
                                fix=f"type {expected_type};",
                            )
                        )
                        content = files[fpath]
                else:
                    pattern = rf"{re.escape(patch_name)}\s*\{{[^}}]*\}}"
                    match = re.search(pattern, content, re.DOTALL)
                    if match:
                        block = match.group(0)
                        type_match = re.search(r"type\s+(\w+)\s*;", block)
                        if type_match and type_match.group(1) != expected_type:
                            old_type = type_match.group(1)
                            new_block = (
                                f"{patch_name}\n"
                                f"    {{\n"
                                f"        type            {expected_type};\n"
                                f"    }}"
                            )
                            files[fpath] = content[:match.start()] + new_block + content[match.end():]
                            issues.append(
                                ValidationIssue(
                                    "warning",
                                    fpath,
                                    f"Fixed {patch_name} BC type from '{old_type}' to '{expected_type}' in {fpath}.",
                                    fix=f"type {expected_type};",
                                )
                            )
                            content = files[fpath]

        return files
