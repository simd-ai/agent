# simd_agent/solvers/simpleFoam/solver.py
"""simpleFoam solver plugin.

Steady-state, incompressible, isothermal RANS solver using the SIMPLE
algorithm.  The simplest OpenFOAM solver — no energy equation, no
thermodynamics, no gravity.
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


class SimpleFoamSolver(SolverPlugin):
    """simpleFoam — steady incompressible isothermal RANS."""

    name = "simpleFoam"
    algorithm = "SIMPLE"
    pressure_field = "p"
    is_transient = False
    is_compressible = False
    supports_energy = False
    needs_gravity = False
    is_multiphase = False

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

        # Disqualifiers
        if heat or compressible or multiphase:
            return MatchResult(0.0, "simpleFoam cannot handle heat/compressible/multiphase")

        if transient:
            return MatchResult(
                0.1,
                "simpleFoam is steady-state; pimpleFoam would be better for transient",
                warnings=["Consider pimpleFoam for transient simulations"],
            )

        # Perfect match: steady, incompressible, no heat, no multiphase
        return MatchResult(0.95, "Steady incompressible isothermal flow — ideal for simpleFoam")

    # ── Required files ────────────────────────────────────────────────────

    def required_files(self, config: dict[str, Any]) -> list[str]:
        turb_model = self._get_turb_model_from_config(config)

        files = [
            "system/controlDict",
            # system/fvSchemes and system/fvSolution are NOT here — they are
            # generated deterministically in validate(), not by the LLM.
            "constant/transportProperties",
            "constant/turbulenceProperties",
            "0/U",
            "0/p",
        ]

        # Turbulence fields
        for f in self.turbulence_fields(turb_model):
            files.append(f"0/{f}")

        return files

    # ── Validation ────────────────────────────────────────────────────────

    def validate(
        self, files: dict[str, str], config: dict[str, Any]
    ) -> ValidationResult:
        issues: list[ValidationIssue] = []
        fixed = dict(files)

        # ── Deterministic fvSolution + fvSchemes ───────────────────────
        # Generated entirely from config — LLM never touches these files.
        # Any LLM-generated versions are discarded.
        # Both methods live in SolverPlugin base class — single source of
        # truth for all solvers (mesh-quality-aware, velocity-aware).
        fixed["system/fvSolution"] = self._build_fv_solution(config)
        fixed["system/fvSchemes"] = self._build_fv_schemes(config)
        issues.append(
            ValidationIssue(
                "info",
                "system/fvSolution",
                "Deterministic fvSolution (solver settings, relaxation, mesh-quality-aware).",
            )
        )
        issues.append(
            ValidationIssue(
                "info",
                "system/fvSchemes",
                "Deterministic fvSchemes (mesh-quality-aware laplacian/snGrad).",
            )
        )

        # ── Common checks on other files ─────────────────────────────
        fixed = self._fix_controldict_solver(fixed, issues)
        fixed = self._fix_pressure_field(fixed, issues)
        fixed = self._fix_pressure_value(fixed, issues)
        fixed = self._remove_unneeded_thermo(fixed, issues)

        # simpleFoam-specific: remove energy field if LLM generated it
        for energy_file in ["0/T", "0/h", "0/e"]:
            if energy_file in fixed:
                issues.append(
                    ValidationIssue(
                        "warning",
                        energy_file,
                        f"simpleFoam does not solve energy. Removing {energy_file}.",
                    )
                )
                del fixed[energy_file]

        # Ensure endTime is integer for steady solver
        fixed = self._fix_steady_end_time(fixed, issues, config)

        # Initialize U with inlet velocity for faster convergence
        fixed = self._fix_velocity_initialization(fixed, issues, config)

        # Fix outlet BCs: zeroGradient → inletOutlet for backflow prevention
        fixed = self._fix_outlet_velocity_bc(fixed, issues, config)
        fixed = self._fix_outlet_turbulence_bc(fixed, issues, config)

        # Floor turbulence ICs (prevent division-by-zero in wall functions)
        fixed = self._fix_turbulence_ic_floors(fixed, issues)

        # Patch coverage check
        fixed = self._check_patch_coverage(fixed, issues, config)

        # 2D validation: ensure empty/wedge patches have correct BC type
        fixed = self._fix_2d_patches(fixed, issues, config)

        return ValidationResult(files=fixed, issues=issues)

    # ── simpleFoam-specific validators ───────────────────────────────────

    def _fix_steady_end_time(
        self,
        files: dict[str, str],
        issues: list[ValidationIssue],
        config: dict[str, Any],
    ) -> dict[str, str]:
        """Ensure endTime is an integer for steady-state solvers."""
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
                    issues.append(
                        ValidationIssue(
                            "warning",
                            "system/controlDict",
                            f"endTime should be integer for steady solver. Fixing {val} -> {ival}.",
                            fix=f"endTime {ival};",
                        )
                    )
                    files["system/controlDict"] = re.sub(
                        r"endTime\s+[\d.eE+\-]+\s*;",
                        f"endTime     {ival};",
                        cd,
                    )
            except ValueError:
                pass
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

    def _fix_2d_patches(
        self,
        files: dict[str, str],
        issues: list[ValidationIssue],
        config: dict[str, Any],
    ) -> dict[str, str]:
        """Ensure empty/wedge patches have the correct BC type in all 0/* files.

        For 2D cases, every field file must have the empty/wedge patches with
        the matching BC type.  If the LLM used a wrong type (e.g. fixedValue
        instead of empty), fix it.
        """
        mesh = config.get("mesh", {}) or {}
        mesh_patches = mesh.get("patches", [])
        bcs = config.get("boundary_conditions", {}) or {}

        # Build map of patch name → expected 2D type ("empty" or "wedge")
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

        # Also check from BC patch_type
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
                    # Patch missing entirely — add it
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
                    # Patch exists — verify it has the correct type
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

    def _fix_velocity_initialization(
        self,
        files: dict[str, str],
        issues: list[ValidationIssue],
        config: dict[str, Any],
    ) -> dict[str, str]:
        """Set U internalField to inlet velocity instead of (0 0 0).

        Steady-state solvers converge significantly faster when initialized
        with a reasonable velocity field rather than starting from rest.
        """
        u_content = files.get("0/U", "")
        if not u_content:
            return files

        # Only fix if currently (0 0 0)
        m = re.search(
            r"internalField\s+uniform\s+\(\s*0\s+0\s+0\s*\)", u_content
        )
        if not m:
            return files

        # Extract inlet velocity from inlet patch fixedValue in the
        # generated file itself — this is the most reliable source.
        bcs = config.get("boundary_conditions", {}) or {}
        inlet_vel = None
        for name in bcs:
            if "inlet" not in name.lower():
                continue
            pat = rf"{re.escape(name)}\s*\{{[^}}]*\}}"
            pm = re.search(pat, u_content, re.DOTALL)
            if pm:
                vm = re.search(
                    r"value\s+uniform\s+\(([^)]+)\)", pm.group(0)
                )
                if vm:
                    vel = vm.group(1).strip()
                    if vel != "0 0 0":
                        inlet_vel = vel
                        break

        if not inlet_vel:
            return files

        files["0/U"] = (
            u_content[: m.start()]
            + f"internalField   uniform ({inlet_vel})"
            + u_content[m.end() :]
        )
        issues.append(
            ValidationIssue(
                "warning",
                "0/U",
                f"Initialized U with inlet velocity ({inlet_vel}) instead "
                f"of (0 0 0) for faster steady-state convergence.",
                fix=f"internalField uniform ({inlet_vel});",
            )
        )
        return files

    # ── Outlet BC fixes ──────────────────────────────────────────────────

    def _fix_outlet_velocity_bc(
        self,
        files: dict[str, str],
        issues: list[ValidationIssue],
        config: dict[str, Any],
    ) -> dict[str, str]:
        """Convert outlet U BC from zeroGradient to inletOutlet.

        inletOutlet prevents reverse-flow contamination at the outlet,
        which is a common cause of divergence in recirculating flows
        (backward-facing step, sudden expansion, cylinder wake).
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
        inletOutlet uses internalField values for reverse flow.
        """
        outlets = self._get_outlet_patches(config)
        if not outlets:
            return files

        turb_files = {"0/k": "k", "0/omega": "omega", "0/epsilon": "epsilon"}

        for fpath, field_name in turb_files.items():
            content = files.get(fpath, "")
            if not content:
                continue

            # Use internalField value as inletValue
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
