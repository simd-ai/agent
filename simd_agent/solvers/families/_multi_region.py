"""Multi-region family — chtMultiRegion{Simple,}Foam.

Conjugate heat transfer (CHT) solvers integrate multiple regions
simultaneously: typically one or more fluids (Navier–Stokes + energy)
coupled with one or more solids (heat conduction only) via mapped
temperature boundary conditions at fluid–solid interfaces.

This base class captures what makes a multi-region solver structurally
different from every other one:

  * **Tree-structured file output** — instead of a flat
    ``{system,constant,0}/<file>``, every region gets its own subtree:
    ``constant/<region>/<file>``, ``system/<region>/<file>``,
    ``0/<region>/<field>``.
  * **regionProperties** lists all regions and their kinds (fluid /
    solid) — read at startup by the solver.
  * **Per-region thermo + turbulence** — fluid regions use
    ``heRhoThermo`` (or similar); solid regions use ``heSolidThermo``
    with no momentum equation and no turbulence.
  * **Coupled BCs** at fluid–solid interfaces — the temperature is
    matched via ``compressible::turbulentTemperatureCoupledBaffleMixed``
    on both sides.

Concrete plugins:

  * ``ChtMultiRegionSimpleFoamSolver(SteadyBase, MultiRegionBase)``
  * ``ChtMultiRegionFoamSolver(TransientBase, MultiRegionBase)``

Status — **architectural skeleton in place**:

  * RegionSpec / CaseRegions strategy ✓
  * Region-tree required_files() ✓
  * regionProperties renderer ✓
  * Per-region thermophysicalProperties rendering ✓ (fluid + solid)
  * Per-region fvSchemes / fvSolution rendering        — TODO (Phase 2)
  * Mapped fluid-solid coupled-BC generation           — TODO (Phase 2)
  * changeDictionaryDict per region                    — TODO (Phase 2)
  * Orchestrator handling of tree-structured files     — TODO (Phase 2)
  * packaging.py multi-region case zip layout          — TODO (Phase 2)

Phase 2 lands the missing pieces case-by-case; this Phase 1 commit
locks the typed contract + identity so the API surface is stable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from simd_agent.solvers.base import SolverPlugin

if TYPE_CHECKING:
    from simd_agent.run.case_spec.strategies import CaseRegions, RegionSpec


class MultiRegionBase(SolverPlugin):
    """Abstract base for multi-region (CHT) solvers.

    Subclasses set ``algorithm`` (SIMPLE / PIMPLE) via a family base
    composed alongside ``MultiRegionBase``.  Use mixin-first ordering:
    ``(MultiRegionBase, SteadyBase)`` so any multi-region overrides win
    over the single-region defaults.
    """

    # Multi-region solvers ALWAYS have an energy equation and ALWAYS need
    # gravity (the buoyancy in fluid regions is what makes CHT interesting).
    supports_energy: bool = True
    needs_gravity: bool = True
    is_compressible: bool = True
    pressure_field: str = "p_rgh"
    energy_var: str = "h"
    # Marks the plugin so the orchestrator / packaging layers can detect
    # multi-region cases and route them through the tree-structured file
    # handler (Phase 2 — orchestration.py / packaging.py work).
    is_multi_region: bool = True

    # ── Region extraction from config ─────────────────────────────────────

    def extract_regions(self, config: dict[str, Any]) -> "CaseRegions":
        """Build the ``CaseRegions`` strategy from a normalised config.

        Config shape expected:

            config["regions"] = {
                "fluid": [
                    {"name": "topAir", "thermo_profile": "gas",
                     "turbulence_model": "kEpsilon", "Cp": 1006, ...},
                    ...
                ],
                "solid": [
                    {"name": "heater", "rho_solid": 8000,
                     "kappa_solid": 80, "Cp_solid": 450},
                    ...
                ],
            }

        Falls back to a minimal default (one fluid + one solid) when no
        region config is provided — useful for smoke tests and template
        generation before the precheck pipeline learns about regions.
        """
        from simd_agent.run.case_spec.strategies import CaseRegions, RegionSpec

        regions_cfg = config.get("regions") or {}
        if not isinstance(regions_cfg, dict):
            regions_cfg = {}

        fluid_raw = regions_cfg.get("fluid") or []
        solid_raw = regions_cfg.get("solid") or []

        # Default: single fluid + single solid (mirrors the OF
        # multiRegionHeater minimal layout).
        if not fluid_raw:
            fluid_raw = [{
                "name": "fluid",
                "thermo_profile": "gas",
                "turbulence_model": "kEpsilon",
            }]

        fluids = [
            RegionSpec(
                name=r["name"],
                kind="fluid",
                thermo_profile=r.get("thermo_profile", "gas"),
                turbulence_model=r.get("turbulence_model", "kEpsilon"),
                Cp=r.get("Cp", 1006.0),
                mol_weight=r.get("mol_weight", 28.97),
                mu=r.get("mu", 1.8e-5),
                Pr=r.get("Pr", 0.7),
            )
            for r in fluid_raw
        ]
        solids = [
            RegionSpec(
                name=r["name"],
                kind="solid",
                thermo_profile="solid",
                turbulence_model=None,
                rho_solid=r.get("rho_solid", 8000.0),
                kappa_solid=r.get("kappa_solid", 80.0),
                Cp_solid=r.get("Cp_solid", 450.0),
            )
            for r in solid_raw
        ]

        return CaseRegions(fluid_regions=fluids, solid_regions=solids)

    # ── Tree-structured file manifest ─────────────────────────────────────

    def required_files(self, config: dict[str, Any]) -> list[str]:
        """Per-region file tree.

        Returns a flat list of paths like ``constant/<region>/file``
        (the manifest is still flat; the *paths* carry the region
        namespace).  The orchestrator unpacks this into the right
        directory layout.

        Phase 1 manifest: top-level controlDict + per-region thermo +
        per-region 0/T (the absolute minimum to instantiate the solver
        in the OF source code's ``createMeshes.H`` step).  Phase 2 adds
        per-region fvSchemes / fvSolution / changeDictionaryDict / 0-fields.
        """
        regions = self.extract_regions(config)
        files = [
            "system/controlDict",
            "constant/regionProperties",
        ]
        for r in regions.all_regions:
            files.append(f"constant/{r.name}/thermophysicalProperties")
            if r.kind == "fluid":
                files.append(f"constant/{r.name}/turbulenceProperties")
                files.append(f"constant/{r.name}/g")
            # 0/<region>/T is required for every region (the coupled
            # field).  Solid regions only solve for T; fluids solve U, p, T, k, ε.
            files.append(f"0/{r.name}/T")
            if r.kind == "fluid":
                files.append(f"0/{r.name}/U")
                files.append(f"0/{r.name}/p")
                files.append(f"0/{r.name}/p_rgh")
                if r.turbulence_model and r.turbulence_model not in ("laminar", "none"):
                    for f in self.turbulence_fields(r.turbulence_model):
                        files.append(f"0/{r.name}/{f}")
        return files

    # ── constant/regionProperties ─────────────────────────────────────────

    def build_region_properties(self, config: dict[str, Any]) -> str:
        """Render ``constant/regionProperties`` from the resolved regions.

        OF 4.x format::

            regions
            (
                fluid       (topAir bottomWater)
                solid       (heater leftSolid rightSolid)
            );
        """
        regions = self.extract_regions(config)
        fluid_list = " ".join(r.name for r in regions.fluid_regions)
        solid_list = " ".join(r.name for r in regions.solid_regions) or ""
        body = "regions\n(\n"
        body += f"    fluid       ({fluid_list})\n"
        if solid_list:
            body += f"    solid       ({solid_list})\n"
        body += ");\n"
        return (
            self._foam_file_header("regionProperties") + body
            + self._foam_file_footer()
        )

    # ── Per-region thermophysicalProperties ───────────────────────────────

    @staticmethod
    def build_fluid_thermo(region: "RegionSpec") -> str:
        """Fluid-region ``constant/<region>/thermophysicalProperties``.

        Matches the OF chtMultiRegionFoam tutorial pattern:
        ``heRhoThermo`` + ``pureMixture`` + ``const`` transport + ``hConst``
        + ``perfectGas`` + ``sensibleEnthalpy``.  Production calls will
        plug in actual ν/Pr/Cp from the user's fluid config.
        """
        return (
            "FoamFile\n{\n"
            "    version     2.0;\n    format      ascii;\n"
            "    class       dictionary;\n"
            f"    location    \"constant/{region.name}\";\n"
            "    object      thermophysicalProperties;\n}\n"
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
            "thermoType\n{\n"
            "    type            heRhoThermo;\n"
            "    mixture         pureMixture;\n"
            "    transport       const;\n"
            "    thermo          hConst;\n"
            "    equationOfState perfectGas;\n"
            "    specie          specie;\n"
            "    energy          sensibleEnthalpy;\n"
            "}\n\n"
            "mixture\n{\n"
            "    specie\n    {\n"
            "        nMoles          1;\n"
            f"        molWeight       {region.mol_weight:g};\n"
            "    }\n"
            "    thermodynamics\n    {\n"
            f"        Cp              {region.Cp:g};\n"
            "        Hf              0;\n"
            "    }\n"
            "    transport\n    {\n"
            f"        mu              {region.mu:g};\n"
            f"        Pr              {region.Pr:g};\n"
            "    }\n"
            "}\n\n"
            "// ************************************************************************* //\n"
        )

    @staticmethod
    def build_solid_thermo(region: "RegionSpec") -> str:
        """Solid-region ``constant/<region>/thermophysicalProperties``.

        OF ``heSolidThermo`` + ``rhoConst`` + ``constIso`` (constant
        isotropic conductivity).  No equation of state for compressibility,
        no transport viscosity — just ρ, κ, Cp.
        """
        return (
            "FoamFile\n{\n"
            "    version     2.0;\n    format      ascii;\n"
            "    class       dictionary;\n"
            f"    location    \"constant/{region.name}\";\n"
            "    object      thermophysicalProperties;\n}\n"
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
            "thermoType\n{\n"
            "    type            heSolidThermo;\n"
            "    mixture         pureMixture;\n"
            "    transport       constIso;\n"
            "    thermo          hConst;\n"
            "    equationOfState rhoConst;\n"
            "    specie          specie;\n"
            "    energy          sensibleEnthalpy;\n"
            "}\n\n"
            "mixture\n{\n"
            "    specie\n    {\n"
            "        nMoles      1;\n"
            "        molWeight   50;\n"
            "    }\n"
            "    transport\n    {\n"
            f"        kappa       {region.kappa_solid:g};\n"
            "    }\n"
            "    thermodynamics\n    {\n"
            "        Hf          0;\n"
            f"        Cp          {region.Cp_solid:g};\n"
            "    }\n"
            "    equationOfState\n    {\n"
            f"        rho         {region.rho_solid:g};\n"
            "    }\n"
            "}\n\n"
            "// ************************************************************************* //\n"
        )

    # ── Deterministic-files registry override ─────────────────────────────

    def render_deterministic_files(self, config: dict[str, Any]) -> dict[str, str]:
        """Render the multi-region deterministic file tree.

        Phase 1 — emits the minimum CHT skeleton:
          * ``constant/regionProperties``
          * ``constant/<region>/thermophysicalProperties`` for each region
            (fluid uses heRhoThermo + perfectGas; solid uses heSolidThermo)

        Phase 2 will add: per-region turbulenceProperties, per-region
        fvSchemes / fvSolution, per-region 0/T with coupled fluid-solid
        boundary patches, top-level fvSchemes / fvSolution (shared).
        """
        files: dict[str, str] = {}
        regions = self.extract_regions(config)

        # 1. regionProperties.
        files["constant/regionProperties"] = self.build_region_properties(config)

        # 2. Per-region thermophysicalProperties.
        for r in regions.fluid_regions:
            files[f"constant/{r.name}/thermophysicalProperties"] = (
                self.build_fluid_thermo(r)
            )
        for r in regions.solid_regions:
            files[f"constant/{r.name}/thermophysicalProperties"] = (
                self.build_solid_thermo(r)
            )

        return files

    # ── Abstract surface — concrete plugins fill these in Phase 2 ─────────

    def _build_fv_solution(self, config: dict[str, Any]) -> str:
        """Top-level ``system/fvSolution``.

        Phase 1 — emits a placeholder shared block.  Phase 2 will:
          (a) emit a minimal top-level PIMPLE / SIMPLE outer-loop block
              that controls the multi-region coupling (nNonOrthogonalCorrectors,
              residual control across regions), and
          (b) emit per-region ``system/<region>/fvSolution`` files via
              ``render_deterministic_files`` so each region's solver
              settings live in its own file.
        """
        return (
            self._foam_file_header("fvSolution")
            + "// Multi-region top-level fvSolution.\n"
            + "// Per-region solver settings live in system/<region>/fvSolution\n"
            + "// (rendered separately by the deterministic-files step — TODO Phase 2).\n\n"
            + "PIMPLE\n{\n"
            + "    nOuterCorrectors    1;\n"
            + "    nCorrectors         1;\n"
            + "    nNonOrthogonalCorrectors 0;\n"
            + "}\n"
            + self._foam_file_footer()
        )

    def _build_fv_schemes(self, config: dict[str, Any]) -> str:
        """Top-level ``system/fvSchemes`` — also mostly per-region (Phase 2)."""
        return (
            self._foam_file_header("fvSchemes")
            + "// Multi-region top-level fvSchemes (per-region overrides TODO Phase 2).\n\n"
            + "ddtSchemes      { default Euler; }\n"
            + "gradSchemes     { default Gauss linear; }\n"
            + "divSchemes      { default none; }\n"
            + "laplacianSchemes { default Gauss linear corrected; }\n"
            + "interpolationSchemes { default linear; }\n"
            + "snGradSchemes   { default corrected; }\n"
            + self._foam_file_footer()
        )