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

    # ── Helper: marshal extra region fields from config dict ──────────────

    @staticmethod
    def _region_kwargs_from_dict(raw: dict[str, Any], kind: str) -> dict[str, Any]:
        """Translate a region dict (user-supplied JSON / YAML) to RegionSpec kwargs.

        Accepts both snake_case (precheck shape) and camelCase (frontend
        shape) keys.  Validates ``interfaces`` shape (must be list of str).
        """
        kw: dict[str, Any] = {
            "name": raw["name"],
            "kind": kind,
        }
        if kind == "fluid":
            kw["thermo_profile"] = raw.get("thermo_profile") or raw.get(
                "thermoProfile", "gas"
            )
            kw["turbulence_model"] = (
                raw.get("turbulence_model")
                or raw.get("turbulenceModel", "kEpsilon")
            )
            for src, dst in (
                ("Cp", "Cp"), ("mol_weight", "mol_weight"),
                ("molWeight", "mol_weight"), ("mu", "mu"), ("Pr", "Pr"),
                ("T_init", "T_init"), ("p_init", "p_init"),
                ("k_init", "k_init"), ("epsilon_init", "epsilon_init"),
            ):
                if src in raw:
                    kw[dst] = raw[src]
            if "U_init" in raw:
                _u = raw["U_init"]
                if isinstance(_u, (list, tuple)) and len(_u) == 3:
                    kw["U_init"] = tuple(float(c) for c in _u)
        else:  # solid
            kw["thermo_profile"] = "solid"
            kw["turbulence_model"] = None
            for src, dst in (
                ("rho_solid", "rho_solid"), ("rhoSolid", "rho_solid"),
                ("kappa_solid", "kappa_solid"), ("kappaSolid", "kappa_solid"),
                ("Cp_solid", "Cp_solid"), ("CpSolid", "Cp_solid"),
                ("T_init", "T_init"),
            ):
                if src in raw:
                    kw[dst] = raw[src]

        # Interfaces — names of regions this one touches.
        _ifaces = raw.get("interfaces") or raw.get("coupled_regions") or ()
        if isinstance(_ifaces, (list, tuple)):
            kw["interfaces"] = tuple(str(x) for x in _ifaces)

        return kw

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
            fluid_raw = [{"name": "fluid"}]

        fluids = [
            RegionSpec(**self._region_kwargs_from_dict(r, kind="fluid"))
            for r in fluid_raw
        ]
        solids = [
            RegionSpec(**self._region_kwargs_from_dict(r, kind="solid"))
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

    # ── Per-region turbulenceProperties + g (fluid regions) ───────────────

    @staticmethod
    def build_fluid_turbulence_properties(region: "RegionSpec") -> str:
        """Per-region ``constant/<fluid>/turbulenceProperties``.

        Always RAS in CHT context — LES on a multi-region case is unusual.
        Model defaults to ``kEpsilon`` (the OF multiRegionHeater choice);
        the user / precheck can override via ``region.turbulence_model``.
        """
        model = region.turbulence_model or "kEpsilon"
        if model.lower() == "laminar":
            body = "simulationType  laminar;\n"
        else:
            body = (
                "simulationType  RAS;\n\n"
                "RAS\n{\n"
                f"    RASModel        {model};\n"
                "    turbulence      on;\n"
                "    printCoeffs     on;\n"
                "}\n"
            )
        return (
            "FoamFile\n{\n"
            "    version     2.0;\n    format      ascii;\n"
            "    class       dictionary;\n"
            f"    location    \"constant/{region.name}\";\n"
            "    object      turbulenceProperties;\n}\n"
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
            + body
            + "\n// ************************************************************************* //\n"
        )

    @staticmethod
    def build_region_gravity(region: "RegionSpec") -> str:
        """Per-region ``constant/<fluid>/g`` — Earth gravity in -y."""
        return (
            "FoamFile\n{\n"
            "    version     2.0;\n    format      ascii;\n"
            "    class       uniformDimensionedVectorField;\n"
            f"    location    \"constant/{region.name}\";\n"
            "    object      g;\n}\n"
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
            "dimensions      [0 1 -2 0 0 0 0];\n"
            "value           (0 -9.81 0);\n\n"
            "// ************************************************************************* //\n"
        )

    # ── Per-region fvSchemes ──────────────────────────────────────────────

    @staticmethod
    def build_fluid_fv_schemes(region: "RegionSpec") -> str:
        """Per-region ``system/<fluid>/fvSchemes`` — full N-S + energy + turb.

        Mirrors the OF ``multiRegionHeater/bottomWater/fvSchemes`` pattern:
        upwind everywhere for robustness, ``div(phi,K) Gauss linear``
        (kinetic energy in the energy equation), incompressible-form
        viscous stress.
        """
        return (
            "FoamFile\n{\n"
            "    version     2.0;\n    format      ascii;\n"
            "    class       dictionary;\n"
            f"    location    \"system/{region.name}\";\n"
            "    object      fvSchemes;\n}\n"
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
            "ddtSchemes\n{\n    default         Euler;\n}\n\n"
            "gradSchemes\n{\n    default         Gauss linear;\n}\n\n"
            "divSchemes\n{\n"
            "    default         none;\n"
            "    div(phi,U)      bounded Gauss upwind;\n"
            "    div(phi,K)      Gauss linear;\n"
            "    div(phi,h)      bounded Gauss upwind;\n"
            "    div(phi,k)      bounded Gauss upwind;\n"
            "    div(phi,epsilon) bounded Gauss upwind;\n"
            "    div(phi,omega)  bounded Gauss upwind;\n"
            "    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;\n"
            "}\n\n"
            "laplacianSchemes\n{\n    default         Gauss linear corrected;\n}\n\n"
            "interpolationSchemes\n{\n    default         linear;\n}\n\n"
            "snGradSchemes\n{\n    default         corrected;\n}\n\n"
            "// ************************************************************************* //\n"
        )

    @staticmethod
    def build_solid_fv_schemes(region: "RegionSpec") -> str:
        """Per-region ``system/<solid>/fvSchemes`` — only ``laplacian(alpha,h)``.

        Solid regions solve only the heat equation; no momentum, no
        turbulence, no convection.  The single relevant scheme is the
        laplacian for enthalpy diffusion.
        """
        return (
            "FoamFile\n{\n"
            "    version     2.0;\n    format      ascii;\n"
            "    class       dictionary;\n"
            f"    location    \"system/{region.name}\";\n"
            "    object      fvSchemes;\n}\n"
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
            "ddtSchemes\n{\n    default         Euler;\n}\n\n"
            "gradSchemes\n{\n    default         Gauss linear;\n}\n\n"
            "divSchemes\n{\n    default         none;\n}\n\n"
            "laplacianSchemes\n{\n"
            "    default             none;\n"
            "    laplacian(alpha,h)  Gauss linear corrected;\n"
            "}\n\n"
            "interpolationSchemes\n{\n    default         linear;\n}\n\n"
            "snGradSchemes\n{\n    default         corrected;\n}\n\n"
            "// ************************************************************************* //\n"
        )

    # ── Per-region fvSolution ─────────────────────────────────────────────

    def build_fluid_fv_solution(self, region: "RegionSpec") -> str:
        """Per-region ``system/<fluid>/fvSolution``.

        Solvers for rho, p_rgh, (U|h|k|epsilon|omega) + Final variants.
        SIMPLE / PIMPLE outer-loop control read from the plugin's
        ``algorithm``.
        """
        outer_block_name = self.algorithm  # SIMPLE or PIMPLE
        # Build the equation regex from the fluid region's turbulence model.
        eq_fields = ["U", "h"]
        m = (region.turbulence_model or "").lower()
        if "komegasst" in m or "komega" in m:
            eq_fields += ["k", "omega"]
        elif "kepsilon" in m:
            eq_fields += ["k", "epsilon"]
        eq_regex = '"(' + "|".join(eq_fields) + ')"'
        eq_final_regex = '"(' + "|".join(eq_fields) + ')Final"'

        return (
            "FoamFile\n{\n"
            "    version     2.0;\n    format      ascii;\n"
            "    class       dictionary;\n"
            f"    location    \"system/{region.name}\";\n"
            "    object      fvSolution;\n}\n"
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
            "solvers\n{\n"
            "    rho\n"
            "    {\n"
            "        solver          PCG;\n"
            "        preconditioner  DIC;\n"
            "        tolerance       1e-7;\n"
            "        relTol          0.1;\n"
            "    }\n\n"
            "    rhoFinal\n"
            "    {\n"
            "        $rho;\n"
            "        tolerance       1e-7;\n"
            "        relTol          0;\n"
            "    }\n\n"
            "    p_rgh\n"
            "    {\n"
            "        solver          GAMG;\n"
            "        smoother        GaussSeidel;\n"
            "        nCellsInCoarsestLevel 20;\n"
            "        tolerance       1e-7;\n"
            "        relTol          0.01;\n"
            "    }\n\n"
            "    p_rghFinal\n"
            "    {\n"
            "        $p_rgh;\n"
            "        tolerance       1e-7;\n"
            "        relTol          0;\n"
            "    }\n\n"
            f"    {eq_regex}\n"
            "    {\n"
            "        solver          PBiCG;\n"
            "        preconditioner  DILU;\n"
            "        tolerance       1e-7;\n"
            "        relTol          0.1;\n"
            "    }\n\n"
            f"    {eq_final_regex}\n"
            f"    {{\n"
            f"        ${eq_fields[0]};\n"
            "        tolerance       1e-7;\n"
            "        relTol          0;\n"
            "    }\n"
            "}\n\n"
            f"{outer_block_name}\n"
            "{\n"
            "    momentumPredictor   yes;\n"
            "    nCorrectors         2;\n"
            "    nNonOrthogonalCorrectors 0;\n"
            "}\n\n"
            "relaxationFactors\n{\n"
            "    equations\n"
            "    {\n"
            '        "h.*"           1;\n'
            '        "U.*"           1;\n'
            "    }\n"
            "}\n\n"
            "// ************************************************************************* //\n"
        )

    def build_solid_fv_solution(self, region: "RegionSpec") -> str:
        """Per-region ``system/<solid>/fvSolution`` — only enthalpy h."""
        outer_block_name = self.algorithm
        return (
            "FoamFile\n{\n"
            "    version     2.0;\n    format      ascii;\n"
            "    class       dictionary;\n"
            f"    location    \"system/{region.name}\";\n"
            "    object      fvSolution;\n}\n"
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
            "solvers\n{\n"
            "    h\n"
            "    {\n"
            "        solver          PCG;\n"
            "        preconditioner  DIC;\n"
            "        tolerance       1e-06;\n"
            "        relTol          0.1;\n"
            "    }\n\n"
            "    hFinal\n"
            "    {\n"
            "        $h;\n"
            "        tolerance       1e-06;\n"
            "        relTol          0;\n"
            "    }\n"
            "}\n\n"
            f"{outer_block_name}\n"
            "{\n"
            "    nNonOrthogonalCorrectors 0;\n"
            "}\n\n"
            "// ************************************************************************* //\n"
        )

    # ── Per-region 0/<region>/* field files ───────────────────────────────

    @staticmethod
    def _coupled_T_patches(region: "RegionSpec") -> str:
        """Render the coupled BC blocks for every interface this region has.

        Fluid sides use ``kappaMethod fluidThermo``; solid sides use
        ``kappaMethod solidThermo``.  Patch names follow the OF
        ``<self>_to_<other>`` convention.
        """
        if not region.interfaces:
            return ""
        kappa_method = "fluidThermo" if region.kind == "fluid" else "solidThermo"
        blocks = []
        for nbr in region.interfaces:
            patch = f"{region.name}_to_{nbr}"
            blocks.append(
                f"    {patch}\n"
                "    {\n"
                "        type            compressible::turbulentTemperatureCoupledBaffleMixed;\n"
                "        Tnbr            T;\n"
                f"        kappaMethod     {kappa_method};\n"
                f"        value           uniform {region.T_init:g};\n"
                "    }\n"
            )
        return "\n".join(blocks)

    def build_region_0_T(self, region: "RegionSpec") -> str:
        """``0/<region>/T`` with coupled BCs at every interface."""
        coupled = self._coupled_T_patches(region)
        # Solids only have wall + coupled patches; fluids also have inlet/outlet.
        if region.kind == "fluid":
            other_patches = (
                "    \".*\"\n"
                "    {\n"
                "        type            zeroGradient;\n"
                "    }\n"
            )
        else:
            other_patches = (
                "    \".*\"\n"
                "    {\n"
                "        type            zeroGradient;\n"
                f"        value           uniform {region.T_init:g};\n"
                "    }\n"
            )
        return (
            "FoamFile\n{\n"
            "    version     2.0;\n    format      ascii;\n"
            "    class       volScalarField;\n"
            f"    location    \"0/{region.name}\";\n"
            "    object      T;\n}\n"
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
            "dimensions      [0 0 0 1 0 0 0];\n"
            f"internalField   uniform {region.T_init:g};\n\n"
            "boundaryField\n{\n"
            f"{other_patches}"
            f"{coupled}"
            "}\n\n"
            "// ************************************************************************* //\n"
        )

    @staticmethod
    def build_region_0_U(region: "RegionSpec") -> str:
        """``0/<fluid>/U`` — wall noSlip, catch-all fixedValue at init."""
        u = region.U_init
        return (
            "FoamFile\n{\n"
            "    version     2.0;\n    format      ascii;\n"
            "    class       volVectorField;\n"
            f"    location    \"0/{region.name}\";\n"
            "    object      U;\n}\n"
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
            "dimensions      [0 1 -1 0 0 0 0];\n"
            f"internalField   uniform ({u[0]:g} {u[1]:g} {u[2]:g});\n\n"
            "boundaryField\n{\n"
            "    \".*\"\n"
            "    {\n"
            "        type            fixedValue;\n"
            "        value           uniform (0 0 0);\n"
            "    }\n"
            "}\n\n"
            "// ************************************************************************* //\n"
        )

    @staticmethod
    def build_region_0_p_rgh(region: "RegionSpec") -> str:
        """``0/<fluid>/p_rgh`` — fixedFluxPressure on all patches."""
        return (
            "FoamFile\n{\n"
            "    version     2.0;\n    format      ascii;\n"
            "    class       volScalarField;\n"
            f"    location    \"0/{region.name}\";\n"
            "    object      p_rgh;\n}\n"
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
            "dimensions      [1 -1 -2 0 0 0 0];\n"
            f"internalField   uniform {region.p_init:g};\n\n"
            "boundaryField\n{\n"
            "    \".*\"\n"
            "    {\n"
            "        type            fixedFluxPressure;\n"
            f"        value           uniform {region.p_init:g};\n"
            "    }\n"
            "}\n\n"
            "// ************************************************************************* //\n"
        )

    @staticmethod
    def build_region_0_p(region: "RegionSpec") -> str:
        """``0/<fluid>/p`` — calculated (synthesised from p_rgh + ρgh)."""
        return (
            "FoamFile\n{\n"
            "    version     2.0;\n    format      ascii;\n"
            "    class       volScalarField;\n"
            f"    location    \"0/{region.name}\";\n"
            "    object      p;\n}\n"
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
            "dimensions      [1 -1 -2 0 0 0 0];\n"
            f"internalField   uniform {region.p_init:g};\n\n"
            "boundaryField\n{\n"
            "    \".*\"\n"
            "    {\n"
            "        type            calculated;\n"
            f"        value           uniform {region.p_init:g};\n"
            "    }\n"
            "}\n\n"
            "// ************************************************************************* //\n"
        )

    @staticmethod
    def build_region_0_k(region: "RegionSpec") -> str:
        """``0/<fluid>/k`` — kqRWallFunction on walls."""
        return (
            "FoamFile\n{\n"
            "    version     2.0;\n    format      ascii;\n"
            "    class       volScalarField;\n"
            f"    location    \"0/{region.name}\";\n"
            "    object      k;\n}\n"
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
            "dimensions      [0 2 -2 0 0 0 0];\n"
            f"internalField   uniform {region.k_init:g};\n\n"
            "boundaryField\n{\n"
            "    \".*\"\n"
            "    {\n"
            "        type            kqRWallFunction;\n"
            f"        value           uniform {region.k_init:g};\n"
            "    }\n"
            "}\n\n"
            "// ************************************************************************* //\n"
        )

    @staticmethod
    def build_region_0_epsilon(region: "RegionSpec") -> str:
        """``0/<fluid>/epsilon`` — epsilonWallFunction on walls."""
        return (
            "FoamFile\n{\n"
            "    version     2.0;\n    format      ascii;\n"
            "    class       volScalarField;\n"
            f"    location    \"0/{region.name}\";\n"
            "    object      epsilon;\n}\n"
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
            "dimensions      [0 2 -3 0 0 0 0];\n"
            f"internalField   uniform {region.epsilon_init:g};\n\n"
            "boundaryField\n{\n"
            "    \".*\"\n"
            "    {\n"
            "        type            epsilonWallFunction;\n"
            f"        value           uniform {region.epsilon_init:g};\n"
            "    }\n"
            "}\n\n"
            "// ************************************************************************* //\n"
        )

    # ── changeDictionaryDict per region ───────────────────────────────────

    def build_change_dictionary_dict(self, region: "RegionSpec") -> str:
        """Per-region ``system/<region>/changeDictionaryDict``.

        Applied after ``splitMeshRegions`` to:
          1. Reclassify any boundary types if needed (mainly for solids
             whose boundaries become ``patch`` after splitting).
          2. Set the coupled fluid-solid temperature BC on the interface
             patches with the right ``kappaMethod`` per side.
        """
        coupled = self._coupled_T_patches(region)
        # We keep the changeDictionaryDict minimal — just the T overrides
        # at the interfaces.  More elaborate patches (renaming, type
        # changes) can be added per-case later.
        body = (
            "T\n"
            "{\n"
            f"    internalField   uniform {region.T_init:g};\n\n"
            "    boundaryField\n"
            "    {\n"
            f"{coupled}"
            "    }\n"
            "}\n"
        )
        return (
            "FoamFile\n{\n"
            "    version     2.0;\n    format      ascii;\n"
            "    class       dictionary;\n"
            f"    location    \"system/{region.name}\";\n"
            "    object      changeDictionaryDict;\n}\n"
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
            + body
            + "\n// ************************************************************************* //\n"
        )

    # ── Deterministic-files registry override ─────────────────────────────

    def render_deterministic_files(self, config: dict[str, Any]) -> dict[str, str]:
        """Render the full multi-region deterministic file tree (Phase 2).

        Emits:

          * ``constant/regionProperties`` — region listing
          * Per-region ``constant/<region>/thermophysicalProperties``
            (fluid: ``heRhoThermo`` + ``perfectGas``; solid: ``heSolidThermo``)
          * Per-fluid ``constant/<region>/turbulenceProperties``
          * Per-fluid ``constant/<region>/g``  (Earth gravity in -y)
          * Per-region ``system/<region>/fvSchemes`` and ``fvSolution``
            (fluid: full N-S + energy + turb; solid: only ``laplacian(alpha,h)``)
          * Per-region ``system/<region>/changeDictionaryDict``
            (coupled T BCs on interface patches)
          * Per-region ``0/<region>/T`` (with
            ``compressible::turbulentTemperatureCoupledBaffleMixed`` at
            every interface patch)
          * Per-fluid ``0/<region>/{U, p, p_rgh, k, epsilon}``

        Still TODO outside this method:

          * Top-level ``system/fvSchemes`` / ``system/fvSolution`` (kept
            as placeholders by the plugin's ``_build_fv_*`` methods).
          * Orchestrator wiring to actually emit the tree of files.
          * packaging.py zip layout for multi-region cases.
        """
        files: dict[str, str] = {}
        regions = self.extract_regions(config)

        # 1. regionProperties.
        files["constant/regionProperties"] = self.build_region_properties(config)

        # 2. Per-region thermophysicalProperties + (fluid only) turb + g.
        for r in regions.fluid_regions:
            files[f"constant/{r.name}/thermophysicalProperties"] = (
                self.build_fluid_thermo(r)
            )
            files[f"constant/{r.name}/turbulenceProperties"] = (
                self.build_fluid_turbulence_properties(r)
            )
            files[f"constant/{r.name}/g"] = self.build_region_gravity(r)
        for r in regions.solid_regions:
            files[f"constant/{r.name}/thermophysicalProperties"] = (
                self.build_solid_thermo(r)
            )

        # 3. Per-region system/<region>/{fvSchemes, fvSolution, changeDictionaryDict}.
        for r in regions.fluid_regions:
            files[f"system/{r.name}/fvSchemes"] = self.build_fluid_fv_schemes(r)
            files[f"system/{r.name}/fvSolution"] = self.build_fluid_fv_solution(r)
            files[f"system/{r.name}/changeDictionaryDict"] = (
                self.build_change_dictionary_dict(r)
            )
        for r in regions.solid_regions:
            files[f"system/{r.name}/fvSchemes"] = self.build_solid_fv_schemes(r)
            files[f"system/{r.name}/fvSolution"] = self.build_solid_fv_solution(r)
            files[f"system/{r.name}/changeDictionaryDict"] = (
                self.build_change_dictionary_dict(r)
            )

        # 4. Per-region 0/<region>/* fields.
        for r in regions.fluid_regions:
            files[f"0/{r.name}/T"] = self.build_region_0_T(r)
            files[f"0/{r.name}/U"] = self.build_region_0_U(r)
            files[f"0/{r.name}/p"] = self.build_region_0_p(r)
            files[f"0/{r.name}/p_rgh"] = self.build_region_0_p_rgh(r)
            tm = (r.turbulence_model or "").lower()
            if "kepsilon" in tm or "komega" in tm:
                files[f"0/{r.name}/k"] = self.build_region_0_k(r)
            if "kepsilon" in tm:
                files[f"0/{r.name}/epsilon"] = self.build_region_0_epsilon(r)
        for r in regions.solid_regions:
            files[f"0/{r.name}/T"] = self.build_region_0_T(r)

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