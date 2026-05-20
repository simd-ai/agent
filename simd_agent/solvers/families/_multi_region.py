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

import logging
from typing import TYPE_CHECKING, Any

from simd_agent.solvers.base import (
    SolverPlugin,
    ValidationIssue,
    ValidationResult,
)

logger = logging.getLogger(__name__)

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

    # ── Turbulence field tweak — CHT needs alphat on every fluid region ──
    #
    # The single-region default ``TURBULENCE_FIELDS`` (in ``solvers/base.py``)
    # only lists ``k``, ``omega``/``epsilon``, ``nut`` — no ``alphat``.  For
    # incompressible single-region that's correct (no energy equation).
    # CHT solvers are compressible by construction and always solve the
    # energy equation, so each fluid region needs its own ``alphat``
    # (turbulent thermal diffusivity, the energy-equation companion of
    # ``nut``).  Without it the solver aborts at startup with
    # ``cannot find file 0/<fluid>/alphat``.
    def turbulence_fields(self, turb_model: str) -> list[str]:
        base = super().turbulence_fields(turb_model)
        if not base:
            return base  # laminar / none — no alphat needed
        return list(base) + (["alphat"] if "alphat" not in base else [])

    # ── Validation entry point — skip the single-region legacy validator ──
    #
    # The shared ``validate_generated_files`` in ``run/genai_codegen.py``
    # was built for flat single-region cases.  It asserts things like "all
    # mesh patches appear in every 0/<field> file", which is wrong here:
    # CHT cases have ``0/<region>/<field>`` and each region only carries
    # the patches that belong to it (plus the coupled interface patches
    # auto-created by ``splitMeshRegions``).  Running it on a multi-region
    # file tree produces false errors and tries to "fix" things that are
    # already correct.  We override the orchestrator entry point to skip
    # the legacy step entirely; the plugin's own ``validate()`` (which
    # invokes ``render_deterministic_files``) is authoritative, and the
    # universal constraint-patch fixer is the only post-pass we run.

    def validate_full(
        self, files: dict[str, Any], config: dict[str, Any]
    ) -> ValidationResult:
        pre_issues: list[ValidationIssue] = []
        files = self._fix_brace_balance(files, pre_issues)

        plugin_result = self.validate(files, config)
        fixed = plugin_result.files
        issues = list(pre_issues) + list(plugin_result.issues)

        # Universal constraint-patch BC fix — runs on every 0/<region>/<field>
        # file just like it does for single-region 0/<field>.
        fixed = self._fix_constraint_patch_bcs(fixed, config, issues)

        return ValidationResult(files=fixed, issues=issues)

    # ── Region presets ────────────────────────────────────────────────────
    #
    # Per-preset physics values used to fill ``RegionSpec`` defaults.  The
    # user supplies ``"fluid_preset": "water"`` (or ``"solid_preset":
    # "copper"``) on a region config dict; the renderer auto-populates
    # Cp / μ / Pr / ρ / κ / mol_weight from the matching table.  Explicit
    # per-field values still win — these are *defaults*, not constraints.
    #
    # Values sourced from NIST / standard tables at the reference
    # temperature noted in each entry.  Where Pr isn't tabulated it's
    # computed as μ·Cp/κ.

    # Per-preset ``rho_nominal`` is the operating-point density used to
    # derive SIMPLE-block ``rhoMin``/``rhoMax`` safety bounds for
    # chtMultiRegionSimpleFoam (see ``build_fluid_fv_solution``).  Without
    # bounded ρ the steady-CHT outer loop drifts: a small ρ overshoot on
    # iteration N feeds an enthalpy overshoot on N+1 which feeds a ρ
    # overshoot on N+2 — observed as ``Min/max rho: 0  9801`` and
    # ``Min/max T: -179  437`` from a 300 K initial field.
    #
    # Sources (saturated-liquid values at 1 atm for cryogenics, 293 K for
    # gases / oil / water):
    # All values verified against authoritative references:
    #
    #   air      1.2041 kg/m³ at 293.15 K  — Wikipedia "Density of air"
    #   water    998.21 kg/m³ at 293.15 K  — Wikipedia "Water (data page)"
    #   oil      880    kg/m³ at 293 K     — SAE 30, Wikipedia "Motor oil"
    #                                        (range 870–892; 880 is mid)
    #   helium   0.1664 kg/m³ at 293 K     — derived from STP 0.1786 g/L
    #                                        × (273.15/293.15)
    #   ln2      808    kg/m³ at 77 K      — Wikipedia "Nitrogen" (b.p.)
    #   lox      1141   kg/m³ at 90 K      — Wikipedia "Liquid oxygen"
    #   lh2      70.85  kg/m³ at 20 K      — Wikipedia "Liquid hydrogen"
    #   lng      422.8  kg/m³ at 111 K     — Wikipedia "Methane"
    #                                        (pure CH₄ at −162 °C; real
    #                                        LNG mixtures: 410–500)
    #   lhe      125    kg/m³ at 4.2 K     — Wikipedia "Liquid helium" (He-4 b.p.)
    FLUID_REGION_PRESETS: dict[str, dict[str, float | str]] = {
        # Air at 293.15 K, 1 atm.  All values NIST-traceable:
        #   ρ  = 1.2041 kg/m³     (Wikipedia "Density of air")
        #   Cp = 1006   J/(kg·K)  (NIST WebBook, dry air at 20 °C)
        #   μ  = 1.81 µPa·s        (NIST WebBook, 1.813e-5 Pa·s)
        #   Pr = 0.714             (μCp/k, k=0.02551 W/(m·K))
        "air": {
            "thermo_profile": "gas",
            "Cp": 1006.0, "mol_weight": 28.97,
            "mu": 1.81e-5, "Pr": 0.714,
            "rho_nominal": 1.2041,
        },
        # Water at 293.15 K, 1 atm.  Density 998.21 (Wikipedia
        # "Water (data page)" — interpolated 0.99820 g/cm³).
        "water": {
            "thermo_profile": "gas",          # Boussinesq treats it as p,T-driven
            "Cp": 4182.0, "mol_weight": 18.02,
            "mu": 1.002e-3, "Pr": 7.0,
            "rho_nominal": 998.21,
        },
        # SAE 30 oil at 293 K.  Density ~880 (range 870–892 across base
        # oils; Wikipedia "Motor oil" reports 888 for SAE 20W as the
        # closest published reference).
        "oil": {
            "thermo_profile": "gas",
            "Cp": 1900.0, "mol_weight": 250.0,
            "mu": 0.29, "Pr": 3800.0,         # very viscous, very high Pr
            "rho_nominal": 880.0,
        },
        # Liquid nitrogen at 77.36 K, 1 atm.  Density 808 g/L (Wikipedia
        # "Nitrogen" — value "when liquid (at b.p.)").
        "ln2": {
            "thermo_profile": "cryogenic",
            "Cp": 2042.0, "mol_weight": 28.01,
            "mu": 1.58e-4, "Pr": 2.3,
            "rho_nominal": 808.0,
        },
        # Liquid oxygen at 90.19 K, 1 atm.  Density 1141 g/L (Wikipedia
        # "Liquid oxygen": "density of 1.141 kg/L").
        "lox": {
            "thermo_profile": "cryogenic",
            "Cp": 1699.0, "mol_weight": 32.00,
            "mu": 1.95e-4, "Pr": 2.2,
            "rho_nominal": 1141.0,
        },
        # Liquid hydrogen at 20.27 K, 1 atm.  Density 70.85 (Wikipedia
        # "Liquid hydrogen": "only 70.85 kg/m³ (at 20 K)").
        "lh2": {
            "thermo_profile": "cryogenic",
            "Cp": 9668.0, "mol_weight": 2.016,
            "mu": 1.33e-5, "Pr": 1.3,
            "rho_nominal": 70.85,
        },
        # Liquefied natural gas at 111.65 K — modelled as pure methane.
        # Density 422.8 g/L (Wikipedia "Methane": "422.8 g/L (liquid,
        # −162 °C)").  Real-world LNG mixtures range 410–500 depending
        # on composition; the rhoMin/rhoMax band (0.2×–2.0×) covers this.
        "lng": {
            "thermo_profile": "cryogenic",
            "Cp": 3500.0, "mol_weight": 16.04,
            "mu": 1.2e-4, "Pr": 2.3,
            "rho_nominal": 422.8,
        },
        # Helium GAS at 293 K (room-temperature He).  Density 0.1664
        # derived from STP value 0.1786 g/L × (273.15/293.15) ideal-gas
        # scaling (Wikipedia "Helium" element infobox at STP).
        "helium": {
            "thermo_profile": "gas",
            "Cp": 5193.0, "mol_weight": 4.003,
            "mu": 1.96e-5, "Pr": 0.67,
            "rho_nominal": 0.1664,
        },
        # Liquid helium-4 (He I) at 4.222 K, 1 atm — saturated liquid at NBP.
        # Properties from NIST REFPROP / Barron, "Cryogenic Heat Transfer":
        #   ρ  = 125 kg/m³        (Wikipedia "Liquid helium": "about 125 g/L")
        #   Cp = 4480 J/(kg·K)    (NIST sat-liquid at 4.222 K — diverges near
        #                          the lambda point at 2.17 K, this is the
        #                          standard NBP engineering value)
        #   μ  = 3.5 µPa·s         (NIST sat-liquid; He I normal liquid,
        #                          not the superfluid regime)
        #   Pr = μ·Cp/k = 3.5e-6 × 4480 / 0.019 ≈ 0.83
        # Kept as a separate preset from the gas-phase ``helium`` above —
        # the operating density differs by three orders of magnitude
        # (125 vs 0.1664 kg/m³) and so do the rhoMin/rhoMax bounds.
        # Region-detection in
        # :func:`simd_agent.run.multi_region.region_detection.fluid_preset_for`
        # routes ``lhe`` / ``liquid helium`` here.
        "lhe": {
            "thermo_profile": "cryogenic",
            "Cp": 4480.0, "mol_weight": 4.003,
            "mu": 3.5e-6, "Pr": 0.83,
            "rho_nominal": 125.0,
        },
    }

    SOLID_REGION_PRESETS: dict[str, dict[str, float]] = {
        # Carbon steel at 293 K.
        "steel": {
            "rho_solid": 8000.0, "kappa_solid": 80.0, "Cp_solid": 450.0,
        },
        # Pure copper at 293 K.
        "copper": {
            "rho_solid": 8960.0, "kappa_solid": 400.0, "Cp_solid": 385.0,
        },
        # Pure aluminum at 293 K.
        "aluminum": {
            "rho_solid": 2700.0, "kappa_solid": 237.0, "Cp_solid": 900.0,
        },
        # Standard concrete at 293 K.
        "concrete": {
            "rho_solid": 2300.0, "kappa_solid": 1.4, "Cp_solid": 880.0,
        },
        # Soda-lime glass at 293 K.
        "glass": {
            "rho_solid": 2500.0, "kappa_solid": 1.05, "Cp_solid": 840.0,
        },
        # Mild stainless 304 at 293 K.
        "stainless": {
            "rho_solid": 7900.0, "kappa_solid": 16.2, "Cp_solid": 500.0,
        },
    }

    # ── Region extraction from config ─────────────────────────────────────

    # ── Helper: marshal extra region fields from config dict ──────────────

    @classmethod
    def _region_kwargs_from_dict(
        cls, raw: dict[str, Any], kind: str
    ) -> dict[str, Any]:
        """Translate a region dict (user-supplied JSON / YAML) to RegionSpec kwargs.

        Two layers of values are applied, in order:

          1. **Preset defaults** — when ``raw["fluid_preset"]`` or
             ``raw["solid_preset"]`` names a known entry in
             ``FLUID_REGION_PRESETS`` / ``SOLID_REGION_PRESETS``, those
             physics values (Cp / μ / Pr / ρ / κ / mol_weight) are
             applied first.  This is what lets the user say
             ``{"name": "topWater", "fluid_preset": "water"}`` and get
             Cp=4182 / μ=1e-3 / Pr=7.0 automatically.

          2. **Explicit overrides** — any per-field value in ``raw``
             (``Cp``, ``mu``, ``rho_solid``, …) wins over the preset.

        Accepts both snake_case (precheck) and camelCase (frontend) keys.
        """
        kw: dict[str, Any] = {
            "name": raw["name"],
            "kind": kind,
        }

        # ── Layer 1: apply preset defaults ──
        preset_name = (
            raw.get("fluid_preset" if kind == "fluid" else "solid_preset")
            or raw.get("fluidPreset" if kind == "fluid" else "solidPreset")
            or raw.get("preset")
        )
        preset_table = (
            cls.FLUID_REGION_PRESETS if kind == "fluid"
            else cls.SOLID_REGION_PRESETS
        )
        if preset_name and preset_name in preset_table:
            kw.update(preset_table[preset_name])
            # Stash the preset name on the spec for traceability / debugging.
            kw[f"{kind}_preset"] = preset_name

        # ── Layer 2: kind-specific fields ──
        if kind == "fluid":
            # thermo_profile / turbulence_model: explicit > preset > default.
            if "thermo_profile" not in kw:
                kw["thermo_profile"] = raw.get("thermo_profile") or raw.get(
                    "thermoProfile", "gas"
                )
            elif "thermo_profile" in raw:
                kw["thermo_profile"] = raw["thermo_profile"]
            kw["turbulence_model"] = (
                raw.get("turbulence_model")
                or raw.get("turbulenceModel")
                or kw.get("turbulence_model", "kEpsilon")
            )
            # Explicit fluid-physics overrides.
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
        """LLM-generated files for a multi-region case — ``system/controlDict`` only.

        Every other file in the case (``constant/regionProperties``, per-region
        thermo / turb / g, per-region ``system/<region>/{fvSchemes,fvSolution,
        changeDictionaryDict}``, per-region ``0/<region>/{T,U,p,p_rgh,k,ε}`` with
        the coupled ``compressible::turbulentTemperatureCoupledBaffleMixed`` BCs
        at interfaces) is **rendered deterministically** in
        :meth:`render_deterministic_files` and merged in by ``validate_full``.

        Returning the full per-region tree here would put the LLM in the loop
        for those files, producing wrong, unprompted output that the
        deterministic step then has to overwrite — slow and prone to drift.
        Restricting the manifest to ``system/controlDict`` keeps the LLM's
        role narrow (case-level time controls + function objects) and lets
        the deterministic renderer be authoritative for the rest.
        """
        return ["system/controlDict"]

    def all_case_files(self, config: dict[str, Any]) -> list[str]:
        """Full file manifest for the multi-region case (LLM + deterministic).

        Useful for documentation / API responses that want to enumerate
        everything the case ships with.  Not consumed by the LLM codegen
        path — see :meth:`required_files` for that.
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

        ``wallDist { method meshWave; }`` is mandatory in OF 2406 for any
        fluid region with wall-function turbulence (kOmegaSST / kEpsilon).
        Without it the solver aborts at startup with
        ``Entry 'method' not found in dictionary wallDist``.  ``meshWave``
        is the OF tutorial default — fast and works on any mesh.
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
            "wallDist\n{\n    method          meshWave;\n}\n\n"
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

    def _rho_bounds_for_region(self, region: "RegionSpec") -> tuple[float, float]:
        """Return ``(rhoMin, rhoMax)`` safety bounds for the SIMPLE block.

        Pulls ``rho_nominal`` from the matching :attr:`FLUID_REGION_PRESETS`
        entry and brackets it at **0.2× / 2×** — the band used by the ESI
        v1906 ``multiRegionHeaterRadiation`` tutorial for air (rho≈1.16,
        bounds 0.2–2.0).  These are *safety* bounds: tight enough to halt
        an incipient ρ blow-up before it cascades through the h/p
        coupling, loose enough that physical T-driven density excursions
        (cryogenic warm-up, hot wall on a cold liquid) still fit.

        Falls back to **air bounds (0.2, 2.0)** when:

          * the region has no ``fluid_preset`` set,
          * the preset name isn't in :attr:`FLUID_REGION_PRESETS`,
          * the preset entry lacks ``rho_nominal`` (legacy preset).

        Per-fluid nominal densities (kg/m³, at the natural operating point):

          * air      1.20    →  bounds 0.24 – 2.4
          * water    998     →  bounds 200  – 2000
          * oil      870     →  bounds 174  – 1740
          * helium   0.166   →  bounds 0.033 – 0.332
          * ln2      808     →  bounds 162  – 1616
          * lox      1141    →  bounds 228  – 2282
          * lh2      71      →  bounds 14   – 142
          * lng      422     →  bounds 84   – 844
          * lhe      125     →  bounds 25   – 250

        The 0.2× lower bound is critical — without it the ``divide``
        operation in ``compressibleTurbulenceModel`` faults on the first
        cell that drifts toward zero density.
        """
        preset_name = getattr(region, "fluid_preset", None)
        preset = (
            self.FLUID_REGION_PRESETS.get(preset_name)
            if isinstance(preset_name, str) else None
        )
        rho_nom = preset.get("rho_nominal") if isinstance(preset, dict) else None
        if not isinstance(rho_nom, (int, float)) or rho_nom <= 0:
            # Unknown preset — air-like bounds.  Better to clip than to
            # let ρ run unbounded on a steady CHT outer loop.
            return (0.2, 2.0)
        return (0.2 * float(rho_nom), 2.0 * float(rho_nom))

    # ── Per-algorithm tuning constants ────────────────────────────────────
    #
    # SIMPLE (steady chtMultiRegionSimpleFoam) and PIMPLE (transient
    # chtMultiRegionFoam) have fundamentally different pressure-velocity
    # coupling, so they get different default solver settings.  Both
    # tables are derived from the ESI v1906 ``multiRegionHeaterRadiation``
    # tutorial + field experience on buoyancy-driven cryogenic CHT.

    # SIMPLE: a single outer iteration per pseudo-step, so the pressure
    # correction MUST be heavily underrelaxed to keep the buoyancy term
    # from oscillating against itself.  Without ``p_rgh 0.3`` the steady
    # solver shows p_rgh climbing/oscillating around 1e-2 forever while
    # velocity/temperature converge to 1e-5 — observed on the Regascold
    # LN2-water case at 322 steps, 7/9 fields converged but p_rgh
    # diverging.  One non-orthogonal corrector helps if the mesh has any
    # non-orthogonality at the CHT interface (concentric quads usually do).
    _SIMPLE_RELAX_PRGH:  float = 0.3
    _SIMPLE_RELAX_U:     float = 0.3
    _SIMPLE_RELAX_H:     float = 0.7
    _SIMPLE_RELAX_TURB:  float = 0.7
    _SIMPLE_NON_ORTHO:   int   = 1

    # PIMPLE: ``nOuterCorrectors`` is the "mini SIMPLE inside each time
    # step" loop that handles the rho-h-p-U coupling natively, so the
    # static underrelaxation can be looser.  Final-iteration relaxation
    # = 1.0 (canonical OpenFOAM convention) is applied via the
    # ``"<field>Final"`` regex pattern.  Non-orthogonal correctors are
    # the same as SIMPLE; momentumPredictor stays on for buoyant cases.
    _PIMPLE_RELAX_PRGH:    float = 0.7
    _PIMPLE_RELAX_U:       float = 0.7
    _PIMPLE_RELAX_H:       float = 0.7
    _PIMPLE_RELAX_TURB:    float = 0.7
    _PIMPLE_N_OUTER:       int   = 50
    _PIMPLE_N_INNER:       int   = 1
    _PIMPLE_NON_ORTHO:     int   = 1

    def build_fluid_fv_solution(self, region: "RegionSpec") -> str:
        """Per-region ``system/<fluid>/fvSolution`` — branches on algorithm.

        SIMPLE (steady, chtMultiRegionSimpleFoam)
        -----------------------------------------
        Single outer iteration → tight p_rgh underrelaxation needed.
        ``rhoMin/rhoMax`` + ``pRefCell/pRefValue`` bound the steady
        solution; ``nNonOrthogonalCorrectors 1`` for mesh imperfections.
        Field relaxation: rho=1.0, **p_rgh=0.3** (was 0.7, raised oscillation
        on buoyancy-driven CHT).  Equation relaxation: U=0.3, h=0.7,
        turbulence 0.7.  Without these the steady solver oscillates on
        p_rgh around 1e-2 indefinitely.

        PIMPLE (transient, chtMultiRegionFoam)
        --------------------------------------
        ``nOuterCorrectors 50`` runs a mini SIMPLE inside each time step,
        which handles the buoyancy coupling natively — so static
        underrelaxation stays at the looser canonical values (0.7 for
        every field, with ``*Final`` regex resetting the last outer
        corrector to 1.0).  ``nCorrectors 1`` PISO step inside each
        outer.  Same rhoMin/rhoMax safety bounds and pRefCell.

        Both branches use the same solver block (PCG/DIC for rho, GAMG/
        GaussSeidel for p_rgh, PBiCGStab/DILU for U/h/k/ω/ε).
        """
        # Build the equation regex from the fluid region's turbulence model.
        eq_fields = ["U", "h"]
        m = (region.turbulence_model or "").lower()
        is_komega = "komega" in m  # also catches komegasst
        is_keps   = "kepsilon" in m
        if is_komega:
            eq_fields += ["k", "omega"]
        elif is_keps:
            eq_fields += ["k", "epsilon"]
        eq_regex = '"(' + "|".join(eq_fields) + ')"'
        eq_final_regex = '"(' + "|".join(eq_fields) + ')Final"'

        # Per-fluid density bounds — air-like (1.2) bounds 0.2/2.0 are
        # 4 orders of magnitude wrong for water (998) and LOX (1141);
        # see ``_rho_bounds_for_region`` for the per-preset values.
        rho_min, rho_max = self._rho_bounds_for_region(region)

        # Algorithm-specific blocks (single source of truth for each).
        if self.algorithm == "SIMPLE":
            algo_block, relax_block = self._build_simple_blocks(
                region, rho_min, rho_max, is_komega, is_keps,
            )
            # Steady: tighter p_rgh solve (``relTol 0.001`` not the
            # canonical 0.01) because the pressure correction is the
            # only place the buoyancy term gets resolved per outer step.
            p_rgh_reltol = "0.001"
        else:
            algo_block, relax_block = self._build_pimple_blocks(
                region, rho_min, rho_max, is_komega, is_keps,
            )
            # Transient: canonical 0.01 — each PIMPLE outer corrector
            # resolves the residual budget further.
            p_rgh_reltol = "0.01"

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
            "        tolerance       1e-7;\n"
            f"        relTol          {p_rgh_reltol};\n"
            "    }\n\n"
            "    p_rghFinal\n"
            "    {\n"
            "        $p_rgh;\n"
            "        tolerance       1e-7;\n"
            "        relTol          0;\n"
            "    }\n\n"
            f"    {eq_regex}\n"
            "    {\n"
            "        solver          PBiCGStab;\n"
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
            + algo_block + "\n"
            + relax_block + "\n"
            "// ************************************************************************* //\n"
        )

    # ── Algorithm-specific block builders ─────────────────────────────────

    def _build_simple_blocks(
        self,
        region: "RegionSpec",
        rho_min: float,
        rho_max: float,
        is_komega: bool,
        is_keps: bool,
    ) -> tuple[str, str]:
        """SIMPLE algorithm + relaxation blocks for steady chtMultiRegionSimpleFoam.

        Returns ``(algorithm_block, relaxation_block)`` as two formatted
        strings ready to concatenate.  See class-level ``_SIMPLE_RELAX_*``
        constants for the tunings + why they're set this way.
        """
        algo = (
            "SIMPLE\n{\n"
            "    momentumPredictor        yes;\n"
            f"    nNonOrthogonalCorrectors {self._SIMPLE_NON_ORTHO};\n"
            "    pRefCell                 0;\n"
            f"    pRefValue                {region.p_init:g};\n"
            f"    rhoMin                   {rho_min:g};\n"
            f"    rhoMax                   {rho_max:g};\n"
            "}\n"
        )
        # Turbulence equation relaxation, regex-matched
        turb_relax = ""
        if is_komega:
            turb_relax = f'        "(k|omega)"   {self._SIMPLE_RELAX_TURB};\n'
        elif is_keps:
            turb_relax = f'        "(k|epsilon)" {self._SIMPLE_RELAX_TURB};\n'
        relax = (
            "relaxationFactors\n{\n"
            "    fields\n"
            "    {\n"
            "        rho           1.0;\n"
            f"        p_rgh         {self._SIMPLE_RELAX_PRGH};\n"
            "    }\n"
            "    equations\n"
            "    {\n"
            f"        U             {self._SIMPLE_RELAX_U};\n"
            f"        h             {self._SIMPLE_RELAX_H};\n"
            + turb_relax +
            "    }\n"
            "}\n"
        )
        return algo, relax

    def _build_pimple_blocks(
        self,
        region: "RegionSpec",
        rho_min: float,
        rho_max: float,
        is_komega: bool,
        is_keps: bool,
    ) -> tuple[str, str]:
        """PIMPLE algorithm + relaxation blocks for transient chtMultiRegionFoam.

        Notable differences from the SIMPLE branch:

          * ``nOuterCorrectors``/``nCorrectors`` instead of single outer
            iteration — handles pressure-velocity coupling per timestep.
          * No tight ``p_rgh 0.3`` here — the outer correctors do the job
            naturally.  Static relaxation stays at the canonical 0.7.
          * ``*Final`` regex resets the LAST outer iteration's relaxation
            to 1.0 (OpenFOAM convention; transient stability depends on
            the final corrector solving the un-relaxed equation).
          * ``residualControl`` lets PIMPLE exit early when the inner
            loop converges, instead of always running 50 outer iterations.
        """
        algo = (
            "PIMPLE\n{\n"
            "    momentumPredictor        yes;\n"
            f"    nOuterCorrectors         {self._PIMPLE_N_OUTER};\n"
            f"    nCorrectors              {self._PIMPLE_N_INNER};\n"
            f"    nNonOrthogonalCorrectors {self._PIMPLE_NON_ORTHO};\n"
            "    pRefCell                 0;\n"
            f"    pRefValue                {region.p_init:g};\n"
            f"    rhoMin                   {rho_min:g};\n"
            f"    rhoMax                   {rho_max:g};\n"
            # Residual control — PIMPLE exits the outer loop early when
            # residuals drop below these thresholds, saving compute on
            # well-behaved time steps.
            "    residualControl\n"
            "    {\n"
            "        \"(U|h|k|epsilon|omega)\"\n"
            "        {\n"
            "            relTol          0;\n"
            "            tolerance       1e-4;\n"
            "        }\n"
            "        p_rgh\n"
            "        {\n"
            "            relTol          0;\n"
            "            tolerance       1e-4;\n"
            "        }\n"
            "    }\n"
            "}\n"
        )
        # Field relaxation — *Final regex resets last outer corrector to 1.0.
        turb_relax = ""
        if is_komega:
            turb_relax = (
                f'        "(k|omega)"      {self._PIMPLE_RELAX_TURB};\n'
                '        "(k|omega)Final" 1.0;\n'
            )
        elif is_keps:
            turb_relax = (
                f'        "(k|epsilon)"      {self._PIMPLE_RELAX_TURB};\n'
                '        "(k|epsilon)Final" 1.0;\n'
            )
        relax = (
            "relaxationFactors\n{\n"
            "    fields\n"
            "    {\n"
            "        rho           1.0;\n"
            f'        "p_rgh.*"     {self._PIMPLE_RELAX_PRGH};\n'
            "        p_rghFinal    1.0;\n"
            "    }\n"
            "    equations\n"
            "    {\n"
            f'        U             {self._PIMPLE_RELAX_U};\n'
            "        UFinal        1.0;\n"
            f'        h             {self._PIMPLE_RELAX_H};\n'
            "        hFinal        1.0;\n"
            + turb_relax +
            "    }\n"
            "}\n"
        )
        return algo, relax

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
    #
    # The actual builders live in :mod:`._multi_region_bcs` — they're the
    # only MultiRegionBase concern that needs ``config["boundary_conditions"]``
    # and isolating them there keeps this class focused on topology +
    # presets + file dispatching.  We re-export the coupled-T helper here
    # because ``build_change_dictionary_dict`` (below, still part of the
    # base class) reuses the same coupled-block formatting.

    @staticmethod
    def _coupled_T_patches(region: "RegionSpec") -> str:
        """Render the coupled CHT temperature blocks for every interface.

        Kept here for ``build_change_dictionary_dict``; the per-region
        ``0/<region>/T`` renderer in :mod:`._multi_region_bcs` builds the
        same blocks via its private ``_coupled_T_block`` helper.
        """
        from simd_agent.solvers.families import _multi_region_bcs
        if not region.interfaces:
            return ""
        return "".join(
            _multi_region_bcs._coupled_T_block(region, nbr)
            for nbr in region.interfaces
        )

    # ── controlDict function objects (A6) ─────────────────────────────────

    def build_region_function_objects(self, config: dict[str, Any]) -> str:
        """Return a ``functions { … }`` block for the multi-region controlDict.

        Emits, per region:
          * ``volAvg_T_<region>`` — volume-averaged temperature.
          * ``patchAvg_<patch>``  — per-patch area-averaged T (+ U/p for
            fluid regions) on every owned patch.  Wall / symmetry / empty
            patches are skipped because their averages are trivial.

        Every function object carries the OF ``region`` keyword so it
        scopes to that region's mesh after ``splitMeshRegions``.  Output
        goes to the per-region postProcessing tree which the runner's
        B2 log parser already understands.
        """
        from simd_agent.solvers.families import _multi_region_bcs

        regions = self.extract_regions(config)
        blocks: list[str] = []

        for r in regions.all_regions:
            # Volume-averaged temperature for every region (fluid + solid).
            blocks.append(
                f"    volAvg_T_{r.name}\n"
                "    {\n"
                "        type            volFieldValue;\n"
                "        libs            (fieldFunctionObjects);\n"
                f"        region          {r.name};\n"
                "        fields          (T);\n"
                "        operation       volAverage;\n"
                "        regionType      all;\n"
                "        writeFields     false;\n"
                "        writeControl    timeStep;\n"
                "        writeInterval   1;\n"
                "        log             true;\n"
                "    }\n"
            )

            # Per-patch area averages for inlet/outlet patches.  Solid
            # regions have no inlet/outlet so this block emits nothing
            # for them — we only emit walls when explicitly useful.
            patches = _multi_region_bcs.region_patches(r.name, config)
            for patch_name in patches:
                role = _multi_region_bcs.patch_role(patch_name, config)
                if role not in ("inlet", "outlet"):
                    continue
                if r.kind == "fluid":
                    fields = "(T U p)"
                else:
                    fields = "(T)"
                blocks.append(
                    f"    patchAvg_{patch_name}\n"
                    "    {\n"
                    "        type            surfaceFieldValue;\n"
                    "        libs            (fieldFunctionObjects);\n"
                    f"        region          {r.name};\n"
                    "        regionType      patch;\n"
                    f"        name            {patch_name};\n"
                    f"        fields          {fields};\n"
                    "        operation       areaAverage;\n"
                    "        writeFields     false;\n"
                    "        writeControl    timeStep;\n"
                    "        writeInterval   1;\n"
                    "        log             true;\n"
                    "    }\n"
                )

        if not blocks:
            return ""
        return "\nfunctions\n{\n" + "\n".join(blocks) + "}\n"

    def inject_function_objects(
        self, files: dict[str, str], config: dict[str, Any],
    ) -> dict[str, str]:
        """Append the per-region ``functions { … }`` block to ``system/controlDict``.

        Idempotent — if the LLM already wrote a ``functions { … }`` block
        we replace it with the deterministic one so the function-object
        names and ``region`` scoping are always correct.  Single-region
        cases pass through unchanged because :meth:`build_region_function_objects`
        returns an empty string when no regions are detected.
        """
        block = self.build_region_function_objects(config)
        if not block:
            return files
        path = "system/controlDict"
        content = files.get(path, "")
        if not content:
            return files

        import re
        # Strip an existing functions { … } block (depth-balanced) so we
        # don't accumulate duplicates across iterations.
        m = re.search(r"\bfunctions\b\s*\{", content)
        if m:
            start = m.start()
            i = m.end()
            depth = 1
            while i < len(content) and depth > 0:
                ch = content[i]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            if depth == 0:
                content = content[:start] + content[i + 1:]

        # Append before the closing FOAM divider if present, else at end.
        divider = "// *********"
        idx = content.rfind(divider)
        if idx == -1:
            new_content = content.rstrip() + "\n" + block
        else:
            new_content = content[:idx].rstrip() + "\n" + block + "\n" + content[idx:]

        out = dict(files)
        out[path] = new_content
        return out

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

        # 0. Top-level placeholders.  Multi-region cases store all *real*
        # schemes / solver settings under system/<region>/, but OpenFOAM
        # utilities that run before splitMeshRegions (checkMesh, the
        # solver's own startup phase that walks the top-level case before
        # diving into regions) still try to open top-level
        # system/{fvSchemes,fvSolution} and abort if they're missing
        # ("cannot find file …/case/system/fvSchemes").  Emit the
        # placeholder versions defined by ``_build_fv_{solution,schemes}``
        # so those tools have something to read.
        files["system/fvSchemes"] = self._build_fv_schemes(config)
        files["system/fvSolution"] = self._build_fv_solution(config)

        # 1. regionProperties.
        files["constant/regionProperties"] = self.build_region_properties(config)

        # 1b. Top-level constant/g — gravity vector.
        #
        # The Foundation OpenFOAM tutorials (4.x, 10) keep gravity per-fluid
        # at ``constant/<region>/g``.  ESI OpenFOAM v2406's
        # ``chtMultiRegionSimpleFoam`` builds its IOobject for ``g`` such
        # that the lookup resolves to ``case/constant/g`` (top level) and
        # MUST_READ — the solver aborts immediately if it isn't there:
        #
        #     cannot find file ".../case/constant/g"
        #
        # Emit a top-level ``constant/g`` alongside the per-region copies
        # below so the case works on both build families.  Same vector in
        # both files; zero physics cost (~200 bytes).
        if regions.fluid_regions:
            # Reuse the per-region builder (same dimensions/value, same -y
            # Earth gravity by construction), then rewrite the location
            # header so the file isn't tagged as belonging to a region.
            _top_g = self.build_region_gravity(regions.fluid_regions[0])
            files["constant/g"] = _top_g.replace(
                f'location    "constant/{regions.fluid_regions[0].name}"',
                'location    "constant"',
            )

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

        # 4. Per-region 0/<region>/* fields.  Delegated to
        # ``_multi_region_bcs`` so each builder can read per-patch BCs
        # from ``config["boundary_conditions"]`` (A2 — BC bridge).
        from simd_agent.solvers.families import _multi_region_bcs as bcs
        # Per-turbulence-field builders — single source of truth.  The
        # list of fields each model needs lives in ``TURBULENCE_FIELDS``
        # (in ``solvers/base.py``); ``MultiRegionBase.turbulence_fields``
        # appends ``alphat`` for CHT.  Whenever we add a new turbulence
        # model, the only thing that has to change is that table — the
        # renderer is auto-driven.
        _TURB_FIELD_BUILDERS = {
            "k":       bcs.build_region_0_k,
            "epsilon": bcs.build_region_0_epsilon,
            "omega":   bcs.build_region_0_omega,
            "nut":     bcs.build_region_0_nut,
            "alphat":  bcs.build_region_0_alphat,
        }

        for r in regions.fluid_regions:
            files[f"0/{r.name}/T"]     = bcs.build_region_0_T(r, config)
            files[f"0/{r.name}/U"]     = bcs.build_region_0_U(r, config)
            files[f"0/{r.name}/p"]     = bcs.build_region_0_p(r, config)
            files[f"0/{r.name}/p_rgh"] = bcs.build_region_0_p_rgh(r, config)
            for field_name in self.turbulence_fields(r.turbulence_model or ""):
                builder = _TURB_FIELD_BUILDERS.get(field_name)
                if builder is None:
                    # Unknown field name (e.g. nuTilda for Spalart-Allmaras
                    # if a builder hasn't been added yet) — log + skip so
                    # the rest of the case still renders rather than
                    # crashing on a KeyError.
                    logger.warning(
                        "[CHT] no per-region builder for turbulence field %r "
                        "(model=%r, region=%s)",
                        field_name, r.turbulence_model, r.name,
                    )
                    continue
                files[f"0/{r.name}/{field_name}"] = builder(r, config)
        # Solid regions: T is the only field solid thermo actually solves,
        # but ESI v2406's chtMultiRegionSimpleFoam walks every region's
        # objectRegistry at startup and aborts when 0/<solid>/{p,p_rgh,U}
        # don't exist ("cannot find file 0/<solid>/p").  Foundation
        # OpenFOAM-4.x ships top-level 0/{p,p_rgh,U} templates that
        # changeDictionaryDict distributes to solid regions for the same
        # reason.  Render them here with passive content (calculated /
        # noSlip everywhere) so the field is present without affecting
        # the solid heat-equation solution.
        for r in regions.solid_regions:
            files[f"0/{r.name}/T"]     = bcs.build_region_0_T(r, config)
            files[f"0/{r.name}/p"]     = bcs.build_region_0_p(r, config)
            files[f"0/{r.name}/p_rgh"] = bcs.build_region_0_p_rgh(r, config)
            files[f"0/{r.name}/U"]     = bcs.build_region_0_U(r, config)

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