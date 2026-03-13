# simd_agent/case_spec.py
"""CaseSpec — single source of truth for all parallel file generators.

Computed ONCE from (solver, validated_config) before any LLM calls.
All file-generation tasks read from CaseSpec only; they never chain
content from one generated file to another.

Dependency graph (N0-N5 are pure-Python; N6-N10 are LLM generation):

  N0 MeshPatchTruth ──┐
  N1 SolverSpec ───────┼──► N5 FieldSet ──► N7_* ZeroFields (parallel)
  N2 PhaseSpec ────────┤                └──► N8 fvSchemes
  N3 ThermoSpec ───────┤                └──► N9 fvSolution
  N4 TurbulenceSpec ───┘                └──► N10 controlDict
                                         └──► N6 ConstantFiles (parallel)

Execution groups once CaseSpec is ready:
  Group A (parallel): N6, N8, N9, N10
  Group B (parallel): all N7_<field>
  Then: N11 validation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Solver property lookup — single mapping that drives all decisions
# ──────────────────────────────────────────────────────────────────────────────

_SOLVER_PROPS: dict[str, dict[str, Any]] = {
    "simpleFoam": {
        "algorithm": "SIMPLE",
        "pressure_field": "p",
        "transient": False,
        "compressible": False,
        "multiphase": False,
        "energy": "none",
        "needs_gravity": False,
    },
    "rhoSimpleFoam": {
        "algorithm": "SIMPLE",
        "pressure_field": "p",
        "transient": False,
        "compressible": True,
        "multiphase": False,
        "energy": "he",
        "needs_gravity": False,
    },
    "pimpleFoam": {
        "algorithm": "PIMPLE",
        "pressure_field": "p",
        "transient": True,
        "compressible": False,
        "multiphase": False,
        "energy": "none",
        "needs_gravity": False,
    },
    "rhoPimpleFoam": {
        "algorithm": "PIMPLE",
        "pressure_field": "p",
        "transient": True,
        "compressible": True,
        "multiphase": False,
        "energy": "he",
        "needs_gravity": False,
    },
    "icoFoam": {
        "algorithm": "PISO",
        "pressure_field": "p",
        "transient": True,
        "compressible": False,
        "multiphase": False,
        "energy": "none",
        "needs_gravity": False,
    },
    "interFoam": {
        "algorithm": "PIMPLE",
        "pressure_field": "p_rgh",
        "transient": True,
        "compressible": False,
        "multiphase": True,
        "energy": "none",
        "needs_gravity": True,
    },
    "interIsoFoam": {
        "algorithm": "PIMPLE",
        "pressure_field": "p_rgh",
        "transient": True,
        "compressible": False,
        "multiphase": True,
        "energy": "none",
        "needs_gravity": True,
    },
    "compressibleInterFoam": {
        "algorithm": "PIMPLE",
        "pressure_field": "p_rgh",
        "transient": True,
        "compressible": True,
        "multiphase": True,
        "energy": "he",
        "needs_gravity": True,
    },
    "compressibleInterIsoFoam": {
        "algorithm": "PIMPLE",
        "pressure_field": "p_rgh",
        "transient": True,
        "compressible": True,
        "multiphase": True,
        "energy": "he",
        "needs_gravity": True,
    },
    "compressibleMultiphaseInterFoam": {
        "algorithm": "PIMPLE",
        "pressure_field": "p_rgh",
        "transient": True,
        "compressible": True,
        "multiphase": True,
        "energy": "he",
        "needs_gravity": True,
    },
}

# Fields each turbulence model contributes to the 0/ directory
_TURB_FIELDS: dict[str, list[str]] = {
    "laminar": [],
    "none":    [],
    "kOmegaSST": ["k", "omega", "nut"],
    "kEpsilon":  ["k", "epsilon", "nut"],
    "SpalartAllmaras": ["nuTilda", "nut"],
    "kOmega":    ["k", "omega", "nut"],
}


# ──────────────────────────────────────────────────────────────────────────────
# CaseSpec dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CaseSpec:
    """Single source of truth: all metadata needed to generate a complete case.

    Computed in pure Python — no LLM involved.  All parallel file-generation
    tasks read from this object only; they never share generated file content.
    """

    # ── N0 MeshPatchTruth ────────────────────────────────────────────────────
    patch_names: list[str]               # exact names from mesh config
    patch_type_by_name: dict[str, str]   # name → "wall" | "patch" | "empty" | "symmetry"
    is_2d: bool                          # True when any patch has type "empty"

    # ── N1 SolverSpec ────────────────────────────────────────────────────────
    solver: str
    algorithm: str          # "SIMPLE" | "PIMPLE" | "PISO"
    pressure_field: str     # "p" | "p_rgh"
    transient: bool
    compressible: bool
    needs_gravity: bool

    # ── N2 PhaseSpec ─────────────────────────────────────────────────────────
    multiphase: bool
    phase_names: list[str]   # ["water", "air"] etc.
    alpha_fields: list[str]  # ["alpha.water"] etc.

    # ── N3 ThermoSpec ────────────────────────────────────────────────────────
    energy: str              # "none" | "he"
    energy_field: str | None # None | "h" | "e"

    # ── N4 TurbulenceSpec ────────────────────────────────────────────────────
    turbulence_model: str   # "laminar" | "kOmegaSST" | ...
    sim_type: str           # "laminar" | "RAS" | "LES"
    turbulence_fields: list[str]  # ["k", "omega", "nut"] etc.

    # ── N5 FieldSet ──────────────────────────────────────────────────────────
    required_0_fields: list[str]
    required_constant_files: list[str]

    # ── Time control ─────────────────────────────────────────────────────────
    end_time: float
    delta_t: float

    # ── Fields with defaults must come last ───────────────────────────────────
    required_system_files: list[str] = field(
        default_factory=lambda: ["system/controlDict", "system/fvSchemes", "system/fvSolution"]
    )

    # ── fvOptions temperature limits ──────────────────────────────────────────
    # Computed deterministically for compressible energy solvers.
    # The LLM chooses the final max value — these fields provide context and constraints.
    fv_options_t_min: float | None = None
    # Hard EOS ceiling: T where icoPolynomial ρ(T)→0.  max in fvOptions MUST stay below
    # this or density goes negative → SIGFPE.  None for perfectGas/rhoConst (no EOS limit).
    fv_options_eos_t_ceiling: float | None = None
    # All BC temperatures found in the case (K) — LLM uses these to reason about range.
    fv_options_bc_temps: list[float] = field(default_factory=list)

    # ── Computed energy BC values (J/kg) ─────────────────────────────────────
    # Deterministically computed as T * Cp for each temperature-bearing patch.
    # Keys: "inlet", "wall", "internal" — values in J/kg (or None if unknown).
    # Use these in 0/h or 0/e prompts instead of raw temperature values.
    energy_bc_values: dict[str, float | None] = field(default_factory=dict)

    # ── Physical properties (from user config) ────────────────────────────────
    fluid_name: str = ""           # e.g. "liquid nitrogen", "water"
    inlet_velocity: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    inlet_temperature: float | None = None
    wall_temperature: float | None = None
    operating_pressure: float = 101325.0
    nu: float | None = None        # kinematic viscosity [m²/s]
    rho: float | None = None       # density [kg/m³]
    mu: float | None = None        # dynamic viscosity [Pa·s]
    cp: float | None = None        # heat capacity [J/kg·K]
    prandtl: float | None = None   # Prandtl number

    # ── Turbulence initial values ─────────────────────────────────────────────
    # Pre-computed from the frontend's `turbulence` block (k, omega, epsilon, nut).
    # Keys match OpenFOAM field names; used as internalField and inlet BC values.
    turbulence_initial_values: dict[str, float] = field(default_factory=dict)

    # ── Full boundary conditions (raw from validated_config) ─────────────────
    # Stored so individual file generators can look up per-patch, per-field values
    # without needing to re-read validated_config.
    boundary_conditions: dict = field(default_factory=dict)

    # ── VOF initial domain state (multiphase / inter solvers only) ────────────
    # Describes the phase distribution at t=0, before any fluid enters from BCs.
    #   "uniform_gas"          — domain filled with gas/vapour (alpha.liquid = 0 everywhere)
    #   "uniform_liquid"       — domain filled with liquid     (alpha.liquid = 1 everywhere)
    #   "liquid_region_in_gas" — liquid occupies a geometric sub-region (requires setFields)
    #   "gas_region_in_liquid" — gas bubble inside liquid      (requires setFields)
    # For "liquid_region_in_gas" / "gas_region_in_liquid":
    #   • 0/alpha.<phase>.orig is generated (uniform template)
    #   • system/setFieldsDict is generated (geometry specification)
    #   • The runner must execute: cp 0/alpha.<phase>.orig 0/alpha.<phase> && setFields
    initial_phase_layout: str = "uniform_gas"
    # Initial pressure in the domain at t=0 (Pa).  May differ from operating/outlet
    # pressure — e.g. a tank starts pressurised, or an empty pipe starts at ambient.
    initial_domain_pressure: float | None = None
    # Initial bulk temperature in the domain at t=0 (K).  May differ from inlet
    # temperature — e.g. domain starts at saturation temp, inlet injects sub-cooled liquid.
    initial_domain_temperature: float | None = None

    def as_prompt_dict_for_file(self, file_path: str) -> dict[str, Any]:
        """Return a file-specific compact dict — boundary_conditions only for 0/* files.

        Reduces prompt noise for system/constant files that don't need per-patch BC details.
        """
        end_time_val: Any = int(self.end_time) if self.end_time == int(self.end_time) else self.end_time
        base: dict[str, Any] = {
            "solver":           self.solver,
            "algorithm":        self.algorithm,
            "pressure_field":   self.pressure_field,
            "transient":        self.transient,
            "compressible":     self.compressible,
            "is_2d":            self.is_2d,
            "energy":           self.energy,
            "energy_field":     self.energy_field,
            "turbulence_model": self.turbulence_model,
            "sim_type":         self.sim_type,
            "turbulence_fields": self.turbulence_fields,
            "patch_names":      self.patch_names,
            "patch_types":      self.patch_type_by_name,
            "end_time":         end_time_val,
            "delta_t":          self.delta_t,
            "operating_pressure": self.operating_pressure,
            "fluid_name":       self.fluid_name if self.fluid_name else None,
            "rho":  self.rho,
            "nu":   self.nu,
            "mu":   self.mu,
            "cp":   self.cp,
            "Pr":   self.prandtl,
            "turbulence_initial_values": self.turbulence_initial_values if self.turbulence_initial_values else None,
        }
        # Include fvOptions temperature limits for the fvOptions file
        if file_path == "system/fvOptions":
            base["fv_options_t_min"] = self.fv_options_t_min
            base["fv_options_eos_t_ceiling"] = self.fv_options_eos_t_ceiling
            base["fv_options_bc_temps"] = self.fv_options_bc_temps
            base["inlet_temperature"] = self.inlet_temperature
            base["wall_temperature"] = self.wall_temperature
        # Include temperature info for EOS selection in thermophysicalProperties
        if file_path == "constant/thermophysicalProperties":
            base["inlet_temperature"] = self.inlet_temperature
            base["wall_temperature"] = self.wall_temperature
            base["has_heat_transfer"] = self.energy == "he"
            _t_in = self.inlet_temperature
            _t_wall = self.wall_temperature
            base["temperatures_differ"] = (
                _t_in is not None and _t_wall is not None and abs(_t_in - _t_wall) > 1.0
            )
        # Only include heavy boundary_conditions blob for 0/* field files
        if file_path.startswith("0/"):
            base["boundary_conditions"] = self.boundary_conditions
            base["inlet_velocity"] = self.inlet_velocity
            base["inlet_temperature"] = self.inlet_temperature
            base["wall_temperature"] = self.wall_temperature
            base["energy_bc_values_J_per_kg"] = self.energy_bc_values if self.energy_bc_values else None
        # Only include alpha fields and phase info for multiphase solvers
        if self.multiphase:
            base["alpha_fields"] = self.alpha_fields
            base["phase_names"] = self.phase_names
            base["initial_phase_layout"] = self.initial_phase_layout
            base["initial_domain_pressure"] = self.initial_domain_pressure
            base["initial_domain_temperature"] = self.initial_domain_temperature
        # setFieldsDict needs full boundary conditions + geometry context
        if file_path == "system/setFieldsDict":
            base["boundary_conditions"] = self.boundary_conditions
        return base

    def as_prompt_dict(self) -> dict[str, Any]:
        """Return a compact dict for injection into LLM prompts."""
        # Format end_time as int when it is a whole number (avoids "88.0" in controlDict)
        end_time_val: Any = int(self.end_time) if self.end_time == int(self.end_time) else self.end_time
        return {
            "solver":           self.solver,
            "algorithm":        self.algorithm,
            "pressure_field":   self.pressure_field,
            "transient":        self.transient,
            "compressible":     self.compressible,
            "is_2d":            self.is_2d,
            "energy_field":     self.energy_field,
            "turbulence_model": self.turbulence_model,
            "sim_type":         self.sim_type,
            "turbulence_fields": self.turbulence_fields,
            "patch_names":      self.patch_names,
            "patch_types":      self.patch_type_by_name,
            "alpha_fields":     self.alpha_fields,
            "end_time":         end_time_val,
            "delta_t":          self.delta_t,
            "inlet_velocity":   self.inlet_velocity,
            "inlet_temperature": self.inlet_temperature,
            "wall_temperature":  self.wall_temperature,
            "operating_pressure": self.operating_pressure,
            "nu":   self.nu,
            "rho":  self.rho,
            "mu":   self.mu,
            "cp":   self.cp,
            "Pr":   self.prandtl,
            # Full per-patch boundary conditions so LLM can see exact types/values
            "boundary_conditions": self.boundary_conditions,
            # Pre-computed enthalpy/energy BC values (J/kg) — use these for 0/h or 0/e,
            # NEVER use raw temperature values in an energy field.
            "energy_bc_values_J_per_kg": self.energy_bc_values if self.energy_bc_values else None,
            # Pre-computed turbulence field initial values from frontend (k, omega, epsilon, nut).
            # Use these for internalField and inlet fixedValue in 0/k, 0/omega, etc.
            "turbulence_initial_values": self.turbulence_initial_values if self.turbulence_initial_values else None,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Builder
# ──────────────────────────────────────────────────────────────────────────────

def build_case_spec(solver: str, validated_config: dict[str, Any]) -> CaseSpec:
    """Compute CaseSpec deterministically — no LLM, no side effects.

    This is the ONLY function that reads validated_config.  All downstream
    generators read from the returned CaseSpec only.
    """
    props = _SOLVER_PROPS.get(solver, _SOLVER_PROPS["simpleFoam"])
    phys = validated_config.get("physics", {}) or {}

    # ── N0: Mesh patch map ───────────────────────────────────────────────────
    mesh_raw = validated_config.get("mesh", {})
    mesh = mesh_raw if isinstance(mesh_raw, dict) else {}
    mesh_patches_raw = mesh.get("patches", [])

    patch_names: list[str] = []
    patch_type_by_name: dict[str, str] = {}
    is_2d = False

    for mp in mesh_patches_raw:
        if isinstance(mp, dict):
            name = mp.get("name", "")
            ptype = mp.get("type", "patch")
        elif hasattr(mp, "name"):
            name = mp.name
            ptype = getattr(mp, "type", "patch")
        else:
            continue
        if not name:
            continue
        # Force frontAndBack → empty (2D constraint)
        nl = name.lower().replace("_", "")
        if nl in ("frontandback", "frontback"):
            ptype = "empty"
            is_2d = True
        if ptype == "empty":
            is_2d = True
        patch_names.append(name)
        patch_type_by_name[name] = ptype

    # If no mesh patches in config, derive from boundary_conditions keys
    if not patch_names:
        bcs = validated_config.get("boundary_conditions", {}) or {}
        for k in bcs:
            nl = k.lower().replace("_", "")
            ptype = "empty" if nl in ("frontandback", "frontback") else "patch"
            if nl in ("frontandback", "frontback"):
                is_2d = True
                ptype = "empty"
            elif nl == "wall":
                ptype = "wall"
            patch_names.append(k)
            patch_type_by_name[k] = ptype

    # ── N1: Solver spec ──────────────────────────────────────────────────────
    algorithm     = props["algorithm"]
    pressure_field = props["pressure_field"]
    transient     = props["transient"]
    compressible  = props["compressible"]
    needs_gravity = props["needs_gravity"]

    # ── N2: Phase spec ───────────────────────────────────────────────────────
    multiphase = props["multiphase"]
    phase_names_cfg = (
        validated_config.get("phases")
        or phys.get("phases")
        or []
    )
    if multiphase and not phase_names_cfg:
        # Derive phase names from fluid name — avoid hardcoding "water/air" for cryogenic cases
        _fluid_name_lower = (
            validated_config.get("fluid", {}) or phys.get("fluid", {}) or {}
        )
        _fluid_name_lower = (_fluid_name_lower.get("name") or "").lower()
        if "helium" in _fluid_name_lower or "lhe" in _fluid_name_lower:
            phase_names_cfg = ["liquidHelium", "heliumVapour"]
        elif "nitrogen" in _fluid_name_lower or "ln2" in _fluid_name_lower:
            phase_names_cfg = ["liquidNitrogen", "nitrogenVapour"]
        elif "hydrogen" in _fluid_name_lower or "lh2" in _fluid_name_lower:
            phase_names_cfg = ["liquidHydrogen", "hydrogenVapour"]
        elif "oxygen" in _fluid_name_lower or "lox" in _fluid_name_lower:
            phase_names_cfg = ["liquidOxygen", "oxygenVapour"]
        elif "methane" in _fluid_name_lower or "lng" in _fluid_name_lower:
            phase_names_cfg = ["liquidMethane", "methaneVapour"]
        elif "water" in _fluid_name_lower or "h2o" in _fluid_name_lower:
            phase_names_cfg = ["water", "air"]
        else:
            phase_names_cfg = ["liquid", "gas"]
    alpha_fields: list[str] = []
    if multiphase and phase_names_cfg:
        # VOF: only the PRIMARY alpha field for 2-phase
        alpha_fields = [f"alpha.{phase_names_cfg[0]}"]

    # ── N3: Thermo spec ──────────────────────────────────────────────────────
    energy = props["energy"]
    energy_field: str | None = None
    if energy == "he":
        # Default: use sensibleEnthalpy → "h" for rho* solvers.
        # energy_field is used in fvSchemes/fvSolution only.
        # We do NOT generate a 0/h or 0/e file — the thermo package initialises
        # the energy field from 0/T at startup (standard OpenFOAM tutorial pattern).
        # Generating 0/h causes "Negative initial temperature" crashes because
        # back-conversion from h to T is reference-point dependent.
        energy_field = "h"
        if validated_config.get("energy_formulation", "") == "sensibleInternalEnergy":
            energy_field = "e"

    # ── N4: Turbulence spec ──────────────────────────────────────────────────
    turb_model = (
        validated_config.get("turbulence_model")
        or phys.get("turbulence_model")
        or ""
    )
    flow_regime = (
        validated_config.get("flow_regime")
        or phys.get("flow_regime")
        or "turbulent"
    )
    if not turb_model or flow_regime == "laminar":
        turb_model = "laminar"
    turb_fields = _TURB_FIELDS.get(turb_model, _TURB_FIELDS["kOmegaSST"])
    if turb_model == "laminar":
        sim_type = "laminar"
    elif turb_model.startswith("S"):  # SpalartAllmaras, Smagorinsky → LES possible
        sim_type = "RAS"
    else:
        sim_type = "RAS"

    # If energy solver + turbulence: add alphat
    if compressible and turb_fields:
        if "alphat" not in turb_fields:
            turb_fields = list(turb_fields) + ["alphat"]

    # ── N4b: Turbulence initial values (from frontend's turbulence block) ────
    # The frontend pre-computes k, omega, epsilon, nut.  Read them once here so
    # all downstream generators share the same values — no re-derivation needed.
    _turb_cfg = validated_config.get("turbulence", {}) or {}
    turbulence_initial_values: dict[str, float] = {}
    for _tf, _aliases in [
        ("k",       ["k"]),
        ("omega",   ["omega"]),
        ("epsilon", ["epsilon"]),
        ("nut",     ["nut"]),
    ]:
        for _alias in _aliases:
            _v = _safe_float(_turb_cfg.get(_alias))
            if _v is not None:
                turbulence_initial_values[_tf] = _v
                break

    # ── N5: Field set ────────────────────────────────────────────────────────
    required_0 = ["U", pressure_field]  # always
    # compressibleInterFoam/compressibleInterIsoFoam/compressibleMultiphaseInterFoam
    # read BOTH p (absolute, MUST_READ) AND p_rgh (modified pressure) at startup.
    # interFoam/interIsoFoam only need p_rgh.
    if pressure_field == "p_rgh" and compressible:
        required_0.append("p")
    if energy == "he":
        # Include 0/T for temperature boundary conditions.
        # Do NOT include the energy field (h/e) — the thermo package reads T
        # and initialises h/e internally.  Providing 0/h causes "Negative
        # initial temperature" because the back-conversion is reference-sensitive.
        required_0.append("T")
    required_0 += alpha_fields
    required_0 += [f for f in turb_fields if f not in required_0]

    required_0_fields = [f"0/{f}" for f in required_0]

    # ── N6: Constant files ───────────────────────────────────────────────────
    required_const: list[str] = []
    if compressible:
        required_const.append("constant/thermophysicalProperties")
        # compressibleInterFoam family uses per-phase thermo files:
        # constant/thermophysicalProperties.<phase1> and .<phase2>
        if multiphase and phase_names_cfg:
            for _phase in phase_names_cfg:
                required_const.append(f"constant/thermophysicalProperties.{_phase}")
    else:
        required_const.append("constant/transportProperties")
    # turbulenceProperties always except icoFoam (laminar, no model needed strictly,
    # but we include it for safety)
    required_const.append("constant/turbulenceProperties")
    if needs_gravity:
        required_const.append("constant/g")

    # ── Time control ─────────────────────────────────────────────────────────
    solver_cfg = validated_config.get("solver", {})
    if not isinstance(solver_cfg, dict):
        solver_cfg = {}

    if not transient:
        end_time = float(
            validated_config.get("max_iterations")
            or solver_cfg.get("max_iterations")
            or solver_cfg.get("maxIterations")
            or 1000
        )
        delta_t = 1.0
    else:
        end_time = float(
            validated_config.get("end_time")
            or solver_cfg.get("endTime")
            or solver_cfg.get("end_time")
            or validated_config.get("max_iterations")
            or 10.0
        )
        delta_t = float(
            validated_config.get("delta_t")
            or solver_cfg.get("deltaT")
            or solver_cfg.get("delta_t")
            or 0.001
        )

    # ── Physical properties ───────────────────────────────────────────────────
    fluid = (
        validated_config.get("fluid", {})
        or validated_config.get("material", {})
        or phys.get("fluid", {})
        or {}
    )
    bcs = validated_config.get("boundary_conditions", {}) or {}
    inlet_bc = bcs.get("inlet", {}) or {}

    def _bc_value(field_entry: Any) -> Any:
        """Unwrap a BC entry that may be a flat value OR {"type": ..., "value": ...} dict."""
        if isinstance(field_entry, dict):
            return field_entry.get("value")
        return field_entry

    def _vel(bc: dict) -> list[float]:
        raw = bc.get("velocity") or bc.get("U")

        if isinstance(raw, dict) and raw.get("type") == "flowRateInletVelocity":
            mv = raw.get("meanVelocity")
            if isinstance(mv, (int, float)):
                return [float(mv), 0.0, 0.0]
            return [0.0, 0.0, 0.0]

        v = _bc_value(raw)
        if isinstance(v, (list, tuple)) and len(v) >= 3:
            return [float(v[0]), float(v[1]), float(v[2])]
        if isinstance(v, (int, float)):
            return [float(v), 0.0, 0.0]
        mag = bc.get("velocity_magnitude") or bc.get("speed") or bc.get("U_mag")
        if mag is not None:
            return [float(_bc_value(mag) or mag), 0.0, 0.0]
        return [0.0, 0.0, 0.0]

    inlet_vel = _vel(inlet_bc)
    inlet_T = _safe_float(
        _bc_value(inlet_bc.get("temperature") or inlet_bc.get("T"))
        or validated_config.get("inlet_temperature")
    )
    wall_bc = bcs.get("wall", {}) or {}
    wall_T = _safe_float(
        _bc_value(wall_bc.get("temperature") or wall_bc.get("T"))
        or validated_config.get("wall_temperature")
    )
    # Operating pressure: prefer explicit config value; fall back to outlet fixedValue, then inlet fixedValue
    _inlet_p_raw = inlet_bc.get("pressure")
    _inlet_p_val = _bc_value(_inlet_p_raw) if isinstance(_inlet_p_raw, dict) else None
    # Only use inlet pressure value if it looks like an absolute pressure (> 1000 Pa)
    _inlet_p = _inlet_p_val if (_inlet_p_val is not None and isinstance(_inlet_p_val, (int, float)) and _inlet_p_val > 1000) else None
    outlet_bc = bcs.get("outlet", {}) or {}
    _outlet_p_raw = outlet_bc.get("pressure")
    _outlet_p_val = _bc_value(_outlet_p_raw) if isinstance(_outlet_p_raw, dict) else None
    _outlet_p = _outlet_p_val if (_outlet_p_val is not None and isinstance(_outlet_p_val, (int, float)) and _outlet_p_val > 1000) else None
    op_p = _safe_float(
        validated_config.get("operating_pressure")
        or validated_config.get("pressure")
        or _outlet_p
        or _inlet_p
    ) or 101325.0

    fluid_name = (fluid.get("name") or "").strip()
    nu  = _safe_float(fluid.get("nu") or fluid.get("kinematic_viscosity"))
    rho = _safe_float(fluid.get("rho") or fluid.get("density"))
    mu  = _safe_float(fluid.get("mu") or fluid.get("dynamic_viscosity"))
    cp  = _safe_float(fluid.get("Cp") or fluid.get("cp") or fluid.get("specific_heat"))
    pr  = _safe_float(fluid.get("Pr") or fluid.get("prandtl"))

    # Derive nu from mu/rho if only those are given
    if nu is None and mu is not None and rho is not None and rho > 0:
        nu = mu / rho

    def _normalize_inlet_u_bc(ibc: dict, _rho: float | None) -> None:
        """Ensure flowRateInletVelocity contains required keys and a safe placeholder value.

        For incompressible solvers (icoFoam, pimpleFoam, simpleFoam):
          - massFlowRate is converted to volumetricFlowRate = mdot / rho so the LLM
            never needs to write a `rho` keyword (those solvers have no rho field).
          - If rho is unavailable, fall back to rhoInlet only (no `rho` word keyword).
        For compressible solvers (rhoPimpleFoam, rhoSimpleFoam):
          - massFlowRate is kept as-is; rhoInlet is added as a startup-fallback scalar.
          - `rho rho;` is NOT injected — the BC defaults to the "rho" field implicitly,
            and rhoInlet covers the startup phase before the field is available.
        """
        u = ibc.get("velocity") or ibc.get("U")
        if not isinstance(u, dict):
            return
        if u.get("type") != "flowRateInletVelocity":
            return
        for key in ("massFlowRate", "volumetricFlowRate", "meanVelocity"):
            if key in ibc and key not in u:
                u[key] = ibc[key]
        if "massFlowRate" in u:
            if not compressible:
                # Incompressible solvers (icoFoam, pimpleFoam, simpleFoam) have NO rho
                # field — writing `rho rho;` causes a fatal IO error at runtime.
                # Convert to volumetricFlowRate = mdot / rho so the BC is density-free.
                if _rho is not None and _rho > 0:
                    try:
                        _mdot = float(u["massFlowRate"])
                        _q = _mdot / _rho
                        del u["massFlowRate"]
                        u["volumetricFlowRate"] = _q
                        # No rho or rhoInlet needed for volumetricFlowRate
                    except (TypeError, ValueError):
                        # Conversion failed — keep massFlowRate + rhoInlet, no rho word
                        u.setdefault("rhoInlet", float(_rho))
                # else: no rho available — leave as-is, LLM must handle
            else:
                # Compressible: rho field exists at runtime; rhoInlet is the startup fallback.
                # Do NOT inject `rho rho;` — the field name defaults to "rho" implicitly.
                if _rho is not None:
                    u.setdefault("rhoInlet", float(_rho))
        u["value"] = [0.0, 0.0, 0.0]

    _normalize_inlet_u_bc(inlet_bc, rho)

    # ── fvOptions temperature limits ─────────────────────────────────────────
    # Provides context and constraints for the LLM to choose appropriate limitTemperature
    # bounds.  The LLM reasons about the right max; Python enforces the hard EOS ceiling.
    fv_options_t_min: float | None = None
    fv_options_eos_t_ceiling: float | None = None
    fv_options_bc_temps: list[float] = []
    _required_system = ["system/controlDict", "system/fvSchemes", "system/fvSolution"]

    if energy == "he":
        # Collect ALL BC temperatures — every patch, every temperature key variant
        _all_bc_t: list[float] = []
        for _pbc in bcs.values():
            if not isinstance(_pbc, dict):
                continue
            for _tk in ("temperature", "T", "T_inlet", "T_wall", "inlet_temperature", "wall_temperature"):
                _tv_raw = _pbc.get(_tk)
                if isinstance(_tv_raw, dict):
                    _tv = _tv_raw.get("value") or _tv_raw.get("uniform")
                elif isinstance(_tv_raw, (int, float)):
                    _tv = float(_tv_raw)
                else:
                    _tv = None
                if _tv is not None:
                    try:
                        _all_bc_t.append(float(_tv))
                    except (TypeError, ValueError):
                        pass
        # Also include top-level temperature values from validated_config
        for _top_key in ("inlet_temperature", "wall_temperature", "temperature"):
            _top_v = _safe_float(validated_config.get(_top_key))
            if _top_v is not None:
                _all_bc_t.append(_top_v)
        # Include explicitly resolved inlet_T / wall_T
        if inlet_T is not None and inlet_T not in _all_bc_t:
            _all_bc_t.append(inlet_T)
        if wall_T is not None and wall_T not in _all_bc_t:
            _all_bc_t.append(wall_T)
        fv_options_bc_temps = sorted(set(_all_bc_t))

        _min_t = min(_all_bc_t) if _all_bc_t else (inlet_T or 200.0)
        fv_options_t_min = max(1.0, _min_t * 0.5)

        # Hard EOS ceiling: for icoPolynomial ρ(T) = a0 + a1·T, ρ→0 at T = a0/|a1|.
        # The LLM must choose max < eos_t_ceiling or density goes negative → SIGFPE.
        # Threshold 30 kg/m³ catches all cryogenic liquids including LHe (~125 kg/m³)
        # and LH2 (~71 kg/m³) which are below the old 200 kg/m³ cutoff.
        if rho is not None and rho > 30.0:
            _T_ref = inlet_T if inlet_T is not None else 300.0
            if _T_ref < 8.0:
                _a1 = -5.0   # liquid helium (He-4): ~125 kg/m³ at 4.2 K
            elif _T_ref < 35.0:
                _a1 = -0.7   # liquid hydrogen: ~71 kg/m³ at 20 K
            elif _T_ref < 100.0:
                _a1 = -4.7   # LN2 / LOX / LAr: ~800–1140 kg/m³
            else:
                _a1 = -0.5   # other dense liquids (water, oil, etc.)
            _a0 = rho - _a1 * _T_ref
            fv_options_eos_t_ceiling = abs(_a0) / abs(_a1)

        # limitTemperature fvOption calls he() on the thermo object.
        # twoPhaseMixtureThermo (used by compressibleInterFoam, compressibleInterIsoFoam,
        # compressibleMultiphaseInterFoam) does NOT implement he() — including fvOptions
        # with limitTemperature crashes those solvers immediately at startup:
        #   "Not implemented: twoPhaseMixtureThermo::he()"
        _INTER_SOLVERS_NO_FVOPTIONS = {
            "compressibleInterFoam",
            "compressibleInterIsoFoam",
            "compressibleMultiphaseInterFoam",
        }

        # fvOptions limitTemperature is needed ONLY when a wall is genuinely hotter
        # than the inlet fluid — i.e. there is actual heat exchange that could drive
        # the bulk temperature above the icoPolynomial EOS ceiling.
        #
        # Two cases that do NOT need fvOptions:
        #   1. Adiabatic walls (no fixedValue T at all) — temperature stays near inlet T.
        #   2. Isothermal walls (wall_T ≈ inlet_T) — wall keeps fluid at inlet temperature,
        #      no net heat flux, temperature never drifts above inlet T.
        #
        # Only case that DOES need fvOptions:
        #   Wall is HOTTER than inlet (wall_T > inlet_T + 10 K) → fluid heats up and
        #   could eventually reach or exceed the EOS ceiling → need limitTemperature.
        #
        # Note: inlet_T is NOT heat transfer — it's just the EOS initialisation value.

        # Collect all wall temperatures
        _wall_temps_found: list[float] = []
        if wall_T is not None:
            _wall_temps_found.append(wall_T)
        for _pname in patch_names:
            if patch_type_by_name.get(_pname) != "wall":
                continue
            _pbc = bcs.get(_pname, {}) or {}
            _t_raw = _pbc.get("temperature") or _pbc.get("T")
            _t_val = _bc_value(_t_raw) if isinstance(_t_raw, dict) else _t_raw
            if _t_val is not None:
                try:
                    _wall_temps_found.append(float(_t_val))
                except (TypeError, ValueError):
                    pass

        _wall_heats_fluid = False
        if _wall_temps_found:
            _max_wall_t = max(_wall_temps_found)
            if inlet_T is not None:
                # fvOptions needed only when wall is significantly hotter than inlet
                _wall_heats_fluid = _max_wall_t > inlet_T + 10.0
            else:
                # No inlet temperature to compare — conservative: include fvOptions
                # only if wall temperature suggests a hot-wall scenario (above 200K,
                # which covers most cryogenic inlet temperatures)
                _wall_heats_fluid = _max_wall_t > 200.0

        if _wall_heats_fluid and solver not in _INTER_SOLVERS_NO_FVOPTIONS:
            _required_system.append("system/fvOptions")

    # ── N3-energy: No 0/h or 0/e file is generated ───────────────────────────
    # The thermo package initialises the energy field from 0/T at startup.
    # Providing 0/h causes "Negative initial temperature" because OpenFOAM
    # back-converts h→T using a reference that differs from what the LLM used.
    # energy_field is kept for fvSchemes/fvSolution naming only.
    energy_bc_values: dict[str, float | None] = {}  # intentionally empty — 0/h not generated

    # ── VOF initial domain state (multiphase only) ────────────────────────────
    initial_phase_layout = "uniform_gas"
    initial_domain_pressure: float | None = None
    initial_domain_temperature: float | None = None
    if multiphase:
        # Allow user to override from config
        _cfg_layout = (
            validated_config.get("initial_phase_layout")
            or phys.get("initial_phase_layout")
        )
        if _cfg_layout in (
            "uniform_gas", "uniform_liquid",
            "liquid_region_in_gas", "gas_region_in_liquid",
        ):
            initial_phase_layout = _cfg_layout
        else:
            # Default: gas-filled domain — liquid enters from inlet.
            # Physical reality for cryogenic injection (LN2/LH2/LOX/LHe):
            # pipe/vessel starts as vapour; liquid is injected from inlet.
            initial_phase_layout = "uniform_gas"

        # Initial domain pressure — may differ from operating/outlet pressure
        _cfg_init_p = _safe_float(
            validated_config.get("initial_domain_pressure")
            or phys.get("initial_domain_pressure")
        )
        initial_domain_pressure = _cfg_init_p if _cfg_init_p is not None else op_p

        # Initial domain temperature — may differ from inlet temperature
        _cfg_init_t = _safe_float(
            validated_config.get("initial_domain_temperature")
            or phys.get("initial_domain_temperature")
        )
        # Default: use inlet temperature as a reasonable starting point.
        # For gas-filled domains this means the vapour starts near the inlet temperature.
        # The LLM may choose a different value (e.g. saturation temp) when guided by the
        # fluid pack prompt.
        initial_domain_temperature = _cfg_init_t if _cfg_init_t is not None else inlet_T

        # Non-uniform layouts require setFields utility before the solver runs:
        #   • system/setFieldsDict specifies the geometry
        #   • 0/alpha.<phase>.orig is the uniform template (setFields overwrites alpha.*)
        _needs_set_fields = initial_phase_layout in (
            "liquid_region_in_gas", "gas_region_in_liquid"
        )
        if _needs_set_fields:
            if "system/setFieldsDict" not in _required_system:
                _required_system.append("system/setFieldsDict")
            # Replace 0/alpha.<phase> entries with 0/alpha.<phase>.orig
            required_0_fields = [
                f"{fp}.orig" if fp.startswith("0/alpha.") else fp
                for fp in required_0_fields
            ]

    spec = CaseSpec(
        # N0
        patch_names=patch_names,
        patch_type_by_name=patch_type_by_name,
        is_2d=is_2d,
        # N1
        solver=solver,
        algorithm=algorithm,
        pressure_field=pressure_field,
        transient=transient,
        compressible=compressible,
        needs_gravity=needs_gravity,
        # N2
        multiphase=multiphase,
        phase_names=phase_names_cfg,
        alpha_fields=alpha_fields,
        # N3
        energy=energy,
        energy_field=energy_field,
        # N4
        turbulence_model=turb_model,
        sim_type=sim_type,
        turbulence_fields=turb_fields,
        # N5
        required_0_fields=required_0_fields,
        required_constant_files=required_const,
        required_system_files=_required_system,
        # Time
        end_time=end_time,
        delta_t=delta_t,
        # Physical
        fluid_name=fluid_name,
        inlet_velocity=inlet_vel,
        inlet_temperature=inlet_T,
        wall_temperature=wall_T,
        operating_pressure=op_p,
        nu=nu,
        rho=rho,
        mu=mu,
        cp=cp,
        prandtl=pr,
        # Computed energy BC values (J/kg) — deterministic, not LLM-guessed
        energy_bc_values=energy_bc_values,
        # Pre-computed turbulence initial values (from frontend turbulence block)
        turbulence_initial_values=turbulence_initial_values,
        # Raw BCs — stored for per-field prompt injection
        boundary_conditions=dict(bcs),
        # fvOptions temperature limits
        fv_options_t_min=fv_options_t_min,
        fv_options_eos_t_ceiling=fv_options_eos_t_ceiling,
        fv_options_bc_temps=fv_options_bc_temps,
        # VOF initial domain state
        initial_phase_layout=initial_phase_layout,
        initial_domain_pressure=initial_domain_pressure,
        initial_domain_temperature=initial_domain_temperature,
    )

    logger.info(
        f"[CASE_SPEC] solver={solver}  algorithm={algorithm}  "
        f"pressure={pressure_field}  energy={energy_field}  "
        f"turbulence={turb_model}  patches={patch_names}  "
        f"0/fields={[f.split('/', 1)[1] for f in required_0_fields]}  "
        f"const={required_const}  "
        f"init_layout={initial_phase_layout}  "
        f"init_p={initial_domain_pressure}  init_T={initial_domain_temperature}"
    )
    return spec


def _safe_float(v: Any) -> float | None:
    """Convert value to float, returning None on failure."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
