"""CaseSpec dataclass — the resolved plan all file renderers consume.

CaseSpec is the single source of truth.  It is computed ONCE from
(solver, validated_config, mesh_data) by ``build_case_spec`` and is
deliberately free of LLM-derived strings — every field is either:
  * raw data lifted from the validated config (BC values, fluid props), or
  * a deterministic decision computed in Python (solver, profile, mesh tier).

Phase 1 of the LLM/validator-boundary redesign adds *optional* typed sub-
models (`FluidThermo`, `PressureSolverStrategy`, `CompressibleBounds`,
`InletTurbulence`).  These fields default to ``None`` and are not yet
populated by the resolver nor read by the renderers — Phase 2 wires them
in, Phase 3 deletes the equivalent regex post-validators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .strategies import (
    CompressibleBounds,
    FluidThermo,
    InletTurbulence,
    PressureSolverStrategy,
    TurbulenceSpec,
)


# ── Solver-property lookup (legacy fallback when registry unavailable) ──────

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
    # Buoyancy-driven (natural convection / HVAC / gravity-dominated heat transfer).
    # heRhoThermo — density varies with T. Solves p_rgh; p is a calculated field.
    # constant/g REQUIRED. Both 0/p_rgh (primary) and 0/p (calculated) are generated.
    "buoyantSimpleFoam": {
        "algorithm": "SIMPLE",
        "pressure_field": "p_rgh",
        "transient": False,
        "compressible": True,
        "multiphase": False,
        "energy": "he",
        "needs_gravity": True,
    },
    "buoyantPimpleFoam": {
        "algorithm": "PIMPLE",
        "pressure_field": "p_rgh",
        "transient": True,
        "compressible": True,
        "multiphase": False,
        "energy": "he",
        "needs_gravity": True,
    },
}

# Fields each turbulence model contributes to the 0/ directory.
_TURB_FIELDS: dict[str, list[str]] = {
    "laminar": [],
    "none":    [],
    "kOmegaSST": ["k", "omega", "nut"],
    "kEpsilon":  ["k", "epsilon", "nut"],
    "SpalartAllmaras": ["nuTilda", "nut"],
    "kOmega":    ["k", "omega", "nut"],
}


# ── CaseSpec ─────────────────────────────────────────────────────────────────


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
    energy_field: str | None  # None | "h" | "e"

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
    max_co: float = 0.5               # Courant limit: 2.0 for PIMPLE, 0.5 for others
    write_interval: float = 1.0       # file write interval (seconds for transient, 1 for steady)
    func_write_interval: float = 1.0  # function object write interval (seconds for transient runTime)
    max_delta_t: float = 1.0          # upper bound on adaptive deltaT

    # ── Fields with defaults must come last ───────────────────────────────────
    required_system_files: list[str] = field(
        default_factory=lambda: ["system/controlDict", "system/fvSchemes", "system/fvSolution"]
    )

    # ── fvOptions temperature limits ──────────────────────────────────────────
    # Computed deterministically for compressible energy solvers.
    fv_options_t_min: float | None = None
    # Hard EOS ceiling: T where icoPolynomial ρ(T)→0.  max in fvOptions MUST
    # stay below this or density goes negative → SIGFPE.  None for
    # perfectGas/rhoConst (no EOS limit).
    fv_options_eos_t_ceiling: float | None = None
    fv_options_bc_temps: list[float] = field(default_factory=list)

    # ── Computed energy BC values (J/kg) ─────────────────────────────────────
    # T * Cp for each temperature-bearing patch — used in 0/h or 0/e prompts.
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
    #   "uniform_gas"          — domain filled with gas/vapour
    #   "uniform_liquid"       — domain filled with liquid
    #   "liquid_region_in_gas" — liquid occupies a geometric sub-region (requires setFields)
    #   "gas_region_in_liquid" — gas bubble inside liquid       (requires setFields)
    initial_phase_layout: str = "uniform_gas"
    initial_domain_pressure: float | None = None
    initial_domain_temperature: float | None = None

    # ── Mesh quality-driven numerics (from OpenFOAM checkMesh) ───────────
    mesh_max_non_orthogonality: float | None = None
    mesh_max_skewness: float | None = None
    mesh_max_aspect_ratio: float | None = None
    use_simplec: bool = False
    n_non_ortho_correctors: int = 1
    mesh_quality_tier: str = "unknown"  # "good" | "moderate" | "poor" | "unknown"

    # ── Thermo profile (gas vs cryogenic liquid) ────────────────────────────
    # Drives every numerical choice for compressible solvers.
    thermo_profile: str = "gas"        # "gas" | "cryogenic"
    rho_min: float | None = None       # SIMPLE/PIMPLE density floor (compressible only)
    rho_max: float | None = None       # SIMPLE/PIMPLE density ceiling
    p_min: float | None = None         # SIMPLE/PIMPLE pressure floor
    p_max: float | None = None         # SIMPLE/PIMPLE pressure ceiling
    transonic: bool = False            # Enable transonic flux scheme (Mach > 0.5)
    mach_estimate: float | None = None  # Inlet Mach number estimate

    # ── Phase 1 of the typed-strategies migration ───────────────────────────
    # These hold resolved decisions in a strict, validatable form.  They
    # default to None because:
    #   • The resolver does not populate them yet (Phase 2)
    #   • The renderers do not read them yet  (Phase 3)
    # Existing scalar fields above are kept in sync until Phases 2 & 3 land
    # so byte-for-byte output is preserved across the refactor.
    thermo_strategy: FluidThermo | None = None
    pressure_solver_strategy: PressureSolverStrategy | None = None
    compressible_bounds_strategy: CompressibleBounds | None = None
    inlet_turbulence_strategy: list[InletTurbulence] = field(default_factory=list)
    # Case-wide turbulence — resolved deterministically from solver
    # capabilities + config.  Replaces the (turbulence_model + sim_type)
    # scalars above for renderers in Phase 4+.
    turbulence_spec: TurbulenceSpec | None = None

    # ── Prompt-dict helpers (unchanged interface) ───────────────────────────

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
            # operating_pressure is only relevant for compressible solvers (Pa).
            # For incompressible solvers, pressure is kinematic gauge (m²/s²) — value is 0.
            # Including 101325 in the prompt for incompressible confuses the LLM.
            "operating_pressure": self.operating_pressure if self.compressible else None,
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
        # Include transient time control for controlDict
        if file_path == "system/controlDict":
            base["max_co"] = self.max_co
            base["write_interval"] = self.write_interval
            base["max_delta_t"] = self.max_delta_t
        # Include mesh quality-driven numerics for fvSolution
        if file_path == "system/fvSolution":
            base["use_simplec"] = self.use_simplec
            base["n_non_ortho_correctors"] = self.n_non_ortho_correctors
            base["mesh_quality_tier"] = self.mesh_quality_tier
            base["thermo_profile"] = self.thermo_profile
            base["rho_min"] = self.rho_min
            base["rho_max"] = self.rho_max
            base["transonic"] = self.transonic
        # Thermo profile context for fvSchemes (div scheme choice)
        if file_path == "system/fvSchemes":
            base["thermo_profile"] = self.thermo_profile
            base["mach_estimate"] = self.mach_estimate
        # Thermo profile context for thermophysicalProperties (EOS choice)
        if file_path == "constant/thermophysicalProperties":
            base["thermo_profile"] = self.thermo_profile
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
            "operating_pressure": self.operating_pressure if self.compressible else None,
            "nu":   self.nu,
            "rho":  self.rho,
            "mu":   self.mu,
            "cp":   self.cp,
            "Pr":   self.prandtl,
            "boundary_conditions": self.boundary_conditions,
            "energy_bc_values_J_per_kg": self.energy_bc_values if self.energy_bc_values else None,
            "turbulence_initial_values": self.turbulence_initial_values if self.turbulence_initial_values else None,
        }
