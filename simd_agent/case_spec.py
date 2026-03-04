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

    # ── Computed energy BC values (J/kg) ─────────────────────────────────────
    # Deterministically computed as T * Cp for each temperature-bearing patch.
    # Keys: "inlet", "wall", "internal" — values in J/kg (or None if unknown).
    # Use these in 0/h or 0/e prompts instead of raw temperature values.
    energy_bc_values: dict[str, float | None] = field(default_factory=dict)

    # ── Physical properties (from user config) ────────────────────────────────
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
        phase_names_cfg = ["water", "air"]
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
    # Operating pressure: prefer explicit config value; fall back to inlet fixedValue
    _inlet_p_raw = inlet_bc.get("pressure")
    _inlet_p_val = _bc_value(_inlet_p_raw) if isinstance(_inlet_p_raw, dict) else None
    # Only use inlet pressure value if it looks like an absolute pressure (> 1000 Pa)
    _inlet_p = _inlet_p_val if (_inlet_p_val is not None and isinstance(_inlet_p_val, (int, float)) and _inlet_p_val > 1000) else None
    op_p = _safe_float(
        validated_config.get("operating_pressure")
        or validated_config.get("pressure")
        or _inlet_p
    ) or 101325.0

    nu  = _safe_float(fluid.get("nu") or fluid.get("kinematic_viscosity"))
    rho = _safe_float(fluid.get("rho") or fluid.get("density"))
    mu  = _safe_float(fluid.get("mu") or fluid.get("dynamic_viscosity"))
    cp  = _safe_float(fluid.get("Cp") or fluid.get("cp") or fluid.get("specific_heat"))
    pr  = _safe_float(fluid.get("Pr") or fluid.get("prandtl"))

    # Derive nu from mu/rho if only those are given
    if nu is None and mu is not None and rho is not None and rho > 0:
        nu = mu / rho

    # ── N3-energy: No 0/h or 0/e file is generated ───────────────────────────
    # The thermo package initialises the energy field from 0/T at startup.
    # Providing 0/h causes "Negative initial temperature" because OpenFOAM
    # back-converts h→T using a reference that differs from what the LLM used.
    # energy_field is kept for fvSchemes/fvSolution naming only.
    energy_bc_values: dict[str, float | None] = {}  # intentionally empty — 0/h not generated

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
        # Time
        end_time=end_time,
        delta_t=delta_t,
        # Physical
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
    )

    logger.info(
        f"[CASE_SPEC] solver={solver}  algorithm={algorithm}  "
        f"pressure={pressure_field}  energy={energy_field}  "
        f"turbulence={turb_model}  patches={patch_names}  "
        f"0/fields={[f.split('/')[1] for f in required_0_fields]}  "
        f"const={required_const}"
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
