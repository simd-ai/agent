"""CaseSpec package — single source of truth for OpenFOAM case generation.

Public surface (importable as ``simd_agent.run.case_spec.<name>``):

  * ``CaseSpec``                   — the resolved plan (dataclass)
  * ``build_case_spec``            — pure-Python builder from validated_config
  * ``FluidThermo``                — typed thermo strategy (Phase 1)
  * ``PressureSolverStrategy``     — typed solvers.p strategy (Phase 1)
  * ``CoarsestLevelCorr``          — sub-block of the above (Phase 1)
  * ``CompressibleBounds``         — rhoMin/Max/pMin/Max/transonic (Phase 1)
  * ``InletTurbulence``            — per-inlet TI + length scale (Phase 1)

Private helpers (still imported by ``solvers/base.py`` and ``run/genai_codegen.py``):

  * ``_select_thermo_profile``     — gas vs cryogenic from (fluid, T, ρ, heat)
  * ``_thermo_profile_from_config`` — same, but from a config dict
  * ``_density_bounds_for_profile`` — (rho_min, rho_max) for SIMPLE/PIMPLE block
  * ``_estimate_inlet_mach``       — Mach scalar for the transonic decision
  * ``_mesh_quality_decisions``    — SIMPLEC / non-ortho correctors / tier

Previously this was a single 1351-line ``case_spec.py`` file.  Phase 1 of
the LLM/validator-boundary redesign split it into per-concern modules so
each one stays well below 400 lines.  Imports of ``simd_agent.run.case_spec``
continue to work because this ``__init__`` re-exports every public name.
"""

from .builders import _safe_float, build_case_spec
from .density import _density_bounds_for_profile, _estimate_inlet_mach
from .mesh_quality import _mesh_quality_decisions, _props_from_registry
from .resolvers import (
    resolve_compressible_bounds,
    resolve_div_phi_h_scheme,
    resolve_fv_options_max,
    resolve_pressure_solver_from_config,
    resolve_pressure_solver_strategy,
    resolve_regime_profile,
    resolve_turbulence_spec,
)
from .spec import _SOLVER_PROPS, _TURB_FIELDS, CaseSpec
from .strategies import (
    CaseRegions,
    CompressibleBounds,
    CoarsestLevelCorr,
    FluidThermo,
    InletTurbulence,
    PressureSolverStrategy,
    RegionSpec,
    TurbulenceRegimeProfile,
    TurbulenceSpec,
)
from .thermo_profile import (
    _CRYO_FLUID_KEYWORDS,
    _select_thermo_profile,
    _thermo_profile_from_config,
)

__all__ = [
    # Public — schemas
    "CaseSpec",
    "build_case_spec",
    "FluidThermo",
    "PressureSolverStrategy",
    "CoarsestLevelCorr",
    "CompressibleBounds",
    "InletTurbulence",
    "RegionSpec",
    "CaseRegions",
    "TurbulenceRegimeProfile",
    "TurbulenceSpec",
    # Public — resolvers (Phase 2)
    "resolve_pressure_solver_strategy",
    "resolve_pressure_solver_from_config",
    "resolve_compressible_bounds",
    "resolve_fv_options_max",
    "resolve_div_phi_h_scheme",
    "resolve_regime_profile",
    "resolve_turbulence_spec",
    # Private (kept for backward compat with existing imports across the codebase)
    "_select_thermo_profile",
    "_thermo_profile_from_config",
    "_density_bounds_for_profile",
    "_estimate_inlet_mach",
    "_mesh_quality_decisions",
    "_props_from_registry",
    "_safe_float",
    "_SOLVER_PROPS",
    "_TURB_FIELDS",
    "_CRYO_FLUID_KEYWORDS",
]
