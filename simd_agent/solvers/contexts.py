"""Typed per-build context shared by all fvSolution / fvSchemes renderers.

The context is computed once per ``_build_fv_*`` call by ``SolverPlugin._fv_context``
and then passed (immutably) into every block-renderer helper.  Before Phase 3
the helpers used ``dict[str, Any]`` for this — two helpers could silently
disagree about which keys existed.  Now every field is explicit, typed, and
frozen.

Conventions:

  * **Literal** types are used everywhere the value space is closed by OpenFOAM
    or by our internal vocabulary, so static checkers catch typos.
  * **Tuple** instead of list — frozen dataclasses don't store mutable defaults
    well and the context is immutable per build anyway.
  * The ``mesh_quality`` field carries the raw ``_mesh_quality_decisions`` dict
    for backward compatibility with code that still expects the legacy
    dict-style access; new code should prefer the named fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from simd_agent.run.case_spec.strategies import TurbulenceRegimeProfile

# ── Closed value sets ──────────────────────────────────────────────────────

MeshTier = Literal["good", "moderate", "poor", "unknown"]
"""Mesh quality tier classification from ``_mesh_quality_decisions``."""

SpeedTier = Literal["low", "moderate", "high"]
"""Inlet-velocity tier — drives div(phi,U) and relaxation choices."""

ThermoProfile = Literal["gas", "cryogenic"]
"""Resolved thermo profile (gas: perfectGas; cryogenic: icoPolynomial)."""

TurbulenceModel = Literal[
    "laminar", "none",
    "kOmegaSST", "kOmega", "kEpsilon",
    "SpalartAllmaras",
    "LES",
]
"""Turbulence model name as it appears in ``constant/turbulenceProperties``."""


# ── FvBuildContext ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class FvBuildContext:
    """Per-build context for the fvSolution / fvSchemes renderer helpers.

    Constructed by ``SolverPlugin._fv_context(config)``.  All renderer helpers
    in ``SolverPlugin`` and the per-solver ``_build_fv_*`` recipes consume
    this object via attribute access — no string indexing.

    Fields are grouped by source:

      * Mesh-driven (from ``_mesh_quality_decisions``):
          ``tier``, ``non_ortho``, ``use_simplec``, ``n_non_ortho``
      * Flow-driven (from ``_extract_velocity_magnitude`` + BCs):
          ``vel_mag``, ``speed_tier``, ``bc_temps``
      * Physics-driven (from ``_thermo_profile_from_config`` + config):
          ``profile``, ``heat_transfer_active``, ``turb_model``
      * Legacy bag (kept for code that still expects a dict):
          ``mesh_quality``
    """

    # Mesh quality
    tier: MeshTier
    non_ortho: float          # degrees
    use_simplec: bool
    n_non_ortho: int

    # Flow
    vel_mag: float            # m/s, maximum inlet velocity magnitude
    speed_tier: SpeedTier
    bc_temps: tuple[float, ...]  # sorted, deduplicated Kelvin temperatures from BCs

    # Physics
    profile: ThermoProfile
    heat_transfer_active: bool
    turb_model: str           # kept as `str` because the registry is open-ended
                              #   (LES variants etc.); use TurbulenceModel for
                              #   match checks in the helpers.

    # Legacy compatibility: the raw mesh-quality dict.  Helpers that still
    # need direct access keep this around; new code uses the named fields.
    mesh_quality: dict[str, Any] = field(default_factory=dict)

    # BC pressures — sorted, deduplicated Pa values across all boundary
    # conditions.  Drives the pressure-ratio decision for ``div(phi,U)``
    # (high inlet/outlet ratio forces ``upwind`` for startup safety).
    # Defaults to empty for legacy test callers that don't set it.
    bc_pressures: tuple[float, ...] = ()

    # True iff any inlet patch uses ``flowRateInletVelocity`` (impulsive
    # mass-flow BC).  These cases require special startup handling:
    #   * a non-zero ``0/U.internalField`` seeded from U_bulk so iteration
    #     1 doesn't have to accelerate the fluid from 0 → U_inlet instantly,
    #   * tighter ``maxCo`` (≤ 0.5) in controlDict,
    #   * more PIMPLE outer correctors,
    #   * ``consistent no`` (regular PIMPLE rather than SIMPLEC).
    has_impulsive_inlets: bool = False

    # Estimated bulk velocity (m/s) — the dominant inlet's
    # ``mdot / (ρ_estimate · A_estimate)`` for mass-flow inlets, or the
    # explicit velocity magnitude for ``fixedValue`` inlets.  Used to
    # seed ``0/U.internalField`` and to size the initial ``deltaT``.
    bulk_velocity: float = 0.0

    # Resolved per-regime scheme bundle — laminar / RAS / LES knobs for
    # fvSchemes (ddt, div(phi,*)) and constant/turbulenceProperties.
    # See ``simd_agent.run.case_spec.resolvers.resolve_regime_profile``.
    # Defaults to None so legacy test callers don't have to build a profile;
    # renderers fall back to the previous RAS-only literals when missing.
    regime_profile: "TurbulenceRegimeProfile | None" = None

    # ── Convenience computed views ──────────────────────────────────────

    @property
    def delta_t_bc(self) -> float:
        """Spread between the max and min BC temperature (0 if < 2 BCs)."""
        if len(self.bc_temps) < 2:
            return 0.0
        return max(self.bc_temps) - min(self.bc_temps)

    @property
    def is_laminar(self) -> bool:
        return self.turb_model in ("laminar", "none", "")

    @property
    def pressure_ratio(self) -> float:
        """max_BC_p / min_BC_p across all BCs (1.0 if < 2 pressures known).

        A ratio ≥ 3 — common for compressor inlets, throttle outflows, or any
        case with a high-pressure inlet against an atmospheric outlet — makes
        ``linearUpwindV`` for ``div(phi,U)`` numerically unsafe at startup
        (the gradient correction overshoots into acoustic noise the
        compressible loop cannot absorb).  Renderers consult this to choose
        the safer ``upwind`` scheme.
        """
        if len(self.bc_pressures) < 2:
            return 1.0
        lo, hi = min(self.bc_pressures), max(self.bc_pressures)
        if lo <= 0:
            return 1.0
        return hi / lo
