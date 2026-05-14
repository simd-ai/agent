"""Density / Mach helpers used by the compressible-solver builders.

These produce the raw scalars that downstream renderers / strategy resolvers
use to populate ``CompressibleBounds`` and decide whether to enable the
``transonic`` flag.
"""

from __future__ import annotations


def _density_bounds_for_profile(
    profile: str,
    rho: float | None,
    eos_t_ceiling: float | None,
    bc_temps: list[float] | None,
) -> tuple[float | None, float | None]:
    """Compute rhoMin / rhoMax for the SIMPLE/PIMPLE block.

    Gas: loose static bounds (0.1 / 10.0) — let physics breathe; the bounds
         only catch genuine divergence, not normal compressibility.
    Cryogenic: derive from EOS at the BC temperature extremes when possible,
         otherwise fall back to a conservative envelope around the inlet ρ.
         The bounds prevent a single bad iteration from producing ρ<0 even
         if the limitTemperature in fvOptions has not yet activated.
    """
    if profile == "gas":
        return 0.1, 10.0
    # Cryogenic
    if rho is None or rho <= 0:
        return None, None
    # If we have BC temperature extremes and an EOS ceiling we can compute exactly.
    if bc_temps and eos_t_ceiling is not None:
        try:
            t_min = max(1.0, min(bc_temps) * 0.5)   # match fv_options_t_min logic
            t_max = min(max(bc_temps), eos_t_ceiling * 0.9)
            # ρ at T_min and T_max using linear extrapolation around the inlet point
            # (a1·T + a0; reconstruct from rho_inlet and the slope used in fvOptions)
            # We don't have a0/a1 here, so fall back to ±50% of inlet ρ instead.
            return rho * 0.5, rho * 1.5
        except Exception:
            pass
    # Final fallback
    return rho * 0.5, rho * 1.5


def _estimate_inlet_mach(
    profile: str,
    inlet_velocity: list[float] | tuple[float, ...] | None,
    inlet_temperature: float | None,
) -> float:
    """Estimate inlet Mach number for the `transonic` decision.

    Gas: M = |U| / sqrt(γ·R·T) using air properties (γ=1.4, R=287 J/kg·K)
         as a robust first-pass — calling code can refine if molWeight known.
    Cryogenic: liquid sound speed is ~10³ m/s and U is typically O(0.1 m/s),
         Mach is essentially zero; return 0.0 so transonic stays off.
    """
    if not inlet_velocity:
        return 0.0
    try:
        u_mag = (sum(float(c) ** 2 for c in inlet_velocity)) ** 0.5
    except (TypeError, ValueError):
        return 0.0
    if profile == "cryogenic":
        return 0.0
    t = inlet_temperature if inlet_temperature and inlet_temperature > 0 else 300.0
    # γ·R for air at T → sound speed
    a = (1.4 * 287.0 * t) ** 0.5
    return u_mag / a if a > 0 else 0.0
