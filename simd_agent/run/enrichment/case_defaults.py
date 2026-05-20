# simd_agent/run/enrichment/case_defaults.py
"""Resolve canonical case-level user intent from the raw validated config.

Why this step is the foundation of the pipeline
------------------------------------------------
``CFDLinter._build_validated_config`` writes the user's inputs to
several disjoint places — ``validated["fluid"]["temperature"]``,
``validated["inlet"]["velocity"]``, ``validated["outlet"]["pressure"]``,
``validated["boundary_conditions"][<patch>][<field>]["value"]`` — and
each downstream consumer (``CaseSpec``, ``RegionSpec``,
``_multi_region_bcs``, the prompt pack, the LLM filler) used to
re-derive "what was the inlet velocity?" with its own ad-hoc lookup
logic.  When the user typed values that landed in only one of those
slots, half the consumers saw them and half didn't.

This step normalises the answer once: it picks the best signal
available for each canonical field, writes it to
``config["case_defaults"]``, and every downstream consumer is meant
to read from there.  Single-region and multi-region cases both go
through this step — it is solver-agnostic.

Multi-region scope (important!)
-------------------------------
The ``inlet_velocity`` / ``inlet_temperature`` / ``inlet_pressure``
keys are **single-region semantics**.  They are derived by walking
all inlet-shaped patches and taking the first usable value — which
in a multi-region case picks up *one* region's inlet (typically
``innerFluid_inlet``) and ignores the others.

For multi-region (CHT) cases, the authoritative per-region values
live on ``regions.fluid[*].T_init`` / ``U_init`` / ``p_init`` and are
resolved by :mod:`region_inits` from each region's own inlet patch
BC.  ``case_defaults`` is then *only* a last-resort fallback for the
degenerate "one fluid region in a CHT-shaped config" case.

This split is deliberate: every single-region consumer
(``CaseSpec``, ``linting``, single-region prompt packs) continues to
read from ``case_defaults.inlet_*`` exactly as before.  Multi-region
consumers (``value_filler``, the ``_multi_region_bcs`` renderer)
should prefer the per-region values.

Output contract (``config["case_defaults"]``)
---------------------------------------------
::

    {
        # Inlet / ambient signals — single-region authoritative,
        # multi-region last-resort fallback (see scope note above).
        "inlet_velocity":             tuple[float, float, float] | None,  # m/s
        "inlet_temperature":          float | None,                       # K
        "inlet_pressure":             float | None,                       # Pa
        "ambient_pressure":           float | None,                       # Pa
        "bulk_temperature":           float | None,                       # K

        # Bulk fluid properties — drive transportProperties /
        # thermophysicalProperties / Re computation.
        "bulk_density":               float | None,                       # kg/m³
        "bulk_kinematic_viscosity":   float | None,                       # m²/s
        "bulk_dynamic_viscosity":     float | None,                       # Pa·s
        "bulk_prandtl":               float | None,                       # dimensionless

        # Turbulence — drives 0/k, 0/omega, 0/epsilon, 0/nut seed values
        # when the user didn't pre-compute them on the patch.
        "turbulence_intensity":       float | None,                       # fraction (0..1)

        # Wall boundary conditions — keyed by patch name because a case
        # can carry several walls at different temperatures.  Empty dict
        # means no wall T was set anywhere.  CHT-coupled interfaces
        # (``*_to_*``) are deliberately excluded — those flow through
        # the multi-region renderer, not through user-set BCs.
        "wall_temperatures":          dict[str, float],                   # K per patch
    }

``None`` means "no usable signal found anywhere".  Consumers must
treat ``None`` as "fall back to the solver's role default" — never
as a programming error.  Empty containers (``{}``) have the same
meaning for dict-shaped fields.
"""

from __future__ import annotations

import logging
from typing import Any

from simd_agent.run.enrichment.context import EnrichmentContext

logger = logging.getLogger(__name__)

_STEP = "case_defaults"

# A zero-vector velocity is treated as "the user did not actually
# specify anything meaningful" — same convention used elsewhere in
# the pipeline (RegionExtractor, RegionSpec defaults).
_ZERO_VEC: tuple[float, float, float] = (0.0, 0.0, 0.0)


# ────────────────────────────────────────────────────────────────────────────
# Step entry point
# ────────────────────────────────────────────────────────────────────────────


async def apply(ctx: EnrichmentContext) -> None:
    """Populate ``ctx.config["case_defaults"]`` from the validated config.

    Always idempotent — running the step twice produces the same
    result and never raises.  Existing ``case_defaults`` blocks are
    overwritten because the function is a pure projection of the
    rest of the config.
    """
    config = ctx.config
    defaults = {
        # Inlet / ambient signals
        "inlet_velocity":           _resolve_inlet_velocity(config),
        "inlet_temperature":        _resolve_inlet_temperature(config),
        "inlet_pressure":           _resolve_inlet_pressure(config),
        "ambient_pressure":         _resolve_ambient_pressure(config),
        "bulk_temperature":         _resolve_bulk_temperature(config),
        # Bulk fluid properties
        "bulk_density":             _resolve_bulk_density(config),
        "bulk_kinematic_viscosity": _resolve_bulk_kinematic_viscosity(config),
        "bulk_dynamic_viscosity":   _resolve_bulk_dynamic_viscosity(config),
        "bulk_prandtl":             _resolve_bulk_prandtl(config),
        # Turbulence
        "turbulence_intensity":     _resolve_turbulence_intensity(config),
        # Wall conditions (dict — empty when none set)
        "wall_temperatures":        _resolve_wall_temperatures(config),
    }
    config["case_defaults"] = defaults

    logger.info("[ENRICH:%s] %s", _STEP, defaults)
    ctx.add_info(_STEP, code="RESOLVED", message=str(defaults), payload=dict(defaults))


# ────────────────────────────────────────────────────────────────────────────
# Per-field resolvers — each is a pure function of ``config``
# ────────────────────────────────────────────────────────────────────────────


def _resolve_inlet_velocity(config: dict[str, Any]) -> tuple[float, float, float] | None:
    """Pick the best inlet velocity signal, in priority order.

    1. Legacy ``config["inlet"]["velocity"]`` (linter populates this
       from the first inlet BC it sees).
    2. Any ``boundary_conditions[<patch>]["velocity"]["value"]`` with
       a non-zero magnitude.

    Returns ``None`` when nothing usable is found.  A scalar magnitude
    is interpreted as "in +x" — same convention as RegionExtractor /
    RegionSpec.
    """
    legacy = (config.get("inlet") or {}).get("velocity") if isinstance(config.get("inlet"), dict) else None
    vec = _coerce_vec3(legacy)
    if vec is not None and vec != _ZERO_VEC:
        return vec

    for patch_bc in _iter_patch_bcs(config):
        v = (patch_bc.get("velocity") or {}).get("value")
        cand = _coerce_vec3(v)
        if cand is not None and cand != _ZERO_VEC:
            return cand
    return None


def _resolve_inlet_temperature(config: dict[str, Any]) -> float | None:
    """First inlet patch with an explicit temperature value, else None.

    Falls back to :func:`_resolve_bulk_temperature` if no inlet BC
    carries one — without this, multi-region cases that have a bulk
    fluid temperature but no inlet-specific BC would drop the signal.
    """
    for patch_name, patch_bc in _iter_patch_bcs_with_names(config):
        if not _looks_like_inlet(patch_name, patch_bc):
            continue
        v = (patch_bc.get("temperature") or {}).get("value")
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    return _resolve_bulk_temperature(config)


def _resolve_inlet_pressure(config: dict[str, Any]) -> float | None:
    """Pick an inlet pressure signal — patches first, then outlet fallback."""
    for patch_name, patch_bc in _iter_patch_bcs_with_names(config):
        if not _looks_like_inlet(patch_name, patch_bc):
            continue
        v = (patch_bc.get("pressure") or {}).get("value")
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    return _resolve_ambient_pressure(config)


def _resolve_ambient_pressure(config: dict[str, Any]) -> float | None:
    """Atmospheric / outlet pressure used as the field-init seed."""
    legacy = (config.get("outlet") or {}).get("pressure") if isinstance(config.get("outlet"), dict) else None
    if isinstance(legacy, (int, float)) and legacy > 0:
        return float(legacy)
    for patch_bc in _iter_patch_bcs(config):
        v = (patch_bc.get("pressure") or {}).get("value")
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    return None


def _resolve_bulk_temperature(config: dict[str, Any]) -> float | None:
    """Bulk fluid temperature from the Step-1 wizard's ``fluid`` block."""
    return _first_positive_number(config.get("fluid"), ("temperature", "T"))


def _resolve_bulk_density(config: dict[str, Any]) -> float | None:
    """Bulk fluid density (kg/m³) from the Step-1 wizard's ``fluid`` block."""
    return _first_positive_number(config.get("fluid"), ("density", "rho"))


def _resolve_bulk_kinematic_viscosity(config: dict[str, Any]) -> float | None:
    """Kinematic viscosity ν (m²/s).

    Three sources in priority order:

    1. Direct ``fluid.kinematic_viscosity`` / ``fluid.nu``.
    2. Derived from ``μ / ρ`` when both are present — keeps the case
       consistent when the user typed only the dynamic viscosity in the
       UI (rhoSimpleFoam-style input).
    3. ``None`` when neither path produces a positive number.
    """
    direct = _first_positive_number(
        config.get("fluid"), ("kinematic_viscosity", "nu"),
    )
    if direct is not None:
        return direct
    mu = _resolve_bulk_dynamic_viscosity(config)
    rho = _resolve_bulk_density(config)
    if mu is not None and rho is not None and rho > 0:
        return mu / rho
    return None


def _resolve_bulk_dynamic_viscosity(config: dict[str, Any]) -> float | None:
    """Dynamic viscosity μ (Pa·s).

    Mirrors :func:`_resolve_bulk_kinematic_viscosity` — falls back to
    ``ν · ρ`` when only the kinematic viscosity is given.
    """
    direct = _first_positive_number(
        config.get("fluid"), ("dynamic_viscosity", "mu"),
    )
    if direct is not None:
        return direct
    nu = _first_positive_number(config.get("fluid"), ("kinematic_viscosity", "nu"))
    rho = _resolve_bulk_density(config)
    if nu is not None and rho is not None and rho > 0:
        return nu * rho
    return None


def _resolve_bulk_prandtl(config: dict[str, Any]) -> float | None:
    """Prandtl number (dimensionless) from the ``fluid`` block."""
    return _first_positive_number(
        config.get("fluid"), ("prandtl_number", "prandtl", "Pr"),
    )


def _resolve_turbulence_intensity(config: dict[str, Any]) -> float | None:
    """Turbulence intensity (fraction in ``[0, 1]``) from the ``turbulence`` block.

    The Step-1 wizard stores this as a dimensionless fraction (e.g.
    ``0.05`` for 5 %).  ``None`` means the user didn't set one — the
    turbulence model will fall back to its solver-side default.
    """
    return _first_positive_number(config.get("turbulence"), ("intensity",))


def _resolve_wall_temperatures(config: dict[str, Any]) -> dict[str, float]:
    """Map of ``wall_patch_name → fixed temperature (K)``.

    Only "real" walls are included — CHT-coupled interfaces
    (``*_to_*``) are excluded because their temperature is solved for,
    not user-set.  Returns an empty dict when no wall has an explicit
    temperature BC.

    A dict rather than a single value because cases can legitimately
    carry several walls at different temperatures (think hot/cold heat-
    exchanger plates); collapsing to a scalar would lose information.
    """
    out: dict[str, float] = {}
    for name, patch_bc in _iter_patch_bcs_with_names(config):
        if not _looks_like_wall(name, patch_bc):
            continue
        v = (patch_bc.get("temperature") or {}).get("value")
        if isinstance(v, (int, float)) and v > 0:
            out[name] = float(v)
    return out


# ────────────────────────────────────────────────────────────────────────────
# Small helpers
# ────────────────────────────────────────────────────────────────────────────


def _first_positive_number(
    block: Any,
    keys: tuple[str, ...],
) -> float | None:
    """Walk ``keys`` in order and return the first positive numeric value.

    Treats anything non-positive (zero, negative, NaN-prone strings) as
    "not a real signal" so consumers don't have to second-guess
    placeholder zeros.  Returns ``None`` when ``block`` is not a dict
    or no key carries a usable value.
    """
    if not isinstance(block, dict):
        return None
    for key in keys:
        v = block.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return float(v)
    return None


def _iter_patch_bcs(config: dict[str, Any]):
    """Yield each boundary-condition dict (without its patch name)."""
    bcs = config.get("boundary_conditions") or {}
    if not isinstance(bcs, dict):
        return
    for patch_bc in bcs.values():
        if isinstance(patch_bc, dict):
            yield patch_bc


def _iter_patch_bcs_with_names(config: dict[str, Any]):
    """Yield ``(patch_name, patch_bc_dict)`` pairs."""
    bcs = config.get("boundary_conditions") or {}
    if not isinstance(bcs, dict):
        return
    for name, patch_bc in bcs.items():
        if isinstance(patch_bc, dict):
            yield name, patch_bc


def _looks_like_inlet(patch_name: str, patch_bc: dict[str, Any]) -> bool:
    """Cheap heuristic — the renderer + propagator use the same rule."""
    role = (
        patch_bc.get("patch_class")
        or patch_bc.get("patchClass")
        or patch_bc.get("patch_type")
    )
    if isinstance(role, str) and role.lower() == "inlet":
        return True
    return isinstance(patch_name, str) and patch_name.endswith("_inlet")


def _looks_like_wall(patch_name: str, patch_bc: dict[str, Any]) -> bool:
    """True for "real" walls — i.e. user-facing wall BCs.

    Excludes CHT-coupled interfaces (patches whose name contains
    ``_to_``) because those are owned by the multi-region renderer
    and their temperature is a solver output, not a user-set input.
    Explicit ``patch_class == "wall"`` always wins over name-based
    heuristics.
    """
    if isinstance(patch_name, str) and "_to_" in patch_name:
        return False
    role = (
        patch_bc.get("patch_class")
        or patch_bc.get("patchClass")
        or patch_bc.get("patch_type")
    )
    if isinstance(role, str):
        return role.lower() == "wall"
    if not isinstance(patch_name, str):
        return False
    name_lower = patch_name.lower()
    return (
        name_lower == "wall"
        or name_lower.endswith("_wall")
        or name_lower.startswith("wall_")
    )


def _coerce_vec3(value: Any) -> tuple[float, float, float] | None:
    """Coerce a scalar / list / tuple to a 3-component velocity vector."""
    if isinstance(value, (int, float)):
        return (float(value), 0.0, 0.0)
    if (
        isinstance(value, (list, tuple))
        and len(value) >= 3
        and all(isinstance(v, (int, float)) for v in value[:3])
    ):
        return (float(value[0]), float(value[1]), float(value[2]))
    return None
