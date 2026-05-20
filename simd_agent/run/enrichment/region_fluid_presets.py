# simd_agent/run/enrichment/region_fluid_presets.py
"""Per-region fluid preset inference from temperature + density signature.

Why this matters
----------------
The region auto-detector in :mod:`simd_agent.run.multi_region.region_detection`
classifies fluid regions by name only — ``innerFluid`` / ``outerFluid``
default to ``"air"`` because the substring matchers fall through.  That's
wrong for any CHT case where the case-wide fluid is a known cryogen
(LN2, LH2, LOX, LHe, LNG) — both fluid regions end up with air's
``Cp = 1006 J/kg·K`` even though the user's prompt and bulk density
clearly say LN2.  The downstream thermo (``hConstThermo<perfectGas>``)
then handles a 77 K inlet temperature with air's heat capacity and the
energy equation crashes with ``Negative initial temperature T0``.

This step closes the gap: after :mod:`region_inits` populates each
region's ``T_init``, we look at:

  * the case-wide fluid name (e.g. ``"Liquid Nitrogen (LN2)"``);
  * the case-wide bulk density (LN2 ≈ 808, water ≈ 998, air ≈ 1.2);
  * each region's ``T_init``;

…and replace the heuristic ``"air"`` with the matching preset from
:attr:`MultiRegionBase.FLUID_REGION_PRESETS` (``"ln2"``, ``"water"``,
…).  Regions whose preset was already set by the RegionExtractor LLM
or the upstream UI are never overwritten.

The step is deterministic — no LLM call.  Safe to run on every config;
single-region cases short-circuit on the ``no fluid_regions`` guard.
"""

from __future__ import annotations

import logging
from typing import Any

from simd_agent.run.case_spec.thermo_profile import _CRYO_FLUID_KEYWORDS
from simd_agent.run.enrichment.context import EnrichmentContext

logger = logging.getLogger(__name__)

_STEP = "region_fluid_presets"

# Maps a cryogenic-fluid keyword (as found in the user's free-text
# ``config["fluid"]["name"]``) to the matching preset key in
# :attr:`MultiRegionBase.FLUID_REGION_PRESETS`.  Order matters — the
# first keyword that hits wins, so the longer/more-specific forms come
# first ("liquid nitrogen" before "nitrogen").
_CRYO_KEYWORD_TO_PRESET: tuple[tuple[str, str], ...] = (
    ("liquid nitrogen",  "ln2"),
    ("nitrogen liquid",  "ln2"),
    ("ln2",              "ln2"),
    ("liquid hydrogen",  "lh2"),
    ("hydrogen liquid",  "lh2"),
    ("lh2",              "lh2"),
    ("liquid oxygen",    "lox"),
    ("oxygen liquid",    "lox"),
    ("lox",              "lox"),
    ("liquid helium",    "lhe"),
    ("helium liquid",    "lhe"),
    ("lhe",              "lhe"),
    ("methane liquid",   "lng"),
    ("lng",              "lng"),
    ("liquid argon",     "lar"),  # registered preset may not exist; safe to skip below
    ("argon liquid",     "lar"),
    ("lar",              "lar"),
)

# Nominal densities and characteristic temperatures of each registered
# fluid preset.  Used to fall back when the user's ``fluid.name`` is
# generic ("cryogenic liquid", or empty) but the bulk density alone is
# diagnostic.  Tolerances are wide enough that user-rounded values land
# in the right bucket.
_PRESET_SIGNATURES: tuple[tuple[str, float, float], ...] = (
    # (preset, rho_nominal, T_typical)  — T_typical is the boiling
    # point at 1 atm for cryogens, room temperature for room-temp
    # fluids.
    ("lhe",   125.0,   4.2),
    ("lh2",    70.85, 20.0),
    ("ln2",   808.0,  77.0),
    ("lox",  1141.0,  90.0),
    ("lng",   422.8, 111.0),
    ("water", 998.21, 290.0),
    ("oil",   880.0,  290.0),
    ("air",     1.2,  293.0),
)

# Tolerances for the density / temperature match.  Density: 20 %
# (covers user-rounded inputs and small T-dependence).  Temperature:
# 30 K (the boiling-point ± wall warming envelope a cryogen sees in a
# CHT case).
_RHO_REL_TOLERANCE = 0.20
_T_ABS_TOLERANCE   = 30.0


# ────────────────────────────────────────────────────────────────────────────
# Step entry point
# ────────────────────────────────────────────────────────────────────────────


async def apply(ctx: EnrichmentContext) -> None:
    config = ctx.config
    fluid_regions = (config.get("regions") or {}).get("fluid") or []
    if not fluid_regions:
        return  # single-region or no-region case

    fluid_block = config.get("fluid") or {}
    fluid_name = (fluid_block.get("name") if isinstance(fluid_block, dict) else "") or ""
    defaults = config.get("case_defaults") or {}
    bulk_rho = _as_float(defaults.get("bulk_density"))
    case_inlet_T = _as_float(defaults.get("inlet_temperature"))

    # Identify the "primary" preset implied by the case-wide fluid name.
    # If the user wrote "Liquid Nitrogen (LN2)", this returns "ln2".
    # Otherwise None — fall back to density / temperature inference.
    case_cryo_preset = _preset_from_fluid_name(fluid_name)

    # ── Pass 1: per-region inference ─────────────────────────────
    changed: list[dict[str, Any]] = []
    for region in fluid_regions:
        old_preset = region.get("fluid_preset")
        # Only override when the auto-detector left the default 'air'
        # and the user/extractor didn't explicitly set something.  Any
        # non-air preset is treated as authoritative.
        if old_preset and old_preset != "air":
            continue

        new_preset = _infer_region_preset(
            T_init=_as_float(region.get("T_init")),
            U_init=region.get("U_init"),
            case_cryo_preset=case_cryo_preset,
            case_bulk_rho=bulk_rho,
            case_inlet_T=case_inlet_T,
        )
        if new_preset is None or new_preset == old_preset:
            continue
        region["fluid_preset"] = new_preset
        changed.append({
            "region":  region.get("name"),
            "from":    old_preset,
            "to":      new_preset,
            "T_init":  region.get("T_init"),
            "rule":    "per-region",
        })

    # ── Pass 2: regasifier-topology rule ─────────────────────────
    # When exactly one fluid region is cryogenic AND at least one other
    # fluid region is room-T (250-350 K) still defaulting to "air",
    # flip the room-T region to "water".  A regasifier or LN2-warmed
    # heat exchanger is the canonical CHT case here, and the warm side
    # is almost always a liquid (water / glycol / process fluid) — air
    # is physically possible but a) carries ~1000× less enthalpy per
    # unit volume so the visualization looks lopsided, and b) is rarely
    # what the user actually means in a CHT regasifier setup.
    cryo_count = sum(
        1 for r in fluid_regions
        if _is_cryogenic_preset(r.get("fluid_preset"))
    )
    if cryo_count == 1:
        for region in fluid_regions:
            if region.get("fluid_preset") != "air":
                continue
            T = _as_float(region.get("T_init"))
            if T is None or not (250.0 <= T <= 350.0):
                continue
            region["fluid_preset"] = "water"
            changed.append({
                "region":  region.get("name"),
                "from":    "air",
                "to":      "water",
                "T_init":  region.get("T_init"),
                "rule":    "regasifier-topology",
            })

    if changed:
        logger.info("[ENRICH:%s] %s", _STEP, changed)
        ctx.add_info(
            _STEP,
            code="INFERRED",
            message=f"Inferred fluid preset for {len(changed)} region(s)",
            payload={"changes": changed},
        )


# Cryogenic-preset set — used by the regasifier topology pass to decide
# which configurations qualify.
_CRYOGENIC_PRESETS: frozenset[str] = frozenset({"ln2", "lh2", "lox", "lhe", "lng", "lar"})


def _is_cryogenic_preset(preset: Any) -> bool:
    return isinstance(preset, str) and preset in _CRYOGENIC_PRESETS


# ────────────────────────────────────────────────────────────────────────────
# Inference helpers
# ────────────────────────────────────────────────────────────────────────────


def _infer_region_preset(
    *,
    T_init: float | None,
    U_init: Any,
    case_cryo_preset: str | None,
    case_bulk_rho: float | None,
    case_inlet_T: float | None,
) -> str | None:
    """Pick a preset for one region; return None to leave as-is.

    Priority order (strongest first):

      1. **The case's primary fluid matches this region's T_init.** The
         user said the case is LN2 and this region's inlet is 77 K → it
         IS the LN2 region.  Use the cryogenic preset.

      2. **Cryogenic regime by temperature alone.** Region T_init below
         200 K → must be a cryogen; pick by density bucket (which
         distinguishes LN2/LOX/LH2/LHe).

      3. **Room-temperature liquid by density.** ~290 K AND bulk
         density looks like water (≈ 998) or oil (≈ 880) → that
         preset.  Air's density (1.2) would put it back at air.

      4. ``None`` — no confident inference, keep heuristic default.
    """
    # 1) Case-primary match.  When the case-level fluid name maps to a
    # cryogen AND this region's T_init is near that cryogen's b.p.,
    # it IS the case's primary fluid region.
    if case_cryo_preset is not None and T_init is not None:
        for preset, _rho_nom, T_typ in _PRESET_SIGNATURES:
            if preset == case_cryo_preset and abs(T_init - T_typ) <= _T_ABS_TOLERANCE:
                return preset

    # 2) Cryogenic regime by T alone (case fluid name absent or not
    # cryogenic, but this region sits at cryogen temperatures).
    if T_init is not None and T_init < 200.0:
        # Find the cryogenic preset whose b.p. is closest to T_init.
        best: tuple[str, float] | None = None
        for preset, _rho_nom, T_typ in _PRESET_SIGNATURES:
            if T_typ > 200.0:
                continue  # skip non-cryogens
            delta = abs(T_init - T_typ)
            if delta <= _T_ABS_TOLERANCE and (best is None or delta < best[1]):
                best = (preset, delta)
        if best is not None:
            return best[0]

    # 3) Room-temperature liquid identification by density — only safe
    # when the case-wide bulk density actually describes THIS region.
    # If the case fluid name is a cryogen, the bulk_density is the
    # cryogen's (e.g. 808 for LN2) and reusing it for the non-cryo
    # region would mis-identify it (808 is closer to oil's 880 than to
    # water's 998 → wrong).  Skip the density heuristic in that case.
    if T_init is not None and 250.0 <= T_init <= 350.0 and case_cryo_preset is None:
        rho_signal = case_bulk_rho
        if rho_signal is not None and rho_signal > 100.0:
            # Definitely not a gas at room T.  Pick water vs oil by
            # density proximity.
            return "water" if abs(rho_signal - 998.21) <= abs(rho_signal - 880.0) else "oil"

    return None  # leave the heuristic default alone


def _preset_from_fluid_name(fluid_name: str) -> str | None:
    """Map a free-text fluid name to a preset key, or None if not a known cryogen."""
    name = (fluid_name or "").strip().lower()
    if not name:
        return None
    # Quick reject for non-cryogenic mentions so we don't accidentally
    # match "nitrogen vapour" → ln2.
    cryo_hit = any(kw in name for kw in _CRYO_FLUID_KEYWORDS)
    if not cryo_hit:
        return None
    for keyword, preset in _CRYO_KEYWORD_TO_PRESET:
        if keyword in name:
            return preset
    return None


def _as_float(value: Any) -> float | None:
    """Coerce optional numeric inputs without raising."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
