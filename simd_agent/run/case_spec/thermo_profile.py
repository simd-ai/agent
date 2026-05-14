"""Thermo profile selection — gas vs cryogenic liquid.

Two profiles drive every numerical choice for compressible solvers:

  "gas"        — perfectGas / hePsiThermo. SIMPLEC-friendly, linearUpwind safe,
                 standard relaxation (h≈0.5, U≈0.7). rhoMin/rhoMax act as a
                 loose safety net. Mach can rise to transonic.
  "cryogenic"  — icoPolynomial / heRhoThermo for LN2/LH2/LOX/LHe and any
                 liquid with heat transfer.  h-ρ coupling is stiff; needs
                 conservative relaxation (h=0.05), upwind div schemes, more
                 non-orthogonal correctors, and EOS-ceiling clamps.

Selection is deterministic from the validated config — no LLM involved.
"""

from __future__ import annotations

from typing import Any


_CRYO_FLUID_KEYWORDS: tuple[str, ...] = (
    "ln2", "liquid nitrogen", "nitrogen liquid",
    "lh2", "liquid hydrogen", "hydrogen liquid",
    "lox", "liquid oxygen", "oxygen liquid",
    "lhe", "liquid helium", "helium liquid",
    "lar", "liquid argon", "argon liquid",
    "lng", "methane liquid",
    "cryogenic", "cryogen",
)


def _select_thermo_profile(
    fluid_name: str | None,
    inlet_temperature: float | None,
    rho: float | None,
    has_heat_transfer: bool,
) -> str:
    """Pick the rhoSimpleFoam/rho* numerical profile from physical signature.

    Returns "cryogenic" for any of:
      - fluid name matches a known cryogenic liquid keyword (LN2, LH2, …);
      - inlet temperature is below 200 K (cryogenic regime);
      - liquid with heat transfer (ρ > 200 kg/m³ AND heat transfer active) —
        density of real liquids varies strongly with T, rhoConst is wrong, EOS
        ceiling clamps are needed.

    Returns "gas" otherwise (default — air, N2 vapour, room-temp gas).
    """
    name = (fluid_name or "").strip().lower()
    if name:
        for kw in _CRYO_FLUID_KEYWORDS:
            if kw in name:
                return "cryogenic"
    if inlet_temperature is not None and inlet_temperature < 200.0:
        return "cryogenic"
    if has_heat_transfer and rho is not None and rho > 200.0:
        return "cryogenic"
    return "gas"


def _thermo_profile_from_config(validated_config: dict[str, Any]) -> str:
    """Convenience wrapper: extract the profile inputs from a config dict.

    Used by the deterministic base-class builders (`_build_fv_solution`,
    `_build_fv_schemes`) which receive only the raw config and have no direct
    access to the CaseSpec.
    """
    fluid = validated_config.get("fluid") or {}
    fluid_name = fluid.get("name") if isinstance(fluid, dict) else None
    rho = None
    if isinstance(fluid, dict):
        for k in ("density", "rho"):
            v = fluid.get(k)
            if v is None:
                continue
            try:
                rho = float(v)
                break
            except (TypeError, ValueError):
                pass
    # Inlet temperature — first temperature-bearing BC counts as inlet
    inlet_t: float | None = None
    for pname, pbc in (validated_config.get("boundary_conditions") or {}).items():
        if not isinstance(pbc, dict):
            continue
        t_entry = pbc.get("temperature") or pbc.get("T")
        if isinstance(t_entry, dict):
            t_val = t_entry.get("value") or t_entry.get("uniform")
        else:
            t_val = t_entry
        try:
            inlet_t = float(t_val)
            break
        except (TypeError, ValueError):
            continue
    phys = validated_config.get("physics") or {}
    has_heat = bool(
        validated_config.get("heat_transfer")
        or phys.get("heat_transfer")
        or phys.get("energy")
    )
    return _select_thermo_profile(fluid_name, inlet_t, rho, has_heat)
