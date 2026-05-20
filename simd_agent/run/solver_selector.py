# simd_agent/solver_selector.py
"""OpenFOAM solver selection via LLM with heuristic fallback.

The LLM makes the full solver decision from the user's natural-language description
and physics flags.  Benefits over the old deterministic approach:

  • Understands intent ("high-speed airflow", "Mach 0.5", "density varies") rather
    than requiring exact config key matches.
  • Detects phase-change cases (boiling, cavitation, evaporation, condensation) and
    refuses to silently map them to a plain single-phase solver.
  • Returns structured JSON with reason + confidence + warnings so the orchestrator
    can surface ambiguity or user-visible guidance.

A heuristic fallback (deterministic rules) is applied if the LLM call fails or
returns an invalid/unsupported solver name.
"""

import asyncio
import json
import logging
import re
from typing import Any

from simd_agent.llm import get_provider

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Canonical allowed-solver set (single source of truth)
# ─────────────────────────────────────────────────────────────

# ── Supported single-phase solvers ───────────────────────────────────────────
# Multiphase (interFoam, compressibleInterFoam, etc.) is reserved for a later phase.

# The following sets are derived from the live plugin registry so a new
# solver dropped under ``simd_agent/solvers/`` becomes visible everywhere
# without editing this file.  The static fallback (used only when the
# registry can't be imported, e.g. tooling that pulls this module in
# isolation) covers the original six core solvers.
def _registry_or_fallback(getter, fallback: set[str]) -> set[str]:
    try:
        from simd_agent.solvers import get_registry
        return set(getter(get_registry()))
    except Exception:
        return fallback


P_SOLVERS: set[str] = _registry_or_fallback(
    lambda r: r.p_solvers(),
    {"simpleFoam", "pimpleFoam", "rhoSimpleFoam", "rhoPimpleFoam"},
)

P_RGH_SOLVERS: set[str] = _registry_or_fallback(
    lambda r: r.p_rgh_solvers(),
    {"buoyantSimpleFoam", "buoyantPimpleFoam"},
)

ENERGY_SOLVERS: set[str] = _registry_or_fallback(
    lambda r: r.energy_solvers(),
    {"rhoSimpleFoam", "rhoPimpleFoam", "buoyantSimpleFoam", "buoyantPimpleFoam"},
)

GRAVITY_SOLVERS: set[str] = _registry_or_fallback(
    lambda r: r.gravity_solvers(),
    P_RGH_SOLVERS,
)

THERMO_SOLVERS: set[str] = ENERGY_SOLVERS

ALLOWED_SOLVERS: set[str] = _registry_or_fallback(
    lambda r: r.allowed_solvers(),
    P_SOLVERS | P_RGH_SOLVERS,
)


def _assert_sets_match_registry() -> None:
    """Fail loudly if the hardcoded solver sets drift from the plugin registry.

    The registry is the single source of truth — these sets are kept for
    backward compatibility with code that still imports them.  Any
    divergence means a plugin was added/removed without updating the sets
    (or vice-versa).  Runs once at import time.
    """
    try:
        from simd_agent.solvers import get_registry
    except Exception:
        return  # registry unavailable — skip check
    r = get_registry()
    drift: list[str] = []
    if P_SOLVERS != r.p_solvers():
        drift.append(f"P_SOLVERS {P_SOLVERS} != registry.p_solvers() {r.p_solvers()}")
    if P_RGH_SOLVERS != r.p_rgh_solvers():
        drift.append(f"P_RGH_SOLVERS {P_RGH_SOLVERS} != registry.p_rgh_solvers() {r.p_rgh_solvers()}")
    if ENERGY_SOLVERS != r.energy_solvers():
        drift.append(f"ENERGY_SOLVERS {ENERGY_SOLVERS} != registry.energy_solvers() {r.energy_solvers()}")
    if GRAVITY_SOLVERS != r.gravity_solvers():
        drift.append(f"GRAVITY_SOLVERS {GRAVITY_SOLVERS} != registry.gravity_solvers() {r.gravity_solvers()}")
    if ALLOWED_SOLVERS != r.allowed_solvers():
        drift.append(f"ALLOWED_SOLVERS {ALLOWED_SOLVERS} != registry.allowed_solvers() {r.allowed_solvers()}")
    if drift:
        import logging as _lg
        _lg.getLogger(__name__).error(
            "[SOLVER_SELECTOR] Hardcoded set drift vs registry:\n  " + "\n  ".join(drift)
        )


_assert_sets_match_registry()


# ─────────────────────────────────────────────────────────────
# Physics flag extraction helper (used by heuristic fallback)
# ─────────────────────────────────────────────────────────────

def _extract_flags(validated_config: dict[str, Any]) -> dict[str, Any]:
    """Normalise physics flags for the heuristic fallback."""
    phys = validated_config.get("physics", {}) or {}

    def _get(*keys, default=None):
        for k in keys:
            v = validated_config.get(k)
            if v is not None:
                return v
            v = phys.get(k)
            if v is not None:
                return v
        return default

    heat         = bool(_get("heat_transfer", "enable_heat_transfer", default=False))
    multiphase   = bool(_get("multiphase", default=False))
    phases       = _get("phases", default=[]) or []
    compressible = (_get("compressibility", default="incompressible") or "incompressible") == "compressible"
    transient    = (_get("time_stepping", "time_scheme", default="steady") or "steady") in (
        "transient", "unsteady"
    )
    laminar      = (_get("flow_regime", default="turbulent") or "turbulent") == "laminar"
    # Gravity / buoyancy is a separate signal from heat transfer.  Forced
    # convection (heat + no gravity) is a real case that must NOT collapse
    # into a buoyant solver.
    gravity      = bool(_get("gravity", "buoyancy", default=False))

    # Heat-transfer detection beyond the explicit flag — temperature BCs
    # with a meaningful spread imply heat transfer even if the flag is
    # missing.  Without this, prompts like "hot air at 500K with a 600K
    # wall, no gravity" silently end up at simpleFoam.
    if not heat:
        bc_temps: list[float] = []
        for pbc in (validated_config.get("boundary_conditions") or {}).values():
            if not isinstance(pbc, dict):
                continue
            t_entry = pbc.get("temperature") or pbc.get("T")
            t_val = (
                t_entry.get("value") or t_entry.get("uniform")
                if isinstance(t_entry, dict)
                else t_entry
            )
            try:
                bc_temps.append(float(t_val))
            except (TypeError, ValueError):
                pass
        if len(bc_temps) >= 2 and (max(bc_temps) - min(bc_temps)) > 5.0:
            heat = True

    return {
        "heat":         heat,
        "multiphase":   multiphase,
        "phases":       phases,
        "n_phases":     len(phases) if phases else (2 if multiphase else 1),
        "compressible": compressible,
        "transient":    transient,
        "laminar":      laminar,
        "gravity":      gravity,
    }


# ─────────────────────────────────────────────────────────────
# Heuristic fallback (deterministic, no LLM)
# ─────────────────────────────────────────────────────────────

_CRYOGENIC_BOILING_POINTS: dict[str, float] = {
    "ln2": 77.4, "liquid nitrogen": 77.4, "nitrogen": 77.4,
    "lh2": 20.3, "liquid hydrogen": 20.3, "hydrogen": 20.3,
    "lox": 90.2, "liquid oxygen": 90.2, "oxygen": 90.2,
    "lng": 111.7, "liquid natural": 111.7, "methane": 111.7,
    "lar": 87.3, "liquid argon": 87.3, "argon": 87.3,
    "lhe": 4.2, "liquid helium": 4.2, "helium": 4.2,
}


def _is_cryogenic_liquid(validated_config: dict[str, Any]) -> bool:
    """Return True if the fluid is a known cryogenic liquid (boiling point < 130 K)."""
    fluid = validated_config.get("fluid") or {}
    fluid_name = (fluid.get("name") or "").lower()
    for key in _CRYOGENIC_BOILING_POINTS:
        if key in fluid_name:
            return True
    # Also detect by inlet temperature: fluid delivered below 130 K is cryogenic
    bcs = validated_config.get("boundary_conditions") or {}
    inlet_bc = bcs.get("inlet") or {}
    t_inlet = inlet_bc.get("temperature")
    t_val = t_inlet.get("value") if isinstance(t_inlet, dict) else t_inlet
    try:
        if t_val is not None and float(t_val) < 130.0:
            return True
    except (TypeError, ValueError):
        pass
    return False


def _is_cryogenic_boiling(validated_config: dict[str, Any]) -> bool:
    """Return True if config describes a cryogenic liquid heated above its boiling point."""
    fluid = validated_config.get("fluid") or {}
    fluid_name = (fluid.get("name") or "").lower()
    boiling_k: float | None = None
    for key, bp in _CRYOGENIC_BOILING_POINTS.items():
        if key in fluid_name:
            boiling_k = bp
            break
    if boiling_k is None:
        return False

    bcs = validated_config.get("boundary_conditions") or {}
    bc_temps: list[float] = []
    for bc in bcs.values():
        if not isinstance(bc, dict):
            continue
        t_bc = bc.get("temperature")
        t_val = t_bc.get("value") if isinstance(t_bc, dict) else None
        if t_val is not None:
            try:
                bc_temps.append(float(t_val))
            except (TypeError, ValueError):
                pass
    return bool(bc_temps) and max(bc_temps) > boiling_k + 10


def _heuristic_fallback(
    validated_config: dict[str, Any],
    user_requirements: str = "",
) -> str:
    """Pure-logic fallback when the LLM response is unparseable or invalid.

    Decision logic, in order:

      0. Honor explicit solver names in the user prompt (highest priority —
         user knows what they want; only override on direct contradiction).
      1. Cryogenic liquid (name keyword or T_inlet < 130 K) → rho*  (density-T coupling)
      2. Heat transfer active:
           - with gravity / buoyancy → buoyant*  (natural convection)
           - without gravity         → rho*      (forced convection — simpleFoam
             cannot solve the energy equation and assumes ρ=const, which is
             physically wrong with any temperature gradient)
      3. No heat but explicitly compressible (Mach > 0.3, etc.) → rho*
      4. Otherwise → simple* / pimple* (incompressible isothermal)
    """
    f = _extract_flags(validated_config)
    prompt_squashed = "".join((user_requirements or "").lower().split())

    # 0. Explicit solver mention in the prompt — most specific first.
    _USER_SOLVER_MAP = (
        ("buoyantboussinesqsimplefoam", "buoyantBoussinesqSimpleFoam"),
        ("buoyantboussinesqpimplefoam", "buoyantBoussinesqPimpleFoam"),
        ("chtmultiregionsimplefoam",   "chtMultiRegionSimpleFoam"),
        ("chtmultiregionfoam",         "chtMultiRegionFoam"),
        ("buoyantsimplefoam",          "buoyantSimpleFoam"),
        ("buoyantpimplefoam",          "buoyantPimpleFoam"),
        ("rhosimplefoam",              "rhoSimpleFoam"),
        ("rhopimplefoam",              "rhoPimpleFoam"),
        ("simplefoam",                 "simpleFoam"),
        ("pimplefoam",                 "pimpleFoam"),
    )
    for token, canonical in _USER_SOLVER_MAP:
        if token in prompt_squashed and canonical in ALLOWED_SOLVERS:
            user_is_steady = canonical.endswith("SimpleFoam")
            user_is_transient = (
                canonical.endswith("PimpleFoam")
                or canonical == "chtMultiRegionFoam"
            )
            if (user_is_steady and not f["transient"]) or (user_is_transient and f["transient"]):
                logger.info(
                    f"[SOLVER_SELECT] Heuristic: user named '{canonical}' — honoring"
                )
                return canonical
            logger.warning(
                f"[SOLVER_SELECT] User named '{canonical}' but it contradicts "
                f"the time scheme (transient={f['transient']}) — ignoring"
            )
            break

    # 1. Cryogenic single-phase liquid (LH2, LN2, LOX, LHe, etc.) → rho*
    if _is_cryogenic_liquid(validated_config):
        logger.warning(
            "[SOLVER_SELECT] Heuristic: cryogenic liquid → rho* solver "
            "(density-temperature coupling)"
        )
        return "rhoPimpleFoam" if f["transient"] else "rhoSimpleFoam"

    # 2. Heat transfer active → buoyant (with gravity) or rho* (forced convection)
    #    "Gravity" is detected both from the explicit flag AND from natural-
    #    language mentions in the prompt ("gravity acts downward at 9.81…",
    #    "natural convection", "buoyancy-driven", …) — the form's gravity
    #    checkbox is often left unticked even when the prompt is explicit.
    prompt_gravity_keywords = (
        "gravity", "gravitational", "buoyancy", "buoyant",
        "natural convection", "free convection",
    )
    prompt_no_gravity = any(p in (user_requirements or "").lower() for p in (
        "no gravity", "without gravity", "ignore gravity",
        "no buoyancy", "without buoyancy",
        "forced convection",
    ))
    has_gravity = f["gravity"] or (
        not prompt_no_gravity
        and any(kw in (user_requirements or "").lower() for kw in prompt_gravity_keywords)
    )

    if f["heat"]:
        if has_gravity:
            logger.info(
                "[SOLVER_SELECT] Heuristic: heat transfer + gravity → buoyant solver"
            )
            return "buoyantPimpleFoam" if f["transient"] else "buoyantSimpleFoam"
        logger.info(
            "[SOLVER_SELECT] Heuristic: heat transfer + no gravity → rho* solver "
            "(forced convection — simpleFoam has no energy equation)"
        )
        return "rhoPimpleFoam" if f["transient"] else "rhoSimpleFoam"

    # 3. Compressible without heat (high-speed aero, shocks)
    if f["compressible"]:
        return "rhoPimpleFoam" if f["transient"] else "rhoSimpleFoam"

    # 4. Incompressible isothermal
    if not f["transient"]:
        return "simpleFoam"
    return "pimpleFoam"


# ─────────────────────────────────────────────────────────────
# LLM system prompt — full solver decision
# ─────────────────────────────────────────────────────────────

_LLM_SOLVER_SYSTEM_PROMPT = """\
You are an expert OpenFOAM solver engineer. Your ONLY job is to select the
single most appropriate OpenFOAM solver from the allowed list below.

══════════════════════════════════════════════════════════════
ALLOWED SOLVERS — return EXACTLY one of these names (or null for unsupported cases):

  # No heat transfer, incompressible
  simpleFoam                       — steady-state, standard industrial flows (pipes, ducts, external aero)
  pimpleFoam                       — transient, vortex shedding, moving mesh, pulsating flow

  # Compressible with heat (high-speed, large ΔT, or cryogenic — needs full EOS)
  rhoSimpleFoam                    — compressible steady-state, high-speed aerodynamics, cryogenic
  rhoPimpleFoam                    — compressible transient, pressure waves, acoustics, cryogenic

  # Buoyancy-driven, full compressible (gravity + significant β·ΔT — fire, smoke, large rooms)
  buoyantSimpleFoam                — buoyancy steady-state, heated rooms, HVAC, electronic cooling
  buoyantPimpleFoam                — buoyancy transient, smoke spread, fire, ventilation transients

  # Buoyancy-driven, BOUSSINESQ approximation (small β·ΔT ≲ 10–20 %, faster than full compressible)
  buoyantBoussinesqSimpleFoam      — Boussinesq steady, mild heating, natural convection benchmarks
  buoyantBoussinesqPimpleFoam      — Boussinesq transient, plume oscillation, weak unsteadiness

  # Conjugate heat transfer — fluid AND solid regions, heat passes through solid walls
  chtMultiRegionSimpleFoam         — CHT steady, heat exchangers, electronic packages with metal walls
  chtMultiRegionFoam               — CHT transient, thermal start-up, time-dependent heat transfer

NOTE: Multiphase (interFoam, compressibleInterFoam, etc.) is NOT currently supported.
If the case requires two or more phases, set solver to null and warn the user.

When the user explicitly names "Boussinesq" / "incompressible Boussinesq"
or mentions β (volumetric expansion coefficient) + T_ref, prefer the
buoyantBoussinesq* variant over the full buoyant* variant.

When the user describes multiple regions (a solid wall between two
fluids, a pipe-in-pipe with metal in between, a heat exchanger with
a wall region), pick chtMultiRegion{Simple,}Foam.
══════════════════════════════════════════════════════════════

──────────────────────────────────────────────────────────────
DECISION RULES — apply strictly in this order
──────────────────────────────────────────────────────────────

## 1. Unsupported cases — set solver to null
Any of the following → set solver to null with a clear warning:
  - Multiphase / two-phase flows (water + air, oil + water, VOF interface)
  - Phase change: boiling, cavitation, evaporation, condensation, flashing
  - Moving fluid-front / filling / air-displacement scenarios
  - Cryogenic liquid heated above its boiling point (phase change)

══════════════════════════════════════════════════════════════
HEAT TRANSFER IS THE PRIMARY DISCRIMINATOR
──────────────────────────────────────────────────────────────
"Heat transfer active" means ANY of:
  • The user mentions temperature, hot, cold, heated, cooled, warm, chill,
    thermal, heat flux, isothermal wall at T≠ambient
  • Two patches in the BCs carry different temperatures (e.g. inlet at
    300 K and wall at 500 K) — even if no thermal language is used
  • A cryogenic fluid is named (LN2, LH2, LOX, LHe, …)

When heat transfer is active, simpleFoam / pimpleFoam are ELIMINATED.
They solve only momentum + continuity (ρ=const, no energy equation).
Using them with a temperature gradient gives a result that ignores the
thermal physics entirely.  Pick a compressible-energy solver instead.
══════════════════════════════════════════════════════════════

## 2. Heat transfer + gravity / buoyancy → buoyant solvers
Apply when ALL of:
  - Heat transfer is active (see above)
  - Single-phase only
  - Gravity matters: user mentions gravity, buoyancy, natural convection,
    heated room, HVAC, chimney, solar collector, electronic cooling,
    smoke spread, density stratification — i.e. the flow is driven (in
    part or whole) by Δρ × g

  Steady-state  → buoyantSimpleFoam
  Transient     → buoyantPimpleFoam

## 3. Heat transfer + NO gravity (forced convection) → rho* solvers
Apply when ALL of:
  - Heat transfer is active
  - Single-phase only
  - Gravity is OFF (user explicitly says "no gravity" / "without gravity"
    / "horizontal pipe / duct" / "ignore buoyancy") OR the flow is clearly
    pressure / pump driven (mass flow rate, velocity BC, no buoyancy
    language)
  - This includes ALL forced-convection heat exchangers, heated pipes,
    cooling channels, hot-wall ducts, etc.

  Steady-state  → rhoSimpleFoam
  Transient     → rhoPimpleFoam

  RATIONALE: simpleFoam has no energy equation and assumes ρ = const.  Air
  at 280 K vs 600 K is ρ ≈ 1.26 vs 0.59 kg/m³ — a 53 % density variation
  that simpleFoam cannot represent.  rhoSimpleFoam is the correct choice
  whenever there is any temperature gradient and no buoyancy.

## 4. No heat transfer, high-speed compressible → rho* solvers
Apply when:
  - Heat transfer is NOT active (isothermal)
  - Mach > 0.3, supersonic, transonic, shock, or pressure differential
    > ~10 % of absolute pressure
  - OR cryogenic liquid (always compressible regardless of speed)

  Steady-state  → rhoSimpleFoam
  Transient     → rhoPimpleFoam

## 5. No heat, low Mach, no buoyancy → incompressible
Apply when ALL of:
  - Heat transfer is NOT active
  - Mach < 0.1, no shocks
  - Liquid or gas at moderate, near-uniform conditions

  Steady-state  → simpleFoam
  Transient     → pimpleFoam

## 6. Steady vs transient
Steady: "steady", "RANS", "converge", "time-averaged", "mean flow"
Transient: "transient", "unsteady", "time-varying", "oscillating", "pulsating",
  "start-up", "smoke spread", "moving parts", "ventilation transient"
When ambiguous: default to steady.

## 7. Honor explicit user solver requests (UNLESS they contradict the physics)
If the user explicitly names an OpenFOAM solver in the User Requirements
section ("use rhoPimpleFoam", "with buoyantBoussinesqSimpleFoam", "run
simpleFoam", …), HONOR that choice — return exactly that solver name —
UNLESS it contradicts another statement they made in the same prompt:

  • User says "rhoSimpleFoam" but ALSO says "transient" → CONTRADICTION.
    Pick `rhoPimpleFoam` and add a warning explaining the swap.
  • User says "pimpleFoam" but ALSO says "steady-state" → CONTRADICTION.
    Pick `simpleFoam` and warn.
  • User says "simpleFoam" but ALSO mentions hot/cold walls / heat
    transfer / temperature gradients → CONTRADICTION (simpleFoam has no
    energy equation).  Pick the appropriate compressible-energy or
    buoyant solver and warn.
  • User says "buoyantSimpleFoam" but ALSO says "no gravity / forced
    convection" → CONTRADICTION.  Pick `rhoSimpleFoam` and warn.

Otherwise (no contradiction), the user's explicit name wins over the
deterministic rules above.  Use `confidence: "high"` and put "Honoring
user-requested solver" in `reason`.

──────────────────────────────────────────────────────────────
OUTPUT — call the report_selected_solver tool
──────────────────────────────────────────────────────────────
You MUST emit your decision by calling the ``report_selected_solver``
tool exactly once.  Do NOT respond with free text.  Tool parameters:

  • solver (enum, nullable): the canonical solver name, or null only
    when no supported single-phase solver can model the case (e.g.
    real multiphase / phase-change cases that need interFoam family).
  • confidence: "high" / "medium" / "low".
  • reason: one sentence explaining the key decision.
  • phase_change: true iff the case involves boiling, cavitation,
    evaporation, or condensation.
  • warnings: optional list of strings.
"""


def _build_llm_system_prompt() -> str:
    """Build the solver-selection system prompt, appending the live registry roster.

    The static prompt above encodes the decision rules; the dynamic tail lists
    the solvers actually registered in ``simd_agent/solvers/``.  Adding a new
    plugin package therefore teaches the selector about it automatically,
    without any text edits to this file.
    """
    try:
        from simd_agent.solvers import get_registry
        allowed = sorted(get_registry().allowed_solvers())
    except Exception:
        return _LLM_SOLVER_SYSTEM_PROMPT

    if not allowed:
        return _LLM_SOLVER_SYSTEM_PROMPT

    tail = (
        "\n\n══════════════════════════════════════════════════════════════\n"
        "LIVE REGISTRY (authoritative) — you MUST pick a solver from this\n"
        "list, or return null.  The decision rules above describe the\n"
        "intended semantics; this list is the actual set of plugins\n"
        "registered in this deployment:\n"
        f"  {', '.join(allowed)}\n"
        "══════════════════════════════════════════════════════════════\n"
    )
    return _LLM_SOLVER_SYSTEM_PROMPT + tail


# ─────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────

class SolverSelector:
    """OpenFOAM solver selector: LLM full decision with heuristic fallback.

    Usage::

        selector = SolverSelector()
        solver = await selector.select(
            user_requirements="...",
            simulation_config={...},
            validated_config={...},
        )
    """

    def __init__(self) -> None:
        self._provider = get_provider()
        self.client = self._provider.client
        self.model = self._provider.models.get("super", self._provider.models["default"])

        # Populated after select() — caller can inspect for warnings/metadata
        self.last_result: dict[str, Any] = {}

    async def select(
        self,
        user_requirements: str,
        simulation_config: dict[str, Any],
        validated_config: dict[str, Any] | None = None,
    ) -> str:
        """Select the best OpenFOAM solver.

        0. LLM-fuzzy extraction of explicit user solver mention — if the
           user typed a solver name (even with typos), honor it unless it
           hard-contradicts the time-scheme.
        1. Call the LLM with the full decision prompt.
        2. Parse JSON response — extract solver, confidence, warnings.
        3. Fall back to heuristic if LLM fails or returns invalid solver.

        Returns:
            Solver name string, guaranteed to be in ALLOWED_SOLVERS.
        """
        vconfig = validated_config or {}
        flags = _extract_flags(vconfig)

        # ── Step 0: LLM-fuzzy extraction of explicit user solver name ───────
        # CFD engineers using this tool routinely name the solver they want.
        # We must honor that, including typos ("smiplefoam" → simpleFoam,
        # "BUOYANTBOUSSINESQsimpleFOAM" → buoyantBoussinesqSimpleFoam).  Only
        # veto on a direct steady↔transient mismatch — anything else is the
        # user's call, not ours.
        extracted = await self.extract_user_solver_from_prompt(user_requirements)
        if extracted:
            contradiction = self._time_scheme_contradiction(extracted, flags, user_requirements)
            if contradiction is None:
                self.last_result = {
                    "solver": extracted,
                    "confidence": "high",
                    "reason": "Honoring user-named solver (LLM-fuzzy extraction).",
                    "flags": {},
                    "warnings": [],
                }
                logger.info(f"[SOLVER_SELECT] User-named solver honored: '{extracted}'")
                print(
                    f"\n{'='*70}\n"
                    f"[SOLVER_SELECT] User-named solver honored: '{extracted}'\n"
                    f"  (fuzzy extraction from prompt — no physics override)\n"
                    f"{'='*70}\n"
                )
                return extracted
            logger.warning(
                f"[SOLVER_SELECT] User named '{extracted}' but {contradiction} — "
                f"falling through to physics-based selection"
            )

        # ── LLM full selection (forced tool call) ───────────────────────────
        user_msg = self._build_message(user_requirements, simulation_config, vconfig, flags)
        select_tool = self._build_select_solver_tool()

        for attempt in (1, 2):
            try:
                # 30 s timeout so a slow Pro-model request falls back to
                # the heuristic instead of stalling the entire run.  We
                # never want a single LLM round-trip to block the
                # orchestrator for minutes — solver selection is a fast
                # decision in absolute terms; a long delay means
                # something is wrong upstream and the heuristic
                # fallback is preferable.
                response = await asyncio.wait_for(
                    self.client.aio.models.generate_content(
                        model=self.model,
                        contents=user_msg,
                        config=self._provider.types.GenerateContentConfig(
                            system_instruction=_build_llm_system_prompt(),
                            temperature=0.0,
                            tools=[select_tool],
                            tool_config=self._provider.types.ToolConfig(
                                function_calling_config=self._provider.types.FunctionCallingConfig(
                                    mode="ANY",
                                    allowed_function_names=["report_selected_solver"],
                                ),
                            ),
                        ),
                    ),
                    timeout=30,
                )

                result = self._extract_select_tool_call(response)

                if result is not None:
                    self.last_result = result
                    llm_solver = result.get("solver")
                    confidence = result.get("confidence", "high")
                    reason = result.get("reason", "")
                    warnings = result.get("warnings") or []
                    phase_change = bool(result.get("phase_change", False))

                    # Log warnings prominently
                    if warnings:
                        for w in warnings:
                            logger.warning(f"[SOLVER_SELECT] ⚠ {w}")
                    if phase_change or llm_solver is None:
                        logger.warning(
                            "[SOLVER_SELECT] ⚠ PHASE-CHANGE DETECTED — single-phase solver "
                            "is not physically correct for this case. "
                            f"Warnings: {warnings}"
                        )
                        print(
                            f"\n{'='*70}\n"
                            f"[SOLVER_SELECT] ⚠ PHASE-CHANGE CASE DETECTED\n"
                            f"  {reason}\n"
                            + "".join(f"  WARNING: {w}\n" for w in warnings)
                            + f"{'='*70}\n"
                        )
                        # Phase-change case: multiphase support not yet available; use best heuristic
                        if llm_solver not in ALLOWED_SOLVERS:
                            llm_solver = _heuristic_fallback(validated_config, user_requirements)

                    if llm_solver in ALLOWED_SOLVERS:
                        logger.info(
                            f"[SOLVER_SELECT] LLM → '{llm_solver}' "
                            f"(confidence={confidence}, attempt={attempt})\n"
                            f"  reason: {reason}"
                        )
                        print(
                            f"\n{'='*70}\n"
                            f"[SOLVER_SELECT] LLM selected: '{llm_solver}'\n"
                            f"  confidence: {confidence}\n"
                            f"  reason: {reason}\n"
                            + (f"  warnings: {warnings}\n" if warnings else "")
                            + f"{'='*70}\n"
                        )
                        return llm_solver

                    logger.warning(
                        f"[SOLVER_SELECT] LLM returned invalid/null solver "
                        f"(attempt {attempt}) — retrying"
                    )
                else:
                    logger.warning(
                        f"[SOLVER_SELECT] LLM produced no tool call "
                        f"(attempt {attempt}) — retrying"
                    )

            except asyncio.TimeoutError:
                logger.warning(
                    f"[SOLVER_SELECT] LLM call timed out after 30 s "
                    f"(attempt {attempt}) — retrying once, then falling "
                    "back to the heuristic"
                )
            except Exception as exc:
                logger.warning(f"[SOLVER_SELECT] LLM call failed (attempt {attempt}): {exc}")

        # ── Heuristic fallback ────────────────────────────────────────────────
        fallback = _heuristic_fallback(vconfig, user_requirements)
        logger.info(f"[SOLVER_SELECT] Heuristic fallback: '{fallback}'")
        print(
            f"\n{'='*70}\n"
            f"[SOLVER_SELECT] Heuristic fallback: '{fallback}'\n"
            f"  (LLM unavailable or returned invalid response)\n"
            f"{'='*70}\n"
        )
        return fallback

    # ─────────────────────────────────────────────────────────
    # Forced tool-call builders for the main select() decision
    # ─────────────────────────────────────────────────────────

    def _build_select_solver_tool(self):
        """Build the Gemini Tool used to force a structured solver decision.

        Forcing the LLM to call ``report_selected_solver`` (mode="ANY")
        eliminates the entire class of "model returned partial JSON / wrong
        format / refusal text" failures.  The ``solver`` parameter is
        enum-constrained to the registered allowed solvers (plus the
        ``nullable=True`` escape hatch for unsupported phase-change cases),
        so the model literally cannot emit a name we don't recognise.
        """
        types_ = self._provider.types
        allowed = sorted(ALLOWED_SOLVERS)
        return types_.Tool(
            function_declarations=[
                types_.FunctionDeclaration(
                    name="report_selected_solver",
                    description=(
                        "Report the OpenFOAM solver chosen for this case, "
                        "along with confidence, reason, warnings, and a "
                        "phase-change flag.  Set solver=null only when no "
                        "supported single-phase solver can model the case "
                        "(true multiphase / phase-change scenarios)."
                    ),
                    parameters=types_.Schema(
                        type="OBJECT",
                        properties={
                            "solver": types_.Schema(
                                type="STRING",
                                nullable=True,
                                enum=allowed,
                                description="Canonical solver name, or null for unsupported cases.",
                            ),
                            "confidence": types_.Schema(
                                type="STRING",
                                enum=["high", "medium", "low"],
                                description="Decision confidence.",
                            ),
                            "reason": types_.Schema(
                                type="STRING",
                                description="One sentence explaining the choice.",
                            ),
                            "phase_change": types_.Schema(
                                type="BOOLEAN",
                                description="True if the case involves boiling, cavitation, evaporation, or condensation.",
                            ),
                            "warnings": types_.Schema(
                                type="ARRAY",
                                items=types_.Schema(type="STRING"),
                                description="Optional warnings about assumptions, ambiguities, or phase-change advice.",
                            ),
                        },
                        required=["solver", "confidence", "reason"],
                    ),
                ),
            ],
        )

    @staticmethod
    def _extract_select_tool_call(response) -> dict[str, Any] | None:
        """Walk a Gemini response and return the report_selected_solver args
        as a plain dict, or None if no such tool call exists."""
        for candidate_msg in (getattr(response, "candidates", None) or []):
            content = getattr(candidate_msg, "content", None)
            for part in (getattr(content, "parts", None) or []):
                fc = getattr(part, "function_call", None)
                if fc is None or fc.name != "report_selected_solver":
                    continue
                args = dict(fc.args) if fc.args else {}
                # Normalise warnings to a plain list of strings
                warnings = args.get("warnings") or []
                if not isinstance(warnings, list):
                    warnings = [str(warnings)]
                args["warnings"] = [str(w) for w in warnings]
                return args
        return None

    # ─────────────────────────────────────────────────────────
    # User-intent extraction (LLM-fuzzy, typo-tolerant)
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _time_scheme_contradiction(
        solver: str,
        flags: dict[str, Any],
        user_requirements: str,
    ) -> str | None:
        """Return a reason string if the user-named solver hard-contradicts
        the time scheme, else None.

        We only veto on the steady↔transient axis — every other physics
        consideration (heat, gravity, compressibility, …) is the user's
        call when they typed the solver name themselves.
        """
        # Case-insensitive: "simpleFoam" / "pimpleFoam" use lowercase initials
        # so a literal .endswith("SimpleFoam") would miss them.
        solver_lower = solver.lower()
        user_is_steady = solver_lower.endswith("simplefoam")
        user_is_transient = (
            solver_lower.endswith("pimplefoam") or solver == "chtMultiRegionFoam"
        )
        flag_transient = bool(flags.get("transient"))
        prompt_lower = (user_requirements or "").lower()
        prompt_says_transient = any(
            kw in prompt_lower
            for kw in ("transient", "unsteady", "time-varying", "time varying")
        )
        prompt_says_steady = any(
            kw in prompt_lower
            for kw in ("steady-state", "steady state", " steady ")
        )

        # Trust the prompt over the flags (the flags may not be set yet in
        # precheck).  Fall back to flags only when the prompt is silent.
        is_transient = (
            prompt_says_transient if (prompt_says_transient or prompt_says_steady)
            else flag_transient
        )

        if user_is_steady and is_transient:
            return "prompt/flags indicate transient but the named solver is steady"
        if user_is_transient and not is_transient and prompt_says_steady:
            return "prompt says steady-state but the named solver is transient"
        return None

    async def extract_user_solver_from_prompt(self, user_requirements: str) -> str | None:
        """LLM-fuzzy extraction of an explicit OpenFOAM solver name.

        Uses Gemini function calling (forced, mode="ANY") so the model
        MUST emit a structured tool call with the chosen solver — no
        free-text response, no JSON parsing.  The tool's ``solver``
        parameter is an enum constrained to ``ALLOWED_SOLVERS`` (with
        nullable=True), so the LLM can only return one of the canonical
        names or null.  Returns None when the prompt contains no
        explicit solver name.
        """
        if not user_requirements or not user_requirements.strip():
            return None

        types_ = self._provider.types
        allowed = sorted(ALLOWED_SOLVERS)

        tool = types_.Tool(
            function_declarations=[
                types_.FunctionDeclaration(
                    name="report_explicit_solver",
                    description=(
                        "Report which OpenFOAM solver the user EXPLICITLY named in "
                        "their prompt.  Tolerate typos and casing differences "
                        "(map 'smiplefoam' → simpleFoam, "
                        "'buoyantboussinesqsimplefoam' → "
                        "buoyantBoussinesqSimpleFoam, 'rho pimple' → "
                        "rhoPimpleFoam).  Set solver=null when the user described "
                        "the case but did NOT type a solver name — do not infer "
                        "from physics."
                    ),
                    parameters=types_.Schema(
                        type="OBJECT",
                        properties={
                            "solver": types_.Schema(
                                type="STRING",
                                nullable=True,
                                enum=allowed,
                                description=(
                                    "Canonical solver name matching what the user "
                                    "typed (after typo correction).  Null if the "
                                    "user did not name any solver."
                                ),
                            ),
                        },
                        required=["solver"],
                    ),
                ),
            ],
        )

        config = types_.GenerateContentConfig(
            system_instruction=(
                "You extract one structured fact from a CFD user prompt: which "
                "OpenFOAM solver did the user EXPLICITLY name?  Tolerate typos "
                "and casing.  Do NOT infer a solver from the physics described "
                "in the prompt (heat, gravity, compressibility, etc.) — only "
                "report a solver if the user literally typed its name.  You "
                "MUST call the report_explicit_solver tool exactly once."
            ),
            temperature=0.0,
            tools=[tool],
            tool_config=types_.ToolConfig(
                function_calling_config=types_.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=["report_explicit_solver"],
                ),
            ),
        )

        try:
            # 20 s timeout — the extractor is a single-tool fast call; if
            # it's slower than that something is wrong upstream and we
            # should fall through to the physics-based selector or the
            # heuristic instead of stalling.
            response = await asyncio.wait_for(
                self.client.aio.models.generate_content(
                    model=self.model,
                    contents=user_requirements,
                    config=config,
                ),
                timeout=20,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "[SOLVER_SELECT] User-intent extraction timed out after 20 s "
                "— falling through to physics-based selection"
            )
            return None
        except Exception as exc:
            logger.warning(f"[SOLVER_SELECT] User-intent extraction failed: {exc}")
            return None

        for candidate_msg in (response.candidates or []):
            content = getattr(candidate_msg, "content", None)
            for part in (getattr(content, "parts", None) or []):
                fc = getattr(part, "function_call", None)
                if fc is None or fc.name != "report_explicit_solver":
                    continue
                args = dict(fc.args) if fc.args else {}
                solver = args.get("solver")
                if isinstance(solver, str) and solver in ALLOWED_SOLVERS:
                    logger.info(f"[SOLVER_SELECT] User-intent extractor → {solver!r}")
                    return solver
                return None
        return None

    # ─────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────

    # Known cryogenic fluids: name fragments → (boiling_point_K, display_name)
    _CRYOGENIC_FLUIDS: list[tuple[list[str], float, str]] = [
        (["ln2", "liquid nitrogen", "nitrogen"],  77.4,  "LN2 (liquid nitrogen)"),
        (["lh2", "liquid hydrogen", "hydrogen"],  20.3,  "LH2 (liquid hydrogen)"),
        (["lox", "liquid oxygen",   "oxygen"],    90.2,  "LOX (liquid oxygen)"),
        (["lng", "liquid natural",  "methane"],  111.7,  "LNG/methane"),
        (["lar", "liquid argon",    "argon"],     87.3,  "LAr (liquid argon)"),
        (["lhe", "liquid helium",   "helium"],    4.2,   "LHe (liquid helium)"),
    ]

    def _build_message(
        self,
        user_requirements: str,
        simulation_config: dict[str, Any],
        validated_config: dict[str, Any],
        flags: dict[str, Any],
    ) -> str:
        # Collect fluid and temperature context
        fluid = validated_config.get("fluid") or {}
        fluid_name = (fluid.get("name") or simulation_config.get("fluid_name") or "").strip()
        fluid_rho = fluid.get("rho") or fluid.get("density")
        fluid_temp = fluid.get("temperature")

        # Collect BC temperatures
        bcs = validated_config.get("boundary_conditions") or {}
        bc_temps: list[float] = []
        bc_summary_parts: list[str] = []
        for patch_name, bc in bcs.items():
            if not isinstance(bc, dict):
                continue
            t_bc = bc.get("temperature")
            t_val = t_bc.get("value") if isinstance(t_bc, dict) else None
            v_bc = bc.get("velocity")
            v_val = v_bc.get("value") if isinstance(v_bc, dict) else None
            parts: list[str] = [patch_name]
            if t_val is not None:
                try:
                    bc_temps.append(float(t_val))
                    parts.append(f"T={t_val} K")
                except (TypeError, ValueError):
                    pass
            if v_val is not None:
                parts.append(f"U={v_val} m/s")
            bc_summary_parts.append(", ".join(parts))

        # ── Pre-compute physics signals ────────────────────────────────────────
        delta_t = (max(bc_temps) - min(bc_temps)) if len(bc_temps) >= 2 else 0.0
        t_max_bc = max(bc_temps) if bc_temps else 0.0
        t_min_bc = min(bc_temps) if bc_temps else 0.0

        # Detect cryogenic fluid
        fluid_lower = fluid_name.lower()
        cryo_match: tuple[float, str] | None = None
        for keywords, boiling_k, display in self._CRYOGENIC_FLUIDS:
            if any(kw in fluid_lower for kw in keywords):
                cryo_match = (boiling_k, display)
                break
        # Also check user requirements
        if cryo_match is None:
            req_lower = (user_requirements or "").lower()
            for keywords, boiling_k, display in self._CRYOGENIC_FLUIDS:
                if any(kw in req_lower for kw in keywords):
                    cryo_match = (boiling_k, display)
                    break

        # Build physics analysis section
        analysis_lines: list[str] = []
        phase_change_flag = False

        if cryo_match is not None:
            boiling_k, cryo_display = cryo_match
            analysis_lines.append(f"  Cryogenic fluid detected: {cryo_display}")
            analysis_lines.append(f"  Normal boiling point: {boiling_k} K")
            if bc_temps:
                analysis_lines.append(f"  BC temperature range: {t_min_bc} K – {t_max_bc} K (ΔT = {delta_t:.1f} K)")
            if t_max_bc > boiling_k + 10:
                phase_change_flag = True
                margin = t_max_bc - boiling_k
                analysis_lines.append(
                    f"  ⚠ WALL TEMPERATURE ({t_max_bc} K) IS {margin:.0f} K ABOVE BOILING POINT ({boiling_k} K)"
                )
                analysis_lines.append(
                    f"  ⚠ THIS CASE WILL CAUSE BOILING — PHASE CHANGE IS PHYSICALLY UNAVOIDABLE"
                )
                analysis_lines.append(
                    f"  ⚠ A SINGLE-PHASE SOLVER (rhoPimpleFoam, rhoSimpleFoam) CANNOT MODEL THIS CORRECTLY"
                )
                analysis_lines.append(
                    f"  ⚠ PHASE CHANGE DETECTED: multiphase solvers not yet supported"
                )
        elif delta_t > 0:
            analysis_lines.append(f"  BC temperature range: {t_min_bc} K – {t_max_bc} K (ΔT = {delta_t:.1f} K)")

        # Build context section
        context_lines = []
        if fluid_name:
            context_lines.append(f"Fluid: {fluid_name}")
        if fluid_rho:
            context_lines.append(f"Density: {fluid_rho} kg/m³")
        if fluid_temp:
            context_lines.append(f"Reference temperature: {fluid_temp} K")
        if bc_summary_parts:
            context_lines.append("Boundary conditions: " + "; ".join(bc_summary_parts))

        msg = (
            f"## User Requirements\n{user_requirements or '(none provided)'}\n\n"
            f"## Physics Flags (extracted from config)\n"
            f"```json\n{json.dumps(flags, indent=2)}\n```\n\n"
        )
        if context_lines:
            msg += f"## Fluid & Thermal Context\n" + "\n".join(context_lines) + "\n\n"
        if analysis_lines:
            header = "## ⚠ PRE-COMPUTED PHYSICS ANALYSIS — READ BEFORE DECIDING" if phase_change_flag else "## Pre-computed Physics Analysis"
            msg += f"{header}\n" + "\n".join(analysis_lines) + "\n\n"
        msg += "Select the best solver for this case."
        return msg

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any] | None:
        """Extract and parse the JSON object from LLM response."""
        # Strip markdown code fences if present
        text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
        # Find the outermost JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
