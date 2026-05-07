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

# Solvers where pressure is *kinematic* p (m²/s²) — generate 0/p
P_SOLVERS: set[str] = {
    "simpleFoam",
    "pimpleFoam",
    "rhoSimpleFoam",
    "rhoPimpleFoam",
}

# Solvers where pressure is p_rgh (Pa relative) — generate 0/p_rgh + constant/g
P_RGH_SOLVERS: set[str] = {
    "buoyantSimpleFoam",
    "buoyantPimpleFoam",
}

# Solvers that solve the energy equation and MUST have 0/T
ENERGY_SOLVERS: set[str] = {
    "rhoSimpleFoam",
    "rhoPimpleFoam",
    "buoyantSimpleFoam",
    "buoyantPimpleFoam",
}

# Solvers that require constant/g (gravity vector)
GRAVITY_SOLVERS: set[str] = P_RGH_SOLVERS

# Solvers that require constant/thermophysicalProperties
THERMO_SOLVERS: set[str] = ENERGY_SOLVERS

ALLOWED_SOLVERS: set[str] = P_SOLVERS | P_RGH_SOLVERS


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

    return {
        "heat":         heat,
        "multiphase":   multiphase,
        "phases":       phases,
        "n_phases":     len(phases) if phases else (2 if multiphase else 1),
        "compressible": compressible,
        "transient":    transient,
        "laminar":      laminar,
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


def _heuristic_fallback(validated_config: dict[str, Any]) -> str:
    """Pure-logic fallback when the LLM response is unparseable or invalid."""
    f = _extract_flags(validated_config)

    # Cryogenic single-phase liquid (LH2, LN2, LOX, LHe, etc.) → always use rho* solver.
    if _is_cryogenic_liquid(validated_config):
        logger.warning("[SOLVER_SELECT] Heuristic: cryogenic liquid → rho* solver (density-temperature coupling)")
        return "rhoPimpleFoam" if f["transient"] else "rhoSimpleFoam"

    # Buoyancy-driven / natural convection: heat transfer + incompressible
    if f["heat"] and not f["compressible"]:
        logger.info("[SOLVER_SELECT] Heuristic: heat transfer + incompressible → buoyant solver")
        return "buoyantPimpleFoam" if f["transient"] else "buoyantSimpleFoam"

    # Compressible with heat (high-speed)
    if f["heat"] or f["compressible"]:
        return "rhoPimpleFoam" if f["transient"] else "rhoSimpleFoam"

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
  simpleFoam        — steady-state, standard industrial flows (pipes, ducts, external aero)
  pimpleFoam        — transient, vortex shedding, moving mesh, pulsating flow

  # High-speed compressible with heat (Mach > 0.3)
  rhoSimpleFoam     — compressible steady-state, high-speed aerodynamics
  rhoPimpleFoam     — compressible transient, pressure waves, acoustics

  # Buoyancy-driven (natural/forced convection, gravity-dominated, HVAC)
  buoyantSimpleFoam — buoyancy steady-state, heated rooms, HVAC, electronic cooling
  buoyantPimpleFoam — buoyancy transient, smoke spread, fire, ventilation transients

NOTE: Multiphase (interFoam, compressibleInterFoam, etc.) is NOT currently supported.
If the case requires two or more phases, set solver to null and warn the user.
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

## 2. Single-phase path — Buoyancy / Natural Convection (CHECK FIRST for heat cases)
Apply when ALL of:
  - Heat transfer is active
  - Single-phase only
  - Low-speed (Mach < 0.3)
  - Gravity matters: natural convection, heated room, HVAC, chimney, solar collector,
    electronic cooling, smoke spread, density stratification, buoyancy-driven flow

  Steady-state  → buoyantSimpleFoam
  Transient     → buoyantPimpleFoam

## 3. Single-phase — High-speed Compressible (gravity not dominant)
Compressible signals (any one):
  - Mach > 0.3, supersonic, transonic, shock
  - Density varies significantly with pressure or temperature
  - Gas at large pressure differential (> ~10% of absolute)
  - ⚠ Cryogenic liquid (LH2, LN2, LOX, LHe — boiling point < 130 K):
      ALWAYS compressible; icoPolynomial EOS required.

  Steady-state  → rhoSimpleFoam
  Transient     → rhoPimpleFoam

## 4. Single-phase incompressible, no heat transfer
  - Liquid at moderate conditions (water, oil, glycol — NOT cryogenic)
  - Density constant, Mach < 0.1, no temperature gradients

  Steady-state  → simpleFoam
  Transient     → pimpleFoam

## 5. Steady vs transient
Steady: "steady", "RANS", "converge", "time-averaged", "mean flow"
Transient: "transient", "unsteady", "time-varying", "oscillating", "pulsating",
  "start-up", "smoke spread", "moving parts", "ventilation transient"
When ambiguous: default to steady.

──────────────────────────────────────────────────────────────
OUTPUT FORMAT — return ONLY valid JSON, no other text
──────────────────────────────────────────────────────────────
{
  "solver": "<solver_name_or_null>",
  "confidence": "high" | "medium" | "low",
  "reason": "<one sentence explaining the key decision>",
  "flags": {
    "phases": <integer>,
    "compressible": <bool>,
    "transient": <bool>,
    "heat": <bool>,
    "phase_change": <bool>,
    "sharp_interface": <bool>
  },
  "warnings": ["<optional — note assumptions, ambiguities, or phase-change advice>"]
}
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

        1. Call the LLM with the full decision prompt.
        2. Parse JSON response — extract solver, confidence, warnings.
        3. Fall back to heuristic if LLM fails or returns invalid solver.

        Returns:
            Solver name string, guaranteed to be in ALLOWED_SOLVERS.
        """
        vconfig = validated_config or {}
        flags = _extract_flags(vconfig)

        # ── LLM full selection ────────────────────────────────────────────────
        user_msg = self._build_message(user_requirements, simulation_config, vconfig, flags)

        for attempt in (1, 2):
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=user_msg,
                    config=self._provider.types.GenerateContentConfig(
                        system_instruction=_build_llm_system_prompt(),
                        temperature=0.0,
                        max_output_tokens=512,
                    ),
                )
                raw = (response.text or "").strip()
                result = self._parse_json(raw)

                if result:
                    self.last_result = result
                    llm_solver = result.get("solver")
                    confidence = result.get("confidence", "high")
                    reason = result.get("reason", "")
                    warnings = result.get("warnings") or []
                    phase_change = (result.get("flags") or {}).get("phase_change", False)

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
                            llm_solver = _heuristic_fallback(validated_config)

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
                        f"[SOLVER_SELECT] LLM returned invalid solver '{llm_solver}' "
                        f"(attempt {attempt}) — retrying"
                    )
                    user_msg = (
                        f"{user_msg}\n\n"
                        f"NOTE: '{llm_solver}' is not in the allowed solver list. "
                        f"Choose exactly one of: {sorted(ALLOWED_SOLVERS)}"
                    )

            except Exception as exc:
                logger.warning(f"[SOLVER_SELECT] LLM call failed (attempt {attempt}): {exc}")

        # ── Heuristic fallback ────────────────────────────────────────────────
        fallback = _heuristic_fallback(vconfig)
        logger.info(f"[SOLVER_SELECT] Heuristic fallback: '{fallback}'")
        print(
            f"\n{'='*70}\n"
            f"[SOLVER_SELECT] Heuristic fallback: '{fallback}'\n"
            f"  (LLM unavailable or returned invalid response)\n"
            f"{'='*70}\n"
        )
        return fallback

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
