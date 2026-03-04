# simd_agent/solver_selector.py
"""OpenFOAM solver selection: deterministic rules first, LLM only for ambiguous cases.

Phase 1 of the two-phase codegen pipeline:
  1. SolverSelector  → pick the right application from the allowed list
  2. GenAICodeGenerator → generate the full case using solver-specific prompts

Design philosophy
─────────────────
The vast majority of solver choices are **fully determined** by hard physics flags
(heat_transfer, time_stepping, multiphase, compressibility, flow_regime).  An LLM
has no advantage here and can only introduce errors.

The LLM is therefore called ONLY when there is genuine ambiguity that cannot be
resolved from config flags alone — currently just the choice between the standard
VOF advection scheme (interFoam) and the sharper isoAdvector scheme (interIsoFoam /
compressibleInterIsoFoam).  All other cases are answered deterministically.

Decision tree (applied in order)
─────────────────────────────────
1. Multiphase (N > 2 phases)          → compressibleMultiphaseInterFoam  [deterministic]
2. Two-phase + (compressible OR heat) → compressibleInterFoam / *IsoFoam [LLM tiebreak]
3. Two-phase incompressible           → interFoam / interIsoFoam          [LLM tiebreak]
4. Single-phase + heat OR compressible
      + transient                     → rhoPimpleFoam                     [deterministic]
      + steady                        → rhoSimpleFoam                     [deterministic]
5. Single-phase incompressible, no heat
      + steady                        → simpleFoam                        [deterministic]
      + transient + laminar           → icoFoam                           [deterministic]
      + transient + turbulent         → pimpleFoam                        [deterministic]
"""

import json
import logging
import os
import re
from typing import Any

from google import genai
from google.genai import types

from simd_agent.settings import get_settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Canonical allowed-solver set (single source of truth)
# ─────────────────────────────────────────────────────────────

# Solvers where pressure is *kinematic* p (m²/s²) — generate 0/p
P_SOLVERS: set[str] = {
    "simpleFoam",
    "pimpleFoam",
    "icoFoam",
    "rhoSimpleFoam",
    "rhoPimpleFoam",
}

# Solvers where pressure is p_rgh (Pa relative) — generate 0/p_rgh + constant/g
P_RGH_SOLVERS: set[str] = {
    "interFoam",
    "interIsoFoam",
    "compressibleInterFoam",
    "compressibleInterIsoFoam",
    "compressibleMultiphaseInterFoam",
}

# Solvers that solve the energy equation and MUST have 0/T
ENERGY_SOLVERS: set[str] = {
    "rhoSimpleFoam",
    "rhoPimpleFoam",
    "compressibleInterFoam",
    "compressibleInterIsoFoam",
    "compressibleMultiphaseInterFoam",
}

# Solvers that require constant/g (gravity vector)
GRAVITY_SOLVERS: set[str] = P_RGH_SOLVERS

# Solvers that require constant/thermophysicalProperties
THERMO_SOLVERS: set[str] = ENERGY_SOLVERS

ALLOWED_SOLVERS: set[str] = P_SOLVERS | P_RGH_SOLVERS


# ─────────────────────────────────────────────────────────────
# Physics flag extraction helper
# ─────────────────────────────────────────────────────────────

def _extract_flags(validated_config: dict[str, Any]) -> dict[str, Any]:
    """Normalise the physics flags that drive solver selection.

    Handles both flat configs (keys at top level) and nested configs
    (keys under a ``physics`` sub-dict), as well as common aliases.
    """
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
# Deterministic selector
# ─────────────────────────────────────────────────────────────

def _deterministic_select(
    validated_config: dict[str, Any],
) -> tuple[str | None, set[str]]:
    """Apply hard physics rules to narrow down the solver choice.

    Returns
    -------
    (definitive_solver, candidates)
        • If *definitive_solver* is not None → use it directly, skip LLM.
        • If *definitive_solver* is None → pass *candidates* to the LLM to
          pick among (both are valid; user preference / description decides).
    """
    f = _extract_flags(validated_config)

    # ── MULTIPHASE ──────────────────────────────────────────────────────────
    if f["multiphase"] or f["n_phases"] > 1:
        if f["n_phases"] > 2:
            # Only one solver for 3+ phases
            return "compressibleMultiphaseInterFoam", set()

        if f["compressible"] or f["heat"]:
            # Compressible two-phase: standard vs isoAdvector — genuine ambiguity
            return None, {"compressibleInterFoam", "compressibleInterIsoFoam"}
        else:
            # Incompressible two-phase: standard vs isoAdvector — genuine ambiguity
            return None, {"interFoam", "interIsoFoam"}

    # ── SINGLE-PHASE: energy equation required ──────────────────────────────
    if f["heat"] or f["compressible"]:
        if f["transient"]:
            return "rhoPimpleFoam", set()   # 100 % deterministic
        else:
            return "rhoSimpleFoam", set()   # 100 % deterministic

    # ── SINGLE-PHASE: incompressible, no heat ───────────────────────────────
    if not f["transient"]:
        return "simpleFoam", set()          # 100 % deterministic

    # Transient + incompressible + no heat
    if f["laminar"]:
        return "icoFoam", set()             # 100 % deterministic

    return "pimpleFoam", set()              # 100 % deterministic


# ─────────────────────────────────────────────────────────────
# Heuristic fallback (last resort, no LLM)
# ─────────────────────────────────────────────────────────────

def _heuristic_fallback(validated_config: dict[str, Any]) -> str:
    """Pure-logic fallback when LLM response is unparseable or invalid.

    This is identical in logic to _deterministic_select() but always returns
    a single solver string (picks the safe default for ambiguous VOF cases).
    """
    solver, candidates = _deterministic_select(validated_config)
    if solver:
        return solver
    # For ambiguous VOF cases default to the non-Iso variant (safer default)
    if "compressibleInterFoam" in candidates:
        return "compressibleInterFoam"
    return "interFoam"


# ─────────────────────────────────────────────────────────────
# LLM prompt for genuinely ambiguous cases
# ─────────────────────────────────────────────────────────────

_AMBIGUOUS_SYSTEM_PROMPT = """\
You are an expert OpenFOAM solver selector.
The physics constraints have been checked automatically. The simulation is valid
for EACH solver in the candidate list below.  Your ONLY job is to choose the
best-fit solver based on the user description and preferences.

Criteria for choosing the isoAdvector variant (*IsoFoam):
  • User asks for "sharp interface", "crisp interface", or "isoAdvector"
  • Simulation involves fast impact, thin film, or drop dynamics
  • Otherwise: prefer the standard variant

Respond with ONLY the solver name — no other text, no explanation, no quotes.\
"""


# ─────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────

class SolverSelector:
    """OpenFOAM solver selector: deterministic first, LLM only for ambiguous VOF cases.

    Usage::

        selector = SolverSelector()
        solver = await selector.select(
            user_requirements="...",
            simulation_config={...},      # raw from frontend
            validated_config={...},       # from linting phase
        )
    """

    def __init__(self) -> None:
        settings = get_settings()
        api_key = (
            settings.gemini_api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        if not api_key:
            raise ValueError("GEMINI_API_KEY not configured")
        self.client = genai.Client(api_key=api_key)
        # Use the super model for the (rare) LLM tiebreak — accuracy matters
        self.model = getattr(settings, "gemini_super_model", None) or settings.gemini_model

    async def select(
        self,
        user_requirements: str,
        simulation_config: dict[str, Any],
        validated_config: dict[str, Any] | None = None,
    ) -> str:
        """Select the best OpenFOAM solver.

        1. Run deterministic physics rules → if unambiguous, return immediately.
        2. If ambiguous (interFoam variants), ask the super-model to tiebreak.
        3. Fall back to heuristic if LLM fails.

        Returns:
            Solver name string, guaranteed to be in ALLOWED_SOLVERS.
        """
        vconfig = validated_config or {}

        # ── Step 1: deterministic selection ──────────────────────────────────
        flags = _extract_flags(vconfig)
        solver, candidates = _deterministic_select(vconfig)

        if solver:
            logger.info(
                f"[SOLVER_SELECT] Deterministic rule → '{solver}'  "
                f"(heat={flags['heat']}, compressible={flags['compressible']}, "
                f"transient={flags['transient']}, multiphase={flags['multiphase']}, "
                f"laminar={flags['laminar']})"
            )
            print(
                f"\n{'='*70}\n"
                f"[SOLVER_SELECT] Deterministic selection: '{solver}'\n"
                f"  heat={flags['heat']}  compressible={flags['compressible']}  "
                f"transient={flags['transient']}  multiphase={flags['multiphase']}  "
                f"laminar={flags['laminar']}\n"
                f"  (No LLM needed — physics flags are unambiguous)\n"
                f"{'='*70}\n"
            )
            return solver

        # ── Step 2: LLM tiebreak for genuinely ambiguous cases ────────────────
        # At this point candidates = {interFoam, interIsoFoam} or
        #                            {compressibleInterFoam, compressibleInterIsoFoam}
        logger.info(
            f"[SOLVER_SELECT] Ambiguous candidates {candidates} — calling LLM tiebreak"
        )
        print(
            f"\n{'='*70}\n"
            f"[SOLVER_SELECT] Ambiguous candidates: {candidates}\n"
            f"  Calling LLM to pick between isoAdvector variants...\n"
            f"{'='*70}\n"
        )

        user_msg = self._build_ambiguous_message(user_requirements, simulation_config, vconfig, candidates)

        for attempt in (1, 2):
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=user_msg,
                    config=types.GenerateContentConfig(
                        system_instruction=_AMBIGUOUS_SYSTEM_PROMPT,
                        temperature=0.0,
                        max_output_tokens=64,
                    ),
                )
                raw = (response.text or "").strip()
                llm_solver = self._parse(raw, candidates)
                if llm_solver in candidates:
                    logger.info(f"[SOLVER_SELECT] LLM tiebreak chose '{llm_solver}' (attempt {attempt})")
                    return llm_solver

                logger.warning(
                    f"[SOLVER_SELECT] LLM returned '{llm_solver}' not in candidates "
                    f"{candidates} (attempt {attempt})"
                )
                user_msg = (
                    f"{user_msg}\n\n"
                    f"NOTE: '{raw}' is not in the candidate list {sorted(candidates)}. "
                    "Choose one of those solvers only."
                )
            except Exception as exc:
                logger.warning(f"[SOLVER_SELECT] LLM tiebreak failed (attempt {attempt}): {exc}")

        # ── Step 3: heuristic fallback ────────────────────────────────────────
        fallback = _heuristic_fallback(vconfig)
        logger.info(f"[SOLVER_SELECT] Heuristic fallback: '{fallback}'")
        return fallback

    # ─────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────

    def _build_ambiguous_message(
        self,
        user_requirements: str,
        simulation_config: dict[str, Any],
        validated_config: dict[str, Any],
        candidates: set[str],
    ) -> str:
        flags = _extract_flags(validated_config)
        return (
            f"## Candidate solvers (both are physically valid)\n"
            f"{sorted(candidates)}\n\n"
            f"## User Requirements\n{user_requirements}\n\n"
            f"## Physics Flags\n```json\n{json.dumps(flags, indent=2)}\n```\n\n"
            "Choose the solver that best matches the user's description."
        )

    @staticmethod
    def _parse(text: str, candidates: set[str]) -> str:
        """Extract the best solver name from LLM response, preferring candidates."""
        # Prefer exact match within the candidate set first
        for s in sorted(candidates, key=len, reverse=True):
            if s in text:
                return s
        # Then try any allowed solver
        for s in sorted(ALLOWED_SOLVERS, key=len, reverse=True):
            if s in text:
                return s
        # Fall back: grab first camelCase word
        m = re.search(r'\b([a-zA-Z][a-zA-Z0-9]+Foam|rhoSimpleFoam|rhoPimpleFoam|icoFoam)\b', text)
        if m:
            return m.group(1)
        return text.split()[0] if text.split() else "simpleFoam"
