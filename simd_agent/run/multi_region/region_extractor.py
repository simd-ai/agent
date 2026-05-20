# simd_agent/run/region_extractor.py
"""LLM-based per-region detail extraction for multi-region (CHT) cases.

Companion to :mod:`simd_agent.run.solver_selector`: where that module pulls
a single solver name out of the user's prompt, this one pulls *per-region*
physical inputs (fluid/solid identity, inlet velocity / temperature /
pressure, turbulence model) out of the same prompt and projects them onto
the region tree already detected from the mesh.

Why a separate module?  The orchestrator's region auto-detection
(:func:`simd_agent.run.orchestration._detect_regions_from_mesh`) groups
mesh patch names by prefix to produce ``config["regions"]`` with sensible
preset defaults (``air``/``stainless``/``air``).  Those defaults are wrong
for any real case — a Regascold-style regasifier needs LN₂ inside +
stainless wall + water outside.  The prompt always carries that
information; this extractor parses it via a forced Gemini tool call so
the deterministic per-region renderer fills in correct properties.

Design mirrors :class:`SolverSelector` exactly:

  * Forced tool call (``mode="ANY"``, single tool ``report_region_details``)
    — no JSON parsing failures, no model refusals.
  * Tool parameters are enum-constrained against the registered
    :data:`MultiRegionBase.FLUID_REGION_PRESETS` and
    ``SOLID_REGION_PRESETS`` tables.
  * Region names passed in are echoed back by the model so we can map
    the answer onto the existing region dicts without renames.
  * On any failure (no API key, network, schema mismatch) the function
    returns the input ``regions`` unchanged — the case still runs, just
    with the heuristic defaults from the orchestrator.
"""

from __future__ import annotations

import logging
from typing import Any

from simd_agent.llm import get_provider
from simd_agent.solvers.families._multi_region import MultiRegionBase

logger = logging.getLogger(__name__)


# Preset enum sources — single source of truth for what the tool can return.
_FLUID_PRESETS: list[str] = sorted(MultiRegionBase.FLUID_REGION_PRESETS.keys())
_SOLID_PRESETS: list[str] = sorted(MultiRegionBase.SOLID_REGION_PRESETS.keys())
_TURB_MODELS:  tuple[str, ...] = (
    "laminar", "kEpsilon", "kOmegaSST", "kOmega", "none",
)


class RegionExtractor:
    """Forced-tool-call LLM extractor for per-region CHT physics inputs.

    Use as a stateless helper — instantiate once and call :meth:`extract`
    per orchestration; the underlying LLM provider client is reused
    across calls.
    """

    def __init__(self) -> None:
        self._provider = get_provider()
        self.client = self._provider.client
        # Same model as the solver selector (the "super" model when
        # available) — extraction quality is the bottleneck here.
        self.model = self._provider.models.get(
            "super", self._provider.models["default"],
        )

    async def extract(
        self,
        user_requirements: str,
        regions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Refine an auto-detected region list with details from the prompt.

        Args:
            user_requirements: Full user-authored prompt text (the same
                joined-history string the solver extractor sees).
            regions: List of region dicts with at minimum ``name``,
                ``kind``, ``fluid_preset``/``solid_preset``, and
                ``interfaces`` populated by the orchestrator's heuristic
                detector.

        Returns:
            The input ``regions`` list with each entry merged with any
            LLM-extracted overrides (``fluid_preset`` / ``solid_preset``
            keep their auto-detected value if the LLM didn't provide one;
            ``U_init``, ``T_init``, ``p_init``, ``turbulence_model`` are
            added when the prompt mentions them).  Region order and
            ``interfaces`` are preserved.

        On any failure the input ``regions`` is returned unchanged.  No
        exception escapes this method — the orchestrator should always
        be able to proceed with the heuristic defaults.
        """
        if not regions or not user_requirements or not user_requirements.strip():
            return regions

        region_names = [r["name"] for r in regions if r.get("name")]
        if len(region_names) < 2:
            return regions

        types_ = self._provider.types
        tool = self._build_tool(types_, region_names)
        config = types_.GenerateContentConfig(
            system_instruction=self._system_prompt(regions),
            temperature=0.0,
            tools=[tool],
            tool_config=types_.ToolConfig(
                function_calling_config=types_.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=["report_region_details"],
                ),
            ),
        )

        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=user_requirements,
                config=config,
            )
        except Exception as exc:
            logger.warning(
                f"[REGION_EXTRACT] LLM call failed ({exc}); keeping heuristic "
                f"region defaults"
            )
            return regions

        details = self._extract_tool_call(response)
        if details is None:
            logger.warning(
                "[REGION_EXTRACT] No report_region_details tool call in "
                "response; keeping heuristic defaults"
            )
            return regions

        return self._merge(regions, details)

    # ─────────────────────────────────────────────────────────
    # Tool schema
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _build_tool(types_, region_names: list[str]):
        """Build the Gemini Tool for forced per-region detail reporting."""
        region_entry = types_.Schema(
            type="OBJECT",
            properties={
                "name": types_.Schema(
                    type="STRING",
                    enum=region_names,
                    description=(
                        "Region name — MUST match one of the names "
                        "listed in the system prompt exactly."
                    ),
                ),
                "kind": types_.Schema(
                    type="STRING",
                    enum=["fluid", "solid"],
                    description=(
                        "Whether this region holds a flowing fluid "
                        "(Navier-Stokes + energy) or a stationary solid "
                        "(heat conduction only)."
                    ),
                ),
                "fluid_preset": types_.Schema(
                    type="STRING",
                    nullable=True,
                    enum=_FLUID_PRESETS,
                    description=(
                        "Canonical fluid preset (only for fluid regions). "
                        "Null when the user did not name a fluid or named "
                        "one outside the supported set."
                    ),
                ),
                "solid_preset": types_.Schema(
                    type="STRING",
                    nullable=True,
                    enum=_SOLID_PRESETS,
                    description=(
                        "Canonical solid preset (only for solid regions). "
                        "Null when the user did not name a material."
                    ),
                ),
                "inlet_velocity": types_.Schema(
                    type="NUMBER",
                    nullable=True,
                    description=(
                        "Inlet bulk velocity in m/s, signed along the "
                        "flow direction.  Use positive values for "
                        "flow in +x, negative for -x (counter-flow).  "
                        "Null when no inlet velocity is specified."
                    ),
                ),
                "inlet_temperature": types_.Schema(
                    type="NUMBER",
                    nullable=True,
                    description=(
                        "Inlet temperature in Kelvin.  Null when not "
                        "specified."
                    ),
                ),
                "inlet_pressure": types_.Schema(
                    type="NUMBER",
                    nullable=True,
                    description=(
                        "Inlet pressure in Pa (use 101325 for "
                        "atmospheric / 1 bar).  Null when not specified."
                    ),
                ),
                "turbulence_model": types_.Schema(
                    type="STRING",
                    nullable=True,
                    enum=list(_TURB_MODELS),
                    description=(
                        "RAS turbulence model name (only for fluid "
                        "regions).  Null when not specified — caller "
                        "will default to kEpsilon."
                    ),
                ),
            },
            required=["name", "kind"],
        )

        return types_.Tool(
            function_declarations=[
                types_.FunctionDeclaration(
                    name="report_region_details",
                    description=(
                        "Report per-region physical inputs extracted "
                        "from the user's prompt.  Emit one entry per "
                        "region listed in the system prompt; do NOT "
                        "invent regions that weren't listed."
                    ),
                    parameters=types_.Schema(
                        type="OBJECT",
                        properties={
                            "regions": types_.Schema(
                                type="ARRAY",
                                items=region_entry,
                                description=(
                                    "One entry per region.  Order does "
                                    "not matter; entries are matched "
                                    "by name."
                                ),
                            ),
                        },
                        required=["regions"],
                    ),
                ),
            ],
        )

    @staticmethod
    def _system_prompt(regions: list[dict[str, Any]]) -> str:
        """Build the system prompt — lists region names and asks for details."""
        lines = [
            "You extract structured per-region physical inputs from a CFD "
            "user prompt for a multi-region (conjugate heat transfer, CHT) "
            "OpenFOAM case.  The mesh has already been split into the "
            "following regions:",
            "",
        ]
        for r in regions:
            kind = r.get("kind") or ("fluid" if r.get("fluid_preset") else "solid")
            ifaces = ", ".join(r.get("interfaces") or []) or "(none)"
            lines.append(
                f"  - {r['name']}  (kind={kind}, interfaces={ifaces})"
            )
        lines.extend([
            "",
            "Read the prompt and decide, for EACH listed region:",
            "  * Its kind (fluid vs solid).",
            "  * The fluid preset (for fluid regions) or solid preset "
            "(for solid regions).  Pick from the enum in the tool "
            "schema; return null if the prompt doesn't name a "
            "specific material.",
            "  * Inlet velocity, temperature and pressure if the prompt "
            "specifies them for that region.  Counter-flow setups are "
            "common — encode flow direction in the velocity sign "
            "(positive = +x, negative = -x).  Return null when not "
            "specified.",
            "  * Turbulence model for fluid regions when the prompt "
            "names one.",
            "",
            "Rules:",
            "  1. Emit EXACTLY one entry per region listed above — no "
            "more, no fewer.",
            "  2. Never invent regions the user didn't ask for.",
            "  3. Don't infer values from physics intuition — only "
            "report what the prompt literally states.",
            "  4. You MUST call the ``report_region_details`` tool "
            "exactly once.",
        ])
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────
    # Response handling
    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _extract_tool_call(response) -> list[dict[str, Any]] | None:
        """Pull the ``regions`` array from the model's tool call, or None."""
        for candidate in (getattr(response, "candidates", None) or []):
            content = getattr(candidate, "content", None)
            for part in (getattr(content, "parts", None) or []):
                fc = getattr(part, "function_call", None)
                if fc is None or fc.name != "report_region_details":
                    continue
                args = dict(fc.args) if fc.args else {}
                regions = args.get("regions")
                if isinstance(regions, list):
                    return [dict(r) for r in regions if isinstance(r, dict)]
                return None
        return None

    @staticmethod
    def _merge(
        regions: list[dict[str, Any]],
        details: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge LLM-supplied details into the heuristic-built region list.

        The heuristic detector is authoritative for *topology*
        (``name``, ``interfaces``, ``kind``); the LLM is authoritative
        for *physics* (presets, inlet conditions, turbulence).  When the
        LLM disagrees with the heuristic on ``kind`` we trust the LLM —
        the user's prompt is closer to ground truth than naming
        conventions.
        """
        by_name = {d.get("name"): d for d in details if d.get("name")}
        merged: list[dict[str, Any]] = []
        for r in regions:
            out = dict(r)
            d = by_name.get(r.get("name"))
            if d is None:
                merged.append(out)
                continue

            # Trust the LLM on kind (only relevant if it disagrees).
            if d.get("kind") in ("fluid", "solid"):
                out["kind"] = d["kind"]

            # Presets: LLM wins if it gave a concrete value.
            if d.get("fluid_preset"):
                out["fluid_preset"] = d["fluid_preset"]
                # If the LLM identified it as a fluid, drop any stale
                # solid_preset that the heuristic might have set.
                out.pop("solid_preset", None)
            if d.get("solid_preset"):
                out["solid_preset"] = d["solid_preset"]
                out.pop("fluid_preset", None)

            # Inlet conditions → RegionSpec init values.  Velocity is
            # stored as a 3-tuple (Ux, Uy, Uz) with the magnitude in +x
            # (or -x for counter-flow); the RegionSpec consumer expects
            # this tuple shape.
            u = d.get("inlet_velocity")
            if isinstance(u, (int, float)):
                out["U_init"] = (float(u), 0.0, 0.0)
            t = d.get("inlet_temperature")
            if isinstance(t, (int, float)):
                out["T_init"] = float(t)
            p = d.get("inlet_pressure")
            if isinstance(p, (int, float)):
                out["p_init"] = float(p)
            tm = d.get("turbulence_model")
            if isinstance(tm, str) and tm:
                out["turbulence_model"] = tm

            merged.append(out)
        return merged
