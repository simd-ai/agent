# simd_agent/precheck.py
"""Precheck service — analyzes user prompts and extracts CFD simulation specs.

Architecture (multi-pass):
  Pass 1 – Boundary planner  : single streamed LLM call → BoundaryPlan per patch
  Pass 2 – Patch agents      : one LLM call per patch, run in parallel, stream live
  Pass 3 – Merge             : combine all PatchSpec results → PrecheckResponse
  Pass 4 – Review            : streamed narrative review of the final spec
"""

import asyncio
import copy
import json
import logging
from typing import Any, AsyncIterator

from google import genai
from google.genai import types

from simd_agent.settings import get_settings
from simd_agent.precheck.models import (
    # request / response
    PrecheckRequest, PrecheckResponse, SuggestedConfig,
    SolverSettings, FluidProperties, TurbulenceSettings,
    FieldBC, PatchBoundaryCondition,
    BoundaryHint, VelocityBC, PressureBC, TemperatureBC,
    Interpretation, ConfidenceScores,
    # constants
    FLUID_PRESETS, CRYOGENIC_KEYWORDS,
    PRECHECK_MODEL,
    # new tool schemas
    BOUNDARY_PLAN_TOOL_SCHEMA, PATCH_SPEC_TOOL_SCHEMA,
    BC_KNOWLEDGE_DIR,
)

logger = logging.getLogger(__name__)

# Max concurrent patch-agent LLM calls
_PATCH_CONCURRENCY = 4


def _norm_conf(value: float | int) -> float:
    """Normalise a confidence value to [0, 1] (LLM sometimes returns percentages)."""
    v = float(value)
    return v / 100.0 if v > 1.0 else v


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# PrecheckService
# ---------------------------------------------------------------------------

class PrecheckService:
    """Streaming precheck: planner → parallel patch agents → merge → review."""

    def __init__(self):
        self.settings = get_settings()
        self._client: genai.Client | None = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=self.settings.gemini_api_key)
        return self._client

    # ── Public API ───────────────────────────────────────────────────────────

    async def analyze(self, request: PrecheckRequest) -> PrecheckResponse:
        """Non-streaming wrapper: runs analyze_stream and returns the final spec."""
        validation_error = request.validate_prompt()
        if validation_error:
            return self._create_friendly_error_response(validation_error)
        try:
            async for event in self.analyze_stream(request):
                if event.get("type") == "spec":
                    # Re-parse from the serialised dict so we get a proper model back
                    return PrecheckResponse(**event["data"])
            return self._create_friendly_error_response("No spec produced by analyze_stream.")
        except Exception as e:
            logger.exception(f"[Precheck] analyze() failed: {e}")
            return self._create_friendly_error_response(str(e))

    async def analyze_stream(self, request: PrecheckRequest) -> AsyncIterator[dict[str, Any]]:
        """Async generator for streaming precheck over WebSocket.

        Event sequence:
            start
            planner_start
            planner_thought  (0-N)
            planner_done
            boundary_plan
            patches_start
              patch_start      × N patches
              patch_retrieval  × N patches
              patch_thought    × N patches (0-M each)
              patch_spec       × N patches
              spec_partial     × N patches
              patch_done       × N patches
            patches_done
            merge_start
            spec
            review_start
            review_phase_start
            thought            (0-M, review thinking)
            review_phase_done
            review_item
            review_done
            done
        """
        print(f"[Precheck] analyze_stream — prompt={request.prompt[:60]!r}", flush=True)

        validation_error = request.validate_prompt()
        if validation_error:
            yield {"type": "start"}
            yield {"type": "error", "message": validation_error}
            yield {"type": "done"}
            return

        yield {"type": "start"}

        try:
            # ── Pass 1: Boundary planner ──────────────────────────────────────
            yield {"type": "planner_start"}

            planner_result: dict[str, Any] | None = None
            async for event in self._stream_boundary_planner(request):
                yield event
                if event["type"] == "boundary_plan":
                    planner_result = event["data"]

            if not planner_result:
                yield {"type": "error", "message": "Boundary planner returned no plan."}
                yield {"type": "done"}
                return

            yield {"type": "planner_done"}

            # ── Pass 2: Parallel patch agents ─────────────────────────────────
            patches = planner_result.get("patches", [])
            yield {"type": "patches_start", "count": len(patches)}

            patch_specs: dict[str, dict[str, Any]] = {}
            partial_boundary_state: dict[str, Any] = {}

            async for event in self._stream_patch_agents_parallel(
                request=request,
                boundary_plan=planner_result,
                patch_specs=patch_specs,
                partial_boundary_state=partial_boundary_state,
            ):
                yield event

            yield {"type": "patches_done", "count": len(patch_specs)}

            # ── Pass 3: Merge ─────────────────────────────────────────────────
            yield {"type": "merge_start"}
            result = self._merge_patch_specs_into_response(
                request=request,
                boundary_plan=planner_result,
                patch_specs=patch_specs,
            )
            yield {"type": "spec", "data": result.model_dump(by_alias=True)}

            # ── Pass 4: Review ────────────────────────────────────────────────
            yield {"type": "review_start"}
            async for event in self._llm_review(request, result):
                yield event

        except Exception as e:
            logger.exception(f"[Precheck] Streaming failed: {e}")
            yield {"type": "error", "message": str(e)}
        finally:
            yield {"type": "done"}

    # ── Pass 1: Boundary planner ─────────────────────────────────────────────

    async def _stream_boundary_planner(
        self, request: PrecheckRequest
    ) -> AsyncIterator[dict[str, Any]]:
        prompt = self._build_boundary_planner_prompt(request)
        print(f"[Precheck/planner] prompt={len(prompt)} chars → {PRECHECK_MODEL}", flush=True)

        func_call_args: dict[str, Any] | None = None

        stream = await self.client.aio.models.generate_content_stream(
            model=PRECHECK_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(include_thoughts=True),
                tools=[BOUNDARY_PLAN_TOOL_SCHEMA],
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode="ANY",
                        allowed_function_names=["submit_boundary_plan"],
                    )
                ),
            ),
        )

        async for chunk in stream:
            if not chunk.candidates:
                continue
            for part in chunk.candidates[0].content.parts:
                if getattr(part, "thought", False) and part.text:
                    yield {"type": "planner_thought", "text": part.text}
                elif getattr(part, "function_call", None) is not None:
                    func_call_args = dict(part.function_call.args)

        if not func_call_args:
            yield {"type": "error", "message": "Planner LLM returned no function call."}
            return

        print("[Precheck/planner] plan received:\n" + json.dumps(func_call_args, indent=2, default=str), flush=True)
        yield {"type": "boundary_plan", "data": func_call_args}

    def _build_boundary_planner_prompt(self, request: PrecheckRequest) -> str:
        mesh = request.get_mesh()
        mesh_summary = self._format_mesh_summary(mesh)
        solver_context = self._format_solver_context(request)

        parts = [
            "You are a CFD boundary-condition planner for OpenFOAM-style case generation.",
            "",
            "Your job is NOT to generate the final full CFD spec.",
            "Your job is to create a per-patch boundary plan that decides:",
            "  1. the role of each patch",
            "  2. the physical requirement(s) that apply to that patch",
            "  3. which boundary-condition family to use for each field",
            "  4. which retrieval markdown files must be loaded for the next patch-specific LLM call",
            "",
            "Return ONLY data matching the BoundaryPlan schema via submit_boundary_plan.",
            "",
            "=== USER REQUEST ===",
            request.prompt,
            "",
            "=== MESH SUMMARY ===",
            mesh_summary,
            "",
            "=== SOLVER CONTEXT ===",
            solver_context,
            "",
            "=== MVP PATCH ROLES ===",
            "Support: inlet, outlet, wall, frontAndBack, symmetry, other.",
            "Do not implement reverse-flow / inletOutlet logic yet. Assume no backflow.",
            "",
            "=== TIME SCHEME — SET time_scheme IN YOUR RESPONSE ===",
            "DEFAULT: time_scheme='transient'. Transient is the recommended approach for all CFD",
            "simulations — it captures startup effects, avoids divergence from bad ICs, and",
            "produces more physically meaningful results than a steady solver.",
            "",
            "Use time_scheme='steady' ONLY when the user explicitly requests steady-state:",
            "  e.g. 'steady-state', 'steady flow', 'RANS steady', 'converged solution',",
            "  'I only need the converged result, no time history'.",
            "",
            "Use time_scheme='transient' (and extract end_time) when the user mentions ANY of:",
            "  • A clock duration: 'for 2s', 'for 2 seconds', '2s simulation', 'run 10 seconds',",
            "    'simulate 0.5s', 'end time 2s', 'over 5 seconds', 't=2s', 'for 2 sec'",
            "  • Physics keywords: 'transient', 'unsteady', 'pulsating', 'oscillating',",
            "    'time-dependent', 'time-varying', 'time stepping'",
            "  • Time evolution intent: 'how the flow develops', 'startup', 'watch the flow',",
            "    'initial transient', 'pressure wave', 'flow development'",
            "  • Or simply gives NO indication of wanting a steady solution (default to transient).",
            "",
            "When time_scheme='transient': set end_time = extracted seconds (e.g. '2s' → 2.0).",
            "If the user does not specify a duration, leave end_time null (system defaults to 5s).",
            "Note: end times above 10s are automatically capped to 10s by the system.",
            "For steady-state simulations the system caps iterations to 1000.",
            "",
            "=== SOLVER SELECTION — SINGLE-PHASE vs TWO-PHASE ===",
            "pimpleFoam is a SINGLE-PHASE solver. It assumes the entire domain is already",
            "filled with one fluid from t=0 and simulates transient velocity development.",
            "It cannot represent a moving fluid-front or filling process.",
            "",
            "For filling / moving-interface cases (fluid entering an empty or partially empty",
            "domain, fill front, air displacement, slug front) set time_scheme='transient'.",
            "Multiphase (VOF/filling) solvers are not currently supported.",
            "Use transient single-phase flow with appropriate BCs instead.",
            "",
            "=== CRITICAL PLANNING RULES ===",
            "- Plan EACH patch independently.",
            "- Output EXACTLY one plan item per mesh patch.",
            "- If mass flow rate or volumetric flow rate is given for an inlet:",
            "    hasMassFlowRate/hasVolumetricFlowRate = true",
            "    strategyU.bcFamily = flowRateInletVelocity",
            "    strategyU.selectionMode = derived_bc_family",
            "    Do NOT convert flow rate into a fixed velocity as the primary BC.",
            "    Do NOT set hasStaticPressure=true on the inlet when flow rate is the primary driver.",
            "    The inlet pressure strategy is always zeroGradient when flow rate is set.",
            "",
            "=== OPERATING PRESSURE / SYSTEM PRESSURE RULE ===",
            "- When the user mentions an 'operating pressure', 'system pressure', 'back pressure',",
            "  or gives a pressure value like '4 bar', '400 kPa', '2 atm' WITHOUT saying 'inlet pressure':",
            "    → assign it as staticPressure on the OUTLET patch, NOT the inlet.",
            "    → set hasStaticPressure=true and staticPressure=<value in Pa> on the OUTLET plan.",
            "    → do NOT put it on the inlet plan.",
            "- Conversions: 1 bar = 100000 Pa | 1 atm = 101325 Pa | 1 MPa = 1000000 Pa",
            "- If the user gives BOTH a mass flow rate AND a pressure value:",
            "    mass flow rate → inlet (hasMassFlowRate)",
            "    pressure value → outlet (hasStaticPressure)",
            "",
            "- If no-slip wall: strategyU.bcFamily = noSlip",
            "- If total pressure given: preserve as total, do not downgrade to static pressure.",
            "- If total temperature given: preserve as total, do not downgrade to static temperature.",
            "- If turbulence intensity given: strategyK.bcFamily = turbulentIntensityKineticEnergyInlet",
            "- frontAndBack / empty patches: role = frontAndBack, all strategies = not_needed.",
            "",
            "=== RETRIEVAL DOC RULES ===",
            "For each patch include the relevant markdown paths in retrievalDocs.",
            "Available paths (relative to bc_knowledge/):",
            "  _meta/common_combinations.md                              — always include for every patch",
            "  inlet/overview.md                                         — any inlet (always include)",
            "  inlet/velocity/flow_rate_inlet.md                        — inlet with mass/vol flow rate",
            "  inlet/turbulence/turbulent_intensity_kinetic_energy.md   — inlet turbulence intensity",
            "  outlet/overview.md                                        — any outlet (always include)",
            "  outlet/pressure/total_pressure.md                        — outlet total pressure",
            "  outlet/temperature/total_temperature.md                  — outlet total temperature",
            "  wall/velocity/no_slip.md                                  — no-slip wall",
        ]

        if request.previous_config:
            parts += ["", "=== PREVIOUS CONFIG (user is refining) ===",
                      json.dumps(request.previous_config, indent=2)]

        return "\n".join(parts)

    # ── Pass 2: Parallel patch agents ────────────────────────────────────────

    async def _stream_patch_agents_parallel(
        self,
        request: PrecheckRequest,
        boundary_plan: dict[str, Any],
        patch_specs: dict[str, dict[str, Any]],
        partial_boundary_state: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        semaphore = asyncio.Semaphore(_PATCH_CONCURRENCY)
        patches = boundary_plan.get("patches", [])

        tasks = [
            asyncio.create_task(
                self._run_single_patch_agent(
                    request=request,
                    patch_plan=patch_plan,
                    queue=queue,
                    semaphore=semaphore,
                    patch_specs=patch_specs,
                    partial_boundary_state=partial_boundary_state,
                )
            )
            for patch_plan in patches
        ]

        remaining = len(tasks)
        while remaining > 0:
            event = await queue.get()
            yield event
            if event["type"] == "patch_done":
                remaining -= 1

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for item in results:
            if isinstance(item, Exception):
                logger.warning(f"[Precheck] Patch task raised: {item}")
                yield {"type": "warning", "message": f"Patch task failed: {item}"}

    async def _run_single_patch_agent(
        self,
        request: PrecheckRequest,
        patch_plan: dict[str, Any],
        queue: asyncio.Queue,
        semaphore: asyncio.Semaphore,
        patch_specs: dict[str, dict[str, Any]],
        partial_boundary_state: dict[str, Any],
    ) -> None:
        patch_name = patch_plan.get("patchName", "unknown")
        patch_role = patch_plan.get("patchRole", "other")

        await queue.put({"type": "patch_start", "patchName": patch_name, "patchRole": patch_role})

        async with semaphore:
            # Load retrieval docs — always guarantee role-level overview files
            _role_overviews = {
                "inlet":  ["_meta/common_combinations.md", "inlet/overview.md"],
                "outlet": ["_meta/common_combinations.md", "outlet/overview.md"],
                "wall":   ["_meta/common_combinations.md"],
            }
            _guaranteed = _role_overviews.get(patch_role, ["_meta/common_combinations.md"])
            _planner_docs = patch_plan.get("retrievalDocs", [])
            # merge: guaranteed first, then any additional planner-selected docs, deduplicated
            retrieval_docs = list(dict.fromkeys(_guaranteed + _planner_docs))
            retrieved_markdown = self._load_retrieval_docs(retrieval_docs)
            await queue.put({
                "type": "patch_retrieval",
                "patchName": patch_name,
                "patchRole": patch_role,
                "docs": retrieval_docs,
            })

            prompt = self._build_patch_agent_prompt(
                patch_role=patch_role,
                patch_name=patch_name,
                patch_plan=patch_plan,
                retrieved_markdown=retrieved_markdown,
                request=request,
            )

            func_call_args: dict[str, Any] | None = None
            try:
                stream = await self.client.aio.models.generate_content_stream(
                    model=PRECHECK_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(include_thoughts=True),
                        tools=[PATCH_SPEC_TOOL_SCHEMA],
                        tool_config=types.ToolConfig(
                            function_calling_config=types.FunctionCallingConfig(
                                mode="ANY",
                                allowed_function_names=["submit_patch_spec"],
                            )
                        ),
                    ),
                )

                async for chunk in stream:
                    if not chunk.candidates:
                        continue
                    for part in chunk.candidates[0].content.parts:
                        if getattr(part, "thought", False) and part.text:
                            await queue.put({
                                "type": "patch_thought",
                                "patchName": patch_name,
                                "patchRole": patch_role,
                                "text": part.text,
                            })
                        elif getattr(part, "function_call", None) is not None:
                            func_call_args = dict(part.function_call.args)

            except Exception as e:
                logger.warning(f"[Precheck/patch] {patch_name} LLM call failed: {e}")
                await queue.put({
                    "type": "warning",
                    "patchName": patch_name,
                    "message": f"Patch agent failed: {e}",
                })
                await queue.put({
                    "type": "patch_done",
                    "patchName": patch_name,
                    "patchRole": patch_role,
                    "status": "failed",
                })
                return

            if not func_call_args:
                await queue.put({
                    "type": "warning",
                    "patchName": patch_name,
                    "message": "Patch agent returned no spec.",
                })
                await queue.put({
                    "type": "patch_done",
                    "patchName": patch_name,
                    "patchRole": patch_role,
                    "status": "failed",
                })
                return

            patch_specs[patch_name] = func_call_args
            boundary_fragment = self._patch_spec_args_to_fragment(func_call_args)
            partial_boundary_state[patch_name] = boundary_fragment

            await queue.put({
                "type": "patch_spec",
                "patchName": patch_name,
                "patchRole": patch_role,
                "data": func_call_args,
            })
            await queue.put({
                "type": "spec_partial",
                "patchName": patch_name,
                "data": {"boundaries": copy.deepcopy(partial_boundary_state)},
            })
            await queue.put({
                "type": "patch_done",
                "patchName": patch_name,
                "patchRole": patch_role,
                "status": "ok",
            })

    def _build_patch_agent_prompt(
        self,
        patch_role: str,
        patch_name: str,
        patch_plan: dict[str, Any],
        retrieved_markdown: str,
        request: PrecheckRequest,
    ) -> str:
        solver_context = self._format_solver_context(request)
        patch_plan_json = json.dumps(patch_plan, indent=2)
        _is_laminar = "laminar" in request.prompt.lower()

        base = f"""\
=== PATCH NAME ===
{patch_name}

=== PATCH PLAN ===
{patch_plan_json}

=== SOLVER CONTEXT ===
{solver_context}

=== RETRIEVED KNOWLEDGE ===
{retrieved_markdown}
"""

        if patch_role == "inlet":
            # Extract the concrete values from the plan so the agent cannot miss them
            _mfr   = patch_plan.get("massFlowRate")
            _vfr   = patch_plan.get("volumetricFlowRate")
            _has_mfr = bool(patch_plan.get("hasMassFlowRate") or _mfr)
            _has_vfr = bool(patch_plan.get("hasVolumetricFlowRate") or _vfr)
            _T_in  = patch_plan.get("staticTemperature")
            _turb_I = patch_plan.get("turbulenceIntensity")  # fraction, e.g. 0.05

            _flow_rate_block = ""
            if _has_mfr and _mfr is not None:
                _flow_rate_block = (
                    f"MASS FLOW RATE IS THE PRIMARY INLET DRIVER:\n"
                    f"  fieldU.bcType          = flowRateInletVelocity\n"
                    f"  fieldU.entryMassFlowRate = {_mfr}   ← copy this EXACT number\n"
                    f"  fieldP.bcType          = zeroGradient  ← NEVER fixedValue when mass flow is primary\n"
                    f"  fieldP.entryValue      = null\n"
                    f"  DO NOT also impose a static inlet pressure — that would over-constrain the problem.\n"
                    f"  DO NOT put {_mfr} in entryValue or entryValueVector — it MUST go in entryMassFlowRate."
                )
            elif _has_vfr and _vfr is not None:
                _flow_rate_block = (
                    f"VOLUMETRIC FLOW RATE IS THE PRIMARY INLET DRIVER:\n"
                    f"  fieldU.bcType                  = flowRateInletVelocity\n"
                    f"  fieldU.entryVolumetricFlowRate = {_vfr}   ← copy this EXACT number\n"
                    f"  fieldP.bcType                  = zeroGradient\n"
                    f"  DO NOT also impose a static inlet pressure."
                )

            _temp_block = ""
            if _T_in is not None:
                _temp_block = (
                    f"INLET TEMPERATURE:\n"
                    f"  fieldT.bcType    = fixedValue\n"
                    f"  fieldT.entryValue = {_T_in}   ← fluid temperature in K, NOT wall temperature"
                )

            _turb_block = ""
            if _turb_I is not None:
                _turb_block = (
                    f"TURBULENCE INTENSITY:\n"
                    f"  fieldK.bcType       = turbulentIntensityKineticEnergyInlet\n"
                    f"  fieldK.entryIntensity = {_turb_I}   ← fraction (e.g. 0.05 for 5%)\n"
                    f"  uiInputs: expose 'intensity' as the user-owned input."
                )

            _turb_section = ""
            if _is_laminar:
                _turb_section = (
                    "=== TURBULENCE — LAMINAR FLOW ===\n"
                    "This is a LAMINAR simulation. Do NOT set any turbulence fields.\n"
                    "Set fieldK, fieldOmega, fieldEpsilon, fieldNut, fieldMut all to enabled=false.\n"
                    "Omit turbulence completely — no k, omega, epsilon, nut, or mut values."
                )
            else:
                _turb_section = (
                    f"{_turb_block or '(no turbulence intensity detected — use fixedValue for k/omega with reasonable defaults)'}\n\n"
                    "=== CORE TURBULENCE RULES ===\n"
                    "- fieldNut.bcType = calculated, entryValue = 0.\n"
                    "- fieldOmega.bcType = fixedValue with a physically computed value.\n"
                    "- For omega: compute from k and hydraulic diameter using L = 0.07 * Dh, omega = sqrt(k) / (Cmu^0.25 * L)."
                )

            return f"""\
You are the INLET boundary-condition synthesis agent.
Generate a PatchSpec for ONE inlet patch only.
Return ONLY data matching the PatchSpec schema via submit_patch_spec.

{base}
=== EXTRACTED VALUES FROM PLANNER (USE THESE EXACT NUMBERS) ===
{_flow_rate_block or "(no flow rate detected — use fixedValue U if a velocity was given)"}

{_temp_block or "(no inlet temperature detected)"}

{_turb_section}

=== CORE INLET RULES ===
- When mass flow rate or volumetric flow rate is the primary driver:
    fieldP.bcType = zeroGradient — ALWAYS. NEVER fixedValue or totalPressure at inlet when flow rate is set.
    A staticPressure value in the plan is the OPERATING PRESSURE context, NOT an inlet BC — ignore it for fieldP.

=== STRICT DO NOTS ===
- Do NOT put the mass flow rate number inside entryValue or entryValueVector.
- Do NOT use fixedValue for fieldU when a flow rate is the primary requirement.
- Do NOT set fieldP to fixedValue or totalPressure when mass/vol flow rate is set.
- Do NOT produce specs for other patches.
"""

        if patch_role == "outlet":
            # Extract concrete pressure values from the plan
            _has_sp   = bool(patch_plan.get("hasStaticPressure") or patch_plan.get("staticPressure"))
            _sp       = patch_plan.get("staticPressure")
            _has_tp   = bool(patch_plan.get("hasTotalPressure") or patch_plan.get("totalPressure"))
            _tp       = patch_plan.get("totalPressure")
            _has_st   = bool(patch_plan.get("hasStaticTemperature") or patch_plan.get("staticTemperature"))
            _st       = patch_plan.get("staticTemperature")
            _is_comp  = "compressible" in self._format_solver_context(request)

            _p_block = ""
            if _has_tp and _tp is not None:
                _p_block = (
                    f"TOTAL PRESSURE IS THE OUTLET PRESSURE DRIVER:\n"
                    f"  fieldP.bcType  = totalPressure\n"
                    f"  fieldP.entryP0 = {_tp}   ← copy this EXACT number in Pa"
                )
            elif _has_sp and _sp is not None:
                _p_block = (
                    f"STATIC PRESSURE IS THE OUTLET PRESSURE DRIVER:\n"
                    f"  fieldP.bcType      = fixedValue\n"
                    f"  fieldP.entryValue  = {_sp}   ← copy this EXACT number in Pa"
                )
            else:
                _default_p = "101325.0" if _is_comp else "0"
                _p_block = (
                    f"NO OUTLET PRESSURE SPECIFIED — use safe default:\n"
                    f"  fieldP.bcType     = fixedValue\n"
                    f"  fieldP.entryValue = {_default_p}   "
                    f"({'absolute Pa for compressible' if _is_comp else 'gauge 0 for incompressible'})"
                )

            _T_block = ""
            if _has_st and _st is not None:
                _T_block = (
                    f"OUTLET TEMPERATURE:\n"
                    f"  fieldT.bcType    = fixedValue\n"
                    f"  fieldT.entryValue = {_st}   ← K"
                )

            _outlet_turb_rule = (
                "- Do NOT set any turbulence fields (k, omega, epsilon, nut) — this is a LAMINAR simulation."
                if _is_laminar else
                "- fieldK.bcType = zeroGradient, fieldOmega.bcType = zeroGradient, fieldNut.bcType = calculated."
            )

            return f"""\
You are the OUTLET boundary-condition synthesis agent.
Generate a PatchSpec for ONE outlet patch only.
Return ONLY data matching the PatchSpec schema via submit_patch_spec.

{base}
=== EXTRACTED VALUES FROM PLANNER (USE THESE EXACT NUMBERS) ===
{_p_block}

{_T_block or "(no outlet temperature — use zeroGradient for T)"}

=== CORE OUTLET RULES ===
- MVP assumes no backflow. Record this in assumptions.
- fieldU.bcType = zeroGradient — NEVER impose a fixed outlet velocity.
{_outlet_turb_rule}
- For T: zeroGradient unless a temperature value is listed above.

=== STRICT DO NOTS ===
- Do not impose a fixed outlet velocity unless explicitly required.
- Do not silently downgrade totalPressure to fixedValue static pressure.
- Do not produce specs for other patches.
"""

        if patch_role == "wall":
            _wall_turb_rules = (
                "- LAMINAR FLOW: Do NOT set any turbulence wall functions.\n"
                "  Set fieldK, fieldOmega, fieldEpsilon, fieldNut, fieldMut all to enabled=false.\n"
                "  Only fieldU (noSlip) and fieldP (zeroGradient) are needed."
                if _is_laminar else
                "- Wall turbulence (kOmegaSST model):\n"
                "    fieldK.bcType = kqRWallFunction\n"
                "    fieldOmega.bcType = omegaWallFunction\n"
                "    fieldNut.bcType = nutkWallFunction\n"
                "- For kEpsilon model:\n"
                "    fieldK.bcType = kqRWallFunction\n"
                "    fieldEpsilon.bcType = epsilonWallFunction\n"
                "    fieldNut.bcType = nutkWallFunction"
            )

            return f"""\
You are the WALL boundary-condition synthesis agent.
Generate a PatchSpec for ONE wall patch only.
Return ONLY data matching the PatchSpec schema via submit_patch_spec.

{base}
=== CORE WALL RULES ===
- fieldU.bcType = noSlip (default for stationary wall, unless user specifies moving wall).
- fieldP.bcType = zeroGradient.
- If staticTemperature is present WITH a specific numeric value: fieldT.bcType = fixedValue, fieldT.entryValue = T in K.
- If user says "isothermal" WITHOUT a specific temperature value, OR wall temperature is not specified: fieldT.bcType = zeroGradient (adiabatic — no heat flux). Do NOT set fixedValue with a null value.
- If no temperature at all: fieldT either not_needed or zeroGradient depending on heat transfer flag.
{_wall_turb_rules}

=== STRICT DO NOTS ===
- Do not invent a wall temperature if none is specified.
- Do not choose slip or partialSlip unless explicitly requested.
- Do not produce specs for other patches.
"""

        if patch_role == "frontAndBack":
            return f"""\
You are the FRONT-AND-BACK (empty) patch synthesis agent.
Generate a PatchSpec for ONE empty/frontAndBack patch only.
Return ONLY data matching the PatchSpec schema via submit_patch_spec.

{base}
=== RULES ===
- This is a 2D mesh empty patch. ALL fields get bcType = empty.
- fieldU.bcType = empty, fieldP.bcType = empty, fieldT.bcType = empty (if heat transfer).
- fieldK.bcType = empty, fieldOmega.bcType = empty, fieldNut.bcType = empty.
- confidence = 1.0
- No uiInputs needed.
"""

        # symmetry / other
        return f"""\
You are the GENERIC PATCH synthesis agent.
Generate a PatchSpec for ONE patch only.
Return ONLY data matching the PatchSpec schema via submit_patch_spec.

{base}
=== RULES ===
- If patchRole = symmetry: all fields get bcType = symmetry.
- Otherwise use reasonable defaults for the detected role.
- Do not produce specs for other patches.
"""

    # ── Pass 3: Merge ─────────────────────────────────────────────────────────

    def _merge_patch_specs_into_response(
        self,
        request: PrecheckRequest,
        boundary_plan: dict[str, Any],
        patch_specs: dict[str, dict[str, Any]],
    ) -> PrecheckResponse:
        """Convert all PatchSpec dicts into a PrecheckResponse."""
        mesh = request.get_mesh()

        # Infer global physics from planner patches
        has_heat = False
        is_compressible = False
        is_transient = False
        flow_regime = "turbulent"
        turb_model = "kOmegaSST"

        for p in boundary_plan.get("patches", []):
            # Only count as heat transfer if a concrete temperature VALUE was provided.
            # hasStaticTemperature=True without an actual number (e.g. user says "isothermal"
            # without specifying a value) is NOT heat transfer — it means adiabatic wall.
            if (p.get("hasStaticTemperature") and p.get("staticTemperature") is not None) or \
               (p.get("hasTotalTemperature") and p.get("totalTemperature") is not None):
                has_heat = True

        prompt_lower = request.prompt.lower()
        is_compressible = any(kw in prompt_lower for kw in
                              ("mach", "supersonic", "transonic", "compressible") +
                              CRYOGENIC_KEYWORDS)
        # Default to transient — steady only when the LLM explicitly says so
        is_transient = boundary_plan.get("time_scheme", "transient") != "steady"
        # Laminar detection: prompt keyword OR explicit LLM field from boundary planner
        if "laminar" in prompt_lower or boundary_plan.get("flow_regime") == "laminar":
            flow_regime = "laminar"
            turb_model = "laminar"

        # Fluid preset
        fluid = self._detect_fluid(prompt_lower)

        # Build patch-name → planner plan lookup for fallback value injection
        _plan_by_name: dict[str, dict[str, Any]] = {
            p.get("patchName", ""): p
            for p in boundary_plan.get("patches", [])
        }

        # Build boundary_conditions dict from patch specs
        boundary_conditions: dict[str, PatchBoundaryCondition] = {}
        for patch_name, spec in patch_specs.items():
            planner_plan = _plan_by_name.get(patch_name, {})
            bc = self._patch_spec_args_to_patch_bc(spec, has_heat, flow_regime, planner_plan)
            boundary_conditions[patch_name] = bc

        # Turbulence: estimate from first inlet patch (only meaningful for turbulent flow)
        turb_intensity: float | None = None
        hydraulic_diam: float | None = None
        if flow_regime == "turbulent":
            turb_intensity = 5.0
            hydraulic_diam = 0.1
            for p in boundary_plan.get("patches", []):
                if p.get("patchRole") == "inlet":
                    if p.get("hasTurbulenceIntensity") and p.get("turbulenceIntensity"):
                        turb_intensity = float(p["turbulenceIntensity"]) * 100
                    break

        compressibility_str = "compressible" if is_compressible else "incompressible"
        time_scheme = "transient" if is_transient else "steady"

        # ── Phase-change detection ────────────────────────────────────────────
        _PHASE_CHANGE_KEYWORDS = (
            "boiling", "boil", "nucleate boiling", "film boiling", "pool boiling",
            "cavitation", "cavitat", "bubble collapse", "vapour", "vapor pocket",
            "evaporation", "evaporat", "condensation", "condens",
            "flash evaporation", "flash boiling", "two-phase heat",
            "phase change", "phase-change", "refrigerant cycle",
            "steam generation", "heat pipe",
        )
        phase_change_detected = any(kw in prompt_lower for kw in _PHASE_CHANGE_KEYWORDS)

        # Also detect: cryogenic liquid + significant heating (T_wall >> boiling point)
        # Check BC temperatures: if there's a hot wall temperature alongside a cryogenic fluid
        _bc_temps: list[float] = []
        for spec in patch_specs.values():
            for fld in (spec.get("foamFields") or []):
                if fld.get("fieldName") == "T":
                    _ev = fld.get("entryValue")
                    try:
                        _bc_temps.append(float(_ev))
                    except (TypeError, ValueError):
                        pass
        _is_cryogenic_fluid = any(kw in prompt_lower for kw in CRYOGENIC_KEYWORDS)
        if _is_cryogenic_fluid and _bc_temps:
            _t_max_bc = max(_bc_temps)
            _t_min_bc = min(_bc_temps)
            # LN2 boils at 77K, LH2 at 20K, LOX at 90K — rough boiling threshold: min_BC + 50K
            if _t_max_bc > _t_min_bc + 50.0:
                phase_change_detected = True

        # ── OpenFOAM solver name ──────────────────────────────────────────────
        _multiphase = any(kw in prompt_lower for kw in
                          ("two-phase", "two phase", "multiphase", "interface", "vof",
                           "free surface", "water-air", "oil-water", "liquid-gas"))

        # Filling / moving fluid-front keywords: pimpleFoam assumes a domain already full
        # of one fluid — it cannot model a fill front or air displacement.
        _FILLING_KEYWORDS = (
            "fill", "filling", "empty pipe", "empty domain", "initially empty",
            "fluid entering", "fluid front", "fill front", "slug front",
            "flood", "priming", "invasion", "invading",
        )
        _filling = any(kw in prompt_lower for kw in _FILLING_KEYWORDS)

        if is_compressible or has_heat:
            openfoam_solver = "rhoPimpleFoam" if is_transient else "rhoSimpleFoam"
        elif not is_transient:
            openfoam_solver = "simpleFoam"
        else:
            openfoam_solver = "pimpleFoam"

        # Use LLM-provided end_time when transient; default to 5s if not specified
        # Cap transient simulations to 10s and steady-state to 1000 iterations.
        _MAX_TRANSIENT_END_TIME = 10.0
        _MAX_STEADY_ITERATIONS = 1000

        _end_time: float | None = boundary_plan.get("end_time") if is_transient else None
        _max_iterations: int | None = None
        _end_time_capped_warning: str | None = None

        if is_transient:
            _end_time = float(_end_time) if _end_time else 5.0
            if _end_time > _MAX_TRANSIENT_END_TIME:
                _end_time_capped_warning = (
                    f"Requested simulation duration ({_end_time:.4g} s) exceeds the "
                    f"{_MAX_TRANSIENT_END_TIME:.4g} s cap. End time set to "
                    f"{_MAX_TRANSIENT_END_TIME:.4g} s."
                )
                _end_time = _MAX_TRANSIENT_END_TIME
            _delta_t = max(1e-4, _end_time / 1000.0)
            _write_interval = max(_delta_t, _end_time / 100.0)
        else:
            _end_time = None
            _delta_t = None
            _write_interval = None
            _max_iterations = _MAX_STEADY_ITERATIONS

        suggested_config = SuggestedConfig(
            case_type=self._detect_case_type(prompt_lower),
            flow_regime=flow_regime,
            time_scheme=time_scheme,
            compressibility=compressibility_str,
            enable_heat_transfer=has_heat,
            gravity=any(w in prompt_lower for w in ("gravity", "buoyancy", "natural convection")),
            openfoam_solver=openfoam_solver,
            phase_change_detected=phase_change_detected,
            # solver: only carry temporal user-intent values from the prompt.
            # algorithm / max_iterations / convergence_criteria are NOT set here —
            # they are determined by the run-time normalizer/linter and sent back
            # to the frontend via the `simulation_config_ready` event on ws/run.
            solver=SolverSettings(
                end_time=_end_time,
                delta_t=_delta_t,
                write_interval=_write_interval,
                max_iterations=_max_iterations,
            ),
            fluid=fluid,
            turbulence=TurbulenceSettings(
                model=turb_model,
                turbulence_intensity=turb_intensity,
                turbulence_length_scale=0.07 * hydraulic_diam if hydraulic_diam is not None else None,
                hydraulic_diameter=hydraulic_diam,
                wall_functions=flow_regime == "turbulent",
            ),
            boundary_conditions=boundary_conditions,
        )

        # Collect all assumptions / warnings from patch specs
        all_assumptions: list[str] = []
        all_warnings: list[str] = []
        for spec in patch_specs.values():
            all_assumptions.extend(spec.get("assumptions") or [])
            all_warnings.extend(spec.get("warnings") or [])

        if _filling and not _multiphase and not phase_change_detected:
            all_warnings.append(
                f"Filling / moving fluid-front detected. pimpleFoam assumes the domain is "
                f"already full of one fluid and cannot model a fill front or air displacement. "
                f"Using '{openfoam_solver}' (VOF two-phase solver) which tracks the interface "
                "between the incoming fluid and the air it displaces."
            )

        if _end_time_capped_warning:
            all_warnings.append(_end_time_capped_warning)

        if phase_change_detected:
            all_warnings.append(
                f"Phase-change physics detected (boiling / evaporation / cavitation). "
                f"The selected solver '{openfoam_solver}' does not model phase change — "
                "this is a best-effort approximation using a two-phase compressible solver. "
                "For accurate boiling/cavitation results a dedicated phase-change model is required."
            )

        confidence = _norm_conf(
            sum((spec.get("confidence") or 0.8) for spec in patch_specs.values()) /
            max(len(patch_specs), 1)
        )

        _phase_change_note = " ⚠ Phase-change detected." if phase_change_detected else ""
        response = PrecheckResponse(
            success=True,
            confidence=confidence,
            message=(
                f"Analyzed {len(patch_specs)} patches — {flow_regime} {compressibility_str} flow. "
                f"Solver: {openfoam_solver}.{_phase_change_note}"
            ),
            suggested_config=suggested_config,
            boundary_hints=self._build_boundary_hints(boundary_conditions),
            kpi_targets=None,
            interpretation=Interpretation(
                summary=f"Multi-patch boundary analysis for {suggested_config.case_type.replace('_', ' ')}.",
                simulation_type=suggested_config.case_type.replace("_", " ").title(),
                key_physics=self._detect_key_physics(prompt_lower, has_heat, flow_regime),
                assumptions=all_assumptions[:10],
                clarifications=None,
            ),
            confidence_scores=ConfidenceScores(
                overall=confidence,
                flow_regime=0.85,
                boundary_conditions=confidence,
                physics_settings=0.8,
            ),
            next_step=2 if mesh else 1,
            should_show_mesh_viewer=mesh is not None,
            warnings=all_warnings or None,
        )

        print("=" * 70)
        print("[Precheck] Final merged response:")
        print(json.dumps(response.model_dump(by_alias=True), indent=2, default=str))
        print("=" * 70)
        return response

    def _patch_spec_args_to_patch_bc(
        self,
        spec: dict[str, Any],
        has_heat: bool,
        flow_regime: str,
        planner_plan: dict[str, Any] | None = None,
    ) -> PatchBoundaryCondition:
        """Convert raw patch spec args dict → PatchBoundaryCondition.

        planner_plan: the corresponding BoundaryPlanPatch dict — used as a fallback
        to recover values the patch agent may have forgotten to set (e.g. massFlowRate).
        """
        role = spec.get("patchRole", "wall")
        turb = flow_regime == "turbulent"
        plan = planner_plan or {}

        def _field(key: str) -> FieldBC | None:
            f = spec.get(key)
            if not f or not f.get("enabled", True):
                return None
            bc_type = f.get("bcType") or "zeroGradient"
            # Determine value — check all entry types in priority order.
            # For flowRateInletVelocity, entryValueVector is the OpenFOAM placeholder
            # [0,0,0] — NOT the flow rate. Scalar flow-rate entries take priority.
            value: Any = None
            if bc_type == "flowRateInletVelocity":
                if f.get("entryMassFlowRate") is not None:
                    value = float(f["entryMassFlowRate"])
                elif f.get("entryVolumetricFlowRate") is not None:
                    value = float(f["entryVolumetricFlowRate"])
                elif f.get("entryValue") is not None:
                    value = float(f["entryValue"])
            else:
                if f.get("entryValueVector"):
                    value = list(f["entryValueVector"])
                elif f.get("entryValue") is not None:
                    value = float(f["entryValue"])
                elif f.get("entryMassFlowRate") is not None:
                    value = float(f["entryMassFlowRate"])
                elif f.get("entryVolumetricFlowRate") is not None:
                    value = float(f["entryVolumetricFlowRate"])
                elif f.get("entryP0") is not None:
                    value = float(f["entryP0"])
                elif f.get("entryT0") is not None:
                    value = float(f["entryT0"])

            # recover mass/volumetric flow rate from planner plan
            # The patch agent sometimes forgets to set entryMassFlowRate even
            # when it correctly selects flowRateInletVelocity.
            if value is None and bc_type == "flowRateInletVelocity":
                if plan.get("massFlowRate") is not None:
                    value = float(plan["massFlowRate"])
                elif plan.get("volumetricFlowRate") is not None:
                    value = float(plan["volumetricFlowRate"])

            # Preserve the specific flow-rate key so downstream code knows
            # whether it's massFlowRate or volumetricFlowRate (not just a generic "value").
            extra_entries: dict[str, Any] | None = None
            if bc_type == "flowRateInletVelocity" and value is not None:
                if f.get("entryMassFlowRate") is not None or plan.get("hasMassFlowRate"):
                    extra_entries = {"massFlowRate": value}
                elif f.get("entryVolumetricFlowRate") is not None or plan.get("hasVolumetricFlowRate"):
                    extra_entries = {"volumetricFlowRate": value}
                else:
                    extra_entries = {"massFlowRate": value}

            return FieldBC(type=bc_type, value=value, entries=extra_entries)

        # ── Inlet pressure override: when mass/vol flow is primary, force zeroGradient ──
        # The planner may record staticPressure as operating-pressure context.
        # It must NEVER become a fixedValue inlet BC when flow rate is primary.
        _is_flow_rate_inlet = (
            role == "inlet"
            and (plan.get("hasMassFlowRate") or plan.get("hasVolumetricFlowRate"))
        )

        def _field_p_inlet() -> FieldBC | None:
            """For inlets with a flow-rate primary driver, always return zeroGradient."""
            if _is_flow_rate_inlet:
                return FieldBC(type="zeroGradient", value=None)
            return _field("fieldP")

        # frontAndBack is OpenFOAM's 2D empty patch — normalise to "empty"
        _ROLE_TO_PATCH_CLASS = {
            "frontAndBack": "empty",
            "other":        "wall",   # safe fallback for unrecognised roles
        }
        patch_class = _ROLE_TO_PATCH_CLASS.get(role, role)

        # Build T field — post-process: wall fixedValue with null value → zeroGradient (adiabatic).
        # This happens when the user says "isothermal" without specifying a temperature.
        _T_field = _field("fieldT") if has_heat else None
        if _T_field is not None and role == "wall":
            if _T_field.type == "fixedValue" and _T_field.value is None:
                _T_field = FieldBC(type="zeroGradient", value=None)

        return PatchBoundaryCondition(
            patch_class=patch_class,
            confidence=_norm_conf(spec.get("confidence", 0.8)),
            U=_field("fieldU"),
            p=_field_p_inlet() if _is_flow_rate_inlet else _field("fieldP"),
            T=_T_field,
            k=_field("fieldK") if turb else None,
            epsilon=_field("fieldEpsilon") if turb else None,
            omega=_field("fieldOmega") if turb else None,
            nut=_field("fieldNut") if turb else None,
        )

    def _patch_spec_args_to_fragment(self, spec: dict[str, Any]) -> dict[str, Any]:
        """Lightweight fragment for spec_partial events (no Pydantic)."""
        return {
            "patchName": spec.get("patchName"),
            "patchRole": spec.get("patchRole"),
            "foamFields": {
                k: {"bcType": v.get("bcType"), "enabled": v.get("enabled")}
                for k, v in spec.items()
                if k.startswith("field") and isinstance(v, dict)
            },
            "uiInputs":  spec.get("uiInputs", []),
            "uiDerived": spec.get("uiDerived", []),
        }

    # ── Pass 4: Review ────────────────────────────────────────────────────────

    async def _llm_review(
        self, request: PrecheckRequest, response: PrecheckResponse
    ) -> AsyncIterator[dict[str, Any]]:
        """Streamed narrative review of the merged spec."""
        spec_json = json.dumps(response.model_dump(by_alias=True), indent=2, default=str)
        flow_regime = response.suggested_config.flow_regime  # "laminar" or "turbulent"
        is_laminar = flow_regime == "laminar"

        # Section 2 and summary table columns change depending on the flow regime
        if is_laminar:
            turbulence_section = """\
2. **Flow regime — laminar** — confirm that the flow is laminar (state the Reynolds number
   if it can be derived from the given data) and explain why no turbulence model is needed.
   No $k$, $\\omega$, $\\varepsilon$, $\\nu_t$, or $\\mu_t$ fields are required for laminar flow.
   Confirm these fields are correctly absent from the spec."""
            summary_table_header = (
                "| Patch | Class | `U` | `p` | `T` |\n"
                "|-------|-------|-----|-----|-----|"
            )
        else:
            turbulence_section = """\
2. **Turbulence** — narrate the derivation ($I=0.05$, $C_\\mu=0.09$, $L=0.07 D_h$):
   $$k = 1.5(U I)^2, \\quad \\omega = \\frac{{\\sqrt{{k}}}}{{C_\\mu^{{0.25}} L}}, \\quad \\varepsilon = C_\\mu^{{0.75}}\\frac{{k^{{1.5}}}}{{L}}$$
   State each result to 4 sig. fig."""
            summary_table_header = (
                "| Patch | Class | `U` | `p` | `T` | `k` / $\\omega$ |\n"
                "|-------|-------|-----|-----|-----|----------------|"
            )

        review_prompt = f"""\
You are a CFD expert reviewing an OpenFOAM configuration for physical correctness.

Write your assessment as plain flowing prose — like a colleague explaining the physics.
Use backticks for identifiers (`inlet`, `fixedValue`, `kOmegaSST`),
inline LaTeX for symbols ($k$, $\\omega$, $\\rho$), and display math ($$...$$)
on its own line for any formula with a numeric substitution.

=== USER REQUEST ===
{request.prompt}

=== GENERATED SPEC ===
{spec_json}

=== FLOW REGIME ===
{flow_regime.upper()} — {"No turbulence fields (k, omega, epsilon, nut, mut) are needed or expected." if is_laminar else "Turbulence model is active — compute and verify k, omega, epsilon values."}

=== WHAT TO COVER ===

1. **Fluid & velocity** — identify the fluid; if a mass flow rate was given, show
   the velocity derivation:
   $$A = \\pi\\left(\\frac{{D}}{{2}}\\right)^2, \\quad U = \\frac{{\\dot{{m}}}}{{\\rho A}}$$
   State the result to 4 sig. fig.

{turbulence_section}

3. **Boundary conditions** — one paragraph per patch; describe what each patch does
   physically and confirm the BC types are correct.{"  For laminar flow, confirm that no turbulence wall functions or inlet turbulence values are set." if is_laminar else ""}

4. **Summary** — one paragraph recap, then a markdown table of the final BC values:

{summary_table_header}
"""

        yield {
            "type": "review_phase_start",
            "phase": "review",
            "title": "Physics review",
            "description": "Reviewing boundary conditions, turbulence parameters, and fluid properties...",
        }
        print(f"[Precheck] → review (model={PRECHECK_MODEL})...", flush=True)

        try:
            stream = await self.client.aio.models.generate_content_stream(
                model=PRECHECK_MODEL,
                contents=review_prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(include_thoughts=True),
                ),
            )

            text_parts: list[str] = []
            async for chunk in stream:
                if not chunk.candidates:
                    continue
                for part in chunk.candidates[0].content.parts:
                    if getattr(part, "thought", False) and part.text:
                        yield {"type": "thought", "text": part.text}
                    elif getattr(part, "text", None):
                        text_parts.append(part.text)

            yield {"type": "review_phase_done", "phase": "review", "message": "Review complete."}

            summary = "".join(text_parts).strip()
            if summary:
                yield {
                    "type": "review_item",
                    "status": "ok",
                    "patch": None,
                    "field": "summary",
                    "label": "Configuration review",
                    "detail": summary,
                }

        except Exception as e:
            logger.warning(f"[Precheck] Review failed: {e}")
            yield {
                "type": "review_item",
                "status": "warning",
                "patch": None,
                "field": "summary",
                "label": "Review unavailable",
                "detail": f"Could not complete the review: `{e}`",
            }

        yield {"type": "review_done"}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _format_mesh_summary(self, mesh) -> str:
        if not mesh:
            return "(no mesh)"
        lines = [f"File: {mesh.file_name} | Cells: {mesh.check_mesh.cells:,}",
                 "Patches:"]
        for p in mesh.patches:
            lines.append(f"  - {p.name}  (type: {p.type}, nCells: {p.n_cells})")
        return "\n".join(lines)

    def _format_solver_context(self, request: PrecheckRequest) -> str:
        prompt_lower = request.prompt.lower()
        is_compressible = any(kw in prompt_lower for kw in
                              ("mach", "supersonic", "compressible") + CRYOGENIC_KEYWORDS)
        is_transient = any(w in prompt_lower for w in
                           ("transient", "unsteady", "pulsating"))
        is_laminar = "laminar" in prompt_lower
        flow_regime = "laminar" if is_laminar else "turbulent"
        turb_note = (
            "laminar — NO turbulence fields (k, omega, epsilon, nut, mut) are needed or generated"
            if is_laminar else
            "turbulent (kOmegaSST default)"
        )
        return (
            f"compressibility={'compressible' if is_compressible else 'incompressible'}, "
            f"time={'transient' if is_transient else 'steady (preliminary — LLM decides)'}, "
            f"flow_regime={flow_regime}, "
            f"turbulence={turb_note}, "
            f"MVP: inlet/outlet/wall/frontAndBack, no backflow."
        )

    def _load_retrieval_docs(self, paths: list[str]) -> str:
        chunks: list[str] = []
        for rel_path in paths:
            full_path = BC_KNOWLEDGE_DIR / rel_path
            try:
                content = full_path.read_text(encoding="utf-8")
                chunks.append(f"\n--- {rel_path} ---\n{content}\n")
            except FileNotFoundError:
                chunks.append(f"\n--- {rel_path} ---\n[Missing retrieval file]\n")
        return "\n".join(chunks) if chunks else "(no retrieval docs)"

    def _detect_fluid(self, prompt_lower: str) -> FluidProperties:
        for kw, key in [
            (("lng", "liquefied natural gas"), "lng"),
            (("helium", "lhe", "liquid helium"), "helium"),
            (("ln2", "liquid nitrogen", "nitrogen"), "ln2"),
            (("oil", "lubricant"), "oil"),
            (("water", "hydraulic"), "water"),
        ]:
            if any(k in prompt_lower for k in kw):
                return FLUID_PRESETS[key]
        return FLUID_PRESETS["air"]

    def _detect_case_type(self, prompt_lower: str) -> str:
        if any(w in prompt_lower for w in ("pipe", "duct", "channel", "tube")):
            return "internal_pipe_flow"
        if any(w in prompt_lower for w in ("external", "wind", "airfoil", "wing")):
            return "external_aero"
        if any(w in prompt_lower for w in ("heat exchanger", "exchanger")):
            return "heat_exchanger"
        if any(w in prompt_lower for w in ("mixing", "mixer")):
            return "mixing"
        return "general"

    def _detect_key_physics(
        self, prompt_lower: str, has_heat: bool, flow_regime: str
    ) -> list[str]:
        physics = [flow_regime]
        if has_heat:
            physics.append("heat transfer")
        if any(kw in prompt_lower for kw in CRYOGENIC_KEYWORDS):
            physics.append("cryogenic fluid")
        if any(w in prompt_lower for w in ("mach", "supersonic", "compressible")):
            physics.append("compressibility")
        if any(w in prompt_lower for w in ("gravity", "buoyancy")):
            physics.append("buoyancy")
        return physics

    def _build_boundary_hints(
        self, boundary_conditions: dict[str, PatchBoundaryCondition]
    ) -> dict[str, BoundaryHint]:
        hints: dict[str, BoundaryHint] = {}
        for patch_name, bc in boundary_conditions.items():
            hints[patch_name] = BoundaryHint(
                suggested_type=bc.patch_class if bc.patch_class != "empty" else "wall",
                velocity=VelocityBC(
                    type=bc.U.type,
                    value=bc.U.value if isinstance(bc.U.value, list) else None,
                    magnitude=bc.U.value if isinstance(bc.U.value, (int, float)) else None,
                ) if bc.U else None,
                pressure=PressureBC(
                    type=bc.p.type,
                    value=bc.p.value if isinstance(bc.p.value, (int, float)) else None,
                ) if bc.p else None,
                temperature=TemperatureBC(
                    type=bc.T.type,
                    value=bc.T.value if isinstance(bc.T.value, (int, float)) else None,
                ) if bc.T else None,
                confidence=bc.confidence,
                reasoning=f"Classified as {bc.patch_class}",
            )
        return hints

    # ── Fallback / error helpers ──────────────────────────────────────────────

    def _create_friendly_error_response(self, message: str) -> PrecheckResponse:
        return PrecheckResponse(
            success=False,
            confidence=0.0,
            message=message,
            suggested_config=SuggestedConfig(
                case_type="general",
                flow_regime="turbulent",
                time_scheme="steady",
                compressibility="incompressible",
                enable_heat_transfer=False,
                gravity=False,
                solver=SolverSettings(),
                fluid=FLUID_PRESETS["air"],
                turbulence=TurbulenceSettings(model="kOmegaSST"),
                boundary_conditions={},
            ),
            interpretation=Interpretation(
                summary=message,
                simulation_type="Unknown",
                key_physics=[],
                assumptions=[],
            ),
            confidence_scores=ConfidenceScores(
                overall=0.0, flow_regime=0.0,
                boundary_conditions=0.0, physics_settings=0.0,
            ),
            errors=[message],
        )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_precheck_service: PrecheckService | None = None


def get_precheck_service() -> PrecheckService:
    global _precheck_service
    if _precheck_service is None:
        _precheck_service = PrecheckService()
    return _precheck_service
