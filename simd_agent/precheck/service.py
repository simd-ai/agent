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
import random
from typing import Any, AsyncIterator

from simd_agent.llm import get_provider
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
    CONVERSATION_TOKEN_LIMIT,
    # tool schemas
    BOUNDARY_PLAN_TOOL_SCHEMA, PATCH_SPEC_TOOL_SCHEMA,
    READY_TOOL_SCHEMA,
    BC_KNOWLEDGE_DIR,
)

logger = logging.getLogger(__name__)

# Max concurrent patch-agent LLM calls
_PATCH_CONCURRENCY = 4

# Retry constants for LLM calls (provider.is_retryable_error classifies errors)
_MAX_LLM_RETRIES = 3
_BASE_RETRY_DELAY = 2.0  # seconds


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
        self._provider = get_provider()

    @property
    def client(self):
        return self._provider.client

    @property
    def types(self):
        return self._provider.types

    @property
    def model(self) -> str:
        """Lighter / faster model — chat and summarization."""
        return self._provider.models["default"]

    @property
    def super_model(self) -> str:
        """Heavier / higher-capacity model — planner, patch agents, review."""
        return self._provider.models["super"]

    # ── Conversation system prompt ──────────────────────────────────────────

    CONVERSATION_SYSTEM_PROMPT = """\
You are **SIMD Agent**, a senior CFD (Computational Fluid Dynamics) engineer \
and simulation assistant built by SIMD.

## Your Role
- Chat naturally with the user about CFD simulations
- Help them describe what they want to simulate
- Explain CFD concepts in simple terms when asked
- Guide them through providing the details you need

## What You Need for a Simulation
1. **Mesh file** (.msh) — defines the geometry (patches: inlet, outlet, walls)
2. **Simulation description** — what to simulate (fluid, flow speed, physics)

## Conversation Guidelines
- Be concise, helpful, and friendly — no long walls of text
- Use markdown (bold, bullets) for clarity when explaining concepts
- If the user greets you, greet back and ask what they would like to simulate
- Ask clarifying questions naturally: What fluid? What flow speed? Any heat transfer?
- When the user needs a mesh, tell them to click **"Try a sample"** below the input \
or upload their own .msh file via **"Add mesh"**
- Do NOT use emoji

## Readiness
When the user has uploaded a mesh AND described enough about what they want to \
simulate, call the `signal_ready_to_analyze` tool with a summary of the simulation.

**Do NOT call signal_ready_to_analyze if:**
- No mesh is uploaded yet — ask the user to upload a mesh or try a sample mesh.
- An analysis has **already been completed** (see Current State below). \
The user is just asking follow-up questions — answer them directly. \
Only call signal_ready_to_analyze again if the user **explicitly** asks to \
re-run the analysis, change the simulation setup, or start a new simulation.

## Current State
{context}
"""

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

    def _should_use_conversation_mode(self, request: PrecheckRequest) -> bool:
        """Decide whether to route to conversational chat or full analysis."""
        # Explicit analysis confirmation → always analyze
        if request.confirm_analysis:
            return False
        # Everything else goes through conversation first so the user sees
        # the plan card and clicks "Start analysis" before the full pipeline.
        return True

    async def analyze_stream(self, request: PrecheckRequest) -> AsyncIterator[dict[str, Any]]:
        """Async generator for streaming precheck over WebSocket.

        Routes to conversation mode when the user is chatting (no mesh or
        ongoing multi-turn conversation), or to the full 4-pass analysis
        pipeline when there is enough info + mesh.

        Analysis event sequence:
            start → planner → patches → merge → spec → review → done

        Conversation event sequence:
            start → chat_token (0-N) → [ready_to_analyze] → chat_done → done
        """
        print(f"[Precheck] analyze_stream — prompt={request.prompt[:60]!r}", flush=True)

        # ── Conversation mode ────────────────────────────────────────────────
        if self._should_use_conversation_mode(request):
            async for event in self._stream_conversation(request):
                yield event
            return

        # ── Full analysis mode ───────────────────────────────────────────────
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
            result = await self._merge_patch_specs_into_response(
                request=request,
                boundary_plan=planner_result,
                patch_specs=patch_specs,
            )

            # ── Pass 3b: Physics coherence check ─────────────────────────────
            mesh = request.get_mesh()
            result = self._physics_coherence_check(result, mesh)

            _physics_warnings = []
            _re = result.suggested_config.reynolds_number
            if _re is not None:
                _physics_warnings.append({
                    "type": "info",
                    "message": (
                        f"Reynolds number: {_re:.0f} "
                        f"({result.suggested_config.flow_regime} regime)"
                    ),
                })
            # Collect any regime-mismatch warnings added by the coherence check
            for w in (result.warnings or []):
                if "Reynolds" in w or "regime" in w or "Regime" in w:
                    _physics_warnings.append({"type": "warning", "message": w})

            if _physics_warnings:
                yield {"type": "physics_check", "warnings": _physics_warnings}

            yield {"type": "spec", "data": result.model_dump(by_alias=True)}

            # ── Pass 4: Review ────────────────────────────────────────────────
            yield {"type": "review_start"}
            async for event in self._llm_review(request, result):
                yield event

        except Exception as e:
            logger.exception(f"[Precheck] Streaming failed: {e}")
            if self._provider.is_retryable_error(e):
                yield {
                    "type": "error",
                    "message": (
                        "The AI model is temporarily unavailable due to high demand. "
                        "Please try again in a moment."
                    ),
                }
            else:
                yield {"type": "error", "message": str(e)}
        finally:
            yield {"type": "done"}

    # ── Pass 1: Boundary planner ─────────────────────────────────────────────

    async def _stream_boundary_planner(
        self, request: PrecheckRequest
    ) -> AsyncIterator[dict[str, Any]]:
        prompt = self._build_boundary_planner_prompt(request)
        print(f"[Precheck/planner] prompt={len(prompt)} chars → {self.super_model}", flush=True)

        config = self.types.GenerateContentConfig(
            thinking_config=self.types.ThinkingConfig(include_thoughts=True),
            tools=[BOUNDARY_PLAN_TOOL_SCHEMA],
            tool_config=self.types.ToolConfig(
                function_calling_config=self.types.FunctionCallingConfig(
                    mode="ANY",
                    allowed_function_names=["submit_boundary_plan"],
                )
            ),
        )

        last_error: Exception | None = None
        for attempt in range(_MAX_LLM_RETRIES):
            try:
                func_call_args: dict[str, Any] | None = None

                stream = await self.client.aio.models.generate_content_stream(
                    model=self.super_model,
                    contents=prompt,
                    config=config,
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
                return  # success

            except Exception as e:
                last_error = e
                if not self._provider.is_retryable_error(e) or attempt == _MAX_LLM_RETRIES - 1:
                    raise
                delay = _BASE_RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(
                    "[Precheck/planner] Retryable error (attempt %d/%d), "
                    "retrying in %.1fs: %s",
                    attempt + 1, _MAX_LLM_RETRIES, delay, e,
                )
                await asyncio.sleep(delay)

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
            "DEFAULT: time_scheme='steady'. Steady-state is the standard approach for most CFD",
            "simulations — it is faster, more robust, and gives the converged result directly.",
            "Most users asking 'simulate flow in a pipe' or 'simulate airflow over X' want the",
            "steady-state solution, not a time-resolved animation.",
            "",
            "Use time_scheme='steady' when:",
            "  • The user gives NO indication of wanting time evolution (this is the default).",
            "  • The user explicitly requests steady-state: 'steady-state', 'steady flow',",
            "    'RANS steady', 'converged solution', 'mean flow'.",
            "  • Constant boundary conditions with no mention of time-varying phenomena.",
            "  • Simple flow queries: 'simulate water in a pipe', 'airflow over a wing',",
            "    'flow at 1 m/s inlet velocity', etc.",
            "",
            "Use time_scheme='transient' ONLY when the user indicates time evolution:",
            "  • A clock duration: 'for 2s', 'for 2 seconds', '2s simulation', 'run 10 seconds',",
            "    'simulate 0.5s', 'end time 2s', 'over 5 seconds', 't=2s', 'for 2 sec'",
            "  • Physics keywords: 'transient', 'unsteady', 'pulsating', 'oscillating',",
            "    'time-dependent', 'time-varying', 'time stepping'",
            "  • Time evolution intent: 'how the flow develops', 'startup', 'watch the flow',",
            "    'initial transient', 'pressure wave', 'flow development', 'vortex shedding'",
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
            "=== HEAT TRANSFER — CRITICAL RULE ===",
            "Do NOT enable heat transfer unless the user EXPLICITLY asks for it.",
            "Heat transfer is ONLY needed when the user mentions ANY of:",
            "  • Temperature values: '300K', '400 degrees', 'T=77K'",
            "  • Thermal keywords: 'heat transfer', 'heated wall', 'cooled wall',",
            "    'hot wall', 'cold wall', 'thermal', 'heating', 'cooling',",
            "    'temperature difference', 'heat flux', 'convection (thermal)'",
            "  • Cryogenic fluids: LN2, liquid nitrogen, LH2, liquid hydrogen, etc.",
            "",
            "For SIMPLE FLOW simulations (e.g. 'air flow at 5 m/s through a channel'):",
            "  • Set hasStaticTemperature = false and staticTemperature = null for ALL patches",
            "  • Set hasTotalTemperature = false and totalTemperature = null for ALL patches",
            "  • Set strategyT.bcFamily = 'not_needed' for ALL patches",
            "  • This results in an ISOTHERMAL simulation (no energy equation)",
            "",
            "If in doubt, do NOT add temperature — isothermal is the safe default.",
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
            "- If inlet velocity is given (e.g. '30 m/s', 'velocity 5', 'U = 10 m/s'):",
            "    hasVelocity = true",
            "    velocityMagnitude = <value in m/s>   ← ALWAYS fill this when the user specifies a velocity",
            "    velocityVector = null   ← unless the user specifies direction explicitly",
            "    strategyU.bcFamily = fixedValue",
            "    strategyU.selectionMode = direct_user_value",
            "    Do NOT convert a fixed velocity into a mass/volumetric flow rate.",
            "    hasMassFlowRate = false, hasVolumetricFlowRate = false.",
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
            patch_config = self.types.GenerateContentConfig(
                thinking_config=self.types.ThinkingConfig(include_thoughts=True),
                tools=[PATCH_SPEC_TOOL_SCHEMA],
                tool_config=self.types.ToolConfig(
                    function_calling_config=self.types.FunctionCallingConfig(
                        mode="ANY",
                        allowed_function_names=["submit_patch_spec"],
                    )
                ),
            )
            for attempt in range(_MAX_LLM_RETRIES):
                try:
                    stream = await self.client.aio.models.generate_content_stream(
                        model=self.super_model,
                        contents=prompt,
                        config=patch_config,
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
                    break  # success

                except Exception as e:
                    if self._provider.is_retryable_error(e) and attempt < _MAX_LLM_RETRIES - 1:
                        delay = _BASE_RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(
                            "[Precheck/patch] %s retryable error (attempt %d/%d), "
                            "retrying in %.1fs: %s",
                            patch_name, attempt + 1, _MAX_LLM_RETRIES, delay, e,
                        )
                        await asyncio.sleep(delay)
                        continue
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
            _vel_mag = patch_plan.get("velocityMagnitude")
            _vel_vec = patch_plan.get("velocityVector")
            _has_vel = bool(patch_plan.get("hasVelocity") or _vel_mag or _vel_vec)
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
            elif _has_vel:
                if _vel_vec:
                    _flow_rate_block = (
                        f"FIXED VELOCITY VECTOR IS THE PRIMARY INLET DRIVER:\n"
                        f"  fieldU.bcType          = fixedValue\n"
                        f"  fieldU.entryValueVector = {_vel_vec}   ← copy this EXACT vector\n"
                        f"  fieldU.entryValue      = null\n"
                        f"  fieldP.bcType          = zeroGradient"
                    )
                elif _vel_mag is not None:
                    _flow_rate_block = (
                        f"FIXED VELOCITY MAGNITUDE IS THE PRIMARY INLET DRIVER:\n"
                        f"  The user specified {_vel_mag} m/s.\n"
                        f"  fieldU.bcType          = fixedValue\n"
                        f"  fieldU.entryValue      = {_vel_mag}   ← copy this EXACT number\n"
                        f"  You may also set entryValueVector if the flow direction is known\n"
                        f"  (e.g. [{ _vel_mag}, 0, 0] for +x direction).\n"
                        f"  fieldP.bcType          = zeroGradient"
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
            _is_comp  = "compressibility=compressible," in self._format_solver_context(request)

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

    async def _merge_patch_specs_into_response(
        self,
        request: PrecheckRequest,
        boundary_plan: dict[str, Any],
        patch_specs: dict[str, dict[str, Any]],
    ) -> PrecheckResponse:
        """Convert all PatchSpec dicts into a PrecheckResponse."""
        mesh = request.get_mesh()

        prompt_lower = request.prompt.lower()

        # Infer global physics from planner patches
        has_heat = False
        is_compressible = False
        is_transient = False
        flow_regime = "turbulent"
        turb_model = "kOmegaSST"

        # Heat transfer detection.
        #
        # Trust the planner's explicit signal first: when the planner sets
        # ``strategyT.status == "selected"`` AND attaches a concrete
        # ``staticTemperature`` / ``totalTemperature`` value, the user
        # provided a temperature for that patch — that IS heat transfer.
        #
        # The keyword check below is a softer second signal that catches
        # cases where the planner failed to extract a concrete value but
        # the prompt clearly talks about temperature.  It is NOT a veto on
        # the planner's explicit per-patch decisions any more — previously
        # this guard was stripping legitimate T BCs whenever the user's
        # most-recent message didn't repeat the thermal keywords.
        _HEAT_KEYWORDS = (
            "temperature", "thermal", "heat transfer", "heat flux",
            "heated", "cooled", "hot wall", "cold wall", "hot ", "cold ",
            "heating", "cooling", "convection heat", "conduction",
            "kelvin", " k ", "celsius", "degrees",
        )
        _prompt_has_heat_keyword = (
            any(kw in prompt_lower for kw in _HEAT_KEYWORDS) or
            any(kw in prompt_lower for kw in CRYOGENIC_KEYWORDS)
        )

        for p in boundary_plan.get("patches", []):
            _has_T = (
                (p.get("hasStaticTemperature") and p.get("staticTemperature") is not None)
                or (p.get("hasTotalTemperature") and p.get("totalTemperature") is not None)
            )
            # 1. Planner explicitly chose a T strategy for this patch (status="selected")
            #    AND attached a concrete value → trust it unconditionally.
            _t_strategy_selected = (
                isinstance(p.get("strategyT"), dict)
                and p["strategyT"].get("status") == "selected"
            )
            if _has_T and _t_strategy_selected:
                has_heat = True
                continue
            # 2. Planner attached a value but didn't explicitly mark T as
            #    "selected" — fall back to the keyword check to avoid
            #    autonomous T BCs the user never asked for.
            if _has_T and _prompt_has_heat_keyword:
                has_heat = True
        # Fluid preset (detected early so compressibility can be fluid-aware)
        fluid = self._detect_fluid(prompt_lower)

        # Compressibility: only for explicit keywords or cryogenic fluids.
        # Common liquids (water, oil) are always incompressible at low Mach.
        _INCOMPRESSIBLE_FLUIDS = ("water", "oil")
        _fluid_is_incompressible = fluid.preset_id in _INCOMPRESSIBLE_FLUIDS
        if _fluid_is_incompressible:
            is_compressible = False
        else:
            is_compressible = any(kw in prompt_lower for kw in
                                  ("mach", "supersonic", "transonic", "compressible") +
                                  CRYOGENIC_KEYWORDS)
        # Default to steady — transient only when the LLM explicitly says so
        is_transient = boundary_plan.get("time_scheme", "steady") == "transient"
        # Laminar detection: prompt keyword OR explicit LLM field from boundary planner
        if "laminar" in prompt_lower or boundary_plan.get("flow_regime") == "laminar":
            flow_regime = "laminar"
            turb_model = "laminar"

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
        turb_k: float | None = None
        turb_omega: float | None = None
        turb_epsilon: float | None = None
        turb_nut: float | None = None
        if flow_regime == "turbulent":
            turb_intensity = 5.0
            # Estimate hydraulic diameter from mesh bounding box
            hydraulic_diam = self._estimate_hydraulic_diameter(mesh)
            for p in boundary_plan.get("patches", []):
                if p.get("patchRole") == "inlet":
                    if p.get("hasTurbulenceIntensity") and p.get("turbulenceIntensity"):
                        turb_intensity = float(p["turbulenceIntensity"]) * 100
                    break

            # Derive inlet velocity magnitude from first inlet BC
            _Uavg = self._estimate_inlet_velocity(
                boundary_conditions, boundary_plan, fluid, mesh,
            )

            # Standard turbulence derivations
            _I_frac = turb_intensity / 100.0
            _L = 0.07 * hydraulic_diam
            _Cmu = 0.09
            turb_k = 1.5 * (_Uavg * _I_frac) ** 2
            if _L > 0 and turb_k > 0:
                turb_omega = turb_k ** 0.5 / (_Cmu ** 0.25 * _L)
                turb_epsilon = _Cmu ** 0.75 * turb_k ** 1.5 / _L
                if turb_model == "kEpsilon" and turb_epsilon > 0:
                    turb_nut = _Cmu * turb_k ** 2 / turb_epsilon
                elif turb_model == "kOmegaSST" and turb_omega > 0:
                    turb_nut = turb_k / turb_omega

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

        # Gravity / buoyancy detection — keyword scan with explicit negation guard.
        # Naive substring matching used to trip on "no gravity" because the
        # substring "gravity" was present.  We now check for negation phrases
        # FIRST; if any are present, gravity is off regardless of what keywords
        # appear later in the prompt.
        _NO_GRAVITY_PATTERNS = (
            "no gravity", "without gravity", "no buoyancy", "without buoyancy",
            "ignore gravity", "ignore buoyancy",
            "neglect gravity", "neglect buoyancy",
            "no g ", "no g.", "no g,",
            "g=0", "g = 0",
            "gravity off", "buoyancy off",
            "forced convection",   # user wrote "forced convection" → no buoyancy
        )
        _GRAVITY_KEYWORDS = ("gravity", "gravitational", "buoyancy", "buoyant",
                              "natural convection", "free convection")
        if any(p in prompt_lower for p in _NO_GRAVITY_PATTERNS):
            _has_gravity = False
        else:
            _has_gravity = any(w in prompt_lower for w in _GRAVITY_KEYWORDS)
        # ── LLM-fuzzy extraction of explicit user solver name ───────────────
        # At precheck time there are no validated physics flags yet — only
        # the user's natural-language input.  Feeding empty/all-False flags
        # to the physics-based selector is contra-productive (it biases the
        # LLM toward the wrong family).  Instead we run JUST the user-intent
        # extractor: a single-purpose LLM call that returns a canonical
        # solver name iff the user literally named one (typos tolerated).
        #
        # We pass the user's FULL conversation (all user turns from history
        # + the current message), not just request.prompt.  Why: the chat
        # frontend often hands us a short LLM-generated "ready signal"
        # summary as request.prompt — which can truncate the user's
        # original solver name (e.g. "...using buoyantBoussinesq" with the
        # "SimpleFoam" suffix lost).  The original full prompt with the
        # explicit solver name lives in request.history.
        #
        # If the extractor returns null (no explicit name found anywhere),
        # we fall through to the prompt-derived keyword fallback below.
        _user_turns = [
            (m.get("content") or "").strip()
            for m in (request.history or [])
            if (m.get("role") == "user") and m.get("content")
        ]
        if request.prompt:
            _user_turns.append(request.prompt.strip())
        _full_user_text = "\n\n".join(t for t in _user_turns if t)

        openfoam_solver: str | None = None
        try:
            from simd_agent.run.solver_selector import SolverSelector
            _selector = SolverSelector()
            openfoam_solver = await _selector.extract_user_solver_from_prompt(
                _full_user_text,
            )
            if openfoam_solver:
                print(f"[PRECHECK] User-named solver honored → '{openfoam_solver}'")
            else:
                print("[PRECHECK] No explicit solver named; using keyword fallback")
        except Exception as exc:
            print(f"[PRECHECK] User-intent extraction failed ({exc}); "
                  f"using keyword fallback")

        # Keyword fallback — fires when the user did not name a solver or
        # the extractor failed.  Uses the has_heat / has_gravity /
        # is_transient / is_compressible flags extracted from the prompt
        # earlier in this function.
        if openfoam_solver is None:
            if has_heat and _has_gravity:
                openfoam_solver = "buoyantPimpleFoam" if is_transient else "buoyantSimpleFoam"
            elif is_compressible:
                openfoam_solver = "rhoPimpleFoam" if is_transient else "rhoSimpleFoam"
            elif not is_transient:
                openfoam_solver = "simpleFoam"
            else:
                openfoam_solver = "pimpleFoam"
            print(f"[PRECHECK] Keyword fallback → '{openfoam_solver}'")

        # ── Solver-identity invariant (safety net) ───────────────────────────
        # rhoSimpleFoam / rhoPimpleFoam / buoyantSimpleFoam / buoyantPimpleFoam
        # all solve the energy equation and require ``0/T``.  If any of them was
        # selected (typically by the compressibility path above, which has its
        # own keyword detection) BUT ``has_heat`` is still ``False`` AND the
        # planner extracted at least one temperature, the state is inconsistent.
        # The solver's identity is the stronger signal — force has_heat=True and
        # rebuild the per-patch BCs so the T values survive.
        _ENERGY_SOLVERS = {
            "rhoSimpleFoam", "rhoPimpleFoam",
            "buoyantSimpleFoam", "buoyantPimpleFoam",
        }
        if openfoam_solver in _ENERGY_SOLVERS and not has_heat:
            _planner_has_any_T = any(
                (p.get("hasStaticTemperature") and p.get("staticTemperature") is not None)
                or (p.get("hasTotalTemperature") and p.get("totalTemperature") is not None)
                for p in boundary_plan.get("patches", [])
            )
            if _planner_has_any_T:
                logger.warning(
                    f"[PRECHECK] Solver-identity invariant: {openfoam_solver} is an "
                    "energy solver but has_heat was False despite the planner "
                    "extracting temperature BCs.  Forcing has_heat=True and "
                    "rebuilding boundary conditions so the T values survive."
                )
                has_heat = True
                # Re-run the BC builder for every patch with the corrected has_heat.
                # patch_specs, _plan_by_name and flow_regime are all in scope.
                boundary_conditions = {
                    patch_name: self._patch_spec_args_to_patch_bc(
                        spec, has_heat, flow_regime, _plan_by_name.get(patch_name, {})
                    )
                    for patch_name, spec in patch_specs.items()
                }

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
            _target_snapshots = max(30, min(100, int(_end_time * 10)))
            _write_interval = max(_delta_t, _end_time / _target_snapshots)
        else:
            _end_time = None
            _delta_t = None
            _write_interval = None
            # Use LLM-extracted iterations if available, cap at 5000
            _llm_max_iter = boundary_plan.get("max_iterations")
            if _llm_max_iter and int(_llm_max_iter) > 0:
                _max_iterations = min(int(_llm_max_iter), 5000)
            else:
                _max_iterations = _MAX_STEADY_ITERATIONS

        logger.info(
            f"[PRECHECK] Solver settings: end_time={_end_time}, delta_t={_delta_t}, "
            f"max_iterations={_max_iterations}, write_interval={_write_interval}, "
            f"time_scheme={time_scheme}, transient={is_transient}"
        )

        suggested_config = SuggestedConfig(
            case_type=self._detect_case_type(prompt_lower),
            flow_regime=flow_regime,
            time_scheme=time_scheme,
            compressibility=compressibility_str,
            enable_heat_transfer=has_heat,
            gravity=_has_gravity,
            openfoam_solver=openfoam_solver,
            phase_change_detected=phase_change_detected,
            # solver: algorithm derived from openfoam_solver so the UI shows
            # the correct value immediately.  Temporal fields (end_time, delta_t,
            # write_interval) extracted from the prompt.
            solver=SolverSettings(
                algorithm=self._algorithm_for_solver(openfoam_solver),
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
                k=turb_k,
                omega=turb_omega,
                epsilon=turb_epsilon,
                nut=turb_nut,
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
                f"Filling / moving fluid-front detected. '{openfoam_solver}' is a single-phase "
                f"solver — it assumes the domain is already full of one fluid and cannot model "
                f"a fill front or air displacement. For accurate filling simulations, a VOF "
                f"two-phase solver (e.g. interFoam) is required, which is not yet supported."
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

        # ── Telemetry ────────────────────────────────────────────────────
        from simd_agent.telemetry import get_telemetry, PrecheckCompleted
        get_telemetry().capture(PrecheckCompleted(
            solver=openfoam_solver,
            flow_regime=flow_regime,
            time_scheme=time_scheme,
            compressibility=compressibility_str,
            heat_transfer=has_heat,
            gravity=suggested_config.gravity,
            fluid=fluid.name,
            turbulence_model=turb_model,
            patch_count=len(patch_specs),
            phase_change_detected=phase_change_detected,
        ))

        return response

    # ── Pass 3b: Physics coherence check ─────────────────────────────────────

    def _physics_coherence_check(
        self,
        response: PrecheckResponse,
        mesh,
    ) -> PrecheckResponse:
        """Deterministic physics validation — acts like a CFD engineer reviewing the setup.

        Computes Reynolds number, detects regime/velocity mismatches, and auto-corrects
        the config so the simulation won't blow up at runtime.

        Runs between Pass 3 (Merge) and Pass 4 (Review).
        """
        sc = response.suggested_config
        fluid = sc.fluid
        warnings: list[str] = list(response.warnings or [])

        # ── Gather inputs ────────────────────────────────────────────────────
        D_h = self._estimate_hydraulic_diameter(mesh)
        U_avg = self._estimate_inlet_velocity(
            sc.boundary_conditions, {}, fluid, mesh,
        )
        nu = fluid.mu / fluid.rho if fluid.rho > 0 else 1e-6  # kinematic viscosity

        # ── Compute Reynolds number ──────────────────────────────────────────
        Re: float | None = None
        if U_avg > 0 and D_h > 0 and nu > 0:
            Re = U_avg * D_h / nu

        sc.reynolds_number = round(Re, 1) if Re is not None else None

        if Re is None:
            logger.info("[PRECHECK/physics] Could not compute Re — skipping coherence check")
            return response

        logger.info(
            f"[PRECHECK/physics] Re = {U_avg:.4g} × {D_h:.4g} / {nu:.4g} = {Re:.0f}  "
            f"(regime={sc.flow_regime})"
        )

        # ── Regime thresholds ────────────────────────────────────────────────
        RE_LAMINAR_MAX = 2300
        RE_TURBULENT_MIN = 4000

        user_said_laminar = sc.flow_regime == "laminar"
        user_said_turbulent = sc.flow_regime == "turbulent"

        corrected = False

        # ── Case 1: User requested laminar but Re is turbulent ───────────────
        if user_said_laminar and Re > RE_LAMINAR_MAX:
            U_max_laminar = RE_LAMINAR_MAX * nu / D_h
            if Re > RE_TURBULENT_MIN:
                # Clearly turbulent — correct velocity down
                warnings.append(
                    f"Regime mismatch: you requested laminar flow, but at {U_avg:.4g} m/s "
                    f"the Reynolds number is {Re:.0f} (fully turbulent). "
                    f"For laminar flow with {fluid.name} in this geometry (D_h = {D_h:.4g} m), "
                    f"the maximum velocity is {U_max_laminar:.4g} m/s (Re = {RE_LAMINAR_MAX}). "
                    f"Inlet velocity has been reduced to {U_max_laminar:.4g} m/s."
                )
                self._correct_inlet_velocity(sc, U_max_laminar)
                U_avg = U_max_laminar
                Re = RE_LAMINAR_MAX
                sc.reynolds_number = round(Re, 1)
                corrected = True
            else:
                # Transitional (2300 < Re < 4000) — warn but keep laminar
                warnings.append(
                    f"The Reynolds number is {Re:.0f} (transitional zone, 2300-4000). "
                    f"Laminar flow may be unstable. Consider reducing velocity below "
                    f"{U_max_laminar:.4g} m/s for a stable laminar simulation."
                )

        # ── Case 2: Default turbulent but Re is "laminar" by D_h ────────────
        # Previously: auto-demoted to laminar whenever Re < 2300.  This is
        # unsafe on real meshes — bbox-D_h overshoots true D_h by ~3× on
        # U-bends, tees and multi-inlet ducts, so the computed Re can read
        # ~500 where the physical value is ~12 500.  The bug surfaced in
        # production as ``cfd_physics.flow_regime = laminar`` +
        # ``cfd_turbulence.model = kOmegaSST`` (DB inconsistency) and as
        # ``simulationType laminar`` in turbulenceProperties of a
        # rhoSimpleFoam case that promptly SIGFPE'd.
        #
        # New policy: surface the low Reynolds as an *informational
        # warning* but DO NOT silently flip the regime.  The user only
        # gets a laminar simulation when they explicitly asked for one
        # (planner ``flow_regime: laminar`` OR the keyword "laminar" in
        # the prompt — already handled in ``_build_suggested_config``).
        elif user_said_turbulent and Re < RE_LAMINAR_MAX:
            warnings.append(
                f"At {U_avg:.4g} m/s the bbox-derived Reynolds number is {Re:.0f} "
                f"(below the classical laminar threshold of {RE_LAMINAR_MAX}). "
                f"On complex / multi-inlet geometries D_h estimated from the "
                f"bounding box can overshoot the true value by 3× or more, so "
                f"the regime is kept turbulent.  If you intended a laminar "
                f"simulation, set the flow regime explicitly."
            )

        elif user_said_turbulent and RE_LAMINAR_MAX <= Re < RE_TURBULENT_MIN:
            # Transitional — warn but keep turbulent (safer choice)
            U_min_turbulent = RE_TURBULENT_MIN * nu / D_h
            warnings.append(
                f"The Reynolds number is {Re:.0f} (transitional zone, 2300-4000). "
                f"kOmegaSST will be used, but results may not fully represent "
                f"transitional physics. For clearly turbulent flow, increase velocity "
                f"above {U_min_turbulent:.4g} m/s (Re > {RE_TURBULENT_MIN})."
            )

        # ── Recompute turbulence values if regime changed to turbulent ────────
        # (This handles the unlikely case where we flip from laminar → turbulent,
        #  but mainly ensures values are consistent after velocity correction.)
        if corrected and sc.flow_regime == "turbulent":
            turb = sc.turbulence
            I_frac = (turb.turbulence_intensity or 5.0) / 100.0
            L = 0.07 * D_h
            Cmu = 0.09
            k = 1.5 * (U_avg * I_frac) ** 2
            if L > 0 and k > 0:
                omega = k ** 0.5 / (Cmu ** 0.25 * L)
                epsilon = Cmu ** 0.75 * k ** 1.5 / L
                nut = k / omega if omega > 0 else None
                turb.k = k
                turb.omega = omega
                turb.epsilon = epsilon
                turb.nut = nut
                turb.hydraulic_diameter = D_h
                turb.turbulence_length_scale = L

        # ── Update warnings on response ──────────────────────────────────────
        response.warnings = warnings if warnings else None

        logger.info(
            f"[PRECHECK/physics] Done — Re={Re:.0f}, regime={sc.flow_regime}, "
            f"corrected={corrected}"
        )

        return response

    @staticmethod
    def _correct_inlet_velocity(
        sc: SuggestedConfig,
        new_velocity: float,
    ) -> None:
        """Patch inlet boundary condition velocity to a new value.

        Updates both scalar (entryValue) and vector (entryValueVector) forms.
        """
        for bc in sc.boundary_conditions.values():
            if bc.patch_class != "inlet" or bc.U is None:
                continue
            if bc.U.type in ("fixedValue", "fixedValue;"):
                old = bc.U.value
                if isinstance(old, list) and len(old) >= 3:
                    # Preserve direction, scale magnitude
                    mag = sum(x ** 2 for x in old) ** 0.5
                    if mag > 0:
                        scale = new_velocity / mag
                        bc.U.value = [round(x * scale, 6) for x in old]
                    else:
                        bc.U.value = [round(new_velocity, 6), 0.0, 0.0]
                else:
                    bc.U.value = round(new_velocity, 6)

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

            # recover velocity from planner plan
            # The patch agent sometimes forgets to set entryValue/entryValueVector
            # even when it correctly selects fixedValue for U.
            if value is None and key == "fieldU" and bc_type == "fixedValue":
                if plan.get("velocityVector"):
                    value = list(plan["velocityVector"])
                elif plan.get("velocityMagnitude") is not None:
                    value = float(plan["velocityMagnitude"])

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

        # Per-patch turbulence intensity from the planner output.
        # Planner schema returns I as a fraction (0.05 = 5%); store as-is.
        # None when the user didn't specify a TI for this patch — validator
        # will fall back to a sensible global default.
        _patch_TI: float | None = None
        if turb and role == "inlet" and plan.get("hasTurbulenceIntensity"):
            _ti_raw = plan.get("turbulenceIntensity")
            if _ti_raw is not None:
                try:
                    _patch_TI = float(_ti_raw)
                    # Planner sometimes emits a percentage rather than a
                    # fraction — clamp to the fraction range.
                    if _patch_TI > 1.0:
                        _patch_TI = _patch_TI / 100.0
                except (TypeError, ValueError):
                    pass

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
            turbulence_intensity=_patch_TI,
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
        sc = response.suggested_config
        flow_regime = sc.flow_regime
        is_laminar = flow_regime == "laminar"
        solver_name = sc.openfoam_solver or "simpleFoam"
        turb = sc.turbulence

        # Build governing equations list
        equations = ["Navier-Stokes (momentum)", "Continuity"]
        if sc.enable_heat_transfer:
            equations.append("Energy")
        if not is_laminar:
            if turb.model == "kOmegaSST":
                equations.extend(["$k$ transport", "$\\omega$ transport (SST)"])
            elif turb.model == "kEpsilon":
                equations.extend(["$k$ transport", "$\\varepsilon$ transport"])

        equations_str = ", ".join(equations)

        # Build turbulence summary with pre-computed values
        if is_laminar:
            turbulence_block = (
                "**Turbulence Model:** None (laminar)\n"
                "No $k$, $\\omega$, $\\varepsilon$, or $\\nu_t$ fields are needed."
            )
        else:
            turb_lines = [f"**Turbulence Model:** `{turb.model}`"]
            if turb.turbulence_intensity is not None:
                turb_lines.append(f"- Intensity $I$ = {turb.turbulence_intensity}%")
            if turb.hydraulic_diameter is not None:
                turb_lines.append(f"- Hydraulic diameter $D_h$ = {turb.hydraulic_diameter} m")
            if turb.turbulence_length_scale is not None:
                turb_lines.append(f"- Length scale $L = 0.07 D_h$ = {turb.turbulence_length_scale:.4g} m")
            if turb.k is not None:
                turb_lines.append(f"- $k$ = {turb.k:.4g} m²/s²")
            if turb.omega is not None:
                turb_lines.append(f"- $\\omega$ = {turb.omega:.4g} s⁻¹")
            if turb.epsilon is not None:
                turb_lines.append(f"- $\\varepsilon$ = {turb.epsilon:.4g} m²/s³")
            if turb.nut is not None:
                turb_lines.append(f"- $\\nu_t$ = {turb.nut:.4g} m²/s")
            turbulence_block = "\n".join(turb_lines)

        # Build fluid summary — only include thermal properties when heat transfer is active
        fl = sc.fluid
        _fluid_lines = [
            f"**Fluid:** {fl.name}",
            f"- $\\rho$ = {fl.rho} kg/m³, $\\mu$ = {fl.mu:.4g} Pa·s",
        ]
        if sc.enable_heat_transfer:
            _fluid_lines.append(f"- $C_p$ = {fl.Cp} J/(kg·K), $k_f$ = {fl.k} W/(m·K)")
            _fluid_lines.append(f"- $T_{{ref}}$ = {fl.temperature} K")
        fluid_block = "\n".join(_fluid_lines)

        # Solver description
        algorithm = "PIMPLE" if "pimple" in solver_name.lower() or "Pimple" in solver_name else "SIMPLE"
        if "piso" in solver_name.lower():
            algorithm = "PISO"
        solver_block = (
            f"**Solver:** `{solver_name}` — {algorithm} algorithm, "
            f"{'transient' if sc.time_scheme == 'transient' else 'steady-state'}, "
            f"{'compressible' if sc.compressibility == 'compressible' else 'incompressible'}"
        )
        if sc.enable_heat_transfer:
            solver_block += ", energy equation active"
        if sc.gravity:
            solver_block += ", gravity enabled"

        # Append solver/time settings so the review confirms them
        ss = sc.solver
        if sc.time_scheme == "transient":
            time_parts = []
            if ss.end_time is not None:
                time_parts.append(f"endTime = {ss.end_time}")
            if ss.delta_t is not None:
                time_parts.append(f"deltaT = {ss.delta_t}")
            if ss.write_interval is not None:
                time_parts.append(f"writeInterval = {ss.write_interval}")
            if time_parts:
                solver_block += "\n\n**Time control:** " + ", ".join(time_parts)
        else:
            steady_parts = []
            if ss.max_iterations is not None:
                steady_parts.append(f"maxIterations = {ss.max_iterations}")
            if ss.write_interval is not None:
                steady_parts.append(f"writeInterval = {ss.write_interval}")
            if steady_parts:
                solver_block += "\n\n**Iterations:** " + ", ".join(steady_parts)

        # BC table columns — only include fields that are active in this simulation
        _bc_cols = ["Patch", "Class", "`U`", "`p`"]
        _bc_seps = ["-------", "-------", "-----", "-----"]
        if sc.enable_heat_transfer:
            _bc_cols.append("`T`")
            _bc_seps.append("-----")
        if not is_laminar:
            _bc_cols.append("`k` / $\\omega$")
            _bc_seps.append("----------------")
        bc_table_header = "| " + " | ".join(_bc_cols) + " |\n| " + " | ".join(_bc_seps) + " |"

        # Build the "active fields" instruction so the LLM knows what to include/exclude
        _active_fields = ["`U`", "`p`"]
        if sc.enable_heat_transfer:
            _active_fields.append("`T`")
        if not is_laminar:
            _active_fields.extend(["`k`", "`omega`" if turb.model == "kOmegaSST" else "`epsilon`"])
        _fields_str = ", ".join(_active_fields)

        _inactive_notes: list[str] = []
        if not sc.enable_heat_transfer:
            _inactive_notes.append(
                "- Temperature (`T`) is NOT part of this simulation (no heat transfer). "
                "Do NOT mention `T`, temperature BCs, thermal conductivity ($k_f$), "
                "specific heat ($C_p$), or reference temperature in any section."
            )
        if is_laminar:
            _inactive_notes.append(
                "- Turbulence fields (`k`, `omega`, `epsilon`, `nut`) are NOT part of this simulation (laminar). "
                "Do NOT mention turbulence BCs or wall functions."
            )
        _inactive_block = "\n".join(_inactive_notes) if _inactive_notes else ""

        review_prompt = f"""\
You are a CFD expert producing an OpenFOAM-style problem specification and review.

Write a structured specification with concise prose under each heading.
Use backticks for identifiers (`inlet`, `fixedValue`, `kOmegaSST`),
inline LaTeX for symbols ($k$, $\\omega$, $\\rho$), and display math ($$...$$)
on its own line for any formula with a numeric substitution.

=== ACTIVE FIELDS IN THIS SIMULATION ===
{_fields_str}
Only discuss these fields. Do not mention, list, or create table columns for inactive fields.
{_inactive_block}

=== USER REQUEST ===
{request.prompt}

=== GENERATED SPEC ===
{spec_json}

=== WRITE THE FOLLOWING SECTIONS ===

## Problem Specification

### 1. Solution Domain
Describe the flow domain in 1-2 sentences (internal/external, geometry type,
key dimensions if inferrable from patch names or mesh stats).

### 2. Governing Equations
{equations_str}.
State each equation by name — no need to write full PDEs.

### 3. Solver
{solver_block}

### 4. Transport Properties
{fluid_block}

### 5. Turbulence
{turbulence_block}
{"These are pre-computed values. Present them as-is and show the derivation formulas with the substituted numbers for reference." if not is_laminar else ""}

### 6. Boundary Conditions
One paragraph per patch — describe the physical role, confirm the BC types are
correct, and state the numeric values. Only mention fields that are active: {_fields_str}.{"  For laminar flow, confirm that no turbulence fields are set." if is_laminar else ""}

### 7. Initial Conditions
State the internal field values that follow from the boundary conditions
({", ".join(_active_fields)}).

### 8. Summary
One paragraph recap, then a markdown table with ONLY the columns below (no extra columns):

{bc_table_header}
"""

        yield {
            "type": "review_phase_start",
            "phase": "review",
            "title": "Physics review",
            "description": "Reviewing boundary conditions, turbulence parameters, and fluid properties...",
        }
        print(f"[Precheck] → review (model={self.super_model})...", flush=True)

        review_config = self.types.GenerateContentConfig(
            thinking_config=self.types.ThinkingConfig(include_thoughts=True),
        )
        try:
            for attempt in range(_MAX_LLM_RETRIES):
                try:
                    stream = await self.client.aio.models.generate_content_stream(
                        model=self.super_model,
                        contents=review_prompt,
                        config=review_config,
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
                    break  # success

                except Exception as e:
                    if self._provider.is_retryable_error(e) and attempt < _MAX_LLM_RETRIES - 1:
                        delay = _BASE_RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(
                            "[Precheck/review] Retryable error (attempt %d/%d), "
                            "retrying in %.1fs: %s",
                            attempt + 1, _MAX_LLM_RETRIES, delay, e,
                        )
                        await asyncio.sleep(delay)
                        continue
                    raise

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
        fluid = self._detect_fluid(prompt_lower)
        _INCOMPRESSIBLE_FLUIDS = ("water", "oil")
        if fluid.preset_id in _INCOMPRESSIBLE_FLUIDS:
            is_compressible = False
        else:
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
            f"fluid={fluid.name}, "
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
            (("lox", "liquid oxygen"), "lox"),
            (("lh2", "liquid hydrogen"), "lh2"),
            (("helium", "lhe", "liquid helium"), "helium"),
            (("ln2", "liquid nitrogen", "nitrogen"), "ln2"),
            (("oil", "lubricant"), "oil"),
            (("water", "hydraulic"), "water"),
        ]:
            if any(k in prompt_lower for k in kw):
                return FLUID_PRESETS[key]
        return FLUID_PRESETS["air"]

    @staticmethod
    def _algorithm_for_solver(solver_name: str) -> str:
        """Derive the pressure-velocity coupling algorithm from the solver name.

        Uses the solver plugin registry when available (single source of truth).
        Falls back to a name-based heuristic for legacy/multiphase solvers
        that haven't been ported to plugins yet.
        """
        from simd_agent.solvers.registry import get_registry

        plugin = get_registry().get(solver_name)
        if plugin is not None:
            return plugin.algorithm

        # Fallback for solvers not yet in the plugin registry (legacy multiphase,
        # chtMultiRegionFoam, etc.).  Transient solvers default to PIMPLE.
        _name = solver_name.lower()
        if "simple" in _name:
            return "SIMPLE"
        if "piso" in _name or _name == "icofoam":
            return "PISO"
        # interFoam, compressibleInterFoam, chtMultiRegionFoam, etc. all use PIMPLE
        return "PIMPLE"

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

    @staticmethod
    def _estimate_hydraulic_diameter(mesh) -> float:
        """Estimate hydraulic diameter from mesh bounding box.

        For internal flows, D_h = 4A/P.  We approximate the cross-section as
        the two smallest extents of the bounding box (the largest extent is
        assumed to be the flow direction).  Falls back to 0.1 m.
        """
        if mesh is None:
            return 0.1

        bb = getattr(mesh, "check_mesh", None)
        if bb is None:
            return 0.1

        bbox = getattr(bb, "bounding_box", None) or (
            bb.get("bounding_box") if isinstance(bb, dict) else None
        )
        if not bbox:
            # Try characteristic_length as a fallback
            cl = getattr(bb, "characteristic_length", None)
            if cl and cl > 0:
                return float(cl)
            return 0.1

        try:
            bb_min = bbox.get("min") or bbox.get("Min", [0, 0, 0])
            bb_max = bbox.get("max") or bbox.get("Max", [0, 0, 0])
            extents = [abs(bb_max[i] - bb_min[i]) for i in range(3)]
            extents.sort()
            # Two smallest extents form the cross-section
            a, b = extents[0], extents[1]
            if a <= 0 or b <= 0:
                return extents[2] if extents[2] > 0 else 0.1
            # D_h = 4 * A / P  for a rectangle: 4*a*b / (2*(a+b)) = 2*a*b/(a+b)
            return 2.0 * a * b / (a + b)
        except (IndexError, TypeError, ValueError):
            return 0.1

    @staticmethod
    def _estimate_inlet_velocity(
        boundary_conditions: dict,
        boundary_plan: dict,
        fluid: FluidProperties,
        mesh,
    ) -> float:
        """Estimate inlet velocity magnitude for turbulence derivation.

        Handles three cases:
        1. fixedValue velocity — use the value directly
        2. flowRateInletVelocity with massFlowRate — convert via rho and inlet area
        3. flowRateInletVelocity with volumetricFlowRate — divide by inlet area

        Falls back to 1.0 m/s if nothing can be determined.
        """
        for bc in boundary_conditions.values():
            if bc.patch_class != "inlet" or bc.U is None:
                continue

            bc_type = bc.U.type
            val = bc.U.value

            # Case 1: fixedValue — value is velocity (scalar or vector)
            if bc_type in ("fixedValue", "fixedValue;") and val is not None:
                if isinstance(val, list):
                    mag = sum(x ** 2 for x in val) ** 0.5
                else:
                    mag = abs(float(val))
                if mag > 0:
                    return mag

            # Case 2 & 3: flowRateInletVelocity — need inlet area to convert
            if bc_type == "flowRateInletVelocity" and val is not None:
                # Estimate inlet area from mesh bounding box cross-section
                inlet_area = 1.0  # fallback
                if mesh and hasattr(mesh, "check_mesh"):
                    bb = mesh.check_mesh
                    bbox = getattr(bb, "bounding_box", None)
                    if isinstance(bb, dict):
                        bbox = bb.get("bounding_box")
                    if bbox:
                        try:
                            bb_min = bbox.get("min") or bbox.get("Min", [0, 0, 0])
                            bb_max = bbox.get("max") or bbox.get("Max", [0, 0, 0])
                            extents = sorted(abs(bb_max[i] - bb_min[i]) for i in range(3))
                            # Inlet area ≈ product of two smallest extents
                            if extents[0] > 0 and extents[1] > 0:
                                inlet_area = extents[0] * extents[1]
                        except (IndexError, TypeError, ValueError):
                            pass

                entries = bc.U.entries or {}
                if "massFlowRate" in entries:
                    mfr = abs(float(entries["massFlowRate"]))
                    rho = fluid.rho if fluid.rho > 0 else 1.225
                    return mfr / (rho * inlet_area) if inlet_area > 0 else 1.0
                elif "volumetricFlowRate" in entries:
                    vfr = abs(float(entries["volumetricFlowRate"]))
                    return vfr / inlet_area if inlet_area > 0 else 1.0
                else:
                    # val is probably the flow rate itself (mass or vol)
                    rho = fluid.rho if fluid.rho > 0 else 1.225
                    return abs(float(val)) / (rho * inlet_area) if inlet_area > 0 else 1.0

        return 1.0  # safe fallback

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


    # ── Conversation mode ───────────────────────────────────────────────────

    async def _stream_conversation(
        self, request: PrecheckRequest
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream a conversational response (no full analysis).

        Uses the provider's lighter default model and the signal_ready_to_analyze
        tool.  When the LLM determines the user has provided enough info,
        it calls that tool and we emit a ``ready_to_analyze`` event.
        """
        yield {"type": "start"}

        # ── Summarize if history is too long ─────────────────────────────
        history = list(request.history or [])
        summary = request.conversation_summary
        token_est = self._estimate_token_count(history, summary, request.prompt)

        if token_est > CONVERSATION_TOKEN_LIMIT and history:
            print(f"[Precheck/conv] Token estimate {token_est} > {CONVERSATION_TOKEN_LIMIT}, summarizing…", flush=True)
            summary = await self._summarize_conversation(history, summary)
            yield {"type": "conversation_summary", "summary": summary}
            history = []  # replace history with summary

        # ── Build context block ──────────────────────────────────────────
        ctx_parts: list[str] = []
        mesh = request.get_mesh()
        if mesh:
            patch_names = ", ".join(p.name for p in mesh.patches)
            ctx_parts.append(
                f"Mesh uploaded: **{mesh.file_name}** "
                f"({mesh.check_mesh.cells:,} cells, patches: {patch_names})"
            )
        else:
            ctx_parts.append("No mesh uploaded yet.")

        if summary:
            ctx_parts.append(f"\nPrior conversation summary:\n{summary}")

        # ── Signal if analysis was already completed ─────────────────
        sim_ctx = request.simulation_context
        if sim_ctx and sim_ctx.get("precheckSummary"):
            ctx_parts.append(
                "\n**Analysis already completed.** The user has already gone through "
                "the simulation planning and analysis phase. Do NOT call "
                "signal_ready_to_analyze again unless the user explicitly asks to "
                "change the simulation, modify parameters, or re-run the analysis. "
                "Just answer their questions directly."
            )

        system_prompt = self.CONVERSATION_SYSTEM_PROMPT.replace(
            "{context}", "\n".join(ctx_parts)
        )

        # ── Build Gemini contents from history ───────────────────────────
        types = self.types
        contents: list[types.Content] = []
        for msg in history:
            role = "user" if msg.get("role") == "user" else "model"
            text = msg.get("content", "")
            if text:
                contents.append(types.Content(
                    role=role, parts=[types.Part.from_text(text=text)],
                ))

        contents.append(types.Content(
            role="user", parts=[types.Part.from_text(text=request.prompt)],
        ))

        # ── Stream from Gemini ───────────────────────────────────────────
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=1.0,
            max_output_tokens=4096,
            thinking_config=types.ThinkingConfig(include_thoughts=True),
            tools=[READY_TOOL_SCHEMA],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="AUTO"),
            ),
        )

        collected_text = ""
        ready_info: dict[str, Any] | None = None

        try:
            for attempt in range(_MAX_LLM_RETRIES):
                try:
                    stream = await self.client.aio.models.generate_content_stream(
                        model=self.model,
                        contents=contents,
                        config=config,
                    )

                    async for chunk in stream:
                        if not chunk.candidates:
                            continue
                        for part in chunk.candidates[0].content.parts:
                            if getattr(part, "thought", False):
                                if part.text:
                                    yield {"type": "thinking", "text": part.text}
                            elif getattr(part, "function_call", None) is not None:
                                fc = part.function_call
                                if fc.name == "signal_ready_to_analyze":
                                    args = dict(fc.args) if fc.args else {}
                                    ready_info = {"summary": args.get("summary", "")}
                                    print(f"[Precheck/conv] LLM signaled ready — {ready_info['summary'][:80]}", flush=True)
                            elif part.text:
                                yield {"type": "chat_token", "text": part.text}
                                collected_text += part.text
                    break  # success

                except Exception as e:
                    if self._provider.is_retryable_error(e) and attempt < _MAX_LLM_RETRIES - 1:
                        delay = _BASE_RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(
                            "[Precheck/conv] Retryable error (attempt %d/%d), "
                            "retrying in %.1fs: %s",
                            attempt + 1, _MAX_LLM_RETRIES, delay, e,
                        )
                        await asyncio.sleep(delay)
                        collected_text = ""
                        ready_info = None
                        continue
                    raise

            response_tokens = len(collected_text) // 4
            total_tokens = token_est + response_tokens

            if ready_info:
                yield {"type": "ready_to_analyze", **ready_info}

            yield {"type": "chat_done", "token_count": total_tokens}

        except Exception as e:
            logger.exception(f"[Precheck/conv] Conversation streaming failed: {e}")
            yield {"type": "error", "message": str(e)}

        yield {"type": "done"}

    # ── Token estimation + summarization ─────────────────────────────────

    @staticmethod
    def _estimate_token_count(
        history: list[dict[str, str]],
        summary: str | None,
        prompt: str,
    ) -> int:
        """Rough token estimate (~4 chars per token)."""
        total = len(prompt)
        for msg in history:
            total += len(msg.get("content", ""))
        if summary:
            total += len(summary)
        return total // 4

    async def _summarize_conversation(
        self,
        history: list[dict[str, str]],
        prior_summary: str | None,
    ) -> str:
        """Summarize conversation history using the lighter model."""
        parts: list[str] = []
        if prior_summary:
            parts.append(f"[Previous summary]: {prior_summary}\n")
        for msg in history:
            role = msg.get("role", "user").upper()
            parts.append(f"{role}: {msg.get('content', '')}")

        history_text = "\n\n".join(parts)

        prompt = (
            "Summarize this CFD simulation conversation concisely. "
            "Preserve ALL technical details:\n"
            "- Simulation goals and physics (flow type, fluid, heat transfer)\n"
            "- Numerical values (velocities, pressures, temperatures, flow rates)\n"
            "- Mesh details (file name, patches, cell count)\n"
            "- Decisions made and user preferences\n"
            "- Any unresolved questions\n\n"
            f"CONVERSATION:\n{history_text}"
        )

        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=prompt,
                config=self.types.GenerateContentConfig(
                    max_output_tokens=2048,
                    temperature=0.3,
                ),
            )
            return response.text or ""
        except Exception as e:
            logger.exception("[Precheck/conv] Summarization failed")
            return prior_summary or ""


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_precheck_service: PrecheckService | None = None


def get_precheck_service() -> PrecheckService:
    global _precheck_service
    if _precheck_service is None:
        _precheck_service = PrecheckService()
    return _precheck_service
