# simd_agent/chat/service.py
"""ChatService — streaming Gemini assistant with tool-calling for CFD analysis."""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any, AsyncIterator

from simd_agent.llm import get_provider

from simd_agent.chat.db import build_snapshot, persist_chat_message
from simd_agent.chat.models import (
    ArtifactEvent, ChatRequest, DoneEvent, ErrorEvent,
    QueryIntent, ThinkingEvent, TokenEvent, ToolCallPlan,
    ToolResultEvent, ToolStartEvent,
)
from simd_agent.chat.prompts import SYSTEM_PROMPT, RESPONSE_PROMPT
from simd_agent.chat.query_analyzer import get_query_analyzer
from simd_agent.chat.tools import CHAT_TOOLS_SCHEMA, TOOL_REGISTRY, SimulationSnapshot
from simd_agent.precheck.models import (
    CONVERSATION_MODEL, CONVERSATION_TOKEN_LIMIT, READY_TOOL_SCHEMA,
)
from simd_agent.settings import get_settings

logger = logging.getLogger(__name__)

# Confidence threshold: above this the backend pre-executes tools deterministically.
_CONFIDENCE_THRESHOLD = 0.7

_TOOL_LABELS: dict[str, str] = {
    "compute_field_stats": "Computing field statistics…",
    "compute_residual_trend": "Analyzing residual convergence…",
    "extract_velocity_profile": "Extracting velocity profile…",
    "run_python_analysis": "Running computation…",
    "generate_report": "Generating simulation report…",
    "analyze_chart": "Analyzing chart data…",
    "read_generated_file": "Reading case file…",
    "query_simulation_results": "Loading simulation results…",
    "plot_field_over_iterations": "Plotting field data…",
    "plot_field_values": "Plotting field values…",
    "plot_volume_values": "Plotting volume-averaged values…",
    "compare_runs": "Comparing runs…",
}


class ChatService:
    """Streaming chat service backed by Gemini with function-calling tools."""

    def __init__(self) -> None:
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
        return self._provider.models["default"]

    async def handle_turn(self, request: ChatRequest) -> AsyncIterator[dict[str, Any]]:
        """Process one user turn: analyze → route → execute → respond."""

        sim_id = request.simulation_id or request.context.simulation_id

        # ── Telemetry: track every chat query ──
        from simd_agent.telemetry import get_telemetry, ChatQuery
        get_telemetry().capture(
            ChatQuery(mode=request.mode, has_simulation=bool(sim_id)),
            user_id=request.user_id,
        )

        # ── Auto-promote precheck → chat when a simulation has already run ──
        # ComposeView always sends mode="precheck", but once a run has
        # completed the user is asking about results, not setting up.
        if request.mode == "precheck" and sim_id:
            try:
                import asyncio as _aio
                from uuid import UUID as _UUID
                from simd_agent.services import run_service as _run_svc
                latest = await _aio.wait_for(
                    _run_svc.get_latest(_UUID(sim_id)), timeout=3.0,
                )
                if latest and latest.status in ("succeeded", "completed", "failed"):
                    logger.info("[chat] Overriding mode precheck → chat (sim %s has completed run)", sim_id)
                    request.mode = "chat"
            except Exception:
                pass  # keep original mode on any error

        if sim_id:
            await persist_chat_message(sim_id, "user", request.message)

        # ── Route 1: confirm_analysis → delegate to precheck analysis pipeline ──
        if request.mode == "precheck" and request.confirm_analysis:
            logger.info("[chat] ── ROUTE: confirm_analysis → precheck pipeline ──")
            async for event in self._delegate_to_precheck(request):
                yield event
            return

        # ── Stage 1: Analyze intent ──────────────────────────────────────────
        history_dicts = [{"role": m.role, "content": m.content} for m in request.history]
        analyzer = get_query_analyzer()
        try:
            intent = await analyzer.analyze(request.message, history_dicts, mode=request.mode)
        except Exception:
            logger.exception("[chat] Query analyzer failed — falling back")
            intent = QueryIntent()

        # Deterministic patch: ensure report_type is set from the message
        # so we don't depend on the analyzer LLM extracting it.
        _msg_lower = request.message.lower()
        for tc in intent.tool_plan:
            if tc.tool == "generate_report" and "report_type" not in tc.args:
                tc.args["report_type"] = "expert" if "expert" in _msg_lower else "standard"

        logger.info(
            "[chat] ── STAGE 1: ANALYZE ── category=%s confidence=%.2f "
            "subject=%r tools=%s data_needs=%s",
            intent.category, intent.confidence, intent.resolved_subject,
            [f"{t.tool}({t.args})" for t in intent.tool_plan],
            intent.data_needs.model_dump() if intent.data_needs else None,
        )

        # ── Route 2: setup conversation (precheck mode + setup category) ──
        if intent.category == "setup" and request.mode == "precheck":
            logger.info("[chat] ── ROUTE: setup conversation ──")
            full_response_text = ""
            async for event in self._stream_setup_conversation(request):
                if event.get("type") == "token":
                    full_response_text += event.get("delta", "")
                yield event
            yield DoneEvent().model_dump()
            if sim_id and full_response_text:
                await persist_chat_message(sim_id, "assistant", full_response_text)
            return

        # ── Stage 2: Selective snapshot ──────────────────────────────────────
        use_deterministic = (
            intent.confidence >= _CONFIDENCE_THRESHOLD
            and intent.tool_plan
        )
        logger.info(
            "[chat] ── STAGE 2: ROUTE ── use_deterministic=%s (threshold=%.1f)",
            use_deterministic, _CONFIDENCE_THRESHOLD,
        )
        try:
            snap = await build_snapshot(
                request,
                data_needs=intent.data_needs if use_deterministic else None,
            )
        except Exception as exc:
            logger.exception("[chat] Failed to build snapshot")
            yield ErrorEvent(message=f"Failed to load simulation data: {exc}").model_dump()
            yield DoneEvent().model_dump()
            return

        types = self.types
        contents: list[types.Content] = [
            types.Content(
                role=msg.role if msg.role == "user" else "model",
                parts=[types.Part.from_text(text=msg.content)],
            )
            for msg in request.history
        ]
        contents.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=request.message)],
        ))

        full_response_text = ""
        pre_results: dict[str, dict[str, Any]] = {}

        if use_deterministic:
            # ── Stage 3: Pre-execute tools ───────────────────────────────────
            logger.info("[chat] ── STAGE 3: PRE-EXECUTE ── %d tools planned", len(intent.tool_plan))
            for tc in intent.tool_plan:
                tool_fn = TOOL_REGISTRY.get(tc.tool)
                if tool_fn is None:
                    logger.warning("[chat] Analyzer planned unknown tool: %s", tc.tool)
                    continue

                label = _TOOL_LABELS.get(tc.tool, f"Running {tc.tool}…")
                logger.info("[chat]   → executing %s(%s)", tc.tool, tc.args)
                yield ToolStartEvent(tool=tc.tool, label=label).model_dump()

                try:
                    if inspect.iscoroutinefunction(tool_fn):
                        result_data = await tool_fn(tc.args, snap)
                    else:
                        result_data = tool_fn(tc.args, snap)
                    logger.info(
                        "[chat]   ← %s result: has_chart=%s has_error=%s keys=%s",
                        tc.tool, "chart" in result_data, "error" in result_data,
                        list(result_data.keys()),
                    )
                except Exception as exc:
                    logger.exception("[chat] Pre-executed tool %s failed", tc.tool)
                    result_data = {"error": str(exc)}

                yield ToolResultEvent(tool=tc.tool, data=result_data).model_dump()

                # Emit frontend artifacts (charts, reports)
                async for artifact_event in self._emit_artifacts(tc.tool, result_data):
                    yield artifact_event

                # Store a stripped version for the LLM prompt (chart/report data
                # is huge and already sent to the frontend as artifacts).
                llm_result = {
                    k: v for k, v in result_data.items()
                    if k not in ("chart", "report_markdown", "report_data", "report_request_payload")
                }
                if "chart" in result_data:
                    chart = result_data["chart"]
                    llm_result["chart_generated"] = True
                    llm_result["chart_type"] = chart.get("type")
                    llm_result["chart_lines"] = chart.get("lines")
                    llm_result["chart_data_points"] = len(chart.get("data", []))
                if "report_markdown" in result_data:
                    llm_result["report_generated"] = True
                    llm_result["sections"] = result_data.get("sections_included", [])
                pre_results[tc.tool] = llm_result

            # If ALL pre-executed tools returned errors, fall back to full-tool
            # mode so the LLM can try alternative tools.
            all_failed = pre_results and all("error" in v for v in pre_results.values())
            if all_failed:
                logger.info(
                    "[chat] ── PRE-EXECUTE FALLBACK ── all %d tools failed, switching to full-tool mode",
                    len(pre_results),
                )
                # Re-fetch full snapshot since the deterministic one may be partial
                try:
                    snap = await build_snapshot(request, data_needs=None)
                except Exception:
                    pass  # keep the existing snap
                use_deterministic = False

        if use_deterministic:
            # ── Stage 4a: Response-only streaming ────────────────────────────
            logger.info("[chat] ── STAGE 4a: RESPONSE-ONLY ── pre_results keys=%s", list(pre_results.keys()))
            system_prompt = RESPONSE_PROMPT.replace(
                "{context_json}",
                json.dumps(snap.summary_dict(), indent=2, default=str),
            ).replace(
                "{tool_results_json}",
                json.dumps(pre_results, indent=2, default=str),
            )

            try:
                async for event in self._stream_response_only(
                    system_prompt=system_prompt,
                    contents=contents,
                    snap=snap,
                ):
                    if event.get("type") == "token":
                        full_response_text += event.get("delta", "")
                    yield event
            except Exception as exc:
                logger.exception("[chat] LLM response streaming failed")
                yield ErrorEvent(message=f"LLM error: {exc}").model_dump()
        else:
            # ── Stage 4b: Full tool mode (fallback) ──────────────────────────
            logger.info("[chat] ── STAGE 4b: FULL TOOL MODE (fallback) ──")
            system_prompt = SYSTEM_PROMPT.replace(
                "{context_json}",
                json.dumps(snap.summary_dict(), indent=2, default=str),
            )
            # Inject pre-execution errors so the LLM doesn't re-try failed tools
            # or call redundant chart tools that will also fail for the same reason.
            if pre_results:
                failed_summary = "\n".join(
                    f"- {tool}: {res.get('error', 'failed')}"
                    for tool, res in pre_results.items()
                    if "error" in res
                )
                if failed_summary:
                    system_prompt += (
                        "\n\n## Tools already attempted (FAILED — do NOT retry or call similar chart tools)\n"
                        f"{failed_summary}\n"
                        "Pick ONE alternative tool that can answer the user's question. "
                        "Do NOT call multiple chart tools for the same field."
                    )

            try:
                async for event in self._stream_with_tools(
                    system_prompt=system_prompt,
                    contents=contents,
                    snap=snap,
                ):
                    if event.get("type") == "token":
                        full_response_text += event.get("delta", "")
                    yield event
            except Exception as exc:
                logger.exception("[chat] LLM streaming failed")
                yield ErrorEvent(message=f"LLM error: {exc}").model_dump()

        if not full_response_text:
            full_response_text = "(no response generated)"

        suggested_actions = self._suggest_actions(snap, request.message)
        yield DoneEvent(suggested_actions=suggested_actions).model_dump()

        if sim_id:
            await persist_chat_message(sim_id, "assistant", full_response_text, suggested_actions)

    async def _stream_with_tools(
        self,
        *,
        system_prompt: str,
        contents: list[Any],
        snap: SimulationSnapshot,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream tokens from the LLM with thinking + tool calls (up to 5 rounds).

        Critical: we preserve the *original* Part objects from each chunk so that
        ``thought_signature`` fields on function-call parts are never stripped.
        Reconstructing parts via ``from_function_call(name, args)`` loses the
        signature and causes a 400 on the next turn.
        """
        types = self.types
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=1.0,  # required for thinking models
            max_output_tokens=16000,
            thinking_config=types.ThinkingConfig(include_thoughts=True),
            tools=[CHAT_TOOLS_SCHEMA],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="AUTO"),
            ),
        )

        for _round in range(5):
            stream = await self.client.aio.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=config,
            )

            # Collect parts in two buckets:
            #   model_parts  — non-thought parts to append as the model turn
            #                  (text parts collapsed into one; function_call parts
            #                   kept as-is to preserve thought_signature)
            #   tool_calls   — (FunctionCall, original_Part) pairs for execution
            collected_text = ""
            model_parts: list[types.Part] = []         # function_call parts (original)
            tool_calls: list[tuple[types.FunctionCall, types.Part]] = []

            async for chunk in stream:
                if not chunk.candidates:
                    continue
                for part in chunk.candidates[0].content.parts:
                    if getattr(part, "thought", False):
                        # Thinking token — stream to frontend, do NOT add to model_parts
                        if part.text:
                            yield ThinkingEvent(delta=part.text).model_dump()
                    elif getattr(part, "function_call", None) is not None:
                        # Keep the original Part so thought_signature is intact
                        tool_calls.append((part.function_call, part))
                        model_parts.append(part)
                    elif part.text:
                        yield TokenEvent(delta=part.text).model_dump()
                        collected_text += part.text

            if not tool_calls:
                # Pure text response — no more rounds needed
                return

            # Build the model turn: text first, then function-call parts (original)
            turn_parts: list[types.Part] = []
            if collected_text:
                turn_parts.append(types.Part.from_text(text=collected_text))
            turn_parts.extend(model_parts)
            contents.append(types.Content(role="model", parts=turn_parts))

            # Execute each tool and collect function-response parts
            response_parts: list[types.Part] = []
            for fc, _original_part in tool_calls:
                tool_name = fc.name
                tool_args = dict(fc.args) if fc.args else {}
                label = _TOOL_LABELS.get(tool_name, f"Running {tool_name}…")

                yield ToolStartEvent(tool=tool_name, label=label).model_dump()

                tool_fn = TOOL_REGISTRY.get(tool_name)
                if tool_fn is None:
                    result_data: dict[str, Any] = {"error": f"Unknown tool: {tool_name}"}
                else:
                    try:
                        if inspect.iscoroutinefunction(tool_fn):
                            result_data = await tool_fn(tool_args, snap)
                        else:
                            result_data = tool_fn(tool_args, snap)
                    except Exception as exc:
                        logger.exception(f"[chat] Tool {tool_name} failed")
                        result_data = {"error": str(exc)}

                # ── DEBUG: log full tool result to backend console ─────────────
                import json as _json
                logger.info(
                    "[chat][tool:%s] result =\n%s",
                    tool_name,
                    _json.dumps(result_data, indent=2, default=str),
                )
                if "convergence" in result_data:
                    logger.info(
                        "[chat][tool:%s] convergence_assessment = %s",
                        tool_name,
                        result_data["convergence"],
                    )
                if "chart" in result_data:
                    chart = result_data["chart"]
                    logger.info(
                        "[chat][tool:%s] chart → type=%s lines=%s data_points=%d convergence=%s",
                        tool_name,
                        chart.get("type"),
                        chart.get("lines"),
                        len(chart.get("data", [])),
                        chart.get("convergence"),
                    )
                # ── END DEBUG ───────────────────────────────────────────────────

                yield ToolResultEvent(tool=tool_name, data=result_data).model_dump()

                # Emit frontend-renderable artifacts (charts, reports)
                async for artifact_event in self._emit_artifacts(tool_name, result_data):
                    yield artifact_event

                # Strip large frontend-only keys before feeding back to the LLM
                keys_to_strip: list[str] = []
                if "chart" in result_data:
                    keys_to_strip.append("chart")
                if "report_markdown" in result_data:
                    keys_to_strip.extend(["report_markdown", "report_data"])
                if "report_request_payload" in result_data:
                    keys_to_strip.append("report_request_payload")

                llm_result = {k: v for k, v in result_data.items() if k not in keys_to_strip}
                if "report_markdown" in result_data:
                    llm_result["report_generated"] = True
                    llm_result["sections"] = result_data.get("sections_included", [])

                response_parts.append(
                    types.Part.from_function_response(name=tool_name, response=llm_result)
                )

            contents.append(types.Content(role="user", parts=response_parts))

    async def _emit_artifacts(
        self, tool_name: str, result_data: dict[str, Any],
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield frontend-renderable artifact events from a tool result."""
        if "chart" in result_data:
            yield ArtifactEvent(
                kind="chart",
                content=json.dumps(result_data["chart"]),
            ).model_dump()

        if "report_markdown" in result_data:
            yield ArtifactEvent(
                kind="report",
                content={
                    "markdown": result_data["report_markdown"],
                    "data": result_data.get("report_data", {}),
                    "sections": result_data.get("sections_included", []),
                },
            ).model_dump()

        if "report_request_payload" in result_data:
            yield ArtifactEvent(
                kind="report_request",
                content=result_data["report_request_payload"],
            ).model_dump()

    async def _stream_response_only(
        self,
        *,
        system_prompt: str,
        contents: list[Any],
        snap: SimulationSnapshot,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream a response-only turn (no tool calling).

        Used when the query analyzer pre-executed tools with high confidence.
        The LLM's job is to explain the pre-computed results in natural language.
        """
        types = self.types
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=1.0,
            max_output_tokens=16000,
            thinking_config=types.ThinkingConfig(include_thoughts=True),
        )

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
                        yield ThinkingEvent(delta=part.text).model_dump()
                elif part.text:
                    yield TokenEvent(delta=part.text).model_dump()

    # ------------------------------------------------------------------
    # Setup conversation (precheck mode)
    # ------------------------------------------------------------------

    async def _stream_setup_conversation(
        self, request: ChatRequest,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream a setup conversation in precheck mode.

        Mirrors PrecheckService._stream_conversation but emits chat-compatible
        events (TokenEvent, ThinkingEvent, ArtifactEvent) instead of precheck
        event types (chat_token, ready_to_analyze, etc.).
        """
        from simd_agent.precheck.service import PrecheckService

        # ── Summarize if history is too long ─────────────────────────────
        history = [{"role": m.role, "content": m.content} for m in request.history]
        summary = request.conversation_summary
        token_est = PrecheckService._estimate_token_count(history, summary, request.message)

        if token_est > CONVERSATION_TOKEN_LIMIT and history:
            logger.info("[chat/setup] Token estimate %d > %d, summarizing", token_est, CONVERSATION_TOKEN_LIMIT)
            precheck_svc = PrecheckService()
            summary = await precheck_svc._summarize_conversation(history, summary)
            yield ArtifactEvent(kind="conversation_summary", content=summary).model_dump()
            history = []

        # ── Build context block ──────────────────────────────────────────
        ctx_parts: list[str] = []
        mesh_info = request.mesh_info
        if mesh_info:
            patch_names = ", ".join(p.get("name", "") for p in mesh_info.get("patches", []))
            cells = mesh_info.get("checkMesh", {}).get("cells", "?")
            file_name = mesh_info.get("fileName", "mesh")
            ctx_parts.append(
                f"Mesh uploaded: **{file_name}** ({cells} cells, patches: {patch_names})"
            )
        elif request.has_mesh:
            ctx_parts.append("Mesh uploaded (details not provided).")
        else:
            ctx_parts.append("No mesh uploaded yet.")

        if summary:
            ctx_parts.append(f"\nPrior conversation summary:\n{summary}")

        sim_ctx = request.simulation_context
        if sim_ctx and sim_ctx.get("precheckSummary"):
            ctx_parts.append(
                "\n**Analysis already completed.** The user has already gone through "
                "the simulation planning and analysis phase. Do NOT call "
                "signal_ready_to_analyze again unless the user explicitly asks to "
                "change the simulation, modify parameters, or re-run the analysis. "
                "Just answer their questions directly."
            )

        system_prompt = PrecheckService.CONVERSATION_SYSTEM_PROMPT.replace(
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
            role="user", parts=[types.Part.from_text(text=request.message)],
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

        try:
            stream = await self.client.aio.models.generate_content_stream(
                model=CONVERSATION_MODEL,
                contents=contents,
                config=config,
            )

            async for chunk in stream:
                if not chunk.candidates:
                    continue
                for part in chunk.candidates[0].content.parts:
                    if getattr(part, "thought", False):
                        if part.text:
                            yield ThinkingEvent(delta=part.text).model_dump()
                    elif getattr(part, "function_call", None) is not None:
                        fc = part.function_call
                        if fc.name == "signal_ready_to_analyze":
                            args = dict(fc.args) if fc.args else {}
                            summary_text = args.get("summary", "")
                            logger.info("[chat/setup] LLM signaled ready — %s", summary_text[:80])
                            yield ArtifactEvent(
                                kind="ready_to_analyze",
                                content={"summary": summary_text},
                            ).model_dump()
                    elif part.text:
                        yield TokenEvent(delta=part.text).model_dump()

        except Exception as exc:
            logger.exception("[chat/setup] Conversation streaming failed: %s", exc)
            yield ErrorEvent(message=str(exc)).model_dump()

    # ------------------------------------------------------------------
    # Delegate to precheck analysis pipeline
    # ------------------------------------------------------------------

    async def _delegate_to_precheck(
        self, request: ChatRequest,
    ) -> AsyncIterator[dict[str, Any]]:
        """Delegate to the full precheck analysis pipeline (4-pass).

        Converts a ChatRequest into a PrecheckRequest, runs the analysis
        pipeline, and passes events through as-is.  The frontend already
        handles both chat and precheck event types.
        """
        from simd_agent.precheck.service import get_precheck_service
        from simd_agent.precheck.models import PrecheckRequest, MeshInfo

        # Build PrecheckRequest from ChatRequest fields
        mesh = None
        if request.mesh_info:
            try:
                mesh = MeshInfo(**request.mesh_info)
            except Exception:
                logger.warning("[chat/precheck] Failed to parse mesh_info, proceeding without")

        precheck_req = PrecheckRequest(
            prompt=request.message,
            has_mesh=request.has_mesh,
            mesh_info=mesh,
            history=[{"role": m.role, "content": m.content} for m in request.history],
            conversation_summary=request.conversation_summary,
            confirm_analysis=request.confirm_analysis,
            simulation_context=request.simulation_context,
        )

        precheck_svc = get_precheck_service()
        async for event in precheck_svc.analyze_stream(precheck_req):
            yield event

    # ------------------------------------------------------------------
    # Suggested follow-up actions
    # ------------------------------------------------------------------

    def _suggest_actions(self, snap: SimulationSnapshot, user_msg: str) -> list[str]:
        """Generate contextual follow-up suggestions based on available data."""
        actions: list[str] = []
        lower = user_msg.lower()

        if snap.sim_progress and "residual" not in lower:
            actions.append("Show residual plot")
        if snap.sim_progress and "pressure" not in lower and any(
            s.get("field_ranges") or s.get("fieldRanges") for s in snap.sim_progress[:20]
        ):
            actions.append("Plot pressure over time")
        if snap.vtk_result:
            actions.append("Explain the velocity field")
        if snap.patches:
            actions.append("Review boundary conditions")
        if len(snap.all_runs) > 1 and "compare" not in lower:
            actions.append("Compare runs")
        if not snap.sim_progress and not snap.vtk_result:
            if snap.physics:
                actions.append("Calculate Reynolds number")
            actions.append("Generate simulation report")
        else:
            actions.append("Generate full report")
        if snap.generated_files:
            actions.append("Show controlDict settings")

        return actions[:4]


_service: ChatService | None = None


def get_chat_service() -> ChatService:
    global _service
    if _service is None:
        _service = ChatService()
    return _service
