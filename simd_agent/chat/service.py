# simd_agent/chat/service.py
"""ChatService — streaming Gemini assistant with tool-calling for CFD analysis."""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

from google import genai
from google.genai import types

from simd_agent.chat.db import build_snapshot, persist_chat_message
from simd_agent.chat.models import ArtifactEvent, ChatRequest, DoneEvent, ErrorEvent, ThinkingEvent, TokenEvent, ToolResultEvent, ToolStartEvent
from simd_agent.chat.prompts import SYSTEM_PROMPT
from simd_agent.chat.tools import CHAT_TOOLS_SCHEMA, TOOL_REGISTRY, SimulationSnapshot
from simd_agent.settings import get_settings

logger = logging.getLogger(__name__)

_TOOL_LABELS: dict[str, str] = {
    "compute_field_stats": "Computing field statistics…",
    "compute_residual_trend": "Analyzing residual convergence…",
    "extract_velocity_profile": "Extracting velocity profile…",
    "run_python_analysis": "Running computation…",
    "generate_report": "Generating simulation report…",
    "analyze_chart": "Analyzing chart data…",
    "read_generated_file": "Reading case file…",
}


class ChatService:
    """Streaming chat service backed by Gemini with function-calling tools."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client: genai.Client | None = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            api_key = self.settings.gemini_api_key
            if not api_key:
                raise ValueError("GEMINI_API_KEY not configured")
            self._client = genai.Client(api_key=api_key)
        return self._client

    @property
    def model(self) -> str:
        return self.settings.gemini_model

    async def handle_turn(self, request: ChatRequest) -> AsyncIterator[dict[str, Any]]:
        """Process one user turn and yield chat events as dicts."""
        try:
            snap = await build_snapshot(request)
        except Exception as exc:
            logger.exception("[chat] Failed to build snapshot")
            yield ErrorEvent(message=f"Failed to load simulation data: {exc}").model_dump()
            yield DoneEvent().model_dump()
            return

        system_prompt = SYSTEM_PROMPT.replace(
            "{context_json}",
            json.dumps(snap.summary_dict(), indent=2, default=str),
        )

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

        sim_id = request.simulation_id or request.context.simulation_id
        if sim_id:
            await persist_chat_message(sim_id, "user", request.message)
            await persist_chat_message(sim_id, "assistant", full_response_text, suggested_actions)

    async def _stream_with_tools(
        self,
        *,
        system_prompt: str,
        contents: list[types.Content],
        snap: SimulationSnapshot,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream tokens from Gemini with thinking + tool calls (up to 5 rounds).

        Critical: we preserve the *original* Part objects from each chunk so that
        ``thought_signature`` fields on function-call parts are never stripped.
        Reconstructing parts via ``from_function_call(name, args)`` loses the
        signature and causes a 400 on the next turn.
        """
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

                # Emit frontend-renderable artifacts based on what the tool returned.
                # These are stripped before feeding back to the LLM (frontend-only data).
                keys_to_strip: list[str] = []

                if "chart" in result_data:
                    yield ArtifactEvent(
                        kind="chart",
                        content=_json.dumps(result_data["chart"]),
                    ).model_dump()
                    keys_to_strip.append("chart")

                if "report_markdown" in result_data:
                    # Emit a "report" artifact carrying both the markdown and the
                    # typed data block so the frontend can render / PDF-export it.
                    yield ArtifactEvent(
                        kind="report",
                        content=_json.dumps({
                            "markdown": result_data["report_markdown"],
                            "data": result_data.get("report_data", {}),
                            "sections": result_data.get("sections_included", []),
                        }),
                    ).model_dump()
                    # Strip large keys — LLM only needs a short acknowledgement
                    keys_to_strip.extend(["report_markdown", "report_data"])

                llm_result = {k: v for k, v in result_data.items() if k not in keys_to_strip}
                if "report_markdown" in result_data:
                    llm_result["report_generated"] = True
                    llm_result["sections"] = result_data.get("sections_included", [])

                response_parts.append(
                    types.Part.from_function_response(name=tool_name, response=llm_result)
                )

            contents.append(types.Content(role="user", parts=response_parts))

    def _suggest_actions(self, snap: SimulationSnapshot, user_msg: str) -> list[str]:
        """Generate contextual follow-up suggestions based on available data."""
        actions: list[str] = []
        lower = user_msg.lower()

        if snap.sim_progress and "converge" not in lower and "residual" not in lower:
            actions.append("Show residual convergence")
        if snap.vtk_result:
            actions.append("Explain the velocity field")
        if snap.patches:
            actions.append("Review boundary conditions")
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
