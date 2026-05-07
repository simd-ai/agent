# simd_agent/chat/query_analyzer.py
"""LLM-based query analyzer — classifies user intent before the main chat call.

Runs a fast, cheap LLM call (gemini-flash) to understand the user's question,
resolve pronouns from conversation history, and produce a structured QueryIntent
that the backend uses to deterministically route tool calls.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from simd_agent.chat.models import DataNeeds, QueryIntent, ToolCallPlan
from simd_agent.llm import get_provider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Analyzer system prompt
# ---------------------------------------------------------------------------

_ANALYZER_SYSTEM_PROMPT = """\
You are a query intent classifier for a CFD simulation platform chat assistant.

Given the user's message and recent conversation history, classify the intent so
the backend can decide which tools to call and what data to fetch.

The user may write in broken English, have typos, or use informal language —
focus on MEANING, not grammar.  If the user refers to something from earlier
messages ("plot it", "show me that", "the pressure"), resolve what they mean
from the conversation context.

## Categories

- **setup**: user is describing or configuring a NEW simulation — talking about
  what fluid to use, what boundary conditions, what physics to enable, uploading
  a mesh, or asking what they need before running.  Use this when the context is
  clearly about SETTING UP a simulation, not analyzing results of one that ran.
- **data_plot**: user wants a chart/graph of PHYSICAL field values (pressure,
  temperature, velocity magnitude, etc.) over time/iterations.
- **residuals**: user wants residual/convergence plots or asks about convergence.
- **data_query**: user asks about simulation results, values, statistics, overview,
  recommendations, or "how did the simulation go".
- **file_inspect**: user wants to see/review/explain an OpenFOAM case file
  (e.g. "show me 0/U", "what does fvSolution look like").
- **cross_run**: user wants to compare results across multiple runs.
- **report**: user wants a full report, PDF export, or complete simulation summary.
- **troubleshoot**: user asks about errors, failures, warnings, or why something
  went wrong.
- **theory**: user asks a general CFD/physics/engineering question that does NOT
  require simulation data.
- **general**: anything else, or greetings/chitchat.

**Important:** The request may include ``mode: "precheck"`` indicating the user
is in the simulation setup view.  Use this as a HINT, but always classify by
actual intent.  If the user is in precheck mode but asks "plot the velocity",
that is ``data_plot``, NOT ``setup``.

## Available tools

| Tool | Purpose | Key args |
|------|---------|----------|
| plot_field_values | Plot physical field min/max over time | fields, metric (min/max/both/range) |
| plot_field_over_iterations | Plot residuals/Courant/continuity | fields |
| compute_residual_trend | Residual history analysis | fields |
| compute_field_stats | Field statistics from VTK | field, patch |
| extract_velocity_profile | Velocity at a patch | patch, axis |
| query_simulation_results | Broad results overview | question |
| read_generated_file | Read an OpenFOAM file | path |
| generate_report | Simulation report PDF | report_type (standard/expert, default standard), focus |
| plot_patch_values | Plot patch-averaged values, pressure/temperature drop | fields, patches, quantity (values/drop) |
| plot_volume_values | Plot domain-wide volume averages/integrals | fields |
| compare_runs | Cross-run comparison chart | fields, data_type (residuals/field_values), metric |
| run_python_analysis | Custom computation | code, description |
| analyze_chart | Explain a chart | chart_type, field |

## Output format

Return ONLY valid JSON (no markdown fences, no extra text):
{
  "category": "one of the categories above",
  "resolved_subject": "what the user is asking about, resolved from context if needed",
  "tool_plan": [
    {"tool": "tool_name", "args": {"param": "value"}}
  ],
  "data_needs": {
    "sim_progress": false,
    "field_ranges": false,
    "vtk_result": false,
    "generated_files": false,
    "cross_run": false,
    "patches": false,
    "mesh_info": false
  },
  "confidence": 0.0
}

## Rules

1. **tool_plan** should list the tool(s) the backend should call, with pre-filled
   args derived from the user's intent.  For "theory" and "general" categories,
   leave tool_plan empty.
2. **data_needs** tells the backend which data sources to fetch from the database.
   Only set fields to true when the tools in tool_plan actually need that data.
3. **confidence** (0.0–1.0): how certain you are about the classification.
   Use >= 0.8 for clear, unambiguous queries.  Use 0.4–0.7 for ambiguous ones
   where the main LLM should have all tools available as fallback.
4. **resolved_subject**: if the user says "plot it" or "show me that", resolve
   what "it"/"that" refers to from the conversation history.  If unclear, set
   confidence lower.
5. For **data_query** (broad questions like "how did the simulation go", "what are
   the results"), use query_simulation_results as the tool.
6. For **residuals**, prefer plot_field_over_iterations (visual) unless the user
   explicitly asks for numbers/statistics (then use compute_residual_trend).
7. For **data_plot** with a physical field (pressure, temperature, velocity):
   - DEFAULT: use ``plot_patch_values`` with quantity='values' — this is what engineers
     typically mean ("what is the pressure at inlet/outlet, how does it evolve").
     Set data_needs.sim_progress=true.
   - ONLY use ``plot_field_values`` when the user explicitly asks for "min/max",
     "global values", "domain-wide min/max", or "field range".
   - ONLY use ``plot_volume_values`` when the user explicitly asks for "domain average",
     "volume average", "bulk", or "domain-wide integral".
8. For **pressure drop**, **temperature drop**, or questions about values **at a specific
   patch** (inlet, outlet, wall), use plot_patch_values with quantity='drop' for drops
   or quantity='values' for raw patch averages.  Set data_needs.sim_progress=true.
9. When the user asks for multiple things (e.g. "show me the pressure and explain
   the boundary conditions"), pick the PRIMARY intent and set confidence lower.
10. **Never plan multiple chart tools for the same field.** For example, do NOT
    plan both plot_field_values and plot_field_over_iterations for pressure.
    Pick the single best tool: plot_field_values for physical values, or
    plot_field_over_iterations for residual/convergence data.
"""

# ---------------------------------------------------------------------------
# Analyzer class
# ---------------------------------------------------------------------------


class QueryAnalyzer:
    """Classifies user queries via a fast LLM call before the main chat turn."""

    def __init__(self) -> None:
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

    async def analyze(
        self,
        message: str,
        history: list[dict[str, str]],
        *,
        mode: str = "chat",
    ) -> QueryIntent:
        """Classify a user message and return structured intent.

        Args:
            message: The current user message.
            history: Recent conversation turns, each with ``role`` and ``content``.
                     At most the last 5 turns are used.
            mode: ``"chat"`` or ``"precheck"`` — hint for the classifier.

        Returns:
            A ``QueryIntent`` with category, tool plan, data needs, and confidence.
            On any failure, returns a low-confidence fallback intent so the main
            LLM gets all tools.
        """
        user_input = self._build_input(message, history, mode=mode)
        logger.info("[query_analyzer] input:\n%s", user_input)

        try:
            response = await self.client.aio.models.generate_content(
                model=self.model,
                contents=user_input,
                config=self.types.GenerateContentConfig(
                    system_instruction=_ANALYZER_SYSTEM_PROMPT,
                    temperature=0.0,
                    max_output_tokens=1024,
                    response_mime_type="application/json",
                ),
            )
            raw = (response.text or "").strip()
            logger.info("[query_analyzer] raw LLM output:\n%s", raw)
            parsed = self._parse_json(raw)

            if parsed is None:
                logger.warning("[query_analyzer] Failed to parse JSON: %s", raw[:200])
                return self._fallback_intent(message)

            intent = self._build_intent(parsed)
            logger.info(
                "[query_analyzer] result: category=%s confidence=%.2f subject=%r tools=%s",
                intent.category, intent.confidence, intent.resolved_subject,
                [t.tool for t in intent.tool_plan],
            )
            return intent

        except Exception as exc:
            logger.exception("[query_analyzer] LLM call failed: %s", exc)
            return self._fallback_intent(message)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_input(self, message: str, history: list[dict[str, str]], *, mode: str = "chat") -> str:
        """Format the user message + recent history as a single text block."""
        recent = history[-5:] if len(history) > 5 else history
        parts: list[str] = []

        if mode != "chat":
            parts.append(f"Mode: {mode}")
            parts.append("")

        if recent:
            parts.append("Recent conversation:")
            for turn in recent:
                role = turn.get("role", "user").capitalize()
                content = turn.get("content", "")
                # Truncate long assistant responses to save tokens
                if role == "Assistant" and len(content) > 300:
                    content = content[:300] + "..."
                parts.append(f"{role}: {content}")
            parts.append("")

        parts.append(f"Current message: {message}")
        return "\n".join(parts)

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any] | None:
        """Extract and parse a JSON object from the LLM response.

        Handles truncated output by attempting to close open braces/brackets.
        """
        # Strip markdown code fences if present
        text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
        # Find the outermost JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # Truncated JSON — try to repair by closing open braces/brackets
        # Find the first '{' and work from there
        start = text.find("{")
        if start == -1:
            return None
        fragment = text[start:]
        # Strip trailing incomplete strings/values
        fragment = re.sub(r',\s*"[^"]*$', "", fragment)
        fragment = re.sub(r':\s*"[^"]*$', ': ""', fragment)
        fragment = re.sub(r':\s*$', ': null', fragment)
        # Strip trailing commas (truncated JSON often leaves a dangling
        # comma when the next key-value pair was cut off, e.g.
        # {"a": 1, "b": false,   ← comma with no following pair)
        fragment = re.sub(r',\s*$', '', fragment)
        # Close open brackets/braces
        opens = 0
        open_sq = 0
        for ch in fragment:
            if ch == "{":
                opens += 1
            elif ch == "}":
                opens -= 1
            elif ch == "[":
                open_sq += 1
            elif ch == "]":
                open_sq -= 1
        fragment += "]" * max(0, open_sq)
        fragment += "}" * max(0, opens)
        try:
            return json.loads(fragment)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _build_intent(data: dict[str, Any]) -> QueryIntent:
        """Convert raw parsed JSON into a validated QueryIntent."""
        # Category
        category = data.get("category", "general")
        valid_categories = {
            "setup", "data_plot", "residuals", "data_query", "file_inspect",
            "cross_run", "report", "troubleshoot", "theory", "general",
        }
        if category not in valid_categories:
            category = "general"

        # Tool plan
        raw_plan = data.get("tool_plan") or []
        tool_plan: list[ToolCallPlan] = []
        for entry in raw_plan:
            if isinstance(entry, dict) and "tool" in entry:
                tool_plan.append(ToolCallPlan(
                    tool=entry["tool"],
                    args=entry.get("args", {}),
                ))

        # Data needs
        raw_needs = data.get("data_needs", {})
        data_needs = DataNeeds(
            sim_progress=bool(raw_needs.get("sim_progress", False)),
            field_ranges=bool(raw_needs.get("field_ranges", False)),
            vtk_result=bool(raw_needs.get("vtk_result", False)),
            generated_files=bool(raw_needs.get("generated_files", False)),
            cross_run=bool(raw_needs.get("cross_run", False)),
            patches=bool(raw_needs.get("patches", False)),
            mesh_info=bool(raw_needs.get("mesh_info", False)),
        )

        # Confidence
        try:
            confidence = float(data.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.5

        return QueryIntent(
            category=category,
            resolved_subject=data.get("resolved_subject", ""),
            tool_plan=tool_plan,
            data_needs=data_needs,
            confidence=confidence,
        )

    @staticmethod
    def _fallback_intent(message: str) -> QueryIntent:
        """Return a low-confidence intent that triggers full-tool fallback."""
        return QueryIntent(
            category="general",
            resolved_subject=message[:100],
            tool_plan=[],
            data_needs=DataNeeds(),
            confidence=0.0,
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_analyzer: QueryAnalyzer | None = None


def get_query_analyzer() -> QueryAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = QueryAnalyzer()
    return _analyzer
