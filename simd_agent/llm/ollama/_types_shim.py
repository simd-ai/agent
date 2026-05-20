# simd_agent/llm/ollama/_types_shim.py
"""Gemini-shaped type stand-ins for the Ollama provider.

The rest of the codebase builds prompts using ``provider.types.X(...)``
where ``X`` is one of Google's ``google.genai.types`` classes (Tool,
FunctionDeclaration, Schema, Content, Part, GenerateContentConfig,
ToolConfig, FunctionCallingConfig, ThinkingConfig).  To swap providers
without touching the call sites, this module exposes the same names
with the same constructor signatures, but as plain dataclasses.

The Ollama provider's ``generate()`` / ``generate_stream()`` read these
shimmed objects and translate them into Ollama's native wire format
(see ``_translate.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── prompt-building primitives ───────────────────────────────────


@dataclass
class Schema:
    """JSON-schema-style descriptor.

    Matches ``google.genai.types.Schema`` — ``type`` is a string token
    (``"OBJECT"``, ``"STRING"``, ``"ARRAY"``, ``"NUMBER"``,
    ``"INTEGER"``, ``"BOOLEAN"``) and nested fields are themselves
    ``Schema`` instances.
    """

    type: str | None = None
    description: str | None = None
    properties: dict[str, "Schema"] | None = None
    required: list[str] | None = None
    items: "Schema | None" = None
    enum: list[str] | None = None
    nullable: bool | None = None


@dataclass
class FunctionDeclaration:
    name: str
    description: str | None = None
    parameters: Schema | None = None


@dataclass
class Tool:
    function_declarations: list[FunctionDeclaration] = field(default_factory=list)


@dataclass
class FunctionCallingConfig:
    """Gemini has modes ``AUTO`` / ``ANY`` / ``NONE`` plus optional
    ``allowed_function_names`` to constrain forced calls.

    Ollama has no native equivalent; the provider passes ``tools=[...]``
    and post-filters tool calls against ``allowed_function_names`` when
    set.  Mode is best-effort: ``ANY`` is treated as "tools enabled and
    the model is expected to call one"; ``NONE`` drops the tools list.
    """

    mode: str = "AUTO"
    allowed_function_names: list[str] | None = None


@dataclass
class ToolConfig:
    function_calling_config: FunctionCallingConfig | None = None


@dataclass
class ThinkingConfig:
    """Gemini-only flag — Ollama ignores it.

    Some Gemma 4 builds expose configurable "thinking modes" but the
    knob isn't surfaced through Ollama's OpenAI-compatible chat API.
    Kept for source-compatibility only.
    """

    include_thoughts: bool = False
    thinking_budget: int | None = None


@dataclass
class Part:
    """A single content part.

    Mirrors ``google.genai.types.Part``: either ``text`` is set, or
    ``function_call`` is set (for assistant tool-call turns), or
    ``function_response`` is set (for tool-result turns).
    """

    text: str | None = None
    function_call: "FunctionCall | None" = None
    function_response: "FunctionResponse | None" = None
    thought: bool = False

    @classmethod
    def from_text(cls, text: str) -> "Part":
        return cls(text=text)

    @classmethod
    def from_function_call(cls, name: str, args: dict[str, Any]) -> "Part":
        return cls(function_call=FunctionCall(name=name, args=args))

    @classmethod
    def from_function_response(
        cls, name: str, response: dict[str, Any]
    ) -> "Part":
        return cls(function_response=FunctionResponse(name=name, response=response))


@dataclass
class FunctionCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class FunctionResponse:
    name: str
    response: dict[str, Any] = field(default_factory=dict)


@dataclass
class Content:
    role: str = "user"  # "user" | "model" | "function"
    parts: list[Part] = field(default_factory=list)


# ── generation config ────────────────────────────────────────────


@dataclass
class GenerateContentConfig:
    """Gemini-shaped generation knobs.

    Recognised fields map to Ollama as follows:
        - ``temperature``        → options.temperature
        - ``max_output_tokens``  → options.num_predict
        - ``top_p`` / ``top_k``  → options.top_p / options.top_k
        - ``system_instruction`` → prepended as a system message
        - ``tools``              → translated to Ollama's tool schema
        - ``tool_config``        → drives tool_choice / post-filtering
        - ``response_mime_type`` ``"application/json"`` → format="json"
        - ``response_schema``    → format=<JSON schema> (constrained decoding)
        - ``thinking_config``    → ignored (Gemini-only)
        - ``stop_sequences``     → options.stop
    """

    temperature: float | None = None
    max_output_tokens: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    system_instruction: str | Content | None = None
    tools: list[Tool] | None = None
    tool_config: ToolConfig | None = None
    response_mime_type: str | None = None
    response_schema: Any | None = None
    thinking_config: ThinkingConfig | None = None
    # Catch-all for forward-compat — silently ignored.
    candidate_count: int | None = None
    seed: int | None = None
