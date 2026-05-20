# simd_agent/llm/ollama/_translate.py
"""Translate between Gemini-shaped objects and Ollama wire format.

Two directions:
    1. inbound  — Gemini-shaped ``contents`` + ``config`` (built by the
       rest of the codebase) → Ollama ``messages`` / ``tools`` /
       ``options`` / ``format`` kwargs.
    2. outbound — Ollama response dicts → Gemini-shaped Response objects
       so call sites that read ``resp.text`` /
       ``resp.candidates[0].content.parts[i].function_call`` work
       unchanged.

The codebase passes both this module's shim types *and* real
``google.genai.types`` objects through the provider, so the inbound
converters use duck-typing rather than isinstance checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from simd_agent.llm.ollama._types_shim import (
    Content,
    FunctionCall,
    Part,
)


# ── inbound: shim → Ollama ───────────────────────────────────────


def to_ollama_messages(
    contents: Any,
    system_instruction: str | None = None,
) -> list[dict[str, Any]]:
    """Flatten Gemini ``Content`` / ``Part`` lists into Ollama messages.

    Accepts:
        - a plain string (treated as a single user turn)
        - a list of ``Content`` / dict / string entries
        - a single ``Content`` instance
    """
    messages: list[dict[str, Any]] = []
    if system_instruction:
        messages.append({"role": "system", "content": _as_text(system_instruction)})

    if contents is None:
        return messages
    if isinstance(contents, str):
        messages.append({"role": "user", "content": contents})
        return messages
    if not isinstance(contents, list):
        contents = [contents]

    for c in contents:
        if isinstance(c, str):
            messages.append({"role": "user", "content": c})
            continue
        role = _attr(c, "role", "user")
        parts = _attr(c, "parts", None) or []
        msg = _parts_to_message(role, parts)
        if msg is not None:
            messages.append(msg)
    return messages


def _parts_to_message(role: str, parts: Iterable[Any]) -> dict[str, Any] | None:
    """Collapse a Gemini Content (role + parts list) into one Ollama message.

    Gemini splits assistant turns into a sequence of text Parts and
    function_call Parts; Ollama wants ``content`` + ``tool_calls`` on a
    single message.
    """
    text_chunks: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    for p in parts:
        text = _attr(p, "text", None)
        fc = _attr(p, "function_call", None)
        fr = _attr(p, "function_response", None)
        if text:
            text_chunks.append(text)
        if fc is not None:
            tool_calls.append(
                {
                    "function": {
                        "name": _attr(fc, "name", ""),
                        "arguments": dict(_attr(fc, "args", {}) or {}),
                    }
                }
            )
        if fr is not None:
            tool_results.append(
                {
                    "name": _attr(fr, "name", ""),
                    "response": dict(_attr(fr, "response", {}) or {}),
                }
            )

    ollama_role = _gemini_role_to_ollama(role)
    if tool_results:
        # Each function_response becomes its own tool message in Ollama.
        # If multiple are present we return the first here; callers
        # should split beforehand, but this is the safe fallback.
        return {
            "role": "tool",
            "content": _stringify(tool_results[0]["response"]),
            "name": tool_results[0]["name"],
        }

    msg: dict[str, Any] = {"role": ollama_role, "content": "\n".join(text_chunks)}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def _gemini_role_to_ollama(role: str) -> str:
    if role in ("model", "assistant"):
        return "assistant"
    if role == "function":
        return "tool"
    return "user"


def to_ollama_tools(tools: Any) -> list[dict[str, Any]]:
    """Convert ``[Tool(function_declarations=[FunctionDeclaration, ...])]``
    into Ollama's OpenAI-compatible tools list.

    Duck-typed: works on both this package's shim Tool/FunctionDeclaration
    and ``google.genai.types`` real ones.
    """
    out: list[dict[str, Any]] = []
    if not tools:
        return out
    for tool in tools:
        for fd in _attr(tool, "function_declarations", []) or []:
            params = _schema_to_jsonschema(_attr(fd, "parameters", None))
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": _attr(fd, "name", ""),
                        "description": _attr(fd, "description", "") or "",
                        "parameters": params or {"type": "object", "properties": {}},
                    },
                }
            )
    return out


def _schema_to_jsonschema(schema: Any) -> dict[str, Any] | None:
    """Convert a ``Schema`` (shim or real Gemini) to JSON Schema.

    Gemini uses uppercase type tokens (``"OBJECT"``, ``"STRING"``);
    OpenAI/Ollama want lowercase JSON-schema strings.
    """
    if schema is None:
        return None
    out: dict[str, Any] = {}
    t = _attr(schema, "type", None)
    if t is not None:
        out["type"] = str(t).lower()
    desc = _attr(schema, "description", None)
    if desc:
        out["description"] = desc
    props = _attr(schema, "properties", None)
    if props:
        out["properties"] = {
            k: _schema_to_jsonschema(v) or {} for k, v in props.items()
        }
    req = _attr(schema, "required", None)
    if req:
        out["required"] = list(req)
    items = _attr(schema, "items", None)
    if items is not None:
        out["items"] = _schema_to_jsonschema(items) or {}
    enum = _attr(schema, "enum", None)
    if enum:
        out["enum"] = list(enum)
    nullable = _attr(schema, "nullable", None)
    if nullable:
        out["nullable"] = True
    return out


def build_options_and_format(config: Any) -> tuple[dict[str, Any], Any]:
    """Extract Ollama ``options`` and ``format`` from a Gemini config.

    Returns ``(options, format)`` where ``format`` is:
        - ``None`` for free text
        - ``"json"`` when response_mime_type is application/json without schema
        - a JSON-schema dict when response_schema is set (constrained decoding)
    """
    if config is None:
        return {}, None
    opts: dict[str, Any] = {}
    if (t := _attr(config, "temperature", None)) is not None:
        opts["temperature"] = float(t)
    if (n := _attr(config, "max_output_tokens", None)) is not None:
        opts["num_predict"] = int(n)
    if (tp := _attr(config, "top_p", None)) is not None:
        opts["top_p"] = float(tp)
    if (tk := _attr(config, "top_k", None)) is not None:
        opts["top_k"] = int(tk)
    if (stop := _attr(config, "stop_sequences", None)) is not None:
        opts["stop"] = list(stop)
    if (seed := _attr(config, "seed", None)) is not None:
        opts["seed"] = int(seed)

    fmt: Any = None
    schema = _attr(config, "response_schema", None)
    mime = _attr(config, "response_mime_type", None)
    if schema is not None:
        fmt = _schema_to_jsonschema(schema) if _looks_like_schema(schema) else schema
    elif mime == "application/json":
        fmt = "json"
    return opts, fmt


# ── outbound: Ollama → Gemini-shaped response ────────────────────


@dataclass
class _Candidate:
    content: Content
    finish_reason: str | None = None


@dataclass
class GenerateContentResponse:
    """Mimics ``google.genai.types.GenerateContentResponse``.

    Exposes ``.text`` (concatenated text across parts), ``.candidates``
    (list with one entry holding parts), and ``.function_calls``
    (convenience accessor used by some call sites).
    """

    candidates: list[_Candidate] = field(default_factory=list)

    @property
    def text(self) -> str:
        if not self.candidates:
            return ""
        return "".join(p.text or "" for p in self.candidates[0].content.parts if p.text)

    @property
    def function_calls(self) -> list[FunctionCall]:
        if not self.candidates:
            return []
        return [
            p.function_call
            for p in self.candidates[0].content.parts
            if p.function_call is not None
        ]


def from_ollama_response(resp: Any) -> GenerateContentResponse:
    """Wrap a non-streaming Ollama chat response in Gemini's shape."""
    msg = _get(resp, "message") or {}
    parts: list[Part] = []
    content = _get(msg, "content")
    if content:
        parts.append(Part(text=content))
    for tc in _get(msg, "tool_calls") or []:
        fn = _get(tc, "function") or {}
        args = _get(fn, "arguments") or {}
        if isinstance(args, str):
            try:
                import json
                args = json.loads(args)
            except Exception:
                args = {"_raw": args}
        parts.append(
            Part(function_call=FunctionCall(name=_get(fn, "name") or "", args=dict(args)))
        )
    finish = _get(resp, "done_reason") or ("stop" if _get(resp, "done") else None)
    return GenerateContentResponse(
        candidates=[_Candidate(content=Content(role="model", parts=parts), finish_reason=finish)]
    )


def from_ollama_chunk(chunk: Any) -> GenerateContentResponse:
    """Wrap a streaming Ollama chunk in Gemini's shape.

    Each chunk has the same shape as a full response but with partial
    content; tool calls typically arrive in the terminal chunk.
    """
    return from_ollama_response(chunk)


# ── helpers ──────────────────────────────────────────────────────


def _attr(obj: Any, name: str, default: Any) -> Any:
    """Read attribute or dict key; tolerant of both shim and real types."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _get(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _as_text(obj: Any) -> str:
    if isinstance(obj, str):
        return obj
    parts = _attr(obj, "parts", None)
    if parts:
        return "\n".join(_attr(p, "text", "") or "" for p in parts)
    return str(obj)


def _stringify(obj: Any) -> str:
    import json
    try:
        return json.dumps(obj)
    except Exception:
        return str(obj)


def _looks_like_schema(obj: Any) -> bool:
    """True if ``obj`` is a Gemini-style Schema rather than raw JSON-schema dict."""
    if isinstance(obj, dict):
        # Already a JSON-schema dict — pass through unchanged.
        return False
    return _attr(obj, "type", None) is not None or _attr(obj, "properties", None) is not None
