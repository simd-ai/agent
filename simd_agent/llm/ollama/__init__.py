# simd_agent/llm/ollama/__init__.py
"""Ollama LLM provider — local inference via the Ollama HTTP server.

Default model is ``gemma4`` (Google's Gemma 4, released April 2026,
Apache 2.0).  Pull it with ``ollama pull gemma4`` before first use.

Set ``DEFAULT_PROVIDER=ollama`` in ``.env`` to make this the active
provider; ``OLLAMA_HOST`` (default ``http://localhost:11434``) and
``OLLAMA_MODEL`` (default ``gemma4``) override the endpoint and tag.
"""

from simd_agent.llm.ollama.provider import OllamaProvider

provider_plugin = OllamaProvider()

__all__ = ["provider_plugin"]
