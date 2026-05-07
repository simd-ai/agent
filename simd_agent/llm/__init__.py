# simd_agent/llm/__init__.py
"""LLM provider abstraction layer.

Providers are auto-discovered from sub-packages (e.g. ``gemini/``).
To add a new provider, create ``simd_agent/llm/<name>/`` with::

    # __init__.py
    from simd_agent.llm.<name>.provider import MyProvider
    provider_plugin = MyProvider()

The registry picks it up automatically — no other file needs editing.

Quick start::

    from simd_agent.llm import get_provider

    provider = get_provider()                   # default (gemini)
    resp = await provider.generate("gemini-3-flash-preview", contents=...)
    print(resp.text)
"""

from simd_agent.llm.base import LLMProvider
from simd_agent.llm.registry import get_llm_registry, get_provider

__all__ = ["LLMProvider", "get_llm_registry", "get_provider"]
