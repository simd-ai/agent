# simd_agent/llm/gemini/__init__.py
"""Gemini LLM provider — default provider for SIMD Agent."""

from simd_agent.llm.gemini.provider import GeminiProvider

provider_plugin = GeminiProvider()

__all__ = ["provider_plugin"]
