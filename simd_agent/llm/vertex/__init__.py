# simd_agent/llm/vertex/__init__.py
"""Vertex AI LLM provider — Gemini via Google Cloud Vertex AI."""

from simd_agent.llm.vertex.provider import VertexProvider

provider_plugin = VertexProvider()

__all__ = ["provider_plugin"]
