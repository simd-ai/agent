# simd_agent/run/enrichment/__init__.py
"""Config enrichment pipeline.

Public surface:

  * :func:`enrich_validated_config` — the single entry point the
    orchestrator calls after ``CFDLinter.lint()``.  Mutates the config
    in place and returns the diagnostic issue list.
  * :class:`EnrichmentContext`, :class:`EnrichmentIssue` — exported so
    custom test harnesses or future callers can compose their own
    step lists.

Everything else (the individual step modules, helpers) is internal —
import directly from the submodule only in tests.
"""

from simd_agent.run.enrichment.context import EnrichmentContext, EnrichmentIssue
from simd_agent.run.enrichment.pipeline import (
    DEFAULT_STEPS,
    enrich_validated_config,
)

__all__ = [
    "DEFAULT_STEPS",
    "EnrichmentContext",
    "EnrichmentIssue",
    "enrich_validated_config",
]
