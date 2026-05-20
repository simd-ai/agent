# simd_agent/run/value_filler/__init__.py
"""LLM-driven value filler for OpenFOAM ``0/*`` field files.

Solver-agnostic: handles both single-region (``0/<field>``) and
multi-region / CHT (``0/<region>/<field>``) targets through the same
pipeline.  See the package's submodule docstrings for the breakdown:

  * :mod:`contexts`   — per-file context builders (single / multi auto-route).
  * :mod:`prompts`    — section-based prompt builder (rules are shared).
  * :mod:`validation` — response parsing + structural sanity check.
  * :mod:`filler`     — public entry point, parallel LLM calls, log spine.

Public surface is intentionally narrow — only :func:`fill_field_values`
is exported.  Internal modules are importable by tests but not part of
the package contract.
"""

from simd_agent.run.value_filler.filler import fill_field_values

__all__ = ["fill_field_values"]
