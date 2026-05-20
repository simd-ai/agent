# simd_agent/run/single_region/validator.py
"""Single-region post-generation validator — re-export façade.

Surfaces :func:`validate_generated_files` and the
:class:`ValidationIssue` dataclass under the package-qualified path
``simd_agent.run.single_region.validator``.

The actual validator (~1800 lines of solver-specific auto-fix logic
covering ``rhoSimpleFoam``, ``rhoPimpleFoam``, ``buoyantSimpleFoam``,
``buoyantBoussinesq*``, the compressible VOF family, etc.) lives in
:mod:`simd_agent.run.genai_codegen` because it shares helper functions
with the codegen path.  In the plugin-centric refactor this code is
reached via :meth:`SolverPlugin.validate_full`; the orchestrator only
calls :func:`validate_generated_files` directly as a fallback when a
solver has no registered plugin.
"""

from simd_agent.run.genai_codegen import (
    ValidationIssue,
    validate_generated_files,
)

__all__ = ["ValidationIssue", "validate_generated_files"]
