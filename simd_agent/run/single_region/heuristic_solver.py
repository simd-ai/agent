# simd_agent/run/single_region/heuristic_solver.py
"""Heuristic single-region solver selection — re-export façade.

:func:`determine_solver` is the deterministic fallback that
:class:`SolverSelector` lands on when the LLM call fails or times out:
it picks an OpenFOAM solver name from physics flags
(compressibility / heat transfer / multiphase / time scheme) without
any LLM call.

It is single-region only — the multi-region equivalent is
:func:`simd_agent.run.multi_region.force_cht_solver_if_multi_region`,
which is unconditional (the time scheme uniquely determines
``chtMultiRegionSimpleFoam`` vs ``chtMultiRegionFoam``).

Implementation lives in :mod:`simd_agent.run.genai_codegen`.
"""

from simd_agent.run.genai_codegen import determine_solver

__all__ = ["determine_solver"]
