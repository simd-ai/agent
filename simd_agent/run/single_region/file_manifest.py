# simd_agent/run/single_region/file_manifest.py
"""Single-region required-files manifest — re-export façade.

:func:`build_required_files_list` returns the exact set of case-file
paths a single-region solver expects the LLM to produce (``0/U``,
``0/p`` or ``0/p_rgh``, ``system/controlDict``, …) based on the
validated config.

The multi-region equivalent is the plugin contract
:meth:`SolverPlugin.required_files` on the :class:`MultiRegionBase`
subclasses — they own their own manifest because the per-region tree
shape (``0/<region>/<field>``) has nothing in common with the
single-region flat tree.

Implementation lives in :mod:`simd_agent.run.genai_codegen` for now.
"""

from simd_agent.run.genai_codegen import build_required_files_list

__all__ = ["build_required_files_list"]
