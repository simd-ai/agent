# simd_agent/run/single_region/codegen.py
"""Single-region LLM codegen — re-export façade.

Surfaces :class:`GenAICodeGenerator` (per-file parallel LLM codegen for
flat single-region case trees) and :func:`extract_file_blocks` (the
parser that turns LLM file-block output into a ``{path: content}``
dict) under the package-qualified path
``simd_agent.run.single_region.codegen``.

The implementation lives in :mod:`simd_agent.run.genai_codegen` for
now — physically splitting that 4 kLOC file is a separate change.
New call sites should import from this module so a future physical
relocation is a one-line search-and-replace.
"""

from simd_agent.run.genai_codegen import (
    GenAICodeGenerator,
    extract_file_blocks,
)

__all__ = ["GenAICodeGenerator", "extract_file_blocks"]
