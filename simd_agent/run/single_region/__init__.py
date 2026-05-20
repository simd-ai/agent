# simd_agent/run/single_region/__init__.py
"""Single-region subsystem.

Everything for the flat single-region OpenFOAM case tree
(``0/<field>``, ``system/<dict>``, ``constant/<dict>``) lives here.
The multi-region (CHT) counterpart is
:mod:`simd_agent.run.multi_region`.

Public surface mirrors what callers actually need:

  * :class:`GenAICodeGenerator`, :func:`extract_file_blocks` — LLM
    per-file codegen + the parser for its file-block output.
  * :func:`validate_generated_files`, :class:`ValidationIssue` —
    deterministic post-generation auto-fix layer (reached via
    :meth:`SolverPlugin.validate_full` for plugin-backed solvers; this
    function is the fallback path for legacy solvers without a
    plugin).
  * :func:`build_required_files_list` — the per-solver case-file
    manifest the LLM must produce.
  * :func:`determine_solver` — heuristic fallback solver selection
    when the LLM call fails / times out.
  * :func:`verify` — rule-based single-region verifier used by
    :class:`simd_agent.run.code_verifier.CodeVerifier`.

The codegen / validator implementations currently live in
:mod:`simd_agent.run.genai_codegen` (~4 kLOC).  The modules in this
package are thin re-export façades; a future PR can physically split
the file and the public import paths here stay stable.
"""

from simd_agent.run.single_region.codegen import (
    GenAICodeGenerator,
    extract_file_blocks,
)
from simd_agent.run.single_region.file_manifest import build_required_files_list
from simd_agent.run.single_region.heuristic_solver import determine_solver
from simd_agent.run.single_region.validator import (
    ValidationIssue,
    validate_generated_files,
)
from simd_agent.run.single_region.verifier import verify

__all__ = [
    "GenAICodeGenerator",
    "ValidationIssue",
    "build_required_files_list",
    "determine_solver",
    "extract_file_blocks",
    "validate_generated_files",
    "verify",
]
