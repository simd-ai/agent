# simd_agent/run/multi_region/__init__.py
"""Multi-region (CHT) subsystem.

Everything CHT-specific lives here so the single-region pipeline can be
read, tested and reasoned about without dragging in multi-region code
paths.  The public surface mirrors what the orchestrator actually
needs:

  * :func:`detect_regions_from_mesh`,
    :func:`force_cht_solver_if_multi_region` — turn raw mesh metadata
    into a CHT ``config["regions"]`` tree and pick the canonical
    chtMultiRegion solver for it.
  * :class:`RegionExtractor` — LLM pass that fills RegionSpec fields
    (temperatures, fluid/solid preset, interfaces) from the user's
    natural-language prompt.
  * :func:`lint_regions` — pre-flight checks on the assembled regions
    tree (cellzone match, interface reciprocity, …).
  * :func:`verify` — post-generation verifier (currently a thin no-op
    because :class:`MultiRegionBase` is authoritative; see
    :mod:`simd_agent.run.multi_region.verifier` for the rationale).

Per-region file rendering itself stays where it already lives —
:mod:`simd_agent.solvers.families._multi_region` and
:mod:`simd_agent.solvers.families._multi_region_bcs` — because the
plugin contract (``render_deterministic_files``, ``validate_full``,
``inject_function_objects``) is the natural seam between "this case
is CHT" (orchestrator) and "here are the per-region files" (plugin).

The case-level → per-region bridging (auto-detect, LLM extractor,
inlet-init backfill, BC propagation, topology lint) now lives in the
solver-agnostic :mod:`simd_agent.run.enrichment` package — see its
docstring for the composable step pipeline.

Post-generation value substitution (LLM rewrites of ``0/<region>/<field>``
files using ``case_defaults`` + per-patch BCs + the user prompt) moved
to :mod:`simd_agent.run.value_filler`, which now handles single-region
cases through the same code path.
"""

from simd_agent.run.multi_region.region_detection import (
    build_region_tree,
    detect_regions_from_cell_zones,
    detect_regions_from_mesh,
    detect_regions_from_patch_prefixes,
    fluid_preset_for,
    force_cht_solver_if_multi_region,
    is_solid_name,
    solid_preset_for,
)
from simd_agent.run.multi_region.region_extractor import RegionExtractor
from simd_agent.run.multi_region.region_lint import lint_regions
from simd_agent.run.multi_region.verifier import verify

__all__ = [
    "RegionExtractor",
    "build_region_tree",
    "detect_regions_from_cell_zones",
    "detect_regions_from_mesh",
    "detect_regions_from_patch_prefixes",
    "fluid_preset_for",
    "force_cht_solver_if_multi_region",
    "is_solid_name",
    "lint_regions",
    "solid_preset_for",
    "verify",
]
