# simd_agent/run/region_extractor.py
"""Backward-compat shim — :class:`RegionExtractor` moved to
:mod:`simd_agent.run.multi_region.region_extractor`.

Existing imports (``from simd_agent.run.region_extractor import RegionExtractor``)
keep working; new code should import from the multi_region package.

Private names (``_FLUID_PRESETS``, ``_SOLID_PRESETS``, …) are re-exported
explicitly so existing test modules that reach in via ``from … import _NAME``
keep working without star-import semantics.
"""

from simd_agent.run.multi_region.region_extractor import *  # noqa: F401,F403
from simd_agent.run.multi_region.region_extractor import (  # noqa: F401
    RegionExtractor,
    _FLUID_PRESETS,
    _SOLID_PRESETS,
)

__all__ = ["RegionExtractor", "_FLUID_PRESETS", "_SOLID_PRESETS"]
