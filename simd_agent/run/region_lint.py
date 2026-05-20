# simd_agent/run/region_lint.py
"""Backward-compat shim — :func:`lint_regions` moved to
:mod:`simd_agent.run.multi_region.region_lint`.

Existing imports (``from simd_agent.run.region_lint import lint_regions``)
keep working; new code should import from the multi_region package.
"""

from simd_agent.run.multi_region.region_lint import *  # noqa: F401,F403
from simd_agent.run.multi_region.region_lint import lint_regions  # noqa: F401

__all__ = ["lint_regions"]
