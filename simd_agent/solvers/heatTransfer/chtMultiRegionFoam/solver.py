# simd_agent/solvers/heatTransfer/chtMultiRegionFoam/solver.py
"""chtMultiRegionFoam — transient multi-region conjugate heat transfer.

Transient counterpart of ``chtMultiRegionSimpleFoam``.  Same multi-region
file tree, same per-region thermo, same fluid-solid coupling — just a
PIMPLE outer loop with real time stepping.

**Phase 1 — architectural skeleton.**  Same status caveats as
``chtMultiRegionSimpleFoam.solver``: identity + manifest + deterministic
regionProperties + per-region thermo are functional; per-region
fvSchemes / fvSolution / mapped BCs / changeDictionaryDict land in Phase 2.
"""

from __future__ import annotations

import logging
from typing import Any

from simd_agent.solvers.base import (
    MatchResult,
    ValidationIssue,
    ValidationResult,
)
from simd_agent.solvers.families import MultiRegionBase, TransientBase

logger = logging.getLogger(__name__)


class ChtMultiRegionFoamSolver(MultiRegionBase, TransientBase):
    """chtMultiRegionFoam — transient multi-region CHT.

    Mixin-first MRO so multi-region overrides win over ``TransientBase``
    defaults.
    """

    name = "chtMultiRegionFoam"
    is_transient = True
    is_multiphase = False

    def matches(self, config: dict[str, Any]) -> MatchResult:
        physics = config.get("physics", {}) or {}
        regions_cfg = config.get("regions") or {}
        solids = regions_cfg.get("solid") if isinstance(regions_cfg, dict) else None
        is_multi_region = bool(solids)
        transient = (
            config.get("time_stepping")
            or physics.get("time_scheme", "steady")
        ) in ("transient", "unsteady")

        if not is_multi_region:
            return MatchResult(
                0.0,
                "chtMultiRegionFoam requires fluid + solid regions; "
                "use a single-region buoyant*/rho* solver instead.",
            )
        if not transient:
            return MatchResult(
                0.2,
                "chtMultiRegionFoam is transient; "
                "use chtMultiRegionSimpleFoam for steady CHT.",
            )
        return MatchResult(
            0.95,
            "Transient conjugate heat transfer with fluid + solid regions.",
        )

    def validate(
        self, files: dict[str, str], config: dict[str, Any]
    ) -> ValidationResult:
        """Validate a CHT case: render the deterministic per-region tree,
        ensure ``application`` in ``system/controlDict`` matches the solver
        name.  Gravity is emitted per-region by ``render_deterministic_files``
        so there's no top-level ``constant/g`` to enforce here.
        """
        issues: list[ValidationIssue] = []
        fixed = dict(files)
        fixed.update(self.render_deterministic_files(config))
        fixed = self._fix_controldict_solver(fixed, issues)
        # A6 — per-region function objects (volAverage T, patch averages
        # on inlets/outlets) so the runner's per-region progress parser
        # has structured data to consume on every timestep.
        fixed = self.inject_function_objects(fixed, config)
        return ValidationResult(files=fixed, issues=issues)
