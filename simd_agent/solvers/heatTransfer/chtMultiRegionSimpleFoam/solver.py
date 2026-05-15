# simd_agent/solvers/heatTransfer/chtMultiRegionSimpleFoam/solver.py
"""chtMultiRegionSimpleFoam — steady multi-region conjugate heat transfer.

Couples one or more fluid regions (compressible flow + energy) with one
or more solid regions (heat conduction only) via mapped temperature
boundaries at fluid–solid interfaces.

**Phase 1 (this file) is the architectural skeleton:**

  * Identity attributes correct (multi-region, SIMPLE, p_rgh, gravity).
  * ``required_files()`` returns the per-region file tree.
  * ``render_deterministic_files()`` emits ``regionProperties`` +
    per-region ``thermophysicalProperties`` for fluid and solid regions.
  * ``matches()`` scores multi-region heat-transfer configs.

**Phase 2 (TODO):**

  * Per-region ``system/<region>/fvSchemes`` and ``fvSolution``.
  * Per-region ``0/<region>/`` field files with the right BC types,
    including the coupled fluid-solid
    ``compressible::turbulentTemperatureCoupledBaffleMixed`` patches.
  * ``changeDictionaryDict`` per region.
  * Hook into ``orchestration.py`` to handle the tree-structured manifest
    and into ``packaging.py`` to zip the multi-region case correctly.
"""

from __future__ import annotations

import logging
from typing import Any

from simd_agent.solvers.base import (
    MatchResult,
    ValidationIssue,
    ValidationResult,
)
from simd_agent.solvers.families import MultiRegionBase, SteadyBase

logger = logging.getLogger(__name__)


class ChtMultiRegionSimpleFoamSolver(MultiRegionBase, SteadyBase):
    """chtMultiRegionSimpleFoam — steady multi-region CHT.

    Mixin-first MRO so the multi-region overrides
    (``is_multi_region = True``, ``pressure_field = "p_rgh"``,
    ``needs_gravity = True``) take precedence over the ``SolverPlugin``
    defaults inherited from ``SteadyBase``.
    """

    name = "chtMultiRegionSimpleFoam"
    is_transient = False
    is_multiphase = False  # multi-region ≠ multi-phase

    def matches(self, config: dict[str, Any]) -> MatchResult:
        physics = config.get("physics", {}) or {}
        # A multi-region case is identified by config["regions"] containing
        # a "solid" list with at least one region.  Single-region cases
        # (even with heat) match buoyantSimpleFoam / rhoSimpleFoam.
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
                "chtMultiRegionSimpleFoam requires fluid + solid regions; "
                "use a single-region buoyant*/rho* solver instead.",
            )
        if transient:
            return MatchResult(
                0.2,
                "chtMultiRegionSimpleFoam is steady; "
                "use chtMultiRegionFoam for transient CHT.",
            )
        return MatchResult(
            0.95,
            "Steady conjugate heat transfer with fluid + solid regions.",
        )

    def validate(
        self, files: dict[str, str], config: dict[str, Any]
    ) -> ValidationResult:
        """Phase 1 validate — emits the deterministic CHT skeleton.

        Most of the heavy lifting will live in Phase 2 (mapped BCs,
        per-region fvSchemes/fvSolution).  For now we just write the
        ``regionProperties`` + per-region ``thermophysicalProperties``
        files so the case can be loaded by ``createMeshes.H``.
        """
        issues: list[ValidationIssue] = []
        fixed = dict(files)

        # Deterministic CHT skeleton (regionProperties + per-region thermo).
        fixed.update(self.render_deterministic_files(config))

        # Universal fixers from base.py — operate on top-level files only,
        # so they're safe to apply on a multi-region case.
        fixed = self._fix_controldict_solver(fixed, issues)

        # constant/g — every fluid region needs one; the top-level constant/g
        # is typically copied into each region's directory by the case
        # setup.  Phase 2 will emit per-region constant/<region>/g; for now
        # ensure the top-level file exists.
        fixed = self._ensure_gravity(fixed, issues)

        return ValidationResult(files=fixed, issues=issues)
