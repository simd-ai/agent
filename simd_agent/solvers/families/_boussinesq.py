"""Boussinesq / buoyancy-driven mixin.

Composed via MRO with a family base.  Adds the buoyancy-specific bits
that ``buoyantSimpleFoam`` / ``buoyantPimpleFoam`` need on top of the
standard SIMPLE / PIMPLE machinery:

  * ``needs_gravity = True``  — drives ``_ensure_gravity`` validator
  * ``pressure_field = "p_rgh"`` — buoyant solvers use modified pressure
                                   ``p_rgh = p − ρgh`` (Boussinesq + hydrostatic)
  * ``_ensure_gravity`` validator — inserts ``constant/g`` when missing

Composed solvers:

  * ``BuoyantSimpleFoamSolver(SteadyBase, BoussinesqMixin)``
  * ``BuoyantPimpleFoamSolver(TransientBase, BoussinesqMixin)``
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simd_agent.solvers.base import ValidationIssue


class BoussinesqMixin:
    """Methods + identity overrides shared by buoyancy-driven solvers.

    The mixin assumes the composed class inherits ``SolverPlugin`` via a
    family base.  Buoyant solvers use ``p_rgh`` (modified pressure) and
    always require ``constant/g``.
    """

    needs_gravity: bool = True
    pressure_field: str = "p_rgh"

    def _ensure_gravity(
        self,
        files: dict[str, str],
        issues: "list[ValidationIssue]",
    ) -> dict[str, str]:
        """Insert a default ``constant/g`` if missing.

        Without this file a buoyantFoam case crashes at startup with
        ``cannot find file constant/g``.  Default is Earth gravity along
        −y (the OpenFOAM tutorial convention; the user can override).
        """
        from simd_agent.solvers.base import ValidationIssue  # avoid cycle

        # Method lives on the mixin so it ONLY runs for buoyant solvers.
        # ``self.name`` / ``self.needs_gravity`` come from the composed
        # plugin via MRO.
        if not getattr(self, "needs_gravity", False):
            return files
        if "constant/g" in files:
            return files

        issues.append(
            ValidationIssue(
                "error",
                "constant/g",
                f"'{getattr(self, 'name', '?')}' requires constant/g. "
                "Adding default.",
                fix="Added constant/g",
            )
        )
        files["constant/g"] = (
            "FoamFile\n{\n    version 2.0;\n    format ascii;\n"
            "    class uniformDimensionedVectorField;\n    object g;\n}\n"
            "dimensions [0 1 -2 0 0 0 0];\nvalue (0 -9.81 0);\n"
        )
        return files
