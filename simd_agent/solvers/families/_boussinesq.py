"""Boussinesq / buoyancy-driven mixins.

Two flavours of buoyancy here:

  * ``BoussinesqMixin``               — composed with ``CompressibleMixin``
                                        by ``buoyant{Simple,Pimple}Foam``.
                                        Uses thermophysicalProperties +
                                        the full compressible energy form.
  * ``IncompressibleBoussinesqMixin`` — for ``buoyantBoussinesq{Simple,
                                        Pimple}Foam``.  Constant ρ; the
                                        Boussinesq approximation enters
                                        via a buoyancy source term
                                        ``-ρ₀·β·(T−T_ref)·g`` rather
                                        than a real density transport.
                                        Energy variable is **T**, not
                                        h or e; thermo file is
                                        **transportProperties**.

Both share:

  * ``needs_gravity = True``  — drives ``_ensure_gravity`` validator
  * ``pressure_field = "p_rgh"`` — modified pressure ``p_rgh = p − ρgh``
  * ``_ensure_gravity`` — inserts ``constant/g`` when missing

Composed solvers:

  * ``BuoyantSimpleFoamSolver(SteadyBase, CompressibleMixin, BoussinesqMixin)``
  * ``BuoyantPimpleFoamSolver(TransientBase, CompressibleMixin, BoussinesqMixin)``
  * ``BuoyantBoussinesqSimpleFoamSolver(SteadyBase, IncompressibleBoussinesqMixin)``
  * ``BuoyantBoussinesqPimpleFoamSolver(TransientBase, IncompressibleBoussinesqMixin)``
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


class IncompressibleBoussinesqMixin(BoussinesqMixin):
    """For ``buoyantBoussinesq{Simple,Pimple}Foam`` — incompressible Boussinesq.

    Differences from the parent ``BoussinesqMixin``:

      * ``is_compressible = False`` — constant ρ; the buoyancy enters
        the momentum equation as a source term
        ``-ρ₀·β·(T−T_ref)·g`` rather than via ρ(T).
      * ``energy_var = "T"`` — the energy equation transports T directly
        (no enthalpy or internal energy variable).  Drives
        ``div(phi,T)`` in fvSchemes and the ``T`` solver block.
      * Uses ``constant/transportProperties`` (Newtonian, ν, β, T_ref,
        Pr, Prt) instead of ``constant/thermophysicalProperties``.

    Inherits ``pressure_field = "p_rgh"``, ``needs_gravity = True``,
    and ``_ensure_gravity`` from ``BoussinesqMixin`` unchanged.

    Composed solvers:

      * ``BuoyantBoussinesqSimpleFoamSolver(SteadyBase, IncompressibleBoussinesqMixin)``
      * ``BuoyantBoussinesqPimpleFoamSolver(TransientBase, IncompressibleBoussinesqMixin)``
    """

    is_compressible: bool = False
    energy_var: str = "T"

    @staticmethod
    def build_transport_properties(
        rho: float = 1.0,
        nu: float = 1e-5,
        beta: float = 3e-3,
        t_ref: float = 300.0,
        Pr: float = 0.9,
        Prt: float = 0.7,
    ) -> str:
        """Render ``constant/transportProperties`` for an incompressible Boussinesq case.

        Defaults match the OpenFOAM ``hotRoom`` reference tutorial
        (air-like fluid: ν ≈ 1e-5 m²/s, β ≈ 3e-3 K⁻¹, T_ref = 300 K,
        Pr = 0.9, Prt = 0.7).
        """
        return (
            "FoamFile\n"
            "{\n"
            "    version     2.0;\n"
            "    format      ascii;\n"
            "    class       dictionary;\n"
            "    location    \"constant\";\n"
            "    object      transportProperties;\n"
            "}\n"
            "// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //\n\n"
            "transportModel  Newtonian;\n\n"
            f"nu              [0 2 -1 0 0 0 0] {nu:g};\n"
            f"beta            [0 0 0 -1 0 0 0] {beta:g};\n"
            f"TRef            [0 0 0 1 0 0 0] {t_ref:g};\n"
            f"Pr              [0 0 0 0 0 0 0] {Pr:g};\n"
            f"Prt             [0 0 0 0 0 0 0] {Prt:g};\n\n"
            "// ************************************************************************* //\n"
        )

    @staticmethod
    def _extract_transport_inputs(
        config: dict,
    ) -> tuple[float, float, float, float, float, float]:
        """Read fluid properties from a normalised config dict.

        Falls back to the ``hotRoom`` defaults for any missing field.
        Returns ``(rho, nu, beta, TRef, Pr, Prt)`` — rho is included so
        future variants (e.g. variable-property post-processing) can read
        it without re-walking the config.
        """
        fluid = config.get("fluid") or {}
        if not isinstance(fluid, dict):
            fluid = {}

        # Defaults match the OF hotRoom tutorial (air-like fluid).
        rho = 1.0
        nu = 1e-5
        beta = 3e-3
        t_ref = 300.0
        Pr = 0.9
        Prt = 0.7

        try:
            _r = fluid.get("rho") or fluid.get("density")
            if _r is not None:
                rho = float(_r)
        except (TypeError, ValueError):
            pass

        try:
            _mu = fluid.get("mu") or fluid.get("viscosity")
            if _mu is not None and rho > 0:
                # Kinematic viscosity from dynamic viscosity.
                nu = float(_mu) / rho
        except (TypeError, ValueError):
            pass

        try:
            _b = fluid.get("beta") or fluid.get("thermal_expansion")
            if _b is not None:
                beta = float(_b)
        except (TypeError, ValueError):
            pass

        try:
            _t = fluid.get("temperature") or fluid.get("TRef")
            if _t is not None:
                t_ref = float(_t)
        except (TypeError, ValueError):
            pass

        try:
            _Pr = fluid.get("Pr") or fluid.get("prandtl")
            if _Pr is not None:
                Pr = float(_Pr)
        except (TypeError, ValueError):
            pass

        return rho, nu, beta, t_ref, Pr, Prt
