"""Heat-transfer solver family.

Solvers that combine an energy equation with buoyancy (``p_rgh`` +
``constant/g``) or conjugate heat transfer across multiple regions.
Mirrors the OpenFOAM ``tutorials/heatTransfer`` directory.

Contents:
  * ``buoyantSimpleFoam``                 — steady, compressible Boussinesq
  * ``buoyantPimpleFoam``                 — transient, compressible Boussinesq
  * ``buoyantBoussinesqSimpleFoam`` (TBD) — steady, incompressible Boussinesq
  * ``buoyantBoussinesqPimpleFoam`` (TBD) — transient, incompressible Boussinesq
  * ``chtMultiRegionSimpleFoam``    (TBD) — steady multi-region CHT
  * ``chtMultiRegionFoam``          (TBD) — transient multi-region CHT
"""
