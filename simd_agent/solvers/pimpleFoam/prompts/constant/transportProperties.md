# pimpleFoam — constant/transportProperties

Identical to simpleFoam — single-phase incompressible fluid.

```
transportModel  Newtonian;

nu              [0 2 -1 0 0 0 0] <kinematic_viscosity>;   // m²/s from CaseSpec.nu
```

If only `mu` (dynamic viscosity) and `rho` (density) are given: `nu = mu / rho`.
