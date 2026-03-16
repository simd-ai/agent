# simpleFoam — constant/transportProperties

Single-phase incompressible fluid. Only `nu` is required.

```
transportModel  Newtonian;

nu              [0 2 -1 0 0 0 0] <kinematic_viscosity>;   // m²/s
```

Use `CaseSpec.nu` for the kinematic viscosity value.
If only dynamic viscosity `mu` and density `rho` are given: `nu = mu / rho`.

## Rules

- `transportModel Newtonian;` — always for Newtonian fluids
- Do NOT include `rho` here — simpleFoam is incompressible (density not used by solver)
- Dimension array: `[0 2 -1 0 0 0 0]` — do not omit it
