# pimpleFoam — 0/omega

**Dimensions**: `[0 0 -1 0 0 0 0]` (1/s)

## internalField

Use `CaseSpec.turbulence_initial_values.omega` when available.
Formula: `omega = k^0.5 / (Cmu^0.25 * L)` where Cmu = 0.09, L = 0.07 * D_h.

## BC types

| Patch role | BC type |
|---|---|
| inlet | `fixedValue` |
| outlet | `inletOutlet` | `inletValue uniform <omega_value>` — prevents negative omega from backflow |
| wall | `omegaWallFunction` |
| symmetry | `symmetry` |
| symmetryPlane | `symmetryPlane` |
| empty (2D planar) | `empty` — no `value`, just `type empty;` |
| wedge (2D axi) | `wedge` — no `value`, just `type wedge;` |

```
internalField   uniform <omega_value>;

boundaryField
{
    inlet
    {
        type            fixedValue;
        value           uniform <omega_value>;
    }
    outlet
    {
        type            inletOutlet;
        inletValue      uniform <omega_value>;
        value           uniform <omega_value>;
    }
    walls
    {
        type            omegaWallFunction;
        value           uniform <omega_value>;
    }
}
```

## Critical rules

- `omega` MUST be >= 1.0 — values below this cause numerical instability in kOmegaSST
- Formula: `omega = k^0.5 / (Cmu^0.25 * L)` with Cmu = 0.09, L = 0.07 * D_h
- Outlet MUST use `inletOutlet` (not `zeroGradient`) to prevent negative omega from backflow -> SIGFPE
