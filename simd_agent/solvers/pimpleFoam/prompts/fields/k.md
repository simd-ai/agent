# pimpleFoam — 0/k

**Dimensions**: `[0 2 -2 0 0 0 0]` (m^2/s^2)

## internalField

Use `CaseSpec.turbulence_initial_values.k` when available.
Formula: `k = 1.5 * (I * |U|)^2` where I = 0.05 for internal flows.

## BC types

| Patch role | BC type |
|---|---|
| inlet | `fixedValue` with computed k |
| outlet | `inletOutlet` | `inletValue uniform <k_value>` — prevents negative k from backflow |
| wall | `kqRWallFunction` |
| symmetry | `symmetry` |
| symmetryPlane | `symmetryPlane` |
| empty (2D planar) | `empty` — no `value`, just `type empty;` |
| wedge (2D axi) | `wedge` — no `value`, just `type wedge;` |

```
internalField   uniform <k_value>;

boundaryField
{
    inlet
    {
        type            fixedValue;
        value           uniform <k_value>;
    }
    outlet
    {
        type            inletOutlet;
        inletValue      uniform <k_value>;
        value           uniform <k_value>;
    }
    walls
    {
        type            kqRWallFunction;
        value           uniform <k_value>;
    }
}
```

## Critical rules

- `k` MUST be >= 1e-6 — values below this cause division-by-zero in wall functions
- `k = 1.5 * (I * |U|)^2` with I = 0.05 is the safe default; if CaseSpec provides a value, use it
- Outlet MUST use `inletOutlet` (not `zeroGradient`) to prevent negative k from backflow -> SIGFPE in kOmegaSST
