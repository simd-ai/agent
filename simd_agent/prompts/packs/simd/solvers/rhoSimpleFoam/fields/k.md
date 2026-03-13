# rhoSimpleFoam — 0/k

**Dimensions**: `[0 2 -2 0 0 0 0]` (m²/s²)
**Only for**: kOmegaSST, kEpsilon, kOmega turbulence models

## internalField

Use pre-computed value from `CaseSpec.turbulence_initial_values.k` when available.
Formula when not pre-computed: `k = 1.5 × (I × |U|)²`
where I ≈ 0.05 (5% turbulence intensity for internal flows).

## Typical BC types by patch role

| Patch role | BC type | Notes |
|---|---|---|
| inlet | `fixedValue` | `value uniform <k_value>` from CaseSpec or formula |
| outlet | `zeroGradient` | |
| wall (with wall functions) | `kqRWallFunction` | `value uniform <k_value>` |
| symmetry | `symmetry` | |
| empty (2D) | `empty` | |

## Wall function notes

`kqRWallFunction` is correct for k at walls with wall-function treatment (y+ > 30).
Use the same k value as the inlet for the `value` entry — it's used as an initial guess.

## Template

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
        type            zeroGradient;
    }
    walls
    {
        type            kqRWallFunction;
        value           uniform <k_value>;
    }
}
```
