# rhoSimpleFoam — 0/epsilon

**Dimensions**: `[0 2 -3 0 0 0 0]` (m²/s³)
**Only for**: kEpsilon turbulence model

## Per-inlet epsilon — same length scale, different k per inlet

`ε = Cμ^0.75 · k^1.5 / L` — each inlet has its own k (from its own U
and the shared TI), so each inlet has its own ε.

**What's the SAME everywhere:** turbulence intensity I and length scale L.
**What's DIFFERENT per inlet:** ε, because each inlet has its own k.

**Rules:**
- Use `CaseSpec.turbulence_initial_values.epsilon` for `internalField`.
- For each inlet, compute `ε_i` from that inlet's `k_i` and the shared L.
- Wall `epsilonWallFunction` `value` uses the internalField ε.

## Typical BC types by patch role

| Patch role | BC type | Notes |
|---|---|---|
| inlet | `fixedValue` | `value uniform <epsilon_i>` — computed per inlet |
| outlet | `zeroGradient` | |
| wall | `epsilonWallFunction` | `value uniform <epsilon_internal>` |
| symmetry | `symmetry` | |
| empty (2D) | `empty` | |

## Template

```
internalField   uniform <epsilon_value>;

boundaryField
{
    inlet
    {
        type            fixedValue;
        value           uniform <epsilon_value>;
    }
    outlet
    {
        type            zeroGradient;
    }
    walls
    {
        type            epsilonWallFunction;
        value           uniform <epsilon_value>;
    }
}
```
