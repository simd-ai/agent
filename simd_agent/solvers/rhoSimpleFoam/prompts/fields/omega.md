# rhoSimpleFoam — 0/omega

**Dimensions**: `[0 0 -1 0 0 0 0]` (1/s)
**Only for**: kOmegaSST, kOmega turbulence models

## Per-inlet omega — same length scale, different k per inlet

`ω = √k / (Cμ^0.25 · L)` where `Cμ = 0.09` and `L = 0.07 · D_h` is the
turbulence length scale (a geometry property, the same everywhere).
Since each inlet has its own `k = 1.5 · (U_i · I)²`, each inlet has its
own `ω`:

- `inlet_main` k=0.06    →  `ω = √0.06 / (0.548 · L)`
- `inlet_small` k=0.00375 →  `ω = √0.00375 / (0.548 · L)`

**What's the SAME everywhere:** turbulence intensity I and length scale L.
**What's DIFFERENT per inlet:** ω, because each inlet has its own k.

**Rules:**
- Use `CaseSpec.turbulence_initial_values.omega` for `internalField`.
- For each inlet, compute `ω_i` from that inlet's `k_i` and the shared L.
- Wall `omegaWallFunction` `value` uses the internalField ω.

## Typical BC types by patch role

| Patch role | BC type | Notes |
|---|---|---|
| inlet | `fixedValue` | `value uniform <omega_i>` — computed per inlet |
| outlet | `zeroGradient` | |
| wall | `omegaWallFunction` | `value uniform <omega_internal>` |
| symmetry | `symmetry` | |
| empty (2D) | `empty` | |

## Template

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
        type            zeroGradient;
    }
    walls
    {
        type            omegaWallFunction;
        value           uniform <omega_value>;
    }
}
```
