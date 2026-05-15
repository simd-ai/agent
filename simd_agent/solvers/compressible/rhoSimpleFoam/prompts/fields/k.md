# rhoSimpleFoam — 0/k

**Dimensions**: `[0 2 -2 0 0 0 0]` (m²/s²)
**Only for**: kOmegaSST, kEpsilon, kOmega turbulence models

## Per-inlet k — same TI, different velocities

`k` depends on inlet velocity through `k = 1.5 × (I × U)²`.  With a
shared turbulence intensity `I` (e.g. 5%), every inlet computes its OWN
`k` from its OWN velocity:

- `inlet_main` at U=4 m/s, I=5%  →  `k = 1.5 · (4·0.05)²  = 0.06`
- `inlet_small` at U=1 m/s, I=5%  →  `k = 1.5 · (1·0.05)² = 0.00375`

**What's the SAME everywhere:** the turbulence intensity `I` (5% for
internal flow) and the length scale `L = 0.07 · D_h` (geometry).
**What's DIFFERENT per inlet:** `k` itself, because each inlet's `U`
goes into the formula.

**Rules:**
- Use `CaseSpec.turbulence_initial_values.k` for `internalField` (it's the
  representative bulk-flow value).
- For each turbulent inlet, compute `k_i = 1.5 · (U_i · 0.05)²` using
  that inlet's velocity.
- Wall `kqRWallFunction` `value` is just an initial guess — use the
  internalField k.
- A common LLM mistake is using a different TI per inlet (5% for one,
  1% for another).  Do NOT do that.  Same TI everywhere, k naturally
  differs.

## Typical BC types by patch role

| Patch role | BC type | Notes |
|---|---|---|
| inlet | `fixedValue` | `value uniform <k_i>` — computed from each inlet's U using I=5% |
| outlet | `zeroGradient` | |
| wall (with wall functions) | `kqRWallFunction` | `value uniform <k_internal>` |
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
