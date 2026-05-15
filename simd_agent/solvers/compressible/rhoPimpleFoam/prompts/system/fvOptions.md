# rhoPimpleFoam — system/fvOptions

This file is **REQUIRED** for all compressible energy solvers.
It prevents negative-temperature and negative-density divergence during early PIMPLE iterations.

---

## Temperature limiter (mandatory)

Always generate a `limitTemperature` block. The values come from CaseSpec:

| Key | CaseSpec field | How to use |
|---|---|---|
| `min` | `fv_options_t_min` | Use exactly — 50% of coldest BC temperature |
| `max` | *(you determine)* | See selection rules below |

### Selecting `max`

1. Read `CaseSpec.fv_options_eos_t_ceiling` — **hard upper limit** for icoPolynomial EOS.
   - For `icoPolynomial`: ρ(T) = a0 + a1·T → ρ = 0 at T = `eos_t_ceiling` → SIGFPE.
   - `max` **MUST be strictly below `eos_t_ceiling`**.
2. **Always use `eos_t_ceiling × 0.9` as `max`** — this is the same value used to clamp the wall fixedValue BC in `0/T` (Check 3c3 in the validator). Keeping `max` consistent with the wall BC avoids an artificial inconsistency where the wall face sees one temperature but internal cells are clipped to a different value, which destabilises the PIMPLE outer loop.
   - Do NOT use 0.8× or lower — that makes fvOptions max lower than the clamped wall BC, causing 40–50% of cells to always hit the upper limiter regardless of the actual flow, breaking the energy equation.
3. Special case — **wall temperature within EOS range**: if `wall_temperature < eos_t_ceiling`, use `wall_temperature * 0.95` (slight margin to avoid clipping the wall BC itself).
4. If `eos_t_ceiling` is not provided (non-icoPolynomial EOS): use `max(fv_options_bc_temps) + 50`.

**Do NOT use `max 100000` for icoPolynomial cases — this will crash the solver.**

---

## Template

```
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      fvOptions;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

temperatureLimiter
{
    type            limitTemperature;
    active          yes;

    selectionMode   all;

    min             <CaseSpec.fv_options_t_min>;        // K — numerical floor
    max             <determined from rules above>;      // K — highest physical T
}

// ************************************************************************* //
```

---

## Adding future limiters

Each limiter is an independent named block. Keys must be unique:

```
temperatureLimiter
{
    type            limitTemperature;
    active          yes;
    selectionMode   all;
    min             <CaseSpec.fv_options_t_min>;
    max             <see above>;
}

velocityLimiter
{
    type            limitVelocityMagnitude;
    active          yes;
    selectionMode   all;
    max             500;
}
```

---

## Rules

1. Always include `temperatureLimiter` — never omit fvOptions.
2. `min` = `CaseSpec.fv_options_t_min` (exact).
3. `max` must be below `CaseSpec.fv_options_eos_t_ceiling` for icoPolynomial.
4. `max` should reflect the highest physically meaningful temperature — reason from BCs and fluid type.
5. Do NOT use `min 1` for cryogenic cases.
6. Do NOT use `max 100000` for icoPolynomial — negative density → SIGFPE.
