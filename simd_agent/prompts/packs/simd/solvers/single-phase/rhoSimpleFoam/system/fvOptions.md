# rhoSimpleFoam — system/fvOptions

This file is **REQUIRED** for all compressible energy solvers.
It prevents negative-temperature and negative-density divergence during early SIMPLE iterations.

---

## Temperature limiter (mandatory)

Always generate a `limitTemperature` block. The values come from CaseSpec:

| Key | CaseSpec field | How to use |
|---|---|---|
| `min` | `fv_options_t_min` | Use exactly — 50% of coldest BC temperature |
| `max` | *(you determine)* | See selection rules below |

### Selecting `max`

1. Read `CaseSpec.fv_options_eos_t_ceiling` — **hard upper limit** for icoPolynomial EOS.
   - For `icoPolynomial`: ρ(T) = a0 + a1·T. At T = `eos_t_ceiling`, ρ → 0 → SIGFPE.
   - `max` **MUST be strictly below `eos_t_ceiling`**.
2. **Always use `eos_t_ceiling × 0.9` as `max`** — this is consistent with the wall fixedValue BC clamp in `0/T`. Using a lower value (0.8×) creates an inconsistency: the wall face sees the clamped BC temperature (0.9×ceiling) but internal cells are clipped to a lower value (0.8×ceiling), causing 40–50% of cells to always hit the upper limiter, breaking the energy equation.
3. If wall temperature < eos_t_ceiling: use `wall_temperature * 0.95` instead (slight margin).
4. If `eos_t_ceiling` not provided: use `max(fv_options_bc_temps) + 50`.

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

Each limiter is an independent named block inside the top-level fvOptions dict.
All entries are processed every iteration. Keys must be unique:

```
temperatureLimiter
{
    type            limitTemperature;
    active          yes;
    selectionMode   all;
    min             <CaseSpec.fv_options_t_min>;
    max             <see above>;
}

// Example: clamp velocity magnitude during startup
velocityLimiter
{
    type            limitVelocityMagnitude;
    active          yes;
    selectionMode   all;
    max             500;    // m/s
}
```

---

## Rules

1. Always include `temperatureLimiter` — never omit fvOptions.
2. `min` = `CaseSpec.fv_options_t_min` (exact, no changes).
3. `max` must be below `CaseSpec.fv_options_eos_t_ceiling` for icoPolynomial EOS.
4. `max` should reflect the highest physically meaningful temperature — reason from BC temperatures and fluid type.
5. `selectionMode all` — applies to every cell in the domain.
6. Do NOT use `min 1` for cryogenic cases — 76 K error per iteration for LN2.
7. Do NOT use `max 100000` for icoPolynomial — negative density → SIGFPE.
