# Solver: compressibleInterIsoFoam  ·  OpenFOAM v2406

**Type**: Transient · Compressible · Two-phase VOF · Non-isothermal · isoAdvector/isoAdvection  
**Pressure field**: `p_rgh` (dimensions `[1 -1 -2 0 0 0 0]`) — MUST have `0/p_rgh`  
**Energy equation**: ✅ YES — MUST generate `0/T`  
**Gravity file**: ✅ ALWAYS generate `constant/g` (even if `(0 0 0)`)  
**Alpha field**: ✅ `0/alpha.<phase1Name>` — name follows phase1 name (e.g. `alpha.water`)  
**Thermophysical**: ✅ Base file + separate thermo per phase (same as compressibleInterFoam)  

## Difference from compressibleInterFoam

Uses **isoAdvector geometric interface advection** (isoAdvection) for sharper phase boundaries instead of MULES.  
Choose when the user requests "sharp interface", thin films, or droplets in compressible flow.

The **only structural difference** is the alpha fvSolution block — everything else (pressure field, thermo files, gravity, energy, phase naming) is identical to `compressibleInterFoam`.

## Phase naming (CRITICAL)

Use phase names from config if provided. Let `phase1Name = config.phases[0]`, `phase2Name = config.phases[1]`.  
If phases are not provided, default to `(water air)`.

- Alpha field: `0/alpha.<phase1Name>` — do **NOT** hardcode `alpha.phase1` unless phase1Name literally is `phase1`.
- Thermo files follow the same naming.

## Required files

| File | Notes |
|------|-------|
| `system/controlDict` | `application compressibleInterIsoFoam;` `startFrom startTime; startTime 0;` |
| `system/fvSchemes` | VOF + compressible convective schemes |
| `system/fvSolution` | PIMPLE + isoAdvection alpha block + `pcorr` solver block |
| `0/U` | Velocity |
| `0/p_rgh` | Modified pressure `[1 -1 -2 0 0 0 0]` |
| `0/T` | Temperature `[0 0 0 1 0 0 0]` — ALWAYS required |
| `0/alpha.<phase1Name>` | Volume fraction |
| `constant/g` | Gravity vector — ALWAYS |
| `constant/thermophysicalProperties` | Base: `phases`, `sigma`, `pMin` |
| `constant/thermophysicalProperties.<phase1Name>` | Thermo for phase 1 |
| `constant/thermophysicalProperties.<phase2Name>` | Thermo for phase 2 |
| `constant/turbulenceProperties` | Only if turbulent; otherwise laminar |

> ⚠️ **No** `constant/transportProperties` — `sigma` and `pMin` live in the **base** `constant/thermophysicalProperties` (same as compressibleInterFoam).

For full templates of `constant/thermophysicalProperties`, per-phase thermo, `fvSchemes`, and PIMPLE block — see `compressibleInterFoam` pack. They are identical.

## Key difference: fvSolution alpha block (isoAdvection)

Replace the MULES block with the isoAdvection configuration. Do **NOT** use `solver isoAdvector;` as a line — that is not the expected config style. Do **NOT** use MULES-only options (`MULESCorr`, `nLimiterIter`, `alphaApplyPrevCorr`) here.

```
"alpha.<phase1Name>.*"
{
    nAlphaCorr          1;
    nAlphaSubCycles     1;
    cAlpha              1;

    // isoAdvection reconstruction options
    reconstructionScheme plicRDF;   // or isoAlpha / gradAlpha depending on mesh
    vof2IsoTol          1e-8;
    surfCellTol         1e-6;
    nAlphaBounds        3;
    snapTol             1e-12;
    clip                true;
}
```

Keep standard `p_rgh`, `p_rghFinal`, `pcorr`, `U`, `T` solver blocks identical to `compressibleInterFoam`.

## fvSchemes notes

- Use conservative transient schemes (`Euler` or `localEuler`).
- Keep `div(rhoPhi,U)` and `div(rhoPhi,T)` stable (`linearUpwind`).
- Do **NOT** rely on `interfaceCompression` tricks as the primary sharp-interface mechanism — isoAdvection is the sharpener here.
- Do **NOT** add `interface interfaceCompression` under `interpolationSchemes`.

## Turbulence fields

- If `turbulenceProperties` says **laminar**: do **NOT** generate `k`, `omega`, `epsilon`, `mut`, or `nut`.
- If **RAS/LES** enabled: generate only what the selected model needs.
- Compressible turbulence uses **`0/mut`** (µt) rather than `0/nut`.

## Critical rules

1. **`p_rgh` NOT `p`** — MUST_READ. `0/T` ALWAYS required.
2. `constant/g` ALWAYS required.
3. **No `constant/transportProperties`** — `sigma` and `pMin` in base `constant/thermophysicalProperties`.
4. Base `constant/thermophysicalProperties` (phases, sigma, pMin) + per-phase files required.
5. Alpha field name follows phase1 name: `0/alpha.<phase1Name>`.
6. `pcorr` solver block required in `fvSolution`.
7. Alpha block uses **isoAdvection parameters** — NOT `solver isoAdvector;`, NOT MULES options.
8. `startFrom startTime; startTime 0;` — NEVER `latestTime`.
9. Patch names MUST match mesh boundary exactly. Empty patches (2D) → `{ type empty; }` in ALL `0/*` files.
