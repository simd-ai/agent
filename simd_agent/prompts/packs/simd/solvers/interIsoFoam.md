# Solver: interIsoFoam  ·  OpenFOAM v2406

**Type**: Transient · Incompressible · Two-phase VOF · isoAdvector/isoAdvection (sharper interface)  
**Pressure field**: `p_rgh` (Pa) → MUST have `0/p_rgh`  
**Energy equation**: ❌ No → do NOT generate `0/T`  
**Gravity file**: ✅ REQUIRED → MUST generate `constant/g` (use `(0 0 0)` if gravity=false)

## Key difference vs interFoam

`interIsoFoam` uses isoAdvector-style geometric interface advection (isoAdvection) instead of
the classic MULES-only approach. This produces sharper, less diffuse interfaces. Choose it
when the user mentions:
- "sharp interface"
- "thin film"
- Droplet / jet simulations where interface precision matters
- Reduced numerical diffusion

## A) Required files (minimum working)

| File | Notes |
|------|-------|
| `system/controlDict` | `application interIsoFoam;` `startFrom startTime; startTime 0;` |
| `system/fvSchemes` | VOF schemes (corrected — see below) |
| `system/fvSolution` | PIMPLE/PISO block + isoAdvection alpha controls |
| `constant/g` | ALWAYS required |
| `constant/transportProperties` | Two-phase with sigma |
| `constant/turbulenceProperties` | ONLY if turbulent; otherwise laminar |
| `0/U`, `0/p_rgh`, `0/alpha.<phase1Name>` | Phase-named alpha field |

If turbulence enabled (e.g. kOmegaSST): `0/k`, `0/omega`, `0/nut` + `constant/turbulenceProperties`

## B) Phase naming (CRITICAL)

Let `phase1Name` and `phase2Name` come from config phases.  
If not provided, default to `water` and `air`.  
Alpha file MUST be `0/alpha.<phase1Name>` (e.g. `alpha.water`).  
**Do NOT hardcode `alpha.phase1`** unless the phase name is literally `phase1`.

## C) constant/transportProperties (two-phase)

Must define `phases (phase1Name phase2Name)`, each phase `nu`/`rho`, and `sigma`. Same as interFoam.

## D) constant/g

```
dimensions [0 1 -2 0 0 0 0];
value      (0 -9.81 0);  // or (0 0 0) if gravity=false
```

## E) controlDict time-step control (recommended)

Use automatic timestep control for stability:
- `adjustTimeStep yes;`
- `maxCo <= 1;`
- `maxAlphaCo <= 1` (often 0.5–1.0)
- `maxDeltaT <cap>;`

## F) fvSolution: isoAdvection alpha controls (KEY DIFFERENCE)

Configure isoAdvection via the alpha dictionary block. Do **not** write `solver isoAdvector;`.

```
"alpha.<phase1Name>.*"
{
    nAlphaCorr      1;
    nAlphaSubCycles 1;
    cAlpha          1;

    // isoAdvector / isoAdvection reconstruction options
    reconstructionScheme plicRDF;   // alternatives depend on your setup
    vof2IsoTol      1e-8;
    surfCellTol     1e-6;
    nAlphaBounds    3;
    snapTol         1e-12;
    clip            true;
}
```

Include PIMPLE (or PISO depending on base templates):

```
PIMPLE
{
    momentumPredictor   yes;
    nOuterCorrectors    1;
    nCorrectors         2;
    nNonOrthogonalCorrectors 0;
}
```

## G) fvSchemes (VOF-specific, corrected)

```
ddtSchemes      { default Euler; }
gradSchemes     { default Gauss linear; }
divSchemes
{
    default                                         none;
    div(rhoPhi,U)                                   Gauss linearUpwind grad(U);
    div(phi,alpha)                                  Gauss vanLeer;
    div(phirb,alpha)                                Gauss linear;
    div(((rho*nuEff)*dev2(T(grad(U)))))             Gauss linear;
}
laplacianSchemes    { default Gauss linear corrected; }
interpolationSchemes { default linear; }   // do NOT add nonstandard "interface interfaceCompression"
snGradSchemes       { default corrected; }
wallDist            { method meshWave; }   // only if using wall functions / turbulence
```

## H) 2D mesh constraint patches

If mesh has a patch with `patch_type == empty` (e.g. `frontAndBack`):
- That patch MUST be `{ type empty; }` in `0/U`, `0/p_rgh`, `0/alpha.<phase1Name>`.
- Never invent `front_and_back`; use exact mesh patch names.

## Critical rules

1. Use `p_rgh`, generate `constant/g`, name alpha field after the phase (`alpha.<phase1Name>`).
2. **Do NOT write `solver isoAdvector;`** in the alpha block — this is not the correct config style. isoAdvection sharpness is governed by reconstruction controls (`reconstructionScheme`, `vof2IsoTol`, etc.).
3. Use canonical isoAdvector parameter names: `vof2IsoTol`, `surfCellTol`, `nAlphaBounds`, `snapTol`, `clip`. Do not use `isoFaceTol`.
4. `div(phi,alpha)` scheme: use `Gauss vanLeer` for boundedness. Do NOT require `Gauss isoAdvectorScheme` unless verified available in the build.
5. Do NOT use MULES-only options (`MULESCorr`, `nLimiterIter`) as the primary interface sharpener.
6. `interpolationSchemes` must remain simple (`default linear;`) — no nonstandard `interface interfaceCompression`.
7. `startFrom startTime; startTime 0;` — NEVER `latestTime`.
