# Solver: interIsoFoam  ·  OpenFOAM v2406

**Type**: Transient · Incompressible · Two-phase VOF · isoAdvection (geometric, sharper interface)  
**Pressure field**: `p_rgh` (kinematic pressure minus hydrostatic head), dimensions `[0 2 -2 0 0 0 0]` → MUST generate `0/p_rgh`  
**Energy equation**: ❌ No → do NOT generate `0/T`  
**Gravity file**: ✅ REQUIRED → MUST generate `constant/g` (use `(0 0 0)` if gravity=false)

---

## A) Required files (minimum working)

### system/
| File | Notes |
|------|------|
| `system/controlDict` | `application interIsoFoam;` · transient time control |
| `system/fvSchemes` | transient + VOF schemes (see template) |
| `system/fvSolution` | PIMPLE block + alpha/isoAdvection controls |

### constant/
| File | Notes |
|------|------|
| `constant/g` | REQUIRED (even if value is zero) |
| `constant/phaseProperties` | Preferred for v2406 two-phase configuration |
| `constant/transportProperties` | Only generate if your codebase explicitly expects this instead of phaseProperties |
| `constant/turbulenceProperties` | Always generate (`laminar` / `RAS` / `LES`) |

### 0/
| File | Notes |
|------|------|
| `0/U` | Velocity |
| `0/p_rgh` | Kinematic `p_rgh` |
| `0/alpha.<phase1Name>` | Volume fraction for phase1 |
| `0/k`, `0/omega`, `0/nut` | Only if turbulence model needs them (e.g., kOmegaSST) |
| `0/k`, `0/epsilon`, `0/nut` | Only if kEpsilon |

---

## B) Phase naming (CRITICAL)

Let `phase1Name` and `phase2Name` come from config (`validated_config.physics.phases[]`).

If not provided, default:
- `phase1Name = water`
- `phase2Name = air`

Alpha file MUST be:
- `0/alpha.<phase1Name>` (example: `0/alpha.water`)

Never invent `alpha.phase1` unless the phase name is literally `phase1`.

---

## C) constant/g

```
dimensions [0 1 -2 0 0 0 0];
value      (0 -9.81 0);  // or (0 0 0) if gravity=false
```

---

## D) constant/phaseProperties (preferred, v2406)

Provide phases, surface tension, and per-phase properties as required by your project convention.
Keep structure simple and syntactically valid.

Example minimal pattern (adapt names/values from config):

```
phases ( <phase1Name> <phase2Name> );

<phase1Name>
{
    transportModel  Newtonian;
    nu              [0 2 -1 0 0 0 0] <nu1>;
    rho             [1 -3 0 0 0 0 0] <rho1>;
}

<phase2Name>
{
    transportModel  Newtonian;
    nu              [0 2 -1 0 0 0 0] <nu2>;
    rho             [1 -3 0 0 0 0 0] <rho2>;
}

sigma           [1 0 -2 0 0 0 0] <sigma>;
```

If your existing backend uses `constant/transportProperties` for two-phase instead, keep that file instead of phaseProperties (but do not generate both unless your runtime expects both).

---

## E) constant/turbulenceProperties (always generate)

- laminar:
```
simulationType laminar;
```

- RAS:
```
simulationType RAS;
RAS
{
    RASModel        <modelName>;   // kOmegaSST or kEpsilon, etc.
    turbulence      on;
    printCoeffs     on;
}
```

- LES:
```
simulationType LES;
LES
{
    LESModel        <modelName>;
    turbulence      on;
    printCoeffs     on;
}
```

Only generate `0/k`, `0/omega`/`0/epsilon`, `0/nut` when turbulence is enabled AND the chosen model requires them.

---

## F) controlDict time-step control (recommended)

Use automatic timestep control for stability:

- `adjustTimeStep yes;`
- `maxCo <= 1;`
- `maxAlphaCo <= 1;` (often 0.5–1.0)
- `maxDeltaT <cap>;`

Never use `startFrom latestTime`.

---

## G) fvSolution: PIMPLE + alpha/isoAdvection controls (robust)

Rules:
- Must include `PIMPLE {}`.
- Configure alpha interface advection with bounded controls.
- Do NOT set `solver isoAdvector;` inside an alpha solver dictionary.

```
solvers
{
    p_rgh
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-7;
        relTol          0.01;
    }
    p_rghFinal { $p_rgh; relTol 0; }

    "(U|k|omega|epsilon)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-6;
        relTol          0.1;
    }
    "(U|k|omega|epsilon)Final"
    {
        $U;
        relTol 0;
    }

    // alpha equation is not a linear solver in the same way; keep simple if present
}

PIMPLE
{
    momentumPredictor   yes;
    nOuterCorrectors    1;
    nCorrectors         2;
    nNonOrthogonalCorrectors 0;

    // pRefCell/pRefValue only if all-Neumann pressure
    // pRefCell  0;
    // pRefValue 0;
}

// Alpha / isoAdvection-style controls (robust dictionary; keys may be build-dependent)
alphaControls
{
    nAlphaCorr      1;
    nAlphaSubCycles 1;
    cAlpha          1;

    // isoAdvection reconstruction controls (safe to include)
    reconstructionScheme plicRDF;
    vof2IsoTol      1e-8;
    surfCellTol     1e-6;
    nAlphaBounds    3;
    snapTol         1e-12;
    clip            true;
}
```

If your runtime expects these keys under a specific dictionary name (e.g. `isoAdvection`), use that name instead of `alphaControls`. The important rule is: **do not write `solver isoAdvector;`** and keep the block syntactically correct.

---

## H) fvSchemes (VOF-specific, robust)

Avoid `divSchemes default none` brittleness. Use a stable default and explicitly set alpha schemes:

```
ddtSchemes      { default Euler; }
gradSchemes     { default Gauss linear; }

divSchemes
{
    default                             bounded Gauss upwind;

    // momentum
    div(phi,U)                          bounded Gauss linearUpwind grad(U);

    // alpha advection
    div(phi,alpha)                      Gauss vanLeer;
    div(phirb,alpha)                    Gauss linear;

    // viscous term
    div((nuEff*dev2(T(grad(U)))))       Gauss linear;
}

laplacianSchemes     { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes        { default corrected; }

// include wallDist only when turbulence enabled / wall functions used
wallDist             { method meshWave; }
```

---

## I) 2D mesh constraint patches

If the mesh includes an `empty` patch (e.g. `frontAndBack`):
- In `0/U`, `0/p_rgh`, `0/alpha.<phase1Name>` that patch MUST be `{ type empty; }`.
- Never invent patch names; use exact mesh patch names from config.

---

## Critical rules

1. Generate `0/p_rgh` (dimensions `[0 2 -2 0 0 0 0]`), `0/U`, and `0/alpha.<phase1Name>`.
2. Always generate `constant/g` (use `(0 0 0)` if gravity disabled).
3. Do NOT generate `0/T` or any thermo files.
4. Do NOT write `solver isoAdvector;` — isoAdvection sharpness is controlled via reconstruction parameters.
5. Use `div(phi,alpha) Gauss vanLeer;` for boundedness; avoid requiring nonstandard schemes not guaranteed in the build.
6. Prefer robust `divSchemes default bounded Gauss upwind;` to avoid missing div entries.
7. `startFrom startTime; startTime 0;` — NEVER `latestTime`.
