# Solver: compressibleInterIsoFoam  ·  OpenFOAM v2406

**Type**: Transient · Compressible · Two-phase VOF · Non-isothermal · isoAdvection (geometric interface advection)  
**Pressure field**: `p_rgh` (kinematic pressure minus hydrostatic head), dimensions `[0 2 -2 0 0 0 0]` → MUST generate `0/p_rgh`  
**Energy equation**: ✅ YES → MUST generate `0/T` (Kelvin)  
**Gravity**: ✅ ALWAYS generate `constant/g` (even if `(0 0 0)`)  
**Alpha field**: ✅ `0/alpha.<phase1Name>` (name follows phase1)  
**Thermophysical**: ✅ base `thermophysicalProperties` + per-phase thermo dictionaries

---

## Difference from compressibleInterFoam

`compressibleInterIsoFoam` uses **isoAdvection (geometric VOF advection)** to keep a sharper interface than classic MULES-only approaches.

Practical codegen intent:
- Same overall case structure as `compressibleInterFoam` (p_rgh + T + per-phase thermo + gravity + pcorr),
- BUT the **alpha controls** should be isoAdvection-style (do not emit MULES-only knobs as the "main sharpener").

---

## Phase naming (CRITICAL)

Use phase names from config if provided:
- `phase1Name = config.phases[0]`
- `phase2Name = config.phases[1]`

If not provided, default:
- `(water air)`

Rules:
- Alpha field: `0/alpha.<phase1Name>` (e.g. `0/alpha.water`) — do NOT hardcode `alpha.phase1` unless phase1Name literally is `phase1`.
- Per-phase thermo files:
  - `constant/thermophysicalProperties.<phase1Name>`
  - `constant/thermophysicalProperties.<phase2Name>`

---

## Required files

### system/
| File | Notes |
|------|------|
| `system/controlDict` | `application compressibleInterIsoFoam;` `startFrom startTime; startTime 0;` |
| `system/fvSchemes` | VOF + compressible convection schemes (robust — see below) |
| `system/fvSolution` | PIMPLE + isoAdvection alpha controls + REQUIRED `pcorr` block |

### 0/
| File | Notes |
|------|------|
| `0/U` | Velocity |
| `0/p_rgh` | Kinematic `p_rgh` `[0 2 -2 0 0 0 0]` |
| `0/T` | Temperature `[0 0 0 1 0 0 0]` (Kelvin) |
| `0/alpha.<phase1Name>` | Volume fraction |
| `0/k`, `0/omega` | If turbulence model is kOmegaSST (compressible) |
| `0/k`, `0/epsilon` | If turbulence model is kEpsilon (compressible) |
| `0/mut` | If turbulence enabled (compressible uses `mut`, not `nut`) |

### constant/
| File | Notes |
|------|------|
| `constant/g` | ALWAYS required |
| `constant/thermophysicalProperties` | Base: phases + sigma + pMin (+ any solver-required keys) |
| `constant/thermophysicalProperties.<phase1Name>` | Per-phase thermo |
| `constant/thermophysicalProperties.<phase2Name>` | Per-phase thermo |
| `constant/turbulenceProperties` | Always generate (`laminar` / `RAS` / `LES`) |

**Never generate**:
- `constant/transportProperties` (sigma/pMin live in base `thermophysicalProperties` for this pack)
- `0/nut` (use `0/mut` if turbulent)
- `0/p` unless your runtime explicitly requires it

---

## constant/g

```
dimensions [0 1 -2 0 0 0 0];
value      (0 -9.81 0);  // or (0 0 0) if gravity=false
```

---

## constant/thermophysicalProperties (base) — minimal template

```
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      thermophysicalProperties;
}

phases ( <phase1Name> <phase2Name> );

pMin   [1 -1 -2 0 0 0 0]  10000;
sigma  [1  0 -2 0 0 0 0]  0.07;
```

---

## Per-phase thermo files

Use the same per-phase thermo structure as your `compressibleInterFoam` pack:
- `constant/thermophysicalProperties.<phaseName>`

Rules:
- If config provides thermoType/EOS/transport, use as-is.
- Otherwise choose conservative defaults (do not "police realism").

---

## fvSolution: Key difference — isoAdvection alpha controls (robust)

Do NOT use MULES-only parameters (`MULESCorr`, `nLimiterIter`, `alphaApplyPrevCorr`) as the main sharp-interface mechanism here.

Do NOT write `solver isoAdvector;` as a single line.

Rules:
- `p_rgh` GAMG smoother MUST be `GaussSeidel` (NOT DIC — causes SIGFPE crashes).
- `pcorr` block is REQUIRED.

```
solvers
{
    p_rgh
    {
        solver      GAMG;
        smoother    GaussSeidel;
        tolerance   1e-7;
        relTol      0.01;
    }
    p_rghFinal
    {
        $p_rgh;
        relTol 0;
    }

    pcorr
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-5;
        relTol          0;
    }

    "(U|T|k|omega|epsilon|mut)"
    {
        solver      smoothSolver;
        smoother    symGaussSeidel;
        tolerance   1e-6;
        relTol      0.1;
    }
    "(U|T|k|omega|epsilon|mut)Final"
    {
        $U;
        relTol 0;
    }
}

PIMPLE
{
    momentumPredictor           no;
    nOuterCorrectors            1;
    nCorrectors                 2;
    nNonOrthogonalCorrectors    0;
}

// isoAdvection alpha controls (not a linear solver block)
isoAdvection
{
    nAlphaCorr          1;
    nAlphaSubCycles     1;
    cAlpha              1;

    reconstructionScheme plicRDF;
    vof2IsoTol          1e-8;
    surfCellTol         1e-6;
    nAlphaBounds        3;
    snapTol             1e-12;
    clip                true;
}
```

If your build expects the isoAdvection keys under a differently named dictionary, use that name. The important rule is: do not write `solver isoAdvector;` and keep the block syntactically correct.

---

## fvSchemes (robust)

Use a robust default for divSchemes, then explicitly set alpha and key compressible terms:

```
ddtSchemes      { default Euler; }
gradSchemes     { default Gauss linear; }

divSchemes
{
    default                             bounded Gauss upwind;

    div(rhoPhi,U)                       bounded Gauss linearUpwind grad(U);

    // alpha advection (isoAdvection changes reconstruction, not the fact alpha convects)
    div(phi,alpha)                      Gauss vanLeer;
    div(phirb,alpha)                    Gauss linear;

    div(rhoPhi,T)                       bounded Gauss linearUpwind grad(T);

    // turbulence convection (only if those fields exist)
    div(rhoPhi,k)                       bounded Gauss upwind;
    div(rhoPhi,omega)                   bounded Gauss upwind;
    div(rhoPhi,epsilon)                 bounded Gauss upwind;

    div((rho*nuEff)*dev2(T(grad(U))))   Gauss linear;
}

laplacianSchemes     { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes        { default corrected; }

fluxRequired { default no; p_rgh; pcorr; alpha.<phase1Name>; }

// wallDist only if turbulence enabled / wall functions used
wallDist { method meshWave; }
```

Do NOT add nonstandard `interface interfaceCompression` entries under `interpolationSchemes`.

---

## Turbulence fields

- If `turbulenceProperties` says `laminar`: do NOT generate `k`, `omega`, `epsilon`, `mut`.
- If `RAS/LES`: generate only what the chosen model needs.
- Compressible turbulence uses **`0/mut`** (dynamic turbulent viscosity), not `0/nut`.

---

## Critical rules

1. Generate `0/p_rgh` with kinematic dimensions `[0 2 -2 0 0 0 0]` (not Pa dims).
2. ALWAYS generate `0/T` and `constant/g`.
3. No `constant/transportProperties` for this pack (sigma/pMin in base thermoProperties).
4. Base `constant/thermophysicalProperties` + per-phase files required.
5. Alpha field name follows phase1 name: `0/alpha.<phase1Name>`.
6. `pcorr` solver block is required in `fvSolution`.
7. Alpha controls are isoAdvection-style; do NOT emit MULES-only sharpening knobs as the primary mechanism.
8. `startFrom startTime; startTime 0;` — NEVER `latestTime`.
9. Patch names must match mesh boundaries exactly; `empty` patches must be `type empty` in ALL generated `0/*` files.
