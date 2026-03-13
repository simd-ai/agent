# Solver: compressibleInterFoam  ·  OpenFOAM v2406

**Type**: Transient · Two-phase VOF · Compressible · Non-isothermal  
**Pressure field**: `p_rgh` (kinematic pressure minus hydrostatic head), dimensions `[0 2 -2 0 0 0 0]` → MUST generate `0/p_rgh`  
**Energy equation**: ✅ YES → MUST generate `0/T` (Kelvin)  
**Gravity**: ✅ ALWAYS generate `constant/g` (even if `(0 0 0)`)  
**Alpha field**: ✅ `0/alpha.<phase1Name>` (name follows phase1)  
**Thermophysical**: ✅ base file + per-phase thermo dictionaries

---

## Phase naming (CRITICAL)

Use phase names from config if provided:
- `phase1Name = config.phases[0]`
- `phase2Name = config.phases[1]`

If not provided, default to:
- `(water air)`

Rules:
- Alpha field: `0/alpha.<phase1Name>` (e.g. `0/alpha.water`) — do NOT hardcode `alpha.phase1` unless phase1Name literally is `phase1`.
- Per-phase thermo files follow the same naming:
  - `constant/thermophysicalProperties.<phase1Name>`
  - `constant/thermophysicalProperties.<phase2Name>`

---

## Required files

### system/
| File | Notes |
|------|------|
| `system/controlDict` | `application compressibleInterFoam;` `startFrom startTime; startTime 0;` |
| `system/fvSchemes` | VOF + compressible convection schemes (robust — see template) |
| `system/fvSolution` | PIMPLE + alpha controls + REQUIRED `pcorr` solver block |

### 0/
| File | Notes |
|------|------|
| `0/U` | Velocity |
| `0/p_rgh` | Kinematic `p_rgh` `[0 2 -2 0 0 0 0]` |
| `0/T` | Temperature `[0 0 0 1 0 0 0]` (Kelvin) |
| `0/alpha.<phase1Name>` | Volume fraction |
| `0/k`, `0/omega` | If turbulence model is kOmegaSST (compressible) |
| `0/k`, `0/epsilon` | If turbulence model is kEpsilon (compressible) |
| `0/mut` | If turbulence enabled (compressible models use **mut**, not nut) |

### constant/
| File | Notes |
|------|------|
| `constant/g` | ALWAYS required |
| `constant/thermophysicalProperties` | Base: phases + sigma + pMin (+ any solver-required keys) |
| `constant/thermophysicalProperties.<phase1Name>` | Per-phase thermo |
| `constant/thermophysicalProperties.<phase2Name>` | Per-phase thermo |
| `constant/turbulenceProperties` | Always generate (`laminar` / `RAS` / `LES`) |

**Never generate**:
- `constant/transportProperties` (sigma/pMin live in base thermophysicalProperties for this pack)
- `0/nut` (use `0/mut` if turbulent)
- `0/p` unless your pipeline explicitly asks for it

Optional (only if your runtime expects it): `0/rho` (some builds read/write it, many don't require it)

---

## constant/g

```
dimensions [0 1 -2 0 0 0 0];
value      (0 -9.81 0);  // or (0 0 0) if gravity=false
```

---

## constant/thermophysicalProperties (base) template

Keep this file minimal and syntactically valid.

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

pMin   [1 -1 -2 0 0 0 0]  10000;     // Pa (absolute pressure floor)
sigma  [1  0 -2 0 0 0 0]  0.07;       // N/m
```

Notes:
- `pMin` and `sigma` are Pa and N/m respectively (these are not `p_rgh` units; they are physical model parameters).

---

## constant/thermophysicalProperties.\<phaseName\> template (per phase)

CONFIG-DRIVEN rule:
- If validated_config provides thermoType/EOS/transport, use as-is.
- Otherwise choose conservative defaults per phase (gas often `perfectGas`, liquid often `rhoConst`), but do not "police realism".

Example skeleton:

```
thermoType
{
    type            heRhoThermo;
    mixture         pureMixture;
    transport       const;
    thermo          hConst;
    equationOfState perfectGas;              // or rhoConst if config says so
    specie          specie;
    energy          sensibleInternalEnergy;  // common default; follow config if specified
}

mixture
{
    specie
    {
        nMoles      1;
        molWeight   <molWeight>;
    }
    thermodynamics
    {
        Cp          <Cp>;
        Hf          0;
    }
    transport
    {
        mu          <mu>;
        Pr          <Pr>;
    }
}
```

If EOS is `rhoConst` (only if config says so), ensure the required density parameter is provided in the EOS section according to your runtime convention.

---

## constant/turbulenceProperties (always generate)

- laminar:
```
simulationType laminar;
```

- RAS:
```
simulationType RAS;
RAS
{
    RASModel        <modelName>;   // e.g. kOmegaSST or kEpsilon
    turbulence      on;
    printCoeffs     on;
}
```

- LES similarly if used.

If laminar: do NOT generate `k/omega/epsilon/mut`.

---

## fvSolution template (robust; avoid GAMG+DIC)

Rules:
- `p_rgh` uses `GAMG` with `GaussSeidel` smoother (NOT DIC — causes SIGFPE crashes).
- `pcorr` block is REQUIRED.
- Transport fields use `smoothSolver`/`PBiCGStab`.
- Alpha controls are MULES-style inside the alpha block.

```
solvers
{
    // Alpha controls block (VOF/MULES-style; not a linear solver)
    "alpha.<phase1Name>.*"
    {
        nAlphaCorr          1;
        nAlphaSubCycles     1;
        cAlpha              1;
        MULESCorr           yes;
        nLimiterIter        8;
        alphaApplyPrevCorr  yes;
    }

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
```

---

## fvSchemes template (robust)

Use a robust default for divSchemes, then explicitly set alpha and key compressible terms:

```
ddtSchemes      { default Euler; }
gradSchemes     { default Gauss linear; }

divSchemes
{
    default                             bounded Gauss upwind;

    div(rhoPhi,U)                       bounded Gauss linearUpwind grad(U);

    div(phi,alpha)                      Gauss vanLeer;
    div(phirb,alpha)                    Gauss linear;

    div(rhoPhi,T)                       bounded Gauss linearUpwind grad(T);

    // turbulence convection (only relevant if those fields exist)
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

## Critical rules

1. Generate `0/p_rgh` with kinematic dimensions `[0 2 -2 0 0 0 0]` (not Pa dims).
2. ALWAYS generate `0/T` and `constant/g`.
3. Do NOT generate `constant/transportProperties` for this pack (sigma/pMin are in base thermoProperties).
4. Provide base `constant/thermophysicalProperties` plus per-phase thermo files.
5. Alpha field name follows phase1: `0/alpha.<phase1Name>`.
6. `pcorr` solver block is REQUIRED in `fvSolution`.
7. Do NOT use `smoother DIC` inside GAMG; use `GaussSeidel`.
8. `startFrom startTime; startTime 0;` — NEVER `latestTime`.
9. Patch names must match mesh boundary exactly; `empty` patches (2D) must be `type empty` in ALL generated `0/*` files.
