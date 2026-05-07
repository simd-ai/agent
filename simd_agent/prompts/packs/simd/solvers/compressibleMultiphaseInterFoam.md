# Solver: compressibleMultiphaseInterFoam  ·  OpenFOAM v2406

**Type**: Transient · Compressible · N-phase (N≥3) VOF · Non-isothermal  
**Pressure field**: `p_rgh` (kinematic pressure minus hydrostatic head), dimensions `[0 2 -2 0 0 0 0]` → MUST generate `0/p_rgh`  
**Energy equation**: ✅ YES → MUST generate `0/T` (Kelvin)  
**Gravity**: ✅ ALWAYS generate `constant/g`  
**Alpha fields**: ✅ `0/alphas` (composite) + one `0/alpha.<phaseName>` per phase  
**Thermophysical**: ✅ base file + one thermo dictionary per phase

---

## Phase naming (CRITICAL)

Use phase names from config (`config.phases[]`). N must be ≥ 3.

If phases are not provided, use a safe default list (example):
- `(water oil air)`

Rules:
- Generate `0/alpha.<phaseName>` for every phase name exactly as provided.
- Never invent phase names.
- `0/alphas` is always required for this solver pack.

---

## Required files

### system/
| File | Notes |
|------|------|
| `system/controlDict` | `application compressibleMultiphaseInterFoam;` `startFrom startTime; startTime 0;` |
| `system/fvSchemes` | transient + VOF div schemes + energy div schemes |
| `system/fvSolution` | PIMPLE + alpha/MULES controls + solvers for `p_rgh`, `pcorr` (if used), `U`, `T` |

### 0/
| File | Notes |
|------|------|
| `0/U` | Velocity |
| `0/p_rgh` | Kinematic `p_rgh` `[0 2 -2 0 0 0 0]` |
| `0/T` | Temperature `[0 0 0 1 0 0 0]` (Kelvin) |
| `0/alphas` | ALWAYS required |
| `0/alpha.<phaseName>` | ONE file per phase |
| `0/k`, `0/omega` | If turbulence model is kOmegaSST (compressible) |
| `0/k`, `0/epsilon` | If turbulence model is kEpsilon (compressible) |
| `0/mut` | If turbulence enabled (compressible uses `mut`, not nut) |

### constant/
| File | Notes |
|------|------|
| `constant/g` | ALWAYS required |
| `constant/thermophysicalProperties` | Base: phases + sigmas (per pair) + pMin |
| `constant/thermophysicalProperties.<phaseName>` | One per phase |
| `constant/turbulenceProperties` | Always generate (`laminar` / `RAS` / `LES`) |

**Never generate**:
- `constant/transportProperties` (this pack puts sigmas/pMin in base thermophysicalProperties)
- `0/nut` (use `0/mut` if turbulent)

---

## constant/g

```
dimensions [0 1 -2 0 0 0 0];
value      (0 -9.81 0);  // or (0 0 0) if gravity=false
```

---

## constant/thermophysicalProperties (base) template

Keep it minimal and syntactically valid.

```
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "constant";
    object      thermophysicalProperties;
}

phases ( <phase1> <phase2> <phase3> ... );  // list ALL phases

pMin   [1 -1 -2 0 0 0 0]  10000;   // Pa

sigmas
(
    (<phaseA> <phaseB>)  <sigmaAB>
    (<phaseA> <phaseC>)  <sigmaAC>
    (<phaseB> <phaseC>)  <sigmaBC>
    // include every unique pair present in config
);
```

Rules:
- Include surface tension for each phase pair required by your case. If config provides only one sigma, expand it consistently for all pairs (same value), but do not invent new phase names.
- `pMin` is absolute pressure floor (Pa), not `p_rgh`.

---

## constant/thermophysicalProperties.\<phaseName\> (per phase)

CONFIG-DRIVEN:
- If config provides thermoType/EOS/transport per phase, use it as-is.
- Otherwise choose conservative defaults per phase (do not "police realism").

Example skeleton:

```
thermoType
{
    type            heRhoThermo;              // or hePsiThermo if config says so
    mixture         pureMixture;
    transport       const;
    thermo          hConst;
    equationOfState rhoConst;                // or perfectGas, etc., from config
    specie          specie;
    energy          sensibleInternalEnergy;  // follow config if specified
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

If EOS requires extra parameters (e.g. rho for rhoConst), include them according to your runtime convention.

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
    RASModel        <modelName>;   // kOmegaSST / kEpsilon etc.
    turbulence      on;
    printCoeffs     on;
}
```

If laminar: do NOT generate `k/omega/epsilon/mut`.

---

## fvSolution template (robust)

Key rules:
- Use `GAMG` for `p_rgh` with `GaussSeidel` smoother (NOT DIC — causes SIGFPE crashes).
- Include alpha controls blocks for `alphas.*` and `alpha.*` (MULES-style; not linear solvers).
- Include `PIMPLE {}`.
- Include `pcorr` block (safe to include in most setups).

```
solvers
{
    // Alpha controls blocks (VOF/MULES-style; not linear solvers)
    "alphas.*"
    {
        nAlphaCorr      2;
        nAlphaSubCycles 1;
        cAlpha          1;
        MULESCorr       yes;
        nLimiterIter    3;
    }

    "alpha.*"
    {
        nAlphaCorr      2;
        nAlphaSubCycles 1;
        cAlpha          1;
        MULESCorr       yes;
        nLimiterIter    3;
    }

    p_rgh
    {
        solver      GAMG;
        smoother    GaussSeidel;
        tolerance   1e-7;
        relTol      0.01;
    }
    p_rghFinal { $p_rgh; relTol 0; }

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

Avoid `divSchemes default none` brittleness. Use a safe default and explicitly set alpha schemes:

```
ddtSchemes      { default Euler; }
gradSchemes     { default Gauss linear; }

divSchemes
{
    default                             bounded Gauss upwind;

    div(rhoPhi,U)                       bounded Gauss linearUpwind grad(U);

    // alpha advection + compression
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

fluxRequired { default no; p_rgh; pcorr; alphas; }

// wallDist only if turbulence enabled / wall functions used
wallDist { method meshWave; }
```

---

## Critical rules

1. Generate `0/p_rgh` with kinematic dimensions `[0 2 -2 0 0 0 0]` (not Pa dims).
2. ALWAYS generate `0/T` and `constant/g`.
3. No `constant/transportProperties` for this pack (sigmas + pMin in base thermophysicalProperties).
4. Base `constant/thermophysicalProperties` + per-phase thermo files required.
5. Always generate `0/alphas` and one `0/alpha.<phaseName>` per phase.
6. Use `GAMG` pressure smoother `GaussSeidel` (not DIC).
7. Alpha values should sum to 1 (conceptual rule; codegen should set consistent ICs).
8. `startFrom startTime; startTime 0;` — NEVER `latestTime`.
9. Patch names must match mesh boundaries exactly; `empty` patches must be `type empty` in ALL generated `0/*` files.
