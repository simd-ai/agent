# Solver: simpleFoam  ·  OpenFOAM v2406

**Type**: Steady-state · Incompressible · SIMPLE (RANS optional)  
**Pressure field**: `p` (kinematic pressure), dimensions `[0 2 -2 0 0 0 0]`  
**Energy equation**: ❌ No — do NOT generate `0/T`  
**Gravity file**: ❌ No `constant/g`

## Required files

| File | Notes |
|------|-------|
| `system/controlDict` | `application simpleFoam;` · `deltaT 1;` · `endTime` = max_iterations |
| `system/fvSchemes` | Gauss linearUpwind for div; Gauss linear for grad; wallDist if turbulent |
| `system/fvSolution` | Must have `SIMPLE { }` block + `relaxationFactors` |
| `0/U` | Velocity — MUST_READ — all patches |
| `0/p` | Kinematic pressure `[0 2 -2 0 0 0 0]` — MUST_READ — all patches |
| `constant/transportProperties` | `nu [0 2 -1 0 0 0 0] <value>;` |
| `constant/turbulenceProperties` | **Always generate** — set `simulationType laminar/RAS/LES` |
| `0/k`, `0/omega`, `0/nut` | Only if turbulence model is kOmegaSST |
| `0/k`, `0/epsilon`, `0/nut` | Only if turbulence model is kEpsilon |

**Never generate**: `0/T` · `constant/thermophysicalProperties` · `constant/g`

## controlDict (steady)

Treat `endTime` as iteration count (pseudo-time):

```
startFrom startTime;
startTime 0;
deltaT    1;
endTime   <max_iterations>;
```

Never use `startFrom latestTime`.

## Pressure reference

Only include `pRefCell`/`pRefValue` (in `SIMPLE` dictionary) when pressure is **not** fixed on any patch (e.g. all `p` BCs are `zeroGradient`). If you have a `fixedValue` pressure outlet, you can omit it.

## fvSolution template

> **Solver selection rule:** `GAMG` for symmetric elliptic equations (pressure `p`).
> `smoothSolver` / `PBiCGStab` for asymmetric transport equations (`U`, turbulence).

```
solvers
{
    p
    {
        solver      GAMG;
        smoother    GaussSeidel;
        tolerance   1e-06;
        relTol      0.1;
    }
    pFinal
    {
        $p;
        relTol      0;
    }

    U
    {
        solver      smoothSolver;
        smoother    symGaussSeidel;
        tolerance   1e-05;
        relTol      0.1;
    }

    k            { $U; }
    kFinal       { $k; relTol 0; }

    omega        { $U; }
    omegaFinal   { $omega; relTol 0; }

    epsilon      { $U; }
    epsilonFinal { $epsilon; relTol 0; }
}

SIMPLE
{
    nNonOrthogonalCorrectors 0;

    // Include ONLY if needed (all-Neumann pressure / closed system):
    // pRefCell  0;
    // pRefValue 0;

    residualControl
    {
        p       1e-4;
        U       1e-4;
        k       1e-4;
        omega   1e-4;
        epsilon 1e-4;
    }
}

relaxationFactors
{
    fields
    {
        p       0.3;
    }
    equations
    {
        U       0.7;
        k       0.7;
        omega   0.7;
        epsilon 0.7;
    }
}
```

## fvSchemes template

```
ddtSchemes      { default steadyState; }
gradSchemes     { default Gauss linear; }
divSchemes
{
    default                               none;

    div(phi,U)                            bounded Gauss linearUpwind grad(U);

    // turbulence convection (use only if the fields exist)
    div(phi,k)                            bounded Gauss limitedLinear 1;
    div(phi,omega)                        bounded Gauss limitedLinear 1;
    div(phi,epsilon)                      bounded Gauss limitedLinear 1;

    // viscous stress term (correct form)
    div((nuEff*dev2(T(grad(U)))))         Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes   { default corrected; }

// Only include wallDist if using wall functions / turbulence
wallDist        { method meshWave; }
```

## Critical rules

1. `startFrom startTime; startTime 0; deltaT 1;` — never use `latestTime`.
2. `stopAt endTime; endTime <max_iterations>;`
3. Pressure is **kinematic** — divide absolute pressure by density if needed.
4. Do NOT generate `constant/g`, `constant/thermophysicalProperties`, or `0/T`.
5. For walls: `U` → `noSlip`; `p` → `zeroGradient`; `k`/`omega` → wall functions.
6. `pRefCell`/`pRefValue` only when pressure has no `fixedValue` patch (all-Neumann system).
7. `constant/turbulenceProperties` must **always** be generated; use `simulationType laminar;` for laminar cases.
8. `nut` is a derived field — do NOT add it to `relaxationFactors/equations`.
9. Viscous stress term must be `dev2(T(grad(U)))`, not `dev(T(grad(U)))`.
10. Every mesh patch (including `empty` patches for 2D) must appear in every `0/*` field file.
