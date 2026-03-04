# Solver: pimpleFoam  ·  OpenFOAM v2406

**Type**: Transient · Incompressible · PIMPLE (merged SIMPLE+PISO)
**What it solves**: Incompressible Newtonian flow; turbulence model can be **laminar / RAS / LES** (selected by config).
**Pressure field**: `p` (kinematic pressure) — solver **reads `0/p` and `0/U` at startup**.
**Energy equation**: ❌ No (do **NOT** generate `0/T`)
**Gravity file**: ❌ No `constant/g` (unless you are using a buoyant solver)

---

## Required files

| File                            | Notes                                                                    |
| ------------------------------- | ------------------------------------------------------------------------ |
| `system/controlDict`            | `application pimpleFoam;` · transient time control (`deltaT`, `endTime`) |
| `system/fvSchemes`              | transient schemes + convection + viscous term                            |
| `system/fvSolution`             | **Must contain `PIMPLE {}`**                                             |
| `0/U`                           | Velocity (all patches)                                                   |
| `0/p`                           | Kinematic pressure (all patches)                                         |
| `constant/transportProperties`  | kinematic viscosity `nu`                                                 |
| `constant/turbulenceProperties` | **ONLY if** turbulence enabled (RAS/LES)                                 |
| `0/k`, `0/omega`, `0/nut`       | ONLY if turbulence model needs them (e.g., kOmegaSST)                    |
| `0/k`, `0/epsilon`, `0/nut`     | ONLY if kEpsilon                                                         |

---

## controlDict template (transient)

**Rule**: `endTime` and `deltaT` are **physical time controls**, not "iterations".

```
/// system/controlDict (core parts)
application     pimpleFoam;

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         <solver.endTime>;     // seconds (physical time)
deltaT          <solver.delta_t>;     // seconds

writeControl    timeStep;
writeInterval   <solver.write_interval>;  // e.g. 50
purgeWrite      0;

runTimeModifiable true;

// Optional for robustness:
adjustTimeStep  yes;
maxCo           0.9;
```

---

## constant/transportProperties (incompressible)

Different OpenFOAM "families" accept slightly different syntaxes; both of these are common:

**Variant A (very common):**

```
transportModel  Newtonian;
nu              [0 2 -1 0 0 0 0] 1.5e-05;
```

**Variant B (also common, especially in many tutorials):**

```
transportModel  Newtonian;
nu              nu [0 2 -1 0 0 0 0] 1.5e-05;
```

Pick **one** style and be consistent.

---

## fvSolution template (FIXED)

Important: `PIMPLE` controls looping. `nCorrectors` etc are PIMPLE parameters.

> **Solver selection rule:** `GAMG` for symmetric elliptic equations (pressure `p`).
> `smoothSolver` / `PBiCGStab` for asymmetric transport equations (`U`, turbulence).

```
/// system/fvSolution
solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-06;
        relTol          0.05;
        smoother        GaussSeidel;
    }
    pFinal
    {
        $p;
        relTol          0;
    }

    U
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-05;
        relTol          0.1;
    }
    UFinal
    {
        $U;
        relTol          0;
    }

    // Only include these if turbulence model requires them:
    k
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-05;
        relTol          0.1;
    }
    kFinal
    {
        $k;
        relTol          0;
    }

    omega
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-05;
        relTol          0.1;
    }
    omegaFinal
    {
        $omega;
        relTol          0;
    }

    epsilon
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-05;
        relTol          0.1;
    }
    epsilonFinal
    {
        $epsilon;
        relTol          0;
    }
}

PIMPLE
{
    nOuterCorrectors        1;
    nCorrectors             2;
    nNonOrthogonalCorrectors 0;

    // Only needed if pressure is all-Neumann (no fixedValue p anywhere)
    // pRefCell  0;
    // pRefValue 0;
}
```

**Note:** Under-relaxation is usually not required for transient PIMPLE; if you include `relaxationFactors`, keep it conservative and only when you know you need it.

---

## fvSchemes template

The viscous stress divergence term is commonly written with **`dev2(T(grad(U)))`** (not `dev(T(grad(U)))`).

```
/// system/fvSchemes
ddtSchemes
{
    default         Euler;
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default                             none;
    div(phi,U)                          bounded Gauss linearUpwind grad(U);
    div(phi,k)                          bounded Gauss linearUpwind grad(k);
    div(phi,omega)                      bounded Gauss linearUpwind grad(omega);
    div(phi,epsilon)                    bounded Gauss linearUpwind grad(epsilon);

    div((nuEff*dev2(T(grad(U)))))        Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}

// ONLY if wall functions / wall distance are needed (RAS/LES with walls)
wallDist
{
    method          meshWave;
}
```

---

## Critical rules

1. **Must generate `0/p` and `0/U`** (pimpleFoam reads them as `MUST_READ`).
2. `PIMPLE {}` block is required, and parameters like `nCorrectors` belong there.
3. If `turbulence_model == laminar/none`: **do not generate** `0/k`, `0/omega`, `0/epsilon`, `0/nut`, or `constant/turbulenceProperties`.
4. Don't generate buoyancy files (`constant/g`, `thermophysicalProperties`) for `pimpleFoam`.
5. Never mix field aliases incorrectly: **`omegaFinal { $omega; }`**, **not** `$k`.
6. `startFrom startTime; startTime 0;` — **NEVER** `startFrom latestTime`.