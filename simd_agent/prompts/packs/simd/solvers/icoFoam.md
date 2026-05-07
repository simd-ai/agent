# Solver Pack: icoFoam  ·  OpenFOAM v2406

**Type**: Transient · Incompressible · Laminar (Newtonian) · PISO  
**Pressure field**: `p` (kinematic pressure), dimensions `[0 2 -2 0 0 0 0]`  
**Energy equation**: ❌ No — do NOT generate `0/T`  
**Turbulence**: ❌ NONE — do NOT generate turbulence fields or `constant/turbulenceProperties`  
**Gravity**: ❌ No — do NOT generate `constant/g`

---

## A) Required files (minimum working)

| File | Notes |
|------|-------|
| `system/controlDict` | `application icoFoam;` · `deltaT` + `endTime` are physical time |
| `system/fvSchemes` | Euler ddt; linear schemes; no wallDist |
| `system/fvSolution` | MUST include `PISO { }` block (NOT SIMPLE/PIMPLE) |
| `0/U` | Velocity — all patches |
| `0/p` | Kinematic pressure — all patches |
| `constant/transportProperties` | MUST include `nu` only |

Do NOT generate:
- `constant/turbulenceProperties`
- `constant/g`
- `0/T`
- `0/k`, `0/omega`, `0/epsilon`, `0/nut`

---

## B) Time control rules (transient)

- `endTime` is PHYSICAL TIME (seconds), not iteration count.
- `deltaT` is a physical time step (seconds).

`controlDict` MUST use:

```
application icoFoam;

startFrom   startTime;
startTime   0;
stopAt      endTime;

deltaT      <solver.delta_t>;
endTime     <solver.endTime>;

writeControl  timeStep;
writeInterval <solver.write_interval>;

runTimeModifiable true;

// Optional stability guard (recommended)
adjustTimeStep yes;
maxCo 0.9;
```

Never use `startFrom latestTime` unless explicitly requested.

---

## C) 2D handling (empty patches)

If the mesh contains any patch with `patch_type == empty` (commonly `frontAndBack`):
- In **every** generated `0/*` file (here: `0/U` and `0/p`), that patch MUST be:
```
type empty;
```
- Never invent patch names; use exact mesh patch names from config.

If the mesh patch type is NOT `empty`, you MUST NOT use `type empty;` for that patch.

---

## D) Patch names are the source of truth

- Use EXACT patch names from `config.mesh.patches[].name` (case-sensitive).
- Every patch must appear in BOTH `0/U` and `0/p`.

---

## E) fvSolution template (PISO) — robust

Rules:
- Pressure is symmetric elliptic → use `GAMG` (robust) or `PCG` (small cases).
- `pRefCell/pRefValue` only when pressure system is all-Neumann (no fixedValue p anywhere).

```
solvers
{
    p
    {
        solver          GAMG;
        smoother        GaussSeidel;
        tolerance       1e-6;
        relTol          0.05;
    }
    pFinal
    {
        $p;
        relTol 0;
    }

    U
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-5;
        relTol          0;
    }
}

PISO
{
    nCorrectors              2;
    nNonOrthogonalCorrectors 0;

    // Include ONLY if needed (all-Neumann pressure):
    // pRefCell  0;
    // pRefValue 0;
}
```

---

## F) fvSchemes template (safe defaults)

```
ddtSchemes      { default Euler; }
gradSchemes     { default Gauss linear; }

divSchemes
{
    default     none;
    div(phi,U)  Gauss linear;
}

laplacianSchemes     { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes        { default corrected; }
```

---

## G) constant/transportProperties (correct syntax)

Must define ONLY kinematic viscosity `nu`:

```
transportModel  Newtonian;
nu              [0 2 -1 0 0 0 0] <nu_value>;
```

---

## Critical rules

1. icoFoam is LAMINAR. Never add turbulence files.
2. `constant/transportProperties` contains only `nu`.
3. Use `PISO {}` (NOT SIMPLE or PIMPLE).
4. `0/U` and `0/p` must include all mesh patches with exact names.
5. For 2D meshes: any `empty` patch must be `{ type empty; }` in every `0/*` file.
6. Time settings are physical (`deltaT`, `endTime`), not iteration counts.
