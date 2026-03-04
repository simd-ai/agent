# Solver Pack: icoFoam  ·  OpenFOAM v2406

**Type**: Transient · Incompressible · Laminar (Newtonian) · PISO  
**Primary pressure field**: `p` (kinematic pressure, dimensions `[0 2 -2 0 0 0 0]`)  
**Energy equation**: ❌ No — do NOT generate `0/T`  
**Turbulence**: ❌ NONE — do NOT generate `0/k`, `0/omega`, `0/epsilon`, `0/nut`, or `constant/turbulenceProperties`  
**Gravity**: ❌ No — do NOT generate `constant/g`

## OpenFOAM tutorial notes (important)

- icoFoam is used for laminar, isothermal, incompressible flow. 2D is achieved via a 1-cell-thick mesh and an `empty` `frontAndBack` patch. (Tutorial 2.1)
- For icoFoam, the only physical property required is kinematic viscosity `nu` in `constant/transportProperties`.
- For temporal stability/accuracy with icoFoam, choose `deltaT` so Courant number Co < 1.
- icoFoam uses the PISO algorithm (`pisoControl` in solver source).

## A) Required files (minimum working)

| File | Notes |
|------|-------|
| `system/controlDict` | `application icoFoam;` · deltaT · endTime = physical time |
| `system/fvSchemes` | Euler ddt; linear interpolation; no wallDist needed |
| `system/fvSolution` | MUST include `PISO { }` block — NOT `SIMPLE` or `PIMPLE` |
| `0/U` | Velocity — all patches |
| `0/p` | Kinematic pressure `[0 2 -2 0 0 0 0]` — all patches |
| `constant/transportProperties` | MUST include `nu` only |

Do NOT generate:
- `constant/turbulenceProperties`
- `constant/g`
- `0/T`
- any turbulence fields (`k` / `omega` / `epsilon` / `nut`)

## B) Time control rules (transient)

- `endTime` is PHYSICAL TIME (not iteration count).
- `controlDict` MUST use:
  ```
  startFrom   startTime;
  startTime   0;
  stopAt      endTime;
  endTime     <config.solver.endTime>;   // physical time
  ```
- `deltaT` MUST be chosen/provided from `config.solver.delta_t` and should satisfy Co < 1.

Never use `startFrom latestTime` unless explicitly requested.

## C) 2D handling (frontAndBack)

If the mesh has a patch with `patch_type == empty` (commonly `frontAndBack`), then:
- In **every** `0/*` field file (here: `0/U` and `0/p`), that patch MUST be:
  ```
  { type empty; }
  ```
- The mesh must be 1 cell thick in the 3rd direction and the empty patches must be planar.

If the mesh `patch_type` is NOT `empty`, you MUST NOT use `type empty;` for that patch.

## D) Patch names are the source of truth

- Use EXACT patch names from `config.mesh.patches[].name` (case-sensitive).
- Never invent or rename patches (e.g. do NOT create `front_and_back`).
- Every patch must appear in BOTH `0/U` and `0/p` (OpenFOAM crashes if missing entries).

## E) fvSolution template (PISO)

```
solvers
{
    p
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-6;
        relTol          0.05;
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
        tolerance       1e-5;
        relTol          0;
    }
}

PISO
{
    nCorrectors             2;
    nNonOrthogonalCorrectors 0;
    pRefCell                0;
    pRefValue               0;
}
```

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

## G) constant/transportProperties

Must define `nu` (kinematic viscosity) only. Example:

```
nu  nu [0 2 -1 0 0 0 0]  <value>;
```

## Critical rules

1. icoFoam is LAMINAR. Never add turbulence files.
2. `transportProperties` needs only `nu` (no turbulence viscosity).
3. Use Co < 1 (Courant number): `deltaT` should satisfy Co = U·deltaT/cellSize < 1.
4. `PISO` block — NOT `SIMPLE` or `PIMPLE`.
5. Do NOT generate `constant/turbulenceProperties`.
6. For 2D cases: `frontAndBack` (or equivalent `empty` patch) must use `type empty;` in all field files, and the mesh must be 1 cell thick in the 3rd direction.
7. All patch names must exactly match `config.mesh.patches[].name` — every patch must appear in both `0/U` and `0/p`.