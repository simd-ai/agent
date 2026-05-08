# Solver: pimpleFoam — Identity & Global Rules

**Algorithm**: PIMPLE (transient, pressure-velocity outer iterations)
**Compressible**: no — incompressible
**Pressure field**: `p` — kinematic pressure, m²/s², `[0 2 -2 0 0 0 0]`
**Energy equation**: none — do NOT generate `0/T` (unless heat transfer is enabled)
**Gravity**: no `constant/g` (use interFoam for buoyancy-driven flows)

pimpleFoam solves the transient Reynolds-averaged Navier-Stokes equations for incompressible flow using the PIMPLE algorithm (merged PISO-SIMPLE). It solves:
- **Continuity**: dU/dt + nabla . (UU) - nabla . (nu_eff nabla U) = -nabla p
- **Momentum**: nabla . U = 0
- **Turbulence** (optional): transport equations for the selected model (k-omega SST, k-epsilon, Spalart-Allmaras, or laminar)

---

## Required files

| Directory | Files |
|-----------|-------|
| `system/` | `controlDict`, `fvSchemes`, `fvSolution` |
| `constant/` | `transportProperties`, `turbulenceProperties` |
| `0/` | `U`, `p`, and turbulence fields matching the selected model |

**Never generate**: `0/T`, `0/h`, `0/e`, `constant/thermophysicalProperties`, `constant/g`

---

## 2D vs 3D simulations

OpenFOAM is inherently 3D. A "2D" simulation uses a mesh that is **one cell thick** in the unused direction. Special boundary types on the front/back faces tell the solver to ignore that direction.

CaseSpec provides: `is_2d` (bool), `patch_types` (dict mapping patch name to OpenFOAM type: `"empty"`, `"wedge"`, `"wall"`, `"patch"`, etc.).

### Planar 2D — `empty` patches

Used for: backward-facing step, 2D channel, flat plate, airfoil (2D slice).

- Mesh has exactly **1 cell** in the z-direction
- Front/back patches have type `empty` in `patch_types`
- Every `0/*` field file MUST include these patches with `type empty;`
- `empty` BC takes **no `value` sub-entry** — just `type empty;`
- Velocity z-component MUST be 0: `internalField uniform (<Ux> <Uy> 0)`
- `inletOutlet` / `fixedValue` on other patches: z-component = 0

Example in any `0/*` field:
```
frontAndBack
{
    type            empty;
}
```

### Axisymmetric 2D — `wedge` patches

Used for: pipe flow, nozzle, axisymmetric jet, rotating bodies.

- Mesh is a thin **wedge sector** (typically 5 degrees) with 1 cell in the circumferential direction
- Front/back patches have type `wedge` in `patch_types`
- The centerline axis may have a patch of type `empty` (zero-radius degenerate faces)
- Every `0/*` field file MUST include wedge patches with `type wedge;`
- `wedge` BC takes **no `value` sub-entry** — just `type wedge;`
- Velocity has axial + radial components; circumferential component = 0

Example in any `0/*` field:
```
front
{
    type            wedge;
}
back
{
    type            wedge;
}
axis
{
    type            empty;
}
```

### 3D — standard

- No `empty` or `wedge` patches
- All patches are `patch`, `wall`, `symmetry`, `symmetryPlane`, etc.
- Full 3D velocity `(Ux Uy Uz)` — all components may be non-zero

### Critical 2D rules

1. **Patch coverage**: `empty` and `wedge` patches MUST appear in EVERY `0/*` field file — missing them is a fatal error
2. **Consistent type**: The BC type in `0/*` files must match the mesh patch type exactly — `empty` for empty, `wedge` for wedge
3. **No value keyword**: `empty` and `wedge` BCs have NO sub-entries — just the `type` line
4. **Velocity direction**: For planar 2D in XY plane, Uz = 0 in all velocity BCs and internalField
5. **Symmetry is NOT empty**: `symmetry` patches are different from `empty` — do not confuse them

---

## HARD RULES — no compressible contamination

pimpleFoam is **strictly incompressible**. The following are completely absent from every generated file:

| What | Why |
|---|---|
| `0/rho` | Density is a constant scalar in `transportProperties`, NOT a field |
| `rho` solver in `fvSolution` | No rho equation — never add a `rho { diagonal; }` entry |
| `0/T` | No energy equation unless heat transfer is explicitly enabled |
| `constant/thermophysicalProperties` | Incompressible thermo is in `transportProperties` only |
| `div(phid,p)` in fvSchemes | Compressible pressure-flux term — does not exist in pimpleFoam |
| `div(phi,K)`, `div(phi,Ekp)` | Compressible energy flux terms — do not exist in pimpleFoam |
| `div(((rho*nuEff)*dev2(...)))` | Use `div((nuEff*dev2(T(grad(U)))))` — no `rho` factor |
| `massFlowRate` in `flowRateInletVelocity` | Requires rho field — convert to `volumetricFlowRate = mdot/rho` instead |
| `p` in Pa (`[1 -1 -2 0 0 0 0]`) | pimpleFoam p is kinematic (`[0 2 -2 0 0 0 0]`, m^2/s^2) |
| `alphat` field | Turbulent thermal diffusivity — only for compressible solvers |

---

## Global critical rules

1. Every mesh patch in `patch_names` MUST appear in every `0/*` field file — including `empty` and `wedge` patches.
2. `application` in `controlDict` MUST equal `pimpleFoam`.
3. Pressure is **kinematic** (`[0 2 -2 0 0 0 0]`, m^2/s^2) — NOT Pa.
4. `controlDict` uses physical time: `endTime` in seconds, `deltaT` in seconds.
5. `startFrom startTime; startTime 0;` — never `latestTime`.
6. PIMPLE requires `nOuterCorrectors`, `nCorrectors`, `nNonOrthogonalCorrectors` in `fvSolution`.
7. Do NOT invent fields or patches not listed in CaseSpec.
8. For laminar flow: set `simulationType laminar;` in `turbulenceProperties`, omit ALL turbulence fields (`k`, `omega`, `epsilon`, `nut`), omit `div(phi,k/omega/epsilon)` from fvSchemes, omit `wallDist` block.
9. For 2D cases (`is_2d: true`), ensure all velocity BCs and internalField have 0 in the out-of-plane component.
10. Do NOT add `functions {}` block to `controlDict` — function objects are injected automatically by the validator. If you include them, they may be duplicated or conflict.
