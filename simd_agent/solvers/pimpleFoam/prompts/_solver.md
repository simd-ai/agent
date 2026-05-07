# Solver: pimpleFoam — Identity & Global Rules

**Algorithm**: PIMPLE (transient, pressure-velocity outer iterations)
**Compressible**: no — incompressible
**Pressure field**: `p` — kinematic pressure, m²/s², `[0 2 -2 0 0 0 0]`
**Energy equation**: none — do NOT generate `0/T` (unless heat transfer is enabled)
**Gravity**: no `constant/g` (use interFoam for buoyancy-driven flows)

---

## Required files

| Directory | Files |
|-----------|-------|
| `system/` | `controlDict`, `fvSchemes`, `fvSolution` |
| `constant/` | `transportProperties`, `turbulenceProperties` |
| `0/` | `U`, `p`, and turbulence fields matching the selected model |

**Never generate**: `constant/thermophysicalProperties`, `constant/g` (unless explicitly required)

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
| `massFlowRate` in `flowRateInletVelocity` | Requires rho field — convert to `volumetricFlowRate = ṁ/ρ` instead |
| `p` in Pa (`[1 -1 -2 0 0 0 0]`) | pimpleFoam p is kinematic (`[0 2 -2 0 0 0 0]`, m²/s²) |
| `alphat` field | Turbulent thermal diffusivity — only for compressible solvers |

---

## Global critical rules

1. Every mesh patch in `patch_names` MUST appear in every `0/*` field file.
2. `application` in `controlDict` MUST equal `pimpleFoam`.
3. Pressure is **kinematic** (`[0 2 -2 0 0 0 0]`, m²/s²) — NOT Pa.
4. `controlDict` uses physical time: `endTime` in seconds, `deltaT` in seconds.
5. `startFrom startTime; startTime 0;` — never `latestTime`.
6. PIMPLE requires `nOuterCorrectors`, `nCorrectors`, `nNonOrthogonalCorrectors` in `fvSolution`.
7. Do NOT invent fields or patches not listed in CaseSpec.
8. For laminar flow: set `simulationType laminar;` in `turbulenceProperties`, omit ALL turbulence fields (`k`, `omega`, `epsilon`, `nut`), omit `div(phi,k/omega/epsilon)` from fvSchemes, omit `wallDist` block.
