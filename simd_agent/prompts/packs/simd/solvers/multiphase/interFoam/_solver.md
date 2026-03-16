# Solver: interFoam — Identity & Global Rules

**Algorithm**: PIMPLE + MULES (transient, two-phase VOF)
**Compressible**: no
**Pressure field**: `p_rgh` — kinematic pressure minus hydrostatic head, `[0 2 -2 0 0 0 0]`
**Energy**: NONE — do NOT generate `0/T`
**Gravity**: REQUIRED — always generate `constant/g`

---

## Required files

| Directory | Files |
|-----------|-------|
| `system/` | `controlDict`, `fvSchemes`, `fvSolution` |
| `constant/` | `g`, `phaseProperties`, `turbulenceProperties` |
| `0/` | `U`, `p_rgh`, `alpha.<phase1Name>`, turbulence fields if active |

**Never generate**: `0/T`, `0/p` (only `p_rgh`), `constant/transportProperties` (unless runtime explicitly requires it instead of `phaseProperties`)

---

## Phase naming (CRITICAL)

- Use phase names from `CaseSpec.alpha_fields` or `CaseSpec.phases`.
- Default: `phase1Name = water`, `phase2Name = air`.
- Alpha file: `0/alpha.<phase1Name>` — NEVER `alpha.phase1` unless phase1Name is literally `phase1`.

---

## Global critical rules

1. Generate `0/p_rgh` (NOT `0/p`).
2. Always generate `constant/g` — use `(0 0 0)` if gravity is disabled.
3. Never generate `0/T` or any thermo files.
4. Alpha controls (`nAlphaCorr`, `nAlphaSubCycles`, `cAlpha`) go inside `PIMPLE {}` — not under `solvers {}`.
5. Use `div(phi,alpha) Gauss vanLeer;` and `div(phirb,alpha) Gauss linear;` for interface.
6. GAMG smoother MUST be `GaussSeidel` — never `DIC`.
7. `startFrom startTime; startTime 0;` — never `latestTime`.
8. Every mesh patch must appear in ALL `0/*` files; `empty` patches → `type empty`.
9. Only generate turbulence fields (`k`, `omega`, `epsilon`, `nut`) when turbulence is active.
