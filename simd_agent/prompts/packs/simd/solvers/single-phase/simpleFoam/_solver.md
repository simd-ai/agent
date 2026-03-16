# Solver: simpleFoam — Identity & Global Rules

**Algorithm**: SIMPLE (steady-state)
**Compressible**: no — incompressible
**Pressure field**: `p` — kinematic pressure, m²/s², `[0 2 -2 0 0 0 0]`
**Energy equation**: none — do NOT generate `0/T`
**Gravity**: no `constant/g`

---

## Required files

| Directory | Files |
|-----------|-------|
| `system/` | `controlDict`, `fvSchemes`, `fvSolution` |
| `constant/` | `transportProperties`, `turbulenceProperties` |
| `0/` | `U`, `p`, and turbulence fields matching the selected model |

**Never generate**: `0/T`, `constant/thermophysicalProperties`, `constant/g`

---

## Global critical rules

1. Every mesh patch in `patch_names` MUST appear in every `0/*` field file.
2. `application` in `controlDict` MUST equal `simpleFoam`.
3. Pressure is **kinematic** (`[0 2 -2 0 0 0 0]`, m²/s²) — NOT Pa.
4. `0/p` needs a reference cell/value in `fvSolution` when NO fixed-value pressure BC exists (closed domain). Set `pRefCell 0; pRefValue 0;` in the `SIMPLE {}` block.
5. `controlDict` `endTime` = `max_iterations` (integer); `deltaT 1`.
6. `startFrom startTime; startTime 0;` — never `latestTime`.
7. Do NOT invent fields or patches not listed in CaseSpec.
