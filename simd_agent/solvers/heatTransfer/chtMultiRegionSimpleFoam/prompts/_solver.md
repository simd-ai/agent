# chtMultiRegionSimpleFoam — global rules

Steady multi-region conjugate heat transfer.

## Identity

- **Algorithm:** SIMPLE outer loop with per-region inner solves.
- **Pressure field (fluid regions):** `p_rgh`.
- **Energy variable (fluid):** `h`; **(solid):** `T` (conduction only).
- **Density:** compressible in fluid regions (`heRhoThermo`); constant in
  solid regions (`rhoConst` via `heSolidThermo`).

## Multi-region file layout

```
constant/
  regionProperties              ← lists fluid + solid regions
  <region>/                     ← per region
    thermophysicalProperties
    turbulenceProperties        (fluid only)
    g                           (fluid only)
system/
  controlDict
  fvSchemes, fvSolution         ← top-level (outer-loop control)
  <region>/                     (Phase 2 — per-region schemes)
0/
  <region>/T                    ← coupled at fluid-solid interfaces
  <region>/U, p, p_rgh, k, ε    (fluid only)
```

## Phase 1 status

The deterministic renderer emits:
- ✅ `constant/regionProperties`
- ✅ `constant/<region>/thermophysicalProperties` (fluid + solid)

**TODO (Phase 2):**
- Per-region `system/<region>/fvSchemes` + `fvSolution`.
- Mapped `compressible::turbulentTemperatureCoupledBaffleMixed` BCs at
  fluid–solid interfaces.
- `changeDictionaryDict` per region.
- Multi-region case packaging.

## Reference tutorial

`OpenFOAM-4.x/tutorials/heatTransfer/chtMultiRegionSimpleFoam/multiRegionHeater`.
