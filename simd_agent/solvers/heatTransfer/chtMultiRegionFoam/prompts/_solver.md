# chtMultiRegionFoam — global rules

Transient multi-region conjugate heat transfer.

Same shape as ``chtMultiRegionSimpleFoam`` — see that file for the full
per-region layout, coupled boundary rules, and status.  Only the
algorithm differs:

- **Algorithm:** PIMPLE outer loop with per-region inner solves.
- **ddt:** `Euler`.
- **Final variants:** Every per-region solver block has its `Final` pair
  (`hFinal`, `UFinal`, `p_rghFinal`, `rhoFinal`, `(k|epsilon)Final`)
  for the PIMPLE final outer iteration.

## Status

- ✅ **Phase 1 + 2:** RegionSpec contract + full per-region deterministic
  rendering (thermo, turbulence, g, fvSchemes, fvSolution, 0-fields with
  coupled T BCs, changeDictionaryDict).
- ⏳ **Phase 3:** Orchestrator + packaging tree-structured emission.

## LLM responsibility

Only `system/controlDict` is LLM-generated.  All per-region files are
rendered deterministically.

## Reference tutorial

`OpenFOAM-4.x/tutorials/heatTransfer/chtMultiRegionFoam/multiRegionHeater`.
