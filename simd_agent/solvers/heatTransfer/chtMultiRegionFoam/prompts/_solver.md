# chtMultiRegionFoam — global rules

Transient multi-region conjugate heat transfer.

Same shape as ``chtMultiRegionSimpleFoam`` (see that file for the
per-region layout and Phase 1 / Phase 2 status) — only the algorithm
changes:

- **Algorithm:** PIMPLE outer loop with per-region inner solves.
- **ddt:** ``Euler`` or ``backward`` (Phase 2 picks via regime profile).
- **Per-region Final solvers** (``TFinal``, ``UFinal``, ``p_rghFinal``)
  follow the same PIMPLE Final-coverage invariant as our single-region
  PIMPLE solvers.

## Reference tutorial

`OpenFOAM-4.x/tutorials/heatTransfer/chtMultiRegionFoam/multiRegionHeater`.
