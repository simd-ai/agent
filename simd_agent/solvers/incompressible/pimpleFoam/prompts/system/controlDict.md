# pimpleFoam — system/controlDict

**Transient**: `endTime` in physical seconds, `deltaT` in seconds.

```
application     pimpleFoam;

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         <end_time>;         // physical seconds, from CaseSpec
deltaT          <delta_t>;          // initial time step (seconds), from CaseSpec

writeControl    adjustableRunTime;
writeInterval   <write_interval>;   // from CaseSpec (~100 snapshots)
purgeWrite      0;

writeFormat     ascii;
writePrecision  6;
writeCompression off;

timeFormat      general;
timePrecision   6;

runTimeModifiable true;

adjustTimeStep  yes;
maxCo           <max_co>;           // from CaseSpec (2.0 for PIMPLE)
maxDeltaT       <max_delta_t>;      // from CaseSpec (= writeInterval)
```

## Rules

- `endTime` is in **seconds** (not iterations like SIMPLE solvers)
- `deltaT` is in **seconds** — initial time step; adjustTimeStep adapts it
- `writeControl adjustableRunTime` — ensures time steps land exactly on write times
- `writeInterval`: use `CaseSpec.write_interval` (computed to give ~100 snapshots)
- `application pimpleFoam;` — must match the solver name exactly
- `maxCo`: use `CaseSpec.max_co` — PIMPLE handles Co=2.0 safely via outer corrector loops (0.5 is overly conservative for PIMPLE, that's a PISO value)
- `maxDeltaT`: use `CaseSpec.max_delta_t` (= writeInterval) so deltaT never jumps past a snapshot

## Adaptive time stepping

Always include adaptive time stepping for transient runs:
```
adjustTimeStep  yes;
maxCo           <max_co>;       // from CaseSpec
maxDeltaT       <max_delta_t>;  // from CaseSpec
```

pimpleFoam's outer corrector loops (PIMPLE algorithm) make it stable at Co > 1.
Using maxCo 2.0 typically gives 4-10x speedup over maxCo 0.5 without loss of accuracy.

## 2D / 3D notes

- No changes to controlDict for 2D vs 3D — the solver handles dimensionality via boundary types in field files
- For 2D, the simulation may converge faster — but endTime is still physical seconds from CaseSpec
