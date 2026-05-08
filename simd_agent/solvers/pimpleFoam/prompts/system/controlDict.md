# pimpleFoam — system/controlDict

**Transient**: `endTime` in physical seconds, `deltaT` in seconds.

```
application     pimpleFoam;

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         <end_time>;     // physical seconds, from CaseSpec.end_time
deltaT          <delta_t>;      // physical seconds, from CaseSpec.delta_t

writeControl    runTime;
writeInterval   <writeInterval>;   // endTime / 30 -> ~30 snapshots (in seconds)
purgeWrite      0;

writeFormat     ascii;
writePrecision  6;
writeCompression off;

timeFormat      general;
timePrecision   6;

runTimeModifiable true;

adjustTimeStep  yes;
maxCo           0.5;
maxDeltaT       1;
```

## Rules

- `endTime` is in **seconds** (not iterations like SIMPLE solvers)
- `deltaT` is in **seconds** — initial time step; adjustTimeStep adapts it
- `writeControl runTime` with `writeInterval` in seconds is standard
- `writeInterval`: use `CaseSpec.end_time / 30` to get ~30 snapshots
- `application pimpleFoam;` — must match the solver name exactly
- `adjustTimeStep yes` with `maxCo 0.5` is the standard safe configuration
- `maxCo 0.5` — at 0.9, one bad cell can push CFL > 1 and cause instability
- `maxDeltaT` — upper bound on deltaT; set to 1s or a fraction of endTime

## Adaptive time stepping

Always include adaptive time stepping for transient runs:
```
adjustTimeStep  yes;
maxCo           0.5;
maxDeltaT       1;
```

This lets OpenFOAM automatically reduce deltaT when CFL exceeds 0.5 (any cell),
preventing instability without requiring the user to manually tune deltaT.
Without this, a fixed deltaT that is too large causes immediate divergence.

## 2D / 3D notes

- No changes to controlDict for 2D vs 3D — the solver handles dimensionality via boundary types in field files
- For 2D, the simulation may converge faster — but endTime is still physical seconds from CaseSpec
