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
writeInterval   <writeInterval>;   // seconds between writes (not iterations)
purgeWrite      0;

writeFormat     ascii;
writePrecision  6;
writeCompression off;

timeFormat      general;
timePrecision   6;

runTimeModifiable true;

// Optional: CFL-based adaptive time stepping
// adjustTimeStep  yes;
// maxCo           0.9;
// maxDeltaT       0.01;
```

## Rules

- `endTime` is in **seconds** (not iterations like SIMPLE solvers)
- `deltaT` is in **seconds** — must satisfy CFL < 1 for pure PISO (nOuterCorrectors 1)
- `writeControl runTime` with `writeInterval` in seconds is standard
- If using adaptive time stepping: add `adjustTimeStep yes; maxCo 0.9;`
