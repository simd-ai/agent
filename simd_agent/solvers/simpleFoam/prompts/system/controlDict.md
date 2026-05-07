# simpleFoam — system/controlDict

**Steady-state**: `endTime` = iteration count (integer), `deltaT 1`.

```
application     simpleFoam;

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         <max_iterations>;   // INTEGER — e.g. 1000, never 1000.0
deltaT          1;

writeControl    runTime;
writeInterval   <writeInterval>;
purgeWrite      0;

writeFormat     ascii;
writePrecision  6;
writeCompression off;

timeFormat      general;
timePrecision   6;

runTimeModifiable true;
```

## Rules

- `endTime` MUST be an integer (no decimal point) — e.g. `1000` not `1000.0`
- `deltaT 1` always for steady-state (each iteration is one "time step")
- `application simpleFoam;` — must match the solver name exactly
- `writeInterval`: use `CaseSpec.end_time / 10` or similar to get ~10 snapshots

## 2D / 3D notes

- No changes to controlDict for 2D vs 3D — the solver handles dimensionality via boundary types in field files
- For 2D, iterations typically converge faster — a lower `endTime` (e.g. 500) may suffice, but use CaseSpec value
