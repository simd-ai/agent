# rhoSimpleFoam — system/controlDict

**Steady-state**: `endTime` = iteration count (integer), `deltaT 1`.

```
application     rhoSimpleFoam;

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         <max_iterations>;   // INTEGER — e.g. 1000, never 1000.0
deltaT          1;

writeControl    runTime;
writeInterval   <writeInterval>;    // from CaseSpec (default 100)
purgeWrite      0;

writeFormat     ascii;
writePrecision  6;
writeCompression off;

timeFormat      general;
timePrecision   6;

runTimeModifiable true;
```

## Rules

- `endTime` MUST be an integer — write `1000` not `1000.0`
- `deltaT 1` — always 1 for steady-state (iterations, not seconds)
- `startFrom startTime; startTime 0;` — never `latestTime`
- `application` MUST be `rhoSimpleFoam` — the validation layer enforces this but set it correctly anyway
