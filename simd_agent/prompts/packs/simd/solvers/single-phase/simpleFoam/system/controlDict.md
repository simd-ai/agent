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

- `endTime` MUST be an integer (no decimal point)
- `deltaT 1` always for steady-state
