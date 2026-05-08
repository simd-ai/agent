# buoyantSimpleFoam — system/controlDict

```
application     buoyantSimpleFoam;

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         <max_iterations>;   // integer — iteration counter for steady solver
deltaT          1;                  // steady: 1 (not physical time)
writeControl    timeStep;
writeInterval   <writeInterval>;    // endTime / 20 → ~20 snapshots
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
```

- `endTime` MUST be an INTEGER (e.g. 500 or 1000) — not seconds.
- `deltaT 1` is the iteration counter step for steady SIMPLE.
- `application` MUST be `buoyantSimpleFoam` exactly.
