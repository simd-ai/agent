# buoyantPimpleFoam — system/controlDict

```
application     buoyantPimpleFoam;

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         <end_time_seconds>;  // physical seconds
deltaT          <delta_t_seconds>;   // physical time step (e.g. 0.01 or 0.1)
writeControl    adjustableRunTime;
writeInterval   1;                   // write every N seconds of physical time
adjustTimeStep  yes;
maxCo           0.5;                 // Courant number limit — 0.5 for buoyant cases
maxDeltaT       0.1;                 // upper bound on automatic time-step growth
purgeWrite      0;
writeFormat     ascii;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
```

## Notes

- `endTime` and `deltaT` are in PHYSICAL SECONDS (not iteration counter).
- `adjustTimeStep yes; maxCo 0.5` recommended for stability.
  At Co=1.0 the limiter reacts too late to prevent divergence cascade in buoyant flows.
- For fire/smoke simulations, initial `deltaT` should be small (0.001–0.01 s).
- `application` MUST be `buoyantPimpleFoam` exactly.
