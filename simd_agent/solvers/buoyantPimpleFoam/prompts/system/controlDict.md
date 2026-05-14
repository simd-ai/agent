# buoyantPimpleFoam — system/controlDict

```
application     buoyantPimpleFoam;

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         <end_time>;          // physical seconds, from CaseSpec
deltaT          <delta_t>;           // physical time step, from CaseSpec
writeControl    adjustableRunTime;
writeInterval   <write_interval>;    // from CaseSpec (~100 snapshots)
adjustTimeStep  yes;
maxCo           <max_co>;            // from CaseSpec (2.0 for PIMPLE)
maxDeltaT       <max_delta_t>;       // from CaseSpec (= writeInterval)
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
- `maxCo` from CaseSpec (2.0 for PIMPLE). For buoyant cases with strong natural convection, consider reducing to 1.0 if residuals oscillate.
- For fire/smoke simulations, initial `deltaT` should be small (0.001–0.01 s).
- `application` MUST be `buoyantPimpleFoam` exactly.
- `writeInterval` from CaseSpec gives ~100 snapshots for the full endTime.
- `maxDeltaT` from CaseSpec (= writeInterval) prevents deltaT from jumping past a file write time.
