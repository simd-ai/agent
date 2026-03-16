# compressibleMultiphaseInterFoam — system/controlDict

## Timestep control — CRITICAL

- `maxCo 0.5` — NOT 1.0. Same reason as compressibleInterFoam: icoPolynomial liquid has zero acoustic compressibility; Co spikes are fatal.
- `maxAlphaCo 0.5`.

```
application     compressibleMultiphaseInterFoam;

startFrom       startTime;
startTime       0;
stopAt          endTime;

deltaT          <delta_t>;
endTime         <end_time>;

writeControl    adjustableRunTime;
writeInterval   <write_interval>;

runTimeModifiable true;

adjustTimeStep  yes;
maxCo           0.5;
maxAlphaCo      0.5;
maxDeltaT       <max_delta_t>;
```
