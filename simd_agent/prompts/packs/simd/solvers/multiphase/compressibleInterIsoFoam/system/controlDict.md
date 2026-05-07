# compressibleInterIsoFoam — system/controlDict

## Timestep control — CRITICAL

- `maxCo 0.5` — NOT 1.0. At Co=1, compressibleInterIsoFoam with icoPolynomial liquid can have Co spike to 18+ before deltaT reacts, causing unrecoverable pressure-velocity divergence.
- `maxAlphaCo 0.5` — keeps the alpha isoAdvection step stable.

```
application     compressibleInterIsoFoam;

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
