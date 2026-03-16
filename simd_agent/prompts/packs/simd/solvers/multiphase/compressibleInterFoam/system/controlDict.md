# compressibleInterFoam — system/controlDict

## Timestep control — CRITICAL

- `maxCo 0.5` — NOT 1.0. At Co=1, compressibleInterFoam with icoPolynomial liquid (zero acoustic compressibility) can have Co spike to 18+ before deltaT reacts, and once Co > 10 the pressure-velocity coupling diverges unrecoverably within 3 timesteps.
- `maxAlphaCo 0.5` — keeps the alpha sub-cycling stable.
- `adjustTimeStep yes` — always use adaptive timestep for transient VOF.

```
application     compressibleInterFoam;

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
