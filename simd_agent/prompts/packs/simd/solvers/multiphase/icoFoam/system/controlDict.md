# icoFoam — system/controlDict

**Time is physical** (seconds) — not iteration count.

```
application     icoFoam;

startFrom       startTime;
startTime       0;
stopAt          endTime;

deltaT          <delta_t>;
endTime         <end_time>;

writeControl    timeStep;
writeInterval   <write_interval>;

runTimeModifiable true;

adjustTimeStep  yes;
maxCo           0.9;
```
