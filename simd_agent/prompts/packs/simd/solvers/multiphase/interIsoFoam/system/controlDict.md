# interIsoFoam — system/controlDict

**Time is physical** (seconds). Use adaptive timestep.

```
application     interIsoFoam;

startFrom       startTime;
startTime       0;
stopAt          endTime;

deltaT          <delta_t>;
endTime         <end_time>;

writeControl    adjustableRunTime;
writeInterval   <write_interval>;

runTimeModifiable true;

adjustTimeStep  yes;
maxCo           1;
maxAlphaCo      1;
maxDeltaT       <max_delta_t>;
```
