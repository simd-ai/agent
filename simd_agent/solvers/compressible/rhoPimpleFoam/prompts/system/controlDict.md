# rhoPimpleFoam — system/controlDict

**Time is physical** (seconds), NOT iteration count.

- `endTime` = `CaseSpec.end_time` (seconds)
- `deltaT` = `CaseSpec.delta_t` (seconds)
- `startFrom startTime; startTime 0;` — never `latestTime`

## deltaT — CRITICAL for cryogenic / low-velocity startup

**Never use the user-supplied `delta_t` directly for cryogenic cases without checking it against the flow velocity.**

With `adjustTimeStep yes` and `maxCo <max_co>`, OpenFOAM will halve deltaT every step if Co > maxCo — but it CANNOT rescue a case where the **first** time step produces Co >> 1 and k goes negative (turbulence blow-up; see fvSolution notes). The damage from the first step is irreversible: once k < 0 → nut < 0 → velocity diverges, subsequent halving is too late.

| Condition | Recommended initial deltaT |
|---|---|
| High-Re established flow, U > 1 m/s | `delta_t` from CaseSpec (typical 0.001–0.01 s), `maxCo` from CaseSpec |
| Low-velocity or cryogenic startup (U < 0.5 m/s) | `1e-5` s, `maxCo 0.2` |
| Cryogenic with T span > 100 K (LN2/LH2/LOX) | `1e-5` s, `maxCo 0.2` |

**Rule**: If `fluid.name` is a cryogenic fluid (LN2, liquid nitrogen, LH2, liquid hydrogen, LOX, liquid oxygen) OR the inlet velocity magnitude is < 0.5 m/s, set `deltaT 1e-5` regardless of what `delta_t` says. A small initial deltaT costs nothing (adjustTimeStep will ramp it up automatically) but prevents the turbulence blow-up that kills the run on step 1.

## Template

```
application     rhoPimpleFoam;

startFrom       startTime;
startTime       0;
stopAt          endTime;

// For cryogenic / low-velocity startup: use 1e-5, NOT the user-supplied delta_t.
// adjustTimeStep will ramp it up automatically once the flow is established.
deltaT          <delta_t>;   // replace with 1e-5 for cryogenic cases (see rules above)
endTime         <end_time>;

writeControl    adjustableRunTime;
writeInterval   <write_interval>;   // from CaseSpec (~100 snapshots)

runTimeModifiable true;

adjustTimeStep  yes;
maxCo           <max_co>;           // from CaseSpec (2.0 for PIMPLE; override to 0.2 for cryogenic startup)
maxDeltaT       <max_delta_t>;      // from CaseSpec (= writeInterval)
```
