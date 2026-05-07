# rhoPimpleFoam — 0/U

**Dimensions**: `[0 1 -1 0 0 0 0]` (m/s)
**internalField**: `uniform (0 0 0)`

## BC types by patch role

| Patch | BC type |
|---|---|
| inlet (fixedVelocity) | `fixedValue` + `value uniform (<vx> <vy> <vz>)` |
| inlet (massFlowRate) | `flowRateInletVelocity` — see critical rules below |
| inlet (volumetricFlow) | `flowRateInletVelocity` with `volumetricFlowRate` |
| outlet | `zeroGradient` |
| wall | `noSlip` |
| symmetry | `symmetry` |
| empty (2D) | `empty` |

## flowRateInletVelocity — CRITICAL rules (OF 2406)

### `rho` keyword expects a WORD (field name), NEVER a number

- `rho 880;` → **FOAM FATAL IO ERROR**: "Wrong token type — expected word, found double 880"
- `rho rho;` → looks up density field named "rho" (valid for rhoPimpleFoam — field exists)
- `rhoInlet <scalar>` → constant density used as startup fallback when `rho` field is not yet available

### Mass flow rate (most common for compressible)

```
inlet
{
    type            flowRateInletVelocity;
    massFlowRate    constant <kg_per_s>;
    rhoInlet        <density_kg_per_m3>;    // startup fallback before rho field is initialised
    value           uniform (0 0 0);        // required placeholder — NOT the velocity
}
```

### Volumetric flow rate

```
inlet
{
    type                flowRateInletVelocity;
    volumetricFlowRate  constant <m3_per_s>;
    value               uniform (0 0 0);
}
```

- `constant` qualifier REQUIRED in OF 2406 for `Function1<scalar>` values.
- EXACTLY ONE of `massFlowRate` or `volumetricFlowRate` — never both.
- Do NOT write `rho rho;` alongside `rhoInlet` — redundant and confusing.
- `massFlowRate 0` causes immediate divergence (SIGFPE).
- `value uniform (0 0 0)` is a required placeholder — NEVER the flow rate value.
