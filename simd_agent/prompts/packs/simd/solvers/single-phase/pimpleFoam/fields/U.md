# pimpleFoam — 0/U

**Dimensions**: `[0 1 -1 0 0 0 0]` (m/s)
**internalField**: `uniform (0 0 0)`

## Typical BC types by patch role

| Patch role | BC type | Notes |
|---|---|---|
| inlet (fixed velocity) | `fixedValue` | `value uniform (<Ux> <Uy> <Uz>)` |
| inlet (volumetric flow) | `flowRateInletVelocity` | `volumetricFlowRate` keyword only — see rules |
| outlet | `zeroGradient` | or `inletOutlet` to prevent backflow |
| wall | `noSlip` | |
| symmetry | `symmetry` | |
| empty (2D) | `empty` | |

Use `inletOutlet` at outlets when backflow is expected:
```
outlet
{
    type            inletOutlet;
    inletValue      uniform (0 0 0);
    value           $internalField;
}
```

## flowRateInletVelocity — CRITICAL rules for incompressible solvers

### `rho` keyword — NEVER use a number; NEVER write `rho rho;` for pimpleFoam

- `rho 880;` → **FOAM FATAL IO ERROR**: "Wrong token type — expected word, found double 880"
- `rho rho;` → **FOAM error**: pimpleFoam has no `rho` field — lookup fails with no `rhoInlet` fallback

### Option A — volumetric flow rate (PREFERRED)

The CaseSpec pre-computes `volumetricFlowRate = massFlowRate / rho`:

```
inlet
{
    type                flowRateInletVelocity;
    volumetricFlowRate  constant <Q_m3_per_s>;    // = massFlowRate / rho (pre-computed)
    value               uniform (0 0 0);           // required placeholder — NOT the velocity
}
```

### Option B — mass flow rate with constant density (also valid per OF 2306 docs)

```
inlet
{
    type            flowRateInletVelocity;
    massFlowRate    constant <mdot_kg_per_s>;
    rhoInlet        <density_kg_per_m3>;           // scalar fallback — no `rho` keyword!
    value           uniform (0 0 0);
}
```

- `constant` qualifier REQUIRED in OF 2406 for `Function1<scalar>` values.
- EXACTLY ONE of `volumetricFlowRate` or `massFlowRate` — never both.
- NEVER write `rho <word>;` — pimpleFoam has no density field.
- `value uniform (0 0 0)` is a required placeholder — NEVER the flow rate value.
