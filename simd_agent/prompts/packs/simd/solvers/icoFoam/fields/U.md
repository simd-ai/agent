# icoFoam — 0/U

**Dimensions**: `[0 1 -1 0 0 0 0]` (m/s)
**internalField**: `uniform (0 0 0)`

## BC types

| Patch role | BC type | Notes |
|---|---|---|
| inlet (fixed velocity) | `fixedValue` | `value uniform (<vx> <vy> <vz>)` |
| inlet (volumetric flow) | `flowRateInletVelocity` | `volumetricFlowRate` — see rules |
| inlet (mass flow rate) | `flowRateInletVelocity` | `massFlowRate` + `rhoInlet` — see rules |
| outlet | `zeroGradient` | |
| wall | `noSlip` | |
| symmetry | `symmetry` | |
| empty (2D) | `empty` | |

## flowRateInletVelocity — CRITICAL rules for incompressible solvers

### `rho` keyword — NEVER use a number; NEVER write `rho rho;` for icoFoam

The `rho` keyword expects a **word** (field name), never a scalar number.
- `rho 880;` → **FOAM FATAL IO ERROR**: "Wrong token type — expected word, found double 880"
- `rho rho;` → **FOAM error**: icoFoam has no `rho` field — lookup fails with no `rhoInlet` fallback
- `rho none;` → treats the flow rate as volumetric (m³/s) regardless of which keyword you used

### Option A — volumetric flow rate (PREFERRED — no density needed)

The CaseSpec pre-computes `volumetricFlowRate = massFlowRate / rho`. Use it directly:

```
inlet
{
    type                flowRateInletVelocity;
    volumetricFlowRate  constant <Q_m3_per_s>;    // pre-computed by CaseSpec
    value               uniform (0 0 0);           // required placeholder — NOT the velocity
}
```

### Option B — mass flow rate with constant density (valid per OF docs)

If the BC dict still has `massFlowRate`, use `rhoInlet` as the constant density fallback.
OpenFOAM will try to find the `rho` field, not find it (icoFoam has none), and fall back to `rhoInlet`:

```
inlet
{
    type            flowRateInletVelocity;
    massFlowRate    constant <mdot_kg_per_s>;
    rhoInlet        <density_kg_per_m3>;           // scalar fallback — no `rho` keyword!
    value           uniform (0 0 0);
}
```

**Rules:**
- `constant` qualifier is REQUIRED in OF 2406 for `Function1<scalar>` values.
- EXACTLY ONE of `volumetricFlowRate` or `massFlowRate` — never both.
- `value uniform (0 0 0)` is a required placeholder — NEVER put the flow rate value there.
- NEVER write `rho <word>;` for icoFoam — no density field exists.
