# rhoSimpleFoam — 0/U

**Dimensions**: `[0 1 -1 0 0 0 0]` (m/s)
**internalField**: `uniform (0 0 0)` — safe zero initialization

## Typical BC types by patch role

| Patch role | BC type | Notes |
|---|---|---|
| inlet (velocity specified) | `fixedValue` | `value uniform (<Ux> <Uy> <Uz>)` |
| inlet (mass flow rate) | `flowRateInletVelocity` | See mandatory structure below |
| inlet (volumetric flow rate) | `flowRateInletVelocity` | See mandatory structure below |
| outlet | `inletOutlet` | `inletValue uniform (0 0 0); value $internalField;` |
| wall | `noSlip` | No `value` entry needed |
| symmetry | `symmetry` | |
| empty (2D front/back) | `empty` | |

## flowRateInletVelocity — MANDATORY structure

### Option A — mass flow rate (kg/s)
```
<patchName>
{
    type            flowRateInletVelocity;
    massFlowRate    <value>;          // MANDATORY — actual kg/s, NEVER 0 unless user asked
    rho             rho;              // ONLY if provided in BC table
    rhoInlet        <density>;        // [kg/m³] ONLY if provided in BC table
    value           uniform (0 0 0);  // placeholder — NOT the flow rate
}
```

### Option B — volumetric flow rate (m³/s)
```
<patchName>
{
    type                flowRateInletVelocity;
    volumetricFlowRate  <value>;      // MANDATORY — actual m³/s, NEVER 0
    value               uniform (0 0 0);
}
```

## CRITICAL RULES for flowRateInletVelocity

1. EXACTLY ONE of `massFlowRate` or `volumetricFlowRate` MUST be present.
   OpenFOAM fatal: `"Please supply either volumetricFlowRate or massFlowRate"`.
2. The flow rate value MUST be the actual user-specified value from the BC table — NEVER 0.
   `massFlowRate 0` causes SIGFPE divergence immediately.
3. `value uniform (0 0 0)` is a patch-initialisation placeholder — NEVER put the flow rate here.
4. Include `rho` and `rhoInlet` ONLY if they appear in the BC table. Do NOT invent them.
5. For volumetric flow: NEVER include `rho` or `rhoInlet`.
6. `extrapolateProfile yes` ONLY if explicitly in the BC table — omit by default.

## internalField

Use `uniform (0 0 0)` — a safe zero guess. NEVER put the mass flow rate in internalField.
