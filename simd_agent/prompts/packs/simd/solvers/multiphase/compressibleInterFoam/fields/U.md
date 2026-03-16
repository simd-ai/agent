# compressibleInterFoam — 0/U

**Dimensions**: `[0 1 -1 0 0 0 0]`

## Initialisation
For filling or startup transients:
- `internalField uniform (0 0 0);`

## Inlet
If the user specifies a mass flow rate:
- use `flowRateInletVelocity`
- include:
  - `massFlowRate <kg/s>;`
  - `rho rho;`
- include `rhoInlet <value>;` when a representative inlet density is available
- include:
  - `value uniform (0 0 0);`

If the user specifies a volumetric flow rate:
- use `flowRateInletVelocity`
- include:
  - `volumetricFlowRate <m3/s>;`
  - `value uniform (0 0 0);`
- do NOT include `rho` or `rhoInlet` for volumetric flow rate

If the user specifies a fixed velocity:
- use `fixedValue`
- `value uniform (<vx> <vy> <vz>);`

## Outlet
- `type zeroGradient;`

## Wall
- `type noSlip;`

## frontAndBack
- `type empty;`

## Constraints
- EXACTLY ONE of `massFlowRate` or `volumetricFlowRate` MUST be present when using `flowRateInletVelocity`.
- `value uniform (0 0 0)` is a placeholder for patch initialisation — NEVER put the flow rate value here.
- Do NOT include `rho`/`rhoInlet` for volumetric flow inlets.
