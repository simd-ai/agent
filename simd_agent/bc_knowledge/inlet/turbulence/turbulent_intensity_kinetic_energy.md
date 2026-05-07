# Inlet turbulent kinetic energy from intensity (`turbulentIntensityKineticEnergyInlet`)

## When to use
Use this when:
- the turbulence model needs `k`
- the user provides turbulence intensity
- the user says things like:
  - "5% turbulence intensity"
  - "low turbulence intensity"
  - "inlet intensity"

## Purpose
This BC provides the inlet value of `k` from:
- user turbulence intensity
- inlet velocity magnitude

So the user does not need to provide `k` directly.

## UI fields to expose
- `intensity` (fraction, e.g. 0.05 for 5%)
- optional:
  - `velocitySource`
    - `fromBoundaryU`
    - `fromEstimatedMeanVelocity`

## OpenFOAM mapping
Primary BC family: `type: turbulentIntensityKineticEnergyInlet`

Required entries:
- `intensity`
- `value`

Optional entries:
- `U`
- `phi`

## Derivation behavior
The BC derives `k` from the inlet velocity magnitude and specified turbulence intensity:
```
k = 1.5 * (U * I)²
```

In the UI:
- show `intensity` as the user-owned parameter
- show `estimatedK` as a derived field if helpful

## Important rule for mass-flow inlets
If the inlet velocity comes from a flow-rate BC:
- do not force the user to enter a fixed velocity just to get `k`
- derive the velocity magnitude from:
  - the active inlet velocity field
  - or a UI-only estimated mean velocity
- make it explicit that `k` depends on the chosen inlet flow state

## Reverse-flow note
This BC behaves like an inlet/outlet style condition and uses gradient behavior for reverse flow.
That is acceptable for the MVP.

## Example agent output intent
- `selectedBcFamily = turbulentIntensityKineticEnergyInlet`
- `uiPrimaryInput = intensity`
- `derived.estimatedK.uiOnly = true`
