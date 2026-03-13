# Inlet conditions (MVP summary)

## Patch type
For standard inlets, the mesh boundary patch is a regular `patch`.

## MVP inlet choices
Support these first:
- velocity / flow:
  - fixed velocity (`fixedValue`)
  - mass-flow-driven velocity (`flowRateInletVelocity` with `massFlowRate`)
  - volumetric-flow-driven velocity (`flowRateInletVelocity` with `volumetricFlowRate`)
- pressure:
  - inlet-compatible gradient behavior (`zeroGradient`) — default when flow/velocity is primary
  - total pressure (`totalPressure`) only if explicitly requested for stagnation conditions
- temperature:
  - fixed temperature (`fixedValue`)
  - total temperature (`totalTemperature`) only if explicitly requested
- turbulence:
  - `k` from intensity (`turbulentIntensityKineticEnergyInlet`) — preferred
  - `omega` via `fixedValue` computed from k and L
  - `epsilon` via `fixedValue` computed from k and L
  - `nut` via `calculated`

## Important note
At an inlet, the primary driving condition should usually be flow/velocity, not static pressure,
when the user specifies:
- mass flow rate
- volumetric flow rate
- inlet velocity

Do not impose both a fixed inlet flow and a fixed static inlet pressure unless the user explicitly
requests that combination and the solver/physics supports it.

## Recommended inlet strategy for MVP

- If user gives **mass flow rate**:
  - velocity is primary: `flowRateInletVelocity` with `massFlowRate`
  - pressure: `zeroGradient`

- If user gives **volumetric flow rate**:
  - velocity is primary: `flowRateInletVelocity` with `volumetricFlowRate`
  - pressure: `zeroGradient`

- If user gives **inlet velocity** (m/s or vector):
  - velocity is primary: `fixedValue` with the specified vector
  - pressure: `zeroGradient`

- If user gives **static inlet temperature**:
  - temperature: `fixedValue` with temperature in K

- If user gives **total inlet temperature**:
  - temperature: `totalTemperature` with `T0` in K
  - do NOT downgrade to static `fixedValue`

- If user gives **total inlet pressure** (stagnation pressure):
  - pressure: `totalPressure` with `p0`
  - this is unusual at inlets — only use when explicitly stated

- **Turbulence fields at inlet** (when turbulence model is active):
  - `k`: prefer `turbulentIntensityKineticEnergyInlet` if intensity is known
  - `k`: use `fixedValue` if the user gives k directly
  - `omega`: use `fixedValue` with computed value from $\omega = \sqrt{k} / (C_\mu^{0.25} L)$
  - `epsilon`: use `fixedValue` with computed value from $\varepsilon = C_\mu^{0.75} k^{1.5} / L$
  - `nut`: use `calculated` with `value uniform 0`

## What to avoid
- Do not impose a fixed static pressure at inlet when mass flow rate or velocity is already
  the primary driving condition, unless explicitly requested.
- Do not silently convert a mass-flow inlet into a pressure inlet.
- Do not silently downgrade total temperature to static temperature.
- Do not assign a user-provided pipe operating pressure to the inlet by default if the case
  is a through-flow with a standard outlet — prefer outlet pressure unless clearly stated.
- Do not force the user to enter a fixed velocity just to derive turbulence values; use
  `turbulentIntensityKineticEnergyInlet` which derives `k` from the active inlet velocity.
