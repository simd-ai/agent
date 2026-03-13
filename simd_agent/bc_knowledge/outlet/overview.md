# Outlet conditions (MVP summary)

## Patch type
For standard outlets, the mesh boundary patch is a regular `patch`.

## MVP outlet choices
Support these first:
- pressure:
  - static pressure (`fixedValue`)
  - total pressure (`totalPressure`)
- temperature:
  - fixed temperature (`fixedValue`)
  - total temperature (`totalTemperature`)
- velocity:
  - outlet-compatible gradient behavior (`zeroGradient`)

## Important note
Simple gradient-only outflow is acceptable only when the flow is truly leaving the domain.
If reverse flow is possible, that choice becomes weak or unstable.

Because the MVP assumes no backflow:
- simple outlet behavior is acceptable
- but the planner should record `assumesNoBackflow = true` in assumptions

## Recommended outlet strategy for MVP
- If user gives static outlet pressure:
  - pressure is primary (`fixedValue`)
  - velocity uses `zeroGradient`
- If user gives total pressure:
  - use `totalPressure` BC
- If user gives total temperature:
  - use `totalTemperature` BC
- Turbulence fields at outlet:
  - use `zeroGradient` for `k`, `epsilon`, `omega`
  - use `calculated` for `nut`

## What to avoid
- Do not impose a fixed outlet velocity unless explicitly requested
- Do not silently downgrade total pressure to static pressure
- Do not silently downgrade total temperature to static temperature
