# Outlet temperature from total temperature (`totalTemperature`)

## When to use
Use this when the user explicitly specifies:
- total temperature
- stagnation temperature
- wording like `T0`

Do not use this when the user simply gives a static fluid temperature.

## Purpose
This BC represents total temperature, not ordinary static temperature.

## UI fields to expose
- `totalTemperature` (stored as `T0`, in K)
- optional advanced:
  - `UField`
  - `phiField`
  - thermodynamic field names if needed by the solver family

## OpenFOAM mapping
Primary BC family: `type: totalTemperature`

Typical entries:
- `T0` (the total temperature value)
- `value`

## Planner rule
If the user says outlet total temperature:
- choose this BC family
- do not collapse it into a simple fixed static `T` boundary

## MVP note
If the current solver path does not yet need total temperature:
- keep support in retrieval and schema
- allow the planner to mark it as `supportedButOptional`

## What to avoid
- Do not use this for a simple user-specified static outlet temperature
- Do not confuse `T0` (total temperature) with fluid temperature at inlet
