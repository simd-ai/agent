# Common boundary pairing rules (MVP)

Use this file as a cross-check after selecting individual patch BCs.

## Goal

Avoid invalid or weak combinations between:
- velocity (`U`)
- pressure (`p` / `p_rgh`)
- temperature (`T`)
- turbulence fields (`k`, `epsilon`, `omega`, `nut`)

## Core MVP rules

### Inlet
- If the user specifies **mass flow rate** or **volumetric flow rate**:
  - Prefer a **flow-rate-based velocity BC** (`flowRateInletVelocity`).
  - Do **not** convert the flow rate into a fixed velocity as the primary BC.
  - A derived display value such as estimated average velocity may be computed for UI display only.

- If the user specifies **static pressure** at inlet:
  - Use a pressure BC for pressure.
  - Use a compatible inlet velocity BC that responds to pressure only if the user really asked for pressure-driven inlet behavior.

- If the user specifies **temperature** directly:
  - Use a direct temperature BC (`fixedValue` style) unless total temperature is explicitly requested.

### Outlet
- If the user specifies **static outlet pressure**:
  - Prefer a direct pressure condition (`fixedValue`).
  - Use outlet-compatible velocity behavior (typically `zeroGradient`), not a fixed outlet velocity unless explicitly requested.

- If the user specifies **total pressure**:
  - Use `totalPressure` on the pressure field.

- If the user specifies **total temperature**:
  - Use `totalTemperature` on the temperature field.

- Plain outflow using `zeroGradient` is acceptable only when the flow is truly leaving the domain and reverse flow is not expected.

### Wall
- If the user says **no slip**:
  - Use `noSlip` for velocity.
  - Do not let the model choose slip or partial slip.

- For thermal wall behavior:
  - Only impose wall temperature if the user explicitly gives it.
  - Otherwise choose the wall thermal strategy based on solver setup and user requirement (adiabatic, fixed temperature, heat flux, etc.).

## Output policy

Every patch agent must return:
- the selected BC family
- the reason for selecting it
- whether the value is:
  - directly user-specified
  - derived for UI only
  - derived and required by the OpenFOAM entry

If a value is derived only for display, it must not replace the original physical constraint.
