# buoyantBoussinesqPimpleFoam — global rules

Transient, incompressible, single-phase, buoyancy-driven solver.

## Identity

- **Algorithm:** PIMPLE (transient outer loop).
- **Pressure field:** `p_rgh` (modified pressure = p − ρgh).
- **Energy variable:** `T` (Kelvin) — transport equation.
- **Density:** constant `ρ₀`; buoyancy via `−ρ₀·β·(T − T_ref)·g`.
- **Required files:** `constant/g`, `constant/transportProperties`
  (deterministic), `0/U`, `0/p_rgh`, `0/T`, plus turbulence fields.

## PIMPLE final-iteration coverage

Every solved field gets a `Final` variant automatically — `pFinal`,
`UFinal`, `TFinal`, `kFinal`, `epsilonFinal` / `omegaFinal`.  Don't
emit them manually.

## Do NOT generate

- `constant/thermophysicalProperties` — Boussinesq uses
  transportProperties.
- `0/h` or `0/e` — energy field is `T`.
- `system/fvSchemes`, `system/fvSolution`, `constant/turbulenceProperties`,
  `0/nut`, `0/alphat` — all deterministic.

## Reference tutorial

`OpenFOAM-4.x/tutorials/heatTransfer/buoyantBoussinesqPimpleFoam/hotRoom`.
