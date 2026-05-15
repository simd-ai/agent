# buoyantBoussinesqSimpleFoam — global rules

Steady-state, incompressible, single-phase, buoyancy-driven solver.

## Identity

- **Algorithm:** SIMPLE (no time loop; iterate to convergence).
- **Pressure field:** `p_rgh` (modified pressure = p − ρgh).  Both `0/p_rgh`
  (solved) and `0/p` (synthesised from p_rgh) are present.
- **Energy variable:** `T` (Kelvin) — solved directly via a transport
  equation `∂T/∂t + ∇·(φT) = ∇·(α_eff ∇T)`.
- **Density:** constant `ρ₀`; the Boussinesq approximation enters only
  via the buoyancy source `−ρ₀·β·(T − T_ref)·g` in the momentum equation.
- **Required files:** `constant/g`, `constant/transportProperties`
  (rendered deterministically), `0/U`, `0/p_rgh`, `0/T`, plus the turbulence
  fields for the chosen model.

## Do NOT generate

- `constant/thermophysicalProperties` — Boussinesq uses
  transportProperties (Newtonian, ν, β, T_ref, Pr, Prt).
- `0/h` or `0/e` — the energy field is `T`, not enthalpy or internal energy.
- `system/fvSchemes`, `system/fvSolution`, `constant/turbulenceProperties`,
  `0/nut`, `0/alphat` — all deterministically rendered by the validator.

## Reference tutorial

`OpenFOAM-4.x/tutorials/heatTransfer/buoyantBoussinesqSimpleFoam/hotRoom`.
