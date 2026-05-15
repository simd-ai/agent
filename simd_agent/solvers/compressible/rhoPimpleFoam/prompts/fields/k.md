# rhoPimpleFoam — 0/k

**Dimensions**: `[0 2 -2 0 0 0 0]` (m²/s²)

## Value — MUST be computed from actual flow, NOT a hard-coded default

**NEVER use k = 0.1 as a default for low-speed flows.** This is ~500× too large for typical
cryogenic pipe flow (U ≈ 0.2 m/s) and will destabilize the pressure solve immediately.

Use `CaseSpec.turbulence_initial_values.k` when pre-computed and available.

Otherwise compute from first principles:

1. Estimate inlet velocity: `U = m_dot / (rho × A_inlet)` where `A_inlet = π × (D/2)²`
2. Apply: `k = 1.5 × (I × U)²` with `I = 0.05` (5% turbulence intensity, typical internal pipe flow)

**Example — LN2 pipe flow:**
- m_dot = 0.089 kg/s, rho ≈ 808 kg/m³, D = 0.025 m
- U = 0.089 / (808 × π × 0.0125²) ≈ 0.22 m/s
- k = 1.5 × (0.05 × 0.22)² ≈ 1.8e-4

For low-speed cryogenic pipe flow (U < 0.5 m/s), k is typically in the range **1e-5 to 1e-3**.
Any generated value of 0.01 or above should be treated as a red flag.

## BC types

| Patch | BC type |
|---|---|
| inlet | `fixedValue` + `value uniform <k_value>` |
| outlet | `zeroGradient` |
| wall | `kqRWallFunction` + `value uniform <k_value>` |
| symmetry | `symmetry` |
| empty (2D) | `empty` |
