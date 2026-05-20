cylindrical-cht (heated cylinder in a cavity)
=============================================

> **Note on the name** — the directory is `cylindrical-cht/` for
> historical reasons, but this is *not* a conjugate heat transfer
> case. It's natural convection around a heated cylinder, single
> region, no solid wall to conduct through. Consider renaming the
> directory to something like `heated-cylinder-cavity/` for
> clarity. See the discussion at the bottom of the walkthrough.

Steady-state natural convection of air inside a 2D square enclosure
with a hot horizontal cylinder in the centre. Classical buoyancy-
driven flow benchmark. No inlet, no outlet — closed cavity,
flow is purely buoyancy-induced.

  - solver:     buoyantBoussinesqSimpleFoam
  - turbulence: kOmegaSST (5% intensity, mixing length 0.07 m)
  - regime:     steady, incompressible Boussinesq, buoyant
  - geometry:   1 m × 1 m square outer enclosure;
                cylinder r = 0.1 m centred at (0.5, 0.5)
  - BCs:        cylinder T = 350 K (hot),
                outer walls T = 300 K (cold),
                front/back empty (2D),
                gravity = (0, -9.81, 0)
  - fluid:      air, β = 3.4e-3 1/K, ν = 1.5e-5 m²/s,
                Pr = 0.7, T_ref = 300 K

See `Documentation/examples/cylindrical-cht.md` for the walkthrough.


reproduce with OpenFOAM directly
--------------------------------

    cd case && buoyantBoussinesqSimpleFoam


reproduce via the agent
-----------------------

    simd run examples/cylindrical-cht/prompt.txt \
             examples/cylindrical-cht/mesh/cylindrical-cht.msh

Or upload the mesh + paste the prompt in the frontend at
http://localhost:3000 — same backend.
