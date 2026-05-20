u-shape-pipe (inverted-U duct)
==============================

Compressible flow of hot air through an inverted-U duct with a
cooler secondary stream merging at a side inlet. Heated walls,
two inlets, one outlet.

  - solver:     rhoSimpleFoam
  - turbulence: kOmegaSST
  - regime:     steady, compressible, energy on
  - BCs:        main inlet 0.012 kg/s at 500 K,
                side inlet 0.001 kg/s at 280 K,
                walls at 600 K,
                outlet at 101325 Pa

The prompt explicitly names `rhosimplefoam` — a useful pattern when
you want to pin the solver instead of letting the LLM choose. The
agent honors the explicit pick and skips the solver-selection LLM
call.

See `Documentation/examples/u-shape-pipe.md` for the walkthrough.


reproduce with OpenFOAM directly
--------------------------------

    cd case && rhoSimpleFoam


reproduce via the agent
-----------------------

    # post the prompt + mesh to /api/precheck, then open
    # /ws/run with op=CFD_CODEGEN_RUN.  See Documentation/api/.
    # The bundled frontend at http://localhost:3000 wires this up.

Or upload the mesh + paste the prompt in the frontend at
http://localhost:3000 — same backend.
