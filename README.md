<div align="center">

<img src="Documentation/images/simd-logo.png" width="120" alt="simd logo">

# simd-agent

AI-native CFD simulation agent.

[![License](https://img.shields.io/github/license/simd-ai/agent)](LICENSE)
[![Language](https://img.shields.io/github/languages/top/simd-ai/agent)](https://github.com/simd-ai/agent)
[![Commit activity](https://img.shields.io/github/commit-activity/m/simd-ai/agent)](https://github.com/simd-ai/agent/commits/main)

</div>


what it does
------------

Takes a natural-language prompt and a mesh, picks the right OpenFOAM
solver, generates a complete case (boundary conditions, fvSchemes,
fvSolution, thermophysical properties, …), submits it to a sim-server
running OpenFOAM, streams residuals and 3D fields back, and self-heals
on solver failures.

Supports:

  - incompressible and compressible flow
    (simpleFoam, pimpleFoam, rhoSimpleFoam, rhoPimpleFoam,
    buoyantSimpleFoam, buoyantPimpleFoam, buoyantBoussinesqSimpleFoam,
    buoyantBoussinesqPimpleFoam, icoFoam)
  - laminar and turbulent regimes
    (k-epsilon, k-omega SST, k-omega, Spalart-Allmaras, laminar)
  - conjugate heat transfer between solids and fluids
    (chtMultiRegionSimpleFoam, chtMultiRegionFoam)
  - multiphase flows — experimental
    (compressibleInterFoam, compressibleInterIsoFoam,
    compressibleMultiphaseInterFoam, interFoam, interIsoFoam)
  - LLM providers: Google Gemini, Vertex AI, Ollama (local)
  - object storage: local filesystem or Google Cloud Storage
  - authentication: Neon Auth, or none (open mode)

Coming: digital-twin streaming for live experiments.


what you need
-------------

  - Docker + Docker Compose
  - One LLM credential: a Gemini API key, OR a Vertex AI service-account
    JSON, OR a local Ollama install

That's it. The compose stack ships OpenFOAM v2406, Postgres, the
agent, and the frontend.


quick start
-----------

    git clone https://github.com/simd-ai/agent
    cd agent
    ./install.sh

``install.sh`` is an interactive wizard.  Pick **Docker mode** to
run everything in containers (postgres + agent come up via
``docker compose``), or **bare-metal mode** to use a Python venv
on this machine.  Either way it asks for your LLM key, the
simulation runner URL, and where to store results — then writes
``.env``.  In Docker mode the stack starts automatically; in
bare-metal mode the wizard prints the ``uvicorn`` command to run.

Once the agent is up at ``http://localhost:8000``, drive it through
the WebSocket / HTTP API (see ``Documentation/api/``) or run the
frontend at ``http://localhost:3000``.


how it works
------------

A FastAPI service orchestrates per-file OpenFOAM codegen with an LLM,
validates the output with deterministic plugin-side rules, ships the
case to a sim-server running OpenFOAM, and streams residuals and
post-processed VTK back through a WebSocket. When the solver fails,
the agent diagnoses the error with a smaller LLM call and retries
with focused fixes — up to seven attempts by default. This is the
self-healing loop.

See Documentation/architecture for the full design,
Documentation/self-healing for a walkthrough of one real failure.


examples
--------

Four end-to-end cases ship under `examples/`. Each carries its mesh,
its prompt, and the generated OpenFOAM case files — so you can run
the simulation directly with OpenFOAM, or watch the agent regenerate
it from the prompt.

    examples/u-shape-pipe/        compressible inverted-U duct,
                                  rhoSimpleFoam + kOmegaSST
    examples/z-bend/              transient turbulent water pipe,
                                  pimpleFoam + kOmegaSST
    examples/inner-outer-pipe/    2D LN2/water counter-flow
                                  regasifier, chtMultiRegionSimpleFoam
    examples/cylindrical-cht/     natural convection around a heated
                                  cylinder, buoyantBoussinesqSimpleFoam

Walk-throughs and screenshots in Documentation/examples/.


documentation
-------------

See Documentation/ for installation, deployment, the WebSocket
protocol, the solver plugin contract, and the LLM provider plugin
contract.


contributing
------------

See CONTRIBUTING. New solver plugins drop into
`simd_agent/solvers/<name>/` and are auto-discovered; new LLM
providers drop into `simd_agent/llm/<name>/`. No registry edits
needed.


license
-------

Apache 2.0 — see LICENSE.


acknowledgements
----------------

OpenFOAM® is a registered trade mark of OpenCFD Ltd. This project is
not approved or endorsed by OpenCFD or the OpenFOAM Foundation.
