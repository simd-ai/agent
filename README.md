simd-agent
==========

AI-native CFD simulation agent built on OpenFOAM v2406.

<img src="Documentation/images/header-strip.png" width="900" alt="four example simulations: u-shape pipe, z-bend, inner-outer pipe, cylindrical CHT">


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

    git clone https://github.com/simd-ai/simd-agent
    cd simd-agent
    ./install.sh

`install.sh` generates `.env`, prompts for your Gemini key (or accepts
`--vertex` / `--ollama`), pulls the published images from GHCR, and
brings up Postgres + OpenFOAM + the agent + the frontend. When it
finishes, open http://localhost:3000.

If you'd rather do it by hand:

    cp .env.example .env       # edit GEMINI_API_KEY, save
    docker compose -f docker/docker-compose.yml up -d

Bare-metal installation (Python venv, system OpenFOAM, external
Postgres) is in Documentation/installation.md.


CLI
---

A ``simd`` command-line client ships in this repo. Install it with
``pip install -e .`` and run:

    simd init                                # interactive setup
    simd run examples/u-shape-pipe/prompt.txt \
             examples/u-shape-pipe/mesh/u-shape-pipe.msh

``simd init`` asks where each component should run (bundled docker,
bare-metal local, or remote) and writes the config; ``simd run``
auto-starts the backend when needed, then walks you through mesh
upload, precheck, interactive patch review, and the five-stage
progress display.  No login, no account, no tracking.  Same backend
as the frontend.  Full reference: Documentation/cli.md.


how it works
------------

A FastAPI service orchestrates per-file OpenFOAM codegen with an LLM,
validates the output with deterministic plugin-side rules, ships the
case to a sim-server running OpenFOAM, and streams residuals and
post-processed VTK back through a WebSocket. When the solver fails,
the agent diagnoses the error with a smaller LLM call and retries
with focused fixes — up to seven attempts by default. This is the
self-healing loop.

See Documentation/architecture.md for the full design,
Documentation/self-healing.md for a walkthrough of one real failure.


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

See CONTRIBUTING.md. New solver plugins drop into
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
