simd CLI
========

Command-line interface to the agent. Same backend as the frontend —
just a different UI on top of the existing REST + WebSocket surface.

Use it for: scripting, CI, headless servers, quick local runs without
the frontend, and benchmarking the agent in isolation.


install
-------

The CLI installs as part of the package:

    pip install -e .

After install, a ``simd`` executable lands on your ``$PATH``.

Verify:

    simd --version
    simd 0.1.0


setup: simd init
----------------

    simd init

Interactive wizard.  Asks four questions:

  1. **agent backend**  — local-docker (recommended), local-bare-metal,
     or remote.
  2. **simulation runner backend** — same three options; bare-metal
     warns that OpenFOAM v2406 must be installed locally.
  3. **LLM provider** — gemini (asks for the API key), vertex (asks
     for the SA JSON path), or ollama (local).
  4. **storage** — local filesystem (default) or GCS.

Writes ``~/.simd/.env`` (carries the agent's runtime env vars) and
``~/.config/simd/config.toml`` (CLI lookup state).

You can also skip ``simd init`` and just run ``simd run …`` — the
wizard fires automatically the first time it sees no config.


no login, no account, no tracking
---------------------------------

There's no ``simd login``.  The CLI talks to whatever local agent you
have running — there's no remote service to authenticate against.
On the first ``simd run`` the CLI creates the local Postgres records
it needs (the agent's data model attaches simulations to a "user"
record for relational integrity), caches the resulting ids in
``~/.config/simd/config.toml``, and reuses them on every subsequent
call.  No data leaves your machine, no telemetry, no account.

Configuration is via env vars or flags:

  - ``--agent URL``  (or ``$SIMD_AGENT``)
  - ``$SIMD_PROJECT`` to pin a specific simulation_id


lifecycle: simd up / down / status
-----------------------------------

When the backend is ``local-docker``, the CLI manages it for you:

    simd up        # docker compose up -d (the bundled stack)
    simd down      # docker compose down
    simd status    # what's running, where, how to reach it

``simd run`` checks ``/health`` before each invocation and auto-starts
the stack if it's down — so for everyday use you can skip ``simd up``
entirely.

``simd up`` / ``simd down`` refuse in local-bare-metal or remote modes
(the CLI doesn't own those processes).


commands
--------

  ``simd init``                 interactive setup wizard
  ``simd up``                   start the bundled docker stack
  ``simd down``                 stop the bundled docker stack
  ``simd status``               show backend / runner health + config
  ``simd run PROMPT MESH``      end-to-end interactive run
  ``simd watch RUN_ID``         re-attach to an in-progress run
  ``simd ls``                   list recent runs in the active simulation
  ``simd stop RUN_ID``          gracefully stop a running simulation


simd run
--------

The main one. Upload mesh, run precheck, review patches, kick off
codegen + sim, stream the result.

    simd run examples/u-shape-pipe/prompt.txt \
             examples/u-shape-pipe/mesh/u-shape-pipe.msh

Five stages, surfaced one heading at a time:

    Stage 1/5 — uploading mesh
    Stage 2/5 — running precheck
    Stage 3/5 — generating case files
    Stage 4/5 — running on simulation server
    Stage 5/5 — results

Flags:

  ``-y, --yes``         skip interactive patch review, accept defaults
  ``--max-retries N``   self-healing retry budget (default 7)
  ``--solver NAME``     force a solver, skip the LLM selector
  ``--detach``          return immediately, don't stream
  ``--no-run``          stop after precheck + patch review

Output mode (apply to any subcommand):

  ``--quiet``           only errors and the final result
  default               5-stage summary
  ``--verbose``         every ``AgentEvent`` as it streams
  ``--json``            NDJSON, one event per line (pipe-friendly)


simd watch
----------

Re-attach to a run started earlier (by you or by the frontend). Same
display as ``simd run``'s stages 3–5, no upload/precheck step.

    simd watch e63a4a69-6b47-402e-bf1e-01f42b6d7cb7

The CLI calls ``/api/runs/{id}/status`` to get the last-seen sequence
number, then opens ``/ws/watch/{id}?last_seq=N`` so it replays missed
events before tailing the live stream.


simd ls
-------

    simd ls
    simd ls --simulation <UUID> --limit 20

Lists the most recent runs for the active simulation (or the one you
pass via ``--simulation``). Output:

    run_id                                  status     solver                  op                started
    e63a4a69-…                              succeeded  rhoSimpleFoam           CFD_CODEGEN_RUN   2026-05-20T13:21
    df0df3cc-…                              failed     chtMultiRegion…         CFD_CODEGEN_RUN   2026-05-19T22:11


simd stop
---------

    simd stop <RUN_ID>

Hits ``POST /api/runs/{id}/stop``. The sim-server writes a final time
folder and unwinds; the orchestrator returns control to the CLI when
the stop is acknowledged.


configuration
-------------

Lookup precedence for every option:

    flag > env-var > ~/.config/simd/config.toml > built-in default

Environment variables:

    SIMD_AGENT       agent base URL
    SIMD_PROJECT     active simulation UUID

Config file format (``~/.config/simd/config.toml``):

    agent_url = "http://localhost:8000"
    user_id = "1de0b2cb-…"
    simulation_id = "9b3a25ec-…"
    last_run_id = "e63a4a69-…"

The file is created automatically on the first ``simd run`` and
updated as ``simulation_id`` / ``last_run_id`` change. To switch
agents, delete the file or override ``$SIMD_AGENT``.


using it from scripts
---------------------

NDJSON output makes the CLI pipe-friendly. Example: filter for
solver-side errors only.

    simd run prompt.txt mesh.msh --json --yes \
      | jq -c 'select(.type == "sim_run_failed")'

Exit codes:

    0    success
    1    user cancel, backend error, or run failed
    2    argument/usage error


limitations in v0.1
-------------------

  - The patch-review REPL uses plain ``input()``. v0.2 will switch to
    ``prompt_toolkit`` for arrow-key navigation and autocompletion.
  - No ``simd logs`` (use ``GET /api/runs/{id}/events`` directly).
  - No ``simd vtk`` (use ``GET /api/runs/{id}/vtk-results``).
  - ``--detach`` is parsed but the streaming code is unconditional;
    a follow-up adds the actual detach path.
