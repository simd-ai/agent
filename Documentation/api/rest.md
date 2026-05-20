REST API
========

All endpoints under `http://<agent-host>:8000/`. Auth-disabled mode
serves them open; with Neon Auth enabled, most write endpoints
require a session cookie or `Authorization: Bearer <token>` header.

The interactive OpenAPI explorer lives at
`http://<host>:8000/docs`.


health
------

  - `GET /health` → `{"status": "ok"}`
  - `GET /` → service banner with version, providers, storage backend


precheck
--------

  - `POST /api/precheck` — synchronous precheck of a natural-language
    prompt. Returns linted physics + suggested solver + per-patch
    boundary plan.

  Streaming version is `WS /ws/precheck` (see websocket.md).


runs
----

  - `GET /api/runs/{run_id}` — full run record (status, generated
    files, final result, residuals summary).
  - `GET /api/runs/{run_id}/events` — replay all persisted
    `AgentEvent` objects for a run. Used by the frontend to restore
    the progress panel after a page reload.
  - `GET /api/runs/{run_id}/status` — short JSON
    `{"status": "running" | "succeeded" | "failed" | "stopped" |
    "cancelled" | "pending"}`. Cheaper than fetching the full
    record.
  - `POST /api/runs/{run_id}/cancel` — mark the run as cancelled in
    the DB. Used by the frontend when the user closes the WebSocket
    without a clean shutdown.
  - `POST /api/runs/{run_id}/stop` — gracefully stop a running
    simulation. Tells the sim-server to write a final time folder
    and unwind. Returns once the stop is acknowledged.
  - `POST /api/runs/{run_id}/continue` — extend a stopped run with
    additional iterations. Restarts from the latest checkpoint.


vtk results
-----------

  - `GET /api/runs/{run_id}/vtk-results` — metadata + the latest
    surface VTP URL. Returns 404 if the run never produced VTK
    output (failed early, no time folders written). The frontend
    pre-checks `run.result.sim_run_id` before calling this to avoid
    a noisy 404.

  - `GET /api/runs/{run_id}/vtk/surface.vtp` — latest surface VTP.
  - `GET /api/runs/{run_id}/timesteps` — list of available time
    folders.
  - `GET /api/runs/{run_id}/vtk-timestep/{time}/surface.vtp` —
    per-timestep surface.
  - `GET /api/runs/{run_id}/vtk-timestep/{time}/{region}/surface.vtp`
    — per-region (CHT) per-timestep surface.
  - `GET /api/runs/{run_id}/playback` — Server-Sent Events stream
    that pushes one timestep VTP URL at a time, paced for
    visualization playback.
  - `GET /api/runs/{run_id}/vtk-progress` — VTK download/conversion
    progress.
  - `GET /api/runs/{run_id}/vtk-debug` — internal diagnostic.
  - `POST /api/runs/{run_id}/vtk-clear-cache` — drop the cached
    VTPs for a run; next read re-downloads from the sim-server.


simulations and projects
------------------------

  - `GET /api/simulations` — list simulations for the current user.
  - `GET /api/simulations/{id}/snapshot/primary` — quick essentials
    for the workspace header.
  - `GET /api/simulations/{id}/snapshot/essentials` — chat, mesh,
    precheck, last run.
  - `POST /api/simulations` — create new simulation record.
  - …and the rest of the CRUD: full schema in `app/api/*.py`.


users and auth (when Neon Auth is enabled)
------------------------------------------

  - `POST /auth/login` / `GET /auth/me` — Neon Auth flow.
  - `GET /api/users/{uid}/usage` — current project / run counts vs.
    tier limits.


internal admin endpoints
------------------------

A few admin endpoints (`/api/admin/*`) require an admin email match
against `ADMIN_EMAILS`. Not documented further here — see the
source in `simd_agent/api/admin.py`.


error responses
---------------

All errors return a JSON body:

    {
      "detail": "...human-readable message..."
    }

with the appropriate HTTP status. Validation errors (422) include
the failing field path. Foreign-key constraint violations return
404 instead of 500 (the run / simulation was deleted out from
under us).
