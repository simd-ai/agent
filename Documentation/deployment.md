deployment
==========

Production setup. What you'd run on a server that's not your laptop:
managed Postgres, GCS for object storage, Vertex AI (no rate cap),
optional Neon Auth.


what changes vs. local
----------------------

  - **Postgres** → Neon (or any managed Postgres).
  - **Storage** → Google Cloud Storage instead of local filesystem.
  - **LLM** → Vertex AI instead of public Gemini, to avoid the daily
    cap.
  - **Auth** → Neon Auth, enabling per-user projects and Stripe
    integration.
  - **Sim-server** → a separate VM or container with GPU/CPU
    headroom, not running on the same host as the agent.


.env for production
-------------------

    # Database
    DATABASE_URL=postgresql+asyncpg://user:pass@ep-xxx.aws.neon.tech/simd

    # LLM (Vertex)
    DEFAULT_PROVIDER=vertex
    VERTEX_PROJECT=your-gcp-project
    VERTEX_LOCATION=us-central1
    GOOGLE_APPLICATION_CREDENTIALS=/secrets/sa-key.json

    # Storage
    STORAGE_BACKEND=gcs
    STORAGE_BUCKET=simd-cases-prod
    PROGRESS_GCS_BUCKET=simd-progress-prod
    # GOOGLE_APPLICATION_CREDENTIALS already set above (one SA key
    # covers both Vertex and GCS)

    # Sim-server (separate host)
    SIMULATION_SERVER_URL=https://runner.example.com

    # Auth
    NEON_AUTH_BASE_URL=https://ep-xxx.neonauth.us-east-1.aws.neon.tech

    # Logging
    LOG_LEVEL=INFO

    # Self-healing
    MAX_RETRIES=7


1. Postgres on Neon
-------------------

  1. Sign up at neon.tech.
  2. Create a project. Pick a region near your agent host.
  3. Copy the pooled connection string from the dashboard.
  4. Replace the `postgresql://` prefix with
     `postgresql+asyncpg://` (the agent uses the async driver).

Neon's free tier handles the agent's load comfortably for low-volume
production. For higher traffic, switch to the Scale plan or move to
self-hosted Postgres.


2. GCS for storage
------------------

  1. In the GCP console, create two buckets:
       - `simd-cases-prod` (case ZIPs, meshes, merged VTPs)
       - `simd-progress-prod` (residuals as NDJSON)
     Choose a region near your agent host.
     Enable uniform bucket-level access.

  2. Create a service account, e.g. `simd-agent-prod@…`. Grant:
       - `roles/storage.objectAdmin` on both buckets
       - `roles/aiplatform.user` on the project (for Vertex)

  3. Create a JSON key for that service account, download it.

  4. Mount the JSON into the agent container at
     `/secrets/sa-key.json` (or set
     `GOOGLE_APPLICATION_CREDENTIALS` to wherever you put it on
     bare-metal).

  5. Set `STORAGE_BACKEND=gcs` and `STORAGE_BUCKET=simd-cases-prod`.

The agent reads `GOOGLE_APPLICATION_CREDENTIALS` from `.env` and
injects it into `os.environ` on first use (see
`simd_agent/storage/gcs.py` and `simd_agent/llm/vertex/provider.py`).


3. Vertex AI
------------

Lifts the AI Studio daily request cap. Uses the same `google-genai`
SDK as the public Gemini provider — model IDs are identical.

  1. Enable the Vertex AI API on your GCP project:
     `gcloud services enable aiplatform.googleapis.com`
     (or click "Enable" in the Vertex AI section of the console).

  2. The service account from step 2 above already has
     `roles/aiplatform.user`. One key serves both Vertex and GCS.

  3. In `.env`:

         DEFAULT_PROVIDER=vertex
         VERTEX_PROJECT=your-gcp-project
         VERTEX_LOCATION=us-central1

  4. Restart the agent.

See `llm-providers/vertex.md` for the per-region model availability
matrix.


4. Neon Auth
------------

Optional. Enables per-user projects, Google sign-in, and Stripe-
backed paid tiers.

  1. In the Neon dashboard, enable Neon Auth on your project.
  2. Copy the Neon Auth base URL — looks like
     `https://ep-xxx.neonauth.us-east-1.aws.neon.tech`.
  3. Set `NEON_AUTH_BASE_URL` in the agent's `.env`.
  4. In the frontend's environment, set
     `NEXT_PUBLIC_AUTH_DISABLED=false` and configure
     `NEON_AUTH_*` values per the frontend's README.

If `NEON_AUTH_BASE_URL` is unset, the agent runs in open mode: a
single local user, no signup, no usage limits. Useful for
self-hosted single-tenant deployments.


5. sim-server placement
-----------------------

The sim-server is the heaviest process — it runs OpenFOAM. Two
patterns work:

  - **Co-located VM** — for low-traffic deployments, run the agent
    and the sim-server on the same VM. The agent's
    `SIMULATION_SERVER_URL` is `http://localhost:9000`. Simple, but
    each simulation contends with the agent for CPU.

  - **Dedicated runner** — put the sim-server on a separate VM or
    a Kubernetes pod with more CPU and memory. The agent reaches it
    over HTTP. Multiple sim-server replicas behind a load balancer
    are supported as long as each replica writes to a shared volume
    (or pulls case ZIPs from GCS — both flows are implemented).

For now the sim-server runs one simulation at a time per replica.
Concurrent runs scale by adding replicas.


6. Docker Compose for production
--------------------------------

The shipped `docker-compose.yml` builds for local dev. For
production, copy it and switch:

  - drop the `postgres` service (use managed Postgres)
  - drop the build contexts; point at GHCR tags like
    `ghcr.io/simd-ai/agent:v0.1.0`
  - mount your service-account key as a secret
  - set `restart: always`

The `examples/deploy/` directory will ship a reference compose
file and a Kubernetes manifest — coming soon.


7. observability
----------------

The agent writes structured logs with `[STAGE]` prefixes. For a
quick eyeballed view:

    docker logs -f simd-agent | grep -E '\[(SIM_ERROR|DIAGNOSE)\]'

For real production you'd ship logs to whatever you already use
(Cloud Logging, Datadog, Grafana Loki, …).

A `/api/health` endpoint returns 200 with `{"status":"ok"}` once
the agent is up. Use it for liveness probes.

Telemetry to Umami is opt-in via `TELEMETRY_ENABLED`. Set to `false`
in production to disable.


8. backups
----------

  - **Postgres** — Neon's automatic PITR backups cover this.
    For self-hosted Postgres, `pg_dump` on a cron.
  - **GCS** — versioning + lifecycle rules. The case ZIPs and VTPs
    are deterministic from the prompt + mesh, so losing them is
    annoying but not catastrophic. Mesh files are the only truly
    user-generated artifact.

  - **The mesh bucket is the one to back up religiously.**


9. updating
-----------

    docker compose pull
    docker compose up -d

GHCR tags follow semver: `:latest` tracks `main` (bleeding edge),
`:vX.Y.Z` are immutable releases. Pin to a version in production.
