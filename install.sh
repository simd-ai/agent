#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
# simd-agent — interactive installer
# ══════════════════════════════════════════════════════════════
# Two deployment modes:
#
#   1) Docker      — postgres + agent in containers via
#                    docker compose.  Easiest if you already
#                    have Docker.  Frontend and OpenFOAM runner
#                    are separate repos; install them later if
#                    you want them.
#
#   2) Bare metal  — Python venv + pip install -e . + simd init.
#                    Bring your own Postgres (or run one in a
#                    container).  Point at a remote runner.
#
# Re-run safely — every step is idempotent.
# ══════════════════════════════════════════════════════════════

set -euo pipefail


# ── output helpers ──────────────────────────────────────────────
if [ -t 1 ]; then
  BOLD=$'\033[1m'; NC=$'\033[0m'
  GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; RED=$'\033[0;31m'
  CYAN=$'\033[0;36m'; BLUE=$'\033[0;34m'
else
  BOLD=""; NC=""; GREEN=""; YELLOW=""; RED=""; CYAN=""; BLUE=""
fi

info()    { printf "${BLUE}[INFO]${NC}  %s\n" "$*"; }
ok()      { printf "${GREEN}[OK]${NC}    %s\n" "$*"; }
warn()    { printf "${YELLOW}[WARN]${NC}  %s\n" "$*"; }
err()     { printf "${RED}[ERROR]${NC} %s\n" "$*" >&2; }
fail()    { err "$*"; exit 1; }
header()  { printf "\n${BOLD}${CYAN}── %s ──${NC}\n\n" "$*"; }
hint()    { printf "    ${CYAN}→${NC} %s\n" "$*"; }


# ── prompt helpers ──────────────────────────────────────────────
ask() {
  # ask <prompt> <default> <var>
  local prompt="$1" default="$2" var="$3" input
  if [ -n "$default" ]; then
    read -rp "$(printf "${BOLD}%s${NC} [%s]: " "$prompt" "$default")" input
    eval "$var=\"${input:-$default}\""
  else
    read -rp "$(printf "${BOLD}%s${NC}: " "$prompt")" input
    eval "$var=\"$input\""
  fi
}

ask_yes_no() {
  # ask_yes_no <prompt> <default Y/N>
  local prompt="$1" default="${2:-Y}" answer
  read -rp "$(printf "${BOLD}%s${NC} [%s/%s]: " "$prompt" \
              "$( [ "$default" = "Y" ] && echo "Y" || echo "y")" \
              "$( [ "$default" = "Y" ] && echo "n" || echo "N")")" answer
  answer="${answer:-$default}"
  [[ "$answer" =~ ^[Yy]$ ]]
}

ask_choice() {
  # ask_choice <prompt> <regex> <default> <var>
  local prompt="$1" pattern="$2" default="$3" var="$4" input
  while true; do
    read -rp "$(printf "${BOLD}%s${NC} [%s]: " "$prompt" "$default")" input
    input="${input:-$default}"
    if [[ "$input" =~ $pattern ]]; then
      eval "$var=\"$input\""
      return
    fi
    err "invalid choice: $input"
  done
}


# ── locate ourselves ────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$SCRIPT_DIR"
ENV_FILE="$AGENT_DIR/.env"
cd "$AGENT_DIR"


# ══════════════════════════════════════════════════════════════
# 1. Welcome
# ══════════════════════════════════════════════════════════════
echo
printf "${BOLD}${CYAN}╔══════════════════════════════════════════════════════╗${NC}\n"
printf "${BOLD}${CYAN}║          simd-agent — installation wizard           ║${NC}\n"
printf "${BOLD}${CYAN}╚══════════════════════════════════════════════════════╝${NC}\n"
echo
echo "  this script will set up the simd-agent stack."
echo "  you'll be asked a few questions to configure your deployment."
echo


# ══════════════════════════════════════════════════════════════
# 2. Deployment mode (Docker or Bare metal)
# ══════════════════════════════════════════════════════════════
header "deployment mode"

echo "  1) Docker     — postgres + agent in containers (recommended"
echo "                  if you already have Docker installed)"
echo "  2) Bare metal — Python venv on this machine, you run uvicorn"
echo
ask_choice "choose deployment mode" "^[12]$" "1" DEPLOY_CHOICE

if [ "$DEPLOY_CHOICE" = "1" ]; then
  DEPLOY_MODE="docker"
  ok "Docker deployment selected"
else
  DEPLOY_MODE="bare-metal"
  ok "bare-metal deployment selected"
fi


# ══════════════════════════════════════════════════════════════
# 3. Configuration (asked in both modes)
# ══════════════════════════════════════════════════════════════
header "LLM provider"

echo "  1) Gemini  (Google AI Studio — easiest, has a daily cap)"
echo "  2) Vertex  (GCP Vertex AI — no daily cap, needs SA JSON)"
echo "  3) Ollama  (local — runs models on this machine)"
echo
ask_choice "choose LLM provider" "^[123]$" "1" LLM_CHOICE

GEMINI_API_KEY=""
VERTEX_PROJECT=""
GOOGLE_APPLICATION_CREDENTIALS=""
OLLAMA_HOST=""

if [ "$LLM_CHOICE" = "1" ]; then
  LLM_PROVIDER="gemini"
  while [ -z "$GEMINI_API_KEY" ]; do
    ask "Gemini API key (get one at https://aistudio.google.com/apikey)" "" GEMINI_API_KEY
    [ -z "$GEMINI_API_KEY" ] && err "a Gemini API key is required."
  done
  ok "Gemini configured"
elif [ "$LLM_CHOICE" = "2" ]; then
  LLM_PROVIDER="vertex"
  ask "GCP project ID" "" VERTEX_PROJECT
  ask "path to service-account JSON" "" GOOGLE_APPLICATION_CREDENTIALS
  [ -f "$GOOGLE_APPLICATION_CREDENTIALS" ] || fail "file not found: $GOOGLE_APPLICATION_CREDENTIALS"
  ok "Vertex configured: $VERTEX_PROJECT"
else
  LLM_PROVIDER="ollama"
  ask "Ollama host URL" "http://localhost:11434" OLLAMA_HOST
  ok "Ollama configured: $OLLAMA_HOST"
fi


header "simulation runner"

echo "  the OpenFOAM runner is a separate service (simd-ai/simulation_server)."
echo "  options:"
echo
echo "    a) point at an existing runner (e.g. a server you already have)"
echo "    b) install + run it yourself later (we'll skip for now)"
echo
ask "simulation runner URL (e.g. http://localhost:9000)" \
    "http://localhost:9000" SIM_SERVER_URL


header "object storage"

echo "  meshes, simulation results, and case files are stored here."
echo
echo "  1) local filesystem (default — no setup needed)"
echo "  2) Google Cloud Storage (requires a bucket + SA JSON)"
echo
ask_choice "choose storage backend" "^[12]$" "1" STORAGE_CHOICE

STORAGE_BACKEND="local"
STORAGE_BUCKET=""
GCS_KEY_PATH=""

if [ "$STORAGE_CHOICE" = "2" ]; then
  STORAGE_BACKEND="gcs"
  ask "GCS bucket name" "" STORAGE_BUCKET
  if [ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]; then
    ask "path to GCS service-account JSON" "" GCS_KEY_PATH
    [ -f "$GCS_KEY_PATH" ] || fail "file not found: $GCS_KEY_PATH"
    GOOGLE_APPLICATION_CREDENTIALS="$GCS_KEY_PATH"
  else
    info "reusing GCS credentials from the LLM provider step"
  fi
  ok "GCS configured: $STORAGE_BUCKET"
else
  ok "using local filesystem storage"
fi


header "authentication"

echo "  by default, simd-agent runs without authentication (single local user)."
echo "  enable Neon Auth only if you want multi-user support."
echo
echo "  1) Open  (no authentication — default)"
echo "  2) Neon Auth  (requires a Neon project)"
echo
ask_choice "choose auth mode" "^[12]$" "1" AUTH_CHOICE

NEON_AUTH_URL=""
if [ "$AUTH_CHOICE" = "2" ]; then
  ask "Neon Auth base URL" "" NEON_AUTH_URL
  ok "Neon Auth configured"
else
  ok "authentication disabled (single-user mode)"
fi


header "database"

if [ "$DEPLOY_MODE" = "docker" ]; then
  echo "  1) bundled Postgres  (a postgres container ships with the stack)"
  echo "  2) external Postgres (Neon, RDS, your own host)"
  echo
  ask_choice "choose database" "^[12]$" "1" DB_CHOICE

  if [ "$DB_CHOICE" = "1" ]; then
    DATABASE_URL="postgresql+asyncpg://simd:simd@postgres:5432/simd"
    ok "using bundled Postgres container"
  else
    ask "PostgreSQL connection URL (postgresql://user:pass@host/db)" "" DATABASE_URL
    DATABASE_URL="${DATABASE_URL/postgresql:\/\//postgresql+asyncpg:\/\/}"
    ok "external database configured"
  fi
else
  echo "  bare-metal mode needs you to have Postgres reachable from this machine."
  echo "  options:"
  echo "    a) Neon (managed) — paste your connection string"
  echo "    b) Local Postgres (brew, apt)"
  echo "    c) Postgres in a container:"
  hint "docker run -d --name simd-pg \\"
  hint "  -e POSTGRES_USER=simd -e POSTGRES_PASSWORD=simd \\"
  hint "  -e POSTGRES_DB=simd -p 5432:5432 postgres:16-alpine"
  echo
  ask "PostgreSQL connection URL" \
      "postgresql+asyncpg://simd:simd@localhost:5432/simd" DATABASE_URL
  DATABASE_URL="${DATABASE_URL/postgresql:\/\//postgresql+asyncpg:\/\/}"
  ok "database URL set"
fi


# ══════════════════════════════════════════════════════════════
# 4. Write .env
# ══════════════════════════════════════════════════════════════
header "writing .env"

if [ -f "$ENV_FILE" ]; then
  warn ".env already exists — backing up to .env.bak"
  cp "$ENV_FILE" "$ENV_FILE.bak"
fi

{
  echo "# ─── written by install.sh ────────────────────────────────"
  echo "# Edit by hand or re-run install.sh to regenerate."
  echo
  echo "# ── Database ─────────────────────────────────────────────"
  echo "DATABASE_URL=$DATABASE_URL"
  echo
  echo "# ── Simulation runner ────────────────────────────────────"
  echo "SIMULATION_SERVER_URL=$SIM_SERVER_URL"
  echo
  echo "# ── LLM provider ─────────────────────────────────────────"
  echo "DEFAULT_PROVIDER=$LLM_PROVIDER"
  case "$LLM_PROVIDER" in
    gemini) echo "GEMINI_API_KEY=$GEMINI_API_KEY" ;;
    vertex)
      echo "VERTEX_PROJECT=$VERTEX_PROJECT"
      echo "VERTEX_LOCATION=us-central1"
      echo "GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS"
      ;;
    ollama) echo "OLLAMA_HOST=$OLLAMA_HOST" ;;
  esac
  echo
  echo "# ── Storage ──────────────────────────────────────────────"
  echo "STORAGE_BACKEND=$STORAGE_BACKEND"
  if [ "$STORAGE_BACKEND" = "local" ]; then
    if [ "$DEPLOY_MODE" = "docker" ]; then
      echo "STORAGE_LOCAL_DIR=/app/storage"
    else
      echo "STORAGE_LOCAL_DIR=$AGENT_DIR/storage"
    fi
  else
    echo "STORAGE_BUCKET=$STORAGE_BUCKET"
    [ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ] && \
      echo "GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS"
  fi
  echo
  echo "# ── Auth ─────────────────────────────────────────────────"
  [ -n "$NEON_AUTH_URL" ] && echo "NEON_AUTH_BASE_URL=$NEON_AUTH_URL"
  echo
  echo "# ── Self-healing ─────────────────────────────────────────"
  echo "MAX_RETRIES=7"
} > "$ENV_FILE"

chmod 600 "$ENV_FILE"
ok ".env written to $ENV_FILE"


# ══════════════════════════════════════════════════════════════
# 5A. Docker deployment
# ══════════════════════════════════════════════════════════════
if [ "$DEPLOY_MODE" = "docker" ]; then
  header "Docker deployment"

  command -v docker >/dev/null || fail \
    "Docker isn't installed — get it at https://docs.docker.com/get-docker/"
  docker info >/dev/null 2>&1 || fail \
    "the Docker daemon isn't running — start Docker Desktop / your daemon"
  ok "Docker is available"

  COMPOSE_CMD="docker compose -f docker/docker-compose.yml"

  if [ "$STORAGE_BACKEND" = "gcs" ]; then
    cat > "$AGENT_DIR/docker/docker-compose.gcs.yml" <<GCSYML
services:
  agent:
    volumes:
      - $GOOGLE_APPLICATION_CREDENTIALS:/secrets/gcs-key.json:ro
GCSYML
    COMPOSE_CMD="$COMPOSE_CMD -f docker/docker-compose.gcs.yml"
  fi

  echo
  warn "the GHCR images don't exist yet for this OSS release."
  warn "to start the stack, uncomment the \`build:\` blocks in"
  warn "docker/docker-compose.yml so docker compose builds locally."
  echo

  if ask_yes_no "start the stack now?" "Y"; then
    info "running:  $COMPOSE_CMD up -d"
    $COMPOSE_CMD up -d
    echo
    ok "stack started.  endpoints:"
    hint "Backend:    http://localhost:8000"
    hint "Postgres:   localhost:5432"
    echo
    hint "view logs:  $COMPOSE_CMD logs -f"
    hint "stop:       $COMPOSE_CMD down"
  else
    echo
    info "to start later, run:"
    hint "$COMPOSE_CMD up -d"
  fi

# ══════════════════════════════════════════════════════════════
# 5B. Bare-metal deployment
# ══════════════════════════════════════════════════════════════
else
  header "bare-metal setup"

  # ── Python check ─────────────────────────────────────────────
  command -v python3 >/dev/null || fail "python3 not found.  install Python 3.11+."
  PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  PY_MAJOR="$(echo "$PY_VER" | cut -d. -f1)"
  PY_MINOR="$(echo "$PY_VER" | cut -d. -f2)"
  if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]; }; then
    fail "Python $PY_VER found, but 3.11+ is required."
  fi
  ok "Python $PY_VER"

  # ── venv + install ───────────────────────────────────────────
  if [ -d ".venv" ]; then
    ok ".venv already exists — reusing"
  else
    python3 -m venv .venv
    ok "created .venv/"
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate

  info "installing simd-agent and the CLI …"
  pip install --upgrade pip --quiet
  pip install -e . --quiet
  command -v simd >/dev/null || fail \
    "the 'simd' command isn't on PATH after install — pip install -e . may have failed silently."
  ok "installed  $(simd --version)"

  # ── storage dir ─────────────────────────────────────────────
  if [ "$STORAGE_BACKEND" = "local" ]; then
    mkdir -p "$AGENT_DIR/storage"
    mkdir -p "$AGENT_DIR/progress_data"
    ok "storage directories ready"
  fi

  # ── simd init ───────────────────────────────────────────────
  header "CLI configuration"

  if [ -f "$HOME/.config/simd/config.toml" ]; then
    ok "~/.config/simd/config.toml already exists — skipping wizard"
    hint "re-run \`simd init\` later to reconfigure"
  else
    info "running the simd init wizard …"
    echo
    simd init || warn "simd init cancelled — you can re-run it later"
  fi

  # ── final instructions ──────────────────────────────────────
  header "setup complete"

  cat <<EOF

  next steps:

    # terminal 1 — start the agent (keeps running)
    source .venv/bin/activate
    uvicorn simd_agent.main:app --port 8000

    # terminal 2 — run an example
    source .venv/bin/activate
    simd run examples/u-shape-pipe/prompt.txt \\
             examples/u-shape-pipe/mesh/u-shape-pipe.msh

  to see what's wired up:
    simd status

  to deactivate the venv:
    deactivate

EOF
fi


# ══════════════════════════════════════════════════════════════
# 6. Done
# ══════════════════════════════════════════════════════════════
printf "${BOLD}${CYAN}════════════════════════════════════════════════════════${NC}\n"
printf "${BOLD}${CYAN}  installation complete.                                ${NC}\n"
printf "${BOLD}${CYAN}════════════════════════════════════════════════════════${NC}\n"
echo
