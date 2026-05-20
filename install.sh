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
#   2) Bare metal  — Python venv + pip install -e .  Bring your
#                    own Postgres (or run one in a container).
#                    Point at a remote runner.
#
# Either path writes both .env (for the agent) and
# ~/.config/simd/config.toml (for the CLI) — no separate
# `simd init` step needed after install.sh.
#
# Menu choices use arrow keys (↑ ↓ Enter).  Free-text inputs
# (URLs, API keys, file paths) are typed normally.
#
# Re-run safely — every step is idempotent.
# ══════════════════════════════════════════════════════════════

set -uo pipefail
# NB: not using `-e` because interactive `read` can return non-zero
# in ways that aren't fatal (escape sequences, etc.).  Errors get
# surfaced explicitly via `fail`.


# ── TTY guard ───────────────────────────────────────────────────
if [ ! -t 0 ] || [ ! -t 1 ]; then
  echo "install.sh needs an interactive terminal." >&2
  echo "for non-interactive setup, edit .env by hand and run:" >&2
  echo "    pip install -e . && simd init" >&2
  exit 1
fi


# ── output helpers ──────────────────────────────────────────────
BOLD=$'\033[1m'; NC=$'\033[0m'
GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; RED=$'\033[0;31m'
CYAN=$'\033[0;36m'; BLUE=$'\033[0;34m'; DIM=$'\033[2m'

info()    { printf "${BLUE}[INFO]${NC}  %s\n" "$*"; }
ok()      { printf "${GREEN}[OK]${NC}    %s\n" "$*"; }
warn()    { printf "${YELLOW}[WARN]${NC}  %s\n" "$*"; }
err()     { printf "${RED}[ERROR]${NC} %s\n" "$*" >&2; }
fail()    { err "$*"; exit 1; }
header()  { printf "\n${BOLD}${CYAN}── %s ──${NC}\n\n" "$*"; }
hint()    { printf "    ${CYAN}→${NC} %s\n" "$*"; }


# ── arrow-key menu ──────────────────────────────────────────────
#
# Renders a list, highlights the current row, redraws on each
# keypress.  ↑/↓ (or vim's k/j) move; Enter confirms; q quits.
# 1..9 digits act as direct hotkeys.  The result lands in two
# globals:
#
#   $_ARROW_INDEX   — 0-based index of the chosen option
#   $_ARROW_RESULT  — the chosen option's label string
#
# Caller reads them right after the function returns.

_ARROW_INDEX=0
_ARROW_RESULT=""
# Callers set _ARROW_DEFAULT=N before invoking arrow_choice to pre-select
# option N (0-based).  It auto-resets to 0 after each call so subsequent
# prompts default to "first option" again unless explicitly overridden.
_ARROW_DEFAULT=0

arrow_choice() {
  # arrow_choice <prompt> <option1> [<option2> ...]
  local prompt="$1"
  shift
  local options=("$@")
  local count=${#options[@]}
  local selected=${_ARROW_DEFAULT:-0}
  # Clamp the pre-selected index — a stale value pointing past the
  # current option list would render off-screen.
  if [ "$selected" -lt 0 ] || [ "$selected" -ge "$count" ]; then
    selected=0
  fi
  # Auto-reset so the next arrow_choice gets a clean default.
  _ARROW_DEFAULT=0
  local key key2

  printf "${BOLD}%s${NC}\n" "$prompt"
  printf "${DIM}  (↑/↓ to move, Enter to select, q to quit)${NC}\n"

  tput civis 2>/dev/null || true
  trap '_arrow_cleanup' INT TERM

  _arrow_draw "$selected" "${options[@]}"

  while true; do
    IFS= read -rsn1 key 2>/dev/null || break
    if [[ "$key" == $'\e' ]]; then
      IFS= read -rsn2 -t 0.05 key2 2>/dev/null || key2=""
      case "$key2" in
        '[A'|'OA') ((selected = (selected - 1 + count) % count)) ;;
        '[B'|'OB') ((selected = (selected + 1) % count)) ;;
        *) ;;
      esac
    elif [[ -z "$key" ]]; then
      break  # Enter
    elif [[ "$key" == "k" ]]; then
      ((selected = (selected - 1 + count) % count))
    elif [[ "$key" == "j" ]]; then
      ((selected = (selected + 1) % count))
    elif [[ "$key" =~ ^[0-9]$ ]] && [ "$key" -ge 1 ] && [ "$key" -le "$count" ]; then
      selected=$((key - 1))
      break
    elif [[ "$key" == "q" ]]; then
      _arrow_cleanup
      fail "cancelled."
    fi

    tput cuu "$count" 2>/dev/null || true
    _arrow_draw "$selected" "${options[@]}"
  done

  tput cnorm 2>/dev/null || true
  trap - INT TERM

  _ARROW_INDEX="$selected"
  _ARROW_RESULT="${options[selected]}"
}

_arrow_draw() {
  local sel="$1"; shift
  local opts=("$@")
  local n=${#opts[@]}
  local i
  for ((i=0; i<n; i++)); do
    tput el 2>/dev/null || true
    if [ "$i" -eq "$sel" ]; then
      printf "  ${BOLD}${CYAN}❯${NC} ${BOLD}%s${NC}\n" "${opts[i]}"
    else
      printf "    %s\n" "${opts[i]}"
    fi
  done
}

_arrow_cleanup() {
  tput cnorm 2>/dev/null || true
  echo
}


# ── free-text prompts ───────────────────────────────────────────

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

ask_path() {
  # ask_path <prompt> <var> [<default>] — keeps asking until the file exists.
  # If a default is given and the user presses Enter, the default is used
  # (provided that file actually exists; otherwise we ask again).
  local prompt="$1" var="$2" default="${3:-}" path
  while true; do
    if [ -n "$default" ]; then
      read -rp "$(printf "${BOLD}%s${NC} [%s]: " "$prompt" "$default")" path
      [ -z "$path" ] && path="$default"
    else
      read -rp "$(printf "${BOLD}%s${NC}: " "$prompt")" path
    fi
    path="${path/#\~/$HOME}"
    if [ -f "$path" ]; then
      eval "$var=\"$path\""
      return
    fi
    err "file not found: $path"
  done
}


# ── CLI config writer ──────────────────────────────────────────
#
# Write ~/.config/simd/config.toml directly from the wizard's
# answers.  Replaces the old approach of calling `simd init` at
# the end of install.sh (which asked all the same questions again).
# `simd init` remains a standalone command for users who install
# just the CLI (e.g. future `pip install simd-agent` without
# running install.sh).

write_cli_config() {
  # write_cli_config <agent_url> <agent_mode> <runner_url> <runner_mode>
  local agent_url="$1" agent_mode="$2" runner_url="$3" runner_mode="$4"

  local cfg_dir="$HOME/.config/simd"
  local cfg_file="$cfg_dir/config.toml"
  mkdir -p "$cfg_dir"

  if [ -f "$cfg_file" ]; then
    cp "$cfg_file" "$cfg_file.bak"
    warn "$cfg_file already existed — backed up to config.toml.bak"
  fi

  cat > "$cfg_file" <<TOML
# Written by install.sh.  Edit by hand or re-run install.sh.
agent_url = "$agent_url"
agent_mode = "$agent_mode"
runner_url = "$runner_url"
runner_mode = "$runner_mode"
TOML
  chmod 600 "$cfg_file" 2>/dev/null || true
  ok "CLI config written to $cfg_file"
}


# ── locate ourselves ────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$SCRIPT_DIR"
ENV_FILE="$AGENT_DIR/.env"
cd "$AGENT_DIR"


# ── load any pre-existing .env so re-runs preserve choices ──────
# When the user re-runs install.sh, every prompt should default to
# whatever they picked last time — not the fresh-install defaults.
# Sourcing the prior .env into the shell sets DATABASE_URL,
# DEFAULT_PROVIDER, SIMULATION_SERVER_URL, STORAGE_BACKEND, etc. as
# env vars; the prompts later in the script consult them to compute
# arrow defaults and ``ask`` fallbacks.
PREFILLED=0
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
  PREFILLED=1
fi

# Snapshot prior values into ``PREV_*`` so later assignments inside the
# wizard (eg ``STORAGE_BACKEND="local"`` resetting before the prompt)
# don't clobber what the user picked last time.  Every prompt computes
# its default from the matching ``PREV_*`` variable below.
PREV_DATABASE_URL="${DATABASE_URL:-}"
PREV_DEFAULT_PROVIDER="${DEFAULT_PROVIDER:-}"
PREV_GEMINI_API_KEY="${GEMINI_API_KEY:-}"
PREV_VERTEX_PROJECT="${VERTEX_PROJECT:-}"
PREV_GOOGLE_APPLICATION_CREDENTIALS="${GOOGLE_APPLICATION_CREDENTIALS:-}"
PREV_OLLAMA_HOST="${OLLAMA_HOST:-}"
PREV_SIM_SERVER_URL="${SIMULATION_SERVER_URL:-}"
PREV_STORAGE_BACKEND="${STORAGE_BACKEND:-}"
PREV_STORAGE_BUCKET="${STORAGE_BUCKET:-}"
PREV_NEON_AUTH_URL="${NEON_AUTH_BASE_URL:-}"


# ══════════════════════════════════════════════════════════════
# 1. Welcome
# ══════════════════════════════════════════════════════════════
echo
printf "${BOLD}simd-agent installer${NC}\n"
if [ "$PREFILLED" -eq 1 ]; then
  printf "${DIM}↑/↓ to move, Enter to confirm.  prior .env detected — your previous choices are pre-selected.${NC}\n"
else
  printf "${DIM}↑/↓ to move, Enter to confirm.${NC}\n"
fi
echo


# ══════════════════════════════════════════════════════════════
# 2. Deployment mode
# ══════════════════════════════════════════════════════════════
header "deployment mode"

arrow_choice "where should simd-agent run?" \
  "Docker     — postgres + agent in containers (recommended if Docker is installed)" \
  "Bare metal — Python venv on this machine, you run uvicorn"

case "$_ARROW_INDEX" in
  0) DEPLOY_MODE="docker"     ; ok "Docker deployment selected" ;;
  1) DEPLOY_MODE="bare-metal" ; ok "bare-metal deployment selected" ;;
esac


# ══════════════════════════════════════════════════════════════
# 3. LLM provider
# ══════════════════════════════════════════════════════════════
header "LLM provider"

case "$PREV_DEFAULT_PROVIDER" in
  vertex) _ARROW_DEFAULT=1 ;;
  ollama) _ARROW_DEFAULT=2 ;;
  gemini|*) _ARROW_DEFAULT=0 ;;
esac
arrow_choice "which LLM provider?" \
  "Gemini  — Google AI Studio (easiest, has a daily cap)" \
  "Vertex  — GCP Vertex AI (no daily cap, needs a service-account JSON)" \
  "Ollama  — local (runs models on this machine, no API key)"

GEMINI_API_KEY=""
VERTEX_PROJECT=""
GOOGLE_APPLICATION_CREDENTIALS=""
OLLAMA_HOST=""

case "$_ARROW_INDEX" in
  0)
    LLM_PROVIDER="gemini"
    # Reuse the prior key as the default; an empty default still
    # loops until the user enters something non-empty.
    while [ -z "$GEMINI_API_KEY" ]; do
      ask "Gemini API key (https://aistudio.google.com/apikey)" \
          "$PREV_GEMINI_API_KEY" GEMINI_API_KEY
      [ -z "$GEMINI_API_KEY" ] && err "a Gemini API key is required."
    done
    ok "Gemini configured"
    ;;
  1)
    LLM_PROVIDER="vertex"
    ask_path "path to GCP service-account JSON (e.g. ~/.gcp/key.json)" \
             GOOGLE_APPLICATION_CREDENTIALS \
             "$PREV_GOOGLE_APPLICATION_CREDENTIALS"
    # The project_id is inside the JSON — extract it via python.
    VERTEX_PROJECT="$(python3 -c "
import json, sys
try:
    d = json.load(open('$GOOGLE_APPLICATION_CREDENTIALS'))
    pid = d.get('project_id')
    if not pid:
        sys.exit('no project_id field in JSON')
    print(pid)
except Exception as e:
    sys.exit(f'parse error: {e}')
" 2>&1)" || fail "couldn't read project_id from $GOOGLE_APPLICATION_CREDENTIALS — $VERTEX_PROJECT"
    ok "Vertex configured: project = $VERTEX_PROJECT"
    ;;
  2)
    LLM_PROVIDER="ollama"
    ask "Ollama host URL" "${PREV_OLLAMA_HOST:-http://localhost:11434}" OLLAMA_HOST
    ok "Ollama configured: $OLLAMA_HOST"
    ;;
esac


# ══════════════════════════════════════════════════════════════
# 4. Simulation runner
# ══════════════════════════════════════════════════════════════
header "simulation runner"

echo "  the OpenFOAM runner is a separate service (simd-ai/simulation_server)."
echo "  enter the URL where it's reachable from here."
echo
ask "simulation runner URL" "${PREV_SIM_SERVER_URL:-http://localhost:9000}" SIM_SERVER_URL


# ══════════════════════════════════════════════════════════════
# 5. Storage
# ══════════════════════════════════════════════════════════════
header "object storage"

[ "$PREV_STORAGE_BACKEND" = "gcs" ] && _ARROW_DEFAULT=1 || _ARROW_DEFAULT=0
arrow_choice "where do meshes, VTPs, and case ZIPs live?" \
  "Local filesystem (default — no setup needed)" \
  "Google Cloud Storage (requires a bucket)"

STORAGE_BACKEND="local"
STORAGE_BUCKET=""

if [ "$_ARROW_INDEX" -eq 1 ]; then
  STORAGE_BACKEND="gcs"
  ask "GCS bucket name" "$PREV_STORAGE_BUCKET" STORAGE_BUCKET

  # Reuse the SA JSON from the Vertex step if present.
  if [ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]; then
    ask_path "path to GCS service-account JSON" GOOGLE_APPLICATION_CREDENTIALS \
             "$PREV_GOOGLE_APPLICATION_CREDENTIALS"
  else
    info "reusing the SA JSON from the LLM provider step"
  fi
  ok "GCS configured: $STORAGE_BUCKET"
else
  ok "using local filesystem storage"
fi


# ══════════════════════════════════════════════════════════════
# 6. Authentication
# ══════════════════════════════════════════════════════════════
header "authentication"

[ -n "$PREV_NEON_AUTH_URL" ] && _ARROW_DEFAULT=1 || _ARROW_DEFAULT=0
arrow_choice "auth mode" \
  "Open      — no authentication (single local user, default)" \
  "Neon Auth — multi-user, requires a Neon project"

NEON_AUTH_URL=""
if [ "$_ARROW_INDEX" -eq 1 ]; then
  ask "Neon Auth base URL" "$PREV_NEON_AUTH_URL" NEON_AUTH_URL
  ok "Neon Auth configured"
else
  ok "authentication disabled (single-user mode)"
fi


# ══════════════════════════════════════════════════════════════
# 7. Database
# ══════════════════════════════════════════════════════════════
header "database"

# Classify the previous DATABASE_URL so we can re-select the same option.
# The mapping is intentionally lossy — eg an unknown postgres URL maps
# to "external" (docker) or "local" (bare metal), which is the right
# fallback for any free-form connection string.
classify_db_url() {
  local url="$1" mode="$2"
  case "$url" in
    sqlite*) echo 0 ;;
    *)
      if [ "$mode" = "docker" ]; then
        case "$url" in
          *@postgres:*)         echo 1 ;;  # bundled
          *)                    echo 2 ;;  # external
        esac
      else
        case "$url" in
          *@localhost:5432/simd) echo 1 ;; # container we manage
          *.neon.tech*)          echo 2 ;; # Neon
          *)                     echo 3 ;; # local install / custom
        esac
      fi
      ;;
  esac
}

if [ "$DEPLOY_MODE" = "docker" ]; then
  _ARROW_DEFAULT=$(classify_db_url "$PREV_DATABASE_URL" docker)
  arrow_choice "database" \
    "SQLite   — single file, zero setup (recommended for local installs)" \
    "Postgres — bundled container ships with the stack" \
    "Postgres — external (Neon, RDS, …)"

  case "$_ARROW_INDEX" in
    0)
      # Bind-mount ~/.simd/ into the container so the SQLite file
      # survives container restarts and is visible to the host CLI.
      mkdir -p "$HOME/.simd"
      DATABASE_URL="sqlite+aiosqlite:////data/simd.db"
      ok "using SQLite at ~/.simd/simd.db (mounted into the container)"
      ;;
    1)
      DATABASE_URL="postgresql+asyncpg://simd:simd@postgres:5432/simd"
      ok "using bundled Postgres container"
      ;;
    2)
      ask "PostgreSQL connection URL (postgresql://user:pass@host/db)" \
          "$PREV_DATABASE_URL" DATABASE_URL
      DATABASE_URL="${DATABASE_URL/postgresql:\/\//postgresql+asyncpg:\/\/}"
      ok "external database configured"
      ;;
  esac
else
  _ARROW_DEFAULT=$(classify_db_url "$PREV_DATABASE_URL" bare-metal)
  arrow_choice "database" \
    "SQLite — single file at ~/.simd/simd.db (recommended, zero setup)" \
    "Postgres — run a container now (we'll do docker run for you)" \
    "Postgres — Neon (managed), paste the connection string" \
    "Postgres — local install, already running on this machine"

  case "$_ARROW_INDEX" in
    0)  # SQLite — no service to start
      mkdir -p "$HOME/.simd"
      DATABASE_URL="sqlite+aiosqlite:///$HOME/.simd/simd.db"
      ok "using SQLite at ~/.simd/simd.db"
      ;;
    1)  # Postgres container
      command -v docker >/dev/null || fail \
        "Docker isn't installed.  install Docker or pick another option."
      docker info >/dev/null 2>&1 || fail \
        "Docker daemon isn't running.  start Docker Desktop or pick another option."

      if docker ps -a --format '{{.Names}}' | grep -q '^simd-pg$'; then
        warn "container 'simd-pg' already exists — restarting it"
        docker start simd-pg >/dev/null 2>&1 || true
      else
        info "starting Postgres container 'simd-pg' on localhost:5432 …"
        docker run -d --name simd-pg \
          -e POSTGRES_USER=simd -e POSTGRES_PASSWORD=simd \
          -e POSTGRES_DB=simd -p 5432:5432 \
          postgres:16-alpine >/dev/null || fail "docker run failed"
      fi
      info "waiting for Postgres to accept connections …"
      for i in $(seq 1 20); do
        if docker exec simd-pg pg_isready -U simd >/dev/null 2>&1; then break; fi
        sleep 1
      done
      DATABASE_URL="postgresql+asyncpg://simd:simd@localhost:5432/simd"
      ok "Postgres container ready at $DATABASE_URL"
      ;;
    2)  # Neon
      ask "Neon connection URL (postgresql://user:pass@ep-xxx.neon.tech/db)" \
          "$PREV_DATABASE_URL" DATABASE_URL
      DATABASE_URL="${DATABASE_URL/postgresql:\/\//postgresql+asyncpg:\/\/}"
      ok "Neon database configured"
      ;;
    3)  # Local Postgres (already running)
      ask "PostgreSQL connection URL" \
          "${PREV_DATABASE_URL:-postgresql+asyncpg://simd:simd@localhost:5432/simd}" \
          DATABASE_URL
      DATABASE_URL="${DATABASE_URL/postgresql:\/\//postgresql+asyncpg:\/\/}"
      ok "using local Postgres at $DATABASE_URL"
      ;;
  esac
fi


# ══════════════════════════════════════════════════════════════
# 8. Write .env
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
# 9A. Docker deployment
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

  arrow_choice "start the stack now?" \
    "yes — run docker compose up -d" \
    "no  — I'll start it later"
  if [ "$_ARROW_INDEX" -eq 0 ]; then
    info "running:  $COMPOSE_CMD up -d"
    $COMPOSE_CMD up -d || fail "docker compose failed"
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

  # CLI config — the agent is in a container exposed on localhost:8000.
  # Runner mode is "remote" since SIM_SERVER_URL points outside the stack
  # (the bundled compose file doesn't yet ship a runner image).
  write_cli_config "http://localhost:8000" "local-docker" \
                   "$SIM_SERVER_URL" "remote"

# ══════════════════════════════════════════════════════════════
# 9B. Bare-metal deployment
# ══════════════════════════════════════════════════════════════
else
  header "bare-metal setup"

  command -v python3 >/dev/null || fail "python3 not found.  install Python 3.11+."
  PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
  PY_MAJOR="$(echo "$PY_VER" | cut -d. -f1)"
  PY_MINOR="$(echo "$PY_VER" | cut -d. -f2)"
  if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]; }; then
    fail "Python $PY_VER found, but 3.11+ is required."
  fi
  ok "Python $PY_VER"

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

  if [ "$STORAGE_BACKEND" = "local" ]; then
    mkdir -p "$AGENT_DIR/storage"
    mkdir -p "$AGENT_DIR/progress_data"
    ok "storage directories ready"
  fi

  header "CLI configuration"

  # The CLI needs the agent + runner URLs we already collected.  We
  # write them directly — no need to re-ask via `simd init`.
  #
  # For bare-metal mode the agent runs locally via `uvicorn …
  # --port 8000`.  The runner_mode is "remote" when SIM_SERVER_URL
  # doesn't point at localhost; "local-bare-metal" when it does.
  AGENT_URL_FOR_CLI="http://localhost:8000"
  case "$SIM_SERVER_URL" in
    http://localhost:*|http://127.0.0.1:*) RUNNER_MODE_FOR_CLI="local-bare-metal" ;;
    *)                                     RUNNER_MODE_FOR_CLI="remote" ;;
  esac
  write_cli_config "$AGENT_URL_FOR_CLI" "local-bare-metal" \
                   "$SIM_SERVER_URL" "$RUNNER_MODE_FOR_CLI"

  header "setup complete"

  cat <<EOF

  next steps — two terminals:

    # terminal 1 — start the agent (keeps running)
    cd $AGENT_DIR
    source .venv/bin/activate
    uvicorn simd_agent.main:app --port 8000

    # terminal 2 — run an example (pick one)
    cd $AGENT_DIR
    source .venv/bin/activate
    simd run examples/u-shape-pipe/prompt.txt     examples/u-shape-pipe/mesh/u-shape-pipe.msh
    simd run examples/z-bend/prompt.txt           examples/z-bend/mesh/z-bend.msh
    simd run examples/inner-outer-pipe/prompt.txt examples/inner-outer-pipe/mesh/inner-outer-pipe.msh
    simd run examples/cylindrical-cht/prompt.txt  examples/cylindrical-cht/mesh/cylindrical-cht.msh

  to see what's wired up:
    simd status

  to deactivate the venv when you're done:
    deactivate

EOF
fi


# ── done ────────────────────────────────────────────────────────
printf "${GREEN}installation complete.${NC}\n"
echo
