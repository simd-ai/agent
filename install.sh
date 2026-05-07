#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════
# SIMD Agent — Interactive Installer
# ══════════════════════════════════════════════════════════════
# Sets up the full SIMD Agent stack:
#   - Backend  (simd-ai/agent)
#   - Frontend (simd-ai/ui)
#   - Runner   (OpenFOAM simulation server)
#   - Postgres
#
# Supports two deployment modes:
#   1. Docker  — docker compose up (recommended)
#   2. Bare metal — Python venv + Node.js + local Postgres
# ══════════════════════════════════════════════════════════════

set -euo pipefail

# ── Colors ───────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }
header()  { echo -e "\n${BOLD}${CYAN}── $* ──${NC}\n"; }

# ── Resolve paths ────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="$SCRIPT_DIR"
UI_DIR="$AGENT_DIR/../ui"
SIM_SERVER_DIR="$AGENT_DIR/../simulation_server"
ENV_FILE="$AGENT_DIR/.env"

# ══════════════════════════════════════════════════════════════
# Helper: prompt with default
# ══════════════════════════════════════════════════════════════
ask() {
  local prompt="$1" default="$2" var="$3"
  if [ -n "$default" ]; then
    read -rp "$(echo -e "${BOLD}$prompt${NC} [$default]: ")" input
    eval "$var=\"${input:-$default}\""
  else
    read -rp "$(echo -e "${BOLD}$prompt${NC}: ")" input
    eval "$var=\"$input\""
  fi
}

ask_yes_no() {
  local prompt="$1" default="$2"
  local yn
  read -rp "$(echo -e "${BOLD}$prompt${NC} [${default}]: ")" yn
  yn="${yn:-$default}"
  [[ "$yn" =~ ^[Yy] ]]
}

# ══════════════════════════════════════════════════════════════
# 1. Welcome
# ══════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║          SIMD Agent — Installation Wizard           ║${NC}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════╝${NC}"
echo ""
echo "This script will set up the SIMD Agent stack."
echo "You'll be asked a few questions to configure your deployment."
echo ""

# ══════════════════════════════════════════════════════════════
# 2. Gather configuration
# ══════════════════════════════════════════════════════════════
header "LLM Provider"

GEMINI_API_KEY=""
while [ -z "$GEMINI_API_KEY" ]; do
  ask "Gemini API key (get one at https://aistudio.google.com/apikey)" "" GEMINI_API_KEY
  if [ -z "$GEMINI_API_KEY" ]; then
    error "A Gemini API key is required."
  fi
done
success "API key set"

# ── Storage ──────────────────────────────────────────────────
header "Object Storage"

echo "Meshes, simulation results, and case files are stored in object storage."
echo ""
echo "  1) Local filesystem (default — no setup needed)"
echo "  2) Google Cloud Storage (requires a GCS bucket + service account)"
echo ""

STORAGE_CHOICE=""
while [[ ! "$STORAGE_CHOICE" =~ ^[12]$ ]]; do
  ask "Choose storage backend" "1" STORAGE_CHOICE
done

STORAGE_BACKEND="local"
STORAGE_BUCKET=""
GCS_KEY_PATH=""

if [ "$STORAGE_CHOICE" = "2" ]; then
  STORAGE_BACKEND="gcs"
  ask "GCS bucket name" "" STORAGE_BUCKET
  ask "Path to GCS service account JSON key" "" GCS_KEY_PATH

  if [ ! -f "$GCS_KEY_PATH" ]; then
    error "File not found: $GCS_KEY_PATH"
    exit 1
  fi
  success "GCS configured: bucket=$STORAGE_BUCKET"
else
  success "Using local filesystem storage"
fi

# ── Authentication ───────────────────────────────────────────
header "Authentication"

echo "By default, SIMD Agent runs without authentication (single local user)."
echo "You can enable Neon Auth for multi-user support."
echo ""
echo "  1) Local (no authentication — default)"
echo "  2) Neon Auth (requires a Neon project)"
echo ""

AUTH_CHOICE=""
while [[ ! "$AUTH_CHOICE" =~ ^[12]$ ]]; do
  ask "Choose authentication mode" "1" AUTH_CHOICE
done

AUTH_DISABLED="true"
NEON_AUTH_URL=""

if [ "$AUTH_CHOICE" = "2" ]; then
  AUTH_DISABLED="false"
  ask "Neon Auth base URL (e.g. https://ep-xxx.neonauth.us-east-1.aws.neon.tech)" "" NEON_AUTH_URL
  success "Neon Auth configured"
else
  success "Authentication disabled (single-user mode)"
fi

# ── Database ─────────────────────────────────────────────────
header "Database"

echo "SIMD Agent uses PostgreSQL for run history and events."
echo ""
echo "  1) Bundled Postgres (Docker only — a Postgres container is included)"
echo "  2) External Postgres (provide your own connection string)"
echo ""

DB_CHOICE=""
while [[ ! "$DB_CHOICE" =~ ^[12]$ ]]; do
  ask "Choose database" "1" DB_CHOICE
done

DATABASE_URL="postgresql://simd:simd@postgres:5432/simd"
DATABASE_URL_SYNC=""

if [ "$DB_CHOICE" = "2" ]; then
  ask "PostgreSQL connection URL (postgresql://user:pass@host:5432/dbname)" "" DATABASE_URL
  success "External database configured"
else
  success "Using bundled Postgres container"
fi

# ── Deployment ───────────────────────────────────────────────
header "Deployment Mode"

echo "  1) Docker (recommended — everything runs in containers)"
echo "  2) Bare metal (Python venv + Node.js on your machine)"
echo ""

DEPLOY_CHOICE=""
while [[ ! "$DEPLOY_CHOICE" =~ ^[12]$ ]]; do
  ask "Choose deployment mode" "1" DEPLOY_CHOICE
done

if [ "$DEPLOY_CHOICE" = "1" ]; then
  DEPLOY_MODE="docker"
  success "Docker deployment selected"
else
  DEPLOY_MODE="bare-metal"
  success "Bare metal deployment selected"
fi

# ══════════════════════════════════════════════════════════════
# 3. Clone frontend repo if not present
# ══════════════════════════════════════════════════════════════
header "Frontend (simd-ai/ui)"

if [ -d "$UI_DIR" ]; then
  info "Frontend repo already exists at $UI_DIR"
else
  info "Cloning simd-ai/ui into $UI_DIR ..."
  git clone https://github.com/simd-ai/ui.git "$UI_DIR"
  success "Frontend cloned"
fi

# ══════════════════════════════════════════════════════════════
# 4. Generate .env
# ══════════════════════════════════════════════════════════════
header "Generating .env"

if [ -f "$ENV_FILE" ]; then
  warn ".env already exists — backing up to .env.bak"
  cp "$ENV_FILE" "$ENV_FILE.bak"
fi

# Build the async DATABASE_URL for the agent (asyncpg)
AGENT_DATABASE_URL="$DATABASE_URL"
if [ "$DB_CHOICE" = "1" ]; then
  # Bundled Postgres: agent uses asyncpg driver inside Docker network
  AGENT_DATABASE_URL="postgresql+asyncpg://simd:simd@postgres:5432/simd"
  # Frontend (Next.js) uses sync driver
  DATABASE_URL_SYNC="postgresql://simd:simd@postgres:5432/simd"
elif [[ "$DATABASE_URL" == postgresql://* ]]; then
  # External: convert to asyncpg for the agent
  AGENT_DATABASE_URL="${DATABASE_URL/postgresql:\/\//postgresql+asyncpg:\/\/}"
  DATABASE_URL_SYNC="$DATABASE_URL"
else
  DATABASE_URL_SYNC="$DATABASE_URL"
fi

# Simulation server URL
if [ "$DEPLOY_MODE" = "docker" ]; then
  SIM_SERVER_URL="http://runner:8000"
else
  SIM_SERVER_URL="http://localhost:8001"
fi

cat > "$ENV_FILE" << ENVEOF
# ══════════════════════════════════════════════════════════════
# SIMD Agent — Generated by install.sh
# ══════════════════════════════════════════════════════════════

# ── Database ─────────────────────────────────────────────────
DATABASE_URL=$AGENT_DATABASE_URL

# ── LLM Provider ────────────────────────────────────────────
GEMINI_API_KEY=$GEMINI_API_KEY

# ── Simulation Server ───────────────────────────────────────
SIMULATION_SERVER_URL=$SIM_SERVER_URL

# ── Object Storage ──────────────────────────────────────────
STORAGE_BACKEND=$STORAGE_BACKEND
ENVEOF

if [ "$STORAGE_BACKEND" = "local" ]; then
  if [ "$DEPLOY_MODE" = "docker" ]; then
    echo "STORAGE_LOCAL_DIR=/app/storage" >> "$ENV_FILE"
  else
    echo "STORAGE_LOCAL_DIR=$AGENT_DIR/storage" >> "$ENV_FILE"
  fi
else
  cat >> "$ENV_FILE" << GCSEOF
STORAGE_BUCKET=$STORAGE_BUCKET
GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcs-key.json
GCSEOF
fi

cat >> "$ENV_FILE" << ENVEOF2

# ── Progress Data ───────────────────────────────────────────
PROGRESS_DATA_DIR=/tmp/simd_progress

# ── Authentication ──────────────────────────────────────────
ENVEOF2

if [ "$AUTH_DISABLED" = "true" ]; then
  echo "# Authentication disabled (single-user mode)" >> "$ENV_FILE"
else
  echo "NEON_AUTH_BASE_URL=$NEON_AUTH_URL" >> "$ENV_FILE"
fi

cat >> "$ENV_FILE" << ENVEOF3

# ── Frontend ────────────────────────────────────────────────
NEXT_PUBLIC_AGENT_URL=http://localhost:8000
NEXT_PUBLIC_AGENT_WS_URL=ws://localhost:8000
NEXT_PUBLIC_AUTH_DISABLED=$AUTH_DISABLED

# ── Logging ─────────────────────────────────────────────────
LOG_LEVEL=INFO
MAX_RETRIES=3
ENVEOF3

success ".env generated at $ENV_FILE"

# ══════════════════════════════════════════════════════════════
# 5A. Docker Deployment
# ══════════════════════════════════════════════════════════════
if [ "$DEPLOY_MODE" = "docker" ]; then
  header "Docker Deployment"

  # Check Docker is available
  if ! command -v docker &> /dev/null; then
    error "Docker is not installed. Please install Docker Desktop or Docker Engine."
    error "  https://docs.docker.com/get-docker/"
    exit 1
  fi

  if ! docker info &> /dev/null; then
    error "Docker daemon is not running. Please start Docker and try again."
    exit 1
  fi

  success "Docker is available"

  # Handle GCS key mount
  COMPOSE_CMD="docker compose -f docker/docker-compose.yml"

  if [ "$STORAGE_BACKEND" = "gcs" ]; then
    # Create a compose override for GCS key mount
    cat > "$AGENT_DIR/docker/docker-compose.gcs.yml" << GCSYML
services:
  agent:
    volumes:
      - $GCS_KEY_PATH:/secrets/gcs-key.json:ro
GCSYML
    COMPOSE_CMD="$COMPOSE_CMD -f docker/docker-compose.gcs.yml"
  fi

  info "Building and starting containers..."
  echo ""
  echo -e "  ${CYAN}$COMPOSE_CMD up -d --build${NC}"
  echo ""

  if ask_yes_no "Start the stack now?" "Y"; then
    cd "$AGENT_DIR"
    $COMPOSE_CMD up -d --build

    echo ""
    success "SIMD Agent is running!"
    echo ""
    echo -e "  ${BOLD}Frontend:${NC}  http://localhost:3000"
    echo -e "  ${BOLD}Backend:${NC}   http://localhost:8000"
    echo -e "  ${BOLD}Runner:${NC}    http://localhost:8001"
    echo -e "  ${BOLD}Postgres:${NC}  localhost:5432"
    echo ""
    echo -e "  ${CYAN}View logs:${NC}     $COMPOSE_CMD logs -f"
    echo -e "  ${CYAN}Stop:${NC}          $COMPOSE_CMD down"
    echo -e "  ${CYAN}Restart:${NC}       $COMPOSE_CMD restart"
    echo ""
  else
    echo ""
    info "To start later, run:"
    echo -e "  ${CYAN}cd $AGENT_DIR && $COMPOSE_CMD up -d --build${NC}"
    echo ""
  fi

# ══════════════════════════════════════════════════════════════
# 5B. Bare Metal Deployment
# ══════════════════════════════════════════════════════════════
else
  header "Bare Metal Setup"

  # ── Check prerequisites ────────────────────────────────────
  MISSING=()

  if ! command -v python3 &> /dev/null; then
    MISSING+=("python3 (3.11+)")
  fi

  if ! command -v node &> /dev/null; then
    MISSING+=("node (20+)")
  fi

  if ! command -v npm &> /dev/null; then
    MISSING+=("npm")
  fi

  if [ ${#MISSING[@]} -gt 0 ]; then
    error "Missing prerequisites:"
    for dep in "${MISSING[@]}"; do
      echo -e "  ${RED}-${NC} $dep"
    done
    exit 1
  fi

  success "Python $(python3 --version 2>&1 | awk '{print $2}'), Node $(node --version)"

  # Update .env for bare-metal paths
  sed -i.bak "s|PROGRESS_DATA_DIR=/tmp/simd_progress|PROGRESS_DATA_DIR=$AGENT_DIR/progress_data|g" "$ENV_FILE"
  rm -f "$ENV_FILE.bak"

  if [ "$STORAGE_BACKEND" = "local" ]; then
    mkdir -p "$AGENT_DIR/storage"
  fi
  mkdir -p "$AGENT_DIR/progress_data"

  # ── Backend (Python venv) ──────────────────────────────────
  header "Backend Setup"

  info "Creating Python virtual environment..."
  cd "$AGENT_DIR"

  if [ ! -d "venv" ]; then
    python3 -m venv venv
  fi
  source venv/bin/activate
  success "Virtual environment ready"

  info "Installing backend dependencies..."
  pip install -e ".[dev]" --quiet
  success "Backend dependencies installed"

  # ── Frontend (npm) ─────────────────────────────────────────
  header "Frontend Setup"

  cd "$UI_DIR"
  info "Installing frontend dependencies..."
  npm ci --silent
  success "Frontend dependencies installed"

  # ── Database setup ─────────────────────────────────────────
  if [ "$DB_CHOICE" = "1" ]; then
    warn "You selected bundled Postgres, but bare metal mode requires you to run Postgres yourself."
    echo ""
    echo "  Option A: Install Postgres locally"
    echo "    brew install postgresql@16   # macOS"
    echo "    sudo apt install postgresql  # Ubuntu/Debian"
    echo ""
    echo "  Option B: Run just Postgres in Docker"
    echo -e "    ${CYAN}docker run -d --name simd-postgres \\"
    echo "      -e POSTGRES_USER=simd \\"
    echo "      -e POSTGRES_PASSWORD=simd \\"
    echo "      -e POSTGRES_DB=simd \\"
    echo -e "      -p 5432:5432 postgres:16-alpine${NC}"
    echo ""

    # Update DATABASE_URL for localhost
    sed -i.bak "s|postgresql+asyncpg://simd:simd@postgres:5432/simd|postgresql+asyncpg://simd:simd@localhost:5432/simd|g" "$ENV_FILE"
    rm -f "$ENV_FILE.bak"
  fi

  # ── GCS key path (bare metal) ──────────────────────────────
  if [ "$STORAGE_BACKEND" = "gcs" ]; then
    sed -i.bak "s|GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcs-key.json|GOOGLE_APPLICATION_CREDENTIALS=$GCS_KEY_PATH|g" "$ENV_FILE"
    rm -f "$ENV_FILE.bak"
  fi

  # ── Print start commands ───────────────────────────────────
  header "Setup Complete"

  echo -e "${GREEN}SIMD Agent is ready!${NC} Start each service in a separate terminal:"
  echo ""
  echo -e "  ${BOLD}1. Postgres${NC} (if using Docker for just the DB):"
  echo -e "     ${CYAN}docker run -d --name simd-postgres \\"
  echo "       -e POSTGRES_USER=simd -e POSTGRES_PASSWORD=simd -e POSTGRES_DB=simd \\"
  echo -e "       -p 5432:5432 postgres:16-alpine${NC}"
  echo ""
  echo -e "  ${BOLD}2. Backend${NC}:"
  echo -e "     ${CYAN}cd $AGENT_DIR${NC}"
  echo -e "     ${CYAN}source venv/bin/activate${NC}"
  echo -e "     ${CYAN}uvicorn simd_agent.main:app --reload --port 8000${NC}"
  echo ""
  echo -e "  ${BOLD}3. Frontend${NC}:"
  echo -e "     ${CYAN}cd $UI_DIR${NC}"
  echo -e "     ${CYAN}npm run dev${NC}"
  echo ""
  echo -e "  ${BOLD}4. OpenFOAM Runner${NC} (optional — only if running simulations):"
  echo -e "     ${CYAN}cd $SIM_SERVER_DIR${NC}"
  echo -e "     ${CYAN}python3 -m venv venv && source venv/bin/activate${NC}"
  echo -e "     ${CYAN}pip install -r requirements.txt${NC}"
  echo -e "     ${CYAN}uvicorn app.main:app --port 8001${NC}"
  echo ""
  echo -e "  ${BOLD}Open in browser:${NC} http://localhost:3000"
  echo ""
fi

echo -e "${BOLD}${CYAN}════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${CYAN}  Installation complete! Enjoy SIMD Agent.             ${NC}"
echo -e "${BOLD}${CYAN}════════════════════════════════════════════════════════${NC}"
echo ""
