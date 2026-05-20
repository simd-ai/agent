# ── Stage 1: Builder ─────────────────────────────────────────
FROM python:3.13-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies into a venv so we can copy it cleanly
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# pyproject.toml's ``readme = {file = "README", …}`` makes hatchling
# read the file at metadata-generation time — even for `-e .`.  Copy
# it (plus LICENSE for completeness) alongside pyproject.toml so the
# editable install doesn't fail with "Readme file does not exist".
# These two rarely change so the layer cache stays warm.
COPY pyproject.toml README LICENSE ./
# Install deps first (cached layer — only rebuilds when pyproject.toml changes)
RUN pip install --no-cache-dir -e .

# Copy source code
COPY simd_agent/ ./simd_agent/

# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy source (prompts + solvers are inside simd_agent/)
COPY --from=builder /app/simd_agent ./simd_agent
COPY --from=builder /app/pyproject.toml ./

# Local storage directory (mounted as volume in production)
RUN mkdir -p /app/storage /tmp/simd_progress

# Default env vars (overridden by .env or docker-compose)
ENV STORAGE_BACKEND=local \
    STORAGE_LOCAL_DIR=/app/storage \
    PROGRESS_DATA_DIR=/tmp/simd_progress \
    LOG_LEVEL=INFO

EXPOSE 8000

CMD ["uvicorn", "simd_agent.main:app", "--host", "0.0.0.0", "--port", "8000"]
