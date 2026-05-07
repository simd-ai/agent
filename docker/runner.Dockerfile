# ── OpenFOAM Simulation Runner ───────────────────────────────
# Based on the official OpenFOAM 2406 image from openfoam.com
# Adds Python + FastAPI to serve the runner HTTP API.
#
# The simulation_server/ code is expected to be mounted or copied
# at build time from a local checkout.
# ─────────────────────────────────────────────────────────────

FROM opencfd/openfoam-default:2406

USER root

# Install Python 3 + pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create venv and install runner dependencies
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# OpenFOAM environment
ENV OPENFOAM_ENV=/usr/lib/openfoam/openfoam2406/etc/bashrc
ENV RUNS_DIR=/tmp/simd-runs

RUN mkdir -p /tmp/simd-runs

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
