"""Mesh converter configuration."""

import os
from pathlib import Path

# Temp directory for conversion work (cleaned up after each request)
TMP_DIR = Path(os.getenv("MESH_TMP_DIR", "./tmp")).resolve()
TMP_DIR.mkdir(parents=True, exist_ok=True)
