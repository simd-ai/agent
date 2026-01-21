"""Mesh converter configuration."""

import os
from pathlib import Path

# Directories
STORAGE_DIR = Path(os.getenv("MESH_STORAGE_DIR", "./storage")).resolve()
TMP_DIR = Path(os.getenv("MESH_TMP_DIR", "./tmp")).resolve()

# Public URL for serving static files
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000").rstrip("/")

# Ensure directories exist
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)
