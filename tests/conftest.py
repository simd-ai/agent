# tests/conftest.py
"""Pytest configuration and fixtures."""

import asyncio
import os
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi import WebSocket

# Set test environment variables before importing app modules
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test")
os.environ.setdefault("SIMULATION_SERVER_URL", "http://localhost:9999")


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_websocket() -> MagicMock:
    """Create a mock WebSocket for testing."""
    ws = MagicMock(spec=WebSocket)
    ws.send_json = AsyncMock()
    ws.receive_json = AsyncMock()
    ws.accept = AsyncMock()
    ws.close = AsyncMock()
    return ws


@pytest.fixture
def run_id():
    """Generate a test run ID."""
    return uuid4()


@pytest.fixture
def sample_simulation_config() -> dict:
    """Sample simulation configuration for testing (complete V1 format)."""
    return {
        "mesh": {
            "mesh_id": "test-mesh-123",
            "patches": [
                {"name": "inlet", "type": "patch", "n_faces": 100},
                {"name": "outlet", "type": "patch", "n_faces": 100},
                {"name": "wall", "type": "wall", "n_faces": 5000},
            ],
        },
        "geometry": {
            "type": "pipe",
            "diameter": 0.1,
            "length": 1.0,
        },
        "fluid": {
            "viscosity": 1e-6,
            "density": 1000.0,
        },
        "boundary_conditions": {
            "inlet": {
                "patch_type": "inlet",
                "velocity": {"type": "fixedValue", "value": [1.0, 0, 0]},
            },
            "outlet": {
                "patch_type": "outlet",
                "pressure": {"type": "fixedValue", "value": 0},
            },
            "wall": {
                "patch_type": "wall",
            },
        },
    }


@pytest.fixture
def sample_requirements() -> str:
    """Sample user requirements for testing."""
    return "Simulate flow through a pipe with diameter 0.1m and length 1m. Inlet velocity is 1 m/s. Use water as the fluid."


@pytest.fixture
def turbulent_config() -> dict:
    """Configuration for turbulent flow testing."""
    return {
        "mesh": {"mesh_id": "turbulent-test"},
        "geometry": {
            "type": "pipe",
            "diameter": 0.1,
            "length": 1.0,
        },
        "inlet": {
            "velocity": 10.0,  # High velocity for turbulent flow
        },
        "fluid": {
            "viscosity": 1e-6,
            "density": 1000.0,
        },
        "boundary_conditions": {
            "inlet": {
                "patch_type": "inlet",
                "velocity": {"value": [10.0, 0, 0]},
            },
            "outlet": {
                "patch_type": "outlet",
            },
        },
    }


@pytest.fixture
def laminar_config() -> dict:
    """Configuration for laminar flow testing."""
    return {
        "mesh": {"mesh_id": "laminar-test"},
        "geometry": {
            "type": "pipe",
            "diameter": 0.01,  # Small pipe
            "length": 0.1,
        },
        "inlet": {
            "velocity": 0.01,  # Low velocity
        },
        "fluid": {
            "viscosity": 1e-3,  # High viscosity (honey-like)
            "density": 1000.0,
        },
        "boundary_conditions": {
            "inlet": {
                "patch_type": "inlet",
                "velocity": {"value": [0.01, 0, 0]},
            },
            "outlet": {
                "patch_type": "outlet",
            },
        },
    }


