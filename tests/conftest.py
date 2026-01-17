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
os.environ.setdefault("SANDBOX_BASE_URL", "http://localhost:9999")


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
    """Sample simulation configuration for testing."""
    return {
        "geometry": {
            "type": "pipe",
            "diameter": 0.1,
            "length": 1.0,
        },
        "inlet": {
            "velocity": 1.0,
        },
        "fluid": {
            "viscosity": 1e-6,
            "density": 1000.0,
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
    }


@pytest.fixture
def laminar_config() -> dict:
    """Configuration for laminar flow testing."""
    return {
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
    }


@pytest.fixture
def mock_sandbox_client():
    """Create a mock sandbox client."""
    from simd_agent.models import SandboxArtifact, SandboxState, SandboxStatus
    from simd_agent.sandbox_client import SandboxClient
    
    client = MagicMock(spec=SandboxClient)
    
    # Mock successful submission
    client.submit_run = AsyncMock(return_value=MagicMock(run_id="sandbox-123"))
    
    # Mock status polling
    client.get_status = AsyncMock(return_value=SandboxStatus(
        state=SandboxState.SUCCEEDED,
        exit_code=0,
    ))
    
    # Mock logs
    client.get_logs = AsyncMock(return_value="Simulation completed successfully\n")
    
    # Mock artifacts
    client.get_artifacts = AsyncMock(return_value=MagicMock(
        artifacts=[
            SandboxArtifact(
                name="results.zip",
                path="/results/results.zip",
                size_bytes=1024,
            )
        ]
    ))
    
    # Mock wait for completion
    client.wait_for_completion = AsyncMock(return_value=SandboxStatus(
        state=SandboxState.SUCCEEDED,
        exit_code=0,
    ))
    
    client.close = AsyncMock()
    
    return client


@pytest.fixture
def mock_failing_sandbox_client():
    """Create a mock sandbox client that simulates failures."""
    from simd_agent.models import SandboxState, SandboxStatus
    from simd_agent.sandbox_client import SandboxClient
    
    client = MagicMock(spec=SandboxClient)
    
    client.submit_run = AsyncMock(return_value=MagicMock(run_id="sandbox-fail-123"))
    
    client.get_status = AsyncMock(return_value=SandboxStatus(
        state=SandboxState.FAILED,
        exit_code=1,
    ))
    
    client.get_logs = AsyncMock(return_value="""
--> FOAM FATAL ERROR:
    Cannot find patchField entry for inlet

    file: /case/0/U
    line: 25
""")
    
    client.wait_for_completion = AsyncMock(return_value=SandboxStatus(
        state=SandboxState.FAILED,
        exit_code=1,
    ))
    
    client.close = AsyncMock()
    
    return client
