# simd_agent/services/__init__.py
"""Business logic layer — services orchestrate repositories.

Singleton service instances with their repository dependencies.
Import from here in API routes.
"""

from simd_agent.repositories.user_repo import UserRepository
from simd_agent.repositories.simulation_repo import SimulationRepository
from simd_agent.repositories.config_repo import ConfigRepository
from simd_agent.repositories.run_repo import RunRepository
from simd_agent.repositories.event_repo import EventRepository
from simd_agent.repositories.mesh_repo import MeshRepository
from simd_agent.repositories.patch_repo import PatchRepository
from simd_agent.repositories.chat_repo import ChatRepository
from simd_agent.repositories.precheck_repo import PrecheckRepository
from simd_agent.repositories.lint_repo import LintRepository
from simd_agent.repositories.progress_repo import ProgressRepository

from simd_agent.services.user_service import UserService
from simd_agent.services.simulation_service import SimulationService
from simd_agent.services.run_service import RunService
from simd_agent.services.snapshot_service import SnapshotService
from simd_agent.services.chat_service import ChatService

# ── Repository instances ─────────────────────────────────────────────────
user_repo = UserRepository()
simulation_repo = SimulationRepository()
config_repo = ConfigRepository()
run_repo = RunRepository()
event_repo = EventRepository()
mesh_repo = MeshRepository()
patch_repo = PatchRepository()
chat_repo = ChatRepository()
precheck_repo = PrecheckRepository()
lint_repo = LintRepository()
progress_repo = ProgressRepository()

# ── Service instances ────────────────────────────────────────────────────
user_service = UserService(user_repo)
simulation_service = SimulationService(simulation_repo, config_repo)
run_service = RunService(run_repo, event_repo, progress_repo)
snapshot_service = SnapshotService(
    simulation_repo, config_repo, mesh_repo, patch_repo,
    precheck_repo, lint_repo, run_repo, chat_repo,
)
chat_service = ChatService(chat_repo)
