from __future__ import annotations

# simd_agent/services/user_service.py
"""User business logic."""

from uuid import UUID

from simd_agent.repositories.user_repo import UserRepository
from simd_agent.schemas.user import (
    UserCreate, UserOut, UserUpdate, UserCreateResponse,
    UsageOut, UsageLimits,
)
from simd_agent.settings import get_settings


class UserService:
    def __init__(self, repo: UserRepository):
        self.repo = repo

    async def get_or_create(self, body: UserCreate) -> UserCreateResponse:
        existing = await self.repo.get_by_email(body.email)
        if existing:
            return UserCreateResponse(user=UserOut(**existing), is_new=False)

        created = await self.repo.create({"email": body.email})

        from simd_agent.telemetry import get_telemetry, UserSignedUp
        get_telemetry().capture(UserSignedUp(), user_id=str(created["id"]))

        return UserCreateResponse(user=UserOut(**created), is_new=True)

    async def get(self, user_id: UUID) -> UserOut | None:
        row = await self.repo.get_by_id(user_id)
        return UserOut(**row) if row else None

    async def update(self, user_id: UUID, body: UserUpdate) -> UserOut | None:
        data = body.model_dump(exclude_none=True)
        row = await self.repo.update(user_id, data)
        return UserOut(**row) if row else None

    async def get_by_email(self, email: str) -> UserOut | None:
        row = await self.repo.get_by_email(email)
        return UserOut(**row) if row else None

    async def get_by_stripe_customer_id(self, customer_id: str) -> UserOut | None:
        row = await self.repo.get_by_stripe_customer_id(customer_id)
        return UserOut(**row) if row else None

    def _is_local_mode(self) -> bool:
        """Local/self-hosted mode has no auth — no usage limits apply."""
        return not get_settings().neon_auth_base_url

    async def get_usage(
        self,
        user_id: UUID,
        simulation_id: UUID | None = None,
    ) -> UsageOut:
        """Return usage info for a user.

        When ``simulation_id`` is provided, ``run_count``/``can_start_run`` are
        scoped to that single project (each project gets its own run budget).
        When omitted, ``run_count`` is the total across all of the user's
        projects (used by the UI account/usage screen).
        """
        user = await self.repo.get_by_id(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")

        project_count = await self.repo.count_projects(user_id)
        if simulation_id is not None:
            run_count = await self.repo.count_runs_for_simulation(simulation_id)
        else:
            run_count = await self.repo.count_runs(user_id)

        settings = get_settings()
        max_projects = settings.free_max_projects
        max_runs = settings.free_max_runs

        # Local mode: no limits
        is_pro = (
            self._is_local_mode()
            or user.get("subscription_status") in ("active", "past_due")
        )

        if is_pro:
            limits = UsageLimits(max_projects=999999, max_runs=999999)
            can_create = True
            can_run = True
            projects_remaining = 999999
            runs_remaining = 999999
        else:
            limits = UsageLimits(max_projects=max_projects, max_runs=max_runs)
            can_create = project_count < max_projects
            can_run = run_count < max_runs
            projects_remaining = max(0, max_projects - project_count)
            runs_remaining = max(0, max_runs - run_count)

        return UsageOut(
            project_count=project_count,
            run_count=run_count,
            limits=limits,
            is_pro=is_pro,
            can_create_project=can_create,
            can_start_run=can_run,
            projects_remaining=projects_remaining,
            runs_remaining=runs_remaining,
        )
