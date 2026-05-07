# simd_agent/schemas/user.py
"""User request/response schemas."""

from uuid import UUID

from pydantic import BaseModel


class UserCreate(BaseModel):
    email: str


class UserOut(BaseModel):
    id: UUID
    email: str
    created_at: str
    stripe_customer_id: str | None = None
    subscription_status: str
    subscription_current_period_end: str | None = None


class UserUpdate(BaseModel):
    stripe_customer_id: str | None = None
    subscription_status: str | None = None
    subscription_current_period_end: str | None = None


class UserCreateResponse(BaseModel):
    user: UserOut
    is_new: bool


# ── Tier limits (defaults — env-configurable via settings) ──────────────
FREE_MAX_PROJECTS = 10
FREE_MAX_RUNS = 20


class UsageLimits(BaseModel):
    max_projects: int
    max_runs: int


class UsageOut(BaseModel):
    project_count: int
    run_count: int
    limits: UsageLimits
    is_pro: bool
    can_create_project: bool
    can_start_run: bool
    projects_remaining: int
    runs_remaining: int
