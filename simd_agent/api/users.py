# simd_agent/api/users.py
"""User endpoints."""

from uuid import UUID

from fastapi import APIRouter, HTTPException

from simd_agent.schemas.user import UserCreate, UserCreateResponse, UserOut, UserUpdate, UsageOut
from simd_agent.services import user_service

router = APIRouter(prefix="/api/users", tags=["users"])


@router.post("")
async def get_or_create_user(body: UserCreate) -> UserCreateResponse:
    return await user_service.get_or_create(body)


@router.get("/{user_id}")
async def get_user(user_id: UUID) -> UserOut:
    user = await user_service.get(user_id)
    if not user:
        raise HTTPException(404, f"User {user_id} not found")
    return user


@router.patch("/{user_id}")
async def update_user(user_id: UUID, body: UserUpdate) -> UserOut:
    user = await user_service.update(user_id, body)
    if not user:
        raise HTTPException(404, f"User {user_id} not found")
    return user


@router.get("/{user_id}/usage")
async def get_user_usage(
    user_id: UUID,
    simulation_id: UUID | None = None,
) -> UsageOut:
    """Return usage. Pass ?simulation_id=... to scope run_count to one project."""
    try:
        return await user_service.get_usage(user_id, simulation_id=simulation_id)
    except ValueError:
        raise HTTPException(404, f"User {user_id} not found")


@router.get("/by-stripe-id/{customer_id}")
async def get_user_by_stripe_id(customer_id: str) -> UserOut:
    user = await user_service.get_by_stripe_customer_id(customer_id)
    if not user:
        raise HTTPException(404, f"No user with stripe_customer_id={customer_id}")
    return user
