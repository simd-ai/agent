from __future__ import annotations

# simd_agent/repositories/user_repo.py
"""User data access."""

from typing import Any
from uuid import UUID

from simd_agent.repositories.base import PostgresRepository


class UserRepository(PostgresRepository):
    table = "users"
    pk = "id"
    columns = [
        "id", "email", "created_at::text", "stripe_customer_id",
        "subscription_status", "subscription_current_period_end::text",
    ]

    async def get_by_email(self, email: str) -> dict[str, Any] | None:
        return await self.execute_raw_one(
            f"SELECT {self._select_cols} FROM {self.table} WHERE email = :email",
            {"email": email},
        )

    async def get_by_stripe_customer_id(self, customer_id: str) -> dict[str, Any] | None:
        return await self.execute_raw_one(
            f"SELECT {self._select_cols} FROM {self.table} WHERE stripe_customer_id = :cid",
            {"cid": customer_id},
        )

    async def count_projects(self, user_id: UUID) -> int:
        row = await self.execute_raw_one(
            "SELECT COUNT(*) AS cnt FROM simulations WHERE user_id = :uid",
            {"uid": user_id},
        )
        return row["cnt"] if row else 0

    async def count_runs(self, user_id: UUID) -> int:
        """Count CFD_CODEGEN_RUN runs (the ones that consume compute).

        INNER JOIN intentionally excludes runs with NULL simulation_id
        (orphaned runs can't be attributed to a user).
        """
        row = await self.execute_raw_one(
            "SELECT COUNT(*) AS cnt FROM runs r "
            "JOIN simulations s ON r.simulation_id = s.id "
            "WHERE s.user_id = :uid AND r.op = 'CFD_CODEGEN_RUN'",
            {"uid": user_id},
        )
        return row["cnt"] if row else 0
