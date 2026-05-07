# simd_agent/api/projects.py
"""CRUD endpoints for projects.

Projects group simulations into workspaces.  The frontend creates a
project per user workspace; simulations and meshes belong to a project.
"""

import logging
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text

from simd_agent.db import get_session

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/projects", tags=["projects"])


# ── Request / Response models ────────────────────────────────────────────


class ProjectCreate(BaseModel):
    name: str = Field(..., max_length=255)
    description: str = ""
    owner_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProjectUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    metadata: dict[str, Any] | None = None


class ProjectOut(BaseModel):
    id: UUID
    name: str
    description: str
    owner_id: str | None
    metadata: dict[str, Any]
    created_at: str
    updated_at: str


# ── Endpoints ────────────────────────────────────────────────────────────


@router.post("", status_code=201)
async def create_project(body: ProjectCreate) -> ProjectOut:
    """Create a new project."""
    async with get_session() as session:
        result = await session.execute(
            text("""
                INSERT INTO projects (name, description, owner_id, metadata)
                VALUES (:name, :description, :owner_id, :metadata)
                RETURNING id, name, description, owner_id, metadata,
                          created_at::text, updated_at::text
            """),
            {
                "name": body.name,
                "description": body.description,
                "owner_id": body.owner_id,
                "metadata": str(body.metadata) if body.metadata else "{}",
            },
        )
        row = result.mappings().one()
        return ProjectOut(**row)


@router.get("")
async def list_projects(owner_id: str | None = None) -> list[ProjectOut]:
    """List all projects, optionally filtered by owner."""
    async with get_session() as session:
        if owner_id:
            result = await session.execute(
                text("""
                    SELECT id, name, description, owner_id, metadata,
                           created_at::text, updated_at::text
                    FROM projects WHERE owner_id = :owner_id
                    ORDER BY updated_at DESC
                """),
                {"owner_id": owner_id},
            )
        else:
            result = await session.execute(
                text("""
                    SELECT id, name, description, owner_id, metadata,
                           created_at::text, updated_at::text
                    FROM projects ORDER BY updated_at DESC
                """),
            )
        return [ProjectOut(**row) for row in result.mappings().all()]


@router.get("/{project_id}")
async def get_project(project_id: UUID) -> ProjectOut:
    """Get a single project by ID."""
    async with get_session() as session:
        result = await session.execute(
            text("""
                SELECT id, name, description, owner_id, metadata,
                       created_at::text, updated_at::text
                FROM projects WHERE id = :id
            """),
            {"id": project_id},
        )
        row = result.mappings().one_or_none()
        if not row:
            raise HTTPException(404, f"Project {project_id} not found")
        return ProjectOut(**row)


@router.patch("/{project_id}")
async def update_project(project_id: UUID, body: ProjectUpdate) -> ProjectOut:
    """Update a project."""
    updates = body.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(400, "No fields to update")

    set_clauses = ", ".join(f"{k} = :{k}" for k in updates)
    updates["id"] = project_id
    updates["_now"] = "NOW()"

    async with get_session() as session:
        result = await session.execute(
            text(f"""
                UPDATE projects SET {set_clauses}, updated_at = NOW()
                WHERE id = :id
                RETURNING id, name, description, owner_id, metadata,
                          created_at::text, updated_at::text
            """),
            updates,
        )
        row = result.mappings().one_or_none()
        if not row:
            raise HTTPException(404, f"Project {project_id} not found")
        return ProjectOut(**row)


@router.delete("/{project_id}", status_code=204)
async def delete_project(project_id: UUID) -> None:
    """Delete a project and all its simulations/runs (cascade)."""
    async with get_session() as session:
        result = await session.execute(
            text("DELETE FROM projects WHERE id = :id RETURNING id"),
            {"id": project_id},
        )
        if not result.scalar():
            raise HTTPException(404, f"Project {project_id} not found")
