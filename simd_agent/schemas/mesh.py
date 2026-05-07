# simd_agent/schemas/mesh.py
"""Mesh info and patch config request/response schemas."""

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class MeshInfoUpsert(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)

    mesh_id: str
    file_name: str | None = None
    patches: list[dict[str, Any]] | None = None
    viewer_artifacts: dict[str, Any] | None = None
    check_mesh: dict[str, Any] | None = None


class MeshInfoOut(BaseModel):
    simulation_id: UUID
    mesh_id: str
    file_name: str | None
    uploaded_at: str
    patches: list[dict[str, Any]] | None
    viewer_artifacts: dict[str, Any] | None
    check_mesh: dict[str, Any] | None


class PatchConfigItem(BaseModel):
    patch_name: str
    patch_class: str | None = None
    patch_config: dict[str, Any] | None = None
    patch_info: dict[str, Any] | None = None
    boundary_hint: dict[str, Any] | None = None
    status: str = "needs_config"


class PatchConfigOut(BaseModel):
    id: UUID
    simulation_id: UUID
    patch_name: str
    patch_class: str | None
    patch_config: dict[str, Any] | None
    patch_info: dict[str, Any] | None
    boundary_hint: dict[str, Any] | None
    status: str


class PatchConfigsBatchUpsert(BaseModel):
    patches: list[PatchConfigItem]
