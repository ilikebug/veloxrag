"""Safe HTTP schemas for index generation administration."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator


class _Schema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        hide_input_in_errors=True,
    )


class _RequestSchema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
    )


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("timestamp must use UTC")
    return value


class IndexGenerationCreate(_RequestSchema):
    embedding_profile_id: UUID
    distance: Literal["cosine", "dot", "euclid", "manhattan"] = "cosine"


class SafeIndexGeneration(_Schema):
    id: UUID
    knowledge_base_id: UUID
    embedding_profile_id: UUID
    status: Literal["building", "active", "retiring", "failed"]
    distance: Literal["cosine", "dot", "euclid", "manhattan"]
    created_at: datetime
    validated_at: datetime | None
    activated_at: datetime | None

    _validate_created_at = field_validator("created_at")(_utc)
    _validate_validated_at = field_validator("validated_at")(
        lambda value: None if value is None else _utc(value)
    )
    _validate_activated_at = field_validator("activated_at")(
        lambda value: None if value is None else _utc(value)
    )


class IndexGenerationPage(_Schema):
    items: tuple[SafeIndexGeneration, ...]
    next_cursor: str | None = None


__all__ = ["IndexGenerationCreate", "IndexGenerationPage", "SafeIndexGeneration"]
