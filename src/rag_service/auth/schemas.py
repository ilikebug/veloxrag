from datetime import UTC, datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StringConstraints,
    field_validator,
    model_validator,
)

from rag_service.auth.policies import Capability

ApiKeyName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=120),
]


class _Schema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
    )


def _require_aware_utc(value: datetime | None) -> datetime | None:
    if value is not None and (value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value)):
        raise ValueError("timestamp must use UTC")
    return value


class AdminApiKeyCreate(_Schema):
    name: ApiKeyName
    not_before: datetime | None = None
    expires_at: datetime | None = None

    _validate_not_before = field_validator("not_before")(_require_aware_utc)
    _validate_expires_at = field_validator("expires_at")(_require_aware_utc)

    @model_validator(mode="after")
    def validate_window(self) -> "AdminApiKeyCreate":
        if (
            self.not_before is not None
            and self.expires_at is not None
            and self.expires_at <= self.not_before
        ):
            raise ValueError("expires_at must be later than not_before")
        return self


class AgentApiKeyCreate(_Schema):
    name: ApiKeyName
    capabilities: frozenset[Capability] = Field(default_factory=frozenset, max_length=4)
    knowledge_base_ids: frozenset[UUID] = Field(default_factory=frozenset)
    query_profile_ids: frozenset[UUID] = Field(default_factory=frozenset)
    default_query_profile_id: UUID | None = None
    raw_file_read: bool = False
    requests_per_minute: int = Field(ge=1)
    max_concurrency: int = Field(ge=1)
    not_before: datetime | None = None
    expires_at: datetime | None = None

    _validate_not_before = field_validator("not_before")(_require_aware_utc)
    _validate_expires_at = field_validator("expires_at")(_require_aware_utc)

    @model_validator(mode="after")
    def validate_policy(self) -> "AgentApiKeyCreate":
        if (
            self.not_before is not None
            and self.expires_at is not None
            and self.expires_at <= self.not_before
        ):
            raise ValueError("expires_at must be later than not_before")
        if (
            self.default_query_profile_id is not None
            and self.default_query_profile_id not in self.query_profile_ids
        ):
            raise ValueError("default query profile must be in query profile scope")
        return self


class AgentApiKeyUpdate(_Schema):
    name: ApiKeyName | None = None
    status: Literal["active", "disabled"] | None = None
    capabilities: frozenset[Capability] | None = Field(default=None, max_length=4)
    knowledge_base_ids: frozenset[UUID] | None = None
    query_profile_ids: frozenset[UUID] | None = None
    default_query_profile_id: UUID | None = None
    raw_file_read: bool | None = None
    requests_per_minute: int | None = Field(default=None, ge=1)
    max_concurrency: int | None = Field(default=None, ge=1)
    not_before: datetime | None = None
    expires_at: datetime | None = None

    _validate_not_before = field_validator("not_before")(_require_aware_utc)
    _validate_expires_at = field_validator("expires_at")(_require_aware_utc)

    @model_validator(mode="after")
    def validate_patch(self) -> "AgentApiKeyUpdate":
        if not self.model_fields_set:
            raise ValueError("at least one field is required")
        nullable_fields = {"default_query_profile_id", "not_before", "expires_at"}
        for field_name in self.model_fields_set - nullable_fields:
            if getattr(self, field_name) is None:
                raise ValueError(f"{field_name} must not be null")
        if (
            "query_profile_ids" in self.model_fields_set
            and self.default_query_profile_id is not None
            and self.query_profile_ids is not None
            and self.default_query_profile_id not in self.query_profile_ids
        ):
            raise ValueError("default query profile must be in query profile scope")
        if (
            "not_before" in self.model_fields_set
            and "expires_at" in self.model_fields_set
            and self.not_before is not None
            and self.expires_at is not None
            and self.expires_at <= self.not_before
        ):
            raise ValueError("expires_at must be later than not_before")
        return self


class SafeApiKey(_Schema):
    id: UUID
    public_id: str = Field(min_length=16, max_length=64)
    name: str = Field(min_length=1, max_length=120)
    status: Literal["active", "disabled", "revoked"]
    key_type: Literal["admin", "agent"]
    capabilities: tuple[Capability, ...]
    knowledge_base_ids: tuple[UUID, ...]
    query_profile_ids: tuple[UUID, ...]
    default_query_profile_id: UUID | None
    raw_file_read: bool
    requests_per_minute: int | None
    max_concurrency: int | None
    not_before: datetime | None
    expires_at: datetime | None
    resource_revision: int = Field(ge=1)
    etag: str | None
    created_at: datetime
    updated_at: datetime
    revoked_at: datetime | None

    _validate_not_before = field_validator("not_before")(_require_aware_utc)
    _validate_expires_at = field_validator("expires_at")(_require_aware_utc)
    _validate_created_at = field_validator("created_at")(_require_aware_utc)
    _validate_updated_at = field_validator("updated_at")(_require_aware_utc)
    _validate_revoked_at = field_validator("revoked_at")(_require_aware_utc)


class IssuedApiKey(_Schema):
    api_key: SafeApiKey
    token: SecretStr = Field(repr=False)


class Page[T](_Schema):
    items: tuple[T, ...]
    next_cursor: str | None = None
