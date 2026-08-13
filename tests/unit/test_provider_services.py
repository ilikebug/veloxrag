import asyncio
import base64
import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from types import TracebackType
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from pydantic import SecretStr, ValidationError
from sqlalchemy.exc import IntegrityError

from rag_service.api.cursors import CursorPosition
from rag_service.api.errors import BusinessError
from rag_service.auth.policies import AdminPrincipal
from rag_service.config import Settings
from rag_service.db.models.auth import AuditEvent, IdempotencyRecord
from rag_service.main import create_app
from rag_service.providers import routes as provider_routes
from rag_service.providers.credentials import (
    EncryptedProviderCredential,
    ProviderCredentialKeyring,
    ProviderCredentialUnavailableError,
)
from rag_service.providers.network_policy import (
    CanonicalProviderEndpoint,
    ProviderNetworkPolicyError,
    ResolvedProviderEndpoint,
)
from rag_service.providers.repositories import (
    AdminActorRecord,
    ModelProfileRecord,
    ProviderConfigRecord,
    ProviderConfigSecretSourceRecord,
    ProviderCredentialRecord,
    ProviderRepositories,
    SqlAlchemyProviderAuditRepository,
    SqlAlchemyProviderConfigRepository,
    SqlAlchemyProviderCredentialRepository,
)
from rag_service.providers.routes import (
    create_provider_config,
    create_provider_credential,
    update_provider_config,
    update_provider_credential,
)
from rag_service.providers.schemas import (
    ModelProfileCreate,
    ModelProfilePatch,
    ProviderConfigCreate,
    ProviderConfigPatch,
    ProviderCredentialCreate,
    ProviderCredentialPatch,
    SafeProviderCredential,
)
from rag_service.providers.services import (
    CredentialProviderSecretSource,
    LegacyProviderSecretSource,
    ModelProfileService,
    ProviderConfigService,
    ProviderCredentialService,
    model_profile_etag,
    provider_config_etag,
    provider_credential_etag,
    provider_credential_keyring_from_settings,
    select_provider_config_secret_source,
)

NOW = datetime(2026, 7, 27, 8, 0, tzinfo=UTC)
KEYS = {"2026-07": b"k" * 32}
SECRET = "provider-secret-sentinel"
SNAPSHOT_SENTINEL = "provider-snapshot-traceback-sentinel"
CONFIG_BASE_URL = "https://Provider.Example:443/v1/"
CONFIG_CANONICAL_URL = "https://provider.example/v1"


class FakeTransaction:
    def __init__(self, session: "FakeSession") -> None:
        self._session = session
        self._snapshot: object | None = None

    async def __aenter__(self) -> None:
        self._snapshot = self._session.snapshot()

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exception_type is not None:
            self._session.restore(self._snapshot)


class FakeSession:
    def __init__(self) -> None:
        self.credentials: dict[UUID, ProviderCredentialRecord] = {}
        self.encrypted: dict[UUID, EncryptedProviderCredential] = {}
        self.configs: dict[UUID, ProviderConfigRecord] = {}
        self.config_secret_sources: dict[UUID, ProviderConfigSecretSourceRecord] = {}
        self.profiles: dict[UUID, ModelProfileRecord] = {}
        self.generation_profile_ids: set[UUID] = set()
        self.idempotency: dict[tuple[UUID, str, str], IdempotencyRecord] = {}
        self.audits: list[AuditEvent] = []
        self.admins: dict[UUID, AdminActorRecord] = {}
        self.fail_credential = False
        self.fail_config = False
        self.fail_idempotency = False
        self.fail_audit = False
        self.credential_error: BaseException | None = None
        self.config_error: BaseException | None = None
        self.audit_error: BaseException | None = None
        self.create_count = 0
        self.config_create_count = 0
        self.profile_create_count = 0

    def begin(self) -> FakeTransaction:
        return FakeTransaction(self)

    def begin_nested(self) -> FakeTransaction:
        return FakeTransaction(self)

    def snapshot(self) -> object:
        return deepcopy(
            (
                self.credentials,
                self.encrypted,
                self.configs,
                self.config_secret_sources,
                self.profiles,
                self.generation_profile_ids,
                self.idempotency,
                self.audits,
                self.create_count,
                self.config_create_count,
                self.profile_create_count,
            )
        )

    def restore(self, snapshot: object | None) -> None:
        assert snapshot is not None
        (
            self.credentials,
            self.encrypted,
            self.configs,
            self.config_secret_sources,
            self.profiles,
            self.generation_profile_ids,
            self.idempotency,
            self.audits,
            self.create_count,
            self.config_create_count,
            self.profile_create_count,
        ) = cast(
            tuple[
                dict[UUID, ProviderCredentialRecord],
                dict[UUID, EncryptedProviderCredential],
                dict[UUID, ProviderConfigRecord],
                dict[UUID, ProviderConfigSecretSourceRecord],
                dict[UUID, ModelProfileRecord],
                set[UUID],
                dict[tuple[UUID, str, str], IdempotencyRecord],
                list[AuditEvent],
                int,
                int,
                int,
            ],
            snapshot,
        )


class FakeCredentialRepository:
    def __init__(self, session: FakeSession) -> None:
        self._session = session

    async def list_safe(
        self,
        position: CursorPosition | None,
        limit: int,
    ) -> list[ProviderCredentialRecord]:
        rows = sorted(
            self._session.credentials.values(),
            key=lambda row: (row.created_at, row.id),
        )
        if position is not None:
            rows = [
                row for row in rows if (row.created_at, row.id) > (position.created_at, position.id)
            ]
        return rows[:limit]

    async def get_safe(
        self,
        credential_id: UUID,
        *,
        for_update: bool = False,
    ) -> ProviderCredentialRecord | None:
        del for_update
        return self._session.credentials.get(credential_id)

    async def add_encrypted(
        self,
        credential_id: UUID,
        name: str,
        encrypted: EncryptedProviderCredential,
    ) -> ProviderCredentialRecord:
        if self._session.credential_error is not None:
            raise self._session.credential_error
        self._session.create_count += 1
        row = ProviderCredentialRecord(
            id=credential_id,
            name=name,
            key_version=encrypted.key_version,
            resource_revision=1,
            created_at=NOW,
            updated_at=NOW,
            rotated_at=None,
        )
        self._session.credentials[credential_id] = row
        self._session.encrypted[credential_id] = encrypted
        if self._session.fail_credential:
            raise RuntimeError("unsafe credential failure")
        return row

    async def update_encrypted(
        self,
        credential_id: UUID,
        *,
        name: str | None,
        encrypted: EncryptedProviderCredential | None,
        updated_at: datetime,
        rotated_at: datetime | None,
    ) -> ProviderCredentialRecord:
        if self._session.credential_error is not None:
            raise self._session.credential_error
        current = self._session.credentials[credential_id]
        row = ProviderCredentialRecord(
            id=current.id,
            name=current.name if name is None else name,
            key_version=(current.key_version if encrypted is None else encrypted.key_version),
            resource_revision=current.resource_revision + 1,
            created_at=current.created_at,
            updated_at=updated_at,
            rotated_at=current.rotated_at if encrypted is None else rotated_at,
        )
        self._session.credentials[credential_id] = row
        if encrypted is not None:
            self._session.encrypted[credential_id] = encrypted
        return row


class FakeConfigRepository:
    def __init__(self, session: FakeSession) -> None:
        self._session = session

    async def list_safe(
        self,
        position: CursorPosition | None,
        limit: int,
    ) -> list[ProviderConfigRecord]:
        rows = sorted(
            self._session.configs.values(),
            key=lambda row: (row.created_at, row.id),
        )
        if position is not None:
            rows = [
                row for row in rows if (row.created_at, row.id) > (position.created_at, position.id)
            ]
        return rows[:limit]

    async def get_safe(
        self,
        provider_config_id: UUID,
        *,
        for_update: bool = False,
    ) -> ProviderConfigRecord | None:
        del for_update
        return self._session.configs.get(provider_config_id)

    async def get_secret_source(
        self,
        provider_config_id: UUID,
    ) -> ProviderConfigSecretSourceRecord | None:
        if self._session.config_error is not None:
            raise self._session.config_error
        return self._session.config_secret_sources.get(provider_config_id)

    async def add_validated(
        self,
        provider_config_id: UUID,
        *,
        name: str,
        provider_type: str,
        base_url: str,
        credential_id: UUID,
        default_headers: dict[str, str],
        routing_options: dict[str, object],
        timeout_seconds: Decimal,
        max_concurrency: int,
        requests_per_minute: int,
        enabled: bool,
        endpoint_policy_version: str,
        endpoint_validated_at: datetime,
    ) -> ProviderConfigRecord:
        if self._session.config_error is not None:
            raise self._session.config_error
        self._session.config_create_count += 1
        row = ProviderConfigRecord(
            id=provider_config_id,
            name=name,
            provider_type=provider_type,
            base_url=base_url,
            credential_id=credential_id,
            default_headers=default_headers,
            routing_options=routing_options,
            timeout_seconds=timeout_seconds,
            max_concurrency=max_concurrency,
            requests_per_minute=requests_per_minute,
            enabled=enabled,
            resource_revision=1,
            endpoint_policy_version=endpoint_policy_version,
            endpoint_validated_at=endpoint_validated_at,
            created_at=NOW,
            updated_at=NOW,
        )
        self._session.configs[provider_config_id] = row
        self._session.config_secret_sources[provider_config_id] = ProviderConfigSecretSourceRecord(
            provider_config_id=provider_config_id,
            credential_id=credential_id,
            secret_ref=None,
        )
        if self._session.fail_config:
            raise RuntimeError("unsafe config failure")
        return row

    async def update_validated(
        self,
        provider_config_id: UUID,
        *,
        values: dict[str, object],
        updated_at: datetime,
    ) -> ProviderConfigRecord:
        if self._session.config_error is not None:
            raise self._session.config_error
        current = self._session.configs[provider_config_id]
        row = replace(
            current,
            name=cast(str, values.get("name", current.name)),
            provider_type=cast(str, values.get("provider_type", current.provider_type)),
            base_url=cast(str, values.get("base_url", current.base_url)),
            credential_id=cast(UUID | None, values.get("credential_id", current.credential_id)),
            default_headers=cast(
                dict[str, str],
                values.get("default_headers", current.default_headers),
            ),
            routing_options=cast(
                dict[str, object],
                values.get("routing_options", current.routing_options),
            ),
            timeout_seconds=cast(
                Decimal,
                values.get("timeout_seconds", current.timeout_seconds),
            ),
            max_concurrency=cast(
                int,
                values.get("max_concurrency", current.max_concurrency),
            ),
            requests_per_minute=cast(
                int,
                values.get("requests_per_minute", current.requests_per_minute),
            ),
            enabled=cast(bool, values.get("enabled", current.enabled)),
            resource_revision=current.resource_revision + 1,
            endpoint_policy_version=cast(
                str | None,
                values.get("endpoint_policy_version", current.endpoint_policy_version),
            ),
            endpoint_validated_at=cast(
                datetime | None,
                values.get("endpoint_validated_at", current.endpoint_validated_at),
            ),
            updated_at=updated_at,
        )
        self._session.configs[provider_config_id] = row
        source = self._session.config_secret_sources[provider_config_id]
        if "credential_id" in values:
            source = ProviderConfigSecretSourceRecord(
                provider_config_id=provider_config_id,
                credential_id=cast(UUID, values["credential_id"]),
                secret_ref=None,
            )
            self._session.config_secret_sources[provider_config_id] = source
        if self._session.fail_config:
            raise RuntimeError("unsafe config failure")
        return row


class FakeModelProfileRepository:
    def __init__(self, session: FakeSession) -> None:
        self._session = session

    async def list_safe(
        self,
        position: CursorPosition | None,
        limit: int,
    ) -> list[ModelProfileRecord]:
        rows = sorted(
            self._session.profiles.values(),
            key=lambda row: (row.created_at, row.id),
        )
        if position is not None:
            rows = [
                row for row in rows if (row.created_at, row.id) > (position.created_at, position.id)
            ]
        return rows[:limit]

    async def get_safe(
        self,
        model_profile_id: UUID,
        *,
        for_update: bool = False,
    ) -> ModelProfileRecord | None:
        del for_update
        return self._session.profiles.get(model_profile_id)

    async def add_profile(
        self,
        model_profile_id: UUID,
        *,
        name: str,
        capability: str,
        provider_config_id: UUID,
        model_name: str,
        dimension: int | None,
        max_input_tokens: int,
        batch_size: int,
        timeout_seconds: Decimal,
        vector_config: dict[str, object],
        enabled: bool,
    ) -> ModelProfileRecord:
        self._session.profile_create_count += 1
        row = ModelProfileRecord(
            id=model_profile_id,
            name=name,
            capability=capability,
            provider_config_id=provider_config_id,
            model_name=model_name,
            dimension=dimension,
            max_input_tokens=max_input_tokens,
            batch_size=batch_size,
            timeout_seconds=timeout_seconds,
            vector_config=vector_config,
            enabled=enabled,
            resource_revision=1,
            created_at=NOW,
            updated_at=NOW,
        )
        self._session.profiles[model_profile_id] = row
        return row

    async def update(
        self,
        model_profile_id: UUID,
        *,
        values: dict[str, object],
        updated_at: datetime,
    ) -> ModelProfileRecord:
        current = self._session.profiles[model_profile_id]
        row = replace(
            current,
            name=cast(str, values.get("name", current.name)),
            provider_config_id=cast(
                UUID,
                values.get("provider_config_id", current.provider_config_id),
            ),
            model_name=cast(str, values.get("model_name", current.model_name)),
            dimension=cast(int | None, values.get("dimension", current.dimension)),
            max_input_tokens=cast(
                int,
                values.get("max_input_tokens", current.max_input_tokens),
            ),
            batch_size=cast(int, values.get("batch_size", current.batch_size)),
            timeout_seconds=cast(
                Decimal,
                values.get("timeout_seconds", current.timeout_seconds),
            ),
            vector_config=cast(
                dict[str, object],
                values.get("vector_config", current.vector_config),
            ),
            enabled=cast(bool, values.get("enabled", current.enabled)),
            resource_revision=current.resource_revision + 1,
            updated_at=updated_at,
        )
        self._session.profiles[model_profile_id] = row
        return row

    async def is_referenced_by_generation(self, model_profile_id: UUID) -> bool:
        return model_profile_id in self._session.generation_profile_ids

    async def provider_config_is_referenced_by_generation(
        self,
        provider_config_id: UUID,
        *,
        lock_profiles: bool = False,
    ) -> bool:
        del lock_profiles
        return any(
            profile.id in self._session.generation_profile_ids
            and profile.provider_config_id == provider_config_id
            for profile in self._session.profiles.values()
        )


class FakeAdminRepository:
    def __init__(self, session: FakeSession) -> None:
        self._session = session

    async def get_for_update(self, key_id: UUID) -> AdminActorRecord | None:
        return self._session.admins.get(key_id)


class FakeIdempotencyRepository:
    def __init__(self, session: FakeSession) -> None:
        self._session = session

    async def get(
        self,
        actor_key_id: UUID,
        operation: str,
        idempotency_key: str,
    ) -> IdempotencyRecord | None:
        return self._session.idempotency.get((actor_key_id, operation, idempotency_key))

    async def add(self, record: IdempotencyRecord) -> None:
        key = (record.actor_key_id, record.operation, record.idempotency_key)
        self._session.idempotency[key] = record
        if self._session.fail_idempotency:
            raise RuntimeError("unsafe idempotency failure")


class FakeAuditRepository:
    def __init__(self, session: FakeSession) -> None:
        self._session = session

    async def add(self, event: AuditEvent) -> None:
        self._session.audits.append(event)
        if self._session.audit_error is not None:
            raise self._session.audit_error
        if self._session.fail_audit:
            raise RuntimeError("unsafe audit failure")

    async def get_created_response(
        self,
        actor_key_id: UUID,
        credential_id: UUID,
    ) -> object | None:
        matches = [
            event.metadata_
            for event in self._session.audits
            if event.actor_api_key_id == actor_key_id
            and event.action == "provider_credential.created"
            and event.target_type == "provider_credential"
            and event.target_id == credential_id
        ]
        return deepcopy(matches[0]) if len(matches) == 1 else None

    async def get_provider_config_created_response(
        self,
        actor_key_id: UUID,
        provider_config_id: UUID,
    ) -> object | None:
        matches = [
            event.metadata_
            for event in self._session.audits
            if event.actor_api_key_id == actor_key_id
            and event.action == "provider_config.created"
            and event.target_type == "provider_config"
            and event.target_id == provider_config_id
        ]
        return deepcopy(matches[0]) if len(matches) == 1 else None

    async def get_model_profile_created_response(
        self,
        actor_key_id: UUID,
        model_profile_id: UUID,
    ) -> object | None:
        matches = [
            event.metadata_
            for event in self._session.audits
            if event.actor_api_key_id == actor_key_id
            and event.action == "model_profile.created"
            and event.target_type == "model_profile"
            and event.target_id == model_profile_id
        ]
        return deepcopy(matches[0]) if len(matches) == 1 else None


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        admin_key_hmac_secret=SecretStr("a" * 32),
        agent_key_hmac_secret=SecretStr("b" * 32),
        default_page_size=2,
        max_page_size=3,
    )


def _service(
    session: FakeSession,
    *,
    settings: Settings | None = None,
    keyring_factory: Callable[[], ProviderCredentialKeyring] | None = None,
) -> ProviderCredentialService:
    def repositories(value: object) -> ProviderRepositories:
        assert value is session
        return ProviderRepositories(
            credentials=FakeCredentialRepository(session),
            admins=FakeAdminRepository(session),
            idempotency=FakeIdempotencyRepository(session),
            audits=FakeAuditRepository(session),
            configs=FakeConfigRepository(session),
            profiles=FakeModelProfileRepository(session),
        )

    return ProviderCredentialService(
        session=cast(Any, session),
        settings=_settings() if settings is None else settings,
        keyring_factory=_keyring if keyring_factory is None else keyring_factory,
        repository_factory=cast(Any, repositories),
        clock=lambda: NOW,
    )


class FakeEndpointPolicy:
    policy_version = "provider-endpoint-v1"

    def __init__(self) -> None:
        self.urls: list[str] = []
        self.headers: list[dict[str, str]] = []
        self.reject = False

    def validate_for_persistence(self, raw_url: str) -> ResolvedProviderEndpoint:
        self.urls.append(raw_url)
        if self.reject:
            raise ProviderNetworkPolicyError
        endpoint = CanonicalProviderEndpoint(
            url=CONFIG_CANONICAL_URL if raw_url == CONFIG_BASE_URL else raw_url.rstrip("/"),
            hostname="provider.example",
            port=443,
            path="/v1",
        )
        return ResolvedProviderEndpoint(endpoint=endpoint, addresses=("93.184.216.34",))

    def validate_url(self, raw_url: str) -> CanonicalProviderEndpoint:
        return CanonicalProviderEndpoint(
            url=CONFIG_CANONICAL_URL if raw_url == CONFIG_BASE_URL else raw_url.rstrip("/"),
            hostname="provider.example",
            port=443,
            path="/v1",
        )

    def validate_headers(self, headers: dict[str, str]) -> dict[str, str]:
        copied = dict(headers)
        self.headers.append(copied)
        if self.reject or any(name.lower() == "authorization" for name in copied):
            raise ProviderNetworkPolicyError
        return {
            {
                "http-referer": "HTTP-Referer",
                "x-openrouter-title": "X-OpenRouter-Title",
                "x-title": "X-Title",
            }[name.lower()]: value
            for name, value in copied.items()
        }


def _config_service(
    session: FakeSession,
    policy: FakeEndpointPolicy,
    *,
    clock: Callable[[], datetime] = lambda: NOW,
) -> ProviderConfigService:
    def repositories(value: object) -> ProviderRepositories:
        assert value is session
        return ProviderRepositories(
            credentials=FakeCredentialRepository(session),
            admins=FakeAdminRepository(session),
            idempotency=FakeIdempotencyRepository(session),
            audits=FakeAuditRepository(session),
            configs=FakeConfigRepository(session),
            profiles=FakeModelProfileRepository(session),
        )

    return ProviderConfigService(
        session=cast(Any, session),
        settings=_settings(),
        endpoint_policy_factory=lambda: cast(Any, policy),
        repository_factory=cast(Any, repositories),
        clock=clock,
    )


def _profile_service(session: FakeSession) -> ModelProfileService:
    def repositories(value: object) -> ProviderRepositories:
        assert value is session
        return ProviderRepositories(
            credentials=FakeCredentialRepository(session),
            admins=FakeAdminRepository(session),
            idempotency=FakeIdempotencyRepository(session),
            audits=FakeAuditRepository(session),
            configs=FakeConfigRepository(session),
            profiles=FakeModelProfileRepository(session),
        )

    return ModelProfileService(
        session=cast(Any, session),
        settings=_settings(),
        repository_factory=cast(Any, repositories),
        clock=lambda: NOW,
    )


def _keyring() -> ProviderCredentialKeyring:
    return ProviderCredentialKeyring(
        keys=KEYS,
        active_key_version="2026-07",
    )


def _admin(session: FakeSession) -> AdminPrincipal:
    key_id = uuid4()
    public_id = "admin-public-id-0001"
    session.admins[key_id] = AdminActorRecord(
        id=key_id,
        public_id=public_id,
        key_type="admin",
        status="active",
        not_before=None,
        expires_at=None,
        revoked_at=None,
    )
    return AdminPrincipal(key_id=key_id, public_id=public_id)


def _credential_record(session: FakeSession, *, name: str = "config credential") -> UUID:
    credential_id = uuid4()
    session.credentials[credential_id] = ProviderCredentialRecord(
        id=credential_id,
        name=name,
        key_version="2026-07",
        resource_revision=1,
        created_at=NOW,
        updated_at=NOW,
        rotated_at=None,
    )
    return credential_id


def _config_command(
    credential_id: UUID,
    **updates: object,
) -> ProviderConfigCreate:
    payload: dict[str, object] = {
        "name": "embedding provider",
        "provider_type": "openai_compatible",
        "base_url": CONFIG_BASE_URL,
        "credential_id": credential_id,
        "default_headers": {"x-title": "RAG"},
        "routing_options": {},
        "timeout_seconds": "30.000",
        "max_concurrency": 8,
        "requests_per_minute": 600,
        "enabled": True,
    }
    payload.update(updates)
    return ProviderConfigCreate.model_validate(payload)


def _config_record(
    session: FakeSession,
    *,
    enabled: bool = True,
    name: str = "profile provider",
    provider_type: str = "openai_compatible",
) -> UUID:
    credential_id = _credential_record(session, name=f"{name} credential")
    provider_config_id = uuid4()
    session.configs[provider_config_id] = ProviderConfigRecord(
        id=provider_config_id,
        name=name,
        provider_type=provider_type,
        base_url=CONFIG_CANONICAL_URL,
        credential_id=credential_id,
        default_headers={},
        routing_options={},
        timeout_seconds=Decimal("30"),
        max_concurrency=8,
        requests_per_minute=600,
        enabled=enabled,
        resource_revision=1,
        endpoint_policy_version="provider-endpoint-v1",
        endpoint_validated_at=NOW,
        created_at=NOW,
        updated_at=NOW,
    )
    session.config_secret_sources[provider_config_id] = ProviderConfigSecretSourceRecord(
        provider_config_id=provider_config_id,
        credential_id=credential_id,
        secret_ref=None,
    )
    return provider_config_id


def _profile_command(provider_config_id: UUID, **updates: object) -> ModelProfileCreate:
    payload: dict[str, object] = {
        "name": "text embedding profile",
        "capability": "embedding",
        "provider_config_id": provider_config_id,
        "model_name": "text-embedding-3-small",
        "dimension": 1536,
        "max_input_tokens": 8191,
        "batch_size": 64,
        "timeout_seconds": "30.000",
        "vector_config": {},
        "enabled": True,
    }
    payload.update(updates)
    return ModelProfileCreate.model_validate(payload)


def _decrypt(
    keyring: ProviderCredentialKeyring,
    credential_id: UUID,
    encrypted: EncryptedProviderCredential,
) -> str:
    return keyring.use_decrypted(
        credential_id,
        encrypted,
        lambda buffer: bytes(buffer).decode(),
    )


def _assert_credential_error_graph_redacted(
    error: BusinessError,
    sentinel: str,
) -> None:
    pending: list[BaseException] = [error]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        assert sentinel not in repr(current)
        assert sentinel not in repr(current.args)
        traceback = current.__traceback__
        while traceback is not None:
            frame = traceback.tb_frame
            if "/src/rag_service/" in frame.f_code.co_filename:
                assert sentinel not in repr(frame.f_locals)
                assert all(sentinel not in repr(value) for value in frame.f_locals.values())
            traceback = traceback.tb_next
        for linked in (current.__cause__, current.__context__):
            if linked is not None:
                pending.append(linked)


def _assert_production_traceback_scrubbed(
    error: BaseException,
    *,
    markers: tuple[str, ...],
    forbidden_objects: tuple[object, ...] = (),
) -> None:
    pending: list[BaseException] = [error]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        for marker in markers:
            assert marker not in repr(current)
            assert marker not in repr(current.args)
        traceback = current.__traceback__
        while traceback is not None:
            frame = traceback.tb_frame
            if "/src/rag_service/" in frame.f_code.co_filename:
                local_values = tuple(frame.f_locals.values())
                for forbidden in forbidden_objects:
                    assert all(value is not forbidden for value in local_values)
                for marker in markers:
                    assert marker not in repr(frame.f_locals)
                    assert all(marker not in repr(value) for value in local_values)
            traceback = traceback.tb_next
        for linked in (current.__cause__, current.__context__):
            if linked is not None:
                pending.append(linked)


def _assert_production_frame_locals_scrubbed(
    error: BaseException,
    *,
    markers: tuple[str, ...],
    forbidden_objects: tuple[object, ...] = (),
) -> None:
    traceback = error.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        if "/src/rag_service/" in frame.f_code.co_filename:
            local_values = tuple(frame.f_locals.values())
            for forbidden in forbidden_objects:
                assert all(value is not forbidden for value in local_values)
            for marker in markers:
                assert marker not in repr(frame.f_locals)
                assert all(marker not in repr(value) for value in local_values)
        traceback = traceback.tb_next


def _caller_owned_cancellation(
    prefix: str,
) -> tuple[
    asyncio.CancelledError,
    RuntimeError,
    RuntimeError,
]:
    cause = RuntimeError(f"{prefix}-cause-sensitive")
    context = RuntimeError(f"{prefix}-context-sensitive")
    source = asyncio.CancelledError(
        f"{prefix}-args-sensitive ciphertext-sensitive nonce-sensitive statement-param-sensitive"
    )
    source.__cause__ = cause
    source.__context__ = context
    return source, cause, context


def _assert_caller_owned_cancellation_unchanged(
    source: asyncio.CancelledError,
    cause: RuntimeError,
    context: RuntimeError,
    prefix: str,
) -> None:
    assert source.args == (
        f"{prefix}-args-sensitive ciphertext-sensitive nonce-sensitive statement-param-sensitive",
    )
    assert source.__cause__ is cause
    assert source.__context__ is context
    assert cause.args == (f"{prefix}-cause-sensitive",)
    assert context.args == (f"{prefix}-context-sensitive",)


def test_credential_schemas_hide_secret_and_bound_input() -> None:
    command = ProviderCredentialCreate(name=" cloud key ", secret=SecretStr(SECRET))

    assert command.name == "cloud key"
    assert SECRET not in repr(command)
    assert SECRET not in str(command)

    for secret in ("", "   ", "x" * 8193):
        with pytest.raises(ValidationError) as captured:
            ProviderCredentialCreate(name="cloud", secret=SecretStr(secret))
        if secret.strip():
            assert secret not in str(captured.value)

    with pytest.raises(ValidationError):
        ProviderCredentialPatch()
    with pytest.raises(ValidationError):
        ProviderCredentialPatch(name=None)

    with pytest.raises(ValidationError):
        SafeProviderCredential(
            id=uuid4(),
            name="cloud",
            credential_configured=False,
            key_version="2026-07",
            resource_revision=0,
            created_at=NOW,
            updated_at=NOW,
            rotated_at=None,
        )


@pytest.mark.parametrize(
    "payload",
    (
        {"name": "invalid-\ud800-name", "secret": SecretStr("valid")},
        {"name": "cloud", "secret": SecretStr("invalid-\ud800-secret")},
        {"name": "cloud", "secret": SecretStr("😀" * 3000)},
    ),
)
def test_credential_schemas_reject_non_utf8_or_oversized_utf8_input(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError) as captured:
        ProviderCredentialCreate.model_validate(payload)

    assert "invalid-" not in str(captured.value)


@pytest.mark.asyncio
async def test_provider_credential_create_replays_and_rejects_reused_key() -> None:
    session = FakeSession()
    actor = _admin(session)
    service = _service(session)
    command = ProviderCredentialCreate(name="cloud", secret=SecretStr(SECRET))

    created = await service.create_credential(
        command,
        actor=actor,
        request_id="req-credential-create",
        idempotency_key="credential-create-key",
    )

    assert created.created is True
    safe = created.credential
    assert safe.credential_configured is True
    assert safe.resource_revision == 1
    assert set(safe.model_dump()) == {
        "id",
        "name",
        "credential_configured",
        "key_version",
        "resource_revision",
        "created_at",
        "updated_at",
        "rotated_at",
    }
    encrypted = session.encrypted[safe.id]
    assert encrypted.nonce != b""
    assert SECRET.encode() not in encrypted.ciphertext
    assert _decrypt(_keyring(), safe.id, encrypted) == SECRET
    assert session.create_count == 1
    assert [event.action for event in session.audits] == ["provider_credential.created"]
    assert session.audits[0].metadata_ == safe.model_dump(mode="json")
    serialized_snapshot = json.dumps(session.audits[0].metadata_)
    for forbidden in (SECRET, "secret", "ciphertext", "nonce"):
        assert forbidden not in serialized_snapshot
    record = next(iter(session.idempotency.values()))
    assert len(record.request_fingerprint) == 32
    assert SECRET.encode() not in record.request_fingerprint

    mutated = await service.update_credential(
        safe.id,
        ProviderCredentialPatch(
            name="cloud renamed",
            secret=SecretStr("rotated-after-create"),
        ),
        actor=actor,
        request_id="req-credential-mutate",
        expected_etag=provider_credential_etag(safe.id, 1),
    )
    assert mutated.resource_revision == 2
    assert mutated.name == "cloud renamed"

    replayed = await service.create_credential(
        command,
        actor=actor,
        request_id="req-credential-replay",
        idempotency_key="credential-create-key",
    )
    assert replayed.created is False
    assert replayed.credential == safe
    assert replayed.credential != mutated
    assert session.create_count == 1
    assert len(session.audits) == 2

    with pytest.raises(BusinessError) as conflict:
        await service.create_credential(
            ProviderCredentialCreate(
                name="cloud",
                secret=SecretStr("different-secret"),
            ),
            actor=actor,
            request_id="req-credential-conflict",
            idempotency_key="credential-create-key",
        )
    assert conflict.value.status_code == 409
    assert conflict.value.code == "IDEMPOTENCY_KEY_REUSED"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_flag",
    ("fail_credential", "fail_idempotency", "fail_audit"),
)
async def test_provider_credential_create_rolls_back_every_repository_failure(
    failure_flag: str,
) -> None:
    session = FakeSession()
    actor = _admin(session)
    setattr(session, failure_flag, True)

    with pytest.raises(BusinessError) as captured:
        await _service(session).create_credential(
            ProviderCredentialCreate(name="cloud", secret=SecretStr(SECRET)),
            actor=actor,
            request_id="req-audit-fails",
            idempotency_key="audit-fails-key",
        )

    assert captured.value.code == "INTERNAL_ERROR"
    assert session.credentials == {}
    assert session.encrypted == {}
    assert session.idempotency == {}
    assert session.audits == []
    assert SECRET not in str(captured.value)


@pytest.mark.asyncio
async def test_provider_credential_create_replay_fails_closed_for_missing_snapshot() -> None:
    session = FakeSession()
    actor = _admin(session)
    service = _service(session)
    command = ProviderCredentialCreate(name="cloud", secret=SecretStr(SECRET))
    await service.create_credential(
        command,
        actor=actor,
        request_id="req-create-snapshot",
        idempotency_key="snapshot-key",
    )
    session.audits.clear()

    with pytest.raises(BusinessError) as captured:
        await service.create_credential(
            command,
            actor=actor,
            request_id="req-replay-snapshot",
            idempotency_key="snapshot-key",
        )

    assert (captured.value.status_code, captured.value.code) == (500, "INTERNAL_ERROR")
    assert SECRET not in str(captured.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("snapshot_state", ("malformed", "id_mismatch"))
async def test_provider_credential_create_replay_redacts_invalid_snapshot_tracebacks(
    snapshot_state: str,
) -> None:
    session = FakeSession()
    actor = _admin(session)
    service = _service(session)
    command = ProviderCredentialCreate(name="cloud", secret=SecretStr(SECRET))
    await service.create_credential(
        command,
        actor=actor,
        request_id="req-create-sensitive-snapshot",
        idempotency_key="sensitive-snapshot-key",
    )
    if snapshot_state == "malformed":
        unsafe_snapshot: dict[str, object] = {"name": SNAPSHOT_SENTINEL}
    else:
        unsafe_snapshot = deepcopy(session.audits[0].metadata_)
        unsafe_snapshot["id"] = str(uuid4())
        unsafe_snapshot["name"] = SNAPSHOT_SENTINEL
    session.audits[0].metadata_ = unsafe_snapshot

    with pytest.raises(BusinessError) as captured:
        await service.create_credential(
            command,
            actor=actor,
            request_id="req-replay-sensitive-snapshot",
            idempotency_key="sensitive-snapshot-key",
        )

    assert (captured.value.status_code, captured.value.code) == (500, "INTERNAL_ERROR")
    assert session.audits[0].metadata_ == unsafe_snapshot
    assert SNAPSHOT_SENTINEL in repr(session.audits[0].metadata_)
    _assert_credential_error_graph_redacted(captured.value, SNAPSHOT_SENTINEL)


@pytest.mark.asyncio
async def test_provider_credential_patch_requires_etag_and_rotates_in_place() -> None:
    session = FakeSession()
    actor = _admin(session)
    service = _service(session)
    created = await service.create_credential(
        ProviderCredentialCreate(name="cloud", secret=SecretStr(SECRET)),
        actor=actor,
        request_id="req-create",
        idempotency_key="create-key",
    )
    credential_id = created.credential.id
    first_encrypted = session.encrypted[credential_id]
    first_etag = provider_credential_etag(credential_id, 1)

    with pytest.raises(BusinessError) as missing:
        await service.update_credential(
            credential_id,
            ProviderCredentialPatch(name="renamed"),
            actor=actor,
            request_id="req-missing",
            expected_etag=None,
        )
    assert (missing.value.status_code, missing.value.code) == (
        428,
        "PRECONDITION_REQUIRED",
    )

    with pytest.raises(BusinessError) as stale:
        await service.update_credential(
            credential_id,
            ProviderCredentialPatch(name="renamed"),
            actor=actor,
            request_id="req-stale",
            expected_etag='"provider-credential:stale:r1"',
        )
    assert (stale.value.status_code, stale.value.code) == (
        412,
        "PRECONDITION_FAILED",
    )

    renamed = await service.update_credential(
        credential_id,
        ProviderCredentialPatch(name="renamed"),
        actor=actor,
        request_id="req-rename",
        expected_etag=first_etag,
    )
    assert renamed.name == "renamed"
    assert renamed.resource_revision == 2
    assert renamed.rotated_at is None
    assert session.encrypted[credential_id] == first_encrypted

    rotated = await service.update_credential(
        credential_id,
        ProviderCredentialPatch(secret=SecretStr("rotated-secret")),
        actor=actor,
        request_id="req-rotate",
        expected_etag=provider_credential_etag(credential_id, 2),
    )
    second_encrypted = session.encrypted[credential_id]
    assert rotated.id == credential_id
    assert rotated.resource_revision == 3
    assert rotated.rotated_at == NOW
    assert second_encrypted.nonce != first_encrypted.nonce
    assert second_encrypted.ciphertext != first_encrypted.ciphertext
    assert _decrypt(_keyring(), credential_id, second_encrypted) == "rotated-secret"
    assert [event.action for event in session.audits] == [
        "provider_credential.created",
        "provider_credential.updated",
        "provider_credential.rotated",
    ]
    assert session.audits[1].metadata_ == {}
    assert session.audits[2].metadata_ == {"name_updated": False}


@pytest.mark.asyncio
async def test_provider_credential_patch_rolls_back_credential_and_audit_together() -> None:
    session = FakeSession()
    actor = _admin(session)
    service = _service(session)
    created = await service.create_credential(
        ProviderCredentialCreate(name="cloud", secret=SecretStr(SECRET)),
        actor=actor,
        request_id="req-create-before-patch-failure",
        idempotency_key="patch-failure-create",
    )
    before = session.snapshot()
    session.fail_audit = True

    with pytest.raises(BusinessError) as captured:
        await service.update_credential(
            created.credential.id,
            ProviderCredentialPatch(
                name="must roll back",
                secret=SecretStr("must-not-persist"),
            ),
            actor=actor,
            request_id="req-patch-audit-failure",
            expected_etag=provider_credential_etag(created.credential.id, 1),
        )

    assert (captured.value.status_code, captured.value.code) == (500, "INTERNAL_ERROR")
    current = session.credentials[created.credential.id]
    (
        original_credentials,
        original_encrypted,
        _,
        _,
        _,
        _,
        _,
        original_audits,
        _,
        _,
        _,
    ) = cast(
        tuple[
            dict[UUID, ProviderCredentialRecord],
            dict[UUID, EncryptedProviderCredential],
            dict[UUID, ProviderConfigRecord],
            dict[UUID, ProviderConfigSecretSourceRecord],
            dict[UUID, ModelProfileRecord],
            set[UUID],
            dict[tuple[UUID, str, str], IdempotencyRecord],
            list[AuditEvent],
            int,
            int,
            int,
        ],
        before,
    )
    assert current == original_credentials[created.credential.id]
    assert session.encrypted == original_encrypted
    assert [
        (event.id, event.action, event.target_id, event.metadata_) for event in session.audits
    ] == [(event.id, event.action, event.target_id, event.metadata_) for event in original_audits]


@pytest.mark.asyncio
async def test_provider_credential_safe_list_get_and_cursor_pagination() -> None:
    session = FakeSession()
    actor = _admin(session)
    service = _service(session)
    created_ids: list[UUID] = []
    for index in range(3):
        result = await service.create_credential(
            ProviderCredentialCreate(
                name=f"credential-{index}",
                secret=SecretStr(f"secret-{index}"),
            ),
            actor=actor,
            request_id=f"req-create-{index}",
            idempotency_key=f"create-{index}",
        )
        created_ids.append(result.credential.id)

    first = await service.list_credentials(limit=2)
    assert len(first.items) == 2
    assert first.next_cursor is not None
    second = await service.list_credentials(cursor=first.next_cursor, limit=2)
    assert len(second.items) == 1
    assert second.next_cursor is None
    assert {item.id for item in (*first.items, *second.items)} == set(created_ids)
    assert await service.get_credential(created_ids[0]) in (*first.items, *second.items)

    serialized = json.dumps(
        [item.model_dump(mode="json") for item in (*first.items, *second.items)]
    )
    for forbidden in ("secret", "ciphertext", "nonce", SECRET):
        assert forbidden not in serialized


@pytest.mark.asyncio
async def test_provider_credential_read_paths_and_name_patch_do_not_load_keyring() -> None:
    session = FakeSession()
    actor = _admin(session)
    created = await _service(session).create_credential(
        ProviderCredentialCreate(name="cloud", secret=SecretStr(SECRET)),
        actor=actor,
        request_id="req-lazy-seed",
        idempotency_key="lazy-seed",
    )
    calls = 0

    def unavailable_keyring() -> ProviderCredentialKeyring:
        nonlocal calls
        calls += 1
        raise ProviderCredentialUnavailableError

    service = _service(session, keyring_factory=unavailable_keyring)
    listed = await service.list_credentials()
    fetched = await service.get_credential(created.credential.id)
    renamed = await service.update_credential(
        created.credential.id,
        ProviderCredentialPatch(name="renamed without keyring"),
        actor=actor,
        request_id="req-lazy-rename",
        expected_etag=provider_credential_etag(created.credential.id, 1),
    )

    assert listed.items == (created.credential,)
    assert fetched == created.credential
    assert renamed.resource_revision == 2
    assert calls == 0

    with pytest.raises(BusinessError) as rotated:
        await service.update_credential(
            created.credential.id,
            ProviderCredentialPatch(secret=SecretStr("rotation needs keyring")),
            actor=actor,
            request_id="req-lazy-rotate",
            expected_etag=provider_credential_etag(created.credential.id, 2),
        )
    assert (rotated.value.status_code, rotated.value.code) == (
        503,
        "PROVIDER_CREDENTIAL_UNAVAILABLE",
    )
    assert calls == 1

    empty_session = FakeSession()
    empty_actor = _admin(empty_session)
    with pytest.raises(BusinessError) as creation:
        await _service(
            empty_session,
            keyring_factory=unavailable_keyring,
        ).create_credential(
            ProviderCredentialCreate(name="new", secret=SecretStr(SECRET)),
            actor=empty_actor,
            request_id="req-lazy-create",
            idempotency_key="lazy-create",
        )
    assert (creation.value.status_code, creation.value.code) == (
        503,
        "PROVIDER_CREDENTIAL_UNAVAILABLE",
    )
    assert calls == 2
    assert empty_session.credentials == {}


def test_provider_credential_keyring_version_boundary_is_64_characters() -> None:
    exact = "v" * 64
    exact_settings = Settings(
        _env_file=None,
        environment="test",
        provider_credential_keyring=SecretStr(
            json.dumps({exact: base64.b64encode(b"k" * 32).decode()})
        ),
        provider_credential_active_key_version=exact,
    )
    assert provider_credential_keyring_from_settings(exact_settings).active_key_version == exact

    too_long = "v" * 65
    too_long_settings = Settings(
        _env_file=None,
        environment="test",
        provider_credential_keyring=SecretStr(
            json.dumps({too_long: base64.b64encode(b"k" * 32).decode()})
        ),
        provider_credential_active_key_version=too_long,
    )
    with pytest.raises(BusinessError) as captured:
        provider_credential_keyring_from_settings(too_long_settings)
    assert (captured.value.status_code, captured.value.code) == (
        503,
        "PROVIDER_CREDENTIAL_UNAVAILABLE",
    )


@pytest.mark.asyncio
async def test_provider_credential_not_found_and_invalid_pagination_are_safe() -> None:
    session = FakeSession()
    service = _service(session)
    missing_id = uuid4()

    with pytest.raises(BusinessError) as missing:
        await service.get_credential(missing_id)
    assert (missing.value.status_code, missing.value.code) == (404, "RESOURCE_NOT_FOUND")

    for cursor, limit in (("not-a-cursor", None), (None, 0), (None, 4)):
        with pytest.raises(BusinessError) as invalid:
            await service.list_credentials(cursor=cursor, limit=limit)
        assert (invalid.value.status_code, invalid.value.code) == (422, "VALIDATION_ERROR")


@pytest.mark.asyncio
async def test_provider_credential_writes_revalidate_the_admin_actor() -> None:
    session = FakeSession()
    actor = _admin(session)
    session.admins[actor.key_id] = AdminActorRecord(
        id=actor.key_id,
        public_id=actor.public_id,
        key_type="admin",
        status="revoked",
        not_before=None,
        expires_at=None,
        revoked_at=NOW,
    )

    with pytest.raises(BusinessError) as captured:
        await _service(session).create_credential(
            ProviderCredentialCreate(name="cloud", secret=SecretStr(SECRET)),
            actor=actor,
            request_id="req-revoked-admin",
            idempotency_key="revoked-admin-key",
        )

    assert (captured.value.status_code, captured.value.code) == (401, "INVALID_API_KEY")
    assert session.credentials == {}

    active_session = FakeSession()
    active_actor = _admin(active_session)
    service = _service(active_session)
    created = await service.create_credential(
        ProviderCredentialCreate(name="existing", secret=SecretStr(SECRET)),
        actor=active_actor,
        request_id="req-active-admin-create",
        idempotency_key="active-admin-create",
    )
    before = active_session.credentials[created.credential.id]
    active_session.admins[active_actor.key_id] = AdminActorRecord(
        id=active_actor.key_id,
        public_id=active_actor.public_id,
        key_type="admin",
        status="revoked",
        not_before=None,
        expires_at=None,
        revoked_at=NOW,
    )

    with pytest.raises(BusinessError) as patch_denied:
        await service.update_credential(
            created.credential.id,
            ProviderCredentialPatch(name="must not update"),
            actor=active_actor,
            request_id="req-revoked-admin-patch",
            expected_etag=provider_credential_etag(created.credential.id, 1),
        )

    assert (patch_denied.value.status_code, patch_denied.value.code) == (
        401,
        "INVALID_API_KEY",
    )
    assert active_session.credentials[created.credential.id] == before


class _CancellingExecuteSession:
    def __init__(self) -> None:
        self.statement: object | None = None

    async def execute(self, statement: object) -> object:
        self.statement = statement
        raise asyncio.CancelledError


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("add", "update"))
async def test_sqlalchemy_credential_repository_scrubs_cancelled_write_tracebacks(
    operation: str,
) -> None:
    session = _CancellingExecuteSession()
    repository = SqlAlchemyProviderCredentialRepository(cast(Any, session))
    credential_id = uuid4()
    encrypted = EncryptedProviderCredential(
        ciphertext=b"ciphertext-cancel-sentinel",
        nonce=b"nonce-cancel-sentinel",
        key_version="cancel-key-version-sentinel",
        algorithm="AES-256-GCM",
    )

    with pytest.raises(asyncio.CancelledError) as captured:
        if operation == "add":
            await repository.add_encrypted(credential_id, "cancel-name-sentinel", encrypted)
        else:
            await repository.update_encrypted(
                credential_id,
                name="cancel-name-sentinel",
                encrypted=encrypted,
                updated_at=NOW,
                rotated_at=NOW,
            )

    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert session.statement is not None
    _assert_production_traceback_scrubbed(
        captured.value,
        markers=(
            "ciphertext-cancel-sentinel",
            "nonce-cancel-sentinel",
            "cancel-key-version-sentinel",
            "cancel-name-sentinel",
        ),
        forbidden_objects=(encrypted, session.statement),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("create", "update"))
async def test_provider_credential_service_scrubs_cancelled_write_tracebacks(
    operation: str,
) -> None:
    session = FakeSession()
    actor = _admin(session)
    service = _service(session)
    invocation: Any
    prefix = f"service-{operation}-cancel"
    source, cause, context = _caller_owned_cancellation(prefix)
    if operation == "create":
        command: ProviderCredentialCreate | ProviderCredentialPatch = ProviderCredentialCreate(
            name="cancel-create-name-sentinel",
            secret=SecretStr("cancel-create-secret-sentinel"),
        )
        session.credential_error = source
        invocation = service.create_credential(
            cast(ProviderCredentialCreate, command),
            actor=actor,
            request_id="req-cancel-create",
            idempotency_key="cancel-create-key",
        )
        markers = ("cancel-create-name-sentinel", "cancel-create-secret-sentinel")
    else:
        created = await service.create_credential(
            ProviderCredentialCreate(name="seed", secret=SecretStr(SECRET)),
            actor=actor,
            request_id="req-cancel-update-seed",
            idempotency_key="cancel-update-seed",
        )
        command = ProviderCredentialPatch(
            name="cancel-update-name-sentinel",
            secret=SecretStr("cancel-update-secret-sentinel"),
        )
        session.credential_error = source
        invocation = service.update_credential(
            created.credential.id,
            command,
            actor=actor,
            request_id="req-cancel-update",
            expected_etag=provider_credential_etag(created.credential.id, 1),
        )
        markers = ("cancel-update-name-sentinel", "cancel-update-secret-sentinel")

    with pytest.raises(asyncio.CancelledError) as captured:
        await invocation

    _assert_caller_owned_cancellation_unchanged(source, cause, context, prefix)
    assert captured.value is not source
    assert captured.value.args == ()
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    sensitive_markers = (
        *markers,
        f"{prefix}-args-sensitive",
        f"{prefix}-cause-sensitive",
        f"{prefix}-context-sensitive",
        "ciphertext-sensitive",
        "nonce-sensitive",
        "statement-param-sensitive",
    )
    _assert_production_frame_locals_scrubbed(
        source,
        markers=sensitive_markers,
        forbidden_objects=(command, command.secret),
    )
    _assert_production_traceback_scrubbed(
        captured.value,
        markers=sensitive_markers,
        forbidden_objects=(command, command.secret),
    )


class _CancellingCredentialService:
    def __init__(self, source: asyncio.CancelledError) -> None:
        self._source = source

    async def create_credential(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise self._source

    async def update_credential(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise self._source


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("create", "update"))
async def test_provider_credential_routes_scrub_cancelled_command_tracebacks(
    operation: str,
) -> None:
    session = FakeSession()
    actor = _admin(session)
    prefix = f"route-{operation}-cancel"
    source, cause, context = _caller_owned_cancellation(prefix)
    service = cast(ProviderCredentialService, _CancellingCredentialService(source))
    invocation: Any
    if operation == "create":
        command: ProviderCredentialCreate | ProviderCredentialPatch = ProviderCredentialCreate(
            name="route-create-name-sentinel",
            secret=SecretStr("route-create-secret-sentinel"),
        )
        invocation = create_provider_credential(
            cast(ProviderCredentialCreate, command),
            "req-route-create-cancel",
            actor,
            service,
            "route-create-key",
        )
        markers = ("route-create-name-sentinel", "route-create-secret-sentinel")
    else:
        command = ProviderCredentialPatch(
            name="route-update-name-sentinel",
            secret=SecretStr("route-update-secret-sentinel"),
        )
        invocation = update_provider_credential(
            uuid4(),
            command,
            "req-route-update-cancel",
            actor,
            service,
            '"provider-credential:unused:r1"',
        )
        markers = ("route-update-name-sentinel", "route-update-secret-sentinel")

    with pytest.raises(asyncio.CancelledError) as captured:
        await invocation

    _assert_caller_owned_cancellation_unchanged(source, cause, context, prefix)
    assert captured.value is not source
    assert captured.value.args == ()
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    sensitive_markers = (
        *markers,
        f"{prefix}-args-sensitive",
        f"{prefix}-cause-sensitive",
        f"{prefix}-context-sensitive",
        "ciphertext-sensitive",
        "nonce-sensitive",
        "statement-param-sensitive",
    )
    _assert_production_frame_locals_scrubbed(
        source,
        markers=sensitive_markers,
        forbidden_objects=(command, command.secret),
    )
    _assert_production_traceback_scrubbed(
        captured.value,
        markers=sensitive_markers,
        forbidden_objects=(command, command.secret),
    )


@pytest.mark.asyncio
async def test_provider_credential_service_does_not_mutate_caller_owned_exception() -> None:
    session = FakeSession()
    actor = _admin(session)
    source_cause = RuntimeError("caller-cause-sentinel")
    source: RuntimeError
    try:
        raise RuntimeError("caller-owned-sentinel") from source_cause
    except RuntimeError as caught:
        source = caught
    original_args = source.args
    original_cause = source.__cause__
    original_context = source.__context__
    session.audit_error = source

    with pytest.raises(BusinessError) as captured:
        await _service(session).create_credential(
            ProviderCredentialCreate(name="cloud", secret=SecretStr(SECRET)),
            actor=actor,
            request_id="req-caller-owned",
            idempotency_key="caller-owned-key",
        )

    assert (captured.value.status_code, captured.value.code) == (500, "INTERNAL_ERROR")
    assert source.args == original_args
    assert source.__cause__ is original_cause
    assert source.__context__ is original_context
    assert source_cause.args == ("caller-cause-sentinel",)
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None


@pytest.mark.asyncio
async def test_provider_credential_integrity_mapping_does_not_mutate_source() -> None:
    session = FakeSession()
    actor = _admin(session)

    class ConstraintFailure(RuntimeError):
        constraint_name = "uq_provider_credentials_name"

    nested_cause = RuntimeError("integrity-cause-sentinel")
    original: ConstraintFailure
    try:
        raise ConstraintFailure("integrity-orig-sentinel") from nested_cause
    except ConstraintFailure as caught:
        original = caught
    original_state = (
        original.args,
        original.__traceback__,
        original.__cause__,
        original.__context__,
    )
    source = IntegrityError(
        "INSERT integrity-statement-sentinel",
        {"secret": "integrity-param-sentinel"},
        original,
    )
    source_state = (source.statement, source.params, source.args)
    session.credential_error = source

    with pytest.raises(BusinessError) as captured:
        await _service(session).create_credential(
            ProviderCredentialCreate(name="cloud", secret=SecretStr(SECRET)),
            actor=actor,
            request_id="req-integrity-source",
            idempotency_key="integrity-source-key",
        )

    assert (captured.value.status_code, captured.value.code) == (
        409,
        "RESOURCE_ALREADY_EXISTS",
    )
    assert (source.statement, source.params, source.args) == source_state
    assert (
        original.args,
        original.__traceback__,
        original.__cause__,
        original.__context__,
    ) == original_state
    assert nested_cause.args == ("integrity-cause-sentinel",)


class _HostileIntegrityError(IntegrityError):
    _hostile_attributes = frozenset({"constraint_name", "diag", "orig", "__cause__", "__context__"})

    def __init__(self) -> None:
        super().__init__("hostile statement", {}, RuntimeError("hostile original"))
        self._armed = True

    def __getattribute__(self, name: str) -> object:
        if name in object.__getattribute__(self, "_hostile_attributes") and object.__getattribute__(
            self,
            "__dict__",
        ).get("_armed", False):
            raise asyncio.CancelledError("hostile-accessor-sentinel")
        return super().__getattribute__(name)


@pytest.mark.asyncio
async def test_provider_credential_constraint_inspection_contains_hostile_accessors() -> None:
    session = FakeSession()
    actor = _admin(session)
    source = _HostileIntegrityError()
    session.credential_error = source
    captured: BusinessError | None = None
    leaked: BaseException | None = None
    try:
        await _service(session).create_credential(
            ProviderCredentialCreate(name="cloud", secret=SecretStr(SECRET)),
            actor=actor,
            request_id="req-hostile-integrity",
            idempotency_key="hostile-integrity-key",
        )
    except BusinessError as error:
        captured = error
    except BaseException as error:
        leaked = error
    finally:
        source._armed = False

    assert leaked is None
    assert captured is not None
    assert (captured.status_code, captured.code) == (500, "INTERNAL_ERROR")
    assert captured.__cause__ is None
    assert captured.__context__ is None
    _assert_production_traceback_scrubbed(
        captured,
        markers=("hostile-accessor-sentinel",),
    )


@pytest.mark.asyncio
async def test_provider_credential_replay_fails_closed_for_duplicate_snapshots() -> None:
    session = FakeSession()
    actor = _admin(session)
    service = _service(session)
    command = ProviderCredentialCreate(name="cloud", secret=SecretStr(SECRET))
    created = await service.create_credential(
        command,
        actor=actor,
        request_id="req-duplicate-snapshot-create",
        idempotency_key="duplicate-snapshot-key",
    )
    session.audits.append(deepcopy(session.audits[0]))

    with pytest.raises(BusinessError) as captured:
        await service.create_credential(
            command,
            actor=actor,
            request_id="req-duplicate-snapshot-replay",
            idempotency_key="duplicate-snapshot-key",
        )

    assert (captured.value.status_code, captured.value.code) == (500, "INTERNAL_ERROR")
    assert created.credential.id == session.audits[0].target_id


class _AuditRowsResult:
    def __init__(self, rows: list[tuple[object]]) -> None:
        self._rows = rows

    def all(self) -> list[tuple[object]]:
        return self._rows


class _AuditExecuteSession:
    def __init__(self, rows: list[tuple[object]]) -> None:
        self._rows = rows
        self.statement: object | None = None

    async def execute(self, statement: object) -> _AuditRowsResult:
        self.statement = statement
        return _AuditRowsResult(self._rows)


@pytest.mark.asyncio
async def test_audit_repository_snapshot_query_is_bounded_and_fails_closed_on_duplicates() -> None:
    session = _AuditExecuteSession([({"id": "first"},), ({"id": "second"},)])
    repository = SqlAlchemyProviderAuditRepository(cast(Any, session))

    result = await repository.get_created_response(uuid4(), uuid4())

    assert result is None
    assert session.statement is not None
    sql = str(session.statement).upper()
    assert "ORDER BY" in sql
    assert "LIMIT" in sql


def test_provider_credential_openapi_documents_success_headers_and_statuses() -> None:
    schema = create_app().openapi()
    paths = schema["paths"]
    collection = paths["/v1/admin/provider-credentials"]
    detail = paths["/v1/admin/provider-credentials/{credential_id}"]

    post_responses = collection["post"]["responses"]
    assert "404" not in post_responses
    for status in ("200", "201"):
        assert set(post_responses[status]["headers"]) >= {
            "ETag",
            "Location",
            "Cache-Control",
        }
    assert "Cache-Control" in collection["get"]["responses"]["200"]["headers"]
    for method in ("get", "patch"):
        assert set(detail[method]["responses"]["200"]["headers"]) >= {
            "ETag",
            "Cache-Control",
        }


def test_provider_config_schemas_enforce_mvp_protocol_and_persistence_bounds() -> None:
    credential_id = uuid4()
    compatible = _config_command(credential_id)
    assert compatible.timeout_seconds == Decimal("30.000")
    assert compatible.enabled is True

    openrouter = _config_command(
        credential_id,
        provider_type="openrouter",
        routing_options={
            "order": ["openai", "anthropic"],
            "allow_fallbacks": False,
            "data_collection": "deny",
            "quantizations": ["fp16"],
            "sort": {"by": "latency", "partition": "model"},
            "preferred_max_latency": {"p90": 200},
            "max_price": {"prompt": "0.25", "completion": "1.5"},
        },
    )
    assert openrouter.provider_type == "openrouter"

    invalid_payloads = (
        {"provider_type": "vendor_specific"},
        {"secret_ref": "env:UNSAFE"},
        {"routing_options": {"only": ["not allowed for compatible"]}},
        {"provider_type": "openrouter", "routing_options": {"unknown": True}},
        {"timeout_seconds": 0},
        {"timeout_seconds": "600.001"},
        {"max_concurrency": 0},
        {"max_concurrency": 10001},
        {"requests_per_minute": 0},
        {"requests_per_minute": 1000001},
    )
    base = _config_command(credential_id).model_dump(mode="python")
    for update in invalid_payloads:
        with pytest.raises(ValidationError):
            ProviderConfigCreate.model_validate({**base, **update})

    for patch in ({}, {"enabled": None}, {"provider_type": "vendor_specific"}):
        with pytest.raises(ValidationError):
            ProviderConfigPatch.model_validate(patch)


@pytest.mark.asyncio
async def test_provider_config_create_replays_lists_gets_and_patches_safely() -> None:
    session = FakeSession()
    actor = _admin(session)
    credential_id = _credential_record(session)
    policy = FakeEndpointPolicy()
    service = _config_service(session, policy)
    command = _config_command(credential_id)

    created = await service.create_provider_config(
        command,
        actor=actor,
        request_id="req-config-create",
        idempotency_key="config-create-key",
    )

    assert created.created is True
    safe = created.provider_config
    assert safe.base_url == CONFIG_CANONICAL_URL
    assert safe.credential_id == credential_id
    assert safe.resource_revision == 1
    assert safe.endpoint_policy_version == "provider-endpoint-v1"
    assert safe.endpoint_validated_at == NOW
    assert safe.default_headers == {"X-Title": "RAG"}
    assert policy.urls == [CONFIG_BASE_URL]
    assert session.config_create_count == 1
    assert session.audits[0].action == "provider_config.created"
    serialized = json.dumps(session.audits[0].metadata_)
    assert "secret_ref" not in serialized
    assert "ciphertext" not in serialized
    assert "nonce" not in serialized

    listed = await service.list_provider_configs()
    fetched = await service.get_provider_config(safe.id)
    assert listed.items == (safe,)
    assert fetched == safe

    replayed = await service.create_provider_config(
        command,
        actor=actor,
        request_id="req-config-replay",
        idempotency_key="config-create-key",
    )
    assert replayed.created is False
    assert replayed.provider_config == safe
    assert session.config_create_count == 1
    assert policy.urls == [CONFIG_BASE_URL]

    with pytest.raises(BusinessError) as reused:
        await service.create_provider_config(
            _config_command(credential_id, name="different config"),
            actor=actor,
            request_id="req-config-reused",
            idempotency_key="config-create-key",
        )
    assert (reused.value.status_code, reused.value.code) == (
        409,
        "IDEMPOTENCY_KEY_REUSED",
    )

    operational = await service.update_provider_config(
        safe.id,
        ProviderConfigPatch(timeout_seconds=Decimal("45.000"), enabled=False),
        actor=actor,
        request_id="req-config-operational",
        expected_etag=provider_config_etag(safe.id, 1),
    )
    assert operational.resource_revision == 2
    assert operational.enabled is False
    assert operational.timeout_seconds == Decimal("45.000")
    assert operational.endpoint_validated_at == safe.endpoint_validated_at
    assert policy.urls == [CONFIG_BASE_URL]

    changed_time = datetime(2026, 7, 27, 9, 0, tzinfo=UTC)
    service = _config_service(session, policy, clock=lambda: changed_time)
    changed = await service.update_provider_config(
        safe.id,
        ProviderConfigPatch(base_url="https://NEW.Provider.Example:443/v2/"),
        actor=actor,
        request_id="req-config-base-url",
        expected_etag=provider_config_etag(safe.id, 2),
    )
    assert changed.resource_revision == 3
    assert changed.base_url == "https://NEW.Provider.Example:443/v2"
    assert changed.endpoint_validated_at == changed_time
    assert policy.urls[-1] == "https://NEW.Provider.Example:443/v2/"


@pytest.mark.asyncio
async def test_provider_config_legacy_vendor_row_can_only_be_disabled() -> None:
    session = FakeSession()
    actor = _admin(session)
    credential_id = _credential_record(session)
    config_id = uuid4()
    validated_at = datetime(2026, 7, 20, 8, 0, tzinfo=UTC)
    session.configs[config_id] = ProviderConfigRecord(
        id=config_id,
        name="legacy vendor provider",
        provider_type="vendor_specific",
        base_url=CONFIG_CANONICAL_URL,
        credential_id=credential_id,
        default_headers={},
        routing_options={},
        timeout_seconds=Decimal("30"),
        max_concurrency=4,
        requests_per_minute=60,
        enabled=True,
        resource_revision=1,
        endpoint_policy_version="provider-endpoint-v1",
        endpoint_validated_at=validated_at,
        created_at=NOW,
        updated_at=NOW,
    )
    session.config_secret_sources[config_id] = ProviderConfigSecretSourceRecord(
        provider_config_id=config_id,
        credential_id=credential_id,
        secret_ref=None,
    )
    service = _config_service(session, FakeEndpointPolicy())

    disabled = await service.update_provider_config(
        config_id,
        ProviderConfigPatch(enabled=False),
        actor=actor,
        request_id="req-disable-legacy-vendor",
        expected_etag=provider_config_etag(config_id, 1),
    )

    assert disabled.enabled is False
    assert disabled.resource_revision == 2
    assert disabled.endpoint_validated_at == validated_at
    assert [event.action for event in session.audits] == ["provider_config.updated"]

    rejected_patches = (
        ProviderConfigPatch(enabled=True),
        ProviderConfigPatch(provider_type="openai_compatible"),
        ProviderConfigPatch(credential_id=credential_id),
        ProviderConfigPatch(base_url="https://replacement.example/v1"),
        ProviderConfigPatch(default_headers={"X-Title": "replacement"}),
        ProviderConfigPatch(routing_options={"preferred_min_throughput": 1}),
        ProviderConfigPatch(name="converted vendor provider"),
        ProviderConfigPatch(timeout_seconds=Decimal("45")),
        ProviderConfigPatch(max_concurrency=5),
        ProviderConfigPatch(requests_per_minute=61),
        ProviderConfigPatch(enabled=False, name="mixed legacy vendor patch"),
    )
    for index, patch in enumerate(rejected_patches):
        with pytest.raises(BusinessError) as rejected:
            await service.update_provider_config(
                config_id,
                patch,
                actor=actor,
                request_id=f"req-reject-legacy-vendor-{index}",
                expected_etag=provider_config_etag(config_id, 2),
            )
        assert (rejected.value.status_code, rejected.value.code) == (422, "VALIDATION_ERROR")

    persisted = session.configs[config_id]
    assert persisted.enabled is False
    assert persisted.resource_revision == 2
    assert persisted.endpoint_validated_at == validated_at
    assert [event.action for event in session.audits] == ["provider_config.updated"]


@pytest.mark.asyncio
async def test_provider_config_create_replays_validated_equivalent_numeric_inputs() -> None:
    session = FakeSession()
    actor = _admin(session)
    credential_id = _credential_record(session)
    service = _config_service(session, FakeEndpointPolicy())
    first = _config_command(
        credential_id,
        provider_type="openrouter",
        timeout_seconds=Decimal("30"),
        routing_options={
            "order": ["openai", "anthropic"],
            "preferred_max_latency": {"p90": 200},
            "preferred_min_throughput": 0,
        },
    )
    equivalent = _config_command(
        credential_id,
        provider_type="openrouter",
        timeout_seconds=Decimal("30.000"),
        routing_options={
            "order": ["openai", "anthropic"],
            "preferred_max_latency": {"p90": 200.0},
            "preferred_min_throughput": -0.0,
        },
    )

    created = await service.create_provider_config(
        first,
        actor=actor,
        request_id="req-config-canonical-number-create",
        idempotency_key="config-canonical-number",
    )
    replayed = await service.create_provider_config(
        equivalent,
        actor=actor,
        request_id="req-config-canonical-number-replay",
        idempotency_key="config-canonical-number",
    )

    assert created.created is True
    assert replayed.created is False
    assert replayed.provider_config == created.provider_config
    assert len(session.configs) == 1
    assert len(session.idempotency) == 1
    assert [event.action for event in session.audits] == ["provider_config.created"]

    for different_routing in (
        {
            "order": ["anthropic", "openai"],
            "preferred_max_latency": {"p90": 200},
            "preferred_min_throughput": 0,
        },
        {
            "order": ["openai", "anthropic"],
            "allow_fallbacks": None,
            "preferred_max_latency": {"p90": 200},
            "preferred_min_throughput": 0,
        },
    ):
        with pytest.raises(BusinessError) as reused:
            await service.create_provider_config(
                _config_command(
                    credential_id,
                    provider_type="openrouter",
                    timeout_seconds=Decimal("30.000"),
                    routing_options=different_routing,
                ),
                actor=actor,
                request_id="req-config-canonical-distinct",
                idempotency_key="config-canonical-number",
            )
        assert (reused.value.status_code, reused.value.code) == (
            409,
            "IDEMPOTENCY_KEY_REUSED",
        )


@pytest.mark.asyncio
async def test_provider_config_create_fingerprint_preserves_large_integer_identity() -> None:
    session = FakeSession()
    actor = _admin(session)
    credential_id = _credential_record(session)
    service = _config_service(session, FakeEndpointPolicy())
    first_value = 12345678901234567890123456781
    second_value = 12345678901234567890123456782

    created = await service.create_provider_config(
        _config_command(
            credential_id,
            provider_type="openrouter",
            routing_options={"preferred_min_throughput": first_value},
        ),
        actor=actor,
        request_id="req-config-large-number-create",
        idempotency_key="config-large-number",
    )

    with pytest.raises(BusinessError) as reused:
        await service.create_provider_config(
            _config_command(
                credential_id,
                provider_type="openrouter",
                routing_options={"preferred_min_throughput": second_value},
            ),
            actor=actor,
            request_id="req-config-large-number-conflict",
            idempotency_key="config-large-number",
        )

    assert created.created is True
    assert (reused.value.status_code, reused.value.code) == (
        409,
        "IDEMPOTENCY_KEY_REUSED",
    )
    assert len(session.configs) == 1
    assert next(iter(session.configs.values())).routing_options == {
        "preferred_min_throughput": first_value
    }
    assert len(session.idempotency) == 1
    assert [event.action for event in session.audits] == ["provider_config.created"]


@pytest.mark.asyncio
async def test_provider_config_cursor_pagination_is_stable_and_rejects_invalid_cursor() -> None:
    session = FakeSession()
    actor = _admin(session)
    credential_id = _credential_record(session)
    service = _config_service(session, FakeEndpointPolicy())
    created_ids: set[UUID] = set()
    for index in range(5):
        result = await service.create_provider_config(
            _config_command(credential_id, name=f"paginated config {index}"),
            actor=actor,
            request_id=f"req-config-pagination-{index}",
            idempotency_key=f"config-pagination-{index}",
        )
        created_ids.add(result.provider_config.id)

    cursor: str | None = None
    traversed: list[UUID] = []
    while True:
        page = await service.list_provider_configs(cursor=cursor, limit=2)
        traversed.extend(item.id for item in page.items)
        if page.next_cursor is None:
            break
        cursor = page.next_cursor

    expected = [
        row.id
        for row in sorted(
            session.configs.values(),
            key=lambda row: (row.created_at, row.id),
        )
    ]
    assert traversed == expected
    assert set(traversed) == created_ids
    assert len(traversed) == len(set(traversed)) == 5

    with pytest.raises(BusinessError) as invalid:
        await service.list_provider_configs(cursor="not-a-canonical-cursor", limit=2)
    assert (invalid.value.status_code, invalid.value.code) == (422, "VALIDATION_ERROR")


@pytest.mark.asyncio
async def test_provider_config_rejects_invalid_inputs_before_mutation() -> None:
    session = FakeSession()
    actor = _admin(session)
    policy = FakeEndpointPolicy()
    service = _config_service(session, policy)

    with pytest.raises(BusinessError) as missing:
        await service.create_provider_config(
            _config_command(uuid4()),
            actor=actor,
            request_id="req-config-missing-credential",
            idempotency_key="config-missing-credential",
        )
    assert (missing.value.status_code, missing.value.code) == (404, "RESOURCE_NOT_FOUND")
    assert session.configs == {}
    assert session.idempotency == {}
    assert session.audits == []

    credential_id = _credential_record(session)
    with pytest.raises(BusinessError) as forbidden_header:
        await service.create_provider_config(
            _config_command(
                credential_id,
                default_headers={"Authorization": "Bearer unsafe-header-sentinel"},
            ),
            actor=actor,
            request_id="req-config-header",
            idempotency_key="config-header",
        )
    assert forbidden_header.value.code == "PROVIDER_ENDPOINT_REJECTED"
    assert session.configs == {}
    assert session.idempotency == {}
    assert session.audits == []

    with pytest.raises(ValidationError):
        _config_command(
            credential_id,
            provider_type="openrouter",
            routing_options={"unknown": True},
        )

    policy.reject = True
    with pytest.raises(BusinessError) as endpoint:
        await service.create_provider_config(
            _config_command(credential_id),
            actor=actor,
            request_id="req-config-endpoint",
            idempotency_key="config-endpoint",
        )
    assert endpoint.value.code == "PROVIDER_ENDPOINT_REJECTED"
    assert session.configs == {}
    assert session.idempotency == {}
    assert session.audits == []


@pytest.mark.asyncio
async def test_provider_config_patch_enforces_etag_admin_and_credential() -> None:
    session = FakeSession()
    actor = _admin(session)
    first_credential_id = _credential_record(session, name="first")
    second_credential_id = _credential_record(session, name="second")
    policy = FakeEndpointPolicy()
    service = _config_service(session, policy)
    created = await service.create_provider_config(
        _config_command(first_credential_id),
        actor=actor,
        request_id="req-config-patch-seed",
        idempotency_key="config-patch-seed",
    )
    config_id = created.provider_config.id

    with pytest.raises(BusinessError) as missing_etag:
        await service.update_provider_config(
            config_id,
            ProviderConfigPatch(enabled=False),
            actor=actor,
            request_id="req-config-no-etag",
            expected_etag=None,
        )
    assert (missing_etag.value.status_code, missing_etag.value.code) == (
        428,
        "PRECONDITION_REQUIRED",
    )

    with pytest.raises(BusinessError) as stale:
        await service.update_provider_config(
            config_id,
            ProviderConfigPatch(enabled=False),
            actor=actor,
            request_id="req-config-stale",
            expected_etag=provider_config_etag(config_id, 99),
        )
    assert (stale.value.status_code, stale.value.code) == (412, "PRECONDITION_FAILED")

    with pytest.raises(BusinessError) as missing_credential:
        await service.update_provider_config(
            config_id,
            ProviderConfigPatch(credential_id=uuid4()),
            actor=actor,
            request_id="req-config-new-credential-missing",
            expected_etag=provider_config_etag(config_id, 1),
        )
    assert (missing_credential.value.status_code, missing_credential.value.code) == (
        404,
        "RESOURCE_NOT_FOUND",
    )
    assert session.configs[config_id].resource_revision == 1
    assert session.configs[config_id].credential_id == first_credential_id

    updated = await service.update_provider_config(
        config_id,
        ProviderConfigPatch(credential_id=second_credential_id),
        actor=actor,
        request_id="req-config-new-credential",
        expected_etag=provider_config_etag(config_id, 1),
    )
    assert updated.credential_id == second_credential_id

    session.admins[actor.key_id] = AdminActorRecord(
        id=actor.key_id,
        public_id=actor.public_id,
        key_type="admin",
        status="revoked",
        not_before=None,
        expires_at=None,
        revoked_at=NOW,
    )
    with pytest.raises(BusinessError) as revoked:
        await service.update_provider_config(
            config_id,
            ProviderConfigPatch(enabled=True),
            actor=actor,
            request_id="req-config-revoked",
            expected_etag=provider_config_etag(config_id, 2),
        )
    assert (revoked.value.status_code, revoked.value.code) == (401, "INVALID_API_KEY")
    assert session.configs[config_id].enabled is updated.enabled is True


def test_provider_config_legacy_source_selection_is_explicit_and_never_falls_back() -> None:
    provider_config_id = uuid4()
    credential_id = uuid4()
    locator = "env:LEGACY_PROVIDER_KEY_REPR_SENTINEL"
    legacy_record = ProviderConfigSecretSourceRecord(
        provider_config_id=provider_config_id,
        credential_id=None,
        secret_ref=locator,
    )
    assert repr(legacy_record) == "ProviderConfigSecretSourceRecord(<redacted>)"
    assert locator not in repr(legacy_record)
    credential_source = select_provider_config_secret_source(
        ProviderConfigSecretSourceRecord(
            provider_config_id=provider_config_id,
            credential_id=credential_id,
            secret_ref=None,
        )
    )
    assert credential_source == CredentialProviderSecretSource(credential_id=credential_id)

    legacy_source = select_provider_config_secret_source(legacy_record)
    assert legacy_source == LegacyProviderSecretSource(secret_ref=locator)

    for invalid in (
        ProviderConfigSecretSourceRecord(provider_config_id, None, None),
        ProviderConfigSecretSourceRecord(
            provider_config_id,
            credential_id,
            "env:MUST_NOT_BE_A_FALLBACK",
        ),
    ):
        with pytest.raises(BusinessError) as captured:
            select_provider_config_secret_source(invalid)
        assert captured.value.code == "INTERNAL_ERROR"


class _SecretSourceExecuteSession:
    def __init__(self, source: BaseException) -> None:
        self._source = source
        self.statement: object | None = None

    async def execute(self, statement: object) -> object:
        self.statement = statement
        raise self._source


def _caller_owned_failure(prefix: str) -> tuple[RuntimeError, RuntimeError, RuntimeError]:
    cause = RuntimeError(f"{prefix}-cause-sensitive")
    context = RuntimeError(f"{prefix}-context-sensitive")
    source = RuntimeError(f"{prefix}-args-sensitive")
    source.__cause__ = cause
    source.__context__ = context
    return source, cause, context


def _assert_caller_owned_failure_unchanged(
    source: RuntimeError,
    cause: RuntimeError,
    context: RuntimeError,
    prefix: str,
) -> None:
    assert source.args == (f"{prefix}-args-sensitive",)
    assert source.__cause__ is cause
    assert source.__context__ is context
    assert cause.args == (f"{prefix}-cause-sensitive",)
    assert context.args == (f"{prefix}-context-sensitive",)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ("cancelled", "ordinary"))
async def test_provider_config_secret_source_repository_scrubs_exception_boundaries(
    failure_kind: str,
) -> None:
    prefix = f"secret-source-repository-{failure_kind}-locator"
    source: BaseException
    if failure_kind == "cancelled":
        source, cause, context = _caller_owned_cancellation(prefix)
    else:
        source, cause, context = _caller_owned_failure(prefix)
    session = _SecretSourceExecuteSession(source)
    repository = SqlAlchemyProviderConfigRepository(cast(Any, session))

    expected_exception = asyncio.CancelledError if failure_kind == "cancelled" else RuntimeError
    with pytest.raises(expected_exception) as captured:
        await repository.get_secret_source(uuid4())

    if failure_kind == "cancelled":
        assert isinstance(source, asyncio.CancelledError)
        _assert_caller_owned_cancellation_unchanged(source, cause, context, prefix)
    else:
        assert isinstance(source, RuntimeError)
        _assert_caller_owned_failure_unchanged(source, cause, context, prefix)
    assert captured.value is not source
    assert captured.value.args == ()
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert session.statement is not None
    markers = (
        f"{prefix}-args-sensitive",
        f"{prefix}-cause-sensitive",
        f"{prefix}-context-sensitive",
    )
    _assert_production_frame_locals_scrubbed(
        source,
        markers=markers,
        forbidden_objects=(session.statement,),
    )
    _assert_production_traceback_scrubbed(
        captured.value,
        markers=markers,
        forbidden_objects=(session.statement,),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ("cancelled", "ordinary"))
async def test_provider_config_secret_source_service_scrubs_exception_boundaries(
    failure_kind: str,
) -> None:
    session = FakeSession()
    prefix = f"secret-source-service-{failure_kind}-locator"
    source: BaseException
    if failure_kind == "cancelled":
        source, cause, context = _caller_owned_cancellation(prefix)
    else:
        source, cause, context = _caller_owned_failure(prefix)
    session.config_error = source
    service = _config_service(session, FakeEndpointPolicy())

    expected_exception = asyncio.CancelledError if failure_kind == "cancelled" else BusinessError
    with pytest.raises(expected_exception) as captured:
        await service.get_secret_source(uuid4())

    if failure_kind == "cancelled":
        assert isinstance(source, asyncio.CancelledError)
        _assert_caller_owned_cancellation_unchanged(source, cause, context, prefix)
        assert captured.value.args == ()
    else:
        assert isinstance(source, RuntimeError)
        _assert_caller_owned_failure_unchanged(source, cause, context, prefix)
        assert isinstance(captured.value, BusinessError)
        assert (captured.value.status_code, captured.value.code) == (500, "INTERNAL_ERROR")
    assert captured.value is not source
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    markers = (
        f"{prefix}-args-sensitive",
        f"{prefix}-cause-sensitive",
        f"{prefix}-context-sensitive",
    )
    _assert_production_frame_locals_scrubbed(source, markers=markers)
    _assert_production_traceback_scrubbed(captured.value, markers=markers)


@pytest.mark.asyncio
async def test_provider_config_service_scrubs_cancelled_write_tracebacks() -> None:
    session = FakeSession()
    actor = _admin(session)
    credential_id = _credential_record(session)
    policy = FakeEndpointPolicy()
    command = _config_command(
        credential_id,
        name="config-cancel-name-sentinel",
        base_url="https://config-cancel-host-sentinel.example/v1",
        default_headers={"X-Title": "config-cancel-header-sentinel"},
    )
    source, cause, context = _caller_owned_cancellation("provider-config-service")
    session.config_error = source

    with pytest.raises(asyncio.CancelledError) as captured:
        await _config_service(session, policy).create_provider_config(
            command,
            actor=actor,
            request_id="req-config-cancel",
            idempotency_key="config-cancel-key",
        )

    _assert_caller_owned_cancellation_unchanged(
        source,
        cause,
        context,
        "provider-config-service",
    )
    assert captured.value is not source
    assert captured.value.args == ()
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    markers = (
        "config-cancel-name-sentinel",
        "config-cancel-host-sentinel",
        "config-cancel-header-sentinel",
        "provider-config-service-args-sensitive",
        "provider-config-service-cause-sensitive",
        "provider-config-service-context-sensitive",
        "statement-param-sensitive",
    )
    _assert_production_frame_locals_scrubbed(
        source,
        markers=markers,
        forbidden_objects=(command,),
    )
    _assert_production_traceback_scrubbed(
        captured.value,
        markers=markers,
        forbidden_objects=(command,),
    )


class _CancellingProviderConfigService:
    def __init__(self, source: asyncio.CancelledError) -> None:
        self._source = source

    async def create_provider_config(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise self._source

    async def update_provider_config(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise self._source


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("create", "update"))
async def test_provider_config_routes_scrub_cancelled_command_tracebacks(
    operation: str,
) -> None:
    session = FakeSession()
    actor = _admin(session)
    credential_id = _credential_record(session)
    command: ProviderConfigCreate | ProviderConfigPatch
    source, cause, context = _caller_owned_cancellation(f"provider-config-route-{operation}")
    service = cast(ProviderConfigService, _CancellingProviderConfigService(source))
    if operation == "create":
        command = _config_command(
            credential_id,
            name="route-config-create-name-sentinel",
            base_url="https://route-config-create-host-sentinel.example/v1",
        )
        invocation = create_provider_config(
            command,
            "req-route-config-create",
            actor,
            service,
            "route-config-create-key",
        )
        markers = (
            "route-config-create-name-sentinel",
            "route-config-create-host-sentinel",
        )
    else:
        command = ProviderConfigPatch(
            name="route-config-update-name-sentinel",
            base_url="https://route-config-update-host-sentinel.example/v1",
        )
        invocation = update_provider_config(
            uuid4(),
            command,
            "req-route-config-update",
            actor,
            service,
            '"provider-config:unused:r1"',
        )
        markers = (
            "route-config-update-name-sentinel",
            "route-config-update-host-sentinel",
        )

    with pytest.raises(asyncio.CancelledError) as captured:
        await invocation

    _assert_caller_owned_cancellation_unchanged(
        source,
        cause,
        context,
        f"provider-config-route-{operation}",
    )
    assert captured.value is not source
    assert captured.value.args == ()
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    sensitive_markers = (
        *markers,
        f"provider-config-route-{operation}-args-sensitive",
        f"provider-config-route-{operation}-cause-sensitive",
        f"provider-config-route-{operation}-context-sensitive",
        "statement-param-sensitive",
    )
    _assert_production_frame_locals_scrubbed(
        source,
        markers=sensitive_markers,
        forbidden_objects=(command,),
    )
    _assert_production_traceback_scrubbed(
        captured.value,
        markers=sensitive_markers,
        forbidden_objects=(command,),
    )


def test_provider_config_openapi_documents_safe_headers_and_excludes_secret_ref() -> None:
    schema = create_app().openapi()
    paths = schema["paths"]
    collection = paths["/v1/admin/provider-configs"]
    detail = paths["/v1/admin/provider-configs/{provider_config_id}"]
    for status in ("200", "201"):
        assert set(collection["post"]["responses"][status]["headers"]) >= {
            "ETag",
            "Location",
            "Cache-Control",
        }
    assert "Cache-Control" in collection["get"]["responses"]["200"]["headers"]
    for method in ("get", "patch"):
        assert set(detail[method]["responses"]["200"]["headers"]) >= {
            "ETag",
            "Cache-Control",
        }
    safe_schema = schema["components"]["schemas"]["SafeProviderConfig"]
    assert "secret_ref" not in safe_schema.get("properties", {})
    assert "ciphertext" not in json.dumps(safe_schema).lower()
    assert "nonce" not in json.dumps(safe_schema).lower()


def test_provider_config_openapi_truthfully_documents_writable_contract() -> None:
    schemas = create_app().openapi()["components"]["schemas"]
    create_schema = schemas["ProviderConfigCreate"]
    patch_schema = schemas["ProviderConfigPatch"]
    create_properties = create_schema["properties"]
    patch_properties = patch_schema["properties"]

    assert create_properties["provider_type"]["enum"] == [
        "openai_compatible",
        "openrouter",
    ]
    assert patch_properties["provider_type"]["enum"] == [
        "openai_compatible",
        "openrouter",
    ]
    assert "secret_ref" not in create_properties
    assert "secret_ref" not in patch_properties

    allowed_headers = {
        "HTTP-Referer": 2048,
        "X-OpenRouter-Title": 120,
        "X-Title": 120,
    }
    header_schema = create_properties["default_headers"]
    assert header_schema["additionalProperties"] is False
    assert header_schema["maxProperties"] == 3
    assert set(header_schema["properties"]) == set(allowed_headers)
    for header, maximum in allowed_headers.items():
        assert header_schema["properties"][header]["type"] == "string"
        assert header_schema["properties"][header]["minLength"] == 1
        assert header_schema["properties"][header]["maxLength"] == maximum
    assert patch_properties["default_headers"] == header_schema

    routing_schema = create_properties["routing_options"]
    expected_routing_keys = {
        "order",
        "allow_fallbacks",
        "require_parameters",
        "data_collection",
        "zdr",
        "enforce_distillable_text",
        "only",
        "ignore",
        "quantizations",
        "sort",
        "preferred_min_throughput",
        "preferred_max_latency",
        "max_price",
    }
    assert routing_schema["additionalProperties"] is False
    assert routing_schema["maxProperties"] == len(expected_routing_keys)
    assert set(routing_schema["properties"]) == expected_routing_keys
    assert routing_schema["properties"]["order"]["maxItems"] == 100
    assert routing_schema["properties"]["quantizations"]["maxItems"] == 32
    assert set(routing_schema["properties"]["quantizations"]["items"]["enum"]) == {
        "int4",
        "int8",
        "fp4",
        "fp6",
        "fp8",
        "fp16",
        "bf16",
        "fp32",
        "unknown",
    }
    assert routing_schema["properties"]["data_collection"]["enum"] == [
        "allow",
        "deny",
        None,
    ]
    sort_schema = routing_schema["properties"]["sort"]
    assert sort_schema["additionalProperties"] is False
    assert {branch["type"] for branch in sort_schema["anyOf"]} == {
        "string",
        "object",
        "null",
    }
    assert sort_schema["properties"]["partition"]["enum"] == ["model", "none", None]
    percentile_schema = routing_schema["properties"]["preferred_max_latency"]
    assert percentile_schema["additionalProperties"] is False
    assert {branch["type"] for branch in percentile_schema["anyOf"]} == {
        "number",
        "object",
        "null",
    }
    assert set(percentile_schema["properties"]) == {"p50", "p75", "p90", "p99"}
    for member in percentile_schema["properties"].values():
        assert member["type"] == ["number", "null"]
        assert member["minimum"] == 0
    max_price_schema = routing_schema["properties"]["max_price"]
    assert max_price_schema["additionalProperties"] is False
    assert set(max_price_schema["properties"]) == {
        "audio",
        "prompt",
        "completion",
        "request",
        "image",
    }
    for price in max_price_schema["properties"].values():
        assert price["type"] == "string"
        assert price["minLength"] == 1
        assert price["maxLength"] == 64
        assert price["pattern"]
    assert patch_properties["routing_options"] == routing_schema
    for bounded_field in (
        "timeout_seconds",
        "max_concurrency",
        "requests_per_minute",
    ):
        assert patch_properties[bounded_field] == create_properties[bounded_field]

    for property_schema in patch_properties.values():
        property_type = property_schema.get("type")
        assert property_type != "null"
        assert not isinstance(property_type, list) or "null" not in property_type
        assert None not in property_schema.get("enum", [])
        assert all(branch.get("type") != "null" for branch in property_schema.get("anyOf", []))


def test_model_profile_schemas_accept_rerank_without_a_dimension() -> None:
    provider_config_id = uuid4()

    profile = _profile_command(provider_config_id, capability="rerank", dimension=None)

    assert profile.capability == "rerank"
    assert profile.dimension is None


def test_model_profile_schemas_tie_dimension_to_capability() -> None:
    provider_config_id = uuid4()
    payload = _profile_command(provider_config_id).model_dump(mode="python")

    # Mirrors ck_model_profiles_dimension: a reranker scores pairs and has no
    # vector width, while an embedding profile without one cannot size a
    # collection. Rejecting both here keeps the API from writing a row the
    # database would refuse.
    for update in ({"capability": "rerank"}, {"capability": "embedding", "dimension": None}):
        with pytest.raises(ValidationError):
            ModelProfileCreate.model_validate({**payload, **update})


def test_model_profile_schemas_are_embedding_only_bounded_and_empty_vector_only() -> None:
    create_type = ModelProfileCreate
    patch_type = ModelProfilePatch
    provider_config_id = uuid4()
    valid = _profile_command(provider_config_id)

    assert valid.capability == "embedding"
    assert valid.dimension == 1536
    assert valid.timeout_seconds == Decimal("30.000")
    assert valid.vector_config == {}

    required = {
        "capability",
        "provider_config_id",
        "name",
        "model_name",
        "dimension",
        "max_input_tokens",
        "batch_size",
        "timeout_seconds",
        "vector_config",
        "enabled",
    }
    payload = valid.model_dump(mode="python")
    for field_name in required:
        with pytest.raises(ValidationError):
            create_type.model_validate(
                {key: value for key, value in payload.items() if key != field_name}
            )

    invalid_creates = (
        # `rerank` is accepted now, but only paired with a null dimension; the
        # capability/dimension pairing has its own test above.
        {"capability": "rerank"},
        {"capability": "chat"},
        {"capability": "chat", "dimension": None},
        {"dimension": 0},
        {"dimension": None},
        {"max_input_tokens": 0},
        {"max_input_tokens": 10_000_001},
        {"batch_size": 0},
        {"batch_size": 10_001},
        {"timeout_seconds": 0},
        {"timeout_seconds": "600.001"},
        {"vector_config": {"normalize": True}},
        {"unknown": "forbidden"},
    )
    for update in invalid_creates:
        with pytest.raises(ValidationError):
            create_type.model_validate({**payload, **update})

    for invalid_patch in (
        {},
        {"name": None},
        {"provider_config_id": None},
        {"model_name": None},
        {"dimension": None},
        {"max_input_tokens": None},
        {"batch_size": None},
        {"timeout_seconds": None},
        {"vector_config": None},
        {"enabled": None},
        {"vector_config": {"normalize": True}},
        {"capability": "embedding"},
    ):
        with pytest.raises(ValidationError):
            patch_type.model_validate(invalid_patch)

    assert patch_type.model_validate({"vector_config": {}}).vector_config == {}


@pytest.mark.asyncio
async def test_model_profile_create_replay_patch_and_vector_immutability() -> None:
    session = FakeSession()
    actor = _admin(session)
    provider_config_id = _config_record(session)
    service = _profile_service(session)
    patch_type = ModelProfilePatch
    etag = model_profile_etag

    created = await service.create_model_profile(
        _profile_command(provider_config_id),
        actor=actor,
        request_id="req-model-profile-create",
        idempotency_key="model-profile-create-key",
    )

    assert created.created is True
    safe = created.model_profile
    assert safe.capability == "embedding"
    assert safe.provider_config_id == provider_config_id
    assert safe.vector_config == {}
    assert safe.resource_revision == 1
    assert session.profile_create_count == 1
    assert session.audits[0].action == "model_profile.created"
    assert "secret" not in json.dumps(session.audits[0].metadata_).lower()

    replayed = await service.create_model_profile(
        _profile_command(provider_config_id, timeout_seconds=30.0),
        actor=actor,
        request_id="req-model-profile-replay",
        idempotency_key="model-profile-create-key",
    )
    assert replayed.created is False
    assert replayed.model_profile == safe
    assert session.profile_create_count == 1

    with pytest.raises(BusinessError) as reused:
        await service.create_model_profile(
            _profile_command(provider_config_id, model_name="different-model"),
            actor=actor,
            request_id="req-model-profile-conflict",
            idempotency_key="model-profile-create-key",
        )
    assert (reused.value.status_code, reused.value.code) == (409, "IDEMPOTENCY_KEY_REUSED")

    listed = await service.list_model_profiles()
    fetched = await service.get_model_profile(safe.id)
    assert listed.items == (safe,)
    assert fetched == safe

    disabled_provider_id = _config_record(session, enabled=False, name="disabled profile provider")
    with pytest.raises(BusinessError) as disabled_create:
        await service.create_model_profile(
            _profile_command(disabled_provider_id, name="disabled provider profile"),
            actor=actor,
            request_id="req-model-profile-disabled-provider",
            idempotency_key="model-profile-disabled-provider",
        )
    assert (disabled_create.value.status_code, disabled_create.value.code) == (
        409,
        "PROVIDER_CONFIG_DISABLED",
    )

    session.configs[provider_config_id] = replace(
        session.configs[provider_config_id], enabled=False
    )
    disabled = await service.update_model_profile(
        safe.id,
        patch_type(enabled=False),
        actor=actor,
        request_id="req-model-profile-disable",
        expected_etag=etag(safe.id, 1),
    )
    assert disabled.enabled is False
    assert disabled.resource_revision == 2

    with pytest.raises(BusinessError) as enable_disabled_provider:
        await service.update_model_profile(
            safe.id,
            patch_type(enabled=True),
            actor=actor,
            request_id="req-model-profile-enable-disabled-provider",
            expected_etag=etag(safe.id, 2),
        )
    assert (enable_disabled_provider.value.status_code, enable_disabled_provider.value.code) == (
        409,
        "PROVIDER_CONFIG_DISABLED",
    )
    assert session.profiles[safe.id].resource_revision == 2

    session.configs[provider_config_id] = replace(session.configs[provider_config_id], enabled=True)
    session.generation_profile_ids.add(safe.id)
    no_op = await service.update_model_profile(
        safe.id,
        patch_type(
            provider_config_id=provider_config_id,
            model_name=safe.model_name,
            dimension=safe.dimension,
            max_input_tokens=safe.max_input_tokens,
            vector_config={},
        ),
        actor=actor,
        request_id="req-model-profile-semantic-noop",
        expected_etag=etag(safe.id, 2),
    )
    assert no_op.resource_revision == 3

    other_provider_config_id = _config_record(session, name="other profile provider")
    semantic_patches = (
        patch_type(provider_config_id=other_provider_config_id),
        patch_type(model_name="text-embedding-3-large"),
        patch_type(dimension=3072),
        patch_type(max_input_tokens=8192),
    )
    for index, patch in enumerate(semantic_patches):
        with pytest.raises(BusinessError) as immutable:
            await service.update_model_profile(
                safe.id,
                patch,
                actor=actor,
                request_id=f"req-model-profile-immutable-{index}",
                expected_etag=etag(safe.id, 3),
            )
        assert (immutable.value.status_code, immutable.value.code) == (
            409,
            "IMMUTABLE_INDEX_CONFIGURATION",
        )
        assert session.profiles[safe.id].resource_revision == 3

    session.profiles[safe.id] = replace(
        session.profiles[safe.id],
        vector_config={"legacy_protocol": "v1"},
    )
    with pytest.raises(BusinessError) as immutable_vector:
        await service.update_model_profile(
            safe.id,
            patch_type(vector_config={}),
            actor=actor,
            request_id="req-model-profile-immutable-vector",
            expected_etag=etag(safe.id, 3),
        )
    assert (immutable_vector.value.status_code, immutable_vector.value.code) == (
        409,
        "IMMUTABLE_INDEX_CONFIGURATION",
    )

    session.profiles[safe.id] = replace(session.profiles[safe.id], vector_config={})
    operational = await service.update_model_profile(
        safe.id,
        patch_type(
            name="operationally tuned profile",
            timeout_seconds=Decimal("45"),
            batch_size=32,
            enabled=True,
        ),
        actor=actor,
        request_id="req-model-profile-operational",
        expected_etag=etag(safe.id, 3),
    )
    assert operational.resource_revision == 4
    assert operational.name == "operationally tuned profile"
    assert operational.timeout_seconds == Decimal("45")
    assert operational.batch_size == 32
    assert operational.enabled is True


@pytest.mark.asyncio
async def test_model_profile_rejects_legacy_vendor_provider_without_residue() -> None:
    session = FakeSession()
    actor = _admin(session)
    service = _profile_service(session)
    vendor_provider_id = _config_record(
        session,
        name="legacy vendor profile provider",
        provider_type="vendor_specific",
    )

    with pytest.raises(BusinessError) as create_rejected:
        await service.create_model_profile(
            _profile_command(vendor_provider_id, name="rejected vendor profile"),
            actor=actor,
            request_id="req-rejected-vendor-profile-create",
            idempotency_key="rejected-vendor-profile-create",
        )
    assert (create_rejected.value.status_code, create_rejected.value.code) == (
        422,
        "VALIDATION_ERROR",
    )
    assert session.profiles == {}
    assert session.profile_create_count == 0
    assert session.idempotency == {}
    assert session.audits == []

    supported_provider_id = _config_record(session, name="supported profile provider")
    created = await service.create_model_profile(
        _profile_command(supported_provider_id, name="supported embedding profile"),
        actor=actor,
        request_id="req-supported-profile-create",
        idempotency_key="supported-profile-create",
    )
    profile_id = created.model_profile.id
    create_audit_ids = [event.id for event in session.audits]
    create_idempotency_keys = set(session.idempotency)
    before_provider_change = session.profiles[profile_id]

    with pytest.raises(BusinessError) as provider_change_rejected:
        await service.update_model_profile(
            profile_id,
            ModelProfilePatch(provider_config_id=vendor_provider_id),
            actor=actor,
            request_id="req-rejected-vendor-profile-provider-change",
            expected_etag=model_profile_etag(profile_id, 1),
        )
    assert (provider_change_rejected.value.status_code, provider_change_rejected.value.code) == (
        422,
        "VALIDATION_ERROR",
    )
    assert session.profiles[profile_id] == before_provider_change
    assert [event.id for event in session.audits] == create_audit_ids
    assert set(session.idempotency) == create_idempotency_keys

    session.profiles[profile_id] = replace(
        session.profiles[profile_id],
        provider_config_id=vendor_provider_id,
        enabled=False,
    )
    before_enable = session.profiles[profile_id]
    with pytest.raises(BusinessError) as enable_rejected:
        await service.update_model_profile(
            profile_id,
            ModelProfilePatch(enabled=True),
            actor=actor,
            request_id="req-rejected-vendor-profile-enable",
            expected_etag=model_profile_etag(profile_id, 1),
        )
    assert (enable_rejected.value.status_code, enable_rejected.value.code) == (
        422,
        "VALIDATION_ERROR",
    )
    assert session.profiles[profile_id] == before_enable
    assert [event.id for event in session.audits] == create_audit_ids
    assert set(session.idempotency) == create_idempotency_keys


@pytest.mark.asyncio
async def test_immutable_provider_config_blocks_semantics_before_dns() -> None:
    session = FakeSession()
    actor = _admin(session)
    provider_config_id = _config_record(session)
    session.configs[provider_config_id] = replace(
        session.configs[provider_config_id],
        provider_type="openrouter",
        routing_options={"allow_fallbacks": True},
    )
    profile_service = _profile_service(session)
    profile = await profile_service.create_model_profile(
        _profile_command(provider_config_id),
        actor=actor,
        request_id="req-provider-immutable-profile",
        idempotency_key="provider-immutable-profile",
    )
    session.generation_profile_ids.add(profile.model_profile.id)
    policy = FakeEndpointPolicy()
    service = _config_service(session, policy)

    semantic_patches = (
        ProviderConfigPatch(provider_type="openai_compatible", routing_options={}),
        ProviderConfigPatch(base_url="https://different.example/v1"),
        ProviderConfigPatch(default_headers={"X-Title": "different"}),
        ProviderConfigPatch(routing_options={"allow_fallbacks": False}),
    )
    for index, patch in enumerate(semantic_patches):
        with pytest.raises(BusinessError) as immutable:
            await service.update_provider_config(
                provider_config_id,
                patch,
                actor=actor,
                request_id=f"req-provider-config-immutable-{index}",
                expected_etag=provider_config_etag(provider_config_id, 1),
            )
        assert (immutable.value.status_code, immutable.value.code) == (
            409,
            "IMMUTABLE_INDEX_CONFIGURATION",
        )
        assert session.configs[provider_config_id].resource_revision == 1
    assert policy.urls == []

    replacement_credential_id = _credential_record(session, name="rotated provider credential")
    updated = await service.update_provider_config(
        provider_config_id,
        ProviderConfigPatch(
            name="operational provider",
            credential_id=replacement_credential_id,
            timeout_seconds=Decimal("45"),
            max_concurrency=16,
            requests_per_minute=1200,
            enabled=False,
        ),
        actor=actor,
        request_id="req-provider-config-operational-referenced",
        expected_etag=provider_config_etag(provider_config_id, 1),
    )
    assert updated.resource_revision == 2
    assert updated.credential_id == replacement_credential_id
    assert updated.timeout_seconds == Decimal("45")
    assert updated.max_concurrency == 16
    assert updated.requests_per_minute == 1200
    assert updated.enabled is False


class _CancellingModelProfileService:
    def __init__(self, source: asyncio.CancelledError) -> None:
        self._source = source

    async def create_model_profile(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise self._source

    async def update_model_profile(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise self._source


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("create", "update"))
async def test_model_profile_routes_scrub_cancelled_request_objects(operation: str) -> None:
    session = FakeSession()
    actor = _admin(session)
    provider_config_id = _config_record(session)
    source, cause, context = _caller_owned_cancellation(f"model-profile-route-{operation}")
    service = _CancellingModelProfileService(source)
    command: ModelProfileCreate | ModelProfilePatch
    if operation == "create":
        command = _profile_command(
            provider_config_id,
            name="model-profile-create-name-sentinel",
            model_name="model-profile-create-model-sentinel",
        )
        invocation = provider_routes.create_model_profile(
            command,
            "req-model-profile-route-create",
            actor,
            cast(ModelProfileService, service),
            "model-profile-route-create-key",
        )
        markers = (
            "model-profile-create-name-sentinel",
            "model-profile-create-model-sentinel",
        )
    else:
        patch_type = ModelProfilePatch
        command = patch_type(
            name="model-profile-update-name-sentinel",
            model_name="model-profile-update-model-sentinel",
        )
        invocation = provider_routes.update_model_profile(
            uuid4(),
            command,
            "req-model-profile-route-update",
            actor,
            cast(ModelProfileService, service),
            '"model-profile:unused:r1"',
        )
        markers = (
            "model-profile-update-name-sentinel",
            "model-profile-update-model-sentinel",
        )

    with pytest.raises(asyncio.CancelledError) as captured:
        await invocation

    _assert_caller_owned_cancellation_unchanged(
        source,
        cause,
        context,
        f"model-profile-route-{operation}",
    )
    assert captured.value is not source
    assert captured.value.args == ()
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    sensitive_markers = (
        *markers,
        f"model-profile-route-{operation}-args-sensitive",
        f"model-profile-route-{operation}-cause-sensitive",
        f"model-profile-route-{operation}-context-sensitive",
    )
    _assert_production_frame_locals_scrubbed(
        source,
        markers=sensitive_markers,
        forbidden_objects=(command,),
    )
    _assert_production_traceback_scrubbed(
        captured.value,
        markers=sensitive_markers,
        forbidden_objects=(command,),
    )


def test_model_profile_openapi_is_safe_and_truthful() -> None:
    schema = create_app().openapi()
    paths = schema["paths"]
    collection = paths["/v1/admin/model-profiles"]
    detail = paths["/v1/admin/model-profiles/{model_profile_id}"]
    for status in ("200", "201"):
        assert set(collection["post"]["responses"][status]["headers"]) >= {
            "ETag",
            "Location",
            "Cache-Control",
        }
    assert "Cache-Control" in collection["get"]["responses"]["200"]["headers"]
    for method in ("get", "patch"):
        assert set(detail[method]["responses"]["200"]["headers"]) >= {
            "ETag",
            "Cache-Control",
        }

    schemas = schema["components"]["schemas"]
    create_schema = schemas["ModelProfileCreate"]
    patch_schema = schemas["ModelProfilePatch"]
    safe_schema = schemas["SafeModelProfile"]
    # `chat` is deliberately absent from both: answer generation is the consuming
    # agent's job, so the published contract must not advertise it as creatable.
    assert create_schema["properties"]["capability"]["enum"] == ["embedding", "rerank"]
    assert safe_schema["properties"]["capability"]["enum"] == ["embedding", "rerank"]
    assert set(create_schema["required"]) >= {
        "capability",
        "provider_config_id",
        "name",
        "model_name",
        "dimension",
        "max_input_tokens",
        "batch_size",
        "timeout_seconds",
        "vector_config",
        "enabled",
    }
    vector_schema = create_schema["properties"]["vector_config"]
    assert vector_schema == {
        "additionalProperties": False,
        "maxProperties": 0,
        "type": "object",
        "title": "Vector Config",
    }
    assert patch_schema["properties"]["vector_config"] == vector_schema
    assert "capability" not in patch_schema["properties"]
    for property_schema in patch_schema["properties"].values():
        assert property_schema.get("type") != "null"
        assert None not in property_schema.get("enum", [])
    serialized_safe_schema = json.dumps(safe_schema).lower()
    for forbidden in ("secret_ref", "ciphertext", "nonce", "secret"):
        assert forbidden not in serialized_safe_schema
