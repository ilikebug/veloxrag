import asyncio
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Protocol, cast
from uuid import UUID

from sqlalchemy import and_, insert, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.api.cursors import CursorPosition
from rag_service.db.models.auth import ApiKey, AuditEvent, IdempotencyRecord
from rag_service.db.models.knowledge_bases import KnowledgeBaseIndexGeneration
from rag_service.db.models.providers import ModelProfile, ProviderConfig, ProviderCredential
from rag_service.providers.credentials import EncryptedProviderCredential


@dataclass(frozen=True, slots=True)
class ProviderCredentialRecord:
    id: UUID
    name: str
    key_version: str
    resource_revision: int
    created_at: datetime
    updated_at: datetime
    rotated_at: datetime | None


@dataclass(frozen=True, slots=True)
class ProviderConfigRecord:
    id: UUID
    name: str
    provider_type: str
    base_url: str
    credential_id: UUID | None
    default_headers: dict[str, str]
    routing_options: dict[str, object]
    timeout_seconds: Decimal
    max_concurrency: int
    requests_per_minute: int
    enabled: bool
    resource_revision: int
    endpoint_policy_version: str | None
    endpoint_validated_at: datetime | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ModelProfileRecord:
    id: UUID
    name: str
    capability: str
    provider_config_id: UUID
    model_name: str
    dimension: int | None
    max_input_tokens: int
    batch_size: int
    timeout_seconds: Decimal
    vector_config: dict[str, object]
    enabled: bool
    resource_revision: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True, repr=False)
class ProviderConfigSecretSourceRecord:
    provider_config_id: UUID
    credential_id: UUID | None
    secret_ref: str | None

    def __repr__(self) -> str:
        return "ProviderConfigSecretSourceRecord(<redacted>)"


@dataclass(frozen=True, slots=True)
class AdminActorRecord:
    id: UUID
    public_id: str
    key_type: str
    status: str
    not_before: datetime | None
    expires_at: datetime | None
    revoked_at: datetime | None


class ProviderCredentialRepository(Protocol):
    async def list_safe(
        self,
        position: CursorPosition | None,
        limit: int,
    ) -> list[ProviderCredentialRecord]: ...

    async def get_safe(
        self,
        credential_id: UUID,
        *,
        for_update: bool = False,
    ) -> ProviderCredentialRecord | None: ...

    async def add_encrypted(
        self,
        credential_id: UUID,
        name: str,
        encrypted: EncryptedProviderCredential,
    ) -> ProviderCredentialRecord: ...

    async def update_encrypted(
        self,
        credential_id: UUID,
        *,
        name: str | None,
        encrypted: EncryptedProviderCredential | None,
        updated_at: datetime,
        rotated_at: datetime | None,
    ) -> ProviderCredentialRecord: ...


class ProviderConfigRepository(Protocol):
    async def list_safe(
        self,
        position: CursorPosition | None,
        limit: int,
    ) -> list[ProviderConfigRecord]: ...

    async def get_safe(
        self,
        provider_config_id: UUID,
        *,
        for_update: bool = False,
    ) -> ProviderConfigRecord | None: ...

    async def get_secret_source(
        self,
        provider_config_id: UUID,
    ) -> ProviderConfigSecretSourceRecord | None: ...

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
    ) -> ProviderConfigRecord: ...

    async def update_validated(
        self,
        provider_config_id: UUID,
        *,
        values: dict[str, object],
        updated_at: datetime,
    ) -> ProviderConfigRecord: ...


class ModelProfileRepository(Protocol):
    async def list_safe(
        self,
        position: CursorPosition | None,
        limit: int,
    ) -> list[ModelProfileRecord]: ...

    async def get_safe(
        self,
        model_profile_id: UUID,
        *,
        for_update: bool = False,
    ) -> ModelProfileRecord | None: ...

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
    ) -> ModelProfileRecord: ...

    async def update(
        self,
        model_profile_id: UUID,
        *,
        values: dict[str, object],
        updated_at: datetime,
    ) -> ModelProfileRecord: ...

    async def is_referenced_by_generation(self, model_profile_id: UUID) -> bool: ...

    async def provider_config_is_referenced_by_generation(
        self,
        provider_config_id: UUID,
        *,
        lock_profiles: bool = False,
    ) -> bool: ...


class ProviderAdminRepository(Protocol):
    async def get_for_update(self, key_id: UUID) -> AdminActorRecord | None: ...


class ProviderIdempotencyRepository(Protocol):
    async def get(
        self,
        actor_key_id: UUID,
        operation: str,
        idempotency_key: str,
    ) -> IdempotencyRecord | None: ...

    async def add(self, record: IdempotencyRecord) -> None: ...


class ProviderAuditRepository(Protocol):
    async def add(self, event: AuditEvent) -> None: ...

    async def get_created_response(
        self,
        actor_key_id: UUID,
        credential_id: UUID,
    ) -> object | None: ...

    async def get_provider_config_created_response(
        self,
        actor_key_id: UUID,
        provider_config_id: UUID,
    ) -> object | None: ...

    async def get_model_profile_created_response(
        self,
        actor_key_id: UUID,
        model_profile_id: UUID,
    ) -> object | None: ...


@dataclass(frozen=True, slots=True)
class ProviderRepositories:
    credentials: ProviderCredentialRepository
    admins: ProviderAdminRepository
    idempotency: ProviderIdempotencyRepository
    audits: ProviderAuditRepository
    configs: ProviderConfigRepository | None = None
    profiles: ModelProfileRepository | None = None


_SAFE_CREDENTIAL_COLUMNS = (
    ProviderCredential.id,
    ProviderCredential.name,
    ProviderCredential.key_version,
    ProviderCredential.resource_revision,
    ProviderCredential.created_at,
    ProviderCredential.updated_at,
    ProviderCredential.rotated_at,
)

_SAFE_CONFIG_COLUMNS = (
    ProviderConfig.id,
    ProviderConfig.name,
    ProviderConfig.provider_type,
    ProviderConfig.base_url,
    ProviderConfig.credential_id,
    ProviderConfig.default_headers,
    ProviderConfig.routing_options,
    ProviderConfig.timeout_seconds,
    ProviderConfig.max_concurrency,
    ProviderConfig.requests_per_minute,
    ProviderConfig.enabled,
    ProviderConfig.resource_revision,
    ProviderConfig.endpoint_policy_version,
    ProviderConfig.endpoint_validated_at,
    ProviderConfig.created_at,
    ProviderConfig.updated_at,
)

_SAFE_PROFILE_COLUMNS = (
    ModelProfile.id,
    ModelProfile.name,
    ModelProfile.capability,
    ModelProfile.provider_config_id,
    ModelProfile.model_name,
    ModelProfile.dimension,
    ModelProfile.max_input_tokens,
    ModelProfile.batch_size,
    ModelProfile.timeout_seconds,
    ModelProfile.vector_config,
    ModelProfile.enabled,
    ModelProfile.resource_revision,
    ModelProfile.created_at,
    ModelProfile.updated_at,
)


def _credential_record(row: object) -> ProviderCredentialRecord:
    values = cast(
        tuple[UUID, str, str, int, datetime, datetime, datetime | None],
        row,
    )
    return ProviderCredentialRecord(*values)


def _config_record(row: object) -> ProviderConfigRecord:
    values = cast(
        tuple[
            UUID,
            str,
            str,
            str,
            UUID | None,
            dict[str, str],
            dict[str, object],
            Decimal,
            int,
            int,
            bool,
            int,
            str | None,
            datetime | None,
            datetime,
            datetime,
        ],
        row,
    )
    return ProviderConfigRecord(*values)


def _profile_record(row: object) -> ModelProfileRecord:
    values = cast(
        tuple[
            UUID,
            str,
            str,
            UUID,
            str,
            int | None,
            int,
            int,
            Decimal,
            dict[str, object],
            bool,
            int,
            datetime,
            datetime,
        ],
        row,
    )
    return ModelProfileRecord(*values)


def _redacted_encrypted() -> EncryptedProviderCredential:
    return EncryptedProviderCredential(
        ciphertext=b"",
        nonce=b"",
        key_version="<redacted>",
        algorithm="<redacted>",
    )


class SqlAlchemyProviderCredentialRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_safe(
        self,
        position: CursorPosition | None,
        limit: int,
    ) -> list[ProviderCredentialRecord]:
        statement = select(*_SAFE_CREDENTIAL_COLUMNS)
        if position is not None:
            statement = statement.where(
                or_(
                    ProviderCredential.created_at > position.created_at,
                    and_(
                        ProviderCredential.created_at == position.created_at,
                        ProviderCredential.id > position.id,
                    ),
                )
            )
        statement = statement.order_by(
            ProviderCredential.created_at,
            ProviderCredential.id,
        ).limit(limit)
        rows = (await self._session.execute(statement)).all()
        return [_credential_record(row) for row in rows]

    async def get_safe(
        self,
        credential_id: UUID,
        *,
        for_update: bool = False,
    ) -> ProviderCredentialRecord | None:
        statement = select(*_SAFE_CREDENTIAL_COLUMNS).where(ProviderCredential.id == credential_id)
        if for_update:
            statement = statement.with_for_update()
        row = (await self._session.execute(statement)).one_or_none()
        return None if row is None else _credential_record(row)

    async def add_encrypted(
        self,
        credential_id: UUID,
        name: str,
        encrypted: EncryptedProviderCredential,
    ) -> ProviderCredentialRecord:
        statement: Any = None
        result: Any = None
        row: object | None = None
        cancelled = False
        try:
            statement = (
                insert(ProviderCredential)
                .values(
                    id=credential_id,
                    name=name,
                    ciphertext=encrypted.ciphertext,
                    nonce=encrypted.nonce,
                    algorithm=encrypted.algorithm,
                    key_version=encrypted.key_version,
                    resource_revision=1,
                )
                .returning(*_SAFE_CREDENTIAL_COLUMNS)
            )
            result = await self._session.execute(statement)
            row = result.one()
            return _credential_record(row)
        except asyncio.CancelledError:
            cancelled = True
        finally:
            statement = None
            result = None
            row = None
            name = "<redacted>"
            encrypted = _redacted_encrypted()
        if cancelled:
            raise asyncio.CancelledError from None
        raise AssertionError("unreachable")

    async def update_encrypted(
        self,
        credential_id: UUID,
        *,
        name: str | None,
        encrypted: EncryptedProviderCredential | None,
        updated_at: datetime,
        rotated_at: datetime | None,
    ) -> ProviderCredentialRecord:
        values: dict[str, object] = {}
        statement: Any = None
        result: Any = None
        row: object | None = None
        cancelled = False
        try:
            values = {
                "updated_at": updated_at,
                "resource_revision": ProviderCredential.resource_revision + 1,
            }
            if name is not None:
                values["name"] = name
            if encrypted is not None:
                values.update(
                    {
                        "ciphertext": encrypted.ciphertext,
                        "nonce": encrypted.nonce,
                        "algorithm": encrypted.algorithm,
                        "key_version": encrypted.key_version,
                        "rotated_at": rotated_at,
                    }
                )
            statement = (
                update(ProviderCredential)
                .where(ProviderCredential.id == credential_id)
                .values(**values)
                .returning(*_SAFE_CREDENTIAL_COLUMNS)
            )
            result = await self._session.execute(statement)
            row = result.one_or_none()
            if row is None:
                raise LookupError("provider credential disappeared")
            return _credential_record(row)
        except asyncio.CancelledError:
            cancelled = True
        finally:
            values.clear()
            statement = None
            result = None
            row = None
            name = "<redacted>"
            encrypted = None
        if cancelled:
            raise asyncio.CancelledError from None
        raise AssertionError("unreachable")


class SqlAlchemyProviderConfigRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_safe(
        self,
        position: CursorPosition | None,
        limit: int,
    ) -> list[ProviderConfigRecord]:
        statement = select(*_SAFE_CONFIG_COLUMNS)
        if position is not None:
            statement = statement.where(
                or_(
                    ProviderConfig.created_at > position.created_at,
                    and_(
                        ProviderConfig.created_at == position.created_at,
                        ProviderConfig.id > position.id,
                    ),
                )
            )
        statement = statement.order_by(ProviderConfig.created_at, ProviderConfig.id).limit(limit)
        rows = (await self._session.execute(statement)).all()
        return [_config_record(row) for row in rows]

    async def get_safe(
        self,
        provider_config_id: UUID,
        *,
        for_update: bool = False,
    ) -> ProviderConfigRecord | None:
        statement = select(*_SAFE_CONFIG_COLUMNS).where(ProviderConfig.id == provider_config_id)
        if for_update:
            statement = statement.with_for_update()
        row = (await self._session.execute(statement)).one_or_none()
        return None if row is None else _config_record(row)

    async def get_secret_source(
        self,
        provider_config_id: UUID,
    ) -> ProviderConfigSecretSourceRecord | None:
        statement: Any = None
        result: Any = None
        row: object | None = None
        values: tuple[UUID, UUID | None, str | None] | tuple[()] = ()
        record: ProviderConfigSecretSourceRecord | None = None
        cancelled = False
        failed = False
        try:
            statement = select(
                ProviderConfig.id,
                ProviderConfig.credential_id,
                ProviderConfig.secret_ref,
            ).where(ProviderConfig.id == provider_config_id)
            result = await self._session.execute(statement)
            row = result.one_or_none()
            if row is None:
                return None
            values = cast(tuple[UUID, UUID | None, str | None], row)
            record = ProviderConfigSecretSourceRecord(*values)
            return record
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            failed = True
        finally:
            statement = None
            result = None
            row = None
            values = ()
            record = None
        if cancelled:
            raise asyncio.CancelledError() from None
        if failed:
            raise RuntimeError() from None
        raise AssertionError("unreachable")

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
        statement: Any = None
        result: Any = None
        row: object | None = None
        cancelled = False
        try:
            statement = (
                insert(ProviderConfig)
                .values(
                    id=provider_config_id,
                    name=name,
                    provider_type=provider_type,
                    base_url=base_url,
                    secret_ref=None,
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
                )
                .returning(*_SAFE_CONFIG_COLUMNS)
            )
            result = await self._session.execute(statement)
            row = result.one()
            return _config_record(row)
        except asyncio.CancelledError:
            cancelled = True
        finally:
            statement = None
            result = None
            row = None
            name = "<redacted>"
            provider_type = "<redacted>"
            base_url = "<redacted>"
            default_headers.clear()
            routing_options.clear()
            endpoint_policy_version = "<redacted>"
        if cancelled:
            raise asyncio.CancelledError from None
        raise AssertionError("unreachable")

    async def update_validated(
        self,
        provider_config_id: UUID,
        *,
        values: dict[str, object],
        updated_at: datetime,
    ) -> ProviderConfigRecord:
        statement: Any = None
        result: Any = None
        row: object | None = None
        cancelled = False
        try:
            update_values = dict(values)
            update_values.update(
                {
                    "updated_at": updated_at,
                    "resource_revision": ProviderConfig.resource_revision + 1,
                }
            )
            statement = (
                update(ProviderConfig)
                .where(ProviderConfig.id == provider_config_id)
                .values(**update_values)
                .returning(*_SAFE_CONFIG_COLUMNS)
            )
            result = await self._session.execute(statement)
            row = result.one_or_none()
            if row is None:
                raise LookupError("provider config disappeared")
            return _config_record(row)
        except asyncio.CancelledError:
            cancelled = True
        finally:
            values.clear()
            if "update_values" in locals():
                update_values.clear()
            statement = None
            result = None
            row = None
        if cancelled:
            raise asyncio.CancelledError from None
        raise AssertionError("unreachable")


class SqlAlchemyModelProfileRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_safe(
        self,
        position: CursorPosition | None,
        limit: int,
    ) -> list[ModelProfileRecord]:
        statement = select(*_SAFE_PROFILE_COLUMNS).where(ModelProfile.capability == "embedding")
        if position is not None:
            statement = statement.where(
                or_(
                    ModelProfile.created_at > position.created_at,
                    and_(
                        ModelProfile.created_at == position.created_at,
                        ModelProfile.id > position.id,
                    ),
                )
            )
        statement = statement.order_by(ModelProfile.created_at, ModelProfile.id).limit(limit)
        rows = (await self._session.execute(statement)).all()
        return [_profile_record(row) for row in rows]

    async def get_safe(
        self,
        model_profile_id: UUID,
        *,
        for_update: bool = False,
    ) -> ModelProfileRecord | None:
        statement = select(*_SAFE_PROFILE_COLUMNS).where(ModelProfile.id == model_profile_id)
        if for_update:
            statement = statement.with_for_update()
        row = (await self._session.execute(statement)).one_or_none()
        return None if row is None else _profile_record(row)

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
        statement = (
            insert(ModelProfile)
            .values(
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
            )
            .returning(*_SAFE_PROFILE_COLUMNS)
        )
        row = (await self._session.execute(statement)).one()
        return _profile_record(row)

    async def update(
        self,
        model_profile_id: UUID,
        *,
        values: dict[str, object],
        updated_at: datetime,
    ) -> ModelProfileRecord:
        update_values = dict(values)
        update_values.update(
            {
                "updated_at": updated_at,
                "resource_revision": ModelProfile.resource_revision + 1,
            }
        )
        statement = (
            update(ModelProfile)
            .where(ModelProfile.id == model_profile_id)
            .values(**update_values)
            .returning(*_SAFE_PROFILE_COLUMNS)
        )
        row = (await self._session.execute(statement)).one_or_none()
        values.clear()
        update_values.clear()
        if row is None:
            raise LookupError("model profile disappeared")
        return _profile_record(row)

    async def is_referenced_by_generation(self, model_profile_id: UUID) -> bool:
        statement = (
            select(KnowledgeBaseIndexGeneration.id)
            .where(KnowledgeBaseIndexGeneration.embedding_profile_id == model_profile_id)
            .limit(1)
        )
        return (await self._session.scalar(statement)) is not None

    async def provider_config_is_referenced_by_generation(
        self,
        provider_config_id: UUID,
        *,
        lock_profiles: bool = False,
    ) -> bool:
        profile_statement = (
            select(ModelProfile.id)
            .where(ModelProfile.provider_config_id == provider_config_id)
            .order_by(ModelProfile.id)
        )
        if lock_profiles:
            profile_statement = profile_statement.with_for_update()
        profile_ids = list((await self._session.scalars(profile_statement)).all())
        if not profile_ids:
            return False
        generation_statement = (
            select(KnowledgeBaseIndexGeneration.id)
            .where(KnowledgeBaseIndexGeneration.embedding_profile_id.in_(profile_ids))
            .limit(1)
        )
        return (await self._session.scalar(generation_statement)) is not None


class SqlAlchemyProviderAdminRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_for_update(self, key_id: UUID) -> AdminActorRecord | None:
        row = (
            await self._session.execute(
                select(
                    ApiKey.id,
                    ApiKey.public_id,
                    ApiKey.key_type,
                    ApiKey.status,
                    ApiKey.not_before,
                    ApiKey.expires_at,
                    ApiKey.revoked_at,
                )
                .where(ApiKey.id == key_id)
                .with_for_update()
            )
        ).one_or_none()
        if row is None:
            return None
        values = cast(
            tuple[UUID, str, str, str, datetime | None, datetime | None, datetime | None],
            row,
        )
        return AdminActorRecord(*values)


class SqlAlchemyProviderIdempotencyRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(
        self,
        actor_key_id: UUID,
        operation: str,
        idempotency_key: str,
    ) -> IdempotencyRecord | None:
        return cast(
            IdempotencyRecord | None,
            await self._session.scalar(
                select(IdempotencyRecord).where(
                    IdempotencyRecord.actor_key_id == actor_key_id,
                    IdempotencyRecord.operation == operation,
                    IdempotencyRecord.idempotency_key == idempotency_key,
                )
            ),
        )

    async def add(self, record: IdempotencyRecord) -> None:
        self._session.add(record)
        await self._session.flush()


class SqlAlchemyProviderAuditRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def add(self, event: AuditEvent) -> None:
        self._session.add(event)
        await self._session.flush()

    async def get_created_response(
        self,
        actor_key_id: UUID,
        credential_id: UUID,
    ) -> object | None:
        # Creation audit metadata is the authoritative, immutable idempotency response
        # snapshot. Missing or duplicate snapshots are ambiguous and therefore fail closed.
        statement = (
            select(AuditEvent.metadata_)
            .where(
                AuditEvent.actor_api_key_id == actor_key_id,
                AuditEvent.action == "provider_credential.created",
                AuditEvent.target_type == "provider_credential",
                AuditEvent.target_id == credential_id,
            )
            .order_by(AuditEvent.created_at, AuditEvent.id)
            .limit(2)
        )
        rows = (await self._session.execute(statement)).all()
        return rows[0][0] if len(rows) == 1 else None

    async def get_model_profile_created_response(
        self,
        actor_key_id: UUID,
        model_profile_id: UUID,
    ) -> object | None:
        statement = (
            select(AuditEvent.metadata_)
            .where(
                AuditEvent.actor_api_key_id == actor_key_id,
                AuditEvent.action == "model_profile.created",
                AuditEvent.target_type == "model_profile",
                AuditEvent.target_id == model_profile_id,
            )
            .order_by(AuditEvent.created_at, AuditEvent.id)
            .limit(2)
        )
        rows = (await self._session.execute(statement)).all()
        return rows[0][0] if len(rows) == 1 else None

    async def get_provider_config_created_response(
        self,
        actor_key_id: UUID,
        provider_config_id: UUID,
    ) -> object | None:
        statement = (
            select(AuditEvent.metadata_)
            .where(
                AuditEvent.actor_api_key_id == actor_key_id,
                AuditEvent.action == "provider_config.created",
                AuditEvent.target_type == "provider_config",
                AuditEvent.target_id == provider_config_id,
            )
            .order_by(AuditEvent.created_at, AuditEvent.id)
            .limit(2)
        )
        rows = (await self._session.execute(statement)).all()
        return rows[0][0] if len(rows) == 1 else None


def sqlalchemy_provider_repositories(session: AsyncSession) -> ProviderRepositories:
    return ProviderRepositories(
        credentials=SqlAlchemyProviderCredentialRepository(session),
        admins=SqlAlchemyProviderAdminRepository(session),
        idempotency=SqlAlchemyProviderIdempotencyRepository(session),
        audits=SqlAlchemyProviderAuditRepository(session),
        configs=SqlAlchemyProviderConfigRepository(session),
        profiles=SqlAlchemyModelProfileRepository(session),
    )
