"""Persistence boundaries for the initial index-generation saga."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from hashlib import sha256
from typing import Protocol, cast
from uuid import UUID

from sqlalchemy import String, and_, delete, func, or_, select, update
from sqlalchemy import cast as sql_cast
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from rag_service.api.cursors import CursorPosition
from rag_service.db.models.knowledge_bases import (
    IndexGenerationCleanupClaim,
    IndexGenerationCreationRequest,
    KnowledgeBase,
    KnowledgeBaseIndexGeneration,
    KnowledgeBaseMutation,
)
from rag_service.db.models.providers import ModelProfile, ProviderConfig, ProviderCredential
from rag_service.providers.credentials import EncryptedProviderCredential
from rag_service.providers.repositories import (
    ModelProfileRecord,
    ProviderConfigRecord,
    ProviderCredentialRecord,
)


@dataclass(frozen=True, slots=True)
class EmbeddingConfigurationSource:
    profile: ModelProfileRecord
    provider: ProviderConfigRecord
    credential: ProviderCredentialRecord | None
    legacy_secret_ref: str | None


@dataclass(frozen=True, slots=True)
class CollectionCleanupClaim:
    collection_name: str
    knowledge_base_id: UUID
    generation_id: UUID
    lease_owner: UUID
    lease_epoch: int
    lease_expires_at: datetime
    completed_at: datetime | None


def _profile_record(row: ModelProfile) -> ModelProfileRecord:
    return ModelProfileRecord(
        id=row.id,
        name=row.name,
        capability=row.capability,
        provider_config_id=row.provider_config_id,
        model_name=row.model_name,
        dimension=row.dimension,
        max_input_tokens=row.max_input_tokens,
        batch_size=row.batch_size,
        timeout_seconds=row.timeout_seconds,
        vector_config=cast(dict[str, object], row.vector_config),
        enabled=row.enabled,
        resource_revision=row.resource_revision,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _provider_record(row: ProviderConfig) -> ProviderConfigRecord:
    return ProviderConfigRecord(
        id=row.id,
        name=row.name,
        provider_type=row.provider_type,
        base_url=row.base_url,
        credential_id=row.credential_id,
        default_headers=cast(dict[str, str], row.default_headers),
        routing_options=cast(dict[str, object], row.routing_options),
        timeout_seconds=row.timeout_seconds,
        max_concurrency=row.max_concurrency,
        requests_per_minute=row.requests_per_minute,
        enabled=row.enabled,
        resource_revision=row.resource_revision,
        endpoint_policy_version=row.endpoint_policy_version,
        endpoint_validated_at=row.endpoint_validated_at,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _credential_record(row: ProviderCredential) -> ProviderCredentialRecord:
    return ProviderCredentialRecord(
        id=row.id,
        name=row.name,
        key_version=row.key_version,
        resource_revision=row.resource_revision,
        created_at=row.created_at,
        updated_at=row.updated_at,
        rotated_at=row.rotated_at,
    )


class GenerationRepository(Protocol):
    async def get_knowledge_base_for_update(
        self,
        knowledge_base_id: UUID,
    ) -> KnowledgeBase | None: ...

    async def get_request(
        self,
        actor_key_id: UUID,
        knowledge_base_id: UUID,
        idempotency_key: str,
    ) -> IndexGenerationCreationRequest | None: ...

    async def add_request(self, row: IndexGenerationCreationRequest) -> None: ...

    async def configured_generation_exists(self, knowledge_base_id: UUID) -> bool: ...

    async def load_embedding_source(
        self,
        profile_id: UUID,
    ) -> EmbeddingConfigurationSource | None: ...

    async def load_credential(
        self,
        credential_id: UUID,
    ) -> ProviderCredentialRecord | None: ...

    async def add_generation(self, row: KnowledgeBaseIndexGeneration) -> None: ...

    async def acquire_collection_fence(self, collection_name: str) -> None: ...

    async def claim_collection_cleanup(
        self,
        *,
        collection_name: str,
        knowledge_base_id: UUID,
        generation_id: UUID,
        lease_owner: UUID,
        now: datetime,
        lease_duration: timedelta,
    ) -> CollectionCleanupClaim | None: ...

    async def collection_cleanup_claim_exists(self, collection_name: str) -> bool: ...

    async def get_collection_cleanup_claim(
        self,
        collection_name: str,
    ) -> CollectionCleanupClaim | None: ...

    async def get_collection_cleanup_claim_for_update(
        self,
        collection_name: str,
    ) -> CollectionCleanupClaim | None: ...

    async def list_expired_collection_cleanup_claims(
        self,
        *,
        now: datetime,
        limit: int,
        after_lease_expires_at: datetime | None = None,
        after_collection_name: str | None = None,
    ) -> tuple[CollectionCleanupClaim, ...]: ...

    async def complete_collection_cleanup(
        self,
        collection_name: str,
        *,
        lease_owner: UUID,
        lease_epoch: int,
        now: datetime,
    ) -> bool: ...

    async def expire_collection_cleanup(
        self,
        collection_name: str,
        *,
        lease_owner: UUID,
        lease_epoch: int,
        now: datetime,
    ) -> bool: ...

    async def release_collection_cleanup(
        self,
        collection_name: str,
        *,
        lease_owner: UUID,
        lease_epoch: int,
    ) -> bool: ...

    async def get_generation(
        self,
        generation_id: UUID,
        *,
        for_update: bool = False,
    ) -> KnowledgeBaseIndexGeneration | None: ...

    async def list_generations(
        self,
        knowledge_base_id: UUID,
        position: CursorPosition | None,
        limit: int,
    ) -> list[KnowledgeBaseIndexGeneration]: ...

    async def add_mutation(self, row: KnowledgeBaseMutation) -> None: ...

    async def collection_statuses(
        self,
        collection_names: Sequence[str],
    ) -> dict[str, str]: ...

    async def flush(self) -> None: ...


class SqlAlchemyGenerationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_knowledge_base_for_update(self, knowledge_base_id: UUID) -> KnowledgeBase | None:
        return cast(
            KnowledgeBase | None,
            await self._session.scalar(
                select(KnowledgeBase).where(KnowledgeBase.id == knowledge_base_id).with_for_update()
            ),
        )

    async def get_request(
        self,
        actor_key_id: UUID,
        knowledge_base_id: UUID,
        idempotency_key: str,
    ) -> IndexGenerationCreationRequest | None:
        return cast(
            IndexGenerationCreationRequest | None,
            await self._session.scalar(
                select(IndexGenerationCreationRequest).where(
                    IndexGenerationCreationRequest.actor_api_key_id == actor_key_id,
                    IndexGenerationCreationRequest.knowledge_base_id == knowledge_base_id,
                    IndexGenerationCreationRequest.idempotency_key == idempotency_key,
                )
            ),
        )

    async def add_request(self, row: IndexGenerationCreationRequest) -> None:
        self._session.add(row)
        await self._session.flush()

    async def configured_generation_exists(self, knowledge_base_id: UUID) -> bool:
        statement = (
            select(KnowledgeBaseIndexGeneration.id)
            .where(
                KnowledgeBaseIndexGeneration.knowledge_base_id == knowledge_base_id,
                KnowledgeBaseIndexGeneration.status.in_(("building", "active")),
            )
            .order_by(KnowledgeBaseIndexGeneration.id)
            .limit(1)
        )
        return (await self._session.scalar(statement)) is not None

    async def load_embedding_source(
        self,
        profile_id: UUID,
    ) -> EmbeddingConfigurationSource | None:
        profile = await self._session.get(ModelProfile, profile_id)
        if profile is None:
            return None
        provider = await self._session.get(ProviderConfig, profile.provider_config_id)
        if provider is None:
            return None
        credential = (
            None
            if provider.credential_id is None
            else await self._session.get(ProviderCredential, provider.credential_id)
        )
        return EmbeddingConfigurationSource(
            profile=_profile_record(profile),
            provider=_provider_record(provider),
            credential=None if credential is None else _credential_record(credential),
            legacy_secret_ref=provider.secret_ref,
        )

    async def load_credential(
        self,
        credential_id: UUID,
    ) -> ProviderCredentialRecord | None:
        credential = await self._session.get(ProviderCredential, credential_id)
        return None if credential is None else _credential_record(credential)

    async def add_generation(self, row: KnowledgeBaseIndexGeneration) -> None:
        self._session.add(row)
        await self._session.flush()
        await self._session.refresh(row, attribute_names=["created_at"])

    async def acquire_collection_fence(self, collection_name: str) -> None:
        if type(collection_name) is not str or not collection_name:
            raise ValueError("collection fence is invalid")
        fence_key = int.from_bytes(
            sha256(collection_name.encode("utf-8")).digest()[:8],
            byteorder="big",
            signed=True,
        )
        await self._session.execute(select(func.pg_advisory_xact_lock(fence_key)))

    @staticmethod
    def _cleanup_claim(row: IndexGenerationCleanupClaim) -> CollectionCleanupClaim:
        return CollectionCleanupClaim(
            collection_name=row.collection_name,
            knowledge_base_id=row.knowledge_base_id,
            generation_id=row.generation_id,
            lease_owner=row.lease_owner,
            lease_epoch=row.lease_epoch,
            lease_expires_at=row.lease_expires_at,
            completed_at=row.completed_at,
        )

    async def claim_collection_cleanup(
        self,
        *,
        collection_name: str,
        knowledge_base_id: UUID,
        generation_id: UUID,
        lease_owner: UUID,
        now: datetime,
        lease_duration: timedelta,
    ) -> CollectionCleanupClaim | None:
        current = cast(
            IndexGenerationCleanupClaim | None,
            await self._session.scalar(
                select(IndexGenerationCleanupClaim)
                .where(IndexGenerationCleanupClaim.collection_name == collection_name)
                .with_for_update()
            ),
        )
        lease_expires_at = now + lease_duration
        if current is None:
            current = IndexGenerationCleanupClaim(
                collection_name=collection_name,
                knowledge_base_id=knowledge_base_id,
                generation_id=generation_id,
                lease_owner=lease_owner,
                lease_epoch=1,
                lease_expires_at=lease_expires_at,
                updated_at=now,
            )
            self._session.add(current)
        else:
            if (
                current.knowledge_base_id != knowledge_base_id
                or current.generation_id != generation_id
            ):
                raise ValueError("cleanup claim identity mismatch")
            if current.completed_at is not None:
                return None
            if current.lease_expires_at > now:
                return None
            current.lease_owner = lease_owner
            current.lease_epoch += 1
            current.lease_expires_at = lease_expires_at
            current.updated_at = now
        await self._session.flush()
        return self._cleanup_claim(current)

    async def collection_cleanup_claim_exists(self, collection_name: str) -> bool:
        statement = (
            select(IndexGenerationCleanupClaim.collection_name)
            .where(IndexGenerationCleanupClaim.collection_name == collection_name)
            .limit(1)
        )
        return (await self._session.scalar(statement)) is not None

    async def get_collection_cleanup_claim(
        self,
        collection_name: str,
    ) -> CollectionCleanupClaim | None:
        row = cast(
            IndexGenerationCleanupClaim | None,
            await self._session.get(IndexGenerationCleanupClaim, collection_name),
        )
        return None if row is None else self._cleanup_claim(row)

    async def get_collection_cleanup_claim_for_update(
        self,
        collection_name: str,
    ) -> CollectionCleanupClaim | None:
        row = cast(
            IndexGenerationCleanupClaim | None,
            await self._session.scalar(
                select(IndexGenerationCleanupClaim)
                .where(IndexGenerationCleanupClaim.collection_name == collection_name)
                .with_for_update()
            ),
        )
        return None if row is None else self._cleanup_claim(row)

    async def list_expired_collection_cleanup_claims(
        self,
        *,
        now: datetime,
        limit: int,
        after_lease_expires_at: datetime | None = None,
        after_collection_name: str | None = None,
    ) -> tuple[CollectionCleanupClaim, ...]:
        if (after_lease_expires_at is None) != (after_collection_name is None):
            raise ValueError("expired cleanup cursor is invalid")
        statement = select(IndexGenerationCleanupClaim).where(
            IndexGenerationCleanupClaim.completed_at.is_(None),
            IndexGenerationCleanupClaim.lease_expires_at <= now,
            IndexGenerationCleanupClaim.collection_name
            == func.concat(
                "rag_kb_",
                func.replace(
                    sql_cast(IndexGenerationCleanupClaim.knowledge_base_id, String),
                    "-",
                    "",
                ),
                "_g_",
                func.replace(
                    sql_cast(IndexGenerationCleanupClaim.generation_id, String),
                    "-",
                    "",
                ),
            ),
        )
        if after_lease_expires_at is not None and after_collection_name is not None:
            statement = statement.where(
                or_(
                    IndexGenerationCleanupClaim.lease_expires_at > after_lease_expires_at,
                    and_(
                        IndexGenerationCleanupClaim.lease_expires_at == after_lease_expires_at,
                        IndexGenerationCleanupClaim.collection_name > after_collection_name,
                    ),
                )
            )
        rows = (
            await self._session.scalars(
                statement.order_by(
                    IndexGenerationCleanupClaim.lease_expires_at,
                    IndexGenerationCleanupClaim.collection_name,
                ).limit(limit)
            )
        ).all()
        return tuple(self._cleanup_claim(row) for row in rows)

    async def complete_collection_cleanup(
        self,
        collection_name: str,
        *,
        lease_owner: UUID,
        lease_epoch: int,
        now: datetime,
    ) -> bool:
        completed = await self._session.scalar(
            update(IndexGenerationCleanupClaim)
            .where(
                IndexGenerationCleanupClaim.collection_name == collection_name,
                IndexGenerationCleanupClaim.lease_owner == lease_owner,
                IndexGenerationCleanupClaim.lease_epoch == lease_epoch,
                IndexGenerationCleanupClaim.completed_at.is_(None),
            )
            .values(completed_at=now, updated_at=now)
            .returning(IndexGenerationCleanupClaim.collection_name)
        )
        return completed is not None

    async def expire_collection_cleanup(
        self,
        collection_name: str,
        *,
        lease_owner: UUID,
        lease_epoch: int,
        now: datetime,
    ) -> bool:
        expired = await self._session.scalar(
            update(IndexGenerationCleanupClaim)
            .where(
                IndexGenerationCleanupClaim.collection_name == collection_name,
                IndexGenerationCleanupClaim.lease_owner == lease_owner,
                IndexGenerationCleanupClaim.lease_epoch == lease_epoch,
                IndexGenerationCleanupClaim.completed_at.is_(None),
            )
            .values(lease_expires_at=now, updated_at=now)
            .returning(IndexGenerationCleanupClaim.collection_name)
        )
        return expired is not None

    async def release_collection_cleanup(
        self,
        collection_name: str,
        *,
        lease_owner: UUID,
        lease_epoch: int,
    ) -> bool:
        released = await self._session.scalar(
            delete(IndexGenerationCleanupClaim)
            .where(
                IndexGenerationCleanupClaim.collection_name == collection_name,
                IndexGenerationCleanupClaim.lease_owner == lease_owner,
                IndexGenerationCleanupClaim.lease_epoch == lease_epoch,
                IndexGenerationCleanupClaim.completed_at.is_(None),
            )
            .returning(IndexGenerationCleanupClaim.collection_name)
        )
        return released is not None

    async def get_generation(
        self,
        generation_id: UUID,
        *,
        for_update: bool = False,
    ) -> KnowledgeBaseIndexGeneration | None:
        statement = select(KnowledgeBaseIndexGeneration).where(
            KnowledgeBaseIndexGeneration.id == generation_id
        )
        if for_update:
            statement = statement.with_for_update()
        return cast(
            KnowledgeBaseIndexGeneration | None,
            await self._session.scalar(statement),
        )

    async def list_generations(
        self,
        knowledge_base_id: UUID,
        position: CursorPosition | None,
        limit: int,
    ) -> list[KnowledgeBaseIndexGeneration]:
        statement = select(KnowledgeBaseIndexGeneration).where(
            KnowledgeBaseIndexGeneration.knowledge_base_id == knowledge_base_id
        )
        if position is not None:
            statement = statement.where(
                or_(
                    KnowledgeBaseIndexGeneration.created_at > position.created_at,
                    and_(
                        KnowledgeBaseIndexGeneration.created_at == position.created_at,
                        KnowledgeBaseIndexGeneration.id > position.id,
                    ),
                )
            )
        statement = statement.order_by(
            KnowledgeBaseIndexGeneration.created_at,
            KnowledgeBaseIndexGeneration.id,
        ).limit(limit)
        return list((await self._session.scalars(statement)).all())

    async def add_mutation(self, row: KnowledgeBaseMutation) -> None:
        self._session.add(row)
        await self._session.flush()

    async def collection_statuses(
        self,
        collection_names: Sequence[str],
    ) -> dict[str, str]:
        names = tuple(collection_names)
        if not names:
            return {}
        rows = (
            await self._session.execute(
                select(
                    KnowledgeBaseIndexGeneration.qdrant_collection_name,
                    KnowledgeBaseIndexGeneration.status,
                ).where(KnowledgeBaseIndexGeneration.qdrant_collection_name.in_(names))
            )
        ).all()
        return {name: status for name, status in rows}

    async def flush(self) -> None:
        await self._session.flush()


class SessionProviderCredentialReader:
    """Read the current encrypted credential by the snapshot's stable ID."""

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def get_encrypted(
        self,
        credential_id: UUID,
    ) -> EncryptedProviderCredential | None:
        async with self._session_factory() as session:
            row = await session.get(ProviderCredential, credential_id)
            if row is None:
                return None
            return EncryptedProviderCredential(
                ciphertext=bytes(row.ciphertext),
                nonce=bytes(row.nonce),
                key_version=row.key_version,
                algorithm=row.algorithm,
            )


__all__ = [
    "CollectionCleanupClaim",
    "EmbeddingConfigurationSource",
    "GenerationRepository",
    "SessionProviderCredentialReader",
    "SqlAlchemyGenerationRepository",
]
