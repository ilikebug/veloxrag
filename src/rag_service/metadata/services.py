import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Never, Protocol, cast
from uuid import UUID, uuid4

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.api.cursors import CursorPosition, decode_cursor, encode_cursor
from rag_service.api.errors import BusinessError
from rag_service.api.etags import knowledge_base_etag, require_matching_etag
from rag_service.api.middleware import is_valid_request_id
from rag_service.api.validation import validate_idempotency_key
from rag_service.auth.policies import (
    AgentPrincipal,
    Capability,
    require_capability,
    require_document_read,
    require_raw_file_read,
)
from rag_service.config import Settings
from rag_service.db.models.auth import ApiKey, AuditEvent, IdempotencyRecord
from rag_service.db.models.documents import Job
from rag_service.db.models.knowledge_bases import (
    KnowledgeBase,
    KnowledgeBaseIndexGeneration,
    KnowledgeBaseMutation,
)
from rag_service.ingestion.artifacts import parsed_text_object_key
from rag_service.metadata.document_repositories import (
    DocumentMetadataRepository,
    DocumentRecord,
    DocumentVersionRecord,
    sqlalchemy_document_metadata_repository,
)
from rag_service.metadata.knowledge_base_repositories import (
    KnowledgeBaseRepositories,
    sqlalchemy_knowledge_base_repositories,
)
from rag_service.metadata.schemas import (
    DocumentContent,
    DocumentPage,
    DocumentStatus,
    DocumentVersionPage,
    DocumentVersionStatus,
    FilterSchemaField,
    FilterSchemaReplacement,
    KnowledgeBaseCreate,
    KnowledgeBaseCreateResult,
    KnowledgeBasePage,
    KnowledgeBasePatch,
    KnowledgeBaseStatus,
    SafeDocument,
    SafeFilterSchema,
    SafeKnowledgeBase,
    SafeVersion,
    canonicalize_filter_schema,
    knowledge_base_create_fingerprint,
)

type RepositoryFactory = Callable[[AsyncSession], KnowledgeBaseRepositories]


class DocumentContentStore(Protocol):
    async def read_bytes(
        self,
        object_key: str,
        *,
        expected_checksum: str,
        max_bytes: int,
    ) -> bytes: ...


type DocumentRepositoryFactory = Callable[[AsyncSession], DocumentMetadataRepository]
type Clock = Callable[[], datetime]

_CREATE_OPERATION = "knowledge_base.create"
_IDEMPOTENCY_UNIQUE_CONSTRAINT = "uq_idempotency_actor_operation_key"
_MAX_RETAINED_EXCEPTION_NODES = 32


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _invalid_api_key_error() -> BusinessError:
    return BusinessError(401, "INVALID_API_KEY", "Invalid API key")


def _not_found_error() -> BusinessError:
    return BusinessError(404, "RESOURCE_NOT_FOUND", "Resource not found")


def _idempotency_conflict_error() -> BusinessError:
    return BusinessError(409, "IDEMPOTENCY_CONFLICT", "Idempotency key conflict")


def _state_conflict_error() -> BusinessError:
    return BusinessError(409, "RESOURCE_STATE_CONFLICT", "Resource state conflict")


def _reindex_required_error() -> BusinessError:
    return BusinessError(409, "REINDEX_REQUIRED", "A new index generation is required")


def _validation_error() -> BusinessError:
    return BusinessError(422, "VALIDATION_ERROR", "Invalid knowledge base request")


def _internal_error() -> BusinessError:
    return BusinessError(500, "INTERNAL_ERROR", "Internal server error")


def _sanitize_retained_exception(error: BaseException) -> None:
    pending = [error]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in visited:
            continue
        visited.add(identity)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        for nested_error in (current.__cause__, current.__context__):
            if nested_error is not None:
                pending.append(nested_error)
        if current.__traceback__ is not None:
            traceback.clear_frames(current.__traceback__)
        try:
            current.args = ("<redacted>",)
            current.__traceback__ = None
            current.__cause__ = None
            current.__context__ = None
        except (AttributeError, TypeError):
            continue


def _raise_internal(error: BaseException) -> Never:
    _sanitize_retained_exception(error)
    public_error = _internal_error()
    try:
        raise public_error from None
    finally:
        object.__setattr__(public_error, "__context__", None)
        object.__setattr__(public_error, "__cause__", None)


def _raise_sanitized_business_error(error: BusinessError) -> Never:
    status_code = error.status_code
    code = error.code
    message = error.message
    retryable = error.retryable
    headers = None if error.headers is None else dict(error.headers)
    _sanitize_retained_exception(error)
    public_error = BusinessError(
        status_code,
        code,
        message,
        retryable=retryable,
        headers=headers,
    )
    try:
        raise public_error from None
    finally:
        object.__setattr__(public_error, "__context__", None)
        object.__setattr__(public_error, "__cause__", None)


def _reraise_redacted_base_exception(error: BaseException) -> Never:
    pending = [error]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in visited:
            continue
        visited.add(identity)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        for nested_error in (current.__cause__, current.__context__):
            if nested_error is not None:
                pending.append(nested_error)
        if current.__traceback__ is not None:
            traceback.clear_frames(current.__traceback__)
        current.__traceback__ = None
        current.__cause__ = None
        current.__context__ = None
    try:
        raise error from None
    finally:
        object.__setattr__(error, "__context__", None)
        object.__setattr__(error, "__cause__", None)


def _require_uuid(value: object) -> UUID:
    if not isinstance(value, UUID):
        raise _validation_error()
    return value


def _require_request_id(value: object, max_length: int) -> str:
    if not is_valid_request_id(value, max_length):
        raise _validation_error()
    return value


def _is_currently_active(row: ApiKey, now: datetime) -> bool:
    if row.status != "active" or row.revoked_at is not None:
        return False
    if row.not_before is not None and row.not_before > now:
        return False
    return row.expires_at is None or row.expires_at > now


def _constraint_name(error: IntegrityError) -> str | None:
    pending: list[BaseException] = [error]
    visited: set[int] = set()
    while pending and len(visited) < _MAX_RETAINED_EXCEPTION_NODES:
        current = pending.pop()
        identity = id(current)
        if identity in visited:
            continue
        visited.add(identity)
        name = getattr(current, "constraint_name", None)
        if isinstance(name, str):
            return name
        diagnostic = getattr(current, "diag", None)
        name = getattr(diagnostic, "constraint_name", None)
        if isinstance(name, str):
            return name
        original = getattr(current, "orig", None)
        if isinstance(original, BaseException):
            pending.append(original)
        for nested_error in (current.__cause__, current.__context__):
            if nested_error is not None:
                pending.append(nested_error)
    return None


def _is_idempotency_unique_conflict(error: IntegrityError) -> bool:
    return _constraint_name(error) == _IDEMPOTENCY_UNIQUE_CONSTRAINT


class KnowledgeBaseService:
    def __init__(
        self,
        *,
        session: AsyncSession,
        settings: Settings,
        repository_factory: RepositoryFactory = sqlalchemy_knowledge_base_repositories,
        clock: Clock = _utc_now,
    ) -> None:
        self._session = session
        self._settings = settings
        self._repository_factory = repository_factory
        self._clock = clock

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
            raise _internal_error()
        return value

    def _require_manage_actor(self, actor: AgentPrincipal) -> AgentPrincipal:
        return require_capability(actor, Capability.MANAGE)

    async def _lock_and_revalidate_actor(
        self,
        repositories: KnowledgeBaseRepositories,
        actor: AgentPrincipal,
        now: datetime,
    ) -> ApiKey:
        row = await repositories.actors.get_agent_for_update(actor.key_id)
        if (
            row is None
            or row.public_id != actor.public_id
            or row.key_type != "agent"
            or not _is_currently_active(row, now)
        ):
            raise _invalid_api_key_error()
        capabilities = row.capabilities
        if (
            type(capabilities) is not list
            or any(type(value) is not str for value in capabilities)
            or len(set(capabilities)) != len(capabilities)
        ):
            raise _internal_error()
        if Capability.MANAGE.value not in capabilities:
            raise BusinessError(403, "INSUFFICIENT_CAPABILITY", "Insufficient capability")
        return row

    @staticmethod
    async def _require_rerank_profile(
        repositories: KnowledgeBaseRepositories,
        profile_id: UUID,
    ) -> None:
        """Reject a profile that cannot rerank, before the pointer is stored.

        The foreign key only proves the row exists. Pointing a knowledge base at
        an embedding profile would satisfy it and then fail at query time, where
        the caller has no way to tell a misconfiguration from a provider outage.
        """
        capability = await repositories.model_profiles.capability_of(profile_id)
        if capability != "rerank":
            raise BusinessError(
                422,
                "INVALID_RERANK_PROFILE",
                "Rerank profile is invalid",
            )

    def _safe_knowledge_base(self, row: KnowledgeBase) -> SafeKnowledgeBase:
        try:
            return SafeKnowledgeBase(
                id=row.id,
                name=row.name,
                description=row.description,
                status=cast(KnowledgeBaseStatus, row.status),
                metadata=row.metadata_,
                resource_revision=row.resource_revision,
                mutation_revision=row.mutation_revision,
                filter_schema_revision=row.filter_schema_revision,
                active_index_generation_id=row.active_index_generation_id,
                pending_index_generation_id=row.pending_index_generation_id,
                rerank_profile_id=row.rerank_profile_id,
                created_at=row.created_at,
                updated_at=row.updated_at,
                etag=knowledge_base_etag(row.id, row.resource_revision),
            )
        except (AttributeError, TypeError, ValueError, ValidationError) as error:
            _raise_internal(error)

    def _safe_filter_schema(self, row: KnowledgeBase) -> SafeFilterSchema:
        try:
            stored_fields = row.filter_schema["fields"]
            if type(stored_fields) is not list:
                raise TypeError("stored filter schema fields must be a list")
            fields: list[FilterSchemaField] = []
            for stored_field in stored_fields:
                if type(stored_field) is not dict:
                    raise TypeError("stored filter schema field must be an object")
                fields.append(
                    FilterSchemaField.model_validate(
                        {
                            "name": stored_field["name"],
                            "source_path": stored_field["source_path"],
                            "type": stored_field["type"],
                            "operators": stored_field["operators"],
                        }
                    )
                )
            return SafeFilterSchema(
                fields=tuple(fields),
                resource_revision=row.resource_revision,
                mutation_revision=row.mutation_revision,
                filter_schema_revision=row.filter_schema_revision,
                etag=knowledge_base_etag(row.id, row.resource_revision),
            )
        except (AttributeError, KeyError, TypeError, ValueError, ValidationError) as error:
            _raise_internal(error)

    async def _replay_create(
        self,
        repositories: KnowledgeBaseRepositories,
        actor: AgentPrincipal,
        fingerprint: bytes,
        record: IdempotencyRecord,
    ) -> KnowledgeBaseCreateResult:
        if record.request_fingerprint != fingerprint:
            raise _idempotency_conflict_error()
        if record.result_resource_type != "knowledge_base" or record.http_status != 201:
            raise _internal_error()
        row = await repositories.knowledge_bases.get_scoped(
            actor.key_id,
            record.result_resource_id,
        )
        if row is None:
            raise _not_found_error()
        return KnowledgeBaseCreateResult(
            knowledge_base=self._safe_knowledge_base(row),
            created=False,
        )

    async def create_knowledge_base(
        self,
        command: KnowledgeBaseCreate,
        *,
        actor: AgentPrincipal,
        request_id: str,
        idempotency_key: str,
    ) -> KnowledgeBaseCreateResult:
        agent = self._require_manage_actor(actor)
        _require_request_id(request_id, self._settings.max_request_id_length)
        validated_idempotency_key = validate_idempotency_key(
            idempotency_key,
            self._settings.max_idempotency_key_length,
        )
        if type(command) is not KnowledgeBaseCreate:
            raise _validation_error()
        snapshot = command.model_copy(deep=True)
        fingerprint = knowledge_base_create_fingerprint(snapshot)
        try:
            async with self._session.begin():
                repositories = self._repository_factory(self._session)
                existing = await repositories.idempotency.get(
                    agent.key_id,
                    _CREATE_OPERATION,
                    validated_idempotency_key,
                )
                if existing is not None:
                    await self._lock_and_revalidate_actor(repositories, agent, self._now())
                    return await self._replay_create(
                        repositories,
                        agent,
                        fingerprint,
                        existing,
                    )

                row: KnowledgeBase | None = None
                try:
                    async with self._session.begin_nested():
                        now = self._now()
                        actor_row = await self._lock_and_revalidate_actor(
                            repositories,
                            agent,
                            now,
                        )
                        row = KnowledgeBase(
                            id=uuid4(),
                            name=snapshot.name,
                            description=snapshot.description,
                            status="active",
                            metadata_=snapshot.metadata,
                            filter_schema={"fields": []},
                            resource_revision=1,
                            mutation_revision=0,
                            filter_schema_revision=0,
                            active_index_generation_id=None,
                            pending_index_generation_id=None,
                        )
                        await repositories.knowledge_bases.add(row)
                        await repositories.actors.add_scope(agent.key_id, row.id)
                        actor_row.resource_revision += 1
                        actor_row.updated_at = now
                        await repositories.idempotency.add(
                            IdempotencyRecord(
                                id=uuid4(),
                                actor_key_id=agent.key_id,
                                operation=_CREATE_OPERATION,
                                idempotency_key=validated_idempotency_key,
                                request_fingerprint=fingerprint,
                                result_resource_type="knowledge_base",
                                result_resource_id=row.id,
                                http_status=201,
                            )
                        )
                        await repositories.audits.add(
                            AuditEvent(
                                id=uuid4(),
                                request_id=request_id,
                                actor_api_key_id=agent.key_id,
                                actor_kind="agent_key",
                                action="knowledge_base.created",
                                target_type="knowledge_base",
                                target_id=row.id,
                                metadata_={},
                            )
                        )
                except IntegrityError as error:
                    if not _is_idempotency_unique_conflict(error):
                        raise
                    _sanitize_retained_exception(error)
                    winner = await repositories.idempotency.get(
                        agent.key_id,
                        _CREATE_OPERATION,
                        validated_idempotency_key,
                    )
                    if winner is None:
                        raise _internal_error() from None
                    return await self._replay_create(
                        repositories,
                        agent,
                        fingerprint,
                        winner,
                    )

                if row is None:
                    raise _internal_error()
                return KnowledgeBaseCreateResult(
                    knowledge_base=self._safe_knowledge_base(row),
                    created=True,
                )
        except BusinessError:
            command = KnowledgeBaseCreate(name="<redacted>")
            snapshot = command
            idempotency_key = "<redacted>"
            validated_idempotency_key = "<redacted>"
            raise
        except Exception as error:
            command = KnowledgeBaseCreate(name="<redacted>")
            snapshot = command
            idempotency_key = "<redacted>"
            validated_idempotency_key = "<redacted>"
            _raise_internal(error)
        except BaseException:
            command = KnowledgeBaseCreate(name="<redacted>")
            snapshot = command
            idempotency_key = "<redacted>"
            validated_idempotency_key = "<redacted>"
            raise

    async def list_knowledge_bases(
        self,
        *,
        actor: AgentPrincipal,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> KnowledgeBasePage:
        agent = self._require_manage_actor(actor)
        try:
            position, page_limit = self._page_parameters(cursor, limit)
            repositories = self._repository_factory(self._session)
            rows = await repositories.knowledge_bases.list_scoped(
                agent.key_id,
                position,
                page_limit + 1,
            )
            has_more = len(rows) > page_limit
            visible = rows[:page_limit]
            return KnowledgeBasePage(
                items=tuple(self._safe_knowledge_base(row) for row in visible),
                next_cursor=self._next_cursor(visible, has_more),
            )
        except BusinessError:
            raise
        except Exception as error:
            _raise_internal(error)

    async def get_knowledge_base(
        self,
        knowledge_base_id: UUID,
        *,
        actor: AgentPrincipal,
    ) -> SafeKnowledgeBase:
        agent = self._require_manage_actor(actor)
        identifier = _require_uuid(knowledge_base_id)
        try:
            repositories = self._repository_factory(self._session)
            row = await repositories.knowledge_bases.get_scoped(agent.key_id, identifier)
            if row is None:
                raise _not_found_error()
            return self._safe_knowledge_base(row)
        except BusinessError:
            raise
        except Exception as error:
            _raise_internal(error)

    async def update_knowledge_base(
        self,
        knowledge_base_id: UUID,
        command: KnowledgeBasePatch,
        *,
        actor: AgentPrincipal,
        request_id: str,
        expected_etag: str | None,
    ) -> SafeKnowledgeBase:
        agent = self._require_manage_actor(actor)
        identifier = _require_uuid(knowledge_base_id)
        _require_request_id(request_id, self._settings.max_request_id_length)
        if type(command) is not KnowledgeBasePatch:
            raise _validation_error()
        snapshot = command.model_copy(deep=True)
        try:
            async with self._session.begin():
                repositories = self._repository_factory(self._session)
                now = self._now()
                await self._lock_and_revalidate_actor(repositories, agent, now)
                row = await repositories.knowledge_bases.get_scoped(
                    agent.key_id,
                    identifier,
                    for_update=True,
                )
                if row is None:
                    raise _not_found_error()
                require_matching_etag(
                    expected_etag,
                    knowledge_base_etag(row.id, row.resource_revision),
                )
                if row.status == "deleting":
                    raise _state_conflict_error()

                fields = snapshot.model_fields_set
                if "name" in fields:
                    if snapshot.name is None:
                        raise _validation_error()
                    row.name = snapshot.name
                if "description" in fields:
                    row.description = snapshot.description
                if "metadata" in fields:
                    if snapshot.metadata is None:
                        raise _validation_error()
                    row.metadata_ = snapshot.metadata
                if "status" in fields:
                    if snapshot.status is None:
                        raise _validation_error()
                    row.status = snapshot.status
                if "rerank_profile_id" in fields:
                    if snapshot.rerank_profile_id is not None:
                        await self._require_rerank_profile(repositories, snapshot.rerank_profile_id)
                    row.rerank_profile_id = snapshot.rerank_profile_id
                row.resource_revision += 1
                row.updated_at = now
                await repositories.knowledge_bases.save(row)
                await repositories.audits.add(
                    AuditEvent(
                        id=uuid4(),
                        request_id=request_id,
                        actor_api_key_id=agent.key_id,
                        actor_kind="agent_key",
                        action="knowledge_base.updated",
                        target_type="knowledge_base",
                        target_id=row.id,
                        metadata_={"changed_fields": sorted(fields)},
                    )
                )
                return self._safe_knowledge_base(row)
        except BusinessError:
            command = KnowledgeBasePatch(name="<redacted>")
            snapshot = command
            raise
        except Exception as error:
            command = KnowledgeBasePatch(name="<redacted>")
            snapshot = command
            _raise_internal(error)
        except BaseException:
            command = KnowledgeBasePatch(name="<redacted>")
            snapshot = command
            raise

    async def replace_filter_schema(
        self,
        knowledge_base_id: UUID,
        command: FilterSchemaReplacement,
        *,
        actor: AgentPrincipal,
        request_id: str,
        expected_etag: str | None,
    ) -> SafeFilterSchema:
        agent = self._require_manage_actor(actor)
        identifier = _require_uuid(knowledge_base_id)
        _require_request_id(request_id, self._settings.max_request_id_length)
        if type(command) is not FilterSchemaReplacement:
            raise _validation_error()
        snapshot = command.model_copy(deep=True)
        repositories: KnowledgeBaseRepositories | None = None
        row: KnowledgeBase | None = None
        canonical: dict[str, object] = {}
        stored_fields: list[object] = []
        stored_field: object = {}
        mutation_fields: list[dict[str, object]] = []
        try:
            async with self._session.begin():
                repositories = self._repository_factory(self._session)
                now = self._now()
                await self._lock_and_revalidate_actor(repositories, agent, now)
                row = await repositories.knowledge_bases.get_scoped(
                    agent.key_id,
                    identifier,
                    for_update=True,
                )
                if row is None:
                    raise _not_found_error()
                require_matching_etag(
                    expected_etag,
                    knowledge_base_etag(row.id, row.resource_revision),
                )
                if row.status == "deleting":
                    raise _state_conflict_error()

                canonical = canonicalize_filter_schema(snapshot, row.filter_schema)
                generation_statuses = (
                    await self._session.scalars(
                        select(KnowledgeBaseIndexGeneration.status)
                        .where(KnowledgeBaseIndexGeneration.knowledge_base_id == row.id)
                        .order_by(KnowledgeBaseIndexGeneration.id)
                        .with_for_update()
                    )
                ).all()
                frozen = any(status in {"active", "building"} for status in generation_statuses)
                if frozen and canonical == row.filter_schema:
                    return self._safe_filter_schema(row)
                if frozen:
                    raise _reindex_required_error()
                row.filter_schema = canonical
                row.resource_revision += 1
                row.mutation_revision += 1
                row.filter_schema_revision += 1
                row.updated_at = now
                await repositories.knowledge_bases.save(row)

                stored_fields = cast(list[object], canonical["fields"])
                if type(stored_fields) is not list:
                    raise TypeError("canonical filter schema fields must be a list")
                for stored_field in stored_fields:
                    if type(stored_field) is not dict:
                        raise TypeError("canonical filter schema field must be an object")
                    mutation_fields.append(
                        {
                            "field_id": stored_field["field_id"],
                            "type": stored_field["type"],
                            "operators": stored_field["operators"],
                        }
                    )
                await repositories.mutations.add(
                    KnowledgeBaseMutation(
                        id=uuid4(),
                        knowledge_base_id=row.id,
                        revision=row.mutation_revision,
                        mutation_type="filter_schema_changed",
                        target_type="knowledge_base",
                        target_id=row.id,
                        payload={
                            "filter_schema_revision": row.filter_schema_revision,
                            "fields": mutation_fields,
                        },
                    )
                )
                await repositories.audits.add(
                    AuditEvent(
                        id=uuid4(),
                        request_id=request_id,
                        actor_api_key_id=agent.key_id,
                        actor_kind="agent_key",
                        action="knowledge_base.filter_schema_replaced",
                        target_type="knowledge_base",
                        target_id=row.id,
                        metadata_={
                            "field_count": len(stored_fields),
                            "resource_revision": row.resource_revision,
                            "mutation_revision": row.mutation_revision,
                            "filter_schema_revision": row.filter_schema_revision,
                        },
                    )
                )
                return self._safe_filter_schema(row)
        except BusinessError as error:
            command = FilterSchemaReplacement(fields=())
            snapshot = command
            row = None
            canonical = {}
            stored_fields = []
            stored_field = {}
            mutation_fields = []
            repositories = None
            _raise_sanitized_business_error(error)
        except Exception as error:
            command = FilterSchemaReplacement(fields=())
            snapshot = command
            row = None
            canonical = {}
            stored_fields = []
            stored_field = {}
            mutation_fields = []
            repositories = None
            _raise_internal(error)
        except BaseException as error:
            command = FilterSchemaReplacement(fields=())
            snapshot = command
            row = None
            canonical = {}
            stored_fields = []
            stored_field = {}
            mutation_fields = []
            repositories = None
            _reraise_redacted_base_exception(error)

    async def delete_knowledge_base(
        self,
        knowledge_base_id: UUID,
        *,
        actor: AgentPrincipal,
        request_id: str,
        expected_etag: str | None,
    ) -> SafeKnowledgeBase:
        agent = self._require_manage_actor(actor)
        identifier = _require_uuid(knowledge_base_id)
        _require_request_id(request_id, self._settings.max_request_id_length)
        try:
            async with self._session.begin():
                repositories = self._repository_factory(self._session)
                now = self._now()
                await self._lock_and_revalidate_actor(repositories, agent, now)
                row = await repositories.knowledge_bases.get_scoped(
                    agent.key_id,
                    identifier,
                    for_update=True,
                )
                if row is None:
                    raise _not_found_error()
                require_matching_etag(
                    expected_etag,
                    knowledge_base_etag(row.id, row.resource_revision),
                )
                if row.status == "deleting":
                    return self._safe_knowledge_base(row)

                row.status = "deleting"
                row.resource_revision += 1
                row.updated_at = now
                await repositories.knowledge_bases.save(row)
                # The purge job names the knowledge base only through target_id,
                # which carries no foreign key, so it survives the row it deletes.
                await repositories.jobs.add(
                    Job(
                        id=uuid4(),
                        knowledge_base_id=None,
                        actor_api_key_id=agent.key_id,
                        target_type="knowledge_base",
                        target_id=row.id,
                        operation="purge_knowledge_base",
                        status="queued",
                    )
                )
                await repositories.audits.add(
                    AuditEvent(
                        id=uuid4(),
                        request_id=request_id,
                        actor_api_key_id=agent.key_id,
                        actor_kind="agent_key",
                        action="knowledge_base.deletion_requested",
                        target_type="knowledge_base",
                        target_id=row.id,
                        metadata_={},
                    )
                )
                return self._safe_knowledge_base(row)
        except BusinessError:
            raise
        except Exception as error:
            _raise_internal(error)
        except BaseException:
            raise

    def _page_parameters(
        self,
        cursor: str | None,
        limit: int | None,
    ) -> tuple[CursorPosition | None, int]:
        page_limit = self._settings.default_page_size if limit is None else limit
        if type(page_limit) is not int or not 1 <= page_limit <= self._settings.max_page_size:
            raise _validation_error()
        position = None if cursor is None else decode_cursor(cursor)
        return position, page_limit

    def _next_cursor(self, rows: list[KnowledgeBase], has_more: bool) -> str | None:
        if not has_more or not rows:
            return None
        last = rows[-1]
        try:
            return encode_cursor(CursorPosition(created_at=last.created_at, id=last.id))
        except (AttributeError, TypeError, ValueError) as error:
            _raise_internal(error)


class DocumentMetadataService:
    def __init__(
        self,
        *,
        session: AsyncSession,
        settings: Settings,
        repository_factory: DocumentRepositoryFactory = sqlalchemy_document_metadata_repository,
        object_store: DocumentContentStore | None = None,
    ) -> None:
        self._session = session
        self._settings = settings
        self._repository_factory = repository_factory
        # Only the content read needs it, so it stays optional: every other
        # caller of this service constructs it without a store.
        self._object_store = object_store

    def _page_parameters(
        self,
        cursor: str | None,
        limit: int,
    ) -> tuple[CursorPosition | None, int]:
        maximum = min(self._settings.max_page_size, 100)
        if type(limit) is not int or not 1 <= limit <= maximum:
            raise _validation_error()
        return (None if cursor is None else decode_cursor(cursor), limit)

    @staticmethod
    def _next_cursor(
        rows: list[DocumentRecord] | list[DocumentVersionRecord],
        has_more: bool,
    ) -> str | None:
        if not has_more or not rows:
            return None
        last = rows[-1]
        return encode_cursor(CursorPosition(created_at=last.created_at, id=last.id))

    @staticmethod
    def _safe_document(row: DocumentRecord) -> SafeDocument:
        try:
            if type(row.tags) is not list or any(type(tag) is not str for tag in row.tags):
                raise TypeError("stored document tags are invalid")
            return SafeDocument(
                id=row.id,
                knowledge_base_id=row.knowledge_base_id,
                display_name=row.display_name,
                mime_type=row.mime_type,
                checksum_sha256=row.checksum_sha256,
                current_version_id=row.current_version_id,
                pending_version_id=row.pending_version_id,
                status=cast(DocumentStatus, row.status),
                tags=tuple(row.tags),
                metadata=row.metadata,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
        except (AttributeError, TypeError, ValueError, ValidationError) as error:
            _raise_internal(error)

    @staticmethod
    def _safe_version(row: DocumentVersionRecord) -> SafeVersion:
        try:
            return SafeVersion(
                id=row.id,
                document_id=row.document_id,
                version_number=row.version_number,
                source_checksum_sha256=row.source_checksum_sha256,
                parsed_object_checksum_sha256=row.parsed_object_checksum_sha256,
                declared_mime_type=row.declared_mime_type,
                detected_mime_type=row.detected_mime_type,
                source_extension=row.source_extension,
                base_version_id=row.base_version_id,
                parser_name=row.parser_name,
                parser_version=row.parser_version,
                parser_config=row.parser_config,
                chunker_name=row.chunker_name,
                chunker_version=row.chunker_version,
                chunker_config=row.chunker_config,
                chunk_count=row.chunk_count,
                status=cast(DocumentVersionStatus, row.status),
                activated_at=row.activated_at,
                created_at=row.created_at,
            )
        except (AttributeError, TypeError, ValueError, ValidationError) as error:
            _raise_internal(error)

    async def list_documents(
        self,
        knowledge_base_id: UUID,
        *,
        actor: AgentPrincipal,
        cursor: str | None,
        limit: int,
    ) -> DocumentPage:
        identifier: UUID | None = None
        position: CursorPosition | None = None
        repository: DocumentMetadataRepository | None = None
        parent_id: UUID | None = None
        agent: AgentPrincipal | None = None
        rows: list[DocumentRecord] = []
        visible: list[DocumentRecord] = []
        try:
            identifier = _require_uuid(knowledge_base_id)
            position, page_limit = self._page_parameters(cursor, limit)
            repository = self._repository_factory(self._session)
            parent_id = await repository.get_scoped_parent(actor.key_id, identifier)
            agent = require_document_read(
                actor,
                identifier,
                parent_knowledge_base_exists=parent_id is not None,
            )
            rows = await repository.list_documents(
                agent.key_id,
                identifier,
                position,
                page_limit + 1,
            )
            has_more = len(rows) > page_limit
            visible = rows[:page_limit]
            return DocumentPage(
                items=tuple(self._safe_document(row) for row in visible),
                next_cursor=self._next_cursor(visible, has_more),
            )
        except BusinessError as error:
            knowledge_base_id = UUID(int=0)
            cursor = "<redacted>"
            identifier = None
            position = None
            repository = None
            parent_id = None
            agent = None
            rows = []
            visible = []
            _raise_sanitized_business_error(error)
        except Exception as error:
            knowledge_base_id = UUID(int=0)
            cursor = "<redacted>"
            identifier = None
            position = None
            repository = None
            parent_id = None
            agent = None
            rows = []
            visible = []
            _raise_internal(error)
        except BaseException as error:
            knowledge_base_id = UUID(int=0)
            cursor = "<redacted>"
            identifier = None
            position = None
            repository = None
            parent_id = None
            agent = None
            rows = []
            visible = []
            _reraise_redacted_base_exception(error)

    async def get_document(
        self,
        document_id: UUID,
        *,
        actor: AgentPrincipal,
    ) -> SafeDocument:
        identifier: UUID | None = None
        repository: DocumentMetadataRepository | None = None
        row: DocumentRecord | None = None
        try:
            identifier = _require_uuid(document_id)
            repository = self._repository_factory(self._session)
            row = await repository.get_document(actor.key_id, identifier)
            if row is None:
                raise _not_found_error()
            require_document_read(
                actor,
                row.knowledge_base_id,
                parent_knowledge_base_exists=True,
            )
            return self._safe_document(row)
        except BusinessError as error:
            document_id = UUID(int=0)
            identifier = None
            repository = None
            row = None
            _raise_sanitized_business_error(error)
        except Exception as error:
            document_id = UUID(int=0)
            identifier = None
            repository = None
            row = None
            _raise_internal(error)
        except BaseException as error:
            document_id = UUID(int=0)
            identifier = None
            repository = None
            row = None
            _reraise_redacted_base_exception(error)

    async def list_versions(
        self,
        document_id: UUID,
        *,
        actor: AgentPrincipal,
        cursor: str | None,
        limit: int,
    ) -> DocumentVersionPage:
        identifier: UUID | None = None
        position: CursorPosition | None = None
        repository: DocumentMetadataRepository | None = None
        parent_id: UUID | None = None
        rows: list[DocumentVersionRecord] = []
        visible: list[DocumentVersionRecord] = []
        try:
            identifier = _require_uuid(document_id)
            position, page_limit = self._page_parameters(cursor, limit)
            repository = self._repository_factory(self._session)
            parent_id = await repository.get_document_parent(actor.key_id, identifier)
            if parent_id is None:
                raise _not_found_error()
            require_document_read(
                actor,
                parent_id,
                parent_knowledge_base_exists=True,
            )
            rows = await repository.list_versions(
                actor.key_id,
                identifier,
                position,
                page_limit + 1,
            )
            has_more = len(rows) > page_limit
            visible = rows[:page_limit]
            return DocumentVersionPage(
                items=tuple(self._safe_version(row) for row in visible),
                next_cursor=self._next_cursor(visible, has_more),
            )
        except BusinessError as error:
            document_id = UUID(int=0)
            cursor = "<redacted>"
            identifier = None
            position = None
            repository = None
            parent_id = None
            rows = []
            visible = []
            _raise_sanitized_business_error(error)
        except Exception as error:
            document_id = UUID(int=0)
            cursor = "<redacted>"
            identifier = None
            position = None
            repository = None
            parent_id = None
            rows = []
            visible = []
            _raise_internal(error)
        except BaseException as error:
            document_id = UUID(int=0)
            cursor = "<redacted>"
            identifier = None
            position = None
            repository = None
            parent_id = None
            rows = []
            visible = []
            _reraise_redacted_base_exception(error)

    async def read_document_content(
        self,
        document_id: UUID,
        *,
        actor: AgentPrincipal,
        start_offset: int,
        end_offset: int,
    ) -> DocumentContent:
        """Return a slice of the document's normalized text by codepoint offsets.

        The whole object is fetched and sliced in memory rather than range-read
        from storage, because read_bytes verifies the object against its recorded
        checksum end to end and a ranged read cannot. Documents are bounded by the
        upload limit, so the cost is bounded too; if that stops holding, the
        replacement needs its own integrity story rather than dropping this one.
        """

        identifier: UUID | None = None
        repository: DocumentMetadataRepository | None = None
        row: DocumentRecord | None = None
        version: DocumentVersionRecord | None = None
        payload = b""
        text = ""
        try:
            identifier = _require_uuid(document_id)
            if (
                type(start_offset) is not int
                or type(end_offset) is not int
                or start_offset < 0
                or end_offset < start_offset
            ):
                raise _validation_error()
            store = self._object_store
            if store is None:
                raise _internal_error()
            repository = self._repository_factory(self._session)
            row = await repository.get_document(actor.key_id, identifier)
            if row is None:
                raise _not_found_error()
            require_raw_file_read(
                actor,
                row.knowledge_base_id,
                parent_knowledge_base_exists=True,
            )
            version = await repository.get_current_version(actor.key_id, identifier)
            if version is None or version.parsed_object_checksum_sha256 is None:
                # No active version yet: the document exists but has no text to
                # serve. A conflict rather than a 404, because the document is
                # visible and this is a state the caller can wait out.
                raise _state_conflict_error()
            payload = await store.read_bytes(
                parsed_text_object_key(row.knowledge_base_id, row.id, version.id),
                expected_checksum=version.parsed_object_checksum_sha256,
                max_bytes=self._settings.max_upload_bytes,
            )
            text = payload.decode("utf-8", errors="strict")
            # Clamped, not rejected: a caller widening around a hit near the end
            # of a document should get what exists rather than an error.
            total = len(text)
            begin = min(start_offset, total)
            finish = min(end_offset, total)
            return DocumentContent(
                document_id=row.id,
                version_id=version.id,
                start_offset=begin,
                end_offset=finish,
                total_codepoints=total,
                text=text[begin:finish],
            )
        except BusinessError as error:
            document_id = UUID(int=0)
            identifier = None
            repository = None
            row = None
            version = None
            payload = b""
            text = ""
            _raise_sanitized_business_error(error)
        except Exception as error:
            document_id = UUID(int=0)
            identifier = None
            repository = None
            row = None
            version = None
            payload = b""
            text = ""
            _raise_internal(error)
        except BaseException as error:
            document_id = UUID(int=0)
            identifier = None
            repository = None
            row = None
            version = None
            payload = b""
            text = ""
            _reraise_redacted_base_exception(error)
