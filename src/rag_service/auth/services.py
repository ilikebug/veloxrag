import traceback
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from typing import Literal, Never
from uuid import UUID, uuid4

from pydantic import SecretStr, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.api.cursors import CursorPosition, decode_cursor, encode_cursor
from rag_service.api.errors import BusinessError
from rag_service.api.etags import agent_key_etag, require_matching_etag
from rag_service.api.middleware import is_valid_request_id
from rag_service.auth.codec import (
    GeneratedToken,
    KeyKind,
    ParsedToken,
    generate_token,
    parse_token,
    verify_secret,
)
from rag_service.auth.policies import (
    AdminPrincipal,
    Capability,
    Principal,
    materialize_principal,
)
from rag_service.auth.repositories import AuthRepositories, sqlalchemy_auth_repositories
from rag_service.auth.schemas import (
    AdminApiKeyCreate,
    AgentApiKeyCreate,
    AgentApiKeyUpdate,
    IssuedApiKey,
    Page,
    SafeApiKey,
)
from rag_service.config import Settings
from rag_service.db.models.auth import ApiKey, AuditEvent

type ActorKind = Literal["admin_key", "local_cli"]
type RepositoryFactory = Callable[[AsyncSession], AuthRepositories]
type AuthenticationSessions = Callable[[], AbstractAsyncContextManager[AsyncSession]]
type Clock = Callable[[], datetime]

_MAX_RETAINED_EXCEPTION_NODES = 32
_DUMMY_SECRET_DIGEST = b"\x00" * 32


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _invalid_api_key_error() -> BusinessError:
    return BusinessError(401, "INVALID_API_KEY", "Invalid API key")


def _not_found_error() -> BusinessError:
    return BusinessError(404, "RESOURCE_NOT_FOUND", "Resource not found")


def _state_conflict_error() -> BusinessError:
    return BusinessError(409, "RESOURCE_STATE_CONFLICT", "Resource state conflict")


def _validation_error() -> BusinessError:
    return BusinessError(422, "VALIDATION_ERROR", "Invalid API key policy")


def _internal_error() -> BusinessError:
    return BusinessError(500, "INTERNAL_ERROR", "Internal server error")


def _sanitize_retained_exception(error: BaseException) -> None:
    pending = [error]
    visited: set[int] = set()
    while pending and len(visited) < _MAX_RETAINED_EXCEPTION_NODES:
        current = pending.pop()
        identity = id(current)
        if identity in visited:
            continue
        visited.add(identity)
        for nested_error in (current.__cause__, current.__context__):
            if nested_error is not None:
                pending.append(nested_error)
        if current.__traceback__ is not None:
            traceback.clear_frames(current.__traceback__)
        if isinstance(current, Exception):
            current.args = ("<redacted>",)
        current.__traceback__ = None
        current.__cause__ = None
        current.__context__ = None


def _raise_internal(error: BaseException) -> Never:
    _sanitize_retained_exception(error)
    raise _internal_error() from None


def _redacted_generated_token() -> GeneratedToken:
    return GeneratedToken(token="<redacted>", public_id="<redacted>", digest=b"")


def _redacted_parsed_token() -> ParsedToken:
    return ParsedToken(kind=KeyKind.ADMIN, public_id="<redacted>", secret="<redacted>")


def _require_request_id(request_id: object, max_length: int) -> None:
    if not is_valid_request_id(request_id, max_length):
        raise _validation_error()


def _require_uuid(value: UUID) -> None:
    if not isinstance(value, UUID):
        raise _validation_error()


def _is_currently_active(row: ApiKey, now: datetime) -> bool:
    if row.status != "active" or row.revoked_at is not None:
        return False
    if row.not_before is not None and row.not_before > now:
        return False
    return row.expires_at is None or row.expires_at > now


def _sorted_uuids(values: frozenset[UUID]) -> tuple[UUID, ...]:
    return tuple(sorted(values, key=str))


def _sorted_uuid_strings(values: frozenset[UUID]) -> list[str]:
    return [str(identifier) for identifier in sorted(values, key=str)]


class ApiKeyService:
    def __init__(
        self,
        *,
        session: AsyncSession,
        authentication_sessions: AuthenticationSessions,
        settings: Settings,
        repository_factory: RepositoryFactory = sqlalchemy_auth_repositories,
        clock: Clock = _utc_now,
    ) -> None:
        self._session = session
        self._authentication_sessions = authentication_sessions
        self._settings = settings
        self._repository_factory = repository_factory
        self._clock = clock

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
            raise _internal_error()
        return value

    def _hmac_secret(self, kind: KeyKind) -> SecretStr:
        if kind is KeyKind.ADMIN:
            return self._settings.admin_key_hmac_secret
        if kind is KeyKind.AGENT:
            return self._settings.agent_key_hmac_secret
        raise _internal_error()

    async def _require_admin_actor(
        self,
        repositories: AuthRepositories,
        actor: AdminPrincipal | None,
        now: datetime,
    ) -> tuple[UUID | None, ActorKind]:
        if actor is None:
            return None, "local_cli"
        if type(actor) is not AdminPrincipal:
            raise _invalid_api_key_error()
        row = await repositories.api_keys.get_by_id(
            actor.key_id,
            KeyKind.ADMIN,
            for_update=True,
        )
        if (
            row is None
            or row.public_id != actor.public_id
            or row.key_type != KeyKind.ADMIN.value
            or not _is_currently_active(row, now)
        ):
            raise _invalid_api_key_error()
        return row.id, "admin_key"

    def _validate_agent_limits(self, requests_per_minute: int, max_concurrency: int) -> None:
        if (
            type(requests_per_minute) is not int
            or not 1 <= requests_per_minute <= self._settings.max_api_key_requests_per_minute
            or type(max_concurrency) is not int
            or not 1 <= max_concurrency <= self._settings.max_api_key_concurrency
        ):
            raise _validation_error()

    def _validate_capabilities(self, capabilities: frozenset[Capability]) -> None:
        if len(capabilities) > len(Capability) or any(
            type(capability) is not Capability for capability in capabilities
        ):
            raise _validation_error()

    def _validate_validity_window(
        self,
        not_before: datetime | None,
        expires_at: datetime | None,
    ) -> None:
        for value in (not_before, expires_at):
            if value is not None and (
                value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value)
            ):
                raise _validation_error()
        if not_before is not None and expires_at is not None and expires_at <= not_before:
            raise _validation_error()

    async def _validate_references(
        self,
        repositories: AuthRepositories,
        knowledge_base_ids: frozenset[UUID],
        query_profile_ids: frozenset[UUID],
        default_query_profile_id: UUID | None,
        *,
        validate_knowledge_bases: bool = True,
        validate_query_profiles: bool = True,
    ) -> None:
        if (
            default_query_profile_id is not None
            and default_query_profile_id not in query_profile_ids
        ):
            raise _validation_error()
        if validate_knowledge_bases:
            statuses = await repositories.scopes.get_knowledge_base_statuses(
                knowledge_base_ids,
                for_update=True,
            )
            if set(statuses) != set(knowledge_base_ids) or any(
                status == "deleting" for status in statuses.values()
            ):
                raise _validation_error()
        if validate_query_profiles:
            enabled = await repositories.scopes.get_query_profile_enabled(
                query_profile_ids,
                for_update=True,
            )
            if set(enabled) != set(query_profile_ids) or not all(enabled.values()):
                raise _validation_error()

    async def _safe_api_key(
        self,
        row: ApiKey,
        knowledge_base_ids: frozenset[UUID] = frozenset(),
        query_profile_ids: frozenset[UUID] = frozenset(),
        default_query_profile_id: UUID | None = None,
    ) -> SafeApiKey:
        try:
            if row.key_type == KeyKind.ADMIN.value:
                if (
                    row.capabilities != []
                    or row.raw_file_read is not False
                    or row.requests_per_minute is not None
                    or row.max_concurrency is not None
                    or knowledge_base_ids
                    or query_profile_ids
                    or default_query_profile_id is not None
                ):
                    raise ValueError
                capabilities: tuple[Capability, ...] = ()
                etag = None
            elif row.key_type == KeyKind.AGENT.value:
                capabilities = tuple(
                    sorted((Capability(value) for value in row.capabilities), key=str)
                )
                if len(capabilities) != len(row.capabilities):
                    raise ValueError
                etag = agent_key_etag(row.id, row.resource_revision)
            else:
                raise ValueError
            return SafeApiKey(
                id=row.id,
                public_id=row.public_id,
                name=row.name,
                status=row.status,
                key_type=row.key_type,
                capabilities=capabilities,
                knowledge_base_ids=_sorted_uuids(knowledge_base_ids),
                query_profile_ids=_sorted_uuids(query_profile_ids),
                default_query_profile_id=default_query_profile_id,
                raw_file_read=row.raw_file_read,
                requests_per_minute=row.requests_per_minute,
                max_concurrency=row.max_concurrency,
                not_before=row.not_before,
                expires_at=row.expires_at,
                resource_revision=row.resource_revision,
                etag=etag,
                created_at=row.created_at,
                updated_at=row.updated_at,
                revoked_at=row.revoked_at,
            )
        except (AttributeError, TypeError, ValueError, ValidationError) as error:
            _raise_internal(error)

    async def _safe_agent(
        self,
        repositories: AuthRepositories,
        row: ApiKey,
    ) -> SafeApiKey:
        knowledge_base_ids = await repositories.scopes.get_knowledge_base_scopes(row.id)
        (
            query_profile_ids,
            default_query_profile_id,
        ) = await repositories.scopes.get_query_profile_scopes(row.id)
        return await self._safe_api_key(
            row,
            knowledge_base_ids,
            query_profile_ids,
            default_query_profile_id,
        )

    async def create_admin_key(
        self,
        command: AdminApiKeyCreate,
        *,
        request_id: str,
    ) -> IssuedApiKey:
        _require_request_id(request_id, self._settings.max_request_id_length)
        generated: GeneratedToken | None = None
        try:
            self._validate_validity_window(command.not_before, command.expires_at)
            async with self._session.begin():
                repositories = self._repository_factory(self._session)
                generated = generate_token(
                    KeyKind.ADMIN,
                    self._settings.admin_key_hmac_secret,
                )
                row = ApiKey(
                    id=uuid4(),
                    public_id=generated.public_id,
                    secret_digest=generated.digest,
                    key_type=KeyKind.ADMIN.value,
                    name=command.name,
                    status="active",
                    capabilities=[],
                    raw_file_read=False,
                    requests_per_minute=None,
                    max_concurrency=None,
                    not_before=command.not_before,
                    expires_at=command.expires_at,
                    resource_revision=1,
                )
                await repositories.api_keys.add(row)
                await repositories.audits.add(
                    AuditEvent(
                        request_id=request_id,
                        actor_api_key_id=None,
                        actor_kind="local_cli",
                        action="api_key.created",
                        target_type="api_key",
                        target_id=row.id,
                        metadata_={},
                    )
                )
                safe = await self._safe_api_key(row)
            return IssuedApiKey(api_key=safe, token=SecretStr(generated.token))
        except BusinessError:
            if generated is not None:
                generated = _redacted_generated_token()
            raise
        except Exception as error:
            if generated is not None:
                generated = _redacted_generated_token()
            _raise_internal(error)
        except BaseException:
            if generated is not None:
                generated = _redacted_generated_token()
            raise

    async def list_admin_keys(
        self,
        *,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> Page[SafeApiKey]:
        try:
            position, page_limit = self._page_parameters(cursor, limit)
            repositories = self._repository_factory(self._session)
            rows = await repositories.api_keys.list_by_kind(
                KeyKind.ADMIN,
                position,
                page_limit + 1,
            )
            has_more = len(rows) > page_limit
            visible = rows[:page_limit]
            items = tuple([await self._safe_api_key(row) for row in visible])
            return Page(items=items, next_cursor=self._next_cursor(visible, has_more))
        except BusinessError:
            raise
        except Exception as error:
            _raise_internal(error)

    async def create_agent_key(
        self,
        command: AgentApiKeyCreate,
        *,
        actor: AdminPrincipal | None,
        request_id: str,
    ) -> IssuedApiKey:
        _require_request_id(request_id, self._settings.max_request_id_length)
        generated: GeneratedToken | None = None
        try:
            self._validate_capabilities(command.capabilities)
            self._validate_agent_limits(
                command.requests_per_minute,
                command.max_concurrency,
            )
            self._validate_validity_window(command.not_before, command.expires_at)
            async with self._session.begin():
                repositories = self._repository_factory(self._session)
                now = self._now()
                actor_id, actor_kind = await self._require_admin_actor(
                    repositories,
                    actor,
                    now,
                )
                await self._validate_references(
                    repositories,
                    command.knowledge_base_ids,
                    command.query_profile_ids,
                    command.default_query_profile_id,
                )
                generated = generate_token(
                    KeyKind.AGENT,
                    self._settings.agent_key_hmac_secret,
                )
                row = ApiKey(
                    id=uuid4(),
                    public_id=generated.public_id,
                    secret_digest=generated.digest,
                    key_type=KeyKind.AGENT.value,
                    name=command.name,
                    status="active",
                    capabilities=sorted(capability.value for capability in command.capabilities),
                    raw_file_read=command.raw_file_read,
                    requests_per_minute=command.requests_per_minute,
                    max_concurrency=command.max_concurrency,
                    not_before=command.not_before,
                    expires_at=command.expires_at,
                    resource_revision=1,
                    created_by_api_key_id=actor_id,
                )
                await repositories.api_keys.add(row)
                await repositories.api_keys.replace_kb_scopes(
                    row.id,
                    command.knowledge_base_ids,
                )
                await repositories.api_keys.replace_query_profile_scopes(
                    row.id,
                    command.query_profile_ids,
                    command.default_query_profile_id,
                )
                await repositories.audits.add(
                    AuditEvent(
                        request_id=request_id,
                        actor_api_key_id=actor_id,
                        actor_kind=actor_kind,
                        action="api_key.created",
                        target_type="api_key",
                        target_id=row.id,
                        metadata_={},
                    )
                )
                safe = await self._safe_api_key(
                    row,
                    command.knowledge_base_ids,
                    command.query_profile_ids,
                    command.default_query_profile_id,
                )
            return IssuedApiKey(api_key=safe, token=SecretStr(generated.token))
        except BusinessError:
            if generated is not None:
                generated = _redacted_generated_token()
            raise
        except Exception as error:
            if generated is not None:
                generated = _redacted_generated_token()
            _raise_internal(error)
        except BaseException:
            if generated is not None:
                generated = _redacted_generated_token()
            raise

    async def get_agent_key(self, key_id: UUID) -> SafeApiKey:
        _require_uuid(key_id)
        try:
            repositories = self._repository_factory(self._session)
            row = await repositories.api_keys.get_by_id(key_id, KeyKind.AGENT)
            if row is None:
                raise _not_found_error()
            return await self._safe_agent(repositories, row)
        except BusinessError:
            raise
        except Exception as error:
            _raise_internal(error)

    async def list_agent_keys(
        self,
        *,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> Page[SafeApiKey]:
        try:
            position, page_limit = self._page_parameters(cursor, limit)
            repositories = self._repository_factory(self._session)
            rows = await repositories.api_keys.list_by_kind(
                KeyKind.AGENT,
                position,
                page_limit + 1,
            )
            has_more = len(rows) > page_limit
            visible = rows[:page_limit]
            visible_ids = frozenset(row.id for row in visible)
            knowledge_base_ids_by_key = await repositories.scopes.get_knowledge_base_scopes_batch(
                visible_ids
            )
            query_profile_scopes_by_key = await repositories.scopes.get_query_profile_scopes_batch(
                visible_ids
            )
            items = tuple(
                [
                    await self._safe_api_key(
                        row,
                        knowledge_base_ids_by_key.get(row.id, frozenset()),
                        *query_profile_scopes_by_key.get(row.id, (frozenset(), None)),
                    )
                    for row in visible
                ]
            )
            return Page(items=items, next_cursor=self._next_cursor(visible, has_more))
        except BusinessError:
            raise
        except Exception as error:
            _raise_internal(error)

    async def update_agent_key(
        self,
        key_id: UUID,
        command: AgentApiKeyUpdate,
        *,
        actor: AdminPrincipal | None,
        request_id: str,
        expected_etag: str,
    ) -> SafeApiKey:
        _require_uuid(key_id)
        _require_request_id(request_id, self._settings.max_request_id_length)
        try:
            async with self._session.begin():
                repositories = self._repository_factory(self._session)
                now = self._now()
                actor_id, actor_kind = await self._require_admin_actor(
                    repositories,
                    actor,
                    now,
                )
                row = await repositories.api_keys.get_by_id(
                    key_id,
                    KeyKind.AGENT,
                    for_update=True,
                )
                if row is None:
                    raise _not_found_error()
                require_matching_etag(
                    expected_etag,
                    agent_key_etag(row.id, row.resource_revision),
                )
                if row.status == "revoked":
                    raise _state_conflict_error()

                old_kb_ids = await repositories.scopes.get_knowledge_base_scopes(row.id)
                old_qp_ids, old_default_id = await repositories.scopes.get_query_profile_scopes(
                    row.id
                )
                fields = command.model_fields_set
                new_kb_ids = (
                    command.knowledge_base_ids if "knowledge_base_ids" in fields else old_kb_ids
                )
                new_qp_ids = (
                    command.query_profile_ids if "query_profile_ids" in fields else old_qp_ids
                )
                new_default_id = (
                    command.default_query_profile_id
                    if "default_query_profile_id" in fields
                    else old_default_id
                )
                if new_kb_ids is None or new_qp_ids is None:
                    raise _validation_error()

                capabilities = (
                    command.capabilities
                    if "capabilities" in fields
                    else frozenset(Capability(value) for value in row.capabilities)
                )
                requests_per_minute = (
                    command.requests_per_minute
                    if "requests_per_minute" in fields
                    else row.requests_per_minute
                )
                max_concurrency = (
                    command.max_concurrency if "max_concurrency" in fields else row.max_concurrency
                )
                not_before = command.not_before if "not_before" in fields else row.not_before
                expires_at = command.expires_at if "expires_at" in fields else row.expires_at
                if capabilities is None or requests_per_minute is None or max_concurrency is None:
                    raise _validation_error()
                self._validate_capabilities(capabilities)
                self._validate_agent_limits(requests_per_minute, max_concurrency)
                self._validate_validity_window(not_before, expires_at)
                await self._validate_references(
                    repositories,
                    new_kb_ids,
                    new_qp_ids,
                    new_default_id,
                    validate_knowledge_bases="knowledge_base_ids" in fields,
                    validate_query_profiles=bool(
                        {"query_profile_ids", "default_query_profile_id"} & fields
                    ),
                )

                if "name" in fields:
                    if command.name is None:
                        raise _validation_error()
                    row.name = command.name
                if "status" in fields:
                    if command.status is None:
                        raise _validation_error()
                    row.status = command.status
                if "capabilities" in fields:
                    row.capabilities = sorted(capability.value for capability in capabilities)
                if "raw_file_read" in fields:
                    if command.raw_file_read is None:
                        raise _validation_error()
                    row.raw_file_read = command.raw_file_read
                if "requests_per_minute" in fields:
                    row.requests_per_minute = requests_per_minute
                if "max_concurrency" in fields:
                    row.max_concurrency = max_concurrency
                if "not_before" in fields:
                    row.not_before = not_before
                if "expires_at" in fields:
                    row.expires_at = expires_at
                if "knowledge_base_ids" in fields:
                    await repositories.api_keys.replace_kb_scopes(row.id, new_kb_ids)
                if {"query_profile_ids", "default_query_profile_id"} & fields:
                    await repositories.api_keys.replace_query_profile_scopes(
                        row.id,
                        new_qp_ids,
                        new_default_id,
                    )
                row.resource_revision += 1
                row.updated_at = now

                metadata = {
                    "changed_fields": sorted(fields),
                    "knowledge_base_ids_added": _sorted_uuid_strings(new_kb_ids - old_kb_ids),
                    "knowledge_base_ids_removed": _sorted_uuid_strings(old_kb_ids - new_kb_ids),
                    "query_profile_ids_added": _sorted_uuid_strings(new_qp_ids - old_qp_ids),
                    "query_profile_ids_removed": _sorted_uuid_strings(old_qp_ids - new_qp_ids),
                }
                await repositories.audits.add(
                    AuditEvent(
                        request_id=request_id,
                        actor_api_key_id=actor_id,
                        actor_kind=actor_kind,
                        action="api_key.policy_updated",
                        target_type="api_key",
                        target_id=row.id,
                        metadata_=metadata,
                    )
                )
                return await self._safe_api_key(
                    row,
                    new_kb_ids,
                    new_qp_ids,
                    new_default_id,
                )
        except BusinessError:
            raise
        except Exception as error:
            _raise_internal(error)

    async def revoke_admin_key(
        self,
        key_id: UUID,
        *,
        request_id: str,
    ) -> SafeApiKey:
        _require_uuid(key_id)
        _require_request_id(request_id, self._settings.max_request_id_length)
        try:
            async with self._session.begin():
                repositories = self._repository_factory(self._session)
                row = await repositories.api_keys.get_by_id(
                    key_id,
                    KeyKind.ADMIN,
                    for_update=True,
                )
                if row is None:
                    raise _not_found_error()
                if row.status == "revoked":
                    return await self._safe_api_key(row)
                now = self._now()
                row.status = "revoked"
                row.revoked_at = now
                row.revoked_by_api_key_id = None
                row.resource_revision += 1
                row.updated_at = now
                await repositories.audits.add(
                    AuditEvent(
                        request_id=request_id,
                        actor_api_key_id=None,
                        actor_kind="local_cli",
                        action="api_key.revoked",
                        target_type="api_key",
                        target_id=row.id,
                        metadata_={},
                    )
                )
                return await self._safe_api_key(row)
        except BusinessError:
            raise
        except Exception as error:
            _raise_internal(error)

    async def revoke_agent_key(
        self,
        key_id: UUID,
        *,
        actor: AdminPrincipal | None,
        request_id: str,
        expected_etag: str | None = None,
    ) -> SafeApiKey:
        _require_uuid(key_id)
        _require_request_id(request_id, self._settings.max_request_id_length)
        try:
            async with self._session.begin():
                repositories = self._repository_factory(self._session)
                now = self._now()
                actor_id, actor_kind = await self._require_admin_actor(
                    repositories,
                    actor,
                    now,
                )
                row = await repositories.api_keys.get_by_id(
                    key_id,
                    KeyKind.AGENT,
                    for_update=True,
                )
                if row is None:
                    raise _not_found_error()
                if actor is not None or expected_etag is not None:
                    require_matching_etag(
                        expected_etag,
                        agent_key_etag(row.id, row.resource_revision),
                    )
                if row.status == "revoked":
                    return await self._safe_agent(repositories, row)
                row.status = "revoked"
                row.revoked_at = now
                row.revoked_by_api_key_id = actor_id
                row.resource_revision += 1
                row.updated_at = now
                await repositories.audits.add(
                    AuditEvent(
                        request_id=request_id,
                        actor_api_key_id=actor_id,
                        actor_kind=actor_kind,
                        action="api_key.revoked",
                        target_type="api_key",
                        target_id=row.id,
                        metadata_={},
                    )
                )
                return await self._safe_agent(repositories, row)
        except BusinessError:
            raise
        except Exception as error:
            _raise_internal(error)

    async def authenticate(self, raw_token: str, expected_kind: KeyKind) -> Principal:
        parsed: ParsedToken | None = None
        secret = "<redacted>"
        try:
            parsed = parse_token(raw_token, expected_kind)
            raw_token = "<redacted>"
            public_id = parsed.public_id
            secret = parsed.secret
            async with self._authentication_sessions() as session:
                repositories = self._repository_factory(session)
                row = await repositories.api_keys.get_by_public_id(
                    expected_kind,
                    public_id,
                )
                digest = _DUMMY_SECRET_DIGEST if row is None else row.secret_digest
                digest_matches = verify_secret(
                    secret,
                    digest,
                    self._hmac_secret(expected_kind),
                )
                if row is None or not digest_matches:
                    secret = "<redacted>"
                    parsed = _redacted_parsed_token()
                    raise _invalid_api_key_error()
                secret = "<redacted>"
                parsed = _redacted_parsed_token()
                now = self._now()
                if row.key_type != expected_kind.value or not _is_currently_active(row, now):
                    raise _invalid_api_key_error()
                if expected_kind is KeyKind.AGENT:
                    knowledge_base_ids = await repositories.scopes.get_knowledge_base_scopes(row.id)
                    (
                        query_profile_ids,
                        default_query_profile_id,
                    ) = await repositories.scopes.get_query_profile_scopes(row.id)
                    return materialize_principal(
                        row,
                        knowledge_base_ids=knowledge_base_ids,
                        query_profile_ids=query_profile_ids,
                        default_query_profile_id=default_query_profile_id,
                    )
                return materialize_principal(row)
        except BusinessError:
            raw_token = "<redacted>"
            secret = "<redacted>"
            if parsed is not None:
                parsed = _redacted_parsed_token()
            raise
        except Exception as error:
            raw_token = "<redacted>"
            secret = "<redacted>"
            if parsed is not None:
                parsed = _redacted_parsed_token()
            _raise_internal(error)
        except BaseException:
            raw_token = "<redacted>"
            secret = "<redacted>"
            if parsed is not None:
                parsed = _redacted_parsed_token()
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

    def _next_cursor(self, rows: list[ApiKey], has_more: bool) -> str | None:
        if not has_more or not rows:
            return None
        last = rows[-1]
        try:
            return encode_cursor(CursorPosition(created_at=last.created_at, id=last.id))
        except (AttributeError, TypeError, ValueError) as error:
            _raise_internal(error)
