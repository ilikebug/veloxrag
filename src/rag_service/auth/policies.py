import base64
import binascii
import re
from collections.abc import Collection
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal
from uuid import UUID

from rag_service.api.errors import BusinessError
from rag_service.db.models.auth import ApiKey


class Capability(StrEnum):
    INGEST = "ingest"
    RETRIEVE = "retrieve"
    ANSWER = "answer"
    MANAGE = "manage"


@dataclass(frozen=True, slots=True)
class AdminPrincipal:
    key_id: UUID
    public_id: str
    key_type: Literal["admin"] = "admin"


@dataclass(frozen=True, slots=True)
class AgentPrincipal:
    key_id: UUID
    public_id: str
    capabilities: frozenset[Capability]
    knowledge_base_ids: frozenset[UUID]
    query_profile_ids: frozenset[UUID]
    default_query_profile_id: UUID | None
    raw_file_read: bool
    requests_per_minute: int
    max_concurrency: int
    key_type: Literal["agent"] = "agent"


type Principal = AdminPrincipal | AgentPrincipal

_DOCUMENT_READ_CAPABILITIES = frozenset({Capability.MANAGE, Capability.INGEST, Capability.RETRIEVE})
_MIN_PUBLIC_ID_LENGTH = 16
_MAX_PUBLIC_ID_LENGTH = 64
_MAX_REQUESTS_PER_MINUTE = 10_000
_MAX_CONCURRENCY = 1_000
_BASE64URL_PATTERN = re.compile(r"[A-Za-z0-9_-]+")


def _insufficient_capability_error() -> BusinessError:
    return BusinessError(403, "INSUFFICIENT_CAPABILITY", "Insufficient capability")


def _resource_not_found_error() -> BusinessError:
    return BusinessError(404, "RESOURCE_NOT_FOUND", "Resource not found")


def _internal_error() -> BusinessError:
    return BusinessError(500, "INTERNAL_ERROR", "Internal server error")


def _validated_scope(scope_ids: Collection[UUID]) -> frozenset[UUID]:
    scope = frozenset(scope_ids)
    if any(not isinstance(scope_id, UUID) for scope_id in scope):
        raise ValueError
    return scope


def _is_canonical_public_id(value: object) -> bool:
    if (
        not isinstance(value, str)
        or not _MIN_PUBLIC_ID_LENGTH <= len(value) <= _MAX_PUBLIC_ID_LENGTH
        or _BASE64URL_PATTERN.fullmatch(value) is None
    ):
        return False
    try:
        decoded = base64.b64decode(
            value + "=" * (-len(value) % 4),
            altchars=b"-_",
            validate=True,
        )
    except (ValueError, binascii.Error):
        return False
    canonical = base64.urlsafe_b64encode(decoded).decode("ascii").rstrip("=")
    return canonical == value


def _materialize_principal(
    row: ApiKey,
    knowledge_base_ids: Collection[UUID],
    query_profile_ids: Collection[UUID],
    default_query_profile_id: UUID | None,
) -> Principal:
    key_id = row.id
    public_id = row.public_id
    key_type = row.key_type
    if (
        not isinstance(key_id, UUID)
        or not _is_canonical_public_id(public_id)
        or type(key_type) is not str
    ):
        raise ValueError

    kb_scope = _validated_scope(knowledge_base_ids)
    query_profile_scope = _validated_scope(query_profile_ids)
    if default_query_profile_id is not None and not isinstance(default_query_profile_id, UUID):
        raise ValueError

    if key_type == "admin":
        admin_capabilities = row.capabilities
        admin_raw_file_read = row.raw_file_read
        if (
            type(admin_capabilities) is not list
            or len(admin_capabilities) != 0
            or type(admin_raw_file_read) is not bool
            or admin_raw_file_read
            or row.requests_per_minute is not None
            or row.max_concurrency is not None
            or kb_scope
            or query_profile_scope
            or default_query_profile_id is not None
        ):
            raise ValueError
        return AdminPrincipal(key_id=key_id, public_id=public_id)
    if key_type != "agent":
        raise ValueError

    raw_capabilities = row.capabilities
    if type(raw_capabilities) is not list or len(raw_capabilities) > len(Capability):
        raise ValueError
    if any(type(value) is not str for value in raw_capabilities):
        raise ValueError
    if len(set(raw_capabilities)) != len(raw_capabilities):
        raise ValueError
    capabilities = frozenset(Capability(value) for value in raw_capabilities)
    raw_file_read = row.raw_file_read
    requests_per_minute = row.requests_per_minute
    max_concurrency = row.max_concurrency
    if (
        type(raw_file_read) is not bool
        or type(requests_per_minute) is not int
        or not 1 <= requests_per_minute <= _MAX_REQUESTS_PER_MINUTE
        or type(max_concurrency) is not int
        or not 1 <= max_concurrency <= _MAX_CONCURRENCY
    ):
        raise ValueError
    if default_query_profile_id is not None and default_query_profile_id not in query_profile_scope:
        raise ValueError

    return AgentPrincipal(
        key_id=key_id,
        public_id=public_id,
        capabilities=capabilities,
        knowledge_base_ids=kb_scope,
        query_profile_ids=query_profile_scope,
        default_query_profile_id=default_query_profile_id,
        raw_file_read=raw_file_read,
        requests_per_minute=requests_per_minute,
        max_concurrency=max_concurrency,
    )


def materialize_principal(
    row: ApiKey,
    *,
    knowledge_base_ids: Collection[UUID] = (),
    query_profile_ids: Collection[UUID] = (),
    default_query_profile_id: UUID | None = None,
) -> Principal:
    try:
        return _materialize_principal(
            row,
            knowledge_base_ids,
            query_profile_ids,
            default_query_profile_id,
        )
    except Exception:
        row = ApiKey()
        knowledge_base_ids = ()
        query_profile_ids = ()
        default_query_profile_id = None
    raise _internal_error() from None


def require_admin(principal: Principal) -> AdminPrincipal:
    if (
        type(principal) is not AdminPrincipal
        or type(principal.key_type) is not str
        or principal.key_type != "admin"
    ):
        raise _insufficient_capability_error()
    return principal


def _require_agent(principal: Principal) -> AgentPrincipal:
    if (
        type(principal) is not AgentPrincipal
        or type(principal.key_type) is not str
        or principal.key_type != "agent"
    ):
        raise _insufficient_capability_error()
    return principal


def _has_capability(agent: AgentPrincipal, capability: Capability) -> bool:
    return any(granted is capability for granted in agent.capabilities)


def require_capability(principal: Principal, capability: Capability) -> AgentPrincipal:
    if type(capability) is not Capability:
        capability = Capability.ANSWER
        raise _insufficient_capability_error()
    agent = _require_agent(principal)
    if not _has_capability(agent, capability):
        raise _insufficient_capability_error()
    return agent


def require_knowledge_base_access(
    principal: Principal,
    knowledge_base_id: UUID,
    *,
    resource_exists: bool,
) -> AgentPrincipal:
    agent = _require_agent(principal)
    if resource_exists is not True:
        resource_exists = False
        raise _resource_not_found_error()
    if knowledge_base_id not in agent.knowledge_base_ids:
        raise _resource_not_found_error()
    return agent


def require_document_read(
    principal: Principal,
    parent_knowledge_base_id: UUID,
    *,
    parent_knowledge_base_exists: bool,
) -> AgentPrincipal:
    agent = _require_agent(principal)
    parent_knowledge_base_exists = parent_knowledge_base_exists is True
    require_knowledge_base_access(
        agent,
        parent_knowledge_base_id,
        resource_exists=parent_knowledge_base_exists,
    )
    if not any(_has_capability(agent, capability) for capability in _DOCUMENT_READ_CAPABILITIES):
        raise _insufficient_capability_error()
    return agent


def require_raw_file_read(
    principal: Principal,
    parent_knowledge_base_id: UUID,
    *,
    parent_knowledge_base_exists: bool,
) -> AgentPrincipal:
    """Gate on reading a document's own text, over and above reading its metadata.

    raw_file_read is a per-key flag rather than a capability because it cuts
    across them: a key may legitimately search a knowledge base while not being
    trusted with the source text behind a hit. So both checks apply, and the
    flag is checked second — a key outside the scope must not be able to tell
    the two refusals apart.
    """

    agent = require_document_read(
        principal,
        parent_knowledge_base_id,
        parent_knowledge_base_exists=parent_knowledge_base_exists,
    )
    if agent.raw_file_read is not True:
        raise _insufficient_capability_error()
    return agent
