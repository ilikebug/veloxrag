from uuid import UUID

from rag_service.api.errors import BusinessError


def _etag(prefix: str, resource_id: UUID, revision: int) -> str:
    if type(revision) is not int or revision <= 0:
        raise ValueError("revision must be a positive integer")
    return f'"{prefix}:{resource_id}:r{revision}"'


def knowledge_base_etag(resource_id: UUID, revision: int) -> str:
    return _etag("kb", resource_id, revision)


def agent_key_etag(resource_id: UUID, revision: int) -> str:
    return _etag("agent-key", resource_id, revision)


def require_matching_etag(if_match: str | None, expected: str) -> None:
    if if_match != expected:
        raise BusinessError(412, "PRECONDITION_FAILED", "Precondition failed")
