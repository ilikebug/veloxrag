"""API-key token and authorization domain primitives."""

from rag_service.auth.codec import (
    GeneratedToken,
    KeyKind,
    ParsedToken,
    digest_secret,
    generate_token,
    parse_token,
    verify_secret,
)
from rag_service.auth.policies import (
    AdminPrincipal,
    AgentPrincipal,
    Capability,
    Principal,
    materialize_principal,
    require_admin,
    require_capability,
    require_document_read,
    require_knowledge_base_access,
)

__all__ = [
    "AdminPrincipal",
    "AgentPrincipal",
    "Capability",
    "GeneratedToken",
    "KeyKind",
    "ParsedToken",
    "Principal",
    "digest_secret",
    "generate_token",
    "materialize_principal",
    "parse_token",
    "require_admin",
    "require_capability",
    "require_document_read",
    "require_knowledge_base_access",
    "verify_secret",
]
