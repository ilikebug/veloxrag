"""ORM models registered with the shared declarative metadata."""

from rag_service.db.models.auth import (
    ApiKey,
    ApiKeyKnowledgeBaseScope,
    ApiKeyQueryProfileScope,
    AuditEvent,
    IdempotencyRecord,
)
from rag_service.db.models.documents import (
    Document,
    DocumentIndexState,
    DocumentUploadIdempotency,
    DocumentVersion,
    Job,
)
from rag_service.db.models.knowledge_bases import (
    IndexGenerationCleanupClaim,
    IndexGenerationCreationRequest,
    KnowledgeBase,
    KnowledgeBaseIndexGeneration,
    KnowledgeBaseMutation,
)
from rag_service.db.models.observability import ProviderUsage, QueryLog
from rag_service.db.models.providers import (
    ModelProfile,
    ModelProfileFallback,
    ProviderConfig,
    ProviderCredential,
    QueryProfile,
    SparseProfile,
)

__all__ = [
    "ApiKey",
    "ApiKeyKnowledgeBaseScope",
    "ApiKeyQueryProfileScope",
    "AuditEvent",
    "IdempotencyRecord",
    "KnowledgeBase",
    "KnowledgeBaseIndexGeneration",
    "KnowledgeBaseMutation",
    "Document",
    "DocumentIndexState",
    "DocumentUploadIdempotency",
    "DocumentVersion",
    "Job",
    "IndexGenerationCreationRequest",
    "IndexGenerationCleanupClaim",
    "ModelProfile",
    "ModelProfileFallback",
    "ProviderConfig",
    "ProviderCredential",
    "ProviderUsage",
    "QueryLog",
    "QueryProfile",
    "SparseProfile",
]
