from __future__ import annotations

import base64
import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import cast
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

from rag_service.api.errors import BusinessError
from rag_service.auth.policies import AgentPrincipal, Capability
from rag_service.indexing.identities import canonical_sha256, collection_name, point_id
from rag_service.indexing.qdrant import (
    AsyncQdrantCollectionClient,
    CollectionSpec,
    FakeQdrantClient,
    QdrantFilterCondition,
    QdrantPoint,
    QdrantSearchFilter,
    QdrantSearchPoint,
    QdrantTransientError,
)
from rag_service.observability.metrics import OperationalMetrics
from rag_service.observability.repositories import ProviderUsageContext, QueryLogContext
from rag_service.providers.embeddings import (
    EmbeddingAttempt,
    EmbeddingAttemptObserver,
    EmbeddingConfigSnapshot,
    EmbeddingGatewayError,
    EmbeddingOperationalConfig,
    EmbeddingResult,
)
from rag_service.providers.rerank import RerankedDocument, RerankGatewayError
from rag_service.retrieval import services as retrieval_services
from rag_service.retrieval.repositories import (
    ActiveSearchTarget,
    EmbeddingRuntimeRecord,
    RerankRuntimeRecord,
    VisibleDocument,
)
from rag_service.retrieval.schemas import SearchFilters, SearchRequest, SearchResponse
from rag_service.retrieval.services import (
    SearchService,
    candidate_limit,
    compile_search_filter,
)


def _field(
    *,
    name: str,
    field_type: str,
    operators: Sequence[str],
    identifier: UUID,
) -> dict[str, object]:
    return {
        "name": name,
        "source_path": f"attributes.{name}",
        "type": field_type,
        "operators": list(operators),
        "field_id": "fld_" + base64.urlsafe_b64encode(identifier.bytes).rstrip(b"=").decode(),
        "payload_path": f"metadata.f_{identifier.hex}",
    }


_CATEGORY_ID = UUID("11111111-1111-4111-8111-111111111111")
_PRIORITY_ID = UUID("22222222-2222-4222-8222-222222222222")
_PUBLISHED_ID = UUID("33333333-3333-4333-8333-333333333333")
_FILTER_SNAPSHOT: dict[str, object] = {
    "fields": [
        _field(
            name="category",
            field_type="keyword",
            operators=("eq", "in"),
            identifier=_CATEGORY_ID,
        ),
        _field(
            name="priority",
            field_type="integer",
            operators=("eq", "gte", "in", "lte"),
            identifier=_PRIORITY_ID,
        ),
        _field(
            name="published_at",
            field_type="datetime",
            operators=("eq", "gte", "lte"),
            identifier=_PUBLISHED_ID,
        ),
    ]
}


def _semantic_snapshot(provider_id: UUID, credential_id: UUID) -> tuple[dict[str, object], str]:
    semantic: dict[str, object] = {
        "adapter_schema_version": "openai-embeddings-v1",
        "provider_type": "openai_compatible",
        "base_url": "https://provider.example/v1",
        "default_headers": {"X-Title": "safe"},
        "routing_options": {},
        "model_name": "immutable-embedding-model",
        "dimension": 3,
        "distance": "cosine",
        "max_input_tokens": 8192,
        "vector_config": {},
    }
    return (
        {
            **semantic,
            "provider_config_id": str(provider_id),
            "credential_id": str(credential_id),
        },
        canonical_sha256(semantic),
    )


def _actor(
    knowledge_base_id: UUID,
    *,
    capabilities: frozenset[Capability] = frozenset({Capability.RETRIEVE}),
    scoped: bool = True,
) -> AgentPrincipal:
    return AgentPrincipal(
        key_id=uuid4(),
        public_id="YWdlbnQtcmV0cmlldmFsLXVuaXQ",
        capabilities=capabilities,
        knowledge_base_ids=frozenset({knowledge_base_id}) if scoped else frozenset(),
        query_profile_ids=frozenset(),
        default_query_profile_id=None,
        raw_file_read=False,
        requests_per_minute=60,
        max_concurrency=4,
    )


def _target(
    knowledge_base_id: UUID,
    *,
    profile_id: UUID | None = None,
    provider_id: UUID | None = None,
    credential_id: UUID | None = None,
    rerank_profile_id: UUID | None = None,
) -> ActiveSearchTarget:
    profile_id = profile_id or uuid4()
    provider_id = provider_id or uuid4()
    credential_id = credential_id or uuid4()
    generation_id = uuid4()
    snapshot, semantic_hash = _semantic_snapshot(provider_id, credential_id)
    return ActiveSearchTarget(
        knowledge_base_id=knowledge_base_id,
        knowledge_base_status="active",
        generation_id=generation_id,
        generation_status="active",
        embedding_profile_id=profile_id,
        qdrant_collection_name=collection_name(knowledge_base_id, generation_id),
        embedding_config_snapshot=snapshot,
        embedding_config_hash=semantic_hash,
        index_profile_hash=semantic_hash,
        filter_schema_snapshot=_FILTER_SNAPSHOT,
        rerank_profile_id=rerank_profile_id,
    )


def _runtime(target: ActiveSearchTarget) -> EmbeddingRuntimeRecord:
    provider_id = UUID(cast(str, target.embedding_config_snapshot["provider_config_id"]))
    credential_id = UUID(cast(str, target.embedding_config_snapshot["credential_id"]))
    return EmbeddingRuntimeRecord(
        profile_id=target.embedding_profile_id,
        profile_capability="embedding",
        profile_enabled=True,
        timeout_seconds=Decimal("12.000"),
        batch_size=8,
        provider_id=provider_id,
        provider_enabled=True,
        max_concurrency=3,
        requests_per_minute=90,
        credential_id=credential_id,
        credential_exists=True,
    )


def _payload(
    *,
    knowledge_base_id: UUID,
    document_id: UUID,
    version_id: UUID,
    chunk_index: int,
    text: str,
    title_path: Sequence[str] = ("Guide", "Authentication"),
    metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    return {
        "knowledge_base_id": str(knowledge_base_id),
        "document_id": str(document_id),
        "version_id": str(version_id),
        "chunk_index": chunk_index,
        "chunk_hash": "a" * 64,
        "text": text,
        "title_path": list(title_path),
        "start_offset": 10,
        "end_offset": 10 + len(text),
        "metadata": dict(metadata or {}),
    }


def test_search_request_trims_query_defaults_top_k_and_locks_response_shape() -> None:
    command = SearchRequest.model_validate({"query": "  如何配置认证？\n"})

    assert command.query == "如何配置认证？"
    assert command.top_k == 10
    assert command.filters is None

    response_schema = SearchResponse.model_json_schema()
    assert set(response_schema["properties"]) == {"results", "index"}
    result_ref = response_schema["properties"]["results"]["items"]["$ref"].split("/")[-1]
    result_properties = response_schema["$defs"][result_ref]["properties"]
    assert set(result_properties) == {
        "text",
        "score",
        "document_id",
        "version_id",
        "chunk_index",
        "title",
        "title_path",
        "source",
        "metadata",
    }
    index_ref = response_schema["properties"]["index"]["$ref"].split("/")[-1]
    assert set(response_schema["$defs"][index_ref]["properties"]) == {
        "generation_id",
        "embedding_profile_id",
    }


@pytest.mark.parametrize(
    "document",
    (
        {"query": ""},
        {"query": " \n\t "},
        {"query": "x" * 8001},
        {"query": "valid", "top_k": 0},
        {"query": "valid", "top_k": 51},
        {"query": "valid", "top_k": True},
        {"query": "valid", "unknown": "must-not-appear"},
        {"query": "valid", "filters": {"document_ids": [str(uuid4())] * 201}},
        {
            "query": "valid",
            "filters": {"metadata": {f"field{index}": index for index in range(65)}},
        },
        {
            "query": "valid",
            "filters": {"metadata": {"category": {"in": ["x"] * 101}}},
        },
        {
            "query": "valid",
            "filters": {
                "metadata": {"category": {"in": ["x" * 400] * 100}},
            },
        },
        {
            "query": "valid",
            "filters": {"metadata": {"category": {"eq": "x", "in": ["x"]}}},
        },
        {"query": "valid", "filters": {"metadata": {"category": None}}},
    ),
)
def test_search_request_rejects_malformed_or_oversized_input_without_echo(
    document: dict[str, object],
) -> None:
    with pytest.raises(ValidationError) as invalid:
        SearchRequest.model_validate(document)
    assert "must-not-appear" not in str(invalid.value)


def test_compile_search_filter_uses_only_snapshot_fields_paths_types_and_operators() -> None:
    document_id = uuid4()
    filters = SearchRequest.model_validate(
        {
            "query": "q",
            "filters": {
                "document_ids": [document_id],
                "metadata": {
                    "category": "guide",
                    "priority": {"gte": 2},
                    "published_at": {"lte": "2026-07-30T12:30:00Z"},
                },
            },
        }
    ).filters

    compiled = compile_search_filter(filters, _FILTER_SNAPSHOT)

    assert compiled == QdrantSearchFilter(
        document_ids=(document_id,),
        conditions=(
            QdrantFilterCondition(
                path=f"metadata.f_{_CATEGORY_ID.hex}",
                field_type="keyword",
                operator="eq",
                value="guide",
            ),
            QdrantFilterCondition(
                path=f"metadata.f_{_PRIORITY_ID.hex}",
                field_type="integer",
                operator="gte",
                value=2,
            ),
            QdrantFilterCondition(
                path=f"metadata.f_{_PUBLISHED_ID.hex}",
                field_type="datetime",
                operator="lte",
                value="2026-07-30T12:30:00Z",
            ),
        ),
    )


@pytest.mark.parametrize(
    "metadata",
    (
        {"unknown": "x"},
        {"category": {"gte": "x"}},
        {"category": {"in": [1]}},
        {"priority": 1.5},
        {"priority": {"in": [1, True]}},
        {"published_at": "not-rfc3339"},
    ),
)
def test_compile_search_filter_rejects_unknown_or_type_invalid_metadata_safely(
    metadata: dict[str, object],
) -> None:
    filters = SearchRequest.model_validate(
        {"query": "q", "filters": {"metadata": metadata}}
    ).filters

    with pytest.raises(BusinessError) as rejected:
        compile_search_filter(filters, _FILTER_SNAPSHOT)

    assert (rejected.value.status_code, rejected.value.code) == (422, "INVALID_SEARCH_FILTER")
    assert "not-rfc3339" not in rejected.value.message


def test_compile_search_filter_fails_closed_for_malformed_generation_snapshot() -> None:
    filters = SearchRequest(
        query="q",
        filters=SearchFilters(metadata={"category": "guide"}),
    ).filters
    malformed: dict[str, object] = {
        "fields": [
            {
                **cast(list[dict[str, object]], _FILTER_SNAPSHOT["fields"])[0],
                "payload_path": "metadata.secret",
            }
        ]
    }

    with pytest.raises(BusinessError) as rejected:
        compile_search_filter(filters, malformed)

    assert (rejected.value.status_code, rejected.value.code) == (
        503,
        "RETRIEVAL_CONFIGURATION_UNAVAILABLE",
    )
    assert "secret" not in rejected.value.message


def test_integer_range_filter_rejects_values_not_exactly_representable_by_qdrant() -> None:
    filters = SearchRequest(
        query="q",
        filters=SearchFilters(metadata={"priority": {"gte": 2**53 + 1}}),
    ).filters

    with pytest.raises(BusinessError) as rejected:
        compile_search_filter(filters, _FILTER_SNAPSHOT)

    assert (rejected.value.status_code, rejected.value.code) == (422, "INVALID_SEARCH_FILTER")
    with pytest.raises(ValueError, match="Qdrant filter condition is invalid"):
        QdrantFilterCondition(
            path=f"metadata.f_{_PRIORITY_ID.hex}",
            field_type="integer",
            operator="gte",
            value=2**53 + 1,
        )


def test_integer_range_filter_accepts_exactly_representable_large_value() -> None:
    filters = SearchRequest(
        query="q",
        filters=SearchFilters(metadata={"priority": {"lte": 2**54}}),
    ).filters

    compiled = compile_search_filter(filters, _FILTER_SNAPSHOT)

    assert compiled.conditions[0].value == 2**54


@pytest.mark.parametrize(
    ("top_k", "expected"),
    ((1, 20), (5, 20), (10, 40), (50, 200)),
)
def test_candidate_limit_is_bounded(top_k: int, expected: int) -> None:
    assert candidate_limit(top_k) == expected


class FakeSearchRepository:
    def __init__(
        self,
        target: ActiveSearchTarget | None,
        runtime: EmbeddingRuntimeRecord | None,
        rerank_runtime: RerankRuntimeRecord | None = None,
    ) -> None:
        self.target = target
        self.runtime = runtime
        self.rerank_runtime = rerank_runtime
        self.visible: dict[tuple[UUID, UUID], VisibleDocument] = {}
        self.filter_visible: dict[UUID, VisibleDocument] = {}
        self.target_calls: list[UUID] = []
        self.runtime_calls: list[tuple[UUID, UUID, UUID]] = []
        self.rerank_runtime_calls: list[UUID] = []
        self.document_filter_calls: list[tuple[UUID, UUID, tuple[UUID, ...]]] = []
        self.visibility_calls: list[tuple[UUID, UUID, tuple[tuple[UUID, UUID], ...]]] = []

    async def get_active_target(self, knowledge_base_id: UUID) -> ActiveSearchTarget | None:
        self.target_calls.append(knowledge_base_id)
        return self.target

    async def get_embedding_runtime(
        self,
        profile_id: UUID,
        provider_id: UUID,
        credential_id: UUID,
    ) -> EmbeddingRuntimeRecord | None:
        self.runtime_calls.append((profile_id, provider_id, credential_id))
        return self.runtime

    async def get_rerank_runtime(self, profile_id: UUID) -> RerankRuntimeRecord | None:
        self.rerank_runtime_calls.append(profile_id)
        return self.rerank_runtime

    async def load_visible_document_filters(
        self,
        *,
        knowledge_base_id: UUID,
        generation_id: UUID,
        document_ids: Sequence[UUID],
    ) -> Mapping[UUID, VisibleDocument]:
        canonical = tuple(document_ids)
        self.document_filter_calls.append((knowledge_base_id, generation_id, canonical))
        return {
            document_id: visible
            for document_id, visible in self.filter_visible.items()
            if document_id in canonical
        }

    async def load_visible_documents(
        self,
        *,
        knowledge_base_id: UUID,
        generation_id: UUID,
        identities: Sequence[tuple[UUID, UUID]],
    ) -> Mapping[tuple[UUID, UUID], VisibleDocument]:
        canonical = tuple(identities)
        self.visibility_calls.append((knowledge_base_id, generation_id, canonical))
        return self.visible


class CapturingEmbeddingGateway:
    def __init__(self, result: EmbeddingResult | BaseException | None = None) -> None:
        self.result = result or EmbeddingResult(((1.0, 0.0, 0.0),), {"prompt_tokens": 3})
        self.calls: list[
            tuple[
                EmbeddingConfigSnapshot,
                EmbeddingOperationalConfig,
                tuple[str, ...],
            ]
        ] = []

    async def embed(
        self,
        *,
        snapshot: EmbeddingConfigSnapshot,
        operational: EmbeddingOperationalConfig,
        inputs: Sequence[str],
        attempt_observer: EmbeddingAttemptObserver | None = None,
    ) -> EmbeddingResult:
        self.calls.append((snapshot, operational, tuple(inputs)))
        if attempt_observer is not None:
            observed = attempt_observer(
                EmbeddingAttempt(
                    provider_identifier="openai_compatible",
                    model_identifier=snapshot.model_name,
                    route_identifier="direct",
                    provider_request_id="safe-request-id",
                    input_tokens=3,
                    output_tokens=0,
                    cost_micros=7,
                    currency="USD",
                    latency_ms=4,
                    status="succeeded",
                    error_code=None,
                    degraded=False,
                )
            )
            if isinstance(observed, Awaitable):
                await observed
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


class CapturingSearchIndex:
    def __init__(self, points: Sequence[QdrantSearchPoint] = ()) -> None:
        self.points = tuple(points)
        self.calls: list[tuple[str, tuple[float, ...], int, QdrantSearchFilter]] = []

    async def search_points(
        self,
        collection: str,
        vector: Sequence[float],
        *,
        limit: int,
        query_filter: QdrantSearchFilter,
    ) -> tuple[QdrantSearchPoint, ...]:
        self.calls.append((collection, tuple(vector), limit, query_filter))
        return self.points


class CapturingProviderUsageSink:
    def __init__(self) -> None:
        self.records: list[tuple[ProviderUsageContext, EmbeddingAttempt]] = []

    async def record(self, context: ProviderUsageContext, attempt: EmbeddingAttempt) -> None:
        self.records.append((context, attempt))


class CapturingQueryLogSink:
    def __init__(self, failure: BaseException | None = None) -> None:
        self.failure = failure
        self.records: list[tuple[QueryLogContext, str, int, bool]] = []

    async def record(
        self,
        context: QueryLogContext,
        *,
        status: str,
        latency_ms: int,
        degraded: bool,
    ) -> None:
        self.records.append((context, status, latency_ms, degraded))
        if self.failure is not None:
            raise self.failure


def _service(
    repository: FakeSearchRepository,
    *,
    gateway: CapturingEmbeddingGateway | None = None,
    index: CapturingSearchIndex | None = None,
    usage_sink: CapturingProviderUsageSink | None = None,
    query_sink: CapturingQueryLogSink | None = None,
    clock: Callable[[], float] | None = None,
    metrics: OperationalMetrics | object | None = None,
    rerank_gateway: object | None = None,
) -> tuple[
    SearchService,
    CapturingEmbeddingGateway,
    CapturingSearchIndex,
    CapturingProviderUsageSink,
    CapturingQueryLogSink,
]:
    gateway = gateway or CapturingEmbeddingGateway()
    index = index or CapturingSearchIndex()
    usage_sink = usage_sink or CapturingProviderUsageSink()
    query_sink = query_sink or CapturingQueryLogSink()
    monotonic_values = iter((1.0, 1.004, 1.008, 1.012))
    kwargs: dict[str, object] = {}
    if metrics is not None:
        kwargs["metrics"] = metrics
    if rerank_gateway is not None:
        kwargs["rerank_gateway"] = rerank_gateway
    return (
        SearchService(
            repository=repository,
            embedding_gateway=gateway,
            search_index=index,
            provider_usage_sink=usage_sink,
            query_log_sink=query_sink,
            monotonic_clock=clock or monotonic_values.__next__,
            **kwargs,  # type: ignore[arg-type]
        ),
        gateway,
        index,
        usage_sink,
        query_sink,
    )


class _CollectingRetrievalLogHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.mark.asyncio
async def test_search_requires_retrieve_capability_and_hides_out_of_scope_kb() -> None:
    kb_id = uuid4()
    target = _target(kb_id)
    repository = FakeSearchRepository(target, _runtime(target))
    service, *_ = _service(repository)

    with pytest.raises(BusinessError) as missing_capability:
        await service.search(
            knowledge_base_id=kb_id,
            actor=_actor(kb_id, capabilities=frozenset({Capability.INGEST})),
            request_id="req-search-capability",
            command=SearchRequest(query="safe query"),
        )
    with pytest.raises(BusinessError) as missing_scope:
        await service.search(
            knowledge_base_id=kb_id,
            actor=_actor(kb_id, scoped=False),
            request_id="req-search-scope",
            command=SearchRequest(query="safe query"),
        )

    assert (missing_capability.value.status_code, missing_capability.value.code) == (
        403,
        "INSUFFICIENT_CAPABILITY",
    )
    assert (missing_scope.value.status_code, missing_scope.value.code) == (
        404,
        "RESOURCE_NOT_FOUND",
    )
    assert repository.target_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutate_target", "mutate_runtime", "expected_code"),
    (
        (lambda _target: None, lambda runtime: runtime, "ACTIVE_GENERATION_UNAVAILABLE"),
        (
            lambda target: replace(target, knowledge_base_status="disabled"),
            lambda runtime: runtime,
            "KNOWLEDGE_BASE_UNAVAILABLE",
        ),
        (
            lambda target: replace(target, generation_status="failed"),
            lambda runtime: runtime,
            "ACTIVE_GENERATION_UNAVAILABLE",
        ),
        (
            lambda target: target,
            lambda _runtime: None,
            "MODEL_PROFILE_UNAVAILABLE",
        ),
        (
            lambda target: target,
            lambda runtime: replace(runtime, profile_enabled=False),
            "MODEL_PROFILE_UNAVAILABLE",
        ),
        (
            lambda target: target,
            lambda runtime: replace(runtime, provider_enabled=False),
            "PROVIDER_CONFIG_UNAVAILABLE",
        ),
        (
            lambda target: target,
            lambda runtime: replace(runtime, credential_exists=False),
            "PROVIDER_CREDENTIAL_UNAVAILABLE",
        ),
    ),
)
async def test_search_configuration_failures_are_safe(
    mutate_target: Callable[[ActiveSearchTarget], ActiveSearchTarget | None],
    mutate_runtime: Callable[[EmbeddingRuntimeRecord], EmbeddingRuntimeRecord | None],
    expected_code: str,
) -> None:
    kb_id = uuid4()
    base_target = _target(kb_id)
    target = mutate_target(base_target)
    runtime = mutate_runtime(_runtime(base_target))
    repository = FakeSearchRepository(target, runtime)
    service, *_ = _service(repository)

    with pytest.raises(BusinessError) as unavailable:
        await service.search(
            knowledge_base_id=kb_id,
            actor=_actor(kb_id),
            request_id="req-search-unavailable",
            command=SearchRequest(query="never persist this query"),
        )

    assert unavailable.value.code == expected_code
    assert "never persist this query" not in unavailable.value.message


@pytest.mark.asyncio
async def test_search_rejects_noncanonical_collection_before_provider_or_qdrant() -> None:
    kb_id = uuid4()
    target = _target(kb_id)
    drifted = replace(
        target,
        qdrant_collection_name=collection_name(kb_id, uuid4()),
    )
    repository = FakeSearchRepository(drifted, _runtime(target))
    service, gateway, index, *_ = _service(repository)

    with pytest.raises(BusinessError) as unavailable:
        await service.search(
            knowledge_base_id=kb_id,
            actor=_actor(kb_id),
            request_id="req-collection-drift",
            command=SearchRequest(query="never send upstream"),
        )

    assert (unavailable.value.status_code, unavailable.value.code) == (
        503,
        "RETRIEVAL_CONFIGURATION_UNAVAILABLE",
    )
    assert repository.runtime_calls == []
    assert gateway.calls == []
    assert index.calls == []


@pytest.mark.asyncio
async def test_search_uses_immutable_snapshot_and_preserves_validated_qdrant_order() -> None:
    kb_id = uuid4()
    target = _target(kb_id)
    runtime = _runtime(target)
    first_document, second_document, hidden_document = uuid4(), uuid4(), uuid4()
    first_version, second_version, hidden_version = uuid4(), uuid4(), uuid4()
    first_payload = _payload(
        knowledge_base_id=kb_id,
        document_id=first_document,
        version_id=first_version,
        chunk_index=1,
        text="first result",
        metadata={f"f_{_CATEGORY_ID.hex}": "guide", "f_ffffffffffffffffffffffffffffffff": "drop"},
    )
    second_payload = _payload(
        knowledge_base_id=kb_id,
        document_id=second_document,
        version_id=second_version,
        chunk_index=2,
        text="second result",
        title_path=(),
        metadata={f"f_{_PRIORITY_ID.hex}": 3},
    )
    hidden_payload = _payload(
        knowledge_base_id=kb_id,
        document_id=hidden_document,
        version_id=hidden_version,
        chunk_index=3,
        text="must not escape",
    )
    index = CapturingSearchIndex(
        (
            QdrantSearchPoint(point_id(first_version, 1, "a" * 64), 0.91, first_payload),
            QdrantSearchPoint(point_id(hidden_version, 3, "a" * 64), 0.90, hidden_payload),
            QdrantSearchPoint(point_id(second_version, 2, "a" * 64), 0.42, second_payload),
        )
    )
    repository = FakeSearchRepository(target, runtime)
    repository.visible = {
        (first_document, first_version): VisibleDocument(
            document_id=first_document,
            version_id=first_version,
            source_filename="guide.md",
        ),
        (second_document, second_version): VisibleDocument(
            document_id=second_document,
            version_id=second_version,
            source_filename="notes.txt",
        ),
    }
    requested_document = uuid4()
    repository.filter_visible = {
        requested_document: VisibleDocument(
            document_id=requested_document,
            version_id=uuid4(),
            source_filename="requested.md",
        )
    }
    service, gateway, _, usage_sink, query_sink = _service(repository, index=index)

    response = await service.search(
        knowledge_base_id=kb_id,
        actor=_actor(kb_id),
        request_id="req-search-success",
        command=SearchRequest.model_validate(
            {
                "query": "  authentication  ",
                "top_k": 3,
                "filters": {
                    "document_ids": [requested_document],
                    "metadata": {"category": "guide"},
                },
            }
        ),
    )

    assert [result.document_id for result in response.results] == [first_document, second_document]
    assert [result.score for result in response.results] == [0.91, 0.42]
    assert response.results[0].title == "Authentication"
    assert response.results[0].title_path == ("Guide", "Authentication")
    assert response.results[0].source.filename == "guide.md"
    assert response.results[0].metadata == {"category": "guide"}
    assert response.results[1].title is None
    assert response.results[1].metadata == {"priority": 3}
    assert response.index.generation_id == target.generation_id
    assert response.index.embedding_profile_id == target.embedding_profile_id

    assert len(gateway.calls) == 1
    gateway_snapshot, operational, inputs = gateway.calls[0]
    assert inputs == ("authentication",)
    assert gateway_snapshot.model_name == "immutable-embedding-model"
    assert gateway_snapshot.dimension == 3
    assert operational.provider_config_id == runtime.provider_id
    assert index.calls == [
        (
            target.qdrant_collection_name,
            (1.0, 0.0, 0.0),
            20,
            QdrantSearchFilter(
                document_ids=(requested_document,),
                conditions=(
                    QdrantFilterCondition(
                        path=f"metadata.f_{_CATEGORY_ID.hex}",
                        field_type="keyword",
                        operator="eq",
                        value="guide",
                    ),
                ),
            ),
        )
    ]
    assert repository.visibility_calls == [
        (
            kb_id,
            target.generation_id,
            (
                (first_document, first_version),
                (hidden_document, hidden_version),
                (second_document, second_version),
            ),
        )
    ]
    assert repository.document_filter_calls == [
        (kb_id, target.generation_id, (requested_document,))
    ]
    assert len(usage_sink.records) == 1
    usage_context, usage = usage_sink.records[0]
    assert usage_context.request_id == "req-search-success"
    assert usage_context.actor_api_key_id is not None
    assert usage_context.provider_config_id == runtime.provider_id
    assert usage_context.model_profile_id == target.embedding_profile_id
    assert usage.status == "succeeded"
    assert query_sink.records == [
        (
            QueryLogContext(
                request_id="req-search-success",
                actor_api_key_id=usage_context.actor_api_key_id,
                knowledge_base_ids=(kb_id,),
            ),
            "succeeded",
            12,
            False,
        )
    ]
    serialized = response.model_dump_json()
    for forbidden in (
        "must not escape",
        "drop",
        "f_ffffffffffffffffffffffffffffffff",
        "authentication",
        "X-Title",
    ):
        assert forbidden not in serialized


@pytest.mark.asyncio
async def test_search_snapshots_old_active_generation_across_pointer_race() -> None:
    kb_id = uuid4()
    old_target = _target(kb_id)
    new_target = _target(kb_id)

    class RacingRepository(FakeSearchRepository):
        async def get_active_target(
            self,
            knowledge_base_id: UUID,
        ) -> ActiveSearchTarget | None:
            captured = await super().get_active_target(knowledge_base_id)
            self.target = new_target
            return captured

    repository = RacingRepository(old_target, _runtime(old_target))
    requested_document = uuid4()
    repository.filter_visible = {
        requested_document: VisibleDocument(
            document_id=requested_document,
            version_id=uuid4(),
            source_filename="race.md",
        )
    }
    service, _gateway, index, *_ = _service(repository)

    response = await service.search(
        knowledge_base_id=kb_id,
        actor=_actor(kb_id),
        request_id="req-generation-race",
        command=SearchRequest(
            query="race",
            filters=SearchFilters(document_ids=(requested_document,)),
        ),
    )

    assert repository.target == new_target
    assert response.index.generation_id == old_target.generation_id
    assert index.calls[0][0] == old_target.qdrant_collection_name
    assert repository.document_filter_calls == [
        (kb_id, old_target.generation_id, (requested_document,))
    ]
    assert repository.visibility_calls[0][1] == old_target.generation_id


@pytest.mark.asyncio
async def test_search_document_filter_uses_same_empty_path_for_hidden_and_missing_ids() -> None:
    kb_id = uuid4()
    target = _target(kb_id)
    invisible_document, nonexistent_document = uuid4(), uuid4()
    stale_version = uuid4()
    stale_point = QdrantSearchPoint(
        point_id(stale_version, 0, "a" * 64),
        1.0,
        _payload(
            knowledge_base_id=kb_id,
            document_id=invisible_document,
            version_id=stale_version,
            chunk_index=0,
            text="must not be queried",
        ),
    )
    repository = FakeSearchRepository(target, _runtime(target))
    index = CapturingSearchIndex((stale_point,))
    service, gateway, _, usage_sink, query_sink = _service(repository, index=index)

    responses = []
    for request_id, document_id in (
        ("req-filter-invisible", invisible_document),
        ("req-filter-nonexistent", nonexistent_document),
    ):
        responses.append(
            await service.search(
                knowledge_base_id=kb_id,
                actor=_actor(kb_id),
                request_id=request_id,
                command=SearchRequest(
                    query="same path",
                    filters=SearchFilters(document_ids=(document_id,)),
                ),
            )
        )

    assert [response.results for response in responses] == [(), ()]
    assert repository.document_filter_calls == [
        (kb_id, target.generation_id, (invisible_document,)),
        (kb_id, target.generation_id, (nonexistent_document,)),
    ]
    assert repository.visibility_calls == []
    assert gateway.calls == []
    assert index.calls == []
    assert usage_sink.records == []
    assert [record[1] for record in query_sink.records] == ["succeeded", "succeeded"]


@pytest.mark.asyncio
async def test_search_pushes_only_canonical_visible_document_ids_to_qdrant() -> None:
    kb_id = uuid4()
    target = _target(kb_id)
    visible_document, invisible_document = uuid4(), uuid4()
    repository = FakeSearchRepository(target, _runtime(target))
    repository.filter_visible = {
        visible_document: VisibleDocument(
            document_id=visible_document,
            version_id=uuid4(),
            source_filename="visible.md",
        )
    }
    service, _, index, *_ = _service(repository)

    response = await service.search(
        knowledge_base_id=kb_id,
        actor=_actor(kb_id),
        request_id="req-filter-visible-only",
        command=SearchRequest(
            query="visible only",
            filters=SearchFilters(
                document_ids=(visible_document, invisible_document),
                metadata={"category": "guide"},
            ),
        ),
    )

    assert response.results == ()
    assert repository.document_filter_calls == [
        (
            kb_id,
            target.generation_id,
            (visible_document, invisible_document),
        )
    ]
    assert index.calls[0][3] == QdrantSearchFilter(
        document_ids=(visible_document,),
        conditions=(
            QdrantFilterCondition(
                path=f"metadata.f_{_CATEGORY_ID.hex}",
                field_type="keyword",
                operator="eq",
                value="guide",
            ),
        ),
    )


@pytest.mark.asyncio
async def test_search_returns_no_fallback_when_postgres_rejects_all_candidates() -> None:
    kb_id = uuid4()
    target = _target(kb_id)
    candidates = tuple(
        QdrantSearchPoint(
            point_id(version_id := uuid4(), index, "a" * 64),
            1.0 - index / 10,
            _payload(
                knowledge_base_id=kb_id,
                document_id=uuid4(),
                version_id=version_id,
                chunk_index=index,
                text=f"hidden-{index}",
            ),
        )
        for index in range(4)
    )
    repository = FakeSearchRepository(target, _runtime(target))
    service, *_ = _service(repository, index=CapturingSearchIndex(candidates))

    response = await service.search(
        knowledge_base_id=kb_id,
        actor=_actor(kb_id),
        request_id="req-no-fallback",
        command=SearchRequest(query="q", top_k=1),
    )

    assert response.results == ()


@pytest.mark.asyncio
async def test_search_observes_only_real_qdrant_search_with_safe_counts_and_ids() -> None:
    kb_id = uuid4()
    target = _target(kb_id)
    first_document, second_document, hidden_document = uuid4(), uuid4(), uuid4()
    first_version, second_version, hidden_version = uuid4(), uuid4(), uuid4()
    sentinel = "query-retrieved-chunk-vector-provider-secret-ciphertext-nonce-authorization"
    candidates = (
        QdrantSearchPoint(
            point_id(first_version, 0, "a" * 64),
            0.9,
            _payload(
                knowledge_base_id=kb_id,
                document_id=first_document,
                version_id=first_version,
                chunk_index=0,
                text=f"first-{sentinel}",
            ),
        ),
        QdrantSearchPoint(
            point_id(hidden_version, 1, "a" * 64),
            0.8,
            _payload(
                knowledge_base_id=kb_id,
                document_id=hidden_document,
                version_id=hidden_version,
                chunk_index=1,
                text=f"hidden-{sentinel}",
            ),
        ),
        QdrantSearchPoint(
            point_id(second_version, 2, "a" * 64),
            0.7,
            _payload(
                knowledge_base_id=kb_id,
                document_id=second_document,
                version_id=second_version,
                chunk_index=2,
                text=f"second-{sentinel}",
            ),
        ),
    )
    repository = FakeSearchRepository(target, _runtime(target))
    repository.visible = {
        (first_document, first_version): VisibleDocument(
            document_id=first_document,
            version_id=first_version,
            source_filename="first.md",
        ),
        (second_document, second_version): VisibleDocument(
            document_id=second_document,
            version_id=second_version,
            source_filename="second.md",
        ),
    }
    metrics = OperationalMetrics()
    handler = _CollectingRetrievalLogHandler()
    previous_handlers = list(retrieval_services.logger.handlers)
    previous_propagate = retrieval_services.logger.propagate
    previous_level = retrieval_services.logger.level
    retrieval_services.logger.handlers = [handler]
    retrieval_services.logger.propagate = False
    retrieval_services.logger.setLevel(logging.INFO)
    service, *_ = _service(
        repository,
        index=CapturingSearchIndex(candidates),
        metrics=metrics,
    )
    try:
        response = await service.search(
            knowledge_base_id=kb_id,
            actor=_actor(kb_id),
            request_id="req-qdrant-observed",
            command=SearchRequest(query=sentinel, top_k=3),
        )
    finally:
        retrieval_services.logger.handlers = previous_handlers
        retrieval_services.logger.propagate = previous_propagate
        retrieval_services.logger.setLevel(previous_level)

    assert len(response.results) == 2
    assert (
        metrics.registry.get_sample_value("rag_qdrant_searches_total", {"outcome": "succeeded"})
        == 1
    )
    assert metrics.registry.get_sample_value("rag_qdrant_visibility_drops_total") == 1
    assert metrics.registry.get_sample_value("rag_qdrant_results_total") == 2
    assert len(handler.records) == 1
    record = handler.records[0]
    assert type(record) is logging.LogRecord
    assert record.msg == "qdrant.search.completed"
    assert record.__dict__["request_id"] == "req-qdrant-observed"
    assert record.__dict__["knowledge_base_id"] == str(kb_id)
    assert record.__dict__["generation_id"] == str(target.generation_id)
    assert record.__dict__["candidate_count"] == 3
    assert record.__dict__["result_count"] == 2
    assert record.__dict__["visibility_drop_count"] == 1
    rendered = repr(record.__dict__)
    assert sentinel not in rendered
    for forbidden_key in (
        "query",
        "text",
        "vector",
        "payload",
        "default_headers",
    ):
        assert forbidden_key not in record.__dict__
    assert "Authorization" not in rendered


@pytest.mark.asyncio
async def test_search_observes_qdrant_failure_with_zero_counts_and_no_exception_message() -> None:
    sentinel = "raw-query-body-object-key-traceback-secret"

    class FailingSearchIndex(CapturingSearchIndex):
        async def search_points(
            self,
            collection: str,
            vector: Sequence[float],
            *,
            limit: int,
            query_filter: QdrantSearchFilter,
        ) -> tuple[QdrantSearchPoint, ...]:
            self.calls.append((collection, tuple(vector), limit, query_filter))
            raise QdrantTransientError(sentinel)

    kb_id = uuid4()
    target = _target(kb_id)
    metrics = OperationalMetrics()
    handler = _CollectingRetrievalLogHandler()
    previous_handlers = list(retrieval_services.logger.handlers)
    previous_propagate = retrieval_services.logger.propagate
    previous_level = retrieval_services.logger.level
    retrieval_services.logger.handlers = [handler]
    retrieval_services.logger.propagate = False
    retrieval_services.logger.setLevel(logging.INFO)
    service, *_ = _service(
        FakeSearchRepository(target, _runtime(target)),
        index=FailingSearchIndex(),
        metrics=metrics,
    )
    try:
        with pytest.raises(BusinessError) as failure:
            await service.search(
                knowledge_base_id=kb_id,
                actor=_actor(kb_id),
                request_id="req-qdrant-failed",
                command=SearchRequest(query=sentinel),
            )
    finally:
        retrieval_services.logger.handlers = previous_handlers
        retrieval_services.logger.propagate = previous_propagate
        retrieval_services.logger.setLevel(previous_level)

    assert failure.value.code == "RETRIEVAL_UNAVAILABLE"
    assert (
        metrics.registry.get_sample_value("rag_qdrant_searches_total", {"outcome": "failed"}) == 1
    )
    assert metrics.registry.get_sample_value("rag_qdrant_visibility_drops_total") == 0
    assert metrics.registry.get_sample_value("rag_qdrant_results_total") == 0
    assert len(handler.records) == 1
    record = handler.records[0]
    assert record.__dict__["candidate_count"] == 0
    assert record.__dict__["result_count"] == 0
    assert record.__dict__["visibility_drop_count"] == 0
    assert record.__dict__["error_code"] == "QDRANT_SEARCH_FAILED"
    assert sentinel not in repr(record.__dict__)


@pytest.mark.asyncio
async def test_qdrant_observation_freezes_network_success_before_slow_visibility_failure() -> None:
    class ManualClock:
        now = 0.0

        def __call__(self) -> float:
            return self.now

    clock = ManualClock()
    kb_id = uuid4()
    target = _target(kb_id)
    document_id, version_id = uuid4(), uuid4()
    point = QdrantSearchPoint(
        point_id(version_id, 0, "a" * 64),
        0.9,
        _payload(
            knowledge_base_id=kb_id,
            document_id=document_id,
            version_id=version_id,
            chunk_index=0,
            text="safe result",
        ),
    )

    class TimedIndex(CapturingSearchIndex):
        async def search_points(
            self,
            *args: object,
            **kwargs: object,
        ) -> tuple[QdrantSearchPoint, ...]:
            del args, kwargs
            clock.now += 0.1
            return (point,)

    class SlowFailingRepository(FakeSearchRepository):
        async def load_visible_documents(
            self,
            **kwargs: object,
        ) -> Mapping[tuple[UUID, UUID], VisibleDocument]:
            del kwargs
            clock.now += 5.0
            raise RuntimeError("visibility-database-secret")

    repository = SlowFailingRepository(target, _runtime(target))
    metrics = OperationalMetrics()
    service, *_ = _service(
        repository,
        index=TimedIndex(),
        metrics=metrics,
        clock=clock,
    )

    with pytest.raises(RuntimeError, match="visibility-database-secret"):
        await service.search(
            knowledge_base_id=kb_id,
            actor=_actor(kb_id),
            request_id="req-qdrant-network-boundary",
            command=SearchRequest(query="safe query"),
        )

    assert (
        metrics.registry.get_sample_value("rag_qdrant_searches_total", {"outcome": "succeeded"})
        == 1
    )
    assert (
        metrics.registry.get_sample_value("rag_qdrant_searches_total", {"outcome": "failed"})
        is None
    )
    assert metrics.registry.get_sample_value(
        "rag_qdrant_search_duration_seconds_sum", {"outcome": "succeeded"}
    ) == pytest.approx(0.1)
    assert metrics.registry.get_sample_value("rag_qdrant_results_total") == 0
    assert metrics.registry.get_sample_value("rag_qdrant_visibility_drops_total") == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("payload_mode", ("invalid_uuid", "oversized_title_path"))
async def test_qdrant_success_is_not_reclassified_by_invalid_or_oversized_payload(
    payload_mode: str,
) -> None:
    kb_id = uuid4()
    target = _target(kb_id)
    document_id, version_id = uuid4(), uuid4()
    payload = _payload(
        knowledge_base_id=kb_id,
        document_id=document_id,
        version_id=version_id,
        chunk_index=0,
        text="safe result",
        title_path=("title",) * (65 if payload_mode == "oversized_title_path" else 1),
    )
    point: QdrantSearchPoint
    expected_error: type[BaseException]
    if payload_mode == "invalid_uuid":
        payload["document_id"] = "not-a-uuid"
        point = cast(
            QdrantSearchPoint,
            SimpleNamespace(score=0.9, payload=payload),
        )
        expected_error = ValueError
    else:
        point = QdrantSearchPoint(
            point_id(version_id, 0, "a" * 64),
            0.9,
            payload,
        )
        expected_error = ValidationError
    repository = FakeSearchRepository(target, _runtime(target))
    repository.visible = {
        (document_id, version_id): VisibleDocument(
            document_id=document_id,
            version_id=version_id,
            source_filename="safe.md",
        )
    }
    metrics = OperationalMetrics()
    service, *_ = _service(
        repository,
        index=CapturingSearchIndex((point,)),
        metrics=metrics,
    )

    with pytest.raises(expected_error):
        await service.search(
            knowledge_base_id=kb_id,
            actor=_actor(kb_id),
            request_id="req-qdrant-postprocess-failure",
            command=SearchRequest(query="safe query"),
        )

    assert (
        metrics.registry.get_sample_value("rag_qdrant_searches_total", {"outcome": "succeeded"})
        == 1
    )
    assert (
        metrics.registry.get_sample_value("rag_qdrant_searches_total", {"outcome": "failed"})
        is None
    )
    assert metrics.registry.get_sample_value("rag_qdrant_results_total") == 0
    assert metrics.registry.get_sample_value("rag_qdrant_visibility_drops_total") == 0


@pytest.mark.asyncio
async def test_search_does_not_observe_qdrant_when_document_visibility_returns_early() -> None:
    kb_id = uuid4()
    target = _target(kb_id)
    metrics = OperationalMetrics()
    service, _gateway, index, *_ = _service(
        FakeSearchRepository(target, _runtime(target)),
        metrics=metrics,
    )

    response = await service.search(
        knowledge_base_id=kb_id,
        actor=_actor(kb_id),
        request_id="req-no-qdrant",
        command=SearchRequest(
            query="private-query",
            filters=SearchFilters(document_ids=(uuid4(),)),
        ),
    )

    assert response.results == ()
    assert index.calls == []
    assert (
        metrics.registry.get_sample_value("rag_qdrant_searches_total", {"outcome": "succeeded"})
        is None
    )


@pytest.mark.asyncio
async def test_search_observability_and_timer_failures_do_not_change_the_result() -> None:
    class FailingMetrics:
        def record_qdrant_search(self, **kwargs: object) -> None:
            del kwargs
            raise BaseException("metrics-secret")

    class FailingHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            del record
            raise BaseException("logging-secret")

    def broken_clock() -> float:
        raise BaseException("clock-secret")

    kb_id = uuid4()
    target = _target(kb_id)
    query_sink = CapturingQueryLogSink()
    previous_handlers = list(retrieval_services.logger.handlers)
    previous_propagate = retrieval_services.logger.propagate
    previous_level = retrieval_services.logger.level
    retrieval_services.logger.handlers = [FailingHandler()]
    retrieval_services.logger.propagate = False
    retrieval_services.logger.setLevel(logging.INFO)
    service, *_ = _service(
        FakeSearchRepository(target, _runtime(target)),
        query_sink=query_sink,
        clock=broken_clock,
        metrics=FailingMetrics(),
    )
    try:
        response = await service.search(
            knowledge_base_id=kb_id,
            actor=_actor(kb_id),
            request_id="req-observability-fail-open",
            command=SearchRequest(query="private-query"),
        )
    finally:
        retrieval_services.logger.handlers = previous_handlers
        retrieval_services.logger.propagate = previous_propagate
        retrieval_services.logger.setLevel(previous_level)

    assert response.results == ()
    assert query_sink.records[0][2] == 0


@pytest.mark.asyncio
async def test_search_maps_provider_and_qdrant_failures_without_content_leakage() -> None:
    kb_id = uuid4()
    target = _target(kb_id)
    repository = FakeSearchRepository(target, _runtime(target))
    gateway = CapturingEmbeddingGateway(
        EmbeddingGatewayError(
            "PROVIDER_AUTHENTICATION_FAILED",
            "upstream-secret and query body",
            retryable=False,
        )
    )
    query_sink = CapturingQueryLogSink()
    service, *_ = _service(repository, gateway=gateway, query_sink=query_sink)

    with pytest.raises(BusinessError) as failure:
        await service.search(
            knowledge_base_id=kb_id,
            actor=_actor(kb_id),
            request_id="req-provider-failure",
            command=SearchRequest(query="private query body"),
        )

    assert (failure.value.status_code, failure.value.code, failure.value.retryable) == (
        503,
        "QUERY_EMBEDDING_FAILED",
        False,
    )
    assert "secret" not in failure.value.message
    assert "private query body" not in failure.value.message
    assert query_sink.records[0][1] == "failed"


@pytest.mark.asyncio
async def test_query_log_is_content_free_and_best_effort() -> None:
    kb_id = uuid4()
    target = _target(kb_id)
    repository = FakeSearchRepository(target, _runtime(target))
    query_sink = CapturingQueryLogSink(RuntimeError("database-secret"))
    service, *_ = _service(repository, query_sink=query_sink)

    response = await service.search(
        knowledge_base_id=kb_id,
        actor=_actor(kb_id),
        request_id="req-query-log-best-effort",
        command=SearchRequest(query="do-not-store-this"),
    )

    assert response.results == ()
    context, status, latency_ms, degraded = query_sink.records[0]
    assert set(context.__dataclass_fields__) == {
        "request_id",
        "actor_api_key_id",
        "knowledge_base_ids",
    }
    assert (status, latency_ms, degraded) == ("succeeded", 12, False)
    assert "do-not-store-this" not in repr(context)


@pytest.mark.asyncio
async def test_fake_qdrant_search_applies_safe_filters_and_keeps_raw_score_order() -> None:
    kb_id, generation_id = uuid4(), uuid4()
    spec = CollectionSpec(collection_name(kb_id, generation_id), 3, "cosine", ())
    qdrant = FakeQdrantClient()
    await qdrant.seed_collection(spec, created_at=datetime.now(UTC))
    first_document, second_document = uuid4(), uuid4()
    first_version, second_version = uuid4(), uuid4()
    await qdrant.upsert_points(
        spec.name,
        (
            QdrantPoint(
                point_id(first_version, 0, "a" * 64),
                (1.0, 0.0, 0.0),
                _payload(
                    knowledge_base_id=kb_id,
                    document_id=first_document,
                    version_id=first_version,
                    chunk_index=0,
                    text="first",
                    metadata={f"f_{_CATEGORY_ID.hex}": "guide"},
                ),
            ),
            QdrantPoint(
                point_id(second_version, 0, "a" * 64),
                (0.0, 1.0, 0.0),
                _payload(
                    knowledge_base_id=kb_id,
                    document_id=second_document,
                    version_id=second_version,
                    chunk_index=0,
                    text="second",
                    metadata={f"f_{_CATEGORY_ID.hex}": "other"},
                ),
            ),
        ),
    )

    points = await qdrant.search_points(
        spec.name,
        (1.0, 0.0, 0.0),
        limit=20,
        query_filter=QdrantSearchFilter(
            document_ids=(first_document, uuid4()),
            conditions=(
                QdrantFilterCondition(
                    path=f"metadata.f_{_CATEGORY_ID.hex}",
                    field_type="keyword",
                    operator="eq",
                    value="guide",
                ),
            ),
        ),
    )

    assert len(points) == 1
    assert points[0].score == 1.0
    assert points[0].payload["document_id"] == str(first_document)


@pytest.mark.asyncio
async def test_real_qdrant_adapter_translates_allowlisted_filters_only() -> None:
    kb_id, generation_id = uuid4(), uuid4()
    collection = collection_name(kb_id, generation_id)
    document_id, version_id = uuid4(), uuid4()
    captured: dict[str, object] = {}
    payload = _payload(
        knowledge_base_id=kb_id,
        document_id=document_id,
        version_id=version_id,
        chunk_index=0,
        text="safe",
    )

    class RawClient:
        async def get_collection(self, _collection: str) -> None:
            return None

        async def query_points(self, **kwargs: object) -> models.QueryResponse:
            captured.update(kwargs)
            return models.QueryResponse(
                points=[
                    models.ScoredPoint(
                        id=point_id(version_id, 0, "a" * 64),
                        version=1,
                        score=0.75,
                        payload=payload,
                    )
                ]
            )

    class Adapter(AsyncQdrantCollectionClient):
        async def _managed_info(self, _collection: str) -> models.CollectionInfo:
            return cast(models.CollectionInfo, object())

    adapter = Adapter(cast(AsyncQdrantClient, RawClient()))
    query_filter = QdrantSearchFilter(
        document_ids=(document_id,),
        conditions=(
            QdrantFilterCondition(
                path=f"metadata.f_{_CATEGORY_ID.hex}",
                field_type="keyword",
                operator="in",
                value=("guide", "reference"),
            ),
            QdrantFilterCondition(
                path="metadata.f_55555555555545558555555555555555",
                field_type="boolean",
                operator="in",
                value=(True, False),
            ),
            QdrantFilterCondition(
                path=f"metadata.f_{_PRIORITY_ID.hex}",
                field_type="integer",
                operator="lte",
                value=2**54,
            ),
        ),
    )

    result = await adapter.search_points(
        collection,
        (1.0, 0.0, 0.0),
        limit=20,
        query_filter=query_filter,
    )

    assert len(result) == 1 and result[0].score == 0.75
    translated = captured["query_filter"]
    assert isinstance(translated, models.Filter)
    must_conditions = translated.must
    assert isinstance(must_conditions, list) and len(must_conditions) == 4
    integer_range = must_conditions[3]
    assert isinstance(integer_range, models.FieldCondition)
    assert isinstance(integer_range.range, models.Range)
    assert integer_range.range.lte == float(2**54)
    assert captured["with_payload"] is True
    assert captured["with_vectors"] is False


@dataclass(slots=True)
class _StubRerankGateway:
    """Reverses the candidate order, or raises when told to."""

    error: BaseException | None = None
    calls: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)

    async def rerank(
        self,
        *,
        snapshot: object,
        operational: object,
        query: str,
        documents: tuple[str, ...],
    ) -> tuple[RerankedDocument, ...]:
        self.calls.append((query, documents))
        if self.error is not None:
            raise self.error
        count = len(documents)
        return tuple(
            RerankedDocument(index=index, score=float(index)) for index in range(count - 1, -1, -1)
        )


def _rerank_runtime(profile_id: UUID) -> RerankRuntimeRecord:
    return RerankRuntimeRecord(
        profile_id=profile_id,
        profile_capability="rerank",
        profile_enabled=True,
        model_name="bge-reranker-v2-m3",
        timeout_seconds=Decimal("30.000"),
        provider_id=uuid4(),
        provider_type="openai_compatible",
        base_url="https://provider.example/v1",
        default_headers={},
        provider_enabled=True,
        credential_id=uuid4(),
        credential_exists=True,
    )


def _rerank_candidates(kb_id: UUID, count: int = 4) -> tuple[QdrantSearchPoint, ...]:
    return tuple(
        QdrantSearchPoint(
            point_id(version_id := uuid4(), index, "a" * 64),
            1.0 - index / 10,
            _payload(
                knowledge_base_id=kb_id,
                document_id=uuid4(),
                version_id=version_id,
                chunk_index=index,
                text=f"candidate-{index}",
            ),
        )
        for index in range(count)
    )


def _visible_all(
    repository: FakeSearchRepository, candidates: tuple[QdrantSearchPoint, ...]
) -> None:
    for point in candidates:
        payload = point.payload
        document_id = UUID(cast(str, payload["document_id"]))
        version_id = UUID(cast(str, payload["version_id"]))
        repository.visible[(document_id, version_id)] = VisibleDocument(
            document_id=document_id,
            version_id=version_id,
            source_filename="doc.md",
        )


@pytest.mark.asyncio
async def test_search_reranks_over_candidates_beyond_top_k() -> None:
    kb_id = uuid4()
    profile_id = uuid4()
    target = _target(kb_id, rerank_profile_id=profile_id)
    candidates = _rerank_candidates(kb_id)
    repository = FakeSearchRepository(target, _runtime(target), _rerank_runtime(profile_id))
    _visible_all(repository, candidates)
    gateway = _StubRerankGateway()
    service, *_ = _service(
        repository, index=CapturingSearchIndex(candidates), rerank_gateway=gateway
    )

    response = await service.search(
        knowledge_base_id=kb_id,
        actor=_actor(kb_id),
        request_id="req-rerank",
        command=SearchRequest(query="q", top_k=2, rerank=True),
    )

    # The reranker sees every visible candidate, not just the two that would
    # have been returned — promoting a candidate from below top_k is the whole
    # reason for over-fetching.
    assert gateway.calls[0][1] == tuple(f"candidate-{index}" for index in range(4))
    assert [result.text for result in response.results] == ["candidate-3", "candidate-2"]


@pytest.mark.asyncio
async def test_search_skips_rerank_unless_requested() -> None:
    kb_id = uuid4()
    profile_id = uuid4()
    target = _target(kb_id, rerank_profile_id=profile_id)
    candidates = _rerank_candidates(kb_id)
    repository = FakeSearchRepository(target, _runtime(target), _rerank_runtime(profile_id))
    _visible_all(repository, candidates)
    gateway = _StubRerankGateway()
    service, *_ = _service(
        repository, index=CapturingSearchIndex(candidates), rerank_gateway=gateway
    )

    response = await service.search(
        knowledge_base_id=kb_id,
        actor=_actor(kb_id),
        request_id="req-no-rerank",
        command=SearchRequest(query="q", top_k=2),
    )

    assert gateway.calls == []
    assert [result.text for result in response.results] == ["candidate-0", "candidate-1"]


@pytest.mark.asyncio
async def test_search_falls_back_to_dense_order_when_the_reranker_fails() -> None:
    kb_id = uuid4()
    profile_id = uuid4()
    target = _target(kb_id, rerank_profile_id=profile_id)
    candidates = _rerank_candidates(kb_id)
    repository = FakeSearchRepository(target, _runtime(target), _rerank_runtime(profile_id))
    _visible_all(repository, candidates)
    gateway = _StubRerankGateway(
        error=RerankGatewayError("PROVIDER_UNAVAILABLE", "Provider unavailable", retryable=True)
    )
    service, *_ = _service(
        repository, index=CapturingSearchIndex(candidates), rerank_gateway=gateway
    )

    response = await service.search(
        knowledge_base_id=kb_id,
        actor=_actor(kb_id),
        request_id="req-rerank-down",
        command=SearchRequest(query="q", top_k=2, rerank=True),
    )

    # A reranker refines an order that is already useful, so losing it must not
    # turn a good answer into no answer.
    assert [result.text for result in response.results] == ["candidate-0", "candidate-1"]


@pytest.mark.asyncio
async def test_search_falls_back_when_the_rerank_profile_is_misconfigured() -> None:
    kb_id = uuid4()
    profile_id = uuid4()
    target = _target(kb_id, rerank_profile_id=profile_id)
    candidates = _rerank_candidates(kb_id)
    runtime = _rerank_runtime(profile_id)
    repository = FakeSearchRepository(
        target,
        _runtime(target),
        replace(runtime, provider_enabled=False),
    )
    _visible_all(repository, candidates)
    gateway = _StubRerankGateway()
    service, *_ = _service(
        repository, index=CapturingSearchIndex(candidates), rerank_gateway=gateway
    )

    response = await service.search(
        knowledge_base_id=kb_id,
        actor=_actor(kb_id),
        request_id="req-rerank-misconfigured",
        command=SearchRequest(query="q", top_k=2, rerank=True),
    )

    assert gateway.calls == []
    assert [result.text for result in response.results] == ["candidate-0", "candidate-1"]


@pytest.mark.asyncio
async def test_search_rejects_rerank_when_the_knowledge_base_has_none() -> None:
    kb_id = uuid4()
    target = _target(kb_id)
    candidates = _rerank_candidates(kb_id)
    repository = FakeSearchRepository(target, _runtime(target))
    _visible_all(repository, candidates)
    service, *_ = _service(
        repository, index=CapturingSearchIndex(candidates), rerank_gateway=_StubRerankGateway()
    )

    # Distinguished from a provider failure on purpose: this one is answered by
    # configuring the knowledge base, so silently ignoring it would leave the
    # caller believing reranking was applied.
    with pytest.raises(BusinessError) as raised:
        await service.search(
            knowledge_base_id=kb_id,
            actor=_actor(kb_id),
            request_id="req-rerank-unconfigured",
            command=SearchRequest(query="q", top_k=2, rerank=True),
        )

    assert (raised.value.status_code, raised.value.code) == (409, "RERANK_NOT_CONFIGURED")
