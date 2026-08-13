"""Authorized Dense retrieval over one immutable active index generation."""

from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Literal, Protocol, cast
from uuid import RFC_4122, UUID

from rag_service.api.errors import BusinessError
from rag_service.api.validation import JSONValue
from rag_service.auth.policies import AgentPrincipal, Capability, require_capability
from rag_service.indexing.generation_services import build_filter_snapshot
from rag_service.indexing.identities import canonical_sha256, collection_name
from rag_service.indexing.qdrant import (
    QdrantConfigurationError,
    QdrantFilterCondition,
    QdrantSearchFilter,
    QdrantSearchPoint,
    QdrantTransientError,
)
from rag_service.ingestion.schemas import is_rfc3339_datetime
from rag_service.observability import METRICS, OperationalMetrics, SafeLogContext, emit_safe_log
from rag_service.observability.repositories import ProviderUsageContext, QueryLogContext
from rag_service.providers.embeddings import (
    EmbeddingAttempt,
    EmbeddingAttemptObserver,
    EmbeddingConfigSnapshot,
    EmbeddingGatewayError,
    EmbeddingOperationalConfig,
    EmbeddingResult,
)
from rag_service.providers.rerank import (
    RerankConfigSnapshot,
    RerankGateway,
    RerankOperationalConfig,
)
from rag_service.retrieval.repositories import (
    ActiveSearchTarget,
    RerankRuntimeRecord,
    SearchRepository,
    VisibleDocument,
)
from rag_service.retrieval.schemas import (
    SearchFilters,
    SearchIndex,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SearchSource,
)

_POSTGRES_BIGINT_MAX = 2**63 - 1
_FILTER_TYPES = frozenset({"keyword", "integer", "float", "boolean", "datetime"})
_FILTER_OPERATORS = {
    "keyword": frozenset({"eq", "in"}),
    "integer": frozenset({"eq", "in", "gte", "lte"}),
    "float": frozenset({"eq", "in", "gte", "lte"}),
    "boolean": frozenset({"eq", "in"}),
    "datetime": frozenset({"eq", "in", "gte", "lte"}),
}
_FILTER_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")
_FILTER_SOURCE_PATH = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}(\.[A-Za-z][A-Za-z0-9_]{0,63}){0,3}")

logger = logging.getLogger(__name__)


class EmbeddingGateway(Protocol):
    async def embed(
        self,
        *,
        snapshot: EmbeddingConfigSnapshot,
        operational: EmbeddingOperationalConfig,
        inputs: Sequence[str],
        attempt_observer: EmbeddingAttemptObserver | None = None,
    ) -> EmbeddingResult: ...


class SearchIndexClient(Protocol):
    async def search_points(
        self,
        collection: str,
        vector: Sequence[float],
        *,
        limit: int,
        query_filter: QdrantSearchFilter,
    ) -> tuple[QdrantSearchPoint, ...]: ...


class ProviderUsageSink(Protocol):
    async def record(
        self,
        context: ProviderUsageContext,
        attempt: EmbeddingAttempt,
    ) -> None: ...


class QueryLogSink(Protocol):
    async def record(
        self,
        context: QueryLogContext,
        *,
        status: str,
        latency_ms: int,
        degraded: bool,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class _FilterField:
    name: str
    field_type: str
    operators: frozenset[str]
    payload_path: str


@dataclass(frozen=True, slots=True)
class _EmbeddingConfiguration:
    snapshot: EmbeddingConfigSnapshot
    provider_id: UUID
    credential_id: UUID


def candidate_limit(top_k: int) -> int:
    if type(top_k) is not int or not 1 <= top_k <= 50:
        raise ValueError("top_k is invalid")
    return min(max(top_k * 4, 20), 200)


def _configuration_unavailable() -> BusinessError:
    return BusinessError(
        503,
        "RETRIEVAL_CONFIGURATION_UNAVAILABLE",
        "Retrieval configuration is unavailable",
    )


def _parse_filter_snapshot(snapshot: dict[str, object]) -> tuple[_FilterField, ...]:
    try:
        validated = build_filter_snapshot(snapshot)
        raw_fields = cast(list[dict[str, object]], validated["fields"])
        fields: list[_FilterField] = []
        seen_names: set[str] = set()
        seen_source_paths: set[str] = set()
        for raw in raw_fields:
            name = raw["name"]
            source_path = raw["source_path"]
            field_type = raw["type"]
            operators = raw["operators"]
            field_id = raw["field_id"]
            payload_path = raw["payload_path"]
            if (
                type(name) is not str
                or _FILTER_NAME.fullmatch(name) is None
                or name in seen_names
                or type(source_path) is not str
                or _FILTER_SOURCE_PATH.fullmatch(source_path) is None
                or source_path in seen_source_paths
                or type(field_type) is not str
                or field_type not in _FILTER_TYPES
                or type(operators) is not list
                or not operators
                or len(operators) != len(set(operators))
                or operators != sorted(operators)
                or any(
                    type(operator) is not str or operator not in _FILTER_OPERATORS[field_type]
                    for operator in operators
                )
                or type(field_id) is not str
                or type(payload_path) is not str
            ):
                raise ValueError
            encoded = field_id.removeprefix("fld_")
            identifier_bytes = base64.b64decode(
                f"{encoded}==",
                altchars=b"-_",
                validate=True,
            )
            identifier = UUID(bytes=identifier_bytes)
            canonical_field_id = (
                "fld_" + base64.urlsafe_b64encode(identifier_bytes).rstrip(b"=").decode()
            )
            if (
                field_id != canonical_field_id
                or identifier.variant != RFC_4122
                or identifier.version != 4
                or payload_path != f"metadata.f_{identifier.hex}"
            ):
                raise ValueError
            seen_names.add(name)
            seen_source_paths.add(source_path)
            fields.append(
                _FilterField(
                    name=name,
                    field_type=field_type,
                    operators=frozenset(cast(list[str], operators)),
                    payload_path=payload_path,
                )
            )
        return tuple(fields)
    except (binascii.Error, KeyError, TypeError, ValueError):
        raise _configuration_unavailable() from None


def _valid_filter_value(field_type: str, value: object) -> bool:
    if field_type == "keyword":
        return type(value) is str and len(value) <= 4096 and "\x00" not in value
    if field_type == "integer":
        return type(value) is int and -(2**63) <= value <= 2**63 - 1
    if field_type == "float":
        try:
            return (
                type(value) in {int, float}
                and not isinstance(value, bool)
                and math.isfinite(float(cast(int | float, value)))
            )
        except OverflowError:
            return False
    if field_type == "boolean":
        return type(value) is bool
    if field_type == "datetime":
        return is_rfc3339_datetime(value)
    return False


def compile_search_filter(
    filters: SearchFilters | None,
    snapshot: dict[str, object],
) -> QdrantSearchFilter:
    fields = {field.name: field for field in _parse_filter_snapshot(snapshot)}
    if filters is None:
        return QdrantSearchFilter()
    conditions: list[QdrantFilterCondition] = []
    try:
        for name, expression in filters.metadata.items():
            field = fields.get(name)
            if field is None:
                raise ValueError
            if type(expression) is dict:
                operator, value = next(iter(expression.items()))
            else:
                operator, value = "eq", expression
            if type(operator) is not str or operator not in field.operators:
                raise ValueError
            if operator == "in":
                if type(value) is not tuple or any(
                    not _valid_filter_value(field.field_type, member) for member in value
                ):
                    raise ValueError
            elif not _valid_filter_value(field.field_type, value):
                raise ValueError
            conditions.append(
                QdrantFilterCondition(
                    path=field.payload_path,
                    field_type=cast(
                        Literal["keyword", "integer", "float", "boolean", "datetime"],
                        field.field_type,
                    ),  # narrowed by the validated snapshot above
                    operator=cast(Literal["eq", "in", "gte", "lte"], operator),
                    value=value,
                )
            )
    except (KeyError, StopIteration, TypeError, ValueError):
        raise BusinessError(
            422,
            "INVALID_SEARCH_FILTER",
            "Search filter is invalid",
        ) from None
    return QdrantSearchFilter(document_ids=filters.document_ids, conditions=tuple(conditions))


def _parse_embedding_configuration(
    snapshot: dict[str, object],
    *,
    embedding_config_hash: str,
    index_profile_hash: str,
) -> _EmbeddingConfiguration:
    try:
        expected = {
            "adapter_schema_version",
            "provider_type",
            "base_url",
            "provider_config_id",
            "credential_id",
            "default_headers",
            "routing_options",
            "model_name",
            "dimension",
            "distance",
            "max_input_tokens",
            "vector_config",
        }
        if type(snapshot) is not dict or set(snapshot) != expected:
            raise ValueError
        provider_id = UUID(cast(str, snapshot["provider_config_id"]))
        credential_id = UUID(cast(str, snapshot["credential_id"]))
        if snapshot["provider_config_id"] != str(provider_id) or snapshot["credential_id"] != str(
            credential_id
        ):
            raise ValueError
        gateway_snapshot = EmbeddingConfigSnapshot(
            adapter_schema_version=cast(str, snapshot["adapter_schema_version"]),
            provider_type=cast(str, snapshot["provider_type"]),
            base_url=cast(str, snapshot["base_url"]),
            credential_id=credential_id,
            default_headers=cast(Mapping[str, str], snapshot["default_headers"]),
            routing_options=cast(Mapping[str, object], snapshot["routing_options"]),
            model_name=cast(str, snapshot["model_name"]),
            dimension=cast(int, snapshot["dimension"]),
            distance=cast(str, snapshot["distance"]),
            max_input_tokens=cast(int, snapshot["max_input_tokens"]),
            vector_config=cast(Mapping[str, object], snapshot["vector_config"]),
        )
        semantic: dict[str, object] = {
            "adapter_schema_version": gateway_snapshot.adapter_schema_version,
            "provider_type": gateway_snapshot.provider_type,
            "base_url": gateway_snapshot.base_url,
            "default_headers": dict(gateway_snapshot.default_headers),
            "routing_options": dict(gateway_snapshot.routing_options),
            "model_name": gateway_snapshot.model_name,
            "dimension": gateway_snapshot.dimension,
            "distance": gateway_snapshot.distance,
            "max_input_tokens": gateway_snapshot.max_input_tokens,
            "vector_config": dict(gateway_snapshot.vector_config),
        }
        semantic_hash = canonical_sha256(semantic)
        if semantic_hash != embedding_config_hash or semantic_hash != index_profile_hash:
            raise ValueError
        return _EmbeddingConfiguration(gateway_snapshot, provider_id, credential_id)
    except (KeyError, TypeError, ValueError):
        raise _configuration_unavailable() from None


def _safe_duration_seconds(started_at: float | None, finished_at: float | None) -> float:
    if started_at is None or finished_at is None:
        return 0.0
    elapsed = finished_at - started_at
    if not math.isfinite(elapsed) or elapsed < 0:
        return 0.0
    return elapsed


def _safe_latency_ms(started_at: float | None, finished_at: float | None) -> int:
    elapsed = _safe_duration_seconds(started_at, finished_at)
    return min(round(elapsed * 1000), _POSTGRES_BIGINT_MAX)


@dataclass(frozen=True, slots=True)
class _RerankOutcome:
    applied: bool
    degraded: bool


def _rerank_configuration(
    runtime: RerankRuntimeRecord | None,
) -> tuple[RerankConfigSnapshot, RerankOperationalConfig]:
    """Build the rerank call configuration, refusing anything half-configured.

    Raises rather than returning None so the caller's except branch treats a
    missing provider the same as a failing one: reranking degrades to the dense
    order either way, and a search should not fail because a reranker is
    misconfigured.
    """
    if (
        runtime is None
        or runtime.profile_capability != "rerank"
        or runtime.provider_type is None
        or runtime.base_url is None
        or runtime.credential_id is None
        or not runtime.credential_exists
        or runtime.provider_enabled is not True
    ):
        raise ValueError("rerank configuration is unavailable")
    headers = runtime.default_headers if isinstance(runtime.default_headers, Mapping) else {}
    snapshot = RerankConfigSnapshot(
        provider_type=runtime.provider_type,
        base_url=runtime.base_url,
        credential_id=runtime.credential_id,
        model_name=runtime.model_name,
        default_headers={key: value for key, value in headers.items() if type(value) is str},
    )
    operational = RerankOperationalConfig(
        timeout_seconds=runtime.timeout_seconds,
        provider_enabled=runtime.provider_enabled,
        profile_enabled=runtime.profile_enabled,
    )
    return snapshot, operational


class SearchService:
    def __init__(
        self,
        *,
        repository: SearchRepository,
        embedding_gateway: EmbeddingGateway,
        rerank_gateway: RerankGateway | None = None,
        search_index: SearchIndexClient,
        provider_usage_sink: ProviderUsageSink,
        query_log_sink: QueryLogSink,
        monotonic_clock: Callable[[], float] = time.monotonic,
        metrics: OperationalMetrics = METRICS,
    ) -> None:
        self._repository = repository
        self._embedding_gateway = embedding_gateway
        self._rerank_gateway = rerank_gateway
        self._search_index = search_index
        self._provider_usage_sink = provider_usage_sink
        self._query_log_sink = query_log_sink
        self._monotonic_clock = monotonic_clock
        self._metrics = metrics

    async def search(
        self,
        *,
        knowledge_base_id: UUID,
        actor: AgentPrincipal,
        request_id: str,
        command: SearchRequest,
    ) -> SearchResponse:
        started_at = self._safe_now()
        status = "failed"
        degraded = False
        query_context = QueryLogContext(request_id, actor.key_id, (knowledge_base_id,))
        try:
            authorized = require_capability(actor, Capability.RETRIEVE)
            if knowledge_base_id not in authorized.knowledge_base_ids:
                raise BusinessError(404, "RESOURCE_NOT_FOUND", "Resource not found")

            target = await self._repository.get_active_target(knowledge_base_id)
            if target is None or target.generation_status != "active":
                raise BusinessError(
                    503,
                    "ACTIVE_GENERATION_UNAVAILABLE",
                    "Active index generation is unavailable",
                )
            if target.knowledge_base_status != "active":
                raise BusinessError(
                    503,
                    "KNOWLEDGE_BASE_UNAVAILABLE",
                    "Knowledge base is unavailable",
                )
            if (
                target.knowledge_base_id != knowledge_base_id
                or target.qdrant_collection_name
                != collection_name(knowledge_base_id, target.generation_id)
            ):
                raise _configuration_unavailable()
            embedding = _parse_embedding_configuration(
                target.embedding_config_snapshot,
                embedding_config_hash=target.embedding_config_hash,
                index_profile_hash=target.index_profile_hash,
            )
            filter_snapshot = _parse_filter_snapshot(target.filter_schema_snapshot)
            query_filter = compile_search_filter(command.filters, target.filter_schema_snapshot)

            runtime = await self._repository.get_embedding_runtime(
                target.embedding_profile_id,
                embedding.provider_id,
                embedding.credential_id,
            )
            if (
                runtime is None
                or runtime.profile_id != target.embedding_profile_id
                or runtime.profile_capability != "embedding"
                or not runtime.profile_enabled
            ):
                raise BusinessError(
                    503,
                    "MODEL_PROFILE_UNAVAILABLE",
                    "Model profile is unavailable",
                )
            if (
                runtime.provider_id != embedding.provider_id
                or runtime.provider_enabled is not True
                or runtime.max_concurrency is None
                or runtime.requests_per_minute is None
            ):
                raise BusinessError(
                    503,
                    "PROVIDER_CONFIG_UNAVAILABLE",
                    "Provider configuration is unavailable",
                )
            if runtime.credential_id != embedding.credential_id or not runtime.credential_exists:
                raise BusinessError(
                    503,
                    "PROVIDER_CREDENTIAL_UNAVAILABLE",
                    "Provider credential is unavailable",
                )
            try:
                operational = EmbeddingOperationalConfig(
                    provider_config_id=runtime.provider_id,
                    provider_enabled=runtime.provider_enabled,
                    profile_enabled=runtime.profile_enabled,
                    timeout_seconds=runtime.timeout_seconds,
                    max_concurrency=runtime.max_concurrency,
                    requests_per_minute=runtime.requests_per_minute,
                    batch_size=runtime.batch_size,
                )
            except ValueError:
                raise _configuration_unavailable() from None

            if command.filters is not None and command.filters.document_ids:
                requested_document_ids = command.filters.document_ids
                visible_document_filters = await self._repository.load_visible_document_filters(
                    knowledge_base_id=knowledge_base_id,
                    generation_id=target.generation_id,
                    document_ids=requested_document_ids,
                )
                visible_document_ids = tuple(
                    document_id
                    for document_id in requested_document_ids
                    if document_id in visible_document_filters
                )
                if not visible_document_ids:
                    status = "succeeded"
                    return SearchResponse(
                        results=(),
                        index=SearchIndex(
                            generation_id=target.generation_id,
                            embedding_profile_id=target.embedding_profile_id,
                        ),
                    )
                query_filter = QdrantSearchFilter(
                    document_ids=visible_document_ids,
                    conditions=query_filter.conditions,
                )

            usage_context = ProviderUsageContext(
                request_id=request_id,
                actor_api_key_id=authorized.key_id,
                provider_config_id=runtime.provider_id,
                model_profile_id=target.embedding_profile_id,
            )

            async def observe_attempt(attempt: EmbeddingAttempt) -> None:
                try:
                    await self._provider_usage_sink.record(usage_context, attempt)
                except Exception:
                    return

            try:
                embedded = await self._embedding_gateway.embed(
                    snapshot=embedding.snapshot,
                    operational=operational,
                    inputs=(command.query,),
                    attempt_observer=observe_attempt,
                )
            except EmbeddingGatewayError as error:
                raise BusinessError(
                    503,
                    "QUERY_EMBEDDING_FAILED",
                    "Query embedding failed",
                    retryable=error.retryable,
                ) from None
            degraded = embedded.telemetry_degraded
            if len(embedded.vectors) != 1:
                raise _configuration_unavailable()
            vector = embedded.vectors[0]
            if len(vector) != embedding.snapshot.dimension or any(
                type(value) is not float or not math.isfinite(value) for value in vector
            ):
                raise _configuration_unavailable()
            qdrant_started_at = self._safe_now()
            try:
                candidates = await self._search_index.search_points(
                    target.qdrant_collection_name,
                    vector,
                    limit=candidate_limit(command.top_k),
                    query_filter=query_filter,
                )
            except asyncio.CancelledError:
                self._observe_qdrant_search(
                    outcome="cancelled",
                    duration_seconds=_safe_duration_seconds(
                        qdrant_started_at,
                        self._safe_now(),
                    ),
                    request_id=request_id,
                    knowledge_base_id=knowledge_base_id,
                    generation_id=target.generation_id,
                )
                raise
            except QdrantTransientError:
                self._observe_qdrant_search(
                    outcome="failed",
                    duration_seconds=_safe_duration_seconds(
                        qdrant_started_at,
                        self._safe_now(),
                    ),
                    request_id=request_id,
                    knowledge_base_id=knowledge_base_id,
                    generation_id=target.generation_id,
                    error_code="QDRANT_SEARCH_FAILED",
                )
                raise BusinessError(
                    503,
                    "RETRIEVAL_UNAVAILABLE",
                    "Retrieval is unavailable",
                    retryable=True,
                ) from None
            except (QdrantConfigurationError, ValueError):
                self._observe_qdrant_search(
                    outcome="failed",
                    duration_seconds=_safe_duration_seconds(
                        qdrant_started_at,
                        self._safe_now(),
                    ),
                    request_id=request_id,
                    knowledge_base_id=knowledge_base_id,
                    generation_id=target.generation_id,
                    error_code="QDRANT_SEARCH_FAILED",
                )
                raise _configuration_unavailable() from None
            except BaseException:
                self._observe_qdrant_search(
                    outcome="failed",
                    duration_seconds=_safe_duration_seconds(
                        qdrant_started_at,
                        self._safe_now(),
                    ),
                    request_id=request_id,
                    knowledge_base_id=knowledge_base_id,
                    generation_id=target.generation_id,
                    error_code="QDRANT_SEARCH_FAILED",
                )
                raise

            qdrant_duration_seconds = _safe_duration_seconds(
                qdrant_started_at,
                self._safe_now(),
            )
            try:
                identities = tuple(
                    (
                        UUID(cast(str, point.payload["document_id"])),
                        UUID(cast(str, point.payload["version_id"])),
                    )
                    for point in candidates
                )
                visible = await self._repository.load_visible_documents(
                    knowledge_base_id=knowledge_base_id,
                    generation_id=target.generation_id,
                    identities=identities,
                )
                metadata_names = {
                    field.payload_path.removeprefix("metadata."): field.name
                    for field in filter_snapshot
                }
                visible_results: list[SearchResult] = []
                for point, identity in zip(candidates, identities, strict=True):
                    canonical = visible.get(identity)
                    if canonical is None:
                        continue
                    visible_results.append(self._result(point, canonical, metadata_names))
                # Truncation happens after reranking, not before: the point of
                # over-fetching candidates is to give the reranker something to
                # reorder, and cutting to top_k first would throw away exactly
                # the candidates it exists to promote.
                results, reranked = await self._apply_rerank(
                    target=target,
                    command=command,
                    request_id=request_id,
                    actor_key_id=authorized.key_id,
                    candidates=visible_results,
                )
                degraded = degraded or reranked.degraded
                response = SearchResponse(
                    results=tuple(results),
                    index=SearchIndex(
                        generation_id=target.generation_id,
                        embedding_profile_id=target.embedding_profile_id,
                    ),
                )
            except BaseException:
                self._observe_qdrant_search(
                    outcome="succeeded",
                    duration_seconds=qdrant_duration_seconds,
                    request_id=request_id,
                    knowledge_base_id=knowledge_base_id,
                    generation_id=target.generation_id,
                    candidate_count=len(candidates),
                )
                raise
            self._observe_qdrant_search(
                outcome="succeeded",
                duration_seconds=qdrant_duration_seconds,
                request_id=request_id,
                knowledge_base_id=knowledge_base_id,
                generation_id=target.generation_id,
                candidate_count=len(candidates),
                result_count=len(results),
                visibility_drop_count=sum(identity not in visible for identity in identities),
            )
            status = "succeeded"
            return response
        except BusinessError as error:
            status = "rejected" if error.status_code < 500 else "failed"
            raise
        finally:
            latency_ms = _safe_latency_ms(started_at, self._safe_now())
            with suppress(Exception):
                await self._query_log_sink.record(
                    query_context,
                    status=status,
                    latency_ms=latency_ms,
                    degraded=degraded,
                )

    def _safe_now(self) -> float | None:
        try:
            value = float(self._monotonic_clock())
        except BaseException:
            return None
        return value if math.isfinite(value) else None

    def _observe_qdrant_search(
        self,
        *,
        outcome: Literal["succeeded", "failed", "cancelled"],
        duration_seconds: float,
        request_id: str,
        knowledge_base_id: UUID,
        generation_id: UUID,
        candidate_count: int = 0,
        result_count: int = 0,
        visibility_drop_count: int = 0,
        error_code: str | None = None,
    ) -> None:
        with suppress(BaseException):
            self._metrics.record_qdrant_search(
                outcome=outcome,
                duration_seconds=duration_seconds,
                candidate_count=candidate_count,
                result_count=result_count,
                visibility_drop_count=visibility_drop_count,
            )
        fields: dict[str, object] = {
            "operation": "qdrant_search",
            "outcome": outcome,
            "duration_seconds": duration_seconds,
            "candidate_count": candidate_count,
            "result_count": result_count,
            "visibility_drop_count": visibility_drop_count,
        }
        if error_code is not None:
            fields["error_code"] = error_code
        with suppress(BaseException):
            emit_safe_log(
                logger,
                logging.INFO,
                "qdrant.search.completed",
                context=SafeLogContext(
                    request_id=request_id,
                    knowledge_base_id=knowledge_base_id,
                    generation_id=generation_id,
                ),
                **fields,
            )

    async def _apply_rerank(
        self,
        *,
        target: ActiveSearchTarget,
        command: SearchRequest,
        request_id: str,
        actor_key_id: UUID,
        candidates: list[SearchResult],
    ) -> tuple[list[SearchResult], _RerankOutcome]:
        """Reorder candidates with a cross-encoder, or return them unchanged.

        Never fails the search. A reranker refines an order that is already
        useful — the measured baseline had the answer inside the top ten 93% of
        the time before any reranking — so turning a provider hiccup into a 503
        would trade a good answer for no answer. The response reports the
        degraded flag instead.
        """
        wanted = candidates[: command.top_k]
        if not command.rerank or self._rerank_gateway is None:
            return wanted, _RerankOutcome(applied=False, degraded=False)
        if target.rerank_profile_id is None:
            # Asked for, but the knowledge base has no reranker configured. That
            # is a configuration answer, not a transient one, so it is worth
            # telling the caller rather than silently ignoring the request.
            raise BusinessError(
                409,
                "RERANK_NOT_CONFIGURED",
                "Knowledge base has no rerank profile",
            )
        if not candidates:
            return wanted, _RerankOutcome(applied=False, degraded=False)
        try:
            runtime = await self._repository.get_rerank_runtime(target.rerank_profile_id)
            snapshot, operational = _rerank_configuration(runtime)
            ordered = await self._rerank_gateway.rerank(
                snapshot=snapshot,
                operational=operational,
                query=command.query,
                documents=tuple(result.text for result in candidates),
            )
        except BusinessError:
            raise
        except Exception:
            self._observe_rerank(
                outcome="failed",
                request_id=request_id,
                knowledge_base_id=target.knowledge_base_id,
                actor_key_id=actor_key_id,
            )
            return wanted, _RerankOutcome(applied=False, degraded=True)
        reordered = [candidates[item.index] for item in ordered][: command.top_k]
        self._observe_rerank(
            outcome="succeeded",
            request_id=request_id,
            knowledge_base_id=target.knowledge_base_id,
            actor_key_id=actor_key_id,
            candidate_count=len(candidates),
            result_count=len(reordered),
        )
        return reordered, _RerankOutcome(applied=True, degraded=False)

    def _observe_rerank(
        self,
        *,
        outcome: str,
        request_id: str,
        knowledge_base_id: UUID,
        actor_key_id: UUID,
        candidate_count: int | None = None,
        result_count: int | None = None,
    ) -> None:
        fields: dict[str, object] = {
            "operation": "retrieval_rerank",
            "outcome": outcome,
            "actor_api_key_id": str(actor_key_id),
        }
        if candidate_count is not None:
            fields["candidate_count"] = candidate_count
        if result_count is not None:
            fields["result_count"] = result_count
        with suppress(BaseException):
            emit_safe_log(
                logger,
                logging.INFO,
                "retrieval.rerank.completed",
                context=SafeLogContext(
                    request_id=request_id,
                    knowledge_base_id=knowledge_base_id,
                ),
                **fields,
            )

    @staticmethod
    def _result(
        point: QdrantSearchPoint,
        canonical: VisibleDocument,
        metadata_names: Mapping[str, str],
    ) -> SearchResult:
        payload = point.payload
        raw_title_path = cast(list[str], payload["title_path"])
        raw_metadata = cast(dict[str, object], payload["metadata"])
        metadata: dict[str, JSONValue] = {
            logical_name: cast(JSONValue, raw_metadata[internal_name])
            for internal_name, logical_name in metadata_names.items()
            if internal_name in raw_metadata
        }
        return SearchResult(
            text=cast(str, payload["text"]),
            score=point.score,
            document_id=canonical.document_id,
            version_id=canonical.version_id,
            chunk_index=cast(int, payload["chunk_index"]),
            title=raw_title_path[-1] if raw_title_path else None,
            title_path=tuple(raw_title_path),
            source=SearchSource(
                filename=canonical.source_filename,
                start_offset=cast(int, payload["start_offset"]),
                end_offset=cast(int, payload["end_offset"]),
            ),
            metadata=metadata,
        )


__all__ = ["SearchService", "candidate_limit", "compile_search_filter"]
