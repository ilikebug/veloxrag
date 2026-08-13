import asyncio
import hashlib
import math
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qdrant_models
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.api.errors import BusinessError
from rag_service.db.models.documents import Document, DocumentIndexState, DocumentVersion, Job
from rag_service.db.models.knowledge_bases import (
    KnowledgeBase,
    KnowledgeBaseIndexGeneration,
    KnowledgeBaseMutation,
)
from rag_service.db.models.providers import (
    ModelProfile,
    ProviderConfig,
    ProviderCredential,
)
from rag_service.db.session import Database
from rag_service.indexing.generation_services import payload_indexes_for_filter_snapshot
from rag_service.indexing.identities import canonical_sha256, collection_name, point_id
from rag_service.indexing.qdrant import (
    AsyncQdrantCollectionClient,
    CollectionSpec,
    FakeQdrantClient,
    QdrantClient,
    QdrantPoint,
    qdrant_client_from_url,
)
from rag_service.indexing.repair import (
    GenerationRepairHooks,
    GenerationRepairPipeline,
    GenerationRepairService,
)
from rag_service.infrastructure.minio_store import ObjectStoreError
from rag_service.ingestion.artifacts import (
    chunk_manifest_header_bytes,
    chunk_manifest_record_bytes,
    chunks_object_key,
)
from rag_service.ingestion.chunkers import Chunk, RecursiveTextChunker
from rag_service.ingestion.parsers import ParsedArtifact
from rag_service.jobs.repositories import (
    JobLease,
    JobRepository,
    LostLeaseError,
    SqlAlchemyJobRepository,
)
from rag_service.jobs.runner import ExponentialBackoff, JobRunner, RetryableJobError
from rag_service.observability.repositories import ProviderUsageContext
from rag_service.providers.embeddings import (
    EmbeddingAttempt,
    EmbeddingAttemptObserver,
    EmbeddingConfigSnapshot,
    EmbeddingOperationalConfig,
    EmbeddingResult,
)

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


@dataclass(frozen=True, slots=True)
class SeededGeneration:
    knowledge_base_id: UUID
    generation_id: UUID
    embedding_config_hash: str


def _repair_service(database: Database, qdrant: QdrantClient) -> GenerationRepairService:
    return GenerationRepairService(
        session_factory=database.sessions,
        qdrant=qdrant,
    )


class _QdrantConnection:
    url: str


@pytest.fixture(scope="session")
def repair_qdrant_url(qdrant_connection: _QdrantConnection) -> str:
    return qdrant_connection.url


async def test_real_qdrant_fixture_exposes_collection_churn_without_a_stale_snapshot(
    repair_qdrant_url: str,
) -> None:
    class CountingTransport:
        def __init__(self, client: AsyncQdrantClient) -> None:
            self.client = client
            self.get_collections_calls = 0
            self.get_collection_calls = 0

        def reset_counts(self) -> None:
            self.get_collections_calls = 0
            self.get_collection_calls = 0

        async def get_collections(self) -> Any:
            self.get_collections_calls += 1
            return await self.client.get_collections()

        async def get_collection(self, collection: str) -> Any:
            self.get_collection_calls += 1
            return await self.client.get_collection(collection)

        def __getattr__(self, name: str) -> Any:
            return getattr(self.client, name)

    names = {
        number: collection_name(UUID(int=number), UUID(int=1))
        for number in (10, 15, 20, 25, 30, 40, 45, 50, 55)
    }
    raw = AsyncQdrantClient(url=repair_qdrant_url, timeout=5)
    transport = CountingTransport(raw)
    client = AsyncQdrantCollectionClient(cast(AsyncQdrantClient, transport))
    created = set(names.values())

    async def create_managed(number: int) -> None:
        await client.ensure_collection(CollectionSpec(names[number], 3, "cosine", ()))

    async def scan_from(cursor: str | None) -> tuple[str, ...]:
        scanned: list[str] = []
        while True:
            transport.reset_counts()
            page = await client.list_managed_collections(limit=2, cursor=cursor)
            assert transport.get_collections_calls == 1
            assert transport.get_collection_calls <= 2
            scanned.extend(item.name for item in page.items)
            cursor = page.next_cursor
            if cursor is None:
                return tuple(scanned)

    try:
        for number in (10, 30, 40, 50):
            await create_managed(number)
        await raw.create_collection(
            collection_name=names[20],
            vectors_config=qdrant_models.VectorParams(
                size=3,
                distance=qdrant_models.Distance.COSINE,
            ),
        )

        transport.reset_counts()
        first_page = await client.list_managed_collections(limit=2, cursor=None)
        assert tuple(item.name for item in first_page.items) == (names[10],)
        assert first_page.next_cursor == names[20]
        assert transport.get_collections_calls == 1
        assert transport.get_collection_calls == 2

        await client.delete_collection(names[10])
        await client.delete_collection(names[40])
        for number in (15, 25, 45, 55):
            await create_managed(number)

        current_scan = tuple(item.name for item in first_page.items) + await scan_from(
            first_page.next_cursor
        )
        next_scan = await scan_from(None)
    finally:
        for collection in sorted(created):
            if await raw.collection_exists(collection_name=collection):
                await raw.delete_collection(collection_name=collection)
        await client.aclose()

    assert current_scan == (
        names[10],
        names[25],
        names[30],
        names[45],
        names[50],
        names[55],
    )
    assert next_scan == (
        names[15],
        names[25],
        names[30],
        names[45],
        names[50],
        names[55],
    )


class RepairObjectStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def read_stream(
        self,
        object_key: str,
        *,
        expected_checksum: str,
        max_bytes: int,
    ) -> AsyncIterator[bytes]:
        try:
            content = self.objects[object_key]
        except KeyError:
            raise ObjectStoreError("OBJECT_VERIFICATION_FAILED", retryable=False) from None
        if len(content) > max_bytes or hashlib.sha256(content).hexdigest() != expected_checksum:
            raise ObjectStoreError("OBJECT_VERIFICATION_FAILED", retryable=False)
        for offset in range(0, len(content), 73):
            yield content[offset : offset + 73]


class DeterministicRepairEmbeddingGateway:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    async def embed(
        self,
        *,
        snapshot: EmbeddingConfigSnapshot,
        operational: EmbeddingOperationalConfig,
        inputs: Sequence[str],
        attempt_observer: EmbeddingAttemptObserver | None = None,
    ) -> EmbeddingResult:
        batch = tuple(inputs)
        assert 1 <= len(batch) <= operational.batch_size
        self.calls.append(batch)
        if attempt_observer is not None:
            observed = attempt_observer(
                EmbeddingAttempt(
                    provider_identifier=snapshot.provider_type,
                    model_identifier=snapshot.model_name,
                    route_identifier=None,
                    provider_request_id=f"repair-request-{len(self.calls)}",
                    input_tokens=sum(len(value) for value in batch),
                    output_tokens=0,
                    cost_micros=0,
                    currency="USD",
                    latency_ms=1,
                    status="succeeded",
                    error_code=None,
                    degraded=False,
                )
            )
            if observed is not None:
                await observed
        vectors = tuple(
            tuple(
                float((input_index + dimension_index + 1) % 7) / 7.0
                for dimension_index in range(snapshot.dimension)
            )
            for input_index, _value in enumerate(batch)
        )
        return EmbeddingResult(vectors=vectors, usage={"prompt_tokens": len(batch)})


def _qdrant_vector_digest(vector: object) -> str:
    if (
        type(vector) is not list
        or not vector
        or any(type(value) is not float or not math.isfinite(value) for value in vector)
    ):
        raise AssertionError("Qdrant snapshot vector was invalid")
    return canonical_sha256(vector)


def _qdrant_payload_digest(payload: object) -> str:
    if type(payload) is not dict:
        raise AssertionError("Qdrant snapshot payload was invalid")
    try:
        return canonical_sha256(payload)
    except (TypeError, ValueError, OverflowError):
        raise AssertionError("Qdrant snapshot payload was invalid") from None


def _qdrant_point_snapshot_digest(records: Sequence[object]) -> str:
    normalized: list[dict[str, str]] = []
    for record in records:
        if type(record) is not qdrant_models.Record:
            raise AssertionError("Qdrant point snapshot was invalid")
        try:
            canonical_id = str(UUID(str(record.id)))
        except (TypeError, ValueError, AttributeError):
            raise AssertionError("Qdrant point snapshot identity was invalid") from None
        normalized.append(
            {
                "id": canonical_id,
                "vector_sha256": _qdrant_vector_digest(record.vector),
                "payload_sha256": _qdrant_payload_digest(record.payload),
            }
        )
    normalized.sort(key=lambda item: item["id"])
    return canonical_sha256(normalized)


def _qdrant_search_snapshot_digest(points: Sequence[object]) -> str:
    normalized: list[dict[str, str]] = []
    for point in points:
        if type(point) is not qdrant_models.ScoredPoint:
            raise AssertionError("Qdrant search snapshot was invalid")
        try:
            canonical_id = str(UUID(str(point.id)))
        except (TypeError, ValueError, AttributeError):
            raise AssertionError("Qdrant search snapshot identity was invalid") from None
        if type(point.score) is not float or not math.isfinite(point.score):
            raise AssertionError("Qdrant search snapshot score was invalid")
        normalized.append(
            {
                "id": canonical_id,
                "score": point.score.hex(),
                "vector_sha256": _qdrant_vector_digest(point.vector),
                "payload_sha256": _qdrant_payload_digest(point.payload),
            }
        )
    return canonical_sha256(normalized)


async def test_repair_snapshot_digests_cover_full_vectors_payloads_order_and_scores() -> None:
    first_id = uuid4()
    second_id = uuid4()
    original = qdrant_models.Record(
        id=first_id,
        vector=[0.1, 0.2, 0.3],
        payload={"kind": "original", "ordinal": 1},
    )
    vector_changed = qdrant_models.Record(
        id=first_id,
        vector=[0.1, 0.2, 0.4],
        payload={"kind": "original", "ordinal": 1},
    )
    payload_changed = qdrant_models.Record(
        id=first_id,
        vector=[0.1, 0.2, 0.3],
        payload={"kind": "changed", "ordinal": 1},
    )
    assert _qdrant_point_snapshot_digest((original,)) != _qdrant_point_snapshot_digest(
        (vector_changed,)
    )
    assert _qdrant_point_snapshot_digest((original,)) != _qdrant_point_snapshot_digest(
        (payload_changed,)
    )

    first = qdrant_models.ScoredPoint(
        id=first_id,
        version=1,
        score=0.9,
        vector=[0.1, 0.2, 0.3],
        payload={"kind": "first"},
    )
    second = qdrant_models.ScoredPoint(
        id=second_id,
        version=1,
        score=0.8,
        vector=[0.3, 0.2, 0.1],
        payload={"kind": "second"},
    )
    rescored = qdrant_models.ScoredPoint(
        id=first_id,
        version=1,
        score=0.7,
        vector=[0.1, 0.2, 0.3],
        payload={"kind": "first"},
    )
    assert _qdrant_search_snapshot_digest((first, second)) != _qdrant_search_snapshot_digest(
        (second, first)
    )
    assert _qdrant_search_snapshot_digest((first,)) != _qdrant_search_snapshot_digest((rescored,))


class CapturingRepairUsageSink:
    def __init__(self) -> None:
        self.contexts: list[ProviderUsageContext] = []

    async def record(
        self,
        context: ProviderUsageContext,
        attempt: EmbeddingAttempt,
    ) -> None:
        assert attempt.status == "succeeded"
        self.contexts.append(context)


@dataclass(frozen=True, slots=True)
class SeededManifest:
    document_id: UUID
    version_id: UUID
    chunks: tuple[Chunk, ...]


@asynccontextmanager
async def _job_repository(database: Database) -> AsyncIterator[JobRepository]:
    async with database.sessions() as session, session.begin():
        yield SqlAlchemyJobRepository(session)


def _repair_runner(database: Database, pipeline: Any) -> JobRunner:
    runner = JobRunner(
        repository_context=lambda: _job_repository(database),
        lease_owner="generation-repair-test-worker",
        lease_seconds=30,
        heartbeat_seconds=5,
        poll_interval_seconds=0.01,
        max_concurrency=1,
        backoff=ExponentialBackoff(0.01, 0.01, jitter_fraction=0),
    )
    runner.register("rebuild_generation", pipeline.handle)
    return runner


async def _configured_generation(
    database: Database,
) -> tuple[SeededGeneration, UUID, CollectionSpec]:
    knowledge_base_id = uuid4()
    generation_id = uuid4()
    credential_id = uuid4()
    provider_config_id = uuid4()
    profile_id = uuid4()
    now = datetime.now(UTC)
    semantic: dict[str, object] = {
        "adapter_schema_version": "openai-embeddings-v1",
        "provider_type": "openai_compatible",
        "base_url": "https://provider.example/v1",
        "default_headers": {},
        "routing_options": {},
        "model_name": "text-embedding-test",
        "dimension": 3,
        "distance": "cosine",
        "max_input_tokens": 8192,
        "vector_config": {},
    }
    semantic_hash = canonical_sha256(semantic)
    persisted_snapshot = {
        **semantic,
        "provider_config_id": str(provider_config_id),
        "credential_id": str(credential_id),
    }
    seeded = SeededGeneration(knowledge_base_id, generation_id, semantic_hash)
    async with database.sessions() as session, session.begin():
        session.add_all(
            [
                KnowledgeBase(
                    id=knowledge_base_id,
                    name="Generation repair integration",
                    description=None,
                    status="active",
                    metadata_={},
                    filter_schema={"fields": []},
                    resource_revision=1,
                    mutation_revision=7,
                    filter_schema_revision=0,
                    active_index_generation_id=generation_id,
                    pending_index_generation_id=None,
                ),
                ProviderCredential(
                    id=credential_id,
                    name=f"Repair credential {credential_id.hex[:8]}",
                    ciphertext=b"ciphertext",
                    nonce=b"n" * 12,
                    algorithm="AES-256-GCM",
                    key_version="test-v1",
                    resource_revision=1,
                ),
            ]
        )
        await session.flush()
        session.add(
            ProviderConfig(
                id=provider_config_id,
                name=f"Repair provider {provider_config_id.hex[:8]}",
                provider_type="openai_compatible",
                base_url="https://provider.example/v1",
                credential_id=credential_id,
                secret_ref=None,
                default_headers={},
                routing_options={},
                timeout_seconds=Decimal("10.000"),
                max_concurrency=2,
                requests_per_minute=60,
                enabled=True,
                resource_revision=1,
                endpoint_policy_version="provider-endpoint-v1",
                endpoint_validated_at=now,
            )
        )
        await session.flush()
        session.add(
            ModelProfile(
                id=profile_id,
                name=f"Repair profile {profile_id.hex[:8]}",
                capability="embedding",
                provider_config_id=provider_config_id,
                model_name=cast(str, semantic["model_name"]),
                dimension=cast(int, semantic["dimension"]),
                max_input_tokens=cast(int, semantic["max_input_tokens"]),
                batch_size=2,
                timeout_seconds=Decimal("8.000"),
                vector_config={},
                enabled=True,
                resource_revision=1,
            )
        )
        await session.flush()
        session.add(
            KnowledgeBaseIndexGeneration(
                id=generation_id,
                knowledge_base_id=knowledge_base_id,
                embedding_profile_id=profile_id,
                sparse_profile_id=None,
                index_profile_hash=semantic_hash,
                qdrant_collection_name=collection_name(knowledge_base_id, generation_id),
                status="active",
                rebuild_snapshot_at=now,
                caught_up_revision=7,
                validated_revision=7,
                validation_manifest_hash="a" * 64,
                expected_point_count=0,
                actual_point_count=0,
                validated_at=now,
                activated_at=now,
                retired_at=None,
                distance="cosine",
                embedding_config_snapshot=persisted_snapshot,
                filter_schema_snapshot={"fields": []},
                applied_filter_schema_revision=0,
                embedding_config_hash=semantic_hash,
                safe_error_code=None,
                safe_error_message=None,
            )
        )
        await session.flush()
        session.add(
            KnowledgeBaseMutation(
                id=uuid4(),
                knowledge_base_id=knowledge_base_id,
                revision=7,
                mutation_type="index_config_changed",
                target_type="index_generation",
                target_id=generation_id,
                payload={
                    "generation_id": str(generation_id),
                    "embedding_profile_id": str(profile_id),
                    "index_profile_hash": semantic_hash,
                    "embedding_config_hash": semantic_hash,
                    "applied_filter_schema_revision": 0,
                    "validation_manifest_hash": "a" * 64,
                },
            )
        )
    spec = CollectionSpec(
        collection_name(knowledge_base_id, generation_id),
        3,
        "cosine",
        payload_indexes_for_filter_snapshot({"fields": []}),
    )
    return seeded, profile_id, spec


async def _seed_ready_document(
    database: Database,
    seeded: SeededGeneration,
    *,
    manifest_complete: bool,
) -> tuple[UUID, UUID]:
    document_id = uuid4()
    version_id = uuid4()
    checksum = "1" * 64
    manifest_checksum = "2" * 64 if manifest_complete else None
    manifest_key = f"artifacts/{version_id}/chunks.jsonl" if manifest_complete else None
    chunker_config = {
        "max_chunk_codepoints": 1200,
        "target_overlap_codepoints": 150,
    }
    chunk_config_hash = (
        canonical_sha256(
            {
                "config": chunker_config,
                "name": "recursive_text_v1",
                "version": "1",
            }
        )
        if manifest_complete
        else None
    )
    now = datetime.now(UTC)
    document = Document(
        id=document_id,
        knowledge_base_id=seeded.knowledge_base_id,
        display_name="Repair document",
        mime_type="text/plain",
        checksum_sha256=checksum,
        current_version_id=version_id,
        pending_version_id=None,
        status="active",
        tags=[],
        metadata_={},
        deleted_at=None,
    )
    version = DocumentVersion(
        id=version_id,
        document_id=document_id,
        version_number=1,
        source_object_key=f"sources/{version_id}",
        parsed_object_key=f"artifacts/{version_id}/parsed.txt",
        source_checksum_sha256=checksum,
        parsed_object_checksum_sha256="4" * 64,
        declared_mime_type="text/plain",
        detected_mime_type="text/plain",
        source_extension="txt",
        base_version_id=None,
        parser_name="plain_text_v1",
        parser_version="1",
        parser_config={},
        chunker_name="recursive_text_v1",
        chunker_version="1",
        chunker_config=chunker_config,
        chunk_count=1,
        status="ready",
        activated_at=now,
        chunk_manifest_object_key=manifest_key,
        chunk_manifest_checksum_sha256=manifest_checksum,
        chunk_config_hash=chunk_config_hash,
    )
    state = DocumentIndexState(
        document_version_id=version_id,
        index_generation_id=seeded.generation_id,
        status="validated",
        expected_point_count=1,
        actual_point_count=1,
        error_code=None,
        validated_at=now,
        chunk_manifest_checksum_sha256=manifest_checksum,
        embedding_config_hash=seeded.embedding_config_hash,
        next_chunk_index=1,
        safe_error_message=None,
    )
    async with database.sessions() as session, session.begin():
        session.add(document)
        await session.flush()
        session.add(version)
        await session.flush()
        session.add(state)
    return document_id, version_id


async def _seed_ready_manifest_document(
    database: Database,
    seeded: SeededGeneration,
    store: RepairObjectStore,
    *,
    text: str,
    bump_revision: bool = False,
) -> SeededManifest:
    document_id = uuid4()
    version_id = uuid4()
    created_at = datetime.now(UTC)
    source_checksum = hashlib.sha256(f"source:{text}".encode()).hexdigest()
    parsed_checksum = hashlib.sha256(text.encode()).hexdigest()
    parsed = ParsedArtifact(text, (), "plain_text_v1", "1", {})
    chunker = RecursiveTextChunker(
        max_chunk_codepoints=24,
        target_overlap_codepoints=4,
    )
    chunks = tuple(chunker.chunk(parsed))
    assert chunks
    manifest = chunk_manifest_header_bytes(
        source_checksum_sha256=source_checksum,
        parsed=parsed,
        chunker=chunker,
        document_version_created_at=created_at,
        chunk_count=len(chunks),
    ) + b"".join(chunk_manifest_record_bytes(chunk) for chunk in chunks)
    manifest_key = chunks_object_key(seeded.knowledge_base_id, document_id, version_id)
    manifest_checksum = hashlib.sha256(manifest).hexdigest()
    store.objects[manifest_key] = manifest

    async with database.sessions() as session, session.begin():
        knowledge_base = await session.get(KnowledgeBase, seeded.knowledge_base_id)
        assert knowledge_base is not None
        if bump_revision:
            knowledge_base.mutation_revision += 1
        session.add(
            Document(
                id=document_id,
                knowledge_base_id=seeded.knowledge_base_id,
                display_name=f"Repair manifest {document_id.hex[:8]}",
                mime_type="text/plain",
                checksum_sha256=source_checksum,
                current_version_id=version_id,
                pending_version_id=None,
                status="active",
                tags=[],
                metadata_={"source": "repair-test"},
                deleted_at=None,
            )
        )
        await session.flush()
        session.add(
            DocumentVersion(
                id=version_id,
                document_id=document_id,
                version_number=1,
                source_object_key=f"sources/{version_id}",
                parsed_object_key=f"artifacts/{version_id}/parsed.txt",
                source_checksum_sha256=source_checksum,
                parsed_object_checksum_sha256=parsed_checksum,
                declared_mime_type="text/plain",
                detected_mime_type="text/plain",
                source_extension="txt",
                base_version_id=None,
                parser_name=parsed.parser_name,
                parser_version=parsed.parser_version,
                parser_config=dict(parsed.parser_config),
                chunker_name=chunker.name,
                chunker_version=chunker.version,
                chunker_config=dict(chunker.config),
                chunk_count=len(chunks),
                status="ready",
                activated_at=created_at,
                chunk_manifest_object_key=manifest_key,
                chunk_manifest_checksum_sha256=manifest_checksum,
                chunk_config_hash=chunker.config_hash,
                created_at=created_at,
            )
        )
        await session.flush()
        session.add(
            DocumentIndexState(
                document_version_id=version_id,
                index_generation_id=seeded.generation_id,
                status="validated",
                expected_point_count=len(chunks),
                actual_point_count=len(chunks),
                error_code=None,
                validated_at=created_at,
                chunk_manifest_checksum_sha256=manifest_checksum,
                embedding_config_hash=seeded.embedding_config_hash,
                next_chunk_index=len(chunks),
                safe_error_message=None,
            )
        )
    return SeededManifest(document_id, version_id, chunks)


async def _repair_jobs(database: Database) -> tuple[Job, ...]:
    async with database.sessions() as session:
        return tuple(
            (
                await session.scalars(
                    select(Job)
                    .where(Job.operation == "rebuild_generation")
                    .order_by(Job.created_at, Job.id)
                )
            ).all()
        )


async def _visible_generation_facts(
    database: Database,
    seeded: SeededGeneration,
) -> tuple[object, ...]:
    async with database.sessions() as session:
        knowledge_base = await session.get(KnowledgeBase, seeded.knowledge_base_id)
        generation = await session.get(KnowledgeBaseIndexGeneration, seeded.generation_id)
        assert knowledge_base is not None and generation is not None
        profile = await session.get(ModelProfile, generation.embedding_profile_id)
        assert profile is not None
        documents = tuple(
            (
                row.id,
                row.status,
                row.current_version_id,
                row.pending_version_id,
                row.deleted_at,
            )
            for row in (
                await session.scalars(
                    select(Document)
                    .where(Document.knowledge_base_id == seeded.knowledge_base_id)
                    .order_by(Document.id)
                )
            ).all()
        )
        versions = tuple(
            (row.id, row.status, row.activated_at, row.chunk_count)
            for row in (
                await session.scalars(
                    select(DocumentVersion)
                    .join(Document, Document.id == DocumentVersion.document_id)
                    .where(Document.knowledge_base_id == seeded.knowledge_base_id)
                    .order_by(DocumentVersion.id)
                )
            ).all()
        )
        states = tuple(
            (
                row.document_version_id,
                row.status,
                row.expected_point_count,
                row.actual_point_count,
                row.next_chunk_index,
                row.validated_at,
            )
            for row in (
                await session.scalars(
                    select(DocumentIndexState)
                    .where(DocumentIndexState.index_generation_id == seeded.generation_id)
                    .order_by(DocumentIndexState.document_version_id)
                )
            ).all()
        )
        return (
            knowledge_base.status,
            knowledge_base.resource_revision,
            knowledge_base.mutation_revision,
            knowledge_base.active_index_generation_id,
            knowledge_base.pending_index_generation_id,
            generation.status,
            generation.caught_up_revision,
            generation.validated_revision,
            generation.validation_manifest_hash,
            generation.expected_point_count,
            generation.actual_point_count,
            generation.validated_at,
            generation.activated_at,
            profile.id,
            profile.provider_config_id,
            profile.model_name,
            profile.dimension,
            profile.vector_config,
            profile.resource_revision,
            documents,
            versions,
            states,
        )


def _make_repair_pipeline(
    database: Database,
    store: RepairObjectStore,
    qdrant: QdrantClient,
    *,
    hooks: GenerationRepairHooks | None = None,
) -> tuple[GenerationRepairPipeline, DeterministicRepairEmbeddingGateway, CapturingRepairUsageSink]:
    gateway = DeterministicRepairEmbeddingGateway()
    usage = CapturingRepairUsageSink()
    pipeline = GenerationRepairPipeline(
        session_factory=database.sessions,
        object_store=cast(Any, store),
        embedding_gateway=gateway,
        qdrant=qdrant,
        provider_usage_sink=usage,
        hooks=hooks,
    )
    return pipeline, gateway, usage


async def _make_job_due(database: Database, job_id: UUID) -> None:
    async with database.sessions() as session, session.begin():
        job = await session.get(Job, job_id)
        database_now = await session.scalar(select(func.clock_timestamp()))
        assert job is not None and isinstance(database_now, datetime)
        job.next_retry_at = database_now - timedelta(seconds=1)


async def test_repair_generation_reserves_a_missing_active_collection_without_mutating_facts(
    migrated_database: Database,
) -> None:
    seeded, _profile_id, spec = await _configured_generation(migrated_database)
    qdrant = FakeQdrantClient()

    async with migrated_database.sessions() as session:
        before_kb = await session.get(KnowledgeBase, seeded.knowledge_base_id)
        assert before_kb is not None
        before_revision = before_kb.mutation_revision
        before_pointer = before_kb.active_index_generation_id

    result = await _repair_service(migrated_database, qdrant).reserve(seeded.generation_id)

    assert result == {
        "generation_id": str(seeded.generation_id),
        "job_id": result["job_id"],
        "status": "queued",
    }
    assert UUID(result["job_id"])
    jobs = await _repair_jobs(migrated_database)
    assert len(jobs) == 1
    assert jobs[0].target_type == "index_generation"
    assert jobs[0].target_id == seeded.generation_id
    assert jobs[0].index_generation_id == seeded.generation_id
    assert jobs[0].target_revision == before_revision
    assert jobs[0].mutation_id is not None
    assert jobs[0].actor_api_key_id is None
    assert jobs[0].stage == "indexing"
    async with migrated_database.sessions() as session:
        after_kb = await session.get(KnowledgeBase, seeded.knowledge_base_id)
        assert after_kb is not None
        assert after_kb.mutation_revision == before_revision
        assert after_kb.active_index_generation_id == before_pointer


async def test_repair_generation_allows_only_a_fully_compatible_empty_collection(
    migrated_database: Database,
) -> None:
    seeded, _profile_id, spec = await _configured_generation(migrated_database)
    qdrant = FakeQdrantClient()
    await qdrant.seed_collection(spec, created_at=datetime.now(UTC))

    result = await _repair_service(migrated_database, qdrant).reserve(seeded.generation_id)

    assert result["status"] == "queued"


async def test_repair_generation_rejects_an_incompatible_empty_collection(
    migrated_database: Database,
) -> None:
    seeded, _profile_id, spec = await _configured_generation(migrated_database)
    qdrant = FakeQdrantClient()
    await qdrant.seed_collection(
        CollectionSpec(spec.name, spec.dimension + 1, spec.distance, spec.payload_indexes),
        created_at=datetime.now(UTC),
    )

    with pytest.raises(BusinessError) as raised:
        await _repair_service(migrated_database, qdrant).reserve(seeded.generation_id)

    assert raised.value.code == "GENERATION_REPAIR_COLLECTION_CONFLICT"
    assert await _repair_jobs(migrated_database) == ()


async def test_repair_generation_rejects_a_preexisting_nonempty_collection(
    migrated_database: Database,
) -> None:
    seeded, _profile_id, spec = await _configured_generation(migrated_database)
    qdrant = FakeQdrantClient()
    await qdrant.seed_collection(spec, created_at=datetime.now(UTC))
    version_id = uuid4()
    chunk_hash = "5" * 64
    await qdrant.upsert_points(
        spec.name,
        (
            QdrantPoint(
                id=point_id(version_id, 0, chunk_hash),
                vector=(0.1, 0.2, 0.3),
                payload={
                    "knowledge_base_id": str(seeded.knowledge_base_id),
                    "document_id": str(uuid4()),
                    "version_id": str(version_id),
                    "chunk_index": 0,
                    "chunk_hash": chunk_hash,
                    "text": "x",
                    "title_path": [],
                    "start_offset": 0,
                    "end_offset": 1,
                    "metadata": {},
                },
            ),
        ),
    )

    with pytest.raises(BusinessError) as raised:
        await _repair_service(migrated_database, qdrant).reserve(seeded.generation_id)

    assert raised.value.code == "GENERATION_REPAIR_COLLECTION_NOT_EMPTY"
    assert await _repair_jobs(migrated_database) == ()


async def test_repair_generation_rejects_missing_canonical_manifest(
    migrated_database: Database,
) -> None:
    seeded, _profile_id, spec = await _configured_generation(migrated_database)
    await _seed_ready_document(migrated_database, seeded, manifest_complete=False)

    with pytest.raises(BusinessError) as raised:
        await _repair_service(migrated_database, FakeQdrantClient()).reserve(seeded.generation_id)

    assert raised.value.code == "GENERATION_REPAIR_MANIFEST_UNAVAILABLE"
    assert await _repair_jobs(migrated_database) == ()


async def test_repair_generation_rejects_profile_snapshot_drift(
    migrated_database: Database,
) -> None:
    seeded, profile_id, _spec = await _configured_generation(migrated_database)
    async with migrated_database.sessions() as session, session.begin():
        profile = await session.get(ModelProfile, profile_id)
        assert profile is not None
        profile.model_name = "drifted-model"
        profile.resource_revision += 1

    with pytest.raises(BusinessError) as raised:
        await _repair_service(migrated_database, FakeQdrantClient()).reserve(seeded.generation_id)

    assert raised.value.code == "GENERATION_REPAIR_CONFIGURATION_CONFLICT"
    assert await _repair_jobs(migrated_database) == ()


async def test_active_ingestion_blocks_generation_repair_reservation(
    migrated_database: Database,
) -> None:
    seeded, _profile_id, _spec = await _configured_generation(migrated_database)
    async with migrated_database.sessions() as session, session.begin():
        session.add(
            Job(
                id=uuid4(),
                knowledge_base_id=seeded.knowledge_base_id,
                actor_api_key_id=None,
                target_type="document_version",
                target_id=uuid4(),
                target_revision=7,
                index_generation_id=seeded.generation_id,
                mutation_id=None,
                parent_job_id=None,
                root_job_id=None,
                idempotency_key=None,
                operation="ingest_document",
                stage="parse",
                status="queued",
            )
        )

    with pytest.raises(BusinessError) as raised:
        await _repair_service(migrated_database, FakeQdrantClient()).reserve(seeded.generation_id)

    assert raised.value.code == "GENERATION_REPAIR_INGESTION_IN_PROGRESS"
    assert raised.value.retryable is True
    assert await _repair_jobs(migrated_database) == ()


async def test_repair_reservation_rechecks_collection_after_ingestion_turns_terminal(
    migrated_database: Database,
) -> None:
    seeded, _profile_id, spec = await _configured_generation(migrated_database)
    ingestion_job_id = uuid4()
    polluted_version_id = uuid4()
    polluted_chunk_hash = "6" * 64
    polluted_text = "failed ingestion pollution"

    class RacingQdrant(FakeQdrantClient):
        count_calls = 0

        async def count_points(self, collection: str) -> int:
            count = await super().count_points(collection)
            self.count_calls += 1
            if self.count_calls == 1:
                await self.upsert_points(
                    collection,
                    (
                        QdrantPoint(
                            id=point_id(polluted_version_id, 0, polluted_chunk_hash),
                            vector=(0.1, 0.2, 0.3),
                            payload={
                                "knowledge_base_id": str(seeded.knowledge_base_id),
                                "document_id": str(uuid4()),
                                "version_id": str(polluted_version_id),
                                "chunk_index": 0,
                                "chunk_hash": polluted_chunk_hash,
                                "text": polluted_text,
                                "title_path": [],
                                "start_offset": 0,
                                "end_offset": len(polluted_text),
                                "metadata": {},
                            },
                        ),
                    ),
                )
                async with migrated_database.sessions() as session, session.begin():
                    job = await session.get(Job, ingestion_job_id)
                    assert job is not None
                    job.status = "failed"
                    job.retryable = False
            return count

    qdrant = RacingQdrant()
    await qdrant.seed_collection(spec, created_at=datetime.now(UTC))
    async with migrated_database.sessions() as session, session.begin():
        session.add(
            Job(
                id=ingestion_job_id,
                knowledge_base_id=seeded.knowledge_base_id,
                actor_api_key_id=None,
                target_type="document_version",
                target_id=polluted_version_id,
                target_revision=7,
                index_generation_id=seeded.generation_id,
                mutation_id=None,
                parent_job_id=None,
                root_job_id=None,
                idempotency_key=None,
                operation="ingest_document",
                stage="embed_index",
                status="queued",
            )
        )

    with pytest.raises(BusinessError) as raised:
        await _repair_service(migrated_database, qdrant).reserve(seeded.generation_id)

    assert raised.value.code == "GENERATION_REPAIR_COLLECTION_NOT_EMPTY"
    assert qdrant.count_calls == 2
    assert await _repair_jobs(migrated_database) == ()


@pytest.mark.parametrize("invalid_state", ["generation_not_active", "pointer_mismatch"])
async def test_repair_generation_requires_the_active_generation_pointer(
    migrated_database: Database,
    invalid_state: str,
) -> None:
    seeded, _profile_id, _spec = await _configured_generation(migrated_database)
    async with migrated_database.sessions() as session, session.begin():
        if invalid_state == "generation_not_active":
            generation = await session.get(KnowledgeBaseIndexGeneration, seeded.generation_id)
            assert generation is not None
            generation.status = "failed"
        else:
            knowledge_base = await session.get(KnowledgeBase, seeded.knowledge_base_id)
            assert knowledge_base is not None
            knowledge_base.active_index_generation_id = None

    with pytest.raises(BusinessError) as raised:
        await _repair_service(migrated_database, FakeQdrantClient()).reserve(seeded.generation_id)

    assert raised.value.code == "GENERATION_REPAIR_NOT_ACTIVE"
    assert await _repair_jobs(migrated_database) == ()


async def test_repair_generation_concurrent_reservations_and_revision_changes_keep_one_job(
    migrated_database: Database,
) -> None:
    seeded, _profile_id, _spec = await _configured_generation(migrated_database)
    service = _repair_service(migrated_database, FakeQdrantClient())

    first, second = await asyncio.gather(
        service.reserve(seeded.generation_id),
        service.reserve(seeded.generation_id),
        return_exceptions=True,
    )

    outcomes = (first, second)
    assert sum(type(outcome) is dict for outcome in outcomes) == 1
    conflicts = [outcome for outcome in outcomes if isinstance(outcome, BusinessError)]
    assert len(conflicts) == 1
    assert conflicts[0].code == "GENERATION_REPAIR_IN_PROGRESS"

    async with migrated_database.sessions() as session, session.begin():
        knowledge_base = await session.get(KnowledgeBase, seeded.knowledge_base_id)
        assert knowledge_base is not None
        knowledge_base.mutation_revision += 1

    with pytest.raises(BusinessError) as raised:
        await service.reserve(seeded.generation_id)
    assert raised.value.code == "GENERATION_REPAIR_IN_PROGRESS"
    assert len(await _repair_jobs(migrated_database)) == 1


@pytest.mark.parametrize("drift", ["equivalent_profile", "applied_filter_revision"])
async def test_repair_reservation_fingerprints_configuration_identities_across_probe_race(
    migrated_database: Database,
    drift: str,
) -> None:
    seeded, profile_id, _spec = await _configured_generation(migrated_database)
    drifted = False

    class ConfigurationDriftQdrant(FakeQdrantClient):
        async def collection_exists(self, collection: str) -> bool:
            nonlocal drifted
            if not drifted:
                drifted = True
                async with migrated_database.sessions() as session, session.begin():
                    generation = await session.get(
                        KnowledgeBaseIndexGeneration,
                        seeded.generation_id,
                    )
                    assert generation is not None
                    if drift == "applied_filter_revision":
                        generation.applied_filter_schema_revision = 1
                    else:
                        profile = await session.get(ModelProfile, profile_id)
                        assert profile is not None
                        replacement_id = uuid4()
                        session.add(
                            ModelProfile(
                                id=replacement_id,
                                name=f"Equivalent repair profile {replacement_id.hex[:8]}",
                                capability=profile.capability,
                                provider_config_id=profile.provider_config_id,
                                model_name=profile.model_name,
                                dimension=profile.dimension,
                                max_input_tokens=profile.max_input_tokens,
                                batch_size=profile.batch_size,
                                timeout_seconds=profile.timeout_seconds,
                                vector_config=profile.vector_config,
                                enabled=profile.enabled,
                                resource_revision=1,
                            )
                        )
                        await session.flush()
                        generation.embedding_profile_id = replacement_id
            return await super().collection_exists(collection)

    with pytest.raises(BusinessError) as raised:
        await _repair_service(migrated_database, ConfigurationDriftQdrant()).reserve(
            seeded.generation_id
        )

    assert raised.value.code == "GENERATION_REPAIR_CONFIGURATION_CONFLICT"
    assert raised.value.retryable is False
    assert await _repair_jobs(migrated_database) == ()


@pytest.mark.parametrize("drift", ["equivalent_profile", "applied_filter_revision"])
async def test_repair_worker_rejects_configuration_identity_drift_after_reservation(
    migrated_database: Database,
    drift: str,
) -> None:
    seeded, profile_id, _spec = await _configured_generation(migrated_database)
    qdrant = FakeQdrantClient()
    reservation = await _repair_service(migrated_database, qdrant).reserve(seeded.generation_id)

    async with migrated_database.sessions() as session, session.begin():
        generation = await session.get(
            KnowledgeBaseIndexGeneration,
            seeded.generation_id,
        )
        assert generation is not None
        if drift == "applied_filter_revision":
            generation.applied_filter_schema_revision = 1
        else:
            profile = await session.get(ModelProfile, profile_id)
            assert profile is not None
            replacement_id = uuid4()
            session.add(
                ModelProfile(
                    id=replacement_id,
                    name=f"Post-reservation profile {replacement_id.hex[:8]}",
                    capability=profile.capability,
                    provider_config_id=profile.provider_config_id,
                    model_name=profile.model_name,
                    dimension=profile.dimension,
                    max_input_tokens=profile.max_input_tokens,
                    batch_size=profile.batch_size,
                    timeout_seconds=profile.timeout_seconds,
                    vector_config=profile.vector_config,
                    enabled=profile.enabled,
                    resource_revision=1,
                )
            )
            await session.flush()
            generation.embedding_profile_id = replacement_id

    pipeline, gateway, _usage = _make_repair_pipeline(
        migrated_database,
        RepairObjectStore(),
        qdrant,
    )
    runner = _repair_runner(migrated_database, pipeline)

    assert await runner.run_once(UUID(reservation["job_id"]))
    job = (await _repair_jobs(migrated_database))[0]
    assert job.status == "failed"
    assert job.error_code == "GENERATION_REPAIR_CONFIGURATION_CONFLICT"
    assert gateway.calls == []


@pytest.mark.acceptance
async def test_real_qdrant_reconstructs_multiple_documents_without_mutating_visible_facts(
    migrated_database: Database,
    repair_qdrant_url: str,
) -> None:
    seeded, _profile_id, spec = await _configured_generation(migrated_database)
    store = RepairObjectStore()
    manifests = (
        await _seed_ready_manifest_document(
            migrated_database,
            seeded,
            store,
            text="alpha document has enough material for several deterministic chunks. " * 2,
        ),
        await _seed_ready_manifest_document(
            migrated_database,
            seeded,
            store,
            text="beta document is reconstructed from its canonical manifest. " * 2,
        ),
    )
    before = await _visible_generation_facts(migrated_database, seeded)
    qdrant = qdrant_client_from_url(repair_qdrant_url, timeout_seconds=5)
    raw_qdrant = AsyncQdrantClient(url=repair_qdrant_url, timeout=5)
    try:
        await qdrant.ensure_collection(spec.vector_only())
        await qdrant.ensure_payload_indexes(spec.name, spec.payload_indexes)
        await qdrant.verify_collection(spec)
        assert await qdrant.collection_exists(spec.name)

        canonical_points: list[QdrantPoint] = []
        for manifest in manifests:
            for batch_start in range(0, len(manifest.chunks), 2):
                batch = manifest.chunks[batch_start : batch_start + 2]
                for input_index, chunk in enumerate(batch):
                    vector = tuple(
                        float((input_index + dimension_index + 1) % 7) / 7.0
                        for dimension_index in range(spec.dimension)
                    )
                    canonical_points.append(
                        QdrantPoint(
                            id=point_id(
                                manifest.version_id,
                                chunk.chunk_index,
                                chunk.chunk_hash,
                            ),
                            vector=vector,
                            payload={
                                "knowledge_base_id": str(seeded.knowledge_base_id),
                                "document_id": str(manifest.document_id),
                                "version_id": str(manifest.version_id),
                                "chunk_index": chunk.chunk_index,
                                "chunk_hash": chunk.chunk_hash,
                                "text": chunk.text,
                                "title_path": list(chunk.title_path),
                                "start_offset": chunk.start_offset,
                                "end_offset": chunk.end_offset,
                                "metadata": {},
                            },
                        )
                    )
        expected_total = sum(len(manifest.chunks) for manifest in manifests)
        if len(canonical_points) != expected_total:
            raise AssertionError("Canonical Qdrant baseline point count was invalid")
        await qdrant.upsert_points(spec.name, tuple(canonical_points))
        if await qdrant.count_points(spec.name) != expected_total:
            raise AssertionError("Canonical Qdrant baseline was incomplete")

        expected_ids = tuple(point.id for point in canonical_points)
        before_points = await raw_qdrant.retrieve(
            collection_name=spec.name,
            ids=[str(point_id_value) for point_id_value in expected_ids],
            with_payload=True,
            with_vectors=True,
        )
        if len(before_points) != expected_total:
            raise AssertionError("Canonical Qdrant baseline retrieval was incomplete")
        before_point_digest = _qdrant_point_snapshot_digest(before_points)
        query_filter = qdrant_models.Filter(
            must=(
                qdrant_models.FieldCondition(
                    key="version_id",
                    match=qdrant_models.MatchValue(value=str(manifests[0].version_id)),
                ),
                qdrant_models.FieldCondition(
                    key="chunk_index",
                    match=qdrant_models.MatchAny(any=(0, 1)),
                ),
            )
        )
        before_query = await raw_qdrant.query_points(
            collection_name=spec.name,
            query=[1.0 / 7.0, 2.0 / 7.0, 3.0 / 7.0],
            query_filter=query_filter,
            limit=2,
            with_payload=True,
            with_vectors=True,
        )
        if len(before_query.points) != 2:
            raise AssertionError("Canonical Qdrant baseline search was incomplete")
        before_search_digest = _qdrant_search_snapshot_digest(before_query.points)

        await qdrant.delete_collection(spec.name)
        assert not await qdrant.collection_exists(spec.name)
        reservation = await _repair_service(migrated_database, qdrant).reserve(seeded.generation_id)
        pipeline, gateway, usage = _make_repair_pipeline(
            migrated_database,
            store,
            qdrant,
        )

        assert await _repair_runner(migrated_database, pipeline).run_once(
            UUID(reservation["job_id"])
        )

        jobs = await _repair_jobs(migrated_database)
        assert len(jobs) == 1
        assert jobs[0].status == "succeeded"
        assert jobs[0].stage == "complete"
        assert jobs[0].progress_current == expected_total
        assert jobs[0].progress_total == expected_total
        assert (
            await qdrant.count_points(
                collection_name(seeded.knowledge_base_id, seeded.generation_id)
            )
            == expected_total
        )
        for manifest in manifests:
            assert await qdrant.count_version_points(
                collection_name(seeded.knowledge_base_id, seeded.generation_id),
                manifest.version_id,
            ) == len(manifest.chunks)
            manifest_ids = tuple(
                point_id(manifest.version_id, chunk.chunk_index, chunk.chunk_hash)
                for chunk in manifest.chunks
            )
            inspected_points = await qdrant.retrieve_version_points(
                collection_name(seeded.knowledge_base_id, seeded.generation_id),
                manifest.version_id,
                manifest_ids,
            )
            for chunk, inspected in zip(manifest.chunks, inspected_points, strict=True):
                expected_payload = {
                    "knowledge_base_id": str(seeded.knowledge_base_id),
                    "document_id": str(manifest.document_id),
                    "version_id": str(manifest.version_id),
                    "chunk_index": chunk.chunk_index,
                    "chunk_hash": chunk.chunk_hash,
                    "text": chunk.text,
                    "title_path": list(chunk.title_path),
                    "start_offset": chunk.start_offset,
                    "end_offset": chunk.end_offset,
                    "metadata": {},
                }
                assert inspected.id == point_id(
                    manifest.version_id,
                    chunk.chunk_index,
                    chunk.chunk_hash,
                )
                assert inspected.vector_dimension == 3
                assert inspected.payload_digest_sha256 == canonical_sha256(expected_payload)

        after_points = await raw_qdrant.retrieve(
            collection_name=spec.name,
            ids=[str(point_id_value) for point_id_value in expected_ids],
            with_payload=True,
            with_vectors=True,
        )
        if len(after_points) != expected_total:
            raise AssertionError("Repaired Qdrant baseline retrieval was incomplete")
        assert _qdrant_point_snapshot_digest(after_points) == before_point_digest

        after_query = await raw_qdrant.query_points(
            collection_name=spec.name,
            query=[1.0 / 7.0, 2.0 / 7.0, 3.0 / 7.0],
            query_filter=query_filter,
            limit=2,
            with_payload=True,
            with_vectors=True,
        )
        if len(after_query.points) != 2:
            raise AssertionError("Repaired Qdrant baseline search was incomplete")
        assert _qdrant_search_snapshot_digest(after_query.points) == before_search_digest
        assert gateway.calls
        assert usage.contexts
        assert all(context.actor_api_key_id is None for context in usage.contexts)
        assert await _visible_generation_facts(migrated_database, seeded) == before
    finally:
        await raw_qdrant.close()
        await qdrant.aclose()


async def test_real_qdrant_retry_resumes_a_repair_owned_partial_collection(
    migrated_database: Database,
    repair_qdrant_url: str,
) -> None:
    seeded, _profile_id, _spec = await _configured_generation(migrated_database)
    store = RepairObjectStore()
    manifest = await _seed_ready_manifest_document(
        migrated_database,
        seeded,
        store,
        text="crash recovery must deterministically overwrite the first batch. " * 3,
    )
    crashed = False

    async def checkpoint(name: str) -> None:
        nonlocal crashed
        if name == "indexing.after_upsert" and not crashed:
            crashed = True
            raise RetryableJobError("SIMULATED_REPAIR_CRASH", "Simulated repair crash")

    qdrant = qdrant_client_from_url(repair_qdrant_url, timeout_seconds=5)
    try:
        reservation = await _repair_service(migrated_database, qdrant).reserve(seeded.generation_id)
        pipeline, _gateway, _usage = _make_repair_pipeline(
            migrated_database,
            store,
            qdrant,
            hooks=GenerationRepairHooks(checkpoint),
        )
        runner = _repair_runner(migrated_database, pipeline)
        job_id = UUID(reservation["job_id"])

        assert await runner.run_once(job_id)
        jobs = await _repair_jobs(migrated_database)
        assert jobs[0].status == "retry_wait"
        assert jobs[0].progress_current == 0
        assert (
            await qdrant.count_points(
                collection_name(seeded.knowledge_base_id, seeded.generation_id)
            )
            > 0
        )
        await _make_job_due(migrated_database, job_id)

        assert await runner.run_once(job_id)
        jobs = await _repair_jobs(migrated_database)
        assert jobs[0].status == "succeeded"
        assert jobs[0].progress_current == len(manifest.chunks)
        assert await qdrant.count_points(
            collection_name(seeded.knowledge_base_id, seeded.generation_id)
        ) == len(manifest.chunks)
    finally:
        await qdrant.aclose()


async def test_repair_batch_checkpoints_do_not_reload_every_canonical_target(
    migrated_database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeded, _profile_id, _spec = await _configured_generation(migrated_database)
    store = RepairObjectStore()
    manifest = await _seed_ready_manifest_document(
        migrated_database,
        seeded,
        store,
        text="bounded repair checkpoint work across many batches. " * 24,
    )
    qdrant = FakeQdrantClient()
    reservation = await _repair_service(migrated_database, qdrant).reserve(seeded.generation_id)
    pipeline, gateway, _usage = _make_repair_pipeline(
        migrated_database,
        store,
        qdrant,
    )
    canonical_target_reads = 0
    original = pipeline._state._canonical_targets

    async def counted_canonical_targets(
        session: AsyncSession,
        knowledge_base_id: UUID,
        generation: KnowledgeBaseIndexGeneration,
    ) -> tuple[tuple[Any, ...], int]:
        nonlocal canonical_target_reads
        canonical_target_reads += 1
        return await original(session, knowledge_base_id, generation)

    monkeypatch.setattr(pipeline._state, "_canonical_targets", counted_canonical_targets)
    runner = _repair_runner(migrated_database, pipeline)

    assert len(manifest.chunks) >= 10
    assert await runner.run_once(UUID(reservation["job_id"]))
    assert (await _repair_jobs(migrated_database))[0].status == "succeeded"
    assert len(gateway.calls) >= 5
    assert canonical_target_reads <= 5


@pytest.mark.parametrize("drift_checkpoint", ["validating.after_qdrant", "complete.after_qdrant"])
async def test_late_stage_revision_drift_resets_repair_to_indexing(
    migrated_database: Database,
    repair_qdrant_url: str,
    drift_checkpoint: str,
) -> None:
    seeded, _profile_id, _spec = await _configured_generation(migrated_database)
    store = RepairObjectStore()
    manifest = await _seed_ready_manifest_document(
        migrated_database,
        seeded,
        store,
        text=f"revision drift at {drift_checkpoint}",
    )
    drifted = False

    async def checkpoint(name: str) -> None:
        nonlocal drifted
        if name == drift_checkpoint and not drifted:
            drifted = True
            async with migrated_database.sessions() as session, session.begin():
                knowledge_base = await session.get(KnowledgeBase, seeded.knowledge_base_id)
                assert knowledge_base is not None
                knowledge_base.mutation_revision += 1

    qdrant = qdrant_client_from_url(repair_qdrant_url, timeout_seconds=5)
    try:
        reservation = await _repair_service(migrated_database, qdrant).reserve(seeded.generation_id)
        pipeline, _gateway, _usage = _make_repair_pipeline(
            migrated_database,
            store,
            qdrant,
            hooks=GenerationRepairHooks(checkpoint),
        )
        runner = _repair_runner(migrated_database, pipeline)
        job_id = UUID(reservation["job_id"])

        assert await runner.run_once(job_id)
        jobs = await _repair_jobs(migrated_database)
        assert jobs[0].status == "retry_wait"
        assert jobs[0].stage == "indexing"
        assert jobs[0].resume_stage == "indexing"
        assert jobs[0].progress_current == 0
        assert jobs[0].progress_total == len(manifest.chunks)
        assert jobs[0].target_revision == 8
        await _make_job_due(migrated_database, job_id)

        assert await runner.run_once(job_id)
        assert (await _repair_jobs(migrated_database))[0].status == "succeeded"
    finally:
        await qdrant.aclose()


async def test_target_reset_reloads_revision_after_the_pre_reset_race_window(
    migrated_database: Database,
    repair_qdrant_url: str,
) -> None:
    seeded, _profile_id, _spec = await _configured_generation(migrated_database)
    store = RepairObjectStore()
    manifests = [
        await _seed_ready_manifest_document(
            migrated_database,
            seeded,
            store,
            text="initial repair target",
        )
    ]
    drifted_after_upsert = False
    drifted_before_reset = False

    async def checkpoint(name: str) -> None:
        nonlocal drifted_after_upsert, drifted_before_reset
        if name == "indexing.after_upsert" and not drifted_after_upsert:
            drifted_after_upsert = True
            manifests.append(
                await _seed_ready_manifest_document(
                    migrated_database,
                    seeded,
                    store,
                    text="target added after the stale snapshot was loaded",
                    bump_revision=True,
                )
            )
        elif name == "target.before_reset" and not drifted_before_reset:
            drifted_before_reset = True
            manifests.append(
                await _seed_ready_manifest_document(
                    migrated_database,
                    seeded,
                    store,
                    text="target added immediately before the reset action locks",
                    bump_revision=True,
                )
            )

    qdrant = qdrant_client_from_url(repair_qdrant_url, timeout_seconds=5)
    try:
        reservation = await _repair_service(migrated_database, qdrant).reserve(seeded.generation_id)
        pipeline, _gateway, _usage = _make_repair_pipeline(
            migrated_database,
            store,
            qdrant,
            hooks=GenerationRepairHooks(checkpoint),
        )
        runner = _repair_runner(migrated_database, pipeline)
        job_id = UUID(reservation["job_id"])

        assert await runner.run_once(job_id)
        expected_total = sum(len(manifest.chunks) for manifest in manifests)
        jobs = await _repair_jobs(migrated_database)
        assert len(manifests) == 3
        assert jobs[0].status == "retry_wait"
        assert jobs[0].stage == "indexing"
        assert jobs[0].target_revision == 9
        assert jobs[0].progress_current == 0
        assert jobs[0].progress_total == expected_total
        await _make_job_due(migrated_database, job_id)

        assert await runner.run_once(job_id)
        assert (await _repair_jobs(migrated_database))[0].status == "succeeded"
        assert (
            await qdrant.count_points(
                collection_name(seeded.knowledge_base_id, seeded.generation_id)
            )
            == expected_total
        )
    finally:
        await qdrant.aclose()


async def test_stale_repair_worker_cannot_advance_progress_after_lease_reclaim(
    migrated_database: Database,
) -> None:
    seeded, _profile_id, _spec = await _configured_generation(migrated_database)
    store = RepairObjectStore()
    manifest = await _seed_ready_manifest_document(
        migrated_database,
        seeded,
        store,
        text="stale repair progress fencing",
    )
    reservation = await _repair_service(
        migrated_database,
        FakeQdrantClient(),
    ).reserve(seeded.generation_id)
    job_id = UUID(reservation["job_id"])
    async with migrated_database.sessions() as session, session.begin():
        stale = await SqlAlchemyJobRepository(session).claim_next(
            lease_owner="stale-repair-worker",
            lease_duration=timedelta(seconds=30),
            job_id=job_id,
        )
    assert isinstance(stale, JobLease)
    async with migrated_database.sessions() as session, session.begin():
        job = await session.get(Job, job_id)
        database_now = await session.scalar(select(func.clock_timestamp()))
        assert job is not None and isinstance(database_now, datetime)
        job.lease_expires_at = database_now - timedelta(seconds=1)
    async with migrated_database.sessions() as session, session.begin():
        current = await SqlAlchemyJobRepository(session).claim_next(
            lease_owner="current-repair-worker",
            lease_duration=timedelta(seconds=30),
            job_id=job_id,
        )
    assert isinstance(current, JobLease)

    async with migrated_database.sessions() as session, session.begin():
        with pytest.raises(LostLeaseError):
            await SqlAlchemyJobRepository(session).checkpoint(
                stale,
                stage="indexing",
                resume_stage="indexing",
                progress_current=1,
                progress_total=len(manifest.chunks),
            )

    async with migrated_database.sessions() as session:
        job = await session.get(Job, job_id)
        assert job is not None
        assert job.lease_owner == "current-repair-worker"
        assert job.progress_current == 0


async def test_equal_count_missing_and_extra_point_fails_exact_repair_validation(
    migrated_database: Database,
    repair_qdrant_url: str,
) -> None:
    seeded, _profile_id, _spec = await _configured_generation(migrated_database)
    store = RepairObjectStore()
    manifest = await _seed_ready_manifest_document(
        migrated_database,
        seeded,
        store,
        text="single canonical chunk",
    )
    assert len(manifest.chunks) == 1
    collection = collection_name(seeded.knowledge_base_id, seeded.generation_id)
    qdrant = qdrant_client_from_url(repair_qdrant_url, timeout_seconds=5)
    raw_qdrant = AsyncQdrantClient(url=repair_qdrant_url, timeout=5)
    corrupted = False

    async def checkpoint(name: str) -> None:
        nonlocal corrupted
        if name != "indexing.after_upsert" or corrupted:
            return
        corrupted = True
        chunk = manifest.chunks[0]
        expected_id = point_id(manifest.version_id, chunk.chunk_index, chunk.chunk_hash)
        await raw_qdrant.delete(
            collection_name=collection,
            points_selector=qdrant_models.PointIdsList(points=[str(expected_id)]),
            wait=True,
        )
        extra_hash = "e" * 64
        await qdrant.upsert_points(
            collection,
            (
                QdrantPoint(
                    id=point_id(manifest.version_id, 999, extra_hash),
                    vector=(0.1, 0.2, 0.3),
                    payload={
                        "knowledge_base_id": str(seeded.knowledge_base_id),
                        "document_id": str(manifest.document_id),
                        "version_id": str(manifest.version_id),
                        "chunk_index": 999,
                        "chunk_hash": extra_hash,
                        "text": "extra",
                        "title_path": [],
                        "start_offset": 0,
                        "end_offset": 5,
                        "metadata": {},
                    },
                ),
            ),
        )

    try:
        reservation = await _repair_service(migrated_database, qdrant).reserve(seeded.generation_id)
        pipeline, _gateway, _usage = _make_repair_pipeline(
            migrated_database,
            store,
            qdrant,
            hooks=GenerationRepairHooks(checkpoint),
        )

        assert await _repair_runner(migrated_database, pipeline).run_once(
            UUID(reservation["job_id"])
        )

        jobs = await _repair_jobs(migrated_database)
        assert jobs[0].status == "failed"
        assert jobs[0].error_code == "GENERATION_REPAIR_VALIDATION_FAILED"
        assert await qdrant.count_points(collection) == 1
        assert await qdrant.count_version_points(collection, manifest.version_id) == 1
    finally:
        await raw_qdrant.close()
        await qdrant.aclose()
