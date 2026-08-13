from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import tracemalloc
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import cast
from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.indexing.identities import (
    canonical_json_bytes,
    canonical_sha256,
    collection_name,
    point_id,
)
from rag_service.indexing.qdrant import (
    QdrantClient,
    QdrantConfigurationError,
    QdrantRetrievedPoint,
    QdrantTransientError,
)
from rag_service.infrastructure.minio_store import (
    ArtifactChecksumConflict,
    ObjectStoreError,
    OrphanCandidate,
    OrphanPage,
)
from rag_service.ingestion import pipeline as pipeline_module
from rag_service.ingestion.artifacts import (
    chunk_manifest_header_bytes,
    chunk_manifest_record_bytes,
)
from rag_service.ingestion.chunkers import Chunk, RecursiveTextChunker
from rag_service.ingestion.parsers import StructuralMetadataLimitExceeded, parser_for_extension
from rag_service.ingestion.pipeline import (
    IngestionPipeline,
    IngestionPipelineHooks,
    IngestionPipelineRepository,
    _manifest_stream,
    _ManifestBatchReader,
)
from rag_service.ingestion.repositories import (
    ActivationStageInput,
    ChunkStageInput,
    DocumentActivationConflictError,
    EmbedIndexStageInput,
    ParseStageInput,
    SqlAlchemyIngestionPipelineRepository,
)
from rag_service.ingestion.validation import (
    MAX_DOCUMENT_BYTES,
    DocumentValidationError,
)
from rag_service.jobs.repositories import JobLease, LostLeaseError
from rag_service.jobs.runner import (
    JobExecutionContext,
    JobHandlerOutcome,
    PermanentJobError,
    RetryableJobError,
    RetryableProviderJobError,
)
from rag_service.observability.metrics import OperationalMetrics
from rag_service.providers.embeddings import (
    EmbeddingConfigSnapshot,
    EmbeddingGatewayError,
    EmbeddingOperationalConfig,
)

MAX_CHUNK_MANIFEST_BYTES = 512 * 1024 * 1024


class BlockingManifestLines(Iterator[bytes]):
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.closed = threading.Event()
        self.running = False
        self.close_while_running = False
        self._emitted = False

    def __next__(self) -> bytes:
        if self._emitted:
            raise StopIteration
        self._emitted = True
        self.running = True
        self.started.set()
        try:
            self.release.wait(1)
            return b'{"chunk_index":0}\n'
        finally:
            self.running = False

    def close(self) -> None:
        self.close_while_running = self.running
        self.closed.set()


class BlockingCloseManifestLines(Iterator[bytes]):
    def __init__(self) -> None:
        self.close_started = threading.Event()
        self.close_release = threading.Event()
        self.closed = threading.Event()
        self._emitted = False

    def __next__(self) -> bytes:
        if self._emitted:
            raise StopIteration
        self._emitted = True
        return b'{"chunk_index":0}\n'

    def close(self) -> None:
        self.close_started.set()
        self.close_release.wait(1)
        self.closed.set()


@pytest.mark.asyncio
async def test_owned_cpu_executor_cancelled_work_keeps_slot_until_done_and_close_drains() -> None:
    executor_type = getattr(pipeline_module, "_OwnedCpuExecutor", None)
    assert executor_type is not None
    executor = executor_type(max_concurrency=1)
    first_started = threading.Event()
    first_release = threading.Event()
    second_started = threading.Event()
    second_release = threading.Event()

    def first() -> int:
        first_started.set()
        first_release.wait(1)
        return 1

    def second() -> int:
        second_started.set()
        second_release.wait(1)
        return 2

    first_task = asyncio.create_task(executor.run(first))
    assert await asyncio.to_thread(first_started.wait, 1)
    first_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_task

    second_task = asyncio.create_task(executor.run(second))
    await asyncio.sleep(0.02)
    assert second_started.is_set() is False

    first_release.set()
    assert await asyncio.to_thread(second_started.wait, 1)
    close_task = asyncio.create_task(executor.aclose())
    await asyncio.sleep(0.02)
    assert close_task.done() is False

    second_release.set()
    assert await second_task == 2
    await close_task


@pytest.mark.asyncio
async def test_owned_cpu_executor_close_wins_admission_race_without_submitting_waiter() -> None:
    executor_type = getattr(pipeline_module, "_OwnedCpuExecutor", None)
    assert executor_type is not None
    executor = executor_type(max_concurrency=1)
    active_started = threading.Event()
    active_release = threading.Event()
    waiter_started = threading.Event()

    def active() -> None:
        active_started.set()
        active_release.wait(1)

    def waiter() -> None:
        waiter_started.set()

    active_task = asyncio.create_task(executor.run(active))
    assert await asyncio.to_thread(active_started.wait, 1)
    waiter_task = asyncio.create_task(executor.run(waiter))
    await asyncio.sleep(0.02)
    close_task = asyncio.create_task(executor.aclose())
    await asyncio.sleep(0.02)

    active_release.set()
    await active_task
    with pytest.raises(RuntimeError, match="CPU executor is closed"):
        await waiter_task
    await close_task
    assert waiter_started.is_set() is False


@pytest.mark.asyncio
async def test_owned_cpu_executor_cancelled_close_still_drains_and_shuts_down() -> None:
    executor_type = getattr(pipeline_module, "_OwnedCpuExecutor", None)
    assert executor_type is not None
    executor = executor_type(max_concurrency=1)
    started = threading.Event()
    release = threading.Event()

    def blocking() -> None:
        started.set()
        release.wait(1)

    active_task = asyncio.create_task(executor.run(blocking))
    assert await asyncio.to_thread(started.wait, 1)
    close_task = asyncio.create_task(executor.aclose())
    await asyncio.sleep(0.02)
    close_task.cancel()
    await asyncio.sleep(0.02)
    assert close_task.done() is False

    release.set()
    await active_task
    with pytest.raises(asyncio.CancelledError):
        await close_task
    with pytest.raises(RuntimeError, match="CPU executor is closed"):
        await executor.run(lambda: None)


@pytest.mark.asyncio
async def test_detached_cancelled_cpu_callers_never_submit_beyond_concurrency() -> None:
    executor_type = getattr(pipeline_module, "_OwnedCpuExecutor", None)
    assert executor_type is not None
    executor = executor_type(max_concurrency=2)
    release = threading.Event()
    started = 0
    started_lock = threading.Lock()
    two_started = threading.Event()

    def blocking() -> None:
        nonlocal started
        with started_lock:
            started += 1
            if started == 2:
                two_started.set()
        release.wait(1)

    callers = [asyncio.create_task(executor.run(blocking)) for _ in range(20)]
    assert await asyncio.to_thread(two_started.wait, 1)
    for caller in callers:
        caller.cancel()
    await asyncio.gather(*callers, return_exceptions=True)
    await asyncio.sleep(0.02)
    assert started == 2

    release.set()
    await executor.aclose()
    assert started == 2


@pytest.mark.asyncio
async def test_manifest_stream_waits_for_cancelled_thread_pull_before_close() -> None:
    lines = BlockingManifestLines()
    stream = _manifest_stream(lines)
    pull: asyncio.Future[bytes] = asyncio.ensure_future(anext(stream))
    assert await asyncio.to_thread(lines.started.wait, 1)

    try:
        pull.cancel()
        await asyncio.sleep(0.02)
        assert lines.closed.is_set() is False
    finally:
        lines.release.set()
    with pytest.raises(asyncio.CancelledError):
        await pull
    assert lines.closed.is_set() is True
    assert lines.close_while_running is False


@pytest.mark.asyncio
async def test_shared_manifest_stream_settles_pull_before_close_on_repeated_cancel() -> None:
    executor_type = getattr(pipeline_module, "_OwnedCpuExecutor", None)
    assert executor_type is not None
    executor = executor_type(max_concurrency=2)
    lines = BlockingManifestLines()
    stream = _manifest_stream(lines, executor)
    pull: asyncio.Future[bytes] = asyncio.ensure_future(anext(stream))
    assert await asyncio.to_thread(lines.started.wait, 1)

    try:
        pull.cancel()
        await asyncio.sleep(0.02)
        pull.cancel()
        await asyncio.sleep(0.02)
        assert lines.closed.is_set() is False
    finally:
        lines.release.set()
    with pytest.raises(asyncio.CancelledError):
        await pull
    assert lines.closed.is_set() is True
    assert lines.close_while_running is False
    await executor.aclose()


@pytest.mark.asyncio
async def test_manifest_stream_repeated_cancel_during_close_still_closes_iterator() -> None:
    executor_type = getattr(pipeline_module, "_OwnedCpuExecutor", None)
    assert executor_type is not None
    executor = executor_type(max_concurrency=1)
    lines = BlockingCloseManifestLines()

    async def consume() -> None:
        async for _batch in _manifest_stream(lines, executor):
            pass

    task = asyncio.create_task(consume())
    assert await asyncio.to_thread(lines.close_started.wait, 1)
    try:
        task.cancel()
        await asyncio.sleep(0.02)
        task.cancel()
        await asyncio.sleep(0.02)
        assert task.done() is False
    finally:
        lines.close_release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert lines.closed.is_set() is True
    await executor.aclose()


def test_manifest_batch_reader_splits_an_oversized_record_at_the_batch_bound() -> None:
    body = b"x" * ((64 * 1024) + 123)
    reader = _ManifestBatchReader(iter((body,)))

    batches: list[bytes] = []
    while batch := reader.next_batch():
        batches.append(batch)

    assert b"".join(batches) == body
    assert max(map(len, batches)) <= 64 * 1024


@pytest.mark.parametrize("batch_bytes", [0, -1])
def test_manifest_batch_reader_rejects_non_positive_bounds(batch_bytes: int) -> None:
    with pytest.raises(ValueError, match="manifest batch size is invalid"):
        _ManifestBatchReader(iter((b"line\n",)), batch_bytes=batch_bytes)


def test_high_density_valid_manifest_uses_absolute_budget_not_source_multiplier() -> None:
    source = b"#\nx\n" * ((64 * 1024) // 4)
    parsed = parser_for_extension(".md").parse(source)
    chunker = RecursiveTextChunker()
    analyzer = getattr(pipeline_module, "_analyze_chunk_manifest", None)
    assert analyzer is not None

    chunk_count, manifest_size = analyzer(
        source_checksum_sha256=hashlib.sha256(source).hexdigest(),
        parsed=parsed,
        chunker=chunker,
        document_version_created_at=datetime(2025, 1, 1, tzinfo=UTC),
        max_bytes=MAX_CHUNK_MANIFEST_BYTES,
    )

    assert chunk_count > 0
    assert manifest_size > len(source) * 8
    assert manifest_size <= MAX_CHUNK_MANIFEST_BYTES


def test_typical_exact_50_mib_document_manifest_fits_with_o1_analysis_memory() -> None:
    source = b"a" * MAX_DOCUMENT_BYTES
    parsed = parser_for_extension(".txt").parse(source)
    chunker = RecursiveTextChunker()
    analyzer = getattr(pipeline_module, "_analyze_chunk_manifest", None)
    assert analyzer is not None

    tracemalloc.start()
    try:
        chunk_count, manifest_size = analyzer(
            source_checksum_sha256=hashlib.sha256(source).hexdigest(),
            parsed=parsed,
            chunker=chunker,
            document_version_created_at=datetime(2025, 1, 1, tzinfo=UTC),
            max_bytes=MAX_CHUNK_MANIFEST_BYTES,
        )
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert chunk_count > 40_000
    assert 50 * 1024 * 1024 < manifest_size < MAX_CHUNK_MANIFEST_BYTES
    assert peak < 8 * 1024 * 1024


def test_pipeline_default_manifest_budget_is_an_explicit_512_mib() -> None:
    pipeline = IngestionPipeline(
        repository_context=lambda: None,  # type: ignore[arg-type,return-value]
        object_store=object(),  # type: ignore[arg-type]
        max_document_bytes=1,
    )

    assert pipeline._max_manifest_bytes == MAX_CHUNK_MANIFEST_BYTES


class _ValidationStore:
    def __init__(self, manifest: bytes, object_key: str, checksum: str) -> None:
        self.manifest = manifest
        self.object_key = object_key
        self.checksum = checksum

    async def read_stream(
        self,
        object_key: str,
        *,
        expected_checksum: str,
        max_bytes: int,
    ) -> AsyncIterator[bytes]:
        assert object_key == self.object_key
        assert expected_checksum == self.checksum
        assert len(self.manifest) <= max_bytes
        for offset in range(0, len(self.manifest), 37):
            yield self.manifest[offset : offset + 37]


class _ValidationRepository:
    def __init__(
        self,
        expected: EmbedIndexStageInput,
        activation: ActivationStageInput,
    ) -> None:
        self.expected = expected
        self.activation = activation

    async def load_validate_stage(
        self,
        version_id: UUID,
        generation_id: UUID,
        job_id: UUID,
    ) -> EmbedIndexStageInput:
        assert (version_id, generation_id, job_id) == (
            self.expected.version_id,
            self.expected.generation_id,
            self.activation.job_id,
        )
        return self.expected

    async def load_embed_index_stage(
        self,
        version_id: UUID,
        generation_id: UUID,
        job_id: UUID,
    ) -> EmbedIndexStageInput:
        assert (version_id, generation_id, job_id) == (
            self.expected.version_id,
            self.expected.generation_id,
            self.activation.job_id,
        )
        return self.expected

    async def load_activation_stage(
        self,
        version_id: UUID,
        generation_id: UUID,
        job_id: UUID,
    ) -> ActivationStageInput:
        assert (version_id, generation_id, job_id) == (
            self.activation.version_id,
            self.activation.generation_id,
            self.activation.job_id,
        )
        return self.activation


class _PipelineContext:
    def __init__(
        self,
        lease: JobLease,
        *,
        domain_finalization_enabled: bool = False,
    ) -> None:
        self.lease = lease
        self.domain_finalization_enabled = domain_finalization_enabled
        self.finalized = False
        self.fact_commits = 0
        self.advances = 0
        self.finalizations = 0

    async def commit_stage_facts(self, _action: object) -> None:
        self.fact_commits += 1

    async def advance_stage(
        self,
        _action: object,
        *,
        stage: str,
        resume_stage: str,
        progress_current: int,
        progress_total: int | None,
    ) -> None:
        self.advances += 1
        self.lease = replace(
            self.lease,
            stage=stage,
            resume_stage=resume_stage,
            progress_current=progress_current,
            progress_total=progress_total,
        )

    async def finalize_domain(self, _action: object) -> None:
        self.finalizations += 1
        self.finalized = True


class _ValidationQdrant:
    def __init__(
        self,
        *,
        count: int,
        inspected: tuple[QdrantRetrievedPoint, ...],
        count_error: Exception | None = None,
        retrieve_error: Exception | None = None,
    ) -> None:
        self.count = count
        self.inspected = inspected
        self.count_error = count_error
        self.retrieve_error = retrieve_error
        self.count_calls = 0
        self.retrieve_calls: list[tuple[UUID, ...]] = []

    async def count_version_points(self, _collection: str, _version_id: UUID) -> int:
        self.count_calls += 1
        if self.count_error is not None:
            raise self.count_error
        return self.count

    async def retrieve_version_points(
        self,
        _collection: str,
        _version_id: UUID,
        point_ids: tuple[UUID, ...],
    ) -> tuple[QdrantRetrievedPoint, ...]:
        self.retrieve_calls.append(point_ids)
        if self.retrieve_error is not None:
            raise self.retrieve_error
        by_id = {point.id: point for point in self.inspected}
        return tuple(by_id[point_id] for point_id in point_ids)


def _validation_stage() -> tuple[
    EmbedIndexStageInput,
    ActivationStageInput,
    JobLease,
    bytes,
    tuple[Chunk, ...],
]:
    knowledge_base_id = uuid4()
    generation_id = uuid4()
    document_id = uuid4()
    version_id = uuid4()
    job_id = uuid4()
    source_checksum = "a" * 64
    source = b"alpha beta gamma delta epsilon"
    parsed = parser_for_extension(".txt").parse(source)
    chunker = RecursiveTextChunker(max_chunk_codepoints=9, target_overlap_codepoints=1)
    chunks = tuple(chunker.chunk(parsed))
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    manifest = chunk_manifest_header_bytes(
        source_checksum_sha256=source_checksum,
        parsed=parsed,
        chunker=chunker,
        document_version_created_at=created_at,
        chunk_count=len(chunks),
    ) + b"".join(chunk_manifest_record_bytes(chunk) for chunk in chunks)
    collection = collection_name(knowledge_base_id, generation_id)
    embedding_config_hash = "e" * 64
    provider_config_id = uuid4()
    gateway = EmbeddingConfigSnapshot(
        adapter_schema_version="openai-embeddings-v1",
        provider_type="openai_compatible",
        base_url="https://provider.example/v1",
        credential_id=uuid4(),
        default_headers={},
        routing_options={},
        model_name="text-embedding-test",
        dimension=3,
        distance="cosine",
        max_input_tokens=8192,
        vector_config={},
    )
    operational = EmbeddingOperationalConfig(
        provider_config_id=provider_config_id,
        provider_enabled=True,
        profile_enabled=True,
        timeout_seconds=Decimal("5.000"),
        max_concurrency=2,
        requests_per_minute=60,
        batch_size=2,
    )
    expected = EmbedIndexStageInput(
        knowledge_base_id=knowledge_base_id,
        generation_id=generation_id,
        document_id=document_id,
        version_id=version_id,
        actor_api_key_id=uuid4(),
        model_profile_id=uuid4(),
        provider_config_id=provider_config_id,
        manifest_object_key="kb/chunks.jsonl",
        manifest_checksum_sha256=hashlib.sha256(manifest).hexdigest(),
        source_checksum_sha256=source_checksum,
        parsed_checksum_sha256=parsed.checksum_sha256,
        parser_name=parsed.parser_name,
        parser_version=parsed.parser_version,
        parser_config=dict(parsed.parser_config),
        chunker_name=chunker.name,
        chunker_version=chunker.version,
        chunker_config=dict(chunker.config),
        chunk_config_hash=chunker.config_hash,
        chunk_count=len(chunks),
        version_created_at=created_at,
        version_status="indexing",
        index_state_status="indexing",
        index_state_embedding_config_hash=embedding_config_hash,
        next_chunk_index=len(chunks),
        qdrant_collection_name=collection,
        embedding_config_hash=embedding_config_hash,
        embedding_snapshot_canonical=b"{}",
        filter_snapshot={"fields": []},
        filter_snapshot_canonical=canonical_json_bytes({"fields": []}),
        applied_filter_schema_revision=0,
        document_metadata={},
        document_metadata_canonical=b"{}",
        gateway_snapshot=gateway,
        operational=operational,
    )
    activation = ActivationStageInput(
        knowledge_base_id=knowledge_base_id,
        generation_id=generation_id,
        document_id=document_id,
        version_id=version_id,
        job_id=job_id,
        source_checksum_sha256=source_checksum,
        detected_mime_type="text/plain",
        chunk_count=len(chunks),
        expected_point_count=len(chunks),
        actual_point_count=len(chunks),
    )
    lease = JobLease(
        id=job_id,
        operation="ingest_document",
        target_type="document_version",
        target_id=version_id,
        target_revision=None,
        index_generation_id=generation_id,
        stage="validate",
        resume_stage="validate",
        progress_current=len(chunks),
        progress_total=len(chunks),
        attempt_count=1,
        max_attempts=5,
        lease_owner="unit-worker",
        lease_epoch=1,
        lease_expires_at=datetime.now(UTC),
        cancel_requested_at=None,
    )
    return expected, activation, lease, manifest, chunks


def _inspected_points(
    expected: EmbedIndexStageInput,
    chunks: tuple[Chunk, ...],
) -> tuple[QdrantRetrievedPoint, ...]:
    payload_for = pipeline_module.qdrant_point_payload
    return tuple(
        QdrantRetrievedPoint(
            id=point_id(expected.version_id, chunk.chunk_index, chunk.chunk_hash),
            vector_dimension=expected.gateway_snapshot.dimension,
            payload_digest_sha256=canonical_sha256(payload_for(expected, chunk, {})),
        )
        for chunk in chunks
    )


def _validation_pipeline(
    expected: EmbedIndexStageInput,
    activation: ActivationStageInput,
    manifest: bytes,
    qdrant: _ValidationQdrant,
    *,
    metrics: OperationalMetrics | object | None = None,
    monotonic_clock: Callable[[], float] | None = None,
) -> IngestionPipeline:
    repository = _ValidationRepository(expected, activation)

    @asynccontextmanager
    async def repository_context() -> AsyncIterator[IngestionPipelineRepository]:
        yield cast(IngestionPipelineRepository, repository)

    kwargs: dict[str, object] = {}
    if metrics is not None:
        kwargs["metrics"] = metrics
    if monotonic_clock is not None:
        kwargs["monotonic_clock"] = monotonic_clock
    return IngestionPipeline(
        repository_context=repository_context,
        object_store=cast(
            pipeline_module.PipelineObjectStore,
            _ValidationStore(
                manifest,
                expected.manifest_object_key,
                expected.manifest_checksum_sha256,
            ),
        ),
        max_document_bytes=1024,
        embedding_gateway=None,
        qdrant=cast(QdrantClient, qdrant),
        **kwargs,  # type: ignore[arg-type]
    )


class _CollectingPipelineLogHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.mark.asyncio
async def test_pipeline_observes_each_completed_stage_once_across_continue_reentry() -> None:
    expected, activation, lease, manifest, chunks = _validation_stage()
    qdrant = _ValidationQdrant(
        count=len(chunks),
        inspected=_inspected_points(expected, chunks),
    )
    metrics = OperationalMetrics()
    clock_values = iter((10.0, 10.1, 20.0, 20.2))
    handler = _CollectingPipelineLogHandler()
    previous_handlers = list(pipeline_module.logger.handlers)
    previous_propagate = pipeline_module.logger.propagate
    previous_level = pipeline_module.logger.level
    pipeline_module.logger.handlers = [handler]
    pipeline_module.logger.propagate = False
    pipeline_module.logger.setLevel(logging.INFO)
    pipeline = _validation_pipeline(
        expected,
        activation,
        manifest,
        qdrant,
        metrics=metrics,
        monotonic_clock=lambda: next(clock_values),
    )
    context = _PipelineContext(lease, domain_finalization_enabled=True)
    try:
        assert (
            await pipeline.handle(cast(JobExecutionContext, context)) is JobHandlerOutcome.CONTINUE
        )
        assert (
            await pipeline.handle(cast(JobExecutionContext, context)) is JobHandlerOutcome.COMPLETE
        )
    finally:
        await pipeline.aclose()
        pipeline_module.logger.handlers = previous_handlers
        pipeline_module.logger.propagate = previous_propagate
        pipeline_module.logger.setLevel(previous_level)

    for stage in ("validate", "activate"):
        assert (
            metrics.registry.get_sample_value(
                "rag_ingestion_stage_duration_seconds_count",
                {"stage": stage, "outcome": "succeeded"},
            )
            == 1
        )
    assert [record.msg for record in handler.records] == [
        "ingestion.stage.completed",
        "ingestion.stage.completed",
    ]
    assert [record.__dict__["stage"] for record in handler.records] == [
        "validate",
        "activate",
    ]
    for record in handler.records:
        assert type(record) is logging.LogRecord
        assert record.__dict__["knowledge_base_id"] == str(expected.knowledge_base_id)
        assert record.__dict__["document_id"] == str(expected.document_id)
        assert record.__dict__["version_id"] == str(expected.version_id)
        assert record.__dict__["job_id"] == str(lease.id)
        assert record.__dict__["generation_id"] == str(expected.generation_id)


@pytest.mark.asyncio
async def test_pipeline_failure_observability_uses_bounded_code_and_never_exception_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected, activation, lease, manifest, chunks = _validation_stage()
    metrics = OperationalMetrics()
    handler = _CollectingPipelineLogHandler()
    previous_handlers = list(pipeline_module.logger.handlers)
    previous_propagate = pipeline_module.logger.propagate
    previous_level = pipeline_module.logger.level
    pipeline_module.logger.handlers = [handler]
    pipeline_module.logger.propagate = False
    pipeline_module.logger.setLevel(logging.INFO)
    pipeline = _validation_pipeline(
        expected,
        activation,
        manifest,
        _ValidationQdrant(count=len(chunks), inspected=_inspected_points(expected, chunks)),
        metrics=metrics,
    )

    async def fail(_context: JobExecutionContext) -> None:
        raise PermanentJobError("CUSTOM_QUERY_SECRET", "raw-query-traceback-secret")

    monkeypatch.setattr(pipeline, "_validate_stage", fail)
    try:
        with pytest.raises(PermanentJobError):
            await pipeline.handle(cast(JobExecutionContext, _PipelineContext(lease)))
    finally:
        await pipeline.aclose()
        pipeline_module.logger.handlers = previous_handlers
        pipeline_module.logger.propagate = previous_propagate
        pipeline_module.logger.setLevel(previous_level)

    assert (
        metrics.registry.get_sample_value(
            "rag_ingestion_stage_errors_total",
            {"stage": "validate", "failure_code": "VALIDATION_FAILED"},
        )
        == 1
    )
    assert len(handler.records) == 1
    assert handler.records[0].msg == "ingestion.stage.failed"
    rendered = repr(handler.records[0].__dict__)
    assert "CUSTOM_QUERY_SECRET" not in rendered
    assert "raw-query-traceback-secret" not in rendered


@pytest.mark.asyncio
async def test_pipeline_observes_lost_lease_with_exact_failure_code_and_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected, activation, lease, manifest, chunks = _validation_stage()
    metrics = OperationalMetrics()
    pipeline = _validation_pipeline(
        expected,
        activation,
        manifest,
        _ValidationQdrant(count=len(chunks), inspected=_inspected_points(expected, chunks)),
        metrics=metrics,
    )

    async def lose_lease(_context: JobExecutionContext) -> None:
        raise LostLeaseError

    monkeypatch.setattr(pipeline, "_validate_stage", lose_lease)
    try:
        with pytest.raises(LostLeaseError):
            await pipeline.handle(cast(JobExecutionContext, _PipelineContext(lease)))
    finally:
        await pipeline.aclose()

    assert (
        metrics.registry.get_sample_value(
            "rag_ingestion_stage_errors_total",
            {"stage": "validate", "failure_code": "LEASE_LOST"},
        )
        == 1
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("success", "upsert_failure", "provider_failure"))
async def test_embed_stage_observes_real_qdrant_upsert_and_committed_batches(
    mode: str,
) -> None:
    expected, activation, lease, manifest, chunks = _validation_stage()
    expected = replace(
        expected,
        next_chunk_index=0,
        operational=replace(expected.operational, batch_size=len(chunks)),
    )
    lease = replace(
        lease,
        stage="embed_index",
        resume_stage="embed_index",
        progress_current=0,
    )
    repository = _ValidationRepository(expected, activation)

    @asynccontextmanager
    async def repository_context() -> AsyncIterator[IngestionPipelineRepository]:
        yield cast(IngestionPipelineRepository, repository)

    class Gateway:
        async def embed(self, **kwargs: object) -> object:
            if mode == "provider_failure":
                raise EmbeddingGatewayError(
                    "PROVIDER_TIMEOUT",
                    "provider-query-secret",
                    retryable=True,
                )
            inputs = cast(tuple[str, ...], kwargs["inputs"])
            return SimpleNamespace(vectors=tuple((0.1, 0.2, 0.3) for _ in inputs))

    class UsageSink:
        async def record(self, context: object, attempt: object) -> None:
            del context, attempt

    class UpsertQdrant:
        def __init__(self) -> None:
            self.points: tuple[object, ...] = ()

        async def upsert_points(self, collection: str, points: tuple[object, ...]) -> None:
            assert collection == expected.qdrant_collection_name
            self.points = points
            if mode == "upsert_failure":
                raise QdrantTransientError("payload-text-vector-secret")

    qdrant = UpsertQdrant()
    metrics = OperationalMetrics()
    handler = _CollectingPipelineLogHandler()
    previous_handlers = list(pipeline_module.logger.handlers)
    previous_propagate = pipeline_module.logger.propagate
    previous_level = pipeline_module.logger.level
    pipeline_module.logger.handlers = [handler]
    pipeline_module.logger.propagate = False
    pipeline_module.logger.setLevel(logging.INFO)
    pipeline = IngestionPipeline(
        repository_context=repository_context,
        object_store=cast(
            pipeline_module.PipelineObjectStore,
            _ValidationStore(
                manifest,
                expected.manifest_object_key,
                expected.manifest_checksum_sha256,
            ),
        ),
        max_document_bytes=1024,
        embedding_gateway=cast(pipeline_module.PipelineEmbeddingGateway, Gateway()),
        qdrant=cast(QdrantClient, qdrant),
        provider_usage_sink=cast(pipeline_module.PipelineProviderUsageSink, UsageSink()),
        metrics=metrics,
    )
    try:
        if mode == "upsert_failure":
            with pytest.raises(RetryableJobError):
                await pipeline.handle(cast(JobExecutionContext, _PipelineContext(lease)))
        elif mode == "provider_failure":
            with pytest.raises(RetryableProviderJobError) as captured:
                await pipeline.handle(cast(JobExecutionContext, _PipelineContext(lease)))
            assert captured.value.provider_type == "openai_compatible"
        else:
            assert (
                await pipeline.handle(cast(JobExecutionContext, _PipelineContext(lease)))
                is JobHandlerOutcome.CONTINUE
            )
    finally:
        await pipeline.aclose()
        pipeline_module.logger.handlers = previous_handlers
        pipeline_module.logger.propagate = previous_propagate
        pipeline_module.logger.setLevel(previous_level)

    if mode != "provider_failure":
        outcome = "failed" if mode == "upsert_failure" else "succeeded"
        assert (
            metrics.registry.get_sample_value("rag_qdrant_upserts_total", {"outcome": outcome}) == 1
        )
        assert metrics.registry.get_sample_value(
            "rag_qdrant_upsert_points_total", {"outcome": outcome}
        ) == len(chunks)
    assert metrics.registry.get_sample_value(
        "rag_ingestion_batches_total", {"stage": "embed_index"}
    ) == (1 if mode == "success" else 0)
    rendered = repr([record.__dict__ for record in handler.records])
    assert "payload-text-vector-secret" not in rendered
    assert "provider-query-secret" not in rendered
    assert all(chunk.text not in rendered for chunk in chunks)


@pytest.mark.asyncio
async def test_validate_handler_checks_bounded_batches_and_advances() -> None:
    expected, activation, lease, manifest, chunks = _validation_stage()
    qdrant = _ValidationQdrant(
        count=len(chunks),
        inspected=_inspected_points(expected, chunks),
    )
    pipeline = _validation_pipeline(expected, activation, manifest, qdrant)
    context = _PipelineContext(lease)
    try:
        outcome = await pipeline.handle(cast(JobExecutionContext, context))
    finally:
        await pipeline.aclose()

    assert outcome is JobHandlerOutcome.CONTINUE
    assert context.fact_commits == 1
    assert context.advances == 1
    assert context.lease.stage == "activate"
    assert qdrant.count_calls == 1
    assert [len(batch) for batch in qdrant.retrieve_calls] == [2, len(chunks) - 2]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "exhausted", "expected_code", "finalized"),
    [
        ("count", False, "INDEX_VALIDATION_FAILED", True),
        ("payload", False, "INDEX_VALIDATION_FAILED", True),
        ("configuration", False, "INDEX_VALIDATION_FAILED", True),
        ("transient", False, "QDRANT_UNAVAILABLE", False),
        ("transient", True, "QDRANT_UNAVAILABLE", True),
        ("invalid-stage", False, "INGESTION_STAGE_CONFLICT", True),
    ],
)
async def test_validate_handler_classifies_and_terminalizes_only_terminal_failures(
    mode: str,
    exhausted: bool,
    expected_code: str,
    finalized: bool,
) -> None:
    expected, activation, lease, manifest, chunks = _validation_stage()
    inspected = list(_inspected_points(expected, chunks))
    count = len(chunks)
    count_error: Exception | None = None
    retrieve_error: Exception | None = None
    if mode == "count":
        count += 1
    elif mode == "payload":
        inspected[0] = replace(inspected[0], payload_digest_sha256="0" * 64)
    elif mode == "configuration":
        retrieve_error = QdrantConfigurationError("bad response")
    elif mode == "transient":
        count_error = QdrantTransientError("offline")
    else:
        expected = replace(expected, version_status="embedding")
    qdrant = _ValidationQdrant(
        count=count,
        inspected=tuple(inspected),
        count_error=count_error,
        retrieve_error=retrieve_error,
    )
    pipeline = _validation_pipeline(expected, activation, manifest, qdrant)
    context = _PipelineContext(
        replace(lease, attempt_count=5 if exhausted else 1),
        domain_finalization_enabled=True,
    )
    try:
        if mode == "transient" and not exhausted:
            with pytest.raises(RetryableJobError) as exc_info:
                await pipeline.handle(cast(JobExecutionContext, context))
            assert exc_info.value.code == expected_code
        else:
            outcome = await pipeline.handle(cast(JobExecutionContext, context))
            assert outcome is JobHandlerOutcome.COMPLETE
    finally:
        await pipeline.aclose()

    assert context.finalized is finalized
    assert context.finalizations == int(finalized)


@pytest.mark.asyncio
async def test_activate_handler_delegates_one_domain_finalization() -> None:
    expected, activation, lease, manifest, chunks = _validation_stage()
    qdrant = _ValidationQdrant(
        count=len(chunks),
        inspected=_inspected_points(expected, chunks),
    )
    pipeline = _validation_pipeline(expected, activation, manifest, qdrant)
    context = _PipelineContext(
        replace(lease, stage="activate", resume_stage="activate"),
        domain_finalization_enabled=True,
    )
    try:
        outcome = await pipeline.handle(cast(JobExecutionContext, context))
    finally:
        await pipeline.aclose()

    assert outcome is JobHandlerOutcome.COMPLETE
    assert context.finalized is True
    assert context.finalizations == 1


@pytest.mark.asyncio
async def test_unavailable_qdrant_stage_is_a_safe_permanent_error() -> None:
    expected, activation, lease, manifest, _chunks = _validation_stage()
    repository = _ValidationRepository(expected, activation)

    @asynccontextmanager
    async def repository_context() -> AsyncIterator[IngestionPipelineRepository]:
        yield cast(IngestionPipelineRepository, repository)

    pipeline = IngestionPipeline(
        repository_context=repository_context,
        object_store=cast(
            pipeline_module.PipelineObjectStore,
            _ValidationStore(
                manifest,
                expected.manifest_object_key,
                expected.manifest_checksum_sha256,
            ),
        ),
        max_document_bytes=1024,
    )
    context = _PipelineContext(lease)
    try:
        with pytest.raises(PermanentJobError) as exc_info:
            await pipeline.handle(cast(JobExecutionContext, context))
    finally:
        await pipeline.aclose()

    assert exc_info.value.code == "JOB_STAGE_UNAVAILABLE"


class _ExecutingPipelineContext(_PipelineContext):
    async def _run(self, action: object) -> None:
        callback = cast(Callable[[AsyncSession], Awaitable[None]], action)
        await callback(cast(AsyncSession, object()))

    async def commit_stage_facts(self, action: object) -> None:
        await self._run(action)
        await super().commit_stage_facts(action)

    async def advance_stage(
        self,
        action: object,
        *,
        stage: str,
        resume_stage: str,
        progress_current: int,
        progress_total: int | None,
    ) -> None:
        await self._run(action)
        await super().advance_stage(
            action,
            stage=stage,
            resume_stage=resume_stage,
            progress_current=progress_current,
            progress_total=progress_total,
        )

    async def finalize_domain(self, action: object) -> None:
        await self._run(action)
        await super().finalize_domain(action)


@pytest.mark.asyncio
async def test_validate_activate_and_terminal_callbacks_execute_repository_commits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected, activation, lease, manifest, chunks = _validation_stage()
    commits: list[str] = []

    async def commit_validate_count(
        _repository: SqlAlchemyIngestionPipelineRepository,
        value: EmbedIndexStageInput,
        *,
        actual_count: int,
    ) -> None:
        assert value is expected
        assert actual_count == len(chunks)
        commits.append("count")

    async def commit_validate(
        _repository: SqlAlchemyIngestionPipelineRepository,
        value: EmbedIndexStageInput,
        *,
        actual_count: int,
    ) -> None:
        assert value is expected
        assert actual_count == len(chunks)
        commits.append("validate")

    async def commit_activation(
        _repository: SqlAlchemyIngestionPipelineRepository,
        value: ActivationStageInput,
        current_lease: JobLease,
    ) -> None:
        assert value is activation
        assert current_lease.id == lease.id
        commits.append("activate")

    async def commit_terminal(
        _repository: SqlAlchemyIngestionPipelineRepository,
        current_lease: JobLease,
        *,
        retryable: bool,
        error_code: str,
        safe_error_message: str,
    ) -> None:
        assert current_lease.id == lease.id
        assert retryable is False
        assert error_code == "INGESTION_STAGE_CONFLICT"
        assert safe_error_message == "Ingestion stage conflict"
        commits.append("terminal")

    monkeypatch.setattr(
        SqlAlchemyIngestionPipelineRepository,
        "commit_validate_count",
        commit_validate_count,
    )
    monkeypatch.setattr(
        SqlAlchemyIngestionPipelineRepository,
        "commit_validate_stage",
        commit_validate,
    )
    monkeypatch.setattr(
        SqlAlchemyIngestionPipelineRepository,
        "commit_activation",
        commit_activation,
    )
    monkeypatch.setattr(
        SqlAlchemyIngestionPipelineRepository,
        "commit_terminal_failure",
        commit_terminal,
    )
    qdrant = _ValidationQdrant(
        count=len(chunks),
        inspected=_inspected_points(expected, chunks),
    )
    pipeline = _validation_pipeline(expected, activation, manifest, qdrant)
    try:
        validate_context = _ExecutingPipelineContext(lease)
        assert (
            await pipeline.handle(cast(JobExecutionContext, validate_context))
            is JobHandlerOutcome.CONTINUE
        )
        activate_context = _ExecutingPipelineContext(
            replace(lease, stage="activate", resume_stage="activate"),
            domain_finalization_enabled=True,
        )
        assert (
            await pipeline.handle(cast(JobExecutionContext, activate_context))
            is JobHandlerOutcome.COMPLETE
        )
        terminal_context = _ExecutingPipelineContext(
            replace(lease, operation="cleanup_generation"),
            domain_finalization_enabled=True,
        )
        assert (
            await pipeline.handle(cast(JobExecutionContext, terminal_context))
            is JobHandlerOutcome.COMPLETE
        )
    finally:
        await pipeline.aclose()

    assert commits == ["count", "validate", "activate", "terminal"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_type", "expected_code"),
    [
        (ArtifactChecksumConflict(), PermanentJobError, "ARTIFACT_CHECKSUM_CONFLICT"),
        (
            ObjectStoreError("OBJECT_STORE_UNAVAILABLE", retryable=True),
            RetryableJobError,
            "OBJECT_STORE_UNAVAILABLE",
        ),
        (
            ObjectStoreError("OBJECT_STORE_API_INCOMPATIBLE", retryable=False),
            PermanentJobError,
            "OBJECT_STORE_API_INCOMPATIBLE",
        ),
        (DocumentValidationError("EMPTY_DOCUMENT"), PermanentJobError, "EMPTY_DOCUMENT"),
        (
            StructuralMetadataLimitExceeded(),
            PermanentJobError,
            "STRUCTURAL_METADATA_TOO_LARGE",
        ),
        (
            EmbeddingGatewayError("RATE_LIMITED", "Rate limited", retryable=True),
            RetryableJobError,
            "RATE_LIMITED",
        ),
        (
            EmbeddingGatewayError("BAD_CONFIG", "Bad config", retryable=False),
            PermanentJobError,
            "BAD_CONFIG",
        ),
        (QdrantTransientError("offline"), RetryableJobError, "QDRANT_UNAVAILABLE"),
        (
            QdrantConfigurationError("bad collection"),
            PermanentJobError,
            "QDRANT_CONFIGURATION_CONFLICT",
        ),
        (
            DocumentActivationConflictError("changed"),
            PermanentJobError,
            "DOCUMENT_ACTIVATION_CONFLICT",
        ),
        (ValueError("bad state"), PermanentJobError, "INGESTION_STAGE_CONFLICT"),
        (TypeError("bad state"), PermanentJobError, "INGESTION_STAGE_CONFLICT"),
        (UnicodeError("bad state"), PermanentJobError, "INGESTION_STAGE_CONFLICT"),
    ],
)
async def test_dispatch_maps_stage_failures_to_safe_job_errors(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    expected_type: type[Exception],
    expected_code: str,
) -> None:
    expected, activation, lease, manifest, chunks = _validation_stage()
    pipeline = _validation_pipeline(
        expected,
        activation,
        manifest,
        _ValidationQdrant(
            count=len(chunks),
            inspected=_inspected_points(expected, chunks),
        ),
    )

    async def fail(_context: JobExecutionContext) -> None:
        raise error

    monkeypatch.setattr(pipeline, "_validate_stage", fail)
    try:
        with pytest.raises(expected_type) as exc_info:
            await pipeline.handle(cast(JobExecutionContext, _PipelineContext(lease)))
    finally:
        await pipeline.aclose()

    assert cast(PermanentJobError | RetryableJobError, exc_info.value).code == expected_code


def test_manifest_json_freezing_filter_values_and_point_dimension_are_strict() -> None:
    freeze = pipeline_module._freeze_manifest_json
    thaw = pipeline_module._thaw_manifest_json
    frozen = freeze({"z": [1, {"nested": None}], "a": True})
    assert thaw(frozen) == {"a": True, "z": [1, {"nested": None}]}
    with pytest.raises(ValueError):
        freeze({1, 2})
    with pytest.raises(ValueError):
        thaw([1, 2])

    valid_filter = pipeline_module._valid_filter_value
    assert valid_filter("keyword", "value") is True
    assert valid_filter("keyword", "\x00") is False
    assert valid_filter("integer", 2**63 - 1) is True
    assert valid_filter("integer", True) is False
    assert valid_filter("integer", 2**63) is False
    assert valid_filter("float", 1.25) is True
    assert valid_filter("float", True) is False
    assert valid_filter("float", float("inf")) is False
    assert valid_filter("float", 10**10_000) is False
    assert valid_filter("boolean", False) is True
    assert valid_filter("boolean", 0) is False
    assert valid_filter("datetime", "2026-01-01T00:00:00Z") is True
    assert valid_filter("datetime", "yesterday") is False
    assert valid_filter("unsupported", "value") is False

    expected, _activation, _lease_value, _manifest_value, chunks = _validation_stage()
    point_for = pipeline_module.qdrant_point_for
    with pytest.raises(ValueError, match="embedding result is invalid"):
        point_for(expected, chunks[0], (0.1,), {})
    assert len(point_for(expected, chunks[0], (0.1, 0.2, 0.3), {}).vector) == 3


def test_approved_filter_metadata_accepts_only_declared_typed_values() -> None:
    expected, _activation, _lease_value, _manifest_value, _chunks = _validation_stage()
    fields = [
        {
            "source_path": "nested.keyword",
            "type": "keyword",
            "payload_path": "metadata.f_keyword",
        },
        {
            "source_path": "integer",
            "type": "integer",
            "payload_path": "metadata.f_integer",
        },
        {
            "source_path": "float",
            "type": "float",
            "payload_path": "metadata.f_float",
        },
        {
            "source_path": "boolean",
            "type": "boolean",
            "payload_path": "metadata.f_boolean",
        },
        {
            "source_path": "datetime",
            "type": "datetime",
            "payload_path": "metadata.f_datetime",
        },
        {
            "source_path": "missing.value",
            "type": "keyword",
            "payload_path": "metadata.f_missing",
        },
    ]
    configured = replace(
        expected,
        filter_snapshot={"fields": fields},
        document_metadata={
            "nested": {"keyword": "value"},
            "integer": 7,
            "float": 1.5,
            "boolean": True,
            "datetime": "2026-01-01T00:00:00Z",
        },
    )
    approved = pipeline_module.approved_filter_metadata
    assert approved(configured) == {
        "f_boolean": True,
        "f_datetime": "2026-01-01T00:00:00Z",
        "f_float": 1.5,
        "f_integer": 7,
        "f_keyword": "value",
    }

    invalid_snapshots: tuple[object, ...] = (
        {"fields": "bad"},
        {"fields": ["bad"]},
        {"fields": [{"source_path": 1, "type": "keyword", "payload_path": "metadata.f"}]},
        {"fields": [{"source_path": "integer", "type": "integer", "payload_path": "wrong.f"}]},
        {
            "fields": [
                {
                    "source_path": "nested",
                    "type": "keyword",
                    "payload_path": "metadata.f_nested",
                }
            ]
        },
    )
    for snapshot in invalid_snapshots:
        with pytest.raises(ValueError, match="stage state is invalid"):
            approved(replace(configured, filter_snapshot=cast(dict[str, object], snapshot)))


def test_manifest_expectation_and_documents_reject_noncanonical_data() -> None:
    stage, _activation, _lease_value, value, chunks = _validation_stage()
    expected = pipeline_module.manifest_expectation(stage)
    with pytest.raises(ValueError, match="expectation is invalid"):
        replace(expected, parser_name="")
    with pytest.raises(ValueError, match="expectation is invalid"):
        replace(expected, parser_config={"bad": object()})

    manifest_document = pipeline_module._manifest_document
    for line in (b"[]\n", b'{ "a":1}\n', b"\xff\n"):
        with pytest.raises(ValueError, match="chunk manifest is invalid"):
            manifest_document(line)

    manifest_chunk = pipeline_module._manifest_chunk
    record = json.loads(chunk_manifest_record_bytes(chunks[0]))
    for changed in (
        {**record, "extra": True},
        {**record, "title_path": "bad"},
        {**record, "chunk_index": 99},
    ):
        with pytest.raises(ValueError, match="chunk manifest is invalid"):
            manifest_chunk(changed, 0)

    header = json.loads(value.splitlines(keepends=True)[0])
    with pytest.raises(ValueError, match="chunk manifest is invalid"):
        pipeline_module._validate_manifest_header({**header, "chunk_count": 999}, expected)


@pytest.mark.asyncio
async def test_manifest_streaming_rejects_bad_blocks_lines_and_cardinality() -> None:
    manifest_lines = pipeline_module._manifest_lines

    async def invalid_type() -> AsyncIterator[bytes]:
        yield cast(bytes, "bad")

    async def one_block(value: bytes) -> AsyncIterator[bytes]:
        yield value

    for stream in (
        invalid_type(),
        one_block(b"x" * ((1024 * 1024) + 1)),
        one_block(b"\n"),
        one_block(b"{}"),
    ):
        with pytest.raises(ValueError, match="chunk manifest is invalid"):
            async for _line in manifest_lines(stream):
                pass

    stage, _activation, _lease_value, value, _chunks = _validation_stage()
    expected = pipeline_module.manifest_expectation(stage)
    lines = value.splitlines(keepends=True)
    too_few = b"".join(lines[:-1])
    too_many = value + lines[-1]
    for malformed in (too_few, too_many):
        with pytest.raises(ValueError, match="chunk manifest is invalid"):
            async for _batch in pipeline_module.iter_verified_manifest_batches(
                one_block(malformed),
                expected=expected,
                next_chunk_index=0,
                batch_size=2,
            ):
                pass

    with pytest.raises(ValueError, match="chunk manifest is invalid"):
        async for _batch in pipeline_module.iter_verified_manifest_batches(
            cast(AsyncIterator[bytes], object()),
            expected=expected,
            next_chunk_index=0,
            batch_size=2,
        ):
            pass


def test_pipeline_configuration_parser_and_chunker_snapshots_are_strict() -> None:
    with pytest.raises(ValueError, match="hook is invalid"):
        IngestionPipelineHooks(cast(Callable[[str], Awaitable[None]], 1))
    store = cast(pipeline_module.PipelineObjectStore, object())
    repository_context = cast(pipeline_module.PipelineRepositoryContextFactory, lambda: None)
    with pytest.raises(ValueError):
        IngestionPipeline(
            repository_context=cast(pipeline_module.PipelineRepositoryContextFactory, 1),
            object_store=store,
            max_document_bytes=1,
        )
    with pytest.raises(ValueError):
        IngestionPipeline(
            repository_context=repository_context,
            object_store=store,
            max_document_bytes=0,
        )
    with pytest.raises(ValueError):
        IngestionPipeline(
            repository_context=repository_context,
            object_store=store,
            max_document_bytes=1,
            max_manifest_bytes=0,
        )
    with pytest.raises(ValueError):
        IngestionPipeline(
            repository_context=repository_context,
            object_store=store,
            max_document_bytes=1,
            cpu_concurrency=0,
        )

    parser = parser_for_extension(".txt")
    parse_input = ParseStageInput(
        knowledge_base_id=uuid4(),
        generation_id=uuid4(),
        document_id=uuid4(),
        version_id=uuid4(),
        source_object_key="kb/source.txt",
        source_checksum_sha256="a" * 64,
        source_extension=".txt",
        parser_name=parser.name,
        parser_version=parser.version,
        parser_config=dict(parser.config),
        version_status="uploaded",
    )
    assert pipeline_module._parser_for(parse_input).name == parser.name
    with pytest.raises(ValueError, match="stage state is invalid"):
        pipeline_module._parser_for(replace(parse_input, parser_version="stale"))

    base_chunk = ChunkStageInput(
        knowledge_base_id=parse_input.knowledge_base_id,
        generation_id=parse_input.generation_id,
        document_id=parse_input.document_id,
        version_id=parse_input.version_id,
        source_checksum_sha256=parse_input.source_checksum_sha256,
        source_extension=parse_input.source_extension,
        parsed_object_key="kb/parsed.txt",
        parsed_object_checksum_sha256="b" * 64,
        parser_name=parse_input.parser_name,
        parser_version=parse_input.parser_version,
        parser_config=parse_input.parser_config,
        chunker_name=None,
        chunker_version=None,
        chunker_config={},
        version_status="chunking",
        version_created_at=datetime.now(UTC),
    )
    assert pipeline_module._chunker_for(base_chunk).name == RecursiveTextChunker.name
    # A first chunk stage honours the configured sizing.
    assert pipeline_module._chunker_for(base_chunk, (500, 80)).config == {
        "max_chunk_codepoints": 500,
        "target_overlap_codepoints": 80,
    }
    configured = replace(
        base_chunk,
        chunker_name=RecursiveTextChunker.name,
        chunker_version=RecursiveTextChunker.version,
        chunker_config={"max_chunk_codepoints": 100, "target_overlap_codepoints": 10},
    )
    assert pipeline_module._chunker_for(configured).config == {
        "max_chunk_codepoints": 100,
        "target_overlap_codepoints": 10,
    }
    # A resumed stage keeps the recorded manifest sizing, never the new default,
    # so an in-flight document cannot be rechunked under a changed setting.
    assert pipeline_module._chunker_for(configured, (500, 80)).config == {
        "max_chunk_codepoints": 100,
        "target_overlap_codepoints": 10,
    }
    for invalid in (
        replace(base_chunk, chunker_config={"unexpected": 1}),
        replace(configured, chunker_name="stale"),
        replace(configured, chunker_config={"max_chunk_codepoints": 100}),
        replace(
            configured,
            chunker_config={"max_chunk_codepoints": "100", "target_overlap_codepoints": 10},
        ),
    ):
        with pytest.raises(ValueError, match="stage state is invalid"):
            pipeline_module._chunker_for(invalid)


@pytest.mark.asyncio
async def test_dispatch_and_stage_loads_reject_missing_or_stale_job_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected, activation, lease, manifest, chunks = _validation_stage()
    pipeline = _validation_pipeline(
        expected,
        activation,
        manifest,
        _ValidationQdrant(
            count=len(chunks),
            inspected=_inspected_points(expected, chunks),
        ),
    )
    missing_generation = _PipelineContext(replace(lease, index_generation_id=None))
    try:
        for loader in (
            pipeline._load_parse,
            pipeline._load_chunk,
            pipeline._load_embed_index,
            pipeline._load_validate,
            pipeline._load_activation,
        ):
            with pytest.raises(ValueError, match="stage state is invalid"):
                await loader(cast(JobExecutionContext, missing_generation))

        calls: list[str] = []

        async def reached(context: JobExecutionContext) -> None:
            assert context.lease.target_id == lease.target_id
            calls.append(cast(str, context.lease.stage))

        for stage, method_name in (
            ("parse", "_parse_stage"),
            ("chunk", "_chunk_stage"),
            ("embed_index", "_embed_index_stage"),
        ):
            monkeypatch.setattr(pipeline, method_name, reached)
            await pipeline._dispatch_stage(
                cast(JobExecutionContext, _PipelineContext(replace(lease, stage=stage)))
            )
        assert calls == ["parse", "chunk", "embed_index"]

        for stale in (
            replace(lease, stage="unknown"),
            replace(lease, operation="cleanup_generation"),
            replace(lease, target_type="knowledge_base"),
        ):
            with pytest.raises(PermanentJobError):
                await pipeline._dispatch_stage(cast(JobExecutionContext, _PipelineContext(stale)))
    finally:
        await pipeline.aclose()


class _ReconciliationRepository:
    def __init__(self, referenced: set[str], protected: set[str] | None = None) -> None:
        self.referenced = referenced
        self.protected = set() if protected is None else protected
        self.checked: list[str] = []

    async def object_key_is_referenced(self, object_key: str) -> bool:
        self.checked.append(object_key)
        return object_key in self.referenced

    async def object_key_cleanup_is_allowed(self, object_key: str) -> bool:
        self.checked.append(object_key)
        return object_key not in self.referenced and object_key not in self.protected


class _ReconciliationObjectStore:
    def __init__(self, candidates: tuple[OrphanCandidate, ...]) -> None:
        self.candidates = candidates
        self.list_calls: list[tuple[str, datetime, int, str | None]] = []
        self.deleted: list[str] = []
        self.failures: dict[str, BaseException | bool] = {}

    async def list_older_than(
        self,
        *,
        prefix: str,
        older_than: datetime,
        limit: int,
        start_after: str | None = None,
    ) -> OrphanPage:
        self.list_calls.append((prefix, older_than, limit, start_after))
        eligible = [
            item
            for item in self.candidates
            if item.object_key.startswith(prefix)
            and item.last_modified < older_than
            and (start_after is None or item.object_key > start_after)
        ]
        eligible.sort(key=lambda item: item.object_key)
        page = eligible[:limit]
        next_start_after = page[-1].object_key if len(eligible) > limit else None
        return OrphanPage(tuple(page), next_start_after)

    async def delete_best_effort(self, object_key: str) -> bool:
        failure = self.failures.get(object_key, True)
        if isinstance(failure, BaseException):
            raise failure
        if failure:
            self.deleted.append(object_key)
        return failure

    async def aclose(self) -> None:
        return None


def _reconciliation_pipeline(
    repository: _ReconciliationRepository,
    store: _ReconciliationObjectStore,
    *,
    now: datetime,
    hooks: IngestionPipelineHooks | None = None,
) -> IngestionPipeline:
    @asynccontextmanager
    async def repository_context() -> AsyncIterator[IngestionPipelineRepository]:
        yield cast(IngestionPipelineRepository, repository)

    pipeline = IngestionPipeline(
        repository_context=repository_context,
        object_store=cast(pipeline_module.PipelineObjectStore, store),
        max_document_bytes=1024,
        hooks=hooks,
        reconciliation_clock=lambda: now,
    )
    return pipeline


@pytest.mark.asyncio
async def test_object_reconciliation_is_age_prefix_reference_and_page_bounded() -> None:
    now = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
    cutoff = now - timedelta(hours=24)
    kb_id, document_id, version_id = uuid4(), uuid4(), uuid4()
    prefix = f"knowledge-bases/{kb_id}/documents/{document_id}/versions/{version_id}"
    temp_old = OrphanCandidate(
        "tmp/jobs/job/1/parsed/text.txt",
        1,
        None,
        cutoff - timedelta(seconds=1),
    )
    temp_boundary = OrphanCandidate("tmp/jobs/job/2/parsed/text.txt", 1, None, cutoff)
    source_unreferenced = OrphanCandidate(
        f"{prefix}/source/source.txt", 1, None, cutoff - timedelta(days=1)
    )
    parsed_referenced = OrphanCandidate(
        f"{prefix}/parsed/text.txt", 1, None, cutoff - timedelta(days=1)
    )
    noncanonical = OrphanCandidate(f"{prefix}/private/raw.txt", 1, None, cutoff - timedelta(days=1))
    repository = _ReconciliationRepository({parsed_referenced.object_key})
    store = _ReconciliationObjectStore(
        (temp_old, temp_boundary, source_unreferenced, parsed_referenced, noncanonical)
    )
    pipeline = _reconciliation_pipeline(repository, store, now=now)
    deleted = 0
    cursor = None
    try:
        while True:
            result = await pipeline.reconcile_orphan_objects(
                grace_period=timedelta(hours=24),
                limit=2,
                cursor=cursor,
            )
            deleted += result.deleted_count
            assert result.scanned_count <= 2
            cursor = result.next_cursor
            if cursor is None:
                break
    finally:
        await pipeline.aclose()

    assert deleted == 2
    assert store.deleted == [temp_old.object_key, source_unreferenced.object_key]
    assert temp_boundary.object_key not in store.deleted
    assert repository.checked == [
        temp_old.object_key,
        parsed_referenced.object_key,
        source_unreferenced.object_key,
    ]
    assert all(limit == 2 for _prefix, _older_than, limit, _cursor in store.list_calls)
    assert {prefix for prefix, _older_than, _limit, _cursor in store.list_calls} == {
        "tmp/jobs/",
        "knowledge-bases/",
    }
    assert noncanonical.object_key not in repr(result)


@pytest.mark.asyncio
async def test_object_reconciliation_rechecks_reference_immediately_before_final_delete() -> None:
    now = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
    kb_id, document_id, version_id = uuid4(), uuid4(), uuid4()
    object_key = (
        f"knowledge-bases/{kb_id}/documents/{document_id}/versions/{version_id}"
        "/chunks/recursive_text_v1.jsonl"
    )
    repository = _ReconciliationRepository(set())
    store = _ReconciliationObjectStore(
        (OrphanCandidate(object_key, 1, None, now - timedelta(hours=25)),)
    )

    async def add_reference(name: str) -> None:
        if name == "before_orphan_object_reference_check":
            repository.referenced.add(object_key)

    pipeline = _reconciliation_pipeline(
        repository,
        store,
        now=now,
        hooks=IngestionPipelineHooks(add_reference),
    )
    try:
        first = await pipeline.reconcile_orphan_objects(
            grace_period=timedelta(hours=24),
            limit=10,
            cursor=None,
        )
        assert first.next_cursor is not None
        result = await pipeline.reconcile_orphan_objects(
            grace_period=timedelta(hours=24),
            limit=10,
            cursor=first.next_cursor,
        )
    finally:
        await pipeline.aclose()

    assert result.deleted_count == 0
    assert repository.checked == [object_key]
    assert store.deleted == []


@pytest.mark.asyncio
async def test_object_reconciliation_rechecks_temporary_object_references_before_delete() -> None:
    now = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
    object_key = "tmp/jobs/job-a/1/parsed/text.txt"
    repository = _ReconciliationRepository({object_key})
    store = _ReconciliationObjectStore(
        (OrphanCandidate(object_key, 1, None, now - timedelta(hours=25)),)
    )
    pipeline = _reconciliation_pipeline(repository, store, now=now)
    try:
        result = await pipeline.reconcile_orphan_objects(
            grace_period=timedelta(hours=24),
            limit=10,
            cursor=None,
        )
    finally:
        await pipeline.aclose()

    assert result.deleted_count == 0
    assert repository.checked == [object_key]
    assert store.deleted == []


@pytest.mark.asyncio
async def test_object_reconciliation_preserves_uncommitted_canonical_artifact_for_version() -> None:
    now = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
    kb_id, document_id, version_id = uuid4(), uuid4(), uuid4()
    object_key = (
        f"knowledge-bases/{kb_id}/documents/{document_id}/versions/{version_id}/parsed/text.txt"
    )
    repository = _ReconciliationRepository(set(), {object_key})
    store = _ReconciliationObjectStore(
        (OrphanCandidate(object_key, 1, None, now - timedelta(hours=25)),)
    )
    pipeline = _reconciliation_pipeline(repository, store, now=now)
    try:
        first = await pipeline.reconcile_orphan_objects(
            grace_period=timedelta(hours=24),
            limit=10,
            cursor=None,
        )
        assert first.next_cursor is not None
        result = await pipeline.reconcile_orphan_objects(
            grace_period=timedelta(hours=24),
            limit=10,
            cursor=first.next_cursor,
        )
    finally:
        await pipeline.aclose()

    assert result.deleted_count == 0
    assert repository.checked == [object_key]
    assert store.deleted == []


@pytest.mark.asyncio
async def test_object_reconciliation_isolates_delete_failures_without_logging_keys(
    caplog: pytest.LogCaptureFixture,
) -> None:
    now = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
    secret = "minio-delete-secret"
    first_key = "tmp/jobs/job-secret/1/parsed/text.txt"
    second_key = "tmp/jobs/job-safe/1/parsed/text.txt"
    store = _ReconciliationObjectStore(
        (
            OrphanCandidate(first_key, 1, None, now - timedelta(hours=25)),
            OrphanCandidate(second_key, 1, None, now - timedelta(hours=25)),
        )
    )
    store.failures[first_key] = RuntimeError(f"{secret}:{first_key}")
    repository = _ReconciliationRepository(set())
    pipeline = _reconciliation_pipeline(repository, store, now=now)
    try:
        with caplog.at_level(logging.WARNING, logger="rag_service.ingestion.pipeline"):
            result = await pipeline.reconcile_orphan_objects(
                grace_period=timedelta(hours=24),
                limit=10,
                cursor=None,
            )
    finally:
        await pipeline.aclose()

    assert result.deleted_count == 1
    assert result.failed_count == 1
    assert store.deleted == [second_key]
    assert secret not in caplog.text
    assert first_key not in caplog.text
    assert second_key not in caplog.text
    cleanup_records = [
        record
        for record in caplog.records
        if record.msg == "cleanup.action.completed"
        and getattr(record, "event", None) == "cleanup.action.completed"
        and getattr(record, "operation", None) == "orphan_cleanup"
        and getattr(record, "phase", None) == "temporary"
    ]
    assert len(cleanup_records) == 1
    assert getattr(cleanup_records[0], "operation", None) == "orphan_cleanup"
    assert getattr(cleanup_records[0], "phase", None) == "temporary"
    assert getattr(cleanup_records[0], "outcome", None) == "failed"
    assert getattr(cleanup_records[0], "count", None) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["hook", "authoritative_check", "delete"])
async def test_object_reconciliation_isolates_non_cancel_base_exception_per_candidate(
    failure_stage: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    now = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
    secret = "fatal-minio-candidate-secret"
    first_key = "tmp/jobs/job-fatal/1/parsed/text.txt"
    second_key = "tmp/jobs/job-safe/1/parsed/text.txt"

    class _FatalRepository(_ReconciliationRepository):
        async def object_key_cleanup_is_allowed(self, object_key: str) -> bool:
            if object_key == first_key and failure_stage == "authoritative_check":
                raise BaseException(f"{secret}:{object_key}")
            return await super().object_key_cleanup_is_allowed(object_key)

    hook_calls = 0

    async def hook(name: str) -> None:
        nonlocal hook_calls
        if name == "before_orphan_object_reference_check":
            hook_calls += 1
            if hook_calls == 1 and failure_stage == "hook":
                raise BaseException(f"{secret}:{first_key}")

    store = _ReconciliationObjectStore(
        (
            OrphanCandidate(first_key, 1, None, now - timedelta(hours=25)),
            OrphanCandidate(second_key, 1, None, now - timedelta(hours=25)),
        )
    )
    if failure_stage == "delete":
        store.failures[first_key] = BaseException(f"{secret}:{first_key}")
    pipeline = _reconciliation_pipeline(
        _FatalRepository(set()),
        store,
        now=now,
        hooks=IngestionPipelineHooks(hook),
    )
    try:
        with caplog.at_level(logging.WARNING, logger="rag_service.ingestion.pipeline"):
            result = await pipeline.reconcile_orphan_objects(
                grace_period=timedelta(hours=24),
                limit=10,
                cursor=None,
            )
    finally:
        await pipeline.aclose()

    assert result.deleted_count == 1
    assert result.failed_count == 1
    assert store.deleted == [second_key]
    assert secret not in caplog.text
    assert first_key not in caplog.text
    assert second_key not in caplog.text


@pytest.mark.asyncio
async def test_object_reconciliation_propagates_cancellation() -> None:
    now = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
    first_key = "tmp/jobs/job-a/1/parsed/text.txt"
    second_key = "tmp/jobs/job-b/1/parsed/text.txt"
    store = _ReconciliationObjectStore(
        (
            OrphanCandidate(first_key, 1, None, now - timedelta(hours=25)),
            OrphanCandidate(second_key, 1, None, now - timedelta(hours=25)),
        )
    )
    store.failures[first_key] = asyncio.CancelledError()
    pipeline = _reconciliation_pipeline(_ReconciliationRepository(set()), store, now=now)
    try:
        with pytest.raises(asyncio.CancelledError):
            await pipeline.reconcile_orphan_objects(
                grace_period=timedelta(hours=24),
                limit=10,
                cursor=None,
            )
    finally:
        await pipeline.aclose()

    assert store.deleted == []
