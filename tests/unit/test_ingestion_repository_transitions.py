from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import cast
from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.db.models.documents import Document, DocumentIndexState, DocumentVersion, Job
from rag_service.db.models.knowledge_bases import (
    KnowledgeBase,
    KnowledgeBaseIndexGeneration,
    KnowledgeBaseMutation,
)
from rag_service.db.models.providers import ModelProfile, ProviderConfig
from rag_service.indexing.identities import canonical_json_bytes, canonical_sha256, collection_name
from rag_service.ingestion.repositories import (
    ActivationStageInput,
    ChunkStageInput,
    DocumentActivationConflictError,
    ParseStageInput,
    SqlAlchemyIngestionPipelineRepository,
)
from rag_service.jobs.repositories import (
    ExhaustedJob,
    JobLease,
    LostLeaseError,
    SqlAlchemyJobRepository,
)


class _Result:
    def __init__(self, value: object) -> None:
        self._value = value

    def one_or_none(self) -> object:
        return self._value


class _SequenceSession:
    def __init__(
        self,
        *,
        scalars: list[object] | None = None,
        executes: list[object] | None = None,
        gets: list[object] | None = None,
    ) -> None:
        self.scalars = [] if scalars is None else list(scalars)
        self.executes = [] if executes is None else list(executes)
        self.gets = [] if gets is None else list(gets)
        self.added: list[object] = []

    async def scalar(self, _statement: object) -> object:
        assert self.scalars, "unexpected scalar call"
        return self.scalars.pop(0)

    async def execute(self, _statement: object) -> _Result:
        assert self.executes, "unexpected execute call"
        return _Result(self.executes.pop(0))

    async def get(self, _model: object, _identity: object) -> object:
        assert self.gets, "unexpected get call"
        return self.gets.pop(0)

    def add(self, value: object) -> None:
        self.added.append(value)


def _lease(
    *,
    job_id: UUID,
    version_id: UUID,
    generation_id: UUID | None,
    stage: str = "validate",
) -> JobLease:
    return JobLease(
        id=job_id,
        operation="ingest_document",
        target_type="document_version",
        target_id=version_id,
        target_revision=None,
        index_generation_id=generation_id,
        stage=stage,
        resume_stage=stage,
        progress_current=2,
        progress_total=2,
        attempt_count=1,
        max_attempts=5,
        lease_owner="worker-a",
        lease_epoch=3,
        lease_expires_at=datetime.now(UTC) + timedelta(seconds=30),
        cancel_requested_at=None,
    )


def _graph() -> tuple[
    KnowledgeBase,
    KnowledgeBaseIndexGeneration,
    Document,
    DocumentVersion,
    DocumentIndexState,
    Job,
    JobLease,
    ActivationStageInput,
]:
    knowledge_base_id = uuid4()
    generation_id = uuid4()
    document_id = uuid4()
    version_id = uuid4()
    job_id = uuid4()
    checksum = "a" * 64
    now = datetime.now(UTC)
    knowledge_base = KnowledgeBase(
        id=knowledge_base_id,
        status="active",
        active_index_generation_id=generation_id,
        mutation_revision=7,
    )
    generation = KnowledgeBaseIndexGeneration(
        id=generation_id,
        knowledge_base_id=knowledge_base_id,
        status="active",
    )
    document = Document(
        id=document_id,
        knowledge_base_id=knowledge_base_id,
        status="processing",
        current_version_id=None,
        pending_version_id=version_id,
        checksum_sha256=checksum,
        deleted_at=None,
    )
    version = DocumentVersion(
        id=version_id,
        document_id=document_id,
        base_version_id=None,
        status="indexing",
        source_checksum_sha256=checksum,
        detected_mime_type="text/plain",
        chunk_count=2,
        activated_at=None,
    )
    state = DocumentIndexState(
        document_version_id=version_id,
        index_generation_id=generation_id,
        status="validated",
        expected_point_count=2,
        actual_point_count=2,
        next_chunk_index=2,
        validated_at=now,
    )
    job = Job(
        id=job_id,
        knowledge_base_id=knowledge_base_id,
        operation="ingest_document",
        target_type="document_version",
        target_id=version_id,
        index_generation_id=generation_id,
        stage="activate",
        status="running",
        retryable=True,
        lease_owner="worker-a",
        lease_epoch=3,
        lease_expires_at=now + timedelta(seconds=30),
    )
    lease = _lease(
        job_id=job_id,
        version_id=version_id,
        generation_id=generation_id,
        stage="activate",
    )
    expected = ActivationStageInput(
        knowledge_base_id=knowledge_base_id,
        generation_id=generation_id,
        document_id=document_id,
        version_id=version_id,
        job_id=job_id,
        source_checksum_sha256=checksum,
        detected_mime_type="text/plain",
        chunk_count=2,
        expected_point_count=2,
        actual_point_count=2,
    )
    return knowledge_base, generation, document, version, state, job, lease, expected


def _indexed_graph() -> tuple[
    KnowledgeBase,
    KnowledgeBaseIndexGeneration,
    Document,
    DocumentVersion,
    DocumentIndexState,
    Job,
    ModelProfile,
    ProviderConfig,
]:
    knowledge_base, generation, document, version, state, job, _lease_value, _expected = _graph()
    credential_id = uuid4()
    provider_id = uuid4()
    profile_id = uuid4()
    actor_id = uuid4()
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
    config_hash = canonical_sha256(semantic)
    snapshot = {
        **semantic,
        "provider_config_id": str(provider_id),
        "credential_id": str(credential_id),
    }
    generation.embedding_profile_id = profile_id
    generation.embedding_config_snapshot = snapshot
    generation.embedding_config_hash = config_hash
    generation.index_profile_hash = config_hash
    generation.distance = "cosine"
    generation.qdrant_collection_name = collection_name(knowledge_base.id, generation.id)
    generation.filter_schema_snapshot = {"fields": []}
    generation.applied_filter_schema_revision = 0
    document.metadata_ = {}
    version.source_object_key = "kb/source.txt"
    version.source_extension = ".txt"
    version.parsed_object_key = "kb/parsed.txt"
    version.parsed_object_checksum_sha256 = "b" * 64
    version.parser_name = "plain_text_v1"
    version.parser_version = "1"
    version.parser_config = {}
    version.chunk_manifest_object_key = "kb/chunks.jsonl"
    version.chunk_manifest_checksum_sha256 = "c" * 64
    version.chunk_config_hash = "d" * 64
    version.chunker_name = "recursive_text_v1"
    version.chunker_version = "1"
    version.chunker_config = {
        "max_chunk_codepoints": 1200,
        "target_overlap_codepoints": 150,
    }
    version.created_at = datetime.now(UTC)
    state.status = "indexing"
    state.expected_point_count = version.chunk_count
    state.actual_point_count = None
    state.chunk_manifest_checksum_sha256 = version.chunk_manifest_checksum_sha256
    state.embedding_config_hash = config_hash
    state.next_chunk_index = cast(int, version.chunk_count)
    state.error_code = None
    state.safe_error_message = None
    state.validated_at = None
    job.knowledge_base_id = knowledge_base.id
    job.operation = "ingest_document"
    job.target_type = "document_version"
    job.target_id = version.id
    job.index_generation_id = generation.id
    job.actor_api_key_id = actor_id
    profile = ModelProfile(
        id=profile_id,
        provider_config_id=provider_id,
        capability="embedding",
        enabled=True,
        timeout_seconds=Decimal("5.000"),
        batch_size=2,
    )
    provider = ProviderConfig(
        id=provider_id,
        credential_id=credential_id,
        enabled=True,
        max_concurrency=2,
        requests_per_minute=60,
    )
    return knowledge_base, generation, document, version, state, job, profile, provider


@pytest.mark.asyncio
async def test_load_validate_and_commit_validation_snapshot_happy_path() -> None:
    knowledge_base, generation, document, version, state, job, profile, provider = _indexed_graph()
    load_session = _SequenceSession(
        executes=[(version, document, state)],
        gets=[knowledge_base, generation, job, profile, provider],
    )
    expected = await SqlAlchemyIngestionPipelineRepository(
        cast(AsyncSession, load_session)
    ).load_validate_stage(version.id, generation.id, job.id)

    assert expected.actor_api_key_id == job.actor_api_key_id
    assert expected.next_chunk_index == expected.chunk_count
    assert expected.embedding_config_hash == generation.embedding_config_hash
    assert expected.embedding_snapshot_canonical == canonical_json_bytes(
        generation.embedding_config_snapshot
    )

    count_session = _SequenceSession(scalars=[knowledge_base, generation, document, version, state])
    await SqlAlchemyIngestionPipelineRepository(
        cast(AsyncSession, count_session)
    ).commit_validate_count(expected, actual_count=expected.chunk_count)
    assert state.actual_point_count == expected.chunk_count

    validated_at = datetime.now(UTC)
    validate_session = _SequenceSession(
        scalars=[knowledge_base, generation, document, version, state, validated_at]
    )
    await SqlAlchemyIngestionPipelineRepository(
        cast(AsyncSession, validate_session)
    ).commit_validate_stage(expected, actual_count=expected.chunk_count)
    assert state.status == "validated"
    assert state.validated_at == validated_at


@pytest.mark.asyncio
@pytest.mark.parametrize("actual_count", [-1, 3])
async def test_commit_validate_count_rejects_invalid_or_conflicting_count(
    actual_count: int,
) -> None:
    knowledge_base, generation, document, version, state, job, profile, provider = _indexed_graph()
    load_session = _SequenceSession(
        executes=[(version, document, state)],
        gets=[knowledge_base, generation, job, profile, provider],
    )
    expected = await SqlAlchemyIngestionPipelineRepository(
        cast(AsyncSession, load_session)
    ).load_validate_stage(version.id, generation.id, job.id)
    if actual_count == 3:
        state.actual_point_count = 2
    session = _SequenceSession(scalars=[knowledge_base, generation, document, version, state])
    with pytest.raises(ValueError, match="ingestion stage state is invalid"):
        await SqlAlchemyIngestionPipelineRepository(
            cast(AsyncSession, session)
        ).commit_validate_count(expected, actual_count=actual_count)


@pytest.mark.asyncio
async def test_load_activation_stage_rehydrates_only_complete_validated_counts() -> None:
    knowledge_base, generation, document, version, state, job, _profile, _provider = (
        _indexed_graph()
    )
    state.actual_point_count = state.expected_point_count
    session = _SequenceSession(
        executes=[(version, document, state)],
        gets=[job],
    )
    expected = await SqlAlchemyIngestionPipelineRepository(
        cast(AsyncSession, session)
    ).load_activation_stage(version.id, generation.id, job.id)

    assert expected.knowledge_base_id == knowledge_base.id
    assert expected.document_id == document.id
    assert expected.expected_point_count == expected.actual_point_count == version.chunk_count


@pytest.mark.asyncio
async def test_stage_graph_and_loaders_reject_missing_or_drifted_rows() -> None:
    repository = SqlAlchemyIngestionPipelineRepository(
        cast(AsyncSession, _SequenceSession(executes=[None]))
    )
    with pytest.raises(ValueError, match="ingestion stage state is invalid"):
        await repository.load_activation_stage(uuid4(), uuid4(), uuid4())

    knowledge_base, generation, document, version, state, _job, _profile, _provider = (
        _indexed_graph()
    )
    document.current_version_id = uuid4()
    session = _SequenceSession(scalars=[knowledge_base, generation, document, version, state])
    with pytest.raises(ValueError, match="ingestion stage state is invalid"):
        await SqlAlchemyIngestionPipelineRepository(cast(AsyncSession, session))._lock_stage_graph(
            knowledge_base_id=knowledge_base.id,
            generation_id=generation.id,
            document_id=document.id,
            version_id=version.id,
        )


@pytest.mark.asyncio
async def test_parse_and_chunk_repository_transitions_accept_only_immutable_snapshot() -> None:
    knowledge_base, generation, document, version, state, _job, _lease_value, _activation = _graph()
    version.source_object_key = "kb/source.txt"
    version.source_extension = ".txt"
    version.parser_name = "plain_text_v1"
    version.parser_version = "1"
    version.parser_config = {}
    version.chunker_name = None
    version.chunker_version = None
    version.chunker_config = {}
    version.chunk_manifest_object_key = None
    version.chunk_manifest_checksum_sha256 = None
    version.chunk_config_hash = None
    version.status = "parsing"
    state.status = "queued"
    state.expected_point_count = None
    state.actual_point_count = None
    state.error_code = None
    state.validated_at = None
    state.chunk_manifest_checksum_sha256 = None
    state.embedding_config_hash = None
    state.next_chunk_index = 0
    state.safe_error_message = None
    parse_expected = ParseStageInput(
        knowledge_base_id=knowledge_base.id,
        generation_id=generation.id,
        document_id=document.id,
        version_id=version.id,
        source_object_key=version.source_object_key,
        source_checksum_sha256=version.source_checksum_sha256,
        source_extension=version.source_extension,
        parser_name=version.parser_name,
        parser_version=version.parser_version,
        parser_config={},
        version_status="parsing",
    )
    parse_session = _SequenceSession(scalars=[knowledge_base, generation, document, version, state])
    await SqlAlchemyIngestionPipelineRepository(
        cast(AsyncSession, parse_session)
    ).commit_parse_stage(
        parse_expected,
        parsed_object_key="kb/parsed.txt",
        parsed_checksum_sha256="b" * 64,
        parser_name="plain_text_v1",
        parser_version="1",
        parser_config={},
    )
    assert version.status == "chunking"
    assert version.parsed_object_key == "kb/parsed.txt"

    chunk_expected = ChunkStageInput(
        knowledge_base_id=knowledge_base.id,
        generation_id=generation.id,
        document_id=document.id,
        version_id=version.id,
        source_checksum_sha256=version.source_checksum_sha256,
        source_extension=version.source_extension,
        parsed_object_key=version.parsed_object_key,
        parsed_object_checksum_sha256=cast(str, version.parsed_object_checksum_sha256),
        parser_name=version.parser_name,
        parser_version=version.parser_version,
        parser_config={},
        chunker_name=None,
        chunker_version=None,
        chunker_config={},
        version_status="chunking",
        version_created_at=datetime.now(UTC),
    )
    chunk_session = _SequenceSession(scalars=[knowledge_base, generation, document, version, state])
    await SqlAlchemyIngestionPipelineRepository(
        cast(AsyncSession, chunk_session)
    ).commit_chunk_stage(
        chunk_expected,
        manifest_object_key="kb/chunks.jsonl",
        manifest_checksum_sha256="c" * 64,
        chunk_config_hash="d" * 64,
        chunker_name="recursive_text_v1",
        chunker_version="1",
        chunker_config={
            "max_chunk_codepoints": 1200,
            "target_overlap_codepoints": 150,
        },
        chunk_count=2,
    )
    assert version.status == "embedding"
    assert state.status == "embedding"
    assert state.expected_point_count == 2


@pytest.mark.asyncio
async def test_parse_started_and_embed_checkpoint_transitions_are_idempotent() -> None:
    knowledge_base, generation, document, version, state, _job, _lease_value, _activation = _graph()
    version.source_object_key = "kb/source.txt"
    version.source_extension = ".txt"
    version.parser_name = "plain_text_v1"
    version.parser_version = "1"
    version.parser_config = {}
    version.status = "uploaded"
    state.status = "queued"
    expected = ParseStageInput(
        knowledge_base_id=knowledge_base.id,
        generation_id=generation.id,
        document_id=document.id,
        version_id=version.id,
        source_object_key=version.source_object_key,
        source_checksum_sha256=version.source_checksum_sha256,
        source_extension=version.source_extension,
        parser_name=version.parser_name,
        parser_version=version.parser_version,
        parser_config={},
        version_status="uploaded",
    )
    session = _SequenceSession(scalars=[knowledge_base, generation, document, version, state])
    await SqlAlchemyIngestionPipelineRepository(cast(AsyncSession, session)).commit_parse_started(
        expected
    )
    assert version.status == "parsing"

    knowledge_base, generation, document, version, state, job, profile, provider = _indexed_graph()
    load_session = _SequenceSession(
        executes=[(version, document, state)],
        gets=[knowledge_base, generation, job, profile, provider],
    )
    embed_expected = await SqlAlchemyIngestionPipelineRepository(
        cast(AsyncSession, load_session)
    ).load_embed_index_stage(version.id, generation.id, job.id)
    checkpoint_session = _SequenceSession(
        scalars=[knowledge_base, generation, document, version, state]
    )
    await SqlAlchemyIngestionPipelineRepository(
        cast(AsyncSession, checkpoint_session)
    ).commit_embed_index_batch(
        embed_expected,
        next_chunk_index=embed_expected.chunk_count,
    )
    assert version.status == "indexing"
    assert state.status == "indexing"
    assert state.next_chunk_index == embed_expected.chunk_count


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    ["initial", "existing-current", "foreign-pending", "unsafe-graph"],
)
async def test_commit_terminal_failure_is_fail_closed_without_destroying_active_data(
    mode: str,
) -> None:
    knowledge_base, generation, document, version, state, job, lease, _expected = _graph()
    lease = _lease(
        job_id=job.id,
        version_id=version.id,
        generation_id=generation.id,
    )
    if mode == "existing-current":
        document.status = "active"
        document.current_version_id = uuid4()
    elif mode == "foreign-pending":
        document.pending_version_id = uuid4()
    elif mode == "unsafe-graph":
        generation.knowledge_base_id = uuid4()
    finished_at = datetime.now(UTC)
    session = _SequenceSession(
        executes=[(document.id, knowledge_base.id)],
        scalars=[knowledge_base, generation, document, version, state, job, finished_at],
    )

    await SqlAlchemyIngestionPipelineRepository(
        cast(AsyncSession, session)
    ).commit_terminal_failure(
        lease,
        retryable=True,
        error_code="QDRANT_UNAVAILABLE",
        safe_error_message="Qdrant unavailable",
    )

    assert job.status == "failed"
    assert job.retryable is True
    assert job.error_code == "QDRANT_UNAVAILABLE"
    assert job.lease_owner is None and job.lease_expires_at is None
    assert job.finished_at == finished_at

    if mode == "initial":
        assert document.status == "failed"
        assert document.pending_version_id == version.id
        assert version.status == "failed"
        assert state.status == "failed"
        assert state.error_code == "QDRANT_UNAVAILABLE"
    elif mode == "existing-current":
        assert document.status == "active"
        assert document.current_version_id is not None
        assert document.pending_version_id is None
        assert version.status == "failed" and state.status == "failed"
    elif mode == "foreign-pending":
        assert document.status == "processing"
        assert document.current_version_id is None
        assert document.pending_version_id not in {None, version.id}
        assert version.status == "failed" and state.status == "failed"
    else:
        assert document.status == "processing"
        assert document.pending_version_id == version.id
        assert version.status == "indexing" and state.status == "validated"


@pytest.mark.asyncio
async def test_commit_exhausted_failure_uses_domain_first_locks_and_atomic_facts() -> None:
    knowledge_base, generation, document, version, state, job, _lease_value, _expected = _graph()
    expired_at = datetime.now(UTC) - timedelta(seconds=1)
    job.attempt_count = 5
    job.max_attempts = 5
    job.lease_expires_at = expired_at
    candidate = ExhaustedJob(
        id=job.id,
        knowledge_base_id=knowledge_base.id,
        operation=job.operation,
        target_type=job.target_type,
        target_id=job.target_id,
        target_revision=job.target_revision,
        index_generation_id=job.index_generation_id,
        stage=job.stage,
        status=job.status,
        attempt_count=5,
        max_attempts=5,
        next_retry_at=job.next_retry_at,
        lease_owner=job.lease_owner,
        lease_epoch=job.lease_epoch,
        lease_expires_at=expired_at,
        cancel_requested_at=job.cancel_requested_at,
    )
    finished_at = datetime.now(UTC)
    session = _SequenceSession(
        executes=[(document.id, knowledge_base.id)],
        scalars=[knowledge_base, generation, document, version, state, job, finished_at],
    )

    await SqlAlchemyIngestionPipelineRepository(
        cast(AsyncSession, session)
    ).commit_exhausted_failure(candidate)

    assert document.status == "failed" and document.pending_version_id == version.id
    assert version.status == "failed"
    assert state.status == "failed" and state.error_code == "JOB_ATTEMPTS_EXHAUSTED"
    assert job.status == "failed" and job.retryable is False
    assert job.error_code == "JOB_ATTEMPTS_EXHAUSTED"
    assert job.lease_owner is None and job.lease_expires_at is None
    assert job.finished_at == finished_at

    mismatched = replace(candidate, knowledge_base_id=uuid4())
    mismatch_session = _SequenceSession(executes=[(document.id, knowledge_base.id)])
    with pytest.raises(LostLeaseError):
        await SqlAlchemyIngestionPipelineRepository(
            cast(AsyncSession, mismatch_session)
        ).commit_exhausted_failure(mismatched)


@pytest.mark.asyncio
async def test_commit_terminal_failure_rejects_invalid_input_and_stale_identity() -> None:
    _kb, generation, _document, version, _state, job, lease, _expected = _graph()
    repository = SqlAlchemyIngestionPipelineRepository(cast(AsyncSession, _SequenceSession()))
    invalid_calls = (
        {"retryable": 1, "error_code": "E", "safe_error_message": "safe"},
        {"retryable": False, "error_code": "", "safe_error_message": "safe"},
        {"retryable": False, "error_code": "e" * 65, "safe_error_message": "safe"},
        {"retryable": False, "error_code": "E", "safe_error_message": ""},
        {"retryable": False, "error_code": "E", "safe_error_message": "x" * 501},
    )
    for kwargs in invalid_calls:
        with pytest.raises(ValueError, match="terminal ingestion failure is invalid"):
            await repository.commit_terminal_failure(lease, **kwargs)  # type: ignore[arg-type]

    missing_generation = _lease(
        job_id=job.id,
        version_id=version.id,
        generation_id=None,
    )
    with pytest.raises(ValueError, match="terminal ingestion failure is invalid"):
        await repository.commit_terminal_failure(
            missing_generation,
            retryable=False,
            error_code="E",
            safe_error_message="safe",
        )

    stale = _SequenceSession(executes=[None])
    with pytest.raises(LostLeaseError):
        await SqlAlchemyIngestionPipelineRepository(
            cast(AsyncSession, stale)
        ).commit_terminal_failure(
            _lease(job_id=job.id, version_id=version.id, generation_id=generation.id),
            retryable=False,
            error_code="E",
            safe_error_message="safe",
        )


@pytest.mark.asyncio
async def test_commit_terminal_failure_rejects_missing_locked_graph() -> None:
    knowledge_base, generation, document, version, state, _job, lease, _expected = _graph()
    session = _SequenceSession(
        executes=[(document.id, knowledge_base.id)],
        scalars=[knowledge_base, generation, document, version, state, None],
    )
    with pytest.raises(LostLeaseError):
        await SqlAlchemyIngestionPipelineRepository(
            cast(AsyncSession, session)
        ).commit_terminal_failure(
            lease,
            retryable=False,
            error_code="E",
            safe_error_message="safe",
        )


@pytest.mark.asyncio
async def test_commit_activation_atomically_publishes_and_finishes_job() -> None:
    knowledge_base, generation, document, version, state, job, lease, expected = _graph()
    activated_at = datetime.now(UTC)
    session = _SequenceSession(
        scalars=[
            knowledge_base,
            generation,
            document,
            version,
            state,
            job,
            document.id,
            version.id,
            state.document_version_id,
            activated_at,
        ]
    )

    await SqlAlchemyIngestionPipelineRepository(cast(AsyncSession, session)).commit_activation(
        expected,
        lease,
    )

    assert version.status == "ready" and version.activated_at == activated_at
    assert document.status == "active"
    assert document.current_version_id == version.id
    assert document.pending_version_id is None
    assert knowledge_base.mutation_revision == 8
    assert job.status == "succeeded"
    assert job.retryable is False
    assert job.lease_owner is None and job.lease_expires_at is None
    assert len(session.added) == 1
    mutation = session.added[0]
    assert isinstance(mutation, KnowledgeBaseMutation)
    assert mutation.revision == 8
    assert mutation.target_id == version.id


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["stale-job", "document", "version", "state", "generation"])
async def test_commit_activation_distinguishes_stale_fence_from_business_conflict(
    mode: str,
) -> None:
    knowledge_base, generation, document, version, state, job, lease, expected = _graph()
    if mode == "stale-job":
        scalars: list[object] = [knowledge_base, generation, document, version, state, None]
        expected_error: type[Exception] = LostLeaseError
    else:
        matches: list[object] = [document.id, version.id, state.document_version_id]
        match_index = {"document": 0, "version": 1, "state": 2}.get(mode)
        if match_index is not None:
            matches[match_index] = None
        if mode == "generation":
            generation.status = "retiring"
        scalars = [knowledge_base, generation, document, version, state, job, *matches]
        expected_error = DocumentActivationConflictError
    session = _SequenceSession(scalars=scalars)

    with pytest.raises(expected_error):
        await SqlAlchemyIngestionPipelineRepository(cast(AsyncSession, session)).commit_activation(
            expected, lease
        )

    assert version.status == "indexing"
    assert document.status == "processing"
    assert job.status == "running"
    assert session.added == []


@pytest.mark.asyncio
async def test_sql_job_finalize_domain_requires_callback_to_finish_the_fenced_job() -> None:
    _kb, generation, _document, version, _state, job, lease, _expected = _graph()
    lease = _lease(job_id=job.id, version_id=version.id, generation_id=generation.id)
    calls = 0

    async def action(_session: AsyncSession) -> str:
        nonlocal calls
        calls += 1
        return "done"

    completed = _SequenceSession(scalars=["failed"])
    repository = SqlAlchemyJobRepository(cast(AsyncSession, completed))
    assert await repository.finalize_domain(lease, action) == "done"
    assert calls == 1

    stale = _SequenceSession(scalars=[None])
    with pytest.raises(LostLeaseError):
        await SqlAlchemyJobRepository(cast(AsyncSession, stale)).finalize_domain(lease, action)
    assert calls == 2
