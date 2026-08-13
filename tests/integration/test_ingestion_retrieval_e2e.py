import asyncio
import hashlib
import importlib
import json
import socket
import ssl
import subprocess
import tempfile
import threading
import time
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, Protocol, TypeGuard, cast
from uuid import UUID, uuid4

import httpx
import pytest
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from minio import Minio
from minio.datatypes import ListMultipartUploadsResult
from pydantic import SecretStr
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from urllib3 import PoolManager

from fixtures.provider_stub import PROVIDER_STUB_SECRET, RunningProviderStub
from rag_service.api.errors import BusinessError
from rag_service.auth.dependencies import require_agent_principal
from rag_service.auth.policies import AgentPrincipal, Capability
from rag_service.config import Settings
from rag_service.db.models.auth import ApiKey, ApiKeyKnowledgeBaseScope
from rag_service.db.models.documents import (
    Document,
    DocumentIndexState,
    DocumentUploadIdempotency,
    DocumentVersion,
    Job,
)
from rag_service.db.models.knowledge_bases import (
    KnowledgeBase,
    KnowledgeBaseIndexGeneration,
    KnowledgeBaseMutation,
)
from rag_service.db.models.observability import ProviderUsage, QueryLog
from rag_service.db.models.providers import ModelProfile, ProviderConfig, ProviderCredential
from rag_service.db.session import Database
from rag_service.indexing.generation_services import payload_indexes_for_filter_snapshot
from rag_service.indexing.identities import canonical_sha256, collection_name, point_id
from rag_service.indexing.qdrant import (
    CollectionSpec,
    FakeQdrantClient,
    QdrantPoint,
    QdrantSearchFilter,
    QdrantSearchPoint,
    QdrantTransientError,
)
from rag_service.infrastructure.minio_store import (
    ArtifactChecksumConflict,
    MinioObjectStore,
    ObjectStoreError,
    UploadLimitExceeded,
)
from rag_service.ingestion.artifacts import (
    chunks_object_key,
    parsed_text_object_key,
    source_object_key,
    temporary_object_key,
)
from rag_service.ingestion.pipeline import (
    MAX_CHUNK_MANIFEST_BYTES,
    IngestionPipeline,
    IngestionPipelineHooks,
    IngestionPipelineRepository,
    PipelineEmbeddingGateway,
    PipelineObjectStore,
    PipelineProviderUsageSink,
)
from rag_service.ingestion.repositories import (
    SqlAlchemyIngestionPipelineRepository,
    SqlAlchemyUploadRepository,
    UploadPreflight,
    UploadReservation,
    UploadReservationResult,
)
from rag_service.ingestion.routes import get_document_upload_service
from rag_service.ingestion.schemas import UploadAccepted, UploadForm
from rag_service.ingestion.services import DocumentUploadService, SourceUpload
from rag_service.jobs.repositories import (
    JobLease,
    JobRepository,
    LostLeaseError,
    SqlAlchemyJobRepository,
)
from rag_service.jobs.runner import (
    ExponentialBackoff,
    JobExecutionContext,
    JobRunner,
    PermanentJobError,
    RepositoryContextFactory,
    RetryableJobError,
)
from rag_service.main import create_app
from rag_service.observability.repositories import (
    SqlAlchemyProviderUsageSink,
    SqlAlchemyQueryLogSink,
)
from rag_service.providers.embeddings import (
    EmbeddingAttempt,
    EmbeddingAttemptObserver,
    EmbeddingConfigSnapshot,
    EmbeddingGatewayError,
    EmbeddingOperationalConfig,
    EmbeddingResult,
)
from rag_service.readiness import ReadinessScope, ReadinessSnapshot
from rag_service.retrieval.repositories import SqlAlchemyRetrievalRepository
from rag_service.retrieval.routes import get_retrieval_service
from rag_service.retrieval.schemas import SearchFilters, SearchRequest
from rag_service.retrieval.services import SearchService


class _MinioConnection(Protocol):
    endpoint: str
    url: str
    access_key: str
    secret_key: str
    bucket: str
    secure: bool


class _RedisConnection(Protocol):
    url: str


class _QdrantConnection(Protocol):
    url: str


class _PostgresConnection(Protocol):
    async_url: str


class _FullStackConnections(Protocol):
    postgres: _PostgresConnection
    minio: _MinioConnection
    redis: _RedisConnection
    qdrant: _QdrantConnection
    provider: RunningProviderStub


MinioHttpPoolFactory = Callable[[], PoolManager]


def _assert_sanitized_process(
    completed: subprocess.CompletedProcess[str],
    *,
    returncode: int,
    stdout: str,
    stderr: str,
) -> None:
    if completed.returncode != returncode:
        raise AssertionError(f"PROCESS_RETURN_CODE_MISMATCH:{completed.returncode}") from None
    if completed.stdout != stdout:
        raise AssertionError("PROCESS_STDOUT_MISMATCH") from None
    if completed.stderr != stderr:
        raise AssertionError("PROCESS_STDERR_MISMATCH") from None


def _assert_sanitized_http_status(response: httpx.Response, expected: int) -> None:
    if response.status_code != expected:
        raise AssertionError(f"HTTP_STATUS_MISMATCH:{response.status_code}") from None


def _assert_protected_content_absent(content: str, protected: str) -> None:
    if protected in content:
        raise AssertionError("PROTECTED_CONTENT_EXPOSED") from None


@pytest.mark.acceptance
def test_task16c_diagnostic_assertions_never_echo_protected_content() -> None:
    protected = "PROTECTED_CONTENT_SENTINEL"
    completed = subprocess.CompletedProcess(
        args=["safe-command"],
        returncode=0,
        stdout=protected,
        stderr="",
    )

    with pytest.raises(AssertionError) as process_error:
        _assert_sanitized_process(
            completed,
            returncode=0,
            stdout="expected",
            stderr="",
        )
    if str(process_error.value) != "PROCESS_STDOUT_MISMATCH":
        pytest.fail("SANITIZED_PROCESS_MESSAGE_MISMATCH", pytrace=False)
    if protected in str(process_error.value):
        pytest.fail("SANITIZED_PROCESS_EXPOSED_CONTENT", pytrace=False)

    response = httpx.Response(503, text=protected)
    with pytest.raises(AssertionError) as response_error:
        _assert_sanitized_http_status(response, 202)
    if str(response_error.value) != "HTTP_STATUS_MISMATCH:503":
        pytest.fail("SANITIZED_HTTP_MESSAGE_MISMATCH", pytrace=False)
    if protected in str(response_error.value):
        pytest.fail("SANITIZED_HTTP_EXPOSED_CONTENT", pytrace=False)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.acceptance
async def test_full_stack_fixture_probes_real_dependencies_and_redacts_connections(
    full_stack_connections: _FullStackConnections,
    minio_http_pool_factory: MinioHttpPoolFactory,
) -> None:
    rendered = "\n".join(
        repr(value)
        for value in (
            full_stack_connections,
            full_stack_connections.minio,
            full_stack_connections.redis,
            full_stack_connections.qdrant,
            full_stack_connections.provider,
        )
    )
    assert "<redacted>" in rendered
    assert full_stack_connections.minio.access_key not in rendered
    assert full_stack_connections.minio.secret_key not in rendered
    assert PROVIDER_STUB_SECRET not in rendered

    database = Database(full_stack_connections.postgres.async_url)
    try:
        await database.ping()
    finally:
        await database.close()

    minio_pool = minio_http_pool_factory()
    try:
        minio = Minio(
            full_stack_connections.minio.endpoint,
            access_key=full_stack_connections.minio.access_key,
            secret_key=full_stack_connections.minio.secret_key,
            secure=full_stack_connections.minio.secure,
            http_client=minio_pool,
        )
        assert await asyncio.to_thread(
            minio.bucket_exists,
            full_stack_connections.minio.bucket,
        )
    finally:
        minio_pool.clear()

    redis = Redis.from_url(full_stack_connections.redis.url, decode_responses=True)
    try:
        assert await redis.ping() is True
    finally:
        await redis.aclose()

    qdrant = AsyncQdrantClient(url=full_stack_connections.qdrant.url, timeout=5)
    try:
        assert isinstance((await qdrant.get_collections()).collections, list)
    finally:
        await qdrant.close()

    async with httpx.AsyncClient(
        verify=ssl.create_default_context(
            cafile=full_stack_connections.provider.ca_bundle,
        ),
        trust_env=False,
        timeout=5,
    ) as provider:
        response = await provider.post(
            f"{full_stack_connections.provider.loopback_base_url}/embeddings",
            headers={"Authorization": f"Bearer {PROVIDER_STUB_SECRET}"},
            json={"model": "fixture-probe", "input": ["deterministic fixture probe"]},
        )
    assert response.status_code == 200
    assert len(response.json()["data"][0]["embedding"]) == 3


class MemoryObjectStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.publish_reused: list[bool] = []
        self.delete_calls: list[str] = []

    async def upload_stream(
        self,
        object_key: str,
        stream: AsyncIterable[bytes],
        *,
        content_type: str,
        max_bytes: int,
    ) -> object:
        del content_type
        content = bytearray()
        async for chunk in stream:
            content.extend(chunk)
            assert len(content) <= max_bytes
        stored = bytes(content)
        self.objects[object_key] = stored
        return SimpleNamespace(
            object_key=object_key,
            size=len(stored),
            checksum_sha256=hashlib.sha256(stored).hexdigest(),
        )

    async def publish_temp(
        self,
        temp_key: str,
        final_key: str,
        *,
        expected_size: int,
        expected_checksum: str,
    ) -> object:
        try:
            content = self.objects[temp_key]
            assert len(content) == expected_size
            assert hashlib.sha256(content).hexdigest() == expected_checksum
            existing = self.objects.get(final_key)
            if existing is not None:
                if (
                    len(existing) != expected_size
                    or hashlib.sha256(existing).hexdigest() != expected_checksum
                ):
                    raise ArtifactChecksumConflict
                reused = True
            else:
                self.objects[final_key] = content
                reused = False
            self.publish_reused.append(reused)
            return SimpleNamespace(
                object_key=final_key,
                size=expected_size,
                checksum_sha256=expected_checksum,
                reused=reused,
            )
        finally:
            await self.delete_best_effort(temp_key)

    async def read_bytes(
        self,
        object_key: str,
        *,
        expected_checksum: str,
        max_bytes: int,
    ) -> bytes:
        try:
            content = self.objects[object_key]
        except KeyError:
            raise ObjectStoreError("OBJECT_VERIFICATION_FAILED", retryable=False) from None
        if len(content) > max_bytes or hashlib.sha256(content).hexdigest() != expected_checksum:
            raise ObjectStoreError("OBJECT_VERIFICATION_FAILED", retryable=False)
        return content

    async def read_stream(
        self,
        object_key: str,
        *,
        expected_checksum: str,
        max_bytes: int,
    ) -> AsyncIterator[bytes]:
        content = await self.read_bytes(
            object_key,
            expected_checksum=expected_checksum,
            max_bytes=max_bytes,
        )
        for offset in range(0, len(content), 97):
            yield content[offset : offset + 97]

    async def delete_best_effort(self, object_key: str) -> bool:
        self.delete_calls.append(object_key)
        self.objects.pop(object_key, None)
        return True

    async def verify_object(
        self,
        object_key: str,
        *,
        expected_size: int,
        expected_checksum: str,
    ) -> object:
        try:
            content = self.objects[object_key]
        except KeyError:
            raise ObjectStoreError("OBJECT_VERIFICATION_FAILED", retryable=False) from None
        if (
            len(content) != expected_size
            or hashlib.sha256(content).hexdigest() != expected_checksum
        ):
            raise ObjectStoreError("OBJECT_VERIFICATION_FAILED", retryable=False)
        return SimpleNamespace(
            object_key=object_key,
            size=expected_size,
            checksum_sha256=expected_checksum,
        )


class TrackingMemoryObjectStore(MemoryObjectStore):
    def __init__(self) -> None:
        super().__init__()
        self.opened_streams = 0
        self.closed_streams = 0

    async def read_stream(
        self,
        object_key: str,
        *,
        expected_checksum: str,
        max_bytes: int,
    ) -> AsyncIterator[bytes]:
        content = await self.read_bytes(
            object_key,
            expected_checksum=expected_checksum,
            max_bytes=max_bytes,
        )
        self.opened_streams += 1
        try:
            for offset in range(0, len(content), 97):
                yield content[offset : offset + 97]
        finally:
            self.closed_streams += 1


class SimulatedStageCrash(RuntimeError):
    pass


class BeforeCommitRepository(SqlAlchemyJobRepository):
    async def advance_stage[Result](
        self,
        lease: JobLease,
        action: Callable[[AsyncSession], Awaitable[Result]],
        *,
        stage: str,
        resume_stage: str,
        progress_current: int,
        progress_total: int | None,
    ) -> tuple[Result, JobLease]:
        del lease, action, stage, resume_stage, progress_current, progress_total
        raise SimulatedStageCrash("simulated crash before fencing commit")


class StageCrashStore(MemoryObjectStore):
    def __init__(self) -> None:
        super().__init__()
        self._target_suffix: str | None = None
        self._position: str | None = None

    def arm(self, target_suffix: str, position: str) -> None:
        assert position in {"before_copy", "after_copy"}
        self._target_suffix = target_suffix
        self._position = position

    async def publish_temp(
        self,
        temp_key: str,
        final_key: str,
        *,
        expected_size: int,
        expected_checksum: str,
    ) -> object:
        should_crash = self._target_suffix is not None and final_key.endswith(self._target_suffix)
        if should_crash and self._position == "before_copy":
            self._target_suffix = None
            self._position = None
            raise SimulatedStageCrash("simulated crash before artifact copy")
        published = await super().publish_temp(
            temp_key,
            final_key,
            expected_size=expected_size,
            expected_checksum=expected_checksum,
        )
        if should_crash and self._position == "after_copy":
            self._target_suffix = None
            self._position = None
            raise SimulatedStageCrash("simulated crash after artifact copy")
        return published


class ReclaimAfterPublishStore(MemoryObjectStore):
    def __init__(self, database: Database) -> None:
        super().__init__()
        self._database = database
        self._target_suffix: str | None = None
        self._job_id: UUID | None = None
        self.reclaimed_lease: JobLease | None = None

    def arm(self, target_suffix: str, job_id: UUID) -> None:
        self._target_suffix = target_suffix
        self._job_id = job_id

    async def publish_temp(
        self,
        temp_key: str,
        final_key: str,
        *,
        expected_size: int,
        expected_checksum: str,
    ) -> object:
        published = await super().publish_temp(
            temp_key,
            final_key,
            expected_size=expected_size,
            expected_checksum=expected_checksum,
        )
        if self._target_suffix is not None and final_key.endswith(self._target_suffix):
            assert self._job_id is not None
            self._target_suffix = None
            self.reclaimed_lease = await expire_and_reclaim_upload_job(
                self._database,
                self._job_id,
                lease_owner="pipeline-worker-b",
            )
        return published


class CapturingRepository:
    def __init__(self, kb_id: UUID) -> None:
        self.preflight_value = UploadPreflight(
            kb_id,
            uuid4(),
            {"fields": []},
            0,
            0,
        )
        self.reservations: list[UploadReservation] = []

    async def preflight(self, knowledge_base_id: UUID) -> UploadPreflight:
        assert knowledge_base_id == self.preflight_value.knowledge_base_id
        return self.preflight_value

    async def reserve(self, reservation: UploadReservation) -> UploadReservationResult:
        self.reservations.append(reservation)
        return UploadReservationResult.created(
            reservation.document_id,
            reservation.version_id,
            reservation.job_id,
        )


class NoopNotifier:
    async def notify(self, job_id: UUID) -> bool:
        del job_id
        return True


class DeterministicEmbeddingGateway:
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
        attempt_number = len(self.calls)
        if attempt_observer is not None:
            observed = attempt_observer(
                EmbeddingAttempt(
                    provider_identifier=snapshot.provider_type,
                    model_identifier=snapshot.model_name,
                    route_identifier=None,
                    provider_request_id=f"provider-request-{attempt_number}",
                    input_tokens=sum(len(value) for value in batch),
                    output_tokens=0,
                    cost_micros=0,
                    currency="USD",
                    latency_ms=attempt_number,
                    status="succeeded",
                    error_code=None,
                    degraded=False,
                )
            )
            if observed is not None:
                await observed
        vectors = tuple(
            tuple(float((index + offset) % 7) / 7.0 for offset in range(snapshot.dimension))
            for index, _value in enumerate(batch)
        )
        return EmbeddingResult(vectors=vectors, usage={"prompt_tokens": len(batch)})


async def _emit_embedding_attempt(
    observer: EmbeddingAttemptObserver | None,
    snapshot: EmbeddingConfigSnapshot,
    *,
    status: Literal["succeeded", "failed", "rate_limited", "timeout", "cancelled"],
    error_code: str | None,
) -> None:
    if observer is None:
        return
    observed = observer(
        EmbeddingAttempt(
            provider_identifier=snapshot.provider_type,
            model_identifier=snapshot.model_name,
            route_identifier=None,
            provider_request_id="provider-request-failure",
            input_tokens=1,
            output_tokens=0,
            cost_micros=0,
            currency="USD",
            latency_ms=3,
            status=status,
            error_code=error_code,
            degraded=False,
        )
    )
    if observed is not None:
        await observed


class FailingEmbeddingGateway:
    def __init__(
        self,
        *,
        code: str,
        retryable: bool,
        status: Literal["failed", "rate_limited", "timeout"],
    ) -> None:
        self.code = code
        self.retryable = retryable
        self.status = status
        self.calls = 0

    async def embed(
        self,
        *,
        snapshot: EmbeddingConfigSnapshot,
        operational: EmbeddingOperationalConfig,
        inputs: Sequence[str],
        attempt_observer: EmbeddingAttemptObserver | None = None,
    ) -> EmbeddingResult:
        del operational, inputs
        self.calls += 1
        await _emit_embedding_attempt(
            attempt_observer,
            snapshot,
            status=self.status,
            error_code=self.code,
        )
        raise EmbeddingGatewayError(
            self.code, "Embedding provider unavailable", retryable=self.retryable
        )


class InvalidEmbeddingGateway:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    async def embed(
        self,
        *,
        snapshot: EmbeddingConfigSnapshot,
        operational: EmbeddingOperationalConfig,
        inputs: Sequence[str],
        attempt_observer: EmbeddingAttemptObserver | None = None,
    ) -> EmbeddingResult:
        del operational
        await _emit_embedding_attempt(
            attempt_observer,
            snapshot,
            status="succeeded",
            error_code=None,
        )
        if self.mode == "count":
            return EmbeddingResult(vectors=(), usage={})
        return EmbeddingResult(
            vectors=tuple((0.1,) for _value in inputs),
            usage={},
        )


class FailingUsageSink:
    def __init__(self, failure: BaseException) -> None:
        self.failure = failure
        self.calls = 0

    async def record(self, _context: object, _attempt: object) -> None:
        self.calls += 1
        raise self.failure


def capturing_service(
    kb_id: UUID,
) -> tuple[DocumentUploadService, CapturingRepository, MemoryObjectStore]:
    repository = CapturingRepository(kb_id)
    store = MemoryObjectStore()
    return (
        DocumentUploadService(
            repository=repository,
            object_store=store,
            notifier=NoopNotifier(),
            max_upload_bytes=50 * 1024 * 1024,
        ),
        repository,
        store,
    )


class FailingUploadService:
    async def upload_multipart(self, **kwargs: object) -> UploadAccepted:
        del kwargs
        raise RuntimeError("backend-secret")


def upload_service_dependency(service: object) -> Callable[[], object]:
    def dependency() -> object:
        return service

    return dependency


@dataclass(frozen=True, slots=True)
class SeededUploadContext:
    knowledge_base_id: UUID
    generation_id: UUID
    actor: AgentPrincipal


async def seed_upload_context(database: Database) -> SeededUploadContext:
    knowledge_base_id = uuid4()
    generation_id = uuid4()
    key_id = uuid4()
    public_id = "YWdlbnQtdXBsb2FkLXBvc3RncmVz"
    now = datetime.now(UTC)
    knowledge_base = KnowledgeBase(
        id=knowledge_base_id,
        name="Upload integration",
        description=None,
        status="active",
        metadata_={},
        filter_schema={"fields": []},
        resource_revision=1,
        mutation_revision=0,
        filter_schema_revision=0,
        active_index_generation_id=generation_id,
        pending_index_generation_id=None,
    )
    generation = KnowledgeBaseIndexGeneration(
        id=generation_id,
        knowledge_base_id=knowledge_base_id,
        embedding_profile_id=None,
        sparse_profile_id=None,
        index_profile_hash="a" * 64,
        qdrant_collection_name=f"upload_{generation_id.hex}",
        status="active",
        rebuild_snapshot_at=now,
        caught_up_revision=0,
        validated_revision=0,
        validation_manifest_hash="b" * 64,
        expected_point_count=0,
        actual_point_count=0,
        validated_at=now,
        activated_at=now,
        retired_at=None,
        distance="cosine",
        embedding_config_snapshot={},
        filter_schema_snapshot={"fields": []},
        applied_filter_schema_revision=0,
        embedding_config_hash="c" * 64,
        safe_error_code=None,
        safe_error_message=None,
    )
    api_key = ApiKey(
        id=key_id,
        public_id=public_id,
        secret_digest=b"x" * 32,
        key_type="agent",
        name="Upload agent",
        status="active",
        capabilities=["ingest"],
        raw_file_read=False,
        requests_per_minute=60,
        max_concurrency=4,
        not_before=None,
        expires_at=None,
        resource_revision=1,
        created_by_api_key_id=None,
        revoked_by_api_key_id=None,
        revoked_at=None,
    )
    scope = ApiKeyKnowledgeBaseScope(
        api_key_id=key_id,
        knowledge_base_id=knowledge_base_id,
    )
    async with database.sessions() as session, session.begin():
        session.add_all([knowledge_base, api_key])
        await session.flush()
        session.add(generation)
        await session.flush()
        session.add(scope)
    return SeededUploadContext(
        knowledge_base_id,
        generation_id,
        AgentPrincipal(
            key_id=key_id,
            public_id=public_id,
            capabilities=frozenset({Capability.INGEST}),
            knowledge_base_ids=frozenset({knowledge_base_id}),
            query_profile_ids=frozenset(),
            default_query_profile_id=None,
            raw_file_read=False,
            requests_per_minute=60,
            max_concurrency=4,
        ),
    )


async def configure_embedding_generation(
    database: Database,
    seeded: SeededUploadContext,
    *,
    batch_size: int = 2,
    filter_snapshot: dict[str, object] | None = None,
) -> tuple[UUID, UUID, CollectionSpec]:
    credential_id = uuid4()
    provider_config_id = uuid4()
    profile_id = uuid4()
    snapshot = {"fields": []} if filter_snapshot is None else filter_snapshot
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
    now = datetime.now(UTC)
    async with database.sessions() as session, session.begin():
        session.add(
            ProviderCredential(
                id=credential_id,
                name=f"Ingestion credential {credential_id.hex[:8]}",
                ciphertext=b"ciphertext",
                nonce=b"n" * 12,
                algorithm="AES-256-GCM",
                key_version="test-v1",
                resource_revision=1,
            )
        )
        await session.flush()
        session.add(
            ProviderConfig(
                id=provider_config_id,
                name=f"Ingestion provider {provider_config_id.hex[:8]}",
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
                name=f"Ingestion profile {profile_id.hex[:8]}",
                capability="embedding",
                provider_config_id=provider_config_id,
                model_name="currently-editable-and-ignored",
                dimension=9,
                max_input_tokens=99,
                batch_size=batch_size,
                timeout_seconds=Decimal("8.000"),
                vector_config={"currently": "ignored"},
                enabled=True,
                resource_revision=1,
            )
        )
        generation = await session.get(KnowledgeBaseIndexGeneration, seeded.generation_id)
        knowledge_base = await session.get(KnowledgeBase, seeded.knowledge_base_id)
        assert generation is not None and knowledge_base is not None
        generation.embedding_profile_id = profile_id
        generation.index_profile_hash = semantic_hash
        generation.qdrant_collection_name = collection_name(
            seeded.knowledge_base_id,
            seeded.generation_id,
        )
        generation.distance = "cosine"
        generation.embedding_config_snapshot = persisted_snapshot
        generation.embedding_config_hash = semantic_hash
        generation.filter_schema_snapshot = snapshot
        generation.applied_filter_schema_revision = knowledge_base.filter_schema_revision
        knowledge_base.filter_schema = snapshot
    spec = CollectionSpec(
        collection_name(seeded.knowledge_base_id, seeded.generation_id),
        3,
        "cosine",
        payload_indexes_for_filter_snapshot(snapshot),
    )
    return profile_id, provider_config_id, spec


async def seed_second_knowledge_base(
    database: Database,
    seeded: SeededUploadContext,
) -> SeededUploadContext:
    knowledge_base_id = uuid4()
    generation_id = uuid4()
    now = datetime.now(UTC)
    knowledge_base = KnowledgeBase(
        id=knowledge_base_id,
        name="Second upload integration",
        description=None,
        status="active",
        metadata_={},
        filter_schema={"fields": []},
        resource_revision=1,
        mutation_revision=0,
        filter_schema_revision=0,
        active_index_generation_id=generation_id,
        pending_index_generation_id=None,
    )
    generation = KnowledgeBaseIndexGeneration(
        id=generation_id,
        knowledge_base_id=knowledge_base_id,
        embedding_profile_id=None,
        sparse_profile_id=None,
        index_profile_hash="d" * 64,
        qdrant_collection_name=f"upload_{generation_id.hex}",
        status="active",
        rebuild_snapshot_at=now,
        caught_up_revision=0,
        validated_revision=0,
        validation_manifest_hash="e" * 64,
        expected_point_count=0,
        actual_point_count=0,
        validated_at=now,
        activated_at=now,
        retired_at=None,
        distance="cosine",
        embedding_config_snapshot={},
        filter_schema_snapshot={"fields": []},
        applied_filter_schema_revision=0,
        embedding_config_hash="f" * 64,
        safe_error_code=None,
        safe_error_message=None,
    )
    scope = ApiKeyKnowledgeBaseScope(
        api_key_id=seeded.actor.key_id,
        knowledge_base_id=knowledge_base_id,
    )
    async with database.sessions() as session, session.begin():
        session.add(knowledge_base)
        await session.flush()
        session.add(generation)
        await session.flush()
        session.add(scope)
    return SeededUploadContext(
        knowledge_base_id,
        generation_id,
        replace(
            seeded.actor,
            knowledge_base_ids=seeded.actor.knowledge_base_ids | {knowledge_base_id},
        ),
    )


async def database_upload(
    database: Database,
    seeded: SeededUploadContext,
    store: MemoryObjectStore,
    *,
    content: bytes,
    idempotency_key: str | None,
    metadata: str = "{}",
    filename: str = "document.txt",
    content_type: str = "text/plain",
) -> UploadAccepted:
    async def source() -> AsyncIterable[bytes]:
        yield content

    async with database.sessions() as session:
        service = DocumentUploadService(
            repository=SqlAlchemyUploadRepository(session, store),
            object_store=store,
            notifier=NoopNotifier(),
            max_upload_bytes=50 * 1024 * 1024,
            session=session,
            reservation_session_factory=database.sessions,
            reservation_repository_factory=lambda reservation_session: SqlAlchemyUploadRepository(
                reservation_session,
                store,
            ),
        )
        return await service.upload(
            knowledge_base_id=seeded.knowledge_base_id,
            actor=seeded.actor,
            source=SourceUpload(
                filename=filename,
                content_type=content_type,
                chunks=source(),
            ),
            form=UploadForm.from_multipart(
                display_name="Document",
                metadata=metadata,
                tags='["integration"]',
            ),
            idempotency_key=idempotency_key,
        )


def job_repository_context(database: Database) -> RepositoryContextFactory:
    @asynccontextmanager
    async def context() -> AsyncIterator[JobRepository]:
        async with database.sessions() as session, session.begin():
            yield SqlAlchemyJobRepository(session)

    return context


async def claim_upload_job(database: Database, job_id: UUID) -> JobLease:
    async with database.sessions() as session, session.begin():
        lease = await SqlAlchemyJobRepository(session).claim_next(
            lease_owner="pipeline-worker",
            lease_duration=timedelta(seconds=30),
            job_id=job_id,
        )
    assert isinstance(lease, JobLease)
    return lease


async def expire_and_reclaim_upload_job(
    database: Database,
    job_id: UUID,
    *,
    lease_owner: str,
) -> JobLease:
    async with database.sessions() as session, session.begin():
        job = await session.get(Job, job_id)
        assert job is not None and job.status == "running"
        job.lease_expires_at = datetime.now(UTC) - timedelta(seconds=1)
    async with database.sessions() as session, session.begin():
        lease = await SqlAlchemyJobRepository(session).claim_next(
            lease_owner=lease_owner,
            lease_duration=timedelta(seconds=30),
            job_id=job_id,
        )
    assert isinstance(lease, JobLease)
    return lease


def ingestion_pipeline(
    database: Database,
    store: MemoryObjectStore,
    *,
    cpu_concurrency: int = 1,
    max_manifest_bytes: int = MAX_CHUNK_MANIFEST_BYTES,
    embedding_gateway: PipelineEmbeddingGateway | None = None,
    qdrant: FakeQdrantClient | None = None,
    provider_usage_sink: PipelineProviderUsageSink | None = None,
    hooks: IngestionPipelineHooks | None = None,
) -> IngestionPipeline:
    @asynccontextmanager
    async def stage_repository_context() -> AsyncIterator[IngestionPipelineRepository]:
        async with database.sessions() as session:
            yield SqlAlchemyIngestionPipelineRepository(session)

    return IngestionPipeline(
        repository_context=stage_repository_context,
        object_store=cast(PipelineObjectStore, store),
        max_document_bytes=50 * 1024 * 1024,
        cpu_concurrency=cpu_concurrency,
        max_manifest_bytes=max_manifest_bytes,
        embedding_gateway=embedding_gateway,
        qdrant=qdrant,
        provider_usage_sink=provider_usage_sink,
        hooks=hooks,
    )


def ingestion_job_runner(
    database: Database,
    pipeline: IngestionPipeline,
    *,
    lease_owner: str,
) -> JobRunner:
    runner = JobRunner(
        repository_context=job_repository_context(database),
        lease_owner=lease_owner,
        lease_seconds=30,
        heartbeat_seconds=5,
        poll_interval_seconds=0.1,
        max_concurrency=1,
        backoff=ExponentialBackoff(initial_seconds=1, maximum_seconds=5),
    )
    runner.register(
        "ingest_document",
        pipeline.handle,
        exhaustion_finalizer=pipeline.finalize_exhausted,
    )
    return runner


def principal(kb_id: UUID) -> AgentPrincipal:
    return AgentPrincipal(
        key_id=uuid4(),
        public_id="YWdlbnQtdXBsb2FkLWludGVncmF0aW9u",
        capabilities=frozenset({Capability.INGEST}),
        knowledge_base_ids=frozenset({kb_id}),
        query_profile_ids=frozenset(),
        default_query_profile_id=None,
        raw_file_read=False,
        requests_per_minute=60,
        max_concurrency=4,
    )


@pytest.mark.asyncio
async def test_upload_route_accepts_one_file_and_returns_safe_202() -> None:
    kb_id = uuid4()
    service, repository, _store = capturing_service(kb_id)
    app = create_app()
    app.dependency_overrides[require_agent_principal] = lambda: principal(kb_id)
    app.dependency_overrides[get_document_upload_service] = lambda: service
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/v1/knowledge-bases/{kb_id}/documents",
            headers={"X-Request-ID": "req-upload-route", "Idempotency-Key": "key-1"},
            files={"file": ("guide.md", b"# Guide", "text/markdown")},
            data={"display_name": "Guide", "metadata": "{}", "tags": '["docs"]'},
        )

    assert response.status_code == 202, response.text
    assert set(response.json()) == {"document_id", "version_id", "job_id", "status"}
    reservation = repository.reservations[0]
    assert reservation.preflight is not None
    assert reservation.preflight.knowledge_base_id == kb_id
    assert reservation.idempotency_key == "key-1"
    assert reservation.display_name == "Guide"
    assert reservation.tags == ("docs",)


@pytest.mark.asyncio
async def test_upload_route_rejects_missing_duplicate_and_unknown_file_parts() -> None:
    kb_id = uuid4()
    service, repository, _store = capturing_service(kb_id)
    app = create_app()
    app.dependency_overrides[require_agent_principal] = lambda: principal(kb_id)
    app.dependency_overrides[get_document_upload_service] = lambda: service
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        missing = await client.post(
            f"/v1/knowledge-bases/{kb_id}/documents",
            headers={"X-Request-ID": "req-upload-missing"},
            data={"metadata": "{}"},
        )
        duplicate = await client.post(
            f"/v1/knowledge-bases/{kb_id}/documents",
            headers={"X-Request-ID": "req-upload-duplicate"},
            files=[
                ("file", ("a.txt", b"a", "text/plain")),
                ("file", ("b.txt", b"b", "text/plain")),
            ],
        )
        unknown = await client.post(
            f"/v1/knowledge-bases/{kb_id}/documents",
            headers={"X-Request-ID": "req-upload-unknown"},
            files={"other": ("a.txt", b"a", "text/plain")},
        )

    assert (missing.status_code, missing.json()["error"]["code"]) == (422, "VALIDATION_ERROR")
    assert (duplicate.status_code, duplicate.json()["error"]["code"]) == (
        422,
        "VALIDATION_ERROR",
    )
    assert (unknown.status_code, unknown.json()["error"]["code"]) == (422, "VALIDATION_ERROR")
    assert repository.reservations == []


@pytest.mark.asyncio
async def test_upload_route_does_not_misclassify_service_failures_as_validation() -> None:
    kb_id = uuid4()

    app = create_app()
    app.dependency_overrides[require_agent_principal] = lambda: principal(kb_id)
    app.dependency_overrides[get_document_upload_service] = lambda: FailingUploadService()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/v1/knowledge-bases/{kb_id}/documents",
            headers={"X-Request-ID": "req-upload-internal"},
            files={"file": ("guide.md", b"# Guide", "text/markdown")},
        )

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL_ERROR"
    assert "backend-secret" not in response.text


async def streamed_file_multipart(
    boundary: str,
    size: int,
    *,
    fail_after_bytes: int | None = None,
) -> AsyncIterator[bytes]:
    yield (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="large.txt"\r\n'
        "Content-Type: text/plain\r\n\r\n"
    ).encode()
    emitted = 0
    block = b"x" * (1024 * 1024)
    while emitted < size:
        piece = block[: min(len(block), size - emitted)]
        yield piece
        emitted += len(piece)
        if fail_after_bytes is not None and emitted >= fail_after_bytes:
            raise RuntimeError("disconnect-secret")
    yield f"\r\n--{boundary}--\r\n".encode()


def _accepted_upload_cleanup_key(
    response: httpx.Response,
    knowledge_base_id: UUID,
) -> str | None:
    if response.status_code != 202 or type(knowledge_base_id) is not UUID:
        return None
    try:
        body = response.json()
        if type(body) is not dict:
            return None
        raw_document_id = body.get("document_id")
        raw_version_id = body.get("version_id")
        if type(raw_document_id) is not str or type(raw_version_id) is not str:
            return None
        document_id = UUID(raw_document_id)
        version_id = UUID(raw_version_id)
        if str(document_id) != raw_document_id or str(version_id) != raw_version_id:
            return None
        return source_object_key(
            knowledge_base_id,
            document_id,
            version_id,
            filename="large.txt",
        )
    except (json.JSONDecodeError, TypeError, ValueError, UnicodeDecodeError):
        return None


def _is_expected_upload_source_key(object_key: object, knowledge_base_id: UUID) -> bool:
    if type(object_key) is not str or type(knowledge_base_id) is not UUID:
        return False
    parts = object_key.split("/")
    if (
        len(parts) != 8
        or parts[0] != "knowledge-bases"
        or parts[1] != str(knowledge_base_id)
        or parts[2] != "documents"
        or parts[4] != "versions"
        or parts[6] != "source"
        or parts[7] != "source.txt"
    ):
        return False
    try:
        document_id = UUID(parts[3])
        version_id = UUID(parts[5])
    except ValueError:
        return False
    if str(document_id) != parts[3] or str(version_id) != parts[5]:
        return False
    return (
        source_object_key(
            knowledge_base_id,
            document_id,
            version_id,
            filename="large.txt",
        )
        == object_key
    )


def _accepted_upload_cleanup_candidates(
    response: httpx.Response,
    knowledge_base_id: UUID,
    before_objects: set[str],
    after_objects: set[str],
) -> tuple[str, ...]:
    if (
        type(knowledge_base_id) is not UUID
        or type(before_objects) is not set
        or type(after_objects) is not set
        or any(type(key) is not str for key in before_objects | after_objects)
    ):
        return ()
    new_objects = after_objects - before_objects
    candidates = {
        key for key in new_objects if _is_expected_upload_source_key(key, knowledge_base_id)
    }
    response_key = _accepted_upload_cleanup_key(response, knowledge_base_id)
    if response_key in new_objects and _is_expected_upload_source_key(
        response_key,
        knowledge_base_id,
    ):
        candidates.add(response_key)
    return tuple(sorted(candidates))


@pytest.mark.acceptance
def test_accepted_upload_cleanup_key_is_derived_only_from_validated_response_identity() -> None:
    knowledge_base_id = uuid4()
    document_id = uuid4()
    version_id = uuid4()
    response = httpx.Response(
        202,
        json={
            "document_id": str(document_id),
            "version_id": str(version_id),
        },
    )

    assert _accepted_upload_cleanup_key(response, knowledge_base_id) == (
        f"knowledge-bases/{knowledge_base_id}/documents/{document_id}/versions/"
        f"{version_id}/source/source.txt"
    )
    assert (
        _accepted_upload_cleanup_key(
            httpx.Response(202, json={"document_id": "invalid", "version_id": str(version_id)}),
            knowledge_base_id,
        )
        is None
    )


@pytest.mark.acceptance
def test_accepted_upload_cleanup_candidates_use_only_restricted_baseline_delta() -> None:
    knowledge_base_id = uuid4()
    document_id = uuid4()
    version_id = uuid4()
    expected_key = source_object_key(
        knowledge_base_id,
        document_id,
        version_id,
        filename="large.txt",
    )
    baseline_key = source_object_key(
        knowledge_base_id,
        uuid4(),
        uuid4(),
        filename="large.txt",
    )
    unrelated_key = source_object_key(
        uuid4(),
        uuid4(),
        uuid4(),
        filename="large.txt",
    )
    malformed = httpx.Response(
        202,
        json={"document_id": "invalid", "version_id": "invalid"},
    )
    inconsistent = httpx.Response(
        202,
        json={"document_id": str(uuid4()), "version_id": str(uuid4())},
    )
    before = {baseline_key, "unrelated/existing-object"}
    after = before | {
        expected_key,
        unrelated_key,
        f"knowledge-bases/{knowledge_base_id}/documents/not-a-uuid/versions/"
        f"{version_id}/source/source.txt",
        f"knowledge-bases/{knowledge_base_id}/documents/{document_id}/versions/"
        f"{version_id}/source/unexpected.bin",
    }

    assert _accepted_upload_cleanup_candidates(
        malformed,
        knowledge_base_id,
        before,
        after,
    ) == (expected_key,)
    assert _accepted_upload_cleanup_candidates(
        inconsistent,
        knowledge_base_id,
        before,
        after,
    ) == (expected_key,)
    assert (
        _accepted_upload_cleanup_candidates(
            malformed,
            knowledge_base_id,
            before | {expected_key},
            after,
        )
        == ()
    )


@pytest.mark.asyncio
async def test_upload_route_enforces_exact_50_mib_without_content_length() -> None:
    kb_id = uuid4()
    boundary = "rag-http-size-boundary"
    limit = 50 * 1024 * 1024
    for size, expected_status, expected_code in (
        (limit, 202, None),
        (limit + 1, 413, "FILE_TOO_LARGE"),
    ):
        service, repository, store = capturing_service(kb_id)
        app = create_app()
        app.dependency_overrides[require_agent_principal] = lambda: principal(kb_id)
        app.dependency_overrides[get_document_upload_service] = upload_service_dependency(service)
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                f"/v1/knowledge-bases/{kb_id}/documents",
                headers={
                    "X-Request-ID": f"req-upload-size-{size}",
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                },
                content=streamed_file_multipart(boundary, size),
            )

        assert response.request.headers.get("content-length") is None
        assert response.status_code == expected_status, response.text
        if expected_code is None:
            assert len(repository.reservations) == 1
            assert len(store.objects) == 1
        else:
            assert response.json()["error"]["code"] == expected_code
            assert repository.reservations == []
            assert store.objects == {}


class _UnusedReadinessProvider:
    async def snapshot(
        self,
        scope: ReadinessScope = ReadinessScope.ALL,
    ) -> ReadinessSnapshot:
        del scope
        raise AssertionError("readiness is not part of the live upload boundary test")


@asynccontextmanager
async def running_live_upload_app(app: FastAPI) -> AsyncIterator[str]:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    listener.setblocking(False)
    port = int(listener.getsockname()[1])
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_config=None,
            access_log=False,
            lifespan="on",
            timeout_graceful_shutdown=2,
        )
    )
    task = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        async with asyncio.timeout(5):
            while not server.started:
                if task.done():
                    task.result()
                await asyncio.sleep(0.01)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, timeout=5)
        finally:
            listener.close()


async def upload_graph_counts(
    database: Database,
    knowledge_base_id: UUID,
) -> tuple[int, int, int]:
    async with database.sessions() as session:
        documents = await session.scalar(
            select(func.count())
            .select_from(Document)
            .where(Document.knowledge_base_id == knowledge_base_id)
        )
        versions = await session.scalar(
            select(func.count())
            .select_from(DocumentVersion)
            .join(Document, Document.id == DocumentVersion.document_id)
            .where(Document.knowledge_base_id == knowledge_base_id)
        )
        jobs = await session.scalar(
            select(func.count()).select_from(Job).where(Job.knowledge_base_id == knowledge_base_id)
        )
    return int(documents or 0), int(versions or 0), int(jobs or 0)


async def minio_object_names(client: Minio, bucket: str) -> set[str]:
    return await asyncio.to_thread(
        lambda: {item.object_name for item in client.list_objects(bucket, recursive=True)}
    )


@dataclass(frozen=True, slots=True, repr=False)
class _MinioMultipartUpload:
    object_name: str
    upload_id: str


def _valid_minio_pagination_value(
    value: object,
    *,
    allow_empty: bool = False,
) -> TypeGuard[str]:
    if type(value) is not str or (not value and not allow_empty):
        return False
    try:
        return len(value.encode("utf-8")) <= 2048 and (not value or value.isprintable())
    except UnicodeError:
        return False


async def minio_multipart_uploads(
    client: Minio,
    bucket: str,
    prefix: str,
) -> set[_MinioMultipartUpload]:
    if not _valid_minio_pagination_value(prefix, allow_empty=True):
        raise ValueError("MINIO_MULTIPART_PREFIX_INVALID")

    def list_uploads() -> set[_MinioMultipartUpload]:
        key_marker: str | None = None
        upload_id_marker: str | None = None
        uploads: set[_MinioMultipartUpload] = set()
        for _page in range(10_000):
            arguments: dict[str, object] = {"prefix": prefix}
            if key_marker is not None:
                arguments["key_marker"] = key_marker
            if upload_id_marker is not None:
                arguments["upload_id_marker"] = upload_id_marker
            try:
                result = cast(Any, client)._list_multipart_uploads(bucket, **arguments)
            except BaseException as error:
                raise RuntimeError(f"MINIO_MULTIPART_LIST_FAILED:{type(error).__name__}") from None
            for upload in result.uploads:
                object_name = getattr(upload, "object_name", None)
                upload_id = getattr(upload, "upload_id", None)
                if (
                    not _valid_minio_pagination_value(object_name)
                    or not object_name.startswith(prefix)
                    or not _valid_minio_pagination_value(upload_id)
                ):
                    raise RuntimeError("MINIO_MULTIPART_PAGINATION_INVALID")
                uploads.add(_MinioMultipartUpload(object_name, upload_id))
            if not result.is_truncated:
                return uploads
            next_key_marker = getattr(result, "next_key_marker", None)
            next_upload_id_marker = getattr(result, "next_upload_id_marker", None)
            if next_upload_id_marker is None:
                # minio==7.2.20 stores the parsed marker under this anomalous
                # attribute name while leaving the public dataclass field unset.
                next_upload_id_marker = getattr(
                    result,
                    "self._next_upload_id_marker",
                    None,
                )
            if (
                not _valid_minio_pagination_value(next_key_marker)
                or not next_key_marker.startswith(prefix)
                or not _valid_minio_pagination_value(next_upload_id_marker)
                or (next_key_marker, next_upload_id_marker) == (key_marker, upload_id_marker)
            ):
                raise RuntimeError("MINIO_MULTIPART_PAGINATION_INVALID")
            key_marker = next_key_marker
            upload_id_marker = next_upload_id_marker
        raise RuntimeError("MINIO_MULTIPART_PAGINATION_INVALID")

    return await asyncio.to_thread(list_uploads)


async def minio_multipart_names(client: Minio, bucket: str, prefix: str) -> set[str]:
    return {upload.object_name for upload in await minio_multipart_uploads(client, bucket, prefix)}


async def minio_upload_multipart_names(client: Minio, bucket: str) -> set[str]:
    # Server-side prefix filtering can hide a real in-progress upload. Enumerate the
    # isolated bucket completely, then validate the only allowed multipart namespace.
    uploads = await minio_multipart_uploads(client, bucket, "")
    if any(not upload.object_name.startswith("tmp/uploads/") for upload in uploads):
        raise RuntimeError("MINIO_MULTIPART_NAMESPACE_INVALID")
    return {upload.object_name for upload in uploads}


async def create_isolated_minio_bucket(client: Minio) -> str:
    bucket = f"rag-live-{uuid4().hex}"
    try:
        await asyncio.to_thread(client.make_bucket, bucket)
    except BaseException as error:
        raise RuntimeError(f"MINIO_ISOLATED_BUCKET_CREATE_FAILED:{type(error).__name__}") from None
    return bucket


async def remove_isolated_minio_bucket(client: Minio, bucket: str) -> None:
    try:
        async with asyncio.timeout(15):
            while True:
                uploads = await minio_multipart_uploads(client, bucket, "")
                objects = await minio_object_names(client, bucket)
                if not uploads and not objects:
                    break
                for upload in uploads:
                    await asyncio.to_thread(
                        cast(Any, client)._abort_multipart_upload,
                        bucket,
                        upload.object_name,
                        upload.upload_id,
                    )
                for object_name in objects:
                    await asyncio.to_thread(client.remove_object, bucket, object_name)
                await asyncio.sleep(0.05)
        await asyncio.to_thread(client.remove_bucket, bucket)
        if await asyncio.to_thread(client.bucket_exists, bucket):
            raise RuntimeError("MINIO_ISOLATED_BUCKET_REMOVE_INCOMPLETE")
    except RuntimeError:
        raise
    except BaseException as error:
        raise RuntimeError(f"MINIO_ISOLATED_BUCKET_REMOVE_FAILED:{type(error).__name__}") from None


@pytest.mark.acceptance
@pytest.mark.asyncio
async def test_minio_multipart_listing_paginates_within_knowledge_base_prefix() -> None:
    knowledge_base_id = uuid4()
    prefix = f"{knowledge_base_id}/"

    class FakeMinio:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def _list_multipart_uploads(self, _bucket: str, **kwargs: object) -> object:
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return SimpleNamespace(
                    uploads=[
                        SimpleNamespace(
                            object_name=f"{prefix}first",
                            upload_id="upload-first",
                        )
                    ],
                    is_truncated=True,
                    next_key_marker=f"{prefix}first",
                    next_upload_id_marker="upload-1",
                )
            return SimpleNamespace(
                uploads=[
                    SimpleNamespace(
                        object_name=f"{prefix}second",
                        upload_id="upload-second",
                    )
                ],
                is_truncated=False,
                next_key_marker=None,
                next_upload_id_marker=None,
            )

    client = FakeMinio()
    names = await minio_multipart_names(cast(Any, client), "bucket", prefix)
    if names != {f"{prefix}first", f"{prefix}second"}:
        raise AssertionError("MINIO_MULTIPART_NAMES_MISMATCH") from None
    if client.calls != [
        {"prefix": prefix},
        {
            "prefix": prefix,
            "key_marker": f"{prefix}first",
            "upload_id_marker": "upload-1",
        },
    ]:
        raise AssertionError("MINIO_MULTIPART_PAGINATION_CALLS_MISMATCH") from None


@pytest.mark.acceptance
@pytest.mark.asyncio
async def test_minio_multipart_listing_fails_closed_without_next_marker() -> None:
    prefix = f"{uuid4()}/"

    class FakeMinio:
        def _list_multipart_uploads(self, _bucket: str, **_kwargs: object) -> object:
            return SimpleNamespace(
                uploads=[],
                is_truncated=True,
                next_key_marker=None,
                next_upload_id_marker=None,
            )

    with pytest.raises(RuntimeError, match="^MINIO_MULTIPART_PAGINATION_INVALID$"):
        await minio_multipart_names(cast(Any, FakeMinio()), "bucket", prefix)


@pytest.mark.acceptance
@pytest.mark.asyncio
async def test_minio_multipart_listing_fails_closed_without_next_upload_id_marker() -> None:
    prefix = "tmp/uploads/"

    class FakeMinio:
        def __init__(self) -> None:
            self.calls = 0

        def _list_multipart_uploads(self, _bucket: str, **_kwargs: object) -> object:
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(
                    uploads=[],
                    is_truncated=True,
                    next_key_marker=f"{prefix}same-key",
                    next_upload_id_marker=None,
                )
            return SimpleNamespace(
                uploads=[],
                is_truncated=False,
                next_key_marker=None,
                next_upload_id_marker=None,
            )

    with pytest.raises(RuntimeError, match="^MINIO_MULTIPART_PAGINATION_INVALID$"):
        await minio_multipart_names(cast(Any, FakeMinio()), "bucket", prefix)


@pytest.mark.acceptance
@pytest.mark.asyncio
async def test_minio_multipart_listing_paginates_multiple_uploads_for_same_key() -> None:
    prefix = "tmp/uploads/"
    object_name = f"{prefix}same-key"

    class FakeMinio:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def _list_multipart_uploads(self, _bucket: str, **kwargs: object) -> object:
            self.calls.append(kwargs)
            if len(self.calls) == 1:
                return SimpleNamespace(
                    uploads=[SimpleNamespace(object_name=object_name, upload_id="upload-1")],
                    is_truncated=True,
                    next_key_marker=object_name,
                    next_upload_id_marker="upload-1",
                )
            if len(self.calls) == 2:
                return SimpleNamespace(
                    uploads=[SimpleNamespace(object_name=object_name, upload_id="upload-2")],
                    is_truncated=True,
                    next_key_marker=object_name,
                    next_upload_id_marker="upload-2",
                )
            return SimpleNamespace(
                uploads=[
                    SimpleNamespace(
                        object_name=f"{prefix}next-key",
                        upload_id="upload-next",
                    )
                ],
                is_truncated=False,
                next_key_marker=None,
                next_upload_id_marker=None,
            )

    client = FakeMinio()
    names = await minio_multipart_names(cast(Any, client), "bucket", prefix)
    if names != {object_name, f"{prefix}next-key"}:
        raise AssertionError("MINIO_MULTIPART_NAMES_MISMATCH") from None
    if client.calls != [
        {"prefix": prefix},
        {
            "prefix": prefix,
            "key_marker": object_name,
            "upload_id_marker": "upload-1",
        },
        {
            "prefix": prefix,
            "key_marker": object_name,
            "upload_id_marker": "upload-2",
        },
    ]:
        raise AssertionError("MINIO_MULTIPART_PAGINATION_CALLS_MISMATCH") from None


@pytest.mark.acceptance
@pytest.mark.asyncio
async def test_minio_multipart_listing_handles_pinned_sdk_next_upload_marker_parser_bug() -> None:
    prefix = "tmp/uploads/"

    def parsed_result(*, truncated: bool, upload_id: str) -> ListMultipartUploadsResult:
        next_markers = (
            f"<NextKeyMarker>{prefix}same-key</NextKeyMarker>"
            f"<NextUploadIdMarker>{upload_id}</NextUploadIdMarker>"
            if truncated
            else ""
        )
        body = (
            "<ListMultipartUploadsResult>"
            f"<IsTruncated>{str(truncated).lower()}</IsTruncated>"
            f"{next_markers}"
            "<Upload>"
            f"<Key>{prefix}same-key</Key>"
            f"<UploadId>{upload_id}</UploadId>"
            "</Upload>"
            "</ListMultipartUploadsResult>"
        ).encode()
        return ListMultipartUploadsResult(cast(Any, SimpleNamespace(data=body)))

    class FakeMinio:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def _list_multipart_uploads(self, _bucket: str, **kwargs: object) -> object:
            self.calls.append(kwargs)
            return parsed_result(
                truncated=len(self.calls) == 1,
                upload_id=f"upload-{len(self.calls)}",
            )

    client = FakeMinio()
    names = await minio_multipart_names(cast(Any, client), "bucket", prefix)
    if names != {f"{prefix}same-key"}:
        raise AssertionError("MINIO_MULTIPART_NAMES_MISMATCH") from None
    if client.calls != [
        {"prefix": prefix},
        {
            "prefix": prefix,
            "key_marker": f"{prefix}same-key",
            "upload_id_marker": "upload-1",
        },
    ]:
        raise AssertionError("MINIO_MULTIPART_PAGINATION_CALLS_MISMATCH") from None


@pytest.mark.integration
@pytest.mark.acceptance
@pytest.mark.asyncio
async def test_real_minio_tmp_prefix_finds_multipart_missed_by_knowledge_base_prefix(
    full_stack_connections: _FullStackConnections,
    minio_http_pool_factory: MinioHttpPoolFactory,
) -> None:
    pool = minio_http_pool_factory()
    minio = Minio(
        full_stack_connections.minio.endpoint,
        access_key=full_stack_connections.minio.access_key,
        secret_key=full_stack_connections.minio.secret_key,
        secure=full_stack_connections.minio.secure,
        http_client=pool,
    )
    bucket = await create_isolated_minio_bucket(minio)
    object_name = f"tmp/uploads/{uuid4().hex}"
    try:
        upload_id = await asyncio.to_thread(
            cast(Any, minio)._create_multipart_upload,
            bucket,
            object_name,
            {},
        )
        await asyncio.to_thread(
            cast(Any, minio)._upload_part,
            bucket,
            object_name,
            b"x",
            None,
            upload_id,
            1,
        )
        missed = await minio_multipart_names(minio, bucket, f"{uuid4()}/")
        if missed:
            raise AssertionError("MINIO_KNOWLEDGE_BASE_PREFIX_UNEXPECTED_MATCH") from None
        all_observed = await minio_multipart_names(minio, bucket, "")
        if all_observed != {object_name}:
            raise AssertionError("MINIO_UNFILTERED_LIST_DID_NOT_FIND_MULTIPART") from None
        observed = await minio_upload_multipart_names(minio, bucket)
        if observed != {object_name}:
            raise AssertionError("MINIO_UPLOAD_PREFIX_DID_NOT_FIND_MULTIPART") from None
    finally:
        try:
            await remove_isolated_minio_bucket(minio, bucket)
        finally:
            pool.clear()


async def wait_for_rejected_upload_cleanup(
    *,
    database: Database,
    knowledge_base_id: UUID,
    expected_counts: tuple[int, int, int],
    minio: Minio,
    bucket: str,
    expected_objects: set[str],
) -> None:
    async with asyncio.timeout(15):
        while True:
            counts = await upload_graph_counts(database, knowledge_base_id)
            objects = await minio_object_names(minio, bucket)
            multipart = await minio_upload_multipart_names(minio, bucket)
            if counts == expected_counts and objects == expected_objects and not multipart:
                return
            await asyncio.sleep(0.05)


async def disconnect_live_multipart(
    base_url: str,
    knowledge_base_id: UUID,
    boundary: str,
) -> None:
    parsed = httpx.URL(base_url)
    assert parsed.host is not None and parsed.port is not None
    _reader, writer = await asyncio.open_connection(parsed.host, parsed.port)
    expected_length = 12 * 1024 * 1024
    headers = (
        f"POST /v1/knowledge-bases/{knowledge_base_id}/documents HTTP/1.1\r\n"
        f"Host: {parsed.host}:{parsed.port}\r\n"
        f"Content-Type: multipart/form-data; boundary={boundary}\r\n"
        f"Content-Length: {expected_length}\r\n"
        "Connection: close\r\n\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="file"; filename="disconnect.txt"\r\n'
        "Content-Type: text/plain\r\n\r\n"
    ).encode()
    writer.write(headers)
    await writer.drain()
    block = b"d" * (1024 * 1024)
    for _ in range(6):
        writer.write(block)
        await writer.drain()
    writer.close()
    await writer.wait_closed()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.acceptance
async def test_live_uvicorn_and_real_minio_enforce_50_mib_and_clean_disconnects(
    migrated_database: Database,
    full_stack_connections: _FullStackConnections,
    minio_http_pool_factory: MinioHttpPoolFactory,
) -> None:
    seeded = await seed_upload_context(migrated_database)
    limit = 50 * 1024 * 1024
    pool = minio_http_pool_factory()
    minio = Minio(
        full_stack_connections.minio.endpoint,
        access_key=full_stack_connections.minio.access_key,
        secret_key=full_stack_connections.minio.secret_key,
        secure=full_stack_connections.minio.secure,
        http_client=pool,
    )
    bucket = await create_isolated_minio_bucket(minio)
    if bucket == full_stack_connections.minio.bucket:
        raise AssertionError("MINIO_ISOLATED_BUCKET_COLLISION") from None
    store: MinioObjectStore | None = None
    try:
        store = MinioObjectStore(
            client=cast(Any, minio),
            bucket=bucket,
            buffer_bytes=1024 * 1024,
            part_size_bytes=5 * 1024 * 1024,
            operation_timeout_seconds=120,
        )
        settings = Settings(
            _env_file=None,
            environment="test",
            database_url=SecretStr(full_stack_connections.postgres.async_url),
            redis_url=SecretStr(full_stack_connections.redis.url),
            qdrant_url=full_stack_connections.qdrant.url,
            minio_url=full_stack_connections.minio.url,
            minio_access_key=full_stack_connections.minio.access_key,
            minio_secret_key=SecretStr(full_stack_connections.minio.secret_key),
            minio_bucket=bucket,
            max_upload_bytes=limit,
            upload_buffer_bytes=1024 * 1024,
            minio_multipart_part_size_bytes=5 * 1024 * 1024,
            minio_operation_timeout_seconds=120,
        )
        app = create_app(
            settings=settings,
            readiness_provider=_UnusedReadinessProvider(),
            database=migrated_database,
            upload_object_store=store,
            job_notifier=cast(Any, NoopNotifier()),
        )

        @app.middleware("http")
        async def replace_exact_upload_identity(
            request: Request,
            call_next: Callable[[Request], Awaitable[Response]],
        ) -> Response:
            response = await call_next(request)
            if (
                request.headers.get("X-Request-ID") == "req-live-exact-50-mib"
                and response.status_code == 202
            ):
                return JSONResponse(
                    status_code=202,
                    content={"document_id": "invalid", "version_id": "invalid"},
                )
            return response

        app.dependency_overrides[require_agent_principal] = lambda: seeded.actor
        before_objects = await minio_object_names(minio, bucket)
        if before_objects:
            raise AssertionError("MINIO_ISOLATED_BUCKET_NOT_EMPTY") from None
        if await minio_upload_multipart_names(minio, bucket):
            raise AssertionError("MINIO_ISOLATED_BUCKET_MULTIPART_NOT_EMPTY") from None

        async with running_live_upload_app(app) as base_url:
            boundary = f"rag-live-exact-{uuid4().hex}"
            async with httpx.AsyncClient(
                base_url=base_url,
                trust_env=False,
                timeout=httpx.Timeout(180),
            ) as client:
                accepted = await client.post(
                    f"/v1/knowledge-bases/{seeded.knowledge_base_id}/documents",
                    headers={
                        "X-Request-ID": "req-live-exact-50-mib",
                        "Content-Type": f"multipart/form-data; boundary={boundary}",
                    },
                    content=streamed_file_multipart(boundary, limit),
                )
                accepted_objects = await minio_object_names(minio, bucket)
                accepted_cleanup_keys = _accepted_upload_cleanup_candidates(
                    accepted,
                    seeded.knowledge_base_id,
                    before_objects,
                    accepted_objects,
                )
                if len(accepted_cleanup_keys) != 1:
                    raise AssertionError("MINIO_ACCEPTED_OBJECT_COUNT_MISMATCH") from None
                accepted_source_key = accepted_cleanup_keys[0]
                if accepted.request.headers.get("content-length") is not None:
                    raise AssertionError("STREAMING_UPLOAD_CONTENT_LENGTH_PRESENT") from None
                _assert_sanitized_http_status(accepted, 202)
                if _accepted_upload_cleanup_key(accepted, seeded.knowledge_base_id) is not None:
                    raise AssertionError("INVALID_RESPONSE_IDENTITY_ACCEPTED") from None

                accepted_counts = await upload_graph_counts(
                    migrated_database,
                    seeded.knowledge_base_id,
                )
                if accepted_counts != (1, 1, 1):
                    raise AssertionError("ACCEPTED_UPLOAD_GRAPH_COUNT_MISMATCH") from None
                if accepted_objects != {accepted_source_key}:
                    raise AssertionError("MINIO_ACCEPTED_OBJECT_SET_MISMATCH") from None
                if await minio_upload_multipart_names(minio, bucket):
                    raise AssertionError("MINIO_ACCEPTED_MULTIPART_REMAINED") from None

                boundary = f"rag-live-over-{uuid4().hex}"
                oversized = await client.post(
                    f"/v1/knowledge-bases/{seeded.knowledge_base_id}/documents",
                    headers={
                        "X-Request-ID": "req-live-over-50-mib",
                        "Content-Type": f"multipart/form-data; boundary={boundary}",
                    },
                    content=streamed_file_multipart(boundary, limit + 1),
                )
                _assert_sanitized_http_status(oversized, 413)
                if oversized.json()["error"]["code"] != "FILE_TOO_LARGE":
                    raise AssertionError("OVERSIZED_UPLOAD_ERROR_CODE_MISMATCH") from None
                await wait_for_rejected_upload_cleanup(
                    database=migrated_database,
                    knowledge_base_id=seeded.knowledge_base_id,
                    expected_counts=accepted_counts,
                    minio=minio,
                    bucket=bucket,
                    expected_objects=accepted_objects,
                )
                if await minio_upload_multipart_names(minio, bucket):
                    raise AssertionError("MINIO_OVERSIZED_MULTIPART_REMAINED") from None

            await disconnect_live_multipart(
                base_url,
                seeded.knowledge_base_id,
                f"rag-live-disconnect-{uuid4().hex}",
            )
            await wait_for_rejected_upload_cleanup(
                database=migrated_database,
                knowledge_base_id=seeded.knowledge_base_id,
                expected_counts=accepted_counts,
                minio=minio,
                bucket=bucket,
                expected_objects=accepted_objects,
            )
            if await minio_upload_multipart_names(minio, bucket):
                raise AssertionError("MINIO_DISCONNECT_MULTIPART_REMAINED") from None
    finally:
        try:
            if store is not None:
                await store.aclose()
        finally:
            try:
                await remove_isolated_minio_bucket(minio, bucket)
            finally:
                pool.clear()


@pytest.mark.acceptance
def test_smoke_script_helpers_accept_canonicalized_system_tmp_directory() -> None:
    script = Path(__file__).parents[2] / "scripts" / "smoke_ingestion_retrieval.sh"
    command = """
source "$1"
work_dir="$2"
provider_base_url="https://provider-stub:8443/v1"
write_request_bodies "path-regression"
patch_json_uuid "$2/provider-config.json" "credential_id" \
  "00000000-0000-0000-0000-000000000001"
write_generation_request "00000000-0000-0000-0000-000000000002"
test -s "$2/document.md"
test -s "$2/generation.json"
"""
    with tempfile.TemporaryDirectory(
        prefix="rag-service-ingestion-smoke.",
        dir="/tmp",
    ) as work_dir:
        completed = subprocess.run(
            ["bash", "-c", command, "smoke-helper-test", str(script), work_dir],
            check=False,
            capture_output=True,
            text=True,
        )

    _assert_sanitized_process(completed, returncode=0, stdout="", stderr="")


@pytest.mark.acceptance
def test_smoke_script_compose_capture_preserves_stdout() -> None:
    script = Path(__file__).parents[2] / "scripts" / "smoke_ingestion_retrieval.sh"
    command = """
source "$1"
PATH="$2:$PATH"
compose_capture run --rm api safe-command >"$3"
python3 - "$3" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    body = json.load(source)
if body != {"status": "ok"}:
    raise SystemExit(1)
PY
"""
    with tempfile.TemporaryDirectory(prefix="rag-smoke-compose.", dir="/tmp") as temp_dir:
        temp = Path(temp_dir)
        fake_docker = temp / "docker"
        fake_docker.write_text(
            "#!/usr/bin/env bash\nprintf '%s\\n' '{\"status\":\"ok\"}'\n",
            encoding="utf-8",
        )
        fake_docker.chmod(0o700)
        output = temp / "output.json"
        completed = subprocess.run(
            [
                "bash",
                "-c",
                command,
                "smoke-compose-test",
                str(script),
                temp_dir,
                str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    _assert_sanitized_process(completed, returncode=0, stdout="", stderr="")


@pytest.mark.acceptance
def test_smoke_script_rejects_untrusted_token_before_writing_curl_config() -> None:
    script = Path(__file__).parents[2] / "scripts" / "smoke_ingestion_retrieval.sh"
    command = """
source "$1"
untrusted_token=$(printf '%s\n%s' "$$-$RANDOM" 'header = "unsafe"')
if write_auth_config "$2/auth.conf" "$untrusted_token"; then
  exit 1
fi
test ! -e "$2/auth.conf"
"""
    with tempfile.TemporaryDirectory(prefix="rag-smoke-auth.", dir="/tmp") as temp_dir:
        completed = subprocess.run(
            ["bash", "-c", command, "smoke-auth-test", str(script), temp_dir],
            check=False,
            capture_output=True,
            text=True,
        )

    _assert_sanitized_process(
        completed,
        returncode=0,
        stdout="",
        stderr="RAG ingestion/retrieval smoke test failed: API key token format was invalid\n",
    )
    _assert_protected_content_absent(completed.stderr, "unsafe")


@pytest.mark.acceptance
def test_smoke_script_rejects_multiline_search_validation_code() -> None:
    script = Path(__file__).parents[2] / "scripts" / "smoke_ingestion_retrieval.sh"
    command = """
source "$1"
safe_search_validation_code "$2"
"""
    untrusted_code = "SEARCH_A\nunsafe diagnostic"
    completed = subprocess.run(
        ["bash", "-c", command, "smoke-code-test", str(script), untrusted_code],
        check=False,
        capture_output=True,
        text=True,
    )

    _assert_sanitized_process(
        completed,
        returncode=0,
        stdout="SEARCH_VALIDATION_UNKNOWN\n",
        stderr="",
    )
    _assert_protected_content_absent(completed.stdout, "unsafe diagnostic")


@pytest.mark.acceptance
def test_smoke_script_success_becomes_failure_when_cleanup_is_incomplete() -> None:
    script = Path(__file__).parents[2] / "scripts" / "smoke_ingestion_retrieval.sh"
    command = """
source "$1"
redis_restore_required=0
work_dir=""
best_effort_delete_knowledge_base() { return 1; }
best_effort_disable_provider_resource() { return 0; }
best_effort_revoke_key() { return 0; }
finish
"""
    completed = subprocess.run(
        ["bash", "-c", command, "smoke-cleanup-test", str(script)],
        check=False,
        capture_output=True,
        text=True,
    )

    _assert_sanitized_process(
        completed,
        returncode=1,
        stdout="",
        stderr="RAG ingestion/retrieval smoke cleanup was incomplete\n",
    )


@pytest.mark.acceptance
def test_smoke_job_polling_uses_one_monotonic_60_second_deadline() -> None:
    script = Path(__file__).parents[2] / "scripts" / "smoke_ingestion_retrieval.sh"
    command = """
source "$1"
work_dir="$2"
agent_auth_config="$2/agent.conf"
clock_file="$2/clock"
timeouts_file="$2/timeouts"
sleep_count_file="$2/sleeps"
printf '%s' '0' >"$clock_file"
printf '%s' '0' >"$sleep_count_file"
monotonic_milliseconds() { command cat "$clock_file"; }
advance_clock() {
  current=$(command cat "$clock_file")
  printf '%s' "$((current + $1))" >"$clock_file"
}
timeout_to_milliseconds() {
  whole=${1%%.*}
  fraction=${1#*.}
  printf '%s' "$((10#$whole * 1000 + 10#$fraction))"
}
http_request() {
  advance_clock 20000
  printf '%s\n' 'legacy' >>"$timeouts_file"
  printf '%s' '{"status":"queued"}' >"$4"
  http_status=200
}
http_request_with_timeout() {
  timeout=$1
  printf '%s\n' "$timeout" >>"$timeouts_file"
  advance_clock "$(timeout_to_milliseconds "$timeout")"
  printf '%s' '{"status":"queued"}' >"$5"
  http_status=200
}
expect_status() { return 0; }
sleep() { advance_clock 1000; }
sleep_milliseconds() {
  count=$(command cat "$sleep_count_file")
  printf '%s' "$((count + 1))" >"$sleep_count_file"
  advance_clock "$1"
}
if wait_for_job "00000000-0000-0000-0000-000000000041"; then
  exit 1
fi
test "$(command cat "$clock_file")" = "60000"
test "$(command cat "$sleep_count_file")" = "2"
test "$(command tr '\n' ',' <"$timeouts_file")" = "20.000,20.000,18.000,"
"""
    with tempfile.TemporaryDirectory(prefix="rag-smoke-deadline.", dir="/tmp") as temp_dir:
        completed = subprocess.run(
            ["bash", "-c", command, "smoke-deadline-test", str(script), temp_dir],
            check=False,
            capture_output=True,
            text=True,
        )

    _assert_sanitized_process(
        completed,
        returncode=0,
        stdout="",
        stderr=(
            "RAG ingestion/retrieval smoke test failed: "
            "ingestion job did not succeed within 60 seconds\n"
        ),
    )


@pytest.mark.acceptance
def test_smoke_readiness_polling_never_exceeds_its_30_second_deadline() -> None:
    script = Path(__file__).parents[2] / "scripts" / "smoke_ingestion_retrieval.sh"
    command = """
source "$1"
clock_file="$2/clock"
timeouts_file="$2/timeouts"
printf '%s' '0' >"$clock_file"
monotonic_milliseconds() { command cat "$clock_file"; }
advance_clock() {
  current=$(command cat "$clock_file")
  printf '%s' "$((current + $1))" >"$clock_file"
}
timeout_to_milliseconds() {
  whole=${1%%.*}
  fraction=${1#*.}
  printf '%s' "$((10#$whole * 1000 + 10#$fraction))"
}
curl() {
  advance_clock 3000
  printf '%s\n' 'legacy' >>"$timeouts_file"
  printf '%s' '503'
}
curl_status_with_timeout() {
  printf '%s\n' "$1" >>"$timeouts_file"
  advance_clock "$(timeout_to_milliseconds "$1")"
  printf '%s' '503'
}
sleep() { advance_clock 1000; }
sleep_milliseconds() { advance_clock "$1"; }
if wait_for_api; then
  exit 1
fi
test "$(command cat "$clock_file")" = "30000"
test "$(tail -n 1 "$timeouts_file")" = "2.000"
printf '%s' '0' >"$clock_file"
: >"$timeouts_file"
if wait_for_ingest_readiness; then
  exit 1
fi
test "$(command cat "$clock_file")" = "30000"
test "$(tail -n 1 "$timeouts_file")" = "2.000"
"""
    with tempfile.TemporaryDirectory(prefix="rag-smoke-readiness.", dir="/tmp") as temp_dir:
        completed = subprocess.run(
            ["bash", "-c", command, "smoke-readiness-test", str(script), temp_dir],
            check=False,
            capture_output=True,
            text=True,
        )

    _assert_sanitized_process(
        completed,
        returncode=0,
        stdout="",
        stderr=(
            "RAG ingestion/retrieval smoke test failed: "
            "API health did not become ready within 30 seconds\n"
            "RAG ingestion/retrieval smoke test failed: "
            "ingestion readiness did not recover within 30 seconds\n"
        ),
    )


@pytest.mark.acceptance
def test_make_exposes_one_unified_ingestion_retrieval_acceptance_gate() -> None:
    repo_root = Path(__file__).parents[2]
    completed = subprocess.run(
        ["make", "-n", "acceptance-ingestion-retrieval"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    if completed.returncode != 0:
        raise AssertionError(f"MAKE_ACCEPTANCE_RETURN_CODE:{completed.returncode}") from None
    if completed.stderr:
        raise AssertionError("MAKE_ACCEPTANCE_STDERR_MISMATCH") from None
    focused = "make test-acceptance"
    live = "bash scripts/acceptance_ingestion_retrieval.sh"
    if focused not in completed.stdout:
        raise AssertionError("MAKE_ACCEPTANCE_FOCUSED_STEP_MISSING") from None
    if live not in completed.stdout:
        raise AssertionError("MAKE_ACCEPTANCE_LIVE_STEP_MISSING") from None
    if completed.stdout.index(focused) >= completed.stdout.index(live):
        raise AssertionError("MAKE_ACCEPTANCE_STEP_ORDER_MISMATCH") from None

    verify = subprocess.run(
        ["make", "-n", "verify"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if verify.returncode != 0:
        raise AssertionError(f"MAKE_VERIFY_RETURN_CODE:{verify.returncode}") from None
    if verify.stderr:
        raise AssertionError("MAKE_VERIFY_STDERR_MISMATCH") from None
    if "make acceptance-ingestion-retrieval" not in verify.stdout:
        raise AssertionError("MAKE_VERIFY_ACCEPTANCE_STEP_MISSING") from None

    standalone = subprocess.run(
        ["make", "-n", "smoke-ingestion-retrieval"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if standalone.returncode == 0:
        raise AssertionError("MAKE_STANDALONE_SMOKE_TARGET_STILL_PUBLIC") from None
    if "make test-integration-core" not in verify.stdout:
        raise AssertionError("MAKE_VERIFY_CORE_INTEGRATION_STEP_MISSING") from None
    if "make test-integration\n" in verify.stdout:
        raise AssertionError("MAKE_VERIFY_USES_FULL_INTEGRATION_STEP") from None
    if "integration and not acceptance" not in verify.stdout:
        raise AssertionError("MAKE_VERIFY_REPEATS_ACCEPTANCE_INTEGRATION") from None
    if "--cov-append" in verify.stdout:
        raise AssertionError("MAKE_ACCEPTANCE_APPENDS_STALE_COVERAGE") from None

    full_integration = subprocess.run(
        ["make", "-n", "test-integration"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if full_integration.returncode != 0:
        raise AssertionError(
            f"MAKE_FULL_INTEGRATION_RETURN_CODE:{full_integration.returncode}"
        ) from None
    if full_integration.stderr:
        raise AssertionError("MAKE_FULL_INTEGRATION_STDERR_MISMATCH") from None
    if "-m integration" not in full_integration.stdout:
        raise AssertionError("MAKE_FULL_INTEGRATION_MARKER_MISSING") from None
    if "integration and not acceptance" in full_integration.stdout:
        raise AssertionError("MAKE_FULL_INTEGRATION_EXCLUDES_ACCEPTANCE") from None

    acceptance = subprocess.run(
        ["make", "-n", "test-acceptance"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if acceptance.returncode != 0:
        raise AssertionError(f"MAKE_ACCEPTANCE_RETURN_CODE:{acceptance.returncode}") from None
    if acceptance.stderr:
        raise AssertionError("MAKE_ACCEPTANCE_STDERR_MISMATCH") from None
    acceptance_coverage = str(repo_root / ".coverage.acceptance")
    erase = f"COVERAGE_FILE={acceptance_coverage} uv run coverage erase"
    pytest_with_coverage = f"COVERAGE_FILE={acceptance_coverage} uv run pytest"
    if erase not in acceptance.stdout:
        raise AssertionError("MAKE_ACCEPTANCE_COVERAGE_ERASE_MISSING") from None
    if pytest_with_coverage not in acceptance.stdout:
        raise AssertionError("MAKE_ACCEPTANCE_COVERAGE_FILE_MISSING") from None
    if acceptance.stdout.index(erase) >= acceptance.stdout.index(pytest_with_coverage):
        raise AssertionError("MAKE_ACCEPTANCE_COVERAGE_ERASE_ORDER_INVALID") from None

    expected_combine = (
        "coverage combine --keep .coverage.unit .coverage.integration .coverage.acceptance"
    )
    if expected_combine not in verify.stdout:
        raise AssertionError("MAKE_COMBINED_COVERAGE_INPUTS_MISMATCH") from None


@pytest.mark.acceptance
def test_acceptance_compose_override_removes_fixed_ports_and_uses_unique_image() -> None:
    repo_root = Path(__file__).parents[2]
    project = f"rag-acceptance-{uuid4().hex[:16]}"
    completed = subprocess.run(
        [
            "env",
            "COMPOSE_DISABLE_ENV_FILE=1",
            f"VELOX_IMAGE_TAG={project}",
            "VELOX_IMAGE=veloxrag",
            "docker",
            "compose",
            "--project-name",
            project,
            "-f",
            str(repo_root / "compose.yaml"),
            "-f",
            str(repo_root / "compose.build.yaml"),
            "-f",
            str(repo_root / "compose.acceptance.yaml"),
            "--profile",
            "provider-stub",
            "config",
            "--format",
            "json",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise AssertionError(f"ACCEPTANCE_COMPOSE_CONFIG_FAILED:{completed.returncode}") from None
    try:
        config = json.loads(completed.stdout)
    except json.JSONDecodeError:
        raise AssertionError("ACCEPTANCE_COMPOSE_CONFIG_INVALID_JSON") from None
    services = config.get("services")
    if not isinstance(services, dict):
        raise AssertionError("ACCEPTANCE_COMPOSE_SERVICES_MISSING") from None
    for service_name in ("postgres", "qdrant", "redis", "minio"):
        if services.get(service_name, {}).get("ports"):
            raise AssertionError(f"ACCEPTANCE_FIXED_PORT_PRESENT:{service_name}") from None
    api_ports = services.get("api", {}).get("ports")
    if not isinstance(api_ports, list) or len(api_ports) != 1:
        raise AssertionError("ACCEPTANCE_API_PORT_SHAPE_MISMATCH") from None
    api_port = api_ports[0]
    if (
        api_port.get("target") != 8000
        or api_port.get("published") != "0"
        or api_port.get("host_ip") != "127.0.0.1"
    ):
        raise AssertionError("ACCEPTANCE_API_PORT_NOT_DYNAMIC_LOOPBACK") from None
    expected_image = f"rag-service:{project}"
    for service_name in (
        "api",
        "worker",
        "migrate",
        "minio-init",
        "provider-tls-init",
        "provider-stub",
    ):
        if services.get(service_name, {}).get("image") != expected_image:
            raise AssertionError(f"ACCEPTANCE_IMAGE_TAG_MISMATCH:{service_name}") from None


@pytest.mark.acceptance
def test_unified_acceptance_gate_owns_an_isolated_compose_project() -> None:
    script = Path(__file__).parents[2] / "scripts" / "acceptance_ingestion_retrieval.sh"
    project_suffix = uuid4().hex[:16]
    expected_project = f"rag-acceptance-{project_suffix}"
    command = """
source "$1"
project_log="$2"
fake_project_suffix="$3"
openssl() { printf '%s' "$fake_project_suffix"; }
docker() {
  printf '%s|%s\n' "${COMPOSE_PROJECT_NAME-}" "$*" >>"$project_log"
  return 0
}
initialize_compose_project
sh -c 'test "$COMPOSE_PROJECT_NAME" = "$1"' isolation-child "$COMPOSE_PROJECT_NAME"
cleanup_required=1
finish
"""
    with tempfile.TemporaryDirectory(prefix="rag-acceptance-project.", dir="/tmp") as temp_dir:
        project_log = Path(temp_dir) / "compose.log"
        completed = subprocess.run(
            [
                "bash",
                "-c",
                command,
                "acceptance-project-test",
                str(script),
                str(project_log),
                project_suffix,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        logged_calls = project_log.read_text().splitlines() if project_log.exists() else []

    _assert_sanitized_process(completed, returncode=0, stdout="", stderr="")
    assert len(logged_calls) == 2
    assert all(call.startswith(f"{expected_project}|") for call in logged_calls)
    assert any("down --remove-orphans --volumes" in call for call in logged_calls)
    assert any("ps -a --format json" in call for call in logged_calls)


@pytest.mark.acceptance
def test_unified_acceptance_gate_isolates_image_ports_and_smoke_ownership() -> None:
    script = Path(__file__).parents[2] / "scripts" / "acceptance_ingestion_retrieval.sh"
    command = """
source "$1"
call_log="$2"
export RAG_IMAGE_TAG=local
docker() {
  printf 'docker|%s|%s|%s\n' \
    "${COMPOSE_PROJECT_NAME-}" "${RAG_IMAGE_TAG-}" "$*" >>"$call_log"
  case "$*" in
    *' port api 8000') printf '%s\n' '127.0.0.1:49152' ;;
  esac
  return 0
}
bash() {
  test -n "${RAG_SMOKE_PROVIDER_SECRET-}" || return 91
  printf 'smoke|%s|%s|%s|%s|%s|%s|%s|%s\n' \
    "${COMPOSE_PROJECT_NAME-}" \
    "${RAG_IMAGE_TAG-}" \
    "${RAG_BASE_URL-}" \
    "${RAG_ACCEPTANCE_EPHEMERAL-}" \
    "${RAG_ACCEPTANCE_OWNED_PROJECT-}" \
    "${RAG_ACCEPTANCE_OWNED_IMAGE_TAG-}" \
    "${RAG_ACCEPTANCE_COMPOSE_OVERRIDE-}" \
    "$*" >>"$call_log"
  return 0
}
main
"""
    with tempfile.TemporaryDirectory(prefix="rag-acceptance-build.", dir="/tmp") as temp_dir:
        call_log = Path(temp_dir) / "calls.log"
        completed = subprocess.run(
            ["bash", "-c", command, "acceptance-build-test", str(script), str(call_log)],
            check=False,
            capture_output=True,
            text=True,
        )
        calls = call_log.read_text().splitlines() if call_log.exists() else []

    _assert_sanitized_process(
        completed,
        returncode=0,
        stdout="RAG ingestion/retrieval unified acceptance gate passed\n",
        stderr="",
    )
    projects = {call.split("|", 3)[1] for call in calls}
    if len(projects) != 1:
        raise AssertionError("ACCEPTANCE_PROJECT_OWNERSHIP_MISMATCH") from None
    project = projects.pop()
    if not project.startswith("rag-acceptance-"):
        raise AssertionError("ACCEPTANCE_PROJECT_FORMAT_MISMATCH") from None
    image_tag = project
    docker_calls = [call for call in calls if call.startswith("docker|")]
    compose_calls = [call for call in docker_calls if "|compose " in call]
    if not compose_calls:
        raise AssertionError("ACCEPTANCE_COMPOSE_CALLS_MISSING") from None
    compose_prefix = (
        f"compose --project-name {project} "
        f"-f {script.parents[1] / 'compose.yaml'} "
        f"-f {script.parents[1] / 'compose.acceptance.yaml'} "
        "--profile provider-stub"
    )
    for call in compose_calls:
        _, call_project, call_image_tag, arguments = call.split("|", 3)
        if call_project != project or call_image_tag != image_tag:
            raise AssertionError("ACCEPTANCE_COMPOSE_ENV_OWNERSHIP_MISMATCH") from None
        if not arguments.startswith(compose_prefix):
            raise AssertionError("ACCEPTANCE_COMPOSE_OVERRIDE_MISSING") from None
    build_calls = [call for call in compose_calls if call.endswith(" build api")]
    if len(build_calls) != 1:
        raise AssertionError("ACCEPTANCE_BUILD_COUNT_MISMATCH") from None
    build_index = calls.index(build_calls[0])
    up_index = next(
        index
        for index, call in enumerate(calls)
        if call.endswith(" up -d --no-build --wait --wait-timeout 120")
    )
    port_index = next(index for index, call in enumerate(calls) if call.endswith(" port api 8000"))
    smoke_index = next(index for index, call in enumerate(calls) if call.startswith("smoke|"))
    if not build_index < up_index < port_index < smoke_index:
        raise AssertionError("ACCEPTANCE_BUILD_UP_PORT_SMOKE_ORDER_MISMATCH") from None
    smoke = calls[smoke_index].split("|", 8)
    expected_override = str(script.parents[1] / "compose.acceptance.yaml")
    if smoke[1:8] != [
        project,
        image_tag,
        "http://127.0.0.1:49152",
        "1",
        project,
        image_tag,
        expected_override,
    ]:
        raise AssertionError("ACCEPTANCE_SMOKE_OWNERSHIP_MISMATCH") from None
    expected_image = f"rag-service:{image_tag}"
    if not any(call.endswith(f"image inspect {expected_image}") for call in docker_calls):
        raise AssertionError("ACCEPTANCE_IMAGE_INSPECTION_MISSING") from None
    if not any(call.endswith(f"image rm {expected_image}") for call in docker_calls):
        raise AssertionError("ACCEPTANCE_IMAGE_CLEANUP_MISSING") from None


@pytest.mark.acceptance
def test_unified_acceptance_gate_rejects_non_loopback_dynamic_port() -> None:
    script = Path(__file__).parents[2] / "scripts" / "acceptance_ingestion_retrieval.sh"
    command = """
source "$1"
call_log="$2"
docker() {
  printf 'docker|%s\n' "$*" >>"$call_log"
  case "$*" in
    *' port api 8000') printf '%s\n' '0.0.0.0:49152' ;;
  esac
  return 0
}
bash() {
  printf '%s\n' smoke >>"$call_log"
  return 0
}
main
"""
    with tempfile.TemporaryDirectory(prefix="rag-acceptance-port.", dir="/tmp") as temp_dir:
        call_log = Path(temp_dir) / "calls.log"
        completed = subprocess.run(
            ["bash", "-c", command, "acceptance-port-test", str(script), str(call_log)],
            check=False,
            capture_output=True,
            text=True,
        )
        calls = call_log.read_text().splitlines() if call_log.exists() else []

    _assert_sanitized_process(
        completed,
        returncode=1,
        stdout="",
        stderr=(
            "RAG ingestion/retrieval acceptance gate failed: dynamic API port discovery failed\n"
        ),
    )
    if any(call == "smoke" for call in calls):
        raise AssertionError("ACCEPTANCE_INVALID_PORT_REACHED_SMOKE") from None
    if not any(" down --remove-orphans --volumes" in call for call in calls):
        raise AssertionError("ACCEPTANCE_INVALID_PORT_CLEANUP_MISSING") from None


@pytest.mark.acceptance
def test_unified_acceptance_gate_sanitizes_build_failure_and_cleans_project() -> None:
    script = Path(__file__).parents[2] / "scripts" / "acceptance_ingestion_retrieval.sh"
    command = """
source "$1"
call_log="$2"
docker() {
  printf 'docker|%s|%s\n' "${COMPOSE_PROJECT_NAME-}" "$*" >>"$call_log"
  case "$*" in
    *' build api') return 1 ;;
    *) return 0 ;;
  esac
}
bash() {
  printf 'smoke|%s|%s\n' "${COMPOSE_PROJECT_NAME-}" "$*" >>"$call_log"
  return 0
}
main
"""
    with tempfile.TemporaryDirectory(
        prefix="rag-acceptance-build-failure.", dir="/tmp"
    ) as temp_dir:
        call_log = Path(temp_dir) / "calls.log"
        completed = subprocess.run(
            [
                "bash",
                "-c",
                command,
                "acceptance-build-failure-test",
                str(script),
                str(call_log),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        calls = call_log.read_text().splitlines() if call_log.exists() else []

    _assert_sanitized_process(
        completed,
        returncode=1,
        stdout="",
        stderr=(
            "RAG ingestion/retrieval acceptance gate failed: "
            "Compose application image build failed\n"
        ),
    )
    assert any(call.endswith(" build api") for call in calls)
    assert any(call.endswith(" down --remove-orphans --volumes") for call in calls)
    assert any(call.endswith(" ps -a --format json") for call in calls)
    assert not any(" up " in call for call in calls)
    assert not any(call.startswith("smoke|") for call in calls)


@pytest.mark.acceptance
def test_smoke_script_refuses_direct_invocation_before_external_calls() -> None:
    script = Path(__file__).parents[2] / "scripts" / "smoke_ingestion_retrieval.sh"
    with tempfile.TemporaryDirectory(prefix="rag-smoke-ownership.", dir="/tmp") as temp_dir:
        temp = Path(temp_dir)
        call_log = temp / "external-calls.log"
        for executable in ("curl", "docker"):
            fake = temp / executable
            fake.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' called >>\"${RAG_TEST_CALL_LOG:?}\"\n"
                "exit 97\n",
                encoding="utf-8",
            )
            fake.chmod(0o700)
        completed = subprocess.run(
            ["/bin/bash", str(script)],
            check=False,
            capture_output=True,
            text=True,
            env={
                "PATH": f"{temp_dir}:/usr/local/bin:/usr/bin:/bin",
                "RAG_SMOKE_PROVIDER_SECRET": uuid4().hex,
                "RAG_TEST_CALL_LOG": str(call_log),
            },
        )
        external_called = call_log.exists()

    _assert_sanitized_process(
        completed,
        returncode=1,
        stdout="",
        stderr=(
            "RAG ingestion/retrieval smoke test failed: "
            "ephemeral acceptance ownership contract was invalid\n"
        ),
    )
    if external_called:
        raise AssertionError("DIRECT_SMOKE_PERFORMED_EXTERNAL_CALL") from None


@pytest.mark.acceptance
def test_task16c_scripts_use_isolation_instead_of_global_lock_and_shasum() -> None:
    repo_root = Path(__file__).parents[2]
    acceptance_source = (repo_root / "scripts" / "acceptance_ingestion_retrieval.sh").read_text(
        encoding="utf-8"
    )
    smoke_source = (repo_root / "scripts" / "smoke_ingestion_retrieval.sh").read_text(
        encoding="utf-8"
    )
    if "shasum" in acceptance_source:
        raise AssertionError("ACCEPTANCE_SHASUM_STILL_USED") from None
    if "openssl dgst -sha256 -r" not in acceptance_source:
        raise AssertionError("ACCEPTANCE_OPENSSL_DIGEST_MISSING") from None
    if "rag-service-ingestion-smoke.lock" in smoke_source:
        raise AssertionError("SMOKE_GLOBAL_LOCK_STILL_PRESENT") from None


@pytest.mark.acceptance
def test_smoke_script_json_helpers_fail_without_tracebacks() -> None:
    script = Path(__file__).parents[2] / "scripts" / "smoke_ingestion_retrieval.sh"
    command = """
source "$1"
printf '%s' 'not-json' >"$2"
if json_uuid "$2" "id"; then
  exit 1
fi
"""
    with tempfile.TemporaryDirectory(prefix="rag-smoke-json.", dir="/tmp") as temp_dir:
        invalid = Path(temp_dir) / "invalid.json"
        completed = subprocess.run(
            ["bash", "-c", command, "smoke-json-test", str(script), str(invalid)],
            check=False,
            capture_output=True,
            text=True,
        )

    _assert_sanitized_process(completed, returncode=0, stdout="", stderr="")


@pytest.mark.acceptance
def test_smoke_script_etag_helper_rejects_header_injection() -> None:
    script = Path(__file__).parents[2] / "scripts" / "smoke_ingestion_retrieval.sh"
    resource_id = "00000000-0000-0000-0000-000000000031"
    command = """
source "$1"
if json_etag "$2" "etag" "kb" "$3"; then
  exit 1
fi
"""
    with tempfile.TemporaryDirectory(prefix="rag-smoke-etag.", dir="/tmp") as temp_dir:
        response = Path(temp_dir) / "response.json"
        response.write_text(
            json.dumps({"etag": f'"kb:{resource_id}:r1"\r\nX-Test: injected'}),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [
                "bash",
                "-c",
                command,
                "smoke-etag-test",
                str(script),
                str(response),
                resource_id,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    _assert_sanitized_process(completed, returncode=0, stdout="", stderr="")


@pytest.mark.acceptance
def test_smoke_script_reports_only_safe_http_failure_fields() -> None:
    script = Path(__file__).parents[2] / "scripts" / "smoke_ingestion_retrieval.sh"
    command = """
source "$1"
provider_secret_file="$2/provider-secret"
printf '%s' "$$-$RANDOM" >"$provider_secret_file"
printf '%s' '{"error":{"code":"SAFE_FAILURE","message":"unsafe body sentinel"}}' >"$2/response.json"
http_status="422"
if expect_status "generation" "201" "$2/response.json"; then
  exit 1
fi
"""
    with tempfile.TemporaryDirectory(prefix="rag-smoke-error.", dir="/tmp") as temp_dir:
        completed = subprocess.run(
            ["bash", "-c", command, "smoke-error-test", str(script), temp_dir],
            check=False,
            capture_output=True,
            text=True,
        )

    _assert_sanitized_process(
        completed,
        returncode=0,
        stdout="",
        stderr=(
            "RAG ingestion/retrieval smoke test failed: generation returned HTTP 422 "
            "(expected 201, code SAFE_FAILURE)\n"
        ),
    )
    _assert_protected_content_absent(completed.stderr, "unsafe body sentinel")


@pytest.mark.acceptance
def test_smoke_search_validator_accepts_flattened_index_metadata() -> None:
    script = Path(__file__).parents[2] / "scripts" / "smoke_ingestion_retrieval.sh"
    document_id = "00000000-0000-0000-0000-000000000011"
    version_id = "00000000-0000-0000-0000-000000000012"
    generation_id = "00000000-0000-0000-0000-000000000013"
    profile_id = "00000000-0000-0000-0000-000000000014"
    command = """
source "$1"
assert_search_response "$2" "$3" "$4" "$5" "$6" "$7"
"""
    with tempfile.TemporaryDirectory(prefix="rag-smoke-search.", dir="/tmp") as temp_dir:
        temp = Path(temp_dir)
        document = temp / "document.md"
        document.write_text(
            "# Acceptance Guide\n\nThe cerulean orchard phrase confirms retrieval.\n",
            encoding="utf-8",
        )
        response = temp / "response.json"
        response.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "text": "The cerulean orchard phrase confirms retrieval.",
                            "score": 0.95,
                            "document_id": document_id,
                            "version_id": version_id,
                            "chunk_index": 0,
                            "title": "Acceptance Guide",
                            "title_path": ["Acceptance Guide"],
                            "source": {
                                "filename": "Acceptance Guide",
                                "start_offset": 0,
                                "end_offset": len(document.read_text(encoding="utf-8")),
                            },
                            "metadata": {"department": "acceptance"},
                        }
                    ],
                    "index": {
                        "generation_id": generation_id,
                        "embedding_profile_id": profile_id,
                    },
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [
                "bash",
                "-c",
                command,
                "smoke-search-test",
                str(script),
                str(response),
                document_id,
                version_id,
                generation_id,
                profile_id,
                str(document),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    _assert_sanitized_process(completed, returncode=0, stdout="", stderr="")


@pytest.mark.acceptance
@pytest.mark.parametrize(
    ("title_path", "source_filename", "expected_code"),
    (
        ([], "Acceptance Guide", "SEARCH_TITLE_PATH_MISMATCH"),
        (
            ["Acceptance Guide"],
            "unexpected.md",
            "SEARCH_SOURCE_FILENAME_MISMATCH",
        ),
    ),
)
def test_smoke_search_validator_reports_only_allowlisted_mismatch_code(
    title_path: list[str],
    source_filename: str,
    expected_code: str,
) -> None:
    script = Path(__file__).parents[2] / "scripts" / "smoke_ingestion_retrieval.sh"
    document_id = "00000000-0000-0000-0000-000000000021"
    version_id = "00000000-0000-0000-0000-000000000022"
    generation_id = "00000000-0000-0000-0000-000000000023"
    profile_id = "00000000-0000-0000-0000-000000000024"
    command = """
source "$1"
assert_search_response "$2" "$3" "$4" "$5" "$6" "$7"
"""
    with tempfile.TemporaryDirectory(prefix="rag-smoke-search-code.", dir="/tmp") as temp_dir:
        temp = Path(temp_dir)
        document = temp / "document.md"
        document.write_text(
            "# Acceptance Guide\n\nThe cerulean orchard phrase confirms retrieval.\n",
            encoding="utf-8",
        )
        response = temp / "response.json"
        response.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "text": "The cerulean orchard phrase confirms retrieval.",
                            "score": 0.95,
                            "document_id": document_id,
                            "version_id": version_id,
                            "chunk_index": 0,
                            "title": "Acceptance Guide",
                            "title_path": title_path,
                            "source": {
                                "filename": source_filename,
                                "start_offset": 0,
                                "end_offset": len(document.read_text(encoding="utf-8")),
                            },
                            "metadata": {"department": "acceptance"},
                        }
                    ],
                    "index": {
                        "generation_id": generation_id,
                        "embedding_profile_id": profile_id,
                    },
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [
                "bash",
                "-c",
                command,
                "smoke-search-code-test",
                str(script),
                str(response),
                document_id,
                version_id,
                generation_id,
                profile_id,
                str(document),
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    _assert_sanitized_process(
        completed,
        returncode=1,
        stdout=f"{expected_code}\n",
        stderr="",
    )


@pytest.mark.asyncio
async def test_upload_route_cleans_disconnect_and_rejects_malformed_or_oversized_fields() -> None:
    kb_id = uuid4()
    boundary = "rag-http-invalid-boundary"
    cases: tuple[tuple[AsyncIterable[bytes], int, str], ...] = (
        (
            streamed_file_multipart(
                boundary,
                2 * 1024 * 1024,
                fail_after_bytes=1024 * 1024,
            ),
            500,
            "INTERNAL_ERROR",
        ),
        (
            streamed_file_multipart(boundary, 8),
            202,
            "",
        ),
        (
            iter_async(
                (
                    f'--{boundary}\r\nContent-Disposition: form-data; name="display_name"\r\n\r\n'
                ).encode()
                + b"a" * (32 * 1024 + 1)
                + f"\r\n--{boundary}--\r\n".encode()
            ),
            422,
            "VALIDATION_ERROR",
        ),
        (
            iter_async(
                (
                    f"--{boundary}\r\n"
                    'Content-Disposition: form-data; name="file"; filename="doc.txt"\r\n'
                    "Content-Type: text/plain\r\n\r\nhello"
                ).encode()
            ),
            422,
            "VALIDATION_ERROR",
        ),
    )
    for index, (body, status, code) in enumerate(cases):
        service, repository, store = capturing_service(kb_id)
        app = create_app()
        app.dependency_overrides[require_agent_principal] = lambda: principal(kb_id)
        app.dependency_overrides[get_document_upload_service] = upload_service_dependency(service)
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                f"/v1/knowledge-bases/{kb_id}/documents",
                headers={
                    "X-Request-ID": f"req-upload-invalid-{index}",
                    "Content-Type": f"multipart/form-data; boundary={boundary}",
                },
                content=body,
            )
        assert response.status_code == status, response.text
        if code:
            assert response.json()["error"]["code"] == code
        assert "disconnect-secret" not in response.text
        if status != 202:
            assert repository.reservations == []
            assert store.objects == {}


async def iter_async(value: bytes) -> AsyncIterator[bytes]:
    yield value


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("filename", "content_type", "content", "code"),
    (
        ("doc.txt", "text/plain", b"\xff", "INVALID_TEXT_ENCODING"),
        ("doc.txt", "text/plain", b"\x00binary", "BINARY_CONTENT_REJECTED"),
        ("doc.txt", "text/plain", b"", "EMPTY_DOCUMENT"),
        ("doc.pdf", "text/plain", b"hello", "UNSUPPORTED_DOCUMENT_TYPE"),
        ("doc.txt", "image/png", b"hello", "UNSUPPORTED_DOCUMENT_TYPE"),
    ),
)
async def test_upload_route_returns_safe_content_mime_and_extension_errors(
    filename: str,
    content_type: str,
    content: bytes,
    code: str,
) -> None:
    kb_id = uuid4()
    service, repository, store = capturing_service(kb_id)
    app = create_app()
    app.dependency_overrides[require_agent_principal] = lambda: principal(kb_id)
    app.dependency_overrides[get_document_upload_service] = lambda: service
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/v1/knowledge-bases/{kb_id}/documents",
            headers={"X-Request-ID": f"req-upload-content-{code}"},
            files={"file": (filename, content, content_type)},
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == code
    assert repository.reservations == []
    assert store.objects == {}


@pytest.mark.asyncio
async def test_upload_route_accepts_cjk_and_emoji_utf8_text() -> None:
    kb_id = uuid4()
    content = "中文知识库🙂\n第二行".encode()
    service, repository, store = capturing_service(kb_id)
    app = create_app()
    app.dependency_overrides[require_agent_principal] = lambda: principal(kb_id)
    app.dependency_overrides[get_document_upload_service] = lambda: service
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/v1/knowledge-bases/{kb_id}/documents",
            headers={"X-Request-ID": "req-upload-cjk-emoji"},
            files={"file": ("知识.md", content, "text/markdown")},
        )

    assert response.status_code == 202, response.text
    assert len(repository.reservations) == 1
    assert list(store.objects.values()) == [content]


@pytest.mark.asyncio
async def test_upload_route_enforces_rfc3339_metadata_from_active_generation() -> None:
    kb_id = uuid4()
    service, repository, store = capturing_service(kb_id)
    repository.preflight_value = replace(
        repository.preflight_value,
        filter_schema_snapshot={
            "fields": [
                {
                    "name": "published_at",
                    "source_path": "published_at",
                    "type": "datetime",
                }
            ]
        },
    )
    app = create_app()
    app.dependency_overrides[require_agent_principal] = lambda: principal(kb_id)
    app.dependency_overrides[get_document_upload_service] = lambda: service
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/v1/knowledge-bases/{kb_id}/documents",
            headers={"X-Request-ID": "req-upload-datetime"},
            files={"file": ("doc.txt", b"hello", "text/plain")},
            data={"metadata": '{"published_at":"contains-T-but-invalid"}'},
        )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"
    assert repository.reservations == []
    assert store.objects == {}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_parse_stage_publishes_and_atomically_advances_without_activation(
    migrated_database: Database,
) -> None:
    seeded = await seed_upload_context(migrated_database)
    store = MemoryObjectStore()
    uploaded = await database_upload(
        migrated_database,
        seeded,
        store,
        content=b"\xef\xbb\xbffirst\r\nsecond",
        idempotency_key="parse-stage-success",
    )
    lease = await claim_upload_job(migrated_database, uploaded.job_id)

    pipeline = ingestion_pipeline(migrated_database, store)
    context = JobExecutionContext(
        job_repository_context(migrated_database),
        lease,
        timedelta(seconds=30),
    )

    outcome = await pipeline.handle(context)

    assert outcome.value == "continue"
    assert context.lease.stage == "chunk"
    assert context.lease.attempt_count == 1
    async with migrated_database.sessions() as session:
        version = await session.get(DocumentVersion, uploaded.version_id)
        document = await session.get(Document, uploaded.document_id)
        job = await session.get(Job, uploaded.job_id)
    assert version is not None and document is not None and job is not None
    final_key = parsed_text_object_key(
        seeded.knowledge_base_id,
        uploaded.document_id,
        uploaded.version_id,
    )
    assert version.status == "chunking"
    assert version.parsed_object_key == final_key
    assert version.parsed_object_checksum_sha256 == hashlib.sha256(b"first\nsecond").hexdigest()
    assert version.parser_name == "plain_text_v1"
    assert version.parser_version == "1"
    assert version.parser_config == {}
    assert job.status == "running"
    assert job.stage == "chunk"
    assert job.resume_stage == "chunk"
    assert job.attempt_count == 1
    assert job.lease_owner == "pipeline-worker"
    assert document.status == "processing"
    assert document.current_version_id is None
    assert document.pending_version_id == uploaded.version_id
    assert store.objects[final_key] == b"first\nsecond"
    explicit_temp = temporary_object_key(
        uploaded.job_id,
        lease.lease_epoch,
        "parsed/text.txt",
    )
    assert explicit_temp not in store.objects
    assert explicit_temp in store.delete_calls


@pytest.mark.integration
@pytest.mark.asyncio
async def test_parse_stage_cpu_work_does_not_block_the_event_loop(
    migrated_database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline_module = importlib.import_module("rag_service.ingestion.pipeline")
    parser_module = importlib.import_module("rag_service.ingestion.parsers")
    seeded = await seed_upload_context(migrated_database)
    store = MemoryObjectStore()
    uploaded = await database_upload(
        migrated_database,
        seeded,
        store,
        content=b"nonblocking parser",
        idempotency_key="parse-stage-nonblocking",
    )
    lease = await claim_upload_job(migrated_database, uploaded.job_id)
    real_parser = parser_module.parser_for_extension(".txt")
    started = threading.Event()
    release = threading.Event()

    class SlowParser:
        name = real_parser.name
        version = real_parser.version
        config = real_parser.config

        def parse(self, source: bytes) -> object:
            started.set()
            release.wait(0.4)
            return real_parser.parse(source)

    monkeypatch.setattr(pipeline_module, "parser_for_extension", lambda _extension: SlowParser())
    context = JobExecutionContext(
        job_repository_context(migrated_database),
        lease,
        timedelta(seconds=30),
    )
    handler = asyncio.create_task(ingestion_pipeline(migrated_database, store).handle(context))
    before = time.monotonic()
    try:
        assert await asyncio.to_thread(started.wait, 1)
        async with migrated_database.sessions() as session:
            version = await session.get(DocumentVersion, uploaded.version_id)
        assert version is not None
        assert version.status == "parsing"
        release.set()
        await handler
    finally:
        release.set()
    assert time.monotonic() - before < 0.25


@pytest.mark.integration
@pytest.mark.asyncio
async def test_parse_stage_rejects_oversized_structural_metadata_with_stable_code(
    migrated_database: Database,
) -> None:
    pipeline_module = importlib.import_module("rag_service.ingestion.pipeline")
    seeded = await seed_upload_context(migrated_database)
    store = MemoryObjectStore()
    heading = "界" * (256 * 1024)
    body = "body\n" * ((256 * 1024) // 5)
    uploaded = await database_upload(
        migrated_database,
        seeded,
        store,
        content=f"# {heading}\n{body}".encode(),
        idempotency_key="parse-stage-structural-metadata-limit",
        filename="document.md",
        content_type="text/markdown",
    )
    lease = await claim_upload_job(migrated_database, uploaded.job_id)
    context = JobExecutionContext(
        job_repository_context(migrated_database),
        lease,
        timedelta(seconds=30),
    )
    pipeline = ingestion_pipeline(migrated_database, store)
    try:
        with pytest.raises(pipeline_module.PermanentJobError) as exc_info:
            await pipeline.handle(context)
    finally:
        await pipeline.aclose()

    assert exc_info.value.code == "STRUCTURAL_METADATA_TOO_LARGE"
    assert exc_info.value.safe_message == "Document structural metadata exceeds limit"
    async with migrated_database.sessions() as session:
        version = await session.get(DocumentVersion, uploaded.version_id)
        job = await session.get(Job, uploaded.job_id)
    assert version is not None and job is not None
    assert version.status == "parsing"
    assert version.parsed_object_key is None
    assert job.stage == "parse"
    assert not any(key.endswith("/parsed/text.txt") for key in store.objects)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cancelled_stale_epoch_cpu_work_holds_slot_until_reclaim_retry(
    migrated_database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline_module = importlib.import_module("rag_service.ingestion.pipeline")
    parser_module = importlib.import_module("rag_service.ingestion.parsers")
    seeded = await seed_upload_context(migrated_database)
    store = MemoryObjectStore()
    uploaded = await database_upload(
        migrated_database,
        seeded,
        store,
        content=b"lease loss parser",
        idempotency_key="parse-stage-cpu-lease-loss",
    )
    stale_lease = await claim_upload_job(migrated_database, uploaded.job_id)
    real_parser = parser_module.parser_for_extension(".txt")
    started = (threading.Event(), threading.Event())
    release = (threading.Event(), threading.Event())
    call_lock = threading.Lock()
    call_count = 0

    class BlockingParser:
        name = real_parser.name
        version = real_parser.version
        config = real_parser.config

        def parse(self, source: bytes) -> object:
            nonlocal call_count
            with call_lock:
                call_index = call_count
                call_count += 1
            started[call_index].set()
            release[call_index].wait(2)
            return real_parser.parse(source)

    parser = BlockingParser()
    monkeypatch.setattr(pipeline_module, "parser_for_extension", lambda _extension: parser)
    pipeline = ingestion_pipeline(migrated_database, store, cpu_concurrency=1)
    stale_context = JobExecutionContext(
        job_repository_context(migrated_database),
        stale_lease,
        timedelta(seconds=30),
    )
    stale_task = asyncio.create_task(pipeline.handle(stale_context))
    retry_task: asyncio.Task[object] | None = None
    try:
        assert await asyncio.to_thread(started[0].wait, 1)
        retry_lease = await expire_and_reclaim_upload_job(
            migrated_database,
            uploaded.job_id,
            lease_owner="pipeline-worker-b",
        )
        stale_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stale_task

        retry_context = JobExecutionContext(
            job_repository_context(migrated_database),
            retry_lease,
            timedelta(seconds=30),
        )
        retry_task = asyncio.create_task(pipeline.handle(retry_context))
        await asyncio.sleep(0.05)
        assert call_count == 1
        assert started[1].is_set() is False

        release[0].set()
        assert await asyncio.to_thread(started[1].wait, 1)
        release[1].set()
        await retry_task
        assert retry_context.lease.stage == "chunk"
        assert retry_context.lease.lease_epoch == stale_lease.lease_epoch + 1
    finally:
        release[0].set()
        release[1].set()
        for task in (stale_task, retry_task):
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        await pipeline.aclose()


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("position", ["before_copy", "after_copy"])
async def test_parse_stage_new_epoch_retry_reuses_final_after_object_copy_crash(
    migrated_database: Database,
    *,
    position: str,
) -> None:
    seeded = await seed_upload_context(migrated_database)
    store = StageCrashStore()
    uploaded = await database_upload(
        migrated_database,
        seeded,
        store,
        content=b"parse crash recovery",
        idempotency_key=f"parse-stage-{position}",
    )
    lease = await claim_upload_job(migrated_database, uploaded.job_id)
    context = JobExecutionContext(
        job_repository_context(migrated_database),
        lease,
        timedelta(seconds=30),
    )
    pipeline = ingestion_pipeline(migrated_database, store)
    final_key = parsed_text_object_key(
        seeded.knowledge_base_id,
        uploaded.document_id,
        uploaded.version_id,
    )
    store.arm("/parsed/text.txt", position)

    with pytest.raises(SimulatedStageCrash):
        await pipeline.handle(context)

    async with migrated_database.sessions() as session:
        version = await session.get(DocumentVersion, uploaded.version_id)
        job = await session.get(Job, uploaded.job_id)
    assert version is not None and job is not None
    assert version.status == "parsing"
    assert version.parsed_object_key is None
    assert job.stage == "parse"
    assert (final_key in store.objects) is (position == "after_copy")
    stale_temp = temporary_object_key(
        uploaded.job_id,
        lease.lease_epoch,
        "parsed/text.txt",
    )
    assert stale_temp not in store.objects
    assert stale_temp in store.delete_calls

    retry_lease = await expire_and_reclaim_upload_job(
        migrated_database,
        uploaded.job_id,
        lease_owner="pipeline-worker-b",
    )
    retry_context = JobExecutionContext(
        job_repository_context(migrated_database),
        retry_lease,
        timedelta(seconds=30),
    )
    await pipeline.handle(retry_context)
    assert retry_context.lease.stage == "chunk"
    assert retry_context.lease.lease_epoch == lease.lease_epoch + 1
    assert retry_context.lease.attempt_count == 2
    assert store.objects[final_key] == b"parse crash recovery"
    assert store.publish_reused[-1] is (position == "after_copy")


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("position", ["before_commit", "after_commit"])
async def test_parse_stage_object_and_database_commit_crash_boundaries(
    migrated_database: Database,
    *,
    position: str,
) -> None:
    seeded = await seed_upload_context(migrated_database)
    store = MemoryObjectStore()
    uploaded = await database_upload(
        migrated_database,
        seeded,
        store,
        content=b"parse commit recovery",
        idempotency_key=f"parse-stage-{position}",
    )
    lease = await claim_upload_job(migrated_database, uploaded.job_id)
    repository_uses = 0

    @asynccontextmanager
    async def crashing_job_repository() -> AsyncIterator[JobRepository]:
        nonlocal repository_uses
        repository_uses += 1
        async with migrated_database.sessions() as session:
            if position == "before_commit":
                async with session.begin():
                    yield BeforeCommitRepository(session)
            else:
                async with session.begin():
                    yield SqlAlchemyJobRepository(session)
                if repository_uses == 2:
                    raise SimulatedStageCrash("simulated crash after fencing commit")

    context = JobExecutionContext(
        crashing_job_repository,
        lease,
        timedelta(seconds=30),
    )
    pipeline = ingestion_pipeline(migrated_database, store)
    final_key = parsed_text_object_key(
        seeded.knowledge_base_id,
        uploaded.document_id,
        uploaded.version_id,
    )

    with pytest.raises(SimulatedStageCrash):
        await pipeline.handle(context)

    async with migrated_database.sessions() as session:
        version = await session.get(DocumentVersion, uploaded.version_id)
        job = await session.get(Job, uploaded.job_id)
    assert version is not None and job is not None
    assert final_key in store.objects
    if position == "before_commit":
        assert version.status == "parsing"
        assert version.parsed_object_key is None
        assert job.stage == "parse"
        retry_lease = await expire_and_reclaim_upload_job(
            migrated_database,
            uploaded.job_id,
            lease_owner="pipeline-worker-b",
        )
        retry_context = JobExecutionContext(
            job_repository_context(migrated_database),
            retry_lease,
            timedelta(seconds=30),
        )
        await pipeline.handle(retry_context)
        assert retry_context.lease.stage == "chunk"
        assert retry_context.lease.lease_epoch == lease.lease_epoch + 1
        assert store.publish_reused[-1] is True
    else:
        assert version.status == "chunking"
        assert version.parsed_object_key == final_key
        assert job.stage == "chunk"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_parse_stage_repository_rejects_uploaded_to_chunking_shortcut(
    migrated_database: Database,
) -> None:
    seeded = await seed_upload_context(migrated_database)
    store = MemoryObjectStore()
    uploaded = await database_upload(
        migrated_database,
        seeded,
        store,
        content=b"parse lifecycle guard",
        idempotency_key="parse-stage-lifecycle-guard",
    )

    async with migrated_database.sessions() as session:
        expected = await SqlAlchemyIngestionPipelineRepository(session).load_parse_stage(
            uploaded.version_id,
            seeded.generation_id,
        )
    assert expected.version_status == "uploaded"

    with pytest.raises(ValueError, match="ingestion stage state is invalid"):
        async with migrated_database.sessions() as session, session.begin():
            await SqlAlchemyIngestionPipelineRepository(session).commit_parse_stage(
                expected,
                parsed_object_key="kb/parsed/text.txt",
                parsed_checksum_sha256=hashlib.sha256(b"parsed").hexdigest(),
                parser_name=expected.parser_name,
                parser_version=expected.parser_version,
                parser_config=expected.parser_config,
            )

    async with migrated_database.sessions() as session:
        version = await session.get(DocumentVersion, uploaded.version_id)
    assert version is not None
    assert version.status == "uploaded"
    assert version.parsed_object_key is None


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["source_checksum", "final_conflict"])
async def test_parse_stage_rejects_source_checksum_and_final_artifact_conflicts(
    migrated_database: Database,
    *,
    failure: str,
) -> None:
    pipeline_module = importlib.import_module("rag_service.ingestion.pipeline")
    seeded = await seed_upload_context(migrated_database)
    store = MemoryObjectStore()
    uploaded = await database_upload(
        migrated_database,
        seeded,
        store,
        content=b"parse checksum conflict",
        idempotency_key=f"parse-stage-{failure}",
    )
    lease = await claim_upload_job(migrated_database, uploaded.job_id)
    context = JobExecutionContext(
        job_repository_context(migrated_database),
        lease,
        timedelta(seconds=30),
    )
    if failure == "source_checksum":
        async with migrated_database.sessions() as session:
            version = await session.get(DocumentVersion, uploaded.version_id)
        assert version is not None
        store.objects[version.source_object_key] = b"tampered source bytes"
        expected_code = "OBJECT_VERIFICATION_FAILED"
    else:
        final_key = parsed_text_object_key(
            seeded.knowledge_base_id,
            uploaded.document_id,
            uploaded.version_id,
        )
        store.objects[final_key] = b"conflicting deterministic artifact"
        expected_code = "ARTIFACT_CHECKSUM_CONFLICT"

    with pytest.raises(pipeline_module.PermanentJobError) as exc_info:
        await ingestion_pipeline(migrated_database, store).handle(context)
    assert exc_info.value.code == expected_code
    async with migrated_database.sessions() as session:
        version = await session.get(DocumentVersion, uploaded.version_id)
        job = await session.get(Job, uploaded.job_id)
    assert version is not None and job is not None
    assert version.status == "parsing"
    assert version.parsed_object_key is None
    assert job.stage == "parse"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_terminal_failure_real_runner_atomically_fails_permanent_parse_error(
    migrated_database: Database,
) -> None:
    seeded = await seed_upload_context(migrated_database)
    store = MemoryObjectStore()
    source_content = b"permanent parse failure keeps its checksum claim"
    uploaded = await database_upload(
        migrated_database,
        seeded,
        store,
        content=source_content,
        idempotency_key="runner-parse-terminal",
    )
    async with migrated_database.sessions() as session:
        version_before = await session.get(DocumentVersion, uploaded.version_id)
    assert version_before is not None
    source_key = version_before.source_object_key
    source_checksum = version_before.source_checksum_sha256
    store.objects[source_key] = b"tampered source remains for diagnosis"

    pipeline = ingestion_pipeline(migrated_database, store)
    try:
        runner = ingestion_job_runner(
            migrated_database,
            pipeline,
            lease_owner="runner-parse-terminal-worker",
        )
        assert await runner.run_once(uploaded.job_id) is True
    finally:
        await pipeline.aclose()

    async with migrated_database.sessions() as session:
        document = await session.get(Document, uploaded.document_id)
        version = await session.get(DocumentVersion, uploaded.version_id)
        state = await session.get(
            DocumentIndexState,
            (uploaded.version_id, seeded.generation_id),
        )
        job = await session.get(Job, uploaded.job_id)
    assert document is not None and version is not None and state is not None and job is not None
    assert document.status == "failed"
    assert document.current_version_id is None
    assert document.pending_version_id == uploaded.version_id
    assert document.checksum_sha256 == source_checksum
    assert version.status == "failed"
    assert version.source_object_key == source_key
    assert version.source_checksum_sha256 == source_checksum
    assert version.parsed_object_key is None
    assert state.status == "failed"
    assert state.next_chunk_index == 0
    assert state.error_code == "OBJECT_VERIFICATION_FAILED"
    assert job.status == "failed"
    assert job.retryable is False
    assert job.error_code == "OBJECT_VERIFICATION_FAILED"
    assert job.lease_owner is None and job.lease_expires_at is None
    assert source_key in store.objects

    with pytest.raises(BusinessError) as duplicate:
        await database_upload(
            migrated_database,
            seeded,
            store,
            content=source_content,
            idempotency_key="runner-parse-terminal-duplicate",
        )
    assert duplicate.value.code == "DUPLICATE_DOCUMENT"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chunk_stage_streams_canonical_manifest_and_advances_same_attempt(
    migrated_database: Database,
) -> None:
    seeded = await seed_upload_context(migrated_database)
    store = MemoryObjectStore()
    content = (("alpha beta gamma delta " * 100) + "\n").encode()
    uploaded = await database_upload(
        migrated_database,
        seeded,
        store,
        content=content,
        idempotency_key="chunk-stage-success",
    )
    lease = await claim_upload_job(migrated_database, uploaded.job_id)
    context = JobExecutionContext(
        job_repository_context(migrated_database),
        lease,
        timedelta(seconds=30),
    )
    pipeline = ingestion_pipeline(migrated_database, store)

    await pipeline.handle(context)
    outcome = await pipeline.handle(context)

    assert outcome.value == "continue"
    assert context.lease.stage == "embed_index"
    assert context.lease.attempt_count == 1
    async with migrated_database.sessions() as session:
        version = await session.get(DocumentVersion, uploaded.version_id)
        state = await session.get(
            DocumentIndexState,
            (uploaded.version_id, seeded.generation_id),
        )
        document = await session.get(Document, uploaded.document_id)
        job = await session.get(Job, uploaded.job_id)
    assert version is not None and state is not None and document is not None and job is not None
    manifest_key = chunks_object_key(
        seeded.knowledge_base_id,
        uploaded.document_id,
        uploaded.version_id,
    )
    manifest_bytes = store.objects[manifest_key]
    records = [json.loads(line) for line in manifest_bytes.splitlines()]
    assert records[0]["schema_version"] == "chunks-v1"
    assert records[0]["chunk_count"] == len(records) - 1
    assert [record["chunk_index"] for record in records[1:]] == list(range(len(records) - 1))
    assert version.status == "embedding"
    assert version.chunk_manifest_object_key == manifest_key
    assert version.chunk_manifest_checksum_sha256 == hashlib.sha256(manifest_bytes).hexdigest()
    assert version.chunk_config_hash is not None
    assert version.chunker_name == "recursive_text_v1"
    assert version.chunker_version == "1"
    assert version.chunk_count == len(records) - 1
    assert state.status == "embedding"
    assert state.expected_point_count == len(records) - 1
    assert state.actual_point_count is None
    assert state.chunk_manifest_checksum_sha256 == version.chunk_manifest_checksum_sha256
    assert state.embedding_config_hash is None
    assert state.next_chunk_index == 0
    assert job.status == "running"
    assert job.stage == "embed_index"
    assert job.attempt_count == 1
    assert job.progress_current == 0
    assert job.progress_total == len(records) - 1
    assert document.status == "processing"
    assert document.current_version_id is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chunk_stage_rejects_non_pristine_index_checkpoint_without_rewinding(
    migrated_database: Database,
) -> None:
    pipeline_module = importlib.import_module("rag_service.ingestion.pipeline")
    seeded = await seed_upload_context(migrated_database)
    store = MemoryObjectStore()
    uploaded = await database_upload(
        migrated_database,
        seeded,
        store,
        content=b"checkpoint protection",
        idempotency_key="chunk-stage-pristine-index-state",
    )
    lease = await claim_upload_job(migrated_database, uploaded.job_id)
    context = JobExecutionContext(
        job_repository_context(migrated_database),
        lease,
        timedelta(seconds=30),
    )
    pipeline = ingestion_pipeline(migrated_database, store)
    await pipeline.handle(context)
    async with migrated_database.sessions() as session, session.begin():
        state = await session.get(
            DocumentIndexState,
            (uploaded.version_id, seeded.generation_id),
        )
        assert state is not None
        state.status = "embedding"
        state.expected_point_count = 99
        state.chunk_manifest_checksum_sha256 = "d" * 64
        state.embedding_config_hash = "e" * 64
        state.next_chunk_index = 7

    with pytest.raises(pipeline_module.PermanentJobError) as exc_info:
        await pipeline.handle(context)
    assert exc_info.value.code == "INGESTION_STAGE_CONFLICT"
    async with migrated_database.sessions() as session:
        state = await session.get(
            DocumentIndexState,
            (uploaded.version_id, seeded.generation_id),
        )
        version = await session.get(DocumentVersion, uploaded.version_id)
        job = await session.get(Job, uploaded.job_id)
    assert state is not None and version is not None and job is not None
    assert state.status == "embedding"
    assert state.expected_point_count == 99
    assert state.chunk_manifest_checksum_sha256 == "d" * 64
    assert state.embedding_config_hash == "e" * 64
    assert state.next_chunk_index == 7
    assert version.status == "chunking"
    assert version.chunk_manifest_object_key is None
    assert job.stage == "chunk"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chunk_stage_reports_generated_artifact_limit_without_mislabeling_document(
    migrated_database: Database,
) -> None:
    pipeline_module = importlib.import_module("rag_service.ingestion.pipeline")

    class ManifestLimitStore(MemoryObjectStore):
        async def upload_stream(
            self,
            object_key: str,
            stream: AsyncIterable[bytes],
            *,
            content_type: str,
            max_bytes: int,
        ) -> object:
            if object_key.endswith("/chunks/recursive_text_v1.jsonl"):
                raise UploadLimitExceeded
            return await super().upload_stream(
                object_key,
                stream,
                content_type=content_type,
                max_bytes=max_bytes,
            )

    seeded = await seed_upload_context(migrated_database)
    store = ManifestLimitStore()
    uploaded = await database_upload(
        migrated_database,
        seeded,
        store,
        content=b"generated artifact limit",
        idempotency_key="chunk-stage-generated-artifact-limit",
    )
    lease = await claim_upload_job(migrated_database, uploaded.job_id)
    context = JobExecutionContext(
        job_repository_context(migrated_database),
        lease,
        timedelta(seconds=30),
    )
    pipeline = ingestion_pipeline(migrated_database, store)
    await pipeline.handle(context)

    with pytest.raises(pipeline_module.PermanentJobError) as exc_info:
        await pipeline.handle(context)
    assert exc_info.value.code == "ARTIFACT_TOO_LARGE"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_chunk_stage_hard_budget_fails_before_manifest_upload_side_effect(
    migrated_database: Database,
) -> None:
    pipeline_module = importlib.import_module("rag_service.ingestion.pipeline")

    class ManifestUploadProbeStore(MemoryObjectStore):
        def __init__(self) -> None:
            super().__init__()
            self.manifest_upload_calls = 0

        async def upload_stream(
            self,
            object_key: str,
            stream: AsyncIterable[bytes],
            *,
            content_type: str,
            max_bytes: int,
        ) -> object:
            if object_key.endswith("/chunks/recursive_text_v1.jsonl"):
                self.manifest_upload_calls += 1
            return await super().upload_stream(
                object_key,
                stream,
                content_type=content_type,
                max_bytes=max_bytes,
            )

    seeded = await seed_upload_context(migrated_database)
    store = ManifestUploadProbeStore()
    uploaded = await database_upload(
        migrated_database,
        seeded,
        store,
        content=b"#\nx\n" * 1024,
        idempotency_key="chunk-stage-hard-manifest-budget",
        filename="document.md",
        content_type="text/markdown",
    )
    lease = await claim_upload_job(migrated_database, uploaded.job_id)
    context = JobExecutionContext(
        job_repository_context(migrated_database),
        lease,
        timedelta(seconds=30),
    )
    pipeline = ingestion_pipeline(
        migrated_database,
        store,
        max_manifest_bytes=1024,
    )
    try:
        await pipeline.handle(context)
        with pytest.raises(pipeline_module.PermanentJobError) as exc_info:
            await pipeline.handle(context)
    finally:
        await pipeline.aclose()

    assert exc_info.value.code == "ARTIFACT_TOO_LARGE"
    assert store.manifest_upload_calls == 0
    async with migrated_database.sessions() as session:
        version = await session.get(DocumentVersion, uploaded.version_id)
        job = await session.get(Job, uploaded.job_id)
    assert version is not None and job is not None
    assert version.status == "chunking"
    assert version.chunk_manifest_object_key is None
    assert job.stage == "chunk"


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("position", ["before_copy", "after_copy"])
async def test_chunk_stage_new_epoch_retry_reuses_byte_identical_final_after_copy_crash(
    migrated_database: Database,
    *,
    position: str,
) -> None:
    seeded = await seed_upload_context(migrated_database)
    store = StageCrashStore()
    uploaded = await database_upload(
        migrated_database,
        seeded,
        store,
        content=(("chunk crash boundary " * 100) + "\n").encode(),
        idempotency_key=f"chunk-stage-{position}",
    )
    lease = await claim_upload_job(migrated_database, uploaded.job_id)
    context = JobExecutionContext(
        job_repository_context(migrated_database),
        lease,
        timedelta(seconds=30),
    )
    pipeline = ingestion_pipeline(migrated_database, store)
    await pipeline.handle(context)
    final_key = chunks_object_key(
        seeded.knowledge_base_id,
        uploaded.document_id,
        uploaded.version_id,
    )
    store.arm("/chunks/recursive_text_v1.jsonl", position)

    with pytest.raises(SimulatedStageCrash):
        await pipeline.handle(context)

    async with migrated_database.sessions() as session:
        version = await session.get(DocumentVersion, uploaded.version_id)
        job = await session.get(Job, uploaded.job_id)
    assert version is not None and job is not None
    assert version.status == "chunking"
    assert version.chunk_manifest_object_key is None
    assert job.stage == "chunk"
    first_bytes = store.objects.get(final_key)
    assert (first_bytes is not None) is (position == "after_copy")
    stale_temp = temporary_object_key(
        uploaded.job_id,
        lease.lease_epoch,
        "chunks/recursive_text_v1.jsonl",
    )
    assert stale_temp not in store.objects
    assert stale_temp in store.delete_calls

    retry_lease = await expire_and_reclaim_upload_job(
        migrated_database,
        uploaded.job_id,
        lease_owner="pipeline-worker-b",
    )
    retry_context = JobExecutionContext(
        job_repository_context(migrated_database),
        retry_lease,
        timedelta(seconds=30),
    )
    await pipeline.handle(retry_context)
    assert retry_context.lease.stage == "embed_index"
    assert retry_context.lease.lease_epoch == lease.lease_epoch + 1
    assert retry_context.lease.attempt_count == 2
    assert store.publish_reused[-1] is (position == "after_copy")
    if first_bytes is not None:
        assert store.objects[final_key] == first_bytes


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("position", ["before_commit", "after_commit"])
async def test_chunk_stage_database_commit_crash_boundaries_preserve_atomic_facts(
    migrated_database: Database,
    *,
    position: str,
) -> None:
    seeded = await seed_upload_context(migrated_database)
    store = MemoryObjectStore()
    uploaded = await database_upload(
        migrated_database,
        seeded,
        store,
        content=(("chunk commit recovery " * 100) + "\n").encode(),
        idempotency_key=f"chunk-stage-{position}",
    )
    lease = await claim_upload_job(migrated_database, uploaded.job_id)
    parse_context = JobExecutionContext(
        job_repository_context(migrated_database),
        lease,
        timedelta(seconds=30),
    )
    pipeline = ingestion_pipeline(migrated_database, store)
    await pipeline.handle(parse_context)
    chunk_lease = parse_context.lease

    @asynccontextmanager
    async def crashing_job_repository() -> AsyncIterator[JobRepository]:
        async with migrated_database.sessions() as session:
            if position == "before_commit":
                async with session.begin():
                    yield BeforeCommitRepository(session)
            else:
                async with session.begin():
                    yield SqlAlchemyJobRepository(session)
                raise SimulatedStageCrash("simulated crash after fencing commit")

    context = JobExecutionContext(
        crashing_job_repository,
        chunk_lease,
        timedelta(seconds=30),
    )
    final_key = chunks_object_key(
        seeded.knowledge_base_id,
        uploaded.document_id,
        uploaded.version_id,
    )

    with pytest.raises(SimulatedStageCrash):
        await pipeline.handle(context)

    async with migrated_database.sessions() as session:
        version = await session.get(DocumentVersion, uploaded.version_id)
        state = await session.get(
            DocumentIndexState,
            (uploaded.version_id, seeded.generation_id),
        )
        job = await session.get(Job, uploaded.job_id)
    assert version is not None and state is not None and job is not None
    manifest_bytes = store.objects[final_key]
    if position == "before_commit":
        assert version.status == "chunking"
        assert version.chunk_manifest_object_key is None
        assert state.status == "queued"
        assert state.expected_point_count is None
        assert job.stage == "chunk"
        retry_lease = await expire_and_reclaim_upload_job(
            migrated_database,
            uploaded.job_id,
            lease_owner="pipeline-worker-b",
        )
        retry_context = JobExecutionContext(
            job_repository_context(migrated_database),
            retry_lease,
            timedelta(seconds=30),
        )
        await pipeline.handle(retry_context)
        assert retry_context.lease.stage == "embed_index"
        assert store.publish_reused[-1] is True
        assert store.objects[final_key] == manifest_bytes
    else:
        assert version.status == "embedding"
        assert version.chunk_manifest_object_key == final_key
        assert state.status == "embedding"
        assert state.expected_point_count == version.chunk_count
        assert job.stage == "embed_index"


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["parsed_checksum", "chunker_config", "final_conflict"])
async def test_chunk_stage_rejects_incompatible_inputs_and_artifact_conflicts(
    migrated_database: Database,
    *,
    failure: str,
) -> None:
    pipeline_module = importlib.import_module("rag_service.ingestion.pipeline")
    seeded = await seed_upload_context(migrated_database)
    store = MemoryObjectStore()
    uploaded = await database_upload(
        migrated_database,
        seeded,
        store,
        content=b"chunk conflict protection",
        idempotency_key=f"chunk-stage-{failure}",
    )
    lease = await claim_upload_job(migrated_database, uploaded.job_id)
    context = JobExecutionContext(
        job_repository_context(migrated_database),
        lease,
        timedelta(seconds=30),
    )
    pipeline = ingestion_pipeline(migrated_database, store)
    await pipeline.handle(context)
    if failure == "parsed_checksum":
        async with migrated_database.sessions() as session:
            version = await session.get(DocumentVersion, uploaded.version_id)
        assert version is not None and version.parsed_object_key is not None
        store.objects[version.parsed_object_key] = b"tampered parsed bytes"
    elif failure == "chunker_config":
        async with migrated_database.sessions() as session, session.begin():
            version = await session.get(DocumentVersion, uploaded.version_id)
            assert version is not None
            version.chunker_name = "recursive_text_v1"
            version.chunker_version = None
    else:
        final_key = chunks_object_key(
            seeded.knowledge_base_id,
            uploaded.document_id,
            uploaded.version_id,
        )
        store.objects[final_key] = b"conflicting chunk manifest"

    expected_code = (
        "OBJECT_VERIFICATION_FAILED"
        if failure == "parsed_checksum"
        else "ARTIFACT_CHECKSUM_CONFLICT"
        if failure == "final_conflict"
        else "INGESTION_STAGE_CONFLICT"
    )
    with pytest.raises(pipeline_module.PermanentJobError) as exc_info:
        await pipeline.handle(context)
    assert exc_info.value.code == expected_code
    async with migrated_database.sessions() as session:
        version = await session.get(DocumentVersion, uploaded.version_id)
        job = await session.get(Job, uploaded.job_id)
    assert version is not None and job is not None
    assert version.status == "chunking"
    assert version.chunk_manifest_object_key is None
    assert job.stage == "chunk"


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["parse", "chunk"])
async def test_parse_and_chunk_stage_stale_owner_epoch_after_publish_cannot_write_database(
    migrated_database: Database,
    *,
    stage: str,
) -> None:
    seeded = await seed_upload_context(migrated_database)
    store = ReclaimAfterPublishStore(migrated_database)
    uploaded = await database_upload(
        migrated_database,
        seeded,
        store,
        content=(("stale lease artifact " * 100) + "\n").encode(),
        idempotency_key=f"{stage}-stage-stale-after-publish",
    )
    lease = await claim_upload_job(migrated_database, uploaded.job_id)
    context = JobExecutionContext(
        job_repository_context(migrated_database),
        lease,
        timedelta(seconds=30),
    )
    pipeline = ingestion_pipeline(migrated_database, store)
    if stage == "chunk":
        await pipeline.handle(context)
        suffix = "/chunks/recursive_text_v1.jsonl"
        final_key = chunks_object_key(
            seeded.knowledge_base_id,
            uploaded.document_id,
            uploaded.version_id,
        )
        artifact_name = "chunks/recursive_text_v1.jsonl"
    else:
        suffix = "/parsed/text.txt"
        final_key = parsed_text_object_key(
            seeded.knowledge_base_id,
            uploaded.document_id,
            uploaded.version_id,
        )
        artifact_name = "parsed/text.txt"
    stale_epoch = context.lease.lease_epoch
    store.arm(suffix, uploaded.job_id)

    with pytest.raises(LostLeaseError):
        await pipeline.handle(context)

    async with migrated_database.sessions() as session:
        version = await session.get(DocumentVersion, uploaded.version_id)
        job = await session.get(Job, uploaded.job_id)
    assert version is not None and job is not None and store.reclaimed_lease is not None
    assert final_key in store.objects
    assert job.status == "running"
    assert job.lease_owner == "pipeline-worker-b"
    assert job.lease_epoch == stale_epoch + 1
    assert job.attempt_count == 2
    assert job.stage == stage
    if stage == "parse":
        assert version.status == "parsing"
        assert version.parsed_object_key is None
    else:
        assert version.status == "chunking"
        assert version.chunk_manifest_object_key is None
    stale_temp = temporary_object_key(uploaded.job_id, stale_epoch, artifact_name)
    assert stale_temp not in store.objects
    assert stale_temp in store.delete_calls


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_postgres_upload_reserves_atomic_graph_and_replays_idempotently(
    migrated_database: Database,
) -> None:
    seeded = await seed_upload_context(migrated_database)
    store = MemoryObjectStore()

    first = await database_upload(
        migrated_database,
        seeded,
        store,
        content=b"atomic upload",
        idempotency_key="atomic-key",
    )
    replay = await database_upload(
        migrated_database,
        seeded,
        store,
        content=b"atomic upload",
        idempotency_key="atomic-key",
    )

    assert replay == first
    async with migrated_database.sessions() as session:
        counts = {}
        for model in (
            Document,
            DocumentVersion,
            DocumentIndexState,
            Job,
            KnowledgeBaseMutation,
            DocumentUploadIdempotency,
        ):
            counts[model.__name__] = await session.scalar(select(func.count()).select_from(model))
        knowledge_base = await session.get(KnowledgeBase, seeded.knowledge_base_id)
    assert counts == {
        "Document": 1,
        "DocumentVersion": 1,
        "DocumentIndexState": 1,
        "Job": 1,
        "KnowledgeBaseMutation": 1,
        "DocumentUploadIdempotency": 1,
    }
    assert knowledge_base is not None
    assert knowledge_base.mutation_revision == 1
    assert len(store.objects) == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_source_cleanup_waits_for_reservation_fence_and_preserves_committed_object(
    migrated_database: Database,
) -> None:
    seeded = await seed_upload_context(migrated_database)

    class BlockingVerifyStore(MemoryObjectStore):
        def __init__(self) -> None:
            super().__init__()
            self.verify_started = asyncio.Event()
            self.release_verify = asyncio.Event()
            self.source_key: str | None = None

        async def verify_object(
            self,
            object_key: str,
            *,
            expected_size: int,
            expected_checksum: str,
        ) -> object:
            self.source_key = object_key
            self.verify_started.set()
            await self.release_verify.wait()
            return await super().verify_object(
                object_key,
                expected_size=expected_size,
                expected_checksum=expected_checksum,
            )

    store = BlockingVerifyStore()
    upload_task = asyncio.create_task(
        database_upload(
            migrated_database,
            seeded,
            store,
            content=b"source cleanup reservation fence",
            idempotency_key=None,
        )
    )
    await asyncio.wait_for(store.verify_started.wait(), timeout=1)
    assert store.source_key is not None

    async def authorize_cleanup() -> bool:
        async with migrated_database.sessions() as session, session.begin():
            return await SqlAlchemyIngestionPipelineRepository(
                session
            ).object_key_cleanup_is_allowed(store.source_key or "")

    cleanup_task = asyncio.create_task(authorize_cleanup())
    done, _pending = await asyncio.wait({cleanup_task}, timeout=0.05)
    assert cleanup_task not in done
    store.release_verify.set()
    uploaded = await asyncio.wait_for(upload_task, timeout=1)
    assert await asyncio.wait_for(cleanup_task, timeout=1) is False
    assert store.source_key in store.objects

    async with migrated_database.sessions() as session:
        version = await session.get(DocumentVersion, uploaded.version_id)
    assert version is not None
    assert version.source_object_key == store.source_key


@pytest.mark.integration
@pytest.mark.asyncio
async def test_source_reservation_rejects_object_deleted_under_cleanup_fence(
    migrated_database: Database,
) -> None:
    seeded = await seed_upload_context(migrated_database)

    class PausedPublishStore(MemoryObjectStore):
        def __init__(self) -> None:
            super().__init__()
            self.publish_visible = asyncio.Event()
            self.release_publish = asyncio.Event()
            self.source_key: str | None = None

        async def publish_temp(
            self,
            temp_key: str,
            final_key: str,
            *,
            expected_size: int,
            expected_checksum: str,
        ) -> object:
            published = await super().publish_temp(
                temp_key,
                final_key,
                expected_size=expected_size,
                expected_checksum=expected_checksum,
            )
            self.source_key = final_key
            self.publish_visible.set()
            await self.release_publish.wait()
            return published

    store = PausedPublishStore()
    upload_task = asyncio.create_task(
        database_upload(
            migrated_database,
            seeded,
            store,
            content=b"source cleanup wins fence",
            idempotency_key=None,
        )
    )
    await asyncio.wait_for(store.publish_visible.wait(), timeout=1)
    assert store.source_key is not None
    async with migrated_database.sessions() as session, session.begin():
        repository = SqlAlchemyIngestionPipelineRepository(session)
        assert await repository.object_key_cleanup_is_allowed(store.source_key) is True
        assert await store.delete_best_effort(store.source_key) is True
    store.release_publish.set()

    with pytest.raises(BusinessError) as raised:
        await asyncio.wait_for(upload_task, timeout=1)
    assert raised.value.code == "INGESTION_RESERVATION_UNCERTAIN"
    async with migrated_database.sessions() as session:
        version_count = await session.scalar(select(func.count()).select_from(DocumentVersion))
    assert version_count == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_active_generation_repair_blocks_new_upload_reservation(
    migrated_database: Database,
) -> None:
    seeded = await seed_upload_context(migrated_database)
    async with migrated_database.sessions() as session, session.begin():
        session.add(
            Job(
                id=uuid4(),
                knowledge_base_id=seeded.knowledge_base_id,
                actor_api_key_id=None,
                target_type="index_generation",
                target_id=seeded.generation_id,
                target_revision=0,
                index_generation_id=seeded.generation_id,
                mutation_id=None,
                parent_job_id=None,
                root_job_id=None,
                idempotency_key=None,
                operation="rebuild_generation",
                stage="indexing",
                status="queued",
                progress_total=0,
                resume_stage="indexing",
            )
        )

    with pytest.raises(BusinessError) as raised:
        await database_upload(
            migrated_database,
            seeded,
            MemoryObjectStore(),
            content=b"blocked during generation repair",
            idempotency_key=None,
        )

    assert raised.value.code == "GENERATION_REPAIR_IN_PROGRESS"
    assert raised.value.retryable is True
    async with migrated_database.sessions() as session:
        document_count = await session.scalar(select(func.count()).select_from(Document))
        ingest_count = await session.scalar(
            select(func.count()).select_from(Job).where(Job.operation == "ingest_document")
        )
    assert document_count == 0
    assert ingest_count == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_active_generation_repair_preserves_existing_upload_idempotency_replay(
    migrated_database: Database,
) -> None:
    seeded = await seed_upload_context(migrated_database)
    store = MemoryObjectStore()
    first = await database_upload(
        migrated_database,
        seeded,
        store,
        content=b"stable replay during generation repair",
        idempotency_key="repair-replay-key",
    )
    async with migrated_database.sessions() as session, session.begin():
        ingest_job = await session.get(Job, first.job_id)
        assert ingest_job is not None
        ingest_job.status = "succeeded"
        ingest_job.retryable = False
        ingest_job.finished_at = datetime.now(UTC)
        session.add(
            Job(
                id=uuid4(),
                knowledge_base_id=seeded.knowledge_base_id,
                actor_api_key_id=None,
                target_type="index_generation",
                target_id=seeded.generation_id,
                target_revision=1,
                index_generation_id=seeded.generation_id,
                mutation_id=None,
                parent_job_id=None,
                root_job_id=None,
                idempotency_key=None,
                operation="rebuild_generation",
                stage="indexing",
                status="queued",
                progress_total=0,
                resume_stage="indexing",
            )
        )

    replay = await database_upload(
        migrated_database,
        seeded,
        store,
        content=b"stable replay during generation repair",
        idempotency_key="repair-replay-key",
    )

    assert replay == first
    async with migrated_database.sessions() as session:
        document_count = await session.scalar(select(func.count()).select_from(Document))
        ingest_count = await session.scalar(
            select(func.count()).select_from(Job).where(Job.operation == "ingest_document")
        )
    assert document_count == 1
    assert ingest_count == 1
    assert len(store.objects) == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_postgres_replays_before_current_generation_schema_validation(
    migrated_database: Database,
) -> None:
    seeded = await seed_upload_context(migrated_database)
    store = MemoryObjectStore()
    old_metadata = '{"published_at":"not-rfc3339"}'
    first = await database_upload(
        migrated_database,
        seeded,
        store,
        content=b"stable old request",
        metadata=old_metadata,
        idempotency_key="schema-stable-key",
    )

    now = datetime.now(UTC)
    new_generation_id = uuid4()
    filter_schema = {
        "fields": [
            {
                "name": "published_at",
                "source_path": "published_at",
                "type": "datetime",
            }
        ]
    }
    async with migrated_database.sessions() as session, session.begin():
        knowledge_base = await session.get(KnowledgeBase, seeded.knowledge_base_id)
        old_generation = await session.get(
            KnowledgeBaseIndexGeneration,
            seeded.generation_id,
        )
        assert knowledge_base is not None and old_generation is not None
        new_generation = KnowledgeBaseIndexGeneration(
            id=new_generation_id,
            knowledge_base_id=seeded.knowledge_base_id,
            embedding_profile_id=None,
            sparse_profile_id=None,
            index_profile_hash="1" * 64,
            qdrant_collection_name=f"upload_{new_generation_id.hex}",
            status="active",
            rebuild_snapshot_at=now,
            caught_up_revision=knowledge_base.mutation_revision,
            validated_revision=knowledge_base.mutation_revision,
            validation_manifest_hash="2" * 64,
            expected_point_count=1,
            actual_point_count=1,
            validated_at=now,
            activated_at=now,
            retired_at=None,
            distance="cosine",
            embedding_config_snapshot={"provider": "changed"},
            filter_schema_snapshot=filter_schema,
            applied_filter_schema_revision=1,
            embedding_config_hash="3" * 64,
            safe_error_code=None,
            safe_error_message=None,
        )
        old_generation.status = "retiring"
        old_generation.retired_at = now
        await session.flush()
        session.add(new_generation)
        await session.flush()
        knowledge_base.filter_schema = filter_schema
        knowledge_base.filter_schema_revision = 1
        knowledge_base.active_index_generation_id = new_generation_id

    replay = await database_upload(
        migrated_database,
        replace(seeded, generation_id=new_generation_id),
        store,
        content=b"stable old request",
        metadata=old_metadata,
        idempotency_key="schema-stable-key",
    )
    assert replay == first

    with pytest.raises(BusinessError) as reused:
        await database_upload(
            migrated_database,
            replace(seeded, generation_id=new_generation_id),
            store,
            content=b"changed request",
            metadata=old_metadata,
            idempotency_key="schema-stable-key",
        )
    assert reused.value.code == "IDEMPOTENCY_KEY_REUSED"

    with pytest.raises(BusinessError) as invalid_new:
        await database_upload(
            migrated_database,
            replace(seeded, generation_id=new_generation_id),
            store,
            content=b"new invalid request",
            metadata=old_metadata,
            idempotency_key="new-schema-key",
        )
    assert invalid_new.value.code == "VALIDATION_ERROR"

    async with migrated_database.sessions() as session, session.begin():
        knowledge_base = await session.get(KnowledgeBase, seeded.knowledge_base_id)
        generation = await session.get(KnowledgeBaseIndexGeneration, new_generation_id)
        assert knowledge_base is not None and generation is not None
        generation.status = "retiring"
        generation.retired_at = datetime.now(UTC)
        knowledge_base.active_index_generation_id = None

    replay_without_generation = await database_upload(
        migrated_database,
        replace(seeded, generation_id=new_generation_id),
        store,
        content=b"stable old request",
        metadata=old_metadata,
        idempotency_key="schema-stable-key",
    )
    assert replay_without_generation == first

    with pytest.raises(BusinessError) as reused_without_generation:
        await database_upload(
            migrated_database,
            replace(seeded, generation_id=new_generation_id),
            store,
            content=b"changed without generation",
            metadata=old_metadata,
            idempotency_key="schema-stable-key",
        )
    assert reused_without_generation.value.code == "IDEMPOTENCY_KEY_REUSED"

    with pytest.raises(BusinessError) as unavailable_for_new:
        await database_upload(
            migrated_database,
            replace(seeded, generation_id=new_generation_id),
            store,
            content=b"new without generation",
            metadata="{}",
            idempotency_key="new-no-generation-key",
        )
    assert unavailable_for_new.value.code == "KNOWLEDGE_BASE_NOT_INDEX_CONFIGURED"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_postgres_scopes_same_idempotency_key_per_knowledge_base(
    migrated_database: Database,
) -> None:
    first_context = await seed_upload_context(migrated_database)
    second_context = await seed_second_knowledge_base(migrated_database, first_context)
    first_context = replace(first_context, actor=second_context.actor)
    store = MemoryObjectStore()

    first = await database_upload(
        migrated_database,
        first_context,
        store,
        content=b"same cross knowledge base content",
        idempotency_key="shared-key",
    )
    second = await database_upload(
        migrated_database,
        second_context,
        store,
        content=b"same cross knowledge base content",
        idempotency_key="shared-key",
    )

    assert first.document_id != second.document_id
    async with migrated_database.sessions() as session:
        assert await session.scalar(select(func.count()).select_from(Document)) == 2
        assert (
            await session.scalar(select(func.count()).select_from(DocumentUploadIdempotency)) == 2
        )


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.acceptance
async def test_real_postgres_prioritizes_idempotency_and_rejects_checksum_duplicates(
    migrated_database: Database,
) -> None:
    seeded = await seed_upload_context(migrated_database)
    store = MemoryObjectStore()
    await database_upload(
        migrated_database,
        seeded,
        store,
        content=b"first content",
        idempotency_key="first-key",
    )
    await database_upload(
        migrated_database,
        seeded,
        store,
        content=b"second content",
        idempotency_key="second-key",
    )

    with pytest.raises(BusinessError) as reused:
        await database_upload(
            migrated_database,
            seeded,
            store,
            content=b"second content",
            idempotency_key="first-key",
        )
    assert reused.value.code == "IDEMPOTENCY_KEY_REUSED"

    for key in (None, "third-key"):
        with pytest.raises(BusinessError) as duplicate:
            await database_upload(
                migrated_database,
                seeded,
                store,
                content=b"first content",
                idempotency_key=key,
            )
        assert duplicate.value.code == "DUPLICATE_DOCUMENT"

    async with migrated_database.sessions() as session:
        assert await session.scalar(select(func.count()).select_from(Document)) == 2
        assert (
            await session.scalar(select(func.count()).select_from(DocumentUploadIdempotency)) == 2
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_postgres_concurrent_same_key_commits_one_graph(
    migrated_database: Database,
) -> None:
    seeded = await seed_upload_context(migrated_database)
    store = MemoryObjectStore()
    first, second = await __import__("asyncio").gather(
        database_upload(
            migrated_database,
            seeded,
            store,
            content=b"concurrent content",
            idempotency_key="concurrent-key",
        ),
        database_upload(
            migrated_database,
            seeded,
            store,
            content=b"concurrent content",
            idempotency_key="concurrent-key",
        ),
    )

    assert first == second
    async with migrated_database.sessions() as session:
        assert await session.scalar(select(func.count()).select_from(Document)) == 1
        assert await session.scalar(select(func.count()).select_from(Job)) == 1
        assert await session.scalar(select(func.count()).select_from(KnowledgeBaseMutation)) == 1
        assert (
            await session.scalar(select(func.count()).select_from(DocumentUploadIdempotency)) == 1
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_postgres_rechecks_scope_before_capability_at_reservation(
    migrated_database: Database,
) -> None:
    seeded = await seed_upload_context(migrated_database)

    class RevokeOnPublishStore(MemoryObjectStore):
        async def publish_temp(
            self,
            temp_key: str,
            final_key: str,
            *,
            expected_size: int,
            expected_checksum: str,
        ) -> object:
            published = await super().publish_temp(
                temp_key,
                final_key,
                expected_size=expected_size,
                expected_checksum=expected_checksum,
            )
            async with migrated_database.sessions() as session, session.begin():
                scope = await session.get(
                    ApiKeyKnowledgeBaseScope,
                    (seeded.actor.key_id, seeded.knowledge_base_id),
                )
                key = await session.get(ApiKey, seeded.actor.key_id)
                assert scope is not None and key is not None
                await session.delete(scope)
                key.capabilities = []
            return published

    store = RevokeOnPublishStore()
    with pytest.raises(BusinessError) as hidden:
        await database_upload(
            migrated_database,
            seeded,
            store,
            content=b"scope revoked",
            idempotency_key=None,
        )
    assert hidden.value.code == "RESOURCE_NOT_FOUND"
    assert store.objects == {}
    async with migrated_database.sessions() as session:
        assert await session.scalar(select(func.count()).select_from(Document)) == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_postgres_rechecks_generation_and_rolls_back_reservation(
    migrated_database: Database,
) -> None:
    seeded = await seed_upload_context(migrated_database)

    class ChangeGenerationOnPublishStore(MemoryObjectStore):
        async def publish_temp(
            self,
            temp_key: str,
            final_key: str,
            *,
            expected_size: int,
            expected_checksum: str,
        ) -> object:
            published = await super().publish_temp(
                temp_key,
                final_key,
                expected_size=expected_size,
                expected_checksum=expected_checksum,
            )
            async with migrated_database.sessions() as session, session.begin():
                knowledge_base = await session.get(
                    KnowledgeBase,
                    seeded.knowledge_base_id,
                )
                assert knowledge_base is not None
                knowledge_base.filter_schema_revision += 1
            return published

    store = ChangeGenerationOnPublishStore()
    with pytest.raises(BusinessError) as conflict:
        await database_upload(
            migrated_database,
            seeded,
            store,
            content=b"generation changed",
            idempotency_key="generation-key",
        )
    assert conflict.value.code == "GENERATION_CONFIGURATION_CONFLICT"
    assert store.objects == {}
    async with migrated_database.sessions() as session:
        for model in (Document, Job, KnowledgeBaseMutation, DocumentUploadIdempotency):
            assert await session.scalar(select(func.count()).select_from(model)) == 0


def _embedding_filter_snapshot(field_type: str = "keyword") -> dict[str, object]:
    field_id = "1234567890abcdef1234567890abcdef"
    return {
        "fields": [
            {
                "name": "Tenant",
                "source_path": "tenant.id",
                "type": field_type,
                "operators": ["eq", "in"],
                "field_id": field_id,
                "payload_path": f"metadata.f_{field_id}",
            }
        ]
    }


async def prepare_embed_index_stage(
    database: Database,
    *,
    idempotency_key: str | None,
    batch_size: int = 2,
    filter_snapshot: dict[str, object] | None = None,
    metadata: str = "{}",
) -> tuple[
    SeededUploadContext,
    CollectionSpec,
    FakeQdrantClient,
    MemoryObjectStore,
    UploadAccepted,
    JobExecutionContext,
]:
    seeded = await seed_upload_context(database)
    _profile_id, _provider_id, spec = await configure_embedding_generation(
        database,
        seeded,
        batch_size=batch_size,
        filter_snapshot=filter_snapshot,
    )
    qdrant = FakeQdrantClient()
    await qdrant.seed_collection(spec, created_at=datetime.now(UTC))
    store = MemoryObjectStore()
    uploaded = await database_upload(
        database,
        seeded,
        store,
        content=(f"{idempotency_key or 'no-key'} content " * 220).encode(),
        idempotency_key=idempotency_key,
        metadata=metadata,
    )
    lease = await claim_upload_job(database, uploaded.job_id)
    context = JobExecutionContext(
        job_repository_context(database),
        lease,
        timedelta(seconds=30),
    )
    bootstrap = ingestion_pipeline(database, store)
    try:
        await bootstrap.handle(context)
        await bootstrap.handle(context)
    finally:
        await bootstrap.aclose()
    assert context.lease.stage == "embed_index"
    return seeded, spec, qdrant, store, uploaded, context


@pytest.mark.integration
@pytest.mark.asyncio
async def test_embed_index_no_idempotency_key_persists_actor_on_every_usage(
    migrated_database: Database,
) -> None:
    seeded, spec, qdrant, store, uploaded, context = await prepare_embed_index_stage(
        migrated_database,
        idempotency_key=None,
        batch_size=2,
    )
    pipeline = ingestion_pipeline(
        migrated_database,
        store,
        embedding_gateway=DeterministicEmbeddingGateway(),
        qdrant=qdrant,
        provider_usage_sink=SqlAlchemyProviderUsageSink(migrated_database.sessions),
    )
    try:
        await pipeline.handle(context)
    finally:
        await pipeline.aclose()

    async with migrated_database.sessions() as session:
        version = await session.get(DocumentVersion, uploaded.version_id)
        usages = (await session.scalars(select(ProviderUsage))).all()
        idempotency_count = await session.scalar(
            select(func.count()).select_from(DocumentUploadIdempotency)
        )
    assert version is not None
    assert context.lease.stage == "validate"
    assert len(usages) >= 1
    assert all(usage.actor_api_key_id == seeded.actor.key_id for usage in usages)
    assert idempotency_count == 0
    assert await qdrant.count_points(spec.name) == version.chunk_count


@pytest.mark.integration
@pytest.mark.asyncio
async def test_embed_index_legacy_job_falls_back_to_idempotency_actor(
    migrated_database: Database,
) -> None:
    seeded, spec, qdrant, store, uploaded, context = await prepare_embed_index_stage(
        migrated_database,
        idempotency_key="legacy-actor-fallback",
    )
    async with migrated_database.sessions() as session, session.begin():
        job = await session.get(Job, uploaded.job_id)
        assert job is not None
        job.actor_api_key_id = None

    pipeline = ingestion_pipeline(
        migrated_database,
        store,
        embedding_gateway=DeterministicEmbeddingGateway(),
        qdrant=qdrant,
        provider_usage_sink=SqlAlchemyProviderUsageSink(migrated_database.sessions),
    )
    try:
        await pipeline.handle(context)
    finally:
        await pipeline.aclose()

    async with migrated_database.sessions() as session:
        usages = (await session.scalars(select(ProviderUsage))).all()
    assert context.lease.stage == "validate"
    assert usages
    assert all(usage.actor_api_key_id == seeded.actor.key_id for usage in usages)
    assert await qdrant.count_points(spec.name) > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_embed_index_missing_actor_fails_closed_before_provider(
    migrated_database: Database,
) -> None:
    _seeded, spec, qdrant, store, uploaded, context = await prepare_embed_index_stage(
        migrated_database,
        idempotency_key=None,
    )
    async with migrated_database.sessions() as session, session.begin():
        job = await session.get(Job, uploaded.job_id)
        assert job is not None
        job.actor_api_key_id = None

    gateway = DeterministicEmbeddingGateway()
    pipeline = ingestion_pipeline(
        migrated_database,
        store,
        embedding_gateway=gateway,
        qdrant=qdrant,
        provider_usage_sink=SqlAlchemyProviderUsageSink(migrated_database.sessions),
    )
    try:
        with pytest.raises(PermanentJobError) as exc_info:
            await pipeline.handle(context)
    finally:
        await pipeline.aclose()

    assert exc_info.value.code == "INGESTION_STAGE_CONFLICT"
    assert gateway.calls == []
    assert await qdrant.count_points(spec.name) == 0
    async with migrated_database.sessions() as session:
        usage_count = await session.scalar(select(func.count()).select_from(ProviderUsage))
    assert usage_count == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_embed_index_batches_persist_usage_and_only_approved_payload(
    migrated_database: Database,
) -> None:
    seeded = await seed_upload_context(migrated_database)
    profile_id, provider_config_id, spec = await configure_embedding_generation(
        migrated_database,
        seeded,
        batch_size=1,
        filter_snapshot=_embedding_filter_snapshot(),
    )
    qdrant = FakeQdrantClient()
    await qdrant.seed_collection(spec, created_at=datetime.now(UTC))
    gateway = DeterministicEmbeddingGateway()
    store = TrackingMemoryObjectStore()
    uploaded = await database_upload(
        migrated_database,
        seeded,
        store,
        content=("alpha beta gamma delta " * 350).encode(),
        idempotency_key="embed-index-success",
        metadata='{"tenant":{"id":"acme"},"secret":"never-index"}',
    )
    lease = await claim_upload_job(migrated_database, uploaded.job_id)
    context = JobExecutionContext(
        job_repository_context(migrated_database),
        lease,
        timedelta(seconds=30),
    )
    pipeline = ingestion_pipeline(
        migrated_database,
        store,
        embedding_gateway=gateway,
        qdrant=qdrant,
        provider_usage_sink=SqlAlchemyProviderUsageSink(migrated_database.sessions),
    )

    await pipeline.handle(context)
    await pipeline.handle(context)
    await pipeline.handle(context)
    await pipeline.aclose()

    async with migrated_database.sessions() as session:
        version = await session.get(DocumentVersion, uploaded.version_id)
        state = await session.get(
            DocumentIndexState,
            (uploaded.version_id, seeded.generation_id),
        )
        usages = (
            await session.scalars(
                select(ProviderUsage).order_by(ProviderUsage.created_at, ProviderUsage.id)
            )
        ).all()
    assert version is not None and state is not None
    assert context.lease.stage == "validate"
    assert version.status == "indexing"
    assert state.status == "indexing"
    assert state.next_chunk_index == version.chunk_count
    assert state.embedding_config_hash is not None
    assert len(usages) == len(gateway.calls) >= 2
    assert store.opened_streams == store.closed_streams == 2
    assert all(row.actor_api_key_id == seeded.actor.key_id for row in usages)
    assert all(row.provider_config_id == provider_config_id for row in usages)
    assert all(row.model_profile_id == profile_id for row in usages)
    assert all(row.provider_identifier == "openai_compatible" for row in usages)
    assert all(row.model_identifier == "text-embedding-test" for row in usages)
    assert all(row.status == "succeeded" and row.provider_request_id for row in usages)
    points = await qdrant.stored_points(spec.name)
    assert len(points) == version.chunk_count
    assert {cast(int, point.payload["chunk_index"]) for point in points} == set(
        range(version.chunk_count or 0)
    )
    for point in points:
        payload = dict(point.payload)
        assert point.id == point_id(
            uploaded.version_id,
            cast(int, payload["chunk_index"]),
            cast(str, payload["chunk_hash"]),
        )
        assert payload["metadata"] == {"f_1234567890abcdef1234567890abcdef": "acme"}
        rendered = json.dumps(payload, ensure_ascii=False)
        assert "never-index" not in rendered
        assert "object_key" not in rendered
        assert "credential" not in rendered


@pytest.mark.integration
@pytest.mark.asyncio
async def test_embed_index_preflights_manifest_tail_before_any_side_effect(
    migrated_database: Database,
) -> None:
    pipeline_module = importlib.import_module("rag_service.ingestion.pipeline")
    seeded = await seed_upload_context(migrated_database)
    _profile_id, _provider_id, spec = await configure_embedding_generation(
        migrated_database,
        seeded,
    )
    qdrant = FakeQdrantClient()
    await qdrant.seed_collection(spec, created_at=datetime.now(UTC))
    gateway = DeterministicEmbeddingGateway()
    store = MemoryObjectStore()
    uploaded = await database_upload(
        migrated_database,
        seeded,
        store,
        content=("manifest tail validation " * 150).encode(),
        idempotency_key="embed-index-tail-invalid",
    )
    lease = await claim_upload_job(migrated_database, uploaded.job_id)
    context = JobExecutionContext(
        job_repository_context(migrated_database),
        lease,
        timedelta(seconds=30),
    )
    pipeline = ingestion_pipeline(
        migrated_database,
        store,
        embedding_gateway=gateway,
        qdrant=qdrant,
        provider_usage_sink=SqlAlchemyProviderUsageSink(migrated_database.sessions),
    )
    await pipeline.handle(context)
    await pipeline.handle(context)
    async with migrated_database.sessions() as session, session.begin():
        version = await session.get(DocumentVersion, uploaded.version_id)
        state = await session.get(
            DocumentIndexState,
            (uploaded.version_id, seeded.generation_id),
        )
        assert version is not None and state is not None
        assert version.chunk_manifest_object_key is not None
        damaged = store.objects[version.chunk_manifest_object_key] + b'{"unexpected":true}\n'
        damaged_checksum = hashlib.sha256(damaged).hexdigest()
        store.objects[version.chunk_manifest_object_key] = damaged
        version.chunk_manifest_checksum_sha256 = damaged_checksum
        state.chunk_manifest_checksum_sha256 = damaged_checksum

    with pytest.raises(pipeline_module.PermanentJobError) as exc_info:
        await pipeline.handle(context)
    await pipeline.aclose()
    assert exc_info.value.code == "INGESTION_STAGE_CONFLICT"
    assert gateway.calls == []
    assert await qdrant.count_points(spec.name) == 0
    async with migrated_database.sessions() as session:
        state = await session.get(
            DocumentIndexState,
            (uploaded.version_id, seeded.generation_id),
        )
        usage_count = await session.scalar(select(func.count()).select_from(ProviderUsage))
    assert state is not None and state.next_chunk_index == 0
    assert usage_count == 0


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("crash_point", "expected_partial_points"),
    [("embed_index.after_provider", 0), ("embed_index.after_upsert", 2)],
)
async def test_embed_index_crash_retries_duplicate_cost_but_not_point_ids(
    migrated_database: Database,
    *,
    crash_point: str,
    expected_partial_points: int,
) -> None:
    seeded = await seed_upload_context(migrated_database)
    _profile_id, _provider_id, spec = await configure_embedding_generation(
        migrated_database,
        seeded,
        batch_size=2,
    )
    qdrant = FakeQdrantClient()
    await qdrant.seed_collection(spec, created_at=datetime.now(UTC))
    gateway = DeterministicEmbeddingGateway()
    store = MemoryObjectStore()
    uploaded = await database_upload(
        migrated_database,
        seeded,
        store,
        content=("crash recovery batch " * 220).encode(),
        idempotency_key=f"embed-index-crash-{crash_point}",
    )
    lease = await claim_upload_job(migrated_database, uploaded.job_id)
    context = JobExecutionContext(
        job_repository_context(migrated_database),
        lease,
        timedelta(seconds=30),
    )
    armed = True

    async def crash_once(name: str) -> None:
        nonlocal armed
        if armed and name == crash_point:
            armed = False
            raise SimulatedStageCrash(name)

    usage_sink = SqlAlchemyProviderUsageSink(migrated_database.sessions)
    pipeline = ingestion_pipeline(
        migrated_database,
        store,
        embedding_gateway=gateway,
        qdrant=qdrant,
        provider_usage_sink=usage_sink,
        hooks=IngestionPipelineHooks(crash_once),
    )
    await pipeline.handle(context)
    await pipeline.handle(context)
    with pytest.raises(SimulatedStageCrash):
        await pipeline.handle(context)

    assert await qdrant.count_points(spec.name) == expected_partial_points
    async with migrated_database.sessions() as session:
        state = await session.get(
            DocumentIndexState,
            (uploaded.version_id, seeded.generation_id),
        )
        usage_before_retry = await session.scalar(select(func.count()).select_from(ProviderUsage))
    assert state is not None and state.next_chunk_index == 0
    assert usage_before_retry == 1

    retry_lease = await expire_and_reclaim_upload_job(
        migrated_database,
        uploaded.job_id,
        lease_owner="pipeline-worker-b",
    )
    retry_context = JobExecutionContext(
        job_repository_context(migrated_database),
        retry_lease,
        timedelta(seconds=30),
    )
    retry_pipeline = ingestion_pipeline(
        migrated_database,
        store,
        embedding_gateway=gateway,
        qdrant=qdrant,
        provider_usage_sink=usage_sink,
    )
    await retry_pipeline.handle(retry_context)
    await pipeline.aclose()
    await retry_pipeline.aclose()

    async with migrated_database.sessions() as session:
        version = await session.get(DocumentVersion, uploaded.version_id)
        usage_after_retry = await session.scalar(select(func.count()).select_from(ProviderUsage))
    assert version is not None
    assert retry_context.lease.stage == "validate"
    assert await qdrant.count_points(spec.name) == version.chunk_count
    assert usage_after_retry == len(gateway.calls)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_embed_index_resume_uses_exclusive_checkpoint_and_current_batch_size(
    migrated_database: Database,
) -> None:
    seeded = await seed_upload_context(migrated_database)
    profile_id, _provider_id, spec = await configure_embedding_generation(
        migrated_database,
        seeded,
        batch_size=2,
    )
    qdrant = FakeQdrantClient()
    await qdrant.seed_collection(spec, created_at=datetime.now(UTC))
    gateway = DeterministicEmbeddingGateway()
    store = MemoryObjectStore()
    uploaded = await database_upload(
        migrated_database,
        seeded,
        store,
        content=("exclusive checkpoint batch size " * 250).encode(),
        idempotency_key="embed-index-batch-size-change",
    )
    lease = await claim_upload_job(migrated_database, uploaded.job_id)
    context = JobExecutionContext(
        job_repository_context(migrated_database),
        lease,
        timedelta(seconds=30),
    )
    armed = True

    async def crash_after_checkpoint(name: str) -> None:
        nonlocal armed
        if armed and name == "embed_index.after_checkpoint":
            armed = False
            raise SimulatedStageCrash(name)

    usage_sink = SqlAlchemyProviderUsageSink(migrated_database.sessions)
    pipeline = ingestion_pipeline(
        migrated_database,
        store,
        embedding_gateway=gateway,
        qdrant=qdrant,
        provider_usage_sink=usage_sink,
        hooks=IngestionPipelineHooks(crash_after_checkpoint),
    )
    await pipeline.handle(context)
    await pipeline.handle(context)
    with pytest.raises(SimulatedStageCrash):
        await pipeline.handle(context)
    async with migrated_database.sessions() as session, session.begin():
        state = await session.get(
            DocumentIndexState,
            (uploaded.version_id, seeded.generation_id),
        )
        profile = await session.get(ModelProfile, profile_id)
        assert state is not None and profile is not None
        assert state.next_chunk_index == 2
        profile.batch_size = 3

    retry_lease = await expire_and_reclaim_upload_job(
        migrated_database,
        uploaded.job_id,
        lease_owner="pipeline-worker-b",
    )
    retry_context = JobExecutionContext(
        job_repository_context(migrated_database),
        retry_lease,
        timedelta(seconds=30),
    )
    retry_pipeline = ingestion_pipeline(
        migrated_database,
        store,
        embedding_gateway=gateway,
        qdrant=qdrant,
        provider_usage_sink=usage_sink,
    )
    await retry_pipeline.handle(retry_context)
    await pipeline.aclose()
    await retry_pipeline.aclose()

    assert retry_context.lease.stage == "validate"
    assert len(gateway.calls[0]) == 2
    assert len(gateway.calls[1]) == 3
    points = await qdrant.stored_points(spec.name)
    assert len({point.id for point in points}) == len(points)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sink_failure",
    [RuntimeError("usage database unavailable"), TimeoutError("usage write timed out")],
    ids=["failure", "timeout"],
)
async def test_embed_index_usage_persistence_is_fail_open(
    migrated_database: Database,
    sink_failure: BaseException,
) -> None:
    _seeded, spec, qdrant, store, uploaded, context = await prepare_embed_index_stage(
        migrated_database,
        idempotency_key=f"usage-fail-open-{type(sink_failure).__name__}",
    )
    gateway = DeterministicEmbeddingGateway()
    sink = FailingUsageSink(sink_failure)
    pipeline = ingestion_pipeline(
        migrated_database,
        store,
        embedding_gateway=gateway,
        qdrant=qdrant,
        provider_usage_sink=cast(PipelineProviderUsageSink, sink),
    )
    try:
        await pipeline.handle(context)
    finally:
        await pipeline.aclose()

    assert context.lease.stage == "validate"
    assert sink.calls == len(gateway.calls) >= 1
    async with migrated_database.sessions() as session:
        version = await session.get(DocumentVersion, uploaded.version_id)
        usage_count = await session.scalar(select(func.count()).select_from(ProviderUsage))
    assert version is not None
    assert await qdrant.count_points(spec.name) == version.chunk_count
    assert usage_count == 0


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "retryable", "status", "error_type"),
    [
        ("PROVIDER_RATE_LIMITED", True, "rate_limited", RetryableJobError),
        ("PROVIDER_TIMEOUT", True, "timeout", RetryableJobError),
        ("PROVIDER_RESPONSE_INVALID", False, "failed", PermanentJobError),
    ],
)
async def test_embed_index_provider_failures_record_usage_without_checkpoint(
    migrated_database: Database,
    code: str,
    retryable: bool,
    status: Literal["failed", "rate_limited", "timeout"],
    error_type: type[RetryableJobError] | type[PermanentJobError],
) -> None:
    seeded, spec, qdrant, store, uploaded, context = await prepare_embed_index_stage(
        migrated_database,
        idempotency_key=f"provider-failure-{status}",
    )
    gateway = FailingEmbeddingGateway(code=code, retryable=retryable, status=status)
    pipeline = ingestion_pipeline(
        migrated_database,
        store,
        embedding_gateway=gateway,
        qdrant=qdrant,
        provider_usage_sink=SqlAlchemyProviderUsageSink(migrated_database.sessions),
    )
    try:
        with pytest.raises(error_type) as exc_info:
            await pipeline.handle(context)
    finally:
        await pipeline.aclose()

    assert exc_info.value.code == code
    assert gateway.calls == 1
    assert await qdrant.count_points(spec.name) == 0
    async with migrated_database.sessions() as session:
        state = await session.get(
            DocumentIndexState,
            (uploaded.version_id, seeded.generation_id),
        )
        usages = (await session.scalars(select(ProviderUsage))).all()
    assert state is not None and state.next_chunk_index == 0
    assert len(usages) == 1
    assert usages[0].status == status
    assert usages[0].error_code == code


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_result", ["count", "dimension"])
async def test_embed_index_rejects_invalid_vector_results_without_checkpoint(
    migrated_database: Database,
    invalid_result: str,
) -> None:
    seeded, spec, qdrant, store, uploaded, context = await prepare_embed_index_stage(
        migrated_database,
        idempotency_key=f"invalid-vector-{invalid_result}",
    )
    pipeline = ingestion_pipeline(
        migrated_database,
        store,
        embedding_gateway=InvalidEmbeddingGateway(invalid_result),
        qdrant=qdrant,
        provider_usage_sink=SqlAlchemyProviderUsageSink(migrated_database.sessions),
    )
    try:
        with pytest.raises(PermanentJobError) as exc_info:
            await pipeline.handle(context)
    finally:
        await pipeline.aclose()

    assert exc_info.value.code == "INGESTION_STAGE_CONFLICT"
    assert await qdrant.count_points(spec.name) == 0
    async with migrated_database.sessions() as session:
        state = await session.get(
            DocumentIndexState,
            (uploaded.version_id, seeded.generation_id),
        )
        usage_count = await session.scalar(select(func.count()).select_from(ProviderUsage))
    assert state is not None and state.next_chunk_index == 0
    assert usage_count == 1


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tamper", "expected_code"),
    [
        ("manifest_checksum", "OBJECT_VERIFICATION_FAILED"),
        ("chunk_config_hash", "INGESTION_STAGE_CONFLICT"),
        ("embedding_config_hash", "INGESTION_STAGE_CONFLICT"),
        ("job_operation", "INGESTION_STAGE_CONFLICT"),
    ],
)
async def test_embed_index_rejects_tampered_immutable_inputs_before_side_effects(
    migrated_database: Database,
    tamper: str,
    expected_code: str,
) -> None:
    seeded, spec, qdrant, store, uploaded, context = await prepare_embed_index_stage(
        migrated_database,
        idempotency_key=f"immutable-tamper-{tamper}",
    )
    async with migrated_database.sessions() as session, session.begin():
        version = await session.get(DocumentVersion, uploaded.version_id)
        state = await session.get(
            DocumentIndexState,
            (uploaded.version_id, seeded.generation_id),
        )
        generation = await session.get(KnowledgeBaseIndexGeneration, seeded.generation_id)
        job = await session.get(Job, uploaded.job_id)
        assert (
            version is not None and state is not None and generation is not None and job is not None
        )
        if tamper == "manifest_checksum":
            version.chunk_manifest_checksum_sha256 = "f" * 64
            state.chunk_manifest_checksum_sha256 = "f" * 64
        elif tamper == "chunk_config_hash":
            version.chunk_config_hash = "f" * 64
        elif tamper == "embedding_config_hash":
            generation.embedding_config_hash = "f" * 64
        else:
            job.operation = "index_document"

    gateway = DeterministicEmbeddingGateway()
    pipeline = ingestion_pipeline(
        migrated_database,
        store,
        embedding_gateway=gateway,
        qdrant=qdrant,
        provider_usage_sink=SqlAlchemyProviderUsageSink(migrated_database.sessions),
    )
    try:
        with pytest.raises(PermanentJobError) as exc_info:
            await pipeline.handle(context)
    finally:
        await pipeline.aclose()

    assert exc_info.value.code == expected_code
    assert gateway.calls == []
    assert await qdrant.count_points(spec.name) == 0
    async with migrated_database.sessions() as session:
        usage_count = await session.scalar(select(func.count()).select_from(ProviderUsage))
    assert usage_count == 0


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("field_type", "invalid_value"),
    [
        ("keyword", "x" * 4097),
        ("float", 10**400),
        ("integer", 2**63),
    ],
)
async def test_embed_index_rejects_invalid_approved_metadata_before_provider_call(
    migrated_database: Database,
    field_type: str,
    invalid_value: object,
) -> None:
    initial_value: object = "valid"
    if field_type == "float":
        initial_value = 1.5
    elif field_type == "integer":
        initial_value = 1
    seeded, spec, qdrant, store, uploaded, context = await prepare_embed_index_stage(
        migrated_database,
        idempotency_key="invalid-approved-metadata",
        filter_snapshot=_embedding_filter_snapshot(field_type),
        metadata=json.dumps({"tenant": {"id": initial_value}}),
    )
    async with migrated_database.sessions() as session, session.begin():
        version = await session.get(DocumentVersion, uploaded.version_id)
        assert version is not None
        document = await session.get(Document, version.document_id)
        assert document is not None
        document.metadata_ = {"tenant": {"id": invalid_value}}

    gateway = DeterministicEmbeddingGateway()
    pipeline = ingestion_pipeline(
        migrated_database,
        store,
        embedding_gateway=gateway,
        qdrant=qdrant,
        provider_usage_sink=SqlAlchemyProviderUsageSink(migrated_database.sessions),
    )
    try:
        with pytest.raises(PermanentJobError) as exc_info:
            await pipeline.handle(context)
    finally:
        await pipeline.aclose()

    assert exc_info.value.code == "INGESTION_STAGE_CONFLICT"
    assert gateway.calls == []
    assert await qdrant.count_points(spec.name) == 0
    async with migrated_database.sessions() as session:
        state = await session.get(
            DocumentIndexState,
            (uploaded.version_id, seeded.generation_id),
        )
    assert state is not None and state.next_chunk_index == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_embed_index_stale_lease_after_upsert_cannot_checkpoint_progress(
    migrated_database: Database,
) -> None:
    seeded, spec, qdrant, store, uploaded, context = await prepare_embed_index_stage(
        migrated_database,
        idempotency_key="embed-stale-after-upsert",
        batch_size=2,
    )
    reclaimed_lease: JobLease | None = None

    async def reclaim_after_upsert(name: str) -> None:
        nonlocal reclaimed_lease
        if name == "embed_index.after_upsert" and reclaimed_lease is None:
            reclaimed_lease = await expire_and_reclaim_upload_job(
                migrated_database,
                uploaded.job_id,
                lease_owner="pipeline-worker-b",
            )

    pipeline = ingestion_pipeline(
        migrated_database,
        store,
        embedding_gateway=DeterministicEmbeddingGateway(),
        qdrant=qdrant,
        provider_usage_sink=SqlAlchemyProviderUsageSink(migrated_database.sessions),
        hooks=IngestionPipelineHooks(reclaim_after_upsert),
    )
    try:
        with pytest.raises(LostLeaseError):
            await pipeline.handle(context)
    finally:
        await pipeline.aclose()

    assert reclaimed_lease is not None
    assert await qdrant.count_points(spec.name) == 2
    async with migrated_database.sessions() as session:
        state = await session.get(
            DocumentIndexState,
            (uploaded.version_id, seeded.generation_id),
        )
        usage_count = await session.scalar(select(func.count()).select_from(ProviderUsage))
    assert state is not None and state.next_chunk_index == 0
    assert usage_count == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_validate_stage_exactly_checks_all_version_points_and_advances_atomically(
    migrated_database: Database,
) -> None:
    seeded, spec, qdrant, store, uploaded, context = await prepare_embed_index_stage(
        migrated_database,
        idempotency_key="validate-happy",
        batch_size=2,
    )
    pipeline = ingestion_pipeline(
        migrated_database,
        store,
        embedding_gateway=DeterministicEmbeddingGateway(),
        qdrant=qdrant,
        provider_usage_sink=SqlAlchemyProviderUsageSink(migrated_database.sessions),
    )
    try:
        await pipeline.handle(context)
        assert context.lease.stage == "validate"
        await pipeline.handle(context)
    finally:
        await pipeline.aclose()

    assert context.lease.stage == "activate"
    async with migrated_database.sessions() as session:
        version = await session.get(DocumentVersion, uploaded.version_id)
        state = await session.get(
            DocumentIndexState,
            (uploaded.version_id, seeded.generation_id),
        )
    assert version is not None and state is not None
    assert version.status == "indexing"
    assert state.status == "validated"
    assert state.expected_point_count == version.chunk_count
    assert state.actual_point_count == version.chunk_count
    assert state.next_chunk_index == version.chunk_count
    assert state.validated_at is not None
    assert await qdrant.count_version_points(spec.name, uploaded.version_id) == version.chunk_count


@pytest.mark.integration
@pytest.mark.asyncio
async def test_validate_stage_crash_after_count_commit_retries_idempotently(
    migrated_database: Database,
) -> None:
    seeded, spec, qdrant, store, uploaded, context = await prepare_embed_index_stage(
        migrated_database,
        idempotency_key="validate-count-crash-retry",
        batch_size=2,
    )
    armed = True

    async def crash_after_count_commit(name: str) -> None:
        nonlocal armed
        if armed and name == "validate.after_count_commit":
            armed = False
            raise SimulatedStageCrash("simulated validate crash after count commit")

    crashing = ingestion_pipeline(
        migrated_database,
        store,
        embedding_gateway=DeterministicEmbeddingGateway(),
        qdrant=qdrant,
        provider_usage_sink=SqlAlchemyProviderUsageSink(migrated_database.sessions),
        hooks=IngestionPipelineHooks(crash_after_count_commit),
    )
    try:
        await crashing.handle(context)
        assert context.lease.stage == "validate"
        with pytest.raises(SimulatedStageCrash):
            await crashing.handle(context)
    finally:
        await crashing.aclose()

    async with migrated_database.sessions() as session:
        version_after_crash = await session.get(DocumentVersion, uploaded.version_id)
        state_after_crash = await session.get(
            DocumentIndexState,
            (uploaded.version_id, seeded.generation_id),
        )
        job_after_crash = await session.get(Job, uploaded.job_id)
    assert (
        version_after_crash is not None
        and state_after_crash is not None
        and job_after_crash is not None
    )
    assert state_after_crash.actual_point_count == version_after_crash.chunk_count
    assert state_after_crash.status == "indexing"
    assert state_after_crash.validated_at is None
    assert job_after_crash.status == "running" and job_after_crash.stage == "validate"

    retry = ingestion_pipeline(
        migrated_database,
        store,
        embedding_gateway=DeterministicEmbeddingGateway(),
        qdrant=qdrant,
        provider_usage_sink=SqlAlchemyProviderUsageSink(migrated_database.sessions),
    )
    try:
        await retry.handle(context)
    finally:
        await retry.aclose()

    async with migrated_database.sessions() as session:
        state = await session.get(
            DocumentIndexState,
            (uploaded.version_id, seeded.generation_id),
        )
    assert context.lease.stage == "activate"
    assert state is not None
    assert state.status == "validated" and state.validated_at is not None
    assert state.actual_point_count == state.expected_point_count
    assert (
        await qdrant.count_version_points(spec.name, uploaded.version_id)
        == state.actual_point_count
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_validate_stage_stale_lease_cannot_persist_count_or_validation(
    migrated_database: Database,
) -> None:
    seeded, spec, qdrant, store, uploaded, context = await prepare_embed_index_stage(
        migrated_database,
        idempotency_key="validate-stale-lease",
        batch_size=2,
    )
    reclaimed: JobLease | None = None

    async def reclaim_after_count(name: str) -> None:
        nonlocal reclaimed
        if name == "validate.after_count" and reclaimed is None:
            reclaimed = await expire_and_reclaim_upload_job(
                migrated_database,
                uploaded.job_id,
                lease_owner="validate-worker-b",
            )

    pipeline = ingestion_pipeline(
        migrated_database,
        store,
        embedding_gateway=DeterministicEmbeddingGateway(),
        qdrant=qdrant,
        provider_usage_sink=SqlAlchemyProviderUsageSink(migrated_database.sessions),
        hooks=IngestionPipelineHooks(reclaim_after_count),
    )
    try:
        await pipeline.handle(context)
        assert context.lease.stage == "validate"
        with pytest.raises(LostLeaseError):
            await pipeline.handle(context)
    finally:
        await pipeline.aclose()

    assert reclaimed is not None
    async with migrated_database.sessions() as session:
        version = await session.get(DocumentVersion, uploaded.version_id)
        state = await session.get(
            DocumentIndexState,
            (uploaded.version_id, seeded.generation_id),
        )
        job = await session.get(Job, uploaded.job_id)
    assert version is not None and state is not None and job is not None
    assert version.status == "indexing"
    assert state.status == "indexing"
    assert state.actual_point_count is None
    assert state.validated_at is None
    assert state.next_chunk_index == version.chunk_count
    assert job.status == "running" and job.stage == "validate"
    assert job.lease_owner == reclaimed.lease_owner
    assert job.lease_epoch == reclaimed.lease_epoch
    assert await qdrant.count_version_points(spec.name, uploaded.version_id) == version.chunk_count


@pytest.mark.integration
@pytest.mark.asyncio
async def test_terminal_validation_failure_atomically_fails_database_facts_and_retains_points(
    migrated_database: Database,
) -> None:
    seeded, spec, qdrant, store, uploaded, original_context = await prepare_embed_index_stage(
        migrated_database,
        idempotency_key="validate-terminal-count",
        batch_size=2,
    )
    pipeline = ingestion_pipeline(
        migrated_database,
        store,
        embedding_gateway=DeterministicEmbeddingGateway(),
        qdrant=qdrant,
        provider_usage_sink=SqlAlchemyProviderUsageSink(migrated_database.sessions),
    )
    try:
        await pipeline.handle(original_context)
        assert original_context.lease.stage == "validate"
        async with migrated_database.sessions() as session:
            version_before = await session.get(DocumentVersion, uploaded.version_id)
            state_before = await session.get(
                DocumentIndexState,
                (uploaded.version_id, seeded.generation_id),
            )
        assert version_before is not None and state_before is not None
        assert version_before.chunk_count is not None
        extra_hash = "e" * 64
        extra = QdrantPoint(
            id=point_id(uploaded.version_id, version_before.chunk_count, extra_hash),
            vector=tuple(0.1 for _ in range(spec.dimension)),
            payload={
                "knowledge_base_id": str(seeded.knowledge_base_id),
                "document_id": str(version_before.document_id),
                "version_id": str(uploaded.version_id),
                "chunk_index": version_before.chunk_count,
                "chunk_hash": extra_hash,
                "text": "x",
                "title_path": [],
                "start_offset": 0,
                "end_offset": 1,
                "metadata": {},
            },
        )
        await qdrant.upsert_points(spec.name, (extra,))
        context = JobExecutionContext(
            job_repository_context(migrated_database),
            original_context.lease,
            timedelta(seconds=30),
            domain_finalization_enabled=True,
        )
        outcome = await pipeline.handle(context)
    finally:
        await pipeline.aclose()

    assert outcome.value == "complete"
    assert context.finalized is True
    async with migrated_database.sessions() as session:
        document = await session.get(Document, version_before.document_id)
        version = await session.get(DocumentVersion, uploaded.version_id)
        state = await session.get(
            DocumentIndexState,
            (uploaded.version_id, seeded.generation_id),
        )
        job = await session.get(Job, uploaded.job_id)
    assert document is not None and version is not None and state is not None and job is not None
    assert document.status == "failed"
    assert document.current_version_id is None
    assert document.pending_version_id == uploaded.version_id
    assert version.status == "failed"
    assert version.chunk_manifest_object_key == version_before.chunk_manifest_object_key
    assert state.status == "failed"
    assert state.expected_point_count is not None
    assert state.actual_point_count == state.expected_point_count + 1
    assert state.next_chunk_index == state.expected_point_count
    assert state.error_code == "INDEX_VALIDATION_FAILED"
    assert job.status == "failed"
    assert job.retryable is False
    assert job.error_code == "INDEX_VALIDATION_FAILED"
    assert job.lease_owner is None and job.lease_expires_at is None
    assert job.finished_at is not None
    assert (
        await qdrant.count_version_points(spec.name, uploaded.version_id)
        == state.actual_point_count
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_expired_final_attempt_crash_atomically_terminalizes_ingestion_graph(
    migrated_database: Database,
) -> None:
    seeded = await seed_upload_context(migrated_database)
    store = MemoryObjectStore()
    uploaded = await database_upload(
        migrated_database,
        seeded,
        store,
        content=b"final attempt crash source",
        idempotency_key="expired-final-attempt-domain",
    )
    expired_at = datetime.now(UTC) - timedelta(seconds=1)
    async with migrated_database.sessions() as session, session.begin():
        job = await session.get(Job, uploaded.job_id)
        assert job is not None
        job.status = "running"
        job.attempt_count = job.max_attempts
        job.lease_owner = "crashed-final-worker"
        job.lease_epoch = 7
        job.lease_expires_at = expired_at
        job.worker_heartbeat_at = expired_at
        job.started_at = expired_at - timedelta(seconds=1)

    pipeline = ingestion_pipeline(migrated_database, store)
    try:
        runner = ingestion_job_runner(
            migrated_database,
            pipeline,
            lease_owner="exhaustion-finalizer-worker",
        )
        assert await runner.run_once(uploaded.job_id) is True
    finally:
        await pipeline.aclose()

    async with migrated_database.sessions() as session:
        document = await session.get(Document, uploaded.document_id)
        version = await session.get(DocumentVersion, uploaded.version_id)
        state = await session.get(
            DocumentIndexState,
            (uploaded.version_id, seeded.generation_id),
        )
        job = await session.get(Job, uploaded.job_id)
    assert document is not None and version is not None and state is not None and job is not None
    assert document.status == "failed"
    assert document.current_version_id is None
    assert document.pending_version_id == version.id
    assert version.status == "failed"
    assert state.status == "failed"
    assert state.error_code == "JOB_ATTEMPTS_EXHAUSTED"
    assert job.status == "failed"
    assert job.attempt_count == job.max_attempts
    assert job.error_code == "JOB_ATTEMPTS_EXHAUSTED"
    assert job.retryable is False
    assert job.lease_owner is None and job.lease_expires_at is None
    assert job.finished_at is not None
    assert version.source_object_key in store.objects


@pytest.mark.integration
@pytest.mark.asyncio
async def test_job_runner_terminalizes_ingestion_once_without_secondary_failure_or_success_write(
    migrated_database: Database,
) -> None:
    seeded, spec, qdrant, store, uploaded, context = await prepare_embed_index_stage(
        migrated_database,
        idempotency_key="runner-terminal-count",
        batch_size=2,
    )
    pipeline = ingestion_pipeline(
        migrated_database,
        store,
        embedding_gateway=DeterministicEmbeddingGateway(),
        qdrant=qdrant,
        provider_usage_sink=SqlAlchemyProviderUsageSink(migrated_database.sessions),
    )
    try:
        await pipeline.handle(context)
        async with migrated_database.sessions() as session:
            version = await session.get(DocumentVersion, uploaded.version_id)
        assert version is not None and version.chunk_count is not None
        extra_hash = "f" * 64
        await qdrant.upsert_points(
            spec.name,
            (
                QdrantPoint(
                    id=point_id(uploaded.version_id, version.chunk_count, extra_hash),
                    vector=tuple(0.1 for _ in range(spec.dimension)),
                    payload={
                        "knowledge_base_id": str(seeded.knowledge_base_id),
                        "document_id": str(version.document_id),
                        "version_id": str(uploaded.version_id),
                        "chunk_index": version.chunk_count,
                        "chunk_hash": extra_hash,
                        "text": "x",
                        "title_path": [],
                        "start_offset": 0,
                        "end_offset": 1,
                        "metadata": {},
                    },
                ),
            ),
        )
        async with migrated_database.sessions() as session, session.begin():
            await SqlAlchemyJobRepository(session).release_claim(context.lease)

        class SpyJobRepository(SqlAlchemyJobRepository):
            mark_succeeded_calls = 0
            record_failure_calls = 0

            async def mark_succeeded(self, lease: JobLease) -> str:
                type(self).mark_succeeded_calls += 1
                return await super().mark_succeeded(lease)

            async def record_failure(
                self,
                lease: JobLease,
                *,
                retryable: bool,
                error_code: str,
                error_message: str,
                retry_delay: timedelta,
            ) -> str:
                type(self).record_failure_calls += 1
                return await super().record_failure(
                    lease,
                    retryable=retryable,
                    error_code=error_code,
                    error_message=error_message,
                    retry_delay=retry_delay,
                )

        @asynccontextmanager
        async def repository_context() -> AsyncIterator[JobRepository]:
            async with migrated_database.sessions() as session, session.begin():
                yield SpyJobRepository(session)

        runner = JobRunner(
            repository_context=cast(RepositoryContextFactory, repository_context),
            lease_owner="runner-terminal-worker",
            lease_seconds=30,
            heartbeat_seconds=5,
            poll_interval_seconds=0.1,
            max_concurrency=1,
            backoff=ExponentialBackoff(initial_seconds=1, maximum_seconds=5),
        )
        runner.register(
            "ingest_document",
            pipeline.handle,
            exhaustion_finalizer=pipeline.finalize_exhausted,
        )

        assert await runner.run_once(uploaded.job_id) is True
    finally:
        await pipeline.aclose()

    async with migrated_database.sessions() as session:
        job = await session.get(Job, uploaded.job_id)
        version = await session.get(DocumentVersion, uploaded.version_id)
        state = await session.get(
            DocumentIndexState,
            (uploaded.version_id, seeded.generation_id),
        )
    assert job is not None and version is not None and state is not None
    assert job.status == "failed"
    assert job.error_code == "INDEX_VALIDATION_FAILED"
    assert version.status == "failed" and state.status == "failed"
    assert SpyJobRepository.mark_succeeded_calls == 0
    assert SpyJobRepository.record_failure_calls == 0


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("exhausted", [False, True], ids=["retry-wait", "terminal"])
async def test_transient_validate_failure_only_terminalizes_when_attempts_are_exhausted(
    migrated_database: Database,
    monkeypatch: pytest.MonkeyPatch,
    *,
    exhausted: bool,
) -> None:
    idempotency_key = f"validate-transient-{'terminal' if exhausted else 'retry'}"
    seeded, spec, qdrant, store, uploaded, context = await prepare_embed_index_stage(
        migrated_database,
        idempotency_key=idempotency_key,
        batch_size=2,
    )
    pipeline = ingestion_pipeline(
        migrated_database,
        store,
        embedding_gateway=DeterministicEmbeddingGateway(),
        qdrant=qdrant,
        provider_usage_sink=SqlAlchemyProviderUsageSink(migrated_database.sessions),
    )
    try:
        await pipeline.handle(context)
        assert context.lease.stage == "validate"
        points_before = await qdrant.stored_points(spec.name)
        async with migrated_database.sessions() as session:
            version_before = await session.get(DocumentVersion, uploaded.version_id)
            state_before = await session.get(
                DocumentIndexState,
                (uploaded.version_id, seeded.generation_id),
            )
        assert version_before is not None and state_before is not None
        artifact_keys = (
            version_before.source_object_key,
            version_before.parsed_object_key,
            version_before.chunk_manifest_object_key,
        )
        assert all(key is not None and key in store.objects for key in artifact_keys)
        artifacts_before = {cast(str, key): store.objects[cast(str, key)] for key in artifact_keys}
        checkpoint_before = state_before.next_chunk_index

        async with migrated_database.sessions() as session, session.begin():
            await SqlAlchemyJobRepository(session).release_claim(context.lease)
            if exhausted:
                job = await session.get(Job, uploaded.job_id)
                assert job is not None
                job.attempt_count = job.max_attempts - 1

        async def qdrant_unavailable(_collection: str, _version_id: UUID) -> int:
            raise QdrantTransientError("simulated qdrant outage")

        monkeypatch.setattr(qdrant, "count_version_points", qdrant_unavailable)
        runner = ingestion_job_runner(
            migrated_database,
            pipeline,
            lease_owner=f"validate-transient-{'terminal' if exhausted else 'retry'}-worker",
        )
        assert await runner.run_once(uploaded.job_id) is True
    finally:
        await pipeline.aclose()

    async with migrated_database.sessions() as session:
        document = await session.get(Document, uploaded.document_id)
        version = await session.get(DocumentVersion, uploaded.version_id)
        state = await session.get(
            DocumentIndexState,
            (uploaded.version_id, seeded.generation_id),
        )
        job = await session.get(Job, uploaded.job_id)
    assert document is not None and version is not None and state is not None and job is not None
    assert tuple(point.id for point in await qdrant.stored_points(spec.name)) == tuple(
        point.id for point in points_before
    )
    assert state.next_chunk_index == checkpoint_before
    assert all(store.objects[key] == value for key, value in artifacts_before.items())
    assert document.current_version_id is None
    assert document.pending_version_id == uploaded.version_id
    assert job.retryable is True
    assert job.error_code == "QDRANT_UNAVAILABLE"
    assert job.lease_owner is None and job.lease_expires_at is None
    if not exhausted:
        assert job.status == "retry_wait"
        assert document.status == "processing"
        assert version.status == "indexing"
        assert state.status == "indexing"
        assert state.actual_point_count is None
        assert state.error_code is None
        return

    assert job.status == "failed"
    assert document.status == "failed"
    assert version.status == "failed"
    assert state.status == "failed"
    assert state.actual_point_count is None
    assert state.error_code == "QDRANT_UNAVAILABLE"
    with pytest.raises(BusinessError) as duplicate:
        await database_upload(
            migrated_database,
            seeded,
            store,
            content=(f"{idempotency_key} content " * 220).encode(),
            idempotency_key=f"{idempotency_key}-duplicate",
        )
    assert duplicate.value.code == "DUPLICATE_DOCUMENT"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_terminal_failure_real_runner_preserves_index_on_generation_config_drift(
    migrated_database: Database,
) -> None:
    idempotency_key = "runner-generation-config-terminal"
    seeded, spec, qdrant, store, uploaded, context = await prepare_embed_index_stage(
        migrated_database,
        idempotency_key=idempotency_key,
        batch_size=2,
    )
    pipeline = ingestion_pipeline(
        migrated_database,
        store,
        embedding_gateway=DeterministicEmbeddingGateway(),
        qdrant=qdrant,
        provider_usage_sink=SqlAlchemyProviderUsageSink(migrated_database.sessions),
    )
    try:
        await pipeline.handle(context)
        assert context.lease.stage == "validate"
        points_before = await qdrant.stored_points(spec.name)
        async with migrated_database.sessions() as session:
            version_before = await session.get(DocumentVersion, uploaded.version_id)
            state_before = await session.get(
                DocumentIndexState,
                (uploaded.version_id, seeded.generation_id),
            )
        assert version_before is not None and state_before is not None
        artifact_keys = (
            version_before.source_object_key,
            version_before.parsed_object_key,
            version_before.chunk_manifest_object_key,
        )
        assert all(key is not None and key in store.objects for key in artifact_keys)
        artifacts_before = {cast(str, key): store.objects[cast(str, key)] for key in artifact_keys}
        checkpoint_before = state_before.next_chunk_index

        async with migrated_database.sessions() as session, session.begin():
            await SqlAlchemyJobRepository(session).release_claim(context.lease)
            generation = await session.get(
                KnowledgeBaseIndexGeneration,
                seeded.generation_id,
            )
            assert generation is not None
            generation.embedding_config_hash = "f" * 64

        runner = ingestion_job_runner(
            migrated_database,
            pipeline,
            lease_owner="runner-generation-config-terminal-worker",
        )
        assert await runner.run_once(uploaded.job_id) is True
    finally:
        await pipeline.aclose()

    async with migrated_database.sessions() as session:
        document = await session.get(Document, uploaded.document_id)
        version = await session.get(DocumentVersion, uploaded.version_id)
        state = await session.get(
            DocumentIndexState,
            (uploaded.version_id, seeded.generation_id),
        )
        job = await session.get(Job, uploaded.job_id)
    assert document is not None and version is not None and state is not None and job is not None
    assert document.status == "failed"
    assert document.current_version_id is None
    assert document.pending_version_id == uploaded.version_id
    assert version.status == "failed"
    assert state.status == "failed"
    assert state.next_chunk_index == checkpoint_before
    assert state.error_code == "INGESTION_STAGE_CONFLICT"
    assert job.status == "failed"
    assert job.retryable is False
    assert job.error_code == "INGESTION_STAGE_CONFLICT"
    assert job.lease_owner is None and job.lease_expires_at is None
    assert all(store.objects[key] == value for key, value in artifacts_before.items())
    assert tuple(point.id for point in await qdrant.stored_points(spec.name)) == tuple(
        point.id for point in points_before
    )

    with pytest.raises(BusinessError) as duplicate:
        await database_upload(
            migrated_database,
            seeded,
            store,
            content=(f"{idempotency_key} content " * 220).encode(),
            idempotency_key=f"{idempotency_key}-duplicate",
        )
    assert duplicate.value.code == "DUPLICATE_DOCUMENT"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_activation_is_one_fenced_transaction_after_pre_activation_invisibility(
    migrated_database: Database,
) -> None:
    seeded, _spec, qdrant, store, uploaded, context = await prepare_embed_index_stage(
        migrated_database,
        idempotency_key="activate-happy",
        batch_size=2,
    )
    pipeline = ingestion_pipeline(
        migrated_database,
        store,
        embedding_gateway=DeterministicEmbeddingGateway(),
        qdrant=qdrant,
        provider_usage_sink=SqlAlchemyProviderUsageSink(migrated_database.sessions),
    )
    try:
        await pipeline.handle(context)
        await pipeline.handle(context)
        assert context.lease.stage == "activate"
        async with migrated_database.sessions() as session:
            pre_version = await session.get(DocumentVersion, uploaded.version_id)
            assert pre_version is not None
            pre_document = await session.get(Document, pre_version.document_id)
            pre_state = await session.get(
                DocumentIndexState,
                (uploaded.version_id, seeded.generation_id),
            )
            pre_kb = await session.get(KnowledgeBase, seeded.knowledge_base_id)
            pre_mutations = await session.scalar(
                select(func.count()).select_from(KnowledgeBaseMutation)
            )
        assert pre_document is not None and pre_state is not None and pre_kb is not None
        assert pre_document.status == "processing"
        assert pre_document.current_version_id is None
        assert pre_document.pending_version_id == uploaded.version_id
        assert pre_version.status == "indexing"
        assert pre_state.status == "validated"

        outcome = await pipeline.handle(context)
    finally:
        await pipeline.aclose()

    assert outcome.value == "complete"
    assert context.finalized is True
    async with migrated_database.sessions() as session:
        version = await session.get(DocumentVersion, uploaded.version_id)
        assert version is not None
        document = await session.get(Document, version.document_id)
        state = await session.get(
            DocumentIndexState,
            (uploaded.version_id, seeded.generation_id),
        )
        kb = await session.get(KnowledgeBase, seeded.knowledge_base_id)
        job = await session.get(Job, uploaded.job_id)
        activation_mutation = await session.scalar(
            select(KnowledgeBaseMutation).where(
                KnowledgeBaseMutation.knowledge_base_id == seeded.knowledge_base_id,
                KnowledgeBaseMutation.mutation_type == "document_activated",
                KnowledgeBaseMutation.target_id == uploaded.version_id,
            )
        )
        mutation_count = await session.scalar(
            select(func.count()).select_from(KnowledgeBaseMutation)
        )
    assert document is not None and state is not None and kb is not None and job is not None
    assert pre_mutations is not None
    assert version.status == "ready" and version.activated_at is not None
    assert document.status == "active"
    assert document.current_version_id == uploaded.version_id
    assert document.pending_version_id is None
    assert document.mime_type == version.detected_mime_type
    assert state.status == "validated"
    assert state.expected_point_count == state.actual_point_count
    assert kb.mutation_revision == pre_kb.mutation_revision + 1
    assert activation_mutation is not None
    assert activation_mutation.revision == kb.mutation_revision
    assert activation_mutation.payload == {"document_id": str(document.id)}
    assert mutation_count == pre_mutations + 1
    assert job.status == "succeeded"
    assert job.retryable is False
    assert job.lease_owner is None and job.lease_expires_at is None
    assert job.finished_at is not None


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("drift", ["state_count", "pending", "checksum", "generation"])
async def test_activation_business_conflict_terminalizes_without_partial_visibility(
    migrated_database: Database,
    drift: str,
) -> None:
    seeded, _spec, qdrant, store, uploaded, original_context = await prepare_embed_index_stage(
        migrated_database,
        idempotency_key=f"activate-conflict-{drift}",
        batch_size=2,
    )
    pipeline = ingestion_pipeline(
        migrated_database,
        store,
        embedding_gateway=DeterministicEmbeddingGateway(),
        qdrant=qdrant,
        provider_usage_sink=SqlAlchemyProviderUsageSink(migrated_database.sessions),
    )
    try:
        await pipeline.handle(original_context)
        await pipeline.handle(original_context)
        async with migrated_database.sessions() as session, session.begin():
            version = await session.get(DocumentVersion, uploaded.version_id)
            assert version is not None
            document = await session.get(Document, version.document_id)
            state = await session.get(
                DocumentIndexState,
                (uploaded.version_id, seeded.generation_id),
            )
            kb = await session.get(KnowledgeBase, seeded.knowledge_base_id)
            assert document is not None and state is not None and kb is not None
            mutation_revision = kb.mutation_revision
            if drift == "state_count":
                assert state.actual_point_count is not None
                state.actual_point_count += 1
            elif drift == "pending":
                document.pending_version_id = None
            elif drift == "checksum":
                document.checksum_sha256 = "0" * 64
            else:
                generation = await session.get(
                    KnowledgeBaseIndexGeneration,
                    seeded.generation_id,
                )
                assert generation is not None
                generation.status = "retiring"

        context = JobExecutionContext(
            job_repository_context(migrated_database),
            original_context.lease,
            timedelta(seconds=30),
            domain_finalization_enabled=True,
        )
        outcome = await pipeline.handle(context)
    finally:
        await pipeline.aclose()

    assert outcome.value == "complete" and context.finalized is True
    async with migrated_database.sessions() as session:
        version = await session.get(DocumentVersion, uploaded.version_id)
        assert version is not None
        document = await session.get(Document, version.document_id)
        state = await session.get(
            DocumentIndexState,
            (uploaded.version_id, seeded.generation_id),
        )
        job = await session.get(Job, uploaded.job_id)
        kb = await session.get(KnowledgeBase, seeded.knowledge_base_id)
    assert document is not None and state is not None and job is not None and kb is not None
    assert document.status == ("processing" if drift == "pending" else "failed")
    assert document.current_version_id is None
    assert document.pending_version_id == (None if drift == "pending" else uploaded.version_id)
    assert version.status == "failed" and version.activated_at is None
    assert state.status == "failed"
    assert state.error_code == "DOCUMENT_ACTIVATION_CONFLICT"
    assert job.status == "failed"
    assert job.error_code == "DOCUMENT_ACTIVATION_CONFLICT"
    assert kb.mutation_revision == mutation_revision


@pytest.mark.integration
@pytest.mark.asyncio
async def test_activation_conflict_never_overwrites_an_existing_active_version(
    migrated_database: Database,
) -> None:
    seeded, _spec, qdrant, store, uploaded, original_context = await prepare_embed_index_stage(
        migrated_database,
        idempotency_key="activate-existing-current",
        batch_size=2,
    )
    pipeline = ingestion_pipeline(
        migrated_database,
        store,
        embedding_gateway=DeterministicEmbeddingGateway(),
        qdrant=qdrant,
        provider_usage_sink=SqlAlchemyProviderUsageSink(migrated_database.sessions),
    )
    try:
        await pipeline.handle(original_context)
        await pipeline.handle(original_context)
        other_version_id = uuid4()
        async with migrated_database.sessions() as session, session.begin():
            target = await session.get(DocumentVersion, uploaded.version_id)
            assert target is not None
            document = await session.get(Document, target.document_id)
            kb = await session.get(KnowledgeBase, seeded.knowledge_base_id)
            assert document is not None and kb is not None
            mutation_revision = kb.mutation_revision
            session.add(
                DocumentVersion(
                    id=other_version_id,
                    document_id=document.id,
                    version_number=2,
                    source_object_key=target.source_object_key,
                    source_checksum_sha256=target.source_checksum_sha256,
                    detected_mime_type=target.detected_mime_type,
                    status="ready",
                    activated_at=datetime.now(UTC),
                )
            )
            await session.flush()
            document.status = "active"
            document.current_version_id = other_version_id
            document.pending_version_id = target.id

        context = JobExecutionContext(
            job_repository_context(migrated_database),
            original_context.lease,
            timedelta(seconds=30),
            domain_finalization_enabled=True,
        )
        outcome = await pipeline.handle(context)
    finally:
        await pipeline.aclose()

    assert outcome.value == "complete" and context.finalized is True
    async with migrated_database.sessions() as session:
        target = await session.get(DocumentVersion, uploaded.version_id)
        assert target is not None
        document = await session.get(Document, target.document_id)
        state = await session.get(
            DocumentIndexState,
            (uploaded.version_id, seeded.generation_id),
        )
        job = await session.get(Job, uploaded.job_id)
        kb = await session.get(KnowledgeBase, seeded.knowledge_base_id)
    assert document is not None and state is not None and job is not None and kb is not None
    assert document.status == "active"
    assert document.current_version_id == other_version_id
    assert document.pending_version_id is None
    assert target.status == "failed" and target.activated_at is None
    assert state.status == "failed"
    assert job.status == "failed" and job.error_code == "DOCUMENT_ACTIVATION_CONFLICT"
    assert kb.mutation_revision == mutation_revision


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stale_activation_worker_cannot_publish_or_append_mutation(
    migrated_database: Database,
) -> None:
    seeded, _spec, qdrant, store, uploaded, context = await prepare_embed_index_stage(
        migrated_database,
        idempotency_key="activate-stale-lease",
        batch_size=2,
    )
    pipeline = ingestion_pipeline(
        migrated_database,
        store,
        embedding_gateway=DeterministicEmbeddingGateway(),
        qdrant=qdrant,
        provider_usage_sink=SqlAlchemyProviderUsageSink(migrated_database.sessions),
    )
    try:
        await pipeline.handle(context)
        await pipeline.handle(context)
        async with migrated_database.sessions() as session:
            version = await session.get(DocumentVersion, uploaded.version_id)
            kb = await session.get(KnowledgeBase, seeded.knowledge_base_id)
            mutation_count = await session.scalar(
                select(func.count()).select_from(KnowledgeBaseMutation)
            )
        assert version is not None and kb is not None
        reclaimed = await expire_and_reclaim_upload_job(
            migrated_database,
            uploaded.job_id,
            lease_owner="activation-worker-b",
        )

        with pytest.raises(LostLeaseError):
            await pipeline.handle(context)
    finally:
        await pipeline.aclose()

    async with migrated_database.sessions() as session:
        stored_version = await session.get(DocumentVersion, uploaded.version_id)
        assert stored_version is not None
        document = await session.get(Document, stored_version.document_id)
        state = await session.get(
            DocumentIndexState,
            (uploaded.version_id, seeded.generation_id),
        )
        job = await session.get(Job, uploaded.job_id)
        stored_kb = await session.get(KnowledgeBase, seeded.knowledge_base_id)
        stored_mutation_count = await session.scalar(
            select(func.count()).select_from(KnowledgeBaseMutation)
        )
    assert document is not None and state is not None and job is not None and stored_kb is not None
    assert stored_version.status == "indexing" and stored_version.activated_at is None
    assert document.status == "processing" and document.current_version_id is None
    assert document.pending_version_id == uploaded.version_id
    assert state.status == "validated"
    assert job.status == "running"
    assert job.lease_owner == reclaimed.lease_owner
    assert job.lease_epoch == reclaimed.lease_epoch
    assert stored_kb.mutation_revision == kb.mutation_revision
    assert stored_mutation_count == mutation_count


def _search_filter_snapshot() -> dict[str, object]:
    identifier = "44444444444444448444444444444444"
    return {
        "fields": [
            {
                "name": "category",
                "source_path": "attributes.category",
                "type": "keyword",
                "operators": ["eq", "in"],
                "field_id": "fld_RERERERERESERERERERERA",
                "payload_path": f"metadata.f_{identifier}",
            }
        ]
    }


@dataclass(frozen=True, slots=True)
class SeededSearchDocument:
    seeded: SeededUploadContext
    spec: CollectionSpec
    qdrant: FakeQdrantClient
    profile_id: UUID
    provider_id: UUID
    document_id: UUID
    version_id: UUID
    point_id: UUID


async def seed_search_document(database: Database) -> SeededSearchDocument:
    seeded = await seed_upload_context(database)
    profile_id, provider_id, spec = await configure_embedding_generation(
        database,
        seeded,
        filter_snapshot=_search_filter_snapshot(),
    )
    document_id, version_id = uuid4(), uuid4()
    content = "Authentication configuration"
    chunk_hash = "a" * 64
    now = datetime.now(UTC)
    async with database.sessions() as session, session.begin():
        generation = await session.get(
            KnowledgeBaseIndexGeneration,
            seeded.generation_id,
        )
        api_key = await session.get(ApiKey, seeded.actor.key_id)
        assert api_key is not None and generation is not None
        api_key.capabilities = [Capability.RETRIEVE.value]
        document = Document(
            id=document_id,
            knowledge_base_id=seeded.knowledge_base_id,
            display_name="guide.md",
            mime_type="text/markdown",
            checksum_sha256="b" * 64,
            current_version_id=None,
            pending_version_id=None,
            status="active",
            tags=["guide"],
            metadata_={"attributes": {"category": "guide"}},
            deleted_at=None,
        )
        version = DocumentVersion(
            id=version_id,
            document_id=document_id,
            version_number=1,
            source_object_key=f"private/{uuid4().hex}/source.md",
            source_checksum_sha256="b" * 64,
            declared_mime_type="text/markdown",
            detected_mime_type="text/markdown",
            source_extension="md",
            parser_name="markdown_v1",
            parser_version="1",
            parser_config={},
            chunker_name="recursive_text_v1",
            chunker_version="1",
            chunker_config={},
            chunk_count=1,
            status="ready",
            activated_at=now,
            chunk_manifest_object_key=f"private/{uuid4().hex}/chunks.json",
            chunk_manifest_checksum_sha256="c" * 64,
            chunk_config_hash="d" * 64,
        )
        session.add(document)
        await session.flush()
        session.add(version)
        await session.flush()
        document.current_version_id = version_id
        session.add(
            DocumentIndexState(
                document_version_id=version_id,
                index_generation_id=seeded.generation_id,
                status="validated",
                expected_point_count=1,
                actual_point_count=1,
                error_code=None,
                validated_at=now,
                chunk_manifest_checksum_sha256="c" * 64,
                embedding_config_hash=generation.embedding_config_hash,
                next_chunk_index=1,
                safe_error_message=None,
            )
        )

    qdrant = FakeQdrantClient()
    await qdrant.seed_collection(spec, created_at=now)
    identity = point_id(version_id, 0, chunk_hash)
    await qdrant.upsert_points(
        spec.name,
        (
            QdrantPoint(
                identity,
                (0.0, 1.0 / 7.0, 2.0 / 7.0),
                {
                    "knowledge_base_id": str(seeded.knowledge_base_id),
                    "document_id": str(document_id),
                    "version_id": str(version_id),
                    "chunk_index": 0,
                    "chunk_hash": chunk_hash,
                    "text": content,
                    "title_path": ["Guide", "Authentication"],
                    "start_offset": 0,
                    "end_offset": len(content),
                    "metadata": {"f_44444444444444448444444444444444": "guide"},
                },
            ),
        ),
    )
    return SeededSearchDocument(
        seeded=seeded,
        spec=spec,
        qdrant=qdrant,
        profile_id=profile_id,
        provider_id=provider_id,
        document_id=document_id,
        version_id=version_id,
        point_id=identity,
    )


def retrieval_actor(seeded: SeededUploadContext) -> AgentPrincipal:
    return replace(
        seeded.actor,
        capabilities=frozenset({Capability.RETRIEVE}),
    )


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.acceptance
async def test_search_api_returns_only_pg_visible_chunks_and_persists_content_free_usage(
    migrated_database: Database,
) -> None:
    search = await seed_search_document(migrated_database)
    actor = retrieval_actor(search.seeded)
    gateway = DeterministicEmbeddingGateway()
    app = create_app()
    app.dependency_overrides[require_agent_principal] = lambda: actor

    async with migrated_database.sessions() as retrieval_session:
        service = SearchService(
            repository=SqlAlchemyRetrievalRepository(retrieval_session),
            embedding_gateway=gateway,
            search_index=search.qdrant,
            provider_usage_sink=SqlAlchemyProviderUsageSink(migrated_database.sessions),
            query_log_sink=SqlAlchemyQueryLogSink(migrated_database.sessions),
        )
        app.dependency_overrides[get_retrieval_service] = lambda: service
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                f"/v1/knowledge-bases/{search.seeded.knowledge_base_id}/search",
                headers={"X-Request-ID": "req-dense-search-e2e"},
                json={
                    "query": "  private authentication question  ",
                    "filters": {
                        "document_ids": [str(search.document_id), str(uuid4())],
                        "metadata": {"category": "guide"},
                    },
                },
            )

    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"results", "index"}
    assert body["index"] == {
        "generation_id": str(search.seeded.generation_id),
        "embedding_profile_id": str(search.profile_id),
    }
    assert len(body["results"]) == 1
    result = body["results"][0]
    assert result == {
        "text": "Authentication configuration",
        "score": 1.0,
        "document_id": str(search.document_id),
        "version_id": str(search.version_id),
        "chunk_index": 0,
        "title": "Authentication",
        "title_path": ["Guide", "Authentication"],
        "source": {
            "filename": "guide.md",
            "start_offset": 0,
            "end_offset": len("Authentication configuration"),
        },
        "metadata": {"category": "guide"},
    }
    assert response.headers["cache-control"] == "no-store"

    async with migrated_database.sessions() as session:
        query_logs = (await session.scalars(select(QueryLog))).all()
        provider_usage = (await session.scalars(select(ProviderUsage))).all()
    assert len(query_logs) == 1
    assert query_logs[0].request_id == "req-dense-search-e2e"
    assert query_logs[0].actor_api_key_id == actor.key_id
    assert query_logs[0].knowledge_base_ids == [search.seeded.knowledge_base_id]
    assert query_logs[0].status == "succeeded"
    assert len(provider_usage) == 1
    assert provider_usage[0].request_id == "req-dense-search-e2e"
    assert provider_usage[0].actor_api_key_id == actor.key_id
    assert provider_usage[0].provider_config_id == search.provider_id
    assert provider_usage[0].model_profile_id == search.profile_id
    persisted = repr(query_logs[0].__dict__) + repr(provider_usage[0].__dict__)
    for forbidden in (
        "private authentication question",
        "Authentication configuration",
        "private/",
        "ciphertext",
        "Authorization",
    ):
        assert forbidden not in persisted


@pytest.mark.integration
@pytest.mark.asyncio
async def test_search_document_filter_hides_stale_qdrant_point_like_nonexistent_document(
    migrated_database: Database,
) -> None:
    search = await seed_search_document(migrated_database)
    now = datetime.now(UTC)
    async with migrated_database.sessions() as session, session.begin():
        document = await session.get(Document, search.document_id)
        assert document is not None
        document.status = "deleted"
        document.deleted_at = now

    class CapturingIndex:
        def __init__(self, delegate: FakeQdrantClient) -> None:
            self._delegate = delegate
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
            return await self._delegate.search_points(
                collection,
                vector,
                limit=limit,
                query_filter=query_filter,
            )

    gateway = DeterministicEmbeddingGateway()
    index = CapturingIndex(search.qdrant)
    nonexistent_document = uuid4()
    async with migrated_database.sessions() as session:
        service = SearchService(
            repository=SqlAlchemyRetrievalRepository(session),
            embedding_gateway=gateway,
            search_index=index,
            provider_usage_sink=SqlAlchemyProviderUsageSink(migrated_database.sessions),
            query_log_sink=SqlAlchemyQueryLogSink(migrated_database.sessions),
        )
        responses = []
        for request_id, document_id in (
            ("req-stale-filter", search.document_id),
            ("req-missing-filter", nonexistent_document),
        ):
            responses.append(
                await service.search(
                    knowledge_base_id=search.seeded.knowledge_base_id,
                    actor=retrieval_actor(search.seeded),
                    request_id=request_id,
                    command=SearchRequest(
                        query="same path",
                        filters=SearchFilters(document_ids=(document_id,)),
                    ),
                )
            )

    assert [response.results for response in responses] == [(), ()]
    assert gateway.calls == []
    assert index.calls == []
    assert await search.qdrant.search_points(
        search.spec.name,
        (0.0, 1.0 / 7.0, 2.0 / 7.0),
        limit=10,
        query_filter=QdrantSearchFilter(document_ids=(search.document_id,)),
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_search_pg_visibility_drops_old_pending_failed_deleted_cross_kb_and_mismatches(
    migrated_database: Database,
) -> None:
    search = await seed_search_document(migrated_database)
    actor = retrieval_actor(search.seeded)
    now = datetime.now(UTC)
    invalid_points: list[QdrantPoint] = []
    async with migrated_database.sessions() as session, session.begin():
        valid_document = await session.get(Document, search.document_id)
        generation = await session.get(
            KnowledgeBaseIndexGeneration,
            search.seeded.generation_id,
        )
        assert valid_document is not None and generation is not None
        generation_embedding_config_hash = generation.embedding_config_hash

        case_versions: dict[str, tuple[UUID, UUID]] = {}

        async def add_visibility_case(
            name: str,
            *,
            document_status: str = "active",
            version_status: str = "ready",
            deleted_at: datetime | None = None,
            activated_at: datetime | None = now,
            add_state: bool = True,
            validated_at: datetime | None = now,
            chunk_count: int = 1,
            expected_point_count: int = 1,
            actual_point_count: int = 1,
            next_chunk_index: int = 1,
            embedding_hash_matches: bool = True,
            manifest_checksum_matches: bool = True,
        ) -> None:
            document_id, version_id = uuid4(), uuid4()
            source_checksum = hashlib.sha256(f"source-{name}".encode()).hexdigest()
            manifest_checksum = hashlib.sha256(f"manifest-{name}".encode()).hexdigest()
            document = Document(
                id=document_id,
                knowledge_base_id=search.seeded.knowledge_base_id,
                display_name=f"{name}.txt",
                mime_type="text/plain",
                checksum_sha256=source_checksum,
                current_version_id=None,
                pending_version_id=None,
                status=document_status,
                tags=[],
                metadata_={},
                deleted_at=deleted_at,
            )
            version = DocumentVersion(
                id=version_id,
                document_id=document_id,
                version_number=1,
                source_object_key=f"objects/{name}.txt",
                source_checksum_sha256=source_checksum,
                declared_mime_type="text/plain",
                detected_mime_type="text/plain",
                source_extension="txt",
                parser_name="plain_text_v1",
                parser_version="1",
                parser_config={},
                chunker_name="recursive_text_v1",
                chunker_version="1",
                chunker_config={},
                chunk_count=chunk_count,
                status=version_status,
                activated_at=activated_at,
                chunk_manifest_object_key=f"objects/{name}-chunks.json",
                chunk_manifest_checksum_sha256=manifest_checksum,
                chunk_config_hash=hashlib.sha256(f"chunk-{name}".encode()).hexdigest(),
            )
            session.add(document)
            await session.flush()
            session.add(version)
            await session.flush()
            document.current_version_id = version_id
            if add_state:
                session.add(
                    DocumentIndexState(
                        document_version_id=version_id,
                        index_generation_id=search.seeded.generation_id,
                        status="validated",
                        expected_point_count=expected_point_count,
                        actual_point_count=actual_point_count,
                        validated_at=validated_at,
                        chunk_manifest_checksum_sha256=(
                            manifest_checksum
                            if manifest_checksum_matches
                            else hashlib.sha256(f"other-{name}".encode()).hexdigest()
                        ),
                        embedding_config_hash=(
                            generation_embedding_config_hash if embedding_hash_matches else "f" * 64
                        ),
                        next_chunk_index=next_chunk_index,
                    )
                )
            case_versions[name] = (document_id, version_id)
            invalid_points.append(
                QdrantPoint(
                    point_id(version_id, 0, "a" * 64),
                    (0.0, 1.0 / 7.0, 2.0 / 7.0),
                    {
                        "knowledge_base_id": str(search.seeded.knowledge_base_id),
                        "document_id": str(document_id),
                        "version_id": str(version_id),
                        "chunk_index": 0,
                        "chunk_hash": "a" * 64,
                        "text": f"hidden {name}",
                        "title_path": [],
                        "start_offset": 0,
                        "end_offset": len(f"hidden {name}"),
                        "metadata": {},
                    },
                )
            )

        await add_visibility_case(
            "pending",
            document_status="processing",
            version_status="uploaded",
            activated_at=None,
        )
        await add_visibility_case("failed", version_status="failed", activated_at=None)
        await add_visibility_case("deleted", document_status="deleted", deleted_at=now)
        await add_visibility_case("missing-state", add_state=False)
        await add_visibility_case("unactivated", activated_at=None)
        await add_visibility_case("unvalidated-at", validated_at=None)
        await add_visibility_case("next-mismatch", next_chunk_index=0)
        await add_visibility_case("expected-mismatch", expected_point_count=2)
        await add_visibility_case("actual-mismatch", actual_point_count=2)
        await add_visibility_case(
            "point-count-vs-chunks",
            expected_point_count=2,
            actual_point_count=2,
        )
        await add_visibility_case("embedding-hash-mismatch", embedding_hash_matches=False)
        await add_visibility_case("manifest-checksum-mismatch", manifest_checksum_matches=False)

        old_version_id = uuid4()
        session.add(
            DocumentVersion(
                id=old_version_id,
                document_id=search.document_id,
                version_number=2,
                source_object_key="objects/old.txt",
                source_checksum_sha256="f" * 64,
                declared_mime_type="text/plain",
                detected_mime_type="text/plain",
                source_extension="txt",
                parser_name="plain_text_v1",
                parser_version="1",
                parser_config={},
                chunker_name="recursive_text_v1",
                chunker_version="1",
                chunker_config={},
                chunk_count=1,
                status="ready",
                activated_at=now,
                chunk_manifest_object_key="objects/old-chunks.json",
                chunk_manifest_checksum_sha256="e" * 64,
                chunk_config_hash="d" * 64,
            )
        )
        await session.flush()
        session.add(
            DocumentIndexState(
                document_version_id=old_version_id,
                index_generation_id=search.seeded.generation_id,
                status="validated",
                expected_point_count=1,
                actual_point_count=1,
                validated_at=now,
                chunk_manifest_checksum_sha256="e" * 64,
                embedding_config_hash=generation.embedding_config_hash,
                next_chunk_index=1,
            )
        )
        invalid_points.append(
            QdrantPoint(
                point_id(old_version_id, 0, "a" * 64),
                (0.0, 1.0 / 7.0, 2.0 / 7.0),
                {
                    "knowledge_base_id": str(search.seeded.knowledge_base_id),
                    "document_id": str(search.document_id),
                    "version_id": str(old_version_id),
                    "chunk_index": 0,
                    "chunk_hash": "a" * 64,
                    "text": "hidden old version",
                    "title_path": [],
                    "start_offset": 0,
                    "end_offset": len("hidden old version"),
                    "metadata": {},
                },
            )
        )

        mismatched_document_id, mismatched_version_id = case_versions["failed"]
        invalid_points.append(
            QdrantPoint(
                point_id(mismatched_version_id, 1, "a" * 64),
                (0.0, 1.0 / 7.0, 2.0 / 7.0),
                {
                    "knowledge_base_id": str(search.seeded.knowledge_base_id),
                    "document_id": str(search.document_id),
                    "version_id": str(mismatched_version_id),
                    "chunk_index": 1,
                    "chunk_hash": "a" * 64,
                    "text": "hidden mismatched version",
                    "title_path": [],
                    "start_offset": 0,
                    "end_offset": len("hidden mismatched version"),
                    "metadata": {},
                },
            )
        )
        assert mismatched_document_id != search.document_id

    second = await seed_second_knowledge_base(migrated_database, search.seeded)
    cross_document, cross_version = uuid4(), uuid4()
    async with migrated_database.sessions() as session, session.begin():
        generation = await session.get(
            KnowledgeBaseIndexGeneration,
            search.seeded.generation_id,
        )
        assert generation is not None
        document = Document(
            id=cross_document,
            knowledge_base_id=second.knowledge_base_id,
            display_name="cross.txt",
            mime_type="text/plain",
            checksum_sha256="9" * 64,
            status="active",
            tags=[],
            metadata_={},
        )
        version = DocumentVersion(
            id=cross_version,
            document_id=cross_document,
            version_number=1,
            source_object_key="objects/cross.txt",
            source_checksum_sha256="9" * 64,
            declared_mime_type="text/plain",
            detected_mime_type="text/plain",
            source_extension="txt",
            parser_name="plain_text_v1",
            parser_version="1",
            parser_config={},
            chunker_name="recursive_text_v1",
            chunker_version="1",
            chunker_config={},
            chunk_count=1,
            status="ready",
            activated_at=now,
            chunk_manifest_object_key="objects/cross-chunks.json",
            chunk_manifest_checksum_sha256="8" * 64,
            chunk_config_hash="7" * 64,
        )
        session.add(document)
        await session.flush()
        session.add(version)
        await session.flush()
        document.current_version_id = cross_version
        session.add(
            DocumentIndexState(
                document_version_id=cross_version,
                index_generation_id=search.seeded.generation_id,
                status="validated",
                expected_point_count=1,
                actual_point_count=1,
                validated_at=now,
                chunk_manifest_checksum_sha256="8" * 64,
                embedding_config_hash=generation.embedding_config_hash,
                next_chunk_index=1,
            )
        )
    invalid_points.append(
        QdrantPoint(
            point_id(cross_version, 0, "a" * 64),
            (0.0, 1.0 / 7.0, 2.0 / 7.0),
            {
                "knowledge_base_id": str(second.knowledge_base_id),
                "document_id": str(cross_document),
                "version_id": str(cross_version),
                "chunk_index": 0,
                "chunk_hash": "a" * 64,
                "text": "hidden cross kb",
                "title_path": [],
                "start_offset": 0,
                "end_offset": len("hidden cross kb"),
                "metadata": {},
            },
        )
    )
    await search.qdrant.upsert_points(search.spec.name, invalid_points)

    async with migrated_database.sessions() as session:
        service = SearchService(
            repository=SqlAlchemyRetrievalRepository(session),
            embedding_gateway=DeterministicEmbeddingGateway(),
            search_index=search.qdrant,
            provider_usage_sink=SqlAlchemyProviderUsageSink(migrated_database.sessions),
            query_log_sink=SqlAlchemyQueryLogSink(migrated_database.sessions),
        )
        response = await service.search(
            knowledge_base_id=search.seeded.knowledge_base_id,
            actor=actor,
            request_id="req-visibility-matrix",
            command=SearchRequest(query="visibility", top_k=50),
        )

    assert {result.document_id for result in response.results} == {search.document_id}
    assert {result.version_id for result in response.results} == {search.version_id}
    serialized = response.model_dump_json()
    for forbidden in (
        "hidden pending",
        "hidden failed",
        "hidden deleted",
        "hidden missing-state",
        "hidden unactivated",
        "hidden unvalidated-at",
        "hidden next-mismatch",
        "hidden expected-mismatch",
        "hidden actual-mismatch",
        "hidden point-count-vs-chunks",
        "hidden embedding-hash-mismatch",
        "hidden manifest-checksum-mismatch",
        "hidden old version",
        "hidden mismatched version",
        "hidden cross kb",
    ):
        assert forbidden not in serialized
