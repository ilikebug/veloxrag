import asyncio
import gc
import hashlib
import inspect
import json
import logging
from collections.abc import AsyncIterable, AsyncIterator
from contextlib import suppress
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest
from python_multipart.exceptions import MultipartParseError

from rag_service.api.errors import BusinessError
from rag_service.auth.policies import AgentPrincipal, Capability
from rag_service.ingestion import services as ingestion_services
from rag_service.ingestion.repositories import (
    SqlAlchemyUploadRepository,
    UploadPreflight,
    UploadReservation,
    UploadReservationResult,
)
from rag_service.ingestion.routes import upload_document
from rag_service.ingestion.schemas import (
    UploadForm,
    UploadRequestFingerprintInput,
    canonical_upload_filename,
    upload_request_fingerprint,
    validate_metadata_against_filter_snapshot,
)
from rag_service.ingestion.services import (
    DocumentUploadService,
    SourceUpload,
    stage_multipart_upload,
)
from rag_service.observability.metrics import OperationalMetrics


class FakeObjectStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.deleted: list[str] = []
        self.published: list[str] = []
        self.publish_pairs: list[tuple[str, str]] = []
        self.fail_upload = False

    async def upload_stream(
        self,
        object_key: str,
        stream: AsyncIterable[bytes],
        *,
        content_type: str,
        max_bytes: int,
    ) -> object:
        del content_type
        if self.fail_upload:
            raise RuntimeError("secret backend failure")
        content = b""
        async for chunk in stream:
            content += chunk
            if len(content) > max_bytes:
                raise AssertionError("service failed to bound the stream")
        self.objects[object_key] = content
        self.published.append(object_key)
        return SimpleNamespace(
            object_key=object_key,
            size=len(content),
            checksum_sha256=hashlib.sha256(content).hexdigest(),
        )

    async def delete_best_effort(self, object_key: str) -> bool:
        self.deleted.append(object_key)
        self.objects.pop(object_key, None)
        return True

    async def verify_object(
        self,
        object_key: str,
        *,
        expected_size: int,
        expected_checksum: str,
    ) -> object:
        content = self.objects[object_key]
        assert len(content) == expected_size
        assert hashlib.sha256(content).hexdigest() == expected_checksum
        return SimpleNamespace(
            object_key=object_key,
            size=expected_size,
            checksum_sha256=expected_checksum,
        )

    async def publish_temp(
        self,
        temp_key: str,
        final_key: str,
        *,
        expected_size: int,
        expected_checksum: str,
    ) -> object:
        content = self.objects.pop(temp_key)
        assert len(content) == expected_size
        assert hashlib.sha256(content).hexdigest() == expected_checksum
        self.objects[final_key] = content
        self.publish_pairs.append((temp_key, final_key))
        return SimpleNamespace(
            object_key=final_key,
            size=expected_size,
            checksum_sha256=expected_checksum,
        )


class FakeNotifier:
    def __init__(self, result: bool = True) -> None:
        self.result = result
        self.jobs: list[object] = []

    async def notify(self, job_id: object) -> bool:
        self.jobs.append(job_id)
        return self.result


class CollectingLogHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class FakeRepository:
    def __init__(self, preflight: UploadPreflight) -> None:
        self.preflight_value = preflight
        self.result: UploadReservationResult | BaseException | None = None
        self.reservations: list[UploadReservation] = []

    async def preflight(self, knowledge_base_id: object) -> UploadPreflight:
        del knowledge_base_id
        return self.preflight_value

    async def reserve(self, reservation: UploadReservation) -> UploadReservationResult:
        self.reservations.append(reservation)
        if isinstance(self.result, BaseException):
            raise self.result
        assert self.result is not None
        return self.result


async def chunks(*values: bytes) -> AsyncIterator[bytes]:
    for value in values:
        yield value


def multipart_body(
    boundary: str,
    parts: tuple[tuple[str, str | None, str | None, bytes], ...],
) -> bytes:
    body = bytearray()
    for name, filename, content_type, value in parts:
        body.extend(f"--{boundary}\r\n".encode())
        disposition = f'Content-Disposition: form-data; name="{name}"'
        if filename is not None:
            disposition += f'; filename="{filename}"'
        body.extend(f"{disposition}\r\n".encode())
        if content_type is not None:
            body.extend(f"Content-Type: {content_type}\r\n".encode())
        body.extend(b"\r\n")
        body.extend(value)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    return bytes(body)


async def sliced_body(value: bytes, size: int = 7) -> AsyncIterator[bytes]:
    for start in range(0, len(value), size):
        yield value[start : start + size]


def actor(kb_id: UUID) -> AgentPrincipal:
    return AgentPrincipal(
        key_id=uuid4(),
        public_id="YWdlbnQtdXBsb2FkLXRlc3Q",
        capabilities=frozenset({Capability.INGEST}),
        knowledge_base_ids=frozenset({kb_id}),
        query_profile_ids=frozenset(),
        default_query_profile_id=None,
        raw_file_read=False,
        requests_per_minute=60,
        max_concurrency=4,
    )


def preflight(kb_id: UUID) -> UploadPreflight:
    return UploadPreflight(
        knowledge_base_id=kb_id,
        generation_id=uuid4(),
        filter_schema_snapshot={
            "fields": [
                {
                    "name": "department",
                    "source_path": "attributes.department",
                    "type": "keyword",
                    "operators": ["eq"],
                    "field_id": "fld_AAAAAAAAAAAAAAAAAAAAAA",
                    "payload_path": "metadata.f_00000000000000000000000000000000",
                }
            ]
        },
        applied_filter_schema_revision=3,
        current_filter_schema_revision=3,
    )


def test_upload_form_canonicalizes_safe_fields_and_rejects_bad_json() -> None:
    form = UploadForm.from_multipart(
        display_name="  Guide  ",
        metadata=json.dumps({"attributes": {"department": "docs"}}),
        tags='["beta", "alpha"]',
    )
    assert form.display_name == "Guide"
    assert form.tags == ("alpha", "beta")
    assert form.metadata == {"attributes": {"department": "docs"}}
    assert canonical_upload_filename("../../unsafe\\manual.MD") == "manual.MD"

    for metadata, tags in (("[]", "[]"), ("{}", "{}"), ("{", "[]")):
        with pytest.raises(BusinessError) as captured:
            UploadForm.from_multipart(display_name=None, metadata=metadata, tags=tags)
        assert captured.value.code == "VALIDATION_ERROR"


def test_filter_snapshot_validation_checks_revision_path_and_type() -> None:
    current = preflight(uuid4())
    validate_metadata_against_filter_snapshot({"attributes": {"department": "docs"}}, current)

    with pytest.raises(BusinessError) as bad_type:
        validate_metadata_against_filter_snapshot({"attributes": {"department": 7}}, current)
    assert bad_type.value.code == "VALIDATION_ERROR"

    with pytest.raises(BusinessError) as stale:
        validate_metadata_against_filter_snapshot(
            {}, replace(current, current_filter_schema_revision=4)
        )
    assert stale.value.code == "GENERATION_CONFIGURATION_CONFLICT"


@pytest.mark.parametrize(
    ("field_type", "value"),
    [
        ("keyword", "x" * 4097),
        ("keyword", "unsafe\x00value"),
        ("integer", 2**63),
        ("integer", -(2**63) - 1),
        ("float", 10**400),
    ],
)
def test_filter_snapshot_validation_rejects_values_unsafe_for_qdrant(
    field_type: str,
    value: Any,
) -> None:
    current = replace(
        preflight(uuid4()),
        filter_schema_snapshot={
            "fields": [
                {
                    "name": "value",
                    "source_path": "value",
                    "type": field_type,
                }
            ]
        },
    )

    with pytest.raises(BusinessError) as captured:
        validate_metadata_against_filter_snapshot({"value": value}, current)

    assert captured.value.code == "VALIDATION_ERROR"


def test_datetime_filter_values_require_a_strict_rfc3339_timestamp() -> None:
    current = replace(
        preflight(uuid4()),
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
    for value in ("2026-07-29T12:34:56Z", "2026-07-29T12:34:56.123+08:00"):
        validate_metadata_against_filter_snapshot({"published_at": value}, current)

    for value in (
        "contains-T-but-is-not-a-date",
        "2026-07-29T12:34:56",
        "2026-13-29T12:34:56Z",
        "2026-07-29 12:34:56Z",
    ):
        with pytest.raises(BusinessError) as captured:
            validate_metadata_against_filter_snapshot({"published_at": value}, current)
        assert captured.value.code == "VALIDATION_ERROR"


def test_public_upload_route_and_repository_do_not_own_unbounded_or_outer_transactions() -> None:
    assert "request.form(" not in inspect.getsource(upload_document)
    assert "self._session.begin(" not in inspect.getsource(SqlAlchemyUploadRepository.preflight)
    assert "self._session.begin(" not in inspect.getsource(SqlAlchemyUploadRepository.reserve)


def test_fingerprint_covers_every_canonical_request_input() -> None:
    base = UploadRequestFingerprintInput(
        actor_key_id=uuid4(),
        knowledge_base_id=uuid4(),
        source_checksum_sha256="a" * 64,
        filename="manual.md",
        extension=".md",
        declared_mime_type="text/plain",
        detected_mime_type="text/markdown",
        parser_name="markdown_text_v1",
        display_name="Manual",
        tags=("docs",),
        metadata={"department": "docs"},
    )
    original = upload_request_fingerprint(base)
    assert len(original) == 32
    changes: dict[str, Any] = {
        "actor_key_id": uuid4(),
        "knowledge_base_id": uuid4(),
        "source_checksum_sha256": "b" * 64,
        "filename": "other.md",
        "extension": ".markdown",
        "declared_mime_type": "text/markdown",
        "detected_mime_type": "text/plain",
        "parser_name": "other_v1",
        "display_name": "Other",
        "tags": ("other",),
        "metadata": {"department": "other"},
    }
    assert all(
        upload_request_fingerprint(replace(base, **{field: value})) != original
        for field, value in changes.items()
    )


@pytest.mark.asyncio
async def test_service_publishes_before_reservation_and_notifies_after_commit() -> None:
    kb_id = uuid4()
    repo = FakeRepository(preflight(kb_id))
    store = FakeObjectStore()
    notifier = FakeNotifier(result=False)
    accepted = UploadReservationResult.created(uuid4(), uuid4(), uuid4())
    repo.result = accepted
    service = DocumentUploadService(
        repository=repo,
        object_store=store,
        notifier=notifier,
        max_upload_bytes=50 * 1024 * 1024,
    )

    result = await service.upload(
        knowledge_base_id=kb_id,
        actor=actor(kb_id),
        source=SourceUpload(
            filename="manual.md",
            content_type="text/plain; charset=utf-8",
            chunks=chunks(b"# hello", b"\nworld"),
        ),
        form=UploadForm.from_multipart(
            display_name=None,
            metadata='{"attributes":{"department":"docs"}}',
            tags='["docs"]',
        ),
        idempotency_key="upload-key",
    )

    assert result.status == "queued"
    assert len(store.publish_pairs) == 1
    temp_key, final_key = store.publish_pairs[0]
    assert temp_key.startswith("tmp/uploads/")
    assert final_key == repo.reservations[0].source_object_key
    assert repo.reservations[0].source_object_key in store.objects
    assert notifier.jobs == [result.job_id]


@pytest.mark.asyncio
async def test_upload_observability_counts_once_and_logs_only_safe_operational_context() -> None:
    kb_id = uuid4()
    repo = FakeRepository(preflight(kb_id))
    store = FakeObjectStore()
    accepted = UploadReservationResult.created(uuid4(), uuid4(), uuid4())
    repo.result = accepted
    metrics = OperationalMetrics()
    clock_values = iter((10.0, 10.25))
    handler = CollectingLogHandler()
    previous_handlers = list(ingestion_services.logger.handlers)
    previous_propagate = ingestion_services.logger.propagate
    previous_level = ingestion_services.logger.level
    ingestion_services.logger.handlers = [handler]
    ingestion_services.logger.propagate = False
    ingestion_services.logger.setLevel(logging.INFO)
    sentinel = "query-retrieved-chunk-vector-secret-ciphertext-nonce-authorization-object-key-body"
    try:
        result = await DocumentUploadService(
            repository=repo,
            object_store=store,
            notifier=FakeNotifier(),
            max_upload_bytes=1024,
            metrics=metrics,
            monotonic_clock=lambda: next(clock_values),
        ).upload(
            knowledge_base_id=kb_id,
            actor=actor(kb_id),
            source=SourceUpload(
                filename="manual.md",
                content_type="text/markdown",
                chunks=chunks(sentinel.encode()),
            ),
            form=UploadForm.from_multipart(display_name=None, metadata=None, tags=None),
            idempotency_key=None,
            request_id="req-upload-observed",
        )
    finally:
        ingestion_services.logger.handlers = previous_handlers
        ingestion_services.logger.propagate = previous_propagate
        ingestion_services.logger.setLevel(previous_level)

    assert result.job_id == accepted.job_id
    assert metrics.registry.get_sample_value("rag_uploads_total", {"outcome": "succeeded"}) == 1
    assert metrics.registry.get_sample_value(
        "rag_upload_bytes_total", {"outcome": "succeeded"}
    ) == len(sentinel.encode())
    assert (
        metrics.registry.get_sample_value(
            "rag_upload_duration_seconds_count", {"outcome": "succeeded"}
        )
        == 1
    )
    upload_records = [record for record in handler.records if record.msg == "upload.completed"]
    assert len(upload_records) == 1
    record = upload_records[0]
    assert type(record) is logging.LogRecord
    assert record.msg == "upload.completed"
    assert record.__dict__["request_id"] == "req-upload-observed"
    assert record.__dict__["knowledge_base_id"] == str(kb_id)
    assert record.__dict__["document_id"] == str(accepted.document_id)
    assert record.__dict__["version_id"] == str(accepted.version_id)
    assert record.__dict__["job_id"] == str(accepted.job_id)
    rendered = repr(record.__dict__)
    assert sentinel not in rendered
    assert repo.reservations[0].source_object_key not in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize("replay", (False, True))
async def test_upload_observes_queued_only_after_a_new_committed_reservation(
    replay: bool,
) -> None:
    kb_id = uuid4()
    repo = FakeRepository(preflight(kb_id))
    identifiers = (uuid4(), uuid4(), uuid4())
    repo.result = (
        UploadReservationResult.replayed(*identifiers)
        if replay
        else UploadReservationResult.created(*identifiers)
    )
    metrics = OperationalMetrics()
    handler = CollectingLogHandler()
    previous_handlers = list(ingestion_services.logger.handlers)
    previous_propagate = ingestion_services.logger.propagate
    previous_level = ingestion_services.logger.level
    ingestion_services.logger.handlers = [handler]
    ingestion_services.logger.propagate = False
    ingestion_services.logger.setLevel(logging.INFO)
    try:
        result = await DocumentUploadService(
            repository=repo,
            object_store=FakeObjectStore(),
            notifier=FakeNotifier(),
            max_upload_bytes=1024,
            metrics=metrics,
        ).upload(
            knowledge_base_id=kb_id,
            actor=actor(kb_id),
            source=SourceUpload(
                filename="queued.md",
                content_type="text/markdown",
                chunks=chunks(b"queued transition"),
            ),
            form=UploadForm.from_multipart(display_name=None, metadata=None, tags=None),
            idempotency_key="queued-transition",
            request_id="req-upload-queued",
        )
    finally:
        ingestion_services.logger.handlers = previous_handlers
        ingestion_services.logger.propagate = previous_propagate
        ingestion_services.logger.setLevel(previous_level)

    assert result.job_id == identifiers[2]
    assert metrics.registry.get_sample_value(
        "rag_job_state_transitions_total", {"state": "queued"}
    ) == (None if replay else 1)
    queued_records = [record for record in handler.records if record.msg == "job.state.changed"]
    assert len(queued_records) == (0 if replay else 1)
    if queued_records:
        record = queued_records[0]
        assert record.__dict__["request_id"] == "req-upload-queued"
        assert record.__dict__["knowledge_base_id"] == str(kb_id)
        assert record.__dict__["document_id"] == str(identifiers[0])
        assert record.__dict__["version_id"] == str(identifiers[1])
        assert record.__dict__["job_id"] == str(identifiers[2])
        assert record.__dict__["operation"] == "ingest_document"
        assert record.__dict__["state"] == "queued"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "expected_outcome"),
    [
        (BusinessError(422, "VALIDATION_ERROR", "raw-body-secret"), "rejected"),
        (RuntimeError("traceback-query-secret"), "failed"),
        (asyncio.CancelledError(), "cancelled"),
    ],
)
async def test_upload_observability_records_safe_terminal_failure_once(
    failure: BaseException,
    expected_outcome: str,
) -> None:
    kb_id = uuid4()
    repo = FakeRepository(preflight(kb_id))
    metrics = OperationalMetrics()

    class FailingStore(FakeObjectStore):
        async def upload_stream(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            raise failure

    handler = CollectingLogHandler()
    previous_handlers = list(ingestion_services.logger.handlers)
    previous_propagate = ingestion_services.logger.propagate
    previous_level = ingestion_services.logger.level
    ingestion_services.logger.handlers = [handler]
    ingestion_services.logger.propagate = False
    ingestion_services.logger.setLevel(logging.INFO)
    try:
        with pytest.raises(type(failure)):
            await DocumentUploadService(
                repository=repo,
                object_store=FailingStore(),
                notifier=FakeNotifier(),
                max_upload_bytes=1024,
                metrics=metrics,
            ).upload(
                knowledge_base_id=kb_id,
                actor=actor(kb_id),
                source=SourceUpload(
                    filename="doc.txt",
                    content_type="text/plain",
                    chunks=chunks(b"secret"),
                ),
                form=UploadForm.from_multipart(display_name=None, metadata=None, tags=None),
                idempotency_key=None,
                request_id="req-upload-failure",
            )
    finally:
        ingestion_services.logger.handlers = previous_handlers
        ingestion_services.logger.propagate = previous_propagate
        ingestion_services.logger.setLevel(previous_level)

    assert (
        metrics.registry.get_sample_value("rag_uploads_total", {"outcome": expected_outcome}) == 1
    )
    assert (
        metrics.registry.get_sample_value("rag_upload_bytes_total", {"outcome": expected_outcome})
        == 0
    )
    assert len(handler.records) == 1
    assert handler.records[0].msg == "upload.failed"
    rendered = repr(handler.records[0].__dict__)
    assert "raw-body-secret" not in rendered
    assert "traceback-query-secret" not in rendered


@pytest.mark.asyncio
async def test_multipart_upload_observability_counts_the_public_call_only_once() -> None:
    kb_id = uuid4()
    repo = FakeRepository(preflight(kb_id))
    repo.result = UploadReservationResult.created(uuid4(), uuid4(), uuid4())
    metrics = OperationalMetrics()
    boundary = "rag-observed-boundary"
    body = multipart_body(
        boundary,
        (("file", "doc.txt", "text/plain", b"hello multipart"),),
    )

    await DocumentUploadService(
        repository=repo,
        object_store=FakeObjectStore(),
        notifier=FakeNotifier(),
        max_upload_bytes=1024,
        metrics=metrics,
    ).upload_multipart(
        knowledge_base_id=kb_id,
        actor=actor(kb_id),
        body=sliced_body(body),
        content_type=f"multipart/form-data; boundary={boundary}",
        idempotency_key=None,
        request_id="req-multipart-observed",
    )

    assert metrics.registry.get_sample_value("rag_uploads_total", {"outcome": "succeeded"}) == 1
    assert metrics.registry.get_sample_value(
        "rag_upload_bytes_total", {"outcome": "succeeded"}
    ) == len(b"hello multipart")


@pytest.mark.asyncio
async def test_upload_observability_failure_does_not_change_the_business_result() -> None:
    kb_id = uuid4()
    repo = FakeRepository(preflight(kb_id))
    accepted = UploadReservationResult.created(uuid4(), uuid4(), uuid4())
    repo.result = accepted

    class FailingMetrics:
        def record_upload(self, **kwargs: object) -> None:
            del kwargs
            raise BaseException("metrics-secret")

        def record_job_state(self, **kwargs: object) -> None:
            del kwargs
            raise BaseException("metrics-secret")

    class FailingHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            del record
            raise BaseException("logging-secret")

    previous_handlers = list(ingestion_services.logger.handlers)
    previous_propagate = ingestion_services.logger.propagate
    previous_level = ingestion_services.logger.level
    ingestion_services.logger.handlers = [FailingHandler()]
    ingestion_services.logger.propagate = False
    ingestion_services.logger.setLevel(logging.INFO)
    try:
        result = await DocumentUploadService(
            repository=repo,
            object_store=FakeObjectStore(),
            notifier=FakeNotifier(),
            max_upload_bytes=1024,
            metrics=FailingMetrics(),  # type: ignore[arg-type]
        ).upload(
            knowledge_base_id=kb_id,
            actor=actor(kb_id),
            source=SourceUpload(
                filename="doc.txt",
                content_type="text/plain",
                chunks=chunks(b"hello"),
            ),
            form=UploadForm.from_multipart(display_name=None, metadata=None, tags=None),
            idempotency_key=None,
            request_id="req-observer-failure",
        )
    finally:
        ingestion_services.logger.handlers = previous_handlers
        ingestion_services.logger.propagate = previous_propagate
        ingestion_services.logger.setLevel(previous_level)

    assert result.job_id == accepted.job_id


@pytest.mark.asyncio
async def test_notification_failure_after_commit_does_not_change_upload_acceptance() -> None:
    kb_id = uuid4()

    class FailingNotifier:
        async def notify(self, job_id: object) -> bool:
            del job_id
            raise RuntimeError("redis-secret-must-not-escape")

    repo = FakeRepository(preflight(kb_id))
    store = FakeObjectStore()
    accepted = UploadReservationResult.created(uuid4(), uuid4(), uuid4())
    repo.result = accepted
    service = DocumentUploadService(
        repository=repo,
        object_store=store,
        notifier=FailingNotifier(),
        max_upload_bytes=1024,
    )

    result = await service.upload(
        knowledge_base_id=kb_id,
        actor=actor(kb_id),
        source=SourceUpload(
            filename="doc.txt",
            content_type="text/plain",
            chunks=chunks(b"hello"),
        ),
        form=UploadForm.from_multipart(display_name=None, metadata=None, tags=None),
        idempotency_key=None,
    )

    assert result.job_id == accepted.job_id
    assert len(repo.reservations) == 1
    assert list(store.objects) == [repo.reservations[0].source_object_key]


@pytest.mark.asyncio
async def test_notification_timeout_after_commit_does_not_block_acceptance() -> None:
    kb_id = uuid4()
    started = asyncio.Event()
    cancelled = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    class CancellationResistantNotifier:
        async def notify(self, job_id: object) -> bool:
            del job_id
            started.set()
            try:
                await asyncio.Event().wait()
                raise AssertionError("notification wait unexpectedly completed")
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()
            finally:
                finished.set()
            return True

    repo = FakeRepository(preflight(kb_id))
    store = FakeObjectStore()
    accepted = UploadReservationResult.created(uuid4(), uuid4(), uuid4())
    repo.result = accepted
    service = DocumentUploadService(
        repository=repo,
        object_store=store,
        notifier=CancellationResistantNotifier(),
        max_upload_bytes=1024,
        notification_timeout_seconds=0.01,
    )

    upload_task = asyncio.create_task(
        service.upload(
            knowledge_base_id=kb_id,
            actor=actor(kb_id),
            source=SourceUpload(
                filename="doc.txt",
                content_type="text/plain",
                chunks=chunks(b"hello"),
            ),
            form=UploadForm.from_multipart(display_name=None, metadata=None, tags=None),
            idempotency_key=None,
        )
    )
    await asyncio.wait_for(started.wait(), timeout=0.2)
    await asyncio.wait_for(cancelled.wait(), timeout=0.2)

    done, _pending = await asyncio.wait({upload_task}, timeout=0.1)
    returned_before_release = upload_task in done
    release.set()
    await asyncio.wait_for(finished.wait(), timeout=0.2)
    result = await asyncio.wait_for(upload_task, timeout=0.2)

    assert returned_before_release is True
    assert result.job_id == accepted.job_id
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_late_notification_failure_is_consumed_without_raw_loop_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    kb_id = uuid4()
    secret = "late-notification-secret-must-not-escape"
    started = asyncio.Event()
    cancelled = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    class LateFailingNotifier:
        async def notify(self, job_id: object) -> bool:
            del job_id
            started.set()
            try:
                await asyncio.Event().wait()
                raise AssertionError("notification wait unexpectedly completed")
            except asyncio.CancelledError:
                cancelled.set()
                await release.wait()
                raise RuntimeError(secret) from None
            finally:
                finished.set()

    repo = FakeRepository(preflight(kb_id))
    store = FakeObjectStore()
    accepted = UploadReservationResult.created(uuid4(), uuid4(), uuid4())
    repo.result = accepted
    service = DocumentUploadService(
        repository=repo,
        object_store=store,
        notifier=LateFailingNotifier(),
        max_upload_bytes=1024,
        notification_timeout_seconds=0.01,
    )
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    unhandled: list[dict[str, object]] = []
    loop.set_exception_handler(lambda _loop, context: unhandled.append(dict(context)))
    upload_task: asyncio.Task[object] | None = None
    try:
        with caplog.at_level(logging.WARNING, logger="rag_service.ingestion.services"):
            upload_task = asyncio.create_task(
                service.upload(
                    knowledge_base_id=kb_id,
                    actor=actor(kb_id),
                    source=SourceUpload(
                        filename="doc.txt",
                        content_type="text/plain",
                        chunks=chunks(b"hello"),
                    ),
                    form=UploadForm.from_multipart(
                        display_name=None,
                        metadata=None,
                        tags=None,
                    ),
                    idempotency_key=None,
                )
            )
            await asyncio.wait_for(started.wait(), timeout=0.2)
            await asyncio.wait_for(cancelled.wait(), timeout=0.2)
            done, _pending = await asyncio.wait({upload_task}, timeout=0.1)
            assert upload_task in done
            result = await upload_task
            assert result.job_id == accepted.job_id

            release.set()
            await asyncio.wait_for(finished.wait(), timeout=0.2)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
    finally:
        release.set()
        loop.set_exception_handler(previous_handler)
        if upload_task is not None and not upload_task.done():
            upload_task.cancel()
            await asyncio.gather(upload_task, return_exceptions=True)

    assert unhandled == []
    assert secret not in caplog.text


@pytest.mark.asyncio
async def test_request_cancellation_during_notification_wait_is_not_swallowed() -> None:
    kb_id = uuid4()
    started = asyncio.Event()
    finished = asyncio.Event()

    class BlockingNotifier:
        async def notify(self, job_id: object) -> bool:
            del job_id
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                finished.set()
            return True

    repo = FakeRepository(preflight(kb_id))
    store = FakeObjectStore()
    repo.result = UploadReservationResult.created(uuid4(), uuid4(), uuid4())
    service = DocumentUploadService(
        repository=repo,
        object_store=store,
        notifier=BlockingNotifier(),
        max_upload_bytes=1024,
        notification_timeout_seconds=1,
    )
    upload_task = asyncio.create_task(
        service.upload(
            knowledge_base_id=kb_id,
            actor=actor(kb_id),
            source=SourceUpload(
                filename="doc.txt",
                content_type="text/plain",
                chunks=chunks(b"hello"),
            ),
            form=UploadForm.from_multipart(display_name=None, metadata=None, tags=None),
            idempotency_key=None,
        )
    )
    await asyncio.wait_for(started.wait(), timeout=0.2)

    upload_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await upload_task
    await asyncio.wait_for(finished.wait(), timeout=0.2)


@pytest.mark.parametrize(
    "invalid_timeout",
    [True, 0, float("nan"), float("inf")],
)
def test_service_rejects_invalid_notification_timeout(invalid_timeout: object) -> None:
    kb_id = uuid4()
    with pytest.raises(ValueError, match="notification timeout"):
        DocumentUploadService(
            repository=FakeRepository(preflight(kb_id)),
            object_store=FakeObjectStore(),
            notifier=FakeNotifier(),
            max_upload_bytes=1024,
            notification_timeout_seconds=invalid_timeout,  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_notification_cancellation_after_commit_is_not_swallowed() -> None:
    kb_id = uuid4()

    class CancelledNotifier:
        async def notify(self, job_id: object) -> bool:
            del job_id
            raise __import__("asyncio").CancelledError

    repo = FakeRepository(preflight(kb_id))
    store = FakeObjectStore()
    repo.result = UploadReservationResult.created(uuid4(), uuid4(), uuid4())
    service = DocumentUploadService(
        repository=repo,
        object_store=store,
        notifier=CancelledNotifier(),
        max_upload_bytes=1024,
    )

    with pytest.raises(__import__("asyncio").CancelledError):
        await service.upload(
            knowledge_base_id=kb_id,
            actor=actor(kb_id),
            source=SourceUpload(
                filename="doc.txt",
                content_type="text/plain",
                chunks=chunks(b"hello"),
            ),
            form=UploadForm.from_multipart(display_name=None, metadata=None, tags=None),
            idempotency_key=None,
        )

    assert len(repo.reservations) == 1
    assert list(store.objects) == [repo.reservations[0].source_object_key]


@pytest.mark.asyncio
async def test_scope_is_hidden_before_capability_is_disclosed() -> None:
    kb_id = uuid4()
    principal = replace(
        actor(uuid4()),
        capabilities=frozenset({Capability.RETRIEVE}),
    )
    repo = FakeRepository(preflight(kb_id))
    service = DocumentUploadService(
        repository=repo,
        object_store=FakeObjectStore(),
        notifier=FakeNotifier(),
        max_upload_bytes=1024,
    )

    with pytest.raises(BusinessError) as captured:
        await service.upload(
            knowledge_base_id=kb_id,
            actor=principal,
            source=SourceUpload(
                filename="doc.txt",
                content_type="text/plain",
                chunks=chunks(b"hello"),
            ),
            form=UploadForm.from_multipart(display_name=None, metadata=None, tags=None),
            idempotency_key=None,
        )

    assert captured.value.code == "RESOURCE_NOT_FOUND"


@pytest.mark.asyncio
async def test_streaming_multipart_accepts_fields_after_file_without_request_spooling() -> None:
    kb_id = uuid4()
    repo = FakeRepository(preflight(kb_id))
    repo.result = UploadReservationResult.created(uuid4(), uuid4(), uuid4())
    store = FakeObjectStore()
    service = DocumentUploadService(
        repository=repo,
        object_store=store,
        notifier=FakeNotifier(),
        max_upload_bytes=1024,
    )
    boundary = "rag-upload-boundary"
    body = multipart_body(
        boundary,
        (
            ("file", "guide.md", "text/markdown", b"# guide\nhello"),
            ("display_name", None, None, b"Guide"),
            ("tags", None, None, b'["docs"]'),
            ("metadata", None, None, b'{"attributes":{"department":"docs"}}'),
        ),
    )

    result = await service.upload_multipart(
        knowledge_base_id=kb_id,
        actor=actor(kb_id),
        body=sliced_body(body),
        content_type=f"multipart/form-data; boundary={boundary}",
        idempotency_key="streamed",
    )

    assert result.status == "queued"
    assert len(store.publish_pairs) == 1
    assert repo.reservations[0].display_name == "Guide"
    assert repo.reservations[0].tags == ("docs",)


@pytest.mark.asyncio
async def test_streaming_multipart_enforces_actual_file_bytes_at_exact_limit() -> None:
    kb_id = uuid4()
    boundary = "rag-size-boundary"
    for payload, expected_code in ((b"1234", None), (b"12345", "FILE_TOO_LARGE")):
        repo = FakeRepository(preflight(kb_id))
        repo.result = UploadReservationResult.created(uuid4(), uuid4(), uuid4())
        store = FakeObjectStore()
        service = DocumentUploadService(
            repository=repo,
            object_store=store,
            notifier=FakeNotifier(),
            max_upload_bytes=4,
        )
        body = multipart_body(
            boundary,
            (("file", "doc.txt", "text/plain", payload),),
        )
        if expected_code is None:
            await service.upload_multipart(
                knowledge_base_id=kb_id,
                actor=actor(kb_id),
                body=sliced_body(body, 3),
                content_type=f"multipart/form-data; boundary={boundary}",
                idempotency_key=None,
            )
            assert store.objects
        else:
            with pytest.raises(BusinessError) as captured:
                await service.upload_multipart(
                    knowledge_base_id=kb_id,
                    actor=actor(kb_id),
                    body=sliced_body(body, 3),
                    content_type=f"multipart/form-data; boundary={boundary}",
                    idempotency_key=None,
                )
            assert captured.value.code == expected_code
            assert store.objects == {}
            assert repo.reservations == []


@pytest.mark.asyncio
async def test_streaming_multipart_rejects_duplicate_unknown_and_malformed_parts() -> None:
    kb_id = uuid4()
    boundary = "rag-invalid-boundary"
    invalid_parts = (
        (
            ("file", "doc.txt", "text/plain", b"hello"),
            ("file", "other.txt", "text/plain", b"other"),
        ),
        (
            ("file", "doc.txt", "text/plain", b"hello"),
            ("unexpected", None, None, b"value"),
        ),
    )
    for parts in invalid_parts:
        repo = FakeRepository(preflight(kb_id))
        store = FakeObjectStore()
        service = DocumentUploadService(
            repository=repo,
            object_store=store,
            notifier=FakeNotifier(),
            max_upload_bytes=1024,
        )
        with pytest.raises(BusinessError) as captured:
            await service.upload_multipart(
                knowledge_base_id=kb_id,
                actor=actor(kb_id),
                body=sliced_body(multipart_body(boundary, parts), 5),
                content_type=f"multipart/form-data; boundary={boundary}",
                idempotency_key=None,
            )
        assert captured.value.code == "VALIDATION_ERROR"
        assert store.objects == {}
        assert repo.reservations == []


@pytest.mark.asyncio
async def test_service_cleans_final_for_known_replay_and_domain_conflict() -> None:
    kb_id = uuid4()
    for outcome in (
        UploadReservationResult.replayed(uuid4(), uuid4(), uuid4()),
        BusinessError(409, "IDEMPOTENCY_KEY_REUSED", "Idempotency key reused"),
    ):
        repo = FakeRepository(preflight(kb_id))
        repo.result = outcome
        store = FakeObjectStore()
        service = DocumentUploadService(
            repository=repo,
            object_store=store,
            notifier=FakeNotifier(),
            max_upload_bytes=1024,
        )
        source = SourceUpload(
            filename="doc.txt",
            content_type="text/plain",
            chunks=chunks(b"hello"),
        )
        with suppress(BusinessError, RuntimeError):
            await service.upload(
                knowledge_base_id=kb_id,
                actor=actor(kb_id),
                source=source,
                form=UploadForm.from_multipart(display_name=None, metadata=None, tags=None),
                idempotency_key="key",
            )
        assert len(store.deleted) == 1
        assert store.objects == {}


@pytest.mark.asyncio
async def test_unclassified_commit_failure_preserves_final_for_reconciliation(
    caplog: pytest.LogCaptureFixture,
) -> None:
    kb_id = uuid4()
    repo = FakeRepository(preflight(kb_id))
    secret = "connection lost for secret=database-secret"
    repo.result = RuntimeError(secret)
    store = FakeObjectStore()
    service = DocumentUploadService(
        repository=repo,
        object_store=store,
        notifier=FakeNotifier(),
        max_upload_bytes=1024,
    )

    with (
        caplog.at_level(logging.ERROR, logger="rag_service.ingestion.services"),
        pytest.raises(BusinessError) as captured,
    ):
        await service.upload(
            knowledge_base_id=kb_id,
            actor=actor(kb_id),
            source=SourceUpload(
                filename="doc.txt",
                content_type="text/plain",
                chunks=chunks(b"hello"),
            ),
            form=UploadForm.from_multipart(display_name=None, metadata=None, tags=None),
            idempotency_key="key",
        )

    assert captured.value.code == "INGESTION_RESERVATION_UNCERTAIN"
    assert captured.value.retryable is True
    assert secret not in str(captured.value)
    assert store.deleted == []
    final_key = repo.reservations[0].source_object_key
    assert final_key not in str(captured.value)
    assert list(store.objects) == [final_key]
    assert "Upload reservation failed unexpectedly" in caplog.text
    assert secret not in caplog.text
    assert final_key not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_point", ["write", "finalize"])
async def test_multipart_parser_structural_errors_are_sanitized_and_cleaned(
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    secret = "multipart-parser-secret-must-not-escape"

    class StructurallyInvalidParser:
        def __init__(self, boundary: bytes, callbacks: object) -> None:
            del boundary, callbacks

        def write(self, piece: bytes) -> None:
            del piece
            if failure_point == "write":
                raise MultipartParseError(secret)

        def finalize(self) -> None:
            if failure_point == "finalize":
                raise MultipartParseError(secret)

    monkeypatch.setattr(ingestion_services, "MultipartParser", StructurallyInvalidParser)
    store = FakeObjectStore()

    with pytest.raises(BusinessError) as captured:
        await stage_multipart_upload(
            body=chunks(b"malformed multipart"),
            content_type="multipart/form-data; boundary=boundary",
            object_store=store,
            max_file_bytes=1024,
        )

    assert captured.value.code == "VALIDATION_ERROR"
    assert str(captured.value) == "Invalid document upload"
    assert secret not in str(captured.value)
    assert len(store.deleted) == 1
    assert store.objects == {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raised", "expected_type"),
    [
        (BusinessError(409, "DOMAIN_ERROR", "safe domain error"), BusinessError),
        (asyncio.CancelledError(), asyncio.CancelledError),
    ],
)
async def test_multipart_parser_does_not_swallow_domain_or_cancellation_errors(
    monkeypatch: pytest.MonkeyPatch,
    raised: BaseException,
    expected_type: type[BaseException],
) -> None:
    class RaisingParser:
        def __init__(self, boundary: bytes, callbacks: object) -> None:
            del boundary, callbacks

        def write(self, piece: bytes) -> None:
            del piece
            raise raised

        def finalize(self) -> None:
            raise AssertionError

    monkeypatch.setattr(ingestion_services, "MultipartParser", RaisingParser)
    store = FakeObjectStore()

    with pytest.raises(expected_type) as captured:
        await stage_multipart_upload(
            body=chunks(b"multipart"),
            content_type="multipart/form-data; boundary=boundary",
            object_store=store,
            max_file_bytes=1024,
        )

    assert captured.value is raised
    assert len(store.deleted) == 1


@pytest.mark.asyncio
async def test_service_rejects_scope_capability_generation_and_invalid_content_before_db() -> None:
    kb_id = uuid4()
    cases = [
        (actor(uuid4()), preflight(kb_id), b"hello", "RESOURCE_NOT_FOUND"),
        (
            replace(actor(kb_id), capabilities=frozenset({Capability.RETRIEVE})),
            preflight(kb_id),
            b"hello",
            "INSUFFICIENT_CAPABILITY",
        ),
        (actor(kb_id), None, b"hello", "KNOWLEDGE_BASE_NOT_INDEX_CONFIGURED"),
        (actor(kb_id), preflight(kb_id), b"\x00binary", "BINARY_CONTENT_REJECTED"),
    ]
    for principal, found, content, code in cases:
        repo = FakeRepository(found)  # type: ignore[arg-type]
        store = FakeObjectStore()
        service = DocumentUploadService(
            repository=repo,
            object_store=store,
            notifier=FakeNotifier(),
            max_upload_bytes=1024,
        )
        with pytest.raises(BusinessError) as captured:
            await service.upload(
                knowledge_base_id=kb_id,
                actor=principal,
                source=SourceUpload(
                    filename="doc.txt",
                    content_type="text/plain",
                    chunks=chunks(content),
                ),
                form=UploadForm.from_multipart(display_name=None, metadata=None, tags=None),
                idempotency_key=None,
            )
        assert captured.value.code == code
        assert repo.reservations == []
        assert store.objects == {}


@pytest.mark.asyncio
async def test_file_too_large_aborts_before_final_object_publication() -> None:
    kb_id = uuid4()
    repo = FakeRepository(preflight(kb_id))
    store = FakeObjectStore()
    service = DocumentUploadService(
        repository=repo,
        object_store=store,
        notifier=FakeNotifier(),
        max_upload_bytes=4,
    )

    with pytest.raises(BusinessError) as captured:
        await service.upload(
            knowledge_base_id=kb_id,
            actor=actor(kb_id),
            source=SourceUpload(
                filename="doc.txt",
                content_type="text/plain",
                chunks=chunks(b"1234", b"5"),
            ),
            form=UploadForm.from_multipart(display_name=None, metadata=None, tags=None),
            idempotency_key=None,
        )

    assert captured.value.code == "FILE_TOO_LARGE"
    assert store.published == []
    assert repo.reservations == []


@pytest.mark.asyncio
async def test_ambiguous_commit_cancellation_preserves_final_for_reconciliation() -> None:
    kb_id = uuid4()
    entered = __import__("asyncio").Event()

    class BlockingRepository(FakeRepository):
        async def reserve(self, reservation: UploadReservation) -> UploadReservationResult:
            self.reservations.append(reservation)
            entered.set()
            await __import__("asyncio").Event().wait()
            raise AssertionError

    class CancellationSensitiveStore(FakeObjectStore):
        async def delete_best_effort(self, object_key: str) -> bool:
            task = __import__("asyncio").current_task()
            assert task is not None
            if task.cancelling():
                raise __import__("asyncio").CancelledError
            return await super().delete_best_effort(object_key)

    repo = BlockingRepository(preflight(kb_id))
    store = CancellationSensitiveStore()
    service = DocumentUploadService(
        repository=repo,
        object_store=store,
        notifier=FakeNotifier(),
        max_upload_bytes=1024,
        commit_resolution_timeout_seconds=0.01,
    )
    task = __import__("asyncio").create_task(
        service.upload(
            knowledge_base_id=kb_id,
            actor=actor(kb_id),
            source=SourceUpload(
                filename="doc.txt",
                content_type="text/plain",
                chunks=chunks(b"hello"),
            ),
            form=UploadForm.from_multipart(display_name=None, metadata=None, tags=None),
            idempotency_key=None,
        )
    )
    await entered.wait()
    task.cancel()
    with pytest.raises(__import__("asyncio").CancelledError):
        await task

    assert store.deleted == []
    assert list(store.objects) == [repo.reservations[0].source_object_key]


@pytest.mark.asyncio
async def test_cancellation_after_known_success_never_deletes_referenced_final() -> None:
    kb_id = uuid4()
    entered = __import__("asyncio").Event()
    release = __import__("asyncio").Event()

    class CommitAfterCancellationRepository(FakeRepository):
        async def reserve(self, reservation: UploadReservation) -> UploadReservationResult:
            self.reservations.append(reservation)
            entered.set()
            await release.wait()
            return UploadReservationResult.created(
                reservation.document_id,
                reservation.version_id,
                reservation.job_id,
            )

    repo = CommitAfterCancellationRepository(preflight(kb_id))
    store = FakeObjectStore()
    service = DocumentUploadService(
        repository=repo,
        object_store=store,
        notifier=FakeNotifier(),
        max_upload_bytes=1024,
        commit_resolution_timeout_seconds=1,
    )
    task = __import__("asyncio").create_task(
        service.upload(
            knowledge_base_id=kb_id,
            actor=actor(kb_id),
            source=SourceUpload(
                filename="doc.txt",
                content_type="text/plain",
                chunks=chunks(b"hello"),
            ),
            form=UploadForm.from_multipart(display_name=None, metadata=None, tags=None),
            idempotency_key=None,
        )
    )
    await entered.wait()
    task.cancel()
    release.set()
    with pytest.raises(__import__("asyncio").CancelledError):
        await task

    assert store.deleted == []
    assert list(store.objects) == [repo.reservations[0].source_object_key]


@pytest.mark.asyncio
@pytest.mark.parametrize("replay", (False, True))
async def test_known_reservation_result_observes_created_queued_before_propagating_cancellation(
    replay: bool,
) -> None:
    kb_id = uuid4()
    entered = asyncio.Event()
    release = asyncio.Event()

    class CommitAfterCancellationRepository(FakeRepository):
        async def reserve(self, reservation: UploadReservation) -> UploadReservationResult:
            self.reservations.append(reservation)
            entered.set()
            await release.wait()
            factory = (
                UploadReservationResult.replayed if replay else UploadReservationResult.created
            )
            return factory(
                reservation.document_id,
                reservation.version_id,
                reservation.job_id,
            )

    repo = CommitAfterCancellationRepository(preflight(kb_id))
    metrics = OperationalMetrics()
    notifier = FakeNotifier()
    handler = CollectingLogHandler()
    previous_handlers = list(ingestion_services.logger.handlers)
    previous_propagate = ingestion_services.logger.propagate
    previous_level = ingestion_services.logger.level
    ingestion_services.logger.handlers = [handler]
    ingestion_services.logger.propagate = False
    ingestion_services.logger.setLevel(logging.INFO)
    try:
        task = asyncio.create_task(
            DocumentUploadService(
                repository=repo,
                object_store=FakeObjectStore(),
                notifier=notifier,
                max_upload_bytes=1024,
                commit_resolution_timeout_seconds=1,
                metrics=metrics,
            ).upload(
                knowledge_base_id=kb_id,
                actor=actor(kb_id),
                source=SourceUpload(
                    filename="cancelled.txt",
                    content_type="text/plain",
                    chunks=chunks(b"hello"),
                ),
                form=UploadForm.from_multipart(
                    display_name=None,
                    metadata=None,
                    tags=None,
                ),
                idempotency_key=None,
                request_id="req-known-reservation-cancelled",
            )
        )
        await entered.wait()
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        ingestion_services.logger.handlers = previous_handlers
        ingestion_services.logger.propagate = previous_propagate
        ingestion_services.logger.setLevel(previous_level)

    assert metrics.registry.get_sample_value(
        "rag_job_state_transitions_total", {"state": "queued"}
    ) == (None if replay else 1)
    queued_records = [record for record in handler.records if record.msg == "job.state.changed"]
    assert len(queued_records) == (0 if replay else 1)
    if queued_records:
        assert queued_records[0].__dict__["operation"] == "ingest_document"
    assert notifier.jobs == []
    assert metrics.registry.get_sample_value("rag_uploads_total", {"outcome": "cancelled"}) == 1
    for outcome in ("succeeded", "rejected", "failed"):
        assert metrics.registry.get_sample_value("rag_uploads_total", {"outcome": outcome}) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("replay", (False, True))
async def test_late_reservation_result_observes_created_queued_after_cancelled_caller(
    replay: bool,
) -> None:
    kb_id = uuid4()
    entered = asyncio.Event()
    release = asyncio.Event()
    reservation_tasks: list[asyncio.Task[object]] = []

    class LateCommitRepository(FakeRepository):
        async def reserve(self, reservation: UploadReservation) -> UploadReservationResult:
            self.reservations.append(reservation)
            task = asyncio.current_task()
            assert task is not None
            reservation_tasks.append(task)
            entered.set()
            await release.wait()
            factory = (
                UploadReservationResult.replayed if replay else UploadReservationResult.created
            )
            return factory(
                reservation.document_id,
                reservation.version_id,
                reservation.job_id,
            )

    repo = LateCommitRepository(preflight(kb_id))
    metrics = OperationalMetrics()
    notifier = FakeNotifier()
    handler = CollectingLogHandler()
    previous_handlers = list(ingestion_services.logger.handlers)
    previous_propagate = ingestion_services.logger.propagate
    previous_level = ingestion_services.logger.level
    ingestion_services.logger.handlers = [handler]
    ingestion_services.logger.propagate = False
    ingestion_services.logger.setLevel(logging.INFO)
    try:
        upload_task = asyncio.create_task(
            DocumentUploadService(
                repository=repo,
                object_store=FakeObjectStore(),
                notifier=notifier,
                max_upload_bytes=1024,
                commit_resolution_timeout_seconds=0.01,
                metrics=metrics,
            ).upload(
                knowledge_base_id=kb_id,
                actor=actor(kb_id),
                source=SourceUpload(
                    filename="late-cancelled.txt",
                    content_type="text/plain",
                    chunks=chunks(b"hello"),
                ),
                form=UploadForm.from_multipart(
                    display_name=None,
                    metadata=None,
                    tags=None,
                ),
                idempotency_key=None,
                request_id="req-late-reservation-cancelled",
            )
        )
        await entered.wait()
        upload_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await upload_task

        assert len(reservation_tasks) == 1
        assert not reservation_tasks[0].done()
        assert metrics.registry.get_sample_value("rag_uploads_total", {"outcome": "cancelled"}) == 1

        release.set()
        await reservation_tasks[0]
        await asyncio.sleep(0)
    finally:
        ingestion_services.logger.handlers = previous_handlers
        ingestion_services.logger.propagate = previous_propagate
        ingestion_services.logger.setLevel(previous_level)

    assert metrics.registry.get_sample_value(
        "rag_job_state_transitions_total", {"state": "queued"}
    ) == (None if replay else 1)
    queued_records = [record for record in handler.records if record.msg == "job.state.changed"]
    assert len(queued_records) == (0 if replay else 1)
    if queued_records:
        record = queued_records[0]
        reservation = repo.reservations[0]
        assert record.__dict__["request_id"] == "req-late-reservation-cancelled"
        assert record.__dict__["knowledge_base_id"] == str(kb_id)
        assert record.__dict__["document_id"] == str(reservation.document_id)
        assert record.__dict__["version_id"] == str(reservation.version_id)
        assert record.__dict__["job_id"] == str(reservation.job_id)
        assert record.__dict__["operation"] == "ingest_document"
        assert record.__dict__["state"] == "queued"
    assert notifier.jobs == []
    assert metrics.registry.get_sample_value("rag_uploads_total", {"outcome": "cancelled"}) == 1
    for outcome in ("succeeded", "rejected", "failed"):
        assert metrics.registry.get_sample_value("rag_uploads_total", {"outcome": outcome}) is None
    terminal_records = [
        record for record in handler.records if record.msg in {"upload.completed", "upload.failed"}
    ]
    assert len(terminal_records) == 1
    assert terminal_records[0].__dict__["outcome"] == "cancelled"


@pytest.mark.asyncio
@pytest.mark.parametrize("completion_first", (True, False))
@pytest.mark.parametrize("replay", (False, True))
async def test_double_cancellation_completion_race_observes_late_created_once(
    completion_first: bool,
    replay: bool,
) -> None:
    kb_id = uuid4()
    entered = asyncio.Event()
    release = asyncio.Event()
    committed = asyncio.Event()
    reservation_tasks: list[asyncio.Task[object]] = []

    class RacingRepository(FakeRepository):
        async def reserve(self, reservation: UploadReservation) -> UploadReservationResult:
            self.reservations.append(reservation)
            task = asyncio.current_task()
            assert task is not None
            reservation_tasks.append(task)
            entered.set()
            await release.wait()
            committed.set()
            factory = (
                UploadReservationResult.replayed if replay else UploadReservationResult.created
            )
            return factory(
                reservation.document_id,
                reservation.version_id,
                reservation.job_id,
            )

    repo = RacingRepository(preflight(kb_id))
    metrics = OperationalMetrics()
    notifier = FakeNotifier()
    handler = CollectingLogHandler()
    loop = asyncio.get_running_loop()
    previous_exception_handler = loop.get_exception_handler()
    unhandled: list[dict[str, object]] = []
    previous_handlers = list(ingestion_services.logger.handlers)
    previous_propagate = ingestion_services.logger.propagate
    previous_level = ingestion_services.logger.level
    loop.set_exception_handler(lambda _loop, context: unhandled.append(dict(context)))
    ingestion_services.logger.handlers = [handler]
    ingestion_services.logger.propagate = False
    ingestion_services.logger.setLevel(logging.INFO)
    upload_task: asyncio.Task[object] | None = None
    try:
        upload_task = asyncio.create_task(
            DocumentUploadService(
                repository=repo,
                object_store=FakeObjectStore(),
                notifier=notifier,
                max_upload_bytes=1024,
                commit_resolution_timeout_seconds=1,
                metrics=metrics,
            ).upload(
                knowledge_base_id=kb_id,
                actor=actor(kb_id),
                source=SourceUpload(
                    filename="racing-cancelled.txt",
                    content_type="text/plain",
                    chunks=chunks(b"hello"),
                ),
                form=UploadForm.from_multipart(
                    display_name=None,
                    metadata=None,
                    tags=None,
                ),
                idempotency_key=None,
                request_id="req-double-cancellation-race",
            )
        )
        await entered.wait()
        upload_task.cancel()
        await asyncio.sleep(0)

        callbacks = (release.set, upload_task.cancel)
        if not completion_first:
            callbacks = tuple(reversed(callbacks))
        for callback in callbacks:
            loop.call_soon(callback)

        with pytest.raises(asyncio.CancelledError):
            await upload_task
        assert len(reservation_tasks) == 1
        await reservation_tasks[0]
        await asyncio.sleep(0)
    finally:
        release.set()
        loop.set_exception_handler(previous_exception_handler)
        ingestion_services.logger.handlers = previous_handlers
        ingestion_services.logger.propagate = previous_propagate
        ingestion_services.logger.setLevel(previous_level)
        if upload_task is not None and not upload_task.done():
            upload_task.cancel()
            await asyncio.gather(upload_task, return_exceptions=True)

    assert committed.is_set()
    assert metrics.registry.get_sample_value(
        "rag_job_state_transitions_total", {"state": "queued"}
    ) == (None if replay else 1)
    queued_records = [record for record in handler.records if record.msg == "job.state.changed"]
    assert len(queued_records) == (0 if replay else 1)
    if queued_records:
        record = queued_records[0]
        reservation = repo.reservations[0]
        assert record.__dict__["request_id"] == "req-double-cancellation-race"
        assert record.__dict__["knowledge_base_id"] == str(kb_id)
        assert record.__dict__["document_id"] == str(reservation.document_id)
        assert record.__dict__["version_id"] == str(reservation.version_id)
        assert record.__dict__["job_id"] == str(reservation.job_id)
        assert record.__dict__["operation"] == "ingest_document"
        assert record.__dict__["state"] == "queued"
    assert notifier.jobs == []
    assert unhandled == []


@pytest.mark.asyncio
async def test_double_cancellation_consumes_late_reservation_exception_without_secret(
    caplog: pytest.LogCaptureFixture,
) -> None:
    kb_id = uuid4()
    secret = "late-reservation-database-secret"
    entered = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    class LateFailingRepository(FakeRepository):
        async def reserve(self, reservation: UploadReservation) -> UploadReservationResult:
            self.reservations.append(reservation)
            entered.set()
            await release.wait()
            try:
                raise RuntimeError(secret)
            finally:
                finished.set()

    repo = LateFailingRepository(preflight(kb_id))
    store = FakeObjectStore()
    service = DocumentUploadService(
        repository=repo,
        object_store=store,
        notifier=FakeNotifier(),
        max_upload_bytes=1024,
        commit_resolution_timeout_seconds=1,
    )
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    unhandled: list[dict[str, object]] = []
    loop.set_exception_handler(lambda _loop, context: unhandled.append(dict(context)))
    upload_task: asyncio.Task[object] | None = None
    try:
        with caplog.at_level(logging.ERROR, logger="rag_service.ingestion.services"):
            upload_task = asyncio.create_task(
                service.upload(
                    knowledge_base_id=kb_id,
                    actor=actor(kb_id),
                    source=SourceUpload(
                        filename="doc.txt",
                        content_type="text/plain",
                        chunks=chunks(b"hello"),
                    ),
                    form=UploadForm.from_multipart(
                        display_name=None,
                        metadata=None,
                        tags=None,
                    ),
                    idempotency_key=None,
                )
            )
            await asyncio.wait_for(entered.wait(), timeout=0.2)
            upload_task.cancel()
            await asyncio.sleep(0)
            upload_task.cancel()
            done, _pending = await asyncio.wait({upload_task}, timeout=0.2)
            assert upload_task in done
            with pytest.raises(asyncio.CancelledError):
                await upload_task

            release.set()
            await asyncio.wait_for(finished.wait(), timeout=0.2)
            await asyncio.sleep(0)
            gc.collect()
            await asyncio.sleep(0)
    finally:
        release.set()
        loop.set_exception_handler(previous_handler)
        if upload_task is not None and not upload_task.done():
            upload_task.cancel()
            await asyncio.gather(upload_task, return_exceptions=True)

    assert unhandled == []
    assert "Detached upload reservation failed unexpectedly" in caplog.text
    assert secret not in caplog.text


@pytest.mark.asyncio
async def test_detached_reservation_owns_session_after_request_teardown() -> None:
    kb_id = uuid4()
    entered = asyncio.Event()
    release = asyncio.Event()
    reservation_closed = asyncio.Event()
    events: list[str] = []

    class Transaction:
        def __init__(self, name: str) -> None:
            self.name = name

        async def __aenter__(self) -> None:
            events.append(f"{self.name}:transaction-enter")

        async def __aexit__(
            self,
            error_type: type[BaseException] | None,
            error: BaseException | None,
            traceback: object,
        ) -> None:
            del error, traceback
            events.append(f"{self.name}:transaction-exit:{error_type is None}")

    class RequestSession:
        async def __aenter__(self) -> "RequestSession":
            events.append("request:session-enter")
            return self

        async def __aexit__(
            self,
            error_type: type[BaseException] | None,
            error: BaseException | None,
            traceback: object,
        ) -> None:
            del error_type, error, traceback
            events.append("request:session-exit")

        def begin(self) -> Transaction:
            return Transaction("request")

    class ReservationSession:
        async def __aenter__(self) -> "ReservationSession":
            events.append("reservation:session-enter")
            return self

        async def __aexit__(
            self,
            error_type: type[BaseException] | None,
            error: BaseException | None,
            traceback: object,
        ) -> None:
            del error, traceback
            events.append(f"reservation:session-exit:{error_type is None}")
            reservation_closed.set()

        def begin(self) -> Transaction:
            return Transaction("reservation")

    preflight_repository = FakeRepository(preflight(kb_id))

    class BlockingReservationRepository:
        async def preflight(self, knowledge_base_id: UUID) -> UploadPreflight | None:
            del knowledge_base_id
            raise AssertionError("reservation repository must not preflight")

        async def reserve(self, reservation: UploadReservation) -> UploadReservationResult:
            events.append("reservation:reserve")
            entered.set()
            await release.wait()
            return UploadReservationResult.created(
                reservation.document_id,
                reservation.version_id,
                reservation.job_id,
            )

    def reservation_repository_factory(session: object) -> BlockingReservationRepository:
        assert isinstance(session, ReservationSession)
        return BlockingReservationRepository()

    store = FakeObjectStore()
    async with RequestSession() as request_session:
        service = DocumentUploadService(
            repository=preflight_repository,
            object_store=store,
            notifier=FakeNotifier(),
            max_upload_bytes=1024,
            session=request_session,  # type: ignore[arg-type]
            reservation_session_factory=ReservationSession,  # type: ignore[arg-type]
            reservation_repository_factory=reservation_repository_factory,
            commit_resolution_timeout_seconds=0.01,
        )
        upload_task = asyncio.create_task(
            service.upload(
                knowledge_base_id=kb_id,
                actor=actor(kb_id),
                source=SourceUpload(
                    filename="doc.txt",
                    content_type="text/plain",
                    chunks=chunks(b"hello"),
                ),
                form=UploadForm.from_multipart(display_name=None, metadata=None, tags=None),
                idempotency_key=None,
            )
        )
        await entered.wait()
        upload_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await upload_task

    assert events[-1] == "request:session-exit"
    release.set()
    await asyncio.wait_for(reservation_closed.wait(), timeout=0.25)
    assert events[-2:] == [
        "reservation:transaction-exit:True",
        "reservation:session-exit:True",
    ]
    assert events.index("request:session-exit") < events.index("reservation:transaction-exit:True")
    assert preflight_repository.reservations == []


@pytest.mark.asyncio
async def test_failure_after_final_copy_still_cleans_the_deterministic_key() -> None:
    kb_id = uuid4()

    class CopyThenFailStore(FakeObjectStore):
        async def publish_temp(
            self,
            temp_key: str,
            final_key: str,
            *,
            expected_size: int,
            expected_checksum: str,
        ) -> object:
            await super().publish_temp(
                temp_key,
                final_key,
                expected_size=expected_size,
                expected_checksum=expected_checksum,
            )
            raise RuntimeError("failure after immutable copy")

    repo = FakeRepository(preflight(kb_id))
    store = CopyThenFailStore()
    service = DocumentUploadService(
        repository=repo,
        object_store=store,
        notifier=FakeNotifier(),
        max_upload_bytes=1024,
    )
    with pytest.raises(RuntimeError):
        await service.upload(
            knowledge_base_id=kb_id,
            actor=actor(kb_id),
            source=SourceUpload(
                filename="doc.txt",
                content_type="text/plain",
                chunks=chunks(b"hello"),
            ),
            form=UploadForm.from_multipart(display_name=None, metadata=None, tags=None),
            idempotency_key=None,
        )

    assert store.publish_pairs[0][1] in store.deleted
    assert store.objects == {}
