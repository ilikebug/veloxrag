"""Upload orchestration across validation, immutable objects, PostgreSQL, and Redis."""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import AsyncIterable, AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol
from uuid import UUID, uuid4

from python_multipart import MultipartParser
from python_multipart.exceptions import MultipartParseError
from python_multipart.multipart import parse_options_header
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.api.errors import BusinessError
from rag_service.api.validation import validate_idempotency_key
from rag_service.auth.policies import AgentPrincipal, Capability, require_capability
from rag_service.infrastructure.minio_store import ObjectStoreError, UploadLimitExceeded
from rag_service.ingestion.artifacts import source_object_key
from rag_service.ingestion.repositories import (
    UploadPreflight,
    UploadReservation,
    UploadReservationResult,
)
from rag_service.ingestion.schemas import (
    UploadAccepted,
    UploadForm,
    UploadRequestFingerprintInput,
    canonical_upload_filename,
    upload_request_fingerprint,
    validate_metadata_against_filter_snapshot,
)
from rag_service.ingestion.validation import (
    DocumentValidationError,
    IncrementalTextValidator,
    TextValidationSummary,
    validate_document_metadata,
)
from rag_service.observability.logging import SafeLogContext, emit_safe_log
from rag_service.observability.metrics import METRICS, OperationalMetrics

if TYPE_CHECKING:
    from python_multipart.multipart import MultipartCallbacks

logger = logging.getLogger(__name__)


class SourceObjectStore(Protocol):
    async def verify_object(
        self,
        object_key: str,
        *,
        expected_size: int,
        expected_checksum: str,
    ) -> object: ...

    async def upload_stream(
        self,
        object_key: str,
        stream: AsyncIterable[bytes],
        *,
        content_type: str,
        max_bytes: int,
    ) -> object: ...

    async def publish_temp(
        self,
        temp_key: str,
        final_key: str,
        *,
        expected_size: int,
        expected_checksum: str,
    ) -> object: ...

    async def delete_best_effort(self, object_key: str) -> bool: ...


class JobNotifier(Protocol):
    async def notify(self, job_id: UUID) -> bool: ...


class UploadRepository(Protocol):
    async def preflight(self, knowledge_base_id: UUID) -> UploadPreflight | None: ...

    async def reserve(self, reservation: UploadReservation) -> UploadReservationResult: ...


ReservationSessionFactory = Callable[[], AsyncSession]
ReservationRepositoryFactory = Callable[[AsyncSession], UploadRepository]


@dataclass(frozen=True, slots=True)
class SourceUpload:
    filename: str
    content_type: str
    chunks: AsyncIterable[bytes]


@dataclass(frozen=True, slots=True)
class _StagedSource:
    temp_key: str
    filename: str
    content_type: str
    summary: TextValidationSummary


@dataclass(frozen=True, slots=True)
class StagedMultipartUpload:
    temp_key: str
    filename: str
    content_type: str
    form: UploadForm
    summary: TextValidationSummary


class _ValidatingStream:
    def __init__(self, source: AsyncIterable[bytes], validator: IncrementalTextValidator) -> None:
        self._source = source
        self._validator = validator
        self.summary: TextValidationSummary | None = None
        self.error: DocumentValidationError | None = None

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self._source:
                try:
                    self._validator.feed(chunk)
                except DocumentValidationError as error:
                    if error.code == "FILE_TOO_LARGE":
                        raise UploadLimitExceeded from None
                    self.error = error
                    return
                yield chunk
            try:
                self.summary = self._validator.finish()
            except DocumentValidationError as error:
                self.error = error
        finally:
            closer = getattr(self._source, "aclose", None)
            if callable(closer):
                await closer()


def _validation_business_error(error: DocumentValidationError) -> BusinessError:
    status = 413 if error.code == "FILE_TOO_LARGE" else 422
    return BusinessError(status, error.code, str(error))


_MULTIPART_FIELDS = frozenset({"file", "display_name", "metadata", "tags"})
_MAX_MULTIPART_FIELD_BYTES = 32 * 1024
_MAX_MULTIPART_HEADER_BYTES = 16 * 1024
_MAX_MULTIPART_OVERHEAD_BYTES = 256 * 1024
_MULTIPART_SLICE_BYTES = 64 * 1024


def _multipart_error() -> BusinessError:
    return BusinessError(422, "VALIDATION_ERROR", "Invalid document upload")


class _MultipartQueueStream:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=2)

    async def __aiter__(self) -> AsyncIterator[bytes]:
        while True:
            item = await self.queue.get()
            if item is None:
                return
            yield item


class _MultipartState:
    def __init__(
        self,
        *,
        max_file_bytes: int,
        temp_key: str,
        store: SourceObjectStore,
    ) -> None:
        self.max_file_bytes = max_file_bytes
        self.temp_key = temp_key
        self.store = store
        self.part_count = 0
        self.ended = False
        self.current_headers: dict[bytes, bytes] = {}
        self.current_header_name = bytearray()
        self.current_header_value = bytearray()
        self.current_header_bytes = 0
        self.current_name: str | None = None
        self.current_is_file = False
        self.current_field = bytearray()
        self.fields: dict[str, str] = {}
        self.file_count = 0
        self.filename: str | None = None
        self.content_type: str | None = None
        self.validator: IncrementalTextValidator | None = None
        self.summary: TextValidationSummary | None = None
        self.stream: _MultipartQueueStream | None = None
        self.upload_task: asyncio.Task[object] | None = None
        self.pending_file_chunks: list[bytes] = []
        self.file_end_pending = False

    @staticmethod
    def _piece(data: bytes, start: int, end: int) -> bytes:
        return bytes(memoryview(data)[start:end])

    def on_part_begin(self) -> None:
        self.part_count += 1
        if self.part_count > len(_MULTIPART_FIELDS):
            raise _multipart_error()
        self.current_headers = {}
        self.current_header_name.clear()
        self.current_header_value.clear()
        self.current_header_bytes = 0
        self.current_name = None
        self.current_is_file = False
        self.current_field.clear()

    def on_header_begin(self) -> None:
        self.current_header_name.clear()
        self.current_header_value.clear()

    def _append_header(self, target: bytearray, value: bytes) -> None:
        self.current_header_bytes += len(value)
        if self.current_header_bytes > _MAX_MULTIPART_HEADER_BYTES:
            raise _multipart_error()
        target.extend(value)

    def on_header_field(self, data: bytes, start: int, end: int) -> None:
        self._append_header(self.current_header_name, self._piece(data, start, end))

    def on_header_value(self, data: bytes, start: int, end: int) -> None:
        self._append_header(self.current_header_value, self._piece(data, start, end))

    def on_header_end(self) -> None:
        name = bytes(self.current_header_name).strip().lower()
        value = bytes(self.current_header_value).strip()
        if name not in {b"content-disposition", b"content-type"} or not value:
            raise _multipart_error()
        if name in self.current_headers:
            raise _multipart_error()
        self.current_headers[name] = value

    def on_headers_finished(self) -> None:
        disposition = self.current_headers.get(b"content-disposition")
        if disposition is None:
            raise _multipart_error()
        kind, options = parse_options_header(disposition)
        if kind != b"form-data" or b"name" not in options:
            raise _multipart_error()
        try:
            name = options[b"name"].decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            raise _multipart_error() from None
        if name not in _MULTIPART_FIELDS or name in self.fields:
            raise _multipart_error()
        filename_bytes = options.get(b"filename")
        if name == "file":
            if filename_bytes is None or self.file_count != 0:
                raise _multipart_error()
            try:
                filename = canonical_upload_filename(
                    filename_bytes.decode("utf-8", errors="strict")
                )
                declared_content_type = self.current_headers[b"content-type"].decode(
                    "ascii", errors="strict"
                )
                metadata = validate_document_metadata(
                    filename=filename,
                    content_type=declared_content_type,
                )
            except DocumentValidationError as error:
                raise _validation_business_error(error) from None
            except (KeyError, UnicodeError):
                raise _multipart_error() from None
            self.file_count = 1
            self.current_is_file = True
            self.filename = filename
            self.content_type = metadata.content_type
            self.validator = IncrementalTextValidator(
                filename=filename,
                content_type=metadata.content_type,
                max_bytes=self.max_file_bytes,
            )
            self.stream = _MultipartQueueStream()
            self.upload_task = asyncio.create_task(
                self.store.upload_stream(
                    self.temp_key,
                    self.stream,
                    content_type=metadata.content_type,
                    max_bytes=self.max_file_bytes,
                ),
                name="document-multipart-temp-upload",
            )
            return
        if filename_bytes is not None or b"content-type" in self.current_headers:
            raise _multipart_error()
        self.current_name = name

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        piece = self._piece(data, start, end)
        if self.current_is_file:
            validator = self.validator
            if validator is None:
                raise _multipart_error()
            try:
                validator.feed(piece)
            except DocumentValidationError as error:
                raise _validation_business_error(error) from None
            if piece:
                self.pending_file_chunks.append(piece)
            return
        if self.current_name is None:
            raise _multipart_error()
        if len(self.current_field) + len(piece) > _MAX_MULTIPART_FIELD_BYTES:
            raise _multipart_error()
        self.current_field.extend(piece)

    def on_part_end(self) -> None:
        if self.current_is_file:
            validator = self.validator
            if validator is None:
                raise _multipart_error()
            try:
                self.summary = validator.finish()
            except DocumentValidationError as error:
                raise _validation_business_error(error) from None
            self.file_end_pending = True
            return
        if self.current_name is None:
            raise _multipart_error()
        try:
            value = bytes(self.current_field).decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            raise _multipart_error() from None
        self.fields[self.current_name] = value

    def on_end(self) -> None:
        self.ended = True

    def callbacks(self) -> MultipartCallbacks:
        return {
            "on_part_begin": self.on_part_begin,
            "on_header_begin": self.on_header_begin,
            "on_header_field": self.on_header_field,
            "on_header_value": self.on_header_value,
            "on_header_end": self.on_header_end,
            "on_headers_finished": self.on_headers_finished,
            "on_part_data": self.on_part_data,
            "on_part_end": self.on_part_end,
            "on_end": self.on_end,
        }

    async def _send(self, item: bytes | None) -> None:
        stream = self.stream
        upload_task = self.upload_task
        if stream is None or upload_task is None:
            raise _multipart_error()
        if upload_task.done():
            await upload_task
        put_task = asyncio.create_task(stream.queue.put(item))
        done, _pending = await asyncio.wait(
            {put_task, upload_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if upload_task in done:
            put_task.cancel()
            with suppress(BaseException):
                await put_task
            await upload_task
        await put_task

    async def drain(self) -> None:
        chunks, self.pending_file_chunks = self.pending_file_chunks, []
        for chunk in chunks:
            await self._send(chunk)
        if self.file_end_pending:
            self.file_end_pending = False
            await self._send(None)

    async def abort(self) -> None:
        task = self.upload_task
        if task is not None and not task.done():
            task.cancel()
        if task is not None:
            with suppress(BaseException):
                await task


async def _cleanup_multipart_temp(store: SourceObjectStore, temp_key: str) -> None:
    task = asyncio.create_task(store.delete_best_effort(temp_key))
    try:
        async with asyncio.timeout(0.25):
            await asyncio.shield(task)
    except BaseException:
        task.cancel()
        with suppress(BaseException):
            await task


async def stage_multipart_upload(
    *,
    body: AsyncIterable[bytes],
    content_type: str | None,
    object_store: SourceObjectStore,
    max_file_bytes: int,
) -> StagedMultipartUpload:
    if type(content_type) is not str:
        raise _multipart_error()
    media_type, options = parse_options_header(content_type)
    boundary = options.get(b"boundary")
    if (
        media_type != b"multipart/form-data"
        or boundary is None
        or not 1 <= len(boundary) <= 200
        or any(value < 0x21 or value > 0x7E for value in boundary)
    ):
        raise _multipart_error()

    temp_key = f"tmp/uploads/{uuid4().hex}"
    state = _MultipartState(
        max_file_bytes=max_file_bytes,
        temp_key=temp_key,
        store=object_store,
    )
    parser = MultipartParser(boundary, callbacks=state.callbacks())
    total_body_bytes = 0
    try:
        async for chunk in body:
            if type(chunk) is not bytes:
                raise _multipart_error()
            for start in range(0, len(chunk), _MULTIPART_SLICE_BYTES):
                piece = chunk[start : start + _MULTIPART_SLICE_BYTES]
                total_body_bytes += len(piece)
                if total_body_bytes > max_file_bytes + _MAX_MULTIPART_OVERHEAD_BYTES:
                    raise _multipart_error()
                try:
                    parser.write(piece)
                except MultipartParseError:
                    raise _multipart_error() from None
                await state.drain()
        try:
            parser.finalize()
        except MultipartParseError:
            raise _multipart_error() from None
        await state.drain()
        if not state.ended or state.file_count != 1 or state.upload_task is None:
            raise _multipart_error()
        receipt = await state.upload_task
        summary = state.summary
        if (
            summary is None
            or state.filename is None
            or state.content_type is None
            or getattr(receipt, "size", None) != summary.source_size
            or getattr(receipt, "checksum_sha256", None) != summary.source_checksum_sha256
        ):
            raise BusinessError(500, "INTERNAL_ERROR", "Internal server error")
        form = UploadForm.from_multipart(
            display_name=state.fields.get("display_name"),
            metadata=state.fields.get("metadata"),
            tags=state.fields.get("tags"),
        )
        return StagedMultipartUpload(
            temp_key,
            state.filename,
            state.content_type,
            form,
            summary,
        )
    except BaseException:
        await state.abort()
        await _cleanup_multipart_temp(object_store, temp_key)
        raise
    finally:
        closer = getattr(body, "aclose", None)
        if callable(closer):
            with suppress(BaseException):
                await closer()


@asynccontextmanager
async def _without_transaction() -> AsyncIterator[None]:
    yield


class _ReservationCancelled(asyncio.CancelledError):
    def __init__(
        self,
        *,
        safe_to_cleanup: bool,
        result: UploadReservationResult | None = None,
    ) -> None:
        super().__init__()
        self.safe_to_cleanup = safe_to_cleanup
        self.result = result


class DocumentUploadService:
    def __init__(
        self,
        *,
        repository: UploadRepository,
        object_store: SourceObjectStore,
        notifier: JobNotifier,
        max_upload_bytes: int,
        max_idempotency_key_length: int = 128,
        session: AsyncSession | None = None,
        reservation_session_factory: ReservationSessionFactory | None = None,
        reservation_repository_factory: ReservationRepositoryFactory | None = None,
        commit_resolution_timeout_seconds: float = 2.0,
        notification_timeout_seconds: float = 0.25,
        metrics: OperationalMetrics = METRICS,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if type(max_upload_bytes) is not int or max_upload_bytes <= 0:
            raise ValueError("upload byte limit is invalid")
        if type(max_idempotency_key_length) is not int or max_idempotency_key_length <= 0:
            raise ValueError("idempotency key limit is invalid")
        if (
            isinstance(commit_resolution_timeout_seconds, bool)
            or not isinstance(commit_resolution_timeout_seconds, (int, float))
            or not math.isfinite(commit_resolution_timeout_seconds)
            or commit_resolution_timeout_seconds <= 0
        ):
            raise ValueError("commit resolution timeout is invalid")
        if (
            isinstance(notification_timeout_seconds, bool)
            or not isinstance(notification_timeout_seconds, (int, float))
            or not math.isfinite(notification_timeout_seconds)
            or notification_timeout_seconds <= 0
        ):
            raise ValueError("notification timeout is invalid")
        if (reservation_session_factory is None) != (reservation_repository_factory is None):
            raise ValueError("reservation session and repository factories must be paired")
        if session is not None and reservation_session_factory is None:
            raise ValueError("request session cannot own detached reservations")
        self._repository = repository
        self._object_store = object_store
        self._notifier = notifier
        self._max_upload_bytes = max_upload_bytes
        self._max_idempotency_key_length = max_idempotency_key_length
        self._session = session
        self._reservation_session_factory = reservation_session_factory
        self._reservation_repository_factory = reservation_repository_factory
        self._commit_resolution_timeout_seconds = commit_resolution_timeout_seconds
        self._notification_timeout_seconds = notification_timeout_seconds
        self._metrics = metrics
        self._monotonic_clock = monotonic_clock

    def _started_at(self) -> float | None:
        try:
            value = self._monotonic_clock()
        except BaseException:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None

    def _duration_since(self, started_at: float | None) -> float:
        if started_at is None:
            return 0.0
        try:
            finished_at = self._monotonic_clock()
        except BaseException:
            return 0.0
        if isinstance(finished_at, bool) or not isinstance(finished_at, (int, float)):
            return 0.0
        duration = float(finished_at) - started_at
        return duration if math.isfinite(duration) and duration >= 0 else 0.0

    def _observe_upload(
        self,
        *,
        knowledge_base_id: UUID,
        request_id: str | None,
        outcome: str,
        byte_count: int,
        started_at: float | None,
        accepted: UploadAccepted | None = None,
    ) -> None:
        duration_seconds = self._duration_since(started_at)
        with suppress(BaseException):
            self._metrics.record_upload(
                outcome=outcome,
                byte_count=byte_count,
                duration_seconds=duration_seconds,
            )
        try:
            context = SafeLogContext(
                request_id=request_id,
                knowledge_base_id=knowledge_base_id,
                document_id=accepted.document_id if accepted is not None else None,
                version_id=accepted.version_id if accepted is not None else None,
                job_id=accepted.job_id if accepted is not None else None,
            )
            fields: dict[str, object] = {
                "operation": "upload",
                "outcome": outcome,
                "byte_count": byte_count,
                "duration_seconds": duration_seconds,
            }
            if outcome == "rejected":
                fields["error_code"] = "UPLOAD_REJECTED"
            elif outcome == "failed":
                fields["error_code"] = "UPLOAD_FAILED"
            emit_safe_log(
                logger,
                logging.INFO,
                "upload.completed" if outcome == "succeeded" else "upload.failed",
                context=context,
                **fields,
            )
        except BaseException:
            pass

    def _observe_queued_job(
        self,
        *,
        request_id: str | None,
        knowledge_base_id: UUID,
        accepted: UploadAccepted,
    ) -> None:
        with suppress(BaseException):
            self._metrics.record_job_state(state="queued")
        try:
            emit_safe_log(
                logger,
                logging.INFO,
                "job.state.changed",
                context=SafeLogContext(
                    request_id=request_id,
                    knowledge_base_id=knowledge_base_id,
                    document_id=accepted.document_id,
                    version_id=accepted.version_id,
                    job_id=accepted.job_id,
                ),
                operation="ingest_document",
                state="queued",
            )
        except BaseException:
            return

    def _transaction(self) -> object:
        if self._session is None:
            return _without_transaction()
        return self._session.begin()

    async def _preflight(self, knowledge_base_id: UUID) -> UploadPreflight | None:
        async with self._transaction():  # type: ignore[attr-defined]
            return await self._repository.preflight(knowledge_base_id)

    def _consume_background_reservation(
        self,
        task: asyncio.Task[UploadReservationResult],
        *,
        request_id: str | None,
        knowledge_base_id: UUID,
    ) -> None:
        try:
            result = task.result()
        except (asyncio.CancelledError, BusinessError):
            return
        except BaseException:
            with suppress(BaseException):
                logger.error("Detached upload reservation failed unexpectedly")
            return
        try:
            if result.replay:
                return
            self._observe_queued_job(
                request_id=request_id,
                knowledge_base_id=knowledge_base_id,
                accepted=UploadAccepted(
                    document_id=result.document_id,
                    version_id=result.version_id,
                    job_id=result.job_id,
                    status="queued",
                ),
            )
        except BaseException:
            return

    @staticmethod
    def _consume_background_notification(task: asyncio.Task[bool]) -> None:
        with suppress(BaseException):
            task.result()

    async def _reservation_operation(
        self,
        reservation: UploadReservation,
    ) -> UploadReservationResult:
        session_factory = self._reservation_session_factory
        repository_factory = self._reservation_repository_factory
        if session_factory is None or repository_factory is None:
            async with _without_transaction():
                return await self._repository.reserve(reservation)
        async with session_factory() as session:
            repository = repository_factory(session)
            async with session.begin():
                return await repository.reserve(reservation)

    async def _reserve(
        self,
        reservation: UploadReservation,
        *,
        request_id: str | None,
    ) -> UploadReservationResult | None:
        task = asyncio.create_task(
            self._reservation_operation(reservation),
            name="document-upload-reservation",
        )

        def consume_late_result(completed: asyncio.Task[UploadReservationResult]) -> None:
            self._consume_background_reservation(
                completed,
                request_id=request_id,
                knowledge_base_id=reservation.knowledge_base_id,
            )

        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            try:
                async with asyncio.timeout(self._commit_resolution_timeout_seconds):
                    result = await asyncio.shield(task)
            except TimeoutError:
                task.add_done_callback(consume_late_result)
                raise _ReservationCancelled(safe_to_cleanup=False) from None
            except BusinessError:
                raise _ReservationCancelled(safe_to_cleanup=True) from None
            except BaseException:
                task.add_done_callback(consume_late_result)
                raise _ReservationCancelled(safe_to_cleanup=False) from None
            raise _ReservationCancelled(
                safe_to_cleanup=result.replay,
                result=result,
            ) from None
        except BusinessError:
            raise
        except Exception:
            logger.error("Upload reservation failed unexpectedly")
            return None

    async def _notify_after_commit(self, job_id: UUID) -> None:
        task = asyncio.create_task(
            self._notifier.notify(job_id),
            name="document-upload-notification",
        )
        task.add_done_callback(self._consume_background_notification)
        try:
            completed, _pending = await asyncio.wait(
                (task,),
                timeout=self._notification_timeout_seconds,
            )
        except asyncio.CancelledError:
            task.cancel()
            raise
        if task not in completed:
            task.cancel()
            logger.warning(
                "Upload notification timed out after reservation commit",
                extra={"job_id": str(job_id)},
            )
            return
        try:
            task.result()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "Upload notification failed after reservation commit",
                extra={"job_id": str(job_id)},
            )

    async def _cleanup_final(self, object_key: str) -> None:
        cleanup = asyncio.create_task(self._object_store.delete_best_effort(object_key))
        try:
            async with asyncio.timeout(0.25):
                await asyncio.shield(cleanup)
        except BaseException:
            cleanup.cancel()
            with suppress(BaseException):
                await asyncio.shield(cleanup)

    async def _authorize_preflight(
        self,
        *,
        knowledge_base_id: UUID,
        actor: AgentPrincipal,
        idempotency_key: str | None,
    ) -> tuple[UploadPreflight | None, str | None]:
        if knowledge_base_id not in actor.knowledge_base_ids:
            raise BusinessError(404, "RESOURCE_NOT_FOUND", "Resource not found")
        require_capability(actor, Capability.INGEST)
        if idempotency_key is not None:
            idempotency_key = validate_idempotency_key(
                idempotency_key, self._max_idempotency_key_length
            )
        preflight = await self._preflight(knowledge_base_id)
        if preflight is None and idempotency_key is None:
            raise BusinessError(
                409,
                "KNOWLEDGE_BASE_NOT_INDEX_CONFIGURED",
                "Knowledge base has no active index generation",
            )
        return preflight, idempotency_key

    async def _stage_source(self, source: SourceUpload) -> _StagedSource:
        try:
            filename = canonical_upload_filename(source.filename)
            document_metadata = validate_document_metadata(
                filename=filename,
                content_type=source.content_type,
            )
        except DocumentValidationError as error:
            raise _validation_business_error(error) from None
        temp_key = f"tmp/uploads/{uuid4().hex}"
        validator = IncrementalTextValidator(
            filename=filename,
            content_type=document_metadata.content_type,
            max_bytes=self._max_upload_bytes,
        )
        stream = _ValidatingStream(source.chunks, validator)
        try:
            receipt = await self._object_store.upload_stream(
                temp_key,
                stream,
                content_type=document_metadata.content_type,
                max_bytes=self._max_upload_bytes,
            )
            if stream.error is not None:
                raise _validation_business_error(stream.error)
            summary = stream.summary
            if (
                summary is None
                or getattr(receipt, "checksum_sha256", None) != summary.source_checksum_sha256
                or getattr(receipt, "size", None) != summary.source_size
            ):
                raise BusinessError(500, "INTERNAL_ERROR", "Internal server error")
            return _StagedSource(
                temp_key,
                filename,
                document_metadata.content_type,
                summary,
            )
        except ObjectStoreError as error:
            await self._cleanup_final(temp_key)
            status = 413 if error.code == "FILE_TOO_LARGE" else 503
            raise BusinessError(
                status,
                error.code,
                str(error),
                retryable=error.retryable,
            ) from None
        except BaseException:
            await self._cleanup_final(temp_key)
            raise

    async def _publish_and_reserve(
        self,
        *,
        knowledge_base_id: UUID,
        actor: AgentPrincipal,
        preflight: UploadPreflight | None,
        staged: _StagedSource | StagedMultipartUpload,
        form: UploadForm,
        idempotency_key: str | None,
        request_id: str | None,
    ) -> UploadAccepted:
        temp_key = staged.temp_key
        filename = staged.filename
        summary = staged.summary
        document_metadata = validate_document_metadata(
            filename=filename,
            content_type=staged.content_type,
        )
        detected_mime = "text/plain" if document_metadata.extension == ".txt" else "text/markdown"
        parser_name = (
            "plain_text_v1" if document_metadata.extension == ".txt" else "markdown_text_v1"
        )
        document_id, version_id, job_id = uuid4(), uuid4(), uuid4()
        final_key = source_object_key(
            knowledge_base_id,
            document_id,
            version_id,
            filename=filename,
        )
        temp_may_exist = True
        final_may_exist = False
        try:
            if idempotency_key is None:
                if preflight is None:
                    raise BusinessError(
                        409,
                        "KNOWLEDGE_BASE_NOT_INDEX_CONFIGURED",
                        "Knowledge base has no active index generation",
                    )
                validate_metadata_against_filter_snapshot(form.metadata, preflight)
            final_may_exist = True
            published = await self._object_store.publish_temp(
                temp_key,
                final_key,
                expected_size=summary.source_size,
                expected_checksum=summary.source_checksum_sha256,
            )
            temp_may_exist = False
            if (
                getattr(published, "object_key", None) != final_key
                or getattr(published, "size", None) != summary.source_size
                or getattr(published, "checksum_sha256", None) != summary.source_checksum_sha256
            ):
                raise BusinessError(500, "INTERNAL_ERROR", "Internal server error")
            display_name = form.display_name or filename
            fingerprint = upload_request_fingerprint(
                UploadRequestFingerprintInput(
                    actor.key_id,
                    knowledge_base_id,
                    summary.source_checksum_sha256,
                    filename,
                    document_metadata.extension,
                    document_metadata.content_type,
                    detected_mime,
                    parser_name,
                    display_name,
                    form.tags,
                    form.metadata,
                )
            )
            reservation = UploadReservation(
                actor=actor,
                knowledge_base_id=knowledge_base_id,
                preflight=preflight,
                idempotency_key=idempotency_key,
                request_fingerprint=fingerprint,
                document_id=document_id,
                version_id=version_id,
                job_id=job_id,
                source_object_key=final_key,
                source_size=summary.source_size,
                source_checksum_sha256=summary.source_checksum_sha256,
                display_name=display_name,
                tags=form.tags,
                metadata=form.metadata,
                declared_mime_type=document_metadata.content_type,
                detected_mime_type=detected_mime,
                source_extension=document_metadata.extension,
                parser_name=parser_name,
            )
            result = await self._reserve(reservation, request_id=request_id)
            if result is None:
                final_may_exist = False
                raise BusinessError(
                    503,
                    "INGESTION_RESERVATION_UNCERTAIN",
                    "Upload reservation outcome is unknown",
                    retryable=True,
                )
            if result.replay:
                await self._cleanup_final(final_key)
                final_may_exist = False
            accepted = UploadAccepted(
                document_id=result.document_id,
                version_id=result.version_id,
                job_id=result.job_id,
                status="queued",
            )
        except _ReservationCancelled as cancellation:
            resolved = cancellation.result
            if resolved is not None and not resolved.replay:
                accepted = UploadAccepted(
                    document_id=resolved.document_id,
                    version_id=resolved.version_id,
                    job_id=resolved.job_id,
                    status="queued",
                )
                self._observe_queued_job(
                    request_id=request_id,
                    knowledge_base_id=knowledge_base_id,
                    accepted=accepted,
                )
            if temp_may_exist:
                await self._cleanup_final(temp_key)
            if cancellation.safe_to_cleanup and final_may_exist:
                await self._cleanup_final(final_key)
            raise asyncio.CancelledError from None
        except ObjectStoreError as error:
            if temp_may_exist:
                await self._cleanup_final(temp_key)
            if final_may_exist:
                await self._cleanup_final(final_key)
            status = 413 if error.code == "FILE_TOO_LARGE" else 503
            raise BusinessError(status, error.code, str(error), retryable=error.retryable) from None
        except BusinessError:
            if temp_may_exist:
                await self._cleanup_final(temp_key)
            if final_may_exist:
                await self._cleanup_final(final_key)
            raise
        except BaseException:
            if temp_may_exist:
                await self._cleanup_final(temp_key)
            if final_may_exist:
                await self._cleanup_final(final_key)
            raise
        if not result.replay:
            self._observe_queued_job(
                request_id=request_id,
                knowledge_base_id=knowledge_base_id,
                accepted=accepted,
            )
            await self._notify_after_commit(result.job_id)
        return accepted

    async def upload(
        self,
        *,
        knowledge_base_id: UUID,
        actor: AgentPrincipal,
        source: SourceUpload,
        form: UploadForm,
        idempotency_key: str | None,
        request_id: str | None = None,
    ) -> UploadAccepted:
        started_at = self._started_at()
        try:
            preflight, idempotency_key = await self._authorize_preflight(
                knowledge_base_id=knowledge_base_id,
                actor=actor,
                idempotency_key=idempotency_key,
            )
            staged = await self._stage_source(source)
            accepted = await self._publish_and_reserve(
                knowledge_base_id=knowledge_base_id,
                actor=actor,
                preflight=preflight,
                staged=staged,
                form=form,
                idempotency_key=idempotency_key,
                request_id=request_id,
            )
        except asyncio.CancelledError:
            self._observe_upload(
                knowledge_base_id=knowledge_base_id,
                request_id=request_id,
                outcome="cancelled",
                byte_count=0,
                started_at=started_at,
            )
            raise
        except BusinessError as error:
            self._observe_upload(
                knowledge_base_id=knowledge_base_id,
                request_id=request_id,
                outcome="rejected" if error.status_code < 500 else "failed",
                byte_count=0,
                started_at=started_at,
            )
            raise
        except BaseException:
            self._observe_upload(
                knowledge_base_id=knowledge_base_id,
                request_id=request_id,
                outcome="failed",
                byte_count=0,
                started_at=started_at,
            )
            raise
        self._observe_upload(
            knowledge_base_id=knowledge_base_id,
            request_id=request_id,
            outcome="succeeded",
            byte_count=staged.summary.source_size,
            started_at=started_at,
            accepted=accepted,
        )
        return accepted

    async def upload_multipart(
        self,
        *,
        knowledge_base_id: UUID,
        actor: AgentPrincipal,
        body: AsyncIterable[bytes],
        content_type: str | None,
        idempotency_key: str | None,
        request_id: str | None = None,
    ) -> UploadAccepted:
        started_at = self._started_at()
        try:
            preflight, idempotency_key = await self._authorize_preflight(
                knowledge_base_id=knowledge_base_id,
                actor=actor,
                idempotency_key=idempotency_key,
            )
            staged = await stage_multipart_upload(
                body=body,
                content_type=content_type,
                object_store=self._object_store,
                max_file_bytes=self._max_upload_bytes,
            )
        except DocumentValidationError as error:
            business_error = _validation_business_error(error)
            self._observe_upload(
                knowledge_base_id=knowledge_base_id,
                request_id=request_id,
                outcome="rejected" if business_error.status_code < 500 else "failed",
                byte_count=0,
                started_at=started_at,
            )
            raise business_error from None
        except ObjectStoreError as error:
            status = 413 if error.code == "FILE_TOO_LARGE" else 503
            self._observe_upload(
                knowledge_base_id=knowledge_base_id,
                request_id=request_id,
                outcome="rejected" if status < 500 else "failed",
                byte_count=0,
                started_at=started_at,
            )
            raise BusinessError(status, error.code, str(error), retryable=error.retryable) from None
        except asyncio.CancelledError:
            self._observe_upload(
                knowledge_base_id=knowledge_base_id,
                request_id=request_id,
                outcome="cancelled",
                byte_count=0,
                started_at=started_at,
            )
            raise
        except BusinessError as error:
            self._observe_upload(
                knowledge_base_id=knowledge_base_id,
                request_id=request_id,
                outcome="rejected" if error.status_code < 500 else "failed",
                byte_count=0,
                started_at=started_at,
            )
            raise
        except BaseException:
            self._observe_upload(
                knowledge_base_id=knowledge_base_id,
                request_id=request_id,
                outcome="failed",
                byte_count=0,
                started_at=started_at,
            )
            raise
        try:
            accepted = await self._publish_and_reserve(
                knowledge_base_id=knowledge_base_id,
                actor=actor,
                preflight=preflight,
                staged=staged,
                form=staged.form,
                idempotency_key=idempotency_key,
                request_id=request_id,
            )
        except asyncio.CancelledError:
            self._observe_upload(
                knowledge_base_id=knowledge_base_id,
                request_id=request_id,
                outcome="cancelled",
                byte_count=0,
                started_at=started_at,
            )
            raise
        except BusinessError as error:
            self._observe_upload(
                knowledge_base_id=knowledge_base_id,
                request_id=request_id,
                outcome="rejected" if error.status_code < 500 else "failed",
                byte_count=0,
                started_at=started_at,
            )
            raise
        except BaseException:
            self._observe_upload(
                knowledge_base_id=knowledge_base_id,
                request_id=request_id,
                outcome="failed",
                byte_count=0,
                started_at=started_at,
            )
            raise
        self._observe_upload(
            knowledge_base_id=knowledge_base_id,
            request_id=request_id,
            outcome="succeeded",
            byte_count=staged.summary.source_size,
            started_at=started_at,
            accepted=accepted,
        )
        return accepted


__all__ = ["DocumentUploadService", "SourceUpload"]
