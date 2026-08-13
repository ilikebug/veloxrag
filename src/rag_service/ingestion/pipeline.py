"""Fenced, deterministic parse and chunk artifact stages."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import sys
import time
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Iterator,
    Mapping,
    Sequence,
)
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractAsyncContextManager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from typing import Protocol, cast
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.indexing.identities import canonical_json_bytes, canonical_sha256, point_id
from rag_service.indexing.qdrant import (
    QdrantClient,
    QdrantConfigurationError,
    QdrantPoint,
    QdrantTransientError,
)
from rag_service.infrastructure.minio_store import (
    ArtifactChecksumConflict,
    ObjectStoreError,
    OrphanPage,
    UploadLimitExceeded,
)
from rag_service.ingestion.artifacts import (
    canonical_document_artifact_identity,
    chunk_manifest_header_bytes,
    chunk_manifest_record_bytes,
    chunks_object_key,
    iter_chunk_manifest_lines,
    parsed_text_object_key,
    temporary_object_key,
)
from rag_service.ingestion.chunkers import (
    CHUNK_SCHEMA_VERSION,
    DEFAULT_CHUNK_CODEPOINTS,
    DEFAULT_OVERLAP_CODEPOINTS,
    Chunk,
    RecursiveTextChunker,
)
from rag_service.ingestion.parsers import (
    ParsedArtifact,
    Parser,
    StructuralMetadataLimitExceeded,
    parser_for_extension,
)
from rag_service.ingestion.repositories import (
    ActivationStageInput,
    ChunkStageInput,
    DocumentActivationConflictError,
    EmbedIndexStageInput,
    ParseStageInput,
    SqlAlchemyIngestionPipelineRepository,
)
from rag_service.ingestion.schemas import is_rfc3339_datetime
from rag_service.ingestion.validation import DocumentValidationError
from rag_service.jobs.repositories import ExhaustedJob, LostLeaseError
from rag_service.jobs.runner import (
    JobExecutionContext,
    JobExecutionError,
    JobHandlerOutcome,
    PermanentJobError,
    RetryableJobError,
    RetryableProviderJobError,
)
from rag_service.observability.logging import SafeLogContext, emit_safe_log
from rag_service.observability.metrics import METRICS, OperationalMetrics
from rag_service.observability.repositories import ProviderUsageContext
from rag_service.providers.embeddings import (
    EmbeddingAttempt,
    EmbeddingAttemptObserver,
    EmbeddingConfigSnapshot,
    EmbeddingGatewayError,
    EmbeddingOperationalConfig,
    EmbeddingResult,
)

MAX_CHUNK_MANIFEST_BYTES = 512 * 1024 * 1024
_MAX_S3_OBJECT_BYTES = 5 * 1024**4
_MAX_MANIFEST_LINE_BYTES = 1024 * 1024
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _StageObservation:
    knowledge_base_id: UUID | None = None
    document_id: UUID | None = None
    character_count: int = 0
    chunk_count: int = 0
    batch_count: int = 0


_CURRENT_STAGE_OBSERVATION: ContextVar[_StageObservation | None] = ContextVar(
    "rag_ingestion_stage_observation",
    default=None,
)


async def _close_async_iterators(*iterators: object) -> BaseException | None:
    """Close every owned iterator while retaining the first cleanup failure."""

    first_error: BaseException | None = None
    seen: set[int] = set()
    for iterator in iterators:
        if id(iterator) in seen:
            continue
        seen.add(id(iterator))
        closer = getattr(iterator, "aclose", None)
        if not callable(closer):
            continue
        try:
            await closer()
        except BaseException as error:
            if first_error is None:
                first_error = error
    return first_error


async def close_async_iterators_without_masking_primary(*iterators: object) -> None:
    """Close owned async iterators without replacing an active primary error."""

    primary_error = sys.exception()
    cleanup_error = await _close_async_iterators(*iterators)
    if primary_error is None and cleanup_error is not None:
        raise cleanup_error


class ArtifactReceipt(Protocol):
    object_key: str
    size: int
    checksum_sha256: str


class PipelineObjectStore(Protocol):
    async def verify_object(
        self,
        object_key: str,
        *,
        expected_size: int,
        expected_checksum: str,
    ) -> ArtifactReceipt: ...

    async def read_bytes(
        self,
        object_key: str,
        *,
        expected_checksum: str,
        max_bytes: int,
    ) -> bytes: ...

    def read_stream(
        self,
        object_key: str,
        *,
        expected_checksum: str,
        max_bytes: int,
    ) -> AsyncIterator[bytes]: ...

    async def upload_stream(
        self,
        object_key: str,
        stream: AsyncIterable[bytes],
        *,
        content_type: str,
        max_bytes: int,
    ) -> ArtifactReceipt: ...

    async def publish_temp(
        self,
        temp_key: str,
        final_key: str,
        *,
        expected_size: int,
        expected_checksum: str,
    ) -> ArtifactReceipt: ...

    async def delete_best_effort(self, object_key: str) -> bool: ...

    async def list_older_than(
        self,
        *,
        prefix: str,
        older_than: datetime,
        limit: int,
        start_after: str | None = None,
    ) -> OrphanPage: ...


class IngestionPipelineRepository(Protocol):
    async def object_key_is_referenced(self, object_key: str) -> bool: ...

    async def object_key_cleanup_is_allowed(self, object_key: str) -> bool: ...

    async def load_parse_stage(
        self,
        version_id: UUID,
        generation_id: UUID,
    ) -> ParseStageInput: ...

    async def load_chunk_stage(
        self,
        version_id: UUID,
        generation_id: UUID,
    ) -> ChunkStageInput: ...

    async def load_embed_index_stage(
        self,
        version_id: UUID,
        generation_id: UUID,
        job_id: UUID,
    ) -> EmbedIndexStageInput: ...

    async def load_validate_stage(
        self,
        version_id: UUID,
        generation_id: UUID,
        job_id: UUID,
    ) -> EmbedIndexStageInput: ...

    async def load_activation_stage(
        self,
        version_id: UUID,
        generation_id: UUID,
        job_id: UUID,
    ) -> ActivationStageInput: ...


class PipelineEmbeddingGateway(Protocol):
    async def embed(
        self,
        *,
        snapshot: EmbeddingConfigSnapshot,
        operational: EmbeddingOperationalConfig,
        inputs: Sequence[str],
        attempt_observer: EmbeddingAttemptObserver | None = None,
    ) -> EmbeddingResult: ...


class PipelineProviderUsageSink(Protocol):
    async def record(
        self,
        context: ProviderUsageContext,
        attempt: EmbeddingAttempt,
    ) -> None: ...


@dataclass(frozen=True, slots=True, repr=False)
class ObjectReconciliationCursor:
    phase: int
    start_after: str | None = None

    def __post_init__(self) -> None:
        if (
            type(self.phase) is not int
            or self.phase not in {0, 1}
            or (
                self.start_after is not None
                and (
                    type(self.start_after) is not str
                    or not self.start_after
                    or len(self.start_after.encode("utf-8")) > 1024
                    or "\x00" in self.start_after
                )
            )
        ):
            raise ValueError("object reconciliation cursor is invalid")

    def __repr__(self) -> str:
        return f"ObjectReconciliationCursor(phase={self.phase}, <opaque>)"


@dataclass(frozen=True, slots=True, repr=False)
class ObjectReconciliationResult:
    scanned_count: int
    deleted_count: int
    failed_count: int
    next_cursor: ObjectReconciliationCursor | None

    def __repr__(self) -> str:
        return (
            "ObjectReconciliationResult("
            f"scanned_count={self.scanned_count}, "
            f"deleted_count={self.deleted_count}, "
            f"failed_count={self.failed_count}, <cursor>)"
        )


_OBJECT_RECONCILIATION_PREFIXES = ("tmp/jobs/", "knowledge-bases/")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _is_canonical_final_object_key(object_key: str) -> bool:
    return canonical_document_artifact_identity(object_key) is not None


async def _noop_pipeline_checkpoint(_name: str) -> None:
    return None


class IngestionPipelineHooks:
    """Internal deterministic crash seam for batch recovery tests."""

    def __init__(
        self,
        checkpoint: Callable[[str], Awaitable[None]] = _noop_pipeline_checkpoint,
    ) -> None:
        if not callable(checkpoint):
            raise ValueError("ingestion pipeline hook is invalid")
        self._checkpoint = checkpoint

    async def reached(self, name: str) -> None:
        await self._checkpoint(name)


type PipelineRepositoryContextFactory = Callable[
    [], AbstractAsyncContextManager[IngestionPipelineRepository]
]


@dataclass(frozen=True, slots=True, repr=False)
class ManifestExpectation:
    source_checksum_sha256: str
    parsed_checksum_sha256: str
    parser_name: str
    parser_version: str
    parser_config: Mapping[str, object]
    chunker_name: str
    chunker_version: str
    chunker_config: Mapping[str, object]
    chunk_config_hash: str
    document_version_created_at: datetime
    chunk_count: int

    def __post_init__(self) -> None:
        if (
            _SHA256_PATTERN.fullmatch(self.source_checksum_sha256) is None
            or _SHA256_PATTERN.fullmatch(self.parsed_checksum_sha256) is None
            or _SHA256_PATTERN.fullmatch(self.chunk_config_hash) is None
            or type(self.parser_name) is not str
            or not self.parser_name
            or type(self.parser_version) is not str
            or not self.parser_version
            or not isinstance(self.parser_config, Mapping)
            or type(self.chunker_name) is not str
            or not self.chunker_name
            or type(self.chunker_version) is not str
            or not self.chunker_version
            or not isinstance(self.chunker_config, Mapping)
            or type(self.document_version_created_at) is not datetime
            or self.document_version_created_at.tzinfo is None
            or self.document_version_created_at.utcoffset() is None
            or type(self.chunk_count) is not int
            or self.chunk_count < 1
        ):
            raise ValueError("chunk manifest expectation is invalid")
        try:
            parser_config = _freeze_manifest_json(
                json.loads(canonical_json_bytes(dict(self.parser_config)))
            )
            chunker_config = _freeze_manifest_json(
                json.loads(canonical_json_bytes(dict(self.chunker_config)))
            )
            if not isinstance(parser_config, Mapping) or not isinstance(chunker_config, Mapping):
                raise ValueError
            object.__setattr__(self, "parser_config", parser_config)
            object.__setattr__(self, "chunker_config", chunker_config)
        except (TypeError, ValueError, json.JSONDecodeError):
            raise ValueError("chunk manifest expectation is invalid") from None


def _freeze_manifest_json(value: object) -> object:
    if type(value) is dict:
        return MappingProxyType(
            {
                key: _freeze_manifest_json(member)
                for key, member in sorted(cast(dict[str, object], value).items())
            }
        )
    if type(value) is list:
        return tuple(_freeze_manifest_json(member) for member in cast(list[object], value))
    if value is None or type(value) in {str, bool, int, float}:
        return value
    raise ValueError


def _thaw_manifest_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_manifest_json(member) for key, member in sorted(value.items())}
    if type(value) is tuple:
        return [_thaw_manifest_json(member) for member in cast(tuple[object, ...], value)]
    if value is None or type(value) in {str, bool, int, float}:
        return value
    raise ValueError


async def _manifest_lines(stream: AsyncIterable[bytes]) -> AsyncGenerator[bytes, None]:
    buffered = bytearray()
    iterator = stream.__aiter__()
    try:
        async for block in iterator:
            if type(block) is not bytes:
                raise ValueError("chunk manifest is invalid")
            buffered.extend(block)
            if len(buffered) > _MAX_MANIFEST_LINE_BYTES and b"\n" not in buffered:
                raise ValueError("chunk manifest is invalid")
            while True:
                newline = buffered.find(b"\n")
                if newline < 0:
                    break
                if newline + 1 > _MAX_MANIFEST_LINE_BYTES:
                    raise ValueError("chunk manifest is invalid")
                line = bytes(buffered[: newline + 1])
                del buffered[: newline + 1]
                if line == b"\n":
                    raise ValueError("chunk manifest is invalid")
                yield line
        if buffered:
            raise ValueError("chunk manifest is invalid")
    finally:
        await close_async_iterators_without_masking_primary(iterator)


def _manifest_document(line: bytes) -> dict[str, object]:
    try:
        document = json.loads(line)
        if type(document) is not dict or canonical_json_bytes(document) + b"\n" != line:
            raise ValueError
        return cast(dict[str, object], document)
    except (json.JSONDecodeError, TypeError, ValueError, UnicodeError):
        raise ValueError("chunk manifest is invalid") from None


def _validate_manifest_header(document: dict[str, object], expected: ManifestExpectation) -> None:
    created_at = (
        expected.document_version_created_at.astimezone(UTC).isoformat().replace("+00:00", "Z")
    )
    wanted = {
        "schema_version": CHUNK_SCHEMA_VERSION,
        "source_checksum_sha256": expected.source_checksum_sha256,
        "parsed_checksum_sha256": expected.parsed_checksum_sha256,
        "parser": {
            "name": expected.parser_name,
            "version": expected.parser_version,
            "config": _thaw_manifest_json(expected.parser_config),
        },
        "chunker": {
            "name": expected.chunker_name,
            "version": expected.chunker_version,
            "config": _thaw_manifest_json(expected.chunker_config),
        },
        "document_version_created_at": created_at,
        "chunk_count": expected.chunk_count,
    }
    chunk_hash = canonical_sha256(
        {
            "config": _thaw_manifest_json(expected.chunker_config),
            "name": expected.chunker_name,
            "version": expected.chunker_version,
        }
    )
    if document != wanted or chunk_hash != expected.chunk_config_hash:
        raise ValueError("chunk manifest is invalid")


def _manifest_chunk(document: dict[str, object], chunk_index: int) -> Chunk:
    try:
        if set(document) != {
            "chunk_index",
            "text",
            "chunk_hash",
            "start_offset",
            "end_offset",
            "title_path",
            "metadata",
        }:
            raise ValueError
        titles = document["title_path"]
        metadata = document["metadata"]
        if type(titles) is not list or type(metadata) is not dict:
            raise ValueError
        chunk = Chunk(
            chunk_index=cast(int, document["chunk_index"]),
            text=cast(str, document["text"]),
            chunk_hash=cast(str, document["chunk_hash"]),
            start_offset=cast(int, document["start_offset"]),
            end_offset=cast(int, document["end_offset"]),
            title_path=tuple(cast(list[str], titles)),
            metadata=cast(dict[str, object], metadata),
        )
        if chunk.chunk_index != chunk_index:
            raise ValueError
        return chunk
    except (KeyError, TypeError, ValueError):
        raise ValueError("chunk manifest is invalid") from None


async def iter_verified_manifest_batches(
    stream: AsyncIterable[bytes],
    *,
    expected: ManifestExpectation,
    next_chunk_index: int,
    batch_size: int,
) -> AsyncGenerator[tuple[Chunk, ...], None]:
    """Strictly parse canonical JSONL and group only the unfinished exclusive suffix."""

    if (
        not hasattr(stream, "__aiter__")
        or type(expected) is not ManifestExpectation
        or type(next_chunk_index) is not int
        or not 0 <= next_chunk_index <= expected.chunk_count
        or type(batch_size) is not int
        or not 1 <= batch_size <= 10_000
    ):
        raise ValueError("chunk manifest is invalid")
    line_number = 0
    batch: list[Chunk] = []
    lines = _manifest_lines(stream)
    try:
        async for line in lines:
            document = _manifest_document(line)
            if line_number == 0:
                _validate_manifest_header(document, expected)
            else:
                chunk = _manifest_chunk(document, line_number - 1)
                if chunk.chunk_index >= next_chunk_index:
                    batch.append(chunk)
                    if len(batch) == batch_size:
                        yield tuple(batch)
                        batch.clear()
            line_number += 1
            if line_number > expected.chunk_count + 1:
                raise ValueError("chunk manifest is invalid")
        if line_number != expected.chunk_count + 1:
            raise ValueError("chunk manifest is invalid")
        if batch:
            yield tuple(batch)
    finally:
        await close_async_iterators_without_masking_primary(lines)


async def _single_chunk(value: bytes) -> AsyncIterator[bytes]:
    yield value


class _OwnedCpuExecutor:
    """Admit at most ``max_concurrency`` submitted CPU calls and drain them explicitly."""

    def __init__(self, *, max_concurrency: int) -> None:
        if type(max_concurrency) is not int or max_concurrency < 1:
            raise ValueError("CPU concurrency is invalid")
        self._admission = asyncio.Semaphore(max_concurrency)
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrency,
            thread_name_prefix="rag-ingestion-cpu",
        )
        self._active: set[asyncio.Future[object]] = set()
        self._closed = False
        self._shutdown = False

    def _settled(self, future: asyncio.Future[object]) -> None:
        self._active.discard(future)
        with suppress(asyncio.CancelledError, Exception):
            future.exception()
        self._admission.release()

    async def run[Result](self, operation: Callable[[], Result]) -> Result:
        if not callable(operation):
            raise ValueError("CPU operation is invalid")
        await self._admission.acquire()
        if self._closed:
            self._admission.release()
            raise RuntimeError("CPU executor is closed")
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self._executor, operation)
        tracked = cast(asyncio.Future[object], future)
        self._active.add(tracked)
        tracked.add_done_callback(self._settled)
        return await asyncio.shield(future)

    async def aclose(self) -> None:
        self._closed = True
        cancelled = False
        while self._active:
            waiter = asyncio.gather(*tuple(self._active), return_exceptions=True)
            try:
                await asyncio.shield(waiter)
            except asyncio.CancelledError:
                cancelled = True
        if not self._shutdown:
            self._shutdown = True
            self._executor.shutdown(wait=True, cancel_futures=False)
        if cancelled:
            raise asyncio.CancelledError


class _ManifestBudgetExceeded(Exception):
    pass


def _analyze_chunk_manifest(
    *,
    source_checksum_sha256: str,
    parsed: ParsedArtifact,
    chunker: RecursiveTextChunker,
    document_version_created_at: datetime,
    max_bytes: int,
) -> tuple[int, int]:
    """Return exact chunk count/manifest bytes in O(1) memory before object upload."""

    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("chunk manifest budget is invalid")
    chunk_count = 0
    chunk_bytes = 0
    chunks = iter(chunker.chunk(parsed))
    try:
        for chunk in chunks:
            line_size = len(chunk_manifest_record_bytes(chunk))
            if line_size > max_bytes - chunk_bytes:
                raise _ManifestBudgetExceeded
            chunk_bytes += line_size
            chunk_count += 1
    finally:
        closer = getattr(chunks, "close", None)
        if callable(closer):
            closer()
    if chunk_count < 1:
        raise ValueError("ingestion stage state is invalid")
    header_size = len(
        chunk_manifest_header_bytes(
            source_checksum_sha256=source_checksum_sha256,
            parsed=parsed,
            chunker=chunker,
            document_version_created_at=document_version_created_at,
            chunk_count=chunk_count,
        )
    )
    if header_size > max_bytes - chunk_bytes:
        raise _ManifestBudgetExceeded
    return chunk_count, header_size + chunk_bytes


class _ManifestBatchReader:
    """Pull a synchronous manifest iterator in bounded worker-thread batches."""

    def __init__(self, lines: Iterator[bytes], batch_bytes: int = 64 * 1024) -> None:
        if type(batch_bytes) is not int or batch_bytes <= 0:
            raise ValueError("manifest batch size is invalid")
        self._lines = lines
        self._batch_bytes = batch_bytes
        self._pending: bytes | None = None
        self._pending_offset = 0

    def next_batch(self) -> bytes:
        parts: list[bytes] = []
        size = 0
        while True:
            if self._pending is not None:
                line = self._pending
                line_offset = self._pending_offset
            else:
                try:
                    line = next(self._lines)
                except StopIteration:
                    break
                line_offset = 0
            if type(line) is not bytes or not line:
                raise ValueError("chunk manifest is invalid")
            remaining = self._batch_bytes - size
            available = len(line) - line_offset
            take = min(available, remaining)
            parts.append(line[line_offset : line_offset + take])
            size += take
            if take < available:
                self._pending = line
                self._pending_offset = line_offset + take
                break
            self._pending = None
            self._pending_offset = 0
            if size >= self._batch_bytes:
                break
        return b"".join(parts)

    def close(self) -> None:
        closer = getattr(self._lines, "close", None)
        if callable(closer):
            closer()


async def _settle_cancelled_cpu_task(task: asyncio.Task[object]) -> None:
    """Drain one owned CPU call even when cleanup receives repeated cancellation."""

    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
        except BaseException:
            break
    if task.done() and not task.cancelled():
        with suppress(BaseException):
            task.result()


async def _run_cpu_call[Result](
    executor: _OwnedCpuExecutor,
    operation: Callable[[], Result],
) -> Result:
    """Preserve cancellation while keeping iterator calls sequential."""

    task = asyncio.create_task(executor.run(operation))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await _settle_cancelled_cpu_task(cast(asyncio.Task[object], task))
        raise


async def _manifest_stream(
    lines: Iterator[bytes],
    cpu: _OwnedCpuExecutor | None = None,
) -> AsyncIterator[bytes]:
    owned_cpu = cpu is None
    executor = _OwnedCpuExecutor(max_concurrency=1) if cpu is None else cpu
    reader = _ManifestBatchReader(lines)
    try:
        while True:
            batch = await _run_cpu_call(executor, reader.next_batch)
            if not batch:
                break
            yield batch
    finally:
        try:
            await _run_cpu_call(executor, reader.close)
        finally:
            if owned_cpu:
                await executor.aclose()


def _plain_mapping(value: Mapping[str, object]) -> dict[str, object]:
    return dict(value)


def _parser_for(expected: ParseStageInput | ChunkStageInput) -> Parser:
    parser = parser_for_extension(expected.source_extension)
    if (
        expected.parser_name != parser.name
        or expected.parser_version != parser.version
        or expected.parser_config != _plain_mapping(parser.config)
    ):
        raise ValueError("ingestion stage state is invalid")
    return parser


def _chunker_for(
    expected: ChunkStageInput,
    defaults: tuple[int, int] = (DEFAULT_CHUNK_CODEPOINTS, DEFAULT_OVERLAP_CODEPOINTS),
) -> RecursiveTextChunker:
    if expected.chunker_name is None and expected.chunker_version is None:
        if expected.chunker_config:
            raise ValueError("ingestion stage state is invalid")
        # A first chunk stage takes the configured sizing; the resumed branch
        # below must instead reuse whatever the recorded manifest was built with.
        default_maximum, default_overlap = defaults
        return RecursiveTextChunker(
            max_chunk_codepoints=default_maximum,
            target_overlap_codepoints=default_overlap,
        )
    if (
        expected.chunker_name != RecursiveTextChunker.name
        or expected.chunker_version != RecursiveTextChunker.version
        or set(expected.chunker_config) != {"max_chunk_codepoints", "target_overlap_codepoints"}
    ):
        raise ValueError("ingestion stage state is invalid")
    maximum = expected.chunker_config["max_chunk_codepoints"]
    overlap = expected.chunker_config["target_overlap_codepoints"]
    if type(maximum) is not int or type(overlap) is not int:
        raise ValueError("ingestion stage state is invalid")
    return RecursiveTextChunker(
        max_chunk_codepoints=maximum,
        target_overlap_codepoints=overlap,
    )


def manifest_expectation(expected: EmbedIndexStageInput) -> ManifestExpectation:
    return ManifestExpectation(
        source_checksum_sha256=expected.source_checksum_sha256,
        parsed_checksum_sha256=expected.parsed_checksum_sha256,
        parser_name=expected.parser_name,
        parser_version=expected.parser_version,
        parser_config=expected.parser_config,
        chunker_name=expected.chunker_name,
        chunker_version=expected.chunker_version,
        chunker_config=expected.chunker_config,
        chunk_config_hash=expected.chunk_config_hash,
        document_version_created_at=expected.version_created_at,
        chunk_count=expected.chunk_count,
    )


_MISSING_METADATA = object()


def _metadata_value(metadata: Mapping[str, object], source_path: str) -> object:
    current: object = metadata
    for segment in source_path.split("."):
        if not isinstance(current, Mapping) or segment not in current:
            return _MISSING_METADATA
        current = current[segment]
    return current


def _valid_filter_value(field_type: str, value: object) -> bool:
    if field_type == "keyword":
        return type(value) is str and len(value) <= 4096 and "\x00" not in value
    if field_type == "integer":
        return type(value) is int and -(2**63) <= value <= 2**63 - 1
    if field_type == "float":
        if type(value) not in {int, float} or isinstance(value, bool):
            return False
        try:
            return math.isfinite(float(cast(int | float, value)))
        except OverflowError:
            return False
    if field_type == "boolean":
        return type(value) is bool
    if field_type == "datetime":
        return is_rfc3339_datetime(value)
    return False


def approved_filter_metadata(expected: EmbedIndexStageInput) -> dict[str, object]:
    fields = expected.filter_snapshot.get("fields")
    if type(fields) is not list:
        raise ValueError("ingestion stage state is invalid")
    approved: dict[str, object] = {}
    for field in fields:
        if type(field) is not dict:
            raise ValueError("ingestion stage state is invalid")
        source_path = field.get("source_path")
        field_type = field.get("type")
        payload_path = field.get("payload_path")
        if (
            type(source_path) is not str
            or type(field_type) is not str
            or type(payload_path) is not str
            or not payload_path.startswith("metadata.f_")
        ):
            raise ValueError("ingestion stage state is invalid")
        value = _metadata_value(expected.document_metadata, source_path)
        if value is _MISSING_METADATA:
            continue
        if not _valid_filter_value(field_type, value):
            raise ValueError("ingestion stage state is invalid")
        approved[payload_path.removeprefix("metadata.")] = value
    return cast(dict[str, object], json.loads(canonical_json_bytes(approved)))


def qdrant_point_for(
    expected: EmbedIndexStageInput,
    chunk: Chunk,
    vector: tuple[float, ...],
    metadata: dict[str, object],
) -> QdrantPoint:
    if len(vector) != expected.gateway_snapshot.dimension:
        raise ValueError("embedding result is invalid")
    return QdrantPoint(
        id=point_id(expected.version_id, chunk.chunk_index, chunk.chunk_hash),
        vector=vector,
        payload=qdrant_point_payload(expected, chunk, metadata),
    )


def qdrant_point_payload(
    expected: EmbedIndexStageInput,
    chunk: Chunk,
    metadata: dict[str, object],
) -> dict[str, object]:
    return {
        "knowledge_base_id": str(expected.knowledge_base_id),
        "document_id": str(expected.document_id),
        "version_id": str(expected.version_id),
        "chunk_index": chunk.chunk_index,
        "chunk_hash": chunk.chunk_hash,
        "text": chunk.text,
        "title_path": list(chunk.title_path),
        "start_offset": chunk.start_offset,
        "end_offset": chunk.end_offset,
        "metadata": metadata,
    }


def provider_usage_observer(
    sink: PipelineProviderUsageSink,
    context: ProviderUsageContext,
) -> EmbeddingAttemptObserver:
    async def observe(attempt: EmbeddingAttempt) -> None:
        try:
            await sink.record(context, attempt)
        except Exception:
            # Provider usage is intentionally fail-open and uses an independent
            # bounded transaction. Cancellation still escapes.
            return

    return observe


class IngestionPipeline:
    """Execute artifact stages while PostgreSQL remains the visibility authority."""

    def __init__(
        self,
        *,
        repository_context: PipelineRepositoryContextFactory,
        object_store: PipelineObjectStore,
        max_document_bytes: int,
        cpu_concurrency: int = 1,
        max_manifest_bytes: int = MAX_CHUNK_MANIFEST_BYTES,
        chunk_max_codepoints: int = DEFAULT_CHUNK_CODEPOINTS,
        chunk_overlap_codepoints: int = DEFAULT_OVERLAP_CODEPOINTS,
        embedding_gateway: PipelineEmbeddingGateway | None = None,
        qdrant: QdrantClient | None = None,
        provider_usage_sink: PipelineProviderUsageSink | None = None,
        hooks: IngestionPipelineHooks | None = None,
        metrics: OperationalMetrics = METRICS,
        monotonic_clock: Callable[[], float] = time.monotonic,
        reconciliation_clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if (
            not callable(repository_context)
            or not callable(reconciliation_clock)
            or type(max_document_bytes) is not int
            or max_document_bytes <= 0
            or type(max_manifest_bytes) is not int
            or not 1 <= max_manifest_bytes <= _MAX_S3_OBJECT_BYTES
        ):
            raise ValueError("ingestion pipeline configuration is invalid")
        self._repository_context = repository_context
        self._object_store = object_store
        self._max_document_bytes = max_document_bytes
        self._cpu = _OwnedCpuExecutor(max_concurrency=cpu_concurrency)
        # An absolute service budget prevents generated-artifact amplification
        # without rejecting safe high-density inputs based on a source multiplier.
        self._max_manifest_bytes = max_manifest_bytes
        # Reject invalid sizing at construction rather than on the first document.
        RecursiveTextChunker(
            max_chunk_codepoints=chunk_max_codepoints,
            target_overlap_codepoints=chunk_overlap_codepoints,
        )
        self._chunk_defaults = (chunk_max_codepoints, chunk_overlap_codepoints)
        self._embedding_gateway = embedding_gateway
        self._qdrant = qdrant
        self._provider_usage_sink = provider_usage_sink
        self._hooks = IngestionPipelineHooks() if hooks is None else hooks
        self._metrics = metrics
        self._monotonic_clock = monotonic_clock
        self._reconciliation_clock = reconciliation_clock

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

    @staticmethod
    def _failure_code(stage: str, code: str | None) -> str:
        if code in {
            "PARSE_FAILED",
            "CHUNK_FAILED",
            "PROVIDER_ERROR",
            "PROVIDER_RATE_LIMITED",
            "PROVIDER_TIMEOUT",
            "QDRANT_UPSERT_FAILED",
            "VALIDATION_FAILED",
            "ACTIVATION_FAILED",
            "LEASE_LOST",
        }:
            return code
        if code is not None and code.startswith("PROVIDER_"):
            return "PROVIDER_ERROR"
        if code is not None and code.startswith("QDRANT_"):
            return "QDRANT_UPSERT_FAILED"
        return {
            "parse": "PARSE_FAILED",
            "chunk": "CHUNK_FAILED",
            "validate": "VALIDATION_FAILED",
            "activate": "ACTIVATION_FAILED",
        }.get(stage, "OTHER")

    @staticmethod
    def _populate_stage_ids(value: object) -> None:
        observation = _CURRENT_STAGE_OBSERVATION.get()
        if observation is None:
            return
        knowledge_base_id = getattr(value, "knowledge_base_id", None)
        document_id = getattr(value, "document_id", None)
        if type(knowledge_base_id) is UUID:
            observation.knowledge_base_id = knowledge_base_id
        if type(document_id) is UUID:
            observation.document_id = document_id

    def _observe_stage(
        self,
        *,
        context: JobExecutionContext,
        stage: str | None,
        outcome: str,
        started_at: float | None,
        observation: _StageObservation,
        failure_code: str | None = None,
    ) -> None:
        if stage not in {"parse", "chunk", "embed_index", "validate", "activate"}:
            return
        duration_seconds = self._duration_since(started_at)
        safe_failure_code = self._failure_code(stage, failure_code) if outcome == "failed" else None
        with suppress(BaseException):
            self._metrics.record_stage(
                stage=stage,
                outcome=outcome,
                duration_seconds=duration_seconds,
                failure_code=safe_failure_code,
                character_count=observation.character_count,
                chunk_count=observation.chunk_count,
                batch_count=observation.batch_count,
            )
        fields: dict[str, object] = {
            "operation": stage,
            "stage": stage,
            "outcome": outcome,
            "duration_seconds": duration_seconds,
            "character_count": observation.character_count,
            "chunk_count": observation.chunk_count,
            "batch_count": observation.batch_count,
        }
        if safe_failure_code is not None:
            fields["error_code"] = safe_failure_code
        with suppress(BaseException):
            emit_safe_log(
                logger,
                logging.INFO,
                "ingestion.stage.completed" if outcome == "succeeded" else "ingestion.stage.failed",
                context=SafeLogContext(
                    knowledge_base_id=observation.knowledge_base_id,
                    document_id=observation.document_id,
                    version_id=(
                        context.lease.target_id
                        if context.lease.target_type == "document_version"
                        else None
                    ),
                    job_id=context.lease.id,
                    generation_id=context.lease.index_generation_id,
                ),
                **fields,
            )

    def _observe_qdrant_upsert(
        self,
        *,
        context: JobExecutionContext,
        expected: EmbedIndexStageInput,
        outcome: str,
        point_count: int,
    ) -> None:
        with suppress(BaseException):
            self._metrics.record_qdrant_upsert(outcome=outcome, point_count=point_count)
        fields: dict[str, object] = {
            "operation": "qdrant_upsert",
            "outcome": outcome,
            "point_count": point_count,
        }
        if outcome == "failed":
            fields["error_code"] = "QDRANT_UPSERT_FAILED"
        with suppress(BaseException):
            emit_safe_log(
                logger,
                logging.INFO,
                "qdrant.upsert.completed",
                context=SafeLogContext(
                    knowledge_base_id=expected.knowledge_base_id,
                    document_id=expected.document_id,
                    version_id=expected.version_id,
                    job_id=context.lease.id,
                    generation_id=expected.generation_id,
                ),
                **fields,
            )

    async def reconcile_orphan_objects(
        self,
        *,
        grace_period: timedelta,
        limit: int,
        cursor: ObjectReconciliationCursor | None,
    ) -> ObjectReconciliationResult:
        if (
            type(grace_period) is not timedelta
            or grace_period <= timedelta(0)
            or type(limit) is not int
            or not 1 <= limit <= 1000
            or (cursor is not None and type(cursor) is not ObjectReconciliationCursor)
        ):
            raise ValueError("object reconciliation request is invalid")
        now = self._reconciliation_clock()
        if type(now) is not datetime or now.tzinfo is None or now.utcoffset() != UTC.utcoffset(now):
            raise ValueError("object reconciliation clock is invalid")
        phase = 0 if cursor is None else cursor.phase
        start_after = None if cursor is None else cursor.start_after
        page = await self._object_store.list_older_than(
            prefix=_OBJECT_RECONCILIATION_PREFIXES[phase],
            older_than=now - grace_period,
            limit=limit,
            start_after=start_after,
        )
        deleted_count = 0
        failed_count = 0
        for candidate in page.items:
            object_key = candidate.object_key
            if phase == 1 and not _is_canonical_final_object_key(object_key):
                continue
            try:
                await self._hooks.reached("before_orphan_object_reference_check")
                async with self._repository_context() as repository:
                    if not await repository.object_key_cleanup_is_allowed(object_key):
                        continue
                    deleted = await self._object_store.delete_best_effort(object_key)
            except asyncio.CancelledError:
                raise
            except BaseException:
                failed_count += 1
                continue
            if deleted:
                deleted_count += 1
            else:
                failed_count += 1

        phase_name = "temporary" if phase == 0 else "canonical"
        emit_safe_log(
            logger,
            logging.WARNING if failed_count else logging.INFO,
            "cleanup.action.completed",
            operation="orphan_cleanup",
            phase=phase_name,
            outcome="failed" if failed_count else "succeeded",
            count=failed_count if failed_count else deleted_count,
            candidate_count=len(page.items),
        )

        if page.next_start_after is not None:
            next_cursor = ObjectReconciliationCursor(phase, page.next_start_after)
        elif phase + 1 < len(_OBJECT_RECONCILIATION_PREFIXES):
            next_cursor = ObjectReconciliationCursor(phase + 1)
        else:
            next_cursor = None
        return ObjectReconciliationResult(
            scanned_count=len(page.items),
            deleted_count=deleted_count,
            failed_count=failed_count,
            next_cursor=next_cursor,
        )

    async def _load_parse(self, context: JobExecutionContext) -> ParseStageInput:
        generation_id = context.lease.index_generation_id
        if generation_id is None:
            raise ValueError("ingestion stage state is invalid")
        async with self._repository_context() as repository:
            return await repository.load_parse_stage(context.lease.target_id, generation_id)

    async def _load_chunk(self, context: JobExecutionContext) -> ChunkStageInput:
        generation_id = context.lease.index_generation_id
        if generation_id is None:
            raise ValueError("ingestion stage state is invalid")
        async with self._repository_context() as repository:
            return await repository.load_chunk_stage(context.lease.target_id, generation_id)

    async def _load_embed_index(self, context: JobExecutionContext) -> EmbedIndexStageInput:
        generation_id = context.lease.index_generation_id
        if generation_id is None:
            raise ValueError("ingestion stage state is invalid")
        async with self._repository_context() as repository:
            return await repository.load_embed_index_stage(
                context.lease.target_id,
                generation_id,
                context.lease.id,
            )

    async def _load_validate(self, context: JobExecutionContext) -> EmbedIndexStageInput:
        generation_id = context.lease.index_generation_id
        if generation_id is None:
            raise ValueError("ingestion stage state is invalid")
        async with self._repository_context() as repository:
            return await repository.load_validate_stage(
                context.lease.target_id,
                generation_id,
                context.lease.id,
            )

    async def _load_activation(self, context: JobExecutionContext) -> ActivationStageInput:
        generation_id = context.lease.index_generation_id
        if generation_id is None:
            raise ValueError("ingestion stage state is invalid")
        async with self._repository_context() as repository:
            return await repository.load_activation_stage(
                context.lease.target_id,
                generation_id,
                context.lease.id,
            )

    async def _cleanup_temp(self, object_key: str) -> None:
        with suppress(Exception):
            await self._object_store.delete_best_effort(object_key)

    async def _parse_stage(self, context: JobExecutionContext) -> None:
        expected = await self._load_parse(context)
        self._populate_stage_ids(expected)
        if expected.version_status not in {"uploaded", "parsing"}:
            raise ValueError("ingestion stage state is invalid")
        if expected.version_status == "uploaded":

            async def commit_started(session: AsyncSession) -> None:
                await SqlAlchemyIngestionPipelineRepository(session).commit_parse_started(expected)

            await context.commit_stage_facts(commit_started)
            expected = replace(expected, version_status="parsing")
        parser = _parser_for(expected)
        source = await self._object_store.read_bytes(
            expected.source_object_key,
            expected_checksum=expected.source_checksum_sha256,
            max_bytes=self._max_document_bytes,
        )
        parsed = await self._cpu.run(lambda: parser.parse(source))
        observation = _CURRENT_STAGE_OBSERVATION.get()
        if observation is not None:
            observation.character_count = len(parsed.text)
        if (
            parsed.parser_name != expected.parser_name
            or parsed.parser_version != expected.parser_version
            or _plain_mapping(parsed.parser_config) != expected.parser_config
        ):
            raise ValueError("ingestion stage state is invalid")
        temp_key = temporary_object_key(
            context.lease.id,
            context.lease.lease_epoch,
            "parsed/text.txt",
        )
        final_key = parsed_text_object_key(
            expected.knowledge_base_id,
            expected.document_id,
            expected.version_id,
        )
        try:
            stored = await self._object_store.upload_stream(
                temp_key,
                _single_chunk(parsed.normalized_bytes),
                content_type="text/plain; charset=utf-8",
                max_bytes=self._max_document_bytes,
            )
            published = await self._object_store.publish_temp(
                temp_key,
                final_key,
                expected_size=stored.size,
                expected_checksum=stored.checksum_sha256,
            )

            async def commit(session: AsyncSession) -> None:
                await SqlAlchemyIngestionPipelineRepository(session).commit_parse_stage(
                    expected,
                    parsed_object_key=published.object_key,
                    parsed_checksum_sha256=published.checksum_sha256,
                    parser_name=parsed.parser_name,
                    parser_version=parsed.parser_version,
                    parser_config=_plain_mapping(parsed.parser_config),
                )

            await context.advance_stage(
                commit,
                stage="chunk",
                resume_stage="chunk",
                progress_current=0,
                progress_total=None,
            )
        finally:
            await self._cleanup_temp(temp_key)

    async def _chunk_stage(self, context: JobExecutionContext) -> None:
        expected = await self._load_chunk(context)
        self._populate_stage_ids(expected)
        if expected.version_status != "chunking":
            raise ValueError("ingestion stage state is invalid")
        parser = _parser_for(expected)
        parsed_bytes = await self._object_store.read_bytes(
            expected.parsed_object_key,
            expected_checksum=expected.parsed_object_checksum_sha256,
            max_bytes=self._max_document_bytes,
        )
        parsed: ParsedArtifact = await self._cpu.run(lambda: parser.parse(parsed_bytes))
        if parsed.checksum_sha256 != expected.parsed_object_checksum_sha256:
            raise ValueError("ingestion stage state is invalid")
        chunker = _chunker_for(expected, self._chunk_defaults)
        try:
            chunk_count, _manifest_size = await self._cpu.run(
                lambda: _analyze_chunk_manifest(
                    source_checksum_sha256=expected.source_checksum_sha256,
                    parsed=parsed,
                    chunker=chunker,
                    document_version_created_at=expected.version_created_at,
                    max_bytes=self._max_manifest_bytes,
                )
            )
        except _ManifestBudgetExceeded:
            raise PermanentJobError(
                "ARTIFACT_TOO_LARGE",
                "Generated artifact exceeds service limit",
            ) from None
        observation = _CURRENT_STAGE_OBSERVATION.get()
        if observation is not None:
            observation.chunk_count = chunk_count
        lines = iter_chunk_manifest_lines(
            source_checksum_sha256=expected.source_checksum_sha256,
            parsed=parsed,
            chunker=chunker,
            document_version_created_at=expected.version_created_at,
            chunk_count=chunk_count,
            chunks=chunker.chunk(parsed),
        )
        temp_key = temporary_object_key(
            context.lease.id,
            context.lease.lease_epoch,
            "chunks/recursive_text_v1.jsonl",
        )
        final_key = chunks_object_key(
            expected.knowledge_base_id,
            expected.document_id,
            expected.version_id,
        )
        try:
            try:
                stored = await self._object_store.upload_stream(
                    temp_key,
                    _manifest_stream(lines, self._cpu),
                    content_type="application/x-ndjson; charset=utf-8",
                    max_bytes=self._max_manifest_bytes,
                )
            except UploadLimitExceeded:
                raise PermanentJobError(
                    "ARTIFACT_TOO_LARGE",
                    "Generated artifact exceeds object storage limit",
                ) from None
            published = await self._object_store.publish_temp(
                temp_key,
                final_key,
                expected_size=stored.size,
                expected_checksum=stored.checksum_sha256,
            )

            async def commit(session: AsyncSession) -> None:
                await SqlAlchemyIngestionPipelineRepository(session).commit_chunk_stage(
                    expected,
                    manifest_object_key=published.object_key,
                    manifest_checksum_sha256=published.checksum_sha256,
                    chunk_config_hash=chunker.config_hash,
                    chunker_name=chunker.name,
                    chunker_version=chunker.version,
                    chunker_config=_plain_mapping(chunker.config),
                    chunk_count=chunk_count,
                )

            await context.advance_stage(
                commit,
                stage="embed_index",
                resume_stage="embed_index",
                progress_current=0,
                progress_total=chunk_count,
            )
        finally:
            await self._cleanup_temp(temp_key)

    async def _verified_manifest_batches(
        self,
        expected: EmbedIndexStageInput,
        *,
        next_chunk_index: int,
    ) -> AsyncGenerator[tuple[Chunk, ...], None]:
        stream = self._object_store.read_stream(
            expected.manifest_object_key,
            expected_checksum=expected.manifest_checksum_sha256,
            max_bytes=self._max_manifest_bytes,
        )
        batches = iter_verified_manifest_batches(
            stream,
            expected=manifest_expectation(expected),
            next_chunk_index=next_chunk_index,
            batch_size=expected.operational.batch_size,
        )
        try:
            async for batch in batches:
                yield batch
        finally:
            await close_async_iterators_without_masking_primary(batches, stream)

    async def _preflight_manifest(self, expected: EmbedIndexStageInput) -> None:
        async for _batch in self._verified_manifest_batches(
            expected,
            next_chunk_index=expected.chunk_count,
        ):
            raise AssertionError("verified manifest preflight yielded an impossible batch")

    async def _commit_embed_progress(
        self,
        context: JobExecutionContext,
        expected: EmbedIndexStageInput,
        *,
        next_chunk_index: int,
        final: bool,
    ) -> None:
        async def commit(session: AsyncSession) -> None:
            await SqlAlchemyIngestionPipelineRepository(session).commit_embed_index_batch(
                expected,
                next_chunk_index=next_chunk_index,
            )

        if final:
            await context.advance_stage(
                commit,
                stage="validate",
                resume_stage="validate",
                progress_current=expected.chunk_count,
                progress_total=expected.chunk_count,
            )
        else:
            await context.commit_stage_checkpoint(
                commit,
                progress_current=next_chunk_index,
                progress_total=expected.chunk_count,
            )

    async def _embed_index_stage(self, context: JobExecutionContext) -> None:
        if (
            self._embedding_gateway is None
            or self._qdrant is None
            or self._provider_usage_sink is None
        ):
            raise PermanentJobError("JOB_STAGE_UNAVAILABLE", "Job stage is unavailable")
        embedding_gateway = self._embedding_gateway
        qdrant = self._qdrant
        usage_sink = self._provider_usage_sink
        expected = await self._load_embed_index(context)
        self._populate_stage_ids(expected)
        await self._preflight_manifest(expected)
        approved_metadata = approved_filter_metadata(expected)
        if expected.next_chunk_index == expected.chunk_count:
            await self._commit_embed_progress(
                context,
                expected,
                next_chunk_index=expected.chunk_count,
                final=True,
            )
            return

        batches = self._verified_manifest_batches(
            expected,
            next_chunk_index=expected.next_chunk_index,
        )
        try:
            async for batch in batches:
                batch_start = batch[0].chunk_index
                usage_context = ProviderUsageContext(
                    request_id=f"{context.lease.id}:embed:{batch_start}",
                    actor_api_key_id=expected.actor_api_key_id,
                    provider_config_id=expected.provider_config_id,
                    model_profile_id=expected.model_profile_id,
                )

                try:
                    result = await embedding_gateway.embed(
                        snapshot=expected.gateway_snapshot,
                        operational=expected.operational,
                        inputs=tuple(chunk.text for chunk in batch),
                        attempt_observer=provider_usage_observer(usage_sink, usage_context),
                    )
                except EmbeddingGatewayError as error:
                    if error.retryable:
                        raise RetryableProviderJobError(
                            error.code,
                            str(error),
                            provider_type=expected.gateway_snapshot.provider_type,
                        ) from None
                    raise
                await self._hooks.reached("embed_index.after_provider")
                if len(result.vectors) != len(batch):
                    raise ValueError("embedding result is invalid")
                points = tuple(
                    qdrant_point_for(expected, chunk, vector, approved_metadata)
                    for chunk, vector in zip(batch, result.vectors, strict=True)
                )
                try:
                    await qdrant.upsert_points(expected.qdrant_collection_name, points)
                except asyncio.CancelledError:
                    self._observe_qdrant_upsert(
                        context=context,
                        expected=expected,
                        outcome="cancelled",
                        point_count=len(points),
                    )
                    raise
                except BaseException:
                    self._observe_qdrant_upsert(
                        context=context,
                        expected=expected,
                        outcome="failed",
                        point_count=len(points),
                    )
                    raise
                self._observe_qdrant_upsert(
                    context=context,
                    expected=expected,
                    outcome="succeeded",
                    point_count=len(points),
                )
                await self._hooks.reached("embed_index.after_upsert")
                next_chunk_index = batch[-1].chunk_index + 1
                final = next_chunk_index == expected.chunk_count
                await self._commit_embed_progress(
                    context,
                    expected,
                    next_chunk_index=next_chunk_index,
                    final=final,
                )
                observation = _CURRENT_STAGE_OBSERVATION.get()
                if observation is not None:
                    observation.batch_count += 1
                if final:
                    return
                await self._hooks.reached("embed_index.after_checkpoint")
                expected = replace(
                    expected,
                    version_status="indexing",
                    index_state_status="indexing",
                    index_state_embedding_config_hash=expected.embedding_config_hash,
                    next_chunk_index=next_chunk_index,
                )
        finally:
            await close_async_iterators_without_masking_primary(batches)

    async def _validate_stage(self, context: JobExecutionContext) -> None:
        if self._qdrant is None:
            raise PermanentJobError("JOB_STAGE_UNAVAILABLE", "Job stage is unavailable")
        expected = await self._load_validate(context)
        self._populate_stage_ids(expected)
        if (
            expected.version_status != "indexing"
            or expected.index_state_status != "indexing"
            or expected.next_chunk_index != expected.chunk_count
            or expected.index_state_embedding_config_hash != expected.embedding_config_hash
        ):
            raise ValueError("ingestion stage state is invalid")

        actual_count = await self._qdrant.count_version_points(
            expected.qdrant_collection_name,
            expected.version_id,
        )
        await self._hooks.reached("validate.after_count")

        async def commit_count(session: AsyncSession) -> None:
            await SqlAlchemyIngestionPipelineRepository(session).commit_validate_count(
                expected,
                actual_count=actual_count,
            )

        await context.commit_stage_facts(commit_count)
        await self._hooks.reached("validate.after_count_commit")
        if actual_count != expected.chunk_count:
            raise PermanentJobError("INDEX_VALIDATION_FAILED", "Index validation failed")

        approved_metadata = approved_filter_metadata(expected)
        batches = self._verified_manifest_batches(expected, next_chunk_index=0)
        try:
            async for batch in batches:
                expected_ids = tuple(
                    point_id(expected.version_id, chunk.chunk_index, chunk.chunk_hash)
                    for chunk in batch
                )
                try:
                    actual = await self._qdrant.retrieve_version_points(
                        expected.qdrant_collection_name,
                        expected.version_id,
                        expected_ids,
                    )
                except QdrantConfigurationError:
                    raise PermanentJobError(
                        "INDEX_VALIDATION_FAILED",
                        "Index validation failed",
                    ) from None
                for chunk, inspected in zip(batch, actual, strict=True):
                    expected_id = point_id(
                        expected.version_id,
                        chunk.chunk_index,
                        chunk.chunk_hash,
                    )
                    expected_payload_digest = canonical_sha256(
                        qdrant_point_payload(expected, chunk, approved_metadata)
                    )
                    if (
                        inspected.id != expected_id
                        or inspected.vector_dimension != expected.gateway_snapshot.dimension
                        or inspected.payload_digest_sha256 != expected_payload_digest
                    ):
                        raise PermanentJobError(
                            "INDEX_VALIDATION_FAILED",
                            "Index validation failed",
                        )
                await self._hooks.reached("validate.after_retrieve")
                observation = _CURRENT_STAGE_OBSERVATION.get()
                if observation is not None:
                    observation.batch_count += 1
        finally:
            await close_async_iterators_without_masking_primary(batches)

        async def commit_validated(session: AsyncSession) -> None:
            await SqlAlchemyIngestionPipelineRepository(session).commit_validate_stage(
                expected,
                actual_count=actual_count,
            )

        await context.advance_stage(
            commit_validated,
            stage="activate",
            resume_stage="activate",
            progress_current=expected.chunk_count,
            progress_total=expected.chunk_count,
        )

    async def _activate_stage(self, context: JobExecutionContext) -> None:
        expected = await self._load_activation(context)
        self._populate_stage_ids(expected)

        async def commit(session: AsyncSession) -> None:
            await SqlAlchemyIngestionPipelineRepository(session).commit_activation(
                expected,
                context.lease,
            )

        await context.finalize_domain(commit)

    async def _dispatch_stage(self, context: JobExecutionContext) -> None:
        if (
            context.lease.operation != "ingest_document"
            or context.lease.target_type != "document_version"
        ):
            raise PermanentJobError("INGESTION_STAGE_CONFLICT", "Ingestion stage conflict")
        try:
            if context.lease.stage == "parse":
                await self._parse_stage(context)
            elif context.lease.stage == "chunk":
                await self._chunk_stage(context)
            elif context.lease.stage == "embed_index":
                await self._embed_index_stage(context)
            elif context.lease.stage == "validate":
                await self._validate_stage(context)
            elif context.lease.stage == "activate":
                await self._activate_stage(context)
            else:
                raise PermanentJobError("JOB_STAGE_UNAVAILABLE", "Job stage is unavailable")
        except ArtifactChecksumConflict:
            raise PermanentJobError(
                "ARTIFACT_CHECKSUM_CONFLICT",
                "Artifact checksum conflict",
            ) from None
        except ObjectStoreError as error:
            error_type = RetryableJobError if error.retryable else PermanentJobError
            raise error_type(error.code, str(error)) from None
        except DocumentValidationError as error:
            raise PermanentJobError(error.code, str(error)) from None
        except StructuralMetadataLimitExceeded:
            raise PermanentJobError(
                "STRUCTURAL_METADATA_TOO_LARGE",
                "Document structural metadata exceeds limit",
            ) from None
        except EmbeddingGatewayError as error:
            error_type = RetryableJobError if error.retryable else PermanentJobError
            raise error_type(error.code, str(error)) from None
        except QdrantTransientError:
            raise RetryableJobError("QDRANT_UNAVAILABLE", "Qdrant unavailable") from None
        except QdrantConfigurationError:
            raise PermanentJobError(
                "QDRANT_CONFIGURATION_CONFLICT",
                "Qdrant configuration conflict",
            ) from None
        except DocumentActivationConflictError:
            raise PermanentJobError(
                "DOCUMENT_ACTIVATION_CONFLICT",
                "Document activation conflict",
            ) from None
        except (TypeError, ValueError, UnicodeError):
            raise PermanentJobError(
                "INGESTION_STAGE_CONFLICT",
                "Ingestion stage conflict",
            ) from None

    async def _terminal_failure(
        self,
        context: JobExecutionContext,
        error: JobExecutionError,
    ) -> None:
        async def commit(session: AsyncSession) -> None:
            await SqlAlchemyIngestionPipelineRepository(session).commit_terminal_failure(
                context.lease,
                retryable=error.retryable,
                error_code=error.code,
                safe_error_message=error.safe_message,
            )

        await context.finalize_domain(commit)

    async def finalize_exhausted(
        self,
        candidate: ExhaustedJob,
        session: AsyncSession,
    ) -> None:
        """Atomically terminalize an expired final-attempt ingestion snapshot."""

        await SqlAlchemyIngestionPipelineRepository(session).commit_exhausted_failure(candidate)

    async def handle(self, context: JobExecutionContext) -> JobHandlerOutcome:
        """Dispatch one current stage; the runner loops CONTINUE under the same lease."""
        stage = context.lease.stage
        started_at = self._started_at()
        observation = _StageObservation()
        token = _CURRENT_STAGE_OBSERVATION.set(observation)
        terminal_error: JobExecutionError | None = None
        try:
            try:
                await self._dispatch_stage(context)
            except JobExecutionError as error:
                terminal = (
                    not error.retryable or context.lease.attempt_count >= context.lease.max_attempts
                )
                if not terminal or not context.domain_finalization_enabled:
                    raise
                await self._terminal_failure(context, error)
                terminal_error = error
                result = JobHandlerOutcome.COMPLETE
            else:
                result = (
                    JobHandlerOutcome.COMPLETE if context.finalized else JobHandlerOutcome.CONTINUE
                )
        except asyncio.CancelledError:
            self._observe_stage(
                context=context,
                stage=stage,
                outcome="cancelled",
                started_at=started_at,
                observation=observation,
            )
            raise
        except JobExecutionError as error:
            self._observe_stage(
                context=context,
                stage=stage,
                outcome="failed",
                started_at=started_at,
                observation=observation,
                failure_code=error.code,
            )
            raise
        except LostLeaseError:
            self._observe_stage(
                context=context,
                stage=stage,
                outcome="failed",
                started_at=started_at,
                observation=observation,
                failure_code="LEASE_LOST",
            )
            raise
        except BaseException:
            self._observe_stage(
                context=context,
                stage=stage,
                outcome="failed",
                started_at=started_at,
                observation=observation,
            )
            raise
        else:
            self._observe_stage(
                context=context,
                stage=stage,
                outcome="failed" if terminal_error is not None else "succeeded",
                started_at=started_at,
                observation=observation,
                failure_code=terminal_error.code if terminal_error is not None else None,
            )
            return result
        finally:
            _CURRENT_STAGE_OBSERVATION.reset(token)

    async def aclose(self) -> None:
        """Stop admitting CPU work and drain every submitted call before shutdown."""

        await self._cpu.aclose()


__all__ = [
    "IngestionPipeline",
    "IngestionPipelineHooks",
    "IngestionPipelineRepository",
    "MAX_CHUNK_MANIFEST_BYTES",
    "ManifestExpectation",
    "ObjectReconciliationCursor",
    "ObjectReconciliationResult",
    "PipelineObjectStore",
    "PipelineEmbeddingGateway",
    "PipelineProviderUsageSink",
    "PipelineRepositoryContextFactory",
    "approved_filter_metadata",
    "close_async_iterators_without_masking_primary",
    "iter_verified_manifest_batches",
    "manifest_expectation",
    "provider_usage_observer",
    "qdrant_point_for",
    "qdrant_point_payload",
]
