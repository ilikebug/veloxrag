"""Deterministic object identities and canonical JSONL artifacts."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable
from uuid import UUID

from rag_service.indexing.identities import canonical_json_bytes
from rag_service.ingestion.chunkers import CHUNK_SCHEMA_VERSION, Chunk, Chunker
from rag_service.ingestion.parsers import ParsedArtifact
from rag_service.ingestion.validation import canonical_source_extension

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CHECKSUM_WINDOW_CODEPOINTS = 64 * 1024


@dataclass(frozen=True, slots=True)
class DocumentArtifactIdentity:
    knowledge_base_id: UUID
    document_id: UUID
    version_id: UUID
    artifact_type: str


def canonical_document_artifact_identity(object_key: str) -> DocumentArtifactIdentity | None:
    if type(object_key) is not str:
        return None
    parts = object_key.split("/")
    if (
        len(parts) != 8
        or parts[0] != "knowledge-bases"
        or parts[2] != "documents"
        or parts[4] != "versions"
    ):
        return None
    artifact_type, artifact_name = parts[6], parts[7]
    if not (
        (
            artifact_type == "source"
            and artifact_name in {"source.txt", "source.md", "source.markdown"}
        )
        or (artifact_type, artifact_name)
        in {("parsed", "text.txt"), ("chunks", "recursive_text_v1.jsonl")}
    ):
        return None
    try:
        knowledge_base_id = UUID(parts[1])
        document_id = UUID(parts[3])
        version_id = UUID(parts[5])
    except ValueError:
        return None
    if (
        str(knowledge_base_id) != parts[1]
        or str(document_id) != parts[3]
        or str(version_id) != parts[5]
    ):
        return None
    return DocumentArtifactIdentity(
        knowledge_base_id,
        document_id,
        version_id,
        artifact_type,
    )


@runtime_checkable
class _ClosableIterator(Protocol):
    def close(self) -> object: ...


def _version_prefix(
    knowledge_base_id: UUID,
    document_id: UUID,
    version_id: UUID,
) -> str:
    if (
        type(knowledge_base_id) is not UUID
        or type(document_id) is not UUID
        or type(version_id) is not UUID
    ):
        raise ValueError("document artifact identity is invalid")
    return f"knowledge-bases/{knowledge_base_id}/documents/{document_id}/versions/{version_id}"


def source_object_key(
    knowledge_base_id: UUID,
    document_id: UUID,
    version_id: UUID,
    *,
    filename: str,
) -> str:
    extension = canonical_source_extension(filename)
    return f"{_version_prefix(knowledge_base_id, document_id, version_id)}/source/source{extension}"


def parsed_text_object_key(
    knowledge_base_id: UUID,
    document_id: UUID,
    version_id: UUID,
) -> str:
    return f"{_version_prefix(knowledge_base_id, document_id, version_id)}/parsed/text.txt"


def chunks_object_key(
    knowledge_base_id: UUID,
    document_id: UUID,
    version_id: UUID,
) -> str:
    return (
        f"{_version_prefix(knowledge_base_id, document_id, version_id)}"
        "/chunks/recursive_text_v1.jsonl"
    )


def temporary_object_key(job_id: UUID, lease_epoch: int, artifact_name: str) -> str:
    """Build a fenced key from a safe relative artifact name owned by a pipeline stage."""

    if type(job_id) is not UUID or type(lease_epoch) is not int or lease_epoch < 0:
        raise ValueError("temporary artifact identity is invalid")
    if type(artifact_name) is not str or not artifact_name or len(artifact_name) > 512:
        raise ValueError("artifact name is invalid")
    if "\\" in artifact_name or artifact_name.startswith("/"):
        raise ValueError("artifact name is invalid")
    segments = artifact_name.split("/")
    if any(
        not segment
        or segment in {".", ".."}
        or "\x00" in segment
        or any(unicodedata.category(character).startswith("C") for character in segment)
        or len(segment.encode("utf-8")) > 255
        for segment in segments
    ):
        raise ValueError("artifact name is invalid")
    object_key = f"tmp/jobs/{job_id}/{lease_epoch}/{artifact_name}"
    if len(object_key.encode("utf-8")) > 1024:
        raise ValueError("artifact name is invalid")
    return object_key


def canonical_jsonl_bytes(records: list[dict[str, object]]) -> bytes:
    """Encode generic records canonically; artifact-specific builders own their schemas."""

    if (
        type(records) is not list
        or not records
        or any(type(record) is not dict for record in records)
    ):
        raise ValueError("value is not canonical JSONL")
    try:
        return b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    except (TypeError, ValueError, UnicodeError):
        raise ValueError("value is not canonical JSONL") from None


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError("chunk manifest is invalid")
        return {key: _plain_json(value[key]) for key in sorted(value)}
    if type(value) is list:
        return [_plain_json(member) for member in value]
    if type(value) is tuple:
        return [_plain_json(member) for member in value]
    return value


def _stable_utc_timestamp(value: datetime) -> str:
    if type(value) is not datetime or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("chunk manifest is invalid")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parsed_text_checksum_sha256(text: str) -> str:
    digest = hashlib.sha256()
    for start in range(0, len(text), _CHECKSUM_WINDOW_CODEPOINTS):
        digest.update(text[start : start + _CHECKSUM_WINDOW_CODEPOINTS].encode("utf-8"))
    return digest.hexdigest()


def _close_iterator(iterator: object | None) -> None:
    if isinstance(iterator, _ClosableIterator):
        iterator.close()


def _close_owned_iterators(
    expected: Iterator[Chunk] | None,
    actual: Iterator[Chunk] | None,
) -> BaseException | None:
    first_error: BaseException | None = None
    for iterator in (expected, actual if actual is not expected else None):
        try:
            _close_iterator(iterator)
        except BaseException as error:
            if first_error is None:
                first_error = error
    return first_error


def chunk_manifest_header_bytes(
    *,
    source_checksum_sha256: str,
    parsed: ParsedArtifact,
    chunker: Chunker,
    document_version_created_at: datetime,
    chunk_count: int,
) -> bytes:
    """Encode the canonical manifest header without retaining document content."""

    if (
        type(source_checksum_sha256) is not str
        or _SHA256_PATTERN.fullmatch(source_checksum_sha256) is None
        or not isinstance(parsed, ParsedArtifact)
        or not isinstance(chunker, Chunker)
        or type(chunk_count) is not int
        or chunk_count < 0
        or (bool(parsed.text) and chunk_count == 0)
    ):
        raise ValueError("chunk manifest is invalid")
    manifest = {
        "schema_version": CHUNK_SCHEMA_VERSION,
        "source_checksum_sha256": source_checksum_sha256,
        "parsed_checksum_sha256": _parsed_text_checksum_sha256(parsed.text),
        "parser": {
            "name": parsed.parser_name,
            "version": parsed.parser_version,
            "config": _plain_json(parsed.parser_config),
        },
        "chunker": {
            "name": chunker.name,
            "version": chunker.version,
            "config": _plain_json(chunker.config),
        },
        "document_version_created_at": _stable_utc_timestamp(document_version_created_at),
        "chunk_count": chunk_count,
    }
    try:
        return canonical_json_bytes(manifest) + b"\n"
    except (TypeError, ValueError, UnicodeError):
        raise ValueError("chunk manifest is invalid") from None


def chunk_manifest_record_bytes(chunk: Chunk) -> bytes:
    """Encode one canonical chunk line for streaming and exact budget analysis."""

    if type(chunk) is not Chunk:
        raise ValueError("chunk manifest is invalid")
    try:
        return canonical_json_bytes(chunk.as_record()) + b"\n"
    except (TypeError, ValueError, UnicodeError):
        raise ValueError("chunk manifest is invalid") from None


def _iter_chunk_manifest_lines(
    *,
    source_checksum_sha256: str,
    parsed: ParsedArtifact,
    chunker: Chunker,
    document_version_created_at: datetime,
    chunk_count: int,
    actual_chunks: Iterator[Chunk],
) -> Iterator[bytes]:
    expected_chunks: Iterator[Chunk] | None = None
    primary_error: BaseException | None = None
    try:
        expected_chunks = iter(chunker.chunk(parsed))
        yield chunk_manifest_header_bytes(
            source_checksum_sha256=source_checksum_sha256,
            parsed=parsed,
            chunker=chunker,
            document_version_created_at=document_version_created_at,
            chunk_count=chunk_count,
        )
        emitted = 0
        previous_start = -1
        previous_end = 0
        for chunk in actual_chunks:
            try:
                expected = next(expected_chunks)
            except StopIteration:
                raise ValueError("chunk manifest is invalid") from None
            if (
                type(chunk) is not Chunk
                or emitted >= chunk_count
                or chunk.chunk_index != emitted
                or chunk != expected
                or chunk.start_offset < 0
                or chunk.end_offset > len(parsed.text)
                or chunk.text != parsed.text[chunk.start_offset : chunk.end_offset]
                or (emitted == 0 and chunk.start_offset != 0)
                or (
                    emitted > 0
                    and (
                        chunk.start_offset <= previous_start
                        or chunk.start_offset > previous_end
                        or chunk.end_offset <= previous_end
                    )
                )
            ):
                raise ValueError("chunk manifest is invalid")
            yield chunk_manifest_record_bytes(chunk)
            previous_start = chunk.start_offset
            previous_end = chunk.end_offset
            emitted += 1
        try:
            next(expected_chunks)
        except StopIteration:
            expected_exhausted = True
        else:
            expected_exhausted = False
        if (
            emitted != chunk_count
            or not expected_exhausted
            or (emitted > 0 and previous_end != len(parsed.text))
        ):
            raise ValueError("chunk manifest is invalid")
    except (TypeError, ValueError, UnicodeError):
        primary_error = ValueError("chunk manifest is invalid")
        raise primary_error from None
    except BaseException as error:
        primary_error = error
        raise
    finally:
        cleanup_error = _close_owned_iterators(expected_chunks, actual_chunks)
        if cleanup_error is not None and (
            primary_error is None or isinstance(primary_error, GeneratorExit)
        ):
            raise cleanup_error


class _ChunkManifestLines(Iterator[bytes]):
    def __init__(
        self,
        *,
        source_checksum_sha256: str,
        parsed: ParsedArtifact,
        chunker: Chunker,
        document_version_created_at: datetime,
        chunk_count: int,
        actual_chunks: Iterator[Chunk],
    ) -> None:
        self._source_checksum_sha256 = source_checksum_sha256
        self._parsed = parsed
        self._chunker = chunker
        self._document_version_created_at = document_version_created_at
        self._chunk_count = chunk_count
        self._actual_chunks = actual_chunks
        self._lines: Iterator[bytes] | None = None
        self._closed = False

    def __next__(self) -> bytes:
        if self._closed:
            raise StopIteration
        if self._lines is None:
            self._lines = _iter_chunk_manifest_lines(
                source_checksum_sha256=self._source_checksum_sha256,
                parsed=self._parsed,
                chunker=self._chunker,
                document_version_created_at=self._document_version_created_at,
                chunk_count=self._chunk_count,
                actual_chunks=self._actual_chunks,
            )
        try:
            return next(self._lines)
        except BaseException:
            self._closed = True
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._lines is None:
            _close_iterator(self._actual_chunks)
        else:
            _close_iterator(self._lines)


def iter_chunk_manifest_lines(
    *,
    source_checksum_sha256: str,
    parsed: ParsedArtifact,
    chunker: Chunker,
    document_version_created_at: datetime,
    chunk_count: int,
    chunks: Iterable[Chunk],
) -> _ChunkManifestLines:
    """Yield canonical manifest and chunk JSONL records without buffering the artifact."""

    if not isinstance(chunks, Iterable):
        raise ValueError("chunk manifest is invalid")
    try:
        actual_chunks = iter(chunks)
    except (TypeError, ValueError):
        raise ValueError("chunk manifest is invalid") from None
    return _ChunkManifestLines(
        source_checksum_sha256=source_checksum_sha256,
        parsed=parsed,
        chunker=chunker,
        document_version_created_at=document_version_created_at,
        chunk_count=chunk_count,
        actual_chunks=actual_chunks,
    )


__all__ = [
    "canonical_jsonl_bytes",
    "chunk_manifest_header_bytes",
    "chunk_manifest_record_bytes",
    "chunks_object_key",
    "iter_chunk_manifest_lines",
    "parsed_text_object_key",
    "source_object_key",
    "temporary_object_key",
]
