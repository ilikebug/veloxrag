from __future__ import annotations

import hashlib
import json
import math
import tracemalloc
from collections.abc import Generator, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import cast
from uuid import UUID

import pytest

from rag_service.ingestion.artifacts import (
    canonical_jsonl_bytes,
    chunks_object_key,
    iter_chunk_manifest_lines,
    parsed_text_object_key,
    source_object_key,
    temporary_object_key,
)
from rag_service.ingestion.chunkers import Chunk, RecursiveTextChunker
from rag_service.ingestion.parsers import MarkdownTextParser, ParsedArtifact, PlainTextParser
from rag_service.ingestion.validation import (
    MAX_DOCUMENT_BYTES,
    DocumentValidationError,
    IncrementalTextValidator,
    validate_document,
    validate_document_metadata,
)

FIXTURES = Path(__file__).parents[1] / "fixtures" / "documents"
KB_ID = UUID("11111111-1111-4111-8111-111111111111")
DOCUMENT_ID = UUID("22222222-2222-4222-8222-222222222222")
VERSION_ID = UUID("33333333-3333-4333-8333-333333333333")
JOB_ID = UUID("44444444-4444-4444-8444-444444444444")


class _CloseProbeIterator(Iterator[Chunk]):
    def __init__(
        self,
        values: tuple[Chunk, ...],
        *,
        raise_at: int | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self._values = values
        self._raise_at = raise_at
        self._close_error = close_error
        self._index = 0
        self.closed = False

    def __next__(self) -> Chunk:
        if self._raise_at == self._index:
            raise RuntimeError("probe iterator failed")
        if self._index >= len(self._values):
            raise StopIteration
        value = self._values[self._index]
        self._index += 1
        return value

    def close(self) -> None:
        self.closed = True
        if self._close_error is not None:
            raise self._close_error


class _CloseProbeChunker:
    name = "probe_chunker_v1"
    version = "1"
    config: Mapping[str, object] = MappingProxyType({})

    def __init__(self, plan: _CloseProbeIterator) -> None:
        self.plan = plan
        self.call_count = 0

    def chunk(self, artifact: ParsedArtifact) -> Iterator[Chunk]:
        del artifact
        self.call_count += 1
        return self.plan


@pytest.mark.parametrize(
    ("filename", "content_type", "fixture_name", "expected_extension"),
    [
        (" notes.TXT ", " Text/Plain ; charset=UTF-8 ", "plain.txt", ".txt"),
        ("guide.md", "text/markdown; charset=utf-8", "structured.md", ".md"),
        ("guide.MARKDOWN", "TEXT/X-MARKDOWN", "structured.md", ".markdown"),
        ("guide.md", "application/octet-stream", "structured.md", ".md"),
    ],
)
def test_text_document_validation_canonicalizes_filename_extension_and_mime(
    filename: str,
    content_type: str,
    fixture_name: str,
    expected_extension: str,
) -> None:
    content = (FIXTURES / fixture_name).read_bytes()

    document = validate_document(
        content,
        filename=filename,
        content_type=content_type,
    )

    assert document.display_filename == filename.strip()
    assert document.extension == expected_extension
    assert document.content_type in {
        "application/octet-stream",
        "text/markdown",
        "text/plain",
        "text/x-markdown",
    }
    assert document.source_size == len(content)
    assert len(document.source_checksum_sha256) == 64
    assert len(document.parsed_checksum_sha256) == 64
    assert document.normalized_text.strip()


def test_text_document_validation_removes_only_a_leading_bom_and_normalizes_newlines() -> None:
    document = validate_document(
        b"\xef\xbb\xbfheading\r\nbody\rtail\n\xef\xbb\xbfkept",
        filename="document.txt",
        content_type="text/plain",
    )

    assert document.normalized_text == "heading\nbody\ntail\n\ufeffkept"
    assert document.normalized_bytes == "heading\nbody\ntail\n\ufeffkept".encode()


def test_incremental_validator_handles_split_utf8_bom_and_code_points() -> None:
    validator = IncrementalTextValidator(
        filename="café.md",
        content_type="text/markdown",
        max_bytes=64,
    )

    validator.feed(b"\xef")
    validator.feed(b"\xbb\xbfCaf\xc3")
    validator.feed(b"\xa9\r")
    validator.feed(b"\n")
    summary = validator.finish()

    assert summary.source_size == 10
    assert len(summary.source_checksum_sha256) == 64
    assert summary.has_text is True


def test_exact_upload_limit_is_accepted_and_the_first_extra_byte_is_rejected() -> None:
    exact = b"a" * MAX_DOCUMENT_BYTES
    accepted = validate_document(
        exact,
        filename="large.txt",
        content_type="text/plain",
    )

    assert accepted.source_size == MAX_DOCUMENT_BYTES

    with pytest.raises(DocumentValidationError) as exc_info:
        validate_document(
            exact + b"a",
            filename="large.txt",
            content_type="text/plain",
        )

    assert exc_info.value.code == "FILE_TOO_LARGE"


@pytest.mark.parametrize(
    ("content", "expected_code"),
    [
        (b"", "EMPTY_DOCUMENT"),
        (b" \r\n\t ", "EMPTY_DOCUMENT"),
        (b"\xff\xfeinvalid", "INVALID_TEXT_ENCODING"),
        (b"hello\x00world", "BINARY_CONTENT_REJECTED"),
        (b"\x01", "BINARY_CONTENT_REJECTED"),
        (b"\x01\x02\x03\x04text", "BINARY_CONTENT_REJECTED"),
    ],
)
def test_text_document_validation_rejects_invalid_or_non_text_content(
    content: bytes,
    expected_code: str,
) -> None:
    with pytest.raises(DocumentValidationError) as exc_info:
        validate_document(
            content,
            filename="document.txt",
            content_type="text/plain",
        )

    assert exc_info.value.code == expected_code
    assert exc_info.value.retryable is False


def test_text_document_validation_accepts_form_feed_page_breaks() -> None:
    document = validate_document(
        b"Page one\fPage two\fPage three",
        filename="scanned-export.txt",
        content_type="text/plain",
    )

    assert document.normalized_text == "Page one\fPage two\fPage three"


@pytest.mark.parametrize(
    ("filename", "content_type"),
    [
        ("document.pdf", "application/pdf"),
        ("document.txt", "application/pdf"),
        ("../document.txt", "text/plain"),
        ("folder/document.md", "text/markdown"),
        ("folder\\document.md", "text/markdown"),
        (".md", "text/markdown"),
        ("document.md\x00.txt", "text/plain"),
        ("invoice\u202etxt.md", "text/markdown"),
        ("document.txt", "not a mime"),
    ],
)
def test_document_metadata_rejects_unsupported_or_unsafe_names_and_types(
    filename: str,
    content_type: str,
) -> None:
    with pytest.raises(DocumentValidationError) as exc_info:
        validate_document_metadata(filename=filename, content_type=content_type)

    assert exc_info.value.code == "UNSUPPORTED_DOCUMENT_TYPE"
    surface = f"{exc_info.value!s} {exc_info.value!r}"
    assert filename not in surface
    assert content_type not in surface


def test_document_error_surface_is_stable_and_does_not_retain_sensitive_input() -> None:
    secret = "customer-secret-content"

    with pytest.raises(DocumentValidationError) as exc_info:
        validate_document(
            secret.encode() + b"\x00",
            filename="private-customer-secret.txt",
            content_type="text/plain",
        )

    error = exc_info.value
    assert error.code == "BINARY_CONTENT_REJECTED"
    assert error.args == ("Document content is not supported text",)
    assert secret not in str(error)
    assert secret not in repr(error)
    assert "private-customer-secret" not in repr(error)


def test_all_required_public_document_error_codes_are_constructible() -> None:
    required = {
        "FILE_TOO_LARGE",
        "UNSUPPORTED_DOCUMENT_TYPE",
        "INVALID_TEXT_ENCODING",
        "BINARY_CONTENT_REJECTED",
        "EMPTY_DOCUMENT",
        "DUPLICATE_DOCUMENT",
    }

    assert {DocumentValidationError(code).code for code in required} == required


def test_object_key_constructors_use_only_server_controlled_path_segments() -> None:
    source = source_object_key(
        KB_ID,
        DOCUMENT_ID,
        VERSION_ID,
        filename="quarterly report.MD",
    )

    prefix = (
        "knowledge-bases/11111111-1111-4111-8111-111111111111/"
        "documents/22222222-2222-4222-8222-222222222222/"
        "versions/33333333-3333-4333-8333-333333333333"
    )
    assert source == f"{prefix}/source/source.md"
    assert parsed_text_object_key(KB_ID, DOCUMENT_ID, VERSION_ID) == f"{prefix}/parsed/text.txt"
    assert chunks_object_key(KB_ID, DOCUMENT_ID, VERSION_ID) == (
        f"{prefix}/chunks/recursive_text_v1.jsonl"
    )
    assert "quarterly report" not in source
    assert temporary_object_key(JOB_ID, 7, "parsed/text.txt") == (
        "tmp/jobs/44444444-4444-4444-8444-444444444444/7/parsed/text.txt"
    )


@pytest.mark.parametrize(
    "artifact_name",
    [
        "",
        "/parsed.txt",
        "../parsed.txt",
        "parsed/../text.txt",
        "parsed\\text.txt",
        "parsed/unsafe\nname.txt",
        ".",
    ],
)
def test_temporary_object_key_rejects_unsafe_artifact_names(artifact_name: str) -> None:
    with pytest.raises(ValueError, match="artifact name is invalid"):
        temporary_object_key(JOB_ID, 1, artifact_name)


def test_temporary_object_key_rejects_a_full_utf8_key_over_1024_bytes() -> None:
    artifact_name = "/".join("界" * 80 for _ in range(5))
    assert len(artifact_name) <= 512
    assert all(len(segment.encode("utf-8")) <= 255 for segment in artifact_name.split("/"))

    with pytest.raises(ValueError, match="artifact name is invalid"):
        temporary_object_key(JOB_ID, 1, artifact_name)


def test_canonical_jsonl_has_stable_utf8_field_order_and_exact_lf_framing() -> None:
    records = [
        {"schema_version": "chunks-v1", "chunk_count": 1},
        {
            "text": "你好",
            "chunk_index": 0,
            "metadata": {},
            "title_path": ["标题"],
            "start_offset": 0,
            "end_offset": 2,
            "chunk_hash": "a" * 64,
        },
    ]

    encoded = canonical_jsonl_bytes(records)

    assert encoded == (
        b'{"chunk_count":1,"schema_version":"chunks-v1"}\n'
        b'{"chunk_hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
        b'"chunk_index":0,"end_offset":2,"metadata":{},"start_offset":0,'
        b'"text":"\xe4\xbd\xa0\xe5\xa5\xbd","title_path":["\xe6\xa0\x87\xe9\xa2\x98"]}\n'
    )


@pytest.mark.parametrize(
    "records",
    [
        ({"not": "a list"},),
        [MappingProxyType({"a": 1})],
        [{"value": math.nan}],
        [{"value": math.inf}],
        [{"value": (1, 2)}],
        [{1: "non-string-key"}],
    ],
)
def test_canonical_jsonl_rejects_noncanonical_values(records: object) -> None:
    with pytest.raises(ValueError, match="canonical JSONL"):
        canonical_jsonl_bytes(records)  # type: ignore[arg-type]


def test_chunk_manifest_is_byte_deterministic_with_stable_version_time_and_checksums() -> None:
    source = "# 标题\n\n你好 🚚\n".encode()
    source_checksum = hashlib.sha256(source).hexdigest()
    parsed = MarkdownTextParser().parse(source)
    # Pinned rather than defaulted: the golden bytes below cover manifest
    # serialization, so they should not need rewriting whenever the shipped
    # chunk sizing changes.
    chunker = RecursiveTextChunker(max_chunk_codepoints=1200, target_overlap_codepoints=150)
    chunks = tuple(chunker.chunk(parsed))
    created_at = datetime(2026, 7, 26, 1, 2, 3, 456789, tzinfo=UTC)

    first = b"".join(
        iter_chunk_manifest_lines(
            source_checksum_sha256=source_checksum,
            parsed=parsed,
            chunker=chunker,
            document_version_created_at=created_at,
            chunk_count=len(chunks),
            chunks=iter(chunks),
        )
    )
    second = b"".join(
        iter_chunk_manifest_lines(
            source_checksum_sha256=source_checksum,
            parsed=parsed,
            chunker=chunker,
            document_version_created_at=created_at,
            chunk_count=len(chunks),
            chunks=iter(chunks),
        )
    )

    assert first == second
    assert first == (
        b'{"chunk_count":1,"chunker":{"config":{"max_chunk_codepoints":1200,'
        b'"target_overlap_codepoints":150},"name":"recursive_text_v1","version":"1"},'
        b'"document_version_created_at":"2026-07-26T01:02:03.456789Z",'
        b'"parsed_checksum_sha256":"'
        + parsed.checksum_sha256.encode()
        + b'","parser":{"config":{},"name":"markdown_text_v1","version":"1"},'
        b'"schema_version":"chunks-v1","source_checksum_sha256":"'
        + source_checksum.encode()
        + b'"}\n'
        b'{"chunk_hash":"'
        + chunks[0].chunk_hash.encode()
        + b'","chunk_index":0,"end_offset":11,"metadata":{},"start_offset":0,'
        b'"text":"# \xe6\xa0\x87\xe9\xa2\x98\\n\\n\xe4\xbd\xa0\xe5\xa5\xbd \xf0\x9f\x9a\x9a\\n",'
        b'"title_path":["\xe6\xa0\x87\xe9\xa2\x98"]}\n'
    )
    assert b"2026-07-29" not in first
    assert first.endswith(b"\n")
    assert b"\r" not in first
    assert b": " not in first and b", " not in first


def test_chunk_manifest_streams_boundary_dense_compact_artifacts_without_index_expansion() -> None:
    source = b"x\n\n" * ((256 * 1024) // 3)
    parsed = PlainTextParser().parse(source)
    chunker = RecursiveTextChunker()
    chunks = tuple(chunker.chunk(parsed))

    tracemalloc.start()
    try:
        line_count = sum(
            1
            for _line in iter_chunk_manifest_lines(
                source_checksum_sha256=hashlib.sha256(source).hexdigest(),
                parsed=parsed,
                chunker=chunker,
                document_version_created_at=datetime(2026, 7, 29, tzinfo=UTC),
                chunk_count=len(chunks),
                chunks=iter(chunks),
            )
        )
        _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert line_count == len(chunks) + 1
    assert peak_bytes <= 4 * len(source)


def test_chunk_manifest_yields_header_before_consuming_chunks_and_streams_records() -> None:
    source = b"small document"
    parsed = MarkdownTextParser().parse(source)
    chunker = RecursiveTextChunker(max_chunk_codepoints=8, target_overlap_codepoints=2)
    chunks = tuple(chunker.chunk(parsed))
    consumed: list[int] = []

    def observed_chunks() -> Iterator[Chunk]:
        for chunk in chunks:
            consumed.append(chunk.chunk_index)
            yield chunk

    lines = iter_chunk_manifest_lines(
        source_checksum_sha256=hashlib.sha256(source).hexdigest(),
        parsed=parsed,
        chunker=chunker,
        document_version_created_at=datetime(2026, 1, 1, tzinfo=UTC),
        chunk_count=len(chunks),
        chunks=observed_chunks(),
    )

    header = next(lines)
    assert consumed == []
    assert header.startswith(b'{"chunk_count":')
    first_chunk_line = next(lines)
    assert consumed == [0]
    assert first_chunk_line.startswith(b'{"chunk_hash":')
    assert list(lines)
    assert consumed == list(range(len(chunks)))


@pytest.mark.parametrize(
    ("created_at", "source_checksum", "chunk_count"),
    [
        (datetime(2026, 1, 1), "a" * 64, 1),
        (datetime(2026, 1, 1, tzinfo=UTC), "A" * 64, 1),
        (datetime(2026, 1, 1, tzinfo=UTC), "a" * 63, 1),
        (datetime(2026, 1, 1, tzinfo=UTC), "a" * 64, -1),
    ],
)
def test_chunk_manifest_rejects_unstable_or_invalid_header_values(
    created_at: datetime,
    source_checksum: str,
    chunk_count: int,
) -> None:
    parsed = MarkdownTextParser().parse(b"text")
    chunker = RecursiveTextChunker()

    with pytest.raises(ValueError, match="chunk manifest is invalid"):
        next(
            iter_chunk_manifest_lines(
                source_checksum_sha256=source_checksum,
                parsed=parsed,
                chunker=chunker,
                document_version_created_at=created_at,
                chunk_count=chunk_count,
                chunks=chunker.chunk(parsed),
            )
        )


def test_chunk_manifest_validates_declared_count_and_sequential_indices() -> None:
    source = b"one two three"
    parsed = MarkdownTextParser().parse(source)
    chunker = RecursiveTextChunker(max_chunk_codepoints=5, target_overlap_codepoints=1)
    chunks = tuple(chunker.chunk(parsed))

    lines = iter_chunk_manifest_lines(
        source_checksum_sha256=hashlib.sha256(source).hexdigest(),
        parsed=parsed,
        chunker=chunker,
        document_version_created_at=datetime(2026, 1, 1, tzinfo=UTC),
        chunk_count=len(chunks) + 1,
        chunks=iter(chunks),
    )

    with pytest.raises(ValueError, match="chunk manifest is invalid"):
        list(lines)


def _manifest_for(
    parsed: ParsedArtifact,
    chunks: tuple[Chunk, ...],
    *,
    chunk_count: int,
) -> Iterator[bytes]:
    return iter_chunk_manifest_lines(
        source_checksum_sha256="a" * 64,
        parsed=parsed,
        chunker=RecursiveTextChunker(),
        document_version_created_at=datetime(2026, 1, 1, tzinfo=UTC),
        chunk_count=chunk_count,
        chunks=iter(chunks),
    )


def test_chunk_manifest_rejects_chunk_text_forged_from_another_source() -> None:
    parsed = MarkdownTextParser().parse(b"abcdefghij")
    forged = Chunk.from_source(
        chunk_index=0,
        source_text="0123456789",
        start_offset=0,
        end_offset=10,
        title_path=(),
    )

    with pytest.raises(ValueError, match="chunk manifest is invalid"):
        list(_manifest_for(parsed, (forged,), chunk_count=1))


def test_chunk_manifest_rejects_zero_chunks_for_nonempty_parsed_text() -> None:
    parsed = MarkdownTextParser().parse(b"nonempty")

    with pytest.raises(ValueError, match="chunk manifest is invalid"):
        list(_manifest_for(parsed, (), chunk_count=0))


@pytest.mark.parametrize(
    "case",
    ["first_offset", "gap", "no_progress", "end_regression", "short_tail", "range"],
)
def test_chunk_manifest_rejects_invalid_source_coverage(case: str) -> None:
    parsed = MarkdownTextParser().parse(b"abcdefghij")
    source = parsed.text
    chunks: tuple[Chunk, ...]
    if case == "first_offset":
        chunks = (
            Chunk.from_source(
                chunk_index=0,
                source_text=source,
                start_offset=1,
                end_offset=10,
                title_path=(),
            ),
        )
    elif case == "gap":
        chunks = (
            Chunk.from_source(
                chunk_index=0,
                source_text=source,
                start_offset=0,
                end_offset=4,
                title_path=(),
            ),
            Chunk.from_source(
                chunk_index=1,
                source_text=source,
                start_offset=5,
                end_offset=10,
                title_path=(),
            ),
        )
    elif case == "no_progress":
        chunks = (
            Chunk.from_source(
                chunk_index=0,
                source_text=source,
                start_offset=0,
                end_offset=7,
                title_path=(),
            ),
            Chunk.from_source(
                chunk_index=1,
                source_text=source,
                start_offset=0,
                end_offset=10,
                title_path=(),
            ),
        )
    elif case == "end_regression":
        chunks = (
            Chunk.from_source(
                chunk_index=0,
                source_text=source,
                start_offset=0,
                end_offset=8,
                title_path=(),
            ),
            Chunk.from_source(
                chunk_index=1,
                source_text=source,
                start_offset=4,
                end_offset=7,
                title_path=(),
            ),
            Chunk.from_source(
                chunk_index=2,
                source_text=source,
                start_offset=6,
                end_offset=10,
                title_path=(),
            ),
        )
    elif case == "short_tail":
        chunks = (
            Chunk.from_source(
                chunk_index=0,
                source_text=source,
                start_offset=0,
                end_offset=9,
                title_path=(),
            ),
        )
    else:
        chunks = (
            Chunk.from_source(
                chunk_index=0,
                source_text=source + "k",
                start_offset=0,
                end_offset=11,
                title_path=(),
            ),
        )

    with pytest.raises(ValueError, match="chunk manifest is invalid"):
        list(_manifest_for(parsed, chunks, chunk_count=len(chunks)))


def test_chunk_manifest_rejects_title_paths_not_produced_by_declared_chunker() -> None:
    parsed = MarkdownTextParser().parse(b"# Correct\n\nbody")
    forged = Chunk.from_source(
        chunk_index=0,
        source_text=parsed.text,
        start_offset=0,
        end_offset=len(parsed.text),
        title_path=("Forged",),
    )

    with pytest.raises(ValueError, match="chunk manifest is invalid"):
        list(_manifest_for(parsed, (forged,), chunk_count=1))


def test_chunk_manifest_rejects_a_chunk_over_declared_chunker_hard_max() -> None:
    parsed = PlainTextParser().parse(("x" * 1300).encode())
    forged = Chunk.from_source(
        chunk_index=0,
        source_text=parsed.text,
        start_offset=0,
        end_offset=len(parsed.text),
        title_path=(),
    )

    with pytest.raises(ValueError, match="chunk manifest is invalid"):
        list(_manifest_for(parsed, (forged,), chunk_count=1))


def test_chunk_manifest_rejects_chunks_from_different_chunker_config() -> None:
    parsed = PlainTextParser().parse(b"abcdefghijklmnopqrstuvwxyz")
    supplied = tuple(
        RecursiveTextChunker(max_chunk_codepoints=10, target_overlap_codepoints=2).chunk(parsed)
    )

    with pytest.raises(ValueError, match="chunk manifest is invalid"):
        list(_manifest_for(parsed, supplied, chunk_count=len(supplied)))


def test_recursive_chunk_plan_round_trips_after_structural_overlap() -> None:
    source = ("x\n\n" + "A" * 700 + "\n\n" + "B" * 1600).encode()
    parsed = PlainTextParser().parse(source)
    chunker = RecursiveTextChunker()
    chunks = tuple(chunker.chunk(parsed))

    manifest = b"".join(
        iter_chunk_manifest_lines(
            source_checksum_sha256=hashlib.sha256(source).hexdigest(),
            parsed=parsed,
            chunker=chunker,
            document_version_created_at=datetime(2026, 1, 1, tzinfo=UTC),
            chunk_count=len(chunks),
            chunks=iter(chunks),
        )
    )

    assert manifest.count(b"\n") == len(chunks) + 1


def test_chunk_manifest_preserves_title_paths_across_three_headings() -> None:
    first_section = "# A\n\n" + "A" * 500 + "\n\n"
    second_section = "# B\n\n" + "B" * 501 + "\n\n"
    source = (first_section + second_section + "# C\n\n" + "C" * 1600).encode()
    parsed = MarkdownTextParser().parse(source)
    chunker = RecursiveTextChunker()
    chunks = tuple(chunker.chunk(parsed))

    manifest = b"".join(
        iter_chunk_manifest_lines(
            source_checksum_sha256=hashlib.sha256(source).hexdigest(),
            parsed=parsed,
            chunker=chunker,
            document_version_created_at=datetime(2026, 1, 1, tzinfo=UTC),
            chunk_count=len(chunks),
            chunks=iter(chunks),
        )
    )
    records = [json.loads(line) for line in manifest.splitlines()]

    assert [record["title_path"] for record in records[1:3]] == [["A"], ["B"]]
    assert records[1]["end_offset"] == 507
    assert records[2]["start_offset"] == 507
    assert records[2]["end_offset"] == 1015


def _close_probe_manifest(
    parsed: ParsedArtifact,
    *,
    expected: _CloseProbeIterator,
    actual: _CloseProbeIterator,
    chunk_count: int,
) -> Generator[bytes, None, None]:
    return cast(
        Generator[bytes, None, None],
        iter_chunk_manifest_lines(
            source_checksum_sha256="a" * 64,
            parsed=parsed,
            chunker=_CloseProbeChunker(expected),
            document_version_created_at=datetime(2026, 1, 1, tzinfo=UTC),
            chunk_count=chunk_count,
            chunks=actual,
        ),
    )


def test_chunk_manifest_closes_both_iterators_after_normal_exhaustion() -> None:
    parsed = PlainTextParser().parse(b"text")
    chunk = Chunk.from_source(
        chunk_index=0,
        source_text=parsed.text,
        start_offset=0,
        end_offset=len(parsed.text),
        title_path=(),
    )
    expected = _CloseProbeIterator((chunk,))
    actual = _CloseProbeIterator((chunk,))

    assert (
        len(list(_close_probe_manifest(parsed, expected=expected, actual=actual, chunk_count=1)))
        == 2
    )
    assert expected.closed is True
    assert actual.closed is True


def test_chunk_manifest_closes_both_iterators_after_plan_mismatch() -> None:
    parsed = PlainTextParser().parse(b"text")
    expected_chunk = Chunk.from_source(
        chunk_index=0,
        source_text=parsed.text,
        start_offset=0,
        end_offset=len(parsed.text),
        title_path=(),
    )
    forged_chunk = Chunk.from_source(
        chunk_index=0,
        source_text=parsed.text,
        start_offset=0,
        end_offset=len(parsed.text),
        title_path=("forged",),
    )
    expected = _CloseProbeIterator((expected_chunk,))
    actual = _CloseProbeIterator((forged_chunk,))

    with pytest.raises(ValueError, match="chunk manifest is invalid"):
        list(_close_probe_manifest(parsed, expected=expected, actual=actual, chunk_count=1))
    assert expected.closed is True
    assert actual.closed is True


def test_chunk_manifest_closes_both_iterators_when_consumer_closes_after_header() -> None:
    parsed = PlainTextParser().parse(b"text")
    chunk = Chunk.from_source(
        chunk_index=0,
        source_text=parsed.text,
        start_offset=0,
        end_offset=len(parsed.text),
        title_path=(),
    )
    expected = _CloseProbeIterator((chunk,))
    actual = _CloseProbeIterator((chunk,))
    lines = _close_probe_manifest(parsed, expected=expected, actual=actual, chunk_count=1)

    assert next(lines).startswith(b'{"chunk_count":')
    lines.close()

    assert expected.closed is True
    assert actual.closed is True


def test_chunk_manifest_close_before_first_next_owns_only_actual_iterator() -> None:
    parsed = PlainTextParser().parse(b"text")
    chunk = Chunk.from_source(
        chunk_index=0,
        source_text=parsed.text,
        start_offset=0,
        end_offset=len(parsed.text),
        title_path=(),
    )
    expected = _CloseProbeIterator((chunk,))
    actual = _CloseProbeIterator((chunk,))
    chunker = _CloseProbeChunker(expected)
    lines = iter_chunk_manifest_lines(
        source_checksum_sha256="a" * 64,
        parsed=parsed,
        chunker=chunker,
        document_version_created_at=datetime(2026, 1, 1, tzinfo=UTC),
        chunk_count=1,
        chunks=actual,
    )

    lines.close()

    assert actual.closed is True
    assert chunker.call_count == 0
    assert expected.closed is False


@pytest.mark.parametrize("failure", ["actual", "expected"])
def test_chunk_manifest_closes_both_iterators_when_a_plan_iterator_raises(
    failure: str,
) -> None:
    parsed = PlainTextParser().parse(b"text")
    chunk = Chunk.from_source(
        chunk_index=0,
        source_text=parsed.text,
        start_offset=0,
        end_offset=len(parsed.text),
        title_path=(),
    )
    expected = _CloseProbeIterator((chunk,), raise_at=0 if failure == "expected" else None)
    actual = _CloseProbeIterator((chunk,), raise_at=0 if failure == "actual" else None)

    with pytest.raises(RuntimeError, match="probe iterator failed"):
        list(_close_probe_manifest(parsed, expected=expected, actual=actual, chunk_count=1))
    assert expected.closed is True
    assert actual.closed is True


@pytest.mark.parametrize(
    ("failure", "expected_error", "expected_message"),
    [
        ("validation", ValueError, "chunk manifest is invalid"),
        ("iteration", RuntimeError, "probe iterator failed"),
    ],
)
def test_chunk_manifest_cleanup_errors_do_not_replace_the_primary_error(
    failure: str,
    expected_error: type[Exception],
    expected_message: str,
) -> None:
    parsed = PlainTextParser().parse(b"text")
    expected_chunk = Chunk.from_source(
        chunk_index=0,
        source_text=parsed.text,
        start_offset=0,
        end_offset=len(parsed.text),
        title_path=(),
    )
    actual_chunk = (
        Chunk.from_source(
            chunk_index=0,
            source_text=parsed.text,
            start_offset=0,
            end_offset=len(parsed.text),
            title_path=("forged",),
        )
        if failure == "validation"
        else expected_chunk
    )
    expected = _CloseProbeIterator(
        (expected_chunk,),
        close_error=RuntimeError("close-secret-expected"),
    )
    actual = _CloseProbeIterator(
        (actual_chunk,),
        raise_at=0 if failure == "iteration" else None,
        close_error=RuntimeError("close-secret-actual"),
    )

    with pytest.raises(expected_error, match=expected_message) as exc_info:
        list(_close_probe_manifest(parsed, expected=expected, actual=actual, chunk_count=1))

    assert "close-secret" not in str(exc_info.value)
    assert expected.closed is True
    assert actual.closed is True


def test_chunk_manifest_cleanup_error_is_not_hidden_by_an_outer_exception() -> None:
    parsed = PlainTextParser().parse(b"text")
    chunk = Chunk.from_source(
        chunk_index=0,
        source_text=parsed.text,
        start_offset=0,
        end_offset=len(parsed.text),
        title_path=(),
    )
    expected = _CloseProbeIterator(
        (chunk,),
        close_error=RuntimeError("close-secret-expected"),
    )
    actual = _CloseProbeIterator(
        (chunk,),
        close_error=RuntimeError("close-secret-actual"),
    )

    try:
        raise LookupError("unrelated outer error")
    except LookupError:
        with pytest.raises(RuntimeError, match="close-secret-expected"):
            list(_close_probe_manifest(parsed, expected=expected, actual=actual, chunk_count=1))

    assert expected.closed is True
    assert actual.closed is True


def test_chunk_manifest_header_checksum_has_bounded_peak_memory_for_large_text() -> None:
    def header_peak(character_count: int) -> int:
        parsed = ParsedArtifact("x" * character_count, (), "plain_text_v1", "1", {})
        expected = _CloseProbeIterator(())
        actual = _CloseProbeIterator(())
        lines = _close_probe_manifest(parsed, expected=expected, actual=actual, chunk_count=1)

        tracemalloc.start()
        next(lines)
        _current, peak = tracemalloc.get_traced_memory()
        lines.close()
        tracemalloc.stop()
        return peak

    small_peak = header_peak(100_000)
    large_peak = header_peak(1_000_000)

    assert large_peak < small_peak * 3
