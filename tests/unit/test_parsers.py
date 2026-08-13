from __future__ import annotations

import gc
import hashlib
import subprocess
import sys
import tracemalloc
from array import array
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType

import pytest

from rag_service.ingestion import parsers as parser_module
from rag_service.ingestion.chunkers import RecursiveTextChunker
from rag_service.ingestion.parsers import (
    BoundaryKind,
    MarkdownTextParser,
    ParsedArtifact,
    PlainTextParser,
    StructuralBoundaries,
    StructuralBoundary,
    parser_for_extension,
)
from rag_service.ingestion.validation import DocumentValidationError

FIXTURES = Path(__file__).parents[1] / "fixtures" / "documents"


def _fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _boundary_for_text(
    artifact: ParsedArtifact,
    source_text: str,
) -> StructuralBoundary:
    return next(
        boundary
        for boundary in artifact.boundaries
        if artifact.text[boundary.start_offset : boundary.end_offset] == source_text
    )


def test_parser_selection_and_descriptors_are_stable_and_immutable() -> None:
    plain = parser_for_extension(".txt")
    markdown = parser_for_extension(".MD")

    assert isinstance(plain, PlainTextParser)
    assert isinstance(markdown, MarkdownTextParser)
    assert parser_for_extension(".markdown") is markdown
    assert (plain.name, plain.version, dict(plain.config)) == (
        "plain_text_v1",
        "1",
        {},
    )
    assert (markdown.name, markdown.version, dict(markdown.config)) == (
        "markdown_text_v1",
        "1",
        {},
    )
    assert isinstance(plain.config, MappingProxyType)
    assert isinstance(markdown.config, MappingProxyType)

    with pytest.raises(TypeError):
        plain.config["unsafe"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="document parser extension is invalid"):
        parser_for_extension(".pdf")


def test_maximum_size_alternating_markdown_builds_a_compact_chunk_index() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from rag_service.ingestion.chunkers import RecursiveTextChunker; "
                "from rag_service.ingestion.parsers import MarkdownTextParser; "
                "source = b'#\\nx\\n' * ((50 * 1024 * 1024) // 4); "
                "artifact = MarkdownTextParser().parse(source); "
                "first_chunk = next(RecursiveTextChunker().chunk(artifact)); "
                "\nassert len(artifact.boundaries) == 26_214_400"
                "\nassert artifact.boundaries.storage_run_count <= 8"
                "\nassert artifact.boundaries[-1].end_offset == len(artifact.text) - 1"
                "\nassert (first_chunk.start_offset, first_chunk.end_offset) == (0, 4)"
                "\nprint('ok')"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "ok\n"


def test_scaled_alternating_markdown_streams_every_chunk_to_the_end() -> None:
    pair_count = 262_144
    artifact = MarkdownTextParser().parse(b"#\nx\n" * pair_count)
    chunk_count = 0
    last_start = -1
    last_offset = 0

    for chunk in RecursiveTextChunker().chunk(artifact):
        assert chunk.start_offset > last_start
        assert chunk.end_offset > last_offset
        assert len(chunk.text) <= 1200
        assert chunk.text == artifact.text[chunk.start_offset : chunk.end_offset]
        chunk_count += 1
        last_start = chunk.start_offset
        last_offset = chunk.end_offset

    assert chunk_count == pair_count
    assert last_offset == len(artifact.text)


def test_shortest_alternating_markdown_uses_an_exact_periodic_sequence() -> None:
    pair_count = 1_000_001
    artifact = MarkdownTextParser().parse(b"#\nx\n" * pair_count)

    assert len(artifact.boundaries) == pair_count * 2
    assert artifact.boundaries.storage_run_count <= 2
    assert artifact.boundaries[0] == StructuralBoundary("heading", 0, 1, (), 1)
    assert artifact.boundaries[1] == StructuralBoundary("paragraph", 2, 3, ())
    assert artifact.boundaries[-2] == StructuralBoundary(
        "heading", (pair_count - 1) * 4, (pair_count - 1) * 4 + 1, (), 1
    )
    assert artifact.boundaries[-1] == StructuralBoundary(
        "paragraph", (pair_count - 1) * 4 + 2, (pair_count - 1) * 4 + 3, ()
    )


def test_plain_text_fixture_has_golden_normalized_bytes_checksum_and_paragraphs() -> None:
    artifact = PlainTextParser().parse(_fixture("plain.txt"))
    expected_text = (
        "First paragraph has café and 你好.\n"
        "It continues with emoji 🚚.\n\n"
        "Second paragraph stays byte-for-byte.\n"
    )

    assert artifact.text == expected_text
    assert artifact.normalized_bytes == expected_text.encode("utf-8")
    assert artifact.checksum_sha256 == (
        "8bdaa610abcd648d683f229d25b08f81829545f97085b8552f16ac257382887c"
    )
    assert artifact.parser_name == "plain_text_v1"
    assert artifact.parser_version == "1"
    assert dict(artifact.parser_config) == {}
    assert artifact.boundaries == (
        StructuralBoundary(
            kind="paragraph",
            start_offset=0,
            end_offset=expected_text.index("\n\n"),
            title_path=(),
        ),
        StructuralBoundary(
            kind="paragraph",
            start_offset=expected_text.index("\n\n") + 2,
            end_offset=len(expected_text) - 1,
            title_path=(),
        ),
    )


def test_plain_text_normalizes_only_leading_bom_and_line_endings_with_codepoint_offsets() -> None:
    source = "\ufeffCafé 🚚\r\n你好\rsecond\n\nlast".encode()

    artifact = PlainTextParser().parse(source)

    assert artifact.text == "Café 🚚\n你好\nsecond\n\nlast"
    assert artifact.normalized_bytes == "Café 🚚\n你好\nsecond\n\nlast".encode()
    assert artifact.boundaries[0].end_offset == artifact.text.index("\n\n")
    assert artifact.boundaries[0].end_offset < len(
        artifact.text[: artifact.boundaries[0].end_offset].encode()
    )
    assert (
        artifact.text[artifact.boundaries[1].start_offset : artifact.boundaries[1].end_offset]
        == "last"
    )


def test_markdown_fixture_has_golden_text_checksum_and_structural_boundaries() -> None:
    source = _fixture("structured.md")
    artifact = MarkdownTextParser().parse(source)
    expected_text = source.decode()

    assert artifact.text == expected_text
    assert artifact.normalized_bytes == source
    assert artifact.checksum_sha256 == (
        "ce78feed2d4622d9b4a24553019f87cd3513e039e70fa99f087fbde687f62981"
    )
    assert artifact.parser_name == "markdown_text_v1"
    assert artifact.parser_version == "1"
    assert dict(artifact.parser_config) == {}
    assert [boundary.kind for boundary in artifact.boundaries] == [
        "heading",
        "paragraph",
        "heading",
        "list_item",
        "list_item",
        "fenced_code_block",
        "heading",
        "paragraph",
    ]

    expected = [
        ("# Shipment guide", ("Shipment guide",), 1),
        (
            "Intro keeps [carrier link](https://example.invalid/track) unchanged.",
            ("Shipment guide",),
            None,
        ),
        ("## Tracking", ("Shipment guide", "Tracking"), 2),
        ("- Keep the original message.", ("Shipment guide", "Tracking"), None),
        ("- Compare updates in order.", ("Shipment guide", "Tracking"), None),
        (
            '```python\nprint("not executed")\n```',
            ("Shipment guide", "Tracking"),
            None,
        ),
        ("### Notes", ("Shipment guide", "Tracking", "Notes"), 3),
        (
            '!INCLUDE "/tmp/should-not-be-read"\n<script>window.shouldNotRun = true;</script>',
            ("Shipment guide", "Tracking", "Notes"),
            None,
        ),
    ]
    assert [
        (
            artifact.text[boundary.start_offset : boundary.end_offset],
            boundary.title_path,
            boundary.heading_level,
        )
        for boundary in artifact.boundaries
    ] == expected


def test_markdown_offsets_are_unicode_codepoints_not_utf8_bytes() -> None:
    artifact = MarkdownTextParser().parse(
        "# 标题 🚚\r\n\r\n正文 café".encode(),
    )
    paragraph = _boundary_for_text(artifact, "正文 café")

    assert artifact.text == "# 标题 🚚\n\n正文 café"
    assert paragraph.start_offset == artifact.text.index("正文")
    assert paragraph.start_offset != len(artifact.text[: paragraph.start_offset].encode())


def test_markdown_setext_headings_have_full_spans_levels_and_hierarchical_title_paths() -> None:
    source = (
        "根标题 🚚\r\n"
        "=========\r\n\r\n"
        "Child café\r\n"
        "---\r\n\r\n"
        "Body text\r\n\r\n"
        "# Reset\r\n\r\n"
        "After reset"
    ).encode()

    artifact = MarkdownTextParser().parse(source)

    assert artifact.text == (
        "根标题 🚚\n=========\n\nChild café\n---\n\nBody text\n\n# Reset\n\nAfter reset"
    )
    assert [
        (
            boundary.kind,
            artifact.text[boundary.start_offset : boundary.end_offset],
            boundary.heading_level,
            boundary.title_path,
        )
        for boundary in artifact.boundaries
    ] == [
        ("heading", "根标题 🚚\n=========", 1, ("根标题 🚚",)),
        ("heading", "Child café\n---", 2, ("根标题 🚚", "Child café")),
        ("paragraph", "Body text", None, ("根标题 🚚", "Child café")),
        ("heading", "# Reset", 1, ("Reset",)),
        ("paragraph", "After reset", None, ("Reset",)),
    ]
    second_heading = artifact.boundaries[1]
    assert second_heading.start_offset == artifact.text.index("Child café")
    assert second_heading.start_offset != len(artifact.text[: second_heading.start_offset].encode())


def test_markdown_multiline_and_nested_bullet_and_ordered_items_are_logical_boundaries() -> None:
    source = (
        "# Lists\n\n"
        "- Parent bullet\n"
        "  continuation 🚚\n"
        "    - Nested bullet\n"
        "      nested continuation\n"
        "- Sibling bullet\n"
        "  lazy continuation\n"
        "1. Ordered first\n"
        "   ordered continuation\n"
        "   1. Nested ordered\n"
        "      nested ordered continuation\n"
        "2) Ordered second\n"
    )

    artifact = MarkdownTextParser().parse(source.encode())

    assert artifact.text == source
    assert [boundary.kind for boundary in artifact.boundaries] == [
        "heading",
        "list_item",
        "list_item",
        "list_item",
        "list_item",
        "list_item",
        "list_item",
    ]
    assert [
        artifact.text[boundary.start_offset : boundary.end_offset]
        for boundary in artifact.boundaries[1:]
    ] == [
        "- Parent bullet\n  continuation 🚚",
        "    - Nested bullet\n      nested continuation",
        "- Sibling bullet\n  lazy continuation",
        "1. Ordered first\n   ordered continuation",
        "   1. Nested ordered\n      nested ordered continuation",
        "2) Ordered second",
    ]
    assert all(boundary.title_path == ("Lists",) for boundary in artifact.boundaries[1:])


def test_markdown_loose_list_items_keep_indented_paragraphs_across_blank_lines() -> None:
    source = (
        "# Loose lists\n\n"
        "- first paragraph\n\n"
        "  second paragraph\n"
        "  continues here\n"
        "  - nested first paragraph\n\n"
        "    nested second paragraph\n"
        "- sibling item\n"
    )

    artifact = MarkdownTextParser().parse(source.encode())

    assert artifact.text == source
    assert [boundary.kind for boundary in artifact.boundaries] == [
        "heading",
        "list_item",
        "list_item",
        "list_item",
    ]
    assert [
        artifact.text[boundary.start_offset : boundary.end_offset]
        for boundary in artifact.boundaries[1:]
    ] == [
        "- first paragraph\n\n  second paragraph\n  continues here",
        "  - nested first paragraph\n\n    nested second paragraph",
        "- sibling item",
    ]
    assert all(boundary.title_path == ("Loose lists",) for boundary in artifact.boundaries[1:])


def test_markdown_four_space_indented_code_like_markers_need_list_context() -> None:
    source = "    - literal bullet text\n    1. literal ordered text\n"

    artifact = MarkdownTextParser().parse(source.encode())

    assert artifact.text == source
    assert artifact.boundaries == (
        StructuralBoundary(
            kind="paragraph",
            start_offset=0,
            end_offset=len(source) - 1,
            title_path=(),
        ),
    )


def test_markdown_heading_inside_list_is_a_nonoverlapping_structural_boundary() -> None:
    source = "# Outer 🚚\n\n- Parent item\n    ## Inner 🚚\n- Sibling under inner\n"

    artifact = MarkdownTextParser().parse(source.encode())

    assert artifact.text == source
    assert [
        (
            boundary.kind,
            artifact.text[boundary.start_offset : boundary.end_offset],
            boundary.heading_level,
            boundary.title_path,
        )
        for boundary in artifact.boundaries
    ] == [
        ("heading", "# Outer 🚚", 1, ("Outer 🚚",)),
        ("list_item", "- Parent item", None, ("Outer 🚚",)),
        ("heading", "    ## Inner 🚚", 2, ("Outer 🚚", "Inner 🚚")),
        (
            "list_item",
            "- Sibling under inner",
            None,
            ("Outer 🚚", "Inner 🚚"),
        ),
    ]
    inner = artifact.boundaries[2]
    assert inner.start_offset == artifact.text.index("    ## Inner")
    assert inner.start_offset != len(artifact.text[: inner.start_offset].encode())


def test_markdown_setext_heading_inside_list_uses_relative_indentation() -> None:
    source = (
        "# Outer\n\n"
        "- Parent item\n"
        "    Inner setext 🚚\n"
        "    ----------------\n"
        "- Sibling under setext\n"
    )

    artifact = MarkdownTextParser().parse(source.encode())

    assert artifact.text == source
    assert [
        (
            boundary.kind,
            artifact.text[boundary.start_offset : boundary.end_offset],
            boundary.heading_level,
            boundary.title_path,
        )
        for boundary in artifact.boundaries
    ] == [
        ("heading", "# Outer", 1, ("Outer",)),
        ("list_item", "- Parent item", None, ("Outer",)),
        (
            "heading",
            "    Inner setext 🚚\n    ----------------",
            2,
            ("Outer", "Inner setext 🚚"),
        ),
        (
            "list_item",
            "- Sibling under setext",
            None,
            ("Outer", "Inner setext 🚚"),
        ),
    ]


def test_markdown_multiline_setext_heading_inside_list_matches_top_level_semantics() -> None:
    source = (
        "# Outer\n\n"
        "- Parent item\n"
        "  First title 🚚 line\n"
        "  Second café line\n"
        "  -----------------\n"
        "- Sibling under multiline heading\n"
    )

    first = MarkdownTextParser().parse(source.encode())
    second = MarkdownTextParser().parse(source.encode())

    assert first.text == source
    assert [
        (
            boundary.kind,
            first.text[boundary.start_offset : boundary.end_offset],
            boundary.heading_level,
            boundary.title_path,
        )
        for boundary in first.boundaries
    ] == [
        ("heading", "# Outer", 1, ("Outer",)),
        ("list_item", "- Parent item", None, ("Outer",)),
        (
            "heading",
            "  First title 🚚 line\n  Second café line\n  -----------------",
            2,
            ("Outer", "First title 🚚 line Second café line"),
        ),
        (
            "list_item",
            "- Sibling under multiline heading",
            None,
            ("Outer", "First title 🚚 line Second café line"),
        ),
    ]
    assert all(left.end_offset <= right.start_offset for left, right in pairwise(first.boundaries))
    assert second == first


def test_markdown_long_list_continuations_bound_setext_indent_probes_linearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_remove_indent = parser_module._remove_indent
    probe_count = 0

    def counted_remove_indent(line: str, width: int) -> str | None:
        nonlocal probe_count
        probe_count += 1
        return original_remove_indent(line, width)

    monkeypatch.setattr(parser_module, "_remove_indent", counted_remove_indent)

    def parse_with_probe_count(continuation_count: int) -> int:
        nonlocal probe_count
        probe_count = 0
        source = "- parent\n" + "".join(
            f"  continuation {index}\n" for index in range(continuation_count)
        )

        artifact = MarkdownTextParser().parse(source.encode())

        assert artifact.boundaries == (
            StructuralBoundary(
                kind="list_item",
                start_offset=0,
                end_offset=len(source) - 1,
                title_path=(),
            ),
        )
        return probe_count

    small_count = parse_with_probe_count(128)
    large_count = parse_with_probe_count(256)

    assert small_count > 0
    assert large_count <= 4 * 256
    assert large_count < 3 * small_count


def test_markdown_long_list_continuations_keep_peak_memory_at_a_low_input_multiple() -> None:
    line = "  continuation\n"

    def parse_with_peak_memory(target_bytes: int) -> tuple[int, int]:
        continuation_count = (target_bytes - len("- parent\n")) // len(line)
        source = ("- parent\n" + line * continuation_count).encode()
        gc.collect()
        tracemalloc.start()
        try:
            artifact = MarkdownTextParser().parse(source)
            _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert artifact.boundaries == (
            StructuralBoundary(
                kind="list_item",
                start_offset=0,
                end_offset=len(source) - 1,
                title_path=(),
            ),
        )
        return len(source), peak_bytes

    small_source_bytes, small_peak_bytes = parse_with_peak_memory(256 * 1024)
    large_source_bytes, large_peak_bytes = parse_with_peak_memory(512 * 1024)

    assert large_peak_bytes <= 12 * large_source_bytes
    assert large_peak_bytes <= 5 * small_peak_bytes // 2
    assert large_source_bytes > small_source_bytes


def test_markdown_extremely_short_lines_use_compact_line_index_memory() -> None:
    def parse_with_peak_memory(target_bytes: int) -> tuple[int, int]:
        source = b"x\n" * (target_bytes // 2)
        gc.collect()
        tracemalloc.start()
        try:
            artifact = MarkdownTextParser().parse(source)
            _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()

        assert artifact.boundaries == (
            StructuralBoundary(
                kind="paragraph",
                start_offset=0,
                end_offset=len(source) - 1,
                title_path=(),
            ),
        )
        return len(source), peak_bytes

    small_source_bytes, small_peak_bytes = parse_with_peak_memory(256 * 1024)
    large_source_bytes, large_peak_bytes = parse_with_peak_memory(512 * 1024)

    assert large_peak_bytes <= 8 * large_source_bytes
    assert large_peak_bytes <= 5 * small_peak_bytes // 2
    assert large_source_bytes == 2 * small_source_bytes


@pytest.mark.parametrize(
    ("parser", "unit", "kind", "heading_level"),
    [
        (MarkdownTextParser(), b"#\n", "heading", 1),
        (PlainTextParser(), b"x\n\n", "paragraph", None),
    ],
)
def test_boundary_dense_inputs_have_an_exact_compact_sequence(
    parser: PlainTextParser | MarkdownTextParser,
    unit: bytes,
    kind: BoundaryKind,
    heading_level: int | None,
) -> None:
    boundary_count = 4_096
    artifact = parser.parse(unit * boundary_count)

    assert len(artifact.boundaries) == boundary_count
    assert artifact.boundaries.storage_run_count == 1
    for index in (0, boundary_count // 2, boundary_count - 1):
        start = index * len(unit)
        assert artifact.boundaries[index] == StructuralBoundary(
            kind=kind,
            start_offset=start,
            end_offset=start + 1,
            title_path=(),
            heading_level=heading_level,
        )


@pytest.mark.parametrize(
    ("parser", "unit", "peak_multiple"),
    [
        (MarkdownTextParser(), b"#\n", 14),
        (PlainTextParser(), b"x\n\n", 10),
    ],
)
def test_boundary_dense_parser_memory_stays_at_a_low_input_multiple(
    parser: PlainTextParser | MarkdownTextParser,
    unit: bytes,
    peak_multiple: int,
) -> None:
    target_bytes = 512 * 1024
    source = unit * (target_bytes // len(unit))

    gc.collect()
    tracemalloc.start()
    try:
        artifact = parser.parse(source)
        _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert len(artifact.boundaries) == len(source) // len(unit)
    assert artifact.boundaries.storage_run_count == 1
    assert peak_bytes <= peak_multiple * len(source)


def test_noncompressible_unique_headings_use_packed_fallback_storage() -> None:
    boundary_count = 10_000
    source = b"".join(
        (f"# {index:06d}\n" if index % 2 == 0 else f"## {index:05d}\n").encode("ascii")
        for index in range(boundary_count)
    )

    gc.collect()
    tracemalloc.start()
    try:
        artifact = MarkdownTextParser().parse(source)
        _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert len(artifact.boundaries) == boundary_count
    assert artifact.boundaries.storage_run_count == boundary_count
    assert artifact.boundaries[0].title_path == ("000000",)
    assert artifact.boundaries[1].title_path == ("000000", "00001")
    assert artifact.boundaries[-1].title_path == ("009998", "09999")
    assert peak_bytes <= 12 * len(source)


def test_periodic_alternating_structures_use_compact_template_storage() -> None:
    unit = b"# h\n\n- x\n\np\n\n"
    unit_count = 8_192
    source = unit * unit_count

    gc.collect()
    tracemalloc.start()
    try:
        artifact = MarkdownTextParser().parse(source)
        _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert len(artifact.boundaries) == unit_count * 3
    assert artifact.boundaries.storage_run_count == 3
    assert tuple(boundary.kind for boundary in artifact.boundaries[:6]) == (
        "heading",
        "list_item",
        "paragraph",
        "heading",
        "list_item",
        "paragraph",
    )
    assert peak_bytes <= 12 * len(source)


def test_structural_boundaries_reject_direct_array_construction() -> None:
    with pytest.raises(ValueError, match="^parsed artifact is invalid$"):
        StructuralBoundaries(
            _construction_token=object(),
            kinds=array("B", [0]),
            first_start_offsets=array("I", [0]),
            boundary_lengths=array("I", [1]),
            strides=array("I", [0]),
            cumulative_counts=array("I", [1]),
            title_path_ids=array("I", [0]),
            heading_levels=array("B", [1]),
            title_data=b"",
            title_end_offsets=array("I"),
            title_parents=array("I"),
        )


def test_parsed_artifact_normalizes_packed_integer_overflow() -> None:
    boundaries = (
        StructuralBoundary("paragraph", 0, 1, ()),
        StructuralBoundary("paragraph", 2**32, 2**32 + 1, ()),
    )

    with pytest.raises(ValueError, match="^parsed artifact is invalid$"):
        ParsedArtifact("x", boundaries, "plain_text_v1", "1", {})


def test_markdown_indented_fence_inside_list_uses_relative_closing_indentation() -> None:
    source = (
        "# Fences\n\n"
        "1. Parent item\n"
        "     ```python\n"
        "     ## opaque heading\n"
        "      ```\n"
        "2. Sibling item\n"
    )

    artifact = MarkdownTextParser().parse(source.encode())

    assert artifact.text == source
    assert [boundary.kind for boundary in artifact.boundaries] == [
        "heading",
        "list_item",
        "fenced_code_block",
        "list_item",
    ]
    assert [
        artifact.text[boundary.start_offset : boundary.end_offset]
        for boundary in artifact.boundaries
    ] == [
        "# Fences",
        "1. Parent item",
        "     ```python\n     ## opaque heading\n      ```",
        "2. Sibling item",
    ]
    assert artifact.boundaries[2].title_path == ("Fences",)
    assert artifact.boundaries[3].title_path == ("Fences",)


def test_markdown_fences_are_opaque_and_unclosed_fences_extend_to_end_of_text() -> None:
    source = (
        "# Outside\n\n"
        "~~~markdown\n"
        "## Not a heading\n"
        "- not a list boundary\n"
        "[link](https://example.invalid)\n"
    )

    artifact = MarkdownTextParser().parse(source.encode())

    assert [boundary.kind for boundary in artifact.boundaries] == [
        "heading",
        "fenced_code_block",
    ]
    fence = artifact.boundaries[1]
    assert artifact.text[fence.start_offset : fence.end_offset] == source[
        source.index("~~~") :
    ].rstrip("\n")
    assert fence.title_path == ("Outside",)


def test_markdown_parser_does_not_follow_links_include_files_execute_code_or_rewrite() -> None:
    source = (
        b"[local](file:///etc/passwd)\n"
        b'!INCLUDE "/etc/passwd"\n'
        b"<script>raise AssertionError('executed')</script>\n"
        b"```python\nraise AssertionError('executed')\n```\n"
    )

    artifact = MarkdownTextParser().parse(source)

    assert artifact.normalized_bytes == source
    assert artifact.text == source.decode()
    assert "root:x:" not in artifact.text
    assert "raise AssertionError('executed')" in artifact.text


@pytest.mark.parametrize("parser", [PlainTextParser(), MarkdownTextParser()])
@pytest.mark.parametrize(
    ("content", "expected_code"),
    [
        (b"", "EMPTY_DOCUMENT"),
        (b" \r\n\t ", "EMPTY_DOCUMENT"),
        (b"\xffinvalid", "INVALID_TEXT_ENCODING"),
        (b"hello\x00world", "BINARY_CONTENT_REJECTED"),
        (b"\x01\x02\x03\x04text", "BINARY_CONTENT_REJECTED"),
    ],
)
def test_parsers_reject_invalid_utf8_nul_binary_and_empty_input(
    parser: PlainTextParser | MarkdownTextParser,
    content: bytes,
    expected_code: str,
) -> None:
    with pytest.raises(DocumentValidationError) as exc_info:
        parser.parse(content)

    assert exc_info.value.code == expected_code
    decoded = content.decode(errors="ignore")
    if decoded:
        assert decoded not in repr(exc_info.value)


def test_markdown_parser_rejects_oversized_title_path_metadata() -> None:
    source = ("# " + ("界" * (256 * 1024)) + "\nbody\n").encode()

    with pytest.raises(ValueError, match="structural metadata exceeds limit"):
        MarkdownTextParser().parse(source)


def test_markdown_parser_rejects_oversized_aggregate_title_path_metadata() -> None:
    source = "\n".join(f"{'#' * level} {'x' * 900}" for level in range(1, 6)).encode()

    with pytest.raises(ValueError, match="structural metadata exceeds limit"):
        MarkdownTextParser().parse(source)


@pytest.mark.parametrize(
    ("parser", "fixture_name"),
    [
        (PlainTextParser(), "plain.txt"),
        (MarkdownTextParser(), "structured.md"),
    ],
)
def test_parser_output_is_byte_deterministic(
    parser: PlainTextParser | MarkdownTextParser,
    fixture_name: str,
) -> None:
    source = _fixture(fixture_name)

    first = parser.parse(source)
    second = parser.parse(source)

    assert first == second
    assert first.normalized_bytes == second.normalized_bytes
    assert first.checksum_sha256 == hashlib.sha256(first.normalized_bytes).hexdigest()
