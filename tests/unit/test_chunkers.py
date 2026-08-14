from __future__ import annotations

import gc
import random
import tracemalloc
import unicodedata
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from time import perf_counter
from types import MappingProxyType
from typing import cast, overload

import pytest

from rag_service.indexing.identities import canonical_sha256
from rag_service.ingestion.chunkers import (
    CHUNK_SCHEMA_VERSION,
    MAX_CHUNK_CODEPOINTS,
    TARGET_OVERLAP_CODEPOINTS,
    Chunk,
    Chunker,
    RecursiveTextChunker,
    _contains_point,
    _first_point,
    _index_boundaries,
    _last_point,
    _paragraph_end_starting_at,
    chunker_for_name,
)
from rag_service.ingestion.parsers import (
    MAX_TITLE_SEGMENT_UTF8_BYTES,
    BoundaryKind,
    MarkdownTextParser,
    ParsedArtifact,
    PlainTextParser,
    StructuralBoundaries,
    StructuralBoundary,
)


@dataclass(frozen=True)
class _ArtifactStub:
    text: str
    boundaries: Sequence[StructuralBoundary]
    parser_name: str = "plain_text_v1"


class _CountingBoundaries(Sequence[StructuralBoundary]):
    def __init__(self, values: tuple[StructuralBoundary, ...]) -> None:
        self._values = values
        self.access_count = 0

    def __len__(self) -> int:
        return len(self._values)

    @overload
    def __getitem__(self, index: int) -> StructuralBoundary: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[StructuralBoundary]: ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> StructuralBoundary | Sequence[StructuralBoundary]:
        self.access_count += 1
        return self._values[index]


def _ceiling_chunker() -> RecursiveTextChunker:
    """Chunk at the configurable ceiling rather than the shipped default.

    The cases below assert exact offsets for boundary handling, so they pin the
    sizing instead of inheriting it: they cover the splitting algorithm, not
    which default a deployment happens to ship with.
    """
    return RecursiveTextChunker(
        max_chunk_codepoints=MAX_CHUNK_CODEPOINTS,
        target_overlap_codepoints=TARGET_OVERLAP_CODEPOINTS,
    )


def _assert_source_slices(artifact: ParsedArtifact, chunks: tuple[Chunk, ...]) -> None:
    assert chunks
    assert chunks[0].start_offset == 0
    assert chunks[-1].end_offset == len(artifact.text)
    assert all(
        chunk.text == artifact.text[chunk.start_offset : chunk.end_offset] for chunk in chunks
    )
    assert all(0 < len(chunk.text) <= 1200 for chunk in chunks)
    assert all(
        left.start_offset < right.start_offset <= left.end_offset
        for left, right in pairwise(chunks)
    )


def test_recursive_chunker_descriptors_are_stable_immutable_and_pluggable() -> None:
    chunker = _ceiling_chunker()

    assert (chunker.name, chunker.version, dict(chunker.config)) == (
        "recursive_text_v1",
        "1",
        {
            "max_chunk_codepoints": 1200,
            "target_overlap_codepoints": 150,
        },
    )
    assert isinstance(chunker.config, MappingProxyType)
    assert chunker_for_name("recursive_text_v1") is chunker_for_name("recursive_text_v1")

    with pytest.raises(TypeError):
        chunker.config["max_chunk_codepoints"] = 1  # type: ignore[index]
    with pytest.raises(ValueError, match="document chunker name is invalid"):
        chunker_for_name("llm_semantic_v1")

    @dataclass(frozen=True)
    class FutureSemanticChunker:
        name: str = "llm_semantic_v1"
        version: str = "future"
        config: Mapping[str, object] = MappingProxyType({})

        def chunk(self, artifact: ParsedArtifact) -> Iterator[Chunk]:
            del artifact
            return iter(())

    assert isinstance(FutureSemanticChunker(), Chunker)


def test_small_markdown_is_one_exact_slice_with_title_path_and_stable_hash() -> None:
    artifact = MarkdownTextParser().parse("# 标题 🚚\n\n正文 café。\n".encode())
    chunker = _ceiling_chunker()

    first = tuple(chunker.chunk(artifact))
    second = tuple(chunker.chunk(artifact))

    assert first == second
    assert len(first) == 1
    chunk = first[0]
    assert chunk.chunk_index == 0
    assert chunk.text == artifact.text
    assert (chunk.start_offset, chunk.end_offset) == (0, len(artifact.text))
    assert chunk.title_path == ("标题 🚚",)
    assert dict(chunk.metadata) == {}
    assert isinstance(chunk.metadata, MappingProxyType)
    assert chunk.chunk_hash == canonical_sha256(
        {
            "metadata": {},
            "schema_version": CHUNK_SCHEMA_VERSION,
            "text": artifact.text,
            "title_path": ["标题 🚚"],
        }
    )


def test_hard_max_overlap_and_offsets_use_unicode_codepoints() -> None:
    artifact = PlainTextParser().parse(("你好🚚café" * 400).encode())
    chunks = tuple(_ceiling_chunker().chunk(artifact))

    _assert_source_slices(artifact, chunks)
    assert max(len(chunk.text) for chunk in chunks) <= 1200
    assert all(left.end_offset - right.start_offset <= 150 for left, right in pairwise(chunks))
    assert any(left.end_offset - right.start_offset == 150 for left, right in pairwise(chunks))
    assert chunks[1].start_offset != len(artifact.text[: chunks[1].start_offset].encode())
    assert all(
        chunk.end_offset == len(artifact.text)
        or (
            artifact.text[chunk.end_offset - 1] != "\u200d"
            and artifact.text[chunk.end_offset] != "\u200d"
            and not unicodedata.combining(artifact.text[chunk.end_offset])
        )
        for chunk in chunks
    )


@pytest.mark.parametrize(
    ("prefix_length", "cluster"),
    [
        pytest.param(1199, "🇨🇳", id="regional-indicator-flag"),
        pytest.param(1199, "का", id="spacing-combining-mark"),
        pytest.param(1198, "1️⃣", id="enclosing-keycap"),
        pytest.param(1198, "👩‍💻", id="zero-width-joiner"),
        pytest.param(1197, "👩️‍💻", id="emoji-extend-zero-width-joiner"),
        pytest.param(1199, "✈️", id="variation-selector"),
        pytest.param(1199, "👍🏽", id="skin-tone-modifier"),
        pytest.param(
            1199,
            "🏴\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f",
            id="england-emoji-tag-sequence",
        ),
        pytest.param(1198, "각", id="hangul-jamo-lvt"),
        pytest.param(1199, "\u0600ا", id="arabic-prepend"),
        pytest.param(1198, "क्ष", id="indic-virama-conjunct"),
        pytest.param(1199, "a\u200c", id="zero-width-non-joiner"),
        pytest.param(1199, "ｶﾞ", id="halfwidth-voiced-mark"),
        pytest.param(1199, "ﾊﾟ", id="halfwidth-semi-voiced-mark"),
    ],
)
def test_character_fallback_does_not_split_extended_unicode_characters(
    prefix_length: int,
    cluster: str,
) -> None:
    prefix = "a" * prefix_length
    artifact = PlainTextParser().parse((prefix + cluster + "z" * 200).encode())

    chunks = tuple(_ceiling_chunker().chunk(artifact))

    cluster_start = len(prefix)
    cluster_end = cluster_start + len(cluster)
    boundaries = {offset for chunk in chunks for offset in (chunk.start_offset, chunk.end_offset)}
    assert chunks[0].end_offset == cluster_start
    assert not any(cluster_start < offset < cluster_end for offset in boundaries)
    _assert_source_slices(artifact, chunks)


def test_regional_indicators_may_split_between_complete_flag_pairs() -> None:
    prefix = "a" * 1198
    artifact = PlainTextParser().parse((prefix + "🇨🇳🇺🇸" + "z" * 200).encode())

    chunks = tuple(_ceiling_chunker().chunk(artifact))

    assert chunks[0].end_offset == 1200
    assert chunks[0].text.endswith("🇨🇳")
    _assert_source_slices(artifact, chunks)


def test_crlf_is_normalized_before_chunk_boundary_selection() -> None:
    artifact = PlainTextParser().parse(("a" * 1199 + "\r\n" + "z" * 200).encode())

    chunks = tuple(_ceiling_chunker().chunk(artifact))

    assert "\r" not in artifact.text
    assert artifact.text[1199] == "\n"
    _assert_source_slices(artifact, chunks)


def test_sentence_end_moves_past_a_following_combining_mark() -> None:
    source = "a" * 200 + ".\u0301" + "z" * 1600
    artifact = PlainTextParser().parse(source.encode())

    chunks = tuple(_ceiling_chunker().chunk(artifact))

    unsafe_sentence_end = source.index(".") + 1
    assert chunks[0].end_offset == unsafe_sentence_end + 1
    assert all(
        offset != unsafe_sentence_end
        for chunk in chunks
        for offset in (chunk.start_offset, chunk.end_offset)
    )
    _assert_source_slices(artifact, chunks)


def test_sentence_overlap_start_moves_past_a_following_combining_mark() -> None:
    first_paragraph = "a" * 1100 + ".\u0301" + "b" * 98
    artifact = PlainTextParser().parse((first_paragraph + "\n\n" + "c" * 1600).encode())

    chunks = tuple(_ceiling_chunker().chunk(artifact))

    unsafe_sentence_end = first_paragraph.index(".") + 1
    assert chunks[0].end_offset == len(first_paragraph)
    assert chunks[1].start_offset == unsafe_sentence_end + 1
    assert all(
        offset != unsafe_sentence_end
        for chunk in chunks
        for offset in (chunk.start_offset, chunk.end_offset)
    )
    _assert_source_slices(artifact, chunks)


@pytest.mark.parametrize(
    ("prefix_length", "combining_marks"),
    [(1199, "\u0301"), (1198, "\u0301\u0301")],
    ids=["one-mark", "two-marks"],
)
def test_sentence_alignment_falls_back_before_a_cluster_at_the_hard_limit(
    prefix_length: int,
    combining_marks: str,
) -> None:
    source = "a" * prefix_length + "." + combining_marks + "z" * 200
    artifact = PlainTextParser().parse(source.encode())

    chunks = tuple(_ceiling_chunker().chunk(artifact))

    cluster_end = prefix_length + 1 + len(combining_marks)
    assert chunks[0].end_offset == prefix_length
    assert all(
        not prefix_length < offset < cluster_end
        for chunk in chunks
        for offset in (chunk.start_offset, chunk.end_offset)
    )
    _assert_source_slices(artifact, chunks)


def test_unbreakable_combining_sequence_uses_bounded_hard_limit_fallback() -> None:
    artifact = PlainTextParser().parse(("a" + "\u0301" * 20_000).encode())

    started = perf_counter()
    chunks = tuple(_ceiling_chunker().chunk(artifact))
    elapsed = perf_counter() - started

    assert len(chunks) <= 20
    assert elapsed < 1.0
    _assert_source_slices(artifact, chunks)


@pytest.mark.parametrize("middle_length", [699, 700, 701, 850])
def test_overlap_does_not_reselect_a_previous_paragraph_boundary(middle_length: int) -> None:
    artifact = PlainTextParser().parse(
        ("x\n\n" + "A" * middle_length + "\n\n" + "B" * 1600).encode()
    )

    chunks = tuple(_ceiling_chunker().chunk(artifact))

    assert all(
        right.start_offset > left.start_offset and right.end_offset > left.end_offset
        for left, right in pairwise(chunks)
    )
    _assert_source_slices(artifact, chunks)


def test_high_overlap_still_progresses_across_grapheme_aligned_paragraphs() -> None:
    artifact = PlainTextParser().parse((("각" * 12) + "\n\n" + ("각" * 30)).encode())
    chunker = RecursiveTextChunker(max_chunk_codepoints=38, target_overlap_codepoints=36)

    chunks = tuple(chunker.chunk(artifact))

    assert all(
        right.start_offset > left.start_offset and right.end_offset > left.end_offset
        for left, right in pairwise(chunks)
    )
    _assert_source_slices(artifact, chunks)


def test_overlap_shrinks_instead_of_splitting_a_fittable_grapheme() -> None:
    england_flag = "🏴\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f"
    artifact = PlainTextParser().parse(("x" * 10 + england_flag + "z" * 20).encode())
    chunker = RecursiveTextChunker(max_chunk_codepoints=14, target_overlap_codepoints=8)

    chunks = tuple(chunker.chunk(artifact))

    assert (chunks[0].start_offset, chunks[0].end_offset) == (0, 10)
    assert (chunks[1].start_offset, chunks[1].end_offset) == (3, 17)
    assert chunks[0].end_offset - chunks[1].start_offset == 7
    assert all(
        not 10 < offset < 17
        for chunk in chunks
        for offset in (chunk.start_offset, chunk.end_offset)
    )
    assert all(len(chunk.text) <= 14 for chunk in chunks)
    _assert_source_slices(artifact, chunks)


@pytest.mark.parametrize(
    "character",
    ["a", "€", "あ", "🇦"],
    ids=["latin", "euro", "hiragana", "regional-indicator"],
)
def test_non_pictographic_zwj_allows_a_boundary_after_the_joiner(
    character: str,
) -> None:
    artifact = PlainTextParser().parse(("x" + (character + "\u200d") * 800 + "z" * 100).encode())

    chunks = tuple(_ceiling_chunker().chunk(artifact))

    assert chunks[0].end_offset == 1199
    assert chunks[0].text.endswith("\u200d")
    _assert_source_slices(artifact, chunks)


def test_spacing_mark_does_not_count_as_extend_in_an_emoji_zwj_sequence() -> None:
    source = "x" * 1197 + "👩\u093e\u200d💻" + "z" * 100
    artifact = PlainTextParser().parse(source.encode())

    chunks = tuple(_ceiling_chunker().chunk(artifact))

    assert chunks[0].end_offset == 1200
    assert chunks[0].text.endswith("\u200d")
    _assert_source_slices(artifact, chunks)


def test_mc_extend_still_counts_as_extend_in_an_emoji_zwj_sequence() -> None:
    source = "x" * 1197 + "👩\u09be\u200d💻" + "z" * 100
    artifact = PlainTextParser().parse(source.encode())

    chunks = tuple(_ceiling_chunker().chunk(artifact))

    assert chunks[0].end_offset == 1197
    assert "👩\u09be\u200d💻" in chunks[1].text
    assert all(
        not 1197 < offset < 1201
        for chunk in chunks
        for offset in (chunk.start_offset, chunk.end_offset)
    )
    _assert_source_slices(artifact, chunks)


def test_txt_prefers_paragraph_then_sentence_boundaries_before_character_fallback() -> None:
    first_paragraph = "A" * 34
    second_paragraph = "First sentence. Second sentence keeps going " + "B" * 50
    artifact = PlainTextParser().parse(f"{first_paragraph}\n\n{second_paragraph}".encode())
    chunks = tuple(
        RecursiveTextChunker(max_chunk_codepoints=55, target_overlap_codepoints=10).chunk(artifact)
    )

    assert chunks[0].end_offset == artifact.text.index(second_paragraph)
    sentence_end = artifact.text.index("Second sentence")
    assert any(chunk.end_offset == sentence_end for chunk in chunks)
    assert all(len(chunk.text) <= 55 for chunk in chunks)
    assert all(
        chunk.text == artifact.text[chunk.start_offset : chunk.end_offset] for chunk in chunks
    )


def test_txt_short_paragraph_boundary_takes_priority_over_overlap_target() -> None:
    first_paragraph = "A" * 80
    second_paragraph = "B" * 1600
    artifact = PlainTextParser().parse(f"{first_paragraph}\n\n{second_paragraph}".encode())

    chunks = tuple(_ceiling_chunker().chunk(artifact))

    second_paragraph_start = artifact.text.index(second_paragraph)
    assert second_paragraph_start == 82
    assert chunks[0].end_offset == second_paragraph_start
    assert chunks[1].start_offset == second_paragraph_start
    _assert_source_slices(artifact, chunks)


def test_markdown_prefers_heading_sections_and_code_blocks_and_tracks_title_paths() -> None:
    source = (
        "# Top\n\n"
        + "intro "
        + "A" * 32
        + "\n\n## Beta\n\n"
        + "```text\n"
        + "code line\n" * 3
        + "```\n\n"
        + "beta body "
        + "B" * 90
    )
    artifact = MarkdownTextParser().parse(source.encode())
    chunks = tuple(
        RecursiveTextChunker(max_chunk_codepoints=70, target_overlap_codepoints=12).chunk(artifact)
    )

    beta_start = artifact.text.index("## Beta")
    code_end = artifact.text.index("```\n\n", artifact.text.index("```text")) + 3
    assert chunks[0].end_offset == beta_start
    assert any(chunk.end_offset == code_end for chunk in chunks)
    assert chunks[0].title_path == ("Top",)
    assert chunks[-1].title_path == ("Top", "Beta")
    assert all(
        chunk.text == artifact.text[chunk.start_offset : chunk.end_offset] for chunk in chunks
    )


def test_short_markdown_heading_takes_priority_over_overlap_target() -> None:
    source = "# Old\n\n" + "A" * 80 + "\n\n## New\n\n" + "B" * 1600
    artifact = MarkdownTextParser().parse(source.encode())

    chunks = tuple(_ceiling_chunker().chunk(artifact))

    new_heading_start = artifact.text.index("## New")
    assert new_heading_start == 89
    assert chunks[0].end_offset == new_heading_start
    assert chunks[0].title_path == ("Old",)
    assert chunks[1].start_offset == new_heading_start
    assert chunks[1].title_path == ("Old", "New")
    _assert_source_slices(artifact, chunks)


def test_markdown_chunks_stop_at_the_next_title_path_change() -> None:
    first_section = "# A\n\n" + "A" * 500 + "\n\n"
    second_section = "# B\n\n" + "B" * 501 + "\n\n"
    source = first_section + second_section + "# C\n\n" + "C" * 1600
    artifact = MarkdownTextParser().parse(source.encode())

    chunks = tuple(_ceiling_chunker().chunk(artifact))

    second_heading = artifact.text.index("# B")
    third_heading = artifact.text.index("# C")
    assert (second_heading, third_heading) == (507, 1015)
    assert (chunks[0].start_offset, chunks[0].end_offset, chunks[0].title_path) == (
        0,
        second_heading,
        ("A",),
    )
    assert (chunks[1].start_offset, chunks[1].end_offset, chunks[1].title_path) == (
        second_heading,
        third_heading,
        ("B",),
    )
    _assert_source_slices(artifact, chunks)


def test_markdown_final_window_still_stops_at_a_title_path_change() -> None:
    source = "# A\n\n" + "A" * 1300 + "\n\n# B\n\nbrief"
    artifact = MarkdownTextParser().parse(source.encode())

    chunks = tuple(_ceiling_chunker().chunk(artifact))

    second_heading = artifact.text.index("# B")
    assert second_heading == 1307
    assert (chunks[-2].start_offset, chunks[-2].end_offset, chunks[-2].title_path) == (
        1050,
        second_heading,
        ("A",),
    )
    assert (chunks[-1].start_offset, chunks[-1].end_offset, chunks[-1].title_path) == (
        second_heading,
        len(artifact.text),
        ("B",),
    )
    _assert_source_slices(artifact, chunks)


def test_markdown_does_not_emit_a_heading_only_chunk_before_a_code_block() -> None:
    source = "# Code\n\n```text\n" + "x" * 1600 + "\n```\n"
    artifact = MarkdownTextParser().parse(source.encode())

    chunks = tuple(_ceiling_chunker().chunk(artifact))

    code_start = artifact.text.index("```text")
    assert code_start == 8
    assert chunks[0].end_offset > code_start
    assert "```text" in chunks[0].text
    _assert_source_slices(artifact, chunks)


@pytest.mark.parametrize(
    ("source", "expected_end"),
    [
        ("# Code\n\n```text\nx\n```\n\n" + "B" * 1600, 21),
        ("# Heading\n\nshort\n\n" + "B" * 1600, 18),
        ("# Heading\n\n- short\n- " + "B" * 1600, 19),
    ],
    ids=["code-end", "next-paragraph", "next-list-item"],
)
def test_markdown_heading_still_uses_early_structural_boundaries(
    source: str,
    expected_end: int,
) -> None:
    artifact = MarkdownTextParser().parse(source.encode())

    chunks = tuple(_ceiling_chunker().chunk(artifact))

    assert chunks[0].end_offset == expected_end
    assert chunks[0].title_path == artifact.boundaries[0].title_path
    _assert_source_slices(artifact, chunks)


@pytest.mark.parametrize(
    "following_structure",
    [
        "```text\n" + "x" * 1600 + "\n```\n",
        "B" * 1600,
        "- " + "item " * 400,
    ],
    ids=["fenced-code", "paragraph", "list-item"],
)
def test_markdown_early_structure_still_precedes_the_overlap_target(
    following_structure: str,
) -> None:
    source = "A" * 80 + "\n\n" + following_structure
    artifact = MarkdownTextParser().parse(source.encode())

    chunks = tuple(_ceiling_chunker().chunk(artifact))

    assert artifact.text.index(following_structure) == 82
    assert chunks[0].end_offset == 82
    assert chunks[1].start_offset == 82
    _assert_source_slices(artifact, chunks)


def test_markdown_early_structure_keeps_code_priority_over_a_later_paragraph() -> None:
    source = "A" * 20 + "\n\n```text\nx\n```\n\n" + "B" * 1600
    artifact = MarkdownTextParser().parse(source.encode())

    chunks = tuple(_ceiling_chunker().chunk(artifact))

    code_end = artifact.text.index("```\n\n") + 3
    following_paragraph = code_end + 2
    assert code_end < following_paragraph < 150
    assert chunks[0].end_offset == code_end
    _assert_source_slices(artifact, chunks)


def test_heading_split_restarts_at_short_new_section_with_its_title_path() -> None:
    source = "# Root\n\n" + "A" * 70 + "\n\n## Short\n\nbrief"
    artifact = MarkdownTextParser().parse(source.encode())

    chunks = tuple(
        RecursiveTextChunker(max_chunk_codepoints=90, target_overlap_codepoints=20).chunk(artifact)
    )

    short_start = artifact.text.index("## Short")
    assert chunks == (
        Chunk.from_source(
            chunk_index=0,
            source_text=artifact.text,
            start_offset=0,
            end_offset=short_start,
            title_path=("Root",),
        ),
        Chunk.from_source(
            chunk_index=1,
            source_text=artifact.text,
            start_offset=short_start,
            end_offset=len(artifact.text),
            title_path=("Root", "Short"),
        ),
    )


def test_empty_heading_resets_title_path_for_following_chunks() -> None:
    source = "# Root\n\n" + "A" * 70 + "\n\n#\n\n" + "B" * 30
    artifact = MarkdownTextParser().parse(source.encode())
    chunks = tuple(
        RecursiveTextChunker(max_chunk_codepoints=90, target_overlap_codepoints=20).chunk(artifact)
    )

    reset_start = artifact.text.index("#\n\n")
    assert chunks[-1].start_offset == reset_start
    assert chunks[-1].title_path == ()


def test_oversized_markdown_code_block_is_recursively_split_without_rewriting() -> None:
    artifact = MarkdownTextParser().parse(
        ("# Code\n\n```text\n" + "界🚚" * 100 + "\n```\n").encode()
    )
    chunks = tuple(
        RecursiveTextChunker(max_chunk_codepoints=64, target_overlap_codepoints=8).chunk(artifact)
    )

    assert len(chunks) > 3
    assert all(len(chunk.text) <= 64 for chunk in chunks)
    assert all(
        chunk.text == artifact.text[chunk.start_offset : chunk.end_offset] for chunk in chunks
    )
    assert chunks[-1].end_offset == len(artifact.text)


def test_empty_artifacts_are_rejected_by_parser_contract_and_small_text_is_supported() -> None:
    with pytest.raises(ValueError, match="parsed artifact is invalid"):
        ParsedArtifact("", (), "plain_text_v1", "1", {})

    artifact = PlainTextParser().parse(b"x")
    assert tuple(_ceiling_chunker().chunk(artifact))[0].text == "x"


def test_chunk_hash_canonicalizes_equivalent_metadata_before_hashing() -> None:
    left = Chunk.from_source(
        chunk_index=0,
        source_text="prefix text suffix",
        start_offset=7,
        end_offset=11,
        title_path=("Title",),
        metadata={"z": [2, 1], "a": {"nested": True}},
    )
    right = Chunk.from_source(
        chunk_index=0,
        source_text="prefix text suffix",
        start_offset=7,
        end_offset=11,
        title_path=("Title",),
        metadata={"a": {"nested": True}, "z": [2, 1]},
    )

    assert left == right
    assert left.chunk_hash == right.chunk_hash
    assert left.text == "text"
    assert isinstance(left.metadata, MappingProxyType)

    with pytest.raises(ValueError, match="chunk metadata is invalid"):
        Chunk.from_source(
            chunk_index=0,
            source_text="text",
            start_offset=0,
            end_offset=4,
            title_path=(),
            metadata=cast(Mapping[str, object], {"bad": {1, 2}}),
        )


def test_boundary_indexing_is_linear_in_boundaries_not_chunks_times_boundaries() -> None:
    text = "x" * 6_000
    raw_boundaries = tuple(
        StructuralBoundary("paragraph", start, start + 5, ())
        for start in range(0, len(text) - 5, 20)
    )
    boundaries = _CountingBoundaries(raw_boundaries)
    artifact = cast(
        ParsedArtifact,
        _ArtifactStub(text=text, boundaries=boundaries),
    )

    chunks = tuple(
        RecursiveTextChunker(max_chunk_codepoints=32, target_overlap_codepoints=4).chunk(artifact)
    )

    assert len(chunks) > 100
    assert boundaries.access_count <= len(boundaries) * 2 + 10


@pytest.mark.parametrize(
    ("parser", "unit"),
    [
        (MarkdownTextParser(), b"#\n"),
        (PlainTextParser(), b"x\n\n"),
    ],
)
def test_compact_boundary_index_does_not_expand_boundary_dense_inputs(
    parser: PlainTextParser | MarkdownTextParser,
    unit: bytes,
) -> None:
    target_bytes = 512 * 1024
    source = unit * (target_bytes // len(unit))
    artifact = parser.parse(source)

    gc.collect()
    tracemalloc.start()
    try:
        chunks = _ceiling_chunker().chunk(artifact)
        first_chunk = next(chunks)
        _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert first_chunk.text
    assert peak_bytes <= 2 * len(source)


@pytest.mark.parametrize(
    ("parser", "unit"),
    [
        (MarkdownTextParser(), b"#\n"),
        (PlainTextParser(), b"x\n\n"),
    ],
)
def test_compact_boundary_index_preserves_legacy_sequence_chunking(
    parser: PlainTextParser | MarkdownTextParser,
    unit: bytes,
) -> None:
    artifact = parser.parse(unit * 4_096)
    legacy_artifact = cast(
        ParsedArtifact,
        _ArtifactStub(
            text=artifact.text,
            boundaries=tuple(artifact.boundaries),
            parser_name=artifact.parser_name,
        ),
    )
    chunker = _ceiling_chunker()

    compact_chunks = tuple(chunker.chunk(artifact))

    assert compact_chunks == tuple(chunker.chunk(artifact))
    assert compact_chunks == tuple(chunker.chunk(legacy_artifact))
    assert compact_chunks[-1].end_offset == len(artifact.text)


def test_compact_title_path_is_reused_across_many_chunks() -> None:
    title = "t" * MAX_TITLE_SEGMENT_UTF8_BYTES
    artifact = MarkdownTextParser().parse(f"# {title}\n\n{'body ' * 4_000}".encode())

    chunks = tuple(_ceiling_chunker().chunk(artifact))

    assert len(chunks) > 10
    assert all(chunk.title_path == (title,) for chunk in chunks)
    assert len({id(chunk.title_path) for chunk in chunks}) == 1


def test_compact_title_path_reuses_a_bounded_parent_across_child_toggles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    title = "t" * MAX_TITLE_SEGMENT_UTF8_BYTES
    source = f"# {title}\n" + ("## child\nx\n##\nx\n" * 100)
    artifact = MarkdownTextParser().parse(source.encode())
    decode_counts: dict[int, int] = {}
    original = type(artifact.boundaries)._decode_title_node

    def counted_decode(boundaries: StructuralBoundaries, node_id: int) -> str:
        decode_counts[node_id] = decode_counts.get(node_id, 0) + 1
        return original(boundaries, node_id)

    monkeypatch.setattr(type(artifact.boundaries), "_decode_title_node", counted_decode)

    chunks = tuple(_ceiling_chunker().chunk(artifact))
    parent_paths = [chunk.title_path for chunk in chunks if chunk.title_path == (title,)]
    parent_titles = [chunk.title_path[0] for chunk in chunks if chunk.title_path]

    assert len(chunks) > 200
    assert parent_paths
    assert len({id(path) for path in parent_paths}) == 1
    assert len({id(parent_title) for parent_title in parent_titles}) == 1
    assert max(decode_counts.values()) == 1


def test_compact_boundary_index_uses_packed_fallback_for_alternating_structures() -> None:
    unit = b"# h\n\n- x\n\np\n\n"
    source = unit * 8_192
    artifact = MarkdownTextParser().parse(source)

    gc.collect()
    tracemalloc.start()
    try:
        chunks = _ceiling_chunker().chunk(artifact)
        first_chunk = next(chunks)
        _current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert first_chunk.text == unit.decode("ascii")
    assert peak_bytes <= 8 * len(source)


def test_periodic_point_views_match_legacy_across_duplicates_gaps_and_partial_cycles() -> None:
    pattern: tuple[tuple[BoundaryKind, int, int, int | None], ...] = (
        ("heading", 0, 1, 1),
        ("paragraph", 1, 3, None),
        ("list_item", 3, 4, None),
        ("fenced_code_block", 5, 8, None),
    )
    raw_boundaries = [
        StructuralBoundary(kind, start + cycle * 10, end + cycle * 10, (), level)
        for cycle in range(20)
        for kind, start, end, level in pattern
    ]
    raw_boundaries.extend(
        (
            StructuralBoundary("heading", 200, 201, (), 1),
            StructuralBoundary("paragraph", 201, 205, ()),
            StructuralBoundary("list_item", 207, 209, ()),
        )
    )
    artifact = ParsedArtifact(
        "x" * 210,
        tuple(raw_boundaries),
        "markdown_text_v1",
        "1",
        {},
    )
    compact = _index_boundaries(artifact.boundaries)
    legacy = _index_boundaries(tuple(artifact.boundaries))

    for field_name in (
        "headings",
        "structural_starts",
        "code_blocks",
        "paragraphs",
        "paragraph_starts",
        "paragraph_ends",
        "list_items",
    ):
        compact_points = getattr(compact, field_name)
        legacy_points = getattr(legacy, field_name)
        assert tuple(compact_points) == legacy_points
        assert compact_points[-3:] == legacy_points[-3:]
        for lower in range(0, 211, 7):
            for upper in range(lower, min(211, lower + 17)):
                assert _first_point(compact_points, lower, upper) == _first_point(
                    legacy_points, lower, upper
                )
                assert _last_point(compact_points, lower, upper) == _last_point(
                    legacy_points, lower, upper
                )
            assert _contains_point(compact_points, lower) == _contains_point(legacy_points, lower)

    for offset in range(210):
        assert _paragraph_end_starting_at(compact, offset) == _paragraph_end_starting_at(
            legacy, offset
        )


def test_periodic_boundary_sequence_matches_legacy_after_mid_cycle_mismatch() -> None:
    source = ("# h\n\n- x\n\np\n\n" * 40) + "# h\n\n- changed\n"
    artifact = MarkdownTextParser().parse(source.encode())
    legacy_boundaries = tuple(artifact.boundaries)

    assert artifact.boundaries == legacy_boundaries
    assert artifact.boundaries[-9:-1:2] == legacy_boundaries[-9:-1:2]
    assert tuple(_ceiling_chunker().chunk(artifact)) == tuple(
        _ceiling_chunker().chunk(
            cast(
                ParsedArtifact,
                _ArtifactStub(
                    text=artifact.text,
                    boundaries=legacy_boundaries,
                    parser_name=artifact.parser_name,
                ),
            )
        )
    )


def test_seeded_periodic_compact_views_match_legacy_across_segment_shapes() -> None:
    rng = random.Random(20260729)
    raw_boundaries: list[StructuralBoundary] = []
    cursor = 0
    kinds: tuple[BoundaryKind, ...] = (
        "heading",
        "paragraph",
        "list_item",
        "fenced_code_block",
    )
    for group in range(60):
        template: list[StructuralBoundary] = []
        template_cursor = cursor
        title_path = (f"group-{group}",)
        for _ in range(rng.randint(1, 8)):
            kind = rng.choice(kinds)
            length = rng.randint(1, 4)
            template.append(
                StructuralBoundary(
                    kind,
                    template_cursor,
                    template_cursor + length,
                    title_path,
                    rng.randint(1, 6) if kind == "heading" else None,
                )
            )
            template_cursor += length + rng.randint(0, 2)
        cycle_stride = template_cursor - cursor + rng.randint(0, 2)
        repeat_count = rng.randint(2, 8)
        for cycle in range(repeat_count):
            for boundary in template:
                raw_boundaries.append(
                    StructuralBoundary(
                        boundary.kind,
                        boundary.start_offset + cycle * cycle_stride,
                        boundary.end_offset + cycle * cycle_stride,
                        boundary.title_path,
                        boundary.heading_level,
                    )
                )
        cursor = template[-1].end_offset + (repeat_count - 1) * cycle_stride
        if rng.random() < 0.7:
            cursor += rng.randint(0, 2)
            raw_boundaries.append(
                StructuralBoundary("paragraph", cursor, cursor + rng.randint(1, 5), title_path)
            )
            cursor = raw_boundaries[-1].end_offset
        cursor += rng.randint(0, 3)

    artifact = ParsedArtifact(
        "x" * max(1, cursor),
        tuple(raw_boundaries),
        "markdown_text_v1",
        "1",
        {},
    )
    compact_boundaries = artifact.boundaries
    legacy_boundaries = tuple(raw_boundaries)
    compact_index = _index_boundaries(compact_boundaries)
    legacy_index = _index_boundaries(legacy_boundaries)

    assert tuple(compact_boundaries) == legacy_boundaries
    for _ in range(300):
        index = rng.randrange(-len(legacy_boundaries), len(legacy_boundaries))
        assert compact_boundaries[index] == legacy_boundaries[index]
        start = rng.randrange(-len(legacy_boundaries), len(legacy_boundaries) + 1)
        stop = rng.randrange(-len(legacy_boundaries), len(legacy_boundaries) + 1)
        step = rng.choice((-5, -2, -1, 1, 2, 5))
        assert compact_boundaries[start:stop:step] == legacy_boundaries[start:stop:step]

    for field_name in (
        "headings",
        "structural_starts",
        "code_blocks",
        "paragraphs",
        "paragraph_starts",
        "paragraph_ends",
        "list_items",
    ):
        compact_points = getattr(compact_index, field_name)
        legacy_points = getattr(legacy_index, field_name)
        assert tuple(compact_points) == legacy_points
        for _ in range(100):
            left = rng.randrange(0, cursor + 1)
            right = rng.randrange(left, cursor + 1)
            assert _first_point(compact_points, left, right) == _first_point(
                legacy_points, left, right
            )
            assert _last_point(compact_points, left, right) == _last_point(
                legacy_points, left, right
            )


def test_lexical_boundary_memory_does_not_scale_with_total_newline_count() -> None:
    def peak_for(line_count: int) -> int:
        artifact = cast(
            ParsedArtifact,
            _ArtifactStub(text="x\n" * line_count, boundaries=()),
        )
        tracemalloc.start()
        chunks = _ceiling_chunker().chunk(artifact)
        next(chunks)
        _current, peak = tracemalloc.get_traced_memory()
        del chunks
        tracemalloc.stop()
        return peak

    small_peak = peak_for(10_000)
    large_peak = peak_for(50_000)

    assert large_peak < small_peak * 3


def test_blank_windows_are_never_emitted_as_chunks() -> None:
    """A whitespace-only chunk fails the whole document, terminally.

    The embedding gateway rejects blank input with EMBEDDING_INPUT_INVALID, which
    is not retryable, so one such chunk fails the document that contained it. Blank
    windows appear where blank lines cluster — between transcript turns, say — and
    get more likely as the chunk size shrinks, which is exactly when someone is
    tuning. Regression: max=300/overlap=60 over this shape used to emit "\n\n".
    """

    text = "# Title\n\n" + "\n\n".join(f"## Turn {index}\n\n{'word ' * 40}" for index in range(12))
    text += "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n" + "tail paragraph\n"
    artifact = MarkdownTextParser().parse(text.encode())

    for max_codepoints, overlap in ((600, 100), (300, 60), (200, 40), (150, 30)):
        chunks = list(
            RecursiveTextChunker(
                max_chunk_codepoints=max_codepoints,
                target_overlap_codepoints=overlap,
            ).chunk(artifact)
        )

        assert chunks, (max_codepoints, overlap)
        blank = [chunk.text for chunk in chunks if not chunk.text.strip()]
        assert not blank, (max_codepoints, overlap, blank[:2])
        # Skipping a window must not leave a hole in the sequence.
        assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
