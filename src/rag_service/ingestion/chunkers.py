"""Deterministic source-slicing chunkers for normalized parsed artifacts."""

from __future__ import annotations

import math
import re
import unicodedata
from array import array
from bisect import bisect_left, bisect_right
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Literal, Protocol, overload, runtime_checkable

from rag_service.indexing.identities import canonical_sha256
from rag_service.ingestion.parsers import (
    BoundaryKind,
    ParsedArtifact,
    StructuralBoundaries,
    StructuralBoundary,
)

CHUNK_SCHEMA_VERSION: Final = "chunks-v1"
# The MAX_/TARGET_ pair is the ceiling a caller may configure up to; the DEFAULT_
# pair is what an unconfigured install actually chunks at. They were one value
# each until a retrieval measurement on a real corpus showed the ceiling made a
# poor default: at 1200 the answer landed in the top chunk 49% of the time, at
# 600 it was 77% (docs/rag-retrieval-baseline.md). Larger chunks dilute the
# embedding across too many unrelated statements to rank precisely.
MAX_CHUNK_CODEPOINTS: Final = 1200
TARGET_OVERLAP_CODEPOINTS: Final = 150
DEFAULT_CHUNK_CODEPOINTS: Final = 600
DEFAULT_OVERLAP_CODEPOINTS: Final = 100

_SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
_SENTENCE_END: Final = re.compile(r"[.!?。！？]+(?:[\"')\]】）”’」』]*)[ \t]*(?:\n|$)?")
_EMPTY_METADATA: Final[Mapping[str, object]] = MappingProxyType({})
_REGIONAL_INDICATOR_START: Final = 0x1F1E6
_REGIONAL_INDICATOR_END: Final = 0x1F1FF
_TAG_CHARACTER_START: Final = 0xE0020
_TAG_CHARACTER_END: Final = 0xE007F
# Unicode 15.0 Extended_Pictographic property from emoji-data.txt.
_EXTENDED_PICTOGRAPHIC_RANGES: Final = (
    (0x00A9, 0x00A9),
    (0x00AE, 0x00AE),
    (0x203C, 0x203C),
    (0x2049, 0x2049),
    (0x2122, 0x2122),
    (0x2139, 0x2139),
    (0x2194, 0x2199),
    (0x21A9, 0x21AA),
    (0x231A, 0x231B),
    (0x2328, 0x2328),
    (0x2388, 0x2388),
    (0x23CF, 0x23CF),
    (0x23E9, 0x23F3),
    (0x23F8, 0x23FA),
    (0x24C2, 0x24C2),
    (0x25AA, 0x25AB),
    (0x25B6, 0x25B6),
    (0x25C0, 0x25C0),
    (0x25FB, 0x25FE),
    (0x2600, 0x2605),
    (0x2607, 0x2612),
    (0x2614, 0x2685),
    (0x2690, 0x2705),
    (0x2708, 0x2712),
    (0x2714, 0x2714),
    (0x2716, 0x2716),
    (0x271D, 0x271D),
    (0x2721, 0x2721),
    (0x2728, 0x2728),
    (0x2733, 0x2734),
    (0x2744, 0x2744),
    (0x2747, 0x2747),
    (0x274C, 0x274C),
    (0x274E, 0x274E),
    (0x2753, 0x2755),
    (0x2757, 0x2757),
    (0x2763, 0x2767),
    (0x2795, 0x2797),
    (0x27A1, 0x27A1),
    (0x27B0, 0x27B0),
    (0x27BF, 0x27BF),
    (0x2934, 0x2935),
    (0x2B05, 0x2B07),
    (0x2B1B, 0x2B1C),
    (0x2B50, 0x2B50),
    (0x2B55, 0x2B55),
    (0x3030, 0x3030),
    (0x303D, 0x303D),
    (0x3297, 0x3297),
    (0x3299, 0x3299),
    (0x1F000, 0x1F0FF),
    (0x1F10D, 0x1F10F),
    (0x1F12F, 0x1F12F),
    (0x1F16C, 0x1F171),
    (0x1F17E, 0x1F17F),
    (0x1F18E, 0x1F18E),
    (0x1F191, 0x1F19A),
    (0x1F1AD, 0x1F1E5),
    (0x1F201, 0x1F20F),
    (0x1F21A, 0x1F21A),
    (0x1F22F, 0x1F22F),
    (0x1F232, 0x1F23A),
    (0x1F23C, 0x1F23F),
    (0x1F249, 0x1F3FA),
    (0x1F400, 0x1F53D),
    (0x1F546, 0x1F64F),
    (0x1F680, 0x1F6FF),
    (0x1F774, 0x1F77F),
    (0x1F7D5, 0x1F7FF),
    (0x1F80C, 0x1F80F),
    (0x1F848, 0x1F84F),
    (0x1F85A, 0x1F85F),
    (0x1F888, 0x1F88F),
    (0x1F8AE, 0x1F8FF),
    (0x1F90C, 0x1F93A),
    (0x1F93C, 0x1F945),
    (0x1F947, 0x1FAFF),
    (0x1FC00, 0x1FFFD),
)
# Unicode 15.0 Grapheme_Cluster_Break=Extend characters whose general category is Mc.
_GCB_EXTEND_MC_RANGES: Final = (
    (0x09BE, 0x09BE),
    (0x09D7, 0x09D7),
    (0x0B3E, 0x0B3E),
    (0x0B57, 0x0B57),
    (0x0BBE, 0x0BBE),
    (0x0BD7, 0x0BD7),
    (0x0CC2, 0x0CC2),
    (0x0CD5, 0x0CD6),
    (0x0D3E, 0x0D3E),
    (0x0D57, 0x0D57),
    (0x0DCF, 0x0DCF),
    (0x0DDF, 0x0DDF),
    (0x1B35, 0x1B35),
    (0x302E, 0x302F),
    (0x1133E, 0x1133E),
    (0x11357, 0x11357),
    (0x114B0, 0x114B0),
    (0x114BD, 0x114BD),
    (0x115AF, 0x115AF),
    (0x11930, 0x11930),
    (0x1D165, 0x1D165),
    (0x1D16E, 0x1D172),
)
_PREPEND_RANGES: Final = (
    (0x0600, 0x0605),
    (0x06DD, 0x06DD),
    (0x070F, 0x070F),
    (0x0890, 0x0891),
    (0x08E2, 0x08E2),
    (0x0D4E, 0x0D4E),
    (0x110BD, 0x110BD),
    (0x110CD, 0x110CD),
    (0x111C2, 0x111C3),
    (0x113D1, 0x113D1),
    (0x1193F, 0x1193F),
    (0x11941, 0x11941),
    (0x11A3A, 0x11A3A),
    (0x11A84, 0x11A89),
    (0x11D46, 0x11D46),
    (0x11F02, 0x11F02),
)
_HANGUL_NONE: Final = 0
_HANGUL_L: Final = 1
_HANGUL_V: Final = 2
_HANGUL_T: Final = 3
_HANGUL_LV: Final = 4
_HANGUL_LVT: Final = 5


def _freeze_json(value: object, *, error_message: str) -> object:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(error_message)
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError(error_message)
        return MappingProxyType(
            {key: _freeze_json(value[key], error_message=error_message) for key in sorted(value)}
        )
    if type(value) is list:
        return tuple(_freeze_json(member, error_message=error_message) for member in value)
    if type(value) is tuple:
        return tuple(_freeze_json(member, error_message=error_message) for member in value)
    raise ValueError(error_message)


def _freeze_mapping(value: Mapping[str, object], *, error_message: str) -> Mapping[str, object]:
    frozen = _freeze_json(value, error_message=error_message)
    if not isinstance(frozen, Mapping):
        raise ValueError(error_message)
    return frozen


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain_json(member) for key, member in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(member) for member in value]
    return value


def _chunk_hash(text: str, title_path: tuple[str, ...], metadata: Mapping[str, object]) -> str:
    return canonical_sha256(
        {
            "metadata": _plain_json(metadata),
            "schema_version": CHUNK_SCHEMA_VERSION,
            "text": text,
            "title_path": list(title_path),
        }
    )


@dataclass(frozen=True, slots=True, repr=False)
class Chunk:
    """One immutable half-open source span and its canonical content identity."""

    chunk_index: int
    text: str
    chunk_hash: str
    start_offset: int
    end_offset: int
    title_path: tuple[str, ...]
    metadata: Mapping[str, object] = _EMPTY_METADATA

    def __post_init__(self) -> None:
        if (
            type(self.chunk_index) is not int
            or self.chunk_index < 0
            or type(self.text) is not str
            or not self.text
            or "\x00" in self.text
            or "\r" in self.text
            or type(self.start_offset) is not int
            or type(self.end_offset) is not int
            or self.start_offset < 0
            or self.end_offset <= self.start_offset
            or self.end_offset - self.start_offset != len(self.text)
            or type(self.title_path) is not tuple
            or any(
                type(title) is not str or not title or "\x00" in title for title in self.title_path
            )
            or not isinstance(self.metadata, Mapping)
        ):
            raise ValueError("chunk is invalid")
        metadata = _freeze_mapping(self.metadata, error_message="chunk metadata is invalid")
        object.__setattr__(self, "metadata", metadata)
        if (
            type(self.chunk_hash) is not str
            or _SHA256_PATTERN.fullmatch(self.chunk_hash) is None
            or self.chunk_hash != _chunk_hash(self.text, self.title_path, metadata)
        ):
            raise ValueError("chunk is invalid")

    @classmethod
    def from_source(
        cls,
        *,
        chunk_index: int,
        source_text: str,
        start_offset: int,
        end_offset: int,
        title_path: tuple[str, ...],
        metadata: Mapping[str, object] = _EMPTY_METADATA,
    ) -> Chunk:
        """Create a chunk only by slicing a normalized source string."""

        if (
            type(source_text) is not str
            or type(start_offset) is not int
            or type(end_offset) is not int
            or start_offset < 0
            or end_offset <= start_offset
            or end_offset > len(source_text)
            or not isinstance(metadata, Mapping)
        ):
            raise ValueError("chunk source span is invalid")
        frozen_metadata = _freeze_mapping(metadata, error_message="chunk metadata is invalid")
        text = source_text[start_offset:end_offset]
        return cls(
            chunk_index=chunk_index,
            text=text,
            chunk_hash=_chunk_hash(text, title_path, frozen_metadata),
            start_offset=start_offset,
            end_offset=end_offset,
            title_path=title_path,
            metadata=frozen_metadata,
        )

    def as_record(self) -> dict[str, object]:
        """Return a fresh canonical-JSON-compatible manifest record."""

        return {
            "chunk_index": self.chunk_index,
            "text": self.text,
            "chunk_hash": self.chunk_hash,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "title_path": list(self.title_path),
            "metadata": _plain_json(self.metadata),
        }

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(chunk_index={self.chunk_index}, "
            f"start_offset={self.start_offset}, end_offset={self.end_offset}, "
            f"chunk_hash={self.chunk_hash!r})"
        )


@runtime_checkable
class Chunker(Protocol):
    """Pluggable boundary-producing interface, including future semantic chunkers."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def config(self) -> Mapping[str, object]: ...

    def chunk(self, artifact: ParsedArtifact) -> Iterator[Chunk]: ...


@dataclass(frozen=True, slots=True)
class _BoundaryIndex:
    headings: Sequence[int]
    structural_starts: Sequence[int]
    code_blocks: Sequence[int]
    paragraphs: Sequence[int]
    paragraph_starts: Sequence[int]
    paragraph_ends: Sequence[int]
    list_items: Sequence[int]
    title_offsets: Sequence[int]
    title_paths: Sequence[tuple[str, ...]]
    compact_title_paths: _CompactTitlePathLookup | None = None


@dataclass(slots=True)
class _CompactTitlePathLookup:
    boundaries: StructuralBoundaries
    _path_cache: dict[int, tuple[str, ...]] = field(default_factory=lambda: {0: ()})

    def at(self, offset: int) -> tuple[str, ...]:
        path_id = self.boundaries._title_path_id_at_offset(offset)
        missing: list[int] = []
        ancestor_id = path_id
        while ancestor_id not in self._path_cache:
            missing.append(ancestor_id)
            ancestor_id = self.boundaries._title_parent(ancestor_id)
        path = self._path_cache[ancestor_id]
        for node_id in reversed(missing):
            path += (self.boundaries._decode_title_node(node_id),)
            self._path_cache[node_id] = path

        reachable = {0}
        ancestor_id = path_id
        while ancestor_id:
            reachable.add(ancestor_id)
            ancestor_id = self.boundaries._title_parent(ancestor_id)
        for cached_id in tuple(self._path_cache):
            if cached_id not in reachable:
                del self._path_cache[cached_id]
        return self._path_cache[path_id]


_PointMode = Literal["starts", "ends", "both"]


@dataclass(frozen=True, slots=True)
class _PackedRunPoints(Sequence[int]):
    template_points: array[int]
    template_lengths: array[int]
    template_starts: array[int]
    template_counts: array[int]
    cycle_strides: array[int]
    initial_skips: array[int]
    repeat_skips: array[int]
    cumulative_counts: array[int]
    first_points: array[int]
    last_points: array[int]

    def __len__(self) -> int:
        return self.cumulative_counts[-1] if self.cumulative_counts else 0

    def _point_position(self, segment_index: int, local_index: int) -> tuple[int, int]:
        template_count = self.template_counts[segment_index]
        first_count = template_count - self.initial_skips[segment_index]
        if local_index < first_count:
            return 0, self.initial_skips[segment_index] + local_index
        repeat_count = template_count - self.repeat_skips[segment_index]
        repeated_index = local_index - first_count
        cycle_index, template_position = divmod(repeated_index, repeat_count)
        return cycle_index + 1, self.repeat_skips[segment_index] + template_position

    def _point_in_segment(self, segment_index: int, local_index: int) -> int:
        cycle_index, template_position = self._point_position(segment_index, local_index)
        return (
            self.template_points[self.template_starts[segment_index] + template_position]
            + cycle_index * self.cycle_strides[segment_index]
        )

    def _segment_point_count(self, segment_index: int) -> int:
        previous_count = self.cumulative_counts[segment_index - 1] if segment_index else 0
        return self.cumulative_counts[segment_index] - previous_count

    def first(self, lower_inclusive: int, upper_exclusive: int) -> int | None:
        lower_position = 0
        upper_position = len(self.template_starts)
        while lower_position < upper_position:
            middle = (lower_position + upper_position) // 2
            if self.last_points[middle] < lower_inclusive:
                lower_position = middle + 1
            else:
                upper_position = middle
        if lower_position >= len(self.template_starts):
            return None
        local_lower = 0
        local_upper = self._segment_point_count(lower_position)
        while local_lower < local_upper:
            middle = (local_lower + local_upper) // 2
            if self._point_in_segment(lower_position, middle) < lower_inclusive:
                local_lower = middle + 1
            else:
                local_upper = middle
        candidate = self._point_in_segment(lower_position, local_lower)
        return candidate if candidate < upper_exclusive else None

    def last(self, lower_exclusive: int, upper_inclusive: int) -> int | None:
        lower_position = 0
        upper_position = len(self.template_starts)
        while lower_position < upper_position:
            middle = (lower_position + upper_position) // 2
            if self.first_points[middle] <= upper_inclusive:
                lower_position = middle + 1
            else:
                upper_position = middle
        position = lower_position - 1
        if position < 0:
            return None
        local_lower = 0
        local_upper = self._segment_point_count(position)
        while local_lower < local_upper:
            middle = (local_lower + local_upper) // 2
            if self._point_in_segment(position, middle) <= upper_inclusive:
                local_lower = middle + 1
            else:
                local_upper = middle
        candidate_index = local_lower - 1
        if candidate_index < 0:
            return None
        candidate = self._point_in_segment(position, candidate_index)
        return candidate if candidate > lower_exclusive else None

    def _locate(self, offset: int) -> tuple[int, int] | None:
        lower_position = 0
        upper_position = len(self.template_starts)
        while lower_position < upper_position:
            middle = (lower_position + upper_position) // 2
            if self.last_points[middle] < offset:
                lower_position = middle + 1
            else:
                upper_position = middle
        if lower_position >= len(self.template_starts):
            return None
        local_lower = 0
        local_upper = self._segment_point_count(lower_position)
        while local_lower < local_upper:
            middle = (local_lower + local_upper) // 2
            if self._point_in_segment(lower_position, middle) < offset:
                local_lower = middle + 1
            else:
                local_upper = middle
        if (
            local_lower >= self._segment_point_count(lower_position)
            or self._point_in_segment(lower_position, local_lower) != offset
        ):
            return None
        return lower_position, local_lower

    def contains(self, offset: int) -> bool:
        return self._locate(offset) is not None

    def boundary_end_starting_at(self, offset: int) -> int | None:
        located = self._locate(offset)
        if located is None or not self.template_lengths:
            return None
        segment_index, local_index = located
        _cycle_index, template_position = self._point_position(segment_index, local_index)
        return (
            offset + self.template_lengths[self.template_starts[segment_index] + template_position]
        )

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[int, ...]: ...

    def __getitem__(self, index: int | slice) -> int | tuple[int, ...]:
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("boundary point index is out of range")
        position = bisect_right(self.cumulative_counts, index)
        previous_count = self.cumulative_counts[position - 1] if position else 0
        return self._point_in_segment(position, index - previous_count)


def _packed_points(
    boundaries: StructuralBoundaries,
    kinds: frozenset[BoundaryKind],
    mode: _PointMode,
) -> _PackedRunPoints:
    template_points = array("I")
    template_lengths = array("I")
    template_starts = array("I")
    template_counts = array("I")
    cycle_strides = array("I")
    initial_skips = array("B")
    repeat_skips = array("B")
    cumulative_counts = array("I")
    first_points = array("I")
    last_points = array("I")
    total = 0
    previous_last: int | None = None
    for boundary_segment in range(boundaries._storage_segment_count):
        points: list[int] = []
        lengths: list[int] = []
        boundary_template_start = boundaries._segment_template_start(boundary_segment)
        boundary_template_count = boundaries._segment_template_count(boundary_segment)
        for template_index in range(
            boundary_template_start,
            boundary_template_start + boundary_template_count,
        ):
            if boundaries._template_kind(template_index) not in kinds:
                continue
            start_offset = boundaries._template_start_offset(template_index)
            boundary_length = boundaries._template_boundary_length(template_index)
            if mode != "ends" and (not points or points[-1] != start_offset):
                points.append(start_offset)
                if mode == "starts":
                    lengths.append(boundary_length)
            end_offset = start_offset + boundary_length
            if mode != "starts" and (not points or points[-1] != end_offset):
                points.append(end_offset)
        if not points:
            continue
        point_template_start = len(template_points)
        template_points.extend(points)
        if mode == "starts":
            template_lengths.extend(lengths)
        repeat_count = boundaries._segment_repeat_count(boundary_segment)
        cycle_stride = boundaries._segment_cycle_stride(boundary_segment)
        initial_skip = int(previous_last == points[0])
        repeat_skip = int(repeat_count > 1 and points[-1] == points[0] + cycle_stride)
        first_count = len(points) - initial_skip
        later_count = len(points) - repeat_skip
        point_count = first_count + (repeat_count - 1) * later_count
        if point_count <= 0:
            continue
        template_starts.append(point_template_start)
        template_counts.append(len(points))
        cycle_strides.append(cycle_stride)
        initial_skips.append(initial_skip)
        repeat_skips.append(repeat_skip)
        total += point_count
        cumulative_counts.append(total)
        first_point = points[initial_skip] if first_count else points[repeat_skip] + cycle_stride
        last_template_position = len(points) - 1
        last_point = points[last_template_position] + (repeat_count - 1) * cycle_stride
        first_points.append(first_point)
        last_points.append(last_point)
        previous_last = last_point
    return _PackedRunPoints(
        template_points,
        template_lengths,
        template_starts,
        template_counts,
        cycle_strides,
        initial_skips,
        repeat_skips,
        cumulative_counts,
        first_points,
        last_points,
    )


def _index_compact_boundaries(boundaries: StructuralBoundaries) -> _BoundaryIndex:
    headings: frozenset[BoundaryKind] = frozenset({"heading"})
    paragraphs: frozenset[BoundaryKind] = frozenset({"paragraph"})
    list_items: frozenset[BoundaryKind] = frozenset({"list_item"})
    code_blocks: frozenset[BoundaryKind] = frozenset({"fenced_code_block"})
    structural: frozenset[BoundaryKind] = frozenset({"paragraph", "list_item", "fenced_code_block"})
    heading_starts = _packed_points(boundaries, headings, "starts")
    structural_starts = _packed_points(boundaries, structural, "starts")
    paragraph_starts = _packed_points(boundaries, paragraphs, "starts")
    paragraph_ends = _packed_points(boundaries, paragraphs, "ends")
    return _BoundaryIndex(
        headings=heading_starts,
        structural_starts=structural_starts,
        code_blocks=_packed_points(boundaries, code_blocks, "both"),
        paragraphs=_packed_points(boundaries, paragraphs, "both"),
        paragraph_starts=paragraph_starts,
        paragraph_ends=paragraph_ends,
        list_items=_packed_points(boundaries, list_items, "both"),
        title_offsets=(),
        title_paths=(),
        compact_title_paths=_CompactTitlePathLookup(boundaries),
    )


def _append_distinct(points: list[int], offset: int) -> None:
    if not points or points[-1] != offset:
        points.append(offset)


def _index_boundaries(boundaries: Sequence[StructuralBoundary]) -> _BoundaryIndex:
    if isinstance(boundaries, StructuralBoundaries):
        return _index_compact_boundaries(boundaries)
    headings: list[int] = []
    structural_starts: list[int] = []
    code_blocks: list[int] = []
    paragraphs: list[int] = []
    paragraph_starts: list[int] = []
    paragraph_ends: list[int] = []
    list_items: list[int] = []
    title_offsets: list[int] = []
    title_paths: list[tuple[str, ...]] = []
    for boundary in boundaries:
        title_offsets.append(boundary.start_offset)
        title_paths.append(boundary.title_path)
        if boundary.kind == "heading":
            _append_distinct(headings, boundary.start_offset)
            continue
        _append_distinct(structural_starts, boundary.start_offset)
        if boundary.kind == "fenced_code_block":
            target = code_blocks
        elif boundary.kind == "paragraph":
            target = paragraphs
            paragraph_starts.append(boundary.start_offset)
            paragraph_ends.append(boundary.end_offset)
        else:
            target = list_items
        _append_distinct(target, boundary.start_offset)
        _append_distinct(target, boundary.end_offset)
    return _BoundaryIndex(
        headings=tuple(headings),
        structural_starts=tuple(structural_starts),
        code_blocks=tuple(code_blocks),
        paragraphs=tuple(paragraphs),
        paragraph_starts=tuple(paragraph_starts),
        paragraph_ends=tuple(paragraph_ends),
        list_items=tuple(list_items),
        title_offsets=tuple(title_offsets),
        title_paths=tuple(title_paths),
    )


def _last_point(points: Sequence[int], lower_exclusive: int, upper_inclusive: int) -> int | None:
    if isinstance(points, _PackedRunPoints):
        return points.last(lower_exclusive, upper_inclusive)
    index = bisect_right(points, upper_inclusive) - 1
    if index >= 0 and points[index] > lower_exclusive:
        return points[index]
    return None


def _first_point(points: Sequence[int], lower_inclusive: int, upper_exclusive: int) -> int | None:
    if isinstance(points, _PackedRunPoints):
        return points.first(lower_inclusive, upper_exclusive)
    index = bisect_left(points, lower_inclusive)
    if index < len(points) and points[index] < upper_exclusive:
        return points[index]
    return None


def _contains_point(points: Sequence[int], offset: int) -> bool:
    if isinstance(points, _PackedRunPoints):
        return points.contains(offset)
    index = bisect_left(points, offset)
    return index < len(points) and points[index] == offset


def _paragraph_end_starting_at(index: _BoundaryIndex, offset: int) -> int | None:
    if isinstance(index.paragraph_starts, _PackedRunPoints):
        return index.paragraph_starts.boundary_end_starting_at(offset)
    position = bisect_left(index.paragraph_starts, offset)
    if position < len(index.paragraph_starts) and index.paragraph_starts[position] == offset:
        return index.paragraph_ends[position]
    return None


def _last_sentence_end(
    text: str,
    scan_start: int,
    lower_exclusive: int,
    upper_inclusive: int,
) -> int | None:
    selected: int | None = None
    for matched in _SENTENCE_END.finditer(text, scan_start, upper_inclusive):
        end = matched.end()
        if end > lower_exclusive:
            selected = end
    return selected


def _first_sentence_end(
    text: str,
    scan_start: int,
    lower_inclusive: int,
    upper_exclusive: int,
) -> int | None:
    for matched in _SENTENCE_END.finditer(text, scan_start, upper_exclusive):
        end = matched.end()
        if lower_inclusive <= end < upper_exclusive:
            return end
    return None


def _last_newline_point(text: str, lower_exclusive: int, upper_inclusive: int) -> int | None:
    newline = text.rfind("\n", lower_exclusive, upper_inclusive)
    return newline + 1 if newline >= 0 else None


def _first_newline_point(text: str, lower_inclusive: int, upper_exclusive: int) -> int | None:
    newline = text.find("\n", max(0, lower_inclusive - 1), max(0, upper_exclusive - 1))
    return newline + 1 if newline >= 0 else None


def _is_regional_indicator(character: str) -> bool:
    codepoint = ord(character)
    return _REGIONAL_INDICATOR_START <= codepoint <= _REGIONAL_INDICATOR_END


def _in_codepoint_ranges(codepoint: int, ranges: tuple[tuple[int, int], ...]) -> bool:
    return any(start <= codepoint <= end for start, end in ranges)


def _is_grapheme_extend(character: str) -> bool:
    codepoint = ord(character)
    return (
        unicodedata.category(character).startswith("M")
        or codepoint == 0x200C
        or 0xFE00 <= codepoint <= 0xFE0F
        or 0xFF9E <= codepoint <= 0xFF9F
        or 0xE0100 <= codepoint <= 0xE01EF
        or 0x1F3FB <= codepoint <= 0x1F3FF
        or _TAG_CHARACTER_START <= codepoint <= _TAG_CHARACTER_END
    )


def _is_gcb_extend_for_emoji_zwj(character: str) -> bool:
    return _is_grapheme_extend(character) and (
        unicodedata.category(character) != "Mc"
        or _in_codepoint_ranges(ord(character), _GCB_EXTEND_MC_RANGES)
    )


def _is_prepend(character: str) -> bool:
    return _in_codepoint_ranges(ord(character), _PREPEND_RANGES)


def _is_extended_pictographic(character: str) -> bool:
    return _in_codepoint_ranges(ord(character), _EXTENDED_PICTOGRAPHIC_RANGES)


def _hangul_type(character: str) -> int:
    codepoint = ord(character)
    if 0x1100 <= codepoint <= 0x115F or 0xA960 <= codepoint <= 0xA97C:
        return _HANGUL_L
    if 0x1160 <= codepoint <= 0x11A7 or 0xD7B0 <= codepoint <= 0xD7C6:
        return _HANGUL_V
    if 0x11A8 <= codepoint <= 0x11FF or 0xD7CB <= codepoint <= 0xD7FB:
        return _HANGUL_T
    if 0xAC00 <= codepoint <= 0xD7A3:
        return _HANGUL_LV if (codepoint - 0xAC00) % 28 == 0 else _HANGUL_LVT
    return _HANGUL_NONE


def _splits_hangul_syllable(before: str, after: str) -> bool:
    before_type = _hangul_type(before)
    after_type = _hangul_type(after)
    return (
        (before_type == _HANGUL_L and after_type in {_HANGUL_L, _HANGUL_V, _HANGUL_LV, _HANGUL_LVT})
        or (before_type in {_HANGUL_LV, _HANGUL_V} and after_type in {_HANGUL_V, _HANGUL_T})
        or (before_type in {_HANGUL_LVT, _HANGUL_T} and after_type == _HANGUL_T)
    )


def _unicode_script(character: str) -> str:
    name = unicodedata.name(character, "")
    return name.split(" ", 1)[0] if name else ""


def _is_virama(character: str) -> bool:
    return unicodedata.combining(character) == 9 or "VIRAMA" in unicodedata.name(character, "")


def _splits_indic_virama_conjunct(text: str, offset: int, safe_anchor: int) -> bool:
    after = text[offset]
    after_script = _unicode_script(after)
    if not unicodedata.category(after).startswith("L") or not after_script:
        return False
    index = offset - 1
    while index >= safe_anchor:
        character = text[index]
        if _is_virama(character):
            if _unicode_script(character) != after_script:
                return False
            base_index = index - 1
            while base_index >= safe_anchor and _is_grapheme_extend(text[base_index]):
                base_index -= 1
            return (
                base_index >= safe_anchor
                and unicodedata.category(text[base_index]).startswith("L")
                and _unicode_script(text[base_index]) == after_script
            )
        if character != "\u200d" and not _is_grapheme_extend(character):
            return False
        index -= 1
    return False


def _splits_emoji_zwj_sequence(text: str, offset: int, safe_anchor: int) -> bool:
    if text[offset - 1] != "\u200d" or not _is_extended_pictographic(text[offset]):
        return False
    index = offset - 2
    while index >= safe_anchor and _is_gcb_extend_for_emoji_zwj(text[index]):
        index -= 1
    return index >= safe_anchor and _is_extended_pictographic(text[index])


def _splits_regional_indicator_pair(text: str, offset: int, safe_anchor: int) -> bool:
    if not (_is_regional_indicator(text[offset - 1]) and _is_regional_indicator(text[offset])):
        return False
    preceding_count = 0
    index = offset - 1
    while index >= safe_anchor and _is_regional_indicator(text[index]):
        preceding_count += 1
        index -= 1
    return preceding_count % 2 == 1


def _unsafe_boundary(text: str, offset: int, safe_anchor: int) -> bool:
    if offset <= 0 or offset >= len(text):
        return False
    before = text[offset - 1]
    after = text[offset]
    return (
        (before == "\r" and after == "\n")
        or after == "\u200d"
        or _is_grapheme_extend(after)
        or _is_prepend(before)
        or _splits_hangul_syllable(before, after)
        or _splits_indic_virama_conjunct(text, offset, safe_anchor)
        or _splits_emoji_zwj_sequence(text, offset, safe_anchor)
        or _splits_regional_indicator_pair(text, offset, safe_anchor)
    )


def _safe_end(text: str, hard_end: int, start: int) -> int:
    candidate = hard_end
    while candidate > start and _unsafe_boundary(text, candidate, start):
        candidate -= 1
    return hard_end if candidate == start else candidate


def _safe_start(text: str, target: int, end: int, safe_anchor: int) -> int:
    candidate = target
    while candidate < end and _unsafe_boundary(text, candidate, safe_anchor):
        candidate += 1
    return candidate


def _align_sentence_boundary(text: str, target: int, limit: int, safe_anchor: int) -> int:
    candidate = _safe_start(text, target, limit, safe_anchor)
    return (
        _safe_end(text, target, safe_anchor)
        if _unsafe_boundary(text, candidate, safe_anchor)
        else candidate
    )


def _title_path_at(index: _BoundaryIndex, offset: int) -> tuple[str, ...]:
    if index.compact_title_paths is not None:
        return index.compact_title_paths.at(offset)
    position = bisect_right(index.title_offsets, offset) - 1
    return index.title_paths[position] if position >= 0 else ()


def _choose_end(
    *,
    text: str,
    parser_name: str,
    index: _BoundaryIndex,
    start: int,
    previous_end: int,
    minimum_end: int,
    hard_end: int,
    final_window: bool,
) -> tuple[int, bool]:
    preferred_floor = max(previous_end, minimum_end)
    structural: tuple[Sequence[int], ...]
    heading_content_start: int | None = None
    if parser_name == "markdown_text_v1":
        next_heading = _first_point(
            index.headings,
            max(start, previous_end) + 1,
            hard_end + 1,
        )
        if next_heading is not None:
            return next_heading, True
        if final_window:
            return hard_end, False
        structural = (index.code_blocks, index.paragraphs, index.list_items)
        if _contains_point(index.headings, start):
            heading_content_start = _first_point(
                index.structural_starts,
                start + 1,
                minimum_end + 1,
            )
    else:
        if final_window:
            return hard_end, False
        structural = (index.paragraphs,)
    for points in structural:
        selected = _last_point(points, preferred_floor, hard_end)
        if selected is not None:
            return selected, False
    current_paragraph_end = _paragraph_end_starting_at(index, start)
    for points in structural:
        selected = _last_point(points, previous_end, minimum_end)
        if selected == heading_content_start or (
            parser_name == "markdown_text_v1"
            and points is index.paragraphs
            and selected == current_paragraph_end
        ):
            selected = None
        if selected is not None:
            return selected, True
    if parser_name != "markdown_text_v1":
        newline = _last_newline_point(text, preferred_floor, hard_end)
        if newline is not None:
            return newline, False
        newline = _last_newline_point(text, previous_end, minimum_end)
        if newline is not None:
            return newline, True
    sentence = _last_sentence_end(text, start, preferred_floor, hard_end)
    if sentence is not None:
        return _align_sentence_boundary(text, sentence, hard_end, start), False
    sentence = _last_sentence_end(text, start, previous_end, minimum_end)
    if sentence is not None:
        return _align_sentence_boundary(text, sentence, hard_end, start), True
    safe_end = _safe_end(text, hard_end, start)
    return safe_end, False


def _choose_overlap_start(
    *,
    text: str,
    parser_name: str,
    index: _BoundaryIndex,
    chunk_start: int,
    overlap_target: int,
    end: int,
) -> int:
    if _contains_point(index.headings, end):
        return end
    structural = (
        (index.headings, index.code_blocks, index.paragraphs, index.list_items)
        if parser_name == "markdown_text_v1"
        else (index.paragraphs,)
    )
    for points in structural:
        selected = _first_point(points, overlap_target, end)
        if selected is not None:
            return selected
    if parser_name != "markdown_text_v1":
        newline = _first_newline_point(text, overlap_target, end)
        if newline is not None:
            return newline
    sentence = _first_sentence_end(text, chunk_start, overlap_target, end)
    return (
        _align_sentence_boundary(text, sentence, end, chunk_start)
        if sentence is not None
        else _safe_start(text, overlap_target, end, chunk_start)
    )


class RecursiveTextChunker:
    """Split normalized text using stable structural and lexical priorities."""

    __slots__ = ("_config", "_max_chunk_codepoints", "_target_overlap_codepoints")

    name: Final = "recursive_text_v1"
    version: Final = "1"

    def __init__(
        self,
        *,
        max_chunk_codepoints: int = DEFAULT_CHUNK_CODEPOINTS,
        target_overlap_codepoints: int = DEFAULT_OVERLAP_CODEPOINTS,
    ) -> None:
        if (
            type(max_chunk_codepoints) is not int
            or not 1 <= max_chunk_codepoints <= MAX_CHUNK_CODEPOINTS
            or type(target_overlap_codepoints) is not int
            or not 0 <= target_overlap_codepoints < max_chunk_codepoints
            or target_overlap_codepoints > TARGET_OVERLAP_CODEPOINTS
        ):
            raise ValueError("document chunker config is invalid")
        self._max_chunk_codepoints = max_chunk_codepoints
        self._target_overlap_codepoints = target_overlap_codepoints
        self._config: Mapping[str, object] = MappingProxyType(
            {
                "max_chunk_codepoints": max_chunk_codepoints,
                "target_overlap_codepoints": target_overlap_codepoints,
            }
        )

    @property
    def config(self) -> Mapping[str, object]:
        return self._config

    @property
    def config_hash(self) -> str:
        return canonical_sha256(
            {
                "config": dict(self.config),
                "name": self.name,
                "version": self.version,
            }
        )

    def chunk(self, artifact: ParsedArtifact) -> Iterator[Chunk]:
        text = artifact.text
        if not text:
            return

        boundary_index = _index_boundaries(artifact.boundaries)

        start = 0
        previous_end = 0
        chunk_index = 0
        while start < len(text):
            hard_end = min(start + self._max_chunk_codepoints, len(text))
            minimum_end = start + self._target_overlap_codepoints
            end, suppress_overlap = _choose_end(
                text=text,
                parser_name=artifact.parser_name,
                index=boundary_index,
                start=start,
                previous_end=previous_end,
                minimum_end=minimum_end,
                hard_end=hard_end,
                final_window=hard_end == len(text),
            )
            if end <= previous_end:
                reduced_overlap_start = _safe_start(text, start + 1, previous_end, start)
                if reduced_overlap_start <= start or reduced_overlap_start > previous_end:
                    raise ValueError("document chunker could not make progress")
                start = reduced_overlap_start
                continue
            yield Chunk.from_source(
                chunk_index=chunk_index,
                source_text=text,
                start_offset=start,
                end_offset=end,
                title_path=_title_path_at(boundary_index, start),
            )
            if end == len(text):
                return

            if suppress_overlap:
                next_start = end
            else:
                overlap_target = max(start + 1, end - self._target_overlap_codepoints)
                next_start = _choose_overlap_start(
                    text=text,
                    parser_name=artifact.parser_name,
                    index=boundary_index,
                    chunk_start=start,
                    overlap_target=overlap_target,
                    end=end,
                )
            if next_start <= start or next_start > end:
                raise ValueError("document chunker could not make progress")
            previous_end = end
            start = next_start
            chunk_index += 1


_RECURSIVE_TEXT_CHUNKER: Final = RecursiveTextChunker()


def chunker_for_name(name: str) -> Chunker:
    """Select a registered chunker without silently substituting future strategies."""

    if type(name) is not str or name != _RECURSIVE_TEXT_CHUNKER.name:
        raise ValueError("document chunker name is invalid")
    return _RECURSIVE_TEXT_CHUNKER


__all__ = [
    "CHUNK_SCHEMA_VERSION",
    "DEFAULT_CHUNK_CODEPOINTS",
    "DEFAULT_OVERLAP_CODEPOINTS",
    "MAX_CHUNK_CODEPOINTS",
    "TARGET_OVERLAP_CODEPOINTS",
    "Chunk",
    "Chunker",
    "RecursiveTextChunker",
    "chunker_for_name",
]
