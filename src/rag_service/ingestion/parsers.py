"""Deterministic, non-executing parsers for supported text documents."""

from __future__ import annotations

import hashlib
import re
from array import array
from bisect import bisect_right
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal, Protocol, overload

from rag_service.ingestion.validation import validate_document

BoundaryKind = Literal["heading", "paragraph", "list_item", "fenced_code_block"]

_BOUNDARY_KINDS: Final = frozenset({"heading", "paragraph", "list_item", "fenced_code_block"})
_BOUNDARY_KIND_BY_CODE: Final[tuple[BoundaryKind, ...]] = (
    "heading",
    "paragraph",
    "list_item",
    "fenced_code_block",
)
_BOUNDARY_CODE_BY_KIND: Final = MappingProxyType(
    {kind: code for code, kind in enumerate(_BOUNDARY_KIND_BY_CODE)}
)
_BOUNDARY_CONSTRUCTION_TOKEN: Final = object()
_MAX_BOUNDARY_PATTERN_LENGTH: Final = 16
_EMPTY_CONFIG: Final[Mapping[str, object]] = MappingProxyType({})
_ATX_HEADING: Final = re.compile(r"^[ ]{0,3}(?P<marker>#{1,6})(?:[ \t]+(?P<title>.*)|[ \t]*)$")
_SETEXT_UNDERLINE: Final = re.compile(r"^[ ]{0,3}(?P<marker>=+|-+)[ \t]*$")
_LIST_ITEM: Final = re.compile(
    r"^(?P<indent>[ \t]*)(?P<marker>[-+*]|[0-9]{1,9}[.)])(?P<spacing>[ \t]+)"
)
_FENCE_OPEN: Final = re.compile(r"^[ ]{0,3}(?P<fence>`{3,}|~{3,})(?P<info>.*)$")
MAX_TITLE_SEGMENT_UTF8_BYTES: Final = 1024
MAX_TITLE_PATH_UTF8_BYTES: Final = 4096


@dataclass(frozen=True, slots=True)
class StructuralBoundary:
    """A half-open Unicode code-point span in normalized source text."""

    kind: BoundaryKind
    start_offset: int
    end_offset: int
    title_path: tuple[str, ...]
    heading_level: int | None = None

    def __post_init__(self) -> None:
        _validate_title_path(self.title_path)
        if (
            self.kind not in _BOUNDARY_KINDS
            or type(self.start_offset) is not int
            or type(self.end_offset) is not int
            or self.start_offset < 0
            or self.end_offset <= self.start_offset
            or (
                self.kind == "heading"
                and (type(self.heading_level) is not int or not 1 <= self.heading_level <= 6)
            )
            or (self.kind != "heading" and self.heading_level is not None)
        ):
            raise ValueError("parser boundary is invalid")


class ParserError(Exception):
    """A stable, sanitized parser failure safe to expose to job orchestration."""


class StructuralMetadataLimitExceeded(ParserError, ValueError):
    """A permanent parser rejection for bounded structural metadata."""

    def __init__(self) -> None:
        super().__init__("document structural metadata exceeds limit")


def _validate_title_path(title_path: object) -> None:
    if type(title_path) is not tuple:
        raise ValueError("parser boundary is invalid")
    total_bytes = 0
    for title in title_path:
        if type(title) is not str or not title or "\x00" in title:
            raise ValueError("parser boundary is invalid")
        encoded_size = len(title.encode("utf-8"))
        if encoded_size > MAX_TITLE_SEGMENT_UTF8_BYTES:
            raise StructuralMetadataLimitExceeded
        total_bytes += encoded_size
        if total_bytes > MAX_TITLE_PATH_UTF8_BYTES:
            raise StructuralMetadataLimitExceeded


class StructuralBoundaries(Sequence[StructuralBoundary]):
    """Immutable exact boundary sequence backed by packed periodic segments."""

    __slots__ = (
        "_boundary_lengths",
        "_cumulative_counts",
        "_first_start_offsets",
        "_heading_levels",
        "_kinds",
        "_segment_cycle_strides",
        "_segment_first_start_offsets",
        "_segment_repeat_counts",
        "_segment_template_counts",
        "_segment_template_starts",
        "_title_data",
        "_title_end_offsets",
        "_title_parents",
        "_title_path_ids",
    )

    def __init__(
        self,
        *,
        _construction_token: object,
        kinds: array[int],
        first_start_offsets: array[int],
        boundary_lengths: array[int],
        strides: array[int],
        cumulative_counts: array[int],
        title_path_ids: array[int],
        heading_levels: array[int],
        title_data: bytes,
        title_end_offsets: array[int],
        title_parents: array[int],
        segment_template_starts: array[int] | None = None,
        segment_template_counts: array[int] | None = None,
        segment_repeat_counts: array[int] | None = None,
        segment_first_start_offsets: array[int] | None = None,
    ) -> None:
        if _construction_token is not _BOUNDARY_CONSTRUCTION_TOKEN:
            raise ValueError("parsed artifact is invalid")
        if (
            segment_template_starts is None
            or segment_template_counts is None
            or segment_repeat_counts is None
            or segment_first_start_offsets is None
        ):
            raise ValueError("parsed artifact is invalid")
        self._kinds = kinds
        self._first_start_offsets = first_start_offsets
        self._boundary_lengths = boundary_lengths
        self._segment_cycle_strides = strides
        self._cumulative_counts = cumulative_counts
        self._segment_template_starts = segment_template_starts
        self._segment_template_counts = segment_template_counts
        self._segment_repeat_counts = segment_repeat_counts
        self._segment_first_start_offsets = segment_first_start_offsets
        self._title_path_ids = title_path_ids
        self._heading_levels = heading_levels
        self._title_data = title_data
        self._title_end_offsets = title_end_offsets
        self._title_parents = title_parents

    @classmethod
    def from_sequence(cls, boundaries: Sequence[StructuralBoundary]) -> StructuralBoundaries:
        builder = _BoundaryBuilder()
        builder.extend(boundaries)
        return builder.finish()

    @property
    def storage_run_count(self) -> int:
        return len(self._kinds)

    @property
    def _storage_segment_count(self) -> int:
        return len(self._segment_template_starts)

    def _title_path(self, path_id: int) -> tuple[str, ...]:
        titles: list[str] = []
        while path_id:
            titles.append(self._decode_title_node(path_id))
            path_id = self._title_parent(path_id)
        titles.reverse()
        return tuple(titles)

    def _title_parent(self, path_id: int) -> int:
        return self._title_parents[path_id - 1] if path_id else 0

    def _decode_title_node(self, path_id: int) -> str:
        title_index = path_id - 1
        start_offset = self._title_end_offsets[title_index - 1] if title_index else 0
        end_offset = self._title_end_offsets[title_index]
        return self._title_data[start_offset:end_offset].decode("utf-8")

    def _segment_template_start(self, segment_index: int) -> int:
        return self._segment_template_starts[segment_index]

    def _segment_template_count(self, segment_index: int) -> int:
        return self._segment_template_counts[segment_index]

    def _segment_repeat_count(self, segment_index: int) -> int:
        return self._segment_repeat_counts[segment_index]

    def _segment_cycle_stride(self, segment_index: int) -> int:
        return self._segment_cycle_strides[segment_index]

    def _template_kind(self, template_index: int) -> BoundaryKind:
        return _BOUNDARY_KIND_BY_CODE[self._kinds[template_index]]

    def _template_start_offset(self, template_index: int) -> int:
        return self._first_start_offsets[template_index]

    def _template_boundary_length(self, template_index: int) -> int:
        return self._boundary_lengths[template_index]

    def _segment_position(self, segment_index: int, local_index: int) -> tuple[int, int]:
        template_count = self._segment_template_counts[segment_index]
        return divmod(local_index, template_count)

    def _boundary_at_segment_position(
        self,
        segment_index: int,
        cycle_index: int,
        template_position: int,
        title_path: tuple[str, ...] | None = None,
    ) -> StructuralBoundary:
        template_index = self._segment_template_starts[segment_index] + template_position
        start_offset = (
            self._first_start_offsets[template_index]
            + cycle_index * self._segment_cycle_strides[segment_index]
        )
        path = (
            self._title_path(self._title_path_ids[template_index])
            if title_path is None
            else title_path
        )
        heading_level = self._heading_levels[template_index]
        return StructuralBoundary(
            _BOUNDARY_KIND_BY_CODE[self._kinds[template_index]],
            start_offset,
            start_offset + self._boundary_lengths[template_index],
            path,
            heading_level or None,
        )

    def _title_path_id_at_offset(self, offset: int) -> int:
        segment_index = bisect_right(self._segment_first_start_offsets, offset) - 1
        if segment_index < 0:
            return 0
        template_start = self._segment_template_starts[segment_index]
        template_count = self._segment_template_counts[segment_index]
        repeat_count = self._segment_repeat_counts[segment_index]
        stride = self._segment_cycle_strides[segment_index]
        cycle_index = 0
        if repeat_count > 1:
            cycle_index = min(
                repeat_count - 1,
                max(0, (offset - self._first_start_offsets[template_start]) // stride),
            )
        adjusted_offset = offset - cycle_index * stride
        lower = template_start
        upper = template_start + template_count
        while lower < upper:
            middle = (lower + upper) // 2
            if self._first_start_offsets[middle] <= adjusted_offset:
                lower = middle + 1
            else:
                upper = middle
        template_index = lower - 1
        if template_index < template_start and cycle_index:
            template_index = template_start + template_count - 1
        return self._title_path_ids[template_index] if template_index >= template_start else 0

    def _title_path_for_id(self, path_id: int) -> tuple[str, ...]:
        return self._title_path(path_id)

    def _validate_for_text_length(self, text_length: int) -> None:
        previous_end = 0
        previous_count = 0
        expected_template_start = 0
        if not (
            len(self._kinds)
            == len(self._first_start_offsets)
            == len(self._boundary_lengths)
            == len(self._title_path_ids)
            == len(self._heading_levels)
        ):
            raise ValueError("parsed artifact is invalid")
        if not (
            self._storage_segment_count
            == len(self._segment_template_counts)
            == len(self._segment_repeat_counts)
            == len(self._segment_cycle_strides)
            == len(self._segment_first_start_offsets)
            == len(self._cumulative_counts)
        ):
            raise ValueError("parsed artifact is invalid")
        for segment_index in range(self._storage_segment_count):
            template_start = self._segment_template_starts[segment_index]
            template_count = self._segment_template_counts[segment_index]
            repeat_count = self._segment_repeat_counts[segment_index]
            stride = self._segment_cycle_strides[segment_index]
            count = self._cumulative_counts[segment_index] - previous_count
            if (
                template_start != expected_template_start
                or template_count <= 0
                or repeat_count <= 0
                or count != template_count * repeat_count
            ):
                raise ValueError("parsed artifact is invalid")
            first_start = self._first_start_offsets[template_start]
            if first_start != self._segment_first_start_offsets[segment_index]:
                raise ValueError("parsed artifact is invalid")
            cycle_previous_end = first_start
            for template_index in range(template_start, template_start + template_count):
                boundary_length = self._boundary_lengths[template_index]
                template_boundary_start = self._first_start_offsets[template_index]
                if boundary_length <= 0 or template_boundary_start < cycle_previous_end:
                    raise ValueError("parsed artifact is invalid")
                cycle_previous_end = template_boundary_start + boundary_length
            if repeat_count > 1 and stride < cycle_previous_end - first_start:
                raise ValueError("parsed artifact is invalid")
            last_end = cycle_previous_end + stride * (repeat_count - 1)
            if first_start < previous_end or last_end > text_length:
                raise ValueError("parsed artifact is invalid")
            previous_end = last_end
            previous_count = self._cumulative_counts[segment_index]
            expected_template_start += template_count
        if expected_template_start != len(self._kinds):
            raise ValueError("parsed artifact is invalid")

    def __len__(self) -> int:
        return self._cumulative_counts[-1] if self._cumulative_counts else 0

    @overload
    def __getitem__(self, index: int) -> StructuralBoundary: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[StructuralBoundary, ...]: ...

    def __getitem__(
        self,
        index: int | slice,
    ) -> StructuralBoundary | tuple[StructuralBoundary, ...]:
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError("parser boundary index is out of range")
        segment_index = bisect_right(self._cumulative_counts, index)
        previous_count = self._cumulative_counts[segment_index - 1] if segment_index else 0
        cycle_index, template_position = self._segment_position(
            segment_index, index - previous_count
        )
        return self._boundary_at_segment_position(segment_index, cycle_index, template_position)

    def __iter__(self) -> Iterator[StructuralBoundary]:
        previous_path_id = -1
        title_path: tuple[str, ...] = ()
        for segment_index in range(self._storage_segment_count):
            template_start = self._segment_template_starts[segment_index]
            template_count = self._segment_template_counts[segment_index]
            for cycle_index in range(self._segment_repeat_counts[segment_index]):
                for template_position in range(template_count):
                    path_id = self._title_path_ids[template_start + template_position]
                    if path_id != previous_path_id:
                        title_path = self._title_path(path_id)
                        previous_path_id = path_id
                    yield self._boundary_at_segment_position(
                        segment_index,
                        cycle_index,
                        template_position,
                        title_path,
                    )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, Sequence) or len(self) != len(other):
            return False
        return all(left == right for left, right in zip(self, other, strict=True))

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(boundary_count={len(self)}, "
            f"storage_run_count={self.storage_run_count})"
        )


class _BoundaryBuilder:
    __slots__ = (
        "_active_position",
        "_active_repeat_count",
        "_active_stride",
        "_active_template",
        "_boundary_lengths",
        "_cumulative_counts",
        "_first_start_offsets",
        "_finished",
        "_heading_levels",
        "_kinds",
        "_last_end_offset",
        "_last_title_node_ids",
        "_last_title_path",
        "_pending",
        "_segment_first_start_offsets",
        "_segment_repeat_counts",
        "_segment_template_counts",
        "_segment_template_starts",
        "_strides",
        "_title_data",
        "_title_end_offsets",
        "_title_parents",
        "_title_path_ids",
    )

    def __init__(self) -> None:
        self._finished = False
        self._kinds = array("B")
        self._first_start_offsets = array("I")
        self._boundary_lengths = array("I")
        self._strides = array("I")
        self._cumulative_counts = array("I")
        self._segment_template_starts = array("I")
        self._segment_template_counts = array("I")
        self._segment_repeat_counts = array("I")
        self._segment_first_start_offsets = array("I")
        self._title_path_ids = array("I")
        self._heading_levels = array("B")
        self._title_data = bytearray()
        self._title_end_offsets = array("I")
        self._title_parents = array("I")
        self._last_title_path: tuple[str, ...] = ()
        self._last_title_node_ids: tuple[int, ...] = ()
        self._pending: list[tuple[int, int, int, int, int]] = []
        self._active_template: tuple[tuple[int, int, int, int, int], ...] = ()
        self._active_repeat_count = 0
        self._active_stride = 0
        self._active_position = 0
        self._last_end_offset = 0

    def _ensure_not_finished(self) -> None:
        if self._finished:
            raise ValueError("parser boundary builder is already finalized")

    def _title_path_id(self, title_path: tuple[str, ...]) -> int:
        self._ensure_not_finished()
        if title_path == self._last_title_path:
            return self._last_title_node_ids[-1] if self._last_title_node_ids else 0
        prefix_length = 0
        for previous, current in zip(self._last_title_path, title_path, strict=False):
            if previous != current:
                break
            prefix_length += 1
        node_ids = list(self._last_title_node_ids[:prefix_length])
        parent = node_ids[-1] if node_ids else 0
        for title in title_path[prefix_length:]:
            encoded_title = title.encode("utf-8")
            try:
                self._title_data.extend(encoded_title)
                self._title_end_offsets.append(len(self._title_data))
                self._title_parents.append(parent)
            except OverflowError:
                raise ValueError("parser boundary is invalid") from None
            parent = len(self._title_end_offsets)
            node_ids.append(parent)
        self._last_title_path = title_path
        self._last_title_node_ids = tuple(node_ids)
        return parent

    def _append_segment(
        self,
        records: Sequence[tuple[int, int, int, int, int]],
        repeat_count: int = 1,
        cycle_stride: int = 0,
    ) -> None:
        if not records:
            return
        template_start = len(self._kinds)
        previous_total = self._cumulative_counts[-1] if self._cumulative_counts else 0
        try:
            for kind_code, start_offset, boundary_length, title_path_id, heading_level in records:
                self._kinds.append(kind_code)
                self._first_start_offsets.append(start_offset)
                self._boundary_lengths.append(boundary_length)
                self._title_path_ids.append(title_path_id)
                self._heading_levels.append(heading_level)
            self._segment_template_starts.append(template_start)
            self._segment_template_counts.append(len(records))
            self._segment_repeat_counts.append(repeat_count)
            self._segment_first_start_offsets.append(records[0][1])
            self._strides.append(cycle_stride)
            self._cumulative_counts.append(previous_total + len(records) * repeat_count)
        except OverflowError:
            raise ValueError("parser boundary is invalid") from None

    @staticmethod
    def _period_stride(
        left: Sequence[tuple[int, int, int, int, int]],
        right: Sequence[tuple[int, int, int, int, int]],
    ) -> int | None:
        stride = right[0][1] - left[0][1]
        if stride <= 0:
            return None
        for previous, current in zip(left, right, strict=True):
            if (
                current[0] != previous[0]
                or current[2:] != previous[2:]
                or current[1] - previous[1] != stride
            ):
                return None
        return stride

    def _detect_pending_pattern(self) -> bool:
        pending_count = len(self._pending)
        for period in range(1, min(_MAX_BOUNDARY_PATTERN_LENGTH, pending_count // 2) + 1):
            left = self._pending[pending_count - 2 * period : pending_count - period]
            right = self._pending[pending_count - period :]
            stride = self._period_stride(left, right)
            if stride is None:
                continue
            self._append_segment(self._pending[: pending_count - 2 * period])
            self._active_template = tuple(left)
            self._active_repeat_count = 2
            self._active_stride = stride
            self._active_position = 0
            self._pending.clear()
            return True
        return False

    def _buffer_record(self, record: tuple[int, int, int, int, int]) -> None:
        self._pending.append(record)
        if self._detect_pending_pattern():
            return
        if len(self._pending) > _MAX_BOUNDARY_PATTERN_LENGTH * 2:
            self._append_segment(self._pending[:_MAX_BOUNDARY_PATTERN_LENGTH])
            del self._pending[:_MAX_BOUNDARY_PATTERN_LENGTH]

    def _flush_active(self) -> None:
        if not self._active_template:
            return
        self._append_segment(
            self._active_template,
            self._active_repeat_count,
            self._active_stride,
        )
        if self._active_position:
            cycle_offset = self._active_repeat_count * self._active_stride
            self._pending.extend(
                (kind, start + cycle_offset, length, path_id, level)
                for kind, start, length, path_id, level in self._active_template[
                    : self._active_position
                ]
            )
        self._active_template = ()
        self._active_repeat_count = 0
        self._active_stride = 0
        self._active_position = 0

    def _add_record(self, record: tuple[int, int, int, int, int]) -> None:
        if self._active_template:
            expected = self._active_template[self._active_position]
            expected_start = expected[1] + self._active_repeat_count * self._active_stride
            if (
                record[0] == expected[0]
                and record[1] == expected_start
                and record[2:] == expected[2:]
            ):
                self._active_position += 1
                if self._active_position == len(self._active_template):
                    self._active_repeat_count += 1
                    self._active_position = 0
                return
            self._flush_active()
        self._buffer_record(record)

    def add(self, boundary: StructuralBoundary) -> None:
        if type(boundary) is not StructuralBoundary:
            raise ValueError("parser boundary is invalid")
        self.add_fields(
            boundary.kind,
            boundary.start_offset,
            boundary.end_offset,
            boundary.title_path,
            boundary.heading_level,
        )

    def add_fields(
        self,
        kind: BoundaryKind,
        start_offset: int,
        end_offset: int,
        title_path: tuple[str, ...],
        heading_level: int | None = None,
    ) -> None:
        _validate_title_path(title_path)
        if (
            kind not in _BOUNDARY_KINDS
            or type(start_offset) is not int
            or type(end_offset) is not int
            or start_offset < 0
            or end_offset <= start_offset
            or (
                kind == "heading"
                and (type(heading_level) is not int or not 1 <= heading_level <= 6)
            )
            or (kind != "heading" and heading_level is not None)
        ):
            raise ValueError("parser boundary is invalid")
        boundary_length = end_offset - start_offset
        if start_offset < self._last_end_offset:
            raise ValueError("parser boundary is invalid")
        kind_code = _BOUNDARY_CODE_BY_KIND[kind]
        title_path_id = self._title_path_id(title_path)
        self._add_record(
            (kind_code, start_offset, boundary_length, title_path_id, heading_level or 0)
        )
        self._last_end_offset = end_offset

    def extend(self, boundaries: Sequence[StructuralBoundary]) -> None:
        self._ensure_not_finished()
        for boundary in boundaries:
            self.add(boundary)

    def finish(self) -> StructuralBoundaries:
        if self._finished:
            raise ValueError("parser boundary builder is already finalized")
        self._flush_active()
        self._append_segment(self._pending)
        self._pending.clear()
        boundaries = StructuralBoundaries(
            _construction_token=_BOUNDARY_CONSTRUCTION_TOKEN,
            kinds=self._kinds,
            first_start_offsets=self._first_start_offsets,
            boundary_lengths=self._boundary_lengths,
            strides=self._strides,
            cumulative_counts=self._cumulative_counts,
            title_path_ids=self._title_path_ids,
            heading_levels=self._heading_levels,
            title_data=bytes(self._title_data),
            title_end_offsets=self._title_end_offsets,
            title_parents=self._title_parents,
            segment_template_starts=self._segment_template_starts,
            segment_template_counts=self._segment_template_counts,
            segment_repeat_counts=self._segment_repeat_counts,
            segment_first_start_offsets=self._segment_first_start_offsets,
        )
        self._finished = True
        return boundaries


@dataclass(frozen=True, slots=True, repr=False, init=False)
class ParsedArtifact:
    """Normalized text and deterministic structural facts emitted by a parser."""

    text: str
    boundaries: StructuralBoundaries
    parser_name: str
    parser_version: str
    parser_config: Mapping[str, object]

    def __init__(
        self,
        text: str,
        boundaries: Sequence[StructuralBoundary],
        parser_name: str,
        parser_version: str,
        parser_config: Mapping[str, object],
    ) -> None:
        try:
            if isinstance(boundaries, StructuralBoundaries):
                compact_boundaries = boundaries
            elif type(boundaries) is tuple:
                compact_boundaries = StructuralBoundaries.from_sequence(boundaries)
            else:
                raise ValueError("parser boundary is invalid")
        except (TypeError, ValueError, OverflowError, ParserError):
            raise ValueError("parsed artifact is invalid") from None
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "boundaries", compact_boundaries)
        object.__setattr__(self, "parser_name", parser_name)
        object.__setattr__(self, "parser_version", parser_version)
        object.__setattr__(self, "parser_config", parser_config)
        self.__post_init__()

    def __post_init__(self) -> None:
        if (
            type(self.text) is not str
            or not self.text
            or "\x00" in self.text
            or "\r" in self.text
            or type(self.boundaries) is not StructuralBoundaries
            or type(self.parser_name) is not str
            or not self.parser_name
            or type(self.parser_version) is not str
            or not self.parser_version
            or not isinstance(self.parser_config, Mapping)
            or any(type(key) is not str for key in self.parser_config)
        ):
            raise ValueError("parsed artifact is invalid")
        self.boundaries._validate_for_text_length(len(self.text))
        object.__setattr__(
            self,
            "parser_config",
            MappingProxyType(dict(sorted(self.parser_config.items()))),
        )

    @property
    def normalized_bytes(self) -> bytes:
        return self.text.encode("utf-8")

    @property
    def checksum_sha256(self) -> str:
        return hashlib.sha256(self.normalized_bytes).hexdigest()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(parser_name={self.parser_name!r}, "
            f"parser_version={self.parser_version!r}, "
            f"character_count={len(self.text)}, boundary_count={len(self.boundaries)})"
        )


class Parser(Protocol):
    """Versioned deterministic parser used by ingestion stages."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def config(self) -> Mapping[str, object]: ...

    def parse(self, source: bytes) -> ParsedArtifact: ...


@dataclass(frozen=True, slots=True)
class _Line:
    start_offset: int
    content: str

    @property
    def content_end_offset(self) -> int:
        return self.start_offset + len(self.content)


@dataclass(frozen=True, slots=True)
class _LineIndex:
    text: str
    content_end_offsets: array[int]

    @classmethod
    def from_text(cls, text: str) -> _LineIndex:
        content_end_offsets = array("I")
        start = 0
        while start < len(text):
            newline = text.find("\n", start)
            if newline < 0:
                content_end_offsets.append(len(text))
                break
            content_end_offsets.append(newline)
            start = newline + 1
        return cls(text, content_end_offsets)

    def __len__(self) -> int:
        return len(self.content_end_offsets)

    def __getitem__(self, index: int) -> _Line:
        content_end_offset = self.content_end_offsets[index]
        start_offset = 0 if index == 0 else self.content_end_offsets[index - 1] + 1
        return _Line(start_offset, self.text[start_offset:content_end_offset])


@dataclass(frozen=True, slots=True)
class _ListMarker:
    indent: int
    content_indent: int


@dataclass(slots=True)
class _NestedListScanState:
    content_indent: int | None = None
    plain_through_index: int = -1

    def is_known_plain(self, content_indent: int, index: int) -> bool:
        if self.content_indent != content_indent:
            self.content_indent = content_indent
            self.plain_through_index = -1
        return index <= self.plain_through_index

    def mark_plain_through(self, content_indent: int, index: int) -> None:
        if self.content_indent != content_indent:
            self.content_indent = content_indent
            self.plain_through_index = -1
        self.plain_through_index = max(self.plain_through_index, index)


def _iter_lines(text: str) -> Iterator[_Line]:
    start = 0
    while start < len(text):
        newline = text.find("\n", start)
        if newline < 0:
            yield _Line(start, text[start:])
            return
        yield _Line(start, text[start:newline])
        start = newline + 1


def _indent_width(prefix: str) -> int:
    width = 0
    for character in prefix:
        width += 4 - (width % 4) if character == "\t" else 1
    return width


def _leading_indent(line: str) -> int:
    return _indent_width(line[: len(line) - len(line.lstrip(" \t"))])


def _list_marker(line: str) -> _ListMarker | None:
    matched = _LIST_ITEM.match(line)
    if matched is None:
        return None
    return _ListMarker(
        _indent_width(matched.group("indent")),
        _indent_width(line[: matched.end()]),
    )


def _top_level_list_marker(line: str) -> _ListMarker | None:
    marker = _list_marker(line)
    if marker is None or marker.indent > 3:
        return None
    return marker


def _remove_indent(line: str, width: int) -> str | None:
    if _leading_indent(line) < width:
        return None
    consumed_width = 0
    index = 0
    while index < len(line) and consumed_width < width:
        character = line[index]
        if character not in {" ", "\t"}:
            return None
        consumed_width += 4 - (consumed_width % 4) if character == "\t" else 1
        index += 1
    return line[index:]


def _plain_boundaries(text: str) -> StructuralBoundaries:
    boundaries = _BoundaryBuilder()
    paragraph_start: int | None = None
    paragraph_end = 0
    for line in _iter_lines(text):
        if not line.content.strip():
            if paragraph_start is not None:
                boundaries.add_fields("paragraph", paragraph_start, paragraph_end, ())
                paragraph_start = None
            continue
        if paragraph_start is None:
            paragraph_start = line.start_offset
        paragraph_end = line.content_end_offset
    if paragraph_start is not None:
        boundaries.add_fields("paragraph", paragraph_start, paragraph_end, ())
    return boundaries.finish()


def _heading(line: str) -> tuple[int, str] | None:
    matched = _ATX_HEADING.fullmatch(line)
    if matched is None:
        return None
    title = (matched.group("title") or "").strip()
    title = re.sub(r"[ \t]+#+[ \t]*$", "", title).rstrip()
    return len(matched.group("marker")), title


def _setext_heading_level(line: str) -> int | None:
    matched = _SETEXT_UNDERLINE.fullmatch(line)
    if matched is None:
        return None
    return 1 if matched.group("marker").startswith("=") else 2


def _fence(line: str) -> tuple[str, int] | None:
    matched = _FENCE_OPEN.fullmatch(line)
    if matched is None:
        return None
    fence = matched.group("fence")
    info = matched.group("info")
    if fence.startswith("`") and "`" in info:
        return None
    return fence[0], len(fence)


def _closes_fence(line: str, character: str, minimum_length: int) -> bool:
    stripped = line.lstrip(" ")
    if len(line) - len(stripped) > 3:
        return False
    marker = stripped.rstrip(" \t")
    return len(marker) >= minimum_length and set(marker) == {character}


def _title_path(stack: list[tuple[int, str]]) -> tuple[str, ...]:
    return tuple(title for _level, title in stack)


def _apply_heading(
    stack: list[tuple[int, str]],
    level: int,
    title: str,
) -> tuple[str, ...]:
    while stack and stack[-1][0] >= level:
        stack.pop()
    if title:
        stack.append((level, title))
    return _title_path(stack)


def _setext_title(
    lines: _LineIndex,
    start_index: int,
    end_index: int,
    content_indent: int,
) -> str:
    def title_parts() -> Iterator[str]:
        for index in range(start_index, end_index):
            relative = _remove_indent(lines[index].content, content_indent)
            if relative is None:
                raise AssertionError("nested Setext title indentation changed during parsing")
            yield relative.strip()

    return " ".join(title_parts())


def _nested_list_structure(
    lines: _LineIndex,
    index: int,
    content_indent: int,
    headings: list[tuple[int, str]],
    scan_state: _NestedListScanState,
) -> tuple[StructuralBoundary, int] | None:
    if scan_state.is_known_plain(content_indent, index):
        return None
    line = lines[index]
    relative = _remove_indent(line.content, content_indent)
    if relative is None:
        return None
    opening_fence = _fence(relative)
    if opening_fence is not None:
        fence_character, fence_length = opening_fence
        end_index = index
        while end_index + 1 < len(lines):
            end_index += 1
            closing = _remove_indent(lines[end_index].content, content_indent)
            if closing is not None and _closes_fence(
                closing,
                fence_character,
                fence_length,
            ):
                break
        return (
            StructuralBoundary(
                "fenced_code_block",
                line.start_offset,
                lines[end_index].content_end_offset,
                _title_path(headings),
            ),
            end_index + 1,
        )
    heading = _heading(relative)
    next_index = index + 1
    last_plain_index = index
    if heading is None and relative.strip():
        while next_index < len(lines):
            candidate = _remove_indent(lines[next_index].content, content_indent)
            if candidate is None or not candidate.strip():
                break
            setext_level = _setext_heading_level(candidate)
            if setext_level is not None:
                title = _setext_title(
                    lines,
                    index,
                    next_index,
                    content_indent,
                )
                heading = (setext_level, title)
                next_index += 1
                break
            if (
                _fence(candidate) is not None
                or _heading(candidate) is not None
                or _top_level_list_marker(candidate) is not None
            ):
                break
            last_plain_index = next_index
            next_index += 1
    if heading is None:
        scan_state.mark_plain_through(content_indent, last_plain_index)
        return None
    level, title = heading
    return (
        StructuralBoundary(
            "heading",
            line.start_offset,
            lines[next_index - 1].content_end_offset,
            _apply_heading(headings, level, title),
            heading_level=level,
        ),
        next_index,
    )


def _consume_list_boundaries(
    lines: _LineIndex,
    start_index: int,
    headings: list[tuple[int, str]],
) -> tuple[StructuralBoundaries, int]:
    boundaries = _BoundaryBuilder()
    scan_state = _NestedListScanState()
    index = start_index
    while index < len(lines):
        line = lines[index]
        marker = _list_marker(line.content)
        if marker is None:
            break
        item_title_path = _title_path(headings)
        item_end = line.content_end_offset
        nested_structure: tuple[StructuralBoundary, int] | None = None
        index += 1
        list_ended = False
        while index < len(lines):
            candidate = lines[index]
            if _list_marker(candidate.content) is not None:
                break
            if not candidate.content.strip():
                continuation_index = index + 1
                while (
                    continuation_index < len(lines)
                    and not lines[continuation_index].content.strip()
                ):
                    continuation_index += 1
                if continuation_index >= len(lines):
                    list_ended = True
                    break
                continuation = lines[continuation_index]
                if _list_marker(continuation.content) is not None:
                    index = continuation_index
                    break
                nested_structure = _nested_list_structure(
                    lines,
                    continuation_index,
                    marker.content_indent,
                    headings,
                    scan_state,
                )
                if nested_structure is not None:
                    index = continuation_index
                    break
                if _leading_indent(continuation.content) >= marker.content_indent:
                    item_end = continuation.content_end_offset
                    index = continuation_index + 1
                    continue
                list_ended = True
                break
            nested_structure = _nested_list_structure(
                lines,
                index,
                marker.content_indent,
                headings,
                scan_state,
            )
            if nested_structure is not None:
                break
            if _leading_indent(candidate.content) >= marker.content_indent:
                item_end = candidate.content_end_offset
                index += 1
                continue
            if (
                _fence(candidate.content) is not None
                or _heading(candidate.content) is not None
                or _setext_heading_level(candidate.content) is not None
            ):
                list_ended = True
                break
            item_end = candidate.content_end_offset
            index += 1
        boundaries.add_fields("list_item", line.start_offset, item_end, item_title_path)
        if nested_structure is not None:
            nested_boundary, index = nested_structure
            boundaries.add(nested_boundary)
            while index < len(lines):
                continuation_index = index
                while (
                    continuation_index < len(lines)
                    and not lines[continuation_index].content.strip()
                ):
                    continuation_index += 1
                if continuation_index >= len(lines):
                    return boundaries.finish(), index
                if _list_marker(lines[continuation_index].content) is not None:
                    index = continuation_index
                    break
                additional_structure = _nested_list_structure(
                    lines,
                    continuation_index,
                    marker.content_indent,
                    headings,
                    scan_state,
                )
                if additional_structure is None:
                    return boundaries.finish(), index
                additional_boundary, index = additional_structure
                boundaries.add(additional_boundary)
            continue
        if list_ended:
            break
    return boundaries.finish(), index


def _markdown_boundaries(text: str) -> StructuralBoundaries:
    lines = _LineIndex.from_text(text)
    boundaries = _BoundaryBuilder()
    headings: list[tuple[int, str]] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.content.strip():
            index += 1
            continue

        opening_fence = _fence(line.content)
        if opening_fence is not None:
            fence_character, fence_length = opening_fence
            end_index = index
            while end_index + 1 < len(lines):
                end_index += 1
                if _closes_fence(
                    lines[end_index].content,
                    fence_character,
                    fence_length,
                ):
                    break
            boundaries.add_fields(
                "fenced_code_block",
                line.start_offset,
                lines[end_index].content_end_offset,
                _title_path(headings),
            )
            index = end_index + 1
            continue

        heading = _heading(line.content)
        if heading is not None:
            level, title = heading
            boundaries.add_fields(
                "heading",
                line.start_offset,
                line.content_end_offset,
                _apply_heading(headings, level, title),
                heading_level=level,
            )
            index += 1
            continue

        if _top_level_list_marker(line.content) is not None:
            list_boundaries, index = _consume_list_boundaries(
                lines,
                index,
                headings,
            )
            boundaries.extend(list_boundaries)
            continue

        paragraph_start = line.start_offset
        paragraph_end = line.content_end_offset
        paragraph_start_index = index
        setext_underline_index: int | None = None
        setext_level: int | None = None
        index += 1
        while index < len(lines):
            candidate = lines[index]
            setext_level = _setext_heading_level(candidate.content)
            if setext_level is not None:
                setext_underline_index = index
                paragraph_end = candidate.content_end_offset
                index += 1
                break
            if (
                not candidate.content.strip()
                or _fence(candidate.content) is not None
                or _heading(candidate.content) is not None
                or _top_level_list_marker(candidate.content) is not None
            ):
                break
            paragraph_end = candidate.content_end_offset
            index += 1
        if setext_level is not None:
            if setext_underline_index is None:
                raise AssertionError("Setext underline index was not retained")
            title = _setext_title(
                lines,
                paragraph_start_index,
                setext_underline_index,
                0,
            )
            boundaries.add_fields(
                "heading",
                paragraph_start,
                paragraph_end,
                _apply_heading(headings, setext_level, title),
                heading_level=setext_level,
            )
            continue
        boundaries.add_fields(
            "paragraph",
            paragraph_start,
            paragraph_end,
            _title_path(headings),
        )
    return boundaries.finish()


class PlainTextParser:
    """Normalize UTF-8 plain text and expose paragraph boundaries."""

    __slots__ = ()

    name: Final = "plain_text_v1"
    version: Final = "1"
    config: Final = _EMPTY_CONFIG

    def parse(self, source: bytes) -> ParsedArtifact:
        validated = validate_document(
            source,
            filename="source.txt",
            content_type="text/plain",
        )
        return ParsedArtifact(
            validated.normalized_text,
            _plain_boundaries(validated.normalized_text),
            self.name,
            self.version,
            self.config,
        )


class MarkdownTextParser:
    """Lexically identify Markdown structure without evaluating its contents."""

    __slots__ = ()

    name: Final = "markdown_text_v1"
    version: Final = "1"
    config: Final = _EMPTY_CONFIG

    def parse(self, source: bytes) -> ParsedArtifact:
        validated = validate_document(
            source,
            filename="source.md",
            content_type="text/markdown",
        )
        return ParsedArtifact(
            validated.normalized_text,
            _markdown_boundaries(validated.normalized_text),
            self.name,
            self.version,
            self.config,
        )


_PLAIN_TEXT_PARSER: Final = PlainTextParser()
_MARKDOWN_TEXT_PARSER: Final = MarkdownTextParser()


def parser_for_extension(extension: str) -> Parser:
    """Select the only allowed parser from an already validated source extension."""

    if type(extension) is not str:
        raise ValueError("document parser extension is invalid")
    canonical = extension.lower()
    if canonical == ".txt":
        return _PLAIN_TEXT_PARSER
    if canonical in {".md", ".markdown"}:
        return _MARKDOWN_TEXT_PARSER
    raise ValueError("document parser extension is invalid")


__all__ = [
    "BoundaryKind",
    "MarkdownTextParser",
    "MAX_TITLE_PATH_UTF8_BYTES",
    "MAX_TITLE_SEGMENT_UTF8_BYTES",
    "ParsedArtifact",
    "ParserError",
    "Parser",
    "PlainTextParser",
    "StructuralBoundaries",
    "StructuralBoundary",
    "StructuralMetadataLimitExceeded",
    "parser_for_extension",
]
