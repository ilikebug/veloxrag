"""Strict request and response contracts for single-KB Dense search."""

from __future__ import annotations

import math
import re
from typing import Annotated
from uuid import UUID

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from rag_service.api.errors import BusinessError
from rag_service.api.validation import JSONValue, validate_bounded_json

_FILTER_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")
_OPERATORS = frozenset({"eq", "in", "gte", "lte"})
_MAX_DOCUMENT_FILTERS = 200
_MAX_METADATA_FILTERS = 64
_MAX_IN_VALUES = 100


class _Schema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        hide_input_in_errors=True,
        allow_inf_nan=False,
    )


def _filter_scalar(value: object) -> bool:
    if type(value) is str:
        return len(value) <= 4096 and "\x00" not in value
    if type(value) is int:
        return -(2**63) <= value <= 2**63 - 1
    if type(value) is bool:
        return True
    if type(value) is float:
        return math.isfinite(value)
    return False


class SearchFilters(_Schema):
    document_ids: tuple[UUID, ...] = Field(default=(), max_length=_MAX_DOCUMENT_FILTERS)
    metadata: dict[str, object] = Field(default_factory=dict)

    @field_validator("document_ids", mode="before")
    @classmethod
    def validate_document_ids_bound(cls, value: object) -> object:
        if isinstance(value, (list, tuple)):
            if len(value) > _MAX_DOCUMENT_FILTERS:
                raise ValueError("too many document filters")
            canonical: list[UUID] = []
            for member in value:
                if type(member) is UUID:
                    canonical.append(member)
                    continue
                if type(member) is not str:
                    raise ValueError("document filter is invalid")
                try:
                    parsed = UUID(member)
                except ValueError:
                    raise ValueError("document filter is invalid") from None
                if member != str(parsed):
                    raise ValueError("document filter is invalid")
                canonical.append(parsed)
            return tuple(canonical)
        return value

    @field_validator("document_ids")
    @classmethod
    def validate_unique_document_ids(cls, value: tuple[UUID, ...]) -> tuple[UUID, ...]:
        if len(set(value)) != len(value):
            raise ValueError("document filters must be unique")
        return value

    @field_validator("metadata", mode="before")
    @classmethod
    def validate_metadata_filters(cls, value: object) -> object:
        if type(value) is not dict:
            return value
        if len(value) > _MAX_METADATA_FILTERS:
            raise ValueError("too many metadata filters")
        canonical: dict[str, object] = {}
        for name, expression in value.items():
            if type(name) is not str or _FILTER_NAME.fullmatch(name) is None:
                raise ValueError("metadata filter name is invalid")
            if type(expression) is dict:
                if len(expression) != 1:
                    raise ValueError("metadata filter expression is invalid")
                operator, operand = next(iter(expression.items()))
                if type(operator) is not str or operator not in _OPERATORS:
                    raise ValueError("metadata filter operator is invalid")
                if operator == "in":
                    if (
                        not isinstance(operand, (list, tuple))
                        or not 1 <= len(operand) <= _MAX_IN_VALUES
                        or any(not _filter_scalar(item) for item in operand)
                    ):
                        raise ValueError("metadata filter values are invalid")
                    canonical[name] = {operator: tuple(operand)}
                else:
                    if not _filter_scalar(operand):
                        raise ValueError("metadata filter value is invalid")
                    canonical[name] = {operator: operand}
                continue
            if not _filter_scalar(expression):
                raise ValueError("metadata filter value is invalid")
            canonical[name] = expression
        return canonical

    @model_validator(mode="after")
    def validate_aggregate_size(self) -> SearchFilters:
        try:
            validate_bounded_json(self.model_dump(mode="json"))
        except BusinessError:
            raise ValueError("search filters exceed the safe JSON bound") from None
        return self


class SearchRequest(_Schema):
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)]
    top_k: int = Field(default=10, strict=True, ge=1, le=50)
    filters: SearchFilters | None = None
    # Per request rather than per knowledge base: reranking costs a second
    # provider round trip, and only the caller knows whether this particular
    # query is worth it. Which model does the reranking is an operator's choice,
    # so it lives on the knowledge base instead.
    rerank: bool = Field(default=False, strict=True)


class SearchSource(_Schema):
    filename: str = Field(min_length=1, max_length=255)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)


class SearchResult(_Schema):
    text: str = Field(min_length=1)
    score: float
    document_id: UUID
    version_id: UUID
    chunk_index: int = Field(ge=0)
    title: str | None = None
    title_path: tuple[str, ...] = Field(max_length=64)
    source: SearchSource
    metadata: dict[str, JSONValue]


class SearchIndex(_Schema):
    generation_id: UUID
    embedding_profile_id: UUID


class SearchResponse(_Schema):
    results: tuple[SearchResult, ...] = Field(max_length=50)
    index: SearchIndex


__all__ = [
    "SearchFilters",
    "SearchIndex",
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    "SearchSource",
]
