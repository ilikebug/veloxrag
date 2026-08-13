"""Deterministic identities and canonical JSON for index artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from uuid import UUID, uuid5

_POINT_NAMESPACE = UUID("a78b5756-9fef-5bd8-9537-e0c872ea90ac")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def collection_name(knowledge_base_id: UUID, generation_id: UUID) -> str:
    if type(knowledge_base_id) is not UUID or type(generation_id) is not UUID:
        raise ValueError("collection identity is invalid")
    return f"rag_kb_{knowledge_base_id.hex}_g_{generation_id.hex}"


def point_id(version_id: UUID, chunk_index: int, chunk_hash: str) -> UUID:
    if type(version_id) is not UUID:
        raise ValueError("version identity is invalid")
    if type(chunk_index) is not int or chunk_index < 0:
        raise ValueError("chunk index is invalid")
    if type(chunk_hash) is not str or _SHA256_PATTERN.fullmatch(chunk_hash) is None:
        raise ValueError("chunk hash is invalid")
    return uuid5(_POINT_NAMESPACE, f"{version_id}:{chunk_index}:{chunk_hash}")


def _validated_json(value: object) -> object:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("value is not canonical JSON")
        return value
    if type(value) is list:
        return [_validated_json(member) for member in value]
    if type(value) is dict:
        document = value
        if any(type(key) is not str for key in document):
            raise ValueError("value is not canonical JSON")
        return {key: _validated_json(document[key]) for key in sorted(document)}
    if isinstance(value, (Mapping, Sequence)):
        raise ValueError("value is not canonical JSON")
    raise ValueError("value is not canonical JSON")


def canonical_json_bytes(value: object) -> bytes:
    try:
        validated = _validated_json(value)
        return json.dumps(
            validated,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        raise ValueError("value is not canonical JSON") from None


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


__all__ = [
    "canonical_json_bytes",
    "canonical_sha256",
    "collection_name",
    "point_id",
]
