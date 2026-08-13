"""Validated public upload commands and canonical request identities."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from rag_service.api.errors import BusinessError
from rag_service.api.validation import JSONValue, validate_bounded_json
from rag_service.indexing.identities import canonical_json_bytes


def _validation_error() -> BusinessError:
    return BusinessError(422, "VALIDATION_ERROR", "Invalid document upload")


def _configuration_conflict() -> BusinessError:
    return BusinessError(
        409,
        "GENERATION_CONFIGURATION_CONFLICT",
        "Index generation configuration conflict",
    )


def _safe_text(value: str, *, maximum: int) -> str:
    if type(value) is not str:
        raise _validation_error()
    normalized = unicodedata.normalize("NFC", value).strip()
    if (
        not normalized
        or len(normalized) > maximum
        or any(unicodedata.category(character).startswith("C") for character in normalized)
    ):
        raise _validation_error()
    return normalized


def canonical_upload_filename(value: str) -> str:
    if type(value) is not str:
        raise _validation_error()
    basename = value.replace("\\", "/").rsplit("/", 1)[-1]
    basename = "".join(
        character
        for character in unicodedata.normalize("NFC", basename)
        if not unicodedata.category(character).startswith("C")
    ).strip()
    if not basename or basename in {".", ".."} or len(basename.encode("utf-8")) > 255:
        raise _validation_error()
    return basename


@dataclass(frozen=True, slots=True)
class UploadForm:
    display_name: str | None
    metadata: dict[str, JSONValue]
    tags: tuple[str, ...]

    @classmethod
    def from_multipart(
        cls,
        *,
        display_name: str | None,
        metadata: str | None,
        tags: str | None,
    ) -> UploadForm:
        try:
            parsed_metadata: object = {} if metadata is None else json.loads(metadata)
            parsed_tags: object = [] if tags is None else json.loads(tags)
        except (TypeError, ValueError, json.JSONDecodeError):
            raise _validation_error() from None
        if not isinstance(parsed_metadata, dict) or not isinstance(parsed_tags, list):
            raise _validation_error()
        validated = validate_bounded_json(parsed_metadata)
        if not isinstance(validated, dict) or len(parsed_tags) > 64:
            raise _validation_error()
        normalized_tags = tuple(sorted(_safe_text(tag, maximum=64) for tag in parsed_tags))
        if len(set(normalized_tags)) != len(normalized_tags):
            raise _validation_error()
        return cls(
            None if display_name is None else _safe_text(display_name, maximum=255),
            validated,
            normalized_tags,
        )


@dataclass(frozen=True, slots=True)
class UploadRequestFingerprintInput:
    actor_key_id: UUID
    knowledge_base_id: UUID
    source_checksum_sha256: str
    filename: str
    extension: str
    declared_mime_type: str
    detected_mime_type: str
    parser_name: str
    display_name: str
    tags: tuple[str, ...]
    metadata: dict[str, JSONValue]


def upload_request_fingerprint(command: UploadRequestFingerprintInput) -> bytes:
    document = {
        "route_version": "document-upload-v1",
        "actor_key_id": str(command.actor_key_id),
        "knowledge_base_id": str(command.knowledge_base_id),
        "source_checksum_sha256": command.source_checksum_sha256,
        "filename": command.filename,
        "extension": command.extension,
        "declared_mime_type": command.declared_mime_type,
        "detected_mime_type": command.detected_mime_type,
        "parser_name": command.parser_name,
        "display_name": command.display_name,
        "tags": list(command.tags),
        "metadata": command.metadata,
    }
    return hashlib.sha256(canonical_json_bytes(document)).digest()


def _path_value(metadata: dict[str, JSONValue], source_path: str) -> object:
    current: object = metadata
    for segment in source_path.split("."):
        if not isinstance(current, dict) or segment not in current:
            return _MISSING
        current = current[segment]
    return current


_MISSING = object()
_RFC3339_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$"
)


def is_rfc3339_datetime(value: object) -> bool:
    if type(value) is not str or _RFC3339_PATTERN.fullmatch(value) is None:
        return False
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00" if value.endswith("Z") else value)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def validate_metadata_against_filter_snapshot(
    metadata: dict[str, JSONValue], preflight: Any
) -> None:
    if preflight.applied_filter_schema_revision != preflight.current_filter_schema_revision:
        raise _configuration_conflict()
    snapshot = preflight.filter_schema_snapshot
    fields = snapshot.get("fields") if isinstance(snapshot, dict) else None
    if not isinstance(fields, list):
        raise _configuration_conflict()
    for field in fields:
        if not isinstance(field, dict):
            raise _configuration_conflict()
        source_path = field.get("source_path")
        field_type = field.get("type")
        if not isinstance(source_path, str) or not isinstance(field_type, str):
            raise _configuration_conflict()
        value = _path_value(metadata, source_path)
        if value is _MISSING:
            continue
        if field_type == "keyword":
            valid = type(value) is str and len(value) <= 4096 and "\x00" not in value
        elif field_type == "integer":
            valid = type(value) is int and -(2**63) <= value <= 2**63 - 1
        elif field_type == "float":
            try:
                valid = (
                    type(value) in {int, float}
                    and not isinstance(value, bool)
                    and math.isfinite(float(cast(int | float, value)))
                )
            except OverflowError:
                valid = False
        elif field_type == "boolean":
            valid = type(value) is bool
        elif field_type == "datetime":
            valid = is_rfc3339_datetime(value)
        else:
            valid = False
        if not valid:
            raise _validation_error()


class UploadAccepted(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    document_id: UUID
    version_id: UUID
    job_id: UUID
    status: Literal["queued"] = "queued"


__all__ = [
    "UploadAccepted",
    "UploadForm",
    "UploadRequestFingerprintInput",
    "canonical_upload_filename",
    "is_rfc3339_datetime",
    "upload_request_fingerprint",
    "validate_metadata_against_filter_snapshot",
]
