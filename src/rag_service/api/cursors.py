import base64
import binascii
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

from rag_service.api.errors import BusinessError

_MAX_CURSOR_LENGTH = 512
_BASE64URL_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
_TIMESTAMP_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z")


def _invalid_cursor_error() -> BusinessError:
    return BusinessError(422, "VALIDATION_ERROR", "Invalid cursor")


@dataclass(frozen=True, slots=True)
class CursorPosition:
    created_at: datetime
    id: UUID


def _canonical_timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
        raise ValueError("cursor timestamps must be UTC")
    return value.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _cursor_payload(position: CursorPosition) -> bytes:
    return json.dumps(
        {"v": 1, "created_at": _canonical_timestamp(position.created_at), "id": str(position.id)},
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def encode_cursor(position: CursorPosition) -> str:
    return base64.urlsafe_b64encode(_cursor_payload(position)).decode("ascii").rstrip("=")


def _decode_cursor(value: str) -> CursorPosition:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_CURSOR_LENGTH
        or _BASE64URL_PATTERN.fullmatch(value) is None
    ):
        raise ValueError
    padding = "=" * (-len(value) % 4)
    raw = base64.b64decode(value + padding, altchars=b"-_", validate=True)
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"v", "created_at", "id"}:
        raise ValueError
    if type(payload["v"]) is not int or payload["v"] != 1:
        raise ValueError
    timestamp = payload["created_at"]
    identifier = payload["id"]
    if not isinstance(timestamp, str) or _TIMESTAMP_PATTERN.fullmatch(timestamp) is None:
        raise ValueError
    created_at = datetime.fromisoformat(timestamp[:-1] + "+00:00")
    if created_at.tzinfo is None or created_at.utcoffset() != UTC.utcoffset(created_at):
        raise ValueError
    if not isinstance(identifier, str):
        raise ValueError
    cursor_id = UUID(identifier)
    position = CursorPosition(created_at=created_at, id=cursor_id)
    if str(cursor_id) != identifier or encode_cursor(position) != value:
        raise ValueError
    return position


def decode_cursor(value: str) -> CursorPosition:
    try:
        return _decode_cursor(value)
    except (UnicodeDecodeError, ValueError, TypeError, binascii.Error, json.JSONDecodeError):
        value = "<redacted>"
    raise _invalid_cursor_error()
