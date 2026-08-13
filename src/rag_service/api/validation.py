import json
import math
import re
from typing import cast

from rag_service.api.constants import MAX_HEADER_VALUE_LENGTH
from rag_service.api.errors import BusinessError

type JSONValue = None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]

_MAX_JSON_OBJECT_KEYS = 64
_MAX_JSON_DEPTH = 4
_MAX_JSON_SIZE = 32 * 1024
# In the densest valid shape, ``[0,0,...]``, each child consumes two bytes.
_MAX_JSON_NODES = (_MAX_JSON_SIZE + 1) // 2
_IDEMPOTENCY_KEY_PATTERN = re.compile(r"[!-~]+")
_JSON_ENCODER = json.JSONEncoder(
    ensure_ascii=False,
    separators=(",", ":"),
    sort_keys=True,
    allow_nan=False,
)


def _validation_error(message: str) -> BusinessError:
    return BusinessError(422, "VALIDATION_ERROR", message)


def _validate_idempotency_key(value: str, max_length: int) -> str:
    effective_max = min(max_length, MAX_HEADER_VALUE_LENGTH)
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= effective_max
        or _IDEMPOTENCY_KEY_PATTERN.fullmatch(value) is None
    ):
        raise ValueError
    return value


def validate_idempotency_key(value: str, max_length: int = MAX_HEADER_VALUE_LENGTH) -> str:
    try:
        return _validate_idempotency_key(value, max_length)
    except (TypeError, ValueError):
        value = "<redacted>"
    raise _validation_error("Invalid idempotency key")


def _validate_bounded_json(value: object) -> JSONValue:
    object_keys = 0
    node_count = 0
    string_bytes = 0

    def count_string_bytes(text: str) -> None:
        nonlocal string_bytes
        remaining = _MAX_JSON_SIZE - string_bytes
        if len(text) > remaining:
            raise ValueError
        string_bytes += len(text.encode("utf-8"))
        if string_bytes > _MAX_JSON_SIZE:
            raise ValueError

    def walk(node: object, depth: int) -> None:
        nonlocal node_count, object_keys
        node_count += 1
        if node_count > _MAX_JSON_NODES:
            raise ValueError
        if depth > _MAX_JSON_DEPTH:
            raise ValueError
        if node is None or isinstance(node, bool):
            return
        if isinstance(node, str):
            count_string_bytes(node)
            return
        if isinstance(node, int) and not isinstance(node, bool):
            return
        if isinstance(node, float):
            if not math.isfinite(node):
                raise ValueError
            return
        if isinstance(node, list):
            if len(node) > _MAX_JSON_NODES - node_count:
                raise ValueError
            for item in node:
                walk(item, depth + 1)
            return
        if isinstance(node, dict):
            item_count = len(node)
            if item_count > _MAX_JSON_OBJECT_KEYS - object_keys:
                raise ValueError
            if item_count > _MAX_JSON_NODES - node_count:
                raise ValueError
            object_keys += item_count
            for key, item in node.items():
                if not isinstance(key, str):
                    raise ValueError
                count_string_bytes(key)
                walk(item, depth + 1)
            return
        raise ValueError

    walk(value, 0)
    encoded_size = 0
    for chunk in _JSON_ENCODER.iterencode(value):
        encoded_size += len(chunk.encode("utf-8"))
        if encoded_size > _MAX_JSON_SIZE:
            raise ValueError
    return cast(JSONValue, value)


def validate_bounded_json(value: object) -> JSONValue:
    try:
        return _validate_bounded_json(value)
    except (TypeError, ValueError):
        value = "<redacted>"
    raise _validation_error("Invalid JSON payload")
