import base64
import binascii
import hashlib
import hmac
import re
import secrets
import traceback
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import SecretStr

from rag_service.api.errors import BusinessError

_MAX_TOKEN_LENGTH = 256
_MIN_PUBLIC_ID_LENGTH = 16
_MAX_PUBLIC_ID_LENGTH = 64
_PUBLIC_ID_BYTES = 16
_SECRET_BYTES = 32
_DIGEST_BYTES = hashlib.sha256().digest_size
_BASE64URL_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
_REDACTED_HMAC_SECRET = SecretStr("<redacted>")
_REDACTED_EXCEPTION_ARGS = ("<redacted>",)
_MAX_RETAINED_EXCEPTION_NODES = 32


class KeyKind(StrEnum):
    ADMIN = "admin"
    AGENT = "agent"


class _MalformedTokenSecretError(Exception):
    pass


_TOKEN_PREFIXES = {
    KeyKind.ADMIN: "rag_adm_",
    KeyKind.AGENT: "rag_agent_",
}


@dataclass(frozen=True, slots=True)
class GeneratedToken:
    token: str = field(repr=False)
    public_id: str
    digest: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class ParsedToken:
    kind: KeyKind
    public_id: str
    secret: str = field(repr=False)


def _invalid_api_key_error() -> BusinessError:
    return BusinessError(401, "INVALID_API_KEY", "Invalid API key")


def _internal_error() -> BusinessError:
    return BusinessError(500, "INTERNAL_ERROR", "Internal server error")


def _sanitize_retained_exception(error: Exception) -> None:
    pending = [error]
    visited: set[int] = set()
    while pending and len(visited) < _MAX_RETAINED_EXCEPTION_NODES:
        current = pending.pop()
        identity = id(current)
        if identity in visited:
            continue
        visited.add(identity)

        for nested_error in (current.__cause__, current.__context__):
            if isinstance(nested_error, Exception):
                pending.append(nested_error)

        if current.__traceback__ is not None:
            traceback.clear_frames(current.__traceback__)
        current.args = _REDACTED_EXCEPTION_ARGS
        current.__traceback__ = None
        current.__cause__ = None
        current.__context__ = None


def _encode_component(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _decode_canonical_component(value: str) -> bytes:
    if not value or _BASE64URL_PATTERN.fullmatch(value) is None:
        raise ValueError
    padding = "=" * (-len(value) % 4)
    decoded = base64.b64decode(value + padding, altchars=b"-_", validate=True)
    if _encode_component(decoded) != value:
        raise ValueError
    return decoded


def _parse_token(raw: str, expected_kind: KeyKind) -> ParsedToken:
    if type(raw) is not str or not raw or len(raw) > _MAX_TOKEN_LENGTH:
        raise ValueError
    prefix = _TOKEN_PREFIXES[expected_kind]
    if not raw.startswith(prefix):
        raise ValueError
    components = raw[len(prefix) :].split(".")
    if len(components) != 2:
        raise ValueError
    public_id, secret = components
    if not _MIN_PUBLIC_ID_LENGTH <= len(public_id) <= _MAX_PUBLIC_ID_LENGTH:
        raise ValueError
    _decode_canonical_component(public_id)
    if len(_decode_canonical_component(secret)) != _SECRET_BYTES:
        raise ValueError
    return ParsedToken(kind=expected_kind, public_id=public_id, secret=secret)


def parse_token(raw: str, expected_kind: KeyKind) -> ParsedToken:
    if type(expected_kind) is not KeyKind:
        raw = "<redacted>"
        expected_kind = KeyKind.ADMIN
        raise _internal_error() from None
    try:
        return _parse_token(raw, expected_kind)
    except Exception as retained_error:
        _sanitize_retained_exception(retained_error)
        raw = "<redacted>"
    raise _invalid_api_key_error() from None


def _require_canonical_secret(secret: str) -> None:
    if type(secret) is not str:
        raise _MalformedTokenSecretError
    try:
        decoded = _decode_canonical_component(secret)
    except (TypeError, ValueError, binascii.Error):
        raise _MalformedTokenSecretError from None
    if len(decoded) != _SECRET_BYTES:
        raise _MalformedTokenSecretError


def _digest_secret(secret: str, hmac_secret: SecretStr) -> bytes:
    _require_canonical_secret(secret)
    key = hmac_secret.get_secret_value().encode("utf-8")
    return hmac.new(key, secret.encode("ascii"), hashlib.sha256).digest()


def digest_secret(secret: str, hmac_secret: SecretStr) -> bytes:
    try:
        return _digest_secret(secret, hmac_secret)
    except _MalformedTokenSecretError as retained_error:
        _sanitize_retained_exception(retained_error)
        surfaced_error = _invalid_api_key_error()
    except Exception as retained_error:
        _sanitize_retained_exception(retained_error)
        surfaced_error = _internal_error()
    secret = "<redacted>"
    hmac_secret = _REDACTED_HMAC_SECRET
    raise surfaced_error from None


def verify_secret(secret: str, expected_digest: bytes, hmac_secret: SecretStr) -> bool:
    if type(expected_digest) is not bytes or len(expected_digest) != _DIGEST_BYTES:
        return False
    calculated_digest = b""
    matches = False
    try:
        calculated_digest = _digest_secret(secret, hmac_secret)
        matches = hmac.compare_digest(calculated_digest, expected_digest)
    except Exception as retained_error:
        _sanitize_retained_exception(retained_error)
        secret = "<redacted>"
        hmac_secret = _REDACTED_HMAC_SECRET
        expected_digest = b""
        calculated_digest = b""
        matches = False
    return matches is True


def _generate_token(kind: KeyKind, hmac_secret: SecretStr) -> GeneratedToken:
    public_id = _encode_component(secrets.token_bytes(_PUBLIC_ID_BYTES))
    secret = _encode_component(secrets.token_bytes(_SECRET_BYTES))
    token = f"{_TOKEN_PREFIXES[kind]}{public_id}.{secret}"
    if len(token) > _MAX_TOKEN_LENGTH:
        raise RuntimeError("generated API key exceeds supported length")
    return GeneratedToken(
        token=token,
        public_id=public_id,
        digest=_digest_secret(secret, hmac_secret),
    )


def generate_token(kind: KeyKind, hmac_secret: SecretStr) -> GeneratedToken:
    if type(kind) is not KeyKind:
        kind = KeyKind.ADMIN
        hmac_secret = _REDACTED_HMAC_SECRET
        raise _internal_error() from None
    try:
        return _generate_token(kind, hmac_secret)
    except Exception as retained_error:
        _sanitize_retained_exception(retained_error)
        kind = KeyKind.ADMIN
        hmac_secret = _REDACTED_HMAC_SECRET
    raise _internal_error() from None
