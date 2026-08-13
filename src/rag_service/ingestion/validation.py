"""Bounded validation for UTF-8 text and Markdown uploads."""

from __future__ import annotations

import codecs
import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Final

MAX_DOCUMENT_BYTES: Final = 50 * 1024 * 1024

_SUPPORTED_EXTENSIONS = frozenset({".txt", ".md", ".markdown"})
_TEXT_MIME_TYPES = frozenset({"text/plain", "application/octet-stream"})
_MARKDOWN_MIME_TYPES = frozenset(
    {
        "text/plain",
        "text/markdown",
        "text/x-markdown",
        "application/markdown",
        "application/octet-stream",
    }
)
_MIME_TOKEN = re.compile(
    r"^[a-z0-9!#$&^_.+-]+/[a-z0-9!#$&^_.+-]+$",
    flags=re.ASCII,
)
_ERROR_MESSAGES: Final = {
    "FILE_TOO_LARGE": "Document exceeds the upload limit",
    "UNSUPPORTED_DOCUMENT_TYPE": "Unsupported document type",
    "INVALID_TEXT_ENCODING": "Document is not valid UTF-8",
    "BINARY_CONTENT_REJECTED": "Document content is not supported text",
    "EMPTY_DOCUMENT": "Document contains no text",
    "DUPLICATE_DOCUMENT": "Document content already exists",
}


class DocumentValidationError(Exception):
    """A permanent public upload rejection with a stable, sanitized code."""

    __slots__ = ("code", "retryable")

    def __init__(self, code: str) -> None:
        if type(code) is not str or code not in _ERROR_MESSAGES:
            raise ValueError("document validation error code is invalid")
        self.code = code
        self.retryable = False
        super().__init__(_ERROR_MESSAGES[code])

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.code!r})"


@dataclass(frozen=True, slots=True)
class DocumentMetadata:
    display_filename: str
    extension: str
    content_type: str


@dataclass(frozen=True, slots=True, repr=False)
class ValidatedDocument:
    display_filename: str
    extension: str
    content_type: str
    source_size: int
    source_checksum_sha256: str
    parsed_checksum_sha256: str
    normalized_text: str
    normalized_bytes: bytes


@dataclass(frozen=True, slots=True)
class TextValidationSummary:
    display_filename: str
    extension: str
    content_type: str
    source_size: int
    source_checksum_sha256: str
    has_text: bool


def _document_error(code: str) -> DocumentValidationError:
    return DocumentValidationError(code)


def _canonical_filename(filename: str) -> tuple[str, str]:
    if type(filename) is not str:
        raise _document_error("UNSUPPORTED_DOCUMENT_TYPE")
    display_filename = unicodedata.normalize("NFC", filename.strip())
    if (
        not display_filename
        or display_filename in {".", ".."}
        or "/" in display_filename
        or "\\" in display_filename
        or any(unicodedata.category(character).startswith("C") for character in display_filename)
        or len(display_filename.encode("utf-8")) > 255
    ):
        raise _document_error("UNSUPPORTED_DOCUMENT_TYPE")
    stem, separator, suffix = display_filename.rpartition(".")
    extension = f".{suffix.lower()}" if separator else ""
    if not stem or extension not in _SUPPORTED_EXTENSIONS:
        raise _document_error("UNSUPPORTED_DOCUMENT_TYPE")
    return display_filename, extension


def canonical_source_extension(filename: str) -> str:
    """Return the supported lower-case extension without retaining the filename."""

    _display_filename, extension = _canonical_filename(filename)
    return extension


def _canonical_content_type(content_type: str) -> str:
    if type(content_type) is not str:
        raise _document_error("UNSUPPORTED_DOCUMENT_TYPE")
    token = content_type.split(";", 1)[0].strip().lower()
    if _MIME_TOKEN.fullmatch(token) is None:
        raise _document_error("UNSUPPORTED_DOCUMENT_TYPE")
    return token


def validate_document_metadata(*, filename: str, content_type: str) -> DocumentMetadata:
    display_filename, extension = _canonical_filename(filename)
    canonical_content_type = _canonical_content_type(content_type)
    allowed_mime_types = _TEXT_MIME_TYPES if extension == ".txt" else _MARKDOWN_MIME_TYPES
    if canonical_content_type not in allowed_mime_types:
        raise _document_error("UNSUPPORTED_DOCUMENT_TYPE")
    return DocumentMetadata(display_filename, extension, canonical_content_type)


def _is_disallowed_control(character: str) -> bool:
    return unicodedata.category(character) == "Cc" and character not in {
        "\t",
        "\n",
        "\r",
        "\f",
    }


class IncrementalTextValidator:
    """Incrementally validate upload bytes without retaining document content."""

    __slots__ = (
        "_control_count",
        "_decoder",
        "_digest",
        "_has_text",
        "_metadata",
        "_max_bytes",
        "_saw_codepoint",
        "_text_character_count",
        "_total",
        "_finished",
    )

    def __init__(
        self,
        *,
        filename: str,
        content_type: str,
        max_bytes: int = MAX_DOCUMENT_BYTES,
    ) -> None:
        if type(max_bytes) is not int or max_bytes <= 0:
            raise ValueError("document byte limit is invalid")
        self._metadata = validate_document_metadata(
            filename=filename,
            content_type=content_type,
        )
        self._max_bytes = max_bytes
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
        self._digest = hashlib.sha256()
        self._total = 0
        self._control_count = 0
        self._text_character_count = 0
        self._has_text = False
        self._saw_codepoint = False
        self._finished = False

    def feed(self, chunk: bytes) -> None:
        if self._finished or type(chunk) is not bytes:
            raise ValueError("document validator state is invalid")
        if not chunk:
            return
        next_total = self._total + len(chunk)
        if next_total > self._max_bytes:
            raise _document_error("FILE_TOO_LARGE")
        self._total = next_total
        self._digest.update(chunk)
        try:
            decoded = self._decoder.decode(chunk, final=False)
        except UnicodeDecodeError:
            raise _document_error("INVALID_TEXT_ENCODING") from None
        self._inspect(decoded)

    def _inspect(self, decoded: str) -> None:
        if not decoded:
            return
        if not self._saw_codepoint:
            self._saw_codepoint = True
            if decoded.startswith("\ufeff"):
                decoded = decoded[1:]
        if "\x00" in decoded:
            raise _document_error("BINARY_CONTENT_REJECTED")
        self._text_character_count += len(decoded)
        self._control_count += sum(_is_disallowed_control(character) for character in decoded)
        self._has_text = self._has_text or any(not character.isspace() for character in decoded)

    def finish(self) -> TextValidationSummary:
        if self._finished:
            raise ValueError("document validator is already finished")
        self._finished = True
        try:
            trailing = self._decoder.decode(b"", final=True)
        except UnicodeDecodeError:
            raise _document_error("INVALID_TEXT_ENCODING") from None
        self._inspect(trailing)
        if (
            self._control_count
            and (self._control_count / max(1, self._text_character_count)) >= 0.05
        ):
            raise _document_error("BINARY_CONTENT_REJECTED")
        if not self._has_text:
            raise _document_error("EMPTY_DOCUMENT")
        return TextValidationSummary(
            self._metadata.display_filename,
            self._metadata.extension,
            self._metadata.content_type,
            self._total,
            self._digest.hexdigest(),
            True,
        )


def validate_document(
    content: bytes,
    *,
    filename: str,
    content_type: str,
    max_bytes: int = MAX_DOCUMENT_BYTES,
) -> ValidatedDocument:
    if type(content) is not bytes:
        raise ValueError("document content must be bytes")
    validator = IncrementalTextValidator(
        filename=filename,
        content_type=content_type,
        max_bytes=max_bytes,
    )
    validator.feed(content)
    summary = validator.finish()
    try:
        decoded = content.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise _document_error("INVALID_TEXT_ENCODING") from None
    if decoded.startswith("\ufeff"):
        decoded = decoded[1:]
    normalized_text = decoded.replace("\r\n", "\n").replace("\r", "\n")
    normalized_bytes = normalized_text.encode("utf-8")
    return ValidatedDocument(
        summary.display_filename,
        summary.extension,
        summary.content_type,
        summary.source_size,
        summary.source_checksum_sha256,
        hashlib.sha256(normalized_bytes).hexdigest(),
        normalized_text,
        normalized_bytes,
    )


__all__ = [
    "MAX_DOCUMENT_BYTES",
    "DocumentMetadata",
    "DocumentValidationError",
    "IncrementalTextValidator",
    "TextValidationSummary",
    "ValidatedDocument",
    "canonical_source_extension",
    "validate_document",
    "validate_document_metadata",
]
