import base64
import hmac
import re
import secrets
from enum import StrEnum
from unittest.mock import Mock

import pytest
from pydantic import SecretStr

from rag_service.api.errors import BusinessError
from rag_service.auth import codec as codec_module
from rag_service.auth.codec import (
    KeyKind,
    digest_secret,
    generate_token,
    parse_token,
    verify_secret,
)

ADMIN_HMAC_SECRET = SecretStr("admin-hmac-secret-with-at-least-thirty-two-bytes")
AGENT_HMAC_SECRET = SecretStr("agent-hmac-secret-with-at-least-thirty-two-bytes")
BASE64URL_COMPONENT = re.compile(r"[A-Za-z0-9_-]+")
PUBLIC_ID = base64.urlsafe_b64encode(b"public-id-source").decode("ascii").rstrip("=")
TOKEN_SECRET = base64.urlsafe_b64encode(b"s" * 32).decode("ascii").rstrip("=")
RETAINED_HMAC_VALUE = "sensitive-retained-hmac-credential"


class ExplodingHmacSecret(SecretStr):
    def get_secret_value(self) -> str:
        raise RuntimeError("sensitive HMAC accessor failure")


class DigestBytesSubclass(bytes):
    pass


class ForeignKeyKind(StrEnum):
    ADMIN = "admin"


class MockedHmacSecret(SecretStr):
    pass


def _decode_component(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _assert_codec_traceback_is_redacted(
    error: BusinessError,
    *secret_markers: str,
) -> None:
    assert error.__context__ is None
    assert error.__cause__ is None
    codec_frames = 0
    traceback = error.__traceback__
    while traceback is not None:
        module_name = traceback.tb_frame.f_globals.get("__name__", "")
        if module_name == "rag_service.auth.codec":
            codec_frames += 1
            frame_locals = traceback.tb_frame.f_locals
            locals_repr = repr(frame_locals)
            for marker in secret_markers:
                assert marker not in locals_repr
            for value in frame_locals.values():
                if isinstance(value, SecretStr):
                    recovered = value.get_secret_value()
                    for marker in secret_markers:
                        assert recovered != marker
        traceback = traceback.tb_next
    assert codec_frames <= 1


def _retained_exception_graph(marker: str) -> tuple[RuntimeError, tuple[RuntimeError, ...]]:
    cause = RuntimeError(f"{marker}-cause")
    context = RuntimeError(f"{marker}-context")
    error = RuntimeError(f"{marker}-error")
    error.__cause__ = cause
    error.__context__ = context
    cause.__context__ = error
    return error, (error, cause, context)


def _assert_retained_exceptions_are_sanitized(
    retained_errors: tuple[RuntimeError, ...],
    *secret_markers: str,
) -> None:
    for retained_error in retained_errors:
        assert retained_error.args == ("<redacted>",)
        assert retained_error.__cause__ is None
        assert retained_error.__context__ is None
        traceback = retained_error.__traceback__
        while traceback is not None:
            locals_repr = repr(traceback.tb_frame.f_locals)
            for marker in secret_markers:
                assert marker not in locals_repr
            traceback = traceback.tb_next
        assert retained_error.__traceback__ is None
        for marker in secret_markers:
            assert marker not in str(retained_error)


def _install_retained_hmac_failure(
    failure_source: str,
    retained_error: RuntimeError,
    monkeypatch: pytest.MonkeyPatch,
) -> SecretStr:
    hmac_secret: SecretStr = SecretStr(RETAINED_HMAC_VALUE)
    if failure_source == "accessor":
        monkeypatch.setattr(
            MockedHmacSecret,
            "get_secret_value",
            Mock(side_effect=retained_error),
        )
        return MockedHmacSecret(RETAINED_HMAC_VALUE)
    if failure_source == "hmac-new":
        monkeypatch.setattr(hmac, "new", Mock(side_effect=retained_error))
        return hmac_secret
    if failure_source == "digest":
        digest = Mock(side_effect=retained_error)
        hmac_result = Mock()
        hmac_result.digest = digest
        monkeypatch.setattr(hmac, "new", Mock(return_value=hmac_result))
        return hmac_secret
    raise AssertionError(f"unknown failure source: {failure_source}")


@pytest.mark.parametrize(
    ("kind", "prefix", "hmac_secret"),
    [
        (KeyKind.ADMIN, "rag_adm_", ADMIN_HMAC_SECRET),
        (KeyKind.AGENT, "rag_agent_", AGENT_HMAC_SECRET),
    ],
)
def test_generate_token_uses_exact_prefix_and_canonical_url_safe_components(
    kind: KeyKind,
    prefix: str,
    hmac_secret: SecretStr,
) -> None:
    generated = generate_token(kind, hmac_secret)
    public_id, secret = generated.token.removeprefix(prefix).split(".")

    assert generated.token.startswith(prefix)
    assert public_id == generated.public_id
    assert 16 <= len(public_id) <= 64
    assert BASE64URL_COMPONENT.fullmatch(public_id) is not None
    assert BASE64URL_COMPONENT.fullmatch(secret) is not None
    assert "=" not in generated.token
    assert len(_decode_component(secret)) == 32
    assert len(generated.token) <= 256
    assert len(generated.digest) == 32
    assert generated.digest == digest_secret(secret, hmac_secret)
    assert parse_token(generated.token, kind).public_id == public_id


def test_generated_and_parsed_reprs_hide_token_secret_and_digest() -> None:
    generated = generate_token(KeyKind.ADMIN, ADMIN_HMAC_SECRET)
    parsed = parse_token(generated.token, KeyKind.ADMIN)

    assert generated.token not in repr(generated)
    assert repr(generated.digest) not in repr(generated)
    assert parsed.secret not in repr(parsed)
    assert generated.token == f"rag_adm_{parsed.public_id}.{parsed.secret}"


@pytest.mark.parametrize(
    ("raw", "expected_kind"),
    [
        (f"unknown_{PUBLIC_ID}.{TOKEN_SECRET}", KeyKind.ADMIN),
        (f"rag_agent_{PUBLIC_ID}.{TOKEN_SECRET}", KeyKind.ADMIN),
        (f"rag_adm_{PUBLIC_ID}.{TOKEN_SECRET}", KeyKind.AGENT),
        (f"rag_adm_{PUBLIC_ID}{TOKEN_SECRET}", KeyKind.ADMIN),
        (f"rag_adm_.{TOKEN_SECRET}", KeyKind.ADMIN),
        (f"rag_adm_{PUBLIC_ID}.", KeyKind.ADMIN),
        (f"rag_adm_{PUBLIC_ID}.{TOKEN_SECRET}.extra", KeyKind.ADMIN),
        (f"rag_adm_{PUBLIC_ID}=.{TOKEN_SECRET}", KeyKind.ADMIN),
        (f"rag_adm_{PUBLIC_ID}.{TOKEN_SECRET}=", KeyKind.ADMIN),
        (f"rag_adm_{PUBLIC_ID}.{TOKEN_SECRET[:-1]}!", KeyKind.ADMIN),
        (f"rag_adm_A.{TOKEN_SECRET}", KeyKind.ADMIN),
        (f"rag_adm_{PUBLIC_ID}.A", KeyKind.ADMIN),
        ("x" * 257, KeyKind.ADMIN),
    ],
)
def test_parse_rejects_unknown_wrong_malformed_or_oversized_tokens_before_lookup(
    raw: str,
    expected_kind: KeyKind,
) -> None:
    repository_lookup = Mock()

    with pytest.raises(BusinessError) as exc_info:
        parsed = parse_token(raw, expected_kind)
        repository_lookup(parsed.public_id)

    assert exc_info.value == BusinessError(401, "INVALID_API_KEY", "Invalid API key")
    assert raw not in str(exc_info.value)
    repository_lookup.assert_not_called()


def test_parse_failure_does_not_retain_token_or_secret_in_service_traceback_locals() -> None:
    raw = "rag_adm_sensitive-public-id.sensitive-secret-value"

    try:
        parse_token(raw, KeyKind.ADMIN)
    except BusinessError as error:
        captured = error
    else:
        raise AssertionError("invalid token was accepted")

    assert captured.__context__ is None
    assert captured.__cause__ is None
    traceback = captured.__traceback__
    while traceback is not None:
        module_name = traceback.tb_frame.f_globals.get("__name__", "")
        if isinstance(module_name, str) and module_name.startswith("rag_service"):
            locals_repr = repr(traceback.tb_frame.f_locals)
            assert raw not in locals_repr
            assert "sensitive-secret-value" not in locals_repr
        traceback = traceback.tb_next


@pytest.mark.parametrize("failure_source", ["base64", "decode-helper"])
def test_parse_sanitizes_exact_retained_helper_exception(
    failure_source: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = f"rag_adm_{PUBLIC_ID}.{TOKEN_SECRET}"
    marker = f"sensitive-retained-parse-{failure_source}"
    retained_error, retained_errors = _retained_exception_graph(marker)
    if failure_source == "base64":
        monkeypatch.setattr(
            base64,
            "b64decode",
            Mock(side_effect=retained_error),
        )
    else:
        monkeypatch.setattr(
            codec_module,
            "_decode_canonical_component",
            Mock(side_effect=retained_error),
        )

    with pytest.raises(BusinessError) as exc_info:
        parse_token(raw, KeyKind.ADMIN)

    assert exc_info.value == BusinessError(401, "INVALID_API_KEY", "Invalid API key")
    _assert_codec_traceback_is_redacted(
        exc_info.value,
        marker,
        raw,
        PUBLIC_ID,
        TOKEN_SECRET,
    )
    _assert_retained_exceptions_are_sanitized(
        retained_errors,
        marker,
        raw,
        PUBLIC_ID,
        TOKEN_SECRET,
    )


def test_generation_failure_does_not_retain_generated_token_or_secret_in_traceback_locals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_values = iter((b"p" * 16, b"s" * 32))
    monkeypatch.setattr(secrets, "token_bytes", lambda _: next(generated_values))
    public_id = base64.urlsafe_b64encode(b"p" * 16).decode("ascii").rstrip("=")
    secret = base64.urlsafe_b64encode(b"s" * 32).decode("ascii").rstrip("=")
    token = f"rag_adm_{public_id}.{secret}"

    try:
        generate_token(KeyKind.ADMIN, object())  # type: ignore[arg-type]
    except BusinessError as error:
        captured = error
    else:
        raise AssertionError("invalid HMAC configuration was accepted")

    assert captured == BusinessError(500, "INTERNAL_ERROR", "Internal server error")
    assert captured.__context__ is None
    assert captured.__cause__ is None
    traceback = captured.__traceback__
    while traceback is not None:
        module_name = traceback.tb_frame.f_globals.get("__name__", "")
        if isinstance(module_name, str) and module_name.startswith("rag_service"):
            locals_repr = repr(traceback.tb_frame.f_locals)
            assert token not in locals_repr
            assert secret not in locals_repr
        traceback = traceback.tb_next


@pytest.mark.parametrize("failure_source", ["accessor", "hmac-new", "digest"])
def test_generate_sanitizes_exact_retained_hmac_exception(
    failure_source: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generated_values = iter((b"p" * 16, b"s" * 32))
    monkeypatch.setattr(secrets, "token_bytes", lambda _: next(generated_values))
    public_id = base64.urlsafe_b64encode(b"p" * 16).decode("ascii").rstrip("=")
    secret = base64.urlsafe_b64encode(b"s" * 32).decode("ascii").rstrip("=")
    token = f"rag_adm_{public_id}.{secret}"
    marker = f"sensitive-retained-generate-{failure_source}"
    retained_error, retained_errors = _retained_exception_graph(marker)
    hmac_secret = _install_retained_hmac_failure(
        failure_source,
        retained_error,
        monkeypatch,
    )

    with pytest.raises(BusinessError) as exc_info:
        generate_token(KeyKind.ADMIN, hmac_secret)

    assert exc_info.value == BusinessError(500, "INTERNAL_ERROR", "Internal server error")
    _assert_codec_traceback_is_redacted(
        exc_info.value,
        marker,
        token,
        secret,
        RETAINED_HMAC_VALUE,
    )
    _assert_retained_exceptions_are_sanitized(
        retained_errors,
        marker,
        token,
        secret,
        RETAINED_HMAC_VALUE,
    )


def test_digest_malformed_secret_redacts_hmac_credential_from_traceback_accessors() -> None:
    hmac_value = "sensitive-hmac-credential-that-must-not-survive"
    hmac_secret = SecretStr(hmac_value)

    with pytest.raises(BusinessError) as exc_info:
        digest_secret("sensitive-malformed-token-secret", hmac_secret)

    assert exc_info.value == BusinessError(401, "INVALID_API_KEY", "Invalid API key")
    _assert_codec_traceback_is_redacted(
        exc_info.value,
        "sensitive-malformed-token-secret",
        hmac_value,
    )


@pytest.mark.parametrize("failure_source", ["accessor", "hmac-helper"])
def test_digest_hmac_failures_surface_only_safe_internal_error(
    failure_source: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = f"sensitive-{failure_source}-failure"
    hmac_secret: SecretStr = ExplodingHmacSecret(marker)
    if failure_source == "hmac-helper":
        hmac_secret = ADMIN_HMAC_SECRET
        monkeypatch.setattr(
            hmac,
            "new",
            Mock(side_effect=RuntimeError(marker)),
        )

    with pytest.raises(BusinessError) as exc_info:
        digest_secret(TOKEN_SECRET, hmac_secret)

    assert exc_info.value == BusinessError(500, "INTERNAL_ERROR", "Internal server error")
    assert marker not in str(exc_info.value)
    _assert_codec_traceback_is_redacted(exc_info.value, marker)


@pytest.mark.parametrize("failure_source", ["accessor", "hmac-new", "digest"])
def test_digest_sanitizes_exact_retained_hmac_exception(
    failure_source: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = f"sensitive-retained-{failure_source}"
    retained_error, retained_errors = _retained_exception_graph(marker)
    hmac_secret = _install_retained_hmac_failure(
        failure_source,
        retained_error,
        monkeypatch,
    )

    with pytest.raises(BusinessError) as exc_info:
        digest_secret(TOKEN_SECRET, hmac_secret)

    assert exc_info.value == BusinessError(500, "INTERNAL_ERROR", "Internal server error")
    _assert_codec_traceback_is_redacted(
        exc_info.value,
        marker,
        TOKEN_SECRET,
        RETAINED_HMAC_VALUE,
    )
    _assert_retained_exceptions_are_sanitized(
        retained_errors,
        marker,
        TOKEN_SECRET,
        RETAINED_HMAC_VALUE,
    )


def test_digest_is_hmac_sha256_and_uses_independent_key_class_secrets() -> None:
    admin_digest = digest_secret(TOKEN_SECRET, ADMIN_HMAC_SECRET)
    agent_digest = digest_secret(TOKEN_SECRET, AGENT_HMAC_SECRET)

    assert isinstance(admin_digest, bytes)
    assert len(admin_digest) == 32
    assert len(agent_digest) == 32
    assert admin_digest != agent_digest
    assert verify_secret(TOKEN_SECRET, admin_digest, ADMIN_HMAC_SECRET)
    assert not verify_secret(TOKEN_SECRET, admin_digest, AGENT_HMAC_SECRET)


def test_verify_secret_uses_constant_time_comparison(monkeypatch: pytest.MonkeyPatch) -> None:
    expected_digest = digest_secret(TOKEN_SECRET, ADMIN_HMAC_SECRET)
    compare_digest = Mock(return_value=True)
    monkeypatch.setattr(hmac, "compare_digest", compare_digest)

    assert verify_secret(TOKEN_SECRET, expected_digest, ADMIN_HMAC_SECRET) is True
    compare_digest.assert_called_once_with(expected_digest, expected_digest)


def test_verify_secret_fails_closed_for_adversarial_accessor_and_compare_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_digest = digest_secret(TOKEN_SECRET, ADMIN_HMAC_SECRET)

    assert not verify_secret(
        TOKEN_SECRET,
        expected_digest,
        ExplodingHmacSecret("sensitive verify accessor failure"),
    )

    compare_digest = Mock(side_effect=RuntimeError("sensitive compare failure"))
    monkeypatch.setattr(hmac, "compare_digest", compare_digest)
    assert not verify_secret(TOKEN_SECRET, expected_digest, ADMIN_HMAC_SECRET)


def test_verify_secret_fails_closed_for_hmac_helper_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_digest = digest_secret(TOKEN_SECRET, ADMIN_HMAC_SECRET)
    monkeypatch.setattr(
        hmac,
        "new",
        Mock(side_effect=RuntimeError("sensitive verify HMAC helper failure")),
    )

    assert not verify_secret(TOKEN_SECRET, expected_digest, ADMIN_HMAC_SECRET)


@pytest.mark.parametrize(
    "failure_source",
    ["accessor", "hmac-new", "digest", "compare-digest"],
)
def test_verify_sanitizes_exact_retained_exception_before_failing_closed(
    failure_source: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_digest = digest_secret(TOKEN_SECRET, ADMIN_HMAC_SECRET)
    marker = f"sensitive-retained-verify-{failure_source}"
    retained_error, retained_errors = _retained_exception_graph(marker)
    if failure_source == "compare-digest":
        hmac_secret = SecretStr(RETAINED_HMAC_VALUE)
        monkeypatch.setattr(
            hmac,
            "compare_digest",
            Mock(side_effect=retained_error),
        )
    else:
        hmac_secret = _install_retained_hmac_failure(
            failure_source,
            retained_error,
            monkeypatch,
        )

    assert not verify_secret(TOKEN_SECRET, expected_digest, hmac_secret)
    _assert_retained_exceptions_are_sanitized(
        retained_errors,
        marker,
        TOKEN_SECRET,
        RETAINED_HMAC_VALUE,
    )


def test_verify_secret_requires_exact_digest_bytes_and_canonical_secret() -> None:
    expected_digest = digest_secret(TOKEN_SECRET, ADMIN_HMAC_SECRET)

    assert not verify_secret(
        TOKEN_SECRET,
        DigestBytesSubclass(expected_digest),
        ADMIN_HMAC_SECRET,
    )
    assert not verify_secret("sensitive-malformed-secret", expected_digest, ADMIN_HMAC_SECRET)


@pytest.mark.parametrize("expected_digest", [b"", b"short", b"x" * 31, b"x" * 33])
def test_verify_secret_rejects_wrong_digest_length_without_comparison(
    expected_digest: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compare_digest = Mock()
    monkeypatch.setattr(hmac, "compare_digest", compare_digest)

    assert not verify_secret(TOKEN_SECRET, expected_digest, ADMIN_HMAC_SECRET)
    compare_digest.assert_not_called()


@pytest.mark.parametrize("invalid_kind", ["admin", ForeignKeyKind.ADMIN])
def test_public_codec_helpers_require_exact_key_kind(invalid_kind: object) -> None:
    raw = f"rag_adm_{PUBLIC_ID}.{TOKEN_SECRET}"

    with pytest.raises(BusinessError) as parse_error:
        parse_token(raw, invalid_kind)  # type: ignore[arg-type]
    with pytest.raises(BusinessError) as generate_error:
        generate_token(invalid_kind, ADMIN_HMAC_SECRET)  # type: ignore[arg-type]

    expected = BusinessError(500, "INTERNAL_ERROR", "Internal server error")
    assert parse_error.value == expected
    assert generate_error.value == expected
    assert raw not in str(parse_error.value)
