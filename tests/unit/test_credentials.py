import asyncio
from collections import deque
from collections.abc import Iterator, Mapping
from dataclasses import FrozenInstanceError, replace
from typing import Any
from uuid import UUID, uuid4

import pytest
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from rag_service.providers import (
    EncryptedProviderCredential,
    ProviderCredentialKeyring,
    ProviderCredentialUnavailableError,
)
from rag_service.providers import credentials as credentials_module

KEY_V1 = bytes(range(32))
KEY_V2 = bytes(reversed(range(32)))
PLAINTEXT = b"sensitive-openrouter-api-key"
ALGORITHM = "AES-256-GCM"
ERROR_CODE = "PROVIDER_CREDENTIAL_UNAVAILABLE"


def _keyring(
    *,
    keys: dict[str, bytes] | None = None,
    active_key_version: str = "version-one",
) -> ProviderCredentialKeyring:
    selected_keys = keys or {"version-one": KEY_V1}
    return ProviderCredentialKeyring(
        keys=selected_keys,
        active_key_version=active_key_version,
    )


def _assert_safe_unavailable_error(
    error: ProviderCredentialUnavailableError,
    *sensitive_values: object,
) -> None:
    assert error.code == ERROR_CODE
    assert error.args == ("Provider credential unavailable",)
    assert str(error) == "Provider credential unavailable"
    assert error.__cause__ is None
    assert error.__context__ is None
    rendered = f"{error!s} {error!r} {error.args!r}"
    for value in sensitive_values:
        assert str(value) not in rendered
        assert repr(value) not in rendered


def _assert_traceback_has_no_markers(error: BaseException, *markers: object) -> None:
    traceback = error.__traceback__
    while traceback is not None:
        module_name = traceback.tb_frame.f_globals.get("__name__", "")
        if module_name == "rag_service.providers.credentials":
            rendered = repr(traceback.tb_frame.f_locals)
            for marker in markers:
                assert str(marker) not in rendered
                assert repr(marker) not in rendered
        traceback = traceback.tb_next


def _assert_decrypts_to(
    keyring: ProviderCredentialKeyring,
    credential_id: UUID,
    encrypted: EncryptedProviderCredential,
    expected: bytes,
) -> None:
    callback_called = False

    def assert_plaintext(plaintext: bytearray) -> None:
        nonlocal callback_called
        callback_called = True
        assert bytes(plaintext) == expected

    assert keyring.use_decrypted(credential_id, encrypted, assert_plaintext) is None
    assert callback_called


def _consume_decrypted(plaintext: bytearray) -> None:
    del plaintext


def test_aes_gcm_round_trip_uses_persistence_compatible_value() -> None:
    credential_id = uuid4()
    encrypted = _keyring().encrypt(credential_id, PLAINTEXT)

    assert isinstance(encrypted, EncryptedProviderCredential)
    assert encrypted.algorithm == ALGORITHM
    assert encrypted.key_version == "version-one"
    assert isinstance(encrypted.ciphertext, bytes)
    assert len(encrypted.ciphertext) == len(PLAINTEXT) + 16
    assert isinstance(encrypted.nonce, bytes)
    assert len(encrypted.nonce) == 12
    _assert_decrypts_to(_keyring(), credential_id, encrypted, PLAINTEXT)
    assert not hasattr(_keyring(), "decrypt")


def test_default_repeated_encryptions_use_independent_96_bit_nonces() -> None:
    keyring = _keyring()
    credential_id = uuid4()
    encrypted_values = [keyring.encrypt(credential_id, PLAINTEXT) for _ in range(64)]

    assert len({value.nonce for value in encrypted_values}) == 64
    assert all(len(value.nonce) == 12 for value in encrypted_values)


def test_each_encryption_requests_a_fresh_96_bit_nonce(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_nonces = deque(index.to_bytes(12, "big") for index in range(1, 65))

    def next_nonce(size: int) -> bytes:
        assert size == 12
        return expected_nonces.popleft()

    monkeypatch.setattr(credentials_module, "_random_nonce", next_nonce)
    keyring = _keyring()
    credential_id = uuid4()
    encrypted_values = [keyring.encrypt(credential_id, PLAINTEXT) for _ in range(64)]

    assert len({value.nonce for value in encrypted_values}) == 64
    assert all(len(value.nonce) == 12 for value in encrypted_values)
    assert not expected_nonces


def test_keyring_constructor_does_not_expose_nonce_source_injection() -> None:
    with pytest.raises(TypeError):
        ProviderCredentialKeyring(
            keys={"version-one": KEY_V1},
            active_key_version="version-one",
            nonce_source=lambda _: b"n" * 12,  # type: ignore[call-arg]
        )

    assert "_nonce_source" not in ProviderCredentialKeyring.__slots__


@pytest.mark.parametrize(
    "tamper",
    [
        lambda value: replace(
            value,
            ciphertext=value.ciphertext[:-1] + bytes([value.ciphertext[-1] ^ 1]),
        ),
        lambda value: replace(value, nonce=bytes([value.nonce[0] ^ 1]) + value.nonce[1:]),
        lambda value: replace(value, nonce=value.nonce[:-1]),
        lambda value: replace(value, key_version="version-two"),
        lambda value: replace(value, algorithm="AES-128-GCM"),
    ],
    ids=["ciphertext", "nonce", "bad-nonce-length", "key-version", "algorithm"],
)
def test_tampered_stored_fields_fail_with_one_sanitized_domain_error(
    tamper: Any,
) -> None:
    credential_id = uuid4()
    keyring = _keyring(keys={"version-one": KEY_V1, "version-two": KEY_V2})
    encrypted = keyring.encrypt(credential_id, PLAINTEXT)
    tampered = tamper(encrypted)

    with pytest.raises(ProviderCredentialUnavailableError) as exc_info:
        keyring.use_decrypted(credential_id, tampered, _consume_decrypted)

    _assert_safe_unavailable_error(
        exc_info.value,
        PLAINTEXT,
        tampered.ciphertext,
        tampered.nonce,
        tampered.key_version,
        KEY_V1,
        KEY_V2,
    )
    _assert_traceback_has_no_markers(
        exc_info.value,
        PLAINTEXT,
        tampered.ciphertext,
        tampered.nonce,
        tampered.key_version,
        KEY_V1,
        KEY_V2,
    )


def test_ciphertext_is_bound_to_exact_credential_uuid_aad() -> None:
    credential_id = uuid4()
    different_credential_id = uuid4()
    keyring = _keyring()
    encrypted = keyring.encrypt(credential_id, PLAINTEXT)

    with pytest.raises(ProviderCredentialUnavailableError) as exc_info:
        keyring.use_decrypted(
            different_credential_id,
            encrypted,
            _consume_decrypted,
        )

    _assert_safe_unavailable_error(
        exc_info.value,
        PLAINTEXT,
        encrypted.ciphertext,
        encrypted.nonce,
        encrypted.key_version,
        KEY_V1,
    )


@pytest.mark.parametrize(
    "keys",
    [
        {"version-two": KEY_V2},
        {"version-one": b"x" * 32},
    ],
    ids=["missing-historical-version", "wrong-historical-key"],
)
def test_missing_or_wrong_historical_key_fails_safely(keys: dict[str, bytes]) -> None:
    credential_id = uuid4()
    encrypted = _keyring().encrypt(credential_id, PLAINTEXT)
    active_version = next(iter(keys))
    rotated_keyring = _keyring(keys=keys, active_key_version=active_version)

    with pytest.raises(ProviderCredentialUnavailableError) as exc_info:
        rotated_keyring.use_decrypted(credential_id, encrypted, _consume_decrypted)

    _assert_safe_unavailable_error(
        exc_info.value,
        PLAINTEXT,
        encrypted.ciphertext,
        encrypted.nonce,
        encrypted.key_version,
        *keys.values(),
    )


def test_key_rotation_decrypts_old_values_and_encrypts_with_new_active_version() -> None:
    credential_id = uuid4()
    old_keyring = _keyring()
    old_value = old_keyring.encrypt(credential_id, PLAINTEXT)
    rotated_keyring = _keyring(
        keys={"version-one": KEY_V1, "version-two": KEY_V2},
        active_key_version="version-two",
    )

    _assert_decrypts_to(rotated_keyring, credential_id, old_value, PLAINTEXT)
    new_value = rotated_keyring.encrypt(credential_id, PLAINTEXT)
    assert new_value.key_version == "version-two"
    _assert_decrypts_to(rotated_keyring, credential_id, new_value, PLAINTEXT)

    without_old_key = _keyring(keys={"version-two": KEY_V2}, active_key_version="version-two")
    with pytest.raises(ProviderCredentialUnavailableError):
        without_old_key.use_decrypted(credential_id, old_value, _consume_decrypted)


def test_aad_is_exact_canonical_utf8_contract() -> None:
    credential_id = UUID("12345678-1234-5678-9abc-def012345678")
    encrypted = _keyring().encrypt(credential_id, PLAINTEXT)
    expected_aad = b"rag-provider-credential:v1:12345678-1234-5678-9abc-def012345678:version-one"

    assert AESGCM(KEY_V1).decrypt(encrypted.nonce, encrypted.ciphertext, expected_aad) == PLAINTEXT
    for wrong_aad in (
        expected_aad.upper(),
        expected_aad + b" ",
        expected_aad.replace(b"version-one", b"version-two"),
    ):
        with pytest.raises(InvalidTag):
            AESGCM(KEY_V1).decrypt(encrypted.nonce, encrypted.ciphertext, wrong_aad)


def test_encrypted_value_and_keyring_are_immutable_copied_and_repr_safe() -> None:
    credential_id = uuid4()
    caller_keys = {"version-one": KEY_V1}
    keyring = _keyring(keys=caller_keys)
    encrypted = keyring.encrypt(credential_id, PLAINTEXT)

    caller_keys["version-one"] = KEY_V2
    caller_keys["version-two"] = KEY_V2

    _assert_decrypts_to(keyring, credential_id, encrypted, PLAINTEXT)
    with pytest.raises(FrozenInstanceError):
        encrypted.nonce = b"z" * 12  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        keyring.active_key_version = "version-two"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        del keyring._active_key_version

    rendered = f"{encrypted!r} {keyring!r}"
    for sensitive in (
        PLAINTEXT,
        encrypted.ciphertext,
        encrypted.nonce,
        encrypted.key_version,
        KEY_V1,
        KEY_V2,
    ):
        assert str(sensitive) not in rendered
        assert repr(sensitive) not in rendered
    with pytest.raises(TypeError):
        vars(encrypted)
    with pytest.raises(TypeError):
        vars(keyring)


@pytest.mark.parametrize(
    ("keys", "active_key_version"),
    [
        ({}, "version-one"),
        ({"version-one": KEY_V1}, "missing-sensitive-version"),
        ({"": KEY_V1}, ""),
        ({"sensitive-version": b"short"}, "sensitive-version"),
        ({"sensitive-version": "sensitive-key-material"}, "sensitive-version"),
    ],
    ids=["empty", "active-missing", "empty-version", "wrong-size", "wrong-type"],
)
def test_invalid_keyring_configuration_is_rejected_without_echoing_input(
    keys: Any,
    active_key_version: str,
) -> None:
    with pytest.raises(ProviderCredentialUnavailableError) as exc_info:
        ProviderCredentialKeyring(keys=keys, active_key_version=active_key_version)

    sensitive_values = tuple(
        value for value in (active_key_version, *keys.keys(), *keys.values()) if value
    )
    _assert_safe_unavailable_error(exc_info.value, *sensitive_values)
    _assert_traceback_has_no_markers(exc_info.value, *sensitive_values)


def test_late_keyring_initialization_failure_clears_copied_key_material(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure_marker = "sensitive-late-keyring-failure"
    nonce_source_marker = "sensitive-nonce-source-marker"
    original_args = (f"{failure_marker}:{nonce_source_marker}:{PLAINTEXT!r}",)
    retained_error = RuntimeError(*original_args)
    original_cause = RuntimeError("caller-owned-cause")
    original_context = RuntimeError("caller-owned-context")
    retained_error.__cause__ = original_cause
    retained_error.__context__ = original_context

    def fail_after_copy(_: dict[str, bytes]) -> Any:
        raise retained_error

    monkeypatch.setattr(credentials_module, "MappingProxyType", fail_after_copy)

    with pytest.raises(ProviderCredentialUnavailableError) as exc_info:
        ProviderCredentialKeyring(
            keys={"sensitive-version": KEY_V1},
            active_key_version="sensitive-version",
        )

    _assert_safe_unavailable_error(
        exc_info.value,
        failure_marker,
        nonce_source_marker,
        PLAINTEXT,
        "sensitive-version",
        KEY_V1,
    )
    _assert_traceback_has_no_markers(
        exc_info.value,
        failure_marker,
        nonce_source_marker,
        PLAINTEXT,
        "sensitive-version",
        KEY_V1,
    )
    assert retained_error.args == original_args
    assert retained_error.__traceback__ is not None
    assert retained_error.__cause__ is original_cause
    assert retained_error.__context__ is original_context
    _assert_traceback_has_no_markers(
        retained_error,
        nonce_source_marker,
        PLAINTEXT,
        "sensitive-version",
        KEY_V1,
    )


def test_mapping_failure_preserves_source_exception_but_clears_credential_locals() -> None:
    version_marker = "sensitive-mapping-version"
    mapping_marker = "sensitive-mapping-key-material"
    original_args = ("caller-owned-mapping-failure",)
    retained_error = RuntimeError(*original_args)

    class FailingMapping(Mapping[str, bytes]):
        def __getitem__(self, key: str) -> bytes:
            del key
            raise retained_error

        def __iter__(self) -> Iterator[str]:
            raise retained_error

        def __len__(self) -> int:
            return 1

        def __repr__(self) -> str:
            return mapping_marker

    with pytest.raises(ProviderCredentialUnavailableError) as exc_info:
        ProviderCredentialKeyring(
            keys=FailingMapping(),
            active_key_version=version_marker,
        )

    _assert_safe_unavailable_error(
        exc_info.value,
        version_marker,
        mapping_marker,
    )
    assert retained_error.args == original_args
    assert retained_error.__traceback__ is not None
    _assert_traceback_has_no_markers(
        retained_error,
        version_marker,
        mapping_marker,
    )


def test_hostile_mapping_failure_uses_domain_error_without_reading_source_attributes() -> None:
    version_marker = "sensitive-hostile-mapping-version"
    mapping_marker = "sensitive-hostile-mapping-key"
    hostile_marker = "sensitive-hostile-mapping-exception-access"

    class HostileMappingError(Exception):
        def __getattribute__(self, name: str) -> object:
            if name in {"args", "__cause__", "__context__", "__traceback__"}:
                raise RuntimeError(hostile_marker)
            return super().__getattribute__(name)

    hostile_error = HostileMappingError("sensitive hostile mapping failure")

    class HostileMapping(Mapping[str, bytes]):
        def __getitem__(self, key: str) -> bytes:
            del key
            raise hostile_error

        def __iter__(self) -> Iterator[str]:
            raise hostile_error

        def __len__(self) -> int:
            return 1

        def __repr__(self) -> str:
            return mapping_marker

    with pytest.raises(ProviderCredentialUnavailableError) as exc_info:
        ProviderCredentialKeyring(
            keys=HostileMapping(),
            active_key_version=version_marker,
        )

    _assert_safe_unavailable_error(
        exc_info.value,
        version_marker,
        mapping_marker,
        hostile_marker,
    )
    _assert_traceback_has_no_markers(
        exc_info.value,
        version_marker,
        mapping_marker,
        hostile_marker,
    )


def test_sync_callback_success_returns_result_and_clears_plaintext_buffer() -> None:
    credential_id = uuid4()
    keyring = _keyring()
    encrypted = keyring.encrypt(credential_id, PLAINTEXT)
    retained_buffers: list[bytearray] = []
    callback_result = object()

    def callback(plaintext: bytearray) -> object:
        assert bytes(plaintext) == PLAINTEXT
        retained_buffers.append(plaintext)
        return callback_result

    assert keyring.use_decrypted(credential_id, encrypted, callback) is callback_result
    assert retained_buffers == [bytearray()]


def test_sync_callback_exception_propagates_after_plaintext_buffer_cleanup() -> None:
    credential_id = uuid4()
    keyring = _keyring()
    encrypted = keyring.encrypt(credential_id, PLAINTEXT)
    retained_buffers: list[bytearray] = []

    class CallbackFailure(RuntimeError):
        pass

    callback_error = CallbackFailure("safe callback failure")

    def callback(plaintext: bytearray) -> None:
        assert bytes(plaintext) == PLAINTEXT
        retained_buffers.append(plaintext)
        raise callback_error

    with pytest.raises(CallbackFailure) as exc_info:
        keyring.use_decrypted(credential_id, encrypted, callback)

    assert exc_info.value is callback_error
    assert retained_buffers == [bytearray()]
    traceback = callback_error.__traceback__
    while traceback is not None:
        module_name = traceback.tb_frame.f_globals.get("__name__", "")
        if (
            module_name == "rag_service.providers.credentials"
            or traceback.tb_frame.f_code is callback.__code__
        ):
            assert repr(PLAINTEXT) not in repr(traceback.tb_frame.f_locals)
        traceback = traceback.tb_next


def test_sync_callback_memoryview_does_not_mask_error_and_observes_zeroized_buffer() -> None:
    credential_id = uuid4()
    keyring = _keyring()
    encrypted = keyring.encrypt(credential_id, PLAINTEXT)
    retained_buffers: list[bytearray] = []
    retained_views: list[memoryview] = []

    class CallbackFailure(RuntimeError):
        pass

    callback_error = CallbackFailure("safe callback failure")

    def callback(plaintext: bytearray) -> None:
        retained_buffers.append(plaintext)
        retained_views.append(memoryview(plaintext))
        raise callback_error

    with pytest.raises(CallbackFailure) as exc_info:
        keyring.use_decrypted(credential_id, encrypted, callback)

    assert exc_info.value is callback_error
    assert bytes(retained_buffers[0]) == b"\x00" * len(PLAINTEXT)
    assert bytes(retained_views[0]) == b"\x00" * len(PLAINTEXT)
    retained_views[0].release()


@pytest.mark.asyncio
async def test_async_cancellation_propagates_after_plaintext_buffer_cleanup() -> None:
    credential_id = uuid4()
    keyring = _keyring()
    encrypted = keyring.encrypt(credential_id, PLAINTEXT)
    retained_buffers: list[bytearray] = []

    async def callback(plaintext: bytearray) -> None:
        assert bytes(plaintext) == PLAINTEXT
        retained_buffers.append(plaintext)
        raise asyncio.CancelledError("safe cancellation")

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await keyring.use_decrypted_async(credential_id, encrypted, callback)

    assert exc_info.value.args == ("safe cancellation",)
    assert retained_buffers == [bytearray()]
    traceback = exc_info.value.__traceback__
    while traceback is not None:
        module_name = traceback.tb_frame.f_globals.get("__name__", "")
        if (
            module_name == "rag_service.providers.credentials"
            or traceback.tb_frame.f_code is callback.__code__
        ):
            assert repr(PLAINTEXT) not in repr(traceback.tb_frame.f_locals)
        traceback = traceback.tb_next


@pytest.mark.asyncio
async def test_async_callback_success_returns_result_and_clears_plaintext_buffer() -> None:
    credential_id = uuid4()
    keyring = _keyring()
    encrypted = keyring.encrypt(credential_id, PLAINTEXT)
    retained_buffers: list[bytearray] = []
    callback_result = object()

    async def callback(plaintext: bytearray) -> object:
        assert bytes(plaintext) == PLAINTEXT
        retained_buffers.append(plaintext)
        return callback_result

    result = await keyring.use_decrypted_async(credential_id, encrypted, callback)

    assert result is callback_result
    assert retained_buffers == [bytearray()]


@pytest.mark.asyncio
async def test_async_business_exception_propagates_after_plaintext_buffer_cleanup() -> None:
    credential_id = uuid4()
    keyring = _keyring()
    encrypted = keyring.encrypt(credential_id, PLAINTEXT)
    retained_buffers: list[bytearray] = []
    callback_error = RuntimeError("safe async callback failure")

    async def callback(plaintext: bytearray) -> None:
        assert bytes(plaintext) == PLAINTEXT
        retained_buffers.append(plaintext)
        raise callback_error

    with pytest.raises(RuntimeError) as exc_info:
        await keyring.use_decrypted_async(credential_id, encrypted, callback)

    assert exc_info.value is callback_error
    assert retained_buffers == [bytearray()]
    traceback = callback_error.__traceback__
    while traceback is not None:
        module_name = traceback.tb_frame.f_globals.get("__name__", "")
        if (
            module_name == "rag_service.providers.credentials"
            or traceback.tb_frame.f_code is callback.__code__
        ):
            assert repr(PLAINTEXT) not in repr(traceback.tb_frame.f_locals)
        traceback = traceback.tb_next


@pytest.mark.asyncio
async def test_async_memoryview_does_not_mask_cancellation_and_observes_zeroized_buffer() -> None:
    credential_id = uuid4()
    keyring = _keyring()
    encrypted = keyring.encrypt(credential_id, PLAINTEXT)
    retained_buffers: list[bytearray] = []
    retained_views: list[memoryview] = []

    async def callback(plaintext: bytearray) -> None:
        retained_buffers.append(plaintext)
        retained_views.append(memoryview(plaintext))
        raise asyncio.CancelledError("safe cancellation")

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await keyring.use_decrypted_async(credential_id, encrypted, callback)

    assert exc_info.value.args == ("safe cancellation",)
    assert bytes(retained_buffers[0]) == b"\x00" * len(PLAINTEXT)
    assert bytes(retained_views[0]) == b"\x00" * len(PLAINTEXT)
    retained_views[0].release()


def test_reusable_objects_do_not_retain_plaintext() -> None:
    credential_id = uuid4()
    keyring = _keyring()
    encrypted = keyring.encrypt(credential_id, PLAINTEXT)

    _assert_decrypts_to(keyring, credential_id, encrypted, PLAINTEXT)
    assert repr(PLAINTEXT) not in repr(keyring)
    assert repr(PLAINTEXT) not in repr(encrypted)
    for slot in keyring.__slots__:
        assert getattr(keyring, slot) != PLAINTEXT
    for slot in encrypted.__slots__:
        assert getattr(encrypted, slot) != PLAINTEXT


def test_random_source_failure_does_not_mutate_caller_owned_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_args = ("sensitive nonce source failure",)
    retained_error = RuntimeError(*original_args)
    original_cause = RuntimeError("caller-owned-random-cause")
    original_context = RuntimeError("caller-owned-random-context")
    retained_error.__cause__ = original_cause
    retained_error.__context__ = original_context

    def failing_nonce_source(_: int) -> bytes:
        raise retained_error

    monkeypatch.setattr(credentials_module, "_random_nonce", failing_nonce_source)

    with pytest.raises(ProviderCredentialUnavailableError) as exc_info:
        _keyring().encrypt(uuid4(), PLAINTEXT)

    _assert_safe_unavailable_error(
        exc_info.value,
        PLAINTEXT,
        KEY_V1,
        "sensitive nonce source failure",
    )
    assert retained_error.args == original_args
    assert retained_error.__traceback__ is not None
    assert retained_error.__cause__ is original_cause
    assert retained_error.__context__ is original_context
    _assert_traceback_has_no_markers(
        retained_error,
        PLAINTEXT,
        KEY_V1,
        "version-one",
    )
    _assert_traceback_has_no_markers(exc_info.value, PLAINTEXT, KEY_V1, "version-one")


def test_decrypt_failure_does_not_mutate_caller_owned_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential_id = uuid4()
    keyring = _keyring()
    encrypted = keyring.encrypt(credential_id, PLAINTEXT)
    original_args = ("sensitive shared decrypt failure",)
    retained_error = RuntimeError(*original_args)
    original_cause = RuntimeError("caller-owned-decrypt-cause")
    original_context = RuntimeError("caller-owned-decrypt-context")
    retained_error.__cause__ = original_cause
    retained_error.__context__ = original_context

    class FailingAESGCM:
        def __init__(self, key: bytes) -> None:
            assert key == KEY_V1

        def decrypt(self, nonce: bytes, ciphertext: bytes, aad: bytes) -> bytes:
            assert nonce == encrypted.nonce
            assert ciphertext == encrypted.ciphertext
            assert aad.endswith(b":version-one")
            raise retained_error

    monkeypatch.setattr(credentials_module, "AESGCM", FailingAESGCM)

    with pytest.raises(ProviderCredentialUnavailableError) as exc_info:
        keyring.use_decrypted(credential_id, encrypted, _consume_decrypted)

    _assert_safe_unavailable_error(
        exc_info.value,
        PLAINTEXT,
        KEY_V1,
        encrypted.ciphertext,
        encrypted.nonce,
        "version-one",
    )
    assert retained_error.args == original_args
    assert retained_error.__traceback__ is not None
    assert retained_error.__cause__ is original_cause
    assert retained_error.__context__ is original_context
    _assert_traceback_has_no_markers(
        retained_error,
        PLAINTEXT,
        KEY_V1,
        encrypted.ciphertext,
        encrypted.nonce,
        "version-one",
    )


def test_hostile_source_exception_attributes_cannot_escape_safe_error_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hostile_marker = "sensitive-hostile-exception-access"

    class HostileSourceError(Exception):
        def __getattribute__(self, name: str) -> object:
            if name in {"args", "__cause__", "__context__", "__traceback__"}:
                raise RuntimeError(hostile_marker)
            return super().__getattribute__(name)

    hostile_error = HostileSourceError("sensitive hostile source")

    def failing_random_source(_: int) -> bytes:
        raise hostile_error

    monkeypatch.setattr(credentials_module, "_random_nonce", failing_random_source)

    with pytest.raises(ProviderCredentialUnavailableError) as exc_info:
        _keyring().encrypt(uuid4(), PLAINTEXT)

    _assert_safe_unavailable_error(
        exc_info.value,
        hostile_marker,
        PLAINTEXT,
        KEY_V1,
        "version-one",
    )
    _assert_traceback_has_no_markers(
        exc_info.value,
        hostile_marker,
        PLAINTEXT,
        KEY_V1,
        "version-one",
    )
