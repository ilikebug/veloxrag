"""Application-layer encryption for provider credentials."""

import secrets
from collections.abc import Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import FrozenInstanceError, dataclass, field
from types import MappingProxyType
from typing import Final, TypeVar
from uuid import UUID

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_ALGORITHM: Final = "AES-256-GCM"
_NONCE_BYTES: Final = 12
_KEY_BYTES: Final = 32
_ERROR_MESSAGE: Final = "Provider credential unavailable"
_REDACTED_VERSION: Final = "<redacted>"

_Result = TypeVar("_Result")


class ProviderCredentialUnavailableError(Exception):
    """Safe failure exposed when a provider credential cannot be used."""

    __slots__ = ()

    code: Final = "PROVIDER_CREDENTIAL_UNAVAILABLE"

    def __init__(self) -> None:
        super().__init__(_ERROR_MESSAGE)


@dataclass(frozen=True, slots=True)
class EncryptedProviderCredential:
    """Immutable persistence value returned by provider credential encryption."""

    ciphertext: bytes = field(repr=False)
    nonce: bytes = field(repr=False)
    key_version: str = field(repr=False)
    algorithm: str = _ALGORITHM


def _copy_valid_keyring(keys: Mapping[str, bytes], active_key_version: str) -> dict[str, bytes]:
    copied: dict[str, bytes] = {}
    result: dict[str, bytes] = {}
    version = _REDACTED_VERSION
    key = b""
    try:
        if type(active_key_version) is not str or not active_key_version:
            raise ValueError
        copied = dict(keys)
        if not copied or active_key_version not in copied:
            raise ValueError
        for version, key in copied.items():
            if type(version) is not str or not version:
                raise ValueError
            if type(key) is not bytes or len(key) != _KEY_BYTES:
                raise ValueError
        result = {version: bytes(key) for version, key in copied.items()}
        return result
    finally:
        copied.clear()
        result = {}
        version = _REDACTED_VERSION
        key = b""
        keys = {}
        active_key_version = _REDACTED_VERSION


def _canonical_aad(credential_id: UUID, key_version: str) -> bytes:
    if type(credential_id) is not UUID or type(key_version) is not str or not key_version:
        raise ValueError
    return f"rag-provider-credential:v1:{credential_id}:{key_version}".encode()


def _random_nonce(size: int) -> bytes:
    return secrets.token_bytes(size)


def _wipe(buffer: bytearray) -> None:
    for index in range(len(buffer)):
        buffer[index] = 0
    with suppress(BufferError):
        buffer.clear()


@dataclass(slots=True, init=False, repr=False, eq=False)
class ProviderCredentialKeyring:
    """Immutable versioned AES-256-GCM keyring with one active encryption key."""

    _keys: Mapping[str, bytes]
    _active_key_version: str

    def __init__(
        self,
        *,
        keys: Mapping[str, bytes],
        active_key_version: str,
    ) -> None:
        copied: dict[str, bytes] = {}
        frozen_keys: Mapping[str, bytes] | None = None
        try:
            copied = _copy_valid_keyring(keys, active_key_version)
            frozen_keys = MappingProxyType(copied)
            object.__setattr__(self, "_keys", frozen_keys)
            object.__setattr__(self, "_active_key_version", active_key_version)
        except Exception:
            pass
        else:
            return

        copied.clear()
        frozen_keys = None
        keys = {}
        active_key_version = _REDACTED_VERSION
        raise ProviderCredentialUnavailableError from None

    def __setattr__(self, name: str, value: object) -> None:
        raise FrozenInstanceError(f"cannot assign to field {name!r}")

    def __delattr__(self, name: str) -> None:
        raise FrozenInstanceError(f"cannot delete field {name!r}")

    @property
    def active_key_version(self) -> str:
        return self._active_key_version

    def __repr__(self) -> str:
        return "ProviderCredentialKeyring(active_key_version=<redacted>, keys=<redacted>)"

    def _encrypt(
        self,
        credential_id: UUID,
        plaintext: bytes,
    ) -> EncryptedProviderCredential:
        key_version = _REDACTED_VERSION
        key = b""
        nonce = b""
        aad = b""
        ciphertext = b""
        encrypted: EncryptedProviderCredential | None = None
        try:
            if type(plaintext) is not bytes:
                raise TypeError
            key_version = self._active_key_version
            key = self._keys[key_version]
            nonce = _random_nonce(_NONCE_BYTES)
            if type(nonce) is not bytes or len(nonce) != _NONCE_BYTES:
                raise ValueError
            aad = _canonical_aad(credential_id, key_version)
            ciphertext = AESGCM(key).encrypt(nonce, plaintext, aad)
            encrypted = EncryptedProviderCredential(
                ciphertext=ciphertext,
                nonce=nonce,
                key_version=key_version,
            )
            return encrypted
        finally:
            plaintext = b""
            key_version = _REDACTED_VERSION
            key = b""
            nonce = b""
            aad = b""
            ciphertext = b""
            encrypted = None

    def encrypt(
        self,
        credential_id: UUID,
        plaintext: bytes,
    ) -> EncryptedProviderCredential:
        """Encrypt with the active key version and an independent 96-bit nonce."""

        try:
            return self._encrypt(credential_id, plaintext)
        except Exception:
            pass

        plaintext = b""
        raise ProviderCredentialUnavailableError from None

    def _decrypt(
        self,
        credential_id: UUID,
        encrypted: EncryptedProviderCredential,
    ) -> bytes:
        key = b""
        aad = b""
        plaintext = b""
        try:
            if type(encrypted) is not EncryptedProviderCredential:
                raise TypeError
            if encrypted.algorithm != _ALGORITHM:
                raise ValueError
            if type(encrypted.nonce) is not bytes or len(encrypted.nonce) != _NONCE_BYTES:
                raise ValueError
            if type(encrypted.ciphertext) is not bytes or len(encrypted.ciphertext) < 16:
                raise ValueError
            if type(encrypted.key_version) is not str or not encrypted.key_version:
                raise ValueError
            key = self._keys[encrypted.key_version]
            aad = _canonical_aad(credential_id, encrypted.key_version)
            plaintext = AESGCM(key).decrypt(encrypted.nonce, encrypted.ciphertext, aad)
            return plaintext
        finally:
            key = b""
            aad = b""
            plaintext = b""
            encrypted = EncryptedProviderCredential(
                ciphertext=b"",
                nonce=b"",
                key_version=_REDACTED_VERSION,
                algorithm="<redacted>",
            )

    def _decrypt_safely(
        self,
        credential_id: UUID,
        encrypted: EncryptedProviderCredential,
    ) -> bytes:
        try:
            return self._decrypt(credential_id, encrypted)
        except Exception:
            pass

        encrypted = EncryptedProviderCredential(
            ciphertext=b"",
            nonce=b"",
            key_version=_REDACTED_VERSION,
            algorithm="<redacted>",
        )
        raise ProviderCredentialUnavailableError from None

    def use_decrypted(
        self,
        credential_id: UUID,
        encrypted: EncryptedProviderCredential,
        callback: Callable[[bytearray], _Result],
    ) -> _Result:
        """Use plaintext within a callback that must not copy or retain the supplied buffer."""

        plaintext = self._decrypt_safely(credential_id, encrypted)
        buffer = bytearray(plaintext)
        plaintext = b""
        try:
            return callback(buffer)
        finally:
            _wipe(buffer)

    async def use_decrypted_async(
        self,
        credential_id: UUID,
        encrypted: EncryptedProviderCredential,
        callback: Callable[[bytearray], Awaitable[_Result]],
    ) -> _Result:
        """Await a callback that must not copy or retain the supplied plaintext buffer."""

        plaintext = self._decrypt_safely(credential_id, encrypted)
        buffer = bytearray(plaintext)
        plaintext = b""
        try:
            return await callback(buffer)
        finally:
            _wipe(buffer)
