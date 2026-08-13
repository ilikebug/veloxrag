import asyncio
import hashlib
import math
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from time import monotonic
from types import MappingProxyType
from typing import Protocol
from uuid import UUID

from rag_service.providers.credentials import (
    EncryptedProviderCredential,
    ProviderCredentialKeyring,
)

INGEST_GENERATION_STATUSES = frozenset({"active", "building"})
RETRIEVE_GENERATION_STATUSES = frozenset({"active"})
NOT_CHECKED_ERROR = "NotChecked"
REQUIRED_COMPONENTS = frozenset(
    {
        "postgres",
        "qdrant",
        "redis",
        "minio",
        "ingest_credentials",
        "retrieve_credentials",
    }
)

_CredentialCacheKey = tuple[UUID, int, str, str]
_CredentialLoader = Callable[[Collection[str]], Awaitable[Sequence["ReferencedCredential"]]]


class ReadinessScope(StrEnum):
    ALL = "all"
    CORE = "core"
    INGEST = "ingest"
    RETRIEVE = "retrieve"


def provider_credential_keyring_fingerprint(
    serialized_keyring: str,
    active_key_version: str,
) -> str:
    if type(serialized_keyring) is not str or type(active_key_version) is not str:
        raise ValueError("keyring fingerprint input is invalid")
    digest = hashlib.sha256()
    digest.update(b"rag-provider-credential-keyring:v1\x00")
    digest.update(active_key_version.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(serialized_keyring.encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ComponentStatus:
    ok: bool
    latency_ms: float
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ReferencedCredential:
    credential_id: UUID
    resource_revision: int
    encrypted: EncryptedProviderCredential


class ReferencedCredentialValidator:
    """Authenticate only credentials referenced by the requested generation states."""

    def __init__(
        self,
        *,
        load_referenced_credentials: _CredentialLoader,
        cache_ttl_seconds: float = 5.0,
        max_cache_entries: int = 1024,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if not callable(load_referenced_credentials):
            raise TypeError("credential loader must be callable")
        if (
            type(cache_ttl_seconds) not in {int, float}
            or not math.isfinite(cache_ttl_seconds)
            or cache_ttl_seconds <= 0
        ):
            if type(cache_ttl_seconds) in {int, float} and not math.isfinite(cache_ttl_seconds):
                raise ValueError("credential readiness cache TTL must be finite")
            raise ValueError("credential readiness cache TTL must be positive")
        if type(max_cache_entries) is not int or max_cache_entries <= 0:
            raise ValueError("credential readiness cache entry limit must be positive")
        if not callable(clock):
            raise TypeError("credential readiness clock must be callable")
        self._load_referenced_credentials = load_referenced_credentials
        self._cache_ttl_seconds = cache_ttl_seconds
        self._max_cache_entries = max_cache_entries
        self._clock = clock
        self._authenticated_until: dict[_CredentialCacheKey, float] = {}
        self._lock = asyncio.Lock()

    async def validate(
        self,
        *,
        statuses: Collection[str],
        keyring: ProviderCredentialKeyring,
        keyring_fingerprint: str,
    ) -> None:
        selected_statuses = frozenset(statuses)
        if not selected_statuses or not selected_statuses.issubset(INGEST_GENERATION_STATUSES):
            raise ValueError("generation statuses are invalid")
        if type(keyring_fingerprint) is not str or not keyring_fingerprint:
            raise ValueError("keyring fingerprint is invalid")

        credentials = await self._load_referenced_credentials(selected_statuses)
        async with self._lock:
            now = self._clock()
            self._authenticated_until = {
                cache_key: expires_at
                for cache_key, expires_at in self._authenticated_until.items()
                if expires_at > now
            }
            for credential in credentials:
                cache_key = (
                    credential.credential_id,
                    credential.resource_revision,
                    credential.encrypted.key_version,
                    keyring_fingerprint,
                )
                if self._authenticated_until.get(cache_key, 0.0) > now:
                    continue
                keyring.use_decrypted(
                    credential.credential_id,
                    credential.encrypted,
                    lambda _plaintext: None,
                )
                if (
                    cache_key not in self._authenticated_until
                    and len(self._authenticated_until) >= self._max_cache_entries
                ):
                    oldest_cache_key = next(iter(self._authenticated_until))
                    del self._authenticated_until[oldest_cache_key]
                self._authenticated_until[cache_key] = now + self._cache_ttl_seconds


@dataclass(frozen=True, slots=True)
class ReadinessSnapshot:
    components: Mapping[str, ComponentStatus]
    answer_configured: bool

    def __post_init__(self) -> None:
        components = dict(self.components)
        missing_components = sorted(REQUIRED_COMPONENTS.difference(components))
        if missing_components:
            missing_names = ", ".join(missing_components)
            raise ValueError(f"Missing required readiness components: {missing_names}")

        object.__setattr__(self, "components", MappingProxyType(components))

    @property
    def core_ready(self) -> bool:
        return self.components["postgres"].ok and self.components["qdrant"].ok

    def _component_checked(self, name: str) -> bool:
        return self.components[name].error != NOT_CHECKED_ERROR

    @property
    def retrieval_capability(self) -> bool | None:
        if not self._component_checked("postgres") or not self._component_checked("qdrant"):
            return None
        if not self.core_ready:
            return False
        if self._component_checked("retrieve_credentials"):
            return self.components["retrieve_credentials"].ok
        if self._component_checked("ingest_credentials"):
            return True if self.components["ingest_credentials"].ok else None
        return None

    @property
    def ingest_capability(self) -> bool | None:
        if not self._component_checked("postgres") or not self._component_checked("qdrant"):
            return None
        if not self.core_ready:
            return False
        required = ("redis", "minio", "ingest_credentials")
        checked = tuple(self._component_checked(name) for name in required)
        if any(
            was_checked and not self.components[name].ok
            for name, was_checked in zip(required, checked, strict=True)
        ):
            return False
        if all(checked):
            return True
        return None

    @property
    def retrieve_ready(self) -> bool:
        return self.core_ready and self.components["retrieve_credentials"].ok

    @property
    def ingest_ready(self) -> bool:
        return (
            self.core_ready
            and self.components["redis"].ok
            and self.components["minio"].ok
            and self.components["ingest_credentials"].ok
        )

    @property
    def answer_ready(self) -> bool:
        return self.core_ready and self.answer_configured


class ReadinessProvider(Protocol):
    async def snapshot(
        self,
        scope: ReadinessScope = ReadinessScope.ALL,
    ) -> ReadinessSnapshot: ...
