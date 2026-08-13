import asyncio
import base64
import binascii
import hashlib
import hmac
import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, cast
from uuid import UUID, uuid4

from pydantic import SecretStr, ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from rag_service.api.cursors import CursorPosition, decode_cursor, encode_cursor
from rag_service.api.errors import BusinessError
from rag_service.api.etags import require_matching_etag
from rag_service.api.middleware import is_valid_request_id
from rag_service.api.validation import validate_idempotency_key
from rag_service.auth.policies import AdminPrincipal
from rag_service.config import Settings
from rag_service.db.models.auth import AuditEvent, IdempotencyRecord
from rag_service.db.models.providers import ProviderCredential
from rag_service.providers.credentials import (
    EncryptedProviderCredential,
    ProviderCredentialKeyring,
    ProviderCredentialUnavailableError,
)
from rag_service.providers.network_policy import (
    CanonicalProviderEndpoint,
    ProviderEndpointPolicy,
    ProviderNetworkPolicyError,
    ResolvedProviderEndpoint,
)
from rag_service.providers.repositories import (
    AdminActorRecord,
    ModelProfileRecord,
    ModelProfileRepository,
    ProviderConfigRecord,
    ProviderConfigRepository,
    ProviderConfigSecretSourceRecord,
    ProviderCredentialRecord,
    ProviderRepositories,
    sqlalchemy_provider_repositories,
)
from rag_service.providers.schemas import (
    InternalSafeModelProfile,
    ModelProfileCreate,
    ModelProfileCreateResult,
    ModelProfilePage,
    ModelProfilePatch,
    ProviderConfigCreate,
    ProviderConfigCreateResult,
    ProviderConfigPage,
    ProviderConfigPatch,
    ProviderCredentialCreate,
    ProviderCredentialCreateResult,
    ProviderCredentialPage,
    ProviderCredentialPatch,
    SafeModelProfile,
    SafeProviderConfig,
    SafeProviderCredential,
)

type RepositoryFactory = Callable[[AsyncSession], ProviderRepositories]
type Clock = Callable[[], datetime]
type KeyringFactory = Callable[[], ProviderCredentialKeyring]
type EndpointPolicyFactory = Callable[[], ProviderEndpointPolicy]

# Capabilities the model-profile API will hand back. `chat` rows can exist in the
# database but are not part of this service's job, so they stay invisible here
# rather than being returned by a read that callers would then try to use.
_EXPOSED_CAPABILITIES = frozenset({"embedding", "rerank"})
_CREATE_OPERATION = "provider_credential.create"
_IDEMPOTENCY_UNIQUE_CONSTRAINT = "uq_idempotency_actor_operation_key"
_NAME_UNIQUE_CONSTRAINT = "uq_provider_credentials_name"
_MAX_RETAINED_EXCEPTION_NODES = 32
_HARD_MAX_PAGE_SIZE = 100
_MAX_KEY_VERSION_LENGTH = 64
_CONFIG_CREATE_OPERATION = "provider_config.create"
_CONFIG_NAME_UNIQUE_CONSTRAINT = "uq_provider_configs_name"
_PROFILE_CREATE_OPERATION = "model_profile.create"
_PROFILE_NAME_UNIQUE_CONSTRAINT = "uq_model_profiles_name"
_EMBEDDING_PROVIDER_TYPES = frozenset({"openai_compatible", "openrouter"})


class SqlAlchemyProviderCredentialReader:
    """Runtime-only reader for the current encrypted value behind a stable credential ID."""

    __slots__ = ("_session_factory",)

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession] | Callable[[], AsyncSession],
    ) -> None:
        if not callable(session_factory):
            raise ProviderCredentialUnavailableError from None
        self._session_factory = session_factory

    def __repr__(self) -> str:
        return "SqlAlchemyProviderCredentialReader(<redacted>)"

    async def get_encrypted(
        self,
        credential_id: UUID,
    ) -> EncryptedProviderCredential | None:
        statement: object | None = None
        result: Any = None
        row: object | None = None
        encrypted: EncryptedProviderCredential | None = None
        cancelled = False
        failed = False
        try:
            if type(credential_id) is not UUID:
                raise ValueError
            statement = select(
                ProviderCredential.ciphertext,
                ProviderCredential.nonce,
                ProviderCredential.key_version,
                ProviderCredential.algorithm,
            ).where(ProviderCredential.id == credential_id)
            async with self._session_factory() as session:
                if not callable(getattr(session, "execute", None)):
                    raise ValueError
                result = await session.execute(statement)
                row = result.one_or_none()
                if row is None:
                    return None
                ciphertext, nonce, key_version, algorithm = cast(
                    tuple[bytes, bytes, str, str],
                    row,
                )
                if (
                    type(ciphertext) is not bytes
                    or len(ciphertext) < 16
                    or type(nonce) is not bytes
                    or len(nonce) != 12
                    or type(key_version) is not str
                    or not key_version
                    or algorithm != "AES-256-GCM"
                ):
                    raise ValueError
                encrypted = EncryptedProviderCredential(
                    ciphertext=bytes(memoryview(ciphertext)),
                    nonce=bytes(memoryview(nonce)),
                    key_version=str(key_version),
                    algorithm=str(algorithm),
                )
            return encrypted
        except asyncio.CancelledError:
            cancelled = True
        except Exception:
            failed = True
        finally:
            statement = None
            result = None
            row = None
            encrypted = None
        if cancelled:
            raise asyncio.CancelledError from None
        if failed:
            raise ProviderCredentialUnavailableError from None
        raise AssertionError("unreachable")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _invalid_api_key_error() -> BusinessError:
    return BusinessError(401, "INVALID_API_KEY", "Invalid API key")


def _not_found_error() -> BusinessError:
    return BusinessError(404, "RESOURCE_NOT_FOUND", "Resource not found")


def _name_conflict_error() -> BusinessError:
    return BusinessError(409, "RESOURCE_ALREADY_EXISTS", "Resource already exists")


def _idempotency_reused_error() -> BusinessError:
    return BusinessError(
        409,
        "IDEMPOTENCY_KEY_REUSED",
        "Idempotency key was already used for a different request",
    )


def _validation_error() -> BusinessError:
    return BusinessError(422, "VALIDATION_ERROR", "Invalid provider credential request")


def _config_validation_error() -> BusinessError:
    return BusinessError(422, "VALIDATION_ERROR", "Invalid provider configuration request")


def _profile_validation_error() -> BusinessError:
    return BusinessError(422, "VALIDATION_ERROR", "Invalid model profile request")


def _provider_config_disabled_error() -> BusinessError:
    return BusinessError(409, "PROVIDER_CONFIG_DISABLED", "Provider configuration is disabled")


def _immutable_index_configuration_error() -> BusinessError:
    return BusinessError(
        409,
        "IMMUTABLE_INDEX_CONFIGURATION",
        "Index configuration is immutable after generation creation",
    )


def _endpoint_rejected_error() -> BusinessError:
    return BusinessError(422, "PROVIDER_ENDPOINT_REJECTED", "Provider endpoint rejected")


def _precondition_required_error() -> BusinessError:
    return BusinessError(428, "PRECONDITION_REQUIRED", "If-Match is required")


def _precondition_failed_error() -> BusinessError:
    return BusinessError(412, "PRECONDITION_FAILED", "Precondition failed")


def _credential_unavailable_error() -> BusinessError:
    return BusinessError(
        503,
        "PROVIDER_CREDENTIAL_UNAVAILABLE",
        "Provider credential unavailable",
    )


def _internal_error() -> BusinessError:
    return BusinessError(500, "INTERNAL_ERROR", "Internal server error")


def _require_uuid(value: object) -> UUID:
    if type(value) is not UUID:
        raise _validation_error()
    return value


def _require_request_id(value: object, max_length: int) -> str:
    if not is_valid_request_id(value, max_length):
        raise _validation_error()
    return value


def _is_currently_active(row: AdminActorRecord, now: datetime) -> bool:
    if row.status != "active" or row.revoked_at is not None:
        return False
    if row.not_before is not None and row.not_before > now:
        return False
    return row.expires_at is None or row.expires_at > now


def provider_credential_etag(resource_id: UUID, revision: int) -> str:
    if type(resource_id) is not UUID or type(revision) is not int or revision <= 0:
        raise ValueError("provider credential ETag inputs are invalid")
    return f'"provider-credential:{resource_id}:r{revision}"'


def provider_config_etag(resource_id: UUID, revision: int) -> str:
    if type(resource_id) is not UUID or type(revision) is not int or revision <= 0:
        raise ValueError("provider config ETag inputs are invalid")
    return f'"provider-config:{resource_id}:r{revision}"'


def model_profile_etag(resource_id: UUID, revision: int) -> str:
    if type(resource_id) is not UUID or type(revision) is not int or revision <= 0:
        raise ValueError("model profile ETag inputs are invalid")
    return f'"model-profile:{resource_id}:r{revision}"'


_ATTRIBUTE_ACCESS_FAILED = object()


def _protected_attribute(value: object, name: str) -> object:
    try:
        return getattr(value, name, None)
    except BaseException:
        return _ATTRIBUTE_ACCESS_FAILED


def _constraint_name(error: IntegrityError) -> str | None:
    pending: list[BaseException] = [error]
    visited: set[int] = set()
    while pending and len(visited) < _MAX_RETAINED_EXCEPTION_NODES:
        current = pending.pop()
        identity = id(current)
        if identity in visited:
            continue
        visited.add(identity)
        name = _protected_attribute(current, "constraint_name")
        if name is _ATTRIBUTE_ACCESS_FAILED:
            return None
        if isinstance(name, str):
            return name
        diagnostic = _protected_attribute(current, "diag")
        if diagnostic is _ATTRIBUTE_ACCESS_FAILED:
            return None
        name = _protected_attribute(diagnostic, "constraint_name")
        if name is _ATTRIBUTE_ACCESS_FAILED:
            return None
        if isinstance(name, str):
            return name
        original = _protected_attribute(current, "orig")
        if original is _ATTRIBUTE_ACCESS_FAILED:
            return None
        if isinstance(original, BaseException):
            pending.append(original)
        cause = _protected_attribute(current, "__cause__")
        context = _protected_attribute(current, "__context__")
        if cause is _ATTRIBUTE_ACCESS_FAILED or context is _ATTRIBUTE_ACCESS_FAILED:
            return None
        for nested_error in (cause, context):
            if nested_error is not None:
                if not isinstance(nested_error, BaseException):
                    return None
                pending.append(nested_error)
    return None


def _redacted_encrypted() -> EncryptedProviderCredential:
    return EncryptedProviderCredential(
        ciphertext=b"",
        nonce=b"",
        key_version="<redacted>",
        algorithm="<redacted>",
    )


def _redacted_create() -> ProviderCredentialCreate:
    return ProviderCredentialCreate(name="<redacted>", secret=SecretStr("<redacted>"))


def _redacted_patch() -> ProviderCredentialPatch:
    return ProviderCredentialPatch(name="<redacted>")


def _redacted_config_create() -> ProviderConfigCreate:
    return ProviderConfigCreate(
        name="<redacted>",
        provider_type="openai_compatible",
        base_url="https://redacted.invalid",
        credential_id=UUID(int=0),
        timeout_seconds=Decimal("1"),
        max_concurrency=1,
        requests_per_minute=1,
    )


def _redacted_config_patch() -> ProviderConfigPatch:
    return ProviderConfigPatch(enabled=False)


def _redacted_profile_create() -> ModelProfileCreate:
    return ModelProfileCreate(
        name="<redacted>",
        capability="embedding",
        provider_config_id=UUID(int=0),
        model_name="<redacted>",
        dimension=1,
        max_input_tokens=1,
        batch_size=1,
        timeout_seconds=Decimal("1"),
        vector_config={},
        enabled=False,
    )


def _redacted_profile_patch() -> ModelProfilePatch:
    return ModelProfilePatch(enabled=False)


@dataclass(frozen=True, slots=True, repr=False)
class CredentialProviderSecretSource:
    credential_id: UUID

    def __repr__(self) -> str:
        return "CredentialProviderSecretSource(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class LegacyProviderSecretSource:
    secret_ref: str

    def __repr__(self) -> str:
        return "LegacyProviderSecretSource(<redacted>)"


type ProviderSecretSource = CredentialProviderSecretSource | LegacyProviderSecretSource


def select_provider_config_secret_source(
    record: ProviderConfigSecretSourceRecord,
) -> ProviderSecretSource:
    if type(record) is not ProviderConfigSecretSourceRecord:
        raise _internal_error()
    credential_id = record.credential_id
    secret_ref = record.secret_ref
    record = cast(ProviderConfigSecretSourceRecord, None)
    if (credential_id is None) == (secret_ref is None):
        secret_ref = None
        raise _internal_error()
    if credential_id is not None:
        secret_ref = None
        return CredentialProviderSecretSource(credential_id=credential_id)
    if type(secret_ref) is not str or not secret_ref:
        secret_ref = None
        raise _internal_error()
    result = LegacyProviderSecretSource(secret_ref=secret_ref)
    secret_ref = None
    return result


def provider_credential_keyring_from_settings(
    settings: Settings,
) -> ProviderCredentialKeyring:
    secret_json = "<redacted>"
    raw_keyring: object = {}
    encoded_key = "<redacted>"
    decoded_key = b""
    keys: dict[str, bytes] = {}
    keyring: ProviderCredentialKeyring | None = None
    failed = False
    try:
        secret_json = settings.provider_credential_keyring.get_secret_value()
        raw_keyring = json.loads(secret_json)
        if not isinstance(raw_keyring, dict):
            raise ValueError
        active_key_version = settings.provider_credential_active_key_version
        if (
            type(active_key_version) is not str
            or not active_key_version
            or len(active_key_version) > _MAX_KEY_VERSION_LENGTH
        ):
            raise ValueError
        for version, encoded_key in raw_keyring.items():
            if (
                type(version) is not str
                or not version
                or len(version) > _MAX_KEY_VERSION_LENGTH
                or type(encoded_key) is not str
            ):
                raise ValueError
            decoded_key = base64.b64decode(encoded_key, validate=True)
            keys[version] = decoded_key
        keyring = ProviderCredentialKeyring(
            keys=keys,
            active_key_version=active_key_version,
        )
    except (ValueError, TypeError, json.JSONDecodeError, binascii.Error):
        failed = True
    except ProviderCredentialUnavailableError:
        failed = True
    finally:
        secret_json = "<redacted>"
        raw_keyring = {}
        encoded_key = "<redacted>"
        decoded_key = b""
        keys = {}
    if failed or keyring is None:
        raise _credential_unavailable_error() from None
    return keyring


def _create_fingerprint(command: ProviderCredentialCreate, hmac_secret: SecretStr) -> bytes:
    hmac_key = b""
    name = b""
    secret = b""
    payload = b""
    try:
        hmac_key = hmac_secret.get_secret_value().encode("utf-8")
        name = command.name.encode("utf-8")
        secret = command.secret.get_secret_value().encode("utf-8")
        payload = b"provider-credential.create:v1\x00" + b"".join(
            (
                len(name).to_bytes(4, "big"),
                name,
                len(secret).to_bytes(4, "big"),
                secret,
            )
        )
        return hmac.digest(hmac_key, payload, hashlib.sha256)
    finally:
        hmac_key = b""
        name = b""
        secret = b""
        payload = b""
        command = _redacted_create()
        hmac_secret = SecretStr("<redacted>")


def provider_endpoint_policy_from_settings(settings: Settings) -> ProviderEndpointPolicy:
    try:
        return ProviderEndpointPolicy(
            environment=settings.environment,
            allow_private_targets=settings.provider_allow_private_targets,
        )
    except ProviderNetworkPolicyError:
        raise _endpoint_rejected_error() from None


def _config_create_fingerprint(
    command: ProviderConfigCreate,
    endpoint: CanonicalProviderEndpoint,
    headers: dict[str, str],
    hmac_secret: SecretStr,
) -> bytes:
    hmac_key = b""
    payload = b""
    document: dict[str, object] = {}
    canonical_document: object = None
    try:
        hmac_key = hmac_secret.get_secret_value().encode("utf-8")
        document = {
            "schema": "provider-config.create:v1",
            "name": command.name,
            "provider_type": command.provider_type,
            "base_url": endpoint.url,
            "credential_id": str(command.credential_id),
            "default_headers": headers,
            "routing_options": command.routing_options,
            "timeout_seconds": command.timeout_seconds,
            "max_concurrency": command.max_concurrency,
            "requests_per_minute": command.requests_per_minute,
            "enabled": command.enabled,
        }
        canonical_document = _canonical_fingerprint_value(document)
        payload = json.dumps(
            canonical_document,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hmac.digest(hmac_key, payload, hashlib.sha256)
    finally:
        hmac_key = b""
        payload = b""
        document.clear()
        canonical_document = None
        headers.clear()
        command = _redacted_config_create()
        endpoint = CanonicalProviderEndpoint(
            url="https://redacted.invalid",
            hostname="redacted.invalid",
            port=443,
            path="",
        )
        hmac_secret = SecretStr("<redacted>")


def _canonical_fingerprint_number(value: int | float | Decimal) -> str:
    number = value if type(value) is Decimal else Decimal(str(value))
    if not number.is_finite():
        raise ValueError("fingerprint number must be finite")
    if number == 0:
        return "0"
    decimal_tuple = number.as_tuple()
    digits = list(decimal_tuple.digits)
    exponent = cast(int, decimal_tuple.exponent)
    while digits[-1] == 0:
        digits.pop()
        exponent += 1
    coefficient = "".join(str(digit) for digit in digits)
    sign = "-" if decimal_tuple.sign else ""
    return f"{sign}{coefficient}e{exponent}"


def _canonical_fingerprint_value(value: object) -> object:
    if value is None:
        return ["null"]
    if type(value) is bool:
        return ["boolean", value]
    if type(value) is str:
        return ["string", value]
    if type(value) in {int, float, Decimal}:
        return ["number", _canonical_fingerprint_number(cast(int | float | Decimal, value))]
    if type(value) is list:
        return ["array", [_canonical_fingerprint_value(item) for item in value]]
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise ValueError("fingerprint object keys must be strings")
        return [
            "object",
            [
                [key, _canonical_fingerprint_value(value[key])]
                for key in sorted(cast(dict[str, object], value))
            ],
        ]
    raise ValueError("fingerprint value is unsupported")


def _profile_create_fingerprint(
    command: ModelProfileCreate,
    hmac_secret: SecretStr,
) -> bytes:
    hmac_key = b""
    payload = b""
    document: dict[str, object] = {}
    canonical_document: object = None
    try:
        hmac_key = hmac_secret.get_secret_value().encode("utf-8")
        document = {
            "schema": "model-profile.create:v1",
            "name": command.name,
            "capability": command.capability,
            "provider_config_id": str(command.provider_config_id),
            "model_name": command.model_name,
            "dimension": command.dimension,
            "max_input_tokens": command.max_input_tokens,
            "batch_size": command.batch_size,
            "timeout_seconds": command.timeout_seconds,
            "vector_config": command.vector_config,
            "enabled": command.enabled,
        }
        canonical_document = _canonical_fingerprint_value(document)
        payload = json.dumps(
            canonical_document,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hmac.digest(hmac_key, payload, hashlib.sha256)
    finally:
        hmac_key = b""
        payload = b""
        document.clear()
        canonical_document = None
        command = _redacted_profile_create()
        hmac_secret = SecretStr("<redacted>")


def _provider_semantics_changed(
    row: ProviderConfigRecord,
    command: ProviderConfigPatch,
    fields: set[str],
    canonical_endpoint: CanonicalProviderEndpoint | None,
    canonical_headers: dict[str, str] | None,
) -> bool:
    if "provider_type" in fields and command.provider_type != row.provider_type:
        return True
    if "base_url" in fields:
        if canonical_endpoint is None:
            raise _config_validation_error()
        if canonical_endpoint.url != row.base_url:
            return True
    if "default_headers" in fields:
        if canonical_headers is None:
            raise _config_validation_error()
        if canonical_headers != row.default_headers:
            return True
    return "routing_options" in fields and command.routing_options != row.routing_options


class _ProviderCredentialServiceBase:
    def __init__(
        self,
        *,
        session: AsyncSession,
        settings: Settings,
        keyring_factory: KeyringFactory,
        repository_factory: RepositoryFactory = sqlalchemy_provider_repositories,
        clock: Clock = _utc_now,
    ) -> None:
        self._session = session
        self._settings = settings
        self._keyring_factory = keyring_factory
        self._repository_factory = repository_factory
        self._clock = clock

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
            raise _internal_error()
        return value

    async def _require_admin_actor(
        self,
        repositories: ProviderRepositories,
        actor: AdminPrincipal,
        now: datetime,
    ) -> AdminActorRecord:
        if type(actor) is not AdminPrincipal:
            raise _invalid_api_key_error()
        row = await repositories.admins.get_for_update(actor.key_id)
        if (
            row is None
            or row.public_id != actor.public_id
            or row.key_type != "admin"
            or not _is_currently_active(row, now)
        ):
            raise _invalid_api_key_error()
        return row

    def _safe(self, row: ProviderCredentialRecord) -> SafeProviderCredential:
        safe: SafeProviderCredential | None = None
        failed = False
        try:
            safe = SafeProviderCredential(
                id=row.id,
                name=row.name,
                credential_configured=True,
                key_version=row.key_version,
                resource_revision=row.resource_revision,
                created_at=row.created_at,
                updated_at=row.updated_at,
                rotated_at=row.rotated_at,
            )
        except (AttributeError, TypeError, ValueError, ValidationError):
            failed = True
        finally:
            row = cast(ProviderCredentialRecord, None)
        if failed or safe is None:
            raise _internal_error() from None
        return safe

    def _encrypt_secret(
        self,
        credential_id: UUID,
        secret: SecretStr,
    ) -> EncryptedProviderCredential:
        plaintext = b""
        keyring: ProviderCredentialKeyring | None = None
        try:
            plaintext = secret.get_secret_value().encode("utf-8")
            keyring = self._keyring_factory()
            if type(keyring) is not ProviderCredentialKeyring:
                raise ProviderCredentialUnavailableError
            return keyring.encrypt(credential_id, plaintext)
        finally:
            plaintext = b""
            keyring = None
            secret = SecretStr("<redacted>")

    async def _replay_create(
        self,
        repositories: ProviderRepositories,
        fingerprint: bytes,
        record: IdempotencyRecord,
    ) -> ProviderCredentialCreateResult:
        raw_response: object | None = None
        response: dict[str, object] = {}
        safe: SafeProviderCredential | None = None
        result: ProviderCredentialCreateResult | None = None
        invalid_snapshot = False
        try:
            if not hmac.compare_digest(record.request_fingerprint, fingerprint):
                raise _idempotency_reused_error()
            if record.result_resource_type != "provider_credential" or record.http_status != 201:
                raise _internal_error()
            raw_response = await repositories.audits.get_created_response(
                record.actor_key_id,
                record.result_resource_id,
            )
            if type(raw_response) is not dict:
                raise ValueError
            response = deepcopy(raw_response)
            raw_response = None
            safe = SafeProviderCredential.model_validate(response)
            if safe.id != record.result_resource_id:
                raise _internal_error()
            result = ProviderCredentialCreateResult(credential=safe, created=False)
            return result
        except BusinessError:
            raise
        except (TypeError, ValueError, ValidationError):
            invalid_snapshot = True
        finally:
            raw_response = None
            response.clear()
            safe = None
            result = None
            record = cast(IdempotencyRecord, None)
            fingerprint = b""
        if invalid_snapshot:
            raise _internal_error() from None
        raise AssertionError("unreachable")

    async def create_credential(
        self,
        command: ProviderCredentialCreate,
        *,
        actor: AdminPrincipal,
        request_id: str,
        idempotency_key: str,
    ) -> ProviderCredentialCreateResult:
        if type(command) is not ProviderCredentialCreate:
            raise _validation_error()
        snapshot = _redacted_create()
        validated_key = "<redacted>"
        fingerprint = b""
        encrypted = _redacted_encrypted()
        repositories: ProviderRepositories | None = None
        existing: IdempotencyRecord | None = None
        winner: IdempotencyRecord | None = None
        row: ProviderCredentialRecord | None = None
        safe: SafeProviderCredential | None = None
        result: ProviderCredentialCreateResult | None = None
        credential_id: UUID | None = None
        conflict: str | None = None
        integrity_failed = False
        cancelled = False
        provider_unavailable = False
        internal_failure = False
        try:
            _require_request_id(request_id, self._settings.max_request_id_length)
            validated_key = validate_idempotency_key(
                idempotency_key,
                self._settings.max_idempotency_key_length,
            )
            snapshot = command.model_copy(deep=True)
            fingerprint = _create_fingerprint(
                snapshot,
                self._settings.admin_key_hmac_secret,
            )
            async with self._session.begin():
                repositories = self._repository_factory(self._session)
                existing = await repositories.idempotency.get(
                    actor.key_id,
                    _CREATE_OPERATION,
                    validated_key,
                )
                if existing is not None:
                    await self._require_admin_actor(repositories, actor, self._now())
                    result = await self._replay_create(repositories, fingerprint, existing)
                    return result

                credential_id = uuid4()
                try:
                    async with self._session.begin_nested():
                        await self._require_admin_actor(repositories, actor, self._now())
                        winner = await repositories.idempotency.get(
                            actor.key_id,
                            _CREATE_OPERATION,
                            validated_key,
                        )
                        if winner is not None:
                            result = await self._replay_create(
                                repositories,
                                fingerprint,
                                winner,
                            )
                            return result
                        encrypted = self._encrypt_secret(credential_id, snapshot.secret)
                        row = await repositories.credentials.add_encrypted(
                            credential_id,
                            snapshot.name,
                            encrypted,
                        )
                        safe = self._safe(row)
                        await repositories.idempotency.add(
                            IdempotencyRecord(
                                id=uuid4(),
                                actor_key_id=actor.key_id,
                                operation=_CREATE_OPERATION,
                                idempotency_key=validated_key,
                                request_fingerprint=fingerprint,
                                result_resource_type="provider_credential",
                                result_resource_id=credential_id,
                                http_status=201,
                            )
                        )
                        await repositories.audits.add(
                            AuditEvent(
                                id=uuid4(),
                                request_id=request_id,
                                actor_api_key_id=actor.key_id,
                                actor_kind="admin_key",
                                action="provider_credential.created",
                                target_type="provider_credential",
                                target_id=credential_id,
                                metadata_=safe.model_dump(mode="json"),
                            )
                        )
                except IntegrityError as source:
                    integrity_failed = True
                    conflict = _constraint_name(source)
                finally:
                    encrypted = _redacted_encrypted()

                if integrity_failed:
                    if conflict == _IDEMPOTENCY_UNIQUE_CONSTRAINT:
                        winner = await repositories.idempotency.get(
                            actor.key_id,
                            _CREATE_OPERATION,
                            validated_key,
                        )
                        if winner is None:
                            raise _internal_error()
                        result = await self._replay_create(repositories, fingerprint, winner)
                        return result
                    if conflict == _NAME_UNIQUE_CONSTRAINT:
                        raise _name_conflict_error()
                    raise _internal_error()
                if row is None or safe is None:
                    raise _internal_error()
                result = ProviderCredentialCreateResult(
                    credential=safe,
                    created=True,
                )
                return result
        except asyncio.CancelledError:
            cancelled = True
        except ProviderCredentialUnavailableError:
            provider_unavailable = True
        except BusinessError:
            raise
        except Exception:
            internal_failure = True
        finally:
            command = _redacted_create()
            snapshot = command
            actor = cast(AdminPrincipal, None)
            request_id = "<redacted>"
            idempotency_key = "<redacted>"
            validated_key = "<redacted>"
            fingerprint = b""
            encrypted = _redacted_encrypted()
            repositories = None
            existing = None
            winner = None
            row = None
            safe = None
            result = None
            credential_id = None
            conflict = None
        if cancelled:
            raise asyncio.CancelledError() from None
        if provider_unavailable:
            raise _credential_unavailable_error() from None
        if internal_failure:
            raise _internal_error() from None
        raise AssertionError("unreachable")


class ProviderConfigService:
    def __init__(
        self,
        *,
        session: AsyncSession,
        settings: Settings,
        endpoint_policy_factory: EndpointPolicyFactory,
        repository_factory: RepositoryFactory = sqlalchemy_provider_repositories,
        clock: Clock = _utc_now,
    ) -> None:
        self._session = session
        self._settings = settings
        self._endpoint_policy_factory = endpoint_policy_factory
        self._repository_factory = repository_factory
        self._clock = clock

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
            raise _internal_error()
        return value

    def _configs(self, repositories: ProviderRepositories) -> ProviderConfigRepository:
        if repositories.configs is None:
            raise _internal_error()
        return repositories.configs

    def _profiles(self, repositories: ProviderRepositories) -> ModelProfileRepository | None:
        return repositories.profiles

    async def _require_admin_actor(
        self,
        repositories: ProviderRepositories,
        actor: AdminPrincipal,
        now: datetime,
    ) -> AdminActorRecord:
        if type(actor) is not AdminPrincipal:
            raise _invalid_api_key_error()
        row = await repositories.admins.get_for_update(actor.key_id)
        if (
            row is None
            or row.public_id != actor.public_id
            or row.key_type != "admin"
            or not _is_currently_active(row, now)
        ):
            raise _invalid_api_key_error()
        return row

    def _safe(self, row: ProviderConfigRecord) -> SafeProviderConfig:
        safe: SafeProviderConfig | None = None
        failed = False
        try:
            safe = SafeProviderConfig(
                id=row.id,
                name=row.name,
                provider_type=row.provider_type,
                base_url=row.base_url,
                credential_id=row.credential_id,
                default_headers=deepcopy(row.default_headers),
                routing_options=deepcopy(row.routing_options),
                timeout_seconds=row.timeout_seconds,
                max_concurrency=row.max_concurrency,
                requests_per_minute=row.requests_per_minute,
                enabled=row.enabled,
                resource_revision=row.resource_revision,
                endpoint_policy_version=row.endpoint_policy_version,
                endpoint_validated_at=row.endpoint_validated_at,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
        except (AttributeError, TypeError, ValueError, ValidationError):
            failed = True
        finally:
            row = cast(ProviderConfigRecord, None)
        if failed or safe is None:
            raise _internal_error() from None
        return safe

    def _policy(self) -> ProviderEndpointPolicy:
        policy: ProviderEndpointPolicy | None = None
        failed = False
        try:
            policy = self._endpoint_policy_factory()
            for method_name in ("validate_url", "validate_headers", "validate_for_persistence"):
                if not callable(getattr(policy, method_name, None)):
                    raise TypeError
        except ProviderNetworkPolicyError:
            raise _endpoint_rejected_error() from None
        except BusinessError:
            raise
        except Exception:
            failed = True
        if failed or policy is None:
            raise _internal_error() from None
        return policy

    def _pure_configuration(
        self,
        policy: ProviderEndpointPolicy,
        base_url: str,
        headers: dict[str, str],
    ) -> tuple[CanonicalProviderEndpoint, dict[str, str]]:
        endpoint: CanonicalProviderEndpoint | None = None
        canonical_headers: dict[str, str] = {}
        failed = False
        try:
            endpoint = policy.validate_url(base_url)
            canonical_headers = dict(policy.validate_headers(headers))
        except ProviderNetworkPolicyError:
            raise _endpoint_rejected_error() from None
        except Exception:
            failed = True
        finally:
            base_url = "<redacted>"
            headers.clear()
        if failed or endpoint is None:
            canonical_headers.clear()
            raise _internal_error() from None
        return endpoint, canonical_headers

    async def _validate_for_persistence(
        self,
        policy: ProviderEndpointPolicy,
        raw_url: str,
    ) -> ResolvedProviderEndpoint:
        resolved: ResolvedProviderEndpoint | None = None
        cancelled = False
        endpoint_rejected = False
        failed = False
        try:
            resolved = await asyncio.to_thread(policy.validate_for_persistence, raw_url)
        except asyncio.CancelledError:
            cancelled = True
        except ProviderNetworkPolicyError:
            endpoint_rejected = True
        except Exception:
            failed = True
        finally:
            raw_url = "<redacted>"
        if cancelled:
            raise asyncio.CancelledError() from None
        if endpoint_rejected:
            raise _endpoint_rejected_error() from None
        if failed or resolved is None:
            raise _internal_error() from None
        return resolved

    async def _replay_create(
        self,
        repositories: ProviderRepositories,
        fingerprint: bytes,
        record: IdempotencyRecord,
    ) -> ProviderConfigCreateResult:
        raw_response: object | None = None
        response: dict[str, object] = {}
        safe: SafeProviderConfig | None = None
        result: ProviderConfigCreateResult | None = None
        invalid_snapshot = False
        try:
            if not hmac.compare_digest(record.request_fingerprint, fingerprint):
                raise _idempotency_reused_error()
            if record.result_resource_type != "provider_config" or record.http_status != 201:
                raise _internal_error()
            raw_response = await repositories.audits.get_provider_config_created_response(
                record.actor_key_id,
                record.result_resource_id,
            )
            if type(raw_response) is not dict:
                raise ValueError
            response = deepcopy(raw_response)
            raw_response = None
            safe = SafeProviderConfig.model_validate(response)
            if safe.id != record.result_resource_id:
                raise _internal_error()
            result = ProviderConfigCreateResult(provider_config=safe, created=False)
            return result
        except BusinessError:
            raise
        except (TypeError, ValueError, ValidationError):
            invalid_snapshot = True
        finally:
            raw_response = None
            response.clear()
            safe = None
            result = None
            record = cast(IdempotencyRecord, None)
            fingerprint = b""
        if invalid_snapshot:
            raise _internal_error() from None
        raise AssertionError("unreachable")

    async def create_provider_config(
        self,
        command: ProviderConfigCreate,
        *,
        actor: AdminPrincipal,
        request_id: str,
        idempotency_key: str,
    ) -> ProviderConfigCreateResult:
        if type(command) is not ProviderConfigCreate:
            raise _config_validation_error()
        snapshot = _redacted_config_create()
        validated_key = "<redacted>"
        fingerprint = b""
        canonical_headers: dict[str, str] = {}
        policy: ProviderEndpointPolicy | None = None
        endpoint: CanonicalProviderEndpoint | None = None
        resolved: ResolvedProviderEndpoint | None = None
        repositories: ProviderRepositories | None = None
        existing: IdempotencyRecord | None = None
        winner: IdempotencyRecord | None = None
        row: ProviderConfigRecord | None = None
        safe: SafeProviderConfig | None = None
        result: ProviderConfigCreateResult | None = None
        provider_config_id: UUID | None = None
        validated_at: datetime | None = None
        conflict: str | None = None
        integrity_failed = False
        cancelled = False
        endpoint_rejected = False
        internal_failure = False
        try:
            _require_request_id(request_id, self._settings.max_request_id_length)
            validated_key = validate_idempotency_key(
                idempotency_key,
                self._settings.max_idempotency_key_length,
            )
            snapshot = command.model_copy(deep=True)
            policy = self._policy()
            endpoint, canonical_headers = self._pure_configuration(
                policy,
                snapshot.base_url,
                dict(snapshot.default_headers),
            )
            fingerprint = _config_create_fingerprint(
                snapshot,
                endpoint,
                dict(canonical_headers),
                self._settings.admin_key_hmac_secret,
            )
            async with self._session.begin():
                repositories = self._repository_factory(self._session)
                existing = await repositories.idempotency.get(
                    actor.key_id,
                    _CONFIG_CREATE_OPERATION,
                    validated_key,
                )
                await self._require_admin_actor(repositories, actor, self._now())
                if existing is not None:
                    result = await self._replay_create(repositories, fingerprint, existing)
                    return result
                credential = await repositories.credentials.get_safe(snapshot.credential_id)
                if credential is None:
                    raise _not_found_error()

            resolved = await self._validate_for_persistence(policy, snapshot.base_url)
            if resolved.endpoint != endpoint:
                raise _endpoint_rejected_error()
            provider_config_id = uuid4()
            validated_at = self._now()

            async with self._session.begin():
                repositories = self._repository_factory(self._session)
                configs = self._configs(repositories)
                winner = await repositories.idempotency.get(
                    actor.key_id,
                    _CONFIG_CREATE_OPERATION,
                    validated_key,
                )
                await self._require_admin_actor(repositories, actor, self._now())
                if winner is None:
                    winner = await repositories.idempotency.get(
                        actor.key_id,
                        _CONFIG_CREATE_OPERATION,
                        validated_key,
                    )
                if winner is not None:
                    result = await self._replay_create(
                        repositories,
                        fingerprint,
                        winner,
                    )
                    return result
                credential = await repositories.credentials.get_safe(
                    snapshot.credential_id,
                    for_update=True,
                )
                if credential is None:
                    raise _not_found_error()
                if resolved.endpoint != endpoint:
                    raise _endpoint_rejected_error()
                try:
                    async with self._session.begin_nested():
                        row = await configs.add_validated(
                            provider_config_id,
                            name=snapshot.name,
                            provider_type=snapshot.provider_type,
                            base_url=resolved.endpoint.url,
                            credential_id=snapshot.credential_id,
                            default_headers=dict(canonical_headers),
                            routing_options=deepcopy(snapshot.routing_options),
                            timeout_seconds=snapshot.timeout_seconds,
                            max_concurrency=snapshot.max_concurrency,
                            requests_per_minute=snapshot.requests_per_minute,
                            enabled=snapshot.enabled,
                            endpoint_policy_version=resolved.endpoint.policy_version,
                            endpoint_validated_at=validated_at,
                        )
                        safe = self._safe(row)
                        await repositories.idempotency.add(
                            IdempotencyRecord(
                                id=uuid4(),
                                actor_key_id=actor.key_id,
                                operation=_CONFIG_CREATE_OPERATION,
                                idempotency_key=validated_key,
                                request_fingerprint=fingerprint,
                                result_resource_type="provider_config",
                                result_resource_id=provider_config_id,
                                http_status=201,
                            )
                        )
                        await repositories.audits.add(
                            AuditEvent(
                                id=uuid4(),
                                request_id=request_id,
                                actor_api_key_id=actor.key_id,
                                actor_kind="admin_key",
                                action="provider_config.created",
                                target_type="provider_config",
                                target_id=provider_config_id,
                                metadata_=safe.model_dump(mode="json"),
                            )
                        )
                except IntegrityError as source:
                    integrity_failed = True
                    conflict = _constraint_name(source)

                if integrity_failed:
                    if conflict == _IDEMPOTENCY_UNIQUE_CONSTRAINT:
                        winner = await repositories.idempotency.get(
                            actor.key_id,
                            _CONFIG_CREATE_OPERATION,
                            validated_key,
                        )
                        if winner is None:
                            raise _internal_error()
                        result = await self._replay_create(repositories, fingerprint, winner)
                        return result
                    if conflict == _CONFIG_NAME_UNIQUE_CONSTRAINT:
                        raise _name_conflict_error()
                    raise _internal_error()
                if row is None or safe is None:
                    raise _internal_error()
                result = ProviderConfigCreateResult(provider_config=safe, created=True)
                return result
        except asyncio.CancelledError:
            cancelled = True
        except ProviderNetworkPolicyError:
            endpoint_rejected = True
        except BusinessError:
            raise
        except Exception:
            internal_failure = True
        finally:
            command = _redacted_config_create()
            snapshot = command
            actor = cast(AdminPrincipal, None)
            request_id = "<redacted>"
            idempotency_key = "<redacted>"
            validated_key = "<redacted>"
            fingerprint = b""
            canonical_headers.clear()
            policy = None
            endpoint = None
            resolved = None
            repositories = None
            existing = None
            winner = None
            row = None
            safe = None
            result = None
            provider_config_id = None
            validated_at = None
            conflict = None
        if cancelled:
            raise asyncio.CancelledError() from None
        if endpoint_rejected:
            raise _endpoint_rejected_error() from None
        if internal_failure:
            raise _internal_error() from None
        raise AssertionError("unreachable")

    async def list_provider_configs(
        self,
        *,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> ProviderConfigPage:
        result: ProviderConfigPage | None = None
        failed = False
        try:
            position = None if cursor is None else decode_cursor(cursor)
            maximum = min(self._settings.max_page_size, _HARD_MAX_PAGE_SIZE)
            page_limit = min(self._settings.default_page_size, maximum) if limit is None else limit
            if type(page_limit) is not int or not 1 <= page_limit <= maximum:
                raise _config_validation_error()
            repositories = self._repository_factory(self._session)
            rows = await self._configs(repositories).list_safe(position, page_limit + 1)
            has_more = len(rows) > page_limit
            visible = rows[:page_limit]
            next_cursor = None
            if has_more and visible:
                last = visible[-1]
                next_cursor = encode_cursor(CursorPosition(created_at=last.created_at, id=last.id))
            result = ProviderConfigPage(
                items=tuple(self._safe(row) for row in visible),
                next_cursor=next_cursor,
            )
        except BusinessError:
            raise
        except Exception:
            failed = True
        if failed or result is None:
            raise _internal_error() from None
        return result

    async def get_provider_config(self, provider_config_id: UUID) -> SafeProviderConfig:
        identifier = _require_uuid(provider_config_id)
        result: SafeProviderConfig | None = None
        failed = False
        try:
            repositories = self._repository_factory(self._session)
            row = await self._configs(repositories).get_safe(identifier)
            if row is None:
                raise _not_found_error()
            result = self._safe(row)
        except BusinessError:
            raise
        except Exception:
            failed = True
        if failed or result is None:
            raise _internal_error() from None
        return result

    async def get_secret_source(self, provider_config_id: UUID) -> ProviderSecretSource:
        identifier = _require_uuid(provider_config_id)
        repositories: ProviderRepositories | None = None
        record: ProviderConfigSecretSourceRecord | None = None
        result: ProviderSecretSource | None = None
        cancelled = False
        failed = False
        try:
            repositories = self._repository_factory(self._session)
            record = await self._configs(repositories).get_secret_source(identifier)
            if record is None:
                raise _not_found_error()
            result = select_provider_config_secret_source(record)
            return result
        except asyncio.CancelledError:
            cancelled = True
        except BusinessError:
            raise
        except Exception:
            failed = True
        finally:
            repositories = None
            record = None
            result = None
        if cancelled:
            raise asyncio.CancelledError() from None
        if failed:
            raise _internal_error() from None
        raise AssertionError("unreachable")

    async def update_provider_config(
        self,
        provider_config_id: UUID,
        command: ProviderConfigPatch,
        *,
        actor: AdminPrincipal,
        request_id: str,
        expected_etag: str | None,
    ) -> SafeProviderConfig:
        identifier = _require_uuid(provider_config_id)
        if type(command) is not ProviderConfigPatch:
            raise _config_validation_error()
        if expected_etag is None:
            raise _precondition_required_error()
        snapshot = _redacted_config_patch()
        fields: set[str] = set()
        canonical_headers: dict[str, str] | None = None
        canonical_endpoint: CanonicalProviderEndpoint | None = None
        policy: ProviderEndpointPolicy | None = None
        resolved: ResolvedProviderEndpoint | None = None
        repositories: ProviderRepositories | None = None
        row: ProviderConfigRecord | None = None
        locked_row: ProviderConfigRecord | None = None
        updated: ProviderConfigRecord | None = None
        result: SafeProviderConfig | None = None
        values: dict[str, object] = {}
        target_credential_id: UUID | None = None
        locked_target_credential_id: UUID | None = None
        raw_url = "<redacted>"
        locked_raw_url = "<redacted>"
        migrating_legacy = False
        locked_migrating_legacy = False
        revalidate_endpoint = False
        locked_revalidate_endpoint = False
        semantic_changed = False
        locked_semantic_changed = False
        conflict: str | None = None
        integrity_failed = False
        cancelled = False
        endpoint_rejected = False
        internal_failure = False
        try:
            _require_request_id(request_id, self._settings.max_request_id_length)
            snapshot = command.model_copy(deep=True)
            fields = set(snapshot.model_fields_set)
            if "base_url" in fields or "default_headers" in fields:
                policy = self._policy()
            if "base_url" in fields:
                if snapshot.base_url is None or policy is None:
                    raise _config_validation_error()
                try:
                    canonical_endpoint = policy.validate_url(snapshot.base_url)
                except ProviderNetworkPolicyError:
                    raise _endpoint_rejected_error() from None
            if "default_headers" in fields:
                if snapshot.default_headers is None or policy is None:
                    raise _config_validation_error()
                try:
                    canonical_headers = dict(policy.validate_headers(snapshot.default_headers))
                except ProviderNetworkPolicyError:
                    raise _endpoint_rejected_error() from None

            async with self._session.begin():
                repositories = self._repository_factory(self._session)
                configs = self._configs(repositories)
                now = self._now()
                await self._require_admin_actor(repositories, actor, now)
                row = await configs.get_safe(identifier)
                if row is None:
                    raise _not_found_error()
                require_matching_etag(
                    expected_etag,
                    provider_config_etag(row.id, row.resource_revision),
                )

                if row.provider_type == "vendor_specific":
                    if fields != {"enabled"} or snapshot.enabled is not False:
                        raise _config_validation_error()
                else:
                    effective_provider_type = (
                        snapshot.provider_type if "provider_type" in fields else row.provider_type
                    )
                    effective_routing_options = (
                        snapshot.routing_options
                        if "routing_options" in fields
                        else row.routing_options
                    )
                    if effective_provider_type == "vendor_specific":
                        raise _config_validation_error()
                    if effective_provider_type == "openai_compatible" and effective_routing_options:
                        raise _config_validation_error()

                semantic_changed = _provider_semantics_changed(
                    row,
                    snapshot,
                    fields,
                    canonical_endpoint,
                    canonical_headers,
                )
                profiles = self._profiles(repositories)
                if (
                    semantic_changed
                    and profiles is not None
                    and await profiles.provider_config_is_referenced_by_generation(identifier)
                ):
                    raise _immutable_index_configuration_error()

                target_credential_id = (
                    snapshot.credential_id if "credential_id" in fields else row.credential_id
                )
                if "credential_id" in fields:
                    if target_credential_id is None:
                        raise _config_validation_error()
                    credential = await repositories.credentials.get_safe(target_credential_id)
                    if credential is None:
                        raise _not_found_error()

                migrating_legacy = row.credential_id is None and "credential_id" in fields
                if row.credential_id is None and "base_url" in fields and not migrating_legacy:
                    raise _config_validation_error()
                revalidate_endpoint = (
                    "base_url" in fields
                    and canonical_endpoint is not None
                    and canonical_endpoint.url != row.base_url
                ) or migrating_legacy
                if revalidate_endpoint:
                    if policy is None:
                        policy = self._policy()
                    selected_url = snapshot.base_url if "base_url" in fields else row.base_url
                    if selected_url is None:
                        raise _config_validation_error()
                    raw_url = selected_url

            if revalidate_endpoint:
                if policy is None:
                    raise _config_validation_error()
                resolved = await self._validate_for_persistence(policy, raw_url)
                if canonical_endpoint is not None and resolved.endpoint != canonical_endpoint:
                    raise _endpoint_rejected_error()

            async with self._session.begin():
                repositories = self._repository_factory(self._session)
                configs = self._configs(repositories)
                await self._require_admin_actor(repositories, actor, self._now())
                locked_row = await configs.get_safe(identifier, for_update=True)
                if locked_row is None:
                    raise _not_found_error()
                require_matching_etag(
                    expected_etag,
                    provider_config_etag(locked_row.id, locked_row.resource_revision),
                )

                if locked_row.provider_type == "vendor_specific":
                    if fields != {"enabled"} or snapshot.enabled is not False:
                        raise _config_validation_error()
                else:
                    effective_provider_type = (
                        snapshot.provider_type
                        if "provider_type" in fields
                        else locked_row.provider_type
                    )
                    effective_routing_options = (
                        snapshot.routing_options
                        if "routing_options" in fields
                        else locked_row.routing_options
                    )
                    if effective_provider_type == "vendor_specific":
                        raise _config_validation_error()
                    if effective_provider_type == "openai_compatible" and effective_routing_options:
                        raise _config_validation_error()

                locked_semantic_changed = _provider_semantics_changed(
                    locked_row,
                    snapshot,
                    fields,
                    canonical_endpoint,
                    canonical_headers,
                )
                profiles = self._profiles(repositories)
                if (
                    locked_semantic_changed
                    and profiles is not None
                    and await profiles.provider_config_is_referenced_by_generation(
                        identifier,
                        lock_profiles=True,
                    )
                ):
                    raise _immutable_index_configuration_error()

                locked_target_credential_id = (
                    snapshot.credential_id
                    if "credential_id" in fields
                    else locked_row.credential_id
                )
                if "credential_id" in fields:
                    if locked_target_credential_id is None:
                        raise _config_validation_error()
                    credential = await repositories.credentials.get_safe(
                        locked_target_credential_id,
                        for_update=True,
                    )
                    if credential is None:
                        raise _not_found_error()

                locked_migrating_legacy = (
                    locked_row.credential_id is None and "credential_id" in fields
                )
                if (
                    locked_row.credential_id is None
                    and "base_url" in fields
                    and not locked_migrating_legacy
                ):
                    raise _config_validation_error()
                locked_revalidate_endpoint = (
                    "base_url" in fields
                    and canonical_endpoint is not None
                    and canonical_endpoint.url != locked_row.base_url
                ) or locked_migrating_legacy
                if locked_revalidate_endpoint:
                    if policy is None or resolved is None:
                        raise _config_validation_error()
                    selected_url = (
                        snapshot.base_url if "base_url" in fields else locked_row.base_url
                    )
                    if selected_url is None:
                        raise _config_validation_error()
                    locked_raw_url = selected_url
                    locked_endpoint = policy.validate_url(locked_raw_url)
                    if resolved.endpoint != locked_endpoint:
                        raise _endpoint_rejected_error()
                    values.update(
                        {
                            "base_url": resolved.endpoint.url,
                            "endpoint_policy_version": resolved.endpoint.policy_version,
                            "endpoint_validated_at": self._now(),
                        }
                    )

                for field_name in (
                    "name",
                    "provider_type",
                    "routing_options",
                    "timeout_seconds",
                    "max_concurrency",
                    "requests_per_minute",
                    "enabled",
                ):
                    if field_name in fields:
                        value = getattr(snapshot, field_name)
                        if value is None:
                            raise _config_validation_error()
                        values[field_name] = deepcopy(value)
                if canonical_headers is not None:
                    values["default_headers"] = dict(canonical_headers)
                if "credential_id" in fields:
                    if locked_target_credential_id is None:
                        raise _config_validation_error()
                    values["credential_id"] = locked_target_credential_id
                    if locked_migrating_legacy:
                        values["secret_ref"] = None

                try:
                    updated = await configs.update_validated(
                        identifier,
                        values=values,
                        updated_at=self._now(),
                    )
                    await repositories.audits.add(
                        AuditEvent(
                            id=uuid4(),
                            request_id=request_id,
                            actor_api_key_id=actor.key_id,
                            actor_kind="admin_key",
                            action="provider_config.updated",
                            target_type="provider_config",
                            target_id=identifier,
                            metadata_={
                                "updated_fields": sorted(fields),
                                "endpoint_revalidated": locked_revalidate_endpoint,
                            },
                        )
                    )
                except IntegrityError as source:
                    integrity_failed = True
                    conflict = _constraint_name(source)

                if integrity_failed:
                    if conflict == _CONFIG_NAME_UNIQUE_CONSTRAINT:
                        raise _name_conflict_error()
                    raise _internal_error()
                if updated is None:
                    raise _internal_error()
                result = self._safe(updated)
                return result
        except asyncio.CancelledError:
            cancelled = True
        except ProviderNetworkPolicyError:
            endpoint_rejected = True
        except BusinessError:
            raise
        except Exception:
            internal_failure = True
        finally:
            command = _redacted_config_patch()
            snapshot = command
            actor = cast(AdminPrincipal, None)
            request_id = "<redacted>"
            expected_etag = "<redacted>"
            fields.clear()
            if canonical_headers is not None:
                canonical_headers.clear()
            canonical_headers = None
            canonical_endpoint = None
            policy = None
            resolved = None
            repositories = None
            row = None
            locked_row = None
            updated = None
            result = None
            values.clear()
            target_credential_id = None
            locked_target_credential_id = None
            raw_url = "<redacted>"
            locked_raw_url = "<redacted>"
            migrating_legacy = False
            locked_migrating_legacy = False
            revalidate_endpoint = False
            locked_revalidate_endpoint = False
            semantic_changed = False
            locked_semantic_changed = False
            conflict = None
        if cancelled:
            raise asyncio.CancelledError() from None
        if endpoint_rejected:
            raise _endpoint_rejected_error() from None
        if internal_failure:
            raise _internal_error() from None
        raise AssertionError("unreachable")


class ModelProfileService:
    def __init__(
        self,
        *,
        session: AsyncSession,
        settings: Settings,
        repository_factory: RepositoryFactory = sqlalchemy_provider_repositories,
        clock: Clock = _utc_now,
    ) -> None:
        self._session = session
        self._settings = settings
        self._repository_factory = repository_factory
        self._clock = clock

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
            raise _internal_error()
        return value

    def _profiles(self, repositories: ProviderRepositories) -> ModelProfileRepository:
        if repositories.profiles is None:
            raise _internal_error()
        return repositories.profiles

    def _configs(self, repositories: ProviderRepositories) -> ProviderConfigRepository:
        if repositories.configs is None:
            raise _internal_error()
        return repositories.configs

    async def _require_admin_actor(
        self,
        repositories: ProviderRepositories,
        actor: AdminPrincipal,
        now: datetime,
    ) -> AdminActorRecord:
        if type(actor) is not AdminPrincipal:
            raise _invalid_api_key_error()
        row = await repositories.admins.get_for_update(actor.key_id)
        if (
            row is None
            or row.public_id != actor.public_id
            or row.key_type != "admin"
            or not _is_currently_active(row, now)
        ):
            raise _invalid_api_key_error()
        return row

    def _internal_safe(self, row: ModelProfileRecord) -> InternalSafeModelProfile:
        safe: InternalSafeModelProfile | None = None
        failed = False
        try:
            safe = InternalSafeModelProfile(
                id=row.id,
                name=row.name,
                capability=row.capability,
                provider_config_id=row.provider_config_id,
                model_name=row.model_name,
                dimension=row.dimension,
                max_input_tokens=row.max_input_tokens,
                batch_size=row.batch_size,
                timeout_seconds=row.timeout_seconds,
                vector_config=deepcopy(row.vector_config),
                enabled=row.enabled,
                resource_revision=row.resource_revision,
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
        except (AttributeError, TypeError, ValueError, ValidationError):
            failed = True
        finally:
            row = cast(ModelProfileRecord, None)
        if failed or safe is None:
            raise _internal_error() from None
        return safe

    def _safe(self, row: ModelProfileRecord) -> SafeModelProfile:
        internal: InternalSafeModelProfile | None = None
        safe: SafeModelProfile | None = None
        failed = False
        try:
            internal = self._internal_safe(row)
            safe = SafeModelProfile.model_validate(internal.model_dump(mode="python"))
        except (TypeError, ValueError, ValidationError):
            failed = True
        finally:
            internal = None
            row = cast(ModelProfileRecord, None)
        if failed or safe is None:
            raise _internal_error() from None
        return safe

    def _require_profile_provider(self, row: ProviderConfigRecord) -> None:
        if row.provider_type not in _EMBEDDING_PROVIDER_TYPES:
            raise _profile_validation_error()
        if not row.enabled:
            raise _provider_config_disabled_error()

    async def _replay_create(
        self,
        repositories: ProviderRepositories,
        fingerprint: bytes,
        record: IdempotencyRecord,
    ) -> ModelProfileCreateResult:
        raw_response: object | None = None
        response: dict[str, object] = {}
        safe: SafeModelProfile | None = None
        result: ModelProfileCreateResult | None = None
        invalid_snapshot = False
        try:
            if not hmac.compare_digest(record.request_fingerprint, fingerprint):
                raise _idempotency_reused_error()
            if record.result_resource_type != "model_profile" or record.http_status != 201:
                raise _internal_error()
            raw_response = await repositories.audits.get_model_profile_created_response(
                record.actor_key_id,
                record.result_resource_id,
            )
            if type(raw_response) is not dict:
                raise ValueError
            response = deepcopy(raw_response)
            raw_response = None
            safe = SafeModelProfile.model_validate(response)
            if safe.id != record.result_resource_id:
                raise _internal_error()
            result = ModelProfileCreateResult(model_profile=safe, created=False)
            return result
        except BusinessError:
            raise
        except (TypeError, ValueError, ValidationError):
            invalid_snapshot = True
        finally:
            raw_response = None
            response.clear()
            safe = None
            result = None
            record = cast(IdempotencyRecord, None)
            fingerprint = b""
        if invalid_snapshot:
            raise _internal_error() from None
        raise AssertionError("unreachable")

    async def create_model_profile(
        self,
        command: ModelProfileCreate,
        *,
        actor: AdminPrincipal,
        request_id: str,
        idempotency_key: str,
    ) -> ModelProfileCreateResult:
        if type(command) is not ModelProfileCreate:
            raise _profile_validation_error()
        snapshot = _redacted_profile_create()
        validated_key = "<redacted>"
        fingerprint = b""
        result: ModelProfileCreateResult | None = None
        conflict: str | None = None
        cancelled = False
        internal_failure = False
        try:
            _require_request_id(request_id, self._settings.max_request_id_length)
            validated_key = validate_idempotency_key(
                idempotency_key,
                self._settings.max_idempotency_key_length,
            )
            snapshot = command.model_copy(deep=True)
            fingerprint = _profile_create_fingerprint(
                snapshot,
                self._settings.admin_key_hmac_secret,
            )
            async with self._session.begin():
                repositories = self._repository_factory(self._session)
                await self._require_admin_actor(repositories, actor, self._now())
                existing = await repositories.idempotency.get(
                    actor.key_id,
                    _PROFILE_CREATE_OPERATION,
                    validated_key,
                )
                if existing is not None:
                    return await self._replay_create(repositories, fingerprint, existing)
                provider = await self._configs(repositories).get_safe(
                    snapshot.provider_config_id,
                    for_update=True,
                )
                if provider is None:
                    raise _not_found_error()
                self._require_profile_provider(provider)
                model_profile_id = uuid4()
                row: ModelProfileRecord | None = None
                safe: SafeModelProfile | None = None
                try:
                    async with self._session.begin_nested():
                        row = await self._profiles(repositories).add_profile(
                            model_profile_id,
                            name=snapshot.name,
                            capability=snapshot.capability,
                            provider_config_id=snapshot.provider_config_id,
                            model_name=snapshot.model_name,
                            dimension=snapshot.dimension,
                            max_input_tokens=snapshot.max_input_tokens,
                            batch_size=snapshot.batch_size,
                            timeout_seconds=snapshot.timeout_seconds,
                            vector_config=deepcopy(snapshot.vector_config),
                            enabled=snapshot.enabled,
                        )
                        safe = self._safe(row)
                        await repositories.idempotency.add(
                            IdempotencyRecord(
                                id=uuid4(),
                                actor_key_id=actor.key_id,
                                operation=_PROFILE_CREATE_OPERATION,
                                idempotency_key=validated_key,
                                request_fingerprint=fingerprint,
                                result_resource_type="model_profile",
                                result_resource_id=model_profile_id,
                                http_status=201,
                            )
                        )
                        await repositories.audits.add(
                            AuditEvent(
                                id=uuid4(),
                                request_id=request_id,
                                actor_api_key_id=actor.key_id,
                                actor_kind="admin_key",
                                action="model_profile.created",
                                target_type="model_profile",
                                target_id=model_profile_id,
                                metadata_=safe.model_dump(mode="json"),
                            )
                        )
                except IntegrityError as source:
                    conflict = _constraint_name(source)
                if conflict is not None:
                    if conflict == _IDEMPOTENCY_UNIQUE_CONSTRAINT:
                        winner = await repositories.idempotency.get(
                            actor.key_id,
                            _PROFILE_CREATE_OPERATION,
                            validated_key,
                        )
                        if winner is None:
                            raise _internal_error()
                        return await self._replay_create(repositories, fingerprint, winner)
                    if conflict == _PROFILE_NAME_UNIQUE_CONSTRAINT:
                        raise _name_conflict_error()
                    raise _internal_error()
                if row is None or safe is None:
                    raise _internal_error()
                result = ModelProfileCreateResult(model_profile=safe, created=True)
                return result
        except asyncio.CancelledError:
            cancelled = True
        except BusinessError:
            raise
        except Exception:
            internal_failure = True
        finally:
            command = _redacted_profile_create()
            snapshot = command
            actor = cast(AdminPrincipal, None)
            request_id = "<redacted>"
            idempotency_key = "<redacted>"
            validated_key = "<redacted>"
            fingerprint = b""
            result = None
            conflict = None
        if cancelled:
            raise asyncio.CancelledError() from None
        if internal_failure:
            raise _internal_error() from None
        raise AssertionError("unreachable")

    async def list_model_profiles(
        self,
        *,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> ModelProfilePage:
        try:
            position = None if cursor is None else decode_cursor(cursor)
            maximum = min(self._settings.max_page_size, _HARD_MAX_PAGE_SIZE)
            page_limit = min(self._settings.default_page_size, maximum) if limit is None else limit
            if type(page_limit) is not int or not 1 <= page_limit <= maximum:
                raise _profile_validation_error()
            repositories = self._repository_factory(self._session)
            rows = await self._profiles(repositories).list_safe(position, page_limit + 1)
            has_more = len(rows) > page_limit
            visible = rows[:page_limit]
            next_cursor = None
            if has_more and visible:
                last = visible[-1]
                next_cursor = encode_cursor(CursorPosition(created_at=last.created_at, id=last.id))
            return ModelProfilePage(
                items=tuple(self._safe(row) for row in visible),
                next_cursor=next_cursor,
            )
        except BusinessError:
            raise
        except Exception:
            raise _internal_error() from None

    async def get_model_profile(self, model_profile_id: UUID) -> SafeModelProfile:
        identifier = _require_uuid(model_profile_id)
        try:
            repositories = self._repository_factory(self._session)
            row = await self._profiles(repositories).get_safe(identifier)
            if row is None or row.capability not in _EXPOSED_CAPABILITIES:
                raise _not_found_error()
            return self._safe(row)
        except BusinessError:
            raise
        except Exception:
            raise _internal_error() from None

    async def update_model_profile(
        self,
        model_profile_id: UUID,
        command: ModelProfilePatch,
        *,
        actor: AdminPrincipal,
        request_id: str,
        expected_etag: str | None,
    ) -> SafeModelProfile:
        identifier = _require_uuid(model_profile_id)
        if type(command) is not ModelProfilePatch:
            raise _profile_validation_error()
        snapshot = _redacted_profile_patch()
        fields: set[str] = set()
        provider_ids: set[UUID] = set()
        locked_providers: dict[UUID, ProviderConfigRecord | None] = {}
        values: dict[str, object] = {}
        conflict: str | None = None
        cancelled = False
        internal_failure = False
        try:
            _require_request_id(request_id, self._settings.max_request_id_length)
            snapshot = command.model_copy(deep=True)
            fields = set(snapshot.model_fields_set)
            async with self._session.begin():
                repositories = self._repository_factory(self._session)
                await self._require_admin_actor(repositories, actor, self._now())
                profiles = self._profiles(repositories)
                row = await profiles.get_safe(identifier)
                if row is None or row.capability not in _EXPOSED_CAPABILITIES:
                    raise _not_found_error()
                provider_ids.add(row.provider_config_id)
                if "provider_config_id" in fields:
                    if snapshot.provider_config_id is None:
                        raise _profile_validation_error()
                    provider_ids.add(snapshot.provider_config_id)
                configs = self._configs(repositories)
                for provider_id in sorted(provider_ids, key=lambda value: value.int):
                    locked_providers[provider_id] = await configs.get_safe(
                        provider_id,
                        for_update=True,
                    )

                locked_row = await profiles.get_safe(identifier, for_update=True)
                if locked_row is None or locked_row.capability not in _EXPOSED_CAPABILITIES:
                    raise _not_found_error()
                if expected_etag is None:
                    raise _precondition_required_error()
                require_matching_etag(
                    expected_etag,
                    model_profile_etag(locked_row.id, locked_row.resource_revision),
                )
                if locked_row.provider_config_id != row.provider_config_id:
                    raise _precondition_failed_error()

                semantic_fields = {
                    "provider_config_id",
                    "model_name",
                    "dimension",
                    "max_input_tokens",
                    "vector_config",
                }
                semantic_changed = any(
                    field_name in fields
                    and getattr(snapshot, field_name) != getattr(locked_row, field_name)
                    for field_name in semantic_fields
                )
                if semantic_changed and await profiles.is_referenced_by_generation(identifier):
                    raise _immutable_index_configuration_error()

                target_provider_id = (
                    snapshot.provider_config_id
                    if "provider_config_id" in fields
                    else locked_row.provider_config_id
                )
                provider_changed = (
                    "provider_config_id" in fields
                    and target_provider_id != locked_row.provider_config_id
                )
                enabling = (
                    "enabled" in fields and snapshot.enabled is True and not locked_row.enabled
                )
                if provider_changed or enabling:
                    if target_provider_id is None:
                        raise _profile_validation_error()
                    provider = locked_providers.get(target_provider_id)
                    if provider is None:
                        raise _not_found_error()
                    self._require_profile_provider(provider)

                for field_name in fields:
                    value = getattr(snapshot, field_name)
                    if value is None:
                        raise _profile_validation_error()
                    values[field_name] = deepcopy(value)
                try:
                    updated = await profiles.update(
                        identifier,
                        values=values,
                        updated_at=self._now(),
                    )
                    await repositories.audits.add(
                        AuditEvent(
                            id=uuid4(),
                            request_id=request_id,
                            actor_api_key_id=actor.key_id,
                            actor_kind="admin_key",
                            action="model_profile.updated",
                            target_type="model_profile",
                            target_id=identifier,
                            metadata_={"updated_fields": sorted(fields)},
                        )
                    )
                except IntegrityError as source:
                    conflict = _constraint_name(source)
                if conflict is not None:
                    if conflict == _PROFILE_NAME_UNIQUE_CONSTRAINT:
                        raise _name_conflict_error()
                    raise _internal_error()
                return self._safe(updated)
        except asyncio.CancelledError:
            cancelled = True
        except BusinessError:
            raise
        except Exception:
            internal_failure = True
        finally:
            command = _redacted_profile_patch()
            snapshot = command
            actor = cast(AdminPrincipal, None)
            request_id = "<redacted>"
            expected_etag = "<redacted>"
            fields.clear()
            provider_ids.clear()
            locked_providers.clear()
            values.clear()
            conflict = None
        if cancelled:
            raise asyncio.CancelledError() from None
        if internal_failure:
            raise _internal_error() from None
        raise AssertionError("unreachable")


class ProviderCredentialService(_ProviderCredentialServiceBase):
    async def list_credentials(
        self,
        *,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> ProviderCredentialPage:
        result: ProviderCredentialPage | None = None
        failed = False
        try:
            position = None if cursor is None else decode_cursor(cursor)
            maximum = min(self._settings.max_page_size, _HARD_MAX_PAGE_SIZE)
            page_limit = min(self._settings.default_page_size, maximum) if limit is None else limit
            if type(page_limit) is not int or not 1 <= page_limit <= maximum:
                raise _validation_error()
            repositories = self._repository_factory(self._session)
            rows = await repositories.credentials.list_safe(position, page_limit + 1)
            has_more = len(rows) > page_limit
            visible = rows[:page_limit]
            next_cursor = None
            if has_more and visible:
                last = visible[-1]
                next_cursor = encode_cursor(CursorPosition(created_at=last.created_at, id=last.id))
            result = ProviderCredentialPage(
                items=tuple(self._safe(row) for row in visible),
                next_cursor=next_cursor,
            )
        except BusinessError:
            raise
        except Exception:
            failed = True
        if failed or result is None:
            raise _internal_error() from None
        return result

    async def get_credential(self, credential_id: UUID) -> SafeProviderCredential:
        identifier = _require_uuid(credential_id)
        result: SafeProviderCredential | None = None
        failed = False
        try:
            repositories = self._repository_factory(self._session)
            row = await repositories.credentials.get_safe(identifier)
            if row is None:
                raise _not_found_error()
            result = self._safe(row)
        except BusinessError:
            raise
        except Exception:
            failed = True
        if failed or result is None:
            raise _internal_error() from None
        return result

    async def update_credential(
        self,
        credential_id: UUID,
        command: ProviderCredentialPatch,
        *,
        actor: AdminPrincipal,
        request_id: str,
        expected_etag: str | None,
    ) -> SafeProviderCredential:
        identifier = _require_uuid(credential_id)
        if type(command) is not ProviderCredentialPatch:
            raise _validation_error()
        snapshot = _redacted_patch()
        encrypted = _redacted_encrypted()
        encrypted_value: EncryptedProviderCredential | None = None
        repositories: ProviderRepositories | None = None
        row: ProviderCredentialRecord | None = None
        updated: ProviderCredentialRecord | None = None
        result: SafeProviderCredential | None = None
        fields: set[str] = set()
        name: str | None = None
        conflict: str | None = None
        integrity_failed = False
        secret_rotated = False
        cancelled = False
        provider_unavailable = False
        internal_failure = False
        try:
            _require_request_id(request_id, self._settings.max_request_id_length)
            if expected_etag is None:
                raise _precondition_required_error()
            snapshot = command.model_copy(deep=True)
            async with self._session.begin():
                repositories = self._repository_factory(self._session)
                now = self._now()
                await self._require_admin_actor(repositories, actor, now)
                row = await repositories.credentials.get_safe(identifier, for_update=True)
                if row is None:
                    raise _not_found_error()
                require_matching_etag(
                    expected_etag,
                    provider_credential_etag(row.id, row.resource_revision),
                )
                fields = snapshot.model_fields_set
                name = snapshot.name if "name" in fields else None
                secret_rotated = "secret" in fields
                if secret_rotated:
                    if snapshot.secret is None:
                        raise _validation_error()
                    encrypted = self._encrypt_secret(identifier, snapshot.secret)
                    encrypted_value = encrypted

                try:
                    updated = await repositories.credentials.update_encrypted(
                        identifier,
                        name=name,
                        encrypted=encrypted_value,
                        updated_at=now,
                        rotated_at=now if secret_rotated else None,
                    )
                    await repositories.audits.add(
                        AuditEvent(
                            id=uuid4(),
                            request_id=request_id,
                            actor_api_key_id=actor.key_id,
                            actor_kind="admin_key",
                            action=(
                                "provider_credential.rotated"
                                if secret_rotated
                                else "provider_credential.updated"
                            ),
                            target_type="provider_credential",
                            target_id=identifier,
                            metadata_=(
                                {"name_updated": "name" in fields} if secret_rotated else {}
                            ),
                        )
                    )
                except IntegrityError as source:
                    integrity_failed = True
                    conflict = _constraint_name(source)
                finally:
                    encrypted = _redacted_encrypted()
                    encrypted_value = None

                if integrity_failed:
                    if conflict == _NAME_UNIQUE_CONSTRAINT:
                        raise _name_conflict_error()
                    raise _internal_error()
                if updated is None:
                    raise _internal_error()
                result = self._safe(updated)
                return result
        except asyncio.CancelledError:
            cancelled = True
        except ProviderCredentialUnavailableError:
            provider_unavailable = True
        except BusinessError:
            raise
        except Exception:
            internal_failure = True
        finally:
            command = _redacted_patch()
            snapshot = command
            actor = cast(AdminPrincipal, None)
            request_id = "<redacted>"
            expected_etag = "<redacted>"
            encrypted = _redacted_encrypted()
            encrypted_value = None
            repositories = None
            row = None
            updated = None
            result = None
            fields.clear()
            name = None
            conflict = None
        if cancelled:
            raise asyncio.CancelledError() from None
        if provider_unavailable:
            raise _credential_unavailable_error() from None
        if internal_failure:
            raise _internal_error() from None
        raise AssertionError("unreachable")
