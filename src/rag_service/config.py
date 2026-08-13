import base64
import binascii
import json
import math
from enum import StrEnum
from functools import lru_cache

from pydantic import SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag_service.api.constants import GENERATED_REQUEST_ID_LENGTH, MAX_HEADER_VALUE_LENGTH
from rag_service.ingestion.chunkers import (
    DEFAULT_CHUNK_CODEPOINTS,
    DEFAULT_OVERLAP_CODEPOINTS,
    MAX_CHUNK_CODEPOINTS,
    TARGET_OVERLAP_CODEPOINTS,
)

_DEV_DATABASE_URL = "postgresql+psycopg://rag:change-me@localhost:5432/rag"
_DEV_MINIO_ACCESS_KEY = "rag-dev"
_DEV_MINIO_SECRET_KEY = "change-me"
_DEV_ADMIN_KEY_HMAC_SECRET = "change-me-admin-local-32-bytes"
_DEV_AGENT_KEY_HMAC_SECRET = "change-me-agent-local-32-bytes"
_COMPOSE_DEV_ADMIN_KEY_HMAC_SECRET = "local-dev-admin-hmac-secret-not-for-production-01"
_COMPOSE_DEV_AGENT_KEY_HMAC_SECRET = "local-dev-agent-hmac-secret-not-for-production-02"
_UNSAFE_SECRET_MARKERS = ("change-me", "change-me-local")
_MEBIBYTE = 1024 * 1024
_MIN_MINIO_MULTIPART_PART_SIZE_BYTES = 5 * _MEBIBYTE
_MAX_MINIO_MULTIPART_PART_SIZE_BYTES = 5 * 1024 * _MEBIBYTE
_MAX_MINIO_OPERATION_TIMEOUT_SECONDS = 600.0
_DEV_PROVIDER_CREDENTIAL_KEYRING = json.dumps({"local-v1": base64.b64encode(b"d" * 32).decode()})
_NUMERIC_SETTING_FIELDS = (
    "api_port",
    "readiness_timeout_seconds",
    "shutdown_timeout_seconds",
    "default_page_size",
    "max_page_size",
    "max_api_key_requests_per_minute",
    "max_api_key_concurrency",
    "max_request_id_length",
    "max_idempotency_key_length",
    "ingestion_notify_timeout_seconds",
    "max_upload_bytes",
    "upload_buffer_bytes",
    "minio_multipart_part_size_bytes",
    "minio_operation_timeout_seconds",
    "worker_poll_interval_seconds",
    "worker_lease_seconds",
    "worker_heartbeat_seconds",
    "worker_max_attempts",
    "worker_retry_initial_seconds",
    "worker_retry_max_seconds",
    "orphan_object_grace_seconds",
    "qdrant_connect_timeout_seconds",
    "qdrant_request_timeout_seconds",
)


def _redacted_settings_validation_error(message: str) -> ValidationError:
    return ValidationError.from_exception_data(
        Settings.__name__,
        [
            {
                "type": "value_error",
                "loc": (),
                "input": "<redacted>",
                "ctx": {"error": ValueError(message)},
            }
        ],
    )


class Environment(StrEnum):
    LOCAL = "local"
    TEST = "test"
    PRODUCTION = "production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="RAG_",
        extra="ignore",
        case_sensitive=False,
        hide_input_in_errors=True,
    )

    environment: Environment = Environment.LOCAL
    app_name: str = "rag-service"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    database_url: SecretStr = SecretStr(_DEV_DATABASE_URL)
    redis_url: SecretStr = SecretStr("redis://localhost:6379/0")
    qdrant_url: str = "http://localhost:6333"
    qdrant_connect_timeout_seconds: float = 5.0
    qdrant_request_timeout_seconds: float = 30.0
    minio_url: str = "http://localhost:9000"
    minio_access_key: str = _DEV_MINIO_ACCESS_KEY
    minio_secret_key: SecretStr = SecretStr(_DEV_MINIO_SECRET_KEY)
    minio_bucket: str = "rag-documents"
    ingestion_notify_timeout_seconds: float = 0.25
    # Chunk sizing is counted in Unicode code points, so the token count it
    # produces depends on the language: CJK text is roughly one token per code
    # point while Latin text needs about four. Lower these for a CJK corpus.
    chunk_max_codepoints: int = DEFAULT_CHUNK_CODEPOINTS
    chunk_overlap_codepoints: int = DEFAULT_OVERLAP_CODEPOINTS
    max_upload_bytes: int = 50 * _MEBIBYTE
    upload_buffer_bytes: int = _MEBIBYTE
    minio_multipart_part_size_bytes: int = _MIN_MINIO_MULTIPART_PART_SIZE_BYTES
    minio_operation_timeout_seconds: float = 300.0
    worker_poll_interval_seconds: float = 1.0
    worker_lease_seconds: float = 60.0
    worker_heartbeat_seconds: float = 15.0
    worker_max_attempts: int = 5
    worker_retry_initial_seconds: float = 5.0
    worker_retry_max_seconds: float = 300.0
    orphan_object_grace_seconds: float = 24 * 60 * 60
    provider_credential_keyring: SecretStr = SecretStr(_DEV_PROVIDER_CREDENTIAL_KEYRING)
    provider_credential_active_key_version: str = "local-v1"
    # Single-user local installs only: skips bearer authentication and attributes
    # every request to a provisioned local actor. Refused in production below,
    # because a convenience switch that survives a copied configuration file
    # stops being a convenience.
    local_trusted_auth: bool = False
    provider_allow_private_targets: bool = False
    provider_ca_bundle: str | None = None
    readiness_timeout_seconds: float = 2.0
    shutdown_timeout_seconds: float = 2.0
    admin_key_hmac_secret: SecretStr = SecretStr(_DEV_ADMIN_KEY_HMAC_SECRET)
    agent_key_hmac_secret: SecretStr = SecretStr(_DEV_AGENT_KEY_HMAC_SECRET)
    default_page_size: int = 20
    max_page_size: int = 100
    max_api_key_requests_per_minute: int = 10_000
    max_api_key_concurrency: int = 1_000
    max_request_id_length: int = MAX_HEADER_VALUE_LENGTH
    max_idempotency_key_length: int = MAX_HEADER_VALUE_LENGTH

    @field_validator(*_NUMERIC_SETTING_FIELDS, mode="before")
    @classmethod
    def reject_boolean_numeric_values(cls, value: object) -> object:
        if type(value) is bool:
            raise _redacted_settings_validation_error(
                "boolean values are not valid numeric settings"
            )
        return value

    @field_validator("provider_credential_keyring", mode="before")
    @classmethod
    def redact_malformed_provider_credential_keyring(cls, value: object) -> object:
        if not isinstance(value, (SecretStr, str)):
            raise _redacted_settings_validation_error(
                "provider credential keyring must be a secret JSON string"
            )
        return value

    @model_validator(mode="after")
    def validate_limits_and_production_secrets(self) -> "Settings":
        self._validate_provider_credential_keyring()
        numeric_limits = (
            self.api_port,
            self.readiness_timeout_seconds,
            self.shutdown_timeout_seconds,
            self.default_page_size,
            self.max_page_size,
            self.max_api_key_requests_per_minute,
            self.max_api_key_concurrency,
            self.max_request_id_length,
            self.max_idempotency_key_length,
            self.ingestion_notify_timeout_seconds,
            self.chunk_max_codepoints,
            self.max_upload_bytes,
            self.upload_buffer_bytes,
            self.minio_multipart_part_size_bytes,
            self.minio_operation_timeout_seconds,
            self.worker_poll_interval_seconds,
            self.worker_lease_seconds,
            self.worker_heartbeat_seconds,
            self.worker_max_attempts,
            self.worker_retry_initial_seconds,
            self.worker_retry_max_seconds,
            self.orphan_object_grace_seconds,
            self.qdrant_connect_timeout_seconds,
            self.qdrant_request_timeout_seconds,
        )
        if any(
            limit <= 0 or (isinstance(limit, float) and not math.isfinite(limit))
            for limit in numeric_limits
        ):
            raise _redacted_settings_validation_error("numeric limits must be positive")
        if self.default_page_size > self.max_page_size:
            raise _redacted_settings_validation_error(
                "default page size must not exceed maximum page size"
            )
        if not (
            GENERATED_REQUEST_ID_LENGTH <= self.max_request_id_length <= MAX_HEADER_VALUE_LENGTH
        ):
            raise _redacted_settings_validation_error(
                "request ID length is outside supported bounds"
            )
        if self.max_idempotency_key_length > MAX_HEADER_VALUE_LENGTH:
            raise _redacted_settings_validation_error(
                "idempotency key length is outside supported bounds"
            )
        if self.upload_buffer_bytes > self.max_upload_bytes:
            raise _redacted_settings_validation_error(
                "upload buffer size must not exceed upload limit"
            )
        if self.chunk_max_codepoints > MAX_CHUNK_CODEPOINTS:
            raise _redacted_settings_validation_error("chunk size is outside supported bounds")
        if not 0 <= self.chunk_overlap_codepoints < self.chunk_max_codepoints:
            raise _redacted_settings_validation_error(
                "chunk overlap must be smaller than the chunk size"
            )
        # The chunker enforces this ceiling too, but only when it is constructed
        # mid-startup and with a generic error; checking it here fails the
        # process on its settings with a message that names the offending knob.
        if self.chunk_overlap_codepoints > TARGET_OVERLAP_CODEPOINTS:
            raise _redacted_settings_validation_error("chunk overlap is outside supported bounds")
        if not (
            _MIN_MINIO_MULTIPART_PART_SIZE_BYTES
            <= self.minio_multipart_part_size_bytes
            <= _MAX_MINIO_MULTIPART_PART_SIZE_BYTES
        ):
            raise _redacted_settings_validation_error(
                "MinIO multipart part size is outside supported S3 bounds"
            )
        if self.minio_operation_timeout_seconds > _MAX_MINIO_OPERATION_TIMEOUT_SECONDS:
            raise _redacted_settings_validation_error(
                "MinIO operation timeout exceeds the supported maximum"
            )
        if self.worker_heartbeat_seconds >= self.worker_lease_seconds:
            raise _redacted_settings_validation_error(
                "worker heartbeat interval must be shorter than the lease duration"
            )
        if self.worker_retry_initial_seconds > self.worker_retry_max_seconds:
            raise _redacted_settings_validation_error(
                "worker retry initial delay must not exceed maximum delay"
            )

        if self.environment is not Environment.PRODUCTION:
            return self

        database_url = self.database_url.get_secret_value()
        minio_secret = self.minio_secret_key.get_secret_value()
        admin_hmac_secret = self.admin_key_hmac_secret.get_secret_value()
        agent_hmac_secret = self.agent_key_hmac_secret.get_secret_value()
        credentials = (database_url, self.minio_access_key, minio_secret)
        unsafe_hmac_secret = (
            any(
                not secret.strip() or len(secret.encode("utf-8")) < 32
                for secret in (
                    admin_hmac_secret,
                    agent_hmac_secret,
                )
            )
            or admin_hmac_secret == agent_hmac_secret
            or admin_hmac_secret
            in (
                _DEV_ADMIN_KEY_HMAC_SECRET,
                _DEV_AGENT_KEY_HMAC_SECRET,
                _COMPOSE_DEV_ADMIN_KEY_HMAC_SECRET,
                _COMPOSE_DEV_AGENT_KEY_HMAC_SECRET,
            )
            or agent_hmac_secret
            in (
                _DEV_ADMIN_KEY_HMAC_SECRET,
                _DEV_AGENT_KEY_HMAC_SECRET,
                _COMPOSE_DEV_ADMIN_KEY_HMAC_SECRET,
                _COMPOSE_DEV_AGENT_KEY_HMAC_SECRET,
            )
        )
        unsafe = (
            unsafe_hmac_secret
            or self.provider_allow_private_targets
            or self.local_trusted_auth
            or self._provider_credential_keyring_is_unsafe_for_production()
            or any(not credential.strip() for credential in credentials)
            or (
                self.minio_access_key == _DEV_MINIO_ACCESS_KEY
                or any(
                    marker in database_url or marker in minio_secret
                    for marker in _UNSAFE_SECRET_MARKERS
                )
            )
        )
        if unsafe:
            raise _redacted_settings_validation_error(
                "production secrets must replace development defaults"
            )
        return self

    def _validate_provider_credential_keyring(self) -> None:
        keyring_secret = self.provider_credential_keyring.get_secret_value()
        try:
            raw_keyring = json.loads(keyring_secret)
        except json.JSONDecodeError:
            raise _redacted_settings_validation_error(
                "provider credential keyring must be a JSON mapping"
            ) from None
        if not isinstance(raw_keyring, dict) or not raw_keyring:
            raise _redacted_settings_validation_error(
                "provider credential keyring must be a non-empty JSON mapping"
            )
        if self.provider_credential_active_key_version not in raw_keyring:
            raise _redacted_settings_validation_error(
                "active provider credential key version is missing from the keyring"
            )
        for version, encoded_key in raw_keyring.items():
            if (
                not isinstance(version, str)
                or not version.strip()
                or not isinstance(encoded_key, str)
            ):
                raise _redacted_settings_validation_error(
                    "provider credential keyring entries must map versions to base64 keys"
                )
            try:
                decoded_key = base64.b64decode(encoded_key, validate=True)
            except (ValueError, binascii.Error):
                raise _redacted_settings_validation_error(
                    "provider credential keyring entries must contain base64 keys"
                ) from None
            if len(decoded_key) != 32:
                raise _redacted_settings_validation_error(
                    "provider credential keyring entries must contain 32-byte AES keys"
                )

    def _provider_credential_keyring_is_unsafe_for_production(self) -> bool:
        raw_keyring = json.loads(self.provider_credential_keyring.get_secret_value())
        if self.provider_credential_keyring.get_secret_value() == _DEV_PROVIDER_CREDENTIAL_KEYRING:
            return True
        return any(
            len(set(base64.b64decode(encoded_key, validate=True))) == 1
            for encoded_key in raw_keyring.values()
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
