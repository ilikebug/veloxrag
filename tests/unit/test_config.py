import base64
import json

import pytest
from pydantic import SecretStr, ValidationError

from rag_service.config import Environment, Settings


def _production_keyring() -> SecretStr:
    return SecretStr(json.dumps({"2026-07": base64.b64encode(bytes(range(32))).decode()}))


def test_local_settings_have_service_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.environment is Environment.LOCAL
    assert settings.api_host == "0.0.0.0"
    assert settings.api_port == 8000
    assert settings.qdrant_url == "http://localhost:6333"
    assert settings.minio_bucket == "rag-documents"
    assert settings.shutdown_timeout_seconds == 2.0


def test_local_settings_have_ingestion_and_worker_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.chunk_max_codepoints == 600
    assert settings.chunk_overlap_codepoints == 100
    assert settings.max_upload_bytes == 50 * 1024 * 1024
    assert settings.ingestion_notify_timeout_seconds == 0.25
    assert settings.upload_buffer_bytes == 1024 * 1024
    assert settings.minio_multipart_part_size_bytes == 5 * 1024 * 1024
    assert settings.minio_operation_timeout_seconds == 300.0
    assert settings.worker_poll_interval_seconds == 1.0
    assert settings.worker_lease_seconds == 60.0
    assert settings.worker_heartbeat_seconds == 15.0
    assert settings.worker_max_attempts == 5
    assert settings.worker_retry_initial_seconds == 5.0
    assert settings.worker_retry_max_seconds == 300.0
    assert settings.orphan_object_grace_seconds == 24 * 60 * 60
    assert settings.qdrant_connect_timeout_seconds == 5.0
    assert settings.qdrant_request_timeout_seconds == 30.0
    assert settings.provider_allow_private_targets is False
    assert settings.provider_ca_bundle is None
    assert settings.provider_credential_active_key_version == "local-v1"
    assert settings.provider_credential_keyring.get_secret_value() == json.dumps(
        {"local-v1": base64.b64encode(b"d" * 32).decode()}
    )


@pytest.mark.parametrize(
    ("maximum", "overlap"),
    [
        (1201, 150),
        (0, 150),
        (500, -1),
        (500, 500),
    ],
)
def test_chunk_sizing_rejects_values_the_chunker_cannot_honour(
    maximum: int,
    overlap: int,
) -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            chunk_max_codepoints=maximum,
            chunk_overlap_codepoints=overlap,
        )


def test_chunk_sizing_accepts_a_smaller_cjk_friendly_size() -> None:
    settings = Settings(_env_file=None, chunk_max_codepoints=500, chunk_overlap_codepoints=80)

    assert settings.chunk_max_codepoints == 500
    assert settings.chunk_overlap_codepoints == 80


def test_settings_accept_valid_versioned_provider_credential_keyring() -> None:
    keyring = {
        "2026-07": base64.b64encode(b"a" * 32).decode(),
        "2026-06": base64.b64encode(b"b" * 32).decode(),
    }

    settings = Settings(
        provider_credential_keyring=SecretStr(json.dumps(keyring)),
        provider_credential_active_key_version="2026-07",
        _env_file=None,
    )

    assert settings.provider_credential_keyring.get_secret_value() == json.dumps(keyring)
    assert settings.provider_credential_active_key_version == "2026-07"


@pytest.mark.parametrize(
    "overrides",
    [
        {"provider_credential_keyring": SecretStr("not-json")},
        {
            "provider_credential_keyring": SecretStr(
                json.dumps({"local-v1": base64.b64encode(b"too-short").decode()})
            )
        },
        {
            "provider_credential_keyring": SecretStr(
                json.dumps({"local-v1": base64.b64encode(b"k" * 32).decode()})
            ),
            "provider_credential_active_key_version": "missing",
        },
    ],
)
def test_settings_reject_invalid_provider_credential_keyrings(
    overrides: dict[str, SecretStr | str],
) -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate(overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {"worker_lease_seconds": 10.0, "worker_heartbeat_seconds": 10.0},
        {"worker_lease_seconds": 10.0, "worker_heartbeat_seconds": 11.0},
        {"minio_multipart_part_size_bytes": 5 * 1024 * 1024 - 1},
        {"minio_multipart_part_size_bytes": 5 * 1024 * 1024 * 1024 + 1},
        {"worker_retry_initial_seconds": 11.0, "worker_retry_max_seconds": 10.0},
    ],
)
def test_settings_reject_invalid_ingestion_runtime_relationships(
    overrides: dict[str, float | int],
) -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate(overrides)


def test_environment_variables_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAG_API_PORT", "9000")
    monkeypatch.setenv("RAG_QDRANT_URL", "http://qdrant:6333")

    settings = Settings(_env_file=None)

    assert settings.api_port == 9000
    assert settings.qdrant_url == "http://qdrant:6333"


def test_production_rejects_development_secrets() -> None:
    with pytest.raises(ValidationError, match="production secrets"):
        Settings(environment=Environment.PRODUCTION, _env_file=None)


def test_production_rejects_compose_example_secrets() -> None:
    with pytest.raises(ValidationError, match="production secrets"):
        Settings(
            environment=Environment.PRODUCTION,
            database_url=SecretStr("postgresql+psycopg://rag:change-me-local@postgres:5432/rag"),
            minio_access_key="rag-dev",
            minio_secret_key=SecretStr("change-me-local"),
            _env_file=None,
        )


def test_production_accepts_explicit_secrets() -> None:
    settings = Settings(
        environment=Environment.PRODUCTION,
        database_url=SecretStr("postgresql+psycopg://rag:strong-password@postgres:5432/rag"),
        minio_access_key="prod-access-key",
        minio_secret_key=SecretStr("prod-secret-key"),
        admin_key_hmac_secret=SecretStr("admin-secret-that-is-at-least-thirty-two-bytes"),
        agent_key_hmac_secret=SecretStr("agent-secret-that-is-at-least-thirty-two-bytes"),
        provider_credential_keyring=_production_keyring(),
        provider_credential_active_key_version="2026-07",
        _env_file=None,
    )

    assert settings.environment is Environment.PRODUCTION
    assert "strong-password" not in repr(settings)
    assert "prod-secret-key" not in repr(settings)


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"provider_allow_private_targets": True},
        {
            "provider_credential_keyring": SecretStr(
                json.dumps({"2026-07": base64.b64encode(b"x" * 32).decode()})
            ),
            "provider_credential_active_key_version": "2026-07",
        },
    ],
)
def test_production_rejects_development_provider_settings(
    overrides: dict[str, SecretStr | str | bool],
) -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate(
            {
                "environment": Environment.PRODUCTION,
                "database_url": SecretStr(
                    "postgresql+psycopg://rag:strong-password@postgres:5432/rag"
                ),
                "minio_access_key": "prod-access-key",
                "minio_secret_key": SecretStr("prod-secret-key"),
                "admin_key_hmac_secret": SecretStr(
                    "admin-secret-that-is-at-least-thirty-two-bytes"
                ),
                "agent_key_hmac_secret": SecretStr(
                    "agent-secret-that-is-at-least-thirty-two-bytes"
                ),
                **overrides,
            }
        )


def test_provider_credential_keyring_validation_redacts_secret_input() -> None:
    sentinel = '{"2026-07":"credential-keyring-secret-sentinel"}'

    with pytest.raises(ValidationError) as exc_info:
        Settings(provider_credential_keyring=SecretStr(sentinel), _env_file=None)

    rendered = (str(exc_info.value), repr(exc_info.value.errors()), exc_info.value.json())
    assert all(sentinel not in output for output in rendered)


@pytest.mark.parametrize(
    "raw_keyring",
    [
        {"2026-07": "malformed-provider-keyring-secret-sentinel"},
        ["malformed-provider-keyring-secret-sentinel"],
    ],
)
def test_provider_credential_keyring_type_errors_redact_raw_secret_input(
    raw_keyring: dict[str, str] | list[str],
) -> None:
    sentinel = "malformed-provider-keyring-secret-sentinel"

    with pytest.raises(ValidationError) as exc_info:
        Settings.model_validate({"provider_credential_keyring": raw_keyring})

    rendered = (str(exc_info.value), repr(exc_info.value.errors()), exc_info.value.json())
    assert all(sentinel not in output for output in rendered)


@pytest.mark.parametrize(
    ("admin_secret", "agent_secret"),
    [
        ("", "agent-secret-that-is-at-least-thirty-two-bytes"),
        ("admin-secret-that-is-at-least-thirty-two-bytes", ""),
        ("short", "agent-secret-that-is-at-least-thirty-two-bytes"),
        ("admin-secret-that-is-at-least-thirty-two-bytes", "short"),
        ("same-secret-that-is-at-least-thirty-two", "same-secret-that-is-at-least-thirty-two"),
        ("change-me-admin-local-32-bytes", "agent-secret-that-is-at-least-thirty-two-bytes"),
        (
            "local-dev-admin-hmac-secret-not-for-production-01",
            "local-dev-agent-hmac-secret-not-for-production-02",
        ),
    ],
)
def test_production_rejects_unsafe_hmac_secrets(admin_secret: str, agent_secret: str) -> None:
    with pytest.raises(ValidationError, match="production secrets"):
        Settings(
            environment=Environment.PRODUCTION,
            database_url=SecretStr("postgresql+psycopg://rag:strong-password@postgres:5432/rag"),
            minio_access_key="prod-access-key",
            minio_secret_key=SecretStr("prod-secret-key"),
            admin_key_hmac_secret=SecretStr(admin_secret),
            agent_key_hmac_secret=SecretStr(agent_secret),
            _env_file=None,
        )


def test_production_validation_redacts_hmac_secrets() -> None:
    admin_secret = "admin-secret-that-must-never-appear-in-errors"
    agent_secret = admin_secret

    with pytest.raises(ValidationError) as exc_info:
        Settings(
            environment=Environment.PRODUCTION,
            database_url=SecretStr("postgresql+psycopg://rag:strong-password@postgres:5432/rag"),
            minio_access_key="prod-access-key",
            minio_secret_key=SecretStr("prod-secret-key"),
            admin_key_hmac_secret=SecretStr(admin_secret),
            agent_key_hmac_secret=SecretStr(agent_secret),
            _env_file=None,
        )

    assert admin_secret not in str(exc_info.value)
    assert agent_secret not in str(exc_info.value)
    assert admin_secret not in repr(exc_info.value.errors())
    assert agent_secret not in repr(exc_info.value.errors())
    assert admin_secret not in exc_info.value.json()
    assert agent_secret not in exc_info.value.json()


@pytest.mark.parametrize(
    "overrides",
    [
        {"default_page_size": 0},
        {"readiness_timeout_seconds": float("nan")},
        {"ingestion_notify_timeout_seconds": 0},
        {"ingestion_notify_timeout_seconds": float("nan")},
        {"ingestion_notify_timeout_seconds": float("inf")},
        {"shutdown_timeout_seconds": 0},
        {"shutdown_timeout_seconds": float("nan")},
        {"shutdown_timeout_seconds": float("inf")},
        {"max_page_size": 0},
        {"max_api_key_requests_per_minute": 0},
        {"max_api_key_concurrency": 0},
        {"max_request_id_length": 0},
        {"max_idempotency_key_length": 0},
        {"minio_operation_timeout_seconds": 0},
        {"minio_operation_timeout_seconds": float("nan")},
        {"default_page_size": 101, "max_page_size": 100},
    ],
)
def test_settings_reject_invalid_api_limits(overrides: dict[str, float | int]) -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate(overrides)


def test_settings_enforce_minio_operation_timeout_upper_bound() -> None:
    settings = Settings.model_validate({"minio_operation_timeout_seconds": 600})

    assert settings.minio_operation_timeout_seconds == 600

    with pytest.raises(ValidationError):
        Settings.model_validate({"minio_operation_timeout_seconds": 600.01})


@pytest.mark.parametrize(
    "field",
    [
        "api_port",
        "readiness_timeout_seconds",
        "ingestion_notify_timeout_seconds",
        "shutdown_timeout_seconds",
        "default_page_size",
        "max_page_size",
        "max_api_key_requests_per_minute",
        "max_api_key_concurrency",
        "max_request_id_length",
        "max_idempotency_key_length",
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
    ],
)
def test_settings_reject_boolean_numeric_values(field: str) -> None:
    with pytest.raises(ValidationError, match="boolean values"):
        Settings.model_validate({field: True})


def test_settings_accept_numeric_strings_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG_DEFAULT_PAGE_SIZE", "25")
    monkeypatch.setenv("RAG_MAX_PAGE_SIZE", "50")

    settings = Settings(_env_file=None)

    assert settings.default_page_size == 25
    assert settings.max_page_size == 50


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_request_id_length": 27},
        {"max_request_id_length": 129},
        {"max_idempotency_key_length": 129},
    ],
)
def test_settings_enforce_header_length_bounds(overrides: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate(overrides)


def test_boolean_numeric_failure_does_not_retain_production_secrets() -> None:
    sentinels = {
        "database_url": "postgresql+psycopg://rag:bool-db-sentinel@postgres:5432/rag",
        "minio_secret_key": "bool-minio-secret-sentinel",
        "admin_key_hmac_secret": "bool-admin-hmac-secret-sentinel-at-least-32-bytes",
        "agent_key_hmac_secret": "bool-agent-hmac-secret-sentinel-at-least-32-bytes",
    }

    with pytest.raises(ValidationError) as exc_info:
        Settings.model_validate(
            {
                "environment": Environment.PRODUCTION,
                "minio_access_key": "prod-access-key",
                "max_request_id_length": True,
                **sentinels,
            }
        )

    traceback_locals: list[str] = []
    traceback = exc_info.value.__traceback__
    while traceback is not None:
        module_name = traceback.tb_frame.f_globals.get("__name__", "")
        if isinstance(module_name, str) and module_name.startswith("rag_service"):
            traceback_locals.append(repr(traceback.tb_frame.f_locals))
        traceback = traceback.tb_next

    rendered = (
        str(exc_info.value),
        repr(exc_info.value.errors()),
        exc_info.value.json(),
        *traceback_locals,
    )
    for secret in sentinels.values():
        assert all(secret not in output for output in rendered)


@pytest.mark.parametrize(
    "invalid_limits",
    [
        {"default_page_size": 0},
        {"default_page_size": 101, "max_page_size": 100},
    ],
)
def test_model_validation_redacts_production_secrets_when_limits_are_invalid(
    invalid_limits: dict[str, int],
) -> None:
    sentinels = {
        "database_url": "postgresql+psycopg://rag:db-secret-sentinel@postgres:5432/rag",
        "minio_secret_key": "minio-secret-sentinel-unique",
        "admin_key_hmac_secret": "admin-hmac-secret-sentinel-at-least-32-bytes",
        "agent_key_hmac_secret": "agent-hmac-secret-sentinel-at-least-32-bytes",
    }

    with pytest.raises(ValidationError) as exc_info:
        Settings.model_validate(
            {
                "environment": Environment.PRODUCTION,
                "minio_access_key": "prod-access-key",
                **sentinels,
                **invalid_limits,
            }
        )

    rendered = (
        str(exc_info.value),
        repr(exc_info.value.errors()),
        exc_info.value.json(),
    )
    for secret in sentinels.values():
        assert all(secret not in output for output in rendered)


def test_production_validation_redacts_raw_environment_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_secret = "postgresql+psycopg://rag:db-change-me-sentinel@postgres:5432/rag"
    minio_secret = "minio-change-me-sentinel"
    monkeypatch.setenv("RAG_ENVIRONMENT", "production")
    monkeypatch.setenv("RAG_DATABASE_URL", database_secret)
    monkeypatch.setenv("RAG_MINIO_ACCESS_KEY", "prod-access-key")
    monkeypatch.setenv("RAG_MINIO_SECRET_KEY", minio_secret)

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    error = exc_info.value
    assert database_secret not in str(error)
    assert minio_secret not in str(error)
    assert database_secret not in repr(error.errors())
    assert minio_secret not in repr(error.errors())
    assert database_secret not in error.json()
    assert minio_secret not in error.json()


@pytest.mark.parametrize(
    ("database_url", "minio_access_key", "minio_secret_key"),
    [
        ("", "prod-access-key", "prod-secret-key"),
        ("prod-database-url", "", "prod-secret-key"),
        ("prod-database-url", "prod-access-key", ""),
        ("  ", "  ", "  "),
    ],
)
def test_production_rejects_blank_credentials(
    database_url: str,
    minio_access_key: str,
    minio_secret_key: str,
) -> None:
    with pytest.raises(ValidationError, match="production secrets"):
        Settings(
            environment=Environment.PRODUCTION,
            database_url=database_url,
            minio_access_key=minio_access_key,
            minio_secret_key=minio_secret_key,
            _env_file=None,
        )


@pytest.mark.parametrize(
    "unsafe",
    [
        {"provider_allow_private_targets": True},
        {"local_trusted_auth": True},
    ],
)
def test_production_refuses_the_local_only_switches(unsafe: dict[str, bool]) -> None:
    """The compose file defaults both on; this is what stops that reaching production.

    A convenience switch that survives a copied configuration file stops being a
    convenience, so the refusal has to live in the settings rather than in a
    comment telling operators to turn it off.
    """
    safe = {
        "environment": "production",
        "database_url": "postgresql+psycopg://rag:s3cret-not-a-marker@db:5432/rag",
        "minio_access_key": "production-access",
        "minio_secret_key": "production-secret-value",
        "admin_key_hmac_secret": "a" * 40,
        "agent_key_hmac_secret": "b" * 40,
        "provider_credential_keyring": json.dumps(
            {"prod-v1": base64.b64encode(bytes(range(32))).decode()}
        ),
        "provider_credential_active_key_version": "prod-v1",
    }

    with pytest.raises(ValidationError):
        Settings(_env_file=None, **{**safe, **unsafe})  # type: ignore[arg-type]
