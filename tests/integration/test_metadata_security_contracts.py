from typing import Any
from uuid import uuid4

import psycopg
import pytest
from psycopg.types.json import Jsonb


def _provider_config_is_accepted(
    connection: psycopg.Connection[Any],
    *,
    provider_type: str = "openrouter",
    secret_ref: str = "env:SYNTHETIC_PROVIDER_KEY",
    base_url: str = "https://example.invalid/v1",
    default_headers: object | None = None,
    routing_options: object | None = None,
) -> bool:
    try:
        connection.execute(
            """
            INSERT INTO provider_configs (
                id, name, provider_type, base_url, secret_ref,
                default_headers, routing_options, timeout_seconds,
                max_concurrency, requests_per_minute
            ) VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s)
            """,
            (
                uuid4(),
                f"synthetic-provider-{uuid4()}",
                provider_type,
                base_url,
                secret_ref,
                Jsonb({} if default_headers is None else default_headers),
                Jsonb({} if routing_options is None else routing_options),
                30,
                10,
                100,
            ),
        )
    except psycopg.errors.CheckViolation:
        return False
    return True


@pytest.mark.integration
def test_provider_config_secret_reference_and_base_url_contract(
    migrated_autocommit_sync_connection: psycopg.Connection[Any],
) -> None:
    connection = migrated_autocommit_sync_connection
    safe_secret_refs = (
        "env:SYNTHETIC_PROVIDER_KEY",
        "file:/run/secrets/synthetic-provider-key",
        "docker-secret:synthetic-provider-key",
        "vault:secret/data/synthetic#provider-key",
        "aws-secrets-manager:synthetic/provider-key",
        "gcp-secret-manager:projects/synthetic/secrets/provider-key/versions/latest",
        "azure-key-vault:synthetic-vault/provider-key",
    )
    for secret_ref in safe_secret_refs:
        assert _provider_config_is_accepted(connection, secret_ref=secret_ref)

    unsafe_secret_refs = (
        "SYNTHETIC_PLAIN_VALUE_WITHOUT_SCHEME",
        "raw:SYNTHETIC-NOT-A-REAL-CREDENTIAL",
        "plain:SYNTHETIC-NOT-A-REAL-CREDENTIAL",
        "env:",
        "env:SYNTHETIC VALUE WITH SPACE",
        "env:SYNTHETIC\tVALUE",
        "env:SYNTHETIC\nVALUE",
    )
    for secret_ref in unsafe_secret_refs:
        assert not _provider_config_is_accepted(connection, secret_ref=secret_ref)

    safe_base_urls = (
        "https://openrouter.ai/api/v1",
        "http://localhost:8000/v1",
        "http://127.0.0.1:8080/path@segment",
        "http://[::1]:8080/v1",
        "https://example.invalid",
        "HTTPS://example.invalid:65535/v1",
    )
    for base_url in safe_base_urls:
        assert _provider_config_is_accepted(connection, base_url=base_url)

    unsafe_base_urls = (
        "ftp://example.invalid/v1",
        "https://synthetic-user:synthetic-password@example.invalid/v1",
        "https://example.invalid/v1?api_key=SYNTHETIC",
        "https://example.invalid/v1?X-Amz-Signature=SYNTHETIC",
        "https://example.invalid/v1#synthetic-fragment",
        "https:///missing-host",
        "https://example.invalid:not-a-port/v1",
        "https://example.invalid:0/v1",
        "https://example.invalid:00000/v1",
        "https://example.invalid:65536/v1",
        "https://example.invalid:99999/v1",
        "https://example.invalid/path with space",
        "/relative/v1",
    )
    for base_url in unsafe_base_urls:
        assert not _provider_config_is_accepted(
            connection,
            base_url=base_url,
        ), f"base_url contract accepted case {base_url.split(':', 1)[0]!r}"


@pytest.mark.integration
def test_provider_config_default_headers_are_exact_safe_allowlist(
    migrated_autocommit_sync_connection: psycopg.Connection[Any],
) -> None:
    connection = migrated_autocommit_sync_connection
    safe_headers: tuple[object, ...] = (
        {},
        {"HTTP-Referer": "https://portal.example.invalid/path@owner"},
        {"HTTP-Referer": "HTTPS://portal.example.invalid:65535/path"},
        {"X-OpenRouter-Title": "Bearer market data"},
        {"X-Title": "Basic standard"},
        {
            "HTTP-Referer": "http://localhost:3000/app",
            "X-OpenRouter-Title": "sk-SK",
            "X-Title": "Synthetic RAG Service",
        },
    )
    for headers in safe_headers:
        assert _provider_config_is_accepted(connection, default_headers=headers)

    unknown_keys = (
        "Authorization",
        "X-API-Key",
        "apiKey",
        "accessToken",
        "clientSecret",
        "privateKey",
        "accessKey",
        "monkey",
        "tokenizer",
        "secretary",
        "accessibility",
        "private",
        "client",
    )
    for key in unknown_keys:
        assert not _provider_config_is_accepted(
            connection,
            default_headers={key: "SYNTHETIC-VALUE"},
        ), f"default_headers accepted unknown key {key!r}"

    invalid_headers: tuple[object, ...] = (
        [],
        {"http-referer": "https://example.invalid"},
        {"x-openrouter-title": "Synthetic"},
        {"x-title": "Synthetic"},
        {"X-Title": {"Authorization": "SYNTHETIC"}},
        {"X-Title": 42},
        {"X-Title": ""},
        {"X-Title": "x" * 121},
        {"X-Title": "line\nbreak"},
        {"X-OpenRouter-Title": "control\u0007value"},
        {"HTTP-Referer": "ftp://example.invalid"},
        {"HTTP-Referer": "https://user@example.invalid"},
        {"HTTP-Referer": "https://example.invalid/path?source=synthetic"},
        {"HTTP-Referer": "https://example.invalid/path#fragment"},
        {"HTTP-Referer": "https:///missing-host"},
        {"HTTP-Referer": "https://example.invalid:0/path"},
        {"HTTP-Referer": "https://example.invalid:65536/path"},
        {"HTTP-Referer": "https://example.invalid/" + "x" * 4096},
        {
            "HTTP-Referer": "https://example.invalid",
            "X-OpenRouter-Title": "Synthetic",
            "X-Title": "Synthetic",
            "Unknown": "Synthetic",
        },
    )
    for headers in invalid_headers:
        assert not _provider_config_is_accepted(connection, default_headers=headers)


@pytest.mark.integration
def test_openrouter_routing_options_accept_official_typed_provider_object(
    migrated_autocommit_sync_connection: psycopg.Connection[Any],
) -> None:
    connection = migrated_autocommit_sync_connection
    complete_provider_object: dict[str, object] = {
        "order": ["openai", "anthropic"],
        "allow_fallbacks": True,
        "require_parameters": False,
        "data_collection": "deny",
        "zdr": True,
        "enforce_distillable_text": False,
        "only": ["openai"],
        "ignore": ["synthetic-provider"],
        "quantizations": [
            "int4",
            "int8",
            "fp4",
            "fp6",
            "fp8",
            "fp16",
            "bf16",
            "fp32",
            "unknown",
        ],
        "sort": {"by": "throughput", "partition": "model"},
        "preferred_min_throughput": {"p50": 0, "p75": 12.5, "p99": 40},
        "preferred_max_latency": 250,
        "max_price": {
            "audio": "0.5",
            "prompt": "0",
            "completion": "1.5",
            "request": "2",
            "image": "3",
        },
    }
    assert _provider_config_is_accepted(
        connection,
        routing_options=complete_provider_object,
    )

    valid_variants: tuple[object, ...] = (
        {},
        {"data_collection": "allow"},
        {"sort": "price"},
        {"sort": "throughput"},
        {"sort": "latency"},
        {"sort": "exacto"},
        {"sort": {"by": "latency"}},
        {"sort": {"by": "price", "partition": "none"}},
        {"sort": {"by": "exacto", "partition": "none"}},
        {"preferred_min_throughput": 0},
        {"preferred_max_latency": {"p90": 100}},
        {
            "order": None,
            "only": None,
            "ignore": None,
            "allow_fallbacks": None,
            "require_parameters": None,
            "zdr": None,
            "enforce_distillable_text": None,
            "data_collection": None,
            "quantizations": None,
            "sort": None,
            "preferred_min_throughput": None,
            "preferred_max_latency": None,
        },
        {"order": ["Amazon Bedrock", "Google AI Studio", "合成提供商"]},
        {"sort": {}},
        {"sort": {"partition": "none"}},
        {"sort": {"by": None, "partition": "model"}},
        {"sort": {"by": "exacto", "partition": None}},
        {"preferred_min_throughput": {}},
        {
            "preferred_min_throughput": {
                "p50": None,
                "p75": None,
                "p90": None,
                "p99": None,
            }
        },
        {"preferred_max_latency": {"p50": None, "p99": None}},
        {"max_price": {"prompt": "1", "completion": "2"}},
        {"max_price": {"audio": "0.5"}},
        {"max_price": {"request": "1e-6", "image": "1E+3"}},
        {"max_price": {}},
        {"order": [], "only": [], "ignore": [], "quantizations": []},
    )
    for routing_options in valid_variants:
        assert _provider_config_is_accepted(
            connection,
            routing_options=routing_options,
        )

    for provider_type in ("openai_compatible", "vendor_specific"):
        assert _provider_config_is_accepted(
            connection,
            provider_type=provider_type,
            routing_options={},
        )


@pytest.mark.integration
def test_routing_options_reject_unknown_keys_wrong_types_and_unsafe_values(
    migrated_autocommit_sync_connection: psycopg.Connection[Any],
) -> None:
    connection = migrated_autocommit_sync_connection

    for provider_type in ("openai_compatible", "vendor_specific"):
        assert not _provider_config_is_accepted(
            connection,
            provider_type=provider_type,
            routing_options={"order": ["openai"]},
        )

    unknown_keys = (
        "Authorization",
        "apiKey",
        "accessToken",
        "clientSecret",
        "privateKey",
        "accessKey",
        "monkey",
        "tokenizer",
        "secretary",
        "accessibility",
        "private",
        "client",
    )
    for key in unknown_keys:
        assert not _provider_config_is_accepted(
            connection,
            routing_options={key: "SYNTHETIC-VALUE"},
        ), f"routing_options accepted unknown key {key!r}"

    invalid_routing_options: tuple[object, ...] = (
        [],
        {"credentials": {"token": "SYNTHETIC"}},
        {"order": "openai"},
        {"order": [None]},
        {"order": [""]},
        {"order": ["line\nbreak"]},
        {"order": ["control\u0007value"]},
        {"order": ["x" * 256]},
        {"order": [f"provider-{index}" for index in range(101)]},
        {"only": 1},
        {"ignore": {}},
        {"allow_fallbacks": "true"},
        {"require_parameters": 1},
        {"zdr": {}},
        {"enforce_distillable_text": []},
        {"data_collection": "sometimes"},
        {"data_collection": False},
        {"quantizations": "fp16"},
        {"quantizations": [None]},
        {"quantizations": ["int2"]},
        {"quantizations": ["fp16"] * 33},
        {"sort": "random"},
        {"sort": []},
        {"sort": {"by": "price", "unknown": "synthetic"}},
        {"sort": {"by": "random"}},
        {"sort": {"by": False}},
        {"sort": {"by": "price", "partition": "unknown"}},
        {"sort": {"partition": 1}},
        {"preferred_min_throughput": -1},
        {"preferred_min_throughput": "fast"},
        {"preferred_min_throughput": {"p50": -1}},
        {"preferred_min_throughput": {"p50": "fast"}},
        {"preferred_min_throughput": {"p75": False}},
        {"preferred_min_throughput": {"p100": 1}},
        {"preferred_max_latency": False},
        {"preferred_max_latency": []},
        {"preferred_max_latency": {"p99": -1}},
        {"max_price": {"prompt": 1}},
        {"max_price": None},
        {"max_price": {"prompt": None}},
        {"max_price": {"prompt": "-1"}},
        {"max_price": {"prompt": ""}},
        {"max_price": {"prompt": "1 2"}},
        {"max_price": {"prompt": "NaN"}},
        {"max_price": {"prompt": "Infinity"}},
        {"max_price": {"completion": "free"}},
        {"max_price": {"completion": "Bearer synthetic"}},
        {"max_price": {"completion": "aaa.bbb.ccc"}},
        {"max_price": {"completion": "https://price.example.invalid"}},
        {"max_price": {"image": "1" * 65}},
        {"max_price": {"token": "1"}},
        {"max_price": {"request": {"token": "SYNTHETIC"}}},
        {f"unknown-{index}": True for index in range(50_000)},
    )
    for routing_options in invalid_routing_options:
        assert not _provider_config_is_accepted(
            connection,
            routing_options=routing_options,
        )


@pytest.mark.integration
def test_provider_usage_currency_database_check_is_case_sensitive(
    migrated_autocommit_sync_connection: psycopg.Connection[Any],
) -> None:
    connection = migrated_autocommit_sync_connection

    def currency_is_accepted(value: str) -> bool:
        try:
            connection.execute(
                """
                INSERT INTO provider_usage (
                    id, request_id, capability, provider_identifier,
                    model_identifier, currency, latency_ms, status
                ) VALUES (%s, %s, 'embedding', 'synthetic-provider',
                          'synthetic-model', %s, 1, 'succeeded')
                """,
                (uuid4(), f"request-{uuid4()}", value),
            )
        except psycopg.errors.CheckViolation:
            return False
        return True

    assert currency_is_accepted("USD")
    assert not currency_is_accepted("usd")
    assert not currency_is_accepted("Usd")
