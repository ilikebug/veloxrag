from __future__ import annotations

import asyncio
import socket
import ssl
import sys
import threading
import time
from collections.abc import Iterable
from dataclasses import FrozenInstanceError
from typing import Any, cast

import httpcore
import pytest

import rag_service.providers.network_policy as network_policy
import rag_service.providers.transport as provider_transport
from rag_service.config import Environment
from rag_service.providers.network_policy import (
    PROVIDER_ENDPOINT_POLICY_VERSION,
    CanonicalProviderEndpoint,
    ProviderEndpointPolicy,
    ProviderNetworkPolicyError,
    ResolvedProviderEndpoint,
    validate_provider_endpoint_url,
    validate_provider_headers,
)
from rag_service.providers.transport import (
    PinnedProviderNetworkBackend,
    ProviderDestination,
    ProviderHttpResponse,
    SecureProviderTransport,
)


class FakeResolver:
    def __init__(self, *answers: Iterable[str] | BaseException) -> None:
        self._answers = list(answers)
        self.calls: list[tuple[str, int]] = []

    def resolve(self, hostname: str, port: int) -> Iterable[str]:
        self.calls.append((hostname, port))
        answer = self._answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


class SlowResolver:
    def __init__(self, delay: float, answers: Iterable[str]) -> None:
        self.delay = delay
        self.answers = tuple(answers)
        self.calls: list[tuple[str, int]] = []

    def resolve(self, hostname: str, port: int) -> Iterable[str]:
        self.calls.append((hostname, port))
        time.sleep(self.delay)
        return self.answers


class TrackingSlowResolver:
    def __init__(self, delay: float, answers: Iterable[str]) -> None:
        self.delay = delay
        self.answers = tuple(answers)
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def resolve(self, hostname: str, port: int) -> Iterable[str]:
        del hostname, port
        with self._lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay)
            return self.answers
        finally:
            with self._lock:
                self.active -= 1


async def _heartbeat(stop: asyncio.Event, ticks: list[int]) -> None:
    while not stop.is_set():
        ticks[0] += 1
        await asyncio.sleep(0)


def _provider_dns_executor(*, max_workers: int) -> Any:
    executor_type = getattr(network_policy, "ProviderDnsExecutor", None)
    assert executor_type is not None
    return executor_type(max_workers=max_workers)


def _utf8_expanding_path(canonical_length: int) -> str:
    body_length = canonical_length - 1
    multibyte_count, ascii_count = divmod(body_length, 9)
    return "/" + "\u4f60" * multibyte_count + "a" * ascii_count


@pytest.mark.parametrize(
    ("raw_url", "expected_url", "hostname", "port", "path"),
    [
        (
            "HTTPS://Example.COM:443/v1/",
            "https://example.com/v1",
            "example.com",
            443,
            "/v1",
        ),
        (
            "https://Example.COM:8443/openai/v1",
            "https://example.com:8443/openai/v1",
            "example.com",
            8443,
            "/openai/v1",
        ),
        (
            "https://b\u00fccher.example/v1",
            "https://xn--bcher-kva.example/v1",
            "xn--bcher-kva.example",
            443,
            "/v1",
        ),
        (
            "https://[2606:4700:4700::1111]:443/v1/",
            "https://[2606:4700:4700::1111]/v1",
            "2606:4700:4700::1111",
            443,
            "/v1",
        ),
        ("https://example.com", "https://example.com", "example.com", 443, ""),
    ],
)
def test_network_policy_url_validation_returns_a_canonical_endpoint(
    raw_url: str,
    expected_url: str,
    hostname: str,
    port: int,
    path: str,
) -> None:
    endpoint = validate_provider_endpoint_url(raw_url)

    assert endpoint == CanonicalProviderEndpoint(
        url=expected_url,
        hostname=hostname,
        port=port,
        path=path,
        policy_version=PROVIDER_ENDPOINT_POLICY_VERSION,
    )
    assert 1 <= len(endpoint.policy_version) <= 64
    assert endpoint.policy_version.isascii()


@pytest.mark.parametrize(
    ("raw_path", "canonical_path"),
    [
        ("/%76%31", "/v1"),
        ("/caf\u00e9", "/caf%C3%A9"),
        ("/cafe\u0301", "/caf%C3%A9"),
        ("/caf%C3%A9", "/caf%C3%A9"),
        ("/%7euser", "/~user"),
        ("/%e2%9c%93", "/%E2%9C%93"),
    ],
)
def test_network_policy_path_has_one_utf8_nfc_percent_encoded_canonical_form(
    raw_path: str,
    canonical_path: str,
) -> None:
    endpoint = validate_provider_endpoint_url(f"https://provider.example{raw_path}")

    assert endpoint.path == canonical_path
    assert endpoint.url == f"https://provider.example{canonical_path}"


def test_network_policy_path_preserves_safe_literal_percent_without_double_decoding() -> None:
    endpoint = validate_provider_endpoint_url("https://provider.example/discount%2525off")

    assert endpoint.path == "/discount%2525off"


def test_network_policy_canonical_path_enforces_post_encoding_exact_boundary() -> None:
    exact_path = _utf8_expanding_path(1024)
    oversized_path = _utf8_expanding_path(1025)

    endpoint = validate_provider_endpoint_url(f"https://provider.example:443{exact_path}")

    assert len(endpoint.path) == 1024
    assert endpoint.url.startswith("https://provider.example/")
    assert ":443" not in endpoint.url
    with pytest.raises(ProviderNetworkPolicyError):
        validate_provider_endpoint_url(f"https://provider.example:443{oversized_path}")


def test_network_policy_ascii_path_length_boundary_does_not_regress() -> None:
    endpoint = validate_provider_endpoint_url("https://provider.example/" + "a" * 1023)

    assert len(endpoint.path) == 1024
    with pytest.raises(ProviderNetworkPolicyError):
        validate_provider_endpoint_url("https://provider.example/" + "a" * 1024)


def test_network_policy_canonical_url_enforces_post_assembly_exact_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(network_policy, "_MAX_PATH_LENGTH", 2048)
    raw_base_url = "https://b\u00fccher.example:8443"
    canonical_base_url = validate_provider_endpoint_url(raw_base_url).url
    exact_path = _utf8_expanding_path(2048 - len(canonical_base_url))
    oversized_path = _utf8_expanding_path(2049 - len(canonical_base_url))

    endpoint = validate_provider_endpoint_url(raw_base_url + exact_path)

    assert endpoint.hostname == "xn--bcher-kva.example"
    assert endpoint.port == 8443
    assert len(endpoint.url) == 2048
    with pytest.raises(ProviderNetworkPolicyError):
        validate_provider_endpoint_url(raw_base_url + oversized_path)


@pytest.mark.parametrize(
    "raw_url",
    [
        "",
        "http://provider.example/v1",
        "https://user@provider.example/v1",
        "https://user:password@provider.example/v1",
        "https://provider.example/v1?api_key=secret",
        "https://provider.example/v1?",
        "https://provider.example/v1#fragment",
        "https://provider.example/v1#",
        "https:///v1",
        "https://provider.example:0/v1",
        "https://provider.example:65536/v1",
        "https://provider.example:invalid/v1",
        "https://provider.example/v1\\models",
        "https://provider.example//v1",
        "https://provider.example/v1//models",
        "https://provider.example/v1/../admin",
        "https://provider.example/v1/%2e%2e/admin",
        "https://provider.example/v1/%2Fadmin",
        "https://provider.example/v1/%5cadmin",
        "https://provider.example/v1/%FF",
        "https://provider.example/%252e%252e/admin",
        "https://provider.example/%252fadmin",
        "https://provider.example/%255cadmin",
        "https://provider.example/%2500",
        "https://provider.example/%250d%250a",
        "https://provider.example/%257f",
        "https://provider.example/%253fambiguous",
        "https://provider.example/%2523ambiguous",
        "https://provider_example/v1",
        "https://[fe80::1%25en0]/v1",
    ],
)
def test_network_policy_url_validation_rejects_ambiguous_or_unsafe_urls(
    raw_url: str,
) -> None:
    with pytest.raises(ProviderNetworkPolicyError) as exc_info:
        validate_provider_endpoint_url(raw_url)

    assert exc_info.value.code == "PROVIDER_ENDPOINT_REJECTED"
    assert exc_info.value.args == ("Provider endpoint rejected",)
    if raw_url:
        assert raw_url not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_network_policy_url_validation_is_pure_and_does_not_resolve_dns() -> None:
    resolver = FakeResolver(["93.184.216.34"])
    policy = ProviderEndpointPolicy(
        environment=Environment.TEST,
        resolver=resolver,
    )

    endpoint = policy.validate_url("https://provider.example/v1")

    assert endpoint.hostname == "provider.example"
    assert resolver.calls == []


def test_network_policy_persistence_validation_resolves_all_current_answers() -> None:
    resolver = FakeResolver(
        [
            "2606:4700:4700::1111",
            "93.184.216.34",
            "93.184.216.34",
        ]
    )
    policy = ProviderEndpointPolicy(
        environment=Environment.TEST,
        resolver=resolver,
    )

    resolved = policy.validate_for_persistence("https://Provider.Example:443/v1/")

    assert resolved == ResolvedProviderEndpoint(
        endpoint=CanonicalProviderEndpoint(
            url="https://provider.example/v1",
            hostname="provider.example",
            port=443,
            path="/v1",
            policy_version=PROVIDER_ENDPOINT_POLICY_VERSION,
        ),
        addresses=("93.184.216.34", "2606:4700:4700::1111"),
    )
    assert resolver.calls == [("provider.example", 443)]


def test_network_policy_runtime_validation_repeats_resolution_for_every_connection() -> None:
    resolver = FakeResolver(
        ["93.184.216.34"],
        ["1.1.1.1"],
        ["8.8.8.8"],
    )
    policy = ProviderEndpointPolicy(
        environment=Environment.TEST,
        resolver=resolver,
    )
    persisted = policy.validate_for_persistence("https://provider.example:8443/v1")

    first_connection = policy.validate_for_connection(persisted.endpoint)
    second_connection = policy.validate_for_connection(persisted.endpoint)

    assert persisted.addresses == ("93.184.216.34",)
    assert first_connection.addresses == ("1.1.1.1",)
    assert second_connection.addresses == ("8.8.8.8",)
    assert resolver.calls == [
        ("provider.example", 8443),
        ("provider.example", 8443),
        ("provider.example", 8443),
    ]


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "::1",
        "10.0.0.1",
        "172.16.0.1",
        "192.168.0.1",
        "fc00::1",
        "169.254.1.1",
        "fe80::1",
        "224.0.0.1",
        "ff02::1",
        "240.0.0.1",
        "::",
        "0.0.0.0",
        "100.64.0.1",
        "169.254.169.254",
        "169.254.170.2",
        "100.100.100.200",
        "192.0.0.192",
        "fd00:ec2::254",
        "fec0::1",
        "2002:7f00:1::",
        "2002:a00:1::",
        "2002:808:808::",
        "2001:0000:4136:e378:8000:63bf:3fff:fdd2",
    ],
)
def test_network_policy_rejects_non_public_and_cloud_metadata_addresses(
    address: str,
) -> None:
    resolver = FakeResolver([address])
    policy = ProviderEndpointPolicy(
        environment=Environment.TEST,
        resolver=resolver,
    )

    with pytest.raises(ProviderNetworkPolicyError):
        policy.validate_for_persistence("https://provider.example/v1")


def test_network_policy_rejects_the_entire_resolution_when_any_answer_is_forbidden() -> None:
    resolver = FakeResolver(["93.184.216.34", "127.0.0.1", "2606:4700:4700::1111"])
    policy = ProviderEndpointPolicy(
        environment=Environment.TEST,
        resolver=resolver,
    )

    with pytest.raises(ProviderNetworkPolicyError):
        policy.validate_for_persistence("https://provider.example/v1")


@pytest.mark.parametrize(
    "answers",
    [
        [],
        ["not-an-ip-address"],
    ],
)
def test_network_policy_rejects_empty_or_malformed_resolver_answers(
    answers: list[str],
) -> None:
    policy = ProviderEndpointPolicy(
        environment=Environment.TEST,
        resolver=FakeResolver(answers),
    )

    with pytest.raises(ProviderNetworkPolicyError):
        policy.validate_for_persistence("https://provider.example/v1")


def test_network_policy_accepts_at_most_the_bounded_resolver_answer_limit() -> None:
    policy = ProviderEndpointPolicy(
        environment=Environment.TEST,
        resolver=FakeResolver(["93.184.216.34"] * 64),
    )

    resolved = policy.validate_for_persistence("https://provider.example/v1")

    assert resolved.addresses == ("93.184.216.34",)


def test_network_policy_rejects_resolver_answer_count_over_the_bound() -> None:
    policy = ProviderEndpointPolicy(
        environment=Environment.TEST,
        resolver=FakeResolver(["93.184.216.34"] * 65),
    )

    with pytest.raises(ProviderNetworkPolicyError):
        policy.validate_for_persistence("https://provider.example/v1")


def test_network_policy_stops_an_infinite_resolver_after_limit_plus_one_yields() -> None:
    yield_count = 0

    def infinite_answers() -> Iterable[str]:
        nonlocal yield_count
        while True:
            yield_count += 1
            yield "93.184.216.34"

    policy = ProviderEndpointPolicy(
        environment=Environment.TEST,
        resolver=FakeResolver(infinite_answers()),
    )

    with pytest.raises(ProviderNetworkPolicyError):
        policy.validate_for_persistence("https://provider.example/v1")

    assert yield_count == 65


def test_network_policy_rejects_partial_resolver_failure_without_source_leak() -> None:
    sentinel = "resolver-partial-failure-secret-sentinel"

    def partial_answers() -> Iterable[str]:
        yield "93.184.216.34"
        raise RuntimeError(sentinel)

    policy = ProviderEndpointPolicy(
        environment=Environment.TEST,
        resolver=FakeResolver(partial_answers()),
    )

    with pytest.raises(ProviderNetworkPolicyError) as exc_info:
        policy.validate_for_persistence("https://provider.example/v1")

    rendered = (str(exc_info.value), repr(exc_info.value), repr(exc_info.value.args))
    assert all(sentinel not in value for value in rendered)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_network_policy_sanitizes_resolver_failures() -> None:
    sentinel = "resolver-internal-secret-sentinel"
    resolver = FakeResolver(RuntimeError(sentinel))
    policy = ProviderEndpointPolicy(
        environment=Environment.TEST,
        resolver=resolver,
    )

    with pytest.raises(ProviderNetworkPolicyError) as exc_info:
        policy.validate_for_persistence("https://provider.example/v1")

    rendered = (str(exc_info.value), repr(exc_info.value), repr(exc_info.value.args))
    assert all(sentinel not in value for value in rendered)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.parametrize(
    "header_name",
    [
        "Host",
        "authorization",
        "Proxy",
        "Proxy-Authorization",
        "Proxy-Connection",
        "Forwarded",
        "Forwarded-For",
        "X-Forwarded",
        "X-Forwarded-For",
        "x-forwarded-host",
        "X-Forwarded-Proto",
        "X-Forwarded-Port",
        "X-Real-IP",
        "X-Proxy-User",
        "Via",
        "Connection",
        "Transfer-Encoding",
        "Content-Length",
        "Content-Type",
        "Cookie",
        "Upgrade",
        "X-Original-URL",
        "X-Rewrite-URL",
        "X-Custom-Provider-Header",
    ],
)
def test_network_policy_rejects_forbidden_provider_headers(header_name: str) -> None:
    with pytest.raises(ProviderNetworkPolicyError):
        validate_provider_headers({header_name: "header-value-secret-sentinel"})


def test_network_policy_accepts_non_sensitive_routing_headers() -> None:
    headers = {
        "x-title": "RAG fallback",
        "http-referer": "https://app.example",
        "X-OPENROUTER-TITLE": "RAG",
    }

    canonical = validate_provider_headers(headers)

    assert canonical == {
        "HTTP-Referer": "https://app.example",
        "X-OpenRouter-Title": "RAG",
        "X-Title": "RAG fallback",
    }
    assert list(canonical) == ["HTTP-Referer", "X-OpenRouter-Title", "X-Title"]


@pytest.mark.parametrize(
    "referer",
    [
        "HTTP://Example.COM",
        "https://provider.example:1/path",
        "http://provider.example:65535/path",
        "https://192.0.2.1:8443/path/to/app",
        "http://[2001:db8::1]:8080/ref/path",
    ],
)
def test_network_policy_accepts_db_compatible_http_referers_without_rewriting(
    referer: str,
) -> None:
    assert validate_provider_headers({"http-referer": referer}) == {"HTTP-Referer": referer}


def test_network_policy_http_referer_validation_does_not_resolve_dns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_dns_lookup(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("HTTP-Referer validation must stay pure")

    monkeypatch.setattr(socket, "getaddrinfo", unexpected_dns_lookup)

    assert validate_provider_headers(
        {"HTTP-Referer": "https://unresolved-provider.example/path"}
    ) == {"HTTP-Referer": "https://unresolved-provider.example/path"}


@pytest.mark.parametrize(
    "referer",
    [
        "not-a-url",
        "ftp://provider.example/path",
        "https://user@provider.example/path",
        "https://user:password@provider.example/path",
        "https:///path",
        "https://provider_example/path",
        "https://provider..example/path",
        "https://[v1.example]/path",
        "https://2001:db8::1/path",
        "https://provider.example:",
        "https://provider.example:0/path",
        "https://provider.example:000001/path",
        "https://provider.example:65536/path",
        "https://provider.example:99999/path",
        "https://provider.example:invalid/path",
        "https://provider.example/path?query=value",
        "https://provider.example/path?",
        "https://provider.example/path#fragment",
        "https://provider.example/path#",
        "https://provider.example/path with space",
        "https://provider.example/path\x1fcontrol",
    ],
)
def test_network_policy_rejects_http_referers_outside_the_db_contract(
    referer: str,
) -> None:
    with pytest.raises(ProviderNetworkPolicyError):
        validate_provider_headers({"HTTP-Referer": referer})


def test_network_policy_http_referer_failure_is_sanitized() -> None:
    sentinel = "https://provider.example/path?referer-secret-sentinel"

    with pytest.raises(ProviderNetworkPolicyError) as exc_info:
        validate_provider_headers({"HTTP-Referer": sentinel})

    assert sentinel not in str(exc_info.value)
    assert sentinel not in repr(exc_info.value.args)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.parametrize(
    "value",
    [
        "\u4e2d\u6587",
        "caf\u00e9",
        " leading-space",
        "trailing-space ",
        "",
        "nul\x00value",
        "delete\x7fvalue",
    ],
)
def test_network_policy_rejects_unsendable_or_ambiguous_header_values(value: str) -> None:
    with pytest.raises(ProviderNetworkPolicyError):
        validate_provider_headers({"X-Title": value})


def test_network_policy_header_value_byte_limits_have_exact_boundaries() -> None:
    referer_prefix = "https://provider.example/"
    exact_referer = referer_prefix + "a" * (2048 - len(referer_prefix))
    oversized_referer = exact_referer + "a"

    assert validate_provider_headers({"X-Title": "a" * 120}) == {"X-Title": "a" * 120}
    assert validate_provider_headers({"HTTP-Referer": exact_referer}) == {
        "HTTP-Referer": exact_referer
    }
    with pytest.raises(ProviderNetworkPolicyError):
        validate_provider_headers({"X-Title": "a" * 121})
    with pytest.raises(ProviderNetworkPolicyError):
        validate_provider_headers({"HTTP-Referer": oversized_referer})


def test_network_policy_oversized_header_is_rejected_before_ascii_encoding() -> None:
    value = "a" * (1024 * 1024)
    encoded_value = False

    def record_encode_call(
        _frame: object,
        event: str,
        argument: object,
    ) -> None:
        nonlocal encoded_value
        if (
            event == "c_call"
            and getattr(argument, "__name__", None) == "encode"
            and getattr(argument, "__self__", None) is value
        ):
            encoded_value = True

    sys.setprofile(record_encode_call)
    try:
        with pytest.raises(ProviderNetworkPolicyError):
            validate_provider_headers({"X-Title": value})
    finally:
        sys.setprofile(None)

    assert encoded_value is False


def test_network_policy_total_header_wire_byte_limit_is_4096_with_exact_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert network_policy._MAX_PROVIDER_HEADERS_TOTAL_BYTES == 4096
    canonical_names = ("X-OpenRouter-Title", "X-Title")
    wire_overhead = sum(len(name) + len(": ") + len("\r\n") for name in canonical_names)
    first_value_length = (4096 - wire_overhead) // 2
    second_value_length = 4096 - wire_overhead - first_value_length
    monkeypatch.setattr(
        network_policy,
        "_PROVIDER_HEADER_VALUE_LIMIT_BYTES",
        {
            "x-openrouter-title": 4096,
            "x-title": 4096,
        },
    )
    exact_headers = {
        "X-OpenRouter-Title": "a" * first_value_length,
        "X-Title": "b" * second_value_length,
    }

    assert validate_provider_headers(exact_headers) == exact_headers

    with pytest.raises(ProviderNetworkPolicyError):
        validate_provider_headers(
            {
                **exact_headers,
                "X-Title": exact_headers["X-Title"] + "b",
            }
        )


def test_network_policy_header_encoding_failure_is_sanitized() -> None:
    sentinel = "\u79d8\u5bc6-header-sentinel"

    with pytest.raises(ProviderNetworkPolicyError) as exc_info:
        validate_provider_headers({"X-Title": sentinel})

    assert sentinel not in str(exc_info.value)
    assert sentinel not in repr(exc_info.value.args)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_network_policy_rejects_case_insensitive_duplicate_allowed_headers() -> None:
    with pytest.raises(ProviderNetworkPolicyError):
        validate_provider_headers(
            {
                "X-Title": "first",
                "x-title": "second",
            }
        )


@pytest.mark.parametrize(
    "headers",
    [
        {"Bad Header": "value"},
        {"X-Test": "line-one\r\nX-Injected: yes"},
        {"": "value"},
    ],
)
def test_network_policy_rejects_malformed_header_names_and_values(
    headers: dict[str, str],
) -> None:
    with pytest.raises(ProviderNetworkPolicyError):
        validate_provider_headers(headers)


def test_network_policy_local_override_allows_private_addresses() -> None:
    local_policy = ProviderEndpointPolicy(
        environment=Environment.LOCAL,
        allow_private_targets=True,
        resolver=FakeResolver(["127.0.0.1", "::1"]),
    )

    resolved = local_policy.validate_for_persistence("https://localhost:8443/v1")

    assert resolved.addresses == ("127.0.0.1", "::1")


@pytest.mark.parametrize("link_local_address", ["169.254.1.1", "fe80::1"])
def test_network_policy_local_override_never_allows_link_local_addresses(
    link_local_address: str,
) -> None:
    policy = ProviderEndpointPolicy(
        environment=Environment.TEST,
        allow_private_targets=True,
        resolver=FakeResolver([link_local_address]),
    )

    with pytest.raises(ProviderNetworkPolicyError):
        policy.validate_for_persistence("https://link-local.example/v1")


@pytest.mark.parametrize(
    "metadata_address",
    ["169.254.169.254", "::ffff:169.254.169.254", "fd00:ec2::254"],
)
def test_network_policy_local_override_never_allows_metadata_addresses(
    metadata_address: str,
) -> None:
    metadata_policy = ProviderEndpointPolicy(
        environment=Environment.TEST,
        allow_private_targets=True,
        resolver=FakeResolver([metadata_address]),
    )
    with pytest.raises(ProviderNetworkPolicyError):
        metadata_policy.validate_for_persistence("https://metadata.example/v1")


@pytest.mark.parametrize(
    "reserved_address",
    ["240.0.0.1", "255.255.255.254", "::ffff:240.0.0.1"],
)
def test_network_policy_local_override_never_allows_reserved_addresses(
    reserved_address: str,
) -> None:
    policy = ProviderEndpointPolicy(
        environment=Environment.TEST,
        allow_private_targets=True,
        resolver=FakeResolver([reserved_address]),
    )

    with pytest.raises(ProviderNetworkPolicyError):
        policy.validate_for_persistence("https://reserved.example/v1")


def test_network_policy_production_refuses_private_target_override() -> None:
    with pytest.raises(ProviderNetworkPolicyError) as exc_info:
        ProviderEndpointPolicy(
            environment=Environment.PRODUCTION,
            allow_private_targets=True,
            resolver=FakeResolver(["127.0.0.1"]),
        )

    assert exc_info.value.code == "PROVIDER_ENDPOINT_REJECTED"
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_network_policy_values_are_immutable_and_have_safe_representations() -> None:
    endpoint = validate_provider_endpoint_url(
        "https://sensitive-provider.example/v1/tenant-token-sentinel"
    )
    resolved = ResolvedProviderEndpoint(
        endpoint=endpoint,
        addresses=("93.184.216.34", "2606:4700:4700::1111"),
    )

    with pytest.raises(FrozenInstanceError):
        endpoint.url = "https://other.example"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        resolved.addresses = ("1.1.1.1",)  # type: ignore[misc]

    assert repr(endpoint) == "CanonicalProviderEndpoint(<redacted>)"
    assert repr(resolved) == "ResolvedProviderEndpoint(<redacted>)"
    for sentinel in (
        "sensitive-provider.example",
        "tenant-token-sentinel",
        "93.184.216.34",
        "2606:4700:4700::1111",
    ):
        assert sentinel not in repr(endpoint)
        assert sentinel not in repr(resolved)


class _FakeNetworkStream(httpcore.AsyncNetworkStream):
    async def read(  # noqa: ASYNC109 -- httpcore test double interface
        self,
        max_bytes: int,
        timeout: float | None = None,  # noqa: ASYNC109 -- httpcore interface
    ) -> bytes:
        del max_bytes, timeout
        return b""

    async def write(  # noqa: ASYNC109 -- httpcore test double interface
        self,
        buffer: bytes,
        timeout: float | None = None,  # noqa: ASYNC109 -- httpcore interface
    ) -> None:
        del buffer, timeout

    async def aclose(self) -> None:
        return None

    async def start_tls(  # noqa: ASYNC109 -- httpcore test double interface
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,  # noqa: ASYNC109 -- httpcore interface
    ) -> _FakeNetworkStream:
        del ssl_context, server_hostname, timeout
        return self

    def get_extra_info(self, info: str) -> object | None:
        del info
        return None


class _RecordingNetworkBackend(httpcore.AsyncNetworkBackend):
    def __init__(self) -> None:
        self.tcp_calls: list[tuple[str, int, float | None, str | None]] = []

    async def connect_tcp(  # noqa: ASYNC109 -- httpcore test double interface
        self,
        host: str,
        port: int,
        timeout: float | None = None,  # noqa: ASYNC109 -- httpcore interface
        local_address: str | None = None,
        socket_options: object = None,
    ) -> _FakeNetworkStream:
        del socket_options
        self.tcp_calls.append((host, port, timeout, local_address))
        return _FakeNetworkStream()

    async def connect_unix_socket(  # noqa: ASYNC109 -- httpcore test double interface
        self,
        path: str,
        timeout: float | None = None,  # noqa: ASYNC109 -- httpcore interface
        socket_options: object = None,
    ) -> _FakeNetworkStream:
        del path, timeout, socket_options
        raise AssertionError("Provider transport must not use Unix sockets")

    async def sleep(self, seconds: float) -> None:
        del seconds


def test_pinned_network_backend_uses_public_anyio_backend_by_default() -> None:
    policy = ProviderEndpointPolicy(
        environment=Environment.TEST,
        resolver=FakeResolver(["93.184.216.34"]),
    )
    resolved = policy.validate_for_persistence("https://provider.example/v1")

    backend = PinnedProviderNetworkBackend(
        policy=policy,
        destination=ProviderDestination(
            resolved=resolved,
            selected_address="93.184.216.34",
        ),
    )

    assert isinstance(backend._delegate, httpcore.AnyIOBackend)


def test_default_httpcore_backend_factory_fails_explicitly_on_unsupported_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(httpcore, "__version__", "2.0.0")

    with pytest.raises(ProviderNetworkPolicyError):
        provider_transport._default_async_network_backend()


def test_default_httpcore_backend_factory_fails_explicitly_on_interface_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class IncompatibleBackend:
        pass

    monkeypatch.setattr(httpcore, "AnyIOBackend", IncompatibleBackend)

    with pytest.raises(ProviderNetworkPolicyError):
        provider_transport._default_async_network_backend()


@pytest.mark.asyncio
async def test_async_connection_validation_does_not_block_event_loop() -> None:
    policy = ProviderEndpointPolicy(
        environment=Environment.TEST,
        resolver=SlowResolver(0.05, ["93.184.216.34"]),
    )
    endpoint = policy.validate_url("https://provider.example/v1")
    stop = asyncio.Event()
    ticks = [0]
    heartbeat = asyncio.create_task(_heartbeat(stop, ticks))
    try:
        await asyncio.sleep(0)
        resolved = await policy.validate_for_connection_async(
            endpoint,
            timeout_seconds=0.5,
        )
    finally:
        stop.set()
        await heartbeat

    assert resolved.addresses == ("93.184.216.34",)
    assert ticks[0] > 2


@pytest.mark.asyncio
async def test_async_dns_uses_dedicated_executor_instead_of_default_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = _provider_dns_executor(max_workers=2)
    policy = ProviderEndpointPolicy(
        environment=Environment.TEST,
        resolver=FakeResolver(["93.184.216.34"]),
    )
    endpoint = policy.validate_url("https://provider.example/v1")

    async def forbidden_to_thread(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("default executor must not be used for provider DNS")

    monkeypatch.setattr(asyncio, "to_thread", forbidden_to_thread)
    try:
        resolved = await policy.validate_for_connection_async(
            endpoint,
            timeout_seconds=0.5,
            dns_executor=executor,
        )
    finally:
        await executor.aclose()

    assert resolved.addresses == ("93.184.216.34",)
    assert executor.closed is True


@pytest.mark.asyncio
async def test_async_dns_capacity_remains_leased_until_slow_resolver_finishes() -> None:
    # The 100 gathered tasks must all time out while the first three
    # resolutions are still running, otherwise a second batch legitimately
    # leases capacity and `calls` exceeds the executor width. A 0.5s resolver
    # keeps that margin under load instead of assuming the gather finishes
    # within 50ms.
    resolver = TrackingSlowResolver(0.5, ["93.184.216.34"])
    executor = _provider_dns_executor(max_workers=3)
    policy = ProviderEndpointPolicy(environment=Environment.TEST, resolver=resolver)
    endpoint = policy.validate_url("https://provider.example/v1")
    stop = asyncio.Event()
    ticks = [0]
    heartbeat = asyncio.create_task(_heartbeat(stop, ticks))
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    unhandled: list[dict[str, object]] = []
    loop.set_exception_handler(lambda _loop, context: unhandled.append(dict(context)))
    try:
        tasks = [
            asyncio.create_task(
                policy.validate_for_connection_async(
                    endpoint,
                    timeout_seconds=0.005,
                    dns_executor=executor,
                )
            )
            for _ in range(100)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        await executor.aclose()
        await asyncio.sleep(0)
    finally:
        stop.set()
        await heartbeat
        loop.set_exception_handler(previous_handler)
        if not executor.closed:
            await executor.aclose()

    assert all(isinstance(result, TimeoutError) for result in results)
    assert resolver.calls <= 3
    assert resolver.max_active <= 3
    assert resolver.active == 0
    assert ticks[0] > 2
    assert unhandled == []


@pytest.mark.asyncio
async def test_transport_closes_owned_dns_executor_but_not_external_shared_executor() -> None:
    external = _provider_dns_executor(max_workers=1)
    external_policy = ProviderEndpointPolicy(
        environment=Environment.TEST,
        resolver=FakeResolver(["93.184.216.34"]),
    )
    external_transport = SecureProviderTransport(
        policy=external_policy,
        pool_factory=_RecordingProviderPoolFactory(),
        dns_executor=external,
    )
    owned_policy = ProviderEndpointPolicy(
        environment=Environment.TEST,
        resolver=FakeResolver(["93.184.216.34"]),
    )
    owned_transport = SecureProviderTransport(
        policy=owned_policy,
        pool_factory=_RecordingProviderPoolFactory(),
    )
    owned = owned_transport._dns_executor
    try:
        await external_transport.aclose()
        await owned_transport.aclose()

        assert external.closed is False
        assert owned.closed is True
        resolved = await external_policy.validate_for_connection_async(
            external_policy.validate_url("https://provider.example/v1"),
            timeout_seconds=0.5,
            dns_executor=external,
        )
        assert resolved.addresses == ("93.184.216.34",)
    finally:
        await external.aclose()


@pytest.mark.asyncio
async def test_pinned_backend_dns_is_inside_connect_deadline_and_nonblocking() -> None:
    endpoint = validate_provider_endpoint_url("https://provider.example/v1")
    resolved = ResolvedProviderEndpoint(
        endpoint=endpoint,
        addresses=("93.184.216.34",),
    )
    policy = ProviderEndpointPolicy(
        environment=Environment.TEST,
        resolver=SlowResolver(0.05, ["93.184.216.34"]),
    )
    delegate = _RecordingNetworkBackend()
    backend = PinnedProviderNetworkBackend(
        policy=policy,
        destination=ProviderDestination(
            resolved=resolved,
            selected_address="93.184.216.34",
        ),
        delegate=delegate,
    )
    stop = asyncio.Event()
    ticks = [0]
    heartbeat = asyncio.create_task(_heartbeat(stop, ticks))
    try:
        await asyncio.sleep(0)
        with pytest.raises(httpcore.ConnectTimeout):
            await backend.connect_tcp("provider.example", 443, timeout=0.01)
    finally:
        stop.set()
        await heartbeat

    assert ticks[0] > 2
    assert delegate.tcp_calls == []


@pytest.mark.asyncio
async def test_pinned_network_backend_revalidates_immediately_before_each_new_connection() -> None:
    resolver = FakeResolver(["93.184.216.34"], ["93.184.216.34"])
    policy = ProviderEndpointPolicy(
        environment=Environment.TEST,
        resolver=resolver,
    )
    resolved = policy.validate_for_persistence("https://provider.example:8443/v1")
    destination = ProviderDestination(
        resolved=resolved,
        selected_address="93.184.216.34",
    )
    delegate = _RecordingNetworkBackend()
    backend = PinnedProviderNetworkBackend(
        policy=policy,
        destination=destination,
        delegate=delegate,
    )

    stream = await backend.connect_tcp(
        "provider.example",
        8443,
        timeout=2.5,
        local_address=None,
    )

    assert isinstance(stream, _FakeNetworkStream)
    assert resolver.calls == [
        ("provider.example", 8443),
        ("provider.example", 8443),
    ]
    assert len(delegate.tcp_calls) == 1
    host, port, remaining_timeout, local_address = delegate.tcp_calls[0]
    assert (host, port, local_address) == ("93.184.216.34", 8443, None)
    assert remaining_timeout is not None
    assert 0 < remaining_timeout <= 2.5


@pytest.mark.asyncio
async def test_pinned_network_backend_fails_closed_when_dns_rebinds_before_connect() -> None:
    resolver = FakeResolver(["93.184.216.34"], ["127.0.0.1"])
    policy = ProviderEndpointPolicy(
        environment=Environment.TEST,
        resolver=resolver,
    )
    resolved = policy.validate_for_persistence("https://provider.example/v1")
    delegate = _RecordingNetworkBackend()
    backend = PinnedProviderNetworkBackend(
        policy=policy,
        destination=ProviderDestination(
            resolved=resolved,
            selected_address="93.184.216.34",
        ),
        delegate=delegate,
    )

    with pytest.raises(ProviderNetworkPolicyError):
        await backend.connect_tcp("provider.example", 443)

    assert delegate.tcp_calls == []


@pytest.mark.parametrize(
    ("host", "port"),
    [("other.example", 443), ("provider.example", 8443)],
)
@pytest.mark.asyncio
async def test_pinned_network_backend_rejects_pool_origin_mismatch(
    host: str,
    port: int,
) -> None:
    policy = ProviderEndpointPolicy(
        environment=Environment.TEST,
        resolver=FakeResolver(["93.184.216.34"]),
    )
    resolved = policy.validate_for_persistence("https://provider.example/v1")
    delegate = _RecordingNetworkBackend()
    backend = PinnedProviderNetworkBackend(
        policy=policy,
        destination=ProviderDestination(
            resolved=resolved,
            selected_address="93.184.216.34",
        ),
        delegate=delegate,
    )

    with pytest.raises(ProviderNetworkPolicyError):
        await backend.connect_tcp(host, port)

    assert delegate.tcp_calls == []


class _RecordingProviderPool:
    def __init__(self, destination: ProviderDestination) -> None:
        self.destination = destination
        self.calls: list[dict[str, object]] = []
        self.close_calls = 0
        self.closed = asyncio.Event()

    async def post_json(
        self,
        *,
        path: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> ProviderHttpResponse:
        self.calls.append(
            {
                "path": path,
                "headers": dict(headers),
                "payload": dict(payload),
                "timeout_seconds": timeout_seconds,
            }
        )
        return ProviderHttpResponse(status_code=200, headers={}, body=b"{}")

    async def aclose(self) -> None:
        self.close_calls += 1
        self.closed.set()


class _RecordingProviderPoolFactory:
    def __init__(self) -> None:
        self.destinations: list[ProviderDestination] = []
        self.pools: list[_RecordingProviderPool] = []
        self.ssl_contexts: list[ssl.SSLContext] = []

    def create(
        self,
        *,
        destination: ProviderDestination,
        ssl_context: ssl.SSLContext,
    ) -> _RecordingProviderPool:
        self.destinations.append(destination)
        self.ssl_contexts.append(ssl_context)
        pool = _RecordingProviderPool(destination)
        self.pools.append(pool)
        return pool


class _BlockingProviderPool(_RecordingProviderPool):
    def __init__(self, destination: ProviderDestination) -> None:
        super().__init__(destination)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.close_error: BaseException | None = None

    async def post_json(
        self,
        *,
        path: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> ProviderHttpResponse:
        self.calls.append(
            {
                "path": path,
                "headers": dict(headers),
                "payload": dict(payload),
                "timeout_seconds": timeout_seconds,
            }
        )
        self.entered.set()
        await self.release.wait()
        return ProviderHttpResponse(status_code=200, headers={}, body=b"{}")

    async def aclose(self) -> None:
        await super().aclose()
        self.closed.set()
        if self.close_error is not None:
            raise self.close_error


class _FirstPoolBlockingFactory(_RecordingProviderPoolFactory):
    def __init__(self) -> None:
        super().__init__()
        self.created: asyncio.Queue[_RecordingProviderPool] = asyncio.Queue()

    def create(
        self,
        *,
        destination: ProviderDestination,
        ssl_context: ssl.SSLContext,
    ) -> _RecordingProviderPool:
        self.destinations.append(destination)
        self.ssl_contexts.append(ssl_context)
        pool: _RecordingProviderPool
        if not self.pools:
            pool = _BlockingProviderPool(destination)
        else:
            pool = _RecordingProviderPool(destination)
        self.pools.append(pool)
        self.created.put_nowait(pool)
        return pool


class _ChunkedRawResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self.status = 200
        self.headers = [(b"x-request-id", b"upstream-large-response-secret-sentinel")]
        self.content = b""
        self._chunks = chunks
        self.consumed_chunks = 0
        self.closed = False

    async def aiter_stream(self) -> Any:
        for chunk in self._chunks:
            self.consumed_chunks += 1
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


class _RawStreamContext:
    def __init__(self, response: _ChunkedRawResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _ChunkedRawResponse:
        return self._response

    async def __aexit__(self, *args: object) -> None:
        del args
        await self._response.aclose()


class _ChunkedRawPool:
    def __init__(self, response: _ChunkedRawResponse) -> None:
        self.response = response
        self.request_calls = 0
        self.stream_calls = 0

    async def request(self, **kwargs: object) -> _ChunkedRawResponse:
        del kwargs
        self.request_calls += 1
        buffered: list[bytes] = []
        async for chunk in self.response.aiter_stream():
            buffered.append(chunk)
        self.response.content = b"".join(buffered)
        buffered.clear()
        return self.response

    def stream(self, **kwargs: object) -> _RawStreamContext:
        del kwargs
        self.stream_calls += 1
        return _RawStreamContext(self.response)


def _httpcore_provider_pool_with(raw_pool: _ChunkedRawPool) -> Any:
    policy = ProviderEndpointPolicy(
        environment=Environment.TEST,
        resolver=FakeResolver(["93.184.216.34"]),
    )
    resolved = policy.validate_for_persistence("https://provider.example/v1")
    pool = cast(
        Any,
        provider_transport._HttpcoreProviderPool.__new__(provider_transport._HttpcoreProviderPool),
    )
    pool._destination = ProviderDestination(
        resolved=resolved,
        selected_address="93.184.216.34",
    )
    pool._pool = raw_pool
    return pool


@pytest.mark.asyncio
async def test_secure_transport_dns_is_inside_request_deadline_and_nonblocking() -> None:
    resolver = SlowResolver(0.05, ["93.184.216.34"])
    policy = ProviderEndpointPolicy(environment=Environment.TEST, resolver=resolver)
    endpoint = policy.validate_url("https://provider.example/v1")
    factory = _RecordingProviderPoolFactory()
    transport = SecureProviderTransport(policy=policy, pool_factory=factory)
    stop = asyncio.Event()
    ticks = [0]
    heartbeat = asyncio.create_task(_heartbeat(stop, ticks))
    try:
        await asyncio.sleep(0)
        with pytest.raises(httpcore.ConnectTimeout):
            await transport.post_json(
                endpoint=endpoint,
                path="/embeddings",
                headers={"Content-Type": "application/json"},
                payload={"input": ["query"], "model": "embedding-test"},
                timeout_seconds=0.01,
            )
    finally:
        stop.set()
        await heartbeat
        await transport.aclose()

    assert ticks[0] > 2
    assert factory.pools == []


@pytest.mark.asyncio
async def test_httpcore_pool_streams_and_aborts_oversized_response_without_buffering_all() -> None:
    chunk = b"x" * (1024 * 1024)
    raw_response = _ChunkedRawResponse([chunk] * 18)
    raw_pool = _ChunkedRawPool(raw_response)
    pool = _httpcore_provider_pool_with(raw_pool)

    with pytest.raises(Exception) as exc_info:
        await pool.post_json(
            path="/embeddings",
            headers={"Content-Type": "application/json"},
            payload={"input": ["query"], "model": "embedding-test"},
            timeout_seconds=3.0,
        )

    assert raw_pool.request_calls == 0
    assert raw_pool.stream_calls == 1
    assert raw_response.consumed_chunks == 17
    assert raw_response.closed is True
    rendered = str(exc_info.value) + repr(exc_info.value)
    assert "upstream-large-response-secret-sentinel" not in rendered


@pytest.mark.asyncio
async def test_secure_transport_isolates_pools_by_complete_validated_destination_tuple() -> None:
    resolver = FakeResolver(
        ["93.184.216.34"],
        ["93.184.216.34"],
        ["1.1.1.1"],
    )
    policy = ProviderEndpointPolicy(environment=Environment.TEST, resolver=resolver)
    endpoint = policy.validate_url("https://provider.example/v1")
    factory = _RecordingProviderPoolFactory()
    transport = SecureProviderTransport(policy=policy, pool_factory=factory)

    try:
        for _ in range(3):
            response = await transport.post_json(
                endpoint=endpoint,
                path="/embeddings",
                headers={"Content-Type": "application/json"},
                payload={"input": ["query"], "model": "embedding-test"},
                timeout_seconds=3.0,
            )
            assert response.status_code == 200
    finally:
        await transport.aclose()

    assert len(factory.pools) == 2
    assert factory.destinations[0].resolved.addresses == ("93.184.216.34",)
    assert factory.destinations[1].resolved.addresses == ("1.1.1.1",)
    assert len(factory.pools[0].calls) == 2
    assert len(factory.pools[1].calls) == 1
    assert [pool.close_calls for pool in factory.pools] == [1, 1]


@pytest.mark.asyncio
async def test_secure_transport_bounds_destination_pools_with_lru_eviction_and_closes_victim() -> (
    None
):
    resolver = FakeResolver(
        ["93.184.216.34"],
        ["1.1.1.1"],
        ["93.184.216.34"],
        ["8.8.8.8"],
    )
    policy = ProviderEndpointPolicy(environment=Environment.TEST, resolver=resolver)
    endpoint = policy.validate_url("https://provider.example/v1")
    factory = _RecordingProviderPoolFactory()
    transport = SecureProviderTransport(
        policy=policy,
        pool_factory=factory,
        max_destination_pools=2,
    )

    try:
        for _ in range(4):
            await transport.post_json(
                endpoint=endpoint,
                path="/embeddings",
                headers={"Content-Type": "application/json"},
                payload={"input": ["query"], "model": "embedding-test"},
                timeout_seconds=3.0,
            )

        assert len(factory.pools) == 3
        assert factory.destinations[0].resolved.addresses == ("93.184.216.34",)
        assert factory.destinations[1].resolved.addresses == ("1.1.1.1",)
        assert factory.destinations[2].resolved.addresses == ("8.8.8.8",)
        assert [pool.close_calls for pool in factory.pools] == [0, 1, 0]
        assert len(transport._pools) == 2
    finally:
        await transport.aclose()

    assert [pool.close_calls for pool in factory.pools] == [1, 1, 1]


@pytest.mark.asyncio
async def test_destination_pool_waits_for_active_lru_lease_before_eviction() -> None:
    resolver = FakeResolver(["93.184.216.34"], ["1.1.1.1"])
    policy = ProviderEndpointPolicy(environment=Environment.TEST, resolver=resolver)
    endpoint = policy.validate_url("https://provider.example/v1")
    factory = _FirstPoolBlockingFactory()
    transport = SecureProviderTransport(
        policy=policy,
        pool_factory=factory,
        max_destination_pools=1,
    )
    first = asyncio.create_task(
        transport.post_json(
            endpoint=endpoint,
            path="/embeddings",
            headers={"Content-Type": "application/json"},
            payload={"input": ["first"], "model": "embedding-test"},
            timeout_seconds=3.0,
        )
    )
    second: asyncio.Task[ProviderHttpResponse] | None = None
    first_pool: _BlockingProviderPool | None = None
    try:
        created = await factory.created.get()
        assert isinstance(created, _BlockingProviderPool)
        first_pool = created
        await first_pool.entered.wait()
        second = asyncio.create_task(
            transport.post_json(
                endpoint=endpoint,
                path="/embeddings",
                headers={"Content-Type": "application/json"},
                payload={"input": ["second"], "model": "embedding-test"},
                timeout_seconds=3.0,
            )
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert second.done() is False
        assert first_pool.close_calls == 0
        assert len(factory.pools) == 1
        assert len(transport._pools) == 1

        first_pool.release.set()
        assert (await first).status_code == 200
        async with asyncio.timeout(1):
            assert (await second).status_code == 200

        assert len(factory.pools) == 2
        assert [pool.close_calls for pool in factory.pools] == [1, 0]
        assert len(transport._pools) == 1
    finally:
        if first_pool is not None:
            first_pool.release.set()
        for task in (first, second):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *(task for task in (first, second) if task is not None),
            return_exceptions=True,
        )
        await transport.aclose()


@pytest.mark.asyncio
async def test_cancelling_destination_pool_waiter_leaves_bounded_reusable_state() -> None:
    resolver = FakeResolver(["93.184.216.34"], ["1.1.1.1"], ["8.8.8.8"])
    policy = ProviderEndpointPolicy(environment=Environment.TEST, resolver=resolver)
    endpoint = policy.validate_url("https://provider.example/v1")
    factory = _FirstPoolBlockingFactory()
    transport = SecureProviderTransport(
        policy=policy,
        pool_factory=factory,
        max_destination_pools=1,
    )
    first = asyncio.create_task(
        transport.post_json(
            endpoint=endpoint,
            path="/embeddings",
            headers={"Content-Type": "application/json"},
            payload={"input": ["first"], "model": "embedding-test"},
            timeout_seconds=3.0,
        )
    )
    waiting: asyncio.Task[ProviderHttpResponse] | None = None
    first_pool: _BlockingProviderPool | None = None
    try:
        created = await factory.created.get()
        assert isinstance(created, _BlockingProviderPool)
        first_pool = created
        await first_pool.entered.wait()
        waiting = asyncio.create_task(
            transport.post_json(
                endpoint=endpoint,
                path="/embeddings",
                headers={"Content-Type": "application/json"},
                payload={"input": ["cancelled"], "model": "embedding-test"},
                timeout_seconds=3.0,
            )
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert waiting.done() is False
        assert len(factory.pools) == 1

        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert first_pool.close_calls == 0
        assert len(factory.pools) == 1
        assert len(transport._pools) == 1

        first_pool.release.set()
        assert (await first).status_code == 200
        async with asyncio.timeout(1):
            response = await transport.post_json(
                endpoint=endpoint,
                path="/embeddings",
                headers={"Content-Type": "application/json"},
                payload={"input": ["third"], "model": "embedding-test"},
                timeout_seconds=3.0,
            )
        assert response.status_code == 200
        assert len(factory.pools) == 2
        assert [pool.close_calls for pool in factory.pools] == [1, 0]
        assert len(transport._pools) == 1
    finally:
        if first_pool is not None:
            first_pool.release.set()
        for task in (first, waiting):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *(task for task in (first, waiting) if task is not None),
            return_exceptions=True,
        )
        await transport.aclose()


@pytest.mark.asyncio
async def test_transport_aclose_closes_idle_pool_and_waits_for_active_pool_release() -> None:
    resolver = FakeResolver(["93.184.216.34"], ["1.1.1.1"])
    policy = ProviderEndpointPolicy(environment=Environment.TEST, resolver=resolver)
    endpoint = policy.validate_url("https://provider.example/v1")
    factory = _FirstPoolBlockingFactory()
    transport = SecureProviderTransport(
        policy=policy,
        pool_factory=factory,
        max_destination_pools=2,
    )
    active_request = asyncio.create_task(
        transport.post_json(
            endpoint=endpoint,
            path="/embeddings",
            headers={"Content-Type": "application/json"},
            payload={"input": ["active"], "model": "embedding-test"},
            timeout_seconds=3.0,
        )
    )
    close_task: asyncio.Task[None] | None = None
    active_pool: _BlockingProviderPool | None = None
    try:
        created = await factory.created.get()
        assert isinstance(created, _BlockingProviderPool)
        active_pool = created
        await active_pool.entered.wait()
        idle_response = await transport.post_json(
            endpoint=endpoint,
            path="/embeddings",
            headers={"Content-Type": "application/json"},
            payload={"input": ["idle"], "model": "embedding-test"},
            timeout_seconds=3.0,
        )
        assert idle_response.status_code == 200
        assert len(factory.pools) == 2

        close_task = asyncio.create_task(transport.aclose())
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert close_task.done() is False
        assert active_pool.close_calls == 0
        async with asyncio.timeout(1):
            await factory.pools[1].closed.wait()
        assert factory.pools[1].close_calls == 1

        active_pool.release.set()
        assert (await active_request).status_code == 200
        async with asyncio.timeout(1):
            await close_task

        assert [pool.close_calls for pool in factory.pools] == [1, 1]
        assert len(transport._pools) == 0
    finally:
        if active_pool is not None:
            active_pool.release.set()
        if not active_request.done():
            active_request.cancel()
        if close_task is not None and not close_task.done():
            close_task.cancel()
        await asyncio.gather(
            active_request,
            *(task for task in (close_task,) if task is not None),
            return_exceptions=True,
        )
        await transport.aclose()


@pytest.mark.asyncio
async def test_cancelled_aclose_caller_does_not_abandon_active_pool_cleanup() -> None:
    resolver = FakeResolver(["93.184.216.34"])
    policy = ProviderEndpointPolicy(environment=Environment.TEST, resolver=resolver)
    endpoint = policy.validate_url("https://provider.example/v1")
    factory = _FirstPoolBlockingFactory()
    transport = SecureProviderTransport(
        policy=policy,
        pool_factory=factory,
        max_destination_pools=1,
    )
    active_request = asyncio.create_task(
        transport.post_json(
            endpoint=endpoint,
            path="/embeddings",
            headers={"Content-Type": "application/json"},
            payload={"input": ["active"], "model": "embedding-test"},
            timeout_seconds=3.0,
        )
    )
    close_caller: asyncio.Task[None] | None = None
    active_pool: _BlockingProviderPool | None = None
    loop = asyncio.get_running_loop()
    original_exception_handler = loop.get_exception_handler()
    unhandled: list[dict[str, Any]] = []
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
    try:
        created = await factory.created.get()
        assert isinstance(created, _BlockingProviderPool)
        active_pool = created
        await active_pool.entered.wait()

        close_caller = asyncio.create_task(transport.aclose())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert close_caller.done() is False
        tracked_close = transport._close_task
        assert isinstance(tracked_close, asyncio.Task)

        close_caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close_caller
        assert active_pool.close_calls == 0

        active_pool.release.set()
        assert (await active_request).status_code == 200
        async with asyncio.timeout(1):
            await active_pool.closed.wait()
        async with asyncio.timeout(1):
            await asyncio.shield(tracked_close)

        assert active_pool.close_calls == 1
        assert len(transport._pools) == 0
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        orphaned = [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task() and not task.done()
        ]
        assert orphaned == []
        assert unhandled == []
    finally:
        loop.set_exception_handler(original_exception_handler)
        if active_pool is not None:
            active_pool.release.set()
        if not active_request.done():
            active_request.cancel()
        if close_caller is not None and not close_caller.done():
            close_caller.cancel()
        await asyncio.gather(
            active_request,
            *(task for task in (close_caller,) if task is not None),
            return_exceptions=True,
        )
        await transport.aclose()


@pytest.mark.asyncio
async def test_concurrent_aclose_callers_share_tracked_cleanup_and_closed_state_is_idempotent() -> (
    None
):
    resolver = FakeResolver(["93.184.216.34"])
    policy = ProviderEndpointPolicy(environment=Environment.TEST, resolver=resolver)
    endpoint = policy.validate_url("https://provider.example/v1")
    factory = _FirstPoolBlockingFactory()
    transport = SecureProviderTransport(
        policy=policy,
        pool_factory=factory,
        max_destination_pools=1,
    )
    active_request = asyncio.create_task(
        transport.post_json(
            endpoint=endpoint,
            path="/embeddings",
            headers={"Content-Type": "application/json"},
            payload={"input": ["active"], "model": "embedding-test"},
            timeout_seconds=3.0,
        )
    )
    first_close: asyncio.Task[None] | None = None
    second_close: asyncio.Task[None] | None = None
    active_pool: _BlockingProviderPool | None = None
    try:
        created = await factory.created.get()
        assert isinstance(created, _BlockingProviderPool)
        active_pool = created
        await active_pool.entered.wait()

        first_close = asyncio.create_task(transport.aclose())
        await asyncio.sleep(0)
        tracked_close = getattr(transport, "_close_task", None)
        assert isinstance(tracked_close, asyncio.Task)
        second_close = asyncio.create_task(transport.aclose())
        await asyncio.sleep(0)
        assert transport._close_task is tracked_close

        first_close.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first_close
        active_pool.release.set()
        assert (await active_request).status_code == 200
        async with asyncio.timeout(1):
            await second_close
        assert tracked_close.done() is True
        assert transport._close_task is None
        assert active_pool.close_calls == 1
        assert len(transport._pools) == 0

        await transport.aclose()
        assert active_pool.close_calls == 1
        assert len(transport._pools) == 0
    finally:
        if active_pool is not None:
            active_pool.release.set()
        if not active_request.done():
            active_request.cancel()
        for task in (first_close, second_close):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            active_request,
            *(task for task in (first_close, second_close) if task is not None),
            return_exceptions=True,
        )
        await transport.aclose()


@pytest.mark.asyncio
async def test_detached_close_failure_is_consumed_and_observed_by_later_aclose() -> None:
    sentinel = "pool-close-secret-sentinel"
    resolver = FakeResolver(["93.184.216.34"])
    policy = ProviderEndpointPolicy(environment=Environment.TEST, resolver=resolver)
    endpoint = policy.validate_url("https://provider.example/v1")
    factory = _FirstPoolBlockingFactory()
    transport = SecureProviderTransport(
        policy=policy,
        pool_factory=factory,
        max_destination_pools=1,
    )
    active_request = asyncio.create_task(
        transport.post_json(
            endpoint=endpoint,
            path="/embeddings",
            headers={"Content-Type": "application/json"},
            payload={"input": ["active"], "model": "embedding-test"},
            timeout_seconds=3.0,
        )
    )
    close_caller: asyncio.Task[None] | None = None
    active_pool: _BlockingProviderPool | None = None
    loop = asyncio.get_running_loop()
    original_exception_handler = loop.get_exception_handler()
    unhandled: list[dict[str, Any]] = []
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
    try:
        created = await factory.created.get()
        assert isinstance(created, _BlockingProviderPool)
        active_pool = created
        active_pool.close_error = ValueError(sentinel)
        await active_pool.entered.wait()

        close_caller = asyncio.create_task(transport.aclose())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        close_caller.cancel()
        with pytest.raises(asyncio.CancelledError):
            await close_caller

        active_pool.release.set()
        assert (await active_request).status_code == 200
        async with asyncio.timeout(1):
            await active_pool.closed.wait()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert transport._close_task is None
        assert unhandled == []
        with pytest.raises(ExceptionGroup) as exc_info:
            await transport.aclose()
        rendered = str(exc_info.value) + repr(exc_info.value)
        assert "ValueError" in rendered
        assert sentinel not in rendered
    finally:
        loop.set_exception_handler(original_exception_handler)
        if active_pool is not None:
            active_pool.close_error = None
            active_pool.release.set()
        if not active_request.done():
            active_request.cancel()
        if close_caller is not None and not close_caller.done():
            close_caller.cancel()
        await asyncio.gather(
            active_request,
            *(task for task in (close_caller,) if task is not None),
            return_exceptions=True,
        )


@pytest.mark.parametrize("host_header", ["Host", "host", "HOST"])
@pytest.mark.asyncio
async def test_secure_transport_rejects_caller_supplied_host_header_before_dns(
    host_header: str,
) -> None:
    resolver = FakeResolver(["93.184.216.34"])
    policy = ProviderEndpointPolicy(environment=Environment.TEST, resolver=resolver)
    endpoint = policy.validate_url("https://provider.example/v1")
    factory = _RecordingProviderPoolFactory()
    transport = SecureProviderTransport(policy=policy, pool_factory=factory)

    try:
        with pytest.raises(ProviderNetworkPolicyError):
            await transport.post_json(
                endpoint=endpoint,
                path="/embeddings",
                headers={host_header: "attacker.example"},
                payload={"input": ["query"], "model": "embedding-test"},
                timeout_seconds=3.0,
            )
    finally:
        await transport.aclose()

    assert resolver.calls == []
    assert factory.pools == []


@pytest.mark.asyncio
async def test_secure_transport_revalidates_even_when_endpoint_was_prevalidated_for_storage() -> (
    None
):
    resolver = FakeResolver(["93.184.216.34"], ["169.254.169.254"])
    policy = ProviderEndpointPolicy(environment=Environment.TEST, resolver=resolver)
    persisted = policy.validate_for_persistence("https://provider.example/v1")
    factory = _RecordingProviderPoolFactory()
    transport = SecureProviderTransport(policy=policy, pool_factory=factory)

    try:
        with pytest.raises(ProviderNetworkPolicyError):
            await transport.post_json(
                endpoint=persisted.endpoint,
                path="/embeddings",
                headers={},
                payload={},
                timeout_seconds=1.0,
            )
    finally:
        await transport.aclose()

    assert factory.pools == []


@pytest.mark.parametrize(
    "path",
    [
        "embeddings",
        "/../embeddings",
        "//attacker.example/embeddings",
        "/embeddings?api_key=secret",
        "/embeddings#fragment",
    ],
)
@pytest.mark.asyncio
async def test_secure_transport_rejects_ambiguous_request_paths_before_dns(path: str) -> None:
    resolver = FakeResolver(["93.184.216.34"])
    policy = ProviderEndpointPolicy(environment=Environment.TEST, resolver=resolver)
    endpoint = policy.validate_url("https://provider.example/v1")
    factory = _RecordingProviderPoolFactory()
    transport = SecureProviderTransport(policy=policy, pool_factory=factory)

    try:
        with pytest.raises(ProviderNetworkPolicyError):
            await transport.post_json(
                endpoint=endpoint,
                path=path,
                headers={},
                payload={},
                timeout_seconds=1.0,
            )
    finally:
        await transport.aclose()

    assert resolver.calls == []
    assert factory.pools == []


def test_transport_destination_and_response_representations_are_safe_and_immutable() -> None:
    policy = ProviderEndpointPolicy(
        environment=Environment.TEST,
        resolver=FakeResolver(["93.184.216.34"]),
    )
    resolved = policy.validate_for_persistence(
        "https://sensitive-provider.example/v1/tenant-sentinel"
    )
    destination = ProviderDestination(
        resolved=resolved,
        selected_address="93.184.216.34",
    )
    response = ProviderHttpResponse(
        status_code=500,
        headers={"x-request-id": "upstream-secret-sentinel"},
        body=b"upstream-body-secret-sentinel",
    )

    with pytest.raises(FrozenInstanceError):
        destination.selected_address = "1.1.1.1"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        response.status_code = 200  # type: ignore[misc]

    rendered = repr(destination) + repr(response)
    for sentinel in (
        "sensitive-provider.example",
        "tenant-sentinel",
        "93.184.216.34",
        "upstream-secret-sentinel",
        "upstream-body-secret-sentinel",
    ):
        assert sentinel not in rendered


def test_secure_transport_constructor_rejects_non_policy_and_unsafe_ca_inputs() -> None:
    with pytest.raises(ProviderNetworkPolicyError):
        SecureProviderTransport(policy=cast(Any, object()))

    policy = ProviderEndpointPolicy(
        environment=Environment.TEST,
        resolver=FakeResolver(["93.184.216.34"]),
    )
    with pytest.raises(ProviderNetworkPolicyError):
        SecureProviderTransport(policy=policy, ca_bundle="ca-path-secret-sentinel\x00")
