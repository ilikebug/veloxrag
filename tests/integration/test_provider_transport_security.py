from __future__ import annotations

import ipaddress
import json
from decimal import Decimal
from typing import TYPE_CHECKING
from uuid import UUID

import pytest

if TYPE_CHECKING:
    from fixtures.provider_stub import RunningProviderStub

from rag_service.config import Environment
from rag_service.providers.credentials import (
    EncryptedProviderCredential,
    ProviderCredentialKeyring,
)
from rag_service.providers.embeddings import (
    EmbeddingConfigSnapshot,
    EmbeddingGateway,
    EmbeddingOperationalConfig,
)
from rag_service.providers.network_policy import (
    ProviderEndpointPolicy,
    ProviderNetworkPolicyError,
    validate_provider_endpoint_url,
)
from rag_service.providers.transport import ProviderHttpResponse, SecureProviderTransport

PROVIDER_STUB_SECRET = "local-provider-stub-secret-sentinel"


class _EncryptedCredentialReader:
    def __init__(self, encrypted: EncryptedProviderCredential) -> None:
        self.encrypted = encrypted
        self.calls: list[UUID] = []

    async def get_encrypted(
        self,
        credential_id: UUID,
    ) -> EncryptedProviderCredential | None:
        self.calls.append(credential_id)
        return self.encrypted


def test_network_policy_default_resolver_rejects_localhost_without_override() -> None:
    policy = ProviderEndpointPolicy(environment=Environment.TEST)

    with pytest.raises(ProviderNetworkPolicyError):
        policy.validate_for_persistence("https://localhost:8443/v1")


def test_network_policy_default_resolver_allows_controlled_local_test_override() -> None:
    policy = ProviderEndpointPolicy(
        environment=Environment.TEST,
        allow_private_targets=True,
    )

    resolved = policy.validate_for_persistence("https://localhost:8443/v1")

    assert resolved.endpoint.url == "https://localhost:8443/v1"
    assert resolved.addresses
    assert all(ipaddress.ip_address(address).is_loopback for address in resolved.addresses)


def test_network_policy_production_refuses_local_test_override_before_dns() -> None:
    with pytest.raises(ProviderNetworkPolicyError):
        ProviderEndpointPolicy(
            environment=Environment.PRODUCTION,
            allow_private_targets=True,
        )


def _local_policy(
    provider_https_stub: RunningProviderStub,
    *answers: tuple[str, ...],
) -> ProviderEndpointPolicy:
    provider_https_stub.resolver.answers[:] = list(answers)
    provider_https_stub.resolver.calls.clear()
    return ProviderEndpointPolicy(
        environment=Environment.TEST,
        allow_private_targets=True,
        resolver=provider_https_stub.resolver,
    )


async def _post_stub_embedding(
    provider_https_stub: RunningProviderStub,
    transport: SecureProviderTransport,
    *,
    path: str = "/embeddings",
) -> ProviderHttpResponse:
    endpoint = validate_provider_endpoint_url(provider_https_stub.base_url)
    return await transport.post_json(
        endpoint=endpoint,
        path=path,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {PROVIDER_STUB_SECRET}",
            "Content-Type": "application/json",
        },
        payload={"input": ["deterministic input"], "model": "stub-embedding"},
        timeout_seconds=2.0,
    )


@pytest.mark.asyncio
async def test_embedding_gateway_full_https_chain_decrypts_authenticates_and_parses_usage(
    provider_https_stub: RunningProviderStub,
    capsys: pytest.CaptureFixture[str],
) -> None:
    loopback = ("127.0.0.1",)
    policy = _local_policy(provider_https_stub, loopback, loopback)
    transport = SecureProviderTransport(
        policy=policy,
        ca_bundle=provider_https_stub.ca_bundle,
    )
    credential_id = UUID("00000000-0000-4000-8000-000000000101")
    provider_config_id = UUID("00000000-0000-4000-8000-000000000102")
    keyring = ProviderCredentialKeyring(
        keys={"v1": b"i" * 32},
        active_key_version="v1",
    )
    reader = _EncryptedCredentialReader(
        keyring.encrypt(credential_id, PROVIDER_STUB_SECRET.encode())
    )
    gateway = EmbeddingGateway(
        keyring=keyring,
        credential_reader=reader,
        transport=transport,
    )
    snapshot = EmbeddingConfigSnapshot(
        adapter_schema_version="openai-embeddings-v1",
        provider_type="openai_compatible",
        base_url=provider_https_stub.base_url,
        credential_id=credential_id,
        default_headers={},
        routing_options={},
        model_name="stub-embedding",
        dimension=3,
        distance="cosine",
        max_input_tokens=8192,
        vector_config={},
    )
    operational = EmbeddingOperationalConfig(
        provider_config_id=provider_config_id,
        provider_enabled=True,
        profile_enabled=True,
        timeout_seconds=Decimal("2.000"),
        max_concurrency=2,
        requests_per_minute=120,
        batch_size=4,
    )

    try:
        result = await gateway.embed(
            snapshot=snapshot,
            operational=operational,
            inputs=("deterministic input",),
        )
    finally:
        await gateway.aclose()

    assert len(result.vectors) == 1
    assert len(result.vectors[0]) == 3
    assert result.usage == {"prompt_tokens": 2, "total_tokens": 2}
    assert reader.calls == [credential_id]
    assert provider_https_stub.request_records == [
        {
            "authorized": True,
            "host": f"{provider_https_stub.hostname}:{provider_https_stub.port}",
            "input_count": 1,
            "model": "stub-embedding",
            "provider": None,
        }
    ]
    captured = capsys.readouterr()
    rendered = captured.out + captured.err + repr(result) + repr(reader.calls)
    assert PROVIDER_STUB_SECRET not in rendered


@pytest.mark.asyncio
async def test_secure_transport_accesses_controlled_local_https_with_pinned_ip_sni_and_host(
    provider_https_stub: RunningProviderStub,
) -> None:
    loopback = ("127.0.0.1",)
    policy = _local_policy(provider_https_stub, loopback, loopback)
    transport = SecureProviderTransport(
        policy=policy,
        ca_bundle=provider_https_stub.ca_bundle,
    )

    try:
        response = await _post_stub_embedding(provider_https_stub, transport)
    finally:
        await transport.aclose()

    assert response.status_code == 200
    document = json.loads(response.body)
    assert len(document["data"]) == 1
    assert len(document["data"][0]["embedding"]) == 3
    assert provider_https_stub.resolver.calls == [
        (provider_https_stub.hostname, provider_https_stub.port),
        (provider_https_stub.hostname, provider_https_stub.port),
    ]
    assert provider_https_stub.request_records == [
        {
            "authorized": True,
            "host": f"{provider_https_stub.hostname}:{provider_https_stub.port}",
            "input_count": 1,
            "model": "stub-embedding",
            "provider": None,
        }
    ]


@pytest.mark.asyncio
async def test_secure_transport_rejects_host_header_override_before_network(
    provider_https_stub: RunningProviderStub,
) -> None:
    loopback = ("127.0.0.1",)
    policy = _local_policy(provider_https_stub, loopback, loopback)
    transport = SecureProviderTransport(
        policy=policy,
        ca_bundle=provider_https_stub.ca_bundle,
    )
    endpoint = validate_provider_endpoint_url(provider_https_stub.base_url)

    try:
        with pytest.raises(ProviderNetworkPolicyError):
            await transport.post_json(
                endpoint=endpoint,
                path="/embeddings",
                headers={
                    "Authorization": f"Bearer {PROVIDER_STUB_SECRET}",
                    "Content-Type": "application/json",
                    "Host": "attacker.example",
                },
                payload={"input": ["query"], "model": "stub-embedding"},
                timeout_seconds=2.0,
            )
    finally:
        await transport.aclose()

    assert provider_https_stub.resolver.calls == []
    assert provider_https_stub.request_records == []


@pytest.mark.asyncio
async def test_secure_transport_ignores_environment_https_proxy(
    provider_https_stub: RunningProviderStub,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:1")
    loopback = ("127.0.0.1",)
    policy = _local_policy(provider_https_stub, loopback, loopback)
    transport = SecureProviderTransport(
        policy=policy,
        ca_bundle=provider_https_stub.ca_bundle,
    )

    try:
        response = await _post_stub_embedding(provider_https_stub, transport)
    finally:
        await transport.aclose()

    assert response.status_code == 200
    assert len(provider_https_stub.request_records) == 1


@pytest.mark.asyncio
async def test_secure_transport_does_not_follow_provider_redirects(
    provider_https_stub: RunningProviderStub,
) -> None:
    loopback = ("127.0.0.1",)
    policy = _local_policy(provider_https_stub, loopback, loopback)
    transport = SecureProviderTransport(
        policy=policy,
        ca_bundle=provider_https_stub.ca_bundle,
    )

    try:
        response = await _post_stub_embedding(
            provider_https_stub,
            transport,
            path="/redirect",
        )
    finally:
        await transport.aclose()

    assert response.status_code == 307
    assert provider_https_stub.request_records == []


@pytest.mark.asyncio
async def test_secure_transport_fails_closed_on_dns_rebinding_without_second_request(
    provider_https_stub: RunningProviderStub,
) -> None:
    loopback = ("127.0.0.1",)
    policy = _local_policy(
        provider_https_stub,
        loopback,
        loopback,
        ("169.254.169.254",),
    )
    transport = SecureProviderTransport(
        policy=policy,
        ca_bundle=provider_https_stub.ca_bundle,
    )

    try:
        first = await _post_stub_embedding(provider_https_stub, transport)
        with pytest.raises(ProviderNetworkPolicyError):
            await _post_stub_embedding(provider_https_stub, transport)
    finally:
        await transport.aclose()

    assert first.status_code == 200
    assert len(provider_https_stub.request_records) == 1


@pytest.mark.asyncio
async def test_secure_transport_does_not_fallback_to_default_httpx_on_tls_failure(
    provider_https_stub: RunningProviderStub,
) -> None:
    loopback = ("127.0.0.1",)
    policy = _local_policy(provider_https_stub, loopback, loopback)
    transport = SecureProviderTransport(policy=policy)

    try:
        with pytest.raises(Exception) as exc_info:
            await _post_stub_embedding(provider_https_stub, transport)
    finally:
        await transport.aclose()

    rendered = str(exc_info.value) + repr(exc_info.value)
    assert PROVIDER_STUB_SECRET not in rendered
    assert provider_https_stub.request_records == []


def test_provider_https_fixture_serialization_and_records_never_contain_plaintext_secret(
    provider_https_stub: RunningProviderStub,
) -> None:
    rendered = repr(provider_https_stub) + json.dumps(list(provider_https_stub.request_records))

    assert PROVIDER_STUB_SECRET not in rendered
    assert "private_key" not in rendered.lower()
