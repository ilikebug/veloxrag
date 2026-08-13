import base64
import json
import os
from uuid import UUID, uuid4

import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy import update

from fixtures.provider_stub import PROVIDER_STUB_SECRET, RunningProviderStub
from rag_service.auth.policies import Capability
from rag_service.auth.schemas import AdminApiKeyCreate, AgentApiKeyCreate
from rag_service.auth.services import ApiKeyService
from rag_service.config import Settings, get_settings
from rag_service.db.models.providers import ProviderConfig
from rag_service.db.session import Database
from rag_service.main import create_app
from rag_service.providers.gateway_provider import EmbeddingGatewayProvider


def _random_text(byte_count: int = 32) -> str:
    return base64.urlsafe_b64encode(os.urandom(byte_count)).decode("ascii")


def _settings(provider: RunningProviderStub) -> Settings:
    key_version = f"test-{uuid4().hex}"
    return Settings(
        _env_file=None,
        environment="test",
        admin_key_hmac_secret=SecretStr(_random_text()),
        agent_key_hmac_secret=SecretStr(_random_text()),
        provider_credential_keyring=SecretStr(
            json.dumps(
                {key_version: base64.b64encode(os.urandom(32)).decode("ascii")},
                separators=(",", ":"),
            )
        ),
        provider_credential_active_key_version=key_version,
        provider_allow_private_targets=True,
        provider_ca_bundle=provider.ca_bundle,
    )


async def _issued_keys(
    database: Database,
    settings: Settings,
) -> tuple[SecretStr, SecretStr]:
    async with database.session() as session:
        service = ApiKeyService(
            session=session,
            authentication_sessions=database.session,
            settings=settings,
        )
        admin = await service.create_admin_key(
            AdminApiKeyCreate(name="embedding probe administrator"),
            request_id="req-probe-admin-bootstrap",
        )
        agent = await service.create_agent_key(
            AgentApiKeyCreate(
                name="embedding probe agent",
                capabilities=frozenset({Capability.RETRIEVE}),
                requests_per_minute=60,
                max_concurrency=4,
            ),
            actor=None,
            request_id="req-probe-agent-bootstrap",
        )
    return admin.token, agent.token


def _headers(
    issued: SecretStr,
    request_id: str,
    *,
    idempotency_key: str | None = None,
) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {issued.get_secret_value()}",
        "X-Request-ID": request_id,
    }
    if idempotency_key is not None:
        headers["Idempotency-Key"] = idempotency_key
    return headers


async def _create_credential(
    client: httpx.AsyncClient,
    admin: SecretStr,
    *,
    name: str,
    value: str,
) -> UUID:
    response = await client.post(
        "/v1/admin/provider-credentials",
        headers=_headers(admin, f"req-{name}", idempotency_key=f"idem-{name}"),
        json={"name": name, "secret": value},
    )
    assert response.status_code == 201
    return UUID(response.json()["id"])


async def _create_config(
    client: httpx.AsyncClient,
    admin: SecretStr,
    provider: RunningProviderStub,
    *,
    name: str,
    credential_id: UUID,
) -> UUID:
    response = await client.post(
        "/v1/admin/provider-configs",
        headers=_headers(admin, f"req-{name}", idempotency_key=f"idem-{name}"),
        json={
            "name": name,
            "provider_type": "openrouter",
            "base_url": provider.loopback_base_url,
            "credential_id": str(credential_id),
            "default_headers": {},
            "routing_options": {},
            "timeout_seconds": "5",
            "max_concurrency": 2,
            "requests_per_minute": 60,
            "enabled": True,
        },
    )
    assert response.status_code == 201
    return UUID(response.json()["id"])


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_embedding_probe_api_uses_shared_tls_gateway_and_sanitizes_failures(
    migrated_database: Database,
    provider_https_stub: RunningProviderStub,
) -> None:
    settings = _settings(provider_https_stub)
    admin, agent = await _issued_keys(migrated_database, settings)
    app = create_app()
    gateway_provider = EmbeddingGatewayProvider()
    app.state.database = migrated_database
    app.state.embedding_gateway_provider = gateway_provider
    app.dependency_overrides[get_settings] = lambda: settings
    transport = httpx.ASGITransport(app=app)
    bad_value = _random_text()
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            credential_id = await _create_credential(
                client,
                admin,
                name="probe-credential",
                value=PROVIDER_STUB_SECRET,
            )
            provider_config_id = await _create_config(
                client,
                admin,
                provider_https_stub,
                name="probe-config",
                credential_id=credential_id,
            )

            success = await client.post(
                f"/v1/admin/provider-configs/{provider_config_id}/embedding-probe",
                headers=_headers(admin, "req-probe-success"),
                json={"model_name": " fixture/model "},
            )
            assert success.status_code == 200
            assert success.headers["cache-control"] == "no-store"
            assert success.json() == {
                "provider_config_id": str(provider_config_id),
                "model_name": "fixture/model",
                "dimension": 3,
            }

            agent_denied = await client.post(
                f"/v1/admin/provider-configs/{provider_config_id}/embedding-probe",
                headers=_headers(agent, "req-probe-agent-denied"),
                json={"model_name": "fixture/model"},
            )
            assert agent_denied.status_code == 401
            assert agent_denied.headers["cache-control"] == "no-store"

            async with migrated_database.session() as session, session.begin():
                await session.execute(
                    update(ProviderConfig)
                    .where(ProviderConfig.id == provider_config_id)
                    .values(enabled=False)
                )
            disabled = await client.post(
                f"/v1/admin/provider-configs/{provider_config_id}/embedding-probe",
                headers=_headers(admin, "req-probe-disabled"),
                json={"model_name": "fixture/model"},
            )
            assert disabled.status_code == 409
            assert disabled.json()["error"]["code"] == "PROVIDER_CONFIG_DISABLED"

            missing = await client.post(
                f"/v1/admin/provider-configs/{uuid4()}/embedding-probe",
                headers=_headers(admin, "req-probe-missing"),
                json={"model_name": "fixture/model"},
            )
            assert missing.status_code == 404
            assert missing.json()["error"]["code"] == "RESOURCE_NOT_FOUND"

            bad_credential_id = await _create_credential(
                client,
                admin,
                name="probe-bad-credential",
                value=bad_value,
            )
            bad_config_id = await _create_config(
                client,
                admin,
                provider_https_stub,
                name="probe-bad-config",
                credential_id=bad_credential_id,
            )
            provider_failure = await client.post(
                f"/v1/admin/provider-configs/{bad_config_id}/embedding-probe",
                headers=_headers(admin, "req-probe-provider-failure"),
                json={"model_name": "fixture/model"},
            )
            assert provider_failure.status_code == 422
            assert provider_failure.headers["cache-control"] == "no-store"
            assert provider_failure.json()["error"] == {
                "code": "PROVIDER_AUTHENTICATION_FAILED",
                "message": "Provider authentication failed",
                "retryable": False,
                "request_id": "req-probe-provider-failure",
            }
            assert bad_value not in provider_failure.text
    finally:
        bad_value = "<redacted>"
        admin = SecretStr("<redacted>")
        agent = SecretStr("<redacted>")
        await gateway_provider.aclose()
