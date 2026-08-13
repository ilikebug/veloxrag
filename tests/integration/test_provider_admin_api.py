import asyncio
import base64
import json
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from functools import partial
from typing import cast
from uuid import UUID, uuid4

import httpx
import psycopg
import pytest
from fastapi import FastAPI
from pydantic import SecretStr
from sqlalchemy import func, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.api.cursors import CursorPosition
from rag_service.api.errors import BusinessError
from rag_service.auth.codec import KeyKind
from rag_service.auth.policies import AdminPrincipal, Capability
from rag_service.auth.schemas import AdminApiKeyCreate, AgentApiKeyCreate
from rag_service.auth.services import ApiKeyService
from rag_service.config import Settings, get_settings
from rag_service.db.models.auth import ApiKey, AuditEvent, IdempotencyRecord
from rag_service.db.models.knowledge_bases import KnowledgeBase, KnowledgeBaseIndexGeneration
from rag_service.db.models.providers import ModelProfile, ProviderConfig, ProviderCredential
from rag_service.db.session import Database
from rag_service.main import create_app
from rag_service.providers.credentials import (
    EncryptedProviderCredential,
    ProviderCredentialKeyring,
)
from rag_service.providers.network_policy import ProviderEndpointPolicy
from rag_service.providers.repositories import (
    ProviderAuditRepository,
    ProviderConfigRecord,
    ProviderConfigRepository,
    ProviderConfigSecretSourceRecord,
    ProviderCredentialRecord,
    ProviderCredentialRepository,
    ProviderIdempotencyRepository,
    ProviderRepositories,
    sqlalchemy_provider_repositories,
)
from rag_service.providers.schemas import (
    ModelProfilePatch,
    ProviderConfigCreate,
    ProviderConfigCreateResult,
    ProviderConfigPatch,
    ProviderCredentialCreate,
    ProviderCredentialCreateResult,
    ProviderCredentialPatch,
)
from rag_service.providers.services import (
    ModelProfileService,
    ProviderConfigService,
    ProviderCredentialService,
    model_profile_etag,
    provider_config_etag,
    provider_credential_etag,
)

ADMIN_HMAC_SECRET = "admin-provider-test-hmac-secret!!"
AGENT_HMAC_SECRET = "agent-provider-test-hmac-secret!!"
KEY_VERSION = "2026-07"
KEY = b"k" * 32
SECRET = "provider-http-secret-sentinel"
ROTATED_SECRET = "provider-http-rotated-sentinel"
CONFIG_LOCAL_URL = "https://localhost:8443/v1/"
CONFIG_CANONICAL_LOCAL_URL = "https://localhost:8443/v1"


def _settings(database_url: str, *, allow_private_targets: bool = False) -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        database_url=SecretStr(database_url),
        admin_key_hmac_secret=SecretStr(ADMIN_HMAC_SECRET),
        agent_key_hmac_secret=SecretStr(AGENT_HMAC_SECRET),
        provider_credential_keyring=SecretStr(
            json.dumps({KEY_VERSION: base64.b64encode(KEY).decode()})
        ),
        provider_credential_active_key_version=KEY_VERSION,
        provider_allow_private_targets=allow_private_targets,
        default_page_size=2,
        max_page_size=3,
    )


def _app(database: Database, settings: Settings) -> FastAPI:
    app = create_app()
    app.state.database = database
    app.dependency_overrides[get_settings] = lambda: settings
    return app


async def _tokens(database: Database, settings: Settings) -> tuple[str, str]:
    async with database.session() as session:
        service = ApiKeyService(
            session=session,
            authentication_sessions=database.session,
            settings=settings,
        )
        admin = await service.create_admin_key(
            AdminApiKeyCreate(name="provider-admin"),
            request_id="req-provider-admin-bootstrap",
        )
        agent = await service.create_agent_key(
            AgentApiKeyCreate(
                name="provider-agent",
                capabilities=frozenset({Capability.RETRIEVE}),
                requests_per_minute=60,
                max_concurrency=4,
            ),
            actor=None,
            request_id="req-provider-agent-bootstrap",
        )
    return admin.token.get_secret_value(), agent.token.get_secret_value()


async def _credential(database: Database, credential_id: UUID) -> ProviderCredential:
    async with database.session() as session:
        row = await session.get(ProviderCredential, credential_id)
        assert row is not None
        return row


async def _provider_config(database: Database, provider_config_id: UUID) -> ProviderConfig:
    async with database.session() as session:
        row = await session.get(ProviderConfig, provider_config_id)
        assert row is not None
        return row


async def _model_profile(database: Database, model_profile_id: UUID) -> ModelProfile:
    async with database.session() as session:
        row = await session.get(ModelProfile, model_profile_id)
        assert row is not None
        return row


async def _create_credential_http(
    client: httpx.AsyncClient,
    admin_token: str,
    *,
    name: str,
    key: str,
) -> UUID:
    response = await client.post(
        "/v1/admin/provider-credentials",
        headers={
            "Authorization": f"Bearer {admin_token}",
            "Idempotency-Key": key,
        },
        json={"name": name, "secret": SECRET},
    )
    assert response.status_code == 201
    return UUID(response.json()["id"])


def _provider_config_payload(
    credential_id: UUID,
    **updates: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": "integration embedding provider",
        "provider_type": "openai_compatible",
        "base_url": CONFIG_LOCAL_URL,
        "credential_id": str(credential_id),
        "default_headers": {"X-Title": "RAG integration"},
        "routing_options": {},
        "timeout_seconds": 30,
        "max_concurrency": 8,
        "requests_per_minute": 600,
        "enabled": True,
    }
    payload.update(updates)
    return payload


def _model_profile_payload(
    provider_config_id: UUID,
    **updates: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": "integration embedding profile",
        "capability": "embedding",
        "provider_config_id": str(provider_config_id),
        "model_name": "text-embedding-3-small",
        "dimension": 1536,
        "max_input_tokens": 8191,
        "batch_size": 64,
        "timeout_seconds": 30,
        "vector_config": {},
        "enabled": True,
    }
    payload.update(updates)
    return payload


async def _create_provider_config_http(
    client: httpx.AsyncClient,
    admin_token: str,
    credential_id: UUID,
    *,
    name: str,
    key: str,
    enabled: bool = True,
) -> UUID:
    response = await client.post(
        "/v1/admin/provider-configs",
        headers={
            "Authorization": f"Bearer {admin_token}",
            "Idempotency-Key": key,
        },
        json=_provider_config_payload(credential_id, name=name, enabled=enabled),
    )
    assert response.status_code == 201
    return UUID(response.json()["id"])


async def _create_model_profile_http(
    client: httpx.AsyncClient,
    admin_token: str,
    provider_config_id: UUID,
    *,
    name: str,
    key: str,
) -> tuple[UUID, str]:
    response = await client.post(
        "/v1/admin/model-profiles",
        headers={
            "Authorization": f"Bearer {admin_token}",
            "Idempotency-Key": key,
        },
        json=_model_profile_payload(provider_config_id, name=name),
    )
    assert response.status_code == 201
    return UUID(response.json()["id"]), response.headers["etag"]


class _BlockingResolver:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def resolve(self, hostname: str, port: int) -> tuple[str, ...]:
        del hostname, port
        self.entered.set()
        if not self.release.wait(timeout=10):
            raise TimeoutError("provider resolver test barrier timed out")
        return ("127.0.0.1",)


async def _wait_for_resolver(resolver: _BlockingResolver) -> None:
    entered = await asyncio.to_thread(resolver.entered.wait, 5)
    assert entered


async def _capture_sync_failure(action: Callable[[], None]) -> BaseException | None:
    try:
        await asyncio.to_thread(action)
    except BaseException as source:
        return source
    return None


def _lock_admin_row(sync_url: str, actor_id: UUID) -> None:
    with psycopg.connect(sync_url) as connection:
        connection.execute("SET LOCAL lock_timeout = '750ms'")
        row = connection.execute(
            "SELECT id FROM api_keys WHERE id = %s FOR UPDATE",
            (actor_id,),
        ).fetchone()
        assert row == (actor_id,)


def _lock_provider_config_row(sync_url: str, provider_config_id: UUID) -> None:
    with psycopg.connect(sync_url) as connection:
        connection.execute("SET LOCAL lock_timeout = '750ms'")
        row = connection.execute(
            "SELECT id FROM provider_configs WHERE id = %s FOR UPDATE",
            (provider_config_id,),
        ).fetchone()
        assert row == (provider_config_id,)


def _advance_provider_config_revision(sync_url: str, provider_config_id: UUID) -> None:
    with psycopg.connect(sync_url) as connection:
        connection.execute("SET LOCAL lock_timeout = '750ms'")
        updated = connection.execute(
            "UPDATE provider_configs "
            "SET enabled = false, resource_revision = resource_revision + 1 "
            "WHERE id = %s",
            (provider_config_id,),
        )
        assert updated.rowcount == 1


def _revoke_admin_row(sync_url: str, actor_id: UUID) -> None:
    with psycopg.connect(sync_url) as connection:
        connection.execute("SET LOCAL lock_timeout = '750ms'")
        updated = connection.execute(
            "UPDATE api_keys SET status = 'revoked', revoked_at = now() WHERE id = %s",
            (actor_id,),
        )
        assert updated.rowcount == 1


def _delete_provider_credential(sync_url: str, credential_id: UUID) -> None:
    with psycopg.connect(sync_url) as connection:
        connection.execute("SET LOCAL lock_timeout = '750ms'")
        deleted = connection.execute(
            "DELETE FROM provider_credentials WHERE id = %s",
            (credential_id,),
        )
        assert deleted.rowcount == 1


def _advance_legacy_provider_url(sync_url: str, provider_config_id: UUID) -> None:
    with psycopg.connect(sync_url) as connection:
        connection.execute("SET LOCAL lock_timeout = '750ms'")
        updated = connection.execute(
            "UPDATE provider_configs "
            "SET base_url = 'https://localhost:9443/replaced', "
            "resource_revision = resource_revision + 1 "
            "WHERE id = %s",
            (provider_config_id,),
        )
        assert updated.rowcount == 1


async def _audit_count(database: Database, credential_id: UUID, action: str) -> int:
    async with database.session() as session:
        return cast(
            int,
            await session.scalar(
                select(func.count())
                .select_from(AuditEvent)
                .where(
                    AuditEvent.target_id == credential_id,
                    AuditEvent.action == action,
                )
            ),
        )


async def _created_audit_metadata(database: Database, credential_id: UUID) -> object:
    async with database.session() as session:
        return await session.scalar(
            select(AuditEvent.metadata_).where(
                AuditEvent.target_id == credential_id,
                AuditEvent.action == "provider_credential.created",
            )
        )


async def _admin_principal(
    database: Database,
    settings: Settings,
    token: str,
) -> AdminPrincipal:
    async with database.session() as session:
        service = ApiKeyService(
            session=session,
            authentication_sessions=database.session,
            settings=settings,
        )
        return cast(AdminPrincipal, await service.authenticate(token, KeyKind.ADMIN))


class _BarrierIdempotencyRepository:
    def __init__(
        self,
        delegate: ProviderIdempotencyRepository,
        barrier: asyncio.Barrier,
    ) -> None:
        self._delegate = delegate
        self._barrier = barrier
        self._first_get = True

    async def get(
        self,
        actor_key_id: UUID,
        operation: str,
        idempotency_key: str,
    ) -> IdempotencyRecord | None:
        if self._first_get:
            self._first_get = False
            await self._barrier.wait()
        return await self._delegate.get(actor_key_id, operation, idempotency_key)

    async def add(self, record: IdempotencyRecord) -> None:
        await self._delegate.add(record)


def _barrier_repository_factory(
    barrier: asyncio.Barrier,
) -> Callable[[AsyncSession], ProviderRepositories]:
    def factory(session: AsyncSession) -> ProviderRepositories:
        repositories = sqlalchemy_provider_repositories(session)
        return ProviderRepositories(
            credentials=repositories.credentials,
            admins=repositories.admins,
            idempotency=_BarrierIdempotencyRepository(
                repositories.idempotency,
                barrier,
            ),
            audits=repositories.audits,
            configs=repositories.configs,
        )

    return factory


class _BarrierCredentialRepository:
    def __init__(
        self,
        delegate: ProviderCredentialRepository,
        barrier: asyncio.Barrier,
    ) -> None:
        self._delegate = delegate
        self._barrier = barrier
        self._first_update_read = True

    async def list_safe(
        self,
        position: CursorPosition | None,
        limit: int,
    ) -> list[ProviderCredentialRecord]:
        return await self._delegate.list_safe(position, limit)

    async def get_safe(
        self,
        credential_id: UUID,
        *,
        for_update: bool = False,
    ) -> ProviderCredentialRecord | None:
        if for_update and self._first_update_read:
            self._first_update_read = False
            await self._barrier.wait()
        return await self._delegate.get_safe(credential_id, for_update=for_update)

    async def add_encrypted(
        self,
        credential_id: UUID,
        name: str,
        encrypted: EncryptedProviderCredential,
    ) -> ProviderCredentialRecord:
        return await self._delegate.add_encrypted(credential_id, name, encrypted)

    async def update_encrypted(
        self,
        credential_id: UUID,
        *,
        name: str | None,
        encrypted: EncryptedProviderCredential | None,
        updated_at: datetime,
        rotated_at: datetime | None,
    ) -> ProviderCredentialRecord:
        return await self._delegate.update_encrypted(
            credential_id,
            name=name,
            encrypted=encrypted,
            updated_at=updated_at,
            rotated_at=rotated_at,
        )


def _credential_barrier_repository_factory(
    barrier: asyncio.Barrier,
) -> Callable[[AsyncSession], ProviderRepositories]:
    def factory(session: AsyncSession) -> ProviderRepositories:
        repositories = sqlalchemy_provider_repositories(session)
        return ProviderRepositories(
            credentials=_BarrierCredentialRepository(
                repositories.credentials,
                barrier,
            ),
            admins=repositories.admins,
            idempotency=repositories.idempotency,
            audits=repositories.audits,
            configs=repositories.configs,
        )

    return factory


class _FailAfterAuditRepository:
    def __init__(
        self,
        delegate: ProviderAuditRepository,
        action: str,
    ) -> None:
        self._delegate = delegate
        self._action = action

    async def add(self, event: AuditEvent) -> None:
        await self._delegate.add(event)
        if event.action == self._action:
            raise RuntimeError("injected provider audit failure")

    async def get_created_response(
        self,
        actor_key_id: UUID,
        credential_id: UUID,
    ) -> object | None:
        return await self._delegate.get_created_response(actor_key_id, credential_id)

    async def get_provider_config_created_response(
        self,
        actor_key_id: UUID,
        provider_config_id: UUID,
    ) -> object | None:
        return await self._delegate.get_provider_config_created_response(
            actor_key_id,
            provider_config_id,
        )

    async def get_model_profile_created_response(
        self,
        actor_key_id: UUID,
        model_profile_id: UUID,
    ) -> object | None:
        return await self._delegate.get_model_profile_created_response(
            actor_key_id,
            model_profile_id,
        )


def _failing_audit_repository_factory(
    action: str,
) -> Callable[[AsyncSession], ProviderRepositories]:
    def factory(session: AsyncSession) -> ProviderRepositories:
        repositories = sqlalchemy_provider_repositories(session)
        return ProviderRepositories(
            credentials=repositories.credentials,
            admins=repositories.admins,
            idempotency=repositories.idempotency,
            audits=_FailAfterAuditRepository(repositories.audits, action),
            configs=repositories.configs,
        )

    return factory


def _assert_safe_payload(response: httpx.Response, *secrets: str) -> None:
    lowered = response.text.lower()
    for forbidden in ("secret", "ciphertext", "nonce"):
        assert forbidden not in lowered
    for secret in secrets:
        assert secret not in response.text


def _assert_safe_config_payload(response: httpx.Response, *secrets: str) -> None:
    lowered = response.text.lower()
    for forbidden in ("secret_ref", "ciphertext", "nonce"):
        assert forbidden not in lowered
    for secret in secrets:
        assert secret not in response.text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_creates_replays_and_reads_only_safe_credential_fields(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    admin_token, _agent_token = await _tokens(migrated_database, settings)
    app = _app(migrated_database, settings)
    headers = {
        "Authorization": f"Bearer {admin_token}",
        "Idempotency-Key": "http-create-credential",
        "X-Request-ID": "req-http-create-credential",
    }

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        created = await client.post(
            "/v1/admin/provider-credentials",
            headers=headers,
            json={"name": "cloud embedding", "secret": SECRET},
        )
        assert created.status_code == 201
        assert created.headers["cache-control"] == "no-store"
        credential_id = UUID(created.json()["id"])
        assert created.headers["location"] == (f"/v1/admin/provider-credentials/{credential_id}")
        assert created.headers["etag"].endswith(':r1"')
        assert set(created.json()) == {
            "id",
            "name",
            "credential_configured",
            "key_version",
            "resource_revision",
            "created_at",
            "updated_at",
            "rotated_at",
        }
        assert created.json()["credential_configured"] is True
        _assert_safe_payload(created, SECRET, admin_token)

        detail = await client.get(
            f"/v1/admin/provider-credentials/{credential_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert detail.status_code == 200
        assert detail.headers["etag"] == created.headers["etag"]
        assert detail.json() == created.json()
        _assert_safe_payload(detail, SECRET, admin_token)

        listed = await client.get(
            "/v1/admin/provider-credentials",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert listed.status_code == 200
        assert listed.headers["cache-control"] == "no-store"
        assert [item["id"] for item in listed.json()["items"]] == [str(credential_id)]
        _assert_safe_payload(listed, SECRET, admin_token)

        mutated = await client.patch(
            f"/v1/admin/provider-credentials/{credential_id}",
            headers={
                "Authorization": f"Bearer {admin_token}",
                "If-Match": created.headers["etag"],
                "X-Request-ID": "req-http-mutate-after-create",
            },
            json={"name": "cloud embedding renamed", "secret": ROTATED_SECRET},
        )
        assert mutated.status_code == 200
        assert mutated.json()["resource_revision"] == 2

        replayed = await client.post(
            "/v1/admin/provider-credentials",
            headers={**headers, "X-Request-ID": "req-http-replay-credential"},
            json={"name": "cloud embedding", "secret": SECRET},
        )
        assert replayed.status_code == 200
        assert replayed.json() == created.json()
        assert replayed.headers["etag"] == created.headers["etag"]
        _assert_safe_payload(replayed, SECRET, ROTATED_SECRET, admin_token)

        current_detail = await client.get(
            f"/v1/admin/provider-credentials/{credential_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert current_detail.status_code == 200
        assert current_detail.json() == mutated.json()
        assert current_detail.headers["etag"] == mutated.headers["etag"]

    persisted = await _credential(migrated_database, credential_id)
    assert persisted.ciphertext != SECRET.encode()
    assert SECRET.encode() not in persisted.ciphertext
    assert ROTATED_SECRET.encode() not in persisted.ciphertext
    assert len(persisted.nonce) == 12
    assert persisted.key_version == KEY_VERSION
    keyring = ProviderCredentialKeyring(keys={KEY_VERSION: KEY}, active_key_version=KEY_VERSION)
    encrypted = EncryptedProviderCredential(
        ciphertext=persisted.ciphertext,
        nonce=persisted.nonce,
        key_version=persisted.key_version,
        algorithm=persisted.algorithm,
    )
    assert (
        keyring.use_decrypted(
            credential_id,
            encrypted,
            lambda buffer: bytes(buffer).decode(),
        )
        == ROTATED_SECRET
    )
    assert (
        await _audit_count(
            migrated_database,
            credential_id,
            "provider_credential.created",
        )
        == 1
    )
    async with migrated_database.session() as session:
        record = await session.scalar(
            select(IdempotencyRecord).where(IdempotencyRecord.result_resource_id == credential_id)
        )
        assert record is not None
        assert SECRET.encode() not in record.request_fingerprint
    snapshot = await _created_audit_metadata(migrated_database, credential_id)
    assert snapshot == created.json()
    serialized_snapshot = json.dumps(snapshot)
    for forbidden in (SECRET, ROTATED_SECRET, "secret", "ciphertext", "nonce"):
        assert forbidden not in serialized_snapshot
    assert SECRET not in caplog.text
    assert ROTATED_SECRET not in caplog.text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_provider_credential_create_rejects_reused_key_and_non_admins(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    admin_token, agent_token = await _tokens(migrated_database, settings)
    app = _app(migrated_database, settings)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        missing_key = await client.post(
            "/v1/admin/provider-credentials",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"name": "missing key", "secret": SECRET},
        )
        assert missing_key.status_code == 422
        assert missing_key.headers["cache-control"] == "no-store"
        _assert_safe_payload(missing_key, SECRET, admin_token)

        created = await client.post(
            "/v1/admin/provider-credentials",
            headers={
                "Authorization": f"Bearer {admin_token}",
                "Idempotency-Key": "reused-key",
            },
            json={"name": "first", "secret": SECRET},
        )
        assert created.status_code == 201

        reused = await client.post(
            "/v1/admin/provider-credentials",
            headers={
                "Authorization": f"Bearer {admin_token}",
                "Idempotency-Key": "reused-key",
            },
            json={"name": "first", "secret": "different-secret"},
        )
        assert reused.status_code == 409
        assert reused.json()["error"]["code"] == "IDEMPOTENCY_KEY_REUSED"
        assert reused.headers["cache-control"] == "no-store"
        _assert_safe_payload(reused, SECRET, "different-secret", admin_token)

        denied = await client.get(
            "/v1/admin/provider-credentials",
            headers={"Authorization": f"Bearer {agent_token}"},
        )
        assert denied.status_code == 401
        assert denied.headers["cache-control"] == "no-store"
        assert denied.json()["error"]["code"] == "INVALID_API_KEY"
        _assert_safe_payload(denied, agent_token, admin_token)

        missing_id = uuid4()
        missing = await client.get(
            f"/v1/admin/provider-credentials/{missing_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "RESOURCE_NOT_FOUND"

        missing_patch = await client.patch(
            f"/v1/admin/provider-credentials/{missing_id}",
            headers={
                "Authorization": f"Bearer {admin_token}",
                "If-Match": f'"provider-credential:{missing_id}:r1"',
            },
            json={"name": "missing"},
        )
        assert missing_patch.status_code == 404
        assert missing_patch.json()["error"]["code"] == "RESOURCE_NOT_FOUND"

        for params in ({"cursor": "invalid"}, {"limit": 0}, {"limit": 4}):
            invalid_page = await client.get(
                "/v1/admin/provider-credentials",
                params=params,
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert invalid_page.status_code == 422
            assert invalid_page.json()["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_provider_credential_patch_uses_etags_and_rotates_in_place(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    admin_token, _agent_token = await _tokens(migrated_database, settings)
    app = _app(migrated_database, settings)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        created = await client.post(
            "/v1/admin/provider-credentials",
            headers={
                "Authorization": f"Bearer {admin_token}",
                "Idempotency-Key": "rotate-create",
            },
            json={"name": "rotating", "secret": SECRET},
        )
        credential_id = UUID(created.json()["id"])
        first_etag = created.headers["etag"]
        first_row = await _credential(migrated_database, credential_id)

        missing = await client.patch(
            f"/v1/admin/provider-credentials/{credential_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"name": "must not apply"},
        )
        assert missing.status_code == 428
        assert missing.json()["error"]["code"] == "PRECONDITION_REQUIRED"
        assert missing.headers["cache-control"] == "no-store"

        stale = await client.patch(
            f"/v1/admin/provider-credentials/{credential_id}",
            headers={
                "Authorization": f"Bearer {admin_token}",
                "If-Match": '"provider-credential:stale:r1"',
            },
            json={"name": "must not apply"},
        )
        assert stale.status_code == 412
        assert stale.json()["error"]["code"] == "PRECONDITION_FAILED"

        renamed = await client.patch(
            f"/v1/admin/provider-credentials/{credential_id}",
            headers={
                "Authorization": f"Bearer {admin_token}",
                "If-Match": first_etag,
                "X-Request-ID": "req-http-rename-credential",
            },
            json={"name": "renamed"},
        )
        assert renamed.status_code == 200
        assert renamed.json()["resource_revision"] == 2
        assert renamed.json()["rotated_at"] is None
        second_etag = renamed.headers["etag"]

        rotated = await client.patch(
            f"/v1/admin/provider-credentials/{credential_id}",
            headers={
                "Authorization": f"Bearer {admin_token}",
                "If-Match": second_etag,
                "X-Request-ID": "req-http-rotate-credential",
            },
            json={"secret": ROTATED_SECRET},
        )
        assert rotated.status_code == 200
        assert UUID(rotated.json()["id"]) == credential_id
        assert rotated.json()["resource_revision"] == 3
        assert rotated.json()["rotated_at"] is not None
        _assert_safe_payload(rotated, SECRET, ROTATED_SECRET, admin_token)

    second_row = await _credential(migrated_database, credential_id)
    assert second_row.nonce != first_row.nonce
    assert second_row.ciphertext != first_row.ciphertext
    assert second_row.resource_revision == 3
    keyring = ProviderCredentialKeyring(keys={KEY_VERSION: KEY}, active_key_version=KEY_VERSION)
    assert (
        keyring.use_decrypted(
            credential_id,
            EncryptedProviderCredential(
                ciphertext=second_row.ciphertext,
                nonce=second_row.nonce,
                key_version=second_row.key_version,
                algorithm=second_row.algorithm,
            ),
            lambda buffer: bytes(buffer).decode(),
        )
        == ROTATED_SECRET
    )
    assert (
        await _audit_count(
            migrated_database,
            credential_id,
            "provider_credential.updated",
        )
        == 1
    )
    assert (
        await _audit_count(
            migrated_database,
            credential_id,
            "provider_credential.rotated",
        )
        == 1
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_credential_list_uses_stable_cursor_pagination(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    admin_token, _agent_token = await _tokens(migrated_database, settings)
    app = _app(migrated_database, settings)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        ids: set[str] = set()
        for index in range(3):
            response = await client.post(
                "/v1/admin/provider-credentials",
                headers={
                    "Authorization": f"Bearer {admin_token}",
                    "Idempotency-Key": f"page-{index}",
                },
                json={"name": f"page credential {index}", "secret": f"value-{index}"},
            )
            assert response.status_code == 201
            ids.add(response.json()["id"])

        first = await client.get(
            "/v1/admin/provider-credentials?limit=2",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert first.status_code == 200
        assert len(first.json()["items"]) == 2
        cursor = first.json()["next_cursor"]
        assert isinstance(cursor, str)

        second = await client.get(
            "/v1/admin/provider-credentials",
            params={"limit": 2, "cursor": cursor},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert second.status_code == 200
        assert len(second.json()["items"]) == 1
        assert second.json()["next_cursor"] is None
        assert {item["id"] for item in (*first.json()["items"], *second.json()["items"])} == ids


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("same_request", (True, False), ids=("same", "different"))
async def test_provider_credential_concurrent_idempotency_is_serialized(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
    same_request: bool,
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    admin_token, _agent_token = await _tokens(migrated_database, settings)
    actor = await _admin_principal(migrated_database, settings, admin_token)
    barrier = asyncio.Barrier(2)
    repository_factory = _barrier_repository_factory(barrier)
    first_command = ProviderCredentialCreate(
        name="concurrent first",
        secret=SecretStr("concurrent-first-secret"),
    )
    second_command = (
        first_command
        if same_request
        else ProviderCredentialCreate(
            name="concurrent second",
            secret=SecretStr("concurrent-second-secret"),
        )
    )

    async with (
        migrated_database.session() as first_session,
        migrated_database.session() as second_session,
    ):
        first_service = ProviderCredentialService(
            session=first_session,
            settings=settings,
            keyring_factory=lambda: ProviderCredentialKeyring(
                keys={KEY_VERSION: KEY},
                active_key_version=KEY_VERSION,
            ),
            repository_factory=repository_factory,
        )
        second_service = ProviderCredentialService(
            session=second_session,
            settings=settings,
            keyring_factory=lambda: ProviderCredentialKeyring(
                keys={KEY_VERSION: KEY},
                active_key_version=KEY_VERSION,
            ),
            repository_factory=repository_factory,
        )
        outcomes = await asyncio.wait_for(
            asyncio.gather(
                first_service.create_credential(
                    first_command,
                    actor=actor,
                    request_id="req-concurrent-first",
                    idempotency_key="concurrent-idempotency-key",
                ),
                second_service.create_credential(
                    second_command,
                    actor=actor,
                    request_id="req-concurrent-second",
                    idempotency_key="concurrent-idempotency-key",
                ),
                return_exceptions=True,
            ),
            timeout=15,
        )

    successes = [
        outcome for outcome in outcomes if isinstance(outcome, ProviderCredentialCreateResult)
    ]
    failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
    if same_request:
        assert failures == []
        assert len(successes) == 2
        assert {result.created for result in successes} == {True, False}
        assert successes[0].credential == successes[1].credential
    else:
        assert len(successes) == 1
        assert len(failures) == 1
        failure = cast(BusinessError, failures[0])
        assert (failure.status_code, failure.code) == (409, "IDEMPOTENCY_KEY_REUSED")

    async with migrated_database.session() as session:
        credential_rows = (
            (
                await session.execute(
                    select(ProviderCredential).where(
                        ProviderCredential.name.in_(("concurrent first", "concurrent second"))
                    )
                )
            )
            .scalars()
            .all()
        )
        assert len(credential_rows) == 1
        assert (
            await session.scalar(
                select(func.count())
                .select_from(IdempotencyRecord)
                .where(
                    IdempotencyRecord.actor_key_id == actor.key_id,
                    IdempotencyRecord.operation == "provider_credential.create",
                    IdempotencyRecord.idempotency_key == "concurrent-idempotency-key",
                )
            )
            == 1
        )
        assert (
            await session.scalar(
                select(func.count())
                .select_from(AuditEvent)
                .where(
                    AuditEvent.action == "provider_credential.created",
                    AuditEvent.target_id == credential_rows[0].id,
                )
            )
            == 1
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_provider_credential_concurrent_different_keys_same_name_conflict(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    admin_token, _agent_token = await _tokens(migrated_database, settings)
    actor = await _admin_principal(migrated_database, settings, admin_token)
    repository_factory = _barrier_repository_factory(asyncio.Barrier(2))
    shared_name = "concurrent unique provider name"

    async with (
        migrated_database.session() as first_session,
        migrated_database.session() as second_session,
    ):
        services = (
            ProviderCredentialService(
                session=first_session,
                settings=settings,
                keyring_factory=lambda: ProviderCredentialKeyring(
                    keys={KEY_VERSION: KEY},
                    active_key_version=KEY_VERSION,
                ),
                repository_factory=repository_factory,
            ),
            ProviderCredentialService(
                session=second_session,
                settings=settings,
                keyring_factory=lambda: ProviderCredentialKeyring(
                    keys={KEY_VERSION: KEY},
                    active_key_version=KEY_VERSION,
                ),
                repository_factory=repository_factory,
            ),
        )
        outcomes = await asyncio.wait_for(
            asyncio.gather(
                services[0].create_credential(
                    ProviderCredentialCreate(
                        name=shared_name,
                        secret=SecretStr("same-name-first-secret"),
                    ),
                    actor=actor,
                    request_id="req-same-name-first",
                    idempotency_key="same-name-first-key",
                ),
                services[1].create_credential(
                    ProviderCredentialCreate(
                        name=shared_name,
                        secret=SecretStr("same-name-second-secret"),
                    ),
                    actor=actor,
                    request_id="req-same-name-second",
                    idempotency_key="same-name-second-key",
                ),
                return_exceptions=True,
            ),
            timeout=15,
        )

    successes = [
        outcome for outcome in outcomes if isinstance(outcome, ProviderCredentialCreateResult)
    ]
    failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
    assert len(successes) == 1
    assert len(failures) == 1
    failure = cast(BusinessError, failures[0])
    assert (failure.status_code, failure.code) == (409, "RESOURCE_ALREADY_EXISTS")

    async with migrated_database.session() as session:
        credentials = (
            (
                await session.execute(
                    select(ProviderCredential).where(ProviderCredential.name == shared_name)
                )
            )
            .scalars()
            .all()
        )
        assert len(credentials) == 1
        assert (
            await session.scalar(
                select(func.count())
                .select_from(IdempotencyRecord)
                .where(
                    IdempotencyRecord.actor_key_id == actor.key_id,
                    IdempotencyRecord.operation == "provider_credential.create",
                    IdempotencyRecord.idempotency_key.in_(
                        ("same-name-first-key", "same-name-second-key")
                    ),
                )
            )
            == 1
        )
        assert (
            await session.scalar(
                select(func.count())
                .select_from(AuditEvent)
                .where(
                    AuditEvent.action == "provider_credential.created",
                    AuditEvent.target_id == credentials[0].id,
                )
            )
            == 1
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_provider_credential_concurrent_patch_same_etag_has_one_winner(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    first_token, _first_agent = await _tokens(migrated_database, settings)
    second_token, _second_agent = await _tokens(migrated_database, settings)
    first_actor = await _admin_principal(migrated_database, settings, first_token)
    second_actor = await _admin_principal(migrated_database, settings, second_token)

    async with migrated_database.session() as seed_session:
        seed_service = ProviderCredentialService(
            session=seed_session,
            settings=settings,
            keyring_factory=lambda: ProviderCredentialKeyring(
                keys={KEY_VERSION: KEY},
                active_key_version=KEY_VERSION,
            ),
        )
        seeded = await seed_service.create_credential(
            ProviderCredentialCreate(
                name="concurrent patch seed",
                secret=SecretStr("concurrent-patch-secret"),
            ),
            actor=first_actor,
            request_id="req-concurrent-patch-seed",
            idempotency_key="concurrent-patch-seed",
        )

    credential_id = seeded.credential.id
    initial_etag = provider_credential_etag(credential_id, 1)
    repository_factory = _credential_barrier_repository_factory(asyncio.Barrier(2))
    async with (
        migrated_database.session() as first_session,
        migrated_database.session() as second_session,
    ):
        services = (
            ProviderCredentialService(
                session=first_session,
                settings=settings,
                keyring_factory=lambda: ProviderCredentialKeyring(
                    keys={KEY_VERSION: KEY},
                    active_key_version=KEY_VERSION,
                ),
                repository_factory=repository_factory,
            ),
            ProviderCredentialService(
                session=second_session,
                settings=settings,
                keyring_factory=lambda: ProviderCredentialKeyring(
                    keys={KEY_VERSION: KEY},
                    active_key_version=KEY_VERSION,
                ),
                repository_factory=repository_factory,
            ),
        )
        outcomes = await asyncio.wait_for(
            asyncio.gather(
                services[0].update_credential(
                    credential_id,
                    ProviderCredentialPatch(name="concurrent patch first"),
                    actor=first_actor,
                    request_id="req-concurrent-patch-first",
                    expected_etag=initial_etag,
                ),
                services[1].update_credential(
                    credential_id,
                    ProviderCredentialPatch(name="concurrent patch second"),
                    actor=second_actor,
                    request_id="req-concurrent-patch-second",
                    expected_etag=initial_etag,
                ),
                return_exceptions=True,
            ),
            timeout=15,
        )

    successes = [outcome for outcome in outcomes if not isinstance(outcome, BaseException)]
    failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
    assert len(successes) == 1
    assert len(failures) == 1
    failure = cast(BusinessError, failures[0])
    assert (failure.status_code, failure.code) == (412, "PRECONDITION_FAILED")

    persisted = await _credential(migrated_database, credential_id)
    assert persisted.resource_revision == 2
    assert persisted.name in {"concurrent patch first", "concurrent patch second"}
    assert (
        await _audit_count(
            migrated_database,
            credential_id,
            "provider_credential.updated",
        )
        == 1
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_provider_credential_read_and_name_patch_ignore_unavailable_keyring(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    admin_token, _agent_token = await _tokens(migrated_database, settings)
    valid_app = _app(migrated_database, settings)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=valid_app),
        base_url="http://test",
    ) as client:
        created = await client.post(
            "/v1/admin/provider-credentials",
            headers={
                "Authorization": f"Bearer {admin_token}",
                "Idempotency-Key": "lazy-http-seed",
            },
            json={"name": "lazy HTTP credential", "secret": SECRET},
        )
        assert created.status_code == 201

    unavailable_settings = settings.model_copy(
        update={"provider_credential_keyring": SecretStr("not-json")}
    )
    unavailable_app = _app(migrated_database, unavailable_settings)
    credential_id = UUID(created.json()["id"])
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=unavailable_app),
        base_url="http://test",
    ) as client:
        listed = await client.get(
            "/v1/admin/provider-credentials",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert listed.status_code == 200

        detail = await client.get(
            f"/v1/admin/provider-credentials/{credential_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert detail.status_code == 200

        renamed = await client.patch(
            f"/v1/admin/provider-credentials/{credential_id}",
            headers={
                "Authorization": f"Bearer {admin_token}",
                "If-Match": created.headers["etag"],
            },
            json={"name": "lazy HTTP renamed"},
        )
        assert renamed.status_code == 200

        rotate = await client.patch(
            f"/v1/admin/provider-credentials/{credential_id}",
            headers={
                "Authorization": f"Bearer {admin_token}",
                "If-Match": renamed.headers["etag"],
            },
            json={"secret": ROTATED_SECRET},
        )
        assert rotate.status_code == 503
        assert rotate.json()["error"]["code"] == "PROVIDER_CREDENTIAL_UNAVAILABLE"

        create = await client.post(
            "/v1/admin/provider-credentials",
            headers={
                "Authorization": f"Bearer {admin_token}",
                "Idempotency-Key": "lazy-http-unavailable",
            },
            json={"name": "must fail", "secret": SECRET},
        )
        assert create.status_code == 503
        assert create.json()["error"]["code"] == "PROVIDER_CREDENTIAL_UNAVAILABLE"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_provider_credential_real_transactions_rollback_after_audit_failure(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    admin_token, _agent_token = await _tokens(migrated_database, settings)
    actor = await _admin_principal(migrated_database, settings, admin_token)

    async with migrated_database.session() as session:
        failing_create_service = ProviderCredentialService(
            session=session,
            settings=settings,
            keyring_factory=lambda: ProviderCredentialKeyring(
                keys={KEY_VERSION: KEY},
                active_key_version=KEY_VERSION,
            ),
            repository_factory=_failing_audit_repository_factory("provider_credential.created"),
        )
        with pytest.raises(BusinessError) as create_failure:
            await failing_create_service.create_credential(
                ProviderCredentialCreate(
                    name="audit rollback create",
                    secret=SecretStr(SECRET),
                ),
                actor=actor,
                request_id="req-audit-rollback-create",
                idempotency_key="audit-rollback-create",
            )
    assert (create_failure.value.status_code, create_failure.value.code) == (
        500,
        "INTERNAL_ERROR",
    )

    async with migrated_database.session() as session:
        assert (
            await session.scalar(
                select(func.count())
                .select_from(ProviderCredential)
                .where(ProviderCredential.name == "audit rollback create")
            )
            == 0
        )
        assert (
            await session.scalar(
                select(func.count())
                .select_from(IdempotencyRecord)
                .where(
                    IdempotencyRecord.actor_key_id == actor.key_id,
                    IdempotencyRecord.operation == "provider_credential.create",
                    IdempotencyRecord.idempotency_key == "audit-rollback-create",
                )
            )
            == 0
        )
        assert (
            await session.scalar(
                select(func.count())
                .select_from(AuditEvent)
                .where(AuditEvent.request_id == "req-audit-rollback-create")
            )
            == 0
        )

    async with migrated_database.session() as session:
        seed_service = ProviderCredentialService(
            session=session,
            settings=settings,
            keyring_factory=lambda: ProviderCredentialKeyring(
                keys={KEY_VERSION: KEY},
                active_key_version=KEY_VERSION,
            ),
        )
        seeded = await seed_service.create_credential(
            ProviderCredentialCreate(
                name="audit rollback rotation",
                secret=SecretStr(SECRET),
            ),
            actor=actor,
            request_id="req-audit-rollback-seed",
            idempotency_key="audit-rollback-seed",
        )
    credential_id = seeded.credential.id
    before = await _credential(migrated_database, credential_id)

    async with migrated_database.session() as session:
        failing_rotation_service = ProviderCredentialService(
            session=session,
            settings=settings,
            keyring_factory=lambda: ProviderCredentialKeyring(
                keys={KEY_VERSION: KEY},
                active_key_version=KEY_VERSION,
            ),
            repository_factory=_failing_audit_repository_factory("provider_credential.rotated"),
        )
        with pytest.raises(BusinessError) as rotation_failure:
            await failing_rotation_service.update_credential(
                credential_id,
                ProviderCredentialPatch(secret=SecretStr(ROTATED_SECRET)),
                actor=actor,
                request_id="req-audit-rollback-rotation",
                expected_etag=provider_credential_etag(credential_id, 1),
            )
    assert (rotation_failure.value.status_code, rotation_failure.value.code) == (
        500,
        "INTERNAL_ERROR",
    )

    after = await _credential(migrated_database, credential_id)
    assert after.resource_revision == before.resource_revision == 1
    assert after.name == before.name
    assert after.ciphertext == before.ciphertext
    assert after.nonce == before.nonce
    assert after.key_version == before.key_version
    assert after.rotated_at == before.rotated_at is None
    assert (
        await _audit_count(
            migrated_database,
            credential_id,
            "provider_credential.rotated",
        )
        == 0
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_provider_credential_real_transaction_revalidates_revoked_admin(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    admin_token, _agent_token = await _tokens(migrated_database, settings)
    actor = await _admin_principal(migrated_database, settings, admin_token)

    async with migrated_database.session() as session, session.begin():
        await session.execute(
            update(ApiKey)
            .where(ApiKey.id == actor.key_id)
            .values(status="revoked", revoked_at=func.now())
        )

    async with migrated_database.session() as session:
        service = ProviderCredentialService(
            session=session,
            settings=settings,
            keyring_factory=lambda: ProviderCredentialKeyring(
                keys={KEY_VERSION: KEY},
                active_key_version=KEY_VERSION,
            ),
        )
        with pytest.raises(BusinessError) as captured:
            await service.create_credential(
                ProviderCredentialCreate(
                    name="revoked admin must not create",
                    secret=SecretStr(SECRET),
                ),
                actor=actor,
                request_id="req-revoked-admin-real-transaction",
                idempotency_key="revoked-admin-real-transaction",
            )

    assert (captured.value.status_code, captured.value.code) == (401, "INVALID_API_KEY")
    async with migrated_database.session() as session:
        assert (
            await session.scalar(
                select(func.count())
                .select_from(ProviderCredential)
                .where(ProviderCredential.name == "revoked admin must not create")
            )
            == 0
        )
        assert (
            await session.scalar(
                select(func.count())
                .select_from(IdempotencyRecord)
                .where(
                    IdempotencyRecord.actor_key_id == actor.key_id,
                    IdempotencyRecord.operation == "provider_credential.create",
                    IdempotencyRecord.idempotency_key == "revoked-admin-real-transaction",
                )
            )
            == 0
        )
        assert (
            await session.scalar(
                select(func.count())
                .select_from(AuditEvent)
                .where(AuditEvent.request_id == "req-revoked-admin-real-transaction")
            )
            == 0
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_provider_config_http_create_replay_list_get_and_patch_both_mvp_types(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url, allow_private_targets=True)
    admin_token, _agent_token = await _tokens(migrated_database, settings)
    app = _app(migrated_database, settings)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        credential_id = await _create_credential_http(
            client,
            admin_token,
            name="provider config credential",
            key="provider-config-credential",
        )
        headers = {
            "Authorization": f"Bearer {admin_token}",
            "Idempotency-Key": "provider-config-create",
            "X-Request-ID": "req-provider-config-create",
        }
        created = await client.post(
            "/v1/admin/provider-configs",
            headers=headers,
            json=_provider_config_payload(credential_id),
        )
        assert created.status_code == 201
        assert created.headers["cache-control"] == "no-store"
        provider_config_id = UUID(created.json()["id"])
        assert created.headers["location"] == (f"/v1/admin/provider-configs/{provider_config_id}")
        assert created.headers["etag"] == provider_config_etag(provider_config_id, 1)
        assert created.json()["base_url"] == CONFIG_CANONICAL_LOCAL_URL
        assert created.json()["endpoint_policy_version"] == "provider-endpoint-v1"
        assert created.json()["endpoint_validated_at"] is not None
        assert created.json()["credential_id"] == str(credential_id)
        assert created.json()["default_headers"] == {"X-Title": "RAG integration"}
        _assert_safe_config_payload(created, SECRET, admin_token)

        detail = await client.get(
            f"/v1/admin/provider-configs/{provider_config_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert detail.status_code == 200
        assert detail.json() == created.json()
        assert detail.headers["etag"] == created.headers["etag"]

        listed = await client.get(
            "/v1/admin/provider-configs",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert listed.status_code == 200
        assert listed.headers["cache-control"] == "no-store"
        assert [item["id"] for item in listed.json()["items"]] == [str(provider_config_id)]
        _assert_safe_config_payload(listed, SECRET, admin_token)

        original_validated_at = created.json()["endpoint_validated_at"]
        operational = await client.patch(
            f"/v1/admin/provider-configs/{provider_config_id}",
            headers={
                "Authorization": f"Bearer {admin_token}",
                "If-Match": created.headers["etag"],
                "X-Request-ID": "req-provider-config-operational",
            },
            json={"timeout_seconds": 45, "max_concurrency": 4, "enabled": False},
        )
        assert operational.status_code == 200
        assert operational.json()["resource_revision"] == 2
        assert operational.json()["enabled"] is False
        assert operational.json()["endpoint_validated_at"] == original_validated_at

        changed = await client.patch(
            f"/v1/admin/provider-configs/{provider_config_id}",
            headers={
                "Authorization": f"Bearer {admin_token}",
                "If-Match": operational.headers["etag"],
                "X-Request-ID": "req-provider-config-endpoint-change",
            },
            json={"base_url": "https://LOCALHOST:9443/v2/"},
        )
        assert changed.status_code == 200
        assert changed.json()["base_url"] == "https://localhost:9443/v2"
        assert changed.json()["endpoint_validated_at"] != original_validated_at

        replayed = await client.post(
            "/v1/admin/provider-configs",
            headers={**headers, "X-Request-ID": "req-provider-config-replay"},
            json=_provider_config_payload(credential_id),
        )
        assert replayed.status_code == 200
        assert replayed.json() == created.json()
        assert replayed.headers["etag"] == created.headers["etag"]

        openrouter = await client.post(
            "/v1/admin/provider-configs",
            headers={
                "Authorization": f"Bearer {admin_token}",
                "Idempotency-Key": "provider-config-openrouter",
            },
            json=_provider_config_payload(
                credential_id,
                name="integration openrouter",
                provider_type="openrouter",
                routing_options={
                    "order": ["openai", "anthropic"],
                    "allow_fallbacks": False,
                    "data_collection": "deny",
                    "quantizations": ["fp16"],
                    "sort": {"by": "latency", "partition": "model"},
                    "preferred_max_latency": {"p90": 200},
                    "max_price": {"prompt": "0.25"},
                },
            ),
        )
        assert openrouter.status_code == 201
        assert openrouter.json()["provider_type"] == "openrouter"
        assert openrouter.json()["routing_options"]["data_collection"] == "deny"
        assert openrouter.json()["credential_id"] == str(credential_id)
        _assert_safe_config_payload(openrouter, SECRET, admin_token)

    persisted = await _provider_config(migrated_database, provider_config_id)
    assert persisted.base_url == "https://localhost:9443/v2"
    assert persisted.credential_id == credential_id
    assert persisted.secret_ref is None
    assert persisted.endpoint_policy_version == "provider-endpoint-v1"
    assert persisted.endpoint_validated_at is not None
    assert await _audit_count(migrated_database, provider_config_id, "provider_config.created") == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_provider_config_http_replays_validated_equivalent_numeric_inputs(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url, allow_private_targets=True)
    admin_token, _agent_token = await _tokens(migrated_database, settings)
    app = _app(migrated_database, settings)
    idempotency_key = "provider-config-canonical-number"
    config_name = "canonical numeric provider config"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        credential_id = await _create_credential_http(
            client,
            admin_token,
            name="canonical number credential",
            key="canonical-number-credential",
        )
        headers = {
            "Authorization": f"Bearer {admin_token}",
            "Idempotency-Key": idempotency_key,
            "X-Request-ID": "req-config-canonical-number-create",
        }
        created = await client.post(
            "/v1/admin/provider-configs",
            headers=headers,
            json=_provider_config_payload(
                credential_id,
                name=config_name,
                provider_type="openrouter",
                timeout_seconds=30,
                routing_options={
                    "preferred_max_latency": {"p90": 200},
                    "preferred_min_throughput": 0,
                },
            ),
        )
        replayed = await client.post(
            "/v1/admin/provider-configs",
            headers={**headers, "X-Request-ID": "req-config-canonical-number-replay"},
            json=_provider_config_payload(
                credential_id,
                name=config_name,
                provider_type="openrouter",
                timeout_seconds="30.000",
                routing_options={
                    "preferred_max_latency": {"p90": 200.0},
                    "preferred_min_throughput": -0.0,
                },
            ),
        )

        assert created.status_code == 201
        assert replayed.status_code == 200
        assert replayed.json() == created.json()
        for header in ("etag", "location", "cache-control"):
            assert replayed.headers[header] == created.headers[header]
        provider_config_id = UUID(created.json()["id"])

    async with migrated_database.session() as session:
        assert (
            await session.scalar(
                select(func.count())
                .select_from(ProviderConfig)
                .where(ProviderConfig.name == config_name)
            )
            == 1
        )
        assert (
            await session.scalar(
                select(func.count())
                .select_from(IdempotencyRecord)
                .where(
                    IdempotencyRecord.operation == "provider_config.create",
                    IdempotencyRecord.idempotency_key == idempotency_key,
                )
            )
            == 1
        )
        assert (
            await session.scalar(
                select(func.count())
                .select_from(AuditEvent)
                .where(
                    AuditEvent.action == "provider_config.created",
                    AuditEvent.target_id == provider_config_id,
                )
            )
            == 1
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_provider_config_http_fingerprint_preserves_large_integer_identity(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url, allow_private_targets=True)
    admin_token, _agent_token = await _tokens(migrated_database, settings)
    app = _app(migrated_database, settings)
    idempotency_key = "provider-config-large-number"
    config_name = "large numeric provider config"
    first_value = 12345678901234567890123456781
    second_value = 12345678901234567890123456782

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        credential_id = await _create_credential_http(
            client,
            admin_token,
            name="large number credential",
            key="large-number-credential",
        )
        headers = {
            "Authorization": f"Bearer {admin_token}",
            "Idempotency-Key": idempotency_key,
        }
        created = await client.post(
            "/v1/admin/provider-configs",
            headers={**headers, "X-Request-ID": "req-config-large-number-create"},
            json=_provider_config_payload(
                credential_id,
                name=config_name,
                provider_type="openrouter",
                routing_options={"preferred_min_throughput": first_value},
            ),
        )
        conflicted = await client.post(
            "/v1/admin/provider-configs",
            headers={**headers, "X-Request-ID": "req-config-large-number-conflict"},
            json=_provider_config_payload(
                credential_id,
                name=config_name,
                provider_type="openrouter",
                routing_options={"preferred_min_throughput": second_value},
            ),
        )

        assert created.status_code == 201
        assert conflicted.status_code == 409
        assert conflicted.json()["error"]["code"] == "IDEMPOTENCY_KEY_REUSED"
        provider_config_id = UUID(created.json()["id"])

    async with migrated_database.session() as session:
        persisted = await session.get(ProviderConfig, provider_config_id)
        assert persisted is not None
        assert persisted.routing_options == {"preferred_min_throughput": first_value}
        assert (
            await session.scalar(
                select(func.count())
                .select_from(ProviderConfig)
                .where(ProviderConfig.name == config_name)
            )
            == 1
        )
        assert (
            await session.scalar(
                select(func.count())
                .select_from(IdempotencyRecord)
                .where(
                    IdempotencyRecord.operation == "provider_config.create",
                    IdempotencyRecord.idempotency_key == idempotency_key,
                )
            )
            == 1
        )
        assert (
            await session.scalar(
                select(func.count())
                .select_from(AuditEvent)
                .where(
                    AuditEvent.action == "provider_config.created",
                    AuditEvent.target_id == provider_config_id,
                )
            )
            == 1
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_provider_config_http_cursor_pagination_is_stable_and_complete(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url, allow_private_targets=True)
    admin_token, _agent_token = await _tokens(migrated_database, settings)
    app = _app(migrated_database, settings)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        credential_id = await _create_credential_http(
            client,
            admin_token,
            name="provider config pagination credential",
            key="provider-config-pagination-credential",
        )
        for index in range(5):
            created = await client.post(
                "/v1/admin/provider-configs",
                headers={
                    "Authorization": f"Bearer {admin_token}",
                    "Idempotency-Key": f"provider-config-pagination-{index}",
                },
                json=_provider_config_payload(
                    credential_id,
                    name=f"paginated integration config {index}",
                ),
            )
            assert created.status_code == 201

        first = await client.get(
            "/v1/admin/provider-configs",
            headers={"Authorization": f"Bearer {admin_token}"},
            params={"limit": 2},
        )
        repeated_first = await client.get(
            "/v1/admin/provider-configs",
            headers={"Authorization": f"Bearer {admin_token}"},
            params={"limit": 2},
        )
        assert first.status_code == repeated_first.status_code == 200
        assert first.json() == repeated_first.json()

        traversed: list[UUID] = []
        page = first
        while True:
            traversed.extend(UUID(item["id"]) for item in page.json()["items"])
            cursor = page.json()["next_cursor"]
            if cursor is None:
                break
            page = await client.get(
                "/v1/admin/provider-configs",
                headers={"Authorization": f"Bearer {admin_token}"},
                params={"limit": 2, "cursor": cursor},
            )
            assert page.status_code == 200

        invalid = await client.get(
            "/v1/admin/provider-configs",
            headers={"Authorization": f"Bearer {admin_token}"},
            params={"limit": 2, "cursor": "not-a-canonical-cursor"},
        )
        assert invalid.status_code == 422
        assert invalid.json()["error"]["code"] == "VALIDATION_ERROR"

    async with migrated_database.session() as session:
        expected = list(
            (
                await session.scalars(
                    select(ProviderConfig.id)
                    .where(ProviderConfig.name.like("paginated integration config %"))
                    .order_by(ProviderConfig.created_at, ProviderConfig.id)
                )
            ).all()
        )
    assert traversed == expected
    assert len(traversed) == len(set(traversed)) == 5


@pytest.mark.integration
@pytest.mark.asyncio
async def test_provider_config_http_rejections_are_safe_and_private_target_leaves_no_residue(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    permissive_settings = _settings(async_url, allow_private_targets=True)
    admin_token, agent_token = await _tokens(migrated_database, permissive_settings)
    permissive_app = _app(migrated_database, permissive_settings)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=permissive_app),
        base_url="http://test",
    ) as client:
        credential_id = await _create_credential_http(
            client,
            admin_token,
            name="provider rejection credential",
            key="provider-rejection-credential",
        )
        base_payload = _provider_config_payload(credential_id)
        cases = (
            ({**base_payload, "secret_ref": "env:UNSAFE"}, 422),
            ({**base_payload, "provider_type": "vendor_specific"}, 422),
            ({**base_payload, "default_headers": {"Authorization": "Bearer unsafe"}}, 422),
            ({**base_payload, "default_headers": {"X-Unknown": "unsafe"}}, 422),
            ({**base_payload, "routing_options": {"only": ["openai"]}}, 422),
            (
                {
                    **base_payload,
                    "provider_type": "openrouter",
                    "routing_options": {"unknown": True},
                },
                422,
            ),
            ({**base_payload, "timeout_seconds": 0}, 422),
            ({**base_payload, "max_concurrency": 10001}, 422),
            ({**base_payload, "requests_per_minute": 1000001}, 422),
            ({**base_payload, "credential_id": str(uuid4())}, 404),
        )
        for index, (payload, status) in enumerate(cases):
            response = await client.post(
                "/v1/admin/provider-configs",
                headers={
                    "Authorization": f"Bearer {admin_token}",
                    "Idempotency-Key": f"provider-rejection-{index}",
                },
                json=payload,
            )
            assert response.status_code == status
            assert response.headers["cache-control"] == "no-store"
            _assert_safe_config_payload(response, SECRET, admin_token)

        denied = await client.get(
            "/v1/admin/provider-configs",
            headers={"Authorization": f"Bearer {agent_token}"},
        )
        assert denied.status_code == 401
        assert denied.json()["error"]["code"] == "INVALID_API_KEY"

    strict_settings = _settings(async_url)
    strict_app = _app(migrated_database, strict_settings)
    residue_name = "private target must not persist"
    residue_key = "private-target-no-residue"
    residue_request_id = "req-private-target-no-residue"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=strict_app),
        base_url="http://test",
    ) as client:
        rejected = await client.post(
            "/v1/admin/provider-configs",
            headers={
                "Authorization": f"Bearer {admin_token}",
                "Idempotency-Key": residue_key,
                "X-Request-ID": residue_request_id,
            },
            json=_provider_config_payload(
                credential_id,
                name=residue_name,
                base_url="https://127.0.0.1:8443/v1",
            ),
        )
        assert rejected.status_code == 422
        assert rejected.json()["error"]["code"] == "PROVIDER_ENDPOINT_REJECTED"
        _assert_safe_config_payload(rejected, SECRET, admin_token)

    async with migrated_database.session() as session:
        assert (
            await session.scalar(
                select(func.count())
                .select_from(ProviderConfig)
                .where(ProviderConfig.name == residue_name)
            )
            == 0
        )
        assert (
            await session.scalar(
                select(func.count())
                .select_from(IdempotencyRecord)
                .where(
                    IdempotencyRecord.operation == "provider_config.create",
                    IdempotencyRecord.idempotency_key == residue_key,
                )
            )
            == 0
        )
        assert (
            await session.scalar(
                select(func.count())
                .select_from(AuditEvent)
                .where(AuditEvent.request_id == residue_request_id)
            )
            == 0
        )


class _BarrierConfigRepository:
    def __init__(
        self,
        delegate: ProviderConfigRepository,
        barrier: asyncio.Barrier,
    ) -> None:
        self._delegate = delegate
        self._barrier = barrier
        self._first_update_read = True

    async def list_safe(
        self,
        position: CursorPosition | None,
        limit: int,
    ) -> list[ProviderConfigRecord]:
        return await self._delegate.list_safe(position, limit)

    async def get_safe(
        self,
        provider_config_id: UUID,
        *,
        for_update: bool = False,
    ) -> ProviderConfigRecord | None:
        if for_update and self._first_update_read:
            self._first_update_read = False
            await self._barrier.wait()
        return await self._delegate.get_safe(provider_config_id, for_update=for_update)

    async def get_secret_source(
        self,
        provider_config_id: UUID,
    ) -> ProviderConfigSecretSourceRecord | None:
        return await self._delegate.get_secret_source(provider_config_id)

    async def add_validated(
        self,
        provider_config_id: UUID,
        *,
        name: str,
        provider_type: str,
        base_url: str,
        credential_id: UUID,
        default_headers: dict[str, str],
        routing_options: dict[str, object],
        timeout_seconds: Decimal,
        max_concurrency: int,
        requests_per_minute: int,
        enabled: bool,
        endpoint_policy_version: str,
        endpoint_validated_at: datetime,
    ) -> ProviderConfigRecord:
        return await self._delegate.add_validated(
            provider_config_id,
            name=name,
            provider_type=provider_type,
            base_url=base_url,
            credential_id=credential_id,
            default_headers=default_headers,
            routing_options=routing_options,
            timeout_seconds=timeout_seconds,
            max_concurrency=max_concurrency,
            requests_per_minute=requests_per_minute,
            enabled=enabled,
            endpoint_policy_version=endpoint_policy_version,
            endpoint_validated_at=endpoint_validated_at,
        )

    async def update_validated(
        self,
        provider_config_id: UUID,
        *,
        values: dict[str, object],
        updated_at: datetime,
    ) -> ProviderConfigRecord:
        return await self._delegate.update_validated(
            provider_config_id,
            values=values,
            updated_at=updated_at,
        )


class _CrossResourceConfigRepository(_BarrierConfigRepository):
    def __init__(
        self,
        delegate: ProviderConfigRepository,
        session: AsyncSession,
        *,
        role: str,
        provider_config_id: UUID,
        provider_locked: asyncio.Event,
        model_attempting_provider: asyncio.Event,
    ) -> None:
        super().__init__(delegate, asyncio.Barrier(1))
        self._session = session
        self._role = role
        self._provider_config_id = provider_config_id
        self._provider_locked = provider_locked
        self._model_attempting_provider = model_attempting_provider

    async def get_safe(
        self,
        provider_config_id: UUID,
        *,
        for_update: bool = False,
    ) -> ProviderConfigRecord | None:
        if not for_update or provider_config_id != self._provider_config_id:
            return await self._delegate.get_safe(provider_config_id, for_update=for_update)
        await self._session.execute(text("SET LOCAL lock_timeout = '1500ms'"))
        await self._session.execute(text("SET LOCAL statement_timeout = '4s'"))
        if self._role == "provider":
            row = await self._delegate.get_safe(provider_config_id, for_update=True)
            self._provider_locked.set()
            await asyncio.wait_for(self._model_attempting_provider.wait(), timeout=5)
            return row
        self._model_attempting_provider.set()
        await asyncio.wait_for(self._provider_locked.wait(), timeout=5)
        return await self._delegate.get_safe(provider_config_id, for_update=True)


def _cross_resource_repository_factory(
    *,
    role: str,
    provider_config_id: UUID,
    provider_locked: asyncio.Event,
    model_attempting_provider: asyncio.Event,
) -> Callable[[AsyncSession], ProviderRepositories]:
    def factory(session: AsyncSession) -> ProviderRepositories:
        repositories = sqlalchemy_provider_repositories(session)
        assert repositories.configs is not None
        return ProviderRepositories(
            credentials=repositories.credentials,
            admins=repositories.admins,
            idempotency=repositories.idempotency,
            audits=repositories.audits,
            configs=_CrossResourceConfigRepository(
                repositories.configs,
                session,
                role=role,
                provider_config_id=provider_config_id,
                provider_locked=provider_locked,
                model_attempting_provider=model_attempting_provider,
            ),
            profiles=repositories.profiles,
        )

    return factory


def _config_barrier_repository_factory(
    barrier: asyncio.Barrier,
) -> Callable[[AsyncSession], ProviderRepositories]:
    def factory(session: AsyncSession) -> ProviderRepositories:
        repositories = sqlalchemy_provider_repositories(session)
        assert repositories.configs is not None
        return ProviderRepositories(
            credentials=repositories.credentials,
            admins=repositories.admins,
            idempotency=repositories.idempotency,
            audits=repositories.audits,
            configs=_BarrierConfigRepository(repositories.configs, barrier),
        )

    return factory


def _blocking_policy_factory(
    settings: Settings,
    resolver: _BlockingResolver,
) -> Callable[[], ProviderEndpointPolicy]:
    return lambda: ProviderEndpointPolicy(
        environment=settings.environment,
        allow_private_targets=True,
        resolver=resolver,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_provider_config_dns_resolution_holds_no_database_row_locks(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, sync_url = postgres_urls
    settings = _settings(async_url, allow_private_targets=True)
    admin_token, _agent_token = await _tokens(migrated_database, settings)
    actor = await _admin_principal(migrated_database, settings, admin_token)
    app = _app(migrated_database, settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        credential_id = await _create_credential_http(
            client,
            admin_token,
            name="dns lock credential",
            key="dns-lock-credential",
        )

    create_resolver = _BlockingResolver()
    async with migrated_database.session() as create_session:
        create_service = ProviderConfigService(
            session=create_session,
            settings=settings,
            endpoint_policy_factory=_blocking_policy_factory(settings, create_resolver),
        )
        create_task = asyncio.create_task(
            create_service.create_provider_config(
                ProviderConfigCreate.model_validate(
                    _provider_config_payload(credential_id, name="dns lock create config")
                ),
                actor=actor,
                request_id="req-dns-lock-create",
                idempotency_key="dns-lock-create",
            )
        )
        await _wait_for_resolver(create_resolver)
        create_lock_failure = await _capture_sync_failure(
            lambda: _lock_admin_row(sync_url, actor.key_id)
        )
        create_resolver.release.set()
        created = await asyncio.wait_for(create_task, timeout=10)
    assert create_lock_failure is None

    update_resolver = _BlockingResolver()
    async with migrated_database.session() as update_session:
        update_service = ProviderConfigService(
            session=update_session,
            settings=settings,
            endpoint_policy_factory=_blocking_policy_factory(settings, update_resolver),
        )
        update_task = asyncio.create_task(
            update_service.update_provider_config(
                created.provider_config.id,
                ProviderConfigPatch(base_url="https://localhost:9443/v2"),
                actor=actor,
                request_id="req-dns-lock-update",
                expected_etag=provider_config_etag(created.provider_config.id, 1),
            )
        )
        await _wait_for_resolver(update_resolver)
        admin_lock_failure, config_lock_failure = await asyncio.gather(
            _capture_sync_failure(lambda: _lock_admin_row(sync_url, actor.key_id)),
            _capture_sync_failure(
                lambda: _lock_provider_config_row(sync_url, created.provider_config.id)
            ),
        )
        update_resolver.release.set()
        updated = await asyncio.wait_for(update_task, timeout=10)
    assert admin_lock_failure is None
    assert config_lock_failure is None
    assert updated.resource_revision == 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_provider_config_update_rechecks_etag_after_dns_resolution(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, sync_url = postgres_urls
    settings = _settings(async_url, allow_private_targets=True)
    admin_token, _agent_token = await _tokens(migrated_database, settings)
    actor = await _admin_principal(migrated_database, settings, admin_token)
    app = _app(migrated_database, settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        credential_id = await _create_credential_http(
            client,
            admin_token,
            name="dns etag credential",
            key="dns-etag-credential",
        )
        created = await client.post(
            "/v1/admin/provider-configs",
            headers={
                "Authorization": f"Bearer {admin_token}",
                "Idempotency-Key": "dns-etag-config",
            },
            json=_provider_config_payload(credential_id, name="dns etag config"),
        )
        assert created.status_code == 201
    provider_config_id = UUID(created.json()["id"])
    resolver = _BlockingResolver()
    async with migrated_database.session() as session:
        service = ProviderConfigService(
            session=session,
            settings=settings,
            endpoint_policy_factory=_blocking_policy_factory(settings, resolver),
        )
        update_task = asyncio.create_task(
            service.update_provider_config(
                provider_config_id,
                ProviderConfigPatch(base_url="https://localhost:9443/v2"),
                actor=actor,
                request_id="req-dns-etag-stale-update",
                expected_etag=provider_config_etag(provider_config_id, 1),
            )
        )
        await _wait_for_resolver(resolver)
        mutation_failure = await _capture_sync_failure(
            lambda: _advance_provider_config_revision(sync_url, provider_config_id)
        )
        resolver.release.set()
        outcome = await asyncio.gather(update_task, return_exceptions=True)

    assert mutation_failure is None
    assert len(outcome) == 1
    failure = outcome[0]
    assert isinstance(failure, BusinessError)
    assert (failure.status_code, failure.code) == (412, "PRECONDITION_FAILED")
    persisted = await _provider_config(migrated_database, provider_config_id)
    assert persisted.enabled is False
    assert persisted.resource_revision == 2
    assert persisted.base_url == CONFIG_CANONICAL_LOCAL_URL
    assert await _audit_count(migrated_database, provider_config_id, "provider_config.updated") == 0


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("intervening_change", "expected_status", "expected_code"),
    (
        ("admin_revoke", 401, "INVALID_API_KEY"),
        ("credential_delete", 404, "RESOURCE_NOT_FOUND"),
    ),
)
async def test_provider_config_create_rechecks_dependencies_after_dns_resolution(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
    intervening_change: str,
    expected_status: int,
    expected_code: str,
) -> None:
    async_url, sync_url = postgres_urls
    settings = _settings(async_url, allow_private_targets=True)
    admin_token, _agent_token = await _tokens(migrated_database, settings)
    actor = await _admin_principal(migrated_database, settings, admin_token)
    app = _app(migrated_database, settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        credential_id = await _create_credential_http(
            client,
            admin_token,
            name=f"dns dependency {intervening_change}",
            key=f"dns-dependency-{intervening_change}",
        )
    resolver = _BlockingResolver()
    config_name = f"dns dependency config {intervening_change}"
    idempotency_key = f"dns-dependency-config-{intervening_change}"
    request_id = f"req-dns-dependency-{intervening_change}"
    async with migrated_database.session() as session:
        service = ProviderConfigService(
            session=session,
            settings=settings,
            endpoint_policy_factory=_blocking_policy_factory(settings, resolver),
        )
        create_task = asyncio.create_task(
            service.create_provider_config(
                ProviderConfigCreate.model_validate(
                    _provider_config_payload(credential_id, name=config_name)
                ),
                actor=actor,
                request_id=request_id,
                idempotency_key=idempotency_key,
            )
        )
        await _wait_for_resolver(resolver)
        if intervening_change == "admin_revoke":
            mutation = partial(_revoke_admin_row, sync_url, actor.key_id)
        else:
            mutation = partial(_delete_provider_credential, sync_url, credential_id)
        mutation_failure = await _capture_sync_failure(mutation)
        resolver.release.set()
        outcome = await asyncio.gather(create_task, return_exceptions=True)

    assert mutation_failure is None
    failure = outcome[0]
    assert isinstance(failure, BusinessError)
    assert (failure.status_code, failure.code) == (expected_status, expected_code)
    async with migrated_database.session() as verification_session:
        assert (
            await verification_session.scalar(
                select(func.count())
                .select_from(ProviderConfig)
                .where(ProviderConfig.name == config_name)
            )
            == 0
        )
        assert (
            await verification_session.scalar(
                select(func.count())
                .select_from(IdempotencyRecord)
                .where(
                    IdempotencyRecord.operation == "provider_config.create",
                    IdempotencyRecord.idempotency_key == idempotency_key,
                )
            )
            == 0
        )
        assert (
            await verification_session.scalar(
                select(func.count())
                .select_from(AuditEvent)
                .where(AuditEvent.request_id == request_id)
            )
            == 0
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_provider_config_legacy_migration_rechecks_url_and_revision_after_dns(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, sync_url = postgres_urls
    settings = _settings(async_url, allow_private_targets=True)
    admin_token, _agent_token = await _tokens(migrated_database, settings)
    actor = await _admin_principal(migrated_database, settings, admin_token)
    app = _app(migrated_database, settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        credential_id = await _create_credential_http(
            client,
            admin_token,
            name="dns legacy migration credential",
            key="dns-legacy-migration-credential",
        )
    legacy_id = uuid4()
    async with migrated_database.session() as seed_session, seed_session.begin():
        seed_session.add(
            ProviderConfig(
                id=legacy_id,
                name="dns legacy migration config",
                provider_type="openai_compatible",
                base_url=CONFIG_LOCAL_URL,
                secret_ref="env:DNS_LEGACY_PROVIDER_KEY",
                credential_id=None,
                default_headers={},
                routing_options={},
                timeout_seconds=30,
                max_concurrency=4,
                requests_per_minute=60,
                enabled=True,
                resource_revision=1,
                endpoint_policy_version=None,
                endpoint_validated_at=None,
            )
        )

    resolver = _BlockingResolver()
    async with migrated_database.session() as session:
        service = ProviderConfigService(
            session=session,
            settings=settings,
            endpoint_policy_factory=_blocking_policy_factory(settings, resolver),
        )
        migration_task = asyncio.create_task(
            service.update_provider_config(
                legacy_id,
                ProviderConfigPatch(credential_id=credential_id),
                actor=actor,
                request_id="req-dns-legacy-migration",
                expected_etag=provider_config_etag(legacy_id, 1),
            )
        )
        await _wait_for_resolver(resolver)
        mutation_failure = await _capture_sync_failure(
            lambda: _advance_legacy_provider_url(sync_url, legacy_id)
        )
        resolver.release.set()
        outcome = await asyncio.gather(migration_task, return_exceptions=True)

    assert mutation_failure is None
    failure = outcome[0]
    assert isinstance(failure, BusinessError)
    assert (failure.status_code, failure.code) == (412, "PRECONDITION_FAILED")
    persisted = await _provider_config(migrated_database, legacy_id)
    assert persisted.base_url == "https://localhost:9443/replaced"
    assert persisted.resource_revision == 2
    assert persisted.credential_id is None
    assert persisted.secret_ref == "env:DNS_LEGACY_PROVIDER_KEY"
    assert await _audit_count(migrated_database, legacy_id, "provider_config.updated") == 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_provider_config_concurrent_idempotency_name_and_etag_are_serialized(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url, allow_private_targets=True)
    admin_token, _agent_token = await _tokens(migrated_database, settings)
    actor = await _admin_principal(migrated_database, settings, admin_token)
    async with migrated_database.session() as second_admin_session:
        second_admin_service = ApiKeyService(
            session=second_admin_session,
            authentication_sessions=migrated_database.session,
            settings=settings,
        )
        second_admin = await second_admin_service.create_admin_key(
            AdminApiKeyCreate(name="provider-config-concurrency-admin"),
            request_id="req-provider-config-concurrency-admin-bootstrap",
        )
    second_actor = await _admin_principal(
        migrated_database,
        settings,
        second_admin.token.get_secret_value(),
    )

    async with migrated_database.session() as seed_session:
        credential_service = ProviderCredentialService(
            session=seed_session,
            settings=settings,
            keyring_factory=lambda: ProviderCredentialKeyring(
                keys={KEY_VERSION: KEY},
                active_key_version=KEY_VERSION,
            ),
        )
        credential = await credential_service.create_credential(
            ProviderCredentialCreate(
                name="config concurrency credential",
                secret=SecretStr(SECRET),
            ),
            actor=actor,
            request_id="req-config-concurrency-credential",
            idempotency_key="config-concurrency-credential",
        )
    command = ProviderConfigCreate.model_validate(
        _provider_config_payload(credential.credential.id, name="concurrent config")
    )
    idempotency_factory = _barrier_repository_factory(asyncio.Barrier(2))

    def policy_factory() -> ProviderEndpointPolicy:
        return ProviderEndpointPolicy(
            environment=settings.environment,
            allow_private_targets=True,
        )

    async with (
        migrated_database.session() as first_session,
        migrated_database.session() as second_session,
    ):
        services = (
            ProviderConfigService(
                session=first_session,
                settings=settings,
                endpoint_policy_factory=policy_factory,
                repository_factory=idempotency_factory,
            ),
            ProviderConfigService(
                session=second_session,
                settings=settings,
                endpoint_policy_factory=policy_factory,
                repository_factory=idempotency_factory,
            ),
        )
        same_key = await asyncio.wait_for(
            asyncio.gather(
                services[0].create_provider_config(
                    command,
                    actor=actor,
                    request_id="req-config-concurrent-first",
                    idempotency_key="config-concurrent-same-key",
                ),
                services[1].create_provider_config(
                    command,
                    actor=actor,
                    request_id="req-config-concurrent-second",
                    idempotency_key="config-concurrent-same-key",
                ),
            ),
            timeout=15,
        )
    assert {result.created for result in same_key} == {True, False}
    assert same_key[0].provider_config == same_key[1].provider_config
    config_id = same_key[0].provider_config.id

    different_key_factory = _barrier_repository_factory(asyncio.Barrier(2))
    duplicate_name_command = ProviderConfigCreate.model_validate(
        _provider_config_payload(credential.credential.id, name="different-key same config")
    )
    async with (
        migrated_database.session() as first_session,
        migrated_database.session() as second_session,
    ):
        services = (
            ProviderConfigService(
                session=first_session,
                settings=settings,
                endpoint_policy_factory=policy_factory,
                repository_factory=different_key_factory,
            ),
            ProviderConfigService(
                session=second_session,
                settings=settings,
                endpoint_policy_factory=policy_factory,
                repository_factory=different_key_factory,
            ),
        )
        outcomes = await asyncio.wait_for(
            asyncio.gather(
                services[0].create_provider_config(
                    duplicate_name_command,
                    actor=actor,
                    request_id="req-config-different-key-first",
                    idempotency_key="config-different-key-first",
                ),
                services[1].create_provider_config(
                    duplicate_name_command,
                    actor=actor,
                    request_id="req-config-different-key-second",
                    idempotency_key="config-different-key-second",
                ),
                return_exceptions=True,
            ),
            timeout=15,
        )
    assert len([item for item in outcomes if isinstance(item, ProviderConfigCreateResult)]) == 1
    failures = [item for item in outcomes if isinstance(item, BaseException)]
    assert len(failures) == 1
    failure = cast(BusinessError, failures[0])
    assert (failure.status_code, failure.code) == (409, "RESOURCE_ALREADY_EXISTS")

    patch_factory = _config_barrier_repository_factory(asyncio.Barrier(2))
    initial_etag = provider_config_etag(config_id, 1)
    async with (
        migrated_database.session() as first_session,
        migrated_database.session() as second_session,
    ):
        services = (
            ProviderConfigService(
                session=first_session,
                settings=settings,
                endpoint_policy_factory=policy_factory,
                repository_factory=patch_factory,
            ),
            ProviderConfigService(
                session=second_session,
                settings=settings,
                endpoint_policy_factory=policy_factory,
                repository_factory=patch_factory,
            ),
        )
        patch_outcomes = await asyncio.wait_for(
            asyncio.gather(
                services[0].update_provider_config(
                    config_id,
                    ProviderConfigPatch(enabled=False),
                    actor=actor,
                    request_id="req-config-patch-first",
                    expected_etag=initial_etag,
                ),
                services[1].update_provider_config(
                    config_id,
                    ProviderConfigPatch(timeout_seconds=Decimal("35")),
                    actor=second_actor,
                    request_id="req-config-patch-second",
                    expected_etag=initial_etag,
                ),
                return_exceptions=True,
            ),
            timeout=15,
        )
    assert len([item for item in patch_outcomes if not isinstance(item, BaseException)]) == 1
    patch_failures = [item for item in patch_outcomes if isinstance(item, BaseException)]
    assert len(patch_failures) == 1
    assert (
        cast(BusinessError, patch_failures[0]).status_code,
        cast(BusinessError, patch_failures[0]).code,
    ) == (412, "PRECONDITION_FAILED")
    persisted = await _provider_config(migrated_database, config_id)
    assert persisted.resource_revision == 2
    assert await _audit_count(migrated_database, config_id, "provider_config.updated") == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_provider_config_legacy_row_is_safe_and_credential_migration_revalidates_endpoint(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url, allow_private_targets=True)
    admin_token, _agent_token = await _tokens(migrated_database, settings)
    legacy_id = uuid4()
    async with migrated_database.session() as session, session.begin():
        session.add(
            ProviderConfig(
                id=legacy_id,
                name="legacy provider config",
                provider_type="openai_compatible",
                base_url=CONFIG_LOCAL_URL,
                secret_ref="env:LEGACY_PROVIDER_KEY",
                credential_id=None,
                default_headers={},
                routing_options={},
                timeout_seconds=30,
                max_concurrency=4,
                requests_per_minute=60,
                enabled=True,
                resource_revision=1,
                endpoint_policy_version=None,
                endpoint_validated_at=None,
            )
        )
    app = _app(migrated_database, settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        detail = await client.get(
            f"/v1/admin/provider-configs/{legacy_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert detail.status_code == 200
        assert detail.json()["credential_id"] is None
        assert detail.json()["endpoint_policy_version"] is None
        assert detail.json()["endpoint_validated_at"] is None
        _assert_safe_config_payload(detail, "env:LEGACY_PROVIDER_KEY", admin_token)

        credential_id = await _create_credential_http(
            client,
            admin_token,
            name="legacy migration credential",
            key="legacy-migration-credential",
        )
        migrated = await client.patch(
            f"/v1/admin/provider-configs/{legacy_id}",
            headers={
                "Authorization": f"Bearer {admin_token}",
                "If-Match": detail.headers["etag"],
                "X-Request-ID": "req-legacy-config-migrate",
            },
            json={"credential_id": str(credential_id)},
        )
        assert migrated.status_code == 200
        assert migrated.json()["credential_id"] == str(credential_id)
        assert migrated.json()["base_url"] == CONFIG_CANONICAL_LOCAL_URL
        assert migrated.json()["endpoint_policy_version"] == "provider-endpoint-v1"
        assert migrated.json()["endpoint_validated_at"] is not None

    persisted = await _provider_config(migrated_database, legacy_id)
    assert persisted.credential_id == credential_id
    assert persisted.secret_ref is None
    assert persisted.endpoint_policy_version == "provider-endpoint-v1"
    assert persisted.endpoint_validated_at is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_provider_config_legacy_vendor_row_can_only_be_disabled(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url, allow_private_targets=True)
    admin_token, _agent_token = await _tokens(migrated_database, settings)
    legacy_id = uuid4()
    async with migrated_database.session() as session, session.begin():
        session.add(
            ProviderConfig(
                id=legacy_id,
                name="legacy vendor provider config",
                provider_type="vendor_specific",
                base_url=CONFIG_LOCAL_URL,
                secret_ref="env:LEGACY_VENDOR_PROVIDER_KEY",
                credential_id=None,
                default_headers={},
                routing_options={},
                timeout_seconds=30,
                max_concurrency=4,
                requests_per_minute=60,
                enabled=True,
                resource_revision=1,
                endpoint_policy_version=None,
                endpoint_validated_at=None,
            )
        )
    app = _app(migrated_database, settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        detail = await client.get(
            f"/v1/admin/provider-configs/{legacy_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert detail.status_code == 200
        assert detail.json()["endpoint_validated_at"] is None

        disabled = await client.patch(
            f"/v1/admin/provider-configs/{legacy_id}",
            headers={
                "Authorization": f"Bearer {admin_token}",
                "If-Match": detail.headers["etag"],
                "X-Request-ID": "req-disable-legacy-vendor",
            },
            json={"enabled": False},
        )
        assert disabled.status_code == 200
        assert disabled.json()["enabled"] is False
        assert disabled.json()["resource_revision"] == 2
        assert disabled.json()["endpoint_validated_at"] is None

        credential_id = await _create_credential_http(
            client,
            admin_token,
            name="legacy vendor rejected credential",
            key="legacy-vendor-rejected-credential",
        )
        rejected_patches = (
            {"enabled": True},
            {"provider_type": "openai_compatible"},
            {"credential_id": str(credential_id)},
            {"base_url": "https://localhost:9443/v2"},
            {"default_headers": {"X-Title": "replacement"}},
            {"routing_options": {"preferred_min_throughput": 1}},
            {"name": "converted legacy vendor"},
            {"timeout_seconds": 45},
            {"max_concurrency": 5},
            {"requests_per_minute": 61},
            {"enabled": False, "name": "mixed legacy vendor patch"},
        )
        for index, patch in enumerate(rejected_patches):
            rejected = await client.patch(
                f"/v1/admin/provider-configs/{legacy_id}",
                headers={
                    "Authorization": f"Bearer {admin_token}",
                    "If-Match": disabled.headers["etag"],
                    "X-Request-ID": f"req-reject-legacy-vendor-{index}",
                },
                json=patch,
            )
            assert rejected.status_code == 422
            assert rejected.json()["error"]["code"] == "VALIDATION_ERROR"

    persisted = await _provider_config(migrated_database, legacy_id)
    assert persisted.enabled is False
    assert persisted.resource_revision == 2
    assert persisted.endpoint_validated_at is None
    assert await _audit_count(migrated_database, legacy_id, "provider_config.updated") == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_model_profile_http_create_replay_list_get_patch_and_safe_validation(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url, allow_private_targets=True)
    admin_token, agent_token = await _tokens(migrated_database, settings)
    app = _app(migrated_database, settings)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        credential_id = await _create_credential_http(
            client,
            admin_token,
            name="model profile credential",
            key="model-profile-credential",
        )
        provider_config_id = await _create_provider_config_http(
            client,
            admin_token,
            credential_id,
            name="model profile provider",
            key="model-profile-provider",
        )

        missing_key = await client.post(
            "/v1/admin/model-profiles",
            headers={"Authorization": f"Bearer {admin_token}"},
            json=_model_profile_payload(provider_config_id),
        )
        assert missing_key.status_code == 422
        assert missing_key.json()["error"]["code"] == "VALIDATION_ERROR"

        created = await client.post(
            "/v1/admin/model-profiles",
            headers={
                "Authorization": f"Bearer {admin_token}",
                "Idempotency-Key": "model-profile-http-create",
            },
            json=_model_profile_payload(provider_config_id),
        )
        assert created.status_code == 201
        profile_id = UUID(created.json()["id"])
        assert created.headers["location"] == f"/v1/admin/model-profiles/{profile_id}"
        assert created.headers["etag"].endswith(':r1"')
        assert created.headers["cache-control"] == "no-store"
        assert created.json()["vector_config"] == {}
        assert created.json()["resource_revision"] == 1
        serialized = created.text.lower()
        for forbidden in (SECRET.lower(), "secret_ref", "ciphertext", "nonce"):
            assert forbidden not in serialized

        replay = await client.post(
            "/v1/admin/model-profiles",
            headers={
                "Authorization": f"Bearer {admin_token}",
                "Idempotency-Key": "model-profile-http-create",
            },
            json=_model_profile_payload(provider_config_id, timeout_seconds=30.0),
        )
        assert replay.status_code == 200
        assert replay.json() == created.json()
        assert replay.headers["etag"] == created.headers["etag"]

        conflict = await client.post(
            "/v1/admin/model-profiles",
            headers={
                "Authorization": f"Bearer {admin_token}",
                "Idempotency-Key": "model-profile-http-create",
            },
            json=_model_profile_payload(provider_config_id, model_name="different-model"),
        )
        assert conflict.status_code == 409
        assert conflict.json()["error"]["code"] == "IDEMPOTENCY_KEY_REUSED"

        listed = await client.get(
            "/v1/admin/model-profiles",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        detail = await client.get(
            f"/v1/admin/model-profiles/{profile_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert listed.status_code == detail.status_code == 200
        assert [item["id"] for item in listed.json()["items"]] == [str(profile_id)]
        assert detail.json() == created.json()
        assert detail.headers["etag"] == created.headers["etag"]

        missing_match = await client.patch(
            f"/v1/admin/model-profiles/{profile_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"batch_size": 32},
        )
        assert missing_match.status_code == 428
        assert missing_match.json()["error"]["code"] == "PRECONDITION_REQUIRED"

        patched = await client.patch(
            f"/v1/admin/model-profiles/{profile_id}",
            headers={
                "Authorization": f"Bearer {admin_token}",
                "If-Match": created.headers["etag"],
            },
            json={"name": "patched embedding profile", "batch_size": 32},
        )
        assert patched.status_code == 200
        assert patched.json()["resource_revision"] == 2
        assert patched.json()["batch_size"] == 32

        invalid_payloads = (
            _model_profile_payload(provider_config_id, capability="rerank"),
            _model_profile_payload(provider_config_id, capability="chat"),
            _model_profile_payload(provider_config_id, vector_config={"normalize": True}),
            {**_model_profile_payload(provider_config_id), "enabled": None},
        )
        for index, payload in enumerate(invalid_payloads):
            invalid = await client.post(
                "/v1/admin/model-profiles",
                headers={
                    "Authorization": f"Bearer {admin_token}",
                    "Idempotency-Key": f"model-profile-invalid-{index}",
                },
                json=payload,
            )
            assert invalid.status_code == 422
            assert invalid.json()["error"]["code"] == "VALIDATION_ERROR"

        unauthorized = await client.get(
            "/v1/admin/model-profiles",
            headers={"Authorization": f"Bearer {agent_token}"},
        )
        assert unauthorized.status_code == 401

        disabled_provider_id = await _create_provider_config_http(
            client,
            admin_token,
            credential_id,
            name="disabled model profile provider",
            key="disabled-model-profile-provider",
            enabled=False,
        )
        disabled_create = await client.post(
            "/v1/admin/model-profiles",
            headers={
                "Authorization": f"Bearer {admin_token}",
                "Idempotency-Key": "disabled-model-profile-create",
            },
            json=_model_profile_payload(
                disabled_provider_id,
                name="disabled provider model profile",
            ),
        )
        assert disabled_create.status_code == 409
        assert disabled_create.json()["error"]["code"] == "PROVIDER_CONFIG_DISABLED"

        provider_detail = await client.get(
            f"/v1/admin/provider-configs/{provider_config_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        disabled_provider = await client.patch(
            f"/v1/admin/provider-configs/{provider_config_id}",
            headers={
                "Authorization": f"Bearer {admin_token}",
                "If-Match": provider_detail.headers["etag"],
            },
            json={"enabled": False},
        )
        assert disabled_provider.status_code == 200
        disabled_profile = await client.patch(
            f"/v1/admin/model-profiles/{profile_id}",
            headers={
                "Authorization": f"Bearer {admin_token}",
                "If-Match": patched.headers["etag"],
            },
            json={"enabled": False},
        )
        assert disabled_profile.status_code == 200
        enable_profile = await client.patch(
            f"/v1/admin/model-profiles/{profile_id}",
            headers={
                "Authorization": f"Bearer {admin_token}",
                "If-Match": disabled_profile.headers["etag"],
            },
            json={"enabled": True},
        )
        assert enable_profile.status_code == 409
        assert enable_profile.json()["error"]["code"] == "PROVIDER_CONFIG_DISABLED"

    persisted = await _model_profile(migrated_database, profile_id)
    assert persisted.resource_revision == 3
    assert persisted.enabled is False
    assert await _audit_count(migrated_database, profile_id, "model_profile.created") == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_model_profile_rejects_enabled_legacy_vendor_provider_without_residue(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url, allow_private_targets=True)
    admin_token, _agent_token = await _tokens(migrated_database, settings)
    app = _app(migrated_database, settings)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        credential_id = await _create_credential_http(
            client,
            admin_token,
            name="vendor profile credential",
            key="vendor-profile-credential",
        )
        supported_provider_id = await _create_provider_config_http(
            client,
            admin_token,
            credential_id,
            name="supported profile provider",
            key="supported-profile-provider",
        )

        vendor_provider_id = uuid4()
        legacy_vendor_profile_id = uuid4()
        async with migrated_database.session() as session, session.begin():
            session.add(
                ProviderConfig(
                    id=vendor_provider_id,
                    name="enabled legacy vendor provider",
                    provider_type="vendor_specific",
                    base_url=CONFIG_CANONICAL_LOCAL_URL,
                    credential_id=credential_id,
                    secret_ref=None,
                    default_headers={},
                    routing_options={},
                    timeout_seconds=Decimal("30"),
                    max_concurrency=8,
                    requests_per_minute=600,
                    enabled=True,
                    resource_revision=1,
                    endpoint_policy_version="provider-endpoint-v1",
                    endpoint_validated_at=datetime.now(UTC),
                )
            )
            await session.flush()
            session.add(
                ModelProfile(
                    id=legacy_vendor_profile_id,
                    name="disabled legacy vendor embedding profile",
                    capability="embedding",
                    provider_config_id=vendor_provider_id,
                    model_name="legacy-vendor-embedding",
                    dimension=1536,
                    max_input_tokens=8191,
                    batch_size=64,
                    timeout_seconds=Decimal("30"),
                    vector_config={},
                    enabled=False,
                    resource_revision=1,
                )
            )

        create_name = "rejected vendor provider profile"
        create_key = "rejected-vendor-provider-profile"
        create_request_id = "req-rejected-vendor-provider-profile"
        rejected_create = await client.post(
            "/v1/admin/model-profiles",
            headers={
                "Authorization": f"Bearer {admin_token}",
                "Idempotency-Key": create_key,
                "X-Request-ID": create_request_id,
            },
            json=_model_profile_payload(vendor_provider_id, name=create_name),
        )
        assert rejected_create.status_code == 422
        assert rejected_create.json()["error"]["code"] == "VALIDATION_ERROR"

        supported_profile_id, supported_etag = await _create_model_profile_http(
            client,
            admin_token,
            supported_provider_id,
            name="provider change source profile",
            key="provider-change-source-profile",
        )
        rejected_change = await client.patch(
            f"/v1/admin/model-profiles/{supported_profile_id}",
            headers={
                "Authorization": f"Bearer {admin_token}",
                "If-Match": supported_etag,
            },
            json={"provider_config_id": str(vendor_provider_id)},
        )
        assert rejected_change.status_code == 422
        assert rejected_change.json()["error"]["code"] == "VALIDATION_ERROR"

        rejected_enable = await client.patch(
            f"/v1/admin/model-profiles/{legacy_vendor_profile_id}",
            headers={
                "Authorization": f"Bearer {admin_token}",
                "If-Match": model_profile_etag(legacy_vendor_profile_id, 1),
            },
            json={"enabled": True},
        )
        assert rejected_enable.status_code == 422
        assert rejected_enable.json()["error"]["code"] == "VALIDATION_ERROR"

    async with migrated_database.session() as session:
        assert (
            await session.scalar(
                select(func.count())
                .select_from(ModelProfile)
                .where(ModelProfile.name == create_name)
            )
            == 0
        )
        assert (
            await session.scalar(
                select(func.count())
                .select_from(IdempotencyRecord)
                .where(
                    IdempotencyRecord.operation == "model_profile.create",
                    IdempotencyRecord.idempotency_key == create_key,
                )
            )
            == 0
        )
        assert (
            await session.scalar(
                select(func.count())
                .select_from(AuditEvent)
                .where(AuditEvent.request_id == create_request_id)
            )
            == 0
        )
    supported_profile = await _model_profile(migrated_database, supported_profile_id)
    legacy_vendor_profile = await _model_profile(migrated_database, legacy_vendor_profile_id)
    assert supported_profile.provider_config_id == supported_provider_id
    assert supported_profile.resource_revision == 1
    assert legacy_vendor_profile.enabled is False
    assert legacy_vendor_profile.resource_revision == 1
    assert (
        await _audit_count(
            migrated_database,
            supported_profile_id,
            "model_profile.updated",
        )
        == 0
    )
    assert (
        await _audit_count(
            migrated_database,
            legacy_vendor_profile_id,
            "model_profile.updated",
        )
        == 0
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_model_profile_public_api_hides_legacy_non_embedding_profiles(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url, allow_private_targets=True)
    admin_token, _agent_token = await _tokens(migrated_database, settings)
    app = _app(migrated_database, settings)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        credential_id = await _create_credential_http(
            client,
            admin_token,
            name="legacy profile credential",
            key="legacy-profile-credential",
        )
        provider_config_id = await _create_provider_config_http(
            client,
            admin_token,
            credential_id,
            name="legacy profile provider",
            key="legacy-profile-provider",
        )
        embedding_id, _embedding_etag = await _create_model_profile_http(
            client,
            admin_token,
            provider_config_id,
            name="public embedding profile",
            key="public-embedding-profile",
        )

        legacy_ids = (uuid4(), uuid4())
        async with migrated_database.session() as session, session.begin():
            session.add_all(
                [
                    ModelProfile(
                        id=legacy_ids[0],
                        name="legacy rerank profile",
                        capability="rerank",
                        provider_config_id=provider_config_id,
                        model_name="legacy-reranker",
                        dimension=None,
                        max_input_tokens=4096,
                        batch_size=16,
                        timeout_seconds=Decimal("30"),
                        vector_config={"legacy": True},
                        enabled=True,
                        resource_revision=1,
                    ),
                    ModelProfile(
                        id=legacy_ids[1],
                        name="legacy chat profile",
                        capability="chat",
                        provider_config_id=provider_config_id,
                        model_name="legacy-chat",
                        dimension=None,
                        max_input_tokens=8192,
                        batch_size=8,
                        timeout_seconds=Decimal("45"),
                        vector_config={"legacy": True},
                        enabled=True,
                        resource_revision=1,
                    ),
                ]
            )

        listed = await client.get(
            "/v1/admin/model-profiles",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert listed.status_code == 200
        assert [item["id"] for item in listed.json()["items"]] == [str(embedding_id)]
        assert listed.json()["next_cursor"] is None

        for legacy_id in legacy_ids:
            detail = await client.get(
                f"/v1/admin/model-profiles/{legacy_id}",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert detail.status_code == 404
            assert detail.json()["error"]["code"] == "RESOURCE_NOT_FOUND"

            for conditional_headers in (
                {},
                {"If-Match": "malformed"},
                {"If-Match": model_profile_etag(legacy_id, 2)},
                {"If-Match": "*"},
                {"If-Match": model_profile_etag(legacy_id, 1)},
            ):
                patched = await client.patch(
                    f"/v1/admin/model-profiles/{legacy_id}",
                    headers={
                        "Authorization": f"Bearer {admin_token}",
                        **conditional_headers,
                    },
                    json={"name": f"mutated {legacy_id}"},
                )
                assert patched.status_code == 404
                assert patched.json()["error"]["code"] == "RESOURCE_NOT_FOUND"

        rerank = await _model_profile(migrated_database, legacy_ids[0])
        chat = await _model_profile(migrated_database, legacy_ids[1])
        assert (rerank.name, rerank.resource_revision) == ("legacy rerank profile", 1)
        assert (chat.name, chat.resource_revision) == ("legacy chat profile", 1)
        assert (
            await _audit_count(
                migrated_database,
                legacy_ids[0],
                "model_profile.updated",
            )
            == 0
        )
        assert (
            await _audit_count(
                migrated_database,
                legacy_ids[1],
                "model_profile.updated",
            )
            == 0
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_model_profile_concurrent_post_and_same_etag_patch_are_serialized(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url, allow_private_targets=True)
    admin_token, _agent_token = await _tokens(migrated_database, settings)
    app = _app(migrated_database, settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        credential_id = await _create_credential_http(
            client,
            admin_token,
            name="concurrent profile credential",
            key="concurrent-profile-credential",
        )
        provider_config_id = await _create_provider_config_http(
            client,
            admin_token,
            credential_id,
            name="concurrent profile provider",
            key="concurrent-profile-provider",
        )
        headers = {
            "Authorization": f"Bearer {admin_token}",
            "Idempotency-Key": "concurrent-model-profile",
        }
        first, second = await asyncio.gather(
            client.post(
                "/v1/admin/model-profiles",
                headers=headers,
                json=_model_profile_payload(provider_config_id, name="concurrent profile"),
            ),
            client.post(
                "/v1/admin/model-profiles",
                headers=headers,
                json=_model_profile_payload(provider_config_id, name="concurrent profile"),
            ),
        )
        assert sorted((first.status_code, second.status_code)) == [200, 201]
        assert first.json() == second.json()
        profile_id = UUID(first.json()["id"])
        initial_etag = first.headers["etag"]

        patch_one, patch_two = await asyncio.gather(
            client.patch(
                f"/v1/admin/model-profiles/{profile_id}",
                headers={
                    "Authorization": f"Bearer {admin_token}",
                    "If-Match": initial_etag,
                },
                json={"batch_size": 32},
            ),
            client.patch(
                f"/v1/admin/model-profiles/{profile_id}",
                headers={
                    "Authorization": f"Bearer {admin_token}",
                    "If-Match": initial_etag,
                },
                json={"timeout_seconds": 45},
            ),
        )
        assert sorted((patch_one.status_code, patch_two.status_code)) == [200, 412]

    persisted = await _model_profile(migrated_database, profile_id)
    assert persisted.resource_revision == 2
    assert await _audit_count(migrated_database, profile_id, "model_profile.created") == 1
    assert await _audit_count(migrated_database, profile_id, "model_profile.updated") == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_provider_and_model_profile_cross_resource_patches_share_lock_order(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url, allow_private_targets=True)
    first_token, _first_agent = await _tokens(migrated_database, settings)
    second_token, _second_agent = await _tokens(migrated_database, settings)
    first_actor = await _admin_principal(migrated_database, settings, first_token)
    second_actor = await _admin_principal(migrated_database, settings, second_token)
    app = _app(migrated_database, settings)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        credential_id = await _create_credential_http(
            client,
            first_token,
            name="cross resource lock credential",
            key="cross-resource-lock-credential",
        )
        provider_config_id = await _create_provider_config_http(
            client,
            first_token,
            credential_id,
            name="cross resource lock provider",
            key="cross-resource-lock-provider",
        )
        profile_response = await client.post(
            "/v1/admin/model-profiles",
            headers={
                "Authorization": f"Bearer {first_token}",
                "Idempotency-Key": "cross-resource-lock-profile",
            },
            json=_model_profile_payload(
                provider_config_id,
                name="cross resource lock profile",
                enabled=False,
            ),
        )
        assert profile_response.status_code == 201
        profile_id = UUID(profile_response.json()["id"])

    provider_locked = asyncio.Event()
    model_attempting_provider = asyncio.Event()

    def policy_factory() -> ProviderEndpointPolicy:
        return ProviderEndpointPolicy(
            environment=settings.environment,
            allow_private_targets=True,
        )

    async with (
        migrated_database.session() as provider_session,
        migrated_database.session() as profile_session,
    ):
        provider_service = ProviderConfigService(
            session=provider_session,
            settings=settings,
            endpoint_policy_factory=policy_factory,
            repository_factory=_cross_resource_repository_factory(
                role="provider",
                provider_config_id=provider_config_id,
                provider_locked=provider_locked,
                model_attempting_provider=model_attempting_provider,
            ),
        )
        profile_service = ModelProfileService(
            session=profile_session,
            settings=settings,
            repository_factory=_cross_resource_repository_factory(
                role="model",
                provider_config_id=provider_config_id,
                provider_locked=provider_locked,
                model_attempting_provider=model_attempting_provider,
            ),
        )
        outcomes = await asyncio.wait_for(
            asyncio.gather(
                provider_service.update_provider_config(
                    provider_config_id,
                    ProviderConfigPatch(provider_type="openrouter"),
                    actor=first_actor,
                    request_id="req-cross-resource-provider-patch",
                    expected_etag=provider_config_etag(provider_config_id, 1),
                ),
                profile_service.update_model_profile(
                    profile_id,
                    ModelProfilePatch(enabled=True),
                    actor=second_actor,
                    request_id="req-cross-resource-profile-patch",
                    expected_etag=model_profile_etag(profile_id, 1),
                ),
                return_exceptions=True,
            ),
            timeout=10,
        )

    assert all(not isinstance(outcome, BaseException) for outcome in outcomes), outcomes
    persisted_provider = await _provider_config(migrated_database, provider_config_id)
    persisted_profile = await _model_profile(migrated_database, profile_id)
    assert persisted_provider.provider_type == "openrouter"
    assert persisted_provider.resource_revision == 2
    assert persisted_profile.provider_config_id == provider_config_id
    assert persisted_profile.enabled is True
    assert persisted_profile.resource_revision == 2
    assert (
        await _audit_count(
            migrated_database,
            provider_config_id,
            "provider_config.updated",
        )
        == 1
    )
    assert (
        await _audit_count(
            migrated_database,
            profile_id,
            "model_profile.updated",
        )
        == 1
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_immutable_model_profile_and_provider_config_when_generation_references_profile(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url, allow_private_targets=True)
    admin_token, _agent_token = await _tokens(migrated_database, settings)
    app = _app(migrated_database, settings)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        credential_id = await _create_credential_http(
            client,
            admin_token,
            name="immutable profile credential",
            key="immutable-profile-credential",
        )
        provider_config_id = await _create_provider_config_http(
            client,
            admin_token,
            credential_id,
            name="immutable profile provider",
            key="immutable-profile-provider",
        )
        profile_id, profile_etag = await _create_model_profile_http(
            client,
            admin_token,
            provider_config_id,
            name="immutable embedding profile",
            key="immutable-embedding-profile",
        )
        other_provider_config_id = await _create_provider_config_http(
            client,
            admin_token,
            credential_id,
            name="other immutable profile provider",
            key="other-immutable-profile-provider",
        )
        disabled_provider_config_id = await _create_provider_config_http(
            client,
            admin_token,
            credential_id,
            name="disabled immutable profile provider",
            key="disabled-immutable-profile-provider",
            enabled=False,
        )

        knowledge_base_id = uuid4()
        generation_id = uuid4()
        async with migrated_database.session() as session, session.begin():
            session.add(
                KnowledgeBase(
                    id=knowledge_base_id,
                    name="Immutable profile KB",
                    status="active",
                    metadata_={},
                    filter_schema={"fields": []},
                    resource_revision=1,
                    mutation_revision=0,
                    filter_schema_revision=0,
                )
            )
            await session.flush()
            session.add(
                KnowledgeBaseIndexGeneration(
                    id=generation_id,
                    knowledge_base_id=knowledge_base_id,
                    embedding_profile_id=profile_id,
                    sparse_profile_id=None,
                    index_profile_hash="a" * 64,
                    qdrant_collection_name=f"immutable_{generation_id.hex}",
                    status="building",
                    rebuild_snapshot_at=datetime.now(UTC),
                    caught_up_revision=0,
                )
            )

        semantic_profile_patches = (
            {"provider_config_id": str(other_provider_config_id)},
            {"provider_config_id": str(uuid4())},
            {"provider_config_id": str(disabled_provider_config_id)},
            {"model_name": "text-embedding-3-large"},
            {"dimension": 3072},
            {"max_input_tokens": 8192},
        )
        for profile_patch in semantic_profile_patches:
            immutable = await client.patch(
                f"/v1/admin/model-profiles/{profile_id}",
                headers={
                    "Authorization": f"Bearer {admin_token}",
                    "If-Match": profile_etag,
                },
                json=profile_patch,
            )
            assert immutable.status_code == 409
            assert immutable.json()["error"]["code"] == "IMMUTABLE_INDEX_CONFIGURATION"

        no_op = await client.patch(
            f"/v1/admin/model-profiles/{profile_id}",
            headers={
                "Authorization": f"Bearer {admin_token}",
                "If-Match": profile_etag,
            },
            json={"model_name": "text-embedding-3-small", "vector_config": {}},
        )
        assert no_op.status_code == 200
        assert no_op.json()["resource_revision"] == 2
        profile_etag = no_op.headers["etag"]

        operational = await client.patch(
            f"/v1/admin/model-profiles/{profile_id}",
            headers={
                "Authorization": f"Bearer {admin_token}",
                "If-Match": profile_etag,
            },
            json={"name": "operational immutable profile", "batch_size": 32, "enabled": False},
        )
        assert operational.status_code == 200

        provider_detail = await client.get(
            f"/v1/admin/provider-configs/{provider_config_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        provider_etag = provider_detail.headers["etag"]
        semantic_provider_patches = (
            {"provider_type": "openrouter"},
            {"base_url": "https://localhost:9443/v2"},
            {"default_headers": {"X-Title": "different"}},
            {"provider_type": "openrouter", "routing_options": {"allow_fallbacks": False}},
        )
        for provider_patch in semantic_provider_patches:
            immutable = await client.patch(
                f"/v1/admin/provider-configs/{provider_config_id}",
                headers={
                    "Authorization": f"Bearer {admin_token}",
                    "If-Match": provider_etag,
                },
                json=provider_patch,
            )
            assert immutable.status_code == 409
            assert immutable.json()["error"]["code"] == "IMMUTABLE_INDEX_CONFIGURATION"

        replacement_credential_id = await _create_credential_http(
            client,
            admin_token,
            name="immutable replacement credential",
            key="immutable-replacement-credential",
        )
        provider_operational = await client.patch(
            f"/v1/admin/provider-configs/{provider_config_id}",
            headers={
                "Authorization": f"Bearer {admin_token}",
                "If-Match": provider_etag,
            },
            json={
                "name": "operational immutable provider",
                "credential_id": str(replacement_credential_id),
                "timeout_seconds": 45,
                "max_concurrency": 16,
                "requests_per_minute": 1200,
                "enabled": False,
            },
        )
        assert provider_operational.status_code == 200
        assert provider_operational.json()["credential_id"] == str(replacement_credential_id)

    persisted_profile = await _model_profile(migrated_database, profile_id)
    persisted_config = await _provider_config(migrated_database, provider_config_id)
    assert persisted_profile.resource_revision == 3
    assert persisted_profile.model_name == "text-embedding-3-small"
    assert persisted_config.resource_revision == 2
    assert persisted_config.base_url == CONFIG_CANONICAL_LOCAL_URL


@pytest.mark.integration
@pytest.mark.asyncio
async def test_immutable_provider_patch_rechecks_concurrent_generation_insertion_after_dns(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url, allow_private_targets=True)
    admin_token, _agent_token = await _tokens(migrated_database, settings)
    actor = await _admin_principal(migrated_database, settings, admin_token)
    app = _app(migrated_database, settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        credential_id = await _create_credential_http(
            client,
            admin_token,
            name="generation race credential",
            key="generation-race-credential",
        )
        provider_config_id = await _create_provider_config_http(
            client,
            admin_token,
            credential_id,
            name="generation race provider",
            key="generation-race-provider",
        )
        profile_id, _profile_etag = await _create_model_profile_http(
            client,
            admin_token,
            provider_config_id,
            name="generation race profile",
            key="generation-race-profile",
        )

    knowledge_base_id = uuid4()
    async with migrated_database.session() as session, session.begin():
        session.add(
            KnowledgeBase(
                id=knowledge_base_id,
                name="Generation race KB",
                status="active",
                metadata_={},
                filter_schema={"fields": []},
                resource_revision=1,
                mutation_revision=0,
                filter_schema_revision=0,
            )
        )

    resolver = _BlockingResolver()
    async with migrated_database.session() as patch_session:
        service = ProviderConfigService(
            session=patch_session,
            settings=settings,
            endpoint_policy_factory=_blocking_policy_factory(settings, resolver),
        )
        patch_task = asyncio.create_task(
            service.update_provider_config(
                provider_config_id,
                ProviderConfigPatch(base_url="https://localhost:9443/v2"),
                actor=actor,
                request_id="req-generation-race-provider-patch",
                expected_etag=provider_config_etag(provider_config_id, 1),
            )
        )
        await _wait_for_resolver(resolver)
        generation_id = uuid4()
        async with migrated_database.session() as insert_session, insert_session.begin():
            insert_session.add(
                KnowledgeBaseIndexGeneration(
                    id=generation_id,
                    knowledge_base_id=knowledge_base_id,
                    embedding_profile_id=profile_id,
                    sparse_profile_id=None,
                    index_profile_hash="d" * 64,
                    qdrant_collection_name=f"generation_race_{generation_id.hex}",
                    status="building",
                    rebuild_snapshot_at=datetime.now(UTC),
                    caught_up_revision=0,
                )
            )
        resolver.release.set()
        with pytest.raises(BusinessError) as immutable:
            await patch_task

    assert (immutable.value.status_code, immutable.value.code) == (
        409,
        "IMMUTABLE_INDEX_CONFIGURATION",
    )
    persisted = await _provider_config(migrated_database, provider_config_id)
    assert persisted.base_url == CONFIG_CANONICAL_LOCAL_URL
    assert persisted.resource_revision == 1
    assert await _audit_count(migrated_database, provider_config_id, "provider_config.updated") == 0
