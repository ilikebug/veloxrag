from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID, uuid4, uuid5

import httpx
import pytest
from fastapi import FastAPI
from pydantic import SecretStr
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException
from sqlalchemy import null, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.api.errors import BusinessError
from rag_service.auth.policies import AdminPrincipal, Capability
from rag_service.auth.schemas import AdminApiKeyCreate, AgentApiKeyCreate
from rag_service.auth.services import ApiKeyService
from rag_service.config import Settings, get_settings
from rag_service.db.models.knowledge_bases import (
    IndexGenerationCleanupClaim,
    IndexGenerationCreationRequest,
    KnowledgeBase,
    KnowledgeBaseIndexGeneration,
    KnowledgeBaseMutation,
)
from rag_service.db.models.providers import ModelProfile, ProviderConfig, ProviderCredential
from rag_service.db.session import Database
from rag_service.indexing.generation_repositories import (
    CollectionCleanupClaim,
    SessionProviderCredentialReader,
    SqlAlchemyGenerationRepository,
)
from rag_service.indexing.generation_routes import (
    get_generation_clock,
    get_generation_hooks,
    get_generation_qdrant,
)
from rag_service.indexing.generation_services import (
    GenerationSagaHooks,
    GenerationService,
    payload_indexes_for_filter_snapshot,
)
from rag_service.indexing.identities import collection_name
from rag_service.indexing.qdrant import (
    AsyncQdrantCollectionClient,
    CollectionSpec,
    FakeQdrantClient,
    ManagedCollectionPage,
    QdrantClient,
    QdrantConfigurationError,
    QdrantTransientError,
)
from rag_service.main import create_app
from rag_service.metadata.purge import KnowledgeBasePurge
from rag_service.providers.credentials import ProviderCredentialKeyring
from rag_service.providers.embeddings import (
    EmbeddingConfigSnapshot,
    EmbeddingGateway,
    EmbeddingGatewayError,
    EmbeddingOperationalConfig,
    EmbeddingResult,
)
from rag_service.providers.gateway_provider import get_embedding_gateway
from rag_service.providers.transport import ProviderHttpResponse
from rag_service.readiness import ReadinessProvider

ADMIN_HMAC_SECRET = "generation-admin-hmac-secret-32-bytes"
AGENT_HMAC_SECRET = "generation-agent-hmac-secret-32-bytes"
KEY_VERSION = "2026-07"
KEY = b"g" * 32
SECRET = "generation-provider-secret-sentinel"
NOW = datetime(2026, 7, 28, 8, 0, tzinfo=UTC)
PROBE_TEXT = "x"


def _settings(database_url: str) -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        database_url=SecretStr(database_url),
        admin_key_hmac_secret=SecretStr(ADMIN_HMAC_SECRET),
        agent_key_hmac_secret=SecretStr(AGENT_HMAC_SECRET),
        provider_credential_keyring=SecretStr(
            json.dumps({KEY_VERSION: base64.b64encode(KEY).decode("ascii")})
        ),
        provider_credential_active_key_version=KEY_VERSION,
        default_page_size=20,
        max_page_size=100,
    )


class _FakeEmbeddingGateway:
    def __init__(self) -> None:
        self.failure: EmbeddingGatewayError | None = None
        self.before_embed: Callable[[], Awaitable[None]] | None = None
        self.calls: list[
            tuple[EmbeddingConfigSnapshot, EmbeddingOperationalConfig, tuple[str, ...]]
        ] = []

    async def embed(
        self,
        *,
        snapshot: EmbeddingConfigSnapshot,
        operational: EmbeddingOperationalConfig,
        inputs: Sequence[str],
    ) -> EmbeddingResult:
        copied = tuple(inputs)
        self.calls.append((snapshot, operational, copied))
        if self.before_embed is not None:
            await self.before_embed()
        if not operational.provider_enabled:
            raise EmbeddingGatewayError(
                "PROVIDER_DISABLED",
                "Provider is disabled",
                retryable=False,
            )
        if not operational.profile_enabled:
            raise EmbeddingGatewayError(
                "MODEL_PROFILE_DISABLED",
                "Model profile is disabled",
                retryable=False,
            )
        if self.failure is not None:
            raise self.failure
        return EmbeddingResult(vectors=((0.0, 0.5, 1.0),), usage={})


class _MalformedExpiredClaimRepository(SqlAlchemyGenerationRepository):
    def __init__(
        self,
        session: AsyncSession,
        malformed_claims: tuple[CollectionCleanupClaim, ...],
    ) -> None:
        super().__init__(session)
        self._malformed_claims = malformed_claims

    async def list_expired_collection_cleanup_claims(
        self,
        *,
        now: datetime,
        limit: int,
        after_lease_expires_at: datetime | None = None,
        after_collection_name: str | None = None,
    ) -> tuple[CollectionCleanupClaim, ...]:
        valid = await super().list_expired_collection_cleanup_claims(
            now=now,
            limit=limit,
            after_lease_expires_at=after_lease_expires_at,
            after_collection_name=after_collection_name,
        )
        return (*self._malformed_claims, *valid)


class _InjectedCrash(RuntimeError):
    pass


def _hooks(
    checkpoint: str | None = None,
    action: Callable[[], Awaitable[None]] | None = None,
) -> tuple[GenerationSagaHooks, Callable[[], None]]:
    enabled = True

    async def reached(name: str) -> None:
        nonlocal enabled
        if enabled and name == checkpoint:
            enabled = False
            if action is not None:
                await action()
            else:
                raise _InjectedCrash(name)

    def disable() -> None:
        nonlocal enabled
        enabled = False

    return GenerationSagaHooks(reached), disable


def _app(
    database: Database,
    settings: Settings,
    qdrant: QdrantClient,
    gateway: _FakeEmbeddingGateway,
    hooks: GenerationSagaHooks | None = None,
) -> FastAPI:
    app = create_app()
    app.state.database = database
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_generation_qdrant] = lambda: qdrant
    app.dependency_overrides[get_embedding_gateway] = lambda: gateway
    app.dependency_overrides[get_generation_clock] = lambda: lambda: NOW
    app.dependency_overrides[get_generation_hooks] = lambda: (
        GenerationSagaHooks() if hooks is None else hooks
    )
    return app


async def _create_admin(database: Database, settings: Settings) -> tuple[AdminPrincipal, str]:
    async with database.sessions() as session:
        issued = await ApiKeyService(
            session=session,
            authentication_sessions=database.sessions,
            settings=settings,
        ).create_admin_key(
            AdminApiKeyCreate(name=f"generation-admin-{uuid4().hex}"),
            request_id=f"req-generation-admin-{uuid4().hex}",
        )
    return (
        AdminPrincipal(key_id=issued.api_key.id, public_id=issued.api_key.public_id),
        issued.token.get_secret_value(),
    )


async def _create_agent(
    database: Database,
    settings: Settings,
    actor: AdminPrincipal,
) -> str:
    async with database.sessions() as session:
        issued = await ApiKeyService(
            session=session,
            authentication_sessions=database.sessions,
            settings=settings,
        ).create_agent_key(
            AgentApiKeyCreate(
                name=f"generation-agent-{uuid4().hex}",
                capabilities=frozenset({Capability.RETRIEVE}),
                requests_per_minute=60,
                max_concurrency=4,
            ),
            actor=actor,
            request_id=f"req-generation-agent-{uuid4().hex}",
        )
    return issued.token.get_secret_value()


async def _seed_configuration(
    database: Database,
    *,
    filter_schema: dict[str, object] | None = None,
    provider_enabled: bool = True,
    profile_enabled: bool = True,
    capability: str = "embedding",
    include_credential: bool = True,
    max_input_tokens: int = 8192,
) -> tuple[UUID, UUID, UUID, UUID]:
    knowledge_base_id = uuid4()
    credential_id = uuid4()
    provider_config_id = uuid4()
    profile_id = uuid4()
    keyring = ProviderCredentialKeyring(keys={KEY_VERSION: KEY}, active_key_version=KEY_VERSION)
    encrypted = keyring.encrypt(credential_id, SECRET.encode("utf-8"))
    schema = {"fields": []} if filter_schema is None else filter_schema
    async with database.sessions() as session, session.begin():
        session.add(
            KnowledgeBase(
                id=knowledge_base_id,
                name=f"Generation KB {knowledge_base_id.hex[:8]}",
                status="active",
                filter_schema=schema,
                filter_schema_revision=3,
                mutation_revision=7,
                resource_revision=1,
            )
        )
        if include_credential:
            session.add(
                ProviderCredential(
                    id=credential_id,
                    name=f"Generation credential {credential_id.hex[:8]}",
                    ciphertext=encrypted.ciphertext,
                    nonce=encrypted.nonce,
                    algorithm=encrypted.algorithm,
                    key_version=encrypted.key_version,
                    resource_revision=1,
                )
            )
            await session.flush()
        session.add(
            ProviderConfig(
                id=provider_config_id,
                name=f"Generation provider {provider_config_id.hex[:8]}",
                provider_type="openai_compatible",
                base_url="https://provider.example/v1",
                credential_id=credential_id if include_credential else None,
                secret_ref=None if include_credential else "env:legacy-generation-secret",
                default_headers={},
                routing_options={},
                timeout_seconds=Decimal("10.000"),
                max_concurrency=2,
                requests_per_minute=60,
                enabled=provider_enabled,
                resource_revision=1,
                endpoint_policy_version=("provider-endpoint-v1" if include_credential else None),
                endpoint_validated_at=NOW if include_credential else None,
            )
        )
        await session.flush()
        session.add(
            ModelProfile(
                id=profile_id,
                name=f"Generation profile {profile_id.hex[:8]}",
                capability=capability,
                provider_config_id=provider_config_id,
                model_name="text-embedding-test",
                dimension=3 if capability == "embedding" else None,
                max_input_tokens=max_input_tokens,
                batch_size=8,
                timeout_seconds=Decimal("8.000"),
                vector_config={},
                enabled=profile_enabled,
                resource_revision=1,
            )
        )
    return knowledge_base_id, profile_id, provider_config_id, credential_id


def _headers(token: str, request_id: str, idempotency_key: str | None = None) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Request-ID": request_id,
    }
    if idempotency_key is not None:
        headers["Idempotency-Key"] = idempotency_key
    return headers


def _assert_error(
    response: httpx.Response,
    status: int,
    code: str,
    request_id: str,
    *,
    retryable: bool = False,
) -> None:
    assert response.status_code == status
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-request-id"] == request_id
    document = response.json()
    assert set(document) == {"error"}
    assert document["error"]["code"] == code
    assert document["error"]["retryable"] is retryable
    assert document["error"]["request_id"] == request_id
    assert SECRET not in response.text


async def _generation_rows(
    database: Database,
    knowledge_base_id: UUID,
) -> list[KnowledgeBaseIndexGeneration]:
    async with database.sessions() as session:
        return list(
            (
                await session.scalars(
                    select(KnowledgeBaseIndexGeneration)
                    .where(KnowledgeBaseIndexGeneration.knowledge_base_id == knowledge_base_id)
                    .order_by(KnowledgeBaseIndexGeneration.created_at)
                )
            ).all()
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_initial_generation_api_activates_atomically_and_lists_only_safe_fields(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    admin, token = await _create_admin(migrated_database, settings)
    agent_token = await _create_agent(migrated_database, settings, admin)
    filter_schema: dict[str, object] = {
        "fields": [
            {
                "name": "department",
                "source_path": "attributes.department",
                "type": "keyword",
                "operators": ["eq", "in"],
                "field_id": "fld_ERERERERQRGBEREREREREQ",
                "payload_path": "metadata.f_11111111111141118111111111111111",
            }
        ]
    }
    knowledge_base_id, profile_id, _provider_id, credential_id = await _seed_configuration(
        migrated_database,
        filter_schema=filter_schema,
    )
    qdrant = FakeQdrantClient(clock=lambda: NOW)
    gateway = _FakeEmbeddingGateway()
    app = _app(migrated_database, settings, qdrant, gateway)
    path = f"/v1/admin/knowledge-bases/{knowledge_base_id}/index-generations"
    command = {"embedding_profile_id": str(profile_id), "distance": "manhattan"}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        unauthenticated = await client.post(path, json=command)
        assert unauthenticated.status_code == 401

        agent_request_id = "req-generation-agent-denied"
        agent_denied = await client.post(
            path,
            headers=_headers(agent_token, agent_request_id, "generation-agent-denied"),
            json=command,
        )
        _assert_error(agent_denied, 401, "INVALID_API_KEY", agent_request_id)

        missing_key_id = "req-generation-missing-key"
        missing_key = await client.post(
            path,
            headers=_headers(token, missing_key_id),
            json=command,
        )
        _assert_error(missing_key, 422, "VALIDATION_ERROR", missing_key_id)

        request_id = "req-generation-create"
        created = await client.post(
            path,
            headers=_headers(token, request_id, "generation-create-1"),
            json=command,
        )
        assert created.status_code == 201, created.text
        assert created.headers["cache-control"] == "no-store"
        assert created.headers["x-request-id"] == request_id
        document = created.json()
        assert set(document) == {
            "id",
            "knowledge_base_id",
            "embedding_profile_id",
            "status",
            "distance",
            "created_at",
            "validated_at",
            "activated_at",
        }
        assert document["status"] == "active"
        assert document["distance"] == "manhattan"
        serialized = created.text.lower()
        for forbidden in (
            SECRET.lower(),
            str(credential_id),
            "credential_id",
            "embedding_config_snapshot",
            "filter_schema_snapshot",
            "default_headers",
            "routing_options",
        ):
            assert forbidden not in serialized

        listed = await client.get(path, headers=_headers(token, "req-generation-list"))
        assert listed.status_code == 200
        assert listed.headers["cache-control"] == "no-store"
        assert listed.json() == {"items": [document], "next_cursor": None}

    rows = await _generation_rows(migrated_database, knowledge_base_id)
    assert len(rows) == 1
    generation = rows[0]
    assert generation.id == UUID(document["id"])
    assert generation.status == "active"
    assert generation.embedding_config_snapshot is not None
    assert generation.embedding_config_snapshot["credential_id"] == str(credential_id)
    assert generation.filter_schema_snapshot == filter_schema
    assert generation.applied_filter_schema_revision == 3
    assert generation.caught_up_revision == 8
    assert generation.validated_revision == 8
    assert generation.expected_point_count == generation.actual_point_count == 0
    assert generation.validation_manifest_hash is not None
    assert generation.validated_at == generation.activated_at == NOW

    async with migrated_database.sessions() as session:
        kb = await session.get(KnowledgeBase, knowledge_base_id)
        assert kb is not None
        assert kb.mutation_revision == 8
        assert kb.active_index_generation_id == generation.id
        assert kb.pending_index_generation_id is None
        mutation = (
            await session.scalars(
                select(KnowledgeBaseMutation).where(
                    KnowledgeBaseMutation.knowledge_base_id == knowledge_base_id
                )
            )
        ).one()
        assert (
            mutation.revision,
            mutation.mutation_type,
            mutation.target_type,
            mutation.target_id,
        ) == (8, "index_config_changed", "index_generation", generation.id)

    assert len(gateway.calls) == 1
    snapshot, operational, inputs = gateway.calls[0]
    assert inputs == (PROBE_TEXT,)
    assert snapshot.dimension == 3
    assert snapshot.distance == "manhattan"
    assert operational.provider_config_id == _provider_id
    qdrant_spec = await qdrant.describe_collection(generation.qdrant_collection_name)
    assert qdrant_spec == CollectionSpec(
        name=generation.qdrant_collection_name,
        dimension=3,
        distance="manhattan",
        payload_indexes=qdrant_spec.payload_indexes,
    )
    assert any(
        index.path.endswith("11111111111141118111111111111111")
        for index in qdrant_spec.payload_indexes
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generation_list_does_not_construct_external_clients(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    settings = _settings(postgres_urls[0])
    _admin, token = await _create_admin(migrated_database, settings)
    knowledge_base_id, _profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )
    app = _app(
        migrated_database,
        settings,
        FakeQdrantClient(clock=lambda: NOW),
        _FakeEmbeddingGateway(),
    )

    async def unexpected_external_dependency() -> object:
        raise AssertionError("generation list must not construct external clients")

    app.dependency_overrides[get_generation_qdrant] = unexpected_external_dependency
    app.dependency_overrides[get_embedding_gateway] = unexpected_external_dependency

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get(
            f"/v1/admin/knowledge-bases/{knowledge_base_id}/index-generations",
            headers=_headers(token, "req-generation-list-lazy"),
        )

    assert response.status_code == 200
    assert response.json() == {"items": [], "next_cursor": None}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generation_list_is_bounded_cursor_paginated_and_serializes_retiring(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    settings = _settings(postgres_urls[0])
    _admin, token = await _create_admin(migrated_database, settings)
    knowledge_base_id, profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )
    first_id = UUID("10000000-0000-4000-8000-000000000001")
    second_id = UUID("10000000-0000-4000-8000-000000000002")
    async with migrated_database.sessions() as session, session.begin():
        session.add_all(
            [
                KnowledgeBaseIndexGeneration(
                    id=first_id,
                    knowledge_base_id=knowledge_base_id,
                    embedding_profile_id=profile_id,
                    index_profile_hash="1" * 64,
                    qdrant_collection_name=collection_name(knowledge_base_id, first_id),
                    status="retiring",
                    rebuild_snapshot_at=NOW,
                    caught_up_revision=0,
                    created_at=NOW,
                    distance="cosine",
                ),
                KnowledgeBaseIndexGeneration(
                    id=second_id,
                    knowledge_base_id=knowledge_base_id,
                    embedding_profile_id=profile_id,
                    index_profile_hash="2" * 64,
                    qdrant_collection_name=collection_name(knowledge_base_id, second_id),
                    status="failed",
                    rebuild_snapshot_at=NOW,
                    caught_up_revision=0,
                    created_at=NOW + timedelta(seconds=1),
                    distance="cosine",
                    safe_error_code="TEST_FAILURE",
                    safe_error_message="Safe test failure",
                ),
            ]
        )

    app = _app(
        migrated_database,
        settings,
        FakeQdrantClient(clock=lambda: NOW),
        _FakeEmbeddingGateway(),
    )
    path = f"/v1/admin/knowledge-bases/{knowledge_base_id}/index-generations"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        first = await client.get(
            path,
            params={"limit": 1},
            headers=_headers(token, "req-generation-page-first"),
        )
        assert first.status_code == 200, first.text
        first_page = first.json()
        assert [item["id"] for item in first_page["items"]] == [str(first_id)]
        assert first_page["items"][0]["status"] == "retiring"
        assert isinstance(first_page["next_cursor"], str)

        second = await client.get(
            path,
            params={"limit": 1, "cursor": first_page["next_cursor"]},
            headers=_headers(token, "req-generation-page-second"),
        )
        assert second.status_code == 200, second.text
        assert second.json()["items"][0]["id"] == str(second_id)
        assert second.json()["next_cursor"] is None

        invalid = await client.get(
            path,
            params={"limit": settings.max_page_size + 1},
            headers=_headers(token, "req-generation-page-invalid"),
        )
        _assert_error(
            invalid,
            422,
            "VALIDATION_ERROR",
            "req-generation-page-invalid",
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generation_idempotency_replays_and_rejects_payload_or_new_key_conflicts(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    settings = _settings(postgres_urls[0])
    _admin, token = await _create_admin(migrated_database, settings)
    knowledge_base_id, profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )
    qdrant = FakeQdrantClient(clock=lambda: NOW)
    gateway = _FakeEmbeddingGateway()
    app = _app(migrated_database, settings, qdrant, gateway)
    path = f"/v1/admin/knowledge-bases/{knowledge_base_id}/index-generations"
    command = {"embedding_profile_id": str(profile_id), "distance": "cosine"}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        first = await client.post(
            path,
            headers=_headers(token, "req-generation-idem-first", "generation-idem-key"),
            json=command,
        )
        replay = await client.post(
            path,
            headers=_headers(token, "req-generation-idem-replay", "generation-idem-key"),
            json=command,
        )
        assert first.status_code == replay.status_code == 201
        assert first.json() == replay.json()
        assert qdrant.create_calls == 1
        assert len(gateway.calls) == 1

        reused_id = "req-generation-idem-reused"
        reused = await client.post(
            path,
            headers=_headers(token, reused_id, "generation-idem-key"),
            json={**command, "distance": "dot"},
        )
        _assert_error(reused, 409, "IDEMPOTENCY_KEY_REUSED", reused_id)

        configured_id = "req-generation-already-configured"
        configured = await client.post(
            path,
            headers=_headers(token, configured_id, "generation-other-key"),
            json=command,
        )
        _assert_error(
            configured,
            409,
            "INDEX_GENERATION_ALREADY_CONFIGURED",
            configured_id,
        )

    assert len(await _generation_rows(migrated_database, knowledge_base_id)) == 1


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.acceptance
async def test_transient_provider_failure_preserves_building_and_same_key_resumes(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    settings = _settings(postgres_urls[0])
    _admin, token = await _create_admin(migrated_database, settings)
    knowledge_base_id, profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )
    qdrant = FakeQdrantClient(clock=lambda: NOW)
    gateway = _FakeEmbeddingGateway()
    gateway.failure = EmbeddingGatewayError(
        "PROVIDER_UNAVAILABLE",
        "Provider unavailable",
        retryable=True,
    )
    app = _app(migrated_database, settings, qdrant, gateway)
    path = f"/v1/admin/knowledge-bases/{knowledge_base_id}/index-generations"
    headers = _headers(token, "req-generation-transient", "generation-transient-key")
    command = {"embedding_profile_id": str(profile_id), "distance": "cosine"}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        unavailable = await client.post(path, headers=headers, json=command)
        _assert_error(
            unavailable,
            503,
            "PROVIDER_UNAVAILABLE",
            "req-generation-transient",
            retryable=True,
        )
        first_rows = await _generation_rows(migrated_database, knowledge_base_id)
        assert len(first_rows) == 1
        assert first_rows[0].status == "building"
        first_generation_id = first_rows[0].id
        first_collection = first_rows[0].qdrant_collection_name

        gateway.failure = None
        resumed = await client.post(
            path,
            headers=_headers(token, "req-generation-resumed", "generation-transient-key"),
            json=command,
        )
        assert resumed.status_code == 201
        assert UUID(resumed.json()["id"]) == first_generation_id
        assert first_collection in qdrant.collection_names

    rows = await _generation_rows(migrated_database, knowledge_base_id)
    assert len(rows) == 1 and rows[0].status == "active"
    assert qdrant.create_calls == 1


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("transport_error", "secret"),
    [
        (httpx.RemoteProtocolError, "remote-protocol-secret"),
        (httpx.DecodingError, "decoding-secret"),
    ],
)
async def test_wrapped_transient_qdrant_failure_returns_503_preserves_building_and_resumes(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
    transport_error: type[httpx.TransportError],
    secret: str,
) -> None:
    settings = _settings(postgres_urls[0])
    _admin, token = await _create_admin(migrated_database, settings)
    knowledge_base_id, profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )

    class _RawQdrant:
        def __init__(self) -> None:
            self.unavailable = True
            self.collections: dict[str, tuple[models.VectorParams, dict[str, str]]] = {}
            self.payload_schemas: dict[str, dict[str, object]] = {}

        async def collection_exists(self, collection: str) -> bool:
            if self.unavailable:
                raise ResponseHandlingException(
                    transport_error(
                        secret,
                        request=httpx.Request("GET", "http://qdrant.invalid"),
                    )
                )
            return collection in self.collections

        async def create_collection(
            self,
            *,
            collection_name: str,
            vectors_config: models.VectorParams,
            sparse_vectors_config: dict[str, models.SparseVectorParams],
            on_disk_payload: bool,
            hnsw_config: models.HnswConfigDiff,
            quantization_config: object | None,
            metadata: dict[str, str],
        ) -> None:
            assert sparse_vectors_config == {}
            assert on_disk_payload is True
            assert hnsw_config == models.HnswConfigDiff(
                m=16,
                ef_construct=100,
                full_scan_threshold=10_000,
                max_indexing_threads=0,
                on_disk=False,
                payload_m=16,
                inline_storage=False,
            )
            assert quantization_config is None
            self.collections[collection_name] = (vectors_config, metadata)
            self.payload_schemas[collection_name] = {}

        async def get_collection(self, collection: str) -> object:
            vectors, metadata = self.collections[collection]
            return SimpleNamespace(
                config=SimpleNamespace(
                    params=SimpleNamespace(
                        vectors=vectors,
                        sparse_vectors={},
                        on_disk_payload=True,
                    ),
                    hnsw_config=models.HnswConfig(
                        m=16,
                        ef_construct=100,
                        full_scan_threshold=10_000,
                        max_indexing_threads=0,
                        on_disk=False,
                        payload_m=16,
                        inline_storage=False,
                    ),
                    quantization_config=None,
                    metadata=metadata,
                ),
                payload_schema=self.payload_schemas[collection],
            )

        async def create_payload_index(
            self,
            *,
            collection_name: str,
            field_name: str,
            field_schema: models.PayloadFieldSchema,
            wait: bool,
        ) -> None:
            assert wait is True
            self.payload_schemas[collection_name][field_name] = SimpleNamespace(
                data_type=models.PayloadSchemaType(cast(Any, field_schema).type.value),
                params=field_schema,
            )

        async def count(self, **_kwargs: object) -> object:
            return SimpleNamespace(count=0)

        async def close(self) -> None:
            return None

    raw_qdrant = _RawQdrant()
    qdrant = AsyncQdrantCollectionClient(cast(AsyncQdrantClient, raw_qdrant))
    gateway = _FakeEmbeddingGateway()
    app = _app(migrated_database, settings, qdrant, gateway)
    path = f"/v1/admin/knowledge-bases/{knowledge_base_id}/index-generations"
    command = {"embedding_profile_id": str(profile_id), "distance": "cosine"}
    key = "generation-transient-qdrant"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        unavailable = await client.post(
            path,
            headers=_headers(token, "req-transient-qdrant", key),
            json=command,
        )
        _assert_error(
            unavailable,
            503,
            "QDRANT_UNAVAILABLE",
            "req-transient-qdrant",
            retryable=True,
        )
        assert secret not in unavailable.text
        rows = await _generation_rows(migrated_database, knowledge_base_id)
        assert len(rows) == 1 and rows[0].status == "building"
        assert gateway.calls == []
        async with migrated_database.sessions() as session:
            knowledge_base = await session.get(KnowledgeBase, knowledge_base_id)
            assert knowledge_base is not None
            assert knowledge_base.pending_index_generation_id == rows[0].id
            assert knowledge_base.active_index_generation_id is None
        raw_qdrant.unavailable = False
        resumed = await client.post(
            path,
            headers=_headers(token, "req-transient-qdrant-resume", key),
            json=command,
        )

    assert resumed.status_code == 201
    rows = await _generation_rows(migrated_database, knowledge_base_id)
    assert len(rows) == 1 and rows[0].status == "active"
    async with migrated_database.sessions() as session:
        knowledge_base = await session.get(KnowledgeBase, knowledge_base_id)
        assert knowledge_base is not None
        assert knowledge_base.pending_index_generation_id is None
        assert knowledge_base.active_index_generation_id == rows[0].id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generation_requests_share_provider_concurrency_limit_across_knowledge_bases(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    settings = _settings(postgres_urls[0])
    _admin, token = await _create_admin(migrated_database, settings)
    first_kb_id, profile_id, provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )
    second_kb_id = uuid4()
    async with migrated_database.sessions() as session, session.begin():
        session.add(
            KnowledgeBase(
                id=second_kb_id,
                name=f"Generation KB {second_kb_id.hex[:8]}",
                status="active",
                filter_schema={"fields": []},
                filter_schema_revision=3,
                mutation_revision=7,
                resource_revision=1,
            )
        )
        provider = await session.get(ProviderConfig, provider_id)
        assert provider is not None
        provider.max_concurrency = 1

    class _BlockingTransport:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.release = asyncio.Event()
            self.calls = 0
            self.active = 0
            self.peak_active = 0
            self.close_calls = 0

        async def post_json(self, **_kwargs: object) -> ProviderHttpResponse:
            self.calls += 1
            self.active += 1
            self.peak_active = max(self.peak_active, self.active)
            self.started.set()
            try:
                await self.release.wait()
                return ProviderHttpResponse(
                    status_code=200,
                    headers={},
                    body=b'{"data":[{"index":0,"embedding":[0.0,0.5,1.0]}]}',
                )
            finally:
                self.active -= 1

        async def aclose(self) -> None:
            self.close_calls += 1

    provider_transport = _BlockingTransport()
    gateway = EmbeddingGateway(
        keyring=ProviderCredentialKeyring(
            keys={KEY_VERSION: KEY},
            active_key_version=KEY_VERSION,
        ),
        credential_reader=SessionProviderCredentialReader(migrated_database.sessions),
        transport=provider_transport,
    )
    qdrant = FakeQdrantClient(clock=lambda: NOW)
    app = create_app(
        settings=settings,
        database=migrated_database,
        readiness_provider=cast(ReadinessProvider, SimpleNamespace()),
        generation_embedding_gateway=gateway,
    )
    app.dependency_overrides[get_generation_qdrant] = lambda: qdrant
    app.dependency_overrides[get_generation_clock] = lambda: lambda: NOW
    command = {"embedding_profile_id": str(profile_id), "distance": "cosine"}

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client,
    ):
        first = asyncio.create_task(
            client.post(
                f"/v1/admin/knowledge-bases/{first_kb_id}/index-generations",
                headers=_headers(token, "req-shared-gateway-first", "shared-gateway-first"),
                json=command,
            )
        )
        await provider_transport.started.wait()
        second = asyncio.create_task(
            client.post(
                f"/v1/admin/knowledge-bases/{second_kb_id}/index-generations",
                headers=_headers(token, "req-shared-gateway-second", "shared-gateway-second"),
                json=command,
            )
        )
        await asyncio.sleep(0.1)
        assert provider_transport.calls == 1
        assert provider_transport.peak_active == 1
        provider_transport.release.set()
        first_response, second_response = await asyncio.gather(first, second)

    assert first_response.status_code == second_response.status_code == 201
    assert provider_transport.calls == 2
    assert provider_transport.peak_active == 1
    assert provider_transport.close_calls == 0
    await gateway.aclose()
    assert provider_transport.close_calls == 1


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.acceptance
async def test_permanent_provider_failure_is_persisted_and_a_new_key_can_retry_after_fix(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    settings = _settings(postgres_urls[0])
    _admin, token = await _create_admin(migrated_database, settings)
    knowledge_base_id, profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )
    qdrant = FakeQdrantClient(clock=lambda: NOW)
    gateway = _FakeEmbeddingGateway()
    gateway.failure = EmbeddingGatewayError(
        "PROVIDER_MODEL_NOT_FOUND",
        "Provider model not found",
        retryable=False,
    )
    app = _app(migrated_database, settings, qdrant, gateway)
    path = f"/v1/admin/knowledge-bases/{knowledge_base_id}/index-generations"
    command = {"embedding_profile_id": str(profile_id), "distance": "cosine"}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        failed_id = "req-generation-permanent"
        failed = await client.post(
            path,
            headers=_headers(token, failed_id, "generation-permanent-key"),
            json=command,
        )
        _assert_error(failed, 422, "PROVIDER_MODEL_NOT_FOUND", failed_id)
        replay = await client.post(
            path,
            headers=_headers(token, "req-generation-permanent-replay", "generation-permanent-key"),
            json=command,
        )
        _assert_error(
            replay,
            422,
            "PROVIDER_MODEL_NOT_FOUND",
            "req-generation-permanent-replay",
        )
        assert len(gateway.calls) == 1

        rows = await _generation_rows(migrated_database, knowledge_base_id)
        assert len(rows) == 1
        failed_generation = rows[0]
        owner = uuid4()
        async with migrated_database.sessions() as session, session.begin():
            repository = SqlAlchemyGenerationRepository(session)
            await repository.get_knowledge_base_for_update(knowledge_base_id)
            await repository.acquire_collection_fence(failed_generation.qdrant_collection_name)
            claim = await repository.claim_collection_cleanup(
                collection_name=failed_generation.qdrant_collection_name,
                knowledge_base_id=knowledge_base_id,
                generation_id=failed_generation.id,
                lease_owner=owner,
                now=NOW,
                lease_duration=timedelta(minutes=1),
            )
            assert claim is not None
        cleanup_in_progress = await client.post(
            path,
            headers=_headers(
                token,
                "req-generation-permanent-cleanup",
                "generation-permanent-key",
            ),
            json=command,
        )
        _assert_error(
            cleanup_in_progress,
            503,
            "GENERATION_CLEANUP_IN_PROGRESS",
            "req-generation-permanent-cleanup",
            retryable=True,
        )

        await qdrant.delete_collection(failed_generation.qdrant_collection_name)
        async with migrated_database.sessions() as session, session.begin():
            repository = SqlAlchemyGenerationRepository(session)
            await repository.get_knowledge_base_for_update(knowledge_base_id)
            await repository.acquire_collection_fence(failed_generation.qdrant_collection_name)
            assert await repository.complete_collection_cleanup(
                failed_generation.qdrant_collection_name,
                lease_owner=owner,
                lease_epoch=claim.lease_epoch,
                now=NOW + timedelta(seconds=1),
            )
        retired = await client.post(
            path,
            headers=_headers(
                token,
                "req-generation-permanent-retired",
                "generation-permanent-key",
            ),
            json=command,
        )
        _assert_error(
            retired,
            409,
            "GENERATION_COLLECTION_RETIRED",
            "req-generation-permanent-retired",
        )

        gateway.failure = None
        recovered = await client.post(
            path,
            headers=_headers(token, "req-generation-new-key", "generation-repaired-key"),
            json=command,
        )
        assert recovered.status_code == 201

    rows = await _generation_rows(migrated_database, knowledge_base_id)
    assert [row.status for row in rows] == ["failed", "active"]
    assert rows[0].safe_error_code == "PROVIDER_MODEL_NOT_FOUND"
    assert rows[0].safe_error_message == "Provider model not found"


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.acceptance
@pytest.mark.parametrize(
    "checkpoint",
    [
        "after_creation_request",
        "after_generation_insert",
        "after_reservation",
        "after_collection",
        "after_payload_indexes",
        "after_probe",
        "before_activation_commit",
    ],
)
async def test_generation_crash_boundaries_resume_one_generation_and_collection(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
    checkpoint: str,
) -> None:
    settings = _settings(postgres_urls[0])
    _admin, token = await _create_admin(migrated_database, settings)
    knowledge_base_id, profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )
    qdrant = FakeQdrantClient(clock=lambda: NOW)
    gateway = _FakeEmbeddingGateway()
    hooks, disable = _hooks(checkpoint)
    app = _app(migrated_database, settings, qdrant, gateway, hooks)
    path = f"/v1/admin/knowledge-bases/{knowledge_base_id}/index-generations"
    command = {"embedding_profile_id": str(profile_id), "distance": "cosine"}
    key = f"generation-crash-{checkpoint}"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        crashed = await client.post(
            path,
            headers=_headers(token, f"req-crash-{checkpoint}", key),
            json=command,
        )
        _assert_error(
            crashed,
            500,
            "INTERNAL_ERROR",
            f"req-crash-{checkpoint}",
        )
        disable()
        resumed = await client.post(
            path,
            headers=_headers(token, f"req-resume-{checkpoint}", key),
            json=command,
        )
        assert resumed.status_code == 201

    rows = await _generation_rows(migrated_database, knowledge_base_id)
    assert len(rows) == 1
    assert rows[0].status == "active"
    assert tuple(qdrant.collection_names) == (rows[0].qdrant_collection_name,)
    assert qdrant.create_calls == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generation_external_io_runs_without_holding_database_row_locks(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    settings = _settings(postgres_urls[0])
    _admin, token = await _create_admin(migrated_database, settings)
    knowledge_base_id, profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )
    lock_checks: list[str] = []

    async def assert_generation_rows_are_unlocked(label: str) -> None:
        async with migrated_database.sessions() as session, session.begin():
            knowledge_base = await session.scalar(
                select(KnowledgeBase)
                .where(KnowledgeBase.id == knowledge_base_id)
                .with_for_update(nowait=True)
            )
            generations = list(
                (
                    await session.scalars(
                        select(KnowledgeBaseIndexGeneration)
                        .where(KnowledgeBaseIndexGeneration.knowledge_base_id == knowledge_base_id)
                        .with_for_update(nowait=True)
                    )
                ).all()
            )
            assert knowledge_base is not None
            assert len(generations) == 1
        lock_checks.append(label)

    class _LockCheckingQdrant(FakeQdrantClient):
        async def ensure_collection(self, spec: CollectionSpec) -> None:
            await assert_generation_rows_are_unlocked("qdrant")
            await super().ensure_collection(spec)

    qdrant = _LockCheckingQdrant(clock=lambda: NOW)
    gateway = _FakeEmbeddingGateway()
    gateway.before_embed = lambda: assert_generation_rows_are_unlocked("provider")
    app = _app(migrated_database, settings, qdrant, gateway)
    path = f"/v1/admin/knowledge-bases/{knowledge_base_id}/index-generations"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            path,
            headers=_headers(token, "req-generation-no-locks", "generation-no-locks"),
            json={"embedding_profile_id": str(profile_id), "distance": "cosine"},
        )

    assert response.status_code == 201
    assert lock_checks == ["qdrant", "provider"]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generation_qdrant_mismatch_is_permanent_and_never_deletes_collection(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    settings = _settings(postgres_urls[0])
    _admin, token = await _create_admin(migrated_database, settings)
    knowledge_base_id, profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )
    hooks, disable = _hooks("after_reservation")
    qdrant = FakeQdrantClient(clock=lambda: NOW)
    gateway = _FakeEmbeddingGateway()
    app = _app(migrated_database, settings, qdrant, gateway, hooks)
    path = f"/v1/admin/knowledge-bases/{knowledge_base_id}/index-generations"
    command = {"embedding_profile_id": str(profile_id), "distance": "cosine"}
    key = "generation-qdrant-mismatch"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        interrupted = await client.post(
            path,
            headers=_headers(token, "req-generation-mismatch-setup", key),
            json=command,
        )
        _assert_error(
            interrupted,
            500,
            "INTERNAL_ERROR",
            "req-generation-mismatch-setup",
        )
        rows = await _generation_rows(migrated_database, knowledge_base_id)
        assert len(rows) == 1
        await qdrant.seed_collection(
            CollectionSpec(
                rows[0].qdrant_collection_name,
                3,
                "cosine",
                (),
                datatype="uint8",
            ),
            created_at=NOW,
        )
        disable()
        request_id = "req-generation-mismatch"
        mismatch = await client.post(
            path,
            headers=_headers(token, request_id, key),
            json=command,
        )
        _assert_error(
            mismatch,
            409,
            "QDRANT_COLLECTION_MISMATCH",
            request_id,
        )

    rows = await _generation_rows(migrated_database, knowledge_base_id)
    assert len(rows) == 1
    assert rows[0].status == "failed"
    assert rows[0].qdrant_collection_name in qdrant.collection_names
    async with migrated_database.sessions() as session:
        knowledge_base = await session.get(KnowledgeBase, knowledge_base_id)
        assert knowledge_base is not None
        assert knowledge_base.pending_index_generation_id is None
        assert knowledge_base.active_index_generation_id is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generation_reloads_current_operational_controls_before_probe(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    settings = _settings(postgres_urls[0])
    _admin, token = await _create_admin(migrated_database, settings)
    knowledge_base_id, profile_id, provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )

    async def disable_provider() -> None:
        async with migrated_database.sessions() as session, session.begin():
            provider = await session.get(ProviderConfig, provider_id, with_for_update=True)
            profile = await session.get(ModelProfile, profile_id, with_for_update=True)
            assert provider is not None
            assert profile is not None
            provider.enabled = False
            provider.max_concurrency = 7
            provider.requests_per_minute = 240
            profile.timeout_seconds = Decimal("2.500")
            profile.batch_size = 4

    hooks, _disable = _hooks("after_payload_indexes", disable_provider)
    qdrant = FakeQdrantClient(clock=lambda: NOW)
    gateway = _FakeEmbeddingGateway()
    app = _app(migrated_database, settings, qdrant, gateway, hooks)
    path = f"/v1/admin/knowledge-bases/{knowledge_base_id}/index-generations"
    request_id = "req-generation-current-operational"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            path,
            headers=_headers(token, request_id, "generation-current-operational"),
            json={"embedding_profile_id": str(profile_id), "distance": "cosine"},
        )

    _assert_error(response, 422, "PROVIDER_DISABLED", request_id)
    assert len(gateway.calls) == 1
    operational = gateway.calls[0][1]
    assert operational.provider_enabled is False
    assert operational.max_concurrency == 7
    assert operational.requests_per_minute == 240
    assert operational.timeout_seconds == Decimal("2.500")
    assert operational.batch_size == 4
    rows = await _generation_rows(migrated_database, knowledge_base_id)
    assert len(rows) == 1
    assert rows[0].status == "failed"
    async with migrated_database.sessions() as session:
        knowledge_base = await session.get(KnowledgeBase, knowledge_base_id)
        assert knowledge_base is not None
        assert knowledge_base.pending_index_generation_id is None
        assert knowledge_base.active_index_generation_id is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generation_keeps_snapshotted_credential_id_when_provider_credential_changes(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    settings = _settings(postgres_urls[0])
    _admin, token = await _create_admin(migrated_database, settings)
    knowledge_base_id, profile_id, provider_id, credential_id = await _seed_configuration(
        migrated_database
    )
    replacement_credential_id = uuid4()

    async def replace_provider_credential() -> None:
        keyring = ProviderCredentialKeyring(
            keys={KEY_VERSION: KEY},
            active_key_version=KEY_VERSION,
        )
        encrypted = keyring.encrypt(
            replacement_credential_id,
            b"replacement-generation-secret",
        )
        async with migrated_database.sessions() as session, session.begin():
            session.add(
                ProviderCredential(
                    id=replacement_credential_id,
                    name="replacement generation credential",
                    ciphertext=encrypted.ciphertext,
                    nonce=encrypted.nonce,
                    algorithm=encrypted.algorithm,
                    key_version=encrypted.key_version,
                    resource_revision=1,
                )
            )
            await session.flush()
            provider = await session.get(ProviderConfig, provider_id, with_for_update=True)
            assert provider is not None
            provider.credential_id = replacement_credential_id

    hooks, _disable = _hooks("after_payload_indexes", replace_provider_credential)
    qdrant = FakeQdrantClient(clock=lambda: NOW)
    gateway = _FakeEmbeddingGateway()
    app = _app(migrated_database, settings, qdrant, gateway, hooks)
    path = f"/v1/admin/knowledge-bases/{knowledge_base_id}/index-generations"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            path,
            headers=_headers(
                token,
                "req-generation-credential-snapshot",
                "generation-credential-snapshot",
            ),
            json={"embedding_profile_id": str(profile_id), "distance": "cosine"},
        )

    assert response.status_code == 201
    assert len(gateway.calls) == 1
    assert gateway.calls[0][0].credential_id == credential_id
    assert gateway.calls[0][0].credential_id != replacement_credential_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generation_replay_configuration_conflict_clears_stale_pending_state(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    settings = _settings(postgres_urls[0])
    _admin, token = await _create_admin(migrated_database, settings)
    knowledge_base_id, profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )
    hooks, disable = _hooks("after_reservation")
    qdrant = FakeQdrantClient(clock=lambda: NOW)
    gateway = _FakeEmbeddingGateway()
    app = _app(migrated_database, settings, qdrant, gateway, hooks)
    path = f"/v1/admin/knowledge-bases/{knowledge_base_id}/index-generations"
    command = {"embedding_profile_id": str(profile_id), "distance": "cosine"}
    key = "generation-replay-config-conflict"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        interrupted = await client.post(
            path,
            headers=_headers(token, "req-generation-config-setup", key),
            json=command,
        )
        _assert_error(
            interrupted,
            500,
            "INTERNAL_ERROR",
            "req-generation-config-setup",
        )
        disable()
        async with migrated_database.sessions() as session, session.begin():
            profile = await session.get(ModelProfile, profile_id, with_for_update=True)
            assert profile is not None
            profile.model_name = "changed-model-name"
        request_id = "req-generation-config-conflict"
        conflict = await client.post(
            path,
            headers=_headers(token, request_id, key),
            json=command,
        )
        async with migrated_database.sessions() as session, session.begin():
            profile = await session.get(ModelProfile, profile_id, with_for_update=True)
            assert profile is not None
            profile.model_name = "text-embedding-test"
        retried = await client.post(
            path,
            headers=_headers(
                token,
                "req-generation-config-new-key",
                "generation-replay-config-conflict-new-key",
            ),
            json=command,
        )

    _assert_error(conflict, 409, "GENERATION_CONFIGURATION_CONFLICT", request_id)
    assert retried.status_code == 201
    rows = await _generation_rows(migrated_database, knowledge_base_id)
    assert len(rows) == 2
    assert {row.status for row in rows} == {"failed", "active"}
    async with migrated_database.sessions() as session:
        knowledge_base = await session.get(KnowledgeBase, knowledge_base_id)
        assert knowledge_base is not None
        assert knowledge_base.pending_index_generation_id is None
        assert knowledge_base.active_index_generation_id is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generation_rejects_compatible_preexisting_nonempty_collection(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    settings = _settings(postgres_urls[0])
    _admin, token = await _create_admin(migrated_database, settings)
    knowledge_base_id, profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )
    hooks, disable = _hooks("after_reservation")
    qdrant = FakeQdrantClient(clock=lambda: NOW)
    gateway = _FakeEmbeddingGateway()
    app = _app(migrated_database, settings, qdrant, gateway, hooks)
    path = f"/v1/admin/knowledge-bases/{knowledge_base_id}/index-generations"
    command = {"embedding_profile_id": str(profile_id), "distance": "cosine"}
    key = "generation-nonempty-collection"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        interrupted = await client.post(
            path,
            headers=_headers(token, "req-generation-nonempty-setup", key),
            json=command,
        )
        _assert_error(
            interrupted,
            500,
            "INTERNAL_ERROR",
            "req-generation-nonempty-setup",
        )
        rows = await _generation_rows(migrated_database, knowledge_base_id)
        assert len(rows) == 1
        generation = rows[0]
        assert generation.filter_schema_snapshot is not None
        await qdrant.seed_collection(
            CollectionSpec(
                generation.qdrant_collection_name,
                3,
                "cosine",
                payload_indexes_for_filter_snapshot(generation.filter_schema_snapshot),
            ),
            created_at=NOW,
            point_count=1,
        )
        disable()
        request_id = "req-generation-nonempty"
        response = await client.post(
            path,
            headers=_headers(token, request_id, key),
            json=command,
        )

    _assert_error(response, 409, "QDRANT_COLLECTION_NOT_EMPTY", request_id)
    rows = await _generation_rows(migrated_database, knowledge_base_id)
    assert len(rows) == 1
    assert rows[0].status == "failed"
    assert rows[0].actual_point_count == 1
    assert rows[0].expected_point_count == 0


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.acceptance
async def test_filter_revision_cas_conflict_never_activates_stale_snapshot(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    settings = _settings(postgres_urls[0])
    _admin, token = await _create_admin(migrated_database, settings)
    knowledge_base_id, profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )

    async def change_filter_revision() -> None:
        async with migrated_database.sessions() as session, session.begin():
            kb = await session.get(KnowledgeBase, knowledge_base_id, with_for_update=True)
            assert kb is not None
            kb.filter_schema_revision += 1

    hooks, _disable = _hooks("after_probe", change_filter_revision)
    qdrant = FakeQdrantClient(clock=lambda: NOW)
    gateway = _FakeEmbeddingGateway()
    app = _app(migrated_database, settings, qdrant, gateway, hooks)
    path = f"/v1/admin/knowledge-bases/{knowledge_base_id}/index-generations"
    request_id = "req-generation-filter-cas"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            path,
            headers=_headers(token, request_id, "generation-filter-cas-key"),
            json={"embedding_profile_id": str(profile_id), "distance": "cosine"},
        )
    _assert_error(
        response,
        409,
        "GENERATION_CONFIGURATION_CONFLICT",
        request_id,
    )
    rows = await _generation_rows(migrated_database, knowledge_base_id)
    assert len(rows) == 1 and rows[0].status == "failed"
    async with migrated_database.sessions() as session:
        kb = await session.get(KnowledgeBase, knowledge_base_id)
        assert kb is not None
        assert kb.active_index_generation_id is None
        assert kb.pending_index_generation_id is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generation_requires_active_knowledge_base_before_reservation(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    settings = _settings(postgres_urls[0])
    _admin, token = await _create_admin(migrated_database, settings)
    knowledge_base_id, profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )
    async with migrated_database.sessions() as session, session.begin():
        knowledge_base = await session.get(KnowledgeBase, knowledge_base_id, with_for_update=True)
        assert knowledge_base is not None
        knowledge_base.status = "disabled"

    qdrant = FakeQdrantClient(clock=lambda: NOW)
    gateway = _FakeEmbeddingGateway()
    app = _app(migrated_database, settings, qdrant, gateway)
    request_id = "req-generation-disabled-reservation"
    path = f"/v1/admin/knowledge-bases/{knowledge_base_id}/index-generations"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            path,
            headers=_headers(token, request_id, "generation-disabled-reservation"),
            json={"embedding_profile_id": str(profile_id), "distance": "cosine"},
        )

    _assert_error(response, 409, "GENERATION_CONFIGURATION_CONFLICT", request_id)
    assert await _generation_rows(migrated_database, knowledge_base_id) == []
    assert len(qdrant.collection_names) == 0
    assert gateway.calls == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generation_kb_status_change_before_activation_is_terminal_and_recoverable(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    settings = _settings(postgres_urls[0])
    _admin, token = await _create_admin(migrated_database, settings)
    knowledge_base_id, profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )

    async def disable_knowledge_base() -> None:
        async with migrated_database.sessions() as session, session.begin():
            knowledge_base = await session.get(
                KnowledgeBase,
                knowledge_base_id,
                with_for_update=True,
            )
            assert knowledge_base is not None
            knowledge_base.status = "deleting"

    hooks, _disable = _hooks("after_probe", disable_knowledge_base)
    qdrant = FakeQdrantClient(clock=lambda: NOW)
    gateway = _FakeEmbeddingGateway()
    app = _app(migrated_database, settings, qdrant, gateway, hooks)
    path = f"/v1/admin/knowledge-bases/{knowledge_base_id}/index-generations"
    command = {"embedding_profile_id": str(profile_id), "distance": "cosine"}
    key = "generation-kb-status-change"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        failed = await client.post(
            path,
            headers=_headers(token, "req-generation-kb-disabled", key),
            json=command,
        )
        replay = await client.post(
            path,
            headers=_headers(token, "req-generation-kb-disabled-replay", key),
            json=command,
        )
        async with migrated_database.sessions() as session, session.begin():
            knowledge_base = await session.get(
                KnowledgeBase,
                knowledge_base_id,
                with_for_update=True,
            )
            assert knowledge_base is not None
            knowledge_base.status = "active"
        recovered = await client.post(
            path,
            headers=_headers(token, "req-generation-kb-reactivated", f"{key}-new"),
            json=command,
        )

    _assert_error(
        failed,
        409,
        "GENERATION_CONFIGURATION_CONFLICT",
        "req-generation-kb-disabled",
    )
    _assert_error(
        replay,
        409,
        "GENERATION_CONFIGURATION_CONFLICT",
        "req-generation-kb-disabled-replay",
    )
    assert recovered.status_code == 201
    assert len(gateway.calls) == 2
    rows = await _generation_rows(migrated_database, knowledge_base_id)
    assert [row.status for row in rows] == ["failed", "active"]
    async with migrated_database.sessions() as session:
        knowledge_base = await session.get(KnowledgeBase, knowledge_base_id)
        assert knowledge_base is not None
        assert knowledge_base.pending_index_generation_id is None
        assert knowledge_base.active_index_generation_id == rows[1].id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generation_replay_with_invalid_persisted_filter_snapshot_fails_terminally(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    settings = _settings(postgres_urls[0])
    _admin, token = await _create_admin(migrated_database, settings)
    knowledge_base_id, profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )
    hooks, disable = _hooks("after_reservation")
    qdrant = FakeQdrantClient(clock=lambda: NOW)
    gateway = _FakeEmbeddingGateway()
    app = _app(migrated_database, settings, qdrant, gateway, hooks)
    path = f"/v1/admin/knowledge-bases/{knowledge_base_id}/index-generations"
    command = {"embedding_profile_id": str(profile_id), "distance": "cosine"}
    key = "generation-invalid-persisted-filter"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        interrupted = await client.post(
            path,
            headers=_headers(token, "req-invalid-filter-setup", key),
            json=command,
        )
        _assert_error(interrupted, 500, "INTERNAL_ERROR", "req-invalid-filter-setup")
        disable()
        rows = await _generation_rows(migrated_database, knowledge_base_id)
        assert len(rows) == 1
        async with migrated_database.sessions() as session, session.begin():
            generation = await session.get(
                KnowledgeBaseIndexGeneration,
                rows[0].id,
                with_for_update=True,
            )
            assert generation is not None
            generation.filter_schema_snapshot = {"fields": [{"valid_json": "bad shape"}]}
        failed = await client.post(
            path,
            headers=_headers(token, "req-invalid-filter-resume", key),
            json=command,
        )
        replay = await client.post(
            path,
            headers=_headers(token, "req-invalid-filter-replay", key),
            json=command,
        )

    _assert_error(
        failed,
        409,
        "GENERATION_CONFIGURATION_CONFLICT",
        "req-invalid-filter-resume",
    )
    _assert_error(
        replay,
        409,
        "GENERATION_CONFIGURATION_CONFLICT",
        "req-invalid-filter-replay",
    )
    rows = await _generation_rows(migrated_database, knowledge_base_id)
    assert len(rows) == 1 and rows[0].status == "failed"
    assert len(qdrant.collection_names) == 0
    assert gateway.calls == []
    async with migrated_database.sessions() as session:
        knowledge_base = await session.get(KnowledgeBase, knowledge_base_id)
        assert knowledge_base is not None
        assert knowledge_base.pending_index_generation_id is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generation_replay_with_missing_nullable_snapshot_field_fails_terminally(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    settings = _settings(postgres_urls[0])
    _admin, token = await _create_admin(migrated_database, settings)
    knowledge_base_id, profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )
    hooks, disable = _hooks("after_reservation")
    qdrant = FakeQdrantClient(clock=lambda: NOW)
    gateway = _FakeEmbeddingGateway()
    app = _app(migrated_database, settings, qdrant, gateway, hooks)
    path = f"/v1/admin/knowledge-bases/{knowledge_base_id}/index-generations"
    command = {"embedding_profile_id": str(profile_id), "distance": "cosine"}
    key = "generation-missing-persisted-snapshot"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        interrupted = await client.post(
            path,
            headers=_headers(token, "req-missing-snapshot-setup", key),
            json=command,
        )
        _assert_error(interrupted, 500, "INTERNAL_ERROR", "req-missing-snapshot-setup")
        disable()
        rows = await _generation_rows(migrated_database, knowledge_base_id)
        assert len(rows) == 1
        async with migrated_database.sessions() as session, session.begin():
            await session.execute(
                update(KnowledgeBaseIndexGeneration)
                .where(KnowledgeBaseIndexGeneration.id == rows[0].id)
                .values(filter_schema_snapshot=null())
            )
        failed = await client.post(
            path,
            headers=_headers(token, "req-missing-snapshot-resume", key),
            json=command,
        )
        replay = await client.post(
            path,
            headers=_headers(token, "req-missing-snapshot-replay", key),
            json=command,
        )

    _assert_error(
        failed,
        409,
        "GENERATION_CONFIGURATION_CONFLICT",
        "req-missing-snapshot-resume",
    )
    _assert_error(
        replay,
        409,
        "GENERATION_CONFIGURATION_CONFLICT",
        "req-missing-snapshot-replay",
    )
    rows = await _generation_rows(migrated_database, knowledge_base_id)
    assert len(rows) == 1 and rows[0].status == "failed"
    assert not qdrant.collection_names
    assert gateway.calls == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generation_replay_rejects_collection_name_not_bound_to_generation_identity(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    settings = _settings(postgres_urls[0])
    _admin, token = await _create_admin(migrated_database, settings)
    knowledge_base_id, profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )
    hooks, disable = _hooks("after_reservation")
    qdrant = FakeQdrantClient(clock=lambda: NOW)
    gateway = _FakeEmbeddingGateway()
    app = _app(migrated_database, settings, qdrant, gateway, hooks)
    path = f"/v1/admin/knowledge-bases/{knowledge_base_id}/index-generations"
    command = {"embedding_profile_id": str(profile_id), "distance": "cosine"}
    key = "generation-corrupt-collection-identity"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        interrupted = await client.post(
            path,
            headers=_headers(token, "req-collection-identity-setup", key),
            json=command,
        )
        _assert_error(interrupted, 500, "INTERNAL_ERROR", "req-collection-identity-setup")
        disable()
        rows = await _generation_rows(migrated_database, knowledge_base_id)
        assert len(rows) == 1
        async with migrated_database.sessions() as session, session.begin():
            generation = await session.get(
                KnowledgeBaseIndexGeneration,
                rows[0].id,
                with_for_update=True,
            )
            assert generation is not None
            generation.qdrant_collection_name = collection_name(knowledge_base_id, uuid4())
        response = await client.post(
            path,
            headers=_headers(token, "req-collection-identity-resume", key),
            json=command,
        )

    _assert_error(
        response,
        409,
        "GENERATION_CONFIGURATION_CONFLICT",
        "req-collection-identity-resume",
    )
    rows = await _generation_rows(migrated_database, knowledge_base_id)
    assert len(rows) == 1 and rows[0].status == "failed"
    assert qdrant.collection_names == ()
    assert qdrant.create_calls == 0
    assert gateway.calls == []


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tamper",
    [
        "model_name",
        "base_url",
        "dimension",
        "embedding_config_hash",
        "index_profile_hash",
    ],
)
async def test_generation_detects_persisted_snapshot_or_hash_tampering_before_qdrant_io(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
    tamper: str,
) -> None:
    settings = _settings(postgres_urls[0])
    _admin, token = await _create_admin(migrated_database, settings)
    knowledge_base_id, profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )
    hooks, disable = _hooks("after_reservation")
    qdrant = FakeQdrantClient(clock=lambda: NOW)
    gateway = _FakeEmbeddingGateway()
    app = _app(migrated_database, settings, qdrant, gateway, hooks)
    path = f"/v1/admin/knowledge-bases/{knowledge_base_id}/index-generations"
    command = {"embedding_profile_id": str(profile_id), "distance": "cosine"}
    key = f"generation-snapshot-tamper-{tamper}"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        interrupted = await client.post(
            path,
            headers=_headers(token, f"req-{tamper}-setup", key),
            json=command,
        )
        _assert_error(interrupted, 500, "INTERNAL_ERROR", f"req-{tamper}-setup")
        disable()
        rows = await _generation_rows(migrated_database, knowledge_base_id)
        assert len(rows) == 1
        async with migrated_database.sessions() as session, session.begin():
            generation = await session.get(
                KnowledgeBaseIndexGeneration,
                rows[0].id,
                with_for_update=True,
            )
            assert generation is not None
            if tamper == "embedding_config_hash":
                generation.embedding_config_hash = "b" * 64
            elif tamper == "index_profile_hash":
                generation.index_profile_hash = "c" * 64
            else:
                assert generation.embedding_config_snapshot is not None
                snapshot = dict(generation.embedding_config_snapshot)
                replacement: object = {
                    "model_name": "tampered-model",
                    "base_url": "https://tampered.example/v1",
                    "dimension": 4,
                }[tamper]
                snapshot[tamper] = replacement
                generation.embedding_config_snapshot = snapshot
        response = await client.post(
            path,
            headers=_headers(token, f"req-{tamper}-resume", key),
            json=command,
        )

    _assert_error(
        response,
        409,
        "GENERATION_CONFIGURATION_CONFLICT",
        f"req-{tamper}-resume",
    )
    rows = await _generation_rows(migrated_database, knowledge_base_id)
    assert len(rows) == 1 and rows[0].status == "failed"
    assert qdrant.collection_names == ()
    assert qdrant.create_calls == 0
    assert gateway.calls == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_generation_rejects_unowned_same_schema_empty_collection_without_mutation(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    settings = _settings(postgres_urls[0])
    _admin, token = await _create_admin(migrated_database, settings)
    knowledge_base_id, profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )
    hooks, disable = _hooks("after_reservation")
    qdrant = FakeQdrantClient(clock=lambda: NOW)
    gateway = _FakeEmbeddingGateway()
    app = _app(migrated_database, settings, qdrant, gateway, hooks)
    path = f"/v1/admin/knowledge-bases/{knowledge_base_id}/index-generations"
    command = {"embedding_profile_id": str(profile_id), "distance": "cosine"}
    key = "generation-unowned-same-schema"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        interrupted = await client.post(
            path,
            headers=_headers(token, "req-unowned-setup", key),
            json=command,
        )
        _assert_error(interrupted, 500, "INTERNAL_ERROR", "req-unowned-setup")
        rows = await _generation_rows(migrated_database, knowledge_base_id)
        assert len(rows) == 1
        generation = rows[0]
        assert generation.filter_schema_snapshot is not None
        spec = CollectionSpec(
            generation.qdrant_collection_name,
            3,
            "cosine",
            payload_indexes_for_filter_snapshot(generation.filter_schema_snapshot),
        )
        await qdrant.seed_collection(spec, created_at=NOW, managed=False)
        disable()
        response = await client.post(
            path,
            headers=_headers(token, "req-unowned-resume", key),
            json=command,
        )

    _assert_error(
        response,
        409,
        "QDRANT_COLLECTION_MISMATCH",
        "req-unowned-resume",
    )
    assert qdrant.collection_names == (spec.name,)
    assert qdrant.payload_index_create_calls == 0
    assert gateway.calls == []


@pytest.mark.integration
@pytest.mark.asyncio
async def test_minimum_one_token_probe_calls_gateway_and_can_activate(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    settings = _settings(postgres_urls[0])
    _admin, token = await _create_admin(migrated_database, settings)
    knowledge_base_id, profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database,
        max_input_tokens=1,
    )
    qdrant = FakeQdrantClient(clock=lambda: NOW)
    gateway = _FakeEmbeddingGateway()
    app = _app(migrated_database, settings, qdrant, gateway)
    path = f"/v1/admin/knowledge-bases/{knowledge_base_id}/index-generations"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            path,
            headers=_headers(token, "req-one-token-probe", "generation-one-token-probe"),
            json={"embedding_profile_id": str(profile_id), "distance": "cosine"},
        )

    assert response.status_code == 201
    assert len(gateway.calls) == 1
    assert gateway.calls[0][2] == ("x",)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("same_key", [True, False])
async def test_concurrent_generation_requests_obey_idempotency_and_single_generation_constraint(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
    same_key: bool,
) -> None:
    settings = _settings(postgres_urls[0])
    _admin, token = await _create_admin(migrated_database, settings)
    knowledge_base_id, profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )
    qdrant = FakeQdrantClient(clock=lambda: NOW)
    gateway = _FakeEmbeddingGateway()
    app = _app(migrated_database, settings, qdrant, gateway)
    path = f"/v1/admin/knowledge-bases/{knowledge_base_id}/index-generations"
    command = {"embedding_profile_id": str(profile_id), "distance": "cosine"}
    keys = ("generation-concurrent", "generation-concurrent" if same_key else "generation-other")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        first, second = await asyncio.gather(
            client.post(
                path,
                headers=_headers(token, "req-generation-concurrent-1", keys[0]),
                json=command,
            ),
            client.post(
                path,
                headers=_headers(token, "req-generation-concurrent-2", keys[1]),
                json=command,
            ),
        )

    if same_key:
        assert (first.status_code, second.status_code) == (201, 201)
        assert first.json() == second.json()
    else:
        assert sorted((first.status_code, second.status_code)) == [201, 409]
        conflict = first if first.status_code == 409 else second
        assert conflict.json()["error"]["code"] == "INDEX_GENERATION_ALREADY_CONFIGURED"
    assert len(await _generation_rows(migrated_database, knowledge_base_id)) == 1
    assert qdrant.create_calls == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_late_same_key_qdrant_failure_cannot_overwrite_successful_activation(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    settings = _settings(postgres_urls[0])
    _admin, token = await _create_admin(migrated_database, settings)
    knowledge_base_id, profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )
    both_reserved = asyncio.Event()
    activation_committing = asyncio.Event()
    reservation_count = 0
    reservation_lock = asyncio.Lock()

    async def checkpoint(name: str) -> None:
        nonlocal reservation_count
        if name == "after_reservation":
            async with reservation_lock:
                reservation_count += 1
                if reservation_count == 2:
                    both_reserved.set()
            await both_reserved.wait()
        elif name == "before_activation_commit":
            activation_committing.set()

    class _LateFailingQdrant(FakeQdrantClient):
        def __init__(self) -> None:
            super().__init__(clock=lambda: NOW)
            self._ensure_calls = 0
            self._ensure_lock = asyncio.Lock()

        async def ensure_collection(self, spec: CollectionSpec) -> None:
            async with self._ensure_lock:
                self._ensure_calls += 1
                call = self._ensure_calls
            if call == 2:
                await activation_committing.wait()
                raise QdrantConfigurationError("late qdrant mismatch")
            await super().ensure_collection(spec)

    qdrant = _LateFailingQdrant()
    gateway = _FakeEmbeddingGateway()
    app = _app(
        migrated_database,
        settings,
        qdrant,
        gateway,
        GenerationSagaHooks(checkpoint),
    )
    path = f"/v1/admin/knowledge-bases/{knowledge_base_id}/index-generations"
    command = {"embedding_profile_id": str(profile_id), "distance": "cosine"}
    key = "generation-late-qdrant-failure"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        first, second = await asyncio.gather(
            client.post(
                path,
                headers=_headers(token, "req-late-failure-first", key),
                json=command,
            ),
            client.post(
                path,
                headers=_headers(token, "req-late-failure-second", key),
                json=command,
            ),
        )
        replay = await client.post(
            path,
            headers=_headers(token, "req-late-failure-replay", key),
            json=command,
        )

    assert sorted((first.status_code, second.status_code)) == [201, 409]
    assert replay.status_code == 201
    rows = await _generation_rows(migrated_database, knowledge_base_id)
    assert len(rows) == 1 and rows[0].status == "active"
    async with migrated_database.sessions() as session:
        request = await session.scalar(
            select(IndexGenerationCreationRequest).where(
                IndexGenerationCreationRequest.knowledge_base_id == knowledge_base_id,
                IndexGenerationCreationRequest.idempotency_key == key,
            )
        )
        knowledge_base = await session.get(KnowledgeBase, knowledge_base_id)
        assert request is not None and request.state == "succeeded"
        assert knowledge_base is not None
        assert knowledge_base.active_index_generation_id == rows[0].id
        assert knowledge_base.pending_index_generation_id is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_orphan_reconciliation_deletes_only_old_unowned_or_failed_collections(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    settings = _settings(postgres_urls[0])
    _admin, _token = await _create_admin(migrated_database, settings)
    knowledge_base_id, profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )
    valid_generation_id = uuid4()
    failed_generation_id = uuid4()
    orphan_generation_id = uuid4()
    recent_generation_id = uuid4()
    boundary_generation_id = uuid4()
    valid_name = collection_name(knowledge_base_id, valid_generation_id)
    failed_name = collection_name(knowledge_base_id, failed_generation_id)
    orphan_name = collection_name(knowledge_base_id, orphan_generation_id)
    recent_name = collection_name(knowledge_base_id, recent_generation_id)
    boundary_name = collection_name(knowledge_base_id, boundary_generation_id)
    async with migrated_database.sessions() as session, session.begin():
        session.add_all(
            [
                KnowledgeBaseIndexGeneration(
                    id=valid_generation_id,
                    knowledge_base_id=knowledge_base_id,
                    embedding_profile_id=profile_id,
                    index_profile_hash="a" * 64,
                    qdrant_collection_name=valid_name,
                    status="building",
                    rebuild_snapshot_at=NOW,
                    caught_up_revision=0,
                ),
                KnowledgeBaseIndexGeneration(
                    id=failed_generation_id,
                    knowledge_base_id=knowledge_base_id,
                    embedding_profile_id=profile_id,
                    index_profile_hash="b" * 64,
                    qdrant_collection_name=failed_name,
                    status="failed",
                    rebuild_snapshot_at=NOW,
                    caught_up_revision=0,
                    safe_error_code="PROVIDER_MODEL_NOT_FOUND",
                    safe_error_message="Provider model not found",
                ),
            ]
        )

    qdrant = FakeQdrantClient(clock=lambda: NOW)
    old = NOW - timedelta(hours=25)
    for name, created_at in (
        (valid_name, old),
        (failed_name, old),
        (orphan_name, old),
        (recent_name, NOW - timedelta(hours=1)),
        (boundary_name, NOW - timedelta(hours=24)),
    ):
        await qdrant.seed_collection(CollectionSpec(name, 3, "cosine", ()), created_at=created_at)

    service = GenerationService(
        session_factory=migrated_database.sessions,
        qdrant=qdrant,
        embedding_gateway=_FakeEmbeddingGateway(),
        clock=lambda: NOW,
    )
    deleted: list[str] = []
    cursor: str | None = None
    page_count = 0
    while True:
        result = await service.reconcile_orphan_collections(
            grace_period=timedelta(hours=24),
            limit=2,
            cursor=cursor,
        )
        assert len(result.deleted) <= 2
        deleted.extend(result.deleted)
        page_count += 1
        if result.next_cursor is None:
            break
        assert result.next_cursor != cursor
        cursor = result.next_cursor

    assert page_count == 3
    assert set(deleted) == {failed_name, orphan_name}
    assert set(qdrant.collection_names) == {valid_name, recent_name, boundary_name}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_orphan_reconciliation_rechecks_status_before_delete_and_releases_claim(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    settings = _settings(postgres_urls[0])
    _admin, _token = await _create_admin(migrated_database, settings)
    knowledge_base_id, profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )
    generation_id = uuid4()
    name = collection_name(knowledge_base_id, generation_id)
    async with migrated_database.sessions() as session, session.begin():
        session.add(
            KnowledgeBaseIndexGeneration(
                id=generation_id,
                knowledge_base_id=knowledge_base_id,
                embedding_profile_id=profile_id,
                index_profile_hash="a" * 64,
                qdrant_collection_name=name,
                status="failed",
                rebuild_snapshot_at=NOW,
                caught_up_revision=0,
                safe_error_code="PROVIDER_MODEL_NOT_FOUND",
                safe_error_message="Provider model not found",
            )
        )

    qdrant = FakeQdrantClient(clock=lambda: NOW)
    await qdrant.seed_collection(
        CollectionSpec(name, 3, "cosine", ()),
        created_at=NOW - timedelta(hours=25),
    )

    async def make_generation_building() -> None:
        async with migrated_database.sessions() as session, session.begin():
            generation = await session.get(KnowledgeBaseIndexGeneration, generation_id)
            assert generation is not None
            generation.status = "building"
            generation.safe_error_code = None
            generation.safe_error_message = None

    hooks, _disable = _hooks("before_orphan_authoritative_recheck", make_generation_building)
    service = GenerationService(
        session_factory=migrated_database.sessions,
        qdrant=qdrant,
        embedding_gateway=_FakeEmbeddingGateway(),
        clock=lambda: NOW,
        hooks=hooks,
    )

    raced = await service.reconcile_orphan_collections(
        grace_period=timedelta(hours=24),
        limit=10,
        cursor=None,
    )

    assert raced.deleted == ()
    assert qdrant.collection_names == (name,)
    async with migrated_database.sessions() as session:
        repository = SqlAlchemyGenerationRepository(session)
        assert await repository.collection_cleanup_claim_exists(name) is False

    async with migrated_database.sessions() as session, session.begin():
        generation = await session.get(KnowledgeBaseIndexGeneration, generation_id)
        assert generation is not None
        generation.status = "failed"
        generation.safe_error_code = "PROVIDER_MODEL_NOT_FOUND"
        generation.safe_error_message = "Provider model not found"

    recovered = await service.reconcile_orphan_collections(
        grace_period=timedelta(hours=24),
        limit=10,
        cursor=None,
    )
    assert recovered.deleted == (name,)
    assert not qdrant.collection_names


@pytest.mark.integration
@pytest.mark.asyncio
async def test_orphan_reconciliation_reserves_bounded_work_for_expired_cleanup_claims(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    settings = _settings(postgres_urls[0])
    _admin, _token = await _create_admin(migrated_database, settings)
    knowledge_base_id, _profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )
    qdrant_generation_ids = (
        UUID("00000000-0000-0000-0000-000000000001"),
        UUID("00000000-0000-0000-0000-000000000002"),
    )
    claimed_generation_ids = (
        UUID("00000000-0000-0000-0000-000000000003"),
        UUID("00000000-0000-0000-0000-000000000004"),
    )
    qdrant_names = tuple(
        collection_name(knowledge_base_id, generation_id) for generation_id in qdrant_generation_ids
    )
    claimed_names = tuple(
        collection_name(knowledge_base_id, generation_id)
        for generation_id in claimed_generation_ids
    )
    qdrant = FakeQdrantClient(clock=lambda: NOW)
    for name in qdrant_names:
        await qdrant.seed_collection(
            CollectionSpec(name, 3, "cosine", ()),
            created_at=NOW - timedelta(hours=25),
        )
    async with migrated_database.sessions() as session, session.begin():
        repository = SqlAlchemyGenerationRepository(session)
        for name, generation_id in zip(claimed_names, claimed_generation_ids, strict=True):
            await repository.get_knowledge_base_for_update(knowledge_base_id)
            await repository.acquire_collection_fence(name)
            claim = await repository.claim_collection_cleanup(
                collection_name=name,
                knowledge_base_id=knowledge_base_id,
                generation_id=generation_id,
                lease_owner=uuid4(),
                now=NOW - timedelta(minutes=2),
                lease_duration=timedelta(minutes=1),
            )
            assert claim is not None

    result = await GenerationService(
        session_factory=migrated_database.sessions,
        qdrant=qdrant,
        embedding_gateway=_FakeEmbeddingGateway(),
        clock=lambda: NOW,
    ).reconcile_orphan_collections(
        grace_period=timedelta(hours=24),
        limit=2,
        cursor=None,
    )

    assert result.deleted == (claimed_names[0], qdrant_names[0])
    async with migrated_database.sessions() as session:
        repository = SqlAlchemyGenerationRepository(session)
        assert await repository.collection_cleanup_claim_exists(claimed_names[0]) is True
        assert await repository.collection_cleanup_claim_exists(claimed_names[1]) is True
        first_claim = await session.get(IndexGenerationCleanupClaim, claimed_names[0])
        second_claim = await session.get(IndexGenerationCleanupClaim, claimed_names[1])
        assert first_claim is not None and first_claim.completed_at == NOW
        assert second_claim is not None and second_claim.completed_at is None
    assert qdrant.collection_names == (qdrant_names[1],)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_orphan_reconciliation_limit_one_alternates_expired_and_qdrant_work(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    settings = _settings(postgres_urls[0])
    _admin, _token = await _create_admin(migrated_database, settings)
    knowledge_base_id, _profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )
    page_generation_id = UUID("00000000-0000-0000-0000-000000000001")
    expired_generation_id = UUID("ffffffff-ffff-4fff-8fff-ffffffffffff")
    page_name = collection_name(knowledge_base_id, page_generation_id)
    expired_name = collection_name(knowledge_base_id, expired_generation_id)

    class _PermanentExpiredFailureQdrant(FakeQdrantClient):
        async def delete_collection(self, collection: str) -> None:
            if collection == expired_name:
                raise QdrantTransientError("Qdrant unavailable")
            await super().delete_collection(collection)

    qdrant = _PermanentExpiredFailureQdrant(clock=lambda: NOW)
    for name in (page_name, expired_name):
        await qdrant.seed_collection(
            CollectionSpec(name, 3, "cosine", ()),
            created_at=NOW - timedelta(hours=25),
        )
    async with migrated_database.sessions() as session, session.begin():
        repository = SqlAlchemyGenerationRepository(session)
        await repository.get_knowledge_base_for_update(knowledge_base_id)
        await repository.acquire_collection_fence(expired_name)
        claim = await repository.claim_collection_cleanup(
            collection_name=expired_name,
            knowledge_base_id=knowledge_base_id,
            generation_id=expired_generation_id,
            lease_owner=uuid4(),
            now=NOW - timedelta(minutes=2),
            lease_duration=timedelta(minutes=1),
        )
        assert claim is not None

    service = GenerationService(
        session_factory=migrated_database.sessions,
        qdrant=qdrant,
        embedding_gateway=_FakeEmbeddingGateway(),
        clock=lambda: NOW,
    )
    expired_turn = await service.reconcile_orphan_collections(
        grace_period=timedelta(hours=24),
        limit=1,
        cursor=None,
    )
    assert expired_turn.deleted == ()
    assert expired_turn.next_cursor is not None
    assert page_name in qdrant.collection_names

    page_turn = await service.reconcile_orphan_collections(
        grace_period=timedelta(hours=24),
        limit=1,
        cursor=expired_turn.next_cursor,
    )
    assert page_turn.deleted == (page_name,)
    assert qdrant.collection_names == (expired_name,)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_expired_cleanup_keyset_progresses_past_persistent_head_without_skipping_page(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    settings = _settings(postgres_urls[0])
    _admin, _token = await _create_admin(migrated_database, settings)
    knowledge_base_id, _profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )
    generation_ids = (
        UUID("00000000-0000-0000-0000-000000000001"),
        UUID("00000000-0000-0000-0000-000000000002"),
        UUID("00000000-0000-0000-0000-000000000003"),
    )
    names = tuple(
        collection_name(knowledge_base_id, generation_id) for generation_id in generation_ids
    )

    class _TrackingQdrant(FakeQdrantClient):
        def __init__(self) -> None:
            super().__init__(clock=lambda: NOW)
            self.page_calls: list[str | None] = []

        async def list_managed_collections(
            self,
            *,
            limit: int,
            cursor: str | None,
        ) -> ManagedCollectionPage:
            self.page_calls.append(cursor)
            return await super().list_managed_collections(limit=limit, cursor=cursor)

    class _PersistentHeadFailureService(GenerationService):
        claim_attempts: list[str]

        async def _claim_orphan_cleanup(
            self,
            collection: str,
        ) -> CollectionCleanupClaim | None:
            self.claim_attempts.append(collection)
            if collection == names[0]:
                raise _InjectedCrash("persistent expired head")
            return await super()._claim_orphan_cleanup(collection)

    qdrant = _TrackingQdrant()
    for name in names:
        await qdrant.seed_collection(
            CollectionSpec(name, 3, "cosine", ()),
            created_at=NOW - timedelta(hours=25),
        )
    async with migrated_database.sessions() as session, session.begin():
        repository = SqlAlchemyGenerationRepository(session)
        await repository.get_knowledge_base_for_update(knowledge_base_id)
        for offset, (generation_id, name) in enumerate(zip(generation_ids, names, strict=True)):
            await repository.acquire_collection_fence(name)
            claim = await repository.claim_collection_cleanup(
                collection_name=name,
                knowledge_base_id=knowledge_base_id,
                generation_id=generation_id,
                lease_owner=uuid4(),
                now=NOW - timedelta(minutes=3 - offset),
                lease_duration=timedelta(minutes=1),
            )
            assert claim is not None

    service = _PersistentHeadFailureService(
        session_factory=migrated_database.sessions,
        qdrant=qdrant,
        embedding_gateway=_FakeEmbeddingGateway(),
        clock=lambda: NOW,
    )
    service.claim_attempts = []
    cursor = None
    results = []
    for _ in range(3):
        result = await service.reconcile_orphan_collections(
            grace_period=timedelta(hours=24),
            limit=1,
            cursor=cursor,
        )
        results.append(result)
        cursor = result.next_cursor
        assert cursor is not None

    assert qdrant.page_calls == [None]
    assert service.claim_attempts[:3] == [names[0], names[0], names[1]]
    assert results[2].deleted == (names[1],)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_failed_expired_claim_preserves_qdrant_page_progress_across_rounds(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    settings = _settings(postgres_urls[0])
    _admin, _token = await _create_admin(migrated_database, settings)
    knowledge_base_id, _profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )
    recent_name = collection_name(
        knowledge_base_id,
        UUID("00000000-0000-0000-0000-000000000001"),
    )
    orphan_name = collection_name(
        knowledge_base_id,
        UUID("00000000-0000-0000-0000-000000000002"),
    )
    expired_generation_id = UUID("ffffffff-ffff-4fff-8fff-ffffffffffff")
    expired_name = collection_name(knowledge_base_id, expired_generation_id)

    class _PermanentExpiredFailureQdrant(FakeQdrantClient):
        async def delete_collection(self, collection: str) -> None:
            if collection == expired_name:
                raise QdrantTransientError("Qdrant unavailable")
            await super().delete_collection(collection)

    qdrant = _PermanentExpiredFailureQdrant(clock=lambda: NOW)
    await qdrant.seed_collection(
        CollectionSpec(recent_name, 3, "cosine", ()),
        created_at=NOW - timedelta(hours=1),
    )
    for name in (orphan_name, expired_name):
        await qdrant.seed_collection(
            CollectionSpec(name, 3, "cosine", ()),
            created_at=NOW - timedelta(hours=25),
        )
    async with migrated_database.sessions() as session, session.begin():
        repository = SqlAlchemyGenerationRepository(session)
        await repository.get_knowledge_base_for_update(knowledge_base_id)
        await repository.acquire_collection_fence(expired_name)
        claim = await repository.claim_collection_cleanup(
            collection_name=expired_name,
            knowledge_base_id=knowledge_base_id,
            generation_id=expired_generation_id,
            lease_owner=uuid4(),
            now=NOW - timedelta(minutes=2),
            lease_duration=timedelta(minutes=1),
        )
        assert claim is not None

    service = GenerationService(
        session_factory=migrated_database.sessions,
        qdrant=qdrant,
        embedding_gateway=_FakeEmbeddingGateway(),
        clock=lambda: NOW,
    )
    first = await service.reconcile_orphan_collections(
        grace_period=timedelta(hours=24),
        limit=2,
        cursor=None,
    )
    assert first.deleted == ()
    assert first.next_cursor is not None
    second = await service.reconcile_orphan_collections(
        grace_period=timedelta(hours=24),
        limit=2,
        cursor=first.next_cursor,
    )
    assert second.deleted == (orphan_name,)
    assert set(qdrant.collection_names) == {recent_name, expired_name}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_expired_cleanup_claim_query_skips_malformed_rows_before_limit(
    migrated_database: Database,
) -> None:
    malformed_knowledge_base_id = uuid4()
    malformed_generation_id = uuid4()
    valid_knowledge_base_id = uuid4()
    valid_generation_id = uuid4()
    malformed_name = collection_name(malformed_knowledge_base_id, malformed_generation_id)
    valid_name = collection_name(valid_knowledge_base_id, valid_generation_id)

    async with migrated_database.sessions() as session:
        transaction = await session.begin()
        try:
            await session.execute(
                text(
                    "ALTER TABLE index_generation_cleanup_claims "
                    "DROP CONSTRAINT IF EXISTS ck_generation_cleanup_claims_collection_identity"
                )
            )
            await session.execute(
                text(
                    """
                    INSERT INTO index_generation_cleanup_claims (
                        collection_name, knowledge_base_id, generation_id, lease_owner,
                        lease_epoch, lease_expires_at
                    ) VALUES
                        (:malformed_name, :wrong_knowledge_base_id, :malformed_generation_id,
                         :malformed_owner, 1, :malformed_expiry),
                        (:valid_name, :valid_knowledge_base_id, :valid_generation_id,
                         :valid_owner, 1, :valid_expiry)
                    """
                ),
                {
                    "malformed_name": malformed_name,
                    "wrong_knowledge_base_id": uuid4(),
                    "malformed_generation_id": malformed_generation_id,
                    "malformed_owner": uuid4(),
                    "malformed_expiry": NOW - timedelta(minutes=3),
                    "valid_name": valid_name,
                    "valid_knowledge_base_id": valid_knowledge_base_id,
                    "valid_generation_id": valid_generation_id,
                    "valid_owner": uuid4(),
                    "valid_expiry": NOW - timedelta(minutes=2),
                },
            )
            claims = await SqlAlchemyGenerationRepository(
                session
            ).list_expired_collection_cleanup_claims(now=NOW, limit=1)
            assert tuple(claim.collection_name for claim in claims) == (valid_name,)
        finally:
            await transaction.rollback()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_orphan_reconciliation_skips_malformed_expired_claims_without_aborting_round(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    settings = _settings(postgres_urls[0])
    _admin, _token = await _create_admin(migrated_database, settings)
    knowledge_base_id, _profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )
    valid_generation_id = uuid4()
    valid_name = collection_name(knowledge_base_id, valid_generation_id)
    malformed_knowledge_base_id = uuid4()
    malformed_generation_id = uuid4()
    mismatched_name = collection_name(malformed_knowledge_base_id, malformed_generation_id)
    malformed_claims = (
        CollectionCleanupClaim(
            collection_name="not-a-managed-collection",
            knowledge_base_id=uuid4(),
            generation_id=uuid4(),
            lease_owner=uuid4(),
            lease_epoch=1,
            lease_expires_at=NOW - timedelta(minutes=2),
            completed_at=None,
        ),
        CollectionCleanupClaim(
            collection_name=mismatched_name,
            knowledge_base_id=uuid4(),
            generation_id=malformed_generation_id,
            lease_owner=uuid4(),
            lease_epoch=1,
            lease_expires_at=NOW - timedelta(minutes=1),
            completed_at=None,
        ),
    )
    qdrant = FakeQdrantClient(clock=lambda: NOW)
    await qdrant.seed_collection(
        CollectionSpec(valid_name, 3, "cosine", ()),
        created_at=NOW - timedelta(hours=25),
    )
    service = GenerationService(
        session_factory=migrated_database.sessions,
        qdrant=qdrant,
        embedding_gateway=_FakeEmbeddingGateway(),
        repository_factory=lambda session: _MalformedExpiredClaimRepository(
            session,
            malformed_claims,
        ),
        clock=lambda: NOW,
    )

    result = await service.reconcile_orphan_collections(
        grace_period=timedelta(hours=24),
        limit=10,
        cursor=None,
    )

    assert result.deleted == (valid_name,)
    assert qdrant.collection_names == ()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_orphan_cleanup_tombstone_blocks_same_key_without_blocking_new_key(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    settings = _settings(postgres_urls[0])
    admin, token = await _create_admin(migrated_database, settings)
    knowledge_base_id, profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )
    key = "generation-orphan-cleanup-race"
    generation_id = uuid5(
        UUID("72fb80bb-a22e-57a7-ac94-6b94caa14a5c"),
        f"{admin.key_id}:{knowledge_base_id}:{key}",
    )
    orphan_name = collection_name(knowledge_base_id, generation_id)
    replacement_key = "generation-orphan-cleanup-race-replacement"
    replacement_generation_id = uuid5(
        UUID("72fb80bb-a22e-57a7-ac94-6b94caa14a5c"),
        f"{admin.key_id}:{knowledge_base_id}:{replacement_key}",
    )
    replacement_name = collection_name(knowledge_base_id, replacement_generation_id)

    class _BlockingDeleteQdrant(FakeQdrantClient):
        def __init__(self) -> None:
            super().__init__(clock=lambda: NOW)
            self.delete_started = asyncio.Event()
            self.release_delete = asyncio.Event()
            self.delete_attempts = 0

        async def delete_collection(self, collection: str) -> None:
            assert collection == orphan_name
            self.delete_attempts += 1
            if self.delete_attempts == 1:
                self.delete_started.set()
                await self.release_delete.wait()
            await super().delete_collection(collection)

    qdrant = _BlockingDeleteQdrant()
    await qdrant.seed_collection(
        CollectionSpec(orphan_name, 3, "cosine", ()),
        created_at=NOW - timedelta(hours=25),
    )
    gateway = _FakeEmbeddingGateway()
    app = _app(migrated_database, settings, qdrant, gateway)
    clock_now = NOW
    service = GenerationService(
        session_factory=migrated_database.sessions,
        qdrant=qdrant,
        embedding_gateway=gateway,
        clock=lambda: clock_now,
        cleanup_lease_duration=timedelta(minutes=1),
    )
    path = f"/v1/admin/knowledge-bases/{knowledge_base_id}/index-generations"

    reconciliation = asyncio.create_task(
        service.reconcile_orphan_collections(
            grace_period=timedelta(hours=24),
            limit=10,
            cursor=None,
        )
    )
    await asyncio.wait_for(qdrant.delete_started.wait(), timeout=2)
    post: asyncio.Task[httpx.Response] | None = None
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            post = asyncio.create_task(
                client.post(
                    path,
                    headers=_headers(token, "req-orphan-cleanup-race", key),
                    json={"embedding_profile_id": str(profile_id), "distance": "cosine"},
                )
            )
            response = await asyncio.wait_for(asyncio.shield(post), timeout=0.5)
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "GENERATION_CLEANUP_IN_PROGRESS"
            assert (await _generation_rows(migrated_database, knowledge_base_id)) == []

            clock_now = NOW + timedelta(minutes=2)
            recovered_result = await service.reconcile_orphan_collections(
                grace_period=timedelta(hours=24),
                limit=10,
                cursor=None,
            )
            retired = await client.post(
                path,
                headers=_headers(token, "req-orphan-cleanup-race-retry", key),
                json={"embedding_profile_id": str(profile_id), "distance": "cosine"},
            )
            replacement = await client.post(
                path,
                headers=_headers(
                    token,
                    "req-orphan-cleanup-race-replacement",
                    replacement_key,
                ),
                json={"embedding_profile_id": str(profile_id), "distance": "cosine"},
            )
            qdrant.release_delete.set()
            stale_result = await reconciliation
    finally:
        qdrant.release_delete.set()
        pending = [task for task in (post, reconciliation) if task is not None and not task.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    assert retired.status_code == 409
    assert retired.json()["error"]["code"] == "GENERATION_COLLECTION_RETIRED"
    assert replacement.status_code == 201
    assert recovered_result.deleted == (orphan_name,)
    assert stale_result.deleted == ()
    rows = await _generation_rows(migrated_database, knowledge_base_id)
    assert len(rows) == 1
    assert rows[0].id == replacement_generation_id
    assert rows[0].status == "active"
    assert qdrant.collection_names == (replacement_name,)
    await qdrant.verify_collection(
        CollectionSpec(
            replacement_name,
            3,
            "cosine",
            payload_indexes_for_filter_snapshot(rows[0].filter_schema_snapshot or {}),
        )
    )
    async with migrated_database.sessions() as session:
        knowledge_base = await session.get(KnowledgeBase, knowledge_base_id)
        assert knowledge_base is not None
        assert knowledge_base.active_index_generation_id == replacement_generation_id
        assert knowledge_base.pending_index_generation_id is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_orphan_cleanup_does_not_block_same_kb_post_for_a_different_collection(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    settings = _settings(postgres_urls[0])
    admin, token = await _create_admin(migrated_database, settings)
    knowledge_base_id, profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )
    cleanup_key = "generation-orphan-cleanup-old-key"
    cleanup_generation_id = uuid5(
        UUID("72fb80bb-a22e-57a7-ac94-6b94caa14a5c"),
        f"{admin.key_id}:{knowledge_base_id}:{cleanup_key}",
    )
    cleanup_name = collection_name(knowledge_base_id, cleanup_generation_id)
    new_key = "generation-orphan-cleanup-new-key"
    new_generation_id = uuid5(
        UUID("72fb80bb-a22e-57a7-ac94-6b94caa14a5c"),
        f"{admin.key_id}:{knowledge_base_id}:{new_key}",
    )
    new_name = collection_name(knowledge_base_id, new_generation_id)

    class _BlockingDeleteQdrant(FakeQdrantClient):
        def __init__(self) -> None:
            super().__init__(clock=lambda: NOW)
            self.delete_started = asyncio.Event()
            self.release_delete = asyncio.Event()

        async def delete_collection(self, collection: str) -> None:
            assert collection == cleanup_name
            self.delete_started.set()
            await self.release_delete.wait()
            await super().delete_collection(collection)

    qdrant = _BlockingDeleteQdrant()
    await qdrant.seed_collection(
        CollectionSpec(cleanup_name, 3, "cosine", ()),
        created_at=NOW - timedelta(hours=25),
    )
    gateway = _FakeEmbeddingGateway()
    app = _app(migrated_database, settings, qdrant, gateway)
    service = GenerationService(
        session_factory=migrated_database.sessions,
        qdrant=qdrant,
        embedding_gateway=gateway,
        clock=lambda: NOW,
    )
    path = f"/v1/admin/knowledge-bases/{knowledge_base_id}/index-generations"
    reconciliation = asyncio.create_task(
        service.reconcile_orphan_collections(
            grace_period=timedelta(hours=24),
            limit=10,
            cursor=None,
        )
    )
    await asyncio.wait_for(qdrant.delete_started.wait(), timeout=2)
    post: asyncio.Task[httpx.Response] | None = None
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            post = asyncio.create_task(
                client.post(
                    path,
                    headers=_headers(token, "req-orphan-cleanup-different-key", new_key),
                    json={"embedding_profile_id": str(profile_id), "distance": "cosine"},
                )
            )
            response = await asyncio.wait_for(asyncio.shield(post), timeout=0.5)
            assert response.status_code == 201
            qdrant.release_delete.set()
            reconciliation_result = await reconciliation
    finally:
        qdrant.release_delete.set()
        pending = [task for task in (post, reconciliation) if task is not None and not task.done()]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    assert reconciliation_result.deleted == (cleanup_name,)
    rows = await _generation_rows(migrated_database, knowledge_base_id)
    assert len(rows) == 1
    assert rows[0].id == new_generation_id
    assert rows[0].status == "active"
    assert qdrant.collection_names == (new_name,)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cleanup_claim_reclaim_and_completion_are_owner_epoch_fenced(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    settings = _settings(postgres_urls[0])
    _admin, _token = await _create_admin(migrated_database, settings)
    knowledge_base_id, _profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )
    generation_id = uuid4()
    name = collection_name(knowledge_base_id, generation_id)
    first_owner = uuid4()
    second_owner = uuid4()

    async with migrated_database.sessions() as session, session.begin():
        repository = SqlAlchemyGenerationRepository(session)
        await repository.get_knowledge_base_for_update(knowledge_base_id)
        await repository.acquire_collection_fence(name)
        first = await repository.claim_collection_cleanup(
            collection_name=name,
            knowledge_base_id=knowledge_base_id,
            generation_id=generation_id,
            lease_owner=first_owner,
            now=NOW,
            lease_duration=timedelta(minutes=1),
        )
    assert first is not None
    assert first.lease_owner == first_owner
    assert first.lease_epoch == 1
    assert first.lease_expires_at == NOW + timedelta(minutes=1)

    async with migrated_database.sessions() as session, session.begin():
        repository = SqlAlchemyGenerationRepository(session)
        await repository.get_knowledge_base_for_update(knowledge_base_id)
        await repository.acquire_collection_fence(name)
        unavailable = await repository.claim_collection_cleanup(
            collection_name=name,
            knowledge_base_id=knowledge_base_id,
            generation_id=generation_id,
            lease_owner=second_owner,
            now=NOW + timedelta(seconds=30),
            lease_duration=timedelta(minutes=1),
        )
    assert unavailable is None

    reclaim_time = NOW + timedelta(minutes=2)
    async with migrated_database.sessions() as session, session.begin():
        repository = SqlAlchemyGenerationRepository(session)
        await repository.get_knowledge_base_for_update(knowledge_base_id)
        await repository.acquire_collection_fence(name)
        second = await repository.claim_collection_cleanup(
            collection_name=name,
            knowledge_base_id=knowledge_base_id,
            generation_id=generation_id,
            lease_owner=second_owner,
            now=reclaim_time,
            lease_duration=timedelta(minutes=1),
        )
    assert second is not None
    assert second.lease_owner == second_owner
    assert second.lease_epoch == 2

    async with migrated_database.sessions() as session, session.begin():
        repository = SqlAlchemyGenerationRepository(session)
        stale_completed = await repository.complete_collection_cleanup(
            name,
            lease_owner=first_owner,
            lease_epoch=first.lease_epoch,
            now=reclaim_time + timedelta(seconds=15),
        )
        assert stale_completed is False
        assert await repository.collection_cleanup_claim_exists(name) is True

    async with migrated_database.sessions() as session, session.begin():
        repository = SqlAlchemyGenerationRepository(session)
        current_completed = await repository.complete_collection_cleanup(
            name,
            lease_owner=second_owner,
            lease_epoch=second.lease_epoch,
            now=reclaim_time + timedelta(seconds=30),
        )
        assert current_completed is True
        completed = await session.get(IndexGenerationCleanupClaim, name)
        assert completed is not None
        assert completed.completed_at == reclaim_time + timedelta(seconds=30)
        assert await repository.collection_cleanup_claim_exists(name) is True
        assert (
            await repository.list_expired_collection_cleanup_claims(
                now=reclaim_time + timedelta(hours=1),
                limit=10,
            )
            == ()
        )
        await repository.get_knowledge_base_for_update(knowledge_base_id)
        await repository.acquire_collection_fence(name)
        unavailable = await repository.claim_collection_cleanup(
            collection_name=name,
            knowledge_base_id=knowledge_base_id,
            generation_id=generation_id,
            lease_owner=uuid4(),
            now=reclaim_time + timedelta(hours=1),
            lease_duration=timedelta(minutes=1),
        )
        assert unavailable is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_failed_orphan_delete_is_isolated_and_expires_claim_for_immediate_recovery(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = _settings(postgres_urls[0])
    _admin, _token = await _create_admin(migrated_database, settings)
    knowledge_base_id, _profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )
    generation_id = UUID("00000000-0000-0000-0000-000000000001")
    second_generation_id = UUID("00000000-0000-0000-0000-000000000002")
    name = collection_name(knowledge_base_id, generation_id)
    second_name = collection_name(knowledge_base_id, second_generation_id)

    class _FailOnceDeleteQdrant(FakeQdrantClient):
        def __init__(self) -> None:
            super().__init__(clock=lambda: NOW)
            self.attempts = 0

        async def delete_collection(self, collection: str) -> None:
            self.attempts += 1
            if self.attempts == 1:
                raise QdrantTransientError("Qdrant unavailable")
            await super().delete_collection(collection)

    qdrant = _FailOnceDeleteQdrant()
    await qdrant.seed_collection(
        CollectionSpec(name, 3, "cosine", ()),
        created_at=NOW - timedelta(hours=25),
    )
    await qdrant.seed_collection(
        CollectionSpec(second_name, 3, "cosine", ()),
        created_at=NOW - timedelta(hours=25),
    )
    service = GenerationService(
        session_factory=migrated_database.sessions,
        qdrant=qdrant,
        embedding_gateway=_FakeEmbeddingGateway(),
        clock=lambda: NOW,
        cleanup_lease_duration=timedelta(minutes=1),
    )

    first_pass = await service.reconcile_orphan_collections(
        grace_period=timedelta(hours=24),
        limit=10,
        cursor=None,
    )

    async with migrated_database.sessions() as session:
        repository = SqlAlchemyGenerationRepository(session)
        expired = await repository.list_expired_collection_cleanup_claims(now=NOW, limit=10)
    assert tuple(claim.collection_name for claim in expired) == (name,)
    assert first_pass.deleted == (second_name,)
    assert qdrant.collection_names == (name,)
    assert name not in caplog.text
    assert second_name not in caplog.text
    failure_records = [
        record
        for record in caplog.records
        if record.msg == "cleanup.action.completed"
        and getattr(record, "event", None) == "cleanup.action.completed"
        and getattr(record, "operation", None) == "orphan_cleanup"
        and getattr(record, "phase", None) == "qdrant"
        and getattr(record, "outcome", None) == "failed"
    ]
    assert failure_records
    assert all(getattr(record, "count", None) in {0, 1} for record in failure_records)

    recovered = await service.reconcile_orphan_collections(
        grace_period=timedelta(hours=24),
        limit=10,
        cursor=None,
    )
    assert recovered.deleted == (name,)
    assert len(qdrant.collection_names) == 0
    async with migrated_database.sessions() as session:
        tombstone = await session.get(IndexGenerationCleanupClaim, name)
        assert tombstone is not None
        assert tombstone.completed_at == NOW


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["claim", "authorize", "delete", "complete", "expire"])
async def test_orphan_cleanup_isolates_every_candidate_operation_base_exception(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
    failure_stage: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = _settings(postgres_urls[0])
    _admin, _token = await _create_admin(migrated_database, settings)
    knowledge_base_id, _profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )
    first_generation_id = UUID("00000000-0000-0000-0000-000000000001")
    second_generation_id = UUID("00000000-0000-0000-0000-000000000002")
    first_name = collection_name(knowledge_base_id, first_generation_id)
    second_name = collection_name(knowledge_base_id, second_generation_id)

    class _OperationFailureQdrant(FakeQdrantClient):
        async def delete_collection(self, collection: str) -> None:
            if collection == first_name:
                if failure_stage == "delete":
                    raise _InjectedCrash("delete")
                if failure_stage == "expire":
                    raise QdrantTransientError("Qdrant unavailable")
            await super().delete_collection(collection)

    class _OperationFailureService(GenerationService):
        async def _claim_orphan_cleanup(
            self,
            collection: str,
        ) -> CollectionCleanupClaim | None:
            if collection == first_name and failure_stage == "claim":
                raise _InjectedCrash("claim")
            return await super()._claim_orphan_cleanup(collection)

        async def _authorize_orphan_cleanup(self, claim: CollectionCleanupClaim) -> bool:
            if claim.collection_name == first_name and failure_stage == "authorize":
                raise _InjectedCrash("authorize")
            return await super()._authorize_orphan_cleanup(claim)

        async def _complete_orphan_cleanup(self, claim: CollectionCleanupClaim) -> bool:
            if claim.collection_name == first_name and failure_stage == "complete":
                raise _InjectedCrash("complete")
            return await super()._complete_orphan_cleanup(claim)

        async def _expire_orphan_cleanup(self, claim: CollectionCleanupClaim) -> bool:
            if claim.collection_name == first_name and failure_stage == "expire":
                raise _InjectedCrash("expire")
            return await super()._expire_orphan_cleanup(claim)

    qdrant = _OperationFailureQdrant(clock=lambda: NOW)
    for name in (first_name, second_name):
        await qdrant.seed_collection(
            CollectionSpec(name, 3, "cosine", ()),
            created_at=NOW - timedelta(hours=25),
        )
    service = _OperationFailureService(
        session_factory=migrated_database.sessions,
        qdrant=qdrant,
        embedding_gateway=_FakeEmbeddingGateway(),
        clock=lambda: NOW,
    )

    with caplog.at_level(logging.WARNING, logger="rag_service.indexing.generation_services"):
        result = await service.reconcile_orphan_collections(
            grace_period=timedelta(hours=24),
            limit=10,
            cursor=None,
        )

    assert second_name in result.deleted
    failure_records = [
        record
        for record in caplog.records
        if record.msg == "cleanup.action.completed"
        and getattr(record, "event", None) == "cleanup.action.completed"
        and getattr(record, "operation", None) == "orphan_cleanup"
        and getattr(record, "phase", None) == "qdrant"
        and getattr(record, "outcome", None) == "failed"
    ]
    assert len(failure_records) == 1
    assert getattr(failure_records[0], "count", None) == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_orphan_cleanup_propagates_cancellation_without_processing_later_candidates(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    settings = _settings(postgres_urls[0])
    _admin, _token = await _create_admin(migrated_database, settings)
    knowledge_base_id, _profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )
    first_generation_id = UUID("00000000-0000-0000-0000-000000000001")
    second_generation_id = UUID("00000000-0000-0000-0000-000000000002")
    first_name = collection_name(knowledge_base_id, first_generation_id)
    second_name = collection_name(knowledge_base_id, second_generation_id)

    class _CancelledDeleteQdrant(FakeQdrantClient):
        async def delete_collection(self, collection: str) -> None:
            if collection == first_name:
                raise asyncio.CancelledError
            await super().delete_collection(collection)

    qdrant = _CancelledDeleteQdrant(clock=lambda: NOW)
    for name in (first_name, second_name):
        await qdrant.seed_collection(
            CollectionSpec(name, 3, "cosine", ()),
            created_at=NOW - timedelta(hours=25),
        )
    service = GenerationService(
        session_factory=migrated_database.sessions,
        qdrant=qdrant,
        embedding_gateway=_FakeEmbeddingGateway(),
        clock=lambda: NOW,
    )

    with pytest.raises(asyncio.CancelledError):
        await service.reconcile_orphan_collections(
            grace_period=timedelta(hours=24),
            limit=10,
            cursor=None,
        )

    assert qdrant.collection_names == (first_name, second_name)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_crashed_orphan_cleanup_is_recovered_after_lease_expiry(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    settings = _settings(postgres_urls[0])
    _admin, _token = await _create_admin(migrated_database, settings)
    knowledge_base_id, _profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )
    generation_id = uuid4()
    name = collection_name(knowledge_base_id, generation_id)
    qdrant = FakeQdrantClient(clock=lambda: NOW)
    await qdrant.seed_collection(
        CollectionSpec(name, 3, "cosine", ()),
        created_at=NOW - timedelta(hours=25),
    )
    hooks, _disable = _hooks("before_orphan_delete")
    clock_now = NOW
    service = GenerationService(
        session_factory=migrated_database.sessions,
        qdrant=qdrant,
        embedding_gateway=_FakeEmbeddingGateway(),
        clock=lambda: clock_now,
        hooks=hooks,
        cleanup_lease_duration=timedelta(minutes=1),
    )

    failed = await service.reconcile_orphan_collections(
        grace_period=timedelta(hours=24),
        limit=10,
        cursor=None,
    )
    assert failed.deleted == ()
    assert len(qdrant.collection_names) == 1
    assert name in qdrant.collection_names

    still_leased = await service.reconcile_orphan_collections(
        grace_period=timedelta(hours=24),
        limit=10,
        cursor=None,
    )
    assert still_leased.deleted == ()
    clock_now = NOW + timedelta(minutes=2)
    recovered = await service.reconcile_orphan_collections(
        grace_period=timedelta(hours=24),
        limit=10,
        cursor=None,
    )
    assert recovered.deleted == (name,)
    assert len(qdrant.collection_names) == 0
    async with migrated_database.sessions() as session:
        tombstone = await session.get(IndexGenerationCleanupClaim, name)
        assert tombstone is not None
        assert tombstone.completed_at == clock_now


@pytest.mark.integration
@pytest.mark.asyncio
async def test_crash_after_qdrant_delete_recovers_claim_even_when_collection_is_absent(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    settings = _settings(postgres_urls[0])
    admin, token = await _create_admin(migrated_database, settings)
    knowledge_base_id, profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )
    key = "generation-cleanup-crash-after-delete"
    generation_id = uuid5(
        UUID("72fb80bb-a22e-57a7-ac94-6b94caa14a5c"),
        f"{admin.key_id}:{knowledge_base_id}:{key}",
    )
    name = collection_name(knowledge_base_id, generation_id)
    replacement_key = "generation-cleanup-crash-after-delete-replacement"
    replacement_generation_id = uuid5(
        UUID("72fb80bb-a22e-57a7-ac94-6b94caa14a5c"),
        f"{admin.key_id}:{knowledge_base_id}:{replacement_key}",
    )
    replacement_name = collection_name(knowledge_base_id, replacement_generation_id)
    qdrant = FakeQdrantClient(clock=lambda: NOW)
    await qdrant.seed_collection(
        CollectionSpec(name, 3, "cosine", ()),
        created_at=NOW - timedelta(hours=25),
    )
    hooks, _disable = _hooks("after_orphan_delete")
    clock_now = NOW
    service = GenerationService(
        session_factory=migrated_database.sessions,
        qdrant=qdrant,
        embedding_gateway=_FakeEmbeddingGateway(),
        clock=lambda: clock_now,
        hooks=hooks,
        cleanup_lease_duration=timedelta(minutes=1),
    )

    failed = await service.reconcile_orphan_collections(
        grace_period=timedelta(hours=24),
        limit=10,
        cursor=None,
    )
    assert failed.deleted == ()
    assert len(qdrant.collection_names) == 0

    app = _app(migrated_database, settings, qdrant, _FakeEmbeddingGateway())
    path = f"/v1/admin/knowledge-bases/{knowledge_base_id}/index-generations"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        blocked = await client.post(
            path,
            headers=_headers(token, "req-cleanup-crash-after-delete", key),
            json={"embedding_profile_id": str(profile_id), "distance": "cosine"},
        )
        assert blocked.status_code == 503
        assert blocked.json()["error"]["code"] == "GENERATION_CLEANUP_IN_PROGRESS"

        clock_now = NOW + timedelta(minutes=2)
        recovered = await service.reconcile_orphan_collections(
            grace_period=timedelta(hours=24),
            limit=10,
            cursor=None,
        )
        assert recovered.deleted == (name,)

        retired = await client.post(
            path,
            headers=_headers(token, "req-cleanup-crash-after-delete-retry", key),
            json={"embedding_profile_id": str(profile_id), "distance": "cosine"},
        )
        created = await client.post(
            path,
            headers=_headers(
                token,
                "req-cleanup-crash-after-delete-replacement",
                replacement_key,
            ),
            json={"embedding_profile_id": str(profile_id), "distance": "cosine"},
        )
    assert retired.status_code == 409
    assert retired.json()["error"]["code"] == "GENERATION_COLLECTION_RETIRED"
    assert created.status_code == 201
    assert len(qdrant.collection_names) == 1
    assert replacement_name in qdrant.collection_names
    rows = await _generation_rows(migrated_database, knowledge_base_id)
    assert len(rows) == 1 and rows[0].id == replacement_generation_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_activation_refuses_a_collection_claimed_after_reservation(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    settings = _settings(postgres_urls[0])
    admin, token = await _create_admin(migrated_database, settings)
    knowledge_base_id, profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )
    key = "generation-activation-cleanup-claim"
    generation_id = uuid5(
        UUID("72fb80bb-a22e-57a7-ac94-6b94caa14a5c"),
        f"{admin.key_id}:{knowledge_base_id}:{key}",
    )
    name = collection_name(knowledge_base_id, generation_id)
    replacement_key = "generation-activation-cleanup-claim-replacement"
    replacement_generation_id = uuid5(
        UUID("72fb80bb-a22e-57a7-ac94-6b94caa14a5c"),
        f"{admin.key_id}:{knowledge_base_id}:{replacement_key}",
    )
    replacement_name = collection_name(knowledge_base_id, replacement_generation_id)
    owner = uuid4()
    claimed_epoch: int | None = None

    async def claim_before_activation() -> None:
        nonlocal claimed_epoch
        async with migrated_database.sessions() as session, session.begin():
            repository = SqlAlchemyGenerationRepository(session)
            await repository.get_knowledge_base_for_update(knowledge_base_id)
            await repository.acquire_collection_fence(name)
            claim = await repository.claim_collection_cleanup(
                collection_name=name,
                knowledge_base_id=knowledge_base_id,
                generation_id=generation_id,
                lease_owner=owner,
                now=NOW,
                lease_duration=timedelta(minutes=1),
            )
            assert claim is not None
            claimed_epoch = claim.lease_epoch

    hooks, _disable = _hooks("after_probe", claim_before_activation)
    qdrant = FakeQdrantClient(clock=lambda: NOW)
    gateway = _FakeEmbeddingGateway()
    app = _app(migrated_database, settings, qdrant, gateway, hooks)
    path = f"/v1/admin/knowledge-bases/{knowledge_base_id}/index-generations"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        blocked = await client.post(
            path,
            headers=_headers(token, "req-generation-activation-claim", key),
            json={"embedding_profile_id": str(profile_id), "distance": "cosine"},
        )
        assert blocked.status_code == 503
        assert blocked.json()["error"]["code"] == "GENERATION_CLEANUP_IN_PROGRESS"
        rows = await _generation_rows(migrated_database, knowledge_base_id)
        assert len(rows) == 1 and rows[0].status == "building"

        assert claimed_epoch is not None
        await qdrant.delete_collection(name)
        async with migrated_database.sessions() as session, session.begin():
            repository = SqlAlchemyGenerationRepository(session)
            await repository.get_knowledge_base_for_update(knowledge_base_id)
            await repository.acquire_collection_fence(name)
            assert await repository.complete_collection_cleanup(
                name,
                lease_owner=owner,
                lease_epoch=claimed_epoch,
                now=NOW + timedelta(seconds=1),
            )

        retired = await client.post(
            path,
            headers=_headers(token, "req-generation-activation-claim-retry", key),
            json={"embedding_profile_id": str(profile_id), "distance": "cosine"},
        )
        assert retired.status_code == 409
        assert retired.json()["error"]["code"] == "GENERATION_COLLECTION_RETIRED"
        rows = await _generation_rows(migrated_database, knowledge_base_id)
        assert len(rows) == 1 and rows[0].status == "failed"
        async with migrated_database.sessions() as session:
            knowledge_base = await session.get(KnowledgeBase, knowledge_base_id)
            assert knowledge_base is not None
            assert knowledge_base.pending_index_generation_id is None

        recovered = await client.post(
            path,
            headers=_headers(
                token,
                "req-generation-activation-claim-replacement",
                replacement_key,
            ),
            json={"embedding_profile_id": str(profile_id), "distance": "cosine"},
        )
    assert recovered.status_code == 201
    rows = await _generation_rows(migrated_database, knowledge_base_id)
    assert {row.id: row.status for row in rows} == {
        generation_id: "failed",
        replacement_generation_id: "active",
    }
    assert qdrant.collection_names == (replacement_name,)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_activation_terminalizes_generation_completed_by_cleanup_after_reservation(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    settings = _settings(postgres_urls[0])
    admin, token = await _create_admin(migrated_database, settings)
    knowledge_base_id, profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )
    key = "generation-activation-completed-cleanup"
    generation_id = uuid5(
        UUID("72fb80bb-a22e-57a7-ac94-6b94caa14a5c"),
        f"{admin.key_id}:{knowledge_base_id}:{key}",
    )
    name = collection_name(knowledge_base_id, generation_id)
    replacement_key = "generation-activation-completed-cleanup-replacement"
    replacement_generation_id = uuid5(
        UUID("72fb80bb-a22e-57a7-ac94-6b94caa14a5c"),
        f"{admin.key_id}:{knowledge_base_id}:{replacement_key}",
    )
    replacement_name = collection_name(knowledge_base_id, replacement_generation_id)
    owner = uuid4()
    qdrant = FakeQdrantClient(clock=lambda: NOW)

    async def complete_before_activation() -> None:
        await qdrant.delete_collection(name)
        async with migrated_database.sessions() as session, session.begin():
            repository = SqlAlchemyGenerationRepository(session)
            await repository.get_knowledge_base_for_update(knowledge_base_id)
            await repository.acquire_collection_fence(name)
            claim = await repository.claim_collection_cleanup(
                collection_name=name,
                knowledge_base_id=knowledge_base_id,
                generation_id=generation_id,
                lease_owner=owner,
                now=NOW,
                lease_duration=timedelta(minutes=1),
            )
            assert claim is not None
            assert await repository.complete_collection_cleanup(
                name,
                lease_owner=owner,
                lease_epoch=claim.lease_epoch,
                now=NOW + timedelta(seconds=1),
            )

    hooks, _disable = _hooks("after_probe", complete_before_activation)
    app = _app(migrated_database, settings, qdrant, _FakeEmbeddingGateway(), hooks)
    path = f"/v1/admin/knowledge-bases/{knowledge_base_id}/index-generations"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        retired = await client.post(
            path,
            headers=_headers(token, "req-generation-activation-completed", key),
            json={"embedding_profile_id": str(profile_id), "distance": "cosine"},
        )
        assert retired.status_code == 409
        assert retired.json()["error"]["code"] == "GENERATION_COLLECTION_RETIRED"
        rows = await _generation_rows(migrated_database, knowledge_base_id)
        assert len(rows) == 1 and rows[0].status == "failed"
        async with migrated_database.sessions() as session:
            knowledge_base = await session.get(KnowledgeBase, knowledge_base_id)
            assert knowledge_base is not None
            assert knowledge_base.pending_index_generation_id is None

        recovered = await client.post(
            path,
            headers=_headers(
                token,
                "req-generation-activation-completed-replacement",
                replacement_key,
            ),
            json={"embedding_profile_id": str(profile_id), "distance": "cosine"},
        )

    assert recovered.status_code == 201
    rows = await _generation_rows(migrated_database, knowledge_base_id)
    assert {row.id: row.status for row in rows} == {
        generation_id: "failed",
        replacement_generation_id: "active",
    }
    assert qdrant.collection_names == (replacement_name,)


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.acceptance
@pytest.mark.parametrize(
    ("seed_options", "expected_code"),
    [
        ({"capability": "chat"}, "INVALID_EMBEDDING_PROFILE"),
        ({"profile_enabled": False}, "MODEL_PROFILE_DISABLED"),
        ({"provider_enabled": False}, "PROVIDER_CONFIG_DISABLED"),
        ({"include_credential": False}, "PROVIDER_CREDENTIAL_UNAVAILABLE"),
    ],
)
async def test_invalid_authoritative_embedding_configuration_fails_without_secret_leak(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
    seed_options: dict[str, object],
    expected_code: str,
) -> None:
    settings = _settings(postgres_urls[0])
    _admin, token = await _create_admin(migrated_database, settings)
    knowledge_base_id, profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database,
        **seed_options,  # type: ignore[arg-type]
    )
    qdrant = FakeQdrantClient(clock=lambda: NOW)
    gateway = _FakeEmbeddingGateway()
    app = _app(migrated_database, settings, qdrant, gateway)
    path = f"/v1/admin/knowledge-bases/{knowledge_base_id}/index-generations"
    request_id = f"req-generation-invalid-{expected_code.lower()}"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.post(
            path,
            headers=_headers(token, request_id, f"generation-invalid-{expected_code}"),
            json={"embedding_profile_id": str(profile_id), "distance": "cosine"},
        )
    _assert_error(response, 422, expected_code, request_id)
    assert SECRET not in response.text
    assert len(await _generation_rows(migrated_database, knowledge_base_id)) == 0
    assert qdrant.collection_names == ()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_abandoning_a_building_generation_unblocks_the_knowledge_base(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    """A knowledge base wedged by a provider outage can be recovered.

    Creating a generation calls the embedding provider inside the request, and a
    retryable failure there leaves the row `building`. Since a `building`
    generation blocks creating another, the knowledge base ends up with no active
    generation and no way to obtain one — retrying with a fresh idempotency key
    returns 409 forever, and only replaying the original key resumes it.
    """
    settings = _settings(postgres_urls[0])
    admin, _token = await _create_admin(migrated_database, settings)
    knowledge_base_id, profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )
    stuck_id = UUID("20000000-0000-4000-8000-000000000001")
    async with migrated_database.sessions() as session, session.begin():
        session.add(
            KnowledgeBaseIndexGeneration(
                id=stuck_id,
                knowledge_base_id=knowledge_base_id,
                embedding_profile_id=profile_id,
                index_profile_hash="3" * 64,
                qdrant_collection_name=collection_name(knowledge_base_id, stuck_id),
                status="building",
                rebuild_snapshot_at=NOW,
                caught_up_revision=0,
                created_at=NOW,
                distance="cosine",
            )
        )
        knowledge_base = await session.get(KnowledgeBase, knowledge_base_id)
        assert knowledge_base is not None
        knowledge_base.pending_index_generation_id = stuck_id

    service = GenerationService(
        session_factory=migrated_database.sessions,
        qdrant=FakeQdrantClient(clock=lambda: NOW),
        embedding_gateway=_FakeEmbeddingGateway(),
        clock=lambda: NOW,
    )

    abandoned = await service.abandon_generation(knowledge_base_id, stuck_id, actor=admin)

    assert abandoned.status == "failed"
    async with migrated_database.sessions() as session:
        knowledge_base = await session.get(KnowledgeBase, knowledge_base_id)
        assert knowledge_base is not None
        # The dangling pointer has to go too, or the knowledge base still looks
        # like it has a generation in flight.
        assert knowledge_base.pending_index_generation_id is None
        row = await session.get(KnowledgeBaseIndexGeneration, stuck_id)
        assert row is not None
        assert row.safe_error_code == "GENERATION_ABANDONED"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_abandon_refuses_an_active_generation_and_a_foreign_one(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    settings = _settings(postgres_urls[0])
    admin, _token = await _create_admin(migrated_database, settings)
    knowledge_base_id, profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )
    active_id = UUID("20000000-0000-4000-8000-000000000002")
    async with migrated_database.sessions() as session, session.begin():
        session.add(
            KnowledgeBaseIndexGeneration(
                id=active_id,
                knowledge_base_id=knowledge_base_id,
                embedding_profile_id=profile_id,
                index_profile_hash="4" * 64,
                qdrant_collection_name=collection_name(knowledge_base_id, active_id),
                status="active",
                rebuild_snapshot_at=NOW,
                caught_up_revision=0,
                created_at=NOW,
                distance="cosine",
                # ck_kb_index_generations_active_validation_complete requires
                # the whole validation record before a row may claim `active`.
                embedding_config_snapshot={"provider_type": "openai_compatible"},
                filter_schema_snapshot={"fields": []},
                applied_filter_schema_revision=0,
                embedding_config_hash="5" * 64,
                validated_revision=0,
                validation_manifest_hash="6" * 64,
                expected_point_count=0,
                actual_point_count=0,
                validated_at=NOW,
                activated_at=NOW,
            )
        )

    service = GenerationService(
        session_factory=migrated_database.sessions,
        qdrant=FakeQdrantClient(clock=lambda: NOW),
        embedding_gateway=_FakeEmbeddingGateway(),
        clock=lambda: NOW,
    )

    # Abandoning an active generation would leave a knowledge base whose
    # documents are indexed but unsearchable.
    with pytest.raises(BusinessError) as active_refused:
        await service.abandon_generation(knowledge_base_id, active_id, actor=admin)
    # A generation id from another knowledge base must not be reachable through
    # this knowledge base's path.
    with pytest.raises(BusinessError) as unknown_refused:
        await service.abandon_generation(knowledge_base_id, uuid4(), actor=admin)

    assert active_refused.value.code == "GENERATION_NOT_ABANDONABLE"
    assert unknown_refused.value.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_purging_a_deleted_knowledge_base_removes_every_row(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    """Deletion has to be true, not just recorded.

    Marking a knowledge base `deleting` used to be the end of it: nothing
    consumed that status, so the vectors, the objects and every row stayed.
    """
    _settings(postgres_urls[0])
    knowledge_base_id, profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )
    generation_id = UUID("30000000-0000-4000-8000-000000000001")
    collection = collection_name(knowledge_base_id, generation_id)
    async with migrated_database.sessions() as session, session.begin():
        session.add(
            KnowledgeBaseIndexGeneration(
                id=generation_id,
                knowledge_base_id=knowledge_base_id,
                embedding_profile_id=profile_id,
                index_profile_hash="7" * 64,
                qdrant_collection_name=collection,
                status="building",
                rebuild_snapshot_at=NOW,
                caught_up_revision=0,
                created_at=NOW,
                distance="cosine",
            )
        )
        knowledge_base = await session.get(KnowledgeBase, knowledge_base_id)
        assert knowledge_base is not None
        knowledge_base.pending_index_generation_id = generation_id
        knowledge_base.status = "deleting"

    deleted_collections: list[str] = []

    class _Index:
        async def delete_collection(self, collection: str) -> None:
            deleted_collections.append(collection)

    class _Store:
        def __init__(self) -> None:
            self.deleted: list[str] = []

        async def list_older_than(
            self,
            *,
            prefix: str,
            older_than: datetime,
            limit: int,
            start_after: str | None = None,
        ) -> Any:
            if start_after is not None:
                return SimpleNamespace(items=(), next_start_after=None)
            return SimpleNamespace(
                items=(
                    SimpleNamespace(object_key=f"{prefix}documents/a/versions/b/parsed/text.txt"),
                ),
                next_start_after=None,
            )

        async def delete_best_effort(self, object_key: str) -> bool:
            self.deleted.append(object_key)
            return True

    store = _Store()
    purge = KnowledgeBasePurge(
        session_factory=migrated_database.sessions,
        object_store=store,
        search_index=_Index(),
    )

    result = await purge.purge(knowledge_base_id)

    assert result.purged is True
    assert deleted_collections == [collection]
    assert store.deleted == [
        f"knowledge-bases/{knowledge_base_id}/documents/a/versions/b/parsed/text.txt"
    ]
    async with migrated_database.sessions() as session:
        assert await session.get(KnowledgeBase, knowledge_base_id) is None
        assert await session.get(KnowledgeBaseIndexGeneration, generation_id) is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_purge_refuses_a_knowledge_base_that_is_not_being_deleted(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    _settings(postgres_urls[0])
    knowledge_base_id, _profile_id, _provider_id, _credential_id = await _seed_configuration(
        migrated_database
    )

    class _Index:
        async def delete_collection(self, collection: str) -> None:
            raise AssertionError("an active knowledge base must not be touched")

    class _Store:
        async def list_older_than(self, **_: object) -> Any:
            raise AssertionError("an active knowledge base must not be touched")

        async def delete_best_effort(self, object_key: str) -> bool:
            raise AssertionError("an active knowledge base must not be touched")

    purge = KnowledgeBasePurge(
        session_factory=migrated_database.sessions,
        object_store=_Store(),
        search_index=_Index(),
    )

    # A purge job could be replayed long after its knowledge base was recreated
    # under the same id, so the status gate is what keeps it from destroying
    # live data.
    result = await purge.purge(knowledge_base_id)

    assert result.purged is False
    async with migrated_database.sessions() as session:
        assert await session.get(KnowledgeBase, knowledge_base_id) is not None
