from datetime import UTC, datetime
from typing import cast
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI
from fastapi.routing import APIRoute
from pydantic import SecretStr
from sqlalchemy import func, select

from rag_service.api.errors import BusinessError
from rag_service.auth.policies import AdminPrincipal, AgentPrincipal, Capability
from rag_service.auth.schemas import AdminApiKeyCreate, AgentApiKeyCreate, SafeApiKey
from rag_service.auth.services import ApiKeyService
from rag_service.config import Settings, get_settings
from rag_service.db.models.auth import AuditEvent
from rag_service.db.models.knowledge_bases import (
    KnowledgeBase,
    KnowledgeBaseIndexGeneration,
    KnowledgeBaseMutation,
)
from rag_service.db.session import Database
from rag_service.main import create_app
from rag_service.metadata.knowledge_base_routes import (
    get_knowledge_base_service,
)
from rag_service.metadata.knowledge_base_routes import (
    router as knowledge_base_router,
)
from rag_service.metadata.schemas import (
    KnowledgeBasePage,
    SafeFilterSchema,
    SafeKnowledgeBase,
)
from rag_service.metadata.services import KnowledgeBaseService

ADMIN_HMAC_SECRET = "kb-api-admin-test-hmac-secret-32-bytes"
AGENT_HMAC_SECRET = "kb-api-agent-test-hmac-secret-32-bytes"


def _settings(database_url: str) -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        database_url=SecretStr(database_url),
        admin_key_hmac_secret=SecretStr(ADMIN_HMAC_SECRET),
        agent_key_hmac_secret=SecretStr(AGENT_HMAC_SECRET),
        default_page_size=2,
        max_page_size=3,
        max_api_key_requests_per_minute=100,
        max_api_key_concurrency=10,
    )


def _app(database: Database, settings: Settings) -> FastAPI:
    app = create_app()
    app.state.database = database
    app.dependency_overrides[get_settings] = lambda: settings
    return app


async def _create_admin(
    database: Database,
    settings: Settings,
) -> tuple[AdminPrincipal, str]:
    async with database.sessions() as session:
        issued = await ApiKeyService(
            session=session,
            authentication_sessions=database.sessions,
            settings=settings,
        ).create_admin_key(
            AdminApiKeyCreate(name="kb-api-admin"),
            request_id=f"req-kb-api-admin-{uuid4().hex}",
        )
    return (
        AdminPrincipal(
            key_id=issued.api_key.id,
            public_id=issued.api_key.public_id,
        ),
        issued.token.get_secret_value(),
    )


async def _create_agent(
    database: Database,
    settings: Settings,
    admin: AdminPrincipal,
    *,
    name: str,
    capabilities: frozenset[Capability],
) -> tuple[SafeApiKey, str]:
    async with database.sessions() as session:
        issued = await ApiKeyService(
            session=session,
            authentication_sessions=database.sessions,
            settings=settings,
        ).create_agent_key(
            AgentApiKeyCreate(
                name=name,
                capabilities=capabilities,
                knowledge_base_ids=frozenset(),
                query_profile_ids=frozenset(),
                default_query_profile_id=None,
                raw_file_read=False,
                requests_per_minute=60,
                max_concurrency=4,
            ),
            actor=admin,
            request_id=f"req-{name}-create",
        )
    return issued.api_key, issued.token.get_secret_value()


def _headers(
    token: str,
    request_id: str,
    **headers: str,
) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "X-Request-ID": request_id,
        **headers,
    }


def _assert_error(
    response: httpx.Response,
    status_code: int,
    code: str,
    request_id: str,
) -> dict[str, object]:
    assert response.status_code == status_code
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-request-id"] == request_id
    document = response.json()
    assert set(document) == {"error"}
    error = document["error"]
    assert isinstance(error, dict)
    assert set(error) == {"code", "message", "retryable", "request_id"}
    assert error["code"] == code
    assert error["retryable"] is False
    assert error["request_id"] == request_id
    assert "detail" not in document
    return cast(dict[str, object], error)


async def _knowledge_base_record(database: Database, identifier: UUID) -> KnowledgeBase:
    async with database.sessions() as session:
        row = await session.get(KnowledgeBase, identifier)
        assert row is not None
        return row


async def _deletion_audit_count(database: Database, identifier: UUID) -> int:
    async with database.sessions() as session:
        return cast(
            int,
            await session.scalar(
                select(func.count())
                .select_from(AuditEvent)
                .where(
                    AuditEvent.target_id == identifier,
                    AuditEvent.action == "knowledge_base.deletion_requested",
                )
            ),
        )


async def _filter_write_counts(database: Database, identifier: UUID) -> tuple[int, int]:
    async with database.sessions() as session:
        audit_count = cast(
            int,
            await session.scalar(
                select(func.count())
                .select_from(AuditEvent)
                .where(
                    AuditEvent.target_id == identifier,
                    AuditEvent.action == "knowledge_base.filter_schema_replaced",
                )
            ),
        )
        mutation_count = cast(
            int,
            await session.scalar(
                select(func.count())
                .select_from(KnowledgeBaseMutation)
                .where(KnowledgeBaseMutation.knowledge_base_id == identifier)
            ),
        )
        return audit_count, mutation_count


class _UnexpectedFailureService:
    async def list_knowledge_bases(
        self,
        *,
        actor: AgentPrincipal,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> KnowledgeBasePage:
        del actor, cursor, limit
        raise RuntimeError("task14-global-error-boundary")


class _BusinessHeaderFailureService:
    async def list_knowledge_bases(
        self,
        *,
        actor: AgentPrincipal,
        cursor: str | None = None,
        limit: int | None = None,
    ) -> KnowledgeBasePage:
        del actor, cursor, limit
        raise BusinessError(
            401,
            "INVALID_API_KEY",
            "Invalid API key",
            headers={
                "WWW-Authenticate": "Bearer",
                "Cache-Control": "public, max-age=3600",
                "X-Request-ID": "unsafe-overridden-request-id",
                "X-Internal-Secret": "unsafe-header-secret",
            },
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_can_create_replay_update_and_idempotently_delete_a_knowledge_base(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    admin, _admin_token = await _create_admin(migrated_database, settings)
    _agent, agent_token = await _create_agent(
        migrated_database,
        settings,
        admin,
        name="kb-api-owner",
        capabilities=frozenset({Capability.MANAGE}),
    )
    app = _app(migrated_database, settings)
    command = {
        "name": "Product manuals",
        "description": "Support documentation",
        "metadata": {"owner": "support"},
    }

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        missing_key_id = "req-kb-create-missing-key"
        missing_key = await client.post(
            "/v1/knowledge-bases",
            headers=_headers(agent_token, missing_key_id),
            json=command,
        )
        _assert_error(missing_key, 422, "VALIDATION_ERROR", missing_key_id)

        create_request_id = "req-kb-create-first"
        created = await client.post(
            "/v1/knowledge-bases",
            headers=_headers(
                agent_token,
                create_request_id,
                **{"Idempotency-Key": "kb-http-create-1"},
            ),
            json=command,
        )
        assert created.status_code == 201
        assert created.headers["cache-control"] == "no-store"
        assert created.headers["x-request-id"] == create_request_id
        document = created.json()
        assert set(document) == set(SafeKnowledgeBase.model_fields)
        knowledge_base = SafeKnowledgeBase.model_validate_json(created.text)
        first_etag = created.headers["etag"]
        assert knowledge_base.etag == first_etag
        assert created.headers["location"] == f"/v1/knowledge-bases/{knowledge_base.id}"

        replay_request_id = "req-kb-create-replay"
        replay = await client.post(
            "/v1/knowledge-bases",
            headers=_headers(
                agent_token,
                replay_request_id,
                **{"Idempotency-Key": "kb-http-create-1"},
            ),
            json=command,
        )
        assert replay.status_code == 200
        assert replay.headers["etag"] == first_etag
        assert replay.headers["location"] == f"/v1/knowledge-bases/{knowledge_base.id}"
        assert replay.headers["x-request-id"] == replay_request_id
        assert replay.json() == document

        conflict_request_id = "req-kb-create-conflict"
        conflict = await client.post(
            "/v1/knowledge-bases",
            headers=_headers(
                agent_token,
                conflict_request_id,
                **{"Idempotency-Key": "kb-http-create-1"},
            ),
            json={**command, "name": "Different request"},
        )
        _assert_error(conflict, 409, "IDEMPOTENCY_CONFLICT", conflict_request_id)

        detail = await client.get(
            f"/v1/knowledge-bases/{knowledge_base.id}",
            headers=_headers(agent_token, "req-kb-detail-active"),
        )
        assert detail.status_code == 200
        assert detail.headers["etag"] == detail.json()["etag"] == first_etag

        missing_match_id = "req-kb-patch-missing-match"
        missing_match = await client.patch(
            f"/v1/knowledge-bases/{knowledge_base.id}",
            headers=_headers(agent_token, missing_match_id),
            json={"name": "Must not apply"},
        )
        _assert_error(missing_match, 412, "PRECONDITION_FAILED", missing_match_id)

        stale_match_id = "req-kb-patch-stale-match"
        stale_match = await client.patch(
            f"/v1/knowledge-bases/{knowledge_base.id}",
            headers=_headers(
                agent_token,
                stale_match_id,
                **{"If-Match": '"kb:00000000-0000-0000-0000-000000000000:r1"'},
            ),
            json={"name": "Must not apply"},
        )
        _assert_error(stale_match, 412, "PRECONDITION_FAILED", stale_match_id)

        update_request_id = "req-kb-patch-current"
        updated = await client.patch(
            f"/v1/knowledge-bases/{knowledge_base.id}",
            headers=_headers(
                agent_token,
                update_request_id,
                **{"If-Match": first_etag},
            ),
            json={"name": "Updated manuals", "metadata": {"owner": "docs"}},
        )
        assert updated.status_code == 200
        assert updated.headers["x-request-id"] == update_request_id
        updated_document = SafeKnowledgeBase.model_validate_json(updated.text)
        second_etag = updated.headers["etag"]
        assert updated_document.etag == second_etag
        assert updated_document.name == "Updated manuals"
        assert second_etag != first_etag

        stale_delete_id = "req-kb-delete-stale"
        stale_delete = await client.delete(
            f"/v1/knowledge-bases/{knowledge_base.id}",
            headers=_headers(
                agent_token,
                stale_delete_id,
                **{"If-Match": first_etag},
            ),
        )
        _assert_error(stale_delete, 412, "PRECONDITION_FAILED", stale_delete_id)

        delete_request_id = "req-kb-delete-current"
        deleted = await client.delete(
            f"/v1/knowledge-bases/{knowledge_base.id}",
            headers=_headers(
                agent_token,
                delete_request_id,
                **{"If-Match": second_etag},
            ),
        )
        assert deleted.status_code == 204
        assert deleted.content == b""
        assert "content-type" not in deleted.headers
        assert "content-length" not in deleted.headers
        assert deleted.headers["cache-control"] == "no-store"
        assert deleted.headers["x-request-id"] == delete_request_id
        deleting_etag = deleted.headers["etag"]
        assert deleting_etag != second_etag

        listed = await client.get(
            "/v1/knowledge-bases",
            headers=_headers(agent_token, "req-kb-list-after-delete"),
        )
        assert listed.status_code == 200
        assert listed.json() == {"items": [], "next_cursor": None}

        deleting_detail = await client.get(
            f"/v1/knowledge-bases/{knowledge_base.id}",
            headers=_headers(agent_token, "req-kb-detail-deleting"),
        )
        assert deleting_detail.status_code == 200
        assert deleting_detail.headers["etag"] == deleting_etag
        assert deleting_detail.json()["status"] == "deleting"

        deleting_patch_id = "req-kb-patch-deleting"
        deleting_patch = await client.patch(
            f"/v1/knowledge-bases/{knowledge_base.id}",
            headers=_headers(
                agent_token,
                deleting_patch_id,
                **{"If-Match": deleting_etag},
            ),
            json={"name": "Must not revive"},
        )
        _assert_error(
            deleting_patch,
            409,
            "RESOURCE_STATE_CONFLICT",
            deleting_patch_id,
        )

        repeated_request_id = "req-kb-delete-repeated"
        repeated = await client.delete(
            f"/v1/knowledge-bases/{knowledge_base.id}",
            headers=_headers(
                agent_token,
                repeated_request_id,
                **{"If-Match": deleting_etag},
            ),
        )
        assert repeated.status_code == 204
        assert repeated.content == b""
        assert "content-type" not in repeated.headers
        assert "content-length" not in repeated.headers
        assert repeated.headers["etag"] == deleting_etag
        assert repeated.headers["x-request-id"] == repeated_request_id

    persisted = await _knowledge_base_record(migrated_database, knowledge_base.id)
    assert persisted.status == "deleting"
    assert persisted.resource_revision == 3
    assert await _deletion_audit_count(migrated_database, knowledge_base.id) == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_filter_schema_put_is_safe_conditional_scoped_and_rejects_deleting_resources(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    admin, _admin_token = await _create_admin(migrated_database, settings)
    _owner, owner_token = await _create_agent(
        migrated_database,
        settings,
        admin,
        name="kb-api-filter-owner",
        capabilities=frozenset({Capability.MANAGE}),
    )
    _outsider, outsider_token = await _create_agent(
        migrated_database,
        settings,
        admin,
        name="kb-api-filter-outsider",
        capabilities=frozenset({Capability.MANAGE}),
    )
    _reader, reader_token = await _create_agent(
        migrated_database,
        settings,
        admin,
        name="kb-api-filter-reader",
        capabilities=frozenset({Capability.RETRIEVE}),
    )
    app = _app(migrated_database, settings)
    command = {
        "fields": [
            {
                "name": "department",
                "source_path": "attributes.department",
                "type": "keyword",
                "operators": ["in", "eq"],
            }
        ]
    }

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        created = await client.post(
            "/v1/knowledge-bases",
            headers=_headers(
                owner_token,
                "req-kb-filter-create",
                **{"Idempotency-Key": "kb-filter-create"},
            ),
            json={"name": "Filter API knowledge base"},
        )
        assert created.status_code == 201
        knowledge_base = SafeKnowledgeBase.model_validate_json(created.text)

        missing_match_id = "req-kb-filter-missing-match"
        missing_match = await client.put(
            f"/v1/knowledge-bases/{knowledge_base.id}/filter-schema",
            headers=_headers(owner_token, missing_match_id),
            json=command,
        )
        _assert_error(missing_match, 412, "PRECONDITION_FAILED", missing_match_id)

        stale_match_id = "req-kb-filter-stale-match"
        stale_match = await client.put(
            f"/v1/knowledge-bases/{knowledge_base.id}/filter-schema",
            headers=_headers(
                owner_token,
                stale_match_id,
                **{"If-Match": '"kb:00000000-0000-0000-0000-000000000000:r1"'},
            ),
            json=command,
        )
        _assert_error(stale_match, 412, "PRECONDITION_FAILED", stale_match_id)

        hidden_request_id = "req-kb-filter-hidden"
        hidden = await client.put(
            f"/v1/knowledge-bases/{knowledge_base.id}/filter-schema",
            headers=_headers(
                outsider_token,
                hidden_request_id,
                **{"If-Match": knowledge_base.etag},
            ),
            json=command,
        )
        _assert_error(hidden, 404, "RESOURCE_NOT_FOUND", hidden_request_id)

        capability_request_id = "req-kb-filter-capability"
        capability_denied = await client.put(
            f"/v1/knowledge-bases/{knowledge_base.id}/filter-schema",
            headers=_headers(
                reader_token,
                capability_request_id,
                **{"If-Match": knowledge_base.etag},
            ),
            json=command,
        )
        _assert_error(
            capability_denied,
            403,
            "INSUFFICIENT_CAPABILITY",
            capability_request_id,
        )

        validation_request_id = "req-kb-filter-validation"
        invalid = await client.put(
            f"/v1/knowledge-bases/{knowledge_base.id}/filter-schema",
            headers=_headers(
                owner_token,
                validation_request_id,
                **{"If-Match": knowledge_base.etag},
            ),
            json={
                **command,
                "field_id": "fld_must-never-be-client-controlled",
                "credential": "must-not-leak-filter-secret",
            },
        )
        _assert_error(invalid, 422, "VALIDATION_ERROR", validation_request_id)
        assert "must-not-leak-filter-secret" not in invalid.text

        replace_request_id = "req-kb-filter-replace"
        replaced = await client.put(
            f"/v1/knowledge-bases/{knowledge_base.id}/filter-schema",
            headers=_headers(
                owner_token,
                replace_request_id,
                **{"If-Match": knowledge_base.etag},
            ),
            json=command,
        )
        assert replaced.status_code == 200
        assert replaced.headers["cache-control"] == "no-store"
        assert replaced.headers["x-request-id"] == replace_request_id
        result = SafeFilterSchema.model_validate_json(replaced.text)
        assert replaced.headers["etag"] == result.etag
        assert (
            result.resource_revision,
            result.mutation_revision,
            result.filter_schema_revision,
        ) == (2, 1, 1)
        assert result.fields[0].operators == ("eq", "in")
        assert set(replaced.json()) == set(SafeFilterSchema.model_fields)
        assert "field_id" not in replaced.text
        assert "payload_path" not in replaced.text

        old_etag_id = "req-kb-filter-old-etag"
        old_etag = await client.put(
            f"/v1/knowledge-bases/{knowledge_base.id}/filter-schema",
            headers=_headers(
                owner_token,
                old_etag_id,
                **{"If-Match": knowledge_base.etag},
            ),
            json=command,
        )
        _assert_error(old_etag, 412, "PRECONDITION_FAILED", old_etag_id)

        deleted = await client.delete(
            f"/v1/knowledge-bases/{knowledge_base.id}",
            headers=_headers(
                owner_token,
                "req-kb-filter-delete",
                **{"If-Match": result.etag},
            ),
        )
        assert deleted.status_code == 204
        deleting_etag = deleted.headers["etag"]

        deleting_request_id = "req-kb-filter-deleting"
        deleting = await client.put(
            f"/v1/knowledge-bases/{knowledge_base.id}/filter-schema",
            headers=_headers(
                owner_token,
                deleting_request_id,
                **{"If-Match": deleting_etag},
            ),
            json=command,
        )
        _assert_error(
            deleting,
            409,
            "RESOURCE_STATE_CONFLICT",
            deleting_request_id,
        )


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("generation_status", ("active", "building"))
async def test_filter_schema_noop_is_stable_and_semantic_change_requires_reindex_required(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
    generation_status: str,
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    admin, _admin_token = await _create_admin(migrated_database, settings)
    _owner, owner_token = await _create_agent(
        migrated_database,
        settings,
        admin,
        name=f"kb-filter-freeze-{generation_status}",
        capabilities=frozenset({Capability.MANAGE}),
    )
    app = _app(migrated_database, settings)
    initial_schema = {
        "fields": [
            {
                "name": "department",
                "source_path": "attributes.department",
                "type": "keyword",
                "operators": ["eq", "in"],
            }
        ]
    }

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        created = await client.post(
            "/v1/knowledge-bases",
            headers=_headers(
                owner_token,
                f"req-filter-freeze-create-{generation_status}",
                **{"Idempotency-Key": f"filter-freeze-create-{generation_status}"},
            ),
            json={"name": f"Filter freeze {generation_status}"},
        )
        assert created.status_code == 201
        knowledge_base = SafeKnowledgeBase.model_validate_json(created.text)
        initial = await client.put(
            f"/v1/knowledge-bases/{knowledge_base.id}/filter-schema",
            headers=_headers(
                owner_token,
                f"req-filter-freeze-initial-{generation_status}",
                **{"If-Match": knowledge_base.etag},
            ),
            json=initial_schema,
        )
        assert initial.status_code == 200
        initial_document = SafeFilterSchema.model_validate_json(initial.text)
        baseline_counts = await _filter_write_counts(migrated_database, knowledge_base.id)

        generation_id = uuid4()
        now = datetime.now(UTC)
        active_fields: dict[str, object] = {}
        if generation_status == "active":
            active_fields = {
                "validated_revision": 0,
                "validation_manifest_hash": "b" * 64,
                "expected_point_count": 0,
                "actual_point_count": 0,
                "validated_at": now,
                "activated_at": now,
                "distance": "cosine",
                "embedding_config_snapshot": {},
                "filter_schema_snapshot": initial_document.model_dump(mode="json"),
                "applied_filter_schema_revision": 1,
                "embedding_config_hash": "c" * 64,
            }
        async with migrated_database.sessions() as session, session.begin():
            session.add(
                KnowledgeBaseIndexGeneration(
                    id=generation_id,
                    knowledge_base_id=knowledge_base.id,
                    embedding_profile_id=None,
                    sparse_profile_id=None,
                    index_profile_hash="a" * 64,
                    qdrant_collection_name=f"filter_freeze_{generation_id.hex}",
                    status=generation_status,
                    rebuild_snapshot_at=now,
                    caught_up_revision=0,
                    **active_fields,
                )
            )

        no_op = await client.put(
            f"/v1/knowledge-bases/{knowledge_base.id}/filter-schema",
            headers=_headers(
                owner_token,
                f"req-filter-freeze-noop-{generation_status}",
                **{"If-Match": initial.headers["etag"]},
            ),
            json={
                "fields": [
                    {
                        "name": "department",
                        "source_path": "attributes.department",
                        "type": "keyword",
                        "operators": ["in", "eq"],
                    }
                ]
            },
        )
        assert no_op.status_code == 200
        assert no_op.headers["etag"] == initial.headers["etag"]
        assert no_op.json() == initial.json()
        assert await _filter_write_counts(migrated_database, knowledge_base.id) == baseline_counts

        blocked = await client.put(
            f"/v1/knowledge-bases/{knowledge_base.id}/filter-schema",
            headers=_headers(
                owner_token,
                f"req-filter-freeze-change-{generation_status}",
                **{"If-Match": initial.headers["etag"]},
            ),
            json={
                "fields": [
                    {
                        "name": "department",
                        "source_path": "attributes.department",
                        "type": "keyword",
                        "operators": ["eq"],
                    }
                ]
            },
        )
        _assert_error(
            blocked,
            409,
            "REINDEX_REQUIRED",
            f"req-filter-freeze-change-{generation_status}",
        )

    persisted = await _knowledge_base_record(migrated_database, knowledge_base.id)
    assert (
        persisted.resource_revision,
        persisted.mutation_revision,
        persisted.filter_schema_revision,
    ) == (2, 1, 1)
    assert await _filter_write_counts(migrated_database, knowledge_base.id) == baseline_counts


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize("generation_status", ("failed", "retiring"))
async def test_filter_schema_change_does_not_require_reindex_required_for_inactive_generation(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
    generation_status: str,
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    admin, _admin_token = await _create_admin(migrated_database, settings)
    _owner, owner_token = await _create_agent(
        migrated_database,
        settings,
        admin,
        name=f"kb-filter-unfrozen-{generation_status}",
        capabilities=frozenset({Capability.MANAGE}),
    )
    app = _app(migrated_database, settings)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        created = await client.post(
            "/v1/knowledge-bases",
            headers=_headers(
                owner_token,
                f"req-filter-unfrozen-create-{generation_status}",
                **{"Idempotency-Key": f"filter-unfrozen-create-{generation_status}"},
            ),
            json={"name": f"Filter unfrozen {generation_status}"},
        )
        assert created.status_code == 201
        knowledge_base = SafeKnowledgeBase.model_validate_json(created.text)
        initial = await client.put(
            f"/v1/knowledge-bases/{knowledge_base.id}/filter-schema",
            headers=_headers(
                owner_token,
                f"req-filter-unfrozen-initial-{generation_status}",
                **{"If-Match": knowledge_base.etag},
            ),
            json={"fields": []},
        )
        assert initial.status_code == 200
        baseline_counts = await _filter_write_counts(migrated_database, knowledge_base.id)

        generation_id = uuid4()
        async with migrated_database.sessions() as session, session.begin():
            session.add(
                KnowledgeBaseIndexGeneration(
                    id=generation_id,
                    knowledge_base_id=knowledge_base.id,
                    embedding_profile_id=None,
                    sparse_profile_id=None,
                    index_profile_hash="d" * 64,
                    qdrant_collection_name=f"filter_unfrozen_{generation_id.hex}",
                    status=generation_status,
                    rebuild_snapshot_at=datetime.now(UTC),
                    caught_up_revision=0,
                )
            )

        changed = await client.put(
            f"/v1/knowledge-bases/{knowledge_base.id}/filter-schema",
            headers=_headers(
                owner_token,
                f"req-filter-unfrozen-change-{generation_status}",
                **{"If-Match": initial.headers["etag"]},
            ),
            json={
                "fields": [
                    {
                        "name": "department",
                        "source_path": "attributes.department",
                        "type": "keyword",
                        "operators": ["eq"],
                    }
                ]
            },
        )
        assert changed.status_code == 200
        changed_document = SafeFilterSchema.model_validate_json(changed.text)
        assert (
            changed_document.resource_revision,
            changed_document.mutation_revision,
            changed_document.filter_schema_revision,
        ) == (3, 2, 2)

    assert await _filter_write_counts(migrated_database, knowledge_base.id) == (
        baseline_counts[0] + 1,
        baseline_counts[1] + 1,
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_is_scoped_deterministic_paginated_and_validates_all_inputs_safely(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    admin, _admin_token = await _create_admin(migrated_database, settings)
    _agent, agent_token = await _create_agent(
        migrated_database,
        settings,
        admin,
        name="kb-api-pagination",
        capabilities=frozenset({Capability.MANAGE}),
    )
    app = _app(migrated_database, settings)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        created: list[dict[str, object]] = []
        for index in range(3):
            response = await client.post(
                "/v1/knowledge-bases",
                headers=_headers(
                    agent_token,
                    f"req-kb-page-create-{index}",
                    **{"Idempotency-Key": f"kb-page-create-{index}"},
                ),
                json={"name": f"Page knowledge base {index}"},
            )
            assert response.status_code == 201
            created.append(response.json())

        collected: list[SafeKnowledgeBase] = []
        cursor: str | None = None
        while True:
            response = await client.get(
                "/v1/knowledge-bases",
                params={"limit": 1, **({"cursor": cursor} if cursor else {})},
                headers=_headers(agent_token, f"req-kb-page-{len(collected)}"),
            )
            assert response.status_code == 200
            page = KnowledgeBasePage.model_validate_json(response.text)
            assert len(page.items) == 1
            assert page.items[0].etag == (
                f'"kb:{page.items[0].id}:r{page.items[0].resource_revision}"'
            )
            collected.extend(page.items)
            cursor = page.next_cursor
            if cursor is None:
                break

        expected = sorted(
            created,
            key=lambda item: (
                datetime.fromisoformat(cast(str, item["created_at"])),
                UUID(cast(str, item["id"])),
            ),
        )
        assert [str(item.id) for item in collected] == [item["id"] for item in expected]

        invalid_queries = (
            ({"limit": "0"}, "req-kb-limit-zero"),
            ({"limit": "4"}, "req-kb-limit-over-config"),
            ({"limit": "not-an-int"}, "req-kb-limit-type"),
            ({"cursor": "not-a-canonical-cursor"}, "req-kb-cursor-invalid"),
        )
        for params, request_id in invalid_queries:
            invalid = await client.get(
                "/v1/knowledge-bases",
                params=params,
                headers=_headers(agent_token, request_id),
            )
            _assert_error(invalid, 422, "VALIDATION_ERROR", request_id)

        invalid_uuid_id = "req-kb-uuid-invalid"
        invalid_uuid = await client.get(
            "/v1/knowledge-bases/not-a-uuid",
            headers=_headers(agent_token, invalid_uuid_id),
        )
        _assert_error(invalid_uuid, 422, "VALIDATION_ERROR", invalid_uuid_id)

        secret = "must-not-leak-body-credential"
        body_cases: tuple[
            tuple[str, str, dict[str, str], dict[str, str] | None, bytes | None], ...
        ] = (
            (
                "POST",
                "/v1/knowledge-bases",
                {"Idempotency-Key": "kb-invalid-body"},
                {"name": "Safe name", "credential": secret},
                None,
            ),
            (
                "PATCH",
                f"/v1/knowledge-bases/{collected[0].id}",
                {"If-Match": collected[0].etag},
                {},
                None,
            ),
            (
                "POST",
                "/v1/knowledge-bases",
                {"Idempotency-Key": "kb-malformed-json"},
                None,
                b'{"name":',
            ),
        )
        for index, (method, path, extra_headers, body, content) in enumerate(body_cases):
            request_id = f"req-kb-body-invalid-{index}"
            invalid = await client.request(
                method,
                path,
                headers={
                    **_headers(agent_token, request_id),
                    **extra_headers,
                    "Content-Type": "application/json",
                },
                json=body if content is None else None,
                content=content,
            )
            _assert_error(invalid, 422, "VALIDATION_ERROR", request_id)
            assert secret not in invalid.text
            assert "RequestValidationError" not in invalid.text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_auth_capability_and_scope_failures_preserve_resource_hiding(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    admin, admin_token = await _create_admin(migrated_database, settings)
    _owner, owner_token = await _create_agent(
        migrated_database,
        settings,
        admin,
        name="kb-api-scope-owner",
        capabilities=frozenset({Capability.MANAGE}),
    )
    _outsider, outsider_token = await _create_agent(
        migrated_database,
        settings,
        admin,
        name="kb-api-scope-outsider",
        capabilities=frozenset({Capability.MANAGE}),
    )
    _reader, reader_token = await _create_agent(
        migrated_database,
        settings,
        admin,
        name="kb-api-reader",
        capabilities=frozenset({Capability.RETRIEVE}),
    )
    app = _app(migrated_database, settings)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        unauthorized_cases = (
            ({"X-Request-ID": "req-kb-no-auth"}, "req-kb-no-auth"),
            (_headers(admin_token, "req-kb-admin-key"), "req-kb-admin-key"),
            (_headers("rag_agent_invalid", "req-kb-invalid-key"), "req-kb-invalid-key"),
        )
        for headers, request_id in unauthorized_cases:
            denied = await client.get("/v1/knowledge-bases", headers=headers)
            _assert_error(denied, 401, "INVALID_API_KEY", request_id)
            assert denied.headers["www-authenticate"] == "Bearer"

        capability_request_id = "req-kb-capability-denied"
        capability_denied = await client.get(
            "/v1/knowledge-bases",
            headers=_headers(reader_token, capability_request_id),
        )
        _assert_error(
            capability_denied,
            403,
            "INSUFFICIENT_CAPABILITY",
            capability_request_id,
        )

        created = await client.post(
            "/v1/knowledge-bases",
            headers=_headers(
                owner_token,
                "req-kb-private-create",
                **{"Idempotency-Key": "kb-private-create"},
            ),
            json={"name": "Private knowledge base"},
        )
        assert created.status_code == 201
        private = SafeKnowledgeBase.model_validate_json(created.text)

        outsider_list = await client.get(
            "/v1/knowledge-bases",
            headers=_headers(outsider_token, "req-kb-outsider-list"),
        )
        assert outsider_list.status_code == 200
        assert outsider_list.json() == {"items": [], "next_cursor": None}

        hidden_request_id = "req-kb-hidden"
        out_of_scope = await client.get(
            f"/v1/knowledge-bases/{private.id}",
            headers=_headers(outsider_token, hidden_request_id),
        )
        missing = await client.get(
            f"/v1/knowledge-bases/{uuid4()}",
            headers=_headers(outsider_token, hidden_request_id),
        )
        _assert_error(out_of_scope, 404, "RESOURCE_NOT_FOUND", hidden_request_id)
        _assert_error(missing, 404, "RESOURCE_NOT_FOUND", hidden_request_id)
        assert out_of_scope.json() == missing.json()

        hidden_mutations = (
            ("PATCH", {"If-Match": private.etag}, {"name": "Enumeration attempt"}),
            ("DELETE", {"If-Match": private.etag}, None),
        )
        for method, extra_headers, body in hidden_mutations:
            hidden = await client.request(
                method,
                f"/v1/knowledge-bases/{private.id}",
                headers={
                    **_headers(outsider_token, hidden_request_id),
                    **extra_headers,
                },
                json=body,
            )
            _assert_error(hidden, 404, "RESOURCE_NOT_FOUND", hidden_request_id)


@pytest.mark.integration
def test_knowledge_base_router_uses_the_standard_api_route() -> None:
    assert knowledge_base_router.route_class is APIRoute


@pytest.mark.integration
@pytest.mark.asyncio
async def test_unexpected_runtime_error_uses_the_global_safe_error_contract(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    admin, _admin_token = await _create_admin(migrated_database, settings)
    _agent, agent_token = await _create_agent(
        migrated_database,
        settings,
        admin,
        name="kb-api-global-error-boundary",
        capabilities=frozenset({Capability.MANAGE}),
    )
    app = _app(migrated_database, settings)
    app.dependency_overrides[get_knowledge_base_service] = lambda: cast(
        KnowledgeBaseService,
        _UnexpectedFailureService(),
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=True),
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/v1/knowledge-bases",
            headers=_headers(agent_token, "req-kb-global-boundary"),
        )

    _assert_error(response, 500, "INTERNAL_ERROR", "req-kb-global-boundary")
    assert response.json()["error"]["message"] == "Internal server error"
    assert "task14-global-error-boundary" not in response.text


@pytest.mark.integration
@pytest.mark.asyncio
async def test_settings_dependency_is_resolved_once_by_fastapi(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    admin, _admin_token = await _create_admin(migrated_database, settings)
    _agent, agent_token = await _create_agent(
        migrated_database,
        settings,
        admin,
        name="kb-api-settings-dependency",
        capabilities=frozenset({Capability.MANAGE}),
    )
    calls = 0

    def settings_override() -> Settings:
        nonlocal calls
        calls += 1
        return settings

    app = create_app()
    app.state.database = migrated_database
    app.dependency_overrides[get_settings] = settings_override
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/v1/knowledge-bases",
            headers=_headers(agent_token, "req-kb-settings-dependency"),
        )

    assert response.status_code == 200
    assert calls == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_business_error_headers_are_allowlisted_and_cannot_override_safe_headers(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    admin, _admin_token = await _create_admin(migrated_database, settings)
    _agent, agent_token = await _create_agent(
        migrated_database,
        settings,
        admin,
        name="kb-api-safe-error-headers",
        capabilities=frozenset({Capability.MANAGE}),
    )
    app = _app(migrated_database, settings)
    app.dependency_overrides[get_knowledge_base_service] = lambda: cast(
        KnowledgeBaseService,
        _BusinessHeaderFailureService(),
    )
    request_id = "req-kb-safe-error-headers"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get(
            "/v1/knowledge-bases",
            headers=_headers(agent_token, request_id),
        )

    _assert_error(response, 401, "INVALID_API_KEY", request_id)
    assert response.headers["www-authenticate"] == "Bearer"
    assert "x-internal-secret" not in response.headers
    assert "unsafe-header-secret" not in response.text


@pytest.mark.integration
def test_knowledge_base_openapi_declares_bearer_and_nonempty_success_error_schemas() -> None:
    document = create_app().openapi()
    operations = {
        ("/v1/knowledge-bases", "post"): {
            "200",
            "201",
            "401",
            "403",
            "404",
            "409",
            "422",
        },
        ("/v1/knowledge-bases", "get"): {"200", "401", "403", "422"},
        ("/v1/knowledge-bases/{knowledge_base_id}", "get"): {
            "200",
            "401",
            "403",
            "404",
            "422",
        },
        ("/v1/knowledge-bases/{knowledge_base_id}", "patch"): {
            "200",
            "401",
            "403",
            "404",
            "409",
            "412",
            "422",
        },
        ("/v1/knowledge-bases/{knowledge_base_id}", "delete"): {
            "204",
            "401",
            "403",
            "404",
            "412",
            "422",
        },
        ("/v1/knowledge-bases/{knowledge_base_id}/filter-schema", "put"): {
            "200",
            "401",
            "403",
            "404",
            "409",
            "412",
            "422",
        },
    }

    for (path, method), expected_statuses in operations.items():
        operation = document["paths"][path][method]
        assert operation["security"] == [{"HTTPBearer": []}]
        assert expected_statuses <= set(operation["responses"])
        for status_code in expected_statuses - {"204"}:
            schema = operation["responses"][status_code]["content"]["application/json"]["schema"]
            assert schema != {}
            assert "$ref" in schema
        if "204" in expected_statuses:
            assert "content" not in operation["responses"]["204"]

    schemas = document["components"]["schemas"]
    assert set(SafeKnowledgeBase.model_fields) <= set(schemas["SafeKnowledgeBase"]["properties"])
    assert set(SafeFilterSchema.model_fields) == set(schemas["SafeFilterSchema"]["properties"])
    assert "field_id" not in schemas["SafeFilterSchema"]["properties"]
    assert "payload_path" not in schemas["SafeFilterSchema"]["properties"]
    assert set(KnowledgeBasePage.model_fields) <= set(schemas["KnowledgeBasePage"]["properties"])
    assert set(schemas["ErrorEnvelope"]["properties"]) == {"error"}
