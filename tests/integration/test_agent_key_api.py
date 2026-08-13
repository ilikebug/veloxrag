import json
from dataclasses import FrozenInstanceError
from typing import cast
from uuid import UUID

import httpx
import pytest
from fastapi import FastAPI
from pydantic import SecretStr
from sqlalchemy import func, select

from rag_service.api.errors import BusinessError
from rag_service.auth.schemas import AdminApiKeyCreate
from rag_service.auth.services import ApiKeyService
from rag_service.config import Settings, get_settings
from rag_service.db.models.auth import ApiKey, AuditEvent
from rag_service.db.session import Database
from rag_service.main import create_app

ADMIN_HMAC_SECRET = "admin-test-hmac-secret-32-bytes!!"
AGENT_HMAC_SECRET = "agent-test-hmac-secret-32-bytes!!"


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


def _agent_policy(name: str) -> dict[str, object]:
    return {
        "name": name,
        "capabilities": ["retrieve"],
        "knowledge_base_ids": [],
        "query_profile_ids": [],
        "default_query_profile_id": None,
        "raw_file_read": False,
        "requests_per_minute": 60,
        "max_concurrency": 4,
        "not_before": None,
        "expires_at": None,
    }


def _app(database: Database, settings: Settings) -> FastAPI:
    app = create_app()
    app.state.database = database
    app.dependency_overrides[get_settings] = lambda: settings
    return app


async def _admin_token(database: Database, settings: Settings) -> str:
    async with database.session() as session:
        service = ApiKeyService(
            session=session,
            authentication_sessions=database.session,
            settings=settings,
        )
        issued = await service.create_admin_key(
            AdminApiKeyCreate(name="http-admin"),
            request_id="req-http-admin-bootstrap",
        )
    return issued.token.get_secret_value()


async def _audit_count(database: Database, key_id: UUID, action: str) -> int:
    async with database.session() as session:
        return cast(
            int,
            await session.scalar(
                select(func.count())
                .select_from(AuditEvent)
                .where(AuditEvent.target_id == key_id, AuditEvent.action == action)
            ),
        )


async def _audit_request_id(database: Database, key_id: UUID, action: str) -> str:
    async with database.session() as session:
        request_id = await session.scalar(
            select(AuditEvent.request_id).where(
                AuditEvent.target_id == key_id,
                AuditEvent.action == action,
            )
        )
        assert isinstance(request_id, str)
        return request_id


async def _key_record(database: Database, key_id: UUID) -> ApiKey:
    async with database.session() as session:
        key = await session.get(ApiKey, key_id)
        assert key is not None
        return key


def _assert_safe(document: object, *tokens: str) -> None:
    serialized = json.dumps(document, sort_keys=True)
    lowered = serialized.lower()
    assert "token" not in lowered
    assert "digest" not in lowered
    assert "authorization" not in lowered
    for token in tokens:
        assert token not in serialized


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_can_manage_an_agent_key_with_safe_etag_responses(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    admin_token = await _admin_token(migrated_database, settings)
    app = _app(migrated_database, settings)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        created = await client.post(
            "/v1/admin/api-keys",
            headers={
                "Authorization": f"Bearer {admin_token}",
                "X-Request-ID": "req-http-agent-create",
            },
            json=_agent_policy("managed-agent"),
        )
        assert created.status_code == 201
        assert created.headers["cache-control"] == "no-store"
        created_document = created.json()
        assert set(created_document) == {"api_key", "token"}
        token = created_document["token"]
        assert isinstance(token, str) and token.startswith("rag_agent_")
        safe = created_document["api_key"]
        key_id = UUID(safe["id"])
        first_etag = created.headers["etag"]
        assert safe["etag"] == first_etag
        assert created.headers["location"] == f"/v1/admin/api-keys/{key_id}"
        assert created.text.count(token) == 1

        listed = await client.get(
            "/v1/admin/api-keys",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert listed.status_code == 200
        assert listed.headers["cache-control"] == "no-store"
        listed_document = listed.json()
        assert [item["id"] for item in listed_document["items"]] == [str(key_id)]
        _assert_safe(listed_document, token, admin_token)

        detail = await client.get(
            f"/v1/admin/api-keys/{key_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert detail.status_code == 200
        assert detail.headers["etag"] == first_etag
        _assert_safe(detail.json(), token, admin_token)

        updated = await client.patch(
            f"/v1/admin/api-keys/{key_id}",
            headers={"Authorization": f"Bearer {admin_token}", "If-Match": first_etag},
            json={"name": "updated-agent"},
        )
        assert updated.status_code == 200
        second_etag = updated.headers["etag"]
        assert second_etag != first_etag
        assert updated.json()["name"] == "updated-agent"
        _assert_safe(updated.json(), token, admin_token)

        revoked = await client.post(
            f"/v1/admin/api-keys/{key_id}/revoke",
            headers={"Authorization": f"Bearer {admin_token}", "If-Match": second_etag},
        )
        assert revoked.status_code == 200
        third_etag = revoked.headers["etag"]
        assert third_etag != second_etag
        assert revoked.json()["status"] == "revoked"
        _assert_safe(revoked.json(), token, admin_token)

        repeated_revoke = await client.post(
            f"/v1/admin/api-keys/{key_id}/revoke",
            headers={"Authorization": f"Bearer {admin_token}", "If-Match": third_etag},
        )
        assert repeated_revoke.status_code == 200
        assert repeated_revoke.headers["etag"] == third_etag
        _assert_safe(repeated_revoke.json(), token, admin_token)

        missing_precondition = await client.patch(
            f"/v1/admin/api-keys/{key_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
            json={"name": "must-not-apply"},
        )
        assert missing_precondition.status_code == 412
        assert missing_precondition.json()["error"]["code"] == "PRECONDITION_FAILED"
        assert "TypeError" not in missing_precondition.text
        stale_precondition = await client.patch(
            f"/v1/admin/api-keys/{key_id}",
            headers={"Authorization": f"Bearer {admin_token}", "If-Match": second_etag},
            json={"name": "must-not-apply"},
        )
        assert stale_precondition.status_code == 412
        assert stale_precondition.json()["error"]["code"] == "PRECONDITION_FAILED"
        assert "TypeError" not in stale_precondition.text
        revoked_patch = await client.patch(
            f"/v1/admin/api-keys/{key_id}",
            headers={"Authorization": f"Bearer {admin_token}", "If-Match": third_etag},
            json={"name": "must-not-apply"},
        )
        assert revoked_patch.status_code == 409

    persisted = await _key_record(migrated_database, key_id)
    assert persisted.name == "updated-agent"
    assert persisted.resource_revision == 3
    assert await _audit_count(migrated_database, key_id, "api_key.policy_updated") == 1
    assert await _audit_count(migrated_database, key_id, "api_key.revoked") == 1
    assert (
        await _audit_request_id(migrated_database, key_id, "api_key.created")
        == "req-http-agent-create"
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_routes_reject_agent_keys_idempotency_and_invalid_agent_policies(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    admin_token = await _admin_token(migrated_database, settings)
    app = _app(migrated_database, settings)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        created = await client.post(
            "/v1/admin/api-keys",
            headers={"Authorization": f"Bearer {admin_token}"},
            json=_agent_policy("agent-auth-check"),
        )
        assert created.status_code == 201
        agent_token = created.json()["token"]

        agent_denied = await client.get(
            "/v1/admin/api-keys",
            headers={"Authorization": f"Bearer {agent_token}"},
        )
        assert agent_denied.status_code == 401
        assert agent_denied.headers["www-authenticate"] == "Bearer"
        assert agent_denied.json()["error"]["code"] == "INVALID_API_KEY"

        for idempotency_key in ("", "no-replay"):
            rejected = await client.post(
                "/v1/admin/api-keys",
                headers={
                    "Authorization": f"Bearer {admin_token}",
                    "Idempotency-Key": idempotency_key,
                },
                json=_agent_policy(f"idempotency-{idempotency_key or 'empty'}"),
            )
            assert rejected.status_code == 422
            assert rejected.json()["error"]["code"] == "VALIDATION_ERROR"

        invalid_limit = _agent_policy("too-fast")
        invalid_limit["requests_per_minute"] = 101
        rejected_limit = await client.post(
            "/v1/admin/api-keys",
            headers={"Authorization": f"Bearer {admin_token}"},
            json=invalid_limit,
        )
        assert rejected_limit.status_code == 422
        assert rejected_limit.json()["error"]["code"] == "VALIDATION_ERROR"
        assert "TypeError" not in rejected_limit.text

        invalid_reference = _agent_policy("invalid-reference")
        invalid_reference["knowledge_base_ids"] = ["00000000-0000-0000-0000-000000000001"]
        rejected_reference = await client.post(
            "/v1/admin/api-keys",
            headers={"Authorization": f"Bearer {admin_token}"},
            json=invalid_reference,
        )
        assert rejected_reference.status_code == 422
        assert rejected_reference.json()["error"]["code"] == "VALIDATION_ERROR"
        assert "TypeError" not in rejected_reference.text


@pytest.mark.integration
def test_business_error_stays_frozen_for_route_transport_adaptation() -> None:
    error = BusinessError(422, "VALIDATION_ERROR", "Invalid API key policy")
    with pytest.raises(FrozenInstanceError):
        error.message = "must remain immutable"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_admin_agent_key_listing_validates_and_honors_cursor_pagination(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    admin_token = await _admin_token(migrated_database, settings)
    app = _app(migrated_database, settings)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        for name in ("first-agent", "second-agent", "third-agent"):
            created = await client.post(
                "/v1/admin/api-keys",
                headers={"Authorization": f"Bearer {admin_token}"},
                json=_agent_policy(name),
            )
            assert created.status_code == 201

        first_page = await client.get(
            "/v1/admin/api-keys?limit=1",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert first_page.status_code == 200
        first_document = first_page.json()
        assert len(first_document["items"]) == 1
        cursor = first_document["next_cursor"]
        assert isinstance(cursor, str)

        second_page = await client.get(
            "/v1/admin/api-keys",
            params={"limit": 1, "cursor": cursor},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert second_page.status_code == 200
        assert second_page.json()["items"][0]["id"] != first_document["items"][0]["id"]

        for query in ("limit=0", "limit=4", "cursor=invalid"):
            invalid_page = await client.get(
                f"/v1/admin/api-keys?{query}",
                headers={"Authorization": f"Bearer {admin_token}"},
            )
            assert invalid_page.status_code == 422
            assert invalid_page.json()["error"]["code"] == "VALIDATION_ERROR"


@pytest.mark.integration
def test_admin_api_openapi_is_bearer_protected_while_health_is_public(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    app = _app(migrated_database, _settings(async_url))
    document = app.openapi()

    assert document["paths"]["/health"]["get"].get("security") is None
    for path in (
        "/v1/admin/api-keys",
        "/v1/admin/api-keys/{key_id}",
        "/v1/admin/api-keys/{key_id}/revoke",
    ):
        for operation in document["paths"][path].values():
            assert operation["security"] == [{"HTTPBearer": []}]
