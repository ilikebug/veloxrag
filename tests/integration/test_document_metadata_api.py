from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, cast
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import Depends, FastAPI
from pydantic import SecretStr
from sqlalchemy import delete, event, select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.auth.policies import AdminPrincipal, Capability
from rag_service.auth.schemas import AdminApiKeyCreate, AgentApiKeyCreate, SafeApiKey
from rag_service.auth.services import ApiKeyService
from rag_service.config import Settings, get_settings
from rag_service.db.dependencies import get_session
from rag_service.db.models.auth import ApiKeyKnowledgeBaseScope
from rag_service.db.models.documents import Document, DocumentVersion
from rag_service.db.models.knowledge_bases import KnowledgeBase
from rag_service.db.session import Database
from rag_service.main import create_app
from rag_service.metadata.document_repositories import (
    DocumentMetadataRepository,
    DocumentRecord,
    DocumentVersionRecord,
    SqlAlchemyDocumentMetadataRepository,
)
from rag_service.metadata.document_routes import get_document_metadata_service
from rag_service.metadata.services import DocumentMetadataService

ADMIN_HMAC_SECRET = "document-api-admin-test-hmac-secret"
AGENT_HMAC_SECRET = "document-api-agent-test-hmac-secret"

SAFE_DOCUMENT_FIELDS = {
    "id",
    "knowledge_base_id",
    "display_name",
    "mime_type",
    "checksum_sha256",
    "current_version_id",
    "pending_version_id",
    "status",
    "tags",
    "metadata",
    "created_at",
    "updated_at",
}
SAFE_VERSION_FIELDS = {
    "id",
    "document_id",
    "version_number",
    "source_checksum_sha256",
    "parsed_object_checksum_sha256",
    "declared_mime_type",
    "detected_mime_type",
    "source_extension",
    "base_version_id",
    "parser_name",
    "parser_version",
    "parser_config",
    "chunker_name",
    "chunker_version",
    "chunker_config",
    "chunk_count",
    "status",
    "activated_at",
    "created_at",
}
OBJECT_KEY_SECRET = "minio-secret-object-key-do-not-return"
ERROR_SECRET = "embedding-provider-secret-error-do-not-return"


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


def _headers(token: str, request_id: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "X-Request-ID": request_id,
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
    assert OBJECT_KEY_SECRET not in response.text
    assert ERROR_SECRET not in response.text
    return cast(dict[str, object], error)


def test_upload_openapi_reports_the_configured_byte_limit() -> None:
    configured_limit = 7 * 1024 * 1024
    app = create_app(
        settings=Settings(
            _env_file=None,
            max_upload_bytes=configured_limit,
        )
    )

    upload = app.openapi()["paths"]["/v1/knowledge-bases/{knowledge_base_id}/documents"]["post"]
    file_schema = upload["requestBody"]["content"]["multipart/form-data"]["schema"]["properties"][
        "file"
    ]

    assert file_schema["description"] == (
        "UTF-8 TXT or Markdown document, up to 7 MiB (7340032 bytes)."
    )


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
            AdminApiKeyCreate(name="document-api-admin"),
            request_id=f"req-document-admin-{uuid4().hex}",
        )
    return (
        AdminPrincipal(key_id=issued.api_key.id, public_id=issued.api_key.public_id),
        issued.token.get_secret_value(),
    )


async def _create_agent(
    database: Database,
    settings: Settings,
    admin: AdminPrincipal,
    *,
    name: str,
    capabilities: frozenset[Capability],
    knowledge_base_ids: frozenset[UUID],
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
                knowledge_base_ids=knowledge_base_ids,
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


@dataclass(frozen=True, slots=True)
class SeededMetadata:
    active_kb_id: UUID
    disabled_kb_id: UUID
    reindexing_kb_id: UUID
    deleting_kb_id: UUID
    outside_kb_id: UUID
    live_document_ids: tuple[UUID, ...]
    target_document_id: UUID
    deleted_document_id: UUID
    deleting_document_id: UUID
    outside_document_id: UUID
    deleting_parent_document_id: UUID
    version_ids: tuple[UUID, ...]
    manage_key_id: UUID
    manage_token: str
    ingest_token: str
    retrieve_token: str
    answer_token: str
    admin_token: str


async def _seed_metadata(database: Database, settings: Settings) -> SeededMetadata:
    base_time = datetime(2026, 7, 24, 10, 30, tzinfo=UTC)
    active_kb_id = UUID(int=100)
    disabled_kb_id = UUID(int=101)
    reindexing_kb_id = UUID(int=102)
    deleting_kb_id = UUID(int=103)
    outside_kb_id = UUID(int=104)
    live_document_ids = tuple(UUID(int=value) for value in (201, 202, 203, 204))
    target_document_id = live_document_ids[1]
    deleted_document_id = UUID(int=205)
    deleting_document_id = UUID(int=206)
    disabled_document_id = UUID(int=207)
    reindexing_document_id = UUID(int=208)
    deleting_parent_document_id = UUID(int=209)
    outside_document_id = UUID(int=210)
    version_ids = tuple(UUID(int=value) for value in (301, 302, 303))

    async with database.sessions() as session, session.begin():
        session.add_all(
            [
                KnowledgeBase(
                    id=active_kb_id,
                    name="Active KB",
                    status="active",
                    metadata_={},
                    filter_schema={"fields": []},
                    resource_revision=1,
                    mutation_revision=0,
                    filter_schema_revision=0,
                    created_at=base_time,
                    updated_at=base_time,
                ),
                KnowledgeBase(
                    id=disabled_kb_id,
                    name="Disabled KB",
                    status="disabled",
                    metadata_={},
                    filter_schema={"fields": []},
                    resource_revision=1,
                    mutation_revision=0,
                    filter_schema_revision=0,
                    created_at=base_time,
                    updated_at=base_time,
                ),
                KnowledgeBase(
                    id=reindexing_kb_id,
                    name="Reindexing KB",
                    status="reindexing",
                    metadata_={},
                    filter_schema={"fields": []},
                    resource_revision=1,
                    mutation_revision=0,
                    filter_schema_revision=0,
                    created_at=base_time,
                    updated_at=base_time,
                ),
                KnowledgeBase(
                    id=deleting_kb_id,
                    name="Deleting KB",
                    status="deleting",
                    metadata_={},
                    filter_schema={"fields": []},
                    resource_revision=1,
                    mutation_revision=0,
                    filter_schema_revision=0,
                    created_at=base_time,
                    updated_at=base_time,
                ),
                KnowledgeBase(
                    id=outside_kb_id,
                    name="Outside KB",
                    status="active",
                    metadata_={},
                    filter_schema={"fields": []},
                    resource_revision=1,
                    mutation_revision=0,
                    filter_schema_revision=0,
                    created_at=base_time,
                    updated_at=base_time,
                ),
            ]
        )

    admin, admin_token = await _create_admin(database, settings)
    readable_scope = frozenset({active_kb_id, disabled_kb_id, reindexing_kb_id})
    manage_key, manage_token = await _create_agent(
        database,
        settings,
        admin,
        name="document-manage-agent",
        capabilities=frozenset({Capability.MANAGE}),
        knowledge_base_ids=readable_scope,
    )
    _ingest_key, ingest_token = await _create_agent(
        database,
        settings,
        admin,
        name="document-ingest-agent",
        capabilities=frozenset({Capability.INGEST}),
        knowledge_base_ids=frozenset({active_kb_id}),
    )
    _retrieve_key, retrieve_token = await _create_agent(
        database,
        settings,
        admin,
        name="document-retrieve-agent",
        capabilities=frozenset({Capability.RETRIEVE}),
        knowledge_base_ids=frozenset({active_kb_id}),
    )
    _answer_key, answer_token = await _create_agent(
        database,
        settings,
        admin,
        name="document-answer-agent",
        capabilities=frozenset({Capability.ANSWER}),
        knowledge_base_ids=frozenset({active_kb_id}),
    )

    document_time = base_time + timedelta(minutes=1)
    version_time = base_time + timedelta(minutes=2)
    async with database.sessions() as session, session.begin():
        session.add(
            ApiKeyKnowledgeBaseScope(
                api_key_id=manage_key.id,
                knowledge_base_id=deleting_kb_id,
            )
        )
        documents = [
            Document(
                id=document_id,
                knowledge_base_id=active_kb_id,
                display_name=f"Manual {index}",
                mime_type="application/pdf",
                checksum_sha256=f"{index + 1:064x}",
                status="active",
                tags=["manual", f"part-{index}"],
                metadata_={"nested": {"locale": "zh-CN"}, "rank": index},
                created_at=document_time,
                updated_at=document_time,
            )
            for index, document_id in enumerate(live_document_ids)
        ]
        documents.extend(
            [
                Document(
                    id=deleted_document_id,
                    knowledge_base_id=active_kb_id,
                    display_name="Soft deleted",
                    checksum_sha256="a" * 64,
                    status="deleted",
                    tags=[],
                    metadata_={},
                    created_at=document_time,
                    updated_at=document_time,
                    deleted_at=document_time,
                ),
                Document(
                    id=deleting_document_id,
                    knowledge_base_id=active_kb_id,
                    display_name="Deleting document",
                    checksum_sha256="b" * 64,
                    status="deleting",
                    tags=[],
                    metadata_={},
                    created_at=document_time,
                    updated_at=document_time,
                    deleted_at=document_time,
                ),
                Document(
                    id=disabled_document_id,
                    knowledge_base_id=disabled_kb_id,
                    display_name="Disabled parent readable",
                    checksum_sha256="c" * 64,
                    status="active",
                    tags=[],
                    metadata_={"state": "disabled-parent"},
                    created_at=document_time,
                    updated_at=document_time,
                ),
                Document(
                    id=reindexing_document_id,
                    knowledge_base_id=reindexing_kb_id,
                    display_name="Reindexing parent readable",
                    checksum_sha256="d" * 64,
                    status="active",
                    tags=[],
                    metadata_={"state": "reindexing-parent"},
                    created_at=document_time,
                    updated_at=document_time,
                ),
                Document(
                    id=deleting_parent_document_id,
                    knowledge_base_id=deleting_kb_id,
                    display_name="Deleting parent hidden",
                    checksum_sha256="e" * 64,
                    status="active",
                    tags=[],
                    metadata_={},
                    created_at=document_time,
                    updated_at=document_time,
                ),
                Document(
                    id=outside_document_id,
                    knowledge_base_id=outside_kb_id,
                    display_name="Outside scope",
                    checksum_sha256="f" * 64,
                    status="active",
                    tags=[],
                    metadata_={},
                    created_at=document_time,
                    updated_at=document_time,
                ),
            ]
        )
        session.add_all(documents)
        await session.flush()
        versions = [
            DocumentVersion(
                id=version_id,
                document_id=target_document_id,
                version_number=index + 1,
                source_object_key=f"{OBJECT_KEY_SECRET}/source-{index}",
                parsed_object_key=f"{OBJECT_KEY_SECRET}/parsed-{index}",
                source_checksum_sha256=f"{index + 20:064x}",
                parsed_object_checksum_sha256=f"{index + 30:064x}",
                declared_mime_type="application/pdf",
                detected_mime_type="application/pdf",
                source_extension="pdf",
                base_version_id=version_ids[index - 1] if index else None,
                parser_name="safe-parser",
                parser_version="1.2.3",
                parser_config={"ocr": {"languages": ["zh", "en"]}},
                chunker_name="safe-chunker",
                chunker_version="2.0.0",
                chunker_config={"window": 512, "overlap": 64},
                chunk_count=10 + index,
                status="ready" if index < 2 else "failed",
                activated_at=version_time if index < 2 else None,
                created_at=version_time,
            )
            for index, version_id in enumerate(version_ids)
        ]
        session.add_all(versions)
        await session.flush()
        target = next(row for row in documents if row.id == target_document_id)
        target.current_version_id = version_ids[1]
        target.pending_version_id = version_ids[2]

    return SeededMetadata(
        active_kb_id=active_kb_id,
        disabled_kb_id=disabled_kb_id,
        reindexing_kb_id=reindexing_kb_id,
        deleting_kb_id=deleting_kb_id,
        outside_kb_id=outside_kb_id,
        live_document_ids=live_document_ids,
        target_document_id=target_document_id,
        deleted_document_id=deleted_document_id,
        deleting_document_id=deleting_document_id,
        outside_document_id=outside_document_id,
        deleting_parent_document_id=deleting_parent_document_id,
        version_ids=version_ids,
        manage_key_id=manage_key.id,
        manage_token=manage_token,
        ingest_token=ingest_token,
        retrieve_token=retrieve_token,
        answer_token=answer_token,
        admin_token=admin_token,
    )


class _ScopeRevokingDocumentRepository:
    def __init__(
        self,
        delegate: SqlAlchemyDocumentMetadataRepository,
        database: Database,
        actor_key_id: UUID,
        knowledge_base_id: UUID,
    ) -> None:
        self._delegate = delegate
        self._database = database
        self._actor_key_id = actor_key_id
        self._knowledge_base_id = knowledge_base_id
        self.revoked = False

    async def get_scoped_parent(
        self,
        actor_key_id: UUID,
        knowledge_base_id: UUID,
    ) -> UUID | None:
        return await self._delegate.get_scoped_parent(actor_key_id, knowledge_base_id)

    async def list_documents(
        self,
        actor_key_id: UUID,
        knowledge_base_id: UUID,
        position: object,
        limit: int,
    ) -> list[DocumentRecord]:
        return await self._delegate.list_documents(
            actor_key_id,
            knowledge_base_id,
            cast(Any, position),
            limit,
        )

    async def get_document(
        self,
        actor_key_id: UUID,
        document_id: UUID,
    ) -> DocumentRecord | None:
        return await self._delegate.get_document(actor_key_id, document_id)

    async def get_document_parent(
        self,
        actor_key_id: UUID,
        document_id: UUID,
    ) -> UUID | None:
        parent_id = await self._delegate.get_document_parent(actor_key_id, document_id)
        if parent_id is not None and not self.revoked:
            async with self._database.sessions() as session, session.begin():
                await session.execute(
                    delete(ApiKeyKnowledgeBaseScope).where(
                        ApiKeyKnowledgeBaseScope.api_key_id == self._actor_key_id,
                        ApiKeyKnowledgeBaseScope.knowledge_base_id == self._knowledge_base_id,
                    )
                )
            self.revoked = True
        return parent_id

    async def list_versions(self, *args: object) -> list[DocumentVersionRecord]:
        return cast(
            list[DocumentVersionRecord],
            await cast(Any, self._delegate).list_versions(*args),
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_document_reads_require_agent_scope_and_an_allowed_capability(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    seeded = await _seed_metadata(migrated_database, settings)
    app = _app(migrated_database, settings)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        for capability_name, token in (
            ("manage", seeded.manage_token),
            ("ingest", seeded.ingest_token),
            ("retrieve", seeded.retrieve_token),
        ):
            response = await client.get(
                f"/v1/documents/{seeded.target_document_id}",
                headers=_headers(token, f"req-document-{capability_name}"),
            )
            assert response.status_code == 200

        answer_request_id = "req-document-answer-forbidden"
        answer = await client.get(
            f"/v1/documents/{seeded.target_document_id}",
            headers=_headers(seeded.answer_token, answer_request_id),
        )
        _assert_error(answer, 403, "INSUFFICIENT_CAPABILITY", answer_request_id)

        missing_auth_id = "req-document-missing-auth"
        missing_auth = await client.get(
            f"/v1/documents/{seeded.target_document_id}",
            headers={"X-Request-ID": missing_auth_id},
        )
        _assert_error(missing_auth, 401, "INVALID_API_KEY", missing_auth_id)
        assert missing_auth.headers["www-authenticate"] == "Bearer"

        admin_request_id = "req-document-admin-not-agent"
        admin = await client.get(
            f"/v1/documents/{seeded.target_document_id}",
            headers=_headers(seeded.admin_token, admin_request_id),
        )
        _assert_error(admin, 401, "INVALID_API_KEY", admin_request_id)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_upload_route_rejects_missing_and_invalid_agent_authentication(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    seeded = await _seed_metadata(migrated_database, settings)
    app = _app(migrated_database, settings)
    path = f"/v1/knowledge-bases/{seeded.active_kb_id}/documents"

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        missing_request_id = "req-upload-missing-auth"
        missing = await client.post(
            path,
            headers={"X-Request-ID": missing_request_id},
        )
        _assert_error(missing, 401, "INVALID_API_KEY", missing_request_id)
        assert missing.headers["www-authenticate"] == "Bearer"

        invalid_request_id = "req-upload-invalid-auth"
        invalid = await client.post(
            path,
            headers={
                "Authorization": "Bearer invalid-agent-token",
                "X-Request-ID": invalid_request_id,
            },
        )
        _assert_error(invalid, 401, "INVALID_API_KEY", invalid_request_id)
        assert invalid.headers["www-authenticate"] == "Bearer"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_document_reads_hide_missing_scope_soft_deletion_and_deleting_parents(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    seeded = await _seed_metadata(migrated_database, settings)
    app = _app(migrated_database, settings)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        hidden_documents = (
            uuid4(),
            seeded.deleted_document_id,
            seeded.outside_document_id,
            seeded.deleting_parent_document_id,
        )
        messages: set[object] = set()
        for index, document_id in enumerate(hidden_documents):
            request_id = f"req-document-hidden-{index}"
            detail = await client.get(
                f"/v1/documents/{document_id}",
                headers=_headers(seeded.manage_token, request_id),
            )
            error = _assert_error(detail, 404, "RESOURCE_NOT_FOUND", request_id)
            messages.add(error["message"])

            versions_request_id = f"req-version-hidden-{index}"
            versions = await client.get(
                f"/v1/documents/{document_id}/versions",
                headers=_headers(seeded.manage_token, versions_request_id),
            )
            error = _assert_error(
                versions,
                404,
                "RESOURCE_NOT_FOUND",
                versions_request_id,
            )
            messages.add(error["message"])
        assert messages == {"Resource not found"}

        hidden_parents = (uuid4(), seeded.outside_kb_id, seeded.deleting_kb_id)
        for index, knowledge_base_id in enumerate(hidden_parents):
            request_id = f"req-document-list-hidden-parent-{index}"
            response = await client.get(
                f"/v1/knowledge-bases/{knowledge_base_id}/documents",
                headers=_headers(seeded.manage_token, request_id),
            )
            _assert_error(response, 404, "RESOURCE_NOT_FOUND", request_id)

        for state, knowledge_base_id in (
            ("disabled", seeded.disabled_kb_id),
            ("reindexing", seeded.reindexing_kb_id),
        ):
            response = await client.get(
                f"/v1/knowledge-bases/{knowledge_base_id}/documents",
                headers=_headers(seeded.manage_token, f"req-document-{state}-readable"),
            )
            assert response.status_code == 200
            assert len(response.json()["items"]) == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_document_and_version_pages_are_deterministic_safe_and_complete(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    seeded = await _seed_metadata(migrated_database, settings)
    app = _app(migrated_database, settings)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        first = await client.get(
            f"/v1/knowledge-bases/{seeded.active_kb_id}/documents",
            params={"limit": 2},
            headers=_headers(seeded.manage_token, "req-document-page-first"),
        )
        assert first.status_code == 200
        first_page = first.json()
        assert set(first_page) == {"items", "next_cursor"}
        assert first_page["next_cursor"] is not None
        assert all(set(item) == SAFE_DOCUMENT_FIELDS for item in first_page["items"])

        second = await client.get(
            f"/v1/knowledge-bases/{seeded.active_kb_id}/documents",
            params={"limit": 2, "cursor": first_page["next_cursor"]},
            headers=_headers(seeded.manage_token, "req-document-page-second"),
        )
        assert second.status_code == 200
        second_page = second.json()
        assert second_page["next_cursor"] is None
        document_ids = [UUID(item["id"]) for item in [*first_page["items"], *second_page["items"]]]
        assert document_ids == sorted(seeded.live_document_ids)
        assert len(document_ids) == len(set(document_ids))
        assert seeded.deleted_document_id not in document_ids
        assert seeded.deleting_document_id not in document_ids

        detail = await client.get(
            f"/v1/documents/{seeded.target_document_id}",
            headers=_headers(seeded.manage_token, "req-document-safe-detail"),
        )
        assert detail.status_code == 200
        detail_document = detail.json()
        assert set(detail_document) == SAFE_DOCUMENT_FIELDS
        assert detail_document["metadata"] == {
            "nested": {"locale": "zh-CN"},
            "rank": 1,
        }
        assert detail_document["current_version_id"] == str(seeded.version_ids[1])
        assert detail_document["pending_version_id"] == str(seeded.version_ids[2])

        versions_first = await client.get(
            f"/v1/documents/{seeded.target_document_id}/versions",
            params={"limit": 2},
            headers=_headers(seeded.manage_token, "req-version-page-first"),
        )
        assert versions_first.status_code == 200
        versions_first_page = versions_first.json()
        assert versions_first_page["next_cursor"] is not None
        assert all(set(item) == SAFE_VERSION_FIELDS for item in versions_first_page["items"])

        versions_second = await client.get(
            f"/v1/documents/{seeded.target_document_id}/versions",
            params={"limit": 2, "cursor": versions_first_page["next_cursor"]},
            headers=_headers(seeded.manage_token, "req-version-page-second"),
        )
        assert versions_second.status_code == 200
        versions_second_page = versions_second.json()
        assert versions_second_page["next_cursor"] is None
        version_documents = [
            *versions_first_page["items"],
            *versions_second_page["items"],
        ]
        version_ids = [UUID(item["id"]) for item in version_documents]
        assert version_ids == sorted(seeded.version_ids)
        assert len(version_ids) == len(set(version_ids))
        assert version_documents[0]["parser_config"] == {"ocr": {"languages": ["zh", "en"]}}
        assert version_documents[0]["chunker_config"] == {
            "window": 512,
            "overlap": 64,
        }

        serialized = " ".join(
            (first.text, second.text, detail.text, versions_first.text, versions_second.text)
        )
        assert OBJECT_KEY_SECRET not in serialized
        assert ERROR_SECRET not in serialized
        assert seeded.manage_token not in serialized
        assert ADMIN_HMAC_SECRET not in serialized
        assert AGENT_HMAC_SECRET not in serialized
        assert "source_object_key" not in serialized
        assert "parsed_object_key" not in serialized
        assert "deleted_at" not in serialized


@pytest.mark.integration
@pytest.mark.asyncio
async def test_document_routes_validate_pagination_and_expose_stable_openapi(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    seeded = await _seed_metadata(migrated_database, settings)
    app = _app(migrated_database, settings)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        invalid_requests: tuple[tuple[str, dict[str, str | int]], ...] = (
            (
                f"/v1/knowledge-bases/{seeded.active_kb_id}/documents",
                {"cursor": "not-a-cursor"},
            ),
            (
                f"/v1/knowledge-bases/{seeded.active_kb_id}/documents",
                {"limit": 0},
            ),
            (
                f"/v1/knowledge-bases/{seeded.active_kb_id}/documents",
                {"limit": settings.max_page_size + 1},
            ),
            (
                f"/v1/documents/{seeded.target_document_id}/versions",
                {"cursor": "not-a-cursor"},
            ),
            (
                f"/v1/documents/{seeded.target_document_id}/versions",
                {"limit": 0},
            ),
            (
                f"/v1/documents/{seeded.target_document_id}/versions",
                {"limit": settings.max_page_size + 1},
            ),
        )
        for index, (path, params) in enumerate(invalid_requests):
            request_id = f"req-document-invalid-page-{index}"
            response = await client.get(
                path,
                params=params,
                headers=_headers(seeded.manage_token, request_id),
            )
            _assert_error(response, 422, "VALIDATION_ERROR", request_id)

        malformed_id_request = "req-document-invalid-uuid"
        malformed_id = await client.get(
            "/v1/documents/not-a-uuid",
            headers=_headers(seeded.manage_token, malformed_id_request),
        )
        _assert_error(malformed_id, 422, "VALIDATION_ERROR", malformed_id_request)

    openapi = app.openapi()
    document_paths = {path for path in openapi["paths"] if "documents" in path}
    expected_paths = {
        "/v1/knowledge-bases/{knowledge_base_id}/documents",
        "/v1/documents/{document_id}",
        "/v1/documents/{document_id}/versions",
    }
    assert document_paths == expected_paths
    for path in expected_paths:
        expected_methods = (
            {"get", "post"}
            if path == "/v1/knowledge-bases/{knowledge_base_id}/documents"
            else {"get"}
        )
        assert set(openapi["paths"][path]) == expected_methods
        operation = openapi["paths"][path]["get"]
        assert operation["security"] == [{"HTTPBearer": []}]
        assert {"200", "401", "403", "404", "422", "500"} <= set(operation["responses"])
        for response in operation["responses"].values():
            schema = response["content"]["application/json"]["schema"]
            assert "$ref" in schema

    upload = openapi["paths"]["/v1/knowledge-bases/{knowledge_base_id}/documents"]["post"]
    assert upload["security"] == [{"HTTPBearer": []}]
    request_body = upload["requestBody"]
    assert request_body["required"] is True
    assert set(request_body["content"]) == {"multipart/form-data"}
    multipart_schema = request_body["content"]["multipart/form-data"]["schema"]
    assert multipart_schema["type"] == "object"
    assert multipart_schema["required"] == ["file"]
    assert multipart_schema["additionalProperties"] is False
    assert multipart_schema["properties"] == {
        "file": {
            "type": "string",
            "format": "binary",
            "description": "UTF-8 TXT or Markdown document, up to 50 MiB (52428800 bytes).",
        },
        "display_name": {
            "type": "string",
            "description": "Optional display name for the document.",
        },
        "metadata": {
            "type": "string",
            "description": "Optional JSON object encoded as a string.",
        },
        "tags": {
            "type": "string",
            "description": "Optional JSON array of tag strings encoded as a string.",
        },
    }
    assert {"202", "401", "403", "404", "409", "413", "422", "500", "503"} <= set(
        upload["responses"]
    )
    accepted_schema = upload["responses"]["202"]["content"]["application/json"]["schema"]
    assert accepted_schema["$ref"].endswith("/UploadAccepted")
    for status in ("401", "403", "404", "409", "413", "422", "500", "503"):
        schema = upload["responses"][status]["content"]["application/json"]["schema"]
        assert "$ref" in schema


@pytest.mark.integration
@pytest.mark.asyncio
async def test_document_queries_scope_before_loading_and_do_not_lazy_load(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    seeded = await _seed_metadata(migrated_database, settings)
    app = _app(migrated_database, settings)
    select_statements: list[str] = []

    def capture_selects(
        _connection: object,
        _cursor: object,
        statement: str,
        _parameters: object,
        _context: object,
        _executemany: bool,
    ) -> None:
        if statement.lstrip().upper().startswith("SELECT"):
            select_statements.append(statement)

    event.listen(
        migrated_database.engine.sync_engine,
        "before_cursor_execute",
        capture_selects,
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            detail = await client.get(
                f"/v1/documents/{seeded.target_document_id}",
                headers=_headers(seeded.manage_token, "req-document-query-detail"),
            )
            assert detail.status_code == 200
            detail_statements = [
                statement for statement in select_statements if "FROM documents" in statement
            ]
            assert len(detail_statements) == 1
            assert "knowledge_bases" in detail_statements[0]
            assert "api_key_knowledge_base_scopes" in detail_statements[0]
            assert "source_object_key" not in detail_statements[0]

            select_statements.clear()
            document_page = await client.get(
                f"/v1/knowledge-bases/{seeded.active_kb_id}/documents",
                headers=_headers(seeded.manage_token, "req-document-query-list"),
            )
            assert document_page.status_code == 200
            parent_statements = [
                statement
                for statement in select_statements
                if "FROM knowledge_bases" in statement
                and "api_key_knowledge_base_scopes" in statement
            ]
            document_statements = [
                statement for statement in select_statements if "FROM documents" in statement
            ]
            assert len(parent_statements) == 1
            assert len(document_statements) == 1
            assert "knowledge_bases" in document_statements[0]
            assert "api_key_knowledge_base_scopes" in document_statements[0]

            select_statements.clear()
            versions = await client.get(
                f"/v1/documents/{seeded.target_document_id}/versions",
                headers=_headers(seeded.manage_token, "req-version-query-list"),
            )
            assert versions.status_code == 200
            scoped_document_statements = [
                statement for statement in select_statements if "FROM documents" in statement
            ]
            version_statements = [
                statement
                for statement in select_statements
                if "FROM document_versions" in statement
            ]
            assert len(scoped_document_statements) == 1
            assert "knowledge_bases" in scoped_document_statements[0]
            assert "api_key_knowledge_base_scopes" in scoped_document_statements[0]
            assert len(version_statements) == 1
            version_statement = version_statements[0].lower()
            assert "join documents" in version_statement
            assert "knowledge_bases" in version_statement
            assert "api_key_knowledge_base_scopes" in version_statement
            assert "documents.deleted_at is null" in version_statement
            assert "knowledge_bases.status !=" in version_statement
            assert "source_object_key" not in version_statement
            assert "parsed_object_key" not in version_statement
    finally:
        event.remove(
            migrated_database.engine.sync_engine,
            "before_cursor_execute",
            capture_selects,
        )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_version_page_rechecks_scope_after_authorization_before_returning_rows(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    seeded = await _seed_metadata(migrated_database, settings)
    app = _app(migrated_database, settings)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        empty = await client.get(
            f"/v1/documents/{seeded.live_document_ids[0]}/versions",
            headers=_headers(seeded.manage_token, "req-version-static-empty"),
        )
    assert empty.status_code == 200
    assert empty.json() == {"items": [], "next_cursor": None}

    repository_holder: list[_ScopeRevokingDocumentRepository] = []

    async def race_service(
        session: Annotated[AsyncSession, Depends(get_session)],
    ) -> DocumentMetadataService:
        def repository_factory(
            current_session: AsyncSession,
        ) -> DocumentMetadataRepository:
            repository = _ScopeRevokingDocumentRepository(
                SqlAlchemyDocumentMetadataRepository(current_session),
                migrated_database,
                seeded.manage_key_id,
                seeded.active_kb_id,
            )
            repository_holder.append(repository)
            return cast(DocumentMetadataRepository, repository)

        return DocumentMetadataService(
            session=session,
            settings=settings,
            repository_factory=repository_factory,
        )

    app.dependency_overrides[get_document_metadata_service] = race_service
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        raced = await client.get(
            f"/v1/documents/{seeded.target_document_id}/versions",
            headers=_headers(seeded.manage_token, "req-version-scope-revoked-race"),
        )

    assert repository_holder and repository_holder[0].revoked is True
    assert raced.status_code == 200
    assert raced.json() == {"items": [], "next_cursor": None}


@pytest.mark.integration
@pytest.mark.asyncio
async def test_corrupt_stored_document_json_returns_only_a_sanitized_internal_error(
    migrated_database: Database,
    postgres_urls: tuple[str, str],
) -> None:
    async_url, _sync_url = postgres_urls
    settings = _settings(async_url)
    seeded = await _seed_metadata(migrated_database, settings)
    async with migrated_database.sessions() as session, session.begin():
        document = await session.scalar(
            select(Document).where(Document.id == seeded.target_document_id)
        )
        assert document is not None
        document.metadata_ = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {OBJECT_KEY_SECRET: ERROR_SECRET},
                    }
                }
            }
        }

    app = _app(migrated_database, settings)
    request_id = "req-document-corrupt-json"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        response = await client.get(
            f"/v1/documents/{seeded.target_document_id}",
            headers=_headers(seeded.manage_token, request_id),
        )

    error = _assert_error(response, 500, "INTERNAL_ERROR", request_id)
    assert error["message"] == "Internal server error"

    async with migrated_database.sessions() as session, session.begin():
        document = await session.scalar(
            select(Document).where(Document.id == seeded.target_document_id)
        )
        version = await session.scalar(
            select(DocumentVersion).where(DocumentVersion.id == seeded.version_ids[0])
        )
        assert document is not None
        assert version is not None
        document.metadata_ = {}
        version.parser_config = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {OBJECT_KEY_SECRET: ERROR_SECRET},
                    }
                }
            }
        }

    version_request_id = "req-version-corrupt-json"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        version_response = await client.get(
            f"/v1/documents/{seeded.target_document_id}/versions",
            headers=_headers(seeded.manage_token, version_request_id),
        )

    version_error = _assert_error(
        version_response,
        500,
        "INTERNAL_ERROR",
        version_request_id,
    )
    assert version_error["message"] == "Internal server error"
