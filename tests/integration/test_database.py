import json
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from datetime import UTC, datetime
from uuid import uuid4

import psycopg
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.db.session import Database
from rag_service.infrastructure.probes import DatabaseReferencedCredentialLoader
from rag_service.readiness import INGEST_GENERATION_STATUSES, RETRIEVE_GENERATION_STATUSES

HEAD_REVISION = "20260730_0005"


@pytest.mark.integration
def test_alembic_upgrade_reaches_job_actor_head(
    postgres_urls: tuple[str, str], upgrade_head: Callable[[], None]
) -> None:
    upgrade_head()
    _async_url, sync_url = postgres_urls
    with psycopg.connect(sync_url) as connection:
        row = connection.execute("SELECT version_num FROM alembic_version").fetchone()

    assert row == (HEAD_REVISION,)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_migrated_database_engine_can_query(migrated_database: Database) -> None:
    await migrated_database.ping()

    async with migrated_database.engine.connect() as connection:
        value = await connection.scalar(text("SELECT 1"))

    assert value == 1


@pytest.mark.asyncio
@pytest.mark.integration
async def test_db_session_fixture_can_commit_and_query(db_session: AsyncSession) -> None:
    knowledge_base_id = uuid4()
    await db_session.execute(
        text("INSERT INTO knowledge_bases (id, name) VALUES (:id, :name)"),
        {"id": knowledge_base_id, "name": "async fixture transaction"},
    )
    await db_session.commit()

    stored_id = await db_session.scalar(
        text("SELECT id FROM knowledge_bases WHERE id = :id"),
        {"id": knowledge_base_id},
    )
    assert stored_id == knowledge_base_id


@pytest.mark.asyncio
@pytest.mark.integration
async def test_db_session_lifecycle_closes_then_truncates_with_fresh_connection(
    db_session_lifecycle: Callable[[], AbstractAsyncContextManager[AsyncSession]],
    postgres_urls: tuple[str, str],
) -> None:
    knowledge_base_id = uuid4()
    async with db_session_lifecycle() as session:
        await session.execute(
            text("INSERT INTO knowledge_bases (id, name) VALUES (:id, :name)"),
            {"id": knowledge_base_id, "name": "nested lifecycle cleanup"},
        )
        await session.commit()

    _async_url, sync_url = postgres_urls
    with psycopg.connect(sync_url) as connection:
        remaining = connection.execute(
            "SELECT count(*) FROM knowledge_bases WHERE id = %s",
            (knowledge_base_id,),
        ).fetchone()

    assert remaining == (0,)


@pytest.mark.asyncio
@pytest.mark.integration
async def test_real_postgres_readiness_loader_selects_exact_generation_credentials(
    migrated_database: Database,
) -> None:
    knowledge_base_id = uuid4()
    credential_ids = {
        status: uuid4() for status in ("active", "building", "failed", "retiring", "unused")
    }
    now = datetime.now(UTC)
    async with migrated_database.sessions() as session, session.begin():
        await session.execute(
            text("INSERT INTO knowledge_bases (id, name) VALUES (:id, :name)"),
            {"id": knowledge_base_id, "name": "readiness credential selection"},
        )
        for status, credential_id in credential_ids.items():
            await session.execute(
                text(
                    "INSERT INTO provider_credentials "
                    "(id, name, ciphertext, nonce, key_version) "
                    "VALUES (:id, :name, :ciphertext, :nonce, :key_version)"
                ),
                {
                    "id": credential_id,
                    "name": f"readiness-{status}",
                    "ciphertext": b"ciphertext-with-auth-tag",
                    "nonce": b"n" * 12,
                    "key_version": "key-v1",
                },
            )
        for status in ("active", "building", "failed", "retiring"):
            generation_id = uuid4()
            await session.execute(
                text(
                    "INSERT INTO knowledge_base_index_generations "
                    "(id, knowledge_base_id, index_profile_hash, qdrant_collection_name, status, "
                    "rebuild_snapshot_at, caught_up_revision, validated_revision, "
                    "validation_manifest_hash, expected_point_count, actual_point_count, "
                    "validated_at, activated_at, distance, embedding_config_snapshot, "
                    "filter_schema_snapshot, applied_filter_schema_revision, "
                    "embedding_config_hash) VALUES "
                    "(:id, :knowledge_base_id, :index_profile_hash, :collection, :status, "
                    ":now, 0, 0, :manifest_hash, 0, 0, :now, :now, 'cosine', "
                    "CAST(:embedding_snapshot AS jsonb), CAST(:filter_snapshot AS jsonb), 0, "
                    ":embedding_hash)"
                ),
                {
                    "id": generation_id,
                    "knowledge_base_id": knowledge_base_id,
                    "index_profile_hash": "a" * 64,
                    "collection": f"readiness_{generation_id.hex}",
                    "status": status,
                    "now": now,
                    "manifest_hash": "b" * 64,
                    "embedding_snapshot": json.dumps(
                        {"credential_id": str(credential_ids[status])}
                    ),
                    "filter_snapshot": json.dumps({"fields": []}),
                    "embedding_hash": "c" * 64,
                },
            )

    loader = DatabaseReferencedCredentialLoader(migrated_database)
    retrieve = await loader(RETRIEVE_GENERATION_STATUSES)
    ingest = await loader(INGEST_GENERATION_STATUSES)

    assert {credential.credential_id for credential in retrieve} == {credential_ids["active"]}
    assert {credential.credential_id for credential in ingest} == {
        credential_ids["active"],
        credential_ids["building"],
    }
    assert credential_ids["failed"] not in {credential.credential_id for credential in ingest}
    assert credential_ids["retiring"] not in {credential.credential_id for credential in ingest}
    assert credential_ids["unused"] not in {credential.credential_id for credential in ingest}

    async with migrated_database.sessions() as session, session.begin():
        await session.execute(
            text(
                "UPDATE provider_credentials SET resource_revision = 2, key_version = 'key-v2' "
                "WHERE id = :id"
            ),
            {"id": credential_ids["active"]},
        )
    rotated = await loader(RETRIEVE_GENERATION_STATUSES)
    assert [(row.resource_revision, row.encrypted.key_version) for row in rotated] == [
        (2, "key-v2")
    ]

    missing_id = uuid4()
    async with migrated_database.sessions() as session, session.begin():
        await session.execute(
            text(
                "UPDATE knowledge_base_index_generations "
                "SET embedding_config_snapshot = CAST(:snapshot AS jsonb) WHERE status = 'active'"
            ),
            {"snapshot": json.dumps({"credential_id": str(missing_id)})},
        )
    with pytest.raises(ValueError, match="referenced provider credential is missing"):
        await loader(RETRIEVE_GENERATION_STATUSES)
