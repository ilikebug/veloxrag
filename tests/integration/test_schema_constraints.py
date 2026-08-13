import subprocess
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, cast
from uuid import UUID, uuid4

import psycopg
import pytest
from alembic.autogenerate import compare_metadata
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from psycopg.types.json import Jsonb
from sqlalchemy import CheckConstraint, MetaData, UniqueConstraint, create_engine, inspect, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.engine import Dialect

from rag_service.db import models as _models  # noqa: F401
from rag_service.db.base import Base

ALEMBIC_CONFIG = "alembic.ini"
BASELINE_REVISION = "20260723_0001"
PREVIOUS_REVISION = "20260724_0002"
GENERATION_CLEANUP_PARENT_REVISION = "20260726_0003"
HEAD_PARENT_REVISION = "20260728_0004"
HEAD_REVISION = "20260730_0005"


def _insert_generation_cleanup_claim(
    connection: psycopg.Connection[Any],
    *,
    completed: bool,
) -> tuple[str, UUID, UUID, UUID, datetime, datetime | None]:
    knowledge_base_id = uuid4()
    generation_id = uuid4()
    lease_owner = uuid4()
    lease_expires_at = datetime.now(UTC) - timedelta(minutes=1)
    completed_at = datetime.now(UTC) if completed else None
    name = f"rag_kb_{knowledge_base_id.hex}_g_{generation_id.hex}"
    connection.execute(
        """
        INSERT INTO index_generation_cleanup_claims (
            collection_name, knowledge_base_id, generation_id, lease_owner,
            lease_epoch, lease_expires_at, completed_at
        ) VALUES (%s, %s, %s, %s, 1, %s, %s)
        """,
        (
            name,
            knowledge_base_id,
            generation_id,
            lease_owner,
            lease_expires_at,
            completed_at,
        ),
    )
    return (
        name,
        knowledge_base_id,
        generation_id,
        lease_owner,
        lease_expires_at,
        completed_at,
    )


def _postgres_dialect() -> Dialect:
    dialect_factory = cast(Callable[[], Dialect], postgresql.dialect)
    return dialect_factory()


def _snapshot_public_catalog(connection: psycopg.Connection[Any]) -> tuple[tuple[object, ...], ...]:
    rows = connection.execute(
        """
        WITH extension_owned AS (
            SELECT dependency.classid, dependency.objid
            FROM pg_catalog.pg_depend AS dependency
            JOIN pg_catalog.pg_extension AS extension
              ON extension.oid = dependency.refobjid
            WHERE dependency.refclassid = 'pg_catalog.pg_extension'::regclass
              AND dependency.deptype = 'e'
        )
        SELECT 'relation', c.relkind::text, c.relname,
               CASE WHEN c.relkind = 'i' THEN pg_catalog.pg_get_indexdef(c.oid)
                    ELSE NULL::text END
        FROM pg_catalog.pg_class AS c
        JOIN pg_catalog.pg_namespace AS n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public'
          AND c.relkind IN ('r', 'p', 'i', 'S', 'v', 'm', 'f')
          AND NOT EXISTS (
              SELECT 1 FROM extension_owned
              WHERE classid = 'pg_catalog.pg_class'::regclass AND objid = c.oid
          )
        UNION ALL
        SELECT 'constraint', c.contype::text, c.conname,
               pg_catalog.pg_get_constraintdef(c.oid, true)
        FROM pg_catalog.pg_constraint AS c
        JOIN pg_catalog.pg_namespace AS n ON n.oid = c.connamespace
        WHERE n.nspname = 'public'
          AND NOT EXISTS (
              SELECT 1 FROM extension_owned
              WHERE classid = 'pg_catalog.pg_constraint'::regclass AND objid = c.oid
          )
        UNION ALL
        SELECT 'function', p.prokind::text, p.proname,
               pg_catalog.pg_get_function_identity_arguments(p.oid)
        FROM pg_catalog.pg_proc AS p
        JOIN pg_catalog.pg_namespace AS n ON n.oid = p.pronamespace
        WHERE n.nspname = 'public'
          AND NOT EXISTS (
              SELECT 1 FROM extension_owned
              WHERE classid = 'pg_catalog.pg_proc'::regclass AND objid = p.oid
          )
        UNION ALL
        SELECT 'type', t.typtype::text, t.typname, t.typcategory::text
        FROM pg_catalog.pg_type AS t
        JOIN pg_catalog.pg_namespace AS n ON n.oid = t.typnamespace
        WHERE n.nspname = 'public'
          AND NOT EXISTS (
              SELECT 1 FROM extension_owned
              WHERE classid = 'pg_catalog.pg_type'::regclass AND objid = t.oid
          )
        ORDER BY 1, 2, 3, 4
        """
    ).fetchall()
    return tuple(tuple(row) for row in rows)


def _user_schemas(connection: psycopg.Connection[Any]) -> frozenset[str]:
    rows = connection.execute(
        """
        SELECT nspname
        FROM pg_catalog.pg_namespace
        WHERE nspname <> 'information_schema'
          AND nspname NOT LIKE 'pg\\_%' ESCAPE '\\'
        """
    ).fetchall()
    return frozenset(row[0] for row in rows)


def _strip_balanced_outer_parentheses(expression: str) -> str:
    value = expression.strip()
    while value.startswith("(") and value.endswith(")"):
        depth = 0
        quote: str | None = None
        encloses_entire_expression = True
        index = 0
        while index < len(value):
            character = value[index]
            if quote is not None:
                if character == quote:
                    if index + 1 < len(value) and value[index + 1] == quote:
                        index += 2
                        continue
                    quote = None
                index += 1
                continue
            if character in {"'", '"'}:
                quote = character
                index += 1
                continue
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth == 0 and index != len(value) - 1:
                    encloses_entire_expression = False
                    break
                if depth < 0:
                    encloses_entire_expression = False
                    break
            index += 1
        if not encloses_entire_expression or depth != 0 or quote is not None:
            break
        value = value[1:-1].strip()
    return value


def _normalize_check_sql(expression: str) -> str:
    value = expression.strip()
    if value.startswith("CHECK (") and value.endswith(")"):
        value = value[7:-1]
    return _strip_balanced_outer_parentheses(value)


def _insert_api_key(
    connection: psycopg.Connection[Any],
    *,
    public_id: str | None = None,
) -> UUID:
    key_id = uuid4()
    connection.execute(
        """
        INSERT INTO api_keys (
            id, public_id, secret_digest, key_type, name,
            requests_per_minute, max_concurrency
        ) VALUES (%s, %s, %s, 'agent', 'test agent', 100, 10)
        """,
        (key_id, public_id or f"rag_agent_{uuid4().hex}", b"a" * 32),
    )
    return key_id


def _insert_knowledge_base(connection: psycopg.Connection[Any], name: str) -> UUID:
    knowledge_base_id = uuid4()
    connection.execute(
        "INSERT INTO knowledge_bases (id, name) VALUES (%s, %s)",
        (knowledge_base_id, name),
    )
    return knowledge_base_id


def _insert_query_profile(
    connection: psycopg.Connection[Any],
    name: str,
    *,
    enabled: bool = True,
    is_system_default: bool = False,
) -> UUID:
    query_profile_id = uuid4()
    connection.execute(
        """
        INSERT INTO query_profiles (
            id, name, dense_candidate_limit, sparse_candidate_limit,
            rrf_candidate_limit, rerank_candidate_limit, top_k_limit,
            min_rerank_score, min_rrf_score_when_degraded, context_token_budget,
            enabled, is_system_default
        ) VALUES (%s, %s, 20, 20, 20, 10, 5, 0, 0, 4096, %s, %s)
        """,
        (query_profile_id, name, enabled, is_system_default),
    )
    return query_profile_id


def _insert_generation(
    connection: psycopg.Connection[Any],
    knowledge_base_id: UUID,
    *,
    status: str,
) -> UUID:
    generation_id = uuid4()
    is_active = status == "active"
    connection.execute(
        """
        INSERT INTO knowledge_base_index_generations (
            id, knowledge_base_id, index_profile_hash,
            qdrant_collection_name, status, rebuild_snapshot_at,
            distance, embedding_config_snapshot, filter_schema_snapshot,
            applied_filter_schema_revision, embedding_config_hash,
            validated_revision, validation_manifest_hash,
            expected_point_count, actual_point_count, validated_at, activated_at
        ) VALUES (%s, %s, %s, %s, %s, %s,
                  %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            generation_id,
            knowledge_base_id,
            "a" * 64,
            f"collection-{uuid4()}",
            status,
            datetime.now(UTC),
            "cosine" if is_active else None,
            Jsonb({}) if is_active else None,
            Jsonb({"fields": []}) if is_active else None,
            0 if is_active else None,
            "f" * 64 if is_active else None,
            0 if is_active else None,
            "e" * 64 if is_active else None,
            0 if is_active else None,
            0 if is_active else None,
            datetime.now(UTC) if is_active else None,
            datetime.now(UTC) if is_active else None,
        ),
    )
    return generation_id


def _insert_document(connection: psycopg.Connection[Any], knowledge_base_id: UUID) -> UUID:
    document_id = uuid4()
    connection.execute(
        "INSERT INTO documents (id, knowledge_base_id, display_name) VALUES (%s, %s, 'doc')",
        (document_id, knowledge_base_id),
    )
    return document_id


def _insert_document_version(
    connection: psycopg.Connection[Any],
    document_id: UUID,
    *,
    version_number: int = 1,
    base_version_id: UUID | None = None,
) -> UUID:
    version_id = uuid4()
    connection.execute(
        """
        INSERT INTO document_versions (
            id, document_id, version_number, source_object_key,
            source_checksum_sha256, base_version_id
        ) VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (
            version_id,
            document_id,
            version_number,
            f"source/{version_id}",
            "b" * 64,
            base_version_id,
        ),
    )
    return version_id


def _assert_constraint_violation(
    connection: psycopg.Connection[Any],
    sql: str,
    parameters: tuple[object, ...],
) -> None:
    with pytest.raises(psycopg.IntegrityError):
        connection.execute(sql, parameters)
        connection.commit()
    connection.rollback()


@pytest.mark.integration
def test_migration_graph_has_exact_ingestion_retrieval_head() -> None:
    scripts = ScriptDirectory.from_config(Config(ALEMBIC_CONFIG))

    assert scripts.get_heads() == [HEAD_REVISION]
    head = scripts.get_revision(HEAD_REVISION)
    assert head is not None
    assert head.down_revision == HEAD_PARENT_REVISION


@pytest.mark.integration
def test_increment_2_table_and_cleanup_contract_is_fixed(
    increment_2_tables: frozenset[str],
    increment_2_truncate_sql: str,
) -> None:
    approved_tables = frozenset(
        {
            "api_key_knowledge_base_scopes",
            "api_key_query_profile_scopes",
            "api_keys",
            "audit_events",
            "document_index_states",
            "document_versions",
            "documents",
            "idempotency_records",
            "index_generation_cleanup_claims",
            "index_generation_creation_requests",
            "jobs",
            "knowledge_base_index_generations",
            "knowledge_base_mutations",
            "knowledge_bases",
            "model_profile_fallbacks",
            "model_profiles",
            "provider_configs",
            "provider_credentials",
            "provider_usage",
            "query_logs",
            "query_profiles",
            "sparse_profiles",
            "document_upload_idempotency",
        }
    )
    approved_cleanup_sql = (
        'TRUNCATE TABLE "api_key_knowledge_base_scopes", '
        '"api_key_query_profile_scopes", "api_keys", "audit_events", '
        '"document_index_states", "document_upload_idempotency", '
        '"document_versions", "documents", "idempotency_records", '
        '"index_generation_cleanup_claims", "index_generation_creation_requests", '
        '"jobs", "knowledge_base_index_generations", "knowledge_base_mutations", '
        '"knowledge_bases", "model_profile_fallbacks", "model_profiles", '
        '"provider_configs", "provider_credentials", "provider_usage", "query_logs", '
        '"query_profiles", "sparse_profiles" CASCADE'
    )

    assert increment_2_tables == approved_tables
    assert set(Base.metadata.tables) == approved_tables
    assert increment_2_truncate_sql == approved_cleanup_sql


@pytest.mark.integration
def test_migration_round_trip_and_catalog_match_metadata(
    migration_postgres_urls: tuple[str, str],
    run_alembic: Callable[[str, str], None],
    increment_2_tables: frozenset[str],
) -> None:
    assert repr(migration_postgres_urls) == "PostgresUrls(<redacted>)"
    async_url, sync_url = migration_postgres_urls
    assert set(Base.metadata.tables) == increment_2_tables

    run_alembic(async_url, f"upgrade {PREVIOUS_REVISION}")
    with psycopg.connect(sync_url) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            PREVIOUS_REVISION,
        )
        legacy_provider_id = uuid4()
        legacy_knowledge_base_id = uuid4()
        legacy_generation_id = uuid4()
        legacy_running_job_id = uuid4()
        legacy_retry_wait_job_id = uuid4()
        legacy_terminal_job_id = uuid4()
        legacy_heartbeat = datetime(2026, 7, 25, 12, 0, tzinfo=UTC)
        connection.execute(
            """
            INSERT INTO provider_configs (
                id, name, provider_type, base_url, secret_ref,
                timeout_seconds, max_concurrency, requests_per_minute
            ) VALUES (%s, 'legacy provider', 'openai_compatible',
                      'https://example.invalid/v1', 'env:LEGACY_PROVIDER_KEY', 30, 1, 1)
            """,
            (legacy_provider_id,),
        )
        connection.execute(
            "INSERT INTO knowledge_bases (id, name) VALUES (%s, 'legacy active kb')",
            (legacy_knowledge_base_id,),
        )
        connection.execute(
            """
            INSERT INTO knowledge_base_index_generations (
                id, knowledge_base_id, index_profile_hash, qdrant_collection_name,
                status, rebuild_snapshot_at, caught_up_revision, validated_revision,
                validation_manifest_hash, expected_point_count, actual_point_count,
                validated_at, activated_at
            ) VALUES (%s, %s, %s, %s, 'active', %s, 10, 3, %s, 100, 1, %s, %s)
            """,
            (
                legacy_generation_id,
                legacy_knowledge_base_id,
                "a" * 64,
                f"legacy-collection-{uuid4()}",
                legacy_heartbeat,
                "b" * 64,
                legacy_heartbeat,
                legacy_heartbeat,
            ),
        )
        connection.execute(
            "UPDATE knowledge_bases SET active_index_generation_id = %s WHERE id = %s",
            (legacy_generation_id, legacy_knowledge_base_id),
        )
        connection.execute(
            """
            INSERT INTO jobs (
                id, target_type, target_id, operation, stage, status,
                progress_current, progress_total, attempt_count, worker_heartbeat_at
            ) VALUES
                (%s, 'knowledge_base', %s, 'rebuild_generation', 'indexing',
                 'running', 4, 10, 2, %s),
                (%s, 'knowledge_base', %s, 'rebuild_generation', 'validating',
                 'retry_wait', 6, 10, 3, %s),
                (%s, 'knowledge_base', %s, 'rebuild_generation', 'complete',
                 'succeeded', 10, 10, 4, %s)
            """,
            (
                legacy_running_job_id,
                uuid4(),
                legacy_heartbeat,
                legacy_retry_wait_job_id,
                uuid4(),
                legacy_heartbeat,
                legacy_terminal_job_id,
                uuid4(),
                legacy_heartbeat,
            ),
        )
        connection.commit()
        baseline_catalog = _snapshot_public_catalog(connection)
        baseline_schemas = _user_schemas(connection)
        baseline_tables = {
            row[0]
            for row in connection.execute(
                """
                SELECT tablename
                FROM pg_catalog.pg_tables
                WHERE schemaname = 'public' AND tablename <> 'alembic_version'
                """
            ).fetchall()
        }

    run_alembic(async_url, "upgrade head")
    engine = create_engine(sync_url.replace("postgresql://", "postgresql+psycopg://", 1))
    try:
        inspector = inspect(engine)
        actual_tables = set(inspector.get_table_names()) - {"alembic_version"}
        assert actual_tables == baseline_tables | increment_2_tables
        with engine.connect() as connection:
            legacy_provider = connection.execute(
                text(
                    "SELECT secret_ref, credential_id, resource_revision, "
                    "endpoint_policy_version, endpoint_validated_at "
                    "FROM provider_configs WHERE id = :provider_id"
                ),
                {"provider_id": legacy_provider_id},
            ).one()
        assert tuple(legacy_provider) == ("env:LEGACY_PROVIDER_KEY", None, 1, None, None)
        with psycopg.connect(sync_url) as connection:
            assert connection.execute(
                """
                SELECT status, caught_up_revision, validated_revision,
                       expected_point_count, actual_point_count,
                       distance, embedding_config_snapshot, embedding_config_hash,
                       validated_at, activated_at, safe_error_code, safe_error_message
                FROM knowledge_base_index_generations WHERE id = %s
                """,
                (legacy_generation_id,),
            ).fetchone() == (
                "failed",
                10,
                3,
                100,
                1,
                None,
                None,
                None,
                legacy_heartbeat,
                legacy_heartbeat,
                "migration_revalidation_required",
                "Generation deactivated during schema upgrade; rebuild required",
            )
            assert connection.execute(
                """
                SELECT active_index_generation_id, pending_index_generation_id
                FROM knowledge_bases WHERE id = %s
                """,
                (legacy_knowledge_base_id,),
            ).fetchone() == (None, None)
            migrated_jobs = connection.execute(
                """
                SELECT id, status, stage, resume_stage, progress_current, progress_total,
                       attempt_count, worker_heartbeat_at, lease_epoch,
                       lease_owner, lease_expires_at, next_retry_at
                FROM jobs WHERE id = ANY(%s) ORDER BY id
                """,
                ([legacy_running_job_id, legacy_retry_wait_job_id, legacy_terminal_job_id],),
            ).fetchall()
        jobs_by_id = {row[0]: row for row in migrated_jobs}
        running_job = jobs_by_id[legacy_running_job_id]
        assert running_job[1:11] == (
            "retry_wait",
            "indexing",
            "indexing",
            4,
            10,
            2,
            legacy_heartbeat,
            0,
            None,
            None,
        )
        assert running_job[11] is not None
        retry_wait_job = jobs_by_id[legacy_retry_wait_job_id]
        assert retry_wait_job[1:11] == (
            "retry_wait",
            "validating",
            None,
            6,
            10,
            3,
            legacy_heartbeat,
            0,
            None,
            None,
        )
        terminal_job = jobs_by_id[legacy_terminal_job_id]
        assert terminal_job[1:11] == (
            "succeeded",
            "complete",
            None,
            10,
            10,
            4,
            legacy_heartbeat,
            0,
            None,
            None,
        )

        reflected = MetaData()
        reflected.reflect(bind=engine)
        dialect = _postgres_dialect()
        for table_name in sorted(increment_2_tables):
            expected_table = Base.metadata.tables[table_name]
            actual_table = reflected.tables[table_name]
            assert list(actual_table.columns.keys()) == list(expected_table.columns.keys())
            for expected_column in expected_table.columns:
                actual_column = actual_table.columns[expected_column.name]
                assert actual_column.nullable == expected_column.nullable
                assert actual_column.type.compile(dialect=dialect) == expected_column.type.compile(
                    dialect=dialect
                )

        with engine.connect() as connection:
            migration_context = MigrationContext.configure(
                connection,
                opts={"compare_type": True, "compare_server_default": True},
            )
            drift = compare_metadata(migration_context, Base.metadata)
        approved_pointer_exceptions = {
            "fk_knowledge_bases_active_generation_same_parent",
            "fk_knowledge_bases_pending_generation_same_parent",
        }
        unexpected_drift = [
            difference
            for difference in drift
            if not (
                difference[0] == "remove_fk" and difference[1].name in approved_pointer_exceptions
            )
        ]
        assert unexpected_drift == []

        with engine.begin() as connection:
            connection.execute(text("CREATE SCHEMA orm_contract"))
            contract_connection = connection.execution_options(
                schema_translate_map={None: "orm_contract"}
            )
            # Let the same PostgreSQL parser canonicalize ORM DDL and migrated DDL,
            # then compare every named CHECK without weakening expression semantics.
            Base.metadata.create_all(bind=contract_connection)

        for table_name in sorted(increment_2_tables):
            expected_table = Base.metadata.tables[table_name]
            expected_checks = {
                constraint.name
                for constraint in expected_table.constraints
                if isinstance(constraint, CheckConstraint)
            }
            assert None not in expected_checks
            actual_checks = {
                constraint["name"]: _normalize_check_sql(constraint["sqltext"])
                for constraint in inspector.get_check_constraints(table_name)
            }
            orm_catalog_checks = {
                constraint["name"]: _normalize_check_sql(constraint["sqltext"])
                for constraint in inspector.get_check_constraints(
                    table_name,
                    schema="orm_contract",
                )
            }
            compiled_orm_checks = {
                constraint.name: str(constraint.sqltext.compile(dialect=_postgres_dialect()))
                for constraint in expected_table.constraints
                if isinstance(constraint, CheckConstraint)
            }
            assert set(compiled_orm_checks) == expected_checks
            assert set(actual_checks) == expected_checks
            assert actual_checks == orm_catalog_checks, compiled_orm_checks

        unnamed_fk_names: dict[tuple[str, tuple[str, ...]], str] = {
            ("api_keys", ("created_by_api_key_id",)): "fk_api_keys_created_by",
            ("api_keys", ("revoked_by_api_key_id",)): "fk_api_keys_revoked_by",
            (
                "api_key_knowledge_base_scopes",
                ("api_key_id",),
            ): "fk_api_key_kb_scopes_api_key",
            (
                "api_key_knowledge_base_scopes",
                ("knowledge_base_id",),
            ): "fk_api_key_kb_scopes_knowledge_base",
            (
                "api_key_query_profile_scopes",
                ("api_key_id",),
            ): "fk_api_key_query_profile_scopes_api_key",
            (
                "api_key_query_profile_scopes",
                ("query_profile_id",),
            ): "fk_api_key_query_profile_scopes_query_profile",
            ("audit_events", ("actor_api_key_id",)): "fk_audit_events_actor_api_key",
            ("idempotency_records", ("actor_key_id",)): "fk_idempotency_records_actor_key",
            (
                "knowledge_base_index_generations",
                ("knowledge_base_id",),
            ): "fk_kb_index_generations_knowledge_base",
            (
                "knowledge_base_index_generations",
                ("embedding_profile_id",),
            ): "fk_kb_index_generations_embedding_profile",
            (
                "knowledge_base_index_generations",
                ("sparse_profile_id",),
            ): "fk_kb_index_generations_sparse_profile",
            (
                "knowledge_base_mutations",
                ("knowledge_base_id",),
            ): "fk_knowledge_base_mutations_knowledge_base",
        }
        expected_foreign_keys = {
            (
                table_name,
                tuple(column.name for column in foreign_key.columns),
            ): (
                foreign_key.name
                or unnamed_fk_names[
                    (
                        table_name,
                        tuple(column.name for column in foreign_key.columns),
                    )
                ],
                foreign_key.ondelete,
                foreign_key.deferrable,
                foreign_key.initially,
            )
            for table_name in increment_2_tables
            for table in (Base.metadata.tables[table_name],)
            for foreign_key in table.foreign_key_constraints
        }
        actual_foreign_keys = {
            (table_name, tuple(foreign_key["constrained_columns"])): (
                foreign_key["name"],
                foreign_key["options"].get("ondelete"),
                foreign_key["options"].get("deferrable"),
                foreign_key["options"].get("initially"),
            )
            for table_name in increment_2_tables
            for foreign_key in inspector.get_foreign_keys(table_name)
            if not (foreign_key["name"] or "").startswith("fk_knowledge_bases_")
        }
        assert actual_foreign_keys == expected_foreign_keys

        unnamed_unique_names: dict[tuple[str, tuple[str, ...]], str] = {
            ("api_keys", ("public_id",)): "uq_api_keys_public_id",
            (
                "knowledge_base_index_generations",
                ("qdrant_collection_name",),
            ): "uq_kb_index_generations_qdrant_collection",
        }
        for table_name in increment_2_tables:
            table = Base.metadata.tables[table_name]
            expected_unique_constraints = {
                constraint.name
                or unnamed_unique_names[
                    (
                        table_name,
                        tuple(column.name for column in constraint.columns),
                    )
                ]
                for constraint in table.constraints
                if isinstance(constraint, UniqueConstraint)
            }
            actual_unique_constraints = {
                constraint["name"] for constraint in inspector.get_unique_constraints(table_name)
            }
            assert actual_unique_constraints == expected_unique_constraints
            assert inspector.get_pk_constraint(table_name)["name"] == f"{table_name}_pkey"

        expected_composite_foreign_keys = {
            "fk_knowledge_bases_active_generation_same_parent": (
                "knowledge_bases",
                ["active_index_generation_id", "id"],
                "knowledge_base_index_generations",
                ["id", "knowledge_base_id"],
            ),
            "fk_knowledge_bases_pending_generation_same_parent": (
                "knowledge_bases",
                ["pending_index_generation_id", "id"],
                "knowledge_base_index_generations",
                ["id", "knowledge_base_id"],
            ),
            "fk_documents_current_version_same_parent": (
                "documents",
                ["current_version_id", "id"],
                "document_versions",
                ["id", "document_id"],
            ),
            "fk_documents_pending_version_same_parent": (
                "documents",
                ["pending_version_id", "id"],
                "document_versions",
                ["id", "document_id"],
            ),
            "fk_document_versions_base_same_document": (
                "document_versions",
                ["base_version_id", "document_id"],
                "document_versions",
                ["id", "document_id"],
            ),
        }
        pointer_fks = {
            constraint["name"]: constraint
            for table_name in ("knowledge_bases", "documents", "document_versions")
            for constraint in inspector.get_foreign_keys(table_name)
            if constraint["name"] in expected_composite_foreign_keys
        }
        assert set(pointer_fks) == set(expected_composite_foreign_keys)
        for name, (
            table_name,
            columns,
            referred_table,
            referred_columns,
        ) in expected_composite_foreign_keys.items():
            pointer_fk = pointer_fks[name]
            assert pointer_fk["constrained_columns"] == columns
            assert pointer_fk["referred_table"] == referred_table
            assert pointer_fk["referred_columns"] == referred_columns
            assert pointer_fk["options"] == {
                "ondelete": "RESTRICT",
                "initially": "DEFERRED",
                "deferrable": True,
            }
            assert name in {
                constraint["name"] for constraint in inspector.get_foreign_keys(table_name)
            }

        expected_composite_uniques = {
            "knowledge_base_index_generations": {
                "uq_kb_index_generations_id_knowledge_base": ["id", "knowledge_base_id"]
            },
            "document_versions": {"uq_document_versions_id_document": ["id", "document_id"]},
        }
        for table_name, expected_uniques in expected_composite_uniques.items():
            actual_uniques = {
                constraint["name"]: constraint["column_names"]
                for constraint in inspector.get_unique_constraints(table_name)
            }
            for name, columns in expected_uniques.items():
                assert actual_uniques[name] == columns

        expected_indexes = {
            "uq_documents_kb_checksum_live": (
                ["knowledge_base_id", "checksum_sha256"],
                "((deleted_at IS NULL) AND (checksum_sha256 IS NOT NULL))",
                False,
            ),
            "uq_kb_index_generations_one_active": (
                ["knowledge_base_id"],
                "((status)::text = 'active'::text)",
                False,
            ),
            "uq_kb_index_generations_one_building": (
                ["knowledge_base_id"],
                "((status)::text = 'building'::text)",
                False,
            ),
            "uq_query_profiles_enabled_system_default": (
                ["is_system_default"],
                "(enabled AND is_system_default)",
                False,
            ),
            "uq_api_key_query_profile_default": (["api_key_id"], "is_default", False),
            "uq_jobs_active_target": (
                [
                    "operation",
                    "target_type",
                    "target_id",
                    "target_revision",
                    "index_generation_id",
                ],
                "((status)::text = ANY ((ARRAY['queued'::character varying, "
                "'running'::character varying, 'retry_wait'::character varying])::text[]))",
                True,
            ),
        }
        actual_indexes = {
            index["name"]: index
            for table_name in increment_2_tables
            for index in inspector.get_indexes(table_name)
            if index["name"] in expected_indexes
        }
        assert set(actual_indexes) == set(expected_indexes)
        for name, (columns, predicate, nulls_not_distinct) in expected_indexes.items():
            index = actual_indexes[name]
            assert index["unique"] is True
            assert index["column_names"] == columns
            assert index["dialect_options"]["postgresql_where"] == predicate
            assert (
                index["dialect_options"].get("postgresql_nulls_not_distinct", False)
                is nulls_not_distinct
            )
    finally:
        try:
            with engine.begin() as connection:
                connection.execute(text("DROP SCHEMA IF EXISTS orm_contract CASCADE"))
        finally:
            engine.dispose()

    run_alembic(async_url, f"downgrade {PREVIOUS_REVISION}")
    with psycopg.connect(sync_url) as connection:
        assert _snapshot_public_catalog(connection) == baseline_catalog
        assert _user_schemas(connection) == baseline_schemas
        assert connection.execute(
            "SELECT secret_ref FROM provider_configs WHERE id = %s",
            (legacy_provider_id,),
        ).fetchone() == ("env:LEGACY_PROVIDER_KEY",)
        assert connection.execute(
            "SELECT status FROM knowledge_base_index_generations WHERE id = %s",
            (legacy_generation_id,),
        ).fetchone() == ("failed",)
        assert connection.execute(
            "SELECT status, worker_heartbeat_at FROM jobs WHERE id = %s",
            (legacy_running_job_id,),
        ).fetchone() == ("retry_wait", legacy_heartbeat)

    run_alembic(async_url, "upgrade head")
    with psycopg.connect(sync_url) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            HEAD_REVISION,
        )


@pytest.mark.integration
@pytest.mark.parametrize("completed", [False, True], ids=["incomplete", "completed"])
def test_generation_cleanup_claim_blocks_downgrade_atomically(
    migration_postgres_urls: tuple[str, str],
    run_alembic: Callable[[str, str], None],
    completed: bool,
) -> None:
    async_url, sync_url = migration_postgres_urls
    run_alembic(async_url, "upgrade head")
    with psycopg.connect(sync_url) as connection:
        expected = _insert_generation_cleanup_claim(connection, completed=completed)
        connection.commit()

    with pytest.raises(subprocess.CalledProcessError) as raised:
        run_alembic(async_url, f"downgrade {GENERATION_CLEANUP_PARENT_REVISION}")
    assert "index-generation cleanup claims cannot be represented by revision 20260726_0003" in (
        raised.value.stderr or ""
    )

    with psycopg.connect(sync_url) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            HEAD_REVISION,
        )
        assert connection.execute(
            "SELECT to_regclass('public.index_generation_cleanup_claims')"
        ).fetchone() == ("index_generation_cleanup_claims",)
        assert (
            connection.execute(
                """
            SELECT collection_name, knowledge_base_id, generation_id, lease_owner,
                   lease_expires_at, completed_at
            FROM index_generation_cleanup_claims
            """
            ).fetchone()
            == expected
        )


@pytest.mark.integration
def test_generation_cleanup_claim_downgrade_lock_closes_concurrent_insert_race(
    migration_postgres_urls: tuple[str, str],
    run_alembic: Callable[[str, str], None],
) -> None:
    async_url, sync_url = migration_postgres_urls
    run_alembic(async_url, "upgrade head")
    downgrade_errors: list[BaseException] = []

    def downgrade() -> None:
        try:
            run_alembic(async_url, f"downgrade {GENERATION_CLEANUP_PARENT_REVISION}")
        except BaseException as error:
            downgrade_errors.append(error)

    with (
        psycopg.connect(sync_url) as inserter,
        psycopg.connect(sync_url, autocommit=True) as observer,
    ):
        expected = _insert_generation_cleanup_claim(inserter, completed=False)
        downgrade_thread = threading.Thread(target=downgrade, daemon=True)
        downgrade_thread.start()
        try:
            deadline = time.monotonic() + 10
            poll_interval = threading.Event()
            waiting_for_exclusive_lock = False
            while downgrade_thread.is_alive() and time.monotonic() < deadline:
                waiting_for_exclusive_lock = observer.execute(
                    """
                        SELECT EXISTS (
                            SELECT 1
                            FROM pg_catalog.pg_locks AS locks
                            JOIN pg_catalog.pg_class AS relation
                              ON relation.oid = locks.relation
                            WHERE relation.relname = 'index_generation_cleanup_claims'
                              AND locks.mode = 'AccessExclusiveLock'
                              AND NOT locks.granted
                        )
                        """
                ).fetchone() == (True,)
                if waiting_for_exclusive_lock:
                    break
                poll_interval.wait(0.01)
            assert waiting_for_exclusive_lock is True
            inserter.commit()
        finally:
            if not waiting_for_exclusive_lock:
                inserter.rollback()
            downgrade_thread.join(timeout=30)

    assert not downgrade_thread.is_alive()
    assert len(downgrade_errors) == 1
    error = downgrade_errors[0]
    assert isinstance(error, subprocess.CalledProcessError)
    assert "index-generation cleanup claims cannot be represented by revision 20260726_0003" in (
        error.stderr or ""
    )
    with psycopg.connect(sync_url) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            HEAD_REVISION,
        )
        assert (
            connection.execute(
                """
            SELECT collection_name, knowledge_base_id, generation_id, lease_owner,
                   lease_expires_at, completed_at
            FROM index_generation_cleanup_claims
            """
            ).fetchone()
            == expected
        )


@pytest.mark.integration
def test_generation_cleanup_claim_empty_table_downgrade_round_trip(
    migration_postgres_urls: tuple[str, str],
    run_alembic: Callable[[str, str], None],
) -> None:
    async_url, sync_url = migration_postgres_urls
    run_alembic(async_url, "upgrade head")

    run_alembic(async_url, f"downgrade {GENERATION_CLEANUP_PARENT_REVISION}")
    with psycopg.connect(sync_url) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            GENERATION_CLEANUP_PARENT_REVISION,
        )
        assert connection.execute(
            "SELECT to_regclass('public.index_generation_cleanup_claims')"
        ).fetchone() == (None,)

    run_alembic(async_url, "upgrade head")
    with psycopg.connect(sync_url) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            HEAD_REVISION,
        )
        assert connection.execute(
            "SELECT to_regclass('public.index_generation_cleanup_claims')"
        ).fetchone() == ("index_generation_cleanup_claims",)


@pytest.mark.integration
def test_job_actor_blocks_downgrade_atomically_until_removed(
    migration_postgres_urls: tuple[str, str],
    run_alembic: Callable[[str, str], None],
) -> None:
    async_url, sync_url = migration_postgres_urls
    run_alembic(async_url, "upgrade head")
    actor_id = uuid4()
    job_id = uuid4()
    with psycopg.connect(sync_url) as connection:
        connection.execute(
            """
            INSERT INTO api_keys (id, public_id, secret_digest, key_type, name)
            VALUES (%s, 'downgradeactor01', %s, 'admin', 'Downgrade actor')
            """,
            (actor_id, b"a" * 32),
        )
        connection.execute(
            """
            INSERT INTO jobs (
                id, actor_api_key_id, target_type, target_id, operation
            ) VALUES (%s, %s, 'document_version', %s, 'ingest_document')
            """,
            (job_id, actor_id, uuid4()),
        )
        connection.commit()

    with pytest.raises(subprocess.CalledProcessError) as raised:
        run_alembic(async_url, f"downgrade {HEAD_PARENT_REVISION}")
    assert "job actors cannot be represented by revision 20260728_0004" in (
        raised.value.stderr or ""
    )
    with psycopg.connect(sync_url) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            HEAD_REVISION,
        )
        assert connection.execute(
            "SELECT actor_api_key_id FROM jobs WHERE id = %s",
            (job_id,),
        ).fetchone() == (actor_id,)
        connection.execute(
            "UPDATE jobs SET actor_api_key_id = NULL WHERE id = %s",
            (job_id,),
        )
        connection.commit()

    run_alembic(async_url, f"downgrade {HEAD_PARENT_REVISION}")
    with psycopg.connect(sync_url) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            HEAD_PARENT_REVISION,
        )
        assert connection.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'jobs'
                  AND column_name = 'actor_api_key_id'
            )
            """
        ).fetchone() == (False,)


@pytest.mark.integration
def test_generation_cleanup_claim_requires_canonical_collection_identity(
    migrated_sync_connection: psycopg.Connection[Any],
) -> None:
    connection = migrated_sync_connection
    knowledge_base_id = uuid4()
    generation_id = uuid4()
    lease_owner = uuid4()
    canonical_name = f"rag_kb_{knowledge_base_id.hex}_g_{generation_id.hex}"
    insert_sql = """
        INSERT INTO index_generation_cleanup_claims (
            collection_name, knowledge_base_id, generation_id, lease_owner,
            lease_epoch, lease_expires_at
        ) VALUES (%s, %s, %s, %s, 1, %s)
    """
    lease_expires_at = datetime.now(UTC) - timedelta(minutes=1)

    _assert_constraint_violation(
        connection,
        insert_sql,
        (
            f"rag_kb_{knowledge_base_id.hex}_x_{generation_id.hex}",
            knowledge_base_id,
            generation_id,
            lease_owner,
            lease_expires_at,
        ),
    )
    _assert_constraint_violation(
        connection,
        insert_sql,
        (
            canonical_name,
            uuid4(),
            generation_id,
            lease_owner,
            lease_expires_at,
        ),
    )

    connection.execute(
        insert_sql,
        (
            canonical_name,
            knowledge_base_id,
            generation_id,
            lease_owner,
            lease_expires_at,
        ),
    )
    connection.commit()
    assert connection.execute(
        """
        SELECT knowledge_base_id, generation_id
        FROM index_generation_cleanup_claims
        WHERE collection_name = %s
        """,
        (canonical_name,),
    ).fetchone() == (knowledge_base_id, generation_id)


@pytest.mark.integration
def test_credential_only_provider_downgrade_fails_atomically_at_head(
    migration_postgres_urls: tuple[str, str],
    run_alembic: Callable[[str, str], None],
) -> None:
    async_url, sync_url = migration_postgres_urls
    run_alembic(async_url, "upgrade head")
    credential_id = uuid4()
    provider_id = uuid4()
    with psycopg.connect(sync_url) as connection:
        connection.execute(
            """
            INSERT INTO provider_credentials
                (id, name, ciphertext, nonce, key_version)
            VALUES (%s, 'downgrade guard', %s, %s, 'v1')
            """,
            (credential_id, b"encrypted", b"n" * 12),
        )
        connection.execute(
            """
            INSERT INTO provider_configs (
                id, name, provider_type, base_url, credential_id,
                endpoint_policy_version, endpoint_validated_at,
                timeout_seconds, max_concurrency, requests_per_minute
            ) VALUES (%s, 'credential only', 'openai_compatible',
                      'https://example.invalid/v1', %s, 'v1', %s, 30, 1, 1)
            """,
            (provider_id, credential_id, datetime.now(UTC)),
        )
        connection.commit()

    with pytest.raises(subprocess.CalledProcessError) as raised:
        run_alembic(async_url, f"downgrade {PREVIOUS_REVISION}")
    assert "credential-backed provider configs cannot be represented by revision 20260724_0002" in (
        raised.value.stderr or ""
    )
    assert "replace credential_id with secret_ref before retrying the downgrade" in (
        raised.value.stderr or ""
    )

    with psycopg.connect(sync_url) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            HEAD_REVISION,
        )
        assert connection.execute(
            """
            SELECT credential_id, secret_ref, endpoint_policy_version
            FROM provider_configs WHERE id = %s
            """,
            (provider_id,),
        ).fetchone() == (credential_id, None, "v1")
        assert connection.execute(
            "SELECT ciphertext, nonce, key_version FROM provider_credentials WHERE id = %s",
            (credential_id,),
        ).fetchone() == (b"encrypted", b"n" * 12, "v1")


@pytest.mark.integration
def test_downgrade_lock_prevents_credential_insert_race(
    migration_postgres_urls: tuple[str, str],
    run_alembic: Callable[[str, str], None],
) -> None:
    async_url, sync_url = migration_postgres_urls
    run_alembic(async_url, f"upgrade {GENERATION_CLEANUP_PARENT_REVISION}")
    downgrade_errors: list[BaseException] = []

    def downgrade() -> None:
        try:
            run_alembic(async_url, f"downgrade {PREVIOUS_REVISION}")
        except BaseException as error:
            downgrade_errors.append(error)

    with (
        psycopg.connect(sync_url) as blocker,
        psycopg.connect(sync_url, autocommit=True) as observer,
    ):
        blocker.execute("LOCK TABLE provider_credentials IN ACCESS SHARE MODE")
        downgrade_thread = threading.Thread(target=downgrade, daemon=True)
        downgrade_thread.start()
        try:
            deadline = time.monotonic() + 10
            poll_interval = threading.Event()
            lock_state: tuple[int, int] | None = None
            while downgrade_thread.is_alive() and time.monotonic() < deadline:
                lock_state = observer.execute(
                    """
                    SELECT locks.pid,
                           count(*) FILTER (
                               WHERE relation.relname NOT IN (
                                   'provider_configs', 'provider_credentials'
                               )
                                 AND locks.mode = 'AccessExclusiveLock'
                                 AND locks.granted
                           )
                    FROM pg_catalog.pg_locks AS locks
                    JOIN pg_catalog.pg_class AS relation ON relation.oid = locks.relation
                    GROUP BY locks.pid
                    HAVING bool_or(
                               relation.relname = 'provider_configs'
                               AND locks.mode = 'AccessExclusiveLock'
                               AND locks.granted
                           )
                       AND bool_or(
                               relation.relname = 'provider_credentials'
                               AND locks.mode = 'AccessExclusiveLock'
                               AND NOT locks.granted
                           )
                    """,
                ).fetchone()
                if lock_state is not None:
                    break
                poll_interval.wait(0.01)
            assert lock_state is not None
            _downgrade_pid, prior_schema_mutation_locks = lock_state
            assert prior_schema_mutation_locks == 0

            with psycopg.connect(sync_url) as contender:
                contender.execute("SET lock_timeout = '500ms'")
                with pytest.raises(psycopg.errors.LockNotAvailable):
                    contender.execute(
                        """
                        INSERT INTO provider_credentials
                            (id, name, ciphertext, nonce, key_version)
                        VALUES (%s, 'racing credential', %s, %s, 'v1')
                        """,
                        (uuid4(), b"encrypted", b"n" * 12),
                    )
                contender.rollback()
        finally:
            try:
                blocker.rollback()
            finally:
                downgrade_thread.join(timeout=30)

    assert not downgrade_thread.is_alive()
    assert downgrade_errors == []
    with psycopg.connect(sync_url) as connection:
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone() == (
            PREVIOUS_REVISION,
        )
        assert connection.execute(
            "SELECT to_regclass('public.provider_credentials')"
        ).fetchone() == (None,)


@pytest.mark.integration
def test_public_scope_mutation_and_default_uniqueness(
    migrated_sync_connection: psycopg.Connection[Any],
) -> None:
    connection = migrated_sync_connection
    public_id = f"rag_agent_{uuid4().hex}"
    key_id = _insert_api_key(connection, public_id=public_id)
    knowledge_base_id = _insert_knowledge_base(connection, "kb uniqueness")
    first_profile_id = _insert_query_profile(connection, "query one")
    second_profile_id = _insert_query_profile(connection, "query two")
    connection.commit()

    _assert_constraint_violation(
        connection,
        """
        INSERT INTO api_keys (
            id, public_id, secret_digest, key_type, name,
            requests_per_minute, max_concurrency
        ) VALUES (%s, %s, %s, 'agent', 'duplicate', 100, 10)
        """,
        (uuid4(), public_id, b"b" * 32),
    )

    connection.execute(
        "INSERT INTO api_key_knowledge_base_scopes (api_key_id, knowledge_base_id) VALUES (%s, %s)",
        (key_id, knowledge_base_id),
    )
    connection.execute(
        "INSERT INTO knowledge_base_mutations "
        "(id, knowledge_base_id, revision, mutation_type, target_type, target_id) "
        "VALUES (%s, %s, 1, 'metadata_changed', 'knowledge_base', %s)",
        (uuid4(), knowledge_base_id, knowledge_base_id),
    )
    connection.execute(
        "INSERT INTO api_key_query_profile_scopes "
        "(api_key_id, query_profile_id, is_default) VALUES (%s, %s, true)",
        (key_id, first_profile_id),
    )
    connection.commit()

    _assert_constraint_violation(
        connection,
        "INSERT INTO api_key_knowledge_base_scopes (api_key_id, knowledge_base_id) VALUES (%s, %s)",
        (key_id, knowledge_base_id),
    )
    _assert_constraint_violation(
        connection,
        "INSERT INTO knowledge_base_mutations "
        "(id, knowledge_base_id, revision, mutation_type, target_type, target_id) "
        "VALUES (%s, %s, 1, 'metadata_changed', 'knowledge_base', %s)",
        (uuid4(), knowledge_base_id, knowledge_base_id),
    )
    _assert_constraint_violation(
        connection,
        "INSERT INTO api_key_query_profile_scopes "
        "(api_key_id, query_profile_id, is_default) VALUES (%s, %s, true)",
        (key_id, second_profile_id),
    )

    default_scope = connection.execute(
        "SELECT api_key_id, query_profile_id, is_default "
        "FROM api_key_query_profile_scopes WHERE api_key_id = %s",
        (key_id,),
    ).fetchone()
    assert default_scope == (key_id, first_profile_id, True)


@pytest.mark.integration
def test_generation_and_job_partial_uniqueness(
    migrated_sync_connection: psycopg.Connection[Any],
) -> None:
    connection = migrated_sync_connection
    knowledge_base_id = _insert_knowledge_base(connection, "kb generation")
    _insert_generation(connection, knowledge_base_id, status="active")
    _insert_generation(connection, knowledge_base_id, status="building")
    target_id = uuid4()
    connection.execute(
        """
        INSERT INTO jobs (id, target_type, target_id, operation)
        VALUES (%s, 'knowledge_base', %s, 'rebuild_generation')
        """,
        (uuid4(), target_id),
    )
    connection.commit()

    for status in ("active", "building"):
        _assert_constraint_violation(
            connection,
            """
            INSERT INTO knowledge_base_index_generations (
                id, knowledge_base_id, index_profile_hash,
                qdrant_collection_name, status, rebuild_snapshot_at
            ) VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                uuid4(),
                knowledge_base_id,
                "c" * 64,
                f"collection-{uuid4()}",
                status,
                datetime.now(UTC),
            ),
        )

    _assert_constraint_violation(
        connection,
        """
        INSERT INTO jobs (id, target_type, target_id, operation)
        VALUES (%s, 'knowledge_base', %s, 'rebuild_generation')
        """,
        (uuid4(), target_id),
    )

    for terminal_status in ("succeeded", "failed"):
        connection.execute(
            """
            INSERT INTO jobs (id, target_type, target_id, operation, status)
            VALUES (%s, 'knowledge_base', %s, 'rebuild_generation', %s)
            """,
            (uuid4(), target_id, terminal_status),
        )
    connection.commit()


@pytest.mark.integration
def test_enabled_system_default_query_profile_is_globally_unique(
    migrated_sync_connection: psycopg.Connection[Any],
) -> None:
    connection = migrated_sync_connection
    _insert_query_profile(
        connection,
        "enabled default one",
        enabled=True,
        is_system_default=True,
    )
    _insert_query_profile(
        connection,
        "disabled default",
        enabled=False,
        is_system_default=True,
    )
    _insert_query_profile(
        connection,
        "enabled non-default",
        enabled=True,
        is_system_default=False,
    )
    connection.commit()

    with pytest.raises(psycopg.errors.UniqueViolation):
        _insert_query_profile(
            connection,
            "enabled default two",
            enabled=True,
            is_system_default=True,
        )
        connection.commit()
    connection.rollback()

    rows = connection.execute(
        "SELECT enabled, is_system_default, count(*) "
        "FROM query_profiles GROUP BY enabled, is_system_default "
        "ORDER BY enabled, is_system_default"
    ).fetchall()
    assert rows == [(False, True, 1), (True, False, 1), (True, True, 1)]


@pytest.mark.integration
def test_deferred_same_parent_foreign_keys_fail_at_commit(
    migrated_sync_connection: psycopg.Connection[Any],
) -> None:
    connection = migrated_sync_connection
    first_kb_id = _insert_knowledge_base(connection, "kb one")
    second_kb_id = _insert_knowledge_base(connection, "kb two")
    generation_id = _insert_generation(connection, first_kb_id, status="active")
    first_document_id = _insert_document(connection, first_kb_id)
    second_document_id = _insert_document(connection, second_kb_id)
    first_version_id = _insert_document_version(connection, first_document_id)
    connection.commit()

    _assert_constraint_violation(
        connection,
        "UPDATE knowledge_bases SET active_index_generation_id = %s WHERE id = %s",
        (generation_id, second_kb_id),
    )
    _assert_constraint_violation(
        connection,
        "UPDATE knowledge_bases SET pending_index_generation_id = %s WHERE id = %s",
        (generation_id, second_kb_id),
    )
    _assert_constraint_violation(
        connection,
        "UPDATE documents SET current_version_id = %s WHERE id = %s",
        (first_version_id, second_document_id),
    )
    _assert_constraint_violation(
        connection,
        "UPDATE documents SET pending_version_id = %s WHERE id = %s",
        (first_version_id, second_document_id),
    )
    _assert_constraint_violation(
        connection,
        """
        INSERT INTO document_versions (
            id, document_id, version_number, source_object_key,
            source_checksum_sha256, base_version_id
        ) VALUES (%s, %s, 1, %s, %s, %s)
        """,
        (
            uuid4(),
            second_document_id,
            f"source/{uuid4()}",
            "d" * 64,
            first_version_id,
        ),
    )


@pytest.mark.integration
def test_idempotency_reservations_can_precede_resources_but_require_them_at_commit(
    migrated_sync_connection: psycopg.Connection[Any],
) -> None:
    connection = migrated_sync_connection
    actor_key_id = _insert_api_key(connection)
    knowledge_base_id = _insert_knowledge_base(connection, "reservation ordering kb")
    connection.commit()

    generation_id = uuid4()
    connection.execute(
        """
        INSERT INTO index_generation_creation_requests (
            id, actor_api_key_id, knowledge_base_id, idempotency_key,
            request_fingerprint, generation_id
        ) VALUES (%s, %s, %s, 'generation-ordering', %s, %s)
        """,
        (uuid4(), actor_key_id, knowledge_base_id, b"g" * 32, generation_id),
    )
    connection.execute(
        """
        INSERT INTO knowledge_base_index_generations (
            id, knowledge_base_id, index_profile_hash,
            qdrant_collection_name, status, rebuild_snapshot_at
        ) VALUES (%s, %s, %s, %s, 'building', %s)
        """,
        (
            generation_id,
            knowledge_base_id,
            "a" * 64,
            f"collection-{uuid4()}",
            datetime.now(UTC),
        ),
    )
    connection.commit()

    with pytest.raises(psycopg.errors.UniqueViolation):
        connection.execute(
            """
            INSERT INTO index_generation_creation_requests (
                id, actor_api_key_id, knowledge_base_id, idempotency_key,
                request_fingerprint, generation_id
            ) VALUES (%s, %s, %s, 'generation-ordering', %s, %s)
            """,
            (uuid4(), actor_key_id, knowledge_base_id, b"h" * 32, generation_id),
        )
    connection.rollback()

    document_id = uuid4()
    document_version_id = uuid4()
    job_id = uuid4()
    connection.execute(
        """
        INSERT INTO document_upload_idempotency (
            id, actor_api_key_id, knowledge_base_id, idempotency_key,
            request_fingerprint, document_id, document_version_id, job_id
        ) VALUES (%s, %s, %s, 'upload-ordering', %s, %s, %s, %s)
        """,
        (
            uuid4(),
            actor_key_id,
            knowledge_base_id,
            b"u" * 32,
            document_id,
            document_version_id,
            job_id,
        ),
    )
    connection.execute(
        "INSERT INTO documents (id, knowledge_base_id, display_name) VALUES (%s, %s, 'doc')",
        (document_id, knowledge_base_id),
    )
    connection.execute(
        """
        INSERT INTO document_versions (
            id, document_id, version_number, source_object_key, source_checksum_sha256
        ) VALUES (%s, %s, 1, %s, %s)
        """,
        (document_version_id, document_id, f"source/{document_version_id}", "b" * 64),
    )
    connection.execute(
        """
        INSERT INTO jobs (id, target_type, target_id, operation)
        VALUES (%s, 'document_version', %s, 'ingest_document')
        """,
        (job_id, document_version_id),
    )
    connection.commit()

    with pytest.raises(psycopg.errors.UniqueViolation):
        connection.execute(
            """
            INSERT INTO document_upload_idempotency (
                id, actor_api_key_id, knowledge_base_id, idempotency_key,
                request_fingerprint, document_id, document_version_id, job_id
            ) VALUES (%s, %s, %s, 'upload-ordering', %s, %s, %s, %s)
            """,
            (
                uuid4(),
                actor_key_id,
                knowledge_base_id,
                b"v" * 32,
                document_id,
                document_version_id,
                job_id,
            ),
        )
    connection.rollback()

    connection.execute(
        """
        INSERT INTO index_generation_creation_requests (
            id, actor_api_key_id, knowledge_base_id, idempotency_key,
            request_fingerprint, generation_id
        ) VALUES (%s, %s, %s, 'missing-generation', %s, %s)
        """,
        (uuid4(), actor_key_id, knowledge_base_id, b"m" * 32, uuid4()),
    )
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        connection.commit()
    connection.rollback()

    connection.execute(
        """
        INSERT INTO document_upload_idempotency (
            id, actor_api_key_id, knowledge_base_id, idempotency_key,
            request_fingerprint, document_id, document_version_id, job_id
        ) VALUES (%s, %s, %s, 'missing-upload-resources', %s, %s, %s, %s)
        """,
        (uuid4(), actor_key_id, knowledge_base_id, b"n" * 32, uuid4(), uuid4(), uuid4()),
    )
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        connection.commit()
    connection.rollback()


@pytest.mark.integration
def test_idempotency_response_references_reject_cross_parent_resources_at_commit(
    migrated_sync_connection: psycopg.Connection[Any],
) -> None:
    connection = migrated_sync_connection
    actor_key_id = _insert_api_key(connection)
    first_kb_id = _insert_knowledge_base(connection, "idempotency parent one")
    second_kb_id = _insert_knowledge_base(connection, "idempotency parent two")
    generation_id = _insert_generation(connection, first_kb_id, status="building")
    first_document_id = _insert_document(connection, first_kb_id)
    second_document_id = _insert_document(connection, second_kb_id)
    first_version_id = _insert_document_version(connection, first_document_id)
    second_version_id = _insert_document_version(connection, second_document_id)
    first_job_id = uuid4()
    second_job_id = uuid4()
    connection.execute(
        """
        INSERT INTO jobs (id, target_type, target_id, operation) VALUES
            (%s, 'document_version', %s, 'ingest_document'),
            (%s, 'document_version', %s, 'ingest_document')
        """,
        (first_job_id, first_version_id, second_job_id, second_version_id),
    )
    connection.commit()

    connection.execute(
        """
        INSERT INTO index_generation_creation_requests (
            id, actor_api_key_id, knowledge_base_id, idempotency_key,
            request_fingerprint, generation_id
        ) VALUES (%s, %s, %s, 'cross-kb-generation', %s, %s)
        """,
        (uuid4(), actor_key_id, second_kb_id, b"x" * 32, generation_id),
    )
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        connection.commit()
    connection.rollback()

    connection.execute(
        """
        INSERT INTO document_upload_idempotency (
            id, actor_api_key_id, knowledge_base_id, idempotency_key,
            request_fingerprint, document_id, document_version_id, job_id
        ) VALUES (%s, %s, %s, 'cross-kb-document', %s, %s, %s, %s)
        """,
        (
            uuid4(),
            actor_key_id,
            second_kb_id,
            b"y" * 32,
            first_document_id,
            first_version_id,
            first_job_id,
        ),
    )
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        connection.commit()
    connection.rollback()

    connection.execute(
        """
        INSERT INTO document_upload_idempotency (
            id, actor_api_key_id, knowledge_base_id, idempotency_key,
            request_fingerprint, document_id, document_version_id, job_id
        ) VALUES (%s, %s, %s, 'wrong-version-parent', %s, %s, %s, %s)
        """,
        (
            uuid4(),
            actor_key_id,
            first_kb_id,
            b"z" * 32,
            first_document_id,
            second_version_id,
            second_job_id,
        ),
    )
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        connection.commit()
    connection.rollback()


@pytest.mark.integration
def test_scope_delete_actions_are_restrict_and_cascade(
    migrated_sync_connection: psycopg.Connection[Any],
) -> None:
    connection = migrated_sync_connection
    key_id = _insert_api_key(connection)
    knowledge_base_id = _insert_knowledge_base(connection, "kb scoped")
    connection.execute(
        "INSERT INTO api_key_knowledge_base_scopes (api_key_id, knowledge_base_id) VALUES (%s, %s)",
        (key_id, knowledge_base_id),
    )
    connection.commit()

    _assert_constraint_violation(
        connection,
        "DELETE FROM knowledge_bases WHERE id = %s",
        (knowledge_base_id,),
    )
    connection.execute("DELETE FROM api_keys WHERE id = %s", (key_id,))
    connection.commit()
    scope_count = connection.execute(
        "SELECT count(*) FROM api_key_knowledge_base_scopes WHERE api_key_id = %s",
        (key_id,),
    ).fetchone()
    assert scope_count == (0,)


@pytest.mark.integration
def test_enums_nonnegative_values_and_binary_lengths(
    migrated_sync_connection: psycopg.Connection[Any],
) -> None:
    connection = migrated_sync_connection
    key_id = _insert_api_key(connection)
    knowledge_base_id = _insert_knowledge_base(connection, "kb checks")
    connection.commit()

    invalid_statements = (
        (
            "INSERT INTO knowledge_bases (id, name, status) VALUES (%s, 'bad', 'unknown')",
            (uuid4(),),
        ),
        (
            "INSERT INTO api_keys "
            "(id, public_id, secret_digest, key_type, name, requests_per_minute, max_concurrency) "
            "VALUES (%s, %s, %s, 'agent', 'bad digest', 1, 1)",
            (uuid4(), f"rag_agent_{uuid4().hex}", b"short"),
        ),
        (
            "INSERT INTO idempotency_records "
            "(id, actor_key_id, operation, idempotency_key, request_fingerprint, "
            "result_resource_type, result_resource_id, http_status) "
            "VALUES (%s, %s, 'ingest', 'key', %s, 'document', %s, 202)",
            (uuid4(), key_id, b"short", uuid4()),
        ),
        (
            "INSERT INTO knowledge_base_mutations "
            "(id, knowledge_base_id, revision, mutation_type, target_type, target_id) "
            "VALUES (%s, %s, -1, 'metadata_changed', 'knowledge_base', %s)",
            (uuid4(), knowledge_base_id, knowledge_base_id),
        ),
        (
            "INSERT INTO provider_usage "
            "(id, request_id, capability, provider_identifier, model_identifier, "
            "currency, latency_ms, status) "
            "VALUES (%s, 'req', 'unknown', 'provider', 'model', 'USD', 1, 'succeeded')",
            (uuid4(),),
        ),
        (
            "INSERT INTO provider_configs "
            "(id, name, provider_type, base_url, secret_ref, timeout_seconds, "
            "max_concurrency, requests_per_minute) "
            "VALUES (%s, %s, 'unknown', 'https://example.invalid/v1', "
            "'env:SYNTHETIC_KEY', 30, 1, 1)",
            (uuid4(), f"provider-{uuid4()}"),
        ),
        (
            "INSERT INTO provider_configs "
            "(id, name, provider_type, base_url, secret_ref, timeout_seconds, "
            "max_concurrency, requests_per_minute) "
            "VALUES (%s, %s, 'openrouter', 'https://example.invalid/v1', "
            "'env:SYNTHETIC_KEY', 30, -1, 1)",
            (uuid4(), f"provider-{uuid4()}"),
        ),
        (
            "INSERT INTO provider_usage "
            "(id, request_id, capability, provider_identifier, model_identifier, "
            "cost_micros, currency, latency_ms, status) "
            "VALUES (%s, 'req', 'embedding', 'provider', 'model', -1, 'USD', 1, 'succeeded')",
            (uuid4(),),
        ),
        (
            "INSERT INTO provider_usage "
            "(id, request_id, capability, provider_identifier, model_identifier, "
            "currency, latency_ms, status) "
            "VALUES (%s, 'req', 'embedding', 'provider', 'model', 'USD', 1, 'unknown')",
            (uuid4(),),
        ),
        (
            "INSERT INTO knowledge_base_index_generations "
            "(id, knowledge_base_id, index_profile_hash, qdrant_collection_name, status, "
            "rebuild_snapshot_at, expected_point_count) "
            "VALUES (%s, %s, %s, %s, 'failed', %s, -1)",
            (
                uuid4(),
                knowledge_base_id,
                "e" * 64,
                f"collection-{uuid4()}",
                datetime.now(UTC),
            ),
        ),
    )
    for sql, parameters in invalid_statements:
        _assert_constraint_violation(connection, sql, parameters)


@pytest.mark.integration
def test_server_defaults_and_json_values_are_independent(
    migrated_sync_connection: psycopg.Connection[Any],
) -> None:
    connection = migrated_sync_connection
    first_id = _insert_knowledge_base(connection, "defaults one")
    second_id = _insert_knowledge_base(connection, "defaults two")
    key_id = _insert_api_key(connection)
    connection.commit()

    defaults = connection.execute(
        """
        SELECT status, metadata, filter_schema, resource_revision,
               mutation_revision, filter_schema_revision
        FROM knowledge_bases WHERE id = %s
        """,
        (first_id,),
    ).fetchone()
    assert defaults == ("active", {}, {"fields": []}, 1, 0, 0)

    key_defaults = connection.execute(
        "SELECT status, capabilities, raw_file_read, resource_revision FROM api_keys WHERE id = %s",
        (key_id,),
    ).fetchone()
    assert key_defaults == ("active", [], False, 1)

    connection.execute(
        'UPDATE knowledge_bases SET metadata = \'{"owner": "one"}\'::jsonb WHERE id = %s',
        (first_id,),
    )
    second_metadata = connection.execute(
        "SELECT metadata, filter_schema FROM knowledge_bases WHERE id = %s",
        (second_id,),
    ).fetchone()
    assert second_metadata == ({}, {"fields": []})


@pytest.mark.integration
def test_provider_credential_source_and_endpoint_policy_invariants(
    migrated_sync_connection: psycopg.Connection[Any],
) -> None:
    connection = migrated_sync_connection
    credential_id = uuid4()
    connection.execute(
        """
        INSERT INTO provider_credentials
            (id, name, ciphertext, nonce, key_version)
        VALUES (%s, 'primary', %s, %s, 'v1')
        """,
        (credential_id, b"ciphertext", b"n" * 12),
    )
    connection.execute(
        """
        INSERT INTO provider_configs (
            id, name, provider_type, base_url, credential_id,
            endpoint_policy_version, endpoint_validated_at,
            timeout_seconds, max_concurrency, requests_per_minute
        ) VALUES (%s, %s, 'openai_compatible', 'https://example.invalid/v1', %s,
                  'v1', %s, 30, 1, 1)
        """,
        (uuid4(), f"credential-provider-{uuid4()}", credential_id, datetime.now(UTC)),
    )
    connection.commit()

    _assert_constraint_violation(
        connection,
        """
        INSERT INTO provider_configs (
            id, name, provider_type, base_url, secret_ref, credential_id,
            endpoint_policy_version, endpoint_validated_at,
            timeout_seconds, max_concurrency, requests_per_minute
        ) VALUES (%s, %s, 'openai_compatible', 'https://example.invalid/v1',
                  'env:BOTH', %s, 'v1', %s, 30, 1, 1)
        """,
        (uuid4(), f"both-provider-{uuid4()}", credential_id, datetime.now(UTC)),
    )
    _assert_constraint_violation(
        connection,
        """
        INSERT INTO provider_configs (
            id, name, provider_type, base_url, credential_id,
            timeout_seconds, max_concurrency, requests_per_minute
        ) VALUES (%s, %s, 'openai_compatible', 'https://example.invalid/v1', %s, 30, 1, 1)
        """,
        (uuid4(), f"unvalidated-provider-{uuid4()}", credential_id),
    )
    _assert_constraint_violation(
        connection,
        """
        INSERT INTO provider_configs (
            id, name, provider_type, base_url, credential_id,
            endpoint_policy_version, endpoint_validated_at,
            timeout_seconds, max_concurrency, requests_per_minute
        ) VALUES (%s, %s, 'openai_compatible', 'https://example.invalid/v1', %s,
                  '', %s, 30, 1, 1)
        """,
        (
            uuid4(),
            f"empty-policy-provider-{uuid4()}",
            credential_id,
            datetime.now(UTC),
        ),
    )


@pytest.mark.integration
def test_active_generation_and_job_lease_invariants(
    migrated_sync_connection: psycopg.Connection[Any],
) -> None:
    connection = migrated_sync_connection
    knowledge_base_id = _insert_knowledge_base(connection, "invariant kb")
    connection.commit()
    _assert_constraint_violation(
        connection,
        """
        INSERT INTO knowledge_base_index_generations (
            id, knowledge_base_id, index_profile_hash,
            qdrant_collection_name, status, rebuild_snapshot_at
        ) VALUES (%s, %s, %s, %s, 'active', %s)
        """,
        (uuid4(), knowledge_base_id, "a" * 64, f"collection-{uuid4()}", datetime.now(UTC)),
    )
    for expected_count, actual_count, caught_up_revision, validated_revision in (
        (100, 1, 10, 10),
        (100, 100, 10, 3),
    ):
        _assert_constraint_violation(
            connection,
            """
            INSERT INTO knowledge_base_index_generations (
                id, knowledge_base_id, index_profile_hash,
                qdrant_collection_name, status, rebuild_snapshot_at,
                caught_up_revision, distance, embedding_config_snapshot,
                filter_schema_snapshot, applied_filter_schema_revision,
                embedding_config_hash, validated_revision,
                validation_manifest_hash, expected_point_count,
                actual_point_count, validated_at, activated_at
            ) VALUES (%s, %s, %s, %s, 'active', %s, %s, 'cosine',
                      %s, %s, 0, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                uuid4(),
                _insert_knowledge_base(connection, f"invalid active {uuid4()}"),
                "a" * 64,
                f"collection-{uuid4()}",
                datetime.now(UTC),
                caught_up_revision,
                Jsonb({}),
                Jsonb({"fields": []}),
                "b" * 64,
                validated_revision,
                "c" * 64,
                expected_count,
                actual_count,
                datetime.now(UTC),
                datetime.now(UTC),
            ),
        )
    _assert_constraint_violation(
        connection,
        """
        INSERT INTO jobs (id, target_type, target_id, operation, status)
        VALUES (%s, 'knowledge_base', %s, 'rebuild_generation', 'running')
        """,
        (uuid4(), knowledge_base_id),
    )
    queued_job_id = uuid4()
    queued_heartbeat = datetime.now(UTC)
    connection.execute(
        """
        INSERT INTO jobs (
            id, target_type, target_id, operation, status, worker_heartbeat_at
        ) VALUES (%s, 'knowledge_base', %s, 'rebuild_generation', 'queued', %s)
        """,
        (queued_job_id, uuid4(), queued_heartbeat),
    )
    assert connection.execute(
        "SELECT status, lease_epoch, worker_heartbeat_at FROM jobs WHERE id = %s",
        (queued_job_id,),
    ).fetchone() == ("queued", 0, queued_heartbeat)
    retry_wait_job_id = uuid4()
    retry_wait_heartbeat = datetime.now(UTC)
    connection.execute(
        """
        INSERT INTO jobs (
            id, target_type, target_id, operation, status,
            lease_epoch, worker_heartbeat_at, next_retry_at
        ) VALUES (%s, 'knowledge_base', %s, 'rebuild_generation', 'retry_wait',
                  3, %s, %s)
        """,
        (retry_wait_job_id, uuid4(), retry_wait_heartbeat, datetime.now(UTC)),
    )
    claimed_heartbeat = datetime.now(UTC)
    connection.execute(
        """
        UPDATE jobs
        SET status = 'running', lease_owner = 'worker-1',
            lease_epoch = lease_epoch + 1, lease_expires_at = %s,
            worker_heartbeat_at = %s
        WHERE id = %s
        """,
        (datetime.now(UTC), claimed_heartbeat, retry_wait_job_id),
    )
    assert connection.execute(
        "SELECT status, lease_epoch FROM jobs WHERE id = %s",
        (retry_wait_job_id,),
    ).fetchone() == ("running", 4)
    connection.execute(
        """
        UPDATE jobs
        SET status = 'succeeded', lease_owner = NULL, lease_expires_at = NULL
        WHERE id = %s
        """,
        (retry_wait_job_id,),
    )
    assert connection.execute(
        """
        SELECT status, lease_epoch, worker_heartbeat_at, lease_owner, lease_expires_at
        FROM jobs WHERE id = %s
        """,
        (retry_wait_job_id,),
    ).fetchone() == ("succeeded", 4, claimed_heartbeat, None, None)
    connection.commit()

    _assert_constraint_violation(
        connection,
        """
        INSERT INTO jobs (
            id, target_type, target_id, operation, status,
            lease_owner, lease_epoch, lease_expires_at, worker_heartbeat_at
        ) VALUES (%s, 'knowledge_base', %s, 'rebuild_generation', 'queued',
                  'worker-1', 1, %s, %s)
        """,
        (uuid4(), uuid4(), datetime.now(UTC), datetime.now(UTC)),
    )
    _assert_constraint_violation(
        connection,
        """
        INSERT INTO jobs (
            id, target_type, target_id, operation, status,
            lease_owner, lease_epoch, lease_expires_at, worker_heartbeat_at
        ) VALUES (%s, 'knowledge_base', %s, 'rebuild_generation', 'succeeded',
                  'worker-1', 1, %s, %s)
        """,
        (uuid4(), uuid4(), datetime.now(UTC), datetime.now(UTC)),
    )


@pytest.mark.integration
def test_exclusive_chunk_checkpoint_and_terminal_idempotency_state(
    migrated_sync_connection: psycopg.Connection[Any],
) -> None:
    connection = migrated_sync_connection
    actor_key_id = _insert_api_key(connection)
    knowledge_base_id = _insert_knowledge_base(connection, "checkpoint kb")
    generation_id = _insert_generation(connection, knowledge_base_id, status="building")
    document_id = _insert_document(connection, knowledge_base_id)
    version_id = _insert_document_version(connection, document_id)
    job_id = uuid4()
    connection.execute(
        "INSERT INTO jobs (id, target_type, target_id, operation) "
        "VALUES (%s, 'document_version', %s, 'ingest_document')",
        (job_id, version_id),
    )
    connection.commit()
    _assert_constraint_violation(
        connection,
        """
        INSERT INTO document_index_states
            (document_version_id, index_generation_id, next_chunk_index)
        VALUES (%s, %s, -1)
        """,
        (version_id, generation_id),
    )
    _assert_constraint_violation(
        connection,
        """
        INSERT INTO index_generation_creation_requests (
            id, actor_api_key_id, knowledge_base_id, idempotency_key,
            request_fingerprint, generation_id, state
        ) VALUES (%s, %s, %s, 'create-generation', %s, %s, 'succeeded')
        """,
        (uuid4(), actor_key_id, knowledge_base_id, b"f" * 32, generation_id),
    )
    connection.execute(
        """
        INSERT INTO index_generation_creation_requests (
            id, actor_api_key_id, knowledge_base_id, idempotency_key,
            request_fingerprint, generation_id
        ) VALUES (%s, %s, %s, 'building-generation', %s, %s)
        """,
        (uuid4(), actor_key_id, knowledge_base_id, b"g" * 32, generation_id),
    )
    assert connection.execute(
        """
        SELECT state FROM index_generation_creation_requests
        WHERE idempotency_key = 'building-generation'
        """
    ).fetchone() == ("building",)
    connection.execute(
        """
        INSERT INTO document_upload_idempotency (
            id, actor_api_key_id, knowledge_base_id, idempotency_key,
            request_fingerprint, document_id, document_version_id, job_id, result_status
        ) VALUES (%s, %s, %s, 'upload', %s, %s, %s, %s, 'accepted')
        """,
        (uuid4(), actor_key_id, knowledge_base_id, b"u" * 32, document_id, version_id, job_id),
    )
    connection.commit()
