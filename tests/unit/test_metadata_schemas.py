# ruff: noqa: E501

import asyncio
from collections.abc import Callable, Iterable
from copy import deepcopy
from datetime import UTC, datetime
from types import TracebackType
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from pydantic import SecretStr, ValidationError
from sqlalchemy import (
    CHAR,
    BigInteger,
    Boolean,
    CheckConstraint,
    Index,
    Integer,
    LargeBinary,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    create_mock_engine,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import (
    AddConstraint,
    Column,
    CreateIndex,
    CreateTable,
    DefaultClause,
    ForeignKeyConstraint,
    Table,
)
from sqlalchemy.sql.ddl import ExecutableDDLElement
from sqlalchemy.sql.schema import CallableColumnDefault, ScalarElementColumnDefault
from sqlalchemy.sql.type_api import TypeEngine

from rag_service.api.cursors import CursorPosition
from rag_service.api.errors import BusinessError
from rag_service.auth.policies import AgentPrincipal, Capability
from rag_service.config import Settings
from rag_service.db import models as db_models
from rag_service.db.base import Base
from rag_service.db.models import (
    ApiKey,
    ApiKeyKnowledgeBaseScope,
    ApiKeyQueryProfileScope,
    AuditEvent,
    Document,
    DocumentIndexState,
    DocumentVersion,
    IdempotencyRecord,
    Job,
    KnowledgeBase,
    KnowledgeBaseIndexGeneration,
    KnowledgeBaseMutation,
    ModelProfile,
    ModelProfileFallback,
    ProviderConfig,
    ProviderUsage,
    QueryLog,
    QueryProfile,
    SparseProfile,
)
from rag_service.metadata import services as metadata_services
from rag_service.metadata.knowledge_base_repositories import KnowledgeBaseRepositories
from rag_service.metadata.schemas import (
    KnowledgeBaseCreate,
    KnowledgeBasePatch,
    SafeKnowledgeBase,
    knowledge_base_create_fingerprint,
)
from rag_service.metadata.services import KnowledgeBaseService


def _model_table(model: type[Base]) -> Table:
    table = model.__table__
    if not isinstance(table, Table):
        raise AssertionError(f"{model.__name__} is not mapped to a Table")
    return table


def _invoke_callable_default(default: CallableColumnDefault) -> object:
    callable_default = cast(Callable[[object | None], object], default.arg)
    return callable_default(None)


POSTGRES_DIALECT = cast(Callable[[], Dialect], postgresql.dialect)()


TASK_TWO_TABLES = {
    "api_key_knowledge_base_scopes": _model_table(ApiKeyKnowledgeBaseScope),
    "api_key_query_profile_scopes": _model_table(ApiKeyQueryProfileScope),
    "api_keys": _model_table(ApiKey),
    "audit_events": _model_table(AuditEvent),
    "idempotency_records": _model_table(IdempotencyRecord),
    "knowledge_base_index_generations": _model_table(KnowledgeBaseIndexGeneration),
    "knowledge_base_mutations": _model_table(KnowledgeBaseMutation),
    "knowledge_bases": _model_table(KnowledgeBase),
}


def _column_names(columns: Iterable[Column[Any]]) -> tuple[str, ...]:
    return tuple(column.name for column in columns)


def _normalized_sql(value: object) -> str:
    normalized: list[str] = []
    in_literal = False
    pending_space = False
    sql = str(value)
    index = 0
    while index < len(sql):
        character = sql[index]
        if in_literal:
            normalized.append(character)
            if character == "'":
                if index + 1 < len(sql) and sql[index + 1] == "'":
                    normalized.append("'")
                    index += 1
                else:
                    in_literal = False
        elif character == "'":
            if pending_space and normalized:
                normalized.append(" ")
            pending_space = False
            normalized.append(character)
            in_literal = True
        elif character.isspace():
            pending_space = True
        else:
            if pending_space and normalized:
                normalized.append(" ")
            pending_space = False
            normalized.append(character.lower())
        index += 1
    return "".join(normalized).strip()


def test_sql_normalization_preserves_quoted_literal_case() -> None:
    assert _normalized_sql("CHECK (currency ~ '^[A-Z]{3}$')") == ("check (currency ~ '^[A-Z]{3}$')")


def _check_definitions(table: Table) -> dict[str, str]:
    return {
        str(constraint.name): _normalized_sql(constraint.sqltext.compile(dialect=POSTGRES_DIALECT))
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint)
    }


def _server_defaults(table: Table) -> dict[str, str]:
    result: dict[str, str] = {}
    for column in table.columns:
        default = column.server_default
        if default is None:
            continue
        if not isinstance(default, DefaultClause):
            raise AssertionError(f"{table.name}.{column.name} has a non-clause server default")
        result[column.name] = str(default.arg)
    return result


def _orm_default_columns(table: Table) -> set[str]:
    return {column.name for column in table.columns if column.default is not None}


def _foreign_keys(table: Table) -> dict[str, tuple[str, str | None]]:
    result: dict[str, tuple[str, str | None]] = {}
    for column in table.columns:
        for foreign_key in column.foreign_keys:
            result[column.name] = (foreign_key.target_fullname, foreign_key.ondelete)
    return result


def _foreign_key_constraint_contracts(
    table: Table,
) -> set[tuple[str | None, tuple[str, ...], tuple[str, ...], str | None, bool | None, str | None]]:
    return {
        (
            constraint.name if isinstance(constraint.name, str) else None,
            _column_names(constraint.columns),
            tuple(element.target_fullname for element in constraint.elements),
            constraint.ondelete,
            constraint.deferrable,
            constraint.initially,
        )
        for constraint in table.foreign_key_constraints
    }


def _unique_constraints(table: Table) -> set[tuple[str | None, tuple[str, ...]]]:
    return {
        (
            constraint.name if isinstance(constraint.name, str) else None,
            _column_names(constraint.columns),
        )
        for constraint in table.constraints
        if isinstance(constraint, UniqueConstraint)
    }


def _index_contract(index: Index) -> tuple[bool, tuple[str, ...], str | None]:
    predicate = index.dialect_options["postgresql"]["where"]
    return (
        bool(index.unique),
        _column_names(index.columns),
        None if predicate is None else _normalized_sql(predicate),
    )


def _type_signature(type_: TypeEngine[Any]) -> tuple[object, ...]:
    if isinstance(type_, PostgreSQLUUID):
        return ("uuid", type_.as_uuid)
    if isinstance(type_, TIMESTAMP):
        return ("timestamp", type_.timezone)
    if isinstance(type_, JSONB):
        return ("jsonb",)
    if isinstance(type_, ARRAY):
        return ("array",) + _type_signature(type_.item_type)
    if isinstance(type_, CHAR):
        return ("char", type_.length)
    if isinstance(type_, LargeBinary):
        return ("binary", type_.length)
    if isinstance(type_, SmallInteger):
        return ("smallint",)
    if isinstance(type_, BigInteger):
        return ("bigint",)
    if isinstance(type_, Numeric):
        return ("numeric", type_.precision, type_.scale)
    if isinstance(type_, Integer):
        return ("integer",)
    if isinstance(type_, Boolean):
        return ("boolean",)
    if isinstance(type_, Text):
        return ("text",)
    if isinstance(type_, String):
        return ("varchar", type_.length)
    raise AssertionError(f"Unhandled SQLAlchemy type: {type_!r}")


def _column_contract(table: Table) -> tuple[tuple[str, tuple[object, ...], bool, bool], ...]:
    result: list[tuple[str, tuple[object, ...], bool, bool]] = []
    for column in table.columns:
        if column.nullable is None:
            raise AssertionError(f"{table.name}.{column.name} has unresolved nullability")
        result.append(
            (column.name, _type_signature(column.type), column.nullable, column.primary_key)
        )
    return tuple(result)


def test_current_model_import_registers_exact_task_two_tables() -> None:
    assert set(TASK_TWO_TABLES) <= set(Base.metadata.tables)
    assert {name: table.name for name, table in TASK_TWO_TABLES.items()} == {
        name: name for name in TASK_TWO_TABLES
    }


def test_all_task_two_tables_have_exact_ordered_column_contracts() -> None:
    uuid = ("uuid", True)
    timestamptz = ("timestamp", True)
    expected = {
        "api_keys": (
            ("id", uuid, False, True),
            ("public_id", ("varchar", 64), False, False),
            ("secret_digest", ("binary", 32), False, False),
            ("key_type", ("varchar", 16), False, False),
            ("name", ("varchar", 120), False, False),
            ("status", ("varchar", 16), False, False),
            ("capabilities", ("array", "text"), False, False),
            ("raw_file_read", ("boolean",), False, False),
            ("requests_per_minute", ("integer",), True, False),
            ("max_concurrency", ("integer",), True, False),
            ("not_before", timestamptz, True, False),
            ("expires_at", timestamptz, True, False),
            ("resource_revision", ("bigint",), False, False),
            ("created_by_api_key_id", uuid, True, False),
            ("revoked_by_api_key_id", uuid, True, False),
            ("created_at", timestamptz, False, False),
            ("updated_at", timestamptz, False, False),
            ("revoked_at", timestamptz, True, False),
        ),
        "api_key_knowledge_base_scopes": (
            ("api_key_id", uuid, False, True),
            ("knowledge_base_id", uuid, False, True),
            ("created_at", timestamptz, False, False),
        ),
        "api_key_query_profile_scopes": (
            ("api_key_id", uuid, False, True),
            ("query_profile_id", uuid, False, True),
            ("is_default", ("boolean",), False, False),
            ("created_at", timestamptz, False, False),
        ),
        "audit_events": (
            ("id", uuid, False, True),
            ("request_id", ("varchar", 128), False, False),
            ("actor_api_key_id", uuid, True, False),
            ("actor_kind", ("varchar", 16), False, False),
            ("action", ("varchar", 64), False, False),
            ("target_type", ("varchar", 64), False, False),
            ("target_id", uuid, True, False),
            ("metadata", ("jsonb",), False, False),
            ("created_at", timestamptz, False, False),
        ),
        "idempotency_records": (
            ("id", uuid, False, True),
            ("actor_key_id", uuid, False, False),
            ("operation", ("varchar", 64), False, False),
            ("idempotency_key", ("varchar", 128), False, False),
            ("request_fingerprint", ("binary", 32), False, False),
            ("result_resource_type", ("varchar", 64), False, False),
            ("result_resource_id", uuid, False, False),
            ("http_status", ("smallint",), False, False),
            ("created_at", timestamptz, False, False),
            ("updated_at", timestamptz, False, False),
        ),
        "knowledge_bases": (
            ("id", uuid, False, True),
            ("name", ("varchar", 120), False, False),
            ("description", ("varchar", 2000), True, False),
            ("status", ("varchar", 16), False, False),
            ("metadata", ("jsonb",), False, False),
            ("filter_schema", ("jsonb",), False, False),
            ("resource_revision", ("bigint",), False, False),
            ("mutation_revision", ("bigint",), False, False),
            ("filter_schema_revision", ("bigint",), False, False),
            ("active_index_generation_id", uuid, True, False),
            ("pending_index_generation_id", uuid, True, False),
            ("rerank_profile_id", uuid, True, False),
            ("created_at", timestamptz, False, False),
            ("updated_at", timestamptz, False, False),
        ),
        "knowledge_base_index_generations": (
            ("id", uuid, False, True),
            ("knowledge_base_id", uuid, False, False),
            ("embedding_profile_id", uuid, True, False),
            ("sparse_profile_id", uuid, True, False),
            ("index_profile_hash", ("char", 64), False, False),
            ("qdrant_collection_name", ("varchar", 255), False, False),
            ("status", ("varchar", 16), False, False),
            ("rebuild_snapshot_at", timestamptz, False, False),
            ("caught_up_revision", ("bigint",), False, False),
            ("validated_revision", ("bigint",), True, False),
            ("validation_manifest_hash", ("char", 64), True, False),
            ("expected_point_count", ("bigint",), True, False),
            ("actual_point_count", ("bigint",), True, False),
            ("created_at", timestamptz, False, False),
            ("validated_at", timestamptz, True, False),
            ("activated_at", timestamptz, True, False),
            ("retired_at", timestamptz, True, False),
            ("distance", ("varchar", 16), True, False),
            ("embedding_config_snapshot", ("jsonb",), True, False),
            ("filter_schema_snapshot", ("jsonb",), True, False),
            ("applied_filter_schema_revision", ("bigint",), True, False),
            ("embedding_config_hash", ("char", 64), True, False),
            ("safe_error_code", ("varchar", 64), True, False),
            ("safe_error_message", ("varchar", 500), True, False),
        ),
        "knowledge_base_mutations": (
            ("id", uuid, False, True),
            ("knowledge_base_id", uuid, False, False),
            ("revision", ("bigint",), False, False),
            ("mutation_type", ("varchar", 40), False, False),
            ("target_type", ("varchar", 32), False, False),
            ("target_id", uuid, False, False),
            ("payload", ("jsonb",), False, False),
            ("created_at", timestamptz, False, False),
        ),
    }

    assert {name: _column_contract(table) for name, table in TASK_TWO_TABLES.items()} == expected


def test_auth_metadata_has_no_plaintext_secret_column() -> None:
    forbidden = {"authorization_header", "response_body", "secret", "token"}
    for table in TASK_TWO_TABLES.values():
        assert forbidden.isdisjoint(table.columns.keys())


def test_api_key_check_definitions_are_authoritative() -> None:
    assert _check_definitions(_model_table(ApiKey)) == {
        "ck_api_keys_admin_policy": (
            "key_type <> 'admin' or (cardinality(capabilities) = 0 "
            "and raw_file_read = false and requests_per_minute is null "
            "and max_concurrency is null)"
        ),
        "ck_api_keys_capabilities": (
            "cardinality(capabilities) <= 4 and array_position(capabilities, null) is null "
            "and capabilities <@ array['ingest', 'retrieve', 'answer', 'manage']::text[]"
        ),
        "ck_api_keys_key_type": "key_type in ('admin', 'agent')",
        "ck_api_keys_name_length": "char_length(name) between 1 and 120",
        "ck_api_keys_policy_positive": (
            "key_type <> 'agent' or (requests_per_minute is not null "
            "and max_concurrency is not null "
            "and requests_per_minute between 1 and 10000 "
            "and max_concurrency between 1 and 1000)"
        ),
        "ck_api_keys_public_id_length": "char_length(public_id) between 16 and 64",
        "ck_api_keys_revision_positive": "resource_revision >= 1",
        "ck_api_keys_revocation_state": "(status = 'revoked') = (revoked_at is not null)",
        "ck_api_keys_secret_digest_length": "octet_length(secret_digest) = 32",
        "ck_api_keys_status": "status in ('active', 'disabled', 'revoked')",
        "ck_api_keys_validity_window": (
            "not_before is null or expires_at is null or expires_at > not_before"
        ),
    }


def test_audit_and_idempotency_check_definitions_are_authoritative() -> None:
    assert _check_definitions(_model_table(AuditEvent)) == {
        "ck_audit_events_action_length": "char_length(action) between 1 and 64",
        "ck_audit_events_actor_kind": (
            "actor_kind in ('admin_key', 'agent_key', 'local_cli', 'system')"
        ),
        "ck_audit_events_request_id_length": "char_length(request_id) between 1 and 128",
        "ck_audit_events_target_type_length": "char_length(target_type) between 1 and 64",
    }
    assert _check_definitions(_model_table(IdempotencyRecord)) == {
        "ck_idempotency_records_fingerprint_length": "octet_length(request_fingerprint) = 32",
        "ck_idempotency_records_http_status": "http_status between 100 and 599",
        "ck_idempotency_records_key_visible_ascii": (
            "char_length(idempotency_key) between 1 and 128 and idempotency_key ~ '^[!-~]+$'"
        ),
        "ck_idempotency_records_operation_length": ("char_length(operation) between 1 and 64"),
        "ck_idempotency_records_result_type_length": (
            "char_length(result_resource_type) between 1 and 64"
        ),
    }


def test_knowledge_base_check_definitions_are_authoritative() -> None:
    assert _check_definitions(_model_table(KnowledgeBase)) == {
        "ck_knowledge_bases_filter_schema_revision_nonnegative": "filter_schema_revision >= 0",
        "ck_knowledge_bases_generation_pointers_distinct": (
            "active_index_generation_id is null or pending_index_generation_id is null "
            "or active_index_generation_id <> pending_index_generation_id"
        ),
        "ck_knowledge_bases_mutation_revision_nonnegative": "mutation_revision >= 0",
        "ck_knowledge_bases_name_length": "char_length(name) between 1 and 120",
        "ck_knowledge_bases_revision_positive": "resource_revision >= 1",
        "ck_knowledge_bases_status": ("status in ('active', 'reindexing', 'disabled', 'deleting')"),
    }


def test_generation_check_definitions_are_authoritative() -> None:
    assert _check_definitions(_model_table(KnowledgeBaseIndexGeneration)) == {
        "ck_kb_index_generations_actual_count_nonnegative": (
            "actual_point_count is null or actual_point_count >= 0"
        ),
        "ck_kb_index_generations_active_validation_complete": (
            "status <> 'active' or (distance is not null and embedding_config_snapshot is not null "
            "and filter_schema_snapshot is not null and applied_filter_schema_revision is not null "
            "and embedding_config_hash is not null and validated_revision is not null "
            "and validation_manifest_hash is not null and expected_point_count is not null "
            "and actual_point_count is not null and expected_point_count = actual_point_count "
            "and validated_revision = caught_up_revision and validated_at is not null "
            "and activated_at is not null)"
        ),
        "ck_kb_index_generations_caught_up_revision_nonnegative": "caught_up_revision >= 0",
        "ck_kb_index_generations_expected_count_nonnegative": (
            "expected_point_count is null or expected_point_count >= 0"
        ),
        "ck_kb_index_generations_distance": (
            "distance is null or distance in ('cosine', 'dot', 'euclid', 'manhattan')"
        ),
        "ck_kb_index_generations_embedding_config_hash": (
            "embedding_config_hash is null or embedding_config_hash ~ '^[0-9a-f]{64}$'"
        ),
        "ck_kb_index_generations_embedding_config_snapshot_object": (
            "embedding_config_snapshot is null or jsonb_typeof(embedding_config_snapshot) = 'object'"
        ),
        "ck_kb_index_generations_filter_schema_revision_nonnegative": (
            "applied_filter_schema_revision is null or applied_filter_schema_revision >= 0"
        ),
        "ck_kb_index_generations_filter_schema_snapshot_object": (
            "filter_schema_snapshot is null or jsonb_typeof(filter_schema_snapshot) = 'object'"
        ),
        "ck_kb_index_generations_index_profile_hash": ("index_profile_hash ~ '^[0-9a-f]{64}$'"),
        "ck_kb_index_generations_status": (
            "status in ('building', 'active', 'retiring', 'failed')"
        ),
        "ck_kb_index_generations_safe_error_code_length": (
            "safe_error_code is null or char_length(safe_error_code) between 1 and 64"
        ),
        "ck_kb_index_generations_safe_error_message_length": (
            "safe_error_message is null or char_length(safe_error_message) between 1 and 500"
        ),
        "ck_kb_index_generations_validated_revision_nonnegative": (
            "validated_revision is null or validated_revision >= 0"
        ),
        "ck_kb_index_generations_validation_manifest_hash": (
            "validation_manifest_hash is null or validation_manifest_hash ~ '^[0-9a-f]{64}$'"
        ),
    }


def test_mutation_check_definitions_are_authoritative() -> None:
    assert _check_definitions(_model_table(KnowledgeBaseMutation)) == {
        "ck_knowledge_base_mutations_mutation_type": (
            "mutation_type in ('document_version_created', 'document_activated', "
            "'document_deleted', 'metadata_changed', 'filter_schema_changed', "
            "'index_config_changed')"
        ),
        "ck_knowledge_base_mutations_revision_positive": "revision > 0",
        "ck_knowledge_base_mutations_target_type": (
            "target_type in ('knowledge_base', 'document', 'document_version', "
            "'index_generation', 'filter_schema_revision')"
        ),
    }


def test_primary_keys_and_unique_constraints_are_exact() -> None:
    assert {
        name: _column_names(table.primary_key.columns) for name, table in TASK_TWO_TABLES.items()
    } == {
        "api_keys": ("id",),
        "api_key_knowledge_base_scopes": ("api_key_id", "knowledge_base_id"),
        "api_key_query_profile_scopes": ("api_key_id", "query_profile_id"),
        "audit_events": ("id",),
        "idempotency_records": ("id",),
        "knowledge_bases": ("id",),
        "knowledge_base_index_generations": ("id",),
        "knowledge_base_mutations": ("id",),
    }
    assert _unique_constraints(_model_table(ApiKey)) == {(None, ("public_id",))}
    assert _unique_constraints(_model_table(ApiKeyKnowledgeBaseScope)) == set()
    assert _unique_constraints(_model_table(ApiKeyQueryProfileScope)) == set()
    assert _unique_constraints(_model_table(AuditEvent)) == set()
    assert _unique_constraints(_model_table(IdempotencyRecord)) == {
        (
            "uq_idempotency_actor_operation_key",
            ("actor_key_id", "operation", "idempotency_key"),
        )
    }
    assert _unique_constraints(_model_table(KnowledgeBase)) == set()
    assert _unique_constraints(_model_table(KnowledgeBaseIndexGeneration)) == {
        (None, ("qdrant_collection_name",)),
        ("uq_kb_index_generations_id_knowledge_base", ("id", "knowledge_base_id")),
    }
    assert _unique_constraints(_model_table(KnowledgeBaseMutation)) == {
        ("uq_knowledge_base_mutations_kb_revision", ("knowledge_base_id", "revision"))
    }


def test_foreign_key_targets_and_delete_actions_are_exact() -> None:
    assert _foreign_keys(_model_table(ApiKey)) == {
        "created_by_api_key_id": ("api_keys.id", "SET NULL"),
        "revoked_by_api_key_id": ("api_keys.id", "SET NULL"),
    }
    assert _foreign_keys(_model_table(ApiKeyKnowledgeBaseScope)) == {
        "api_key_id": ("api_keys.id", "CASCADE"),
        "knowledge_base_id": ("knowledge_bases.id", "RESTRICT"),
    }
    assert _foreign_keys(_model_table(ApiKeyQueryProfileScope)) == {
        "api_key_id": ("api_keys.id", "CASCADE"),
        "query_profile_id": ("query_profiles.id", "RESTRICT"),
    }
    assert _foreign_keys(_model_table(AuditEvent)) == {
        "actor_api_key_id": ("api_keys.id", "SET NULL")
    }
    assert _foreign_keys(_model_table(IdempotencyRecord)) == {
        "actor_key_id": ("api_keys.id", "RESTRICT")
    }
    assert _foreign_keys(_model_table(KnowledgeBase)) == {
        # RESTRICT, not SET NULL: dropping the pointer would silently turn
        # reranking off for the knowledge base rather than refusing the delete.
        "rerank_profile_id": ("model_profiles.id", "RESTRICT"),
    }
    assert _foreign_keys(_model_table(KnowledgeBaseIndexGeneration)) == {
        "knowledge_base_id": ("knowledge_bases.id", "RESTRICT"),
        "embedding_profile_id": ("model_profiles.id", "RESTRICT"),
        "sparse_profile_id": ("sparse_profiles.id", "RESTRICT"),
    }
    assert _foreign_keys(_model_table(KnowledgeBaseMutation)) == {
        "knowledge_base_id": ("knowledge_bases.id", "RESTRICT")
    }


def test_server_defaults_are_exact_for_every_task_two_table() -> None:
    assert {name: _server_defaults(table) for name, table in TASK_TWO_TABLES.items()} == {
        "api_keys": {
            "status": "'active'",
            "capabilities": "'{}'::text[]",
            "raw_file_read": "false",
            "resource_revision": "1",
            "created_at": "now()",
            "updated_at": "now()",
        },
        "api_key_knowledge_base_scopes": {"created_at": "now()"},
        "api_key_query_profile_scopes": {
            "is_default": "false",
            "created_at": "now()",
        },
        "audit_events": {"metadata": "'{}'::jsonb", "created_at": "now()"},
        "idempotency_records": {"created_at": "now()", "updated_at": "now()"},
        "knowledge_bases": {
            "status": "'active'",
            "metadata": "'{}'::jsonb",
            "filter_schema": "'{\"fields\": []}'::jsonb",
            "resource_revision": "1",
            "mutation_revision": "0",
            "filter_schema_revision": "0",
            "created_at": "now()",
            "updated_at": "now()",
        },
        "knowledge_base_index_generations": {
            "caught_up_revision": "0",
            "created_at": "now()",
        },
        "knowledge_base_mutations": {"payload": "'{}'::jsonb", "created_at": "now()"},
    }


def test_orm_defaults_are_exact_and_mutable_defaults_are_independent() -> None:
    assert {name: _orm_default_columns(table) for name, table in TASK_TWO_TABLES.items()} == {
        "api_keys": {"id", "status", "capabilities", "raw_file_read", "resource_revision"},
        "api_key_knowledge_base_scopes": set(),
        "api_key_query_profile_scopes": {"is_default"},
        "audit_events": {"id", "metadata"},
        "idempotency_records": {"id"},
        "knowledge_bases": {
            "id",
            "status",
            "metadata",
            "filter_schema",
            "resource_revision",
            "mutation_revision",
            "filter_schema_revision",
        },
        "knowledge_base_index_generations": {"id", "caught_up_revision"},
        "knowledge_base_mutations": {"id", "payload"},
    }

    assert ApiKey.__table__.c.status.default.arg == "active"
    assert ApiKey.__table__.c.raw_file_read.default.arg is False
    assert ApiKey.__table__.c.resource_revision.default.arg == 1
    assert ApiKeyQueryProfileScope.__table__.c.is_default.default.arg is False
    assert KnowledgeBase.__table__.c.status.default.arg == "active"
    assert KnowledgeBase.__table__.c.resource_revision.default.arg == 1
    assert KnowledgeBase.__table__.c.mutation_revision.default.arg == 0
    assert KnowledgeBase.__table__.c.filter_schema_revision.default.arg == 0
    assert KnowledgeBaseIndexGeneration.__table__.c.caught_up_revision.default.arg == 0

    mutable_columns = (
        ApiKey.__table__.c.capabilities,
        AuditEvent.__table__.c.metadata,
        KnowledgeBase.__table__.c.metadata,
        KnowledgeBase.__table__.c.filter_schema,
        KnowledgeBaseMutation.__table__.c.payload,
    )
    for column in mutable_columns:
        assert column.default is not None and column.default.is_callable
        assert column.default.arg(None) is not column.default.arg(None)
    assert KnowledgeBase.__table__.c.filter_schema.default.arg(None) == {"fields": []}

    for table in (
        ApiKey.__table__,
        AuditEvent.__table__,
        IdempotencyRecord.__table__,
        KnowledgeBase.__table__,
        KnowledgeBaseIndexGeneration.__table__,
        KnowledgeBaseMutation.__table__,
    ):
        default = table.c.id.default
        assert default is not None and default.is_callable
        first = default.arg(None)
        second = default.arg(None)
        assert isinstance(first, UUID)
        assert first != second
        assert table.c.id.server_default is None


def test_partial_unique_indexes_are_exact_and_compile_for_postgresql() -> None:
    assert {
        str(index.name): _index_contract(index)
        for index in _model_table(ApiKeyQueryProfileScope).indexes
    } == {"uq_api_key_query_profile_default": (True, ("api_key_id",), "is_default")}
    generation_indexes = {
        str(index.name): _index_contract(index)
        for index in _model_table(KnowledgeBaseIndexGeneration).indexes
    }
    assert generation_indexes == {
        "uq_kb_index_generations_one_active": (
            True,
            ("knowledge_base_id",),
            "status = 'active'",
        ),
        "uq_kb_index_generations_one_building": (
            True,
            ("knowledge_base_id",),
            "status = 'building'",
        ),
    }
    for index in _model_table(KnowledgeBaseIndexGeneration).indexes:
        compiled = _normalized_sql(CreateIndex(index).compile(dialect=POSTGRES_DIALECT))
        assert "create unique index" in compiled
        assert f"where {_index_contract(index)[2]}" in compiled


def test_generation_pointer_relationships_include_parent_identity() -> None:
    active_join = _normalized_sql(KnowledgeBase.active_index_generation.property.primaryjoin)
    pending_join = _normalized_sql(KnowledgeBase.pending_index_generation.property.primaryjoin)

    assert "knowledge_bases.active_index_generation_id" in active_join
    assert "knowledge_bases.pending_index_generation_id" in pending_join
    for relationship_join in (active_join, pending_join):
        assert "knowledge_bases.id = knowledge_base_index_generations.knowledge_base_id" in (
            relationship_join
        )
    assert KnowledgeBase.active_index_generation.property.post_update
    assert KnowledgeBase.pending_index_generation.property.post_update


def test_agent_policy_check_rejects_null_limits() -> None:
    policy = _check_definitions(_model_table(ApiKey))["ck_api_keys_policy_positive"]
    assert "requests_per_minute is not null" in policy
    assert "max_concurrency is not null" in policy
    assert "requests_per_minute between 1 and 10000" in policy
    assert "max_concurrency between 1 and 1000" in policy


TASK_THREE_TABLES = {
    "document_index_states": _model_table(DocumentIndexState),
    "document_versions": _model_table(DocumentVersion),
    "documents": _model_table(Document),
    "jobs": _model_table(Job),
    "model_profile_fallbacks": _model_table(ModelProfileFallback),
    "model_profiles": _model_table(ModelProfile),
    "provider_configs": _model_table(ProviderConfig),
    "provider_usage": _model_table(ProviderUsage),
    "query_logs": _model_table(QueryLog),
    "query_profiles": _model_table(QueryProfile),
    "sparse_profiles": _model_table(SparseProfile),
}


def _foreign_key_constraints(
    table: Table,
) -> dict[str, tuple[tuple[str, ...], tuple[str, ...], str | None, bool | None, str | None]]:
    return {
        str(constraint.name): (
            _column_names(constraint.columns),
            tuple(element.target_fullname for element in constraint.elements),
            constraint.ondelete,
            constraint.deferrable,
            constraint.initially,
        )
        for constraint in table.constraints
        if isinstance(constraint, ForeignKeyConstraint)
    }


def test_task_three_models_register_exact_complete_metadata() -> None:
    assert set(Base.metadata.tables) == set(TASK_TWO_TABLES) | set(TASK_THREE_TABLES) | {
        "provider_credentials",
        "index_generation_creation_requests",
        "index_generation_cleanup_claims",
        "document_upload_idempotency",
    }
    sorted_tables = Base.metadata.sorted_tables
    assert {table.name for table in sorted_tables} == set(Base.metadata.tables)
    for table in sorted_tables:
        assert "create table" in _normalized_sql(
            CreateTable(table).compile(dialect=POSTGRES_DIALECT)
        )
        for index in table.indexes:
            assert "create" in _normalized_sql(CreateIndex(index).compile(dialect=POSTGRES_DIALECT))


def test_task_three_exact_ordered_column_contracts() -> None:
    uuid = ("uuid", True)
    timestamptz = ("timestamp", True)
    expected = {
        "documents": (
            ("id", uuid, False, True),
            ("knowledge_base_id", uuid, False, False),
            ("display_name", ("varchar", 255), False, False),
            ("mime_type", ("varchar", 255), True, False),
            ("checksum_sha256", ("char", 64), True, False),
            ("current_version_id", uuid, True, False),
            ("pending_version_id", uuid, True, False),
            ("status", ("varchar", 16), False, False),
            ("tags", ("array", "text"), False, False),
            ("metadata", ("jsonb",), False, False),
            ("created_at", timestamptz, False, False),
            ("updated_at", timestamptz, False, False),
            ("deleted_at", timestamptz, True, False),
        ),
        "document_versions": (
            ("id", uuid, False, True),
            ("document_id", uuid, False, False),
            ("version_number", ("integer",), False, False),
            ("source_object_key", ("varchar", 1024), False, False),
            ("parsed_object_key", ("varchar", 1024), True, False),
            ("source_checksum_sha256", ("char", 64), False, False),
            ("parsed_object_checksum_sha256", ("char", 64), True, False),
            ("declared_mime_type", ("varchar", 255), True, False),
            ("detected_mime_type", ("varchar", 255), True, False),
            ("source_extension", ("varchar", 32), True, False),
            ("base_version_id", uuid, True, False),
            ("parser_name", ("varchar", 120), True, False),
            ("parser_version", ("varchar", 64), True, False),
            ("parser_config", ("jsonb",), False, False),
            ("chunker_name", ("varchar", 120), True, False),
            ("chunker_version", ("varchar", 64), True, False),
            ("chunker_config", ("jsonb",), False, False),
            ("chunk_count", ("integer",), True, False),
            ("status", ("varchar", 24), False, False),
            ("activated_at", timestamptz, True, False),
            ("created_at", timestamptz, False, False),
            ("chunk_manifest_object_key", ("varchar", 1024), True, False),
            ("chunk_manifest_checksum_sha256", ("char", 64), True, False),
            ("chunk_config_hash", ("char", 64), True, False),
        ),
        "document_index_states": (
            ("document_version_id", uuid, False, True),
            ("index_generation_id", uuid, False, True),
            ("status", ("varchar", 16), False, False),
            ("expected_point_count", ("bigint",), True, False),
            ("actual_point_count", ("bigint",), True, False),
            ("error_code", ("varchar", 64), True, False),
            ("created_at", timestamptz, False, False),
            ("validated_at", timestamptz, True, False),
            ("chunk_manifest_checksum_sha256", ("char", 64), True, False),
            ("embedding_config_hash", ("char", 64), True, False),
            ("next_chunk_index", ("bigint",), False, False),
            ("safe_error_message", ("varchar", 500), True, False),
        ),
        "jobs": (
            ("id", uuid, False, True),
            ("knowledge_base_id", uuid, True, False),
            ("target_type", ("varchar", 32), False, False),
            ("target_id", uuid, False, False),
            ("target_revision", ("bigint",), True, False),
            ("index_generation_id", uuid, True, False),
            ("mutation_id", uuid, True, False),
            ("parent_job_id", uuid, True, False),
            ("root_job_id", uuid, True, False),
            ("idempotency_key", ("varchar", 128), True, False),
            ("operation", ("varchar", 48), False, False),
            ("stage", ("varchar", 64), True, False),
            ("status", ("varchar", 16), False, False),
            ("progress_current", ("bigint",), False, False),
            ("progress_total", ("bigint",), True, False),
            ("attempt_count", ("integer",), False, False),
            ("max_attempts", ("integer",), False, False),
            ("next_retry_at", timestamptz, True, False),
            ("worker_heartbeat_at", timestamptz, True, False),
            ("cancel_requested_at", timestamptz, True, False),
            ("error_code", ("varchar", 64), True, False),
            ("error_message_sanitized", ("varchar", 500), True, False),
            ("created_at", timestamptz, False, False),
            ("updated_at", timestamptz, False, False),
            ("started_at", timestamptz, True, False),
            ("finished_at", timestamptz, True, False),
            ("lease_owner", ("varchar", 255), True, False),
            ("lease_epoch", ("bigint",), False, False),
            ("lease_expires_at", timestamptz, True, False),
            ("retryable", ("boolean",), False, False),
            ("resume_stage", ("varchar", 64), True, False),
            ("actor_api_key_id", uuid, True, False),
        ),
        "provider_configs": (
            ("id", uuid, False, True),
            ("name", ("varchar", 120), False, False),
            ("provider_type", ("varchar", 32), False, False),
            ("base_url", ("varchar", 2048), False, False),
            ("secret_ref", ("varchar", 255), True, False),
            ("default_headers", ("jsonb",), False, False),
            ("routing_options", ("jsonb",), False, False),
            ("timeout_seconds", ("numeric", 8, 3), False, False),
            ("max_concurrency", ("integer",), False, False),
            ("requests_per_minute", ("integer",), False, False),
            ("enabled", ("boolean",), False, False),
            ("created_at", timestamptz, False, False),
            ("updated_at", timestamptz, False, False),
            ("credential_id", uuid, True, False),
            ("resource_revision", ("bigint",), False, False),
            ("endpoint_policy_version", ("varchar", 64), True, False),
            ("endpoint_validated_at", timestamptz, True, False),
        ),
        "model_profiles": (
            ("id", uuid, False, True),
            ("name", ("varchar", 120), False, False),
            ("capability", ("varchar", 16), False, False),
            ("provider_config_id", uuid, False, False),
            ("model_name", ("varchar", 255), False, False),
            ("dimension", ("integer",), True, False),
            ("max_input_tokens", ("integer",), False, False),
            ("batch_size", ("integer",), False, False),
            ("timeout_seconds", ("numeric", 8, 3), False, False),
            ("enabled", ("boolean",), False, False),
            ("created_at", timestamptz, False, False),
            ("updated_at", timestamptz, False, False),
            ("resource_revision", ("bigint",), False, False),
            ("vector_config", ("jsonb",), False, False),
        ),
        "model_profile_fallbacks": (
            ("profile_id", uuid, False, True),
            ("fallback_profile_id", uuid, False, True),
            ("priority", ("integer",), False, False),
        ),
        "sparse_profiles": (
            ("id", uuid, False, True),
            ("name", ("varchar", 120), False, False),
            ("algorithm", ("varchar", 64), False, False),
            ("encoder_package", ("varchar", 255), False, False),
            ("encoder_version", ("varchar", 64), False, False),
            ("tokenizer_name", ("varchar", 120), False, False),
            ("tokenizer_version", ("varchar", 64), False, False),
            ("language_config", ("jsonb",), False, False),
            ("term_frequency_config", ("jsonb",), False, False),
            ("length_normalization_config", ("jsonb",), False, False),
            ("idf_object_key", ("varchar", 1024), False, False),
            ("idf_checksum_sha256", ("char", 64), False, False),
            ("oov_config", ("jsonb",), False, False),
            ("config_hash", ("char", 64), False, False),
            ("enabled", ("boolean",), False, False),
            ("created_at", timestamptz, False, False),
            ("updated_at", timestamptz, False, False),
        ),
        "query_profiles": (
            ("id", uuid, False, True),
            ("name", ("varchar", 120), False, False),
            ("rerank_profile_id", uuid, True, False),
            ("chat_profile_id", uuid, True, False),
            ("dense_candidate_limit", ("integer",), False, False),
            ("sparse_candidate_limit", ("integer",), False, False),
            ("rrf_candidate_limit", ("integer",), False, False),
            ("rerank_candidate_limit", ("integer",), False, False),
            ("top_k_limit", ("integer",), False, False),
            ("min_rerank_score", ("numeric", 12, 8), False, False),
            ("min_rrf_score_when_degraded", ("numeric", 12, 8), False, False),
            ("context_token_budget", ("integer",), False, False),
            ("enabled", ("boolean",), False, False),
            ("is_system_default", ("boolean",), False, False),
            ("created_at", timestamptz, False, False),
            ("updated_at", timestamptz, False, False),
        ),
        "query_logs": (
            ("id", uuid, False, True),
            ("request_id", ("varchar", 128), False, False),
            ("actor_api_key_id", uuid, True, False),
            ("knowledge_base_ids", ("array", "uuid", True), False, False),
            ("query_profile_id", uuid, True, False),
            ("latency_ms", ("bigint",), False, False),
            ("status", ("varchar", 16), False, False),
            ("degraded", ("boolean",), False, False),
            ("created_at", timestamptz, False, False),
        ),
        "provider_usage": (
            ("id", uuid, False, True),
            ("request_id", ("varchar", 128), False, False),
            ("actor_api_key_id", uuid, True, False),
            ("provider_config_id", uuid, True, False),
            ("model_profile_id", uuid, True, False),
            ("capability", ("varchar", 16), False, False),
            ("provider_identifier", ("varchar", 120), False, False),
            ("model_identifier", ("varchar", 255), False, False),
            ("route_identifier", ("varchar", 255), True, False),
            ("provider_request_id", ("varchar", 255), True, False),
            ("input_tokens", ("bigint",), False, False),
            ("output_tokens", ("bigint",), False, False),
            ("cost_micros", ("bigint",), False, False),
            ("currency", ("char", 3), False, False),
            ("latency_ms", ("bigint",), False, False),
            ("status", ("varchar", 16), False, False),
            ("error_code", ("varchar", 64), True, False),
            ("degraded", ("boolean",), False, False),
            ("created_at", timestamptz, False, False),
        ),
    }
    assert {name: _column_contract(table) for name, table in TASK_THREE_TABLES.items()} == expected


def test_locked_task_three_attribute_names_have_no_legacy_aliases() -> None:
    expected_attributes = {
        DocumentVersion: {"parsed_object_checksum_sha256"},
        Job: {"worker_heartbeat_at"},
        QueryProfile: {
            "dense_candidate_limit",
            "sparse_candidate_limit",
            "rrf_candidate_limit",
            "rerank_candidate_limit",
            "top_k_limit",
        },
    }
    legacy_attributes = {
        DocumentVersion: {"parsed_checksum_sha256"},
        Job: {"heartbeat_at"},
        QueryProfile: {
            "dense_candidate_count",
            "sparse_candidate_count",
            "rrf_candidate_count",
            "rerank_candidate_count",
            "top_k",
        },
    }
    for model, required in expected_attributes.items():
        mapped_attributes = set(model.__mapper__.attrs.keys())
        assert required <= mapped_attributes
        assert legacy_attributes[model].isdisjoint(mapped_attributes)


def test_document_lineage_and_parent_pointer_foreign_keys_are_exact() -> None:
    assert _foreign_key_constraints(_model_table(Document)) == {
        "fk_documents_current_version_same_parent": (
            ("current_version_id", "id"),
            ("document_versions.id", "document_versions.document_id"),
            "RESTRICT",
            True,
            "DEFERRED",
        ),
        "fk_documents_knowledge_base": (
            ("knowledge_base_id",),
            ("knowledge_bases.id",),
            "RESTRICT",
            None,
            None,
        ),
        "fk_documents_pending_version_same_parent": (
            ("pending_version_id", "id"),
            ("document_versions.id", "document_versions.document_id"),
            "RESTRICT",
            True,
            "DEFERRED",
        ),
    }
    assert _foreign_key_constraints(_model_table(DocumentVersion)) == {
        "fk_document_versions_base_same_document": (
            ("base_version_id", "document_id"),
            ("document_versions.id", "document_versions.document_id"),
            "RESTRICT",
            True,
            "DEFERRED",
        ),
        "fk_document_versions_document": (
            ("document_id",),
            ("documents.id",),
            "RESTRICT",
            None,
            None,
        ),
    }
    assert Document.current_version.property.post_update
    assert Document.pending_version.property.post_update
    pointer_constraints = {
        str(constraint.name): constraint
        for constraint in _model_table(Document).constraints
        if isinstance(constraint, ForeignKeyConstraint)
        and constraint.name
        in {
            "fk_documents_current_version_same_parent",
            "fk_documents_pending_version_same_parent",
        }
    }
    assert set(pointer_constraints) == {
        "fk_documents_current_version_same_parent",
        "fk_documents_pending_version_same_parent",
    }
    assert all(constraint.use_alter for constraint in pointer_constraints.values())


def test_metadata_create_all_emits_document_pointer_fks_as_alter_table() -> None:
    statements: list[str] = []

    def capture(sql: ExecutableDDLElement, *args: object, **kwargs: object) -> None:
        del args, kwargs
        statements.append(_normalized_sql(sql.compile(dialect=POSTGRES_DIALECT)))

    engine = create_mock_engine("postgresql://", capture)
    Base.metadata.create_all(engine, checkfirst=False)

    document_create = next(
        index
        for index, statement in enumerate(statements)
        if statement.startswith("create table documents ")
    )
    version_create = next(
        index
        for index, statement in enumerate(statements)
        if statement.startswith("create table document_versions ")
    )
    pointer_alters = {
        name: next(
            index
            for index, statement in enumerate(statements)
            if statement.startswith("alter table documents add constraint") and name in statement
        )
        for name in (
            "fk_documents_current_version_same_parent",
            "fk_documents_pending_version_same_parent",
        )
    }
    assert all(index > document_create for index in pointer_alters.values())
    assert all(index > version_create for index in pointer_alters.values())


def test_task_three_primary_keys_uniques_and_foreign_keys_are_exact() -> None:
    assert {
        name: _column_names(table.primary_key.columns) for name, table in TASK_THREE_TABLES.items()
    } == {
        "documents": ("id",),
        "document_versions": ("id",),
        "document_index_states": ("document_version_id", "index_generation_id"),
        "jobs": ("id",),
        "provider_configs": ("id",),
        "model_profiles": ("id",),
        "model_profile_fallbacks": ("profile_id", "fallback_profile_id"),
        "sparse_profiles": ("id",),
        "query_profiles": ("id",),
        "query_logs": ("id",),
        "provider_usage": ("id",),
    }
    assert _unique_constraints(_model_table(Document)) == {
        ("uq_documents_id_knowledge_base", ("id", "knowledge_base_id"))
    }
    assert _unique_constraints(_model_table(DocumentVersion)) == {
        ("uq_document_versions_id_document", ("id", "document_id")),
        ("uq_document_versions_document_number", ("document_id", "version_number")),
    }
    assert _unique_constraints(_model_table(ModelProfileFallback)) == {
        ("uq_model_profile_fallbacks_profile_priority", ("profile_id", "priority"))
    }
    assert _unique_constraints(_model_table(ProviderConfig)) == {
        ("uq_provider_configs_name", ("name",))
    }
    assert _unique_constraints(_model_table(ModelProfile)) == {
        ("uq_model_profiles_name", ("name",))
    }
    assert _unique_constraints(_model_table(SparseProfile)) == {
        ("uq_sparse_profiles_name", ("name",)),
        ("uq_sparse_profiles_config_hash", ("config_hash",)),
    }
    assert _unique_constraints(_model_table(QueryProfile)) == {
        ("uq_query_profiles_name", ("name",))
    }
    for table in (
        _model_table(DocumentIndexState),
        _model_table(Job),
        _model_table(QueryLog),
        _model_table(ProviderUsage),
    ):
        assert _unique_constraints(table) == set()
    assert _foreign_keys(_model_table(DocumentIndexState)) == {
        "document_version_id": ("document_versions.id", "CASCADE"),
        "index_generation_id": ("knowledge_base_index_generations.id", "CASCADE"),
    }
    job_fks = _foreign_keys(_model_table(Job))
    assert job_fks == {
        "knowledge_base_id": ("knowledge_bases.id", "RESTRICT"),
        "actor_api_key_id": ("api_keys.id", "RESTRICT"),
        "index_generation_id": ("knowledge_base_index_generations.id", "RESTRICT"),
        "mutation_id": ("knowledge_base_mutations.id", "RESTRICT"),
        "parent_job_id": ("jobs.id", "RESTRICT"),
        "root_job_id": ("jobs.id", "RESTRICT"),
    }
    assert Job.__table__.c.target_id.nullable is False
    assert not Job.__table__.c.target_id.foreign_keys
    assert _foreign_keys(_model_table(ModelProfile)) == {
        "provider_config_id": ("provider_configs.id", "RESTRICT")
    }
    assert _foreign_keys(_model_table(ModelProfileFallback)) == {
        "profile_id": ("model_profiles.id", "CASCADE"),
        "fallback_profile_id": ("model_profiles.id", "RESTRICT"),
    }


def test_task_three_single_column_foreign_key_constraints_have_stable_names() -> None:
    assert _foreign_key_constraints(_model_table(DocumentIndexState)) == {
        "fk_document_index_states_document_version": (
            ("document_version_id",),
            ("document_versions.id",),
            "CASCADE",
            None,
            None,
        ),
        "fk_document_index_states_index_generation": (
            ("index_generation_id",),
            ("knowledge_base_index_generations.id",),
            "CASCADE",
            None,
            None,
        ),
    }
    assert _foreign_key_constraints(_model_table(Job)) == {
        "fk_jobs_knowledge_base": (
            ("knowledge_base_id",),
            ("knowledge_bases.id",),
            "RESTRICT",
            None,
            None,
        ),
        "fk_jobs_actor_api_key": (
            ("actor_api_key_id",),
            ("api_keys.id",),
            "RESTRICT",
            None,
            None,
        ),
        "fk_jobs_index_generation": (
            ("index_generation_id",),
            ("knowledge_base_index_generations.id",),
            "RESTRICT",
            None,
            None,
        ),
        "fk_jobs_mutation": (
            ("mutation_id",),
            ("knowledge_base_mutations.id",),
            "RESTRICT",
            None,
            None,
        ),
        "fk_jobs_parent": (
            ("parent_job_id",),
            ("jobs.id",),
            "RESTRICT",
            None,
            None,
        ),
        "fk_jobs_root": (
            ("root_job_id",),
            ("jobs.id",),
            "RESTRICT",
            None,
            None,
        ),
    }
    assert _foreign_key_constraints(_model_table(ModelProfile)) == {
        "fk_model_profiles_provider_config": (
            ("provider_config_id",),
            ("provider_configs.id",),
            "RESTRICT",
            None,
            None,
        )
    }
    assert _foreign_key_constraints(_model_table(ModelProfileFallback)) == {
        "fk_model_profile_fallbacks_profile": (
            ("profile_id",),
            ("model_profiles.id",),
            "CASCADE",
            None,
            None,
        ),
        "fk_model_profile_fallbacks_fallback": (
            ("fallback_profile_id",),
            ("model_profiles.id",),
            "RESTRICT",
            None,
            None,
        ),
    }
    assert _foreign_key_constraints(_model_table(QueryProfile)) == {
        "fk_query_profiles_rerank_profile": (
            ("rerank_profile_id",),
            ("model_profiles.id",),
            "RESTRICT",
            None,
            None,
        ),
        "fk_query_profiles_chat_profile": (
            ("chat_profile_id",),
            ("model_profiles.id",),
            "RESTRICT",
            None,
            None,
        ),
    }
    assert _foreign_key_constraints(_model_table(QueryLog)) == {
        "fk_query_logs_actor_api_key": (
            ("actor_api_key_id",),
            ("api_keys.id",),
            "SET NULL",
            None,
            None,
        ),
        "fk_query_logs_query_profile": (
            ("query_profile_id",),
            ("query_profiles.id",),
            "SET NULL",
            None,
            None,
        ),
    }
    assert _foreign_key_constraints(_model_table(ProviderUsage)) == {
        "fk_provider_usage_actor_api_key": (
            ("actor_api_key_id",),
            ("api_keys.id",),
            "SET NULL",
            None,
            None,
        ),
        "fk_provider_usage_provider_config": (
            ("provider_config_id",),
            ("provider_configs.id",),
            "SET NULL",
            None,
            None,
        ),
        "fk_provider_usage_model_profile": (
            ("model_profile_id",),
            ("model_profiles.id",),
            "SET NULL",
            None,
            None,
        ),
    }


def test_task_three_check_definitions_are_authoritative() -> None:
    assert _check_definitions(_model_table(Document)) == {
        "ck_documents_checksum": "checksum_sha256 is null or checksum_sha256 ~ '^[0-9a-f]{64}$'",
        "ck_documents_display_name_length": "char_length(display_name) between 1 and 255",
        "ck_documents_mime_type_length": "mime_type is null or char_length(mime_type) between 1 and 255",
        "ck_documents_status": "status in ('processing', 'active', 'failed', 'deleting', 'deleted')",
        "ck_documents_tags": "cardinality(tags) <= 64 and array_position(tags, null) is null",
        "ck_documents_version_pointers_distinct": "current_version_id is null or pending_version_id is null or current_version_id <> pending_version_id",
    }
    assert _check_definitions(_model_table(DocumentVersion)) == {
        "ck_document_versions_chunk_config_hash": "chunk_config_hash is null or chunk_config_hash ~ '^[0-9a-f]{64}$'",
        "ck_document_versions_chunk_count_nonnegative": "chunk_count is null or chunk_count >= 0",
        "ck_document_versions_chunk_manifest_checksum": "chunk_manifest_checksum_sha256 is null or chunk_manifest_checksum_sha256 ~ '^[0-9a-f]{64}$'",
        "ck_document_versions_chunk_manifest_complete": "num_nonnulls(chunk_manifest_object_key, chunk_manifest_checksum_sha256, chunk_config_hash) in (0, 3)",
        "ck_document_versions_chunk_manifest_object_key_length": "chunk_manifest_object_key is null or char_length(chunk_manifest_object_key) between 1 and 1024",
        "ck_document_versions_chunker_name_length": "chunker_name is null or char_length(chunker_name) between 1 and 120",
        "ck_document_versions_chunker_version_length": "chunker_version is null or char_length(chunker_version) between 1 and 64",
        "ck_document_versions_declared_mime_length": "declared_mime_type is null or char_length(declared_mime_type) between 1 and 255",
        "ck_document_versions_detected_mime_length": "detected_mime_type is null or char_length(detected_mime_type) between 1 and 255",
        "ck_document_versions_parsed_object_checksum": "parsed_object_checksum_sha256 is null or parsed_object_checksum_sha256 ~ '^[0-9a-f]{64}$'",
        "ck_document_versions_parsed_object_key_length": "parsed_object_key is null or char_length(parsed_object_key) between 1 and 1024",
        "ck_document_versions_parser_name_length": "parser_name is null or char_length(parser_name) between 1 and 120",
        "ck_document_versions_parser_version_length": "parser_version is null or char_length(parser_version) between 1 and 64",
        "ck_document_versions_source_checksum": "source_checksum_sha256 ~ '^[0-9a-f]{64}$'",
        "ck_document_versions_source_extension_length": "source_extension is null or char_length(source_extension) between 1 and 32",
        "ck_document_versions_source_object_key_length": "char_length(source_object_key) between 1 and 1024",
        "ck_document_versions_status": "status in ('uploaded', 'parsing', 'chunking', 'embedding', 'indexing', 'ready', 'failed', 'conflicted', 'cancelled', 'ocr_required', 'superseded')",
        "ck_document_versions_version_positive": "version_number > 0",
    }
    assert _check_definitions(_model_table(DocumentIndexState)) == {
        "ck_document_index_states_actual_count": "actual_point_count is null or actual_point_count >= 0",
        "ck_document_index_states_embedding_config_hash": "embedding_config_hash is null or embedding_config_hash ~ '^[0-9a-f]{64}$'",
        "ck_document_index_states_error_code_length": "error_code is null or char_length(error_code) between 1 and 64",
        "ck_document_index_states_expected_count": "expected_point_count is null or expected_point_count >= 0",
        "ck_document_index_states_manifest_checksum": "chunk_manifest_checksum_sha256 is null or chunk_manifest_checksum_sha256 ~ '^[0-9a-f]{64}$'",
        "ck_document_index_states_next_chunk_index_nonnegative": "next_chunk_index >= 0",
        "ck_document_index_states_safe_error_message_length": "safe_error_message is null or char_length(safe_error_message) between 1 and 500",
        "ck_document_index_states_status": "status in ('queued', 'embedding', 'indexing', 'validated', 'failed', 'retired')",
    }
    assert _check_definitions(_model_table(Job)) == {
        "ck_jobs_attempt_count": "attempt_count >= 0 and attempt_count <= max_attempts",
        "ck_jobs_error_code_length": "error_code is null or char_length(error_code) between 1 and 64",
        "ck_jobs_error_message_sanitized_length": "error_message_sanitized is null or char_length(error_message_sanitized) between 1 and 500",
        "ck_jobs_idempotency_key": "idempotency_key is null or (char_length(idempotency_key) between 1 and 128 and idempotency_key ~ '^[!-~]+$')",
        "ck_jobs_lease_epoch_nonnegative": "lease_epoch >= 0",
        "ck_jobs_lease_owner_length": "lease_owner is null or char_length(lease_owner) between 1 and 255",
        "ck_jobs_lease_state_invariant": "(status = 'running' and lease_owner is not null and lease_expires_at is not null and worker_heartbeat_at is not null) or (status <> 'running' and lease_owner is null and lease_expires_at is null)",
        "ck_jobs_max_attempts": "max_attempts between 1 and 100",
        "ck_jobs_operation": "operation in ('ingest_document', 'index_document', 'delete_document', 'rebuild_generation', 'apply_filter_schema', 'cleanup_generation', 'cleanup_document_version', 'purge_knowledge_base')",
        "ck_jobs_parent_not_self": "parent_job_id is null or parent_job_id <> id",
        "ck_jobs_progress": "progress_current >= 0 and (progress_total is null or (progress_total >= 0 and progress_current <= progress_total))",
        "ck_jobs_root_not_self": "root_job_id is null or root_job_id <> id",
        "ck_jobs_resume_stage_length": "resume_stage is null or char_length(resume_stage) between 1 and 64",
        "ck_jobs_stage_length": "stage is null or char_length(stage) between 1 and 64",
        "ck_jobs_status": "status in ('queued', 'running', 'retry_wait', 'succeeded', 'failed', 'cancelled')",
        "ck_jobs_target_revision": "target_revision is null or target_revision >= 0",
        "ck_jobs_target_type": "target_type in ('document_version', 'index_generation', 'filter_schema_revision', 'knowledge_base')",
    }


def test_provider_and_query_profile_check_definitions_are_authoritative() -> None:
    provider_checks = _check_definitions(_model_table(ProviderConfig))
    assert set(provider_checks) == {
        "ck_provider_configs_base_url_http",
        "ck_provider_configs_base_url_length",
        "ck_provider_configs_base_url_port",
        "ck_provider_configs_credential_endpoint_validation",
        "ck_provider_configs_credential_source_exactly_one",
        "ck_provider_configs_endpoint_policy_version_length",
        "ck_provider_configs_default_headers_http_referer",
        "ck_provider_configs_default_headers_key_count",
        "ck_provider_configs_default_headers_keys",
        "ck_provider_configs_default_headers_object",
        "ck_provider_configs_default_headers_size",
        "ck_provider_configs_default_headers_titles",
        "ck_provider_configs_default_headers_value_types",
        "ck_provider_configs_max_concurrency",
        "ck_provider_configs_name_length",
        "ck_provider_configs_provider_type",
        "ck_provider_configs_requests_per_minute",
        "ck_provider_configs_revision_positive",
        "ck_provider_configs_routing_options_booleans",
        "ck_provider_configs_routing_options_data_collection",
        "ck_provider_configs_routing_options_key_count",
        "ck_provider_configs_routing_options_keys",
        "ck_provider_configs_routing_options_max_latency",
        "ck_provider_configs_routing_options_max_price",
        "ck_provider_configs_routing_options_min_throughput",
        "ck_provider_configs_routing_options_object",
        "ck_provider_configs_routing_options_provider_arrays",
        "ck_provider_configs_routing_options_provider_scope",
        "ck_provider_configs_routing_options_quantizations",
        "ck_provider_configs_routing_options_size",
        "ck_provider_configs_routing_options_sort",
        "ck_provider_configs_secret_ref_length",
        "ck_provider_configs_secret_ref_scheme",
        "ck_provider_configs_timeout",
    }
    assert provider_checks["ck_provider_configs_base_url_http"] == (
        "base_url ~* '^https?://(\\[[0-9A-Fa-f:.]+\\]|[^:/?#@[:space:][:cntrl:]]+)"
        "(:[0-9]{1,5})?"
        "(/[^?#[:space:][:cntrl:]]*)?$'"
    )
    assert provider_checks["ck_provider_configs_endpoint_policy_version_length"] == (
        "endpoint_policy_version is null or char_length(endpoint_policy_version) between 1 and 64"
    )
    base_url_port_sql = provider_checks["ck_provider_configs_base_url_port"]
    assert "regexp_match(base_url" in base_url_port_sql
    assert "::integer between 1 and 65535" in base_url_port_sql
    assert provider_checks["ck_provider_configs_default_headers_size"] == (
        "pg_column_size(default_headers) <= 4096"
    )
    assert provider_checks["ck_provider_configs_default_headers_key_count"] == (
        "case when jsonb_typeof(default_headers) = 'object' then "
        "jsonb_array_length(jsonb_path_query_array(default_headers, "
        "'$ ? (@.type() == \"object\").keyvalue()')) <= 3 else false end"
    )
    header_keys_sql = provider_checks["ck_provider_configs_default_headers_keys"]
    assert "array['HTTP-Referer', 'X-OpenRouter-Title', 'X-Title']" in header_keys_sql
    assert "= '{}'::jsonb" in header_keys_sql
    header_types_sql = provider_checks["ck_provider_configs_default_headers_value_types"]
    for key in ("HTTP-Referer", "X-OpenRouter-Title", "X-Title"):
        assert f"default_headers ? '{key}'" in header_types_sql
        assert f"jsonb_typeof(default_headers -> '{key}') = 'string'" in header_types_sql
    assert (
        "between 1 and 2048" in provider_checks["ck_provider_configs_default_headers_http_referer"]
    )
    assert "between 1 and 120" in provider_checks["ck_provider_configs_default_headers_titles"]
    assert "[[:cntrl:]]" in provider_checks["ck_provider_configs_default_headers_titles"]

    assert provider_checks["ck_provider_configs_routing_options_size"] == (
        "pg_column_size(routing_options) <= 16384"
    )
    assert provider_checks["ck_provider_configs_routing_options_provider_scope"] == (
        "provider_type = 'openrouter' or routing_options = '{}'::jsonb"
    )
    routing_keys_sql = provider_checks["ck_provider_configs_routing_options_keys"]
    for key in (
        "order",
        "allow_fallbacks",
        "require_parameters",
        "data_collection",
        "zdr",
        "enforce_distillable_text",
        "only",
        "ignore",
        "quantizations",
        "sort",
        "preferred_min_throughput",
        "preferred_max_latency",
        "max_price",
    ):
        assert f"'{key}'" in routing_keys_sql
    arrays_sql = provider_checks["ck_provider_configs_routing_options_provider_arrays"]
    for key in ("order", "only", "ignore"):
        assert f"routing_options ? '{key}'" in arrays_sql
        assert f"jsonb_typeof((routing_options -> '{key}')) = 'null'" in arrays_sql
    assert r"^[^\u0001-\u001F\u007F]{1,255}$" in arrays_sql
    assert "jsonb_array_length" in arrays_sql
    booleans_sql = provider_checks["ck_provider_configs_routing_options_booleans"]
    for key in (
        "allow_fallbacks",
        "require_parameters",
        "zdr",
        "enforce_distillable_text",
    ):
        assert f"routing_options ? '{key}'" in booleans_sql
        assert f"jsonb_typeof(routing_options -> '{key}') in ('boolean', 'null')" in (booleans_sql)
    data_collection_sql = provider_checks["ck_provider_configs_routing_options_data_collection"]
    assert "jsonb_typeof(routing_options -> 'data_collection') = 'null'" in (data_collection_sql)
    assert "('allow', 'deny')" in data_collection_sql
    quantizations_sql = provider_checks["ck_provider_configs_routing_options_quantizations"]
    assert "jsonb_typeof((routing_options -> 'quantizations')) = 'null'" in (quantizations_sql)
    for value in ("int4", "int8", "fp4", "fp6", "fp8", "fp16", "bf16", "fp32", "unknown"):
        assert value in quantizations_sql
    sort_sql = provider_checks["ck_provider_configs_routing_options_sort"]
    for value in ("price", "throughput", "latency", "exacto", "model", "none"):
        assert f"'{value}'" in sort_sql
    assert "jsonb_typeof((routing_options -> 'sort')) = 'null'" in sort_sql
    assert "between 0 and 2" in sort_sql
    assert "not ((routing_options -> 'sort') ? 'by')" in sort_sql
    assert "jsonb_typeof((routing_options -> 'sort') -> 'by') = 'null'" in sort_sql
    assert "jsonb_typeof((routing_options -> 'sort') -> 'partition') = 'null'" in sort_sql
    for check_name in (
        "ck_provider_configs_routing_options_min_throughput",
        "ck_provider_configs_routing_options_max_latency",
    ):
        threshold_sql = provider_checks[check_name]
        for percentile in ("p50", "p75", "p90", "p99"):
            assert f"'{percentile}'" in threshold_sql
        assert "= 'null'" in threshold_sql
        assert "between 0 and 4" in threshold_sql
        assert "::numeric >= 0" in threshold_sql
    max_price_sql = provider_checks["ck_provider_configs_routing_options_max_price"]
    for key in ("audio", "prompt", "completion", "request", "image"):
        assert f"'{key}'" in max_price_sql
        assert f"jsonb_typeof((routing_options -> 'max_price') -> '{key}') = 'string'" in (
            max_price_sql
        )
    assert "between 1 and 64" in max_price_sql
    assert "between 0 and 5" in max_price_sql
    assert "^(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?$" in max_price_sql
    assert "::numeric" not in max_price_sql
    assert all("$.**" not in sql for sql in provider_checks.values())
    assert all("jsonb_object_length" not in sql for sql in provider_checks.values())
    assert all("no_credentials" not in name for name in provider_checks)
    assert _check_definitions(_model_table(ModelProfile)) == {
        "ck_model_profiles_batch_size": "batch_size between 1 and 10000",
        "ck_model_profiles_capability": "capability in ('embedding', 'rerank', 'chat')",
        "ck_model_profiles_dimension": "(capability = 'embedding' and dimension is not null and dimension > 0) or (capability in ('rerank', 'chat') and dimension is null)",
        "ck_model_profiles_max_input_tokens": "max_input_tokens between 1 and 10000000",
        "ck_model_profiles_model_name_length": "char_length(model_name) between 1 and 255",
        "ck_model_profiles_name_length": "char_length(name) between 1 and 120",
        "ck_model_profiles_revision_positive": "resource_revision >= 1",
        "ck_model_profiles_timeout": "timeout_seconds > 0 and timeout_seconds <= 600",
        "ck_model_profiles_vector_config_object": "jsonb_typeof(vector_config) = 'object'",
    }
    assert _check_definitions(_model_table(ModelProfileFallback)) == {
        "ck_model_profile_fallbacks_distinct": "profile_id <> fallback_profile_id",
        "ck_model_profile_fallbacks_priority": "priority between 1 and 100",
    }
    assert _check_definitions(_model_table(SparseProfile)) == {
        "ck_sparse_profiles_algorithm": "algorithm = 'qdrant_bm25_v1'",
        "ck_sparse_profiles_config_hash": "config_hash ~ '^[0-9a-f]{64}$'",
        "ck_sparse_profiles_encoder_package_length": "char_length(encoder_package) between 1 and 255",
        "ck_sparse_profiles_encoder_version_length": "char_length(encoder_version) between 1 and 64",
        "ck_sparse_profiles_idf_checksum": "idf_checksum_sha256 ~ '^[0-9a-f]{64}$'",
        "ck_sparse_profiles_idf_object_key_length": "char_length(idf_object_key) between 1 and 1024",
        "ck_sparse_profiles_name_length": "char_length(name) between 1 and 120",
        "ck_sparse_profiles_tokenizer_name_length": "char_length(tokenizer_name) between 1 and 120",
        "ck_sparse_profiles_tokenizer_version_length": "char_length(tokenizer_version) between 1 and 64",
    }
    assert _check_definitions(_model_table(QueryProfile)) == {
        "ck_query_profiles_candidate_limits": "dense_candidate_limit between 1 and 1000 and sparse_candidate_limit between 1 and 1000 and rrf_candidate_limit between 1 and 1000 and rerank_candidate_limit between 1 and 1000 and rerank_candidate_limit <= rrf_candidate_limit",
        "ck_query_profiles_context_budget": "context_token_budget between 1 and 1000000",
        "ck_query_profiles_min_rerank_score": "min_rerank_score between -1 and 1",
        "ck_query_profiles_min_rrf_degraded": "min_rrf_score_when_degraded between 0 and 1",
        "ck_query_profiles_name_length": "char_length(name) between 1 and 120",
        "ck_query_profiles_top_k_limit": "top_k_limit between 1 and 100 and top_k_limit <= rerank_candidate_limit",
    }


def test_observability_contracts_are_bounded_and_content_free() -> None:
    assert _check_definitions(_model_table(QueryLog)) == {
        "ck_query_logs_kb_ids": "cardinality(knowledge_base_ids) between 1 and 64 and array_position(knowledge_base_ids, null) is null",
        "ck_query_logs_latency": "latency_ms >= 0",
        "ck_query_logs_request_id_length": "char_length(request_id) between 1 and 128",
        "ck_query_logs_status": "status in ('succeeded', 'failed', 'rejected')",
    }
    assert _check_definitions(_model_table(ProviderUsage)) == {
        "ck_provider_usage_capability": "capability in ('embedding', 'rerank', 'chat')",
        "ck_provider_usage_cost": "cost_micros >= 0",
        "ck_provider_usage_currency": "currency ~ '^[A-Z]{3}$'",
        "ck_provider_usage_error_code_length": "error_code is null or char_length(error_code) between 1 and 64",
        "ck_provider_usage_latency": "latency_ms >= 0",
        "ck_provider_usage_model_identifier_length": "char_length(model_identifier) between 1 and 255",
        "ck_provider_usage_provider_identifier_length": "char_length(provider_identifier) between 1 and 120",
        "ck_provider_usage_provider_request_id_length": "provider_request_id is null or char_length(provider_request_id) between 1 and 255",
        "ck_provider_usage_request_id_length": "char_length(request_id) between 1 and 128",
        "ck_provider_usage_route_identifier_length": "route_identifier is null or char_length(route_identifier) between 1 and 255",
        "ck_provider_usage_status": "status in ('succeeded', 'failed', 'rate_limited', 'timeout', 'cancelled')",
        "ck_provider_usage_tokens": "input_tokens >= 0 and output_tokens >= 0",
    }
    forbidden_query_log_columns = {
        "query_text",
        "retrieved_text",
        "retrieved_content",
        "prompt",
        "prompt_text",
        "response",
        "response_text",
        "secret",
        "token_text",
    }
    assert forbidden_query_log_columns.isdisjoint(QueryLog.__table__.columns.keys())
    allowed_token_counters = {"input_tokens", "output_tokens"}
    assert all(
        "token" not in column or column in allowed_token_counters
        for column in ProviderUsage.__table__.columns.keys()  # noqa: SIM118
    )
    assert all(
        not any(fragment in column for fragment in ("prompt", "response", "secret"))
        for column in ProviderUsage.__table__.columns.keys()  # noqa: SIM118
    )
    provider_columns = set(ProviderConfig.__table__.columns.keys())
    assert "secret_ref" in provider_columns
    assert not provider_columns & {"credential", "credential_value", "api_key", "token", "secret"}

    currency_constraint = next(
        constraint
        for constraint in _model_table(ProviderUsage).constraints
        if isinstance(constraint, CheckConstraint)
        and constraint.name == "ck_provider_usage_currency"
    )
    raw_currency_sql = str(currency_constraint.sqltext.compile(dialect=POSTGRES_DIALECT))
    assert "'^[A-Z]{3}$'" in raw_currency_sql
    assert "'^[a-z]{3}$'" not in raw_currency_sql


def test_task_three_defaults_are_exact_and_mutable_values_are_independent() -> None:
    expected_server = {
        "documents": {
            "status": "'processing'",
            "tags": "'{}'::text[]",
            "metadata": "'{}'::jsonb",
            "created_at": "now()",
            "updated_at": "now()",
        },
        "document_versions": {
            "parser_config": "'{}'::jsonb",
            "chunker_config": "'{}'::jsonb",
            "status": "'uploaded'",
            "created_at": "now()",
        },
        "document_index_states": {
            "status": "'queued'",
            "created_at": "now()",
            "next_chunk_index": "0",
        },
        "jobs": {
            "status": "'queued'",
            "progress_current": "0",
            "attempt_count": "0",
            "max_attempts": "5",
            "created_at": "now()",
            "updated_at": "now()",
            "lease_epoch": "0",
            "retryable": "true",
        },
        "provider_configs": {
            "default_headers": "'{}'::jsonb",
            "routing_options": "'{}'::jsonb",
            "enabled": "true",
            "created_at": "now()",
            "updated_at": "now()",
            "resource_revision": "1",
        },
        "model_profiles": {
            "enabled": "true",
            "created_at": "now()",
            "updated_at": "now()",
            "resource_revision": "1",
            "vector_config": "'{}'::jsonb",
        },
        "model_profile_fallbacks": {},
        "sparse_profiles": {
            "language_config": "'{}'::jsonb",
            "term_frequency_config": "'{}'::jsonb",
            "length_normalization_config": "'{}'::jsonb",
            "oov_config": "'{}'::jsonb",
            "enabled": "true",
            "created_at": "now()",
            "updated_at": "now()",
        },
        "query_profiles": {
            "enabled": "true",
            "is_system_default": "false",
            "created_at": "now()",
            "updated_at": "now()",
        },
        "query_logs": {"degraded": "false", "created_at": "now()"},
        "provider_usage": {
            "input_tokens": "0",
            "output_tokens": "0",
            "cost_micros": "0",
            "degraded": "false",
            "created_at": "now()",
        },
    }
    assert {
        name: _server_defaults(table) for name, table in TASK_THREE_TABLES.items()
    } == expected_server
    assert {name: _orm_default_columns(table) for name, table in TASK_THREE_TABLES.items()} == {
        "documents": {"id", "status", "tags", "metadata"},
        "document_versions": {"id", "parser_config", "chunker_config", "status"},
        "document_index_states": {"status", "next_chunk_index"},
        "jobs": {
            "id",
            "status",
            "progress_current",
            "attempt_count",
            "max_attempts",
            "lease_epoch",
            "retryable",
        },
        "provider_configs": {
            "id",
            "default_headers",
            "routing_options",
            "enabled",
            "resource_revision",
        },
        "model_profiles": {"id", "enabled", "resource_revision", "vector_config"},
        "model_profile_fallbacks": set(),
        "sparse_profiles": {
            "id",
            "language_config",
            "term_frequency_config",
            "length_normalization_config",
            "oov_config",
            "enabled",
        },
        "query_profiles": {"id", "enabled", "is_system_default"},
        "query_logs": {"id", "degraded"},
        "provider_usage": {"id", "input_tokens", "output_tokens", "cost_micros", "degraded"},
    }
    mutable_columns = (
        Document.__table__.c.tags,
        Document.__table__.c.metadata,
        DocumentVersion.__table__.c.parser_config,
        DocumentVersion.__table__.c.chunker_config,
        ProviderConfig.__table__.c.default_headers,
        ProviderConfig.__table__.c.routing_options,
        ModelProfile.__table__.c.vector_config,
        SparseProfile.__table__.c.language_config,
        SparseProfile.__table__.c.term_frequency_config,
        SparseProfile.__table__.c.length_normalization_config,
        SparseProfile.__table__.c.oov_config,
    )
    for column in mutable_columns:
        default = column.default
        assert isinstance(default, CallableColumnDefault)
        assert _invoke_callable_default(default) is not _invoke_callable_default(default)

    assert QueryLog.__table__.c.knowledge_base_ids.default is None
    assert QueryLog.__table__.c.knowledge_base_ids.server_default is None

    scalar_defaults = {
        Document.__table__.c.status: "processing",
        DocumentVersion.__table__.c.status: "uploaded",
        DocumentIndexState.__table__.c.status: "queued",
        DocumentIndexState.__table__.c.next_chunk_index: 0,
        Job.__table__.c.status: "queued",
        Job.__table__.c.progress_current: 0,
        Job.__table__.c.attempt_count: 0,
        Job.__table__.c.max_attempts: 5,
        Job.__table__.c.retryable: True,
        ProviderConfig.__table__.c.enabled: True,
        ProviderConfig.__table__.c.resource_revision: 1,
        ModelProfile.__table__.c.enabled: True,
        ModelProfile.__table__.c.resource_revision: 1,
        SparseProfile.__table__.c.enabled: True,
        QueryProfile.__table__.c.enabled: True,
        QueryProfile.__table__.c.is_system_default: False,
        QueryLog.__table__.c.degraded: False,
        ProviderUsage.__table__.c.input_tokens: 0,
        ProviderUsage.__table__.c.output_tokens: 0,
        ProviderUsage.__table__.c.cost_micros: 0,
        ProviderUsage.__table__.c.degraded: False,
    }
    for column, expected in scalar_defaults.items():
        default = column.default
        assert isinstance(default, ScalarElementColumnDefault)
        assert default.arg == expected

    id_tables = tuple(
        table for table in TASK_THREE_TABLES.values() if "id" in table.c and table.c.id.primary_key
    )
    for table in id_tables:
        default = table.c.id.default
        assert isinstance(default, CallableColumnDefault)
        assert isinstance(_invoke_callable_default(default), UUID)
        assert _invoke_callable_default(default) != _invoke_callable_default(default)
        assert table.c.id.server_default is None

    for table in (
        _model_table(Document),
        _model_table(Job),
        _model_table(ProviderConfig),
        _model_table(ModelProfile),
        _model_table(SparseProfile),
        _model_table(QueryProfile),
    ):
        assert table.c.updated_at.onupdate is None
        assert table.c.updated_at.server_onupdate is None


def test_task_three_partial_unique_indexes_are_exact() -> None:
    document_indexes = {
        str(index.name): _index_contract(index) for index in _model_table(Document).indexes
    }
    assert document_indexes == {
        "uq_documents_kb_checksum_live": (
            True,
            ("knowledge_base_id", "checksum_sha256"),
            "deleted_at is null and checksum_sha256 is not null",
        )
    }
    query_indexes = {
        str(index.name): _index_contract(index) for index in _model_table(QueryProfile).indexes
    }
    assert query_indexes == {
        "uq_query_profiles_enabled_system_default": (
            True,
            ("is_system_default",),
            "enabled and is_system_default",
        )
    }
    job_table = _model_table(Job)
    job_indexes = {str(index.name): _index_contract(index) for index in job_table.indexes}
    assert job_indexes == {
        "ix_jobs_expired_leases": (
            False,
            ("lease_expires_at",),
            "status = 'running'",
        ),
        "ix_jobs_polling": (
            False,
            ("status", "next_retry_at", "created_at"),
            "status in ('queued', 'retry_wait')",
        ),
        "uq_jobs_active_target": (
            True,
            ("operation", "target_type", "target_id", "target_revision", "index_generation_id"),
            "status in ('queued', 'running', 'retry_wait')",
        ),
    }
    job_index = next(index for index in job_table.indexes if index.name == "uq_jobs_active_target")
    assert job_index.dialect_options["postgresql"]["nulls_not_distinct"] is True
    compiled = _normalized_sql(CreateIndex(job_index).compile(dialect=POSTGRES_DIALECT))
    assert "create unique index" in compiled
    assert "nulls not distinct" in compiled

    for table in (_model_table(Document), _model_table(DocumentVersion)):
        for constraint in table.constraints:
            if isinstance(constraint, ForeignKeyConstraint) and constraint.deferrable:
                compiled_constraint = _normalized_sql(
                    AddConstraint(constraint).compile(dialect=POSTGRES_DIALECT)
                )
                assert "deferrable initially deferred" in compiled_constraint


def test_knowledge_base_create_is_strict_bounded_and_canonical() -> None:
    command = KnowledgeBaseCreate(
        name="  Product manuals  ",
        description=None,
        metadata={"z": [1, True, None], "a": {"owner": "support"}},
    )

    assert command.name == "Product manuals"
    assert command.description is None
    assert command.metadata == {
        "z": [1, True, None],
        "a": {"owner": "support"},
    }

    invalid_payloads = (
        {"name": ""},
        {"name": "x" * 121},
        {"name": 123},
        {"name": "valid", "description": "x" * 2001},
        {"name": "valid", "unknown": "forbidden"},
    )
    for payload in invalid_payloads:
        with pytest.raises(ValidationError):
            KnowledgeBaseCreate.model_validate(payload)


def test_knowledge_base_metadata_uses_shared_json_bounds_without_echoing_values() -> None:
    sentinel = "sensitive-metadata-value-" * 2_000
    invalid_metadata = (
        {str(index): index for index in range(65)},
        {"a": {"b": {"c": {"d": {"e": 1}}}}},
        {"payload": sentinel},
    )

    for metadata in invalid_metadata:
        with pytest.raises(BusinessError) as raised:
            KnowledgeBaseCreate(name="bounded", metadata=cast(Any, metadata))
        assert (raised.value.status_code, raised.value.code) == (422, "VALIDATION_ERROR")
        assert sentinel not in str(raised.value)


def test_knowledge_base_patch_distinguishes_omitted_and_explicit_null() -> None:
    with pytest.raises(ValidationError):
        KnowledgeBasePatch()

    assert KnowledgeBasePatch(description=None).model_fields_set == {"description"}
    assert KnowledgeBasePatch(name=" renamed ").name == "renamed"
    assert KnowledgeBasePatch(status="disabled").status == "disabled"

    for payload in (
        {"name": None},
        {"metadata": None},
        {"status": None},
        {"status": "deleting"},
        {"unknown": "forbidden"},
    ):
        with pytest.raises(ValidationError):
            KnowledgeBasePatch.model_validate(payload)


def test_knowledge_base_create_fingerprint_is_deterministic_and_sensitive_to_request() -> None:
    first = KnowledgeBaseCreate(
        name="Manuals",
        description="Current manuals",
        metadata={"z": 1, "a": {"two": 2, "one": 1}},
    )
    same = KnowledgeBaseCreate(
        metadata={"a": {"one": 1, "two": 2}, "z": 1},
        description="Current manuals",
        name="Manuals",
    )
    changed = KnowledgeBaseCreate(
        name="Manuals",
        description="Different manuals",
        metadata={"z": 1, "a": {"two": 2, "one": 1}},
    )

    first_fingerprint = knowledge_base_create_fingerprint(first)
    assert first_fingerprint == knowledge_base_create_fingerprint(same)
    assert first_fingerprint != knowledge_base_create_fingerprint(changed)
    assert len(first_fingerprint) == 32


def test_safe_knowledge_base_contains_only_public_fields_and_exact_etag() -> None:
    knowledge_base_id = uuid4()
    now = datetime(2026, 7, 25, 8, 0, tzinfo=UTC)
    safe = SafeKnowledgeBase(
        id=knowledge_base_id,
        name="Manuals",
        description=None,
        status="active",
        metadata={"owner": "support"},
        resource_revision=3,
        mutation_revision=1,
        filter_schema_revision=1,
        active_index_generation_id=None,
        pending_index_generation_id=None,
        created_at=now,
        updated_at=now,
        etag=f'"kb:{knowledge_base_id}:r3"',
    )

    assert set(safe.model_dump()) == {
        "id",
        "name",
        "description",
        "status",
        "metadata",
        "resource_revision",
        "mutation_revision",
        "filter_schema_revision",
        "active_index_generation_id",
        "pending_index_generation_id",
        "rerank_profile_id",
        "created_at",
        "updated_at",
        "etag",
    }
    assert safe.etag == f'"kb:{knowledge_base_id}:r3"'
    with pytest.raises(ValidationError):
        SafeKnowledgeBase.model_validate({**safe.model_dump(), "filter_schema": {"fields": []}})

    for changes in (
        {"etag": f'"kb:{knowledge_base_id}:r2"'},
        {"created_at": now.replace(tzinfo=None)},
        {"metadata": ["not", "an", "object"]},
        {"metadata": {"payload": "x" * (32 * 1024)}},
    ):
        with pytest.raises(ValidationError):
            SafeKnowledgeBase.model_validate({**safe.model_dump(), **changes})
    with pytest.raises(TypeError):
        knowledge_base_create_fingerprint(cast(Any, object()))


class _FakeMetadataTransaction:
    def __init__(self, session: "_FakeMetadataSession", *, nested: bool) -> None:
        self._session = session
        self._nested = nested
        self._snapshot: object | None = None

    async def __aenter__(self) -> None:
        if self._nested:
            self._session.nested_begin_count += 1
        else:
            self._session.begin_count += 1
        self._snapshot = self._session.snapshot()

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exception, traceback
        if exception_type is not None:
            self._session.restore(self._snapshot)


class _FakeMetadataSession:
    def __init__(self) -> None:
        self.begin_count = 0
        self.nested_begin_count = 0
        self.agents: dict[UUID, ApiKey] = {}
        self.knowledge_bases: dict[UUID, KnowledgeBase] = {}
        self.scopes: set[tuple[UUID, UUID]] = set()
        self.idempotency: dict[tuple[UUID, str, str], IdempotencyRecord] = {}
        self.audit_events: list[AuditEvent] = []
        self.mutations: list[KnowledgeBaseMutation] = []
        self.model_profile_capabilities: dict[UUID, str] = {}
        self.jobs: list[Job] = []
        self.audit_error: BaseException | None = None
        self.idempotency_error: IntegrityError | None = None
        self.idempotency_get_count = 0
        self.list_error: BaseException | None = None
        self.get_error: BaseException | None = None
        self.save_error: BaseException | None = None

    def begin(self) -> _FakeMetadataTransaction:
        return _FakeMetadataTransaction(self, nested=False)

    def begin_nested(self) -> _FakeMetadataTransaction:
        return _FakeMetadataTransaction(self, nested=True)

    def snapshot(self) -> object:
        return deepcopy(
            (
                self.agents,
                self.knowledge_bases,
                self.scopes,
                self.idempotency,
                self.audit_events,
                self.mutations,
            )
        )

    def restore(self, snapshot: object | None) -> None:
        assert snapshot is not None
        (
            self.agents,
            self.knowledge_bases,
            self.scopes,
            self.idempotency,
            self.audit_events,
            self.mutations,
        ) = cast(
            tuple[
                dict[UUID, ApiKey],
                dict[UUID, KnowledgeBase],
                set[tuple[UUID, UUID]],
                dict[tuple[UUID, str, str], IdempotencyRecord],
                list[AuditEvent],
                list[KnowledgeBaseMutation],
            ],
            snapshot,
        )


class _FakeKnowledgeBases:
    def __init__(self, session: _FakeMetadataSession) -> None:
        self._session = session

    async def list_scoped(
        self,
        actor_key_id: UUID,
        position: CursorPosition | None,
        limit: int,
    ) -> list[KnowledgeBase]:
        if self._session.list_error is not None:
            raise self._session.list_error
        rows = [
            row
            for row in self._session.knowledge_bases.values()
            if (actor_key_id, row.id) in self._session.scopes and row.status != "deleting"
        ]
        if position is not None:
            rows = [
                row for row in rows if (row.created_at, row.id) > (position.created_at, position.id)
            ]
        return sorted(rows, key=lambda row: (row.created_at, row.id))[:limit]

    async def get_scoped(
        self,
        actor_key_id: UUID,
        knowledge_base_id: UUID,
        *,
        for_update: bool = False,
    ) -> KnowledgeBase | None:
        del for_update
        if self._session.get_error is not None:
            raise self._session.get_error
        if (actor_key_id, knowledge_base_id) not in self._session.scopes:
            return None
        return self._session.knowledge_bases.get(knowledge_base_id)

    async def add(self, row: KnowledgeBase) -> None:
        row.created_at = datetime(2026, 7, 25, 8, 0, tzinfo=UTC)
        row.updated_at = row.created_at
        self._session.knowledge_bases[row.id] = row

    async def save(self, row: KnowledgeBase) -> None:
        if self._session.save_error is not None:
            raise self._session.save_error
        self._session.knowledge_bases[row.id] = row

    async def reload(self, row: KnowledgeBase) -> KnowledgeBase:
        return self._session.knowledge_bases[row.id]


class _FakeMetadataActors:
    def __init__(self, session: _FakeMetadataSession) -> None:
        self._session = session

    async def get_agent_for_update(self, key_id: UUID) -> ApiKey | None:
        return self._session.agents.get(key_id)

    async def add_scope(self, actor_key_id: UUID, knowledge_base_id: UUID) -> None:
        self._session.scopes.add((actor_key_id, knowledge_base_id))


class _FakeMetadataIdempotency:
    def __init__(self, session: _FakeMetadataSession) -> None:
        self._session = session

    async def get(
        self,
        actor_key_id: UUID,
        operation: str,
        idempotency_key: str,
    ) -> IdempotencyRecord | None:
        self._session.idempotency_get_count += 1
        return self._session.idempotency.get((actor_key_id, operation, idempotency_key))

    async def add(self, record: IdempotencyRecord) -> None:
        if self._session.idempotency_error is not None:
            raise self._session.idempotency_error
        record.created_at = datetime(2026, 7, 25, 8, 0, tzinfo=UTC)
        record.updated_at = record.created_at
        self._session.idempotency[
            (record.actor_key_id, record.operation, record.idempotency_key)
        ] = record


class _FakeMetadataAudits:
    def __init__(self, session: _FakeMetadataSession) -> None:
        self._session = session

    async def add(self, event: AuditEvent) -> None:
        self._session.audit_events.append(event)
        if self._session.audit_error is not None:
            raise self._session.audit_error


class _FakeMetadataMutations:
    def __init__(self, session: _FakeMetadataSession) -> None:
        self._session = session

    async def add(self, mutation: KnowledgeBaseMutation) -> None:
        self._session.mutations.append(mutation)


class _FakeMetadataJobs:
    def __init__(self, session: _FakeMetadataSession) -> None:
        self._session = session

    async def add(self, job: Job) -> None:
        self._session.jobs.append(job)


class _FakeModelProfiles:
    def __init__(self, session: _FakeMetadataSession) -> None:
        self._session = session

    async def capability_of(self, profile_id: UUID) -> str | None:
        return self._session.model_profile_capabilities.get(profile_id)


def _fake_metadata_service(
    session: _FakeMetadataSession,
    *,
    clock: Callable[[], datetime] = lambda: datetime(2026, 7, 25, 8, 0, tzinfo=UTC),
) -> KnowledgeBaseService:
    def repositories(candidate: object) -> KnowledgeBaseRepositories:
        assert candidate is session
        return KnowledgeBaseRepositories(
            knowledge_bases=_FakeKnowledgeBases(session),
            actors=_FakeMetadataActors(session),
            idempotency=_FakeMetadataIdempotency(session),
            audits=_FakeMetadataAudits(session),
            mutations=_FakeMetadataMutations(session),
            model_profiles=_FakeModelProfiles(session),
            jobs=_FakeMetadataJobs(session),
        )

    return KnowledgeBaseService(
        session=cast(Any, session),
        settings=Settings(
            _env_file=None,
            environment="test",
            admin_key_hmac_secret=SecretStr("a" * 32),
            agent_key_hmac_secret=SecretStr("b" * 32),
        ),
        repository_factory=cast(Any, repositories),
        clock=clock,
    )


def _fake_manage_actor(session: _FakeMetadataSession) -> AgentPrincipal:
    key_id = uuid4()
    row = ApiKey(
        id=key_id,
        public_id="abcdefghijklmnopqrstuv",
        secret_digest=b"x" * 32,
        key_type="agent",
        name="fake-manager",
        status="active",
        capabilities=["manage"],
        raw_file_read=False,
        requests_per_minute=60,
        max_concurrency=4,
        resource_revision=1,
        created_at=datetime(2026, 7, 25, 8, 0, tzinfo=UTC),
        updated_at=datetime(2026, 7, 25, 8, 0, tzinfo=UTC),
    )
    session.agents[key_id] = row
    return AgentPrincipal(
        key_id=key_id,
        public_id=row.public_id,
        capabilities=frozenset({Capability.MANAGE}),
        knowledge_base_ids=frozenset(),
        query_profile_ids=frozenset(),
        default_query_profile_id=None,
        raw_file_read=False,
        requests_per_minute=60,
        max_concurrency=4,
    )


@pytest.mark.asyncio
async def test_fake_create_uses_one_outer_transaction_and_one_savepoint() -> None:
    session = _FakeMetadataSession()
    actor = _fake_manage_actor(session)

    result = await _fake_metadata_service(session).create_knowledge_base(
        KnowledgeBaseCreate(name="Fake transactional KB"),
        actor=actor,
        request_id="req-fake-create",
        idempotency_key="fake-create",
    )

    assert result.created is True
    assert (session.begin_count, session.nested_begin_count) == (1, 1)
    assert session.agents[actor.key_id].resource_revision == 2
    assert session.scopes == {(actor.key_id, result.knowledge_base.id)}
    assert len(session.idempotency) == 1
    assert [event.action for event in session.audit_events] == ["knowledge_base.created"]


@pytest.mark.asyncio
async def test_fake_lifecycle_covers_scoped_pagination_etags_and_idempotent_delete() -> None:
    session = _FakeMetadataSession()
    actor = _fake_manage_actor(session)
    service = _fake_metadata_service(session)
    command = KnowledgeBaseCreate(
        name="Lifecycle KB",
        description="before",
        metadata={"phase": 1},
    )
    first = await service.create_knowledge_base(
        command,
        actor=actor,
        request_id="req-fake-lifecycle-first",
        idempotency_key="fake-lifecycle-first",
    )
    second = await service.create_knowledge_base(
        KnowledgeBaseCreate(name="Second KB"),
        actor=actor,
        request_id="req-fake-lifecycle-second",
        idempotency_key="fake-lifecycle-second",
    )

    first_page = await service.list_knowledge_bases(actor=actor, limit=1)
    assert len(first_page.items) == 1
    assert first_page.next_cursor is not None
    second_page = await service.list_knowledge_bases(
        actor=actor,
        cursor=first_page.next_cursor,
        limit=1,
    )
    assert len(second_page.items) == 1
    assert second_page.next_cursor is None
    expected_ids = sorted(
        (first.knowledge_base.id, second.knowledge_base.id),
        key=lambda identifier: (
            session.knowledge_bases[identifier].created_at,
            identifier,
        ),
    )
    assert [item.id for item in (*first_page.items, *second_page.items)] == expected_ids
    assert (
        await service.get_knowledge_base(first.knowledge_base.id, actor=actor)
    ) == first.knowledge_base

    with pytest.raises(BusinessError) as missing_precondition:
        await service.update_knowledge_base(
            first.knowledge_base.id,
            KnowledgeBasePatch(name="not-applied"),
            actor=actor,
            request_id="req-fake-update-missing-etag",
            expected_etag=None,
        )
    assert (missing_precondition.value.status_code, missing_precondition.value.code) == (
        412,
        "PRECONDITION_FAILED",
    )

    updated = await service.update_knowledge_base(
        first.knowledge_base.id,
        KnowledgeBasePatch(
            name="Lifecycle KB disabled",
            description=None,
            metadata={"phase": 2},
            status="disabled",
        ),
        actor=actor,
        request_id="req-fake-update",
        expected_etag=first.knowledge_base.etag,
    )
    assert (
        updated.name,
        updated.description,
        updated.metadata,
        updated.status,
        updated.resource_revision,
    ) == ("Lifecycle KB disabled", None, {"phase": 2}, "disabled", 2)

    with pytest.raises(BusinessError) as stale_update:
        await service.update_knowledge_base(
            first.knowledge_base.id,
            KnowledgeBasePatch(name="stale"),
            actor=actor,
            request_id="req-fake-update-stale",
            expected_etag=first.knowledge_base.etag,
        )
    assert (stale_update.value.status_code, stale_update.value.code) == (
        412,
        "PRECONDITION_FAILED",
    )

    deleted = await service.delete_knowledge_base(
        first.knowledge_base.id,
        actor=actor,
        request_id="req-fake-delete",
        expected_etag=updated.etag,
    )
    assert (deleted.status, deleted.resource_revision) == ("deleting", 3)
    repeated = await service.delete_knowledge_base(
        first.knowledge_base.id,
        actor=actor,
        request_id="req-fake-delete-repeat",
        expected_etag=deleted.etag,
    )
    assert repeated == deleted

    with pytest.raises(BusinessError) as stale_delete:
        await service.delete_knowledge_base(
            first.knowledge_base.id,
            actor=actor,
            request_id="req-fake-delete-stale",
            expected_etag=updated.etag,
        )
    assert (stale_delete.value.status_code, stale_delete.value.code) == (
        412,
        "PRECONDITION_FAILED",
    )
    with pytest.raises(BusinessError) as deleting_update:
        await service.update_knowledge_base(
            first.knowledge_base.id,
            KnowledgeBasePatch(name="cannot-update"),
            actor=actor,
            request_id="req-fake-update-deleting",
            expected_etag=deleted.etag,
        )
    assert (deleting_update.value.status_code, deleting_update.value.code) == (
        409,
        "RESOURCE_STATE_CONFLICT",
    )

    visible = await service.list_knowledge_bases(actor=actor)
    assert [item.id for item in visible.items] == [second.knowledge_base.id]
    replay = await service.create_knowledge_base(
        command,
        actor=actor,
        request_id="req-fake-replay",
        idempotency_key="fake-lifecycle-first",
    )
    assert replay.created is False
    assert replay.knowledge_base == deleted
    with pytest.raises(BusinessError) as conflict:
        await service.create_knowledge_base(
            KnowledgeBaseCreate(name="Different request"),
            actor=actor,
            request_id="req-fake-conflict",
            idempotency_key="fake-lifecycle-first",
        )
    assert (conflict.value.status_code, conflict.value.code) == (
        409,
        "IDEMPOTENCY_CONFLICT",
    )

    session.scopes.remove((actor.key_id, first.knowledge_base.id))
    with pytest.raises(BusinessError) as removed_scope:
        await service.create_knowledge_base(
            command,
            actor=actor,
            request_id="req-fake-removed-scope",
            idempotency_key="fake-lifecycle-first",
        )
    assert (removed_scope.value.status_code, removed_scope.value.code) == (
        404,
        "RESOURCE_NOT_FOUND",
    )
    assert [event.action for event in session.audit_events] == [
        "knowledge_base.created",
        "knowledge_base.created",
        "knowledge_base.updated",
        "knowledge_base.deletion_requested",
    ]


@pytest.mark.asyncio
async def test_fake_service_revalidates_actor_and_rejects_invalid_inputs() -> None:
    for capabilities, status, expected in (
        (["manage"], "disabled", (401, "INVALID_API_KEY")),
        ([], "active", (403, "INSUFFICIENT_CAPABILITY")),
        (["manage", "manage"], "active", (500, "INTERNAL_ERROR")),
    ):
        session = _FakeMetadataSession()
        actor = _fake_manage_actor(session)
        session.agents[actor.key_id].capabilities = capabilities
        session.agents[actor.key_id].status = status
        with pytest.raises(BusinessError) as invalid_actor:
            await _fake_metadata_service(session).create_knowledge_base(
                KnowledgeBaseCreate(name="Rejected actor"),
                actor=actor,
                request_id="req-fake-invalid-actor",
                idempotency_key="fake-invalid-actor",
            )
        assert (invalid_actor.value.status_code, invalid_actor.value.code) == expected
        assert session.knowledge_bases == {}

    session = _FakeMetadataSession()
    actor = _fake_manage_actor(session)
    service = _fake_metadata_service(session)
    invalid_calls = (
        service.create_knowledge_base(
            cast(Any, object()),
            actor=actor,
            request_id="req-invalid-command",
            idempotency_key="invalid-command",
        ),
        service.create_knowledge_base(
            KnowledgeBaseCreate(name="Invalid request ID"),
            actor=actor,
            request_id="contains spaces",
            idempotency_key="invalid-request-id",
        ),
        service.create_knowledge_base(
            KnowledgeBaseCreate(name="Invalid idempotency"),
            actor=actor,
            request_id="req-invalid-idempotency",
            idempotency_key="contains spaces",
        ),
        service.list_knowledge_bases(actor=actor, limit=0),
        service.list_knowledge_bases(actor=actor, cursor="invalid cursor"),
        service.get_knowledge_base(cast(Any, "not-a-uuid"), actor=actor),
    )
    for call in invalid_calls:
        with pytest.raises(BusinessError) as invalid:
            await call
        assert (invalid.value.status_code, invalid.value.code) == (
            422,
            "VALIDATION_ERROR",
        )

    missing_id = uuid4()
    with pytest.raises(BusinessError) as missing_update:
        await service.update_knowledge_base(
            missing_id,
            KnowledgeBasePatch(name="missing"),
            actor=actor,
            request_id="req-missing-update",
            expected_etag='"kb:missing:r1"',
        )
    assert (missing_update.value.status_code, missing_update.value.code) == (
        404,
        "RESOURCE_NOT_FOUND",
    )
    with pytest.raises(BusinessError) as missing_delete:
        await service.delete_knowledge_base(
            missing_id,
            actor=actor,
            request_id="req-missing-delete",
            expected_etag='"kb:missing:r1"',
        )
    assert (missing_delete.value.status_code, missing_delete.value.code) == (
        404,
        "RESOURCE_NOT_FOUND",
    )


@pytest.mark.asyncio
async def test_fake_service_normalizes_repository_and_stored_data_failures() -> None:
    session = _FakeMetadataSession()
    actor = _fake_manage_actor(session)
    service = _fake_metadata_service(session)

    session.list_error = RuntimeError("unsafe list details")
    with pytest.raises(BusinessError) as list_failure:
        await service.list_knowledge_bases(actor=actor)
    assert (list_failure.value.status_code, list_failure.value.code) == (
        500,
        "INTERNAL_ERROR",
    )
    session.list_error = None

    session.get_error = RuntimeError("unsafe get details")
    with pytest.raises(BusinessError) as get_failure:
        await service.get_knowledge_base(uuid4(), actor=actor)
    assert (get_failure.value.status_code, get_failure.value.code) == (
        500,
        "INTERNAL_ERROR",
    )
    session.get_error = None

    created = await service.create_knowledge_base(
        KnowledgeBaseCreate(name="Failure normalization KB"),
        actor=actor,
        request_id="req-failure-normalization-create",
        idempotency_key="failure-normalization-create",
    )
    session.save_error = RuntimeError("unsafe save details")
    with pytest.raises(BusinessError) as update_failure:
        await service.update_knowledge_base(
            created.knowledge_base.id,
            KnowledgeBasePatch(name="must-roll-back"),
            actor=actor,
            request_id="req-failure-normalization-update",
            expected_etag=created.knowledge_base.etag,
        )
    assert (update_failure.value.status_code, update_failure.value.code) == (
        500,
        "INTERNAL_ERROR",
    )
    assert session.knowledge_bases[created.knowledge_base.id].name == created.knowledge_base.name

    with pytest.raises(BusinessError) as delete_failure:
        await service.delete_knowledge_base(
            created.knowledge_base.id,
            actor=actor,
            request_id="req-failure-normalization-delete",
            expected_etag=created.knowledge_base.etag,
        )
    assert (delete_failure.value.status_code, delete_failure.value.code) == (
        500,
        "INTERNAL_ERROR",
    )
    assert session.knowledge_bases[created.knowledge_base.id].status == "active"

    session.save_error = None
    session.knowledge_bases[created.knowledge_base.id].status = "invalid-stored-status"
    with pytest.raises(BusinessError) as stored_failure:
        await service.get_knowledge_base(created.knowledge_base.id, actor=actor)
    assert (stored_failure.value.status_code, stored_failure.value.code) == (
        500,
        "INTERNAL_ERROR",
    )


@pytest.mark.asyncio
async def test_fake_service_rejects_invalid_clock_and_replay_record() -> None:
    session = _FakeMetadataSession()
    actor = _fake_manage_actor(session)
    with pytest.raises(BusinessError) as invalid_clock:
        await _fake_metadata_service(
            session,
            clock=lambda: datetime(2026, 7, 25, 8, 0),
        ).create_knowledge_base(
            KnowledgeBaseCreate(name="Invalid clock"),
            actor=actor,
            request_id="req-invalid-clock",
            idempotency_key="invalid-clock",
        )
    assert (invalid_clock.value.status_code, invalid_clock.value.code) == (
        500,
        "INTERNAL_ERROR",
    )

    service = _fake_metadata_service(session)
    command = KnowledgeBaseCreate(name="Replay record KB")
    await service.create_knowledge_base(
        command,
        actor=actor,
        request_id="req-replay-record-create",
        idempotency_key="replay-record",
    )
    record = session.idempotency[(actor.key_id, "knowledge_base.create", "replay-record")]
    record.result_resource_type = "unexpected"
    with pytest.raises(BusinessError) as invalid_record:
        await service.create_knowledge_base(
            command,
            actor=actor,
            request_id="req-replay-record-invalid",
            idempotency_key="replay-record",
        )
    assert (invalid_record.value.status_code, invalid_record.value.code) == (
        500,
        "INTERNAL_ERROR",
    )


def test_constraint_name_requires_the_exact_idempotency_unique_constraint() -> None:
    target = IntegrityError(
        "insert",
        {},
        _FakeConstraintViolation("uq_idempotency_actor_operation_key"),
    )
    other = IntegrityError(
        "insert",
        {},
        _FakeConstraintViolation("uq_some_other_constraint"),
    )

    assert metadata_services._is_idempotency_unique_conflict(target) is True
    assert metadata_services._is_idempotency_unique_conflict(other) is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [RuntimeError("unsafe database payload"), asyncio.CancelledError()],
)
async def test_fake_create_exception_or_cancellation_rolls_back_every_write(
    failure: BaseException,
) -> None:
    session = _FakeMetadataSession()
    actor = _fake_manage_actor(session)
    session.audit_error = failure

    expected_error = BusinessError if isinstance(failure, Exception) else asyncio.CancelledError
    with pytest.raises(expected_error) as raised:
        await _fake_metadata_service(session).create_knowledge_base(
            KnowledgeBaseCreate(name="Must roll back"),
            actor=actor,
            request_id="req-fake-rollback",
            idempotency_key="fake-rollback",
        )

    if isinstance(raised.value, BusinessError):
        assert (raised.value.status_code, raised.value.code, raised.value.message) == (
            500,
            "INTERNAL_ERROR",
            "Internal server error",
        )
        assert "unsafe database payload" not in repr(raised.value)
    assert session.agents[actor.key_id].resource_revision == 1
    assert session.knowledge_bases == {}
    assert session.scopes == set()
    assert session.idempotency == {}
    assert session.audit_events == []


class _FakeConstraintDiagnostic:
    def __init__(self, constraint_name: str) -> None:
        self.constraint_name = constraint_name


class _FakeConstraintViolation(Exception):
    def __init__(self, constraint_name: str) -> None:
        super().__init__("unsafe constraint details")
        self.diag = _FakeConstraintDiagnostic(constraint_name)


class _FakeDirectConstraintViolation(Exception):
    def __init__(self, constraint_name: str) -> None:
        super().__init__("unsafe direct constraint details")
        self.constraint_name = constraint_name


def _captured_secret_exception(secret: str) -> tuple[RuntimeError, TracebackType]:
    retained_secret = secret
    try:
        raise RuntimeError(retained_secret)
    except RuntimeError as error:
        assert error.__traceback__ is not None
        return error, error.__traceback__


def _reachable_exception_nodes(error: BaseException) -> list[BaseException]:
    pending = [error]
    visited: set[int] = set()
    nodes: list[BaseException] = []
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        nodes.append(current)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        for nested in (current.__cause__, current.__context__):
            if nested is not None:
                pending.append(nested)
    return nodes


@pytest.mark.asyncio
async def test_exception_group_children_and_public_context_drop_all_secret_references() -> None:
    secret = "exception-group-child-secret-sentinel"
    child, child_traceback = _captured_secret_exception(secret)
    cause, cause_traceback = _captured_secret_exception(f"{secret}-cause")
    nested_child, nested_traceback = _captured_secret_exception(f"{secret}-nested")
    child.__cause__ = cause
    child.__context__ = cause
    group = ExceptionGroup(
        "safe outer group",
        [child, ExceptionGroup("safe nested group", [nested_child])],
    )
    session = _FakeMetadataSession()
    actor = _fake_manage_actor(session)
    session.audit_error = group

    with pytest.raises(BusinessError) as raised:
        await _fake_metadata_service(session).create_knowledge_base(
            KnowledgeBaseCreate(name="Exception group rollback"),
            actor=actor,
            request_id="req-exception-group-rollback",
            idempotency_key="exception-group-rollback",
        )

    public = raised.value
    assert (public.status_code, public.code, public.message) == (
        500,
        "INTERNAL_ERROR",
        "Internal server error",
    )
    assert public.__cause__ is None
    assert public.__context__ is None
    assert secret not in repr(public)

    nodes = [*_reachable_exception_nodes(group), cause]
    assert len(nodes) == 5
    for node in nodes:
        assert secret not in repr(node.args)
        assert node.__traceback__ is None
        assert node.__cause__ is None
        assert node.__context__ is None
    for retained_traceback in (child_traceback, cause_traceback, nested_traceback):
        current_traceback: TracebackType | None = retained_traceback
        while current_traceback is not None:
            string_locals = (
                value
                for value in current_traceback.tb_frame.f_locals.values()
                if isinstance(value, str)
            )
            assert all(secret not in value for value in string_locals)
            current_traceback = current_traceback.tb_next


def test_constraint_name_supports_direct_driver_attribute_and_exact_matching() -> None:
    target = IntegrityError(
        "insert",
        {},
        _FakeDirectConstraintViolation("uq_idempotency_actor_operation_key"),
    )
    other = IntegrityError(
        "insert",
        {},
        _FakeDirectConstraintViolation("uq_some_other_constraint"),
    )

    assert metadata_services._is_idempotency_unique_conflict(target) is True
    assert metadata_services._is_idempotency_unique_conflict(other) is False


@pytest.mark.asyncio
async def test_fake_create_does_not_treat_other_unique_constraints_as_idempotency() -> None:
    session = _FakeMetadataSession()
    actor = _fake_manage_actor(session)
    session.idempotency_error = IntegrityError(
        "unsafe insert",
        {"unsafe": "parameters"},
        _FakeConstraintViolation("uq_some_other_constraint"),
    )

    with pytest.raises(BusinessError) as raised:
        await _fake_metadata_service(session).create_knowledge_base(
            KnowledgeBaseCreate(name="Must not replay"),
            actor=actor,
            request_id="req-fake-other-unique",
            idempotency_key="fake-other-unique",
        )

    assert (raised.value.status_code, raised.value.code, raised.value.message) == (
        500,
        "INTERNAL_ERROR",
        "Internal server error",
    )
    assert session.idempotency_get_count == 1
    assert session.agents[actor.key_id].resource_revision == 1
    assert session.knowledge_bases == {}
    assert session.scopes == set()
    assert session.idempotency == {}
    assert session.audit_events == []


def test_ingestion_retrieval_models_are_exported_and_registered() -> None:
    expected_exports = {
        "ProviderCredential",
        "IndexGenerationCreationRequest",
        "DocumentUploadIdempotency",
    }
    assert expected_exports <= set(db_models.__all__)
    assert {
        "provider_credentials",
        "index_generation_creation_requests",
        "document_upload_idempotency",
    } <= set(Base.metadata.tables)


def test_ingestion_retrieval_schema_columns_defaults_and_types_are_authoritative() -> None:
    uuid = ("uuid", True)
    timestamptz = ("timestamp", True)
    expected_columns = {
        "provider_credentials": (
            ("id", uuid, False, True),
            ("name", ("varchar", 120), False, False),
            ("ciphertext", ("binary", None), False, False),
            ("nonce", ("binary", 12), False, False),
            ("algorithm", ("varchar", 32), False, False),
            ("key_version", ("varchar", 64), False, False),
            ("resource_revision", ("bigint",), False, False),
            ("created_at", timestamptz, False, False),
            ("updated_at", timestamptz, False, False),
            ("rotated_at", timestamptz, True, False),
        ),
        "index_generation_creation_requests": (
            ("id", uuid, False, True),
            ("actor_api_key_id", uuid, False, False),
            ("knowledge_base_id", uuid, False, False),
            ("idempotency_key", ("varchar", 128), False, False),
            ("request_fingerprint", ("binary", 32), False, False),
            ("generation_id", uuid, False, False),
            ("state", ("varchar", 16), False, False),
            ("final_http_status", ("integer",), True, False),
            ("safe_result", ("jsonb",), True, False),
            ("safe_error_code", ("varchar", 64), True, False),
            ("safe_error_message", ("varchar", 500), True, False),
            ("created_at", timestamptz, False, False),
            ("updated_at", timestamptz, False, False),
        ),
        "document_upload_idempotency": (
            ("id", uuid, False, True),
            ("actor_api_key_id", uuid, False, False),
            ("knowledge_base_id", uuid, False, False),
            ("idempotency_key", ("varchar", 128), False, False),
            ("request_fingerprint", ("binary", 32), False, False),
            ("document_id", uuid, False, False),
            ("document_version_id", uuid, False, False),
            ("job_id", uuid, False, False),
            ("result_status", ("varchar", 16), False, False),
            ("created_at", timestamptz, False, False),
            ("updated_at", timestamptz, False, False),
        ),
    }
    for table_name, columns in expected_columns.items():
        assert _column_contract(Base.metadata.tables[table_name]) == columns

    changed_columns: dict[
        str,
        dict[str, tuple[tuple[object, ...], bool]],
    ] = {
        "provider_configs": {
            "secret_ref": (("varchar", 255), True),
            "credential_id": (uuid, True),
            "resource_revision": (("bigint",), False),
            "endpoint_policy_version": (("varchar", 64), True),
            "endpoint_validated_at": (timestamptz, True),
        },
        "model_profiles": {
            "resource_revision": (("bigint",), False),
            "vector_config": (("jsonb",), False),
        },
        "knowledge_base_index_generations": {
            "distance": (("varchar", 16), True),
            "embedding_config_snapshot": (("jsonb",), True),
            "filter_schema_snapshot": (("jsonb",), True),
            "applied_filter_schema_revision": (("bigint",), True),
            "embedding_config_hash": (("char", 64), True),
            "safe_error_code": (("varchar", 64), True),
            "safe_error_message": (("varchar", 500), True),
        },
        "document_versions": {
            "chunk_manifest_object_key": (("varchar", 1024), True),
            "chunk_manifest_checksum_sha256": (("char", 64), True),
            "chunk_config_hash": (("char", 64), True),
        },
        "document_index_states": {
            "chunk_manifest_checksum_sha256": (("char", 64), True),
            "embedding_config_hash": (("char", 64), True),
            "next_chunk_index": (("bigint",), False),
            "safe_error_message": (("varchar", 500), True),
        },
        "jobs": {
            "actor_api_key_id": (uuid, True),
            "lease_owner": (("varchar", 255), True),
            "lease_epoch": (("bigint",), False),
            "lease_expires_at": (timestamptz, True),
            "retryable": (("boolean",), False),
            "resume_stage": (("varchar", 64), True),
        },
    }
    for table_name, contracts in changed_columns.items():
        actual = {
            name: (_type_signature(column.type), bool(column.nullable))
            for name, column in Base.metadata.tables[table_name].columns.items()
        }
        assert contracts.items() <= actual.items()

    assert _server_defaults(Base.metadata.tables["provider_credentials"]) == {
        "algorithm": "'AES-256-GCM'",
        "resource_revision": "1",
        "created_at": "now()",
        "updated_at": "now()",
    }
    assert (
        _server_defaults(Base.metadata.tables["model_profiles"])["vector_config"] == "'{}'::jsonb"
    )
    assert (
        _server_defaults(Base.metadata.tables["document_index_states"])["next_chunk_index"] == "0"
    )
    assert _server_defaults(Base.metadata.tables["jobs"])["retryable"] == "true"
    assert _server_defaults(Base.metadata.tables["jobs"])["lease_epoch"] == "0"
    assert (
        _server_defaults(Base.metadata.tables["index_generation_creation_requests"])["state"]
        == "'building'"
    )

    generation_request_table = Base.metadata.tables["index_generation_creation_requests"]
    generation_request_checks = _check_definitions(generation_request_table)
    assert generation_request_checks["ck_index_generation_requests_state"] == (
        "state in ('building', 'succeeded', 'failed')"
    )
    assert generation_request_checks["ck_index_generation_requests_terminal_http_status"] == (
        "(state = 'building' and final_http_status is null) or "
        "(state in ('succeeded', 'failed') and final_http_status is not null "
        "and final_http_status between 100 and 599)"
    )
    reconciliation_index = next(
        index
        for index in generation_request_table.indexes
        if index.name == "ix_index_generation_requests_reconciliation"
    )
    assert _index_contract(reconciliation_index) == (
        False,
        ("state", "updated_at"),
        "state = 'building'",
    )


def test_ingestion_retrieval_named_constraints_foreign_keys_uniques_and_indexes() -> None:
    required_checks = {
        "provider_credentials": {
            "ck_provider_credentials_name_length",
            "ck_provider_credentials_nonce_length",
            "ck_provider_credentials_algorithm",
            "ck_provider_credentials_key_version_length",
            "ck_provider_credentials_revision_positive",
        },
        "provider_configs": {
            "ck_provider_configs_credential_source_exactly_one",
            "ck_provider_configs_credential_endpoint_validation",
            "ck_provider_configs_endpoint_policy_version_length",
            "ck_provider_configs_revision_positive",
        },
        "model_profiles": {
            "ck_model_profiles_revision_positive",
            "ck_model_profiles_vector_config_object",
        },
        "knowledge_base_index_generations": {
            "ck_kb_index_generations_distance",
            "ck_kb_index_generations_embedding_config_hash",
            "ck_kb_index_generations_active_validation_complete",
            "ck_kb_index_generations_safe_error_code_length",
            "ck_kb_index_generations_safe_error_message_length",
        },
        "index_generation_creation_requests": {
            "ck_index_generation_requests_idempotency_key",
            "ck_index_generation_requests_fingerprint_length",
            "ck_index_generation_requests_state",
            "ck_index_generation_requests_terminal_http_status",
            "ck_index_generation_requests_safe_error_code_length",
            "ck_index_generation_requests_safe_error_message_length",
        },
        "document_upload_idempotency": {
            "ck_document_upload_idempotency_key",
            "ck_document_upload_fingerprint_length",
            "ck_document_upload_result_status",
        },
        "document_versions": {
            "ck_document_versions_chunk_manifest_object_key_length",
            "ck_document_versions_chunk_manifest_checksum",
            "ck_document_versions_chunk_config_hash",
            "ck_document_versions_chunk_manifest_complete",
        },
        "document_index_states": {
            "ck_document_index_states_manifest_checksum",
            "ck_document_index_states_embedding_config_hash",
            "ck_document_index_states_next_chunk_index_nonnegative",
            "ck_document_index_states_safe_error_message_length",
        },
        "jobs": {
            "ck_jobs_lease_epoch_nonnegative",
            "ck_jobs_lease_owner_length",
            "ck_jobs_resume_stage_length",
            "ck_jobs_lease_state_invariant",
        },
    }
    for table_name, names in required_checks.items():
        assert names <= set(_check_definitions(Base.metadata.tables[table_name]))

    assert _unique_constraints(Base.metadata.tables["provider_credentials"]) == {
        ("uq_provider_credentials_name", ("name",))
    }
    assert (
        "uq_index_generation_requests_actor_kb_key",
        ("actor_api_key_id", "knowledge_base_id", "idempotency_key"),
    ) in _unique_constraints(Base.metadata.tables["index_generation_creation_requests"])
    assert (
        "uq_document_upload_idempotency_actor_kb_key",
        ("actor_api_key_id", "knowledge_base_id", "idempotency_key"),
    ) in _unique_constraints(Base.metadata.tables["document_upload_idempotency"])
    for table_name, constraint_name in (
        (
            "index_generation_creation_requests",
            "uq_index_generation_requests_actor_kb_key",
        ),
        (
            "document_upload_idempotency",
            "uq_document_upload_idempotency_actor_kb_key",
        ),
    ):
        unique_constraint = next(
            constraint
            for constraint in Base.metadata.tables[table_name].constraints
            if isinstance(constraint, UniqueConstraint) and constraint.name == constraint_name
        )
        assert unique_constraint.deferrable is None
        assert unique_constraint.initially is None

    expected_fks = {
        "provider_configs": {"credential_id": ("provider_credentials.id", "RESTRICT")},
    }
    for table_name, expected in expected_fks.items():
        assert expected.items() <= _foreign_keys(Base.metadata.tables[table_name]).items()

    assert _foreign_key_constraint_contracts(
        Base.metadata.tables["index_generation_creation_requests"]
    ) == {
        (
            "fk_index_generation_requests_actor",
            ("actor_api_key_id",),
            ("api_keys.id",),
            "RESTRICT",
            None,
            None,
        ),
        (
            "fk_index_generation_requests_knowledge_base",
            ("knowledge_base_id",),
            ("knowledge_bases.id",),
            "RESTRICT",
            None,
            None,
        ),
        (
            "fk_index_generation_requests_generation_same_kb",
            ("generation_id", "knowledge_base_id"),
            (
                "knowledge_base_index_generations.id",
                "knowledge_base_index_generations.knowledge_base_id",
            ),
            "RESTRICT",
            True,
            "DEFERRED",
        ),
    }
    assert _foreign_key_constraint_contracts(
        Base.metadata.tables["document_upload_idempotency"]
    ) == {
        (
            "fk_document_upload_idempotency_actor",
            ("actor_api_key_id",),
            ("api_keys.id",),
            "RESTRICT",
            None,
            None,
        ),
        (
            "fk_document_upload_idempotency_knowledge_base",
            ("knowledge_base_id",),
            ("knowledge_bases.id",),
            "RESTRICT",
            None,
            None,
        ),
        (
            "fk_document_upload_idempotency_document_same_kb",
            ("document_id", "knowledge_base_id"),
            ("documents.id", "documents.knowledge_base_id"),
            "RESTRICT",
            True,
            "DEFERRED",
        ),
        (
            "fk_document_upload_idempotency_version_same_document",
            ("document_version_id", "document_id"),
            ("document_versions.id", "document_versions.document_id"),
            "RESTRICT",
            True,
            "DEFERRED",
        ),
        (
            "fk_document_upload_idempotency_job",
            ("job_id",),
            ("jobs.id",),
            "RESTRICT",
            True,
            "DEFERRED",
        ),
    }

    expected_indexes = {
        "provider_configs": {"ix_provider_configs_credential_id"},
        "index_generation_creation_requests": {
            "ix_index_generation_requests_generation_id",
            "ix_index_generation_requests_reconciliation",
        },
        "document_upload_idempotency": {
            "ix_document_upload_idempotency_document_id",
            "ix_document_upload_idempotency_reconciliation",
        },
        "jobs": {"ix_jobs_polling", "ix_jobs_expired_leases"},
    }
    for table_name, names in expected_indexes.items():
        assert names <= {index.name for index in Base.metadata.tables[table_name].indexes}


def test_generation_cleanup_claim_schema_is_persistent_and_fenced() -> None:
    table = Base.metadata.tables["index_generation_cleanup_claims"]
    uuid = ("uuid", True)
    timestamptz = ("timestamp", True)
    assert _column_contract(table) == (
        ("collection_name", ("varchar", 255), False, True),
        ("knowledge_base_id", uuid, False, False),
        ("generation_id", uuid, False, False),
        ("lease_owner", uuid, False, False),
        ("lease_epoch", ("bigint",), False, False),
        ("lease_expires_at", timestamptz, False, False),
        ("completed_at", timestamptz, True, False),
        ("created_at", timestamptz, False, False),
        ("updated_at", timestamptz, False, False),
    )
    assert _server_defaults(table) == {
        "lease_epoch": "1",
        "created_at": "now()",
        "updated_at": "now()",
    }
    assert _check_definitions(table) == {
        "ck_generation_cleanup_claims_collection_identity": (
            "collection_name = 'rag_kb_' || "
            "replace(knowledge_base_id::text, '-', '') || '_g_' || "
            "replace(generation_id::text, '-', '')"
        ),
        "ck_generation_cleanup_claims_collection_name_length": (
            "char_length(collection_name) between 1 and 255"
        ),
        "ck_generation_cleanup_claims_lease_epoch_positive": "lease_epoch >= 1",
    }
    assert _foreign_key_constraint_contracts(table) == set()
    expired_index = next(
        index for index in table.indexes if index.name == "ix_generation_cleanup_claims_expired"
    )
    assert _index_contract(expired_index) == (
        False,
        ("lease_expires_at", "collection_name"),
        "completed_at is null",
    )


def test_knowledge_base_patch_parses_a_rerank_profile_id_from_json() -> None:
    profile_id = uuid4()

    # The model is strict, so without an explicit validator a JSON string would
    # be rejected and the field would be unsettable over HTTP.
    assert (
        KnowledgeBasePatch.model_validate({"rerank_profile_id": str(profile_id)}).rerank_profile_id
        == profile_id
    )
    assert KnowledgeBasePatch.model_validate({"rerank_profile_id": None}).rerank_profile_id is None


@pytest.mark.parametrize(
    "value",
    ["not-a-uuid", "{2f9a1f4e-0a0e-4a34-9f5e-2f1a9f9c1a11}", 7],
)
def test_knowledge_base_patch_rejects_an_unusable_rerank_profile_id(value: object) -> None:
    # Non-canonical spellings are refused rather than normalised: the value is
    # compared and stored as given elsewhere in this API.
    with pytest.raises(ValidationError):
        KnowledgeBasePatch.model_validate({"rerank_profile_id": value})
