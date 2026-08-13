import asyncio
from datetime import UTC, datetime
from types import TracebackType
from typing import Any, cast
from uuid import UUID

import pytest
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.api.errors import BusinessError
from rag_service.auth.policies import AgentPrincipal, Capability
from rag_service.config import Settings
from rag_service.metadata.document_repositories import (
    DocumentMetadataRepository,
    DocumentRecord,
    DocumentVersionRecord,
)
from rag_service.metadata.services import DocumentMetadataService

KB_ID = UUID(int=1)
DOCUMENT_ID = UUID(int=2)
VERSION_ID = UUID(int=3)
CREATED_AT = datetime(2026, 7, 24, 8, 0, tzinfo=UTC)


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        database_url=SecretStr("postgresql+psycopg://test:test@localhost/test"),
        admin_key_hmac_secret=SecretStr("document-unit-admin-hmac-secret"),
        agent_key_hmac_secret=SecretStr("document-unit-agent-hmac-secret"),
        default_page_size=2,
        max_page_size=3,
    )


def _actor(capability: Capability) -> AgentPrincipal:
    return AgentPrincipal(
        key_id=UUID(int=10),
        public_id="AAAAAAAAAAAAAAAA",
        capabilities=frozenset({capability}),
        knowledge_base_ids=frozenset({KB_ID}),
        query_profile_ids=frozenset(),
        default_query_profile_id=None,
        raw_file_read=False,
        requests_per_minute=60,
        max_concurrency=4,
    )


def _document(metadata: dict[str, Any]) -> DocumentRecord:
    return DocumentRecord(
        id=DOCUMENT_ID,
        knowledge_base_id=KB_ID,
        display_name="Document",
        mime_type="application/pdf",
        checksum_sha256="a" * 64,
        current_version_id=VERSION_ID,
        pending_version_id=None,
        status="active",
        tags=["manual"],
        metadata=metadata,
        created_at=CREATED_AT,
        updated_at=CREATED_AT,
    )


def _version(
    *,
    parser_config: dict[str, Any] | None = None,
    chunker_config: dict[str, Any] | None = None,
) -> DocumentVersionRecord:
    return DocumentVersionRecord(
        id=VERSION_ID,
        document_id=DOCUMENT_ID,
        version_number=1,
        source_checksum_sha256="b" * 64,
        parsed_object_checksum_sha256="c" * 64,
        declared_mime_type="application/pdf",
        detected_mime_type="application/pdf",
        source_extension="pdf",
        base_version_id=None,
        parser_name="parser",
        parser_version="1",
        parser_config={} if parser_config is None else parser_config,
        chunker_name="chunker",
        chunker_version="1",
        chunker_config={} if chunker_config is None else chunker_config,
        chunk_count=1,
        status="ready",
        activated_at=CREATED_AT,
        created_at=CREATED_AT,
    )


class _FakeDocumentRepository:
    def __init__(
        self,
        *,
        document: DocumentRecord | None = None,
        versions: list[DocumentVersionRecord] | None = None,
        failure: BaseException | None = None,
        failure_marker: str | None = None,
    ) -> None:
        self.document = document
        self.versions = [] if versions is None else versions
        self.failure = failure
        self.failure_marker = failure_marker

    async def get_scoped_parent(
        self,
        _actor_key_id: UUID,
        _knowledge_base_id: UUID,
    ) -> UUID | None:
        retained_repository_marker = self.failure_marker
        if self.failure is not None:
            assert retained_repository_marker is not None
            raise self.failure
        return KB_ID

    async def list_documents(
        self,
        _actor_key_id: UUID,
        _knowledge_base_id: UUID,
        _position: object,
        _limit: int,
    ) -> list[DocumentRecord]:
        return [] if self.document is None else [self.document]

    async def get_document(
        self,
        _actor_key_id: UUID,
        _document_id: UUID,
    ) -> DocumentRecord | None:
        return self.document

    async def get_document_parent(
        self,
        _actor_key_id: UUID,
        _document_id: UUID,
    ) -> UUID | None:
        return None if self.document is None else self.document.knowledge_base_id

    async def list_versions(self, *args: object) -> list[DocumentVersionRecord]:
        assert len(args) in {3, 4}
        return self.versions


def _service(repository: _FakeDocumentRepository) -> DocumentMetadataService:
    return DocumentMetadataService(
        session=cast(AsyncSession, object()),
        settings=_settings(),
        repository_factory=lambda _session: cast(DocumentMetadataRepository, repository),
    )


def _exception_nodes(error: BaseException) -> list[BaseException]:
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


def _assert_marker_absent_from_retained_service_state(
    error: BaseException,
    marker: str,
    *,
    repository_frame_names: frozenset[str] = frozenset(),
) -> None:
    for node in _exception_nodes(error):
        assert marker not in repr(node.args)
        traceback: TracebackType | None = node.__traceback__
        while traceback is not None:
            frame = traceback.tb_frame
            filename = frame.f_code.co_filename
            if "/rag_service/metadata/services.py" in filename or (
                filename.endswith("test_document_metadata_service.py")
                and frame.f_code.co_name in repository_frame_names
            ):
                assert marker not in repr(frame.f_locals)
            traceback = traceback.tb_next
        assert node.__cause__ is None
        assert node.__context__ is None


def _nested_marker(marker: str) -> dict[str, Any]:
    return {"one": {"two": {"three": {"four": {"secret": marker}}}}}


def _captured_exception(
    error: BaseException,
    marker: str,
) -> BaseException:
    retained_inner_marker = marker
    try:
        assert retained_inner_marker
        raise error
    except BaseException as captured:
        return captured


@pytest.mark.asyncio
async def test_document_detail_redacts_corrupt_metadata_from_retained_exception_state() -> None:
    marker = "corrupt-document-metadata-secret-marker"
    repository = _FakeDocumentRepository(document=_document(_nested_marker(marker)))

    with pytest.raises(BusinessError) as raised:
        await _service(repository).get_document(DOCUMENT_ID, actor=_actor(Capability.MANAGE))

    assert (raised.value.status_code, raised.value.code, raised.value.message) == (
        500,
        "INTERNAL_ERROR",
        "Internal server error",
    )
    _assert_marker_absent_from_retained_service_state(raised.value, marker)


@pytest.mark.asyncio
@pytest.mark.parametrize("config_field", ["parser_config", "chunker_config"])
async def test_version_list_redacts_corrupt_config_from_retained_exception_state(
    config_field: str,
) -> None:
    marker = f"corrupt-version-{config_field}-secret-marker"
    version = _version(**{config_field: _nested_marker(marker)})
    repository = _FakeDocumentRepository(
        document=_document({}),
        versions=[version],
    )

    with pytest.raises(BusinessError) as raised:
        await _service(repository).list_versions(
            DOCUMENT_ID,
            actor=_actor(Capability.RETRIEVE),
            cursor=None,
            limit=2,
        )

    assert (raised.value.status_code, raised.value.code, raised.value.message) == (
        500,
        "INTERNAL_ERROR",
        "Internal server error",
    )
    _assert_marker_absent_from_retained_service_state(raised.value, marker)


@pytest.mark.asyncio
async def test_answer_only_error_drops_loaded_document_from_retained_exception_state() -> None:
    marker = "answer-only-loaded-document-secret-marker"
    repository = _FakeDocumentRepository(document=_document({"secret": marker}))

    with pytest.raises(BusinessError) as raised:
        await _service(repository).get_document(DOCUMENT_ID, actor=_actor(Capability.ANSWER))

    assert (raised.value.status_code, raised.value.code, raised.value.message) == (
        403,
        "INSUFFICIENT_CAPABILITY",
        "Insufficient capability",
    )
    _assert_marker_absent_from_retained_service_state(raised.value, marker)


@pytest.mark.asyncio
async def test_list_documents_preserves_cancellation_identity_args_and_control_flow() -> None:
    marker = "repository-cancellation-local-secret-marker"
    cancellation = asyncio.CancelledError("safe cancellation", 7)
    repository = _FakeDocumentRepository(
        failure=cancellation,
        failure_marker=marker,
    )

    with pytest.raises(asyncio.CancelledError) as raised:
        await _service(repository).list_documents(
            KB_ID,
            actor=_actor(Capability.MANAGE),
            cursor=None,
            limit=2,
        )

    assert raised.value is cancellation
    assert raised.value.args == ("safe cancellation", 7)
    _assert_marker_absent_from_retained_service_state(
        raised.value,
        marker,
        repository_frame_names=frozenset({"get_scoped_parent"}),
    )


@pytest.mark.asyncio
async def test_list_documents_preserves_system_exit_and_scrubs_its_entire_traceback() -> None:
    marker = "direct-system-exit-retained-local-secret-marker"
    system_exit = cast(SystemExit, _captured_exception(SystemExit("safe exit", 23), marker))
    original_args = system_exit.args
    repository = _FakeDocumentRepository(
        failure=system_exit,
        failure_marker=marker,
    )

    with pytest.raises(SystemExit) as raised:
        await _service(repository).list_documents(
            KB_ID,
            actor=_actor(Capability.MANAGE),
            cursor=None,
            limit=2,
        )

    assert raised.value is system_exit
    assert raised.value.args == original_args
    _assert_marker_absent_from_retained_service_state(
        raised.value,
        marker,
        repository_frame_names=frozenset({"_captured_exception", "get_scoped_parent"}),
    )


@pytest.mark.asyncio
async def test_list_documents_preserves_base_exception_group_and_scrubs_every_node() -> None:
    marker = "base-exception-group-child-retained-local-secret-marker"
    child = cast(SystemExit, _captured_exception(SystemExit("safe child exit", 31), marker))
    cause = cast(RuntimeError, _captured_exception(RuntimeError("safe cause"), marker))
    nested_child = cast(
        asyncio.CancelledError,
        _captured_exception(asyncio.CancelledError("safe cancellation", 37), marker),
    )
    child.__cause__ = cause
    child.__context__ = cause
    nested = BaseExceptionGroup("safe nested group", [child, nested_child])
    group = BaseExceptionGroup("safe root group", [nested])
    retained_nodes = (group, nested, child, nested_child, cause)
    original_args = {id(node): node.args for node in retained_nodes}
    repository = _FakeDocumentRepository(
        failure=group,
        failure_marker=marker,
    )

    with pytest.raises(BaseExceptionGroup) as raised:
        await _service(repository).list_documents(
            KB_ID,
            actor=_actor(Capability.MANAGE),
            cursor=None,
            limit=2,
        )

    assert raised.value is group
    assert group.exceptions[0] is nested
    assert nested.exceptions == (child, nested_child)
    for node in retained_nodes:
        assert node.args == original_args[id(node)]
        _assert_marker_absent_from_retained_service_state(
            node,
            marker,
            repository_frame_names=frozenset({"_captured_exception", "get_scoped_parent"}),
        )
