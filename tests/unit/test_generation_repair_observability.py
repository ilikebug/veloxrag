from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.indexing import repair as repair_module
from rag_service.indexing.generation_repositories import SqlAlchemyGenerationRepository
from rag_service.indexing.identities import collection_name
from rag_service.indexing.qdrant import CollectionSpec
from rag_service.indexing.repair import GenerationRepairService
from rag_service.observability.metrics import OperationalMetrics


class _CollectingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.mark.asyncio
async def test_generation_repair_observes_queued_only_after_transaction_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    knowledge_base_id = uuid4()
    generation_id = uuid4()
    snapshot = SimpleNamespace(
        knowledge_base_id=knowledge_base_id,
        generation_id=generation_id,
        anchor_mutation_id=uuid4(),
        mutation_revision=7,
        collection_spec=CollectionSpec(
            collection_name(knowledge_base_id, generation_id),
            3,
            "cosine",
            (),
        ),
        point_total=11,
    )
    committed = False
    added: list[object] = []
    scalar_rows: list[object] = [object(), object()]

    @asynccontextmanager
    async def transaction() -> AsyncIterator[None]:
        nonlocal committed
        yield
        committed = True

    class FakeSession:
        def begin(self) -> AbstractAsyncContextManager[None]:
            return transaction()

        async def scalar(self, _statement: object) -> object:
            return scalar_rows.pop(0)

        def add(self, value: object) -> None:
            added.append(value)

        async def flush(self) -> None:
            return None

    @asynccontextmanager
    async def sessions() -> AsyncIterator[AsyncSession]:
        yield cast(AsyncSession, FakeSession())

    class Qdrant:
        async def collection_exists(self, _name: str) -> bool:
            return True

    inner_metrics = OperationalMetrics()

    class CommitCheckingMetrics:
        def record_job_state(self, *, state: str) -> None:
            assert committed is True
            inner_metrics.record_job_state(state=state)

    service = GenerationRepairService(
        session_factory=sessions,
        qdrant=cast(Any, Qdrant()),
        metrics=cast(Any, CommitCheckingMetrics()),
    )

    async def read_snapshot(_session: AsyncSession, _generation_id: object) -> object:
        return snapshot

    async def no_active_repair(_session: AsyncSession, _generation_id: object) -> bool:
        return False

    async def no_active_ingestion(
        _session: AsyncSession,
        _knowledge_base_id: object,
        _generation_id: object,
    ) -> bool:
        return False

    async def return_snapshot(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return snapshot

    async def probe(_spec: CollectionSpec) -> None:
        return None

    async def acquire_fence(_repository: object, _collection_name: str) -> None:
        return None

    monkeypatch.setattr(service, "_read_snapshot", read_snapshot)
    monkeypatch.setattr(service, "_active_repair_exists", no_active_repair)
    monkeypatch.setattr(service, "_active_ingestion_exists", no_active_ingestion)
    monkeypatch.setattr(service, "_snapshot_from_rows", return_snapshot)
    monkeypatch.setattr(service, "_probe_collection", probe)
    monkeypatch.setattr(
        SqlAlchemyGenerationRepository,
        "acquire_collection_fence",
        acquire_fence,
    )

    handler = _CollectingHandler()
    previous_handlers = list(repair_module.logger.handlers)
    previous_propagate = repair_module.logger.propagate
    previous_level = repair_module.logger.level
    repair_module.logger.handlers = [handler]
    repair_module.logger.propagate = False
    repair_module.logger.setLevel(logging.INFO)
    try:
        reservation = await service.reserve(generation_id)
    finally:
        repair_module.logger.handlers = previous_handlers
        repair_module.logger.propagate = previous_propagate
        repair_module.logger.setLevel(previous_level)

    assert committed is True
    assert len(added) == 1
    assert (
        inner_metrics.registry.get_sample_value(
            "rag_job_state_transitions_total", {"state": "queued"}
        )
        == 1
    )
    queued_records = [record for record in handler.records if record.msg == "job.state.changed"]
    assert len(queued_records) == 1
    record = queued_records[0]
    assert record.__dict__["knowledge_base_id"] == str(knowledge_base_id)
    assert record.__dict__["generation_id"] == str(generation_id)
    assert record.__dict__["job_id"] == reservation["job_id"]
    assert record.__dict__["operation"] == "rebuild_generation"
    assert record.__dict__["state"] == "queued"
