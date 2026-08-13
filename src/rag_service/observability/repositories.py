"""Independent, content-free Provider attempt persistence."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.db.models.observability import ProviderUsage, QueryLog
from rag_service.providers.embeddings import EmbeddingAttempt

type SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]


@dataclass(frozen=True, slots=True, repr=False)
class ProviderUsageContext:
    """Application-owned identities; deliberately excludes content and credentials."""

    request_id: str
    actor_api_key_id: UUID | None
    provider_config_id: UUID
    model_profile_id: UUID

    def __post_init__(self) -> None:
        if (
            type(self.request_id) is not str
            or not 1 <= len(self.request_id) <= 128
            or any(ord(character) < 33 or ord(character) > 126 for character in self.request_id)
            or (self.actor_api_key_id is not None and type(self.actor_api_key_id) is not UUID)
            or type(self.provider_config_id) is not UUID
            or type(self.model_profile_id) is not UUID
        ):
            raise ValueError("Provider usage context is invalid")

    def __repr__(self) -> str:
        return f"ProviderUsageContext(request_id={self.request_id!r})"


@dataclass(frozen=True, slots=True, repr=False)
class QueryLogContext:
    """Content-free request identities safe for query-log persistence."""

    request_id: str
    actor_api_key_id: UUID | None
    knowledge_base_ids: tuple[UUID, ...]

    def __post_init__(self) -> None:
        if (
            type(self.request_id) is not str
            or not 1 <= len(self.request_id) <= 128
            or any(ord(character) < 33 or ord(character) > 126 for character in self.request_id)
            or (self.actor_api_key_id is not None and type(self.actor_api_key_id) is not UUID)
            or type(self.knowledge_base_ids) is not tuple
            or not 1 <= len(self.knowledge_base_ids) <= 64
            or any(type(value) is not UUID for value in self.knowledge_base_ids)
            or len(set(self.knowledge_base_ids)) != len(self.knowledge_base_ids)
        ):
            raise ValueError("Query log context is invalid")

    def __repr__(self) -> str:
        return f"QueryLogContext(request_id={self.request_id!r})"


class SqlAlchemyProviderUsageSink:
    """Persist each attempt in its own bounded transaction.

    Callers intentionally use this as a best-effort observer: database failure is
    fail-open for the Provider request and indexing batch, while cancellation is
    propagated so gateway and worker shutdown remain bounded.
    """

    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        timeout_seconds: float = 4.0,
    ) -> None:
        if (
            not callable(session_factory)
            or type(timeout_seconds) not in {int, float}
            or not math.isfinite(float(timeout_seconds))
            or not 0.01 <= float(timeout_seconds) <= 60.0
        ):
            raise ValueError("Provider usage sink configuration is invalid")
        self._session_factory = session_factory
        self._timeout_seconds = float(timeout_seconds)

    async def record(
        self,
        context: ProviderUsageContext,
        attempt: EmbeddingAttempt,
    ) -> None:
        if type(context) is not ProviderUsageContext or type(attempt) is not EmbeddingAttempt:
            raise ValueError("Provider usage record is invalid")
        async with asyncio.timeout(self._timeout_seconds):
            async with self._session_factory() as session, session.begin():
                session.add(
                    ProviderUsage(
                        request_id=context.request_id,
                        actor_api_key_id=context.actor_api_key_id,
                        provider_config_id=context.provider_config_id,
                        model_profile_id=context.model_profile_id,
                        capability="embedding",
                        provider_identifier=attempt.provider_identifier,
                        model_identifier=attempt.model_identifier,
                        route_identifier=attempt.route_identifier,
                        provider_request_id=attempt.provider_request_id,
                        input_tokens=attempt.input_tokens,
                        output_tokens=attempt.output_tokens,
                        cost_micros=attempt.cost_micros,
                        currency=attempt.currency,
                        latency_ms=attempt.latency_ms,
                        status=attempt.status,
                        error_code=attempt.error_code,
                        degraded=attempt.degraded,
                    )
                )


class SqlAlchemyQueryLogSink:
    """Persist one content-free query outcome in an independent bounded transaction."""

    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        timeout_seconds: float = 4.0,
    ) -> None:
        if (
            not callable(session_factory)
            or type(timeout_seconds) not in {int, float}
            or not math.isfinite(float(timeout_seconds))
            or not 0.01 <= float(timeout_seconds) <= 60.0
        ):
            raise ValueError("Query log sink configuration is invalid")
        self._session_factory = session_factory
        self._timeout_seconds = float(timeout_seconds)

    async def record(
        self,
        context: QueryLogContext,
        *,
        status: str,
        latency_ms: int,
        degraded: bool,
    ) -> None:
        if (
            type(context) is not QueryLogContext
            or status not in {"succeeded", "failed", "rejected"}
            or type(latency_ms) is not int
            or latency_ms < 0
            or type(degraded) is not bool
        ):
            raise ValueError("Query log record is invalid")
        async with asyncio.timeout(self._timeout_seconds):
            async with self._session_factory() as session, session.begin():
                session.add(
                    QueryLog(
                        request_id=context.request_id,
                        actor_api_key_id=context.actor_api_key_id,
                        knowledge_base_ids=list(context.knowledge_base_ids),
                        query_profile_id=None,
                        latency_ms=latency_ms,
                        status=status,
                        degraded=degraded,
                    )
                )


__all__ = [
    "ProviderUsageContext",
    "QueryLogContext",
    "SqlAlchemyProviderUsageSink",
    "SqlAlchemyQueryLogSink",
]
