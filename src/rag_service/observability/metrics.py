"""Bounded Prometheus metrics for RAG operational events."""

from __future__ import annotations

import math
from typing import cast

from prometheus_client import CollectorRegistry, Counter, Histogram

_MAX_COUNTER_VALUE = 2**63 - 1
_UPLOAD_OUTCOMES = frozenset({"succeeded", "rejected", "failed", "cancelled"})
_JOB_STATES = frozenset({"queued", "running", "retry_wait", "succeeded", "failed", "cancelled"})
_STAGES = frozenset({"parse", "chunk", "embed_index", "validate", "activate"})
_STAGE_OUTCOMES = frozenset({"succeeded", "failed", "cancelled"})
_STAGE_FAILURE_CODES = frozenset(
    {
        "OTHER",
        "PARSE_FAILED",
        "CHUNK_FAILED",
        "PROVIDER_ERROR",
        "PROVIDER_RATE_LIMITED",
        "PROVIDER_TIMEOUT",
        "QDRANT_UPSERT_FAILED",
        "VALIDATION_FAILED",
        "ACTIVATION_FAILED",
        "LEASE_LOST",
    }
)
_PROVIDER_TYPES = frozenset({"openai_compatible", "openrouter", "vendor_specific"})
_PROVIDER_STATUSES = frozenset({"succeeded", "failed", "rate_limited", "timeout", "cancelled"})
_QDRANT_OUTCOMES = frozenset({"succeeded", "failed", "cancelled"})
_DURATION_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)


def _label(value: object, allowed: frozenset[str]) -> str:
    if type(value) is not str or value not in allowed:
        raise ValueError("metric label is invalid")
    return value


def _count(value: object) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_COUNTER_VALUE:
        raise ValueError("metric value is invalid")
    return value


def _duration(value: object) -> float:
    if type(value) not in {int, float}:
        raise ValueError("metric value is invalid")
    duration = float(cast(int | float, value))
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("metric value is invalid")
    return duration


class OperationalMetrics:
    """An isolated registry whose labels are finite by construction."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry(auto_describe=True)

        self._uploads = Counter(
            "rag_uploads_total",
            "Completed upload attempts.",
            ("outcome",),
            registry=self.registry,
        )
        self._upload_bytes = Counter(
            "rag_upload_bytes_total",
            "Bytes received by completed upload attempts.",
            ("outcome",),
            registry=self.registry,
        )
        self._upload_duration = Histogram(
            "rag_upload_duration_seconds",
            "Upload attempt latency.",
            ("outcome",),
            buckets=_DURATION_BUCKETS,
            registry=self.registry,
        )

        self._job_states = Counter(
            "rag_job_state_transitions_total",
            "Job state transitions observed by workers.",
            ("state",),
            registry=self.registry,
        )
        self._lease_recoveries = Counter(
            "rag_job_lease_recoveries_total",
            "Expired job leases reclaimed by workers.",
            registry=self.registry,
        )

        self._stage_duration = Histogram(
            "rag_ingestion_stage_duration_seconds",
            "Ingestion stage latency.",
            ("stage", "outcome"),
            buckets=_DURATION_BUCKETS,
            registry=self.registry,
        )
        self._stage_errors = Counter(
            "rag_ingestion_stage_errors_total",
            "Failed ingestion stages.",
            ("stage", "failure_code"),
            registry=self.registry,
        )
        self._stage_characters = Counter(
            "rag_ingestion_characters_total",
            "Characters processed by an ingestion stage.",
            ("stage",),
            registry=self.registry,
        )
        self._stage_chunks = Counter(
            "rag_ingestion_chunks_total",
            "Chunks processed by an ingestion stage.",
            ("stage",),
            registry=self.registry,
        )
        self._stage_batches = Counter(
            "rag_ingestion_batches_total",
            "Batches processed by an ingestion stage.",
            ("stage",),
            registry=self.registry,
        )

        self._provider_requests = Counter(
            "rag_provider_requests_total",
            "Provider network attempts.",
            ("provider_type", "status"),
            registry=self.registry,
        )
        self._provider_retries = Counter(
            "rag_provider_retries_total",
            "Provider retries scheduled by the application.",
            ("provider_type",),
            registry=self.registry,
        )
        self._provider_rate_limits = Counter(
            "rag_provider_rate_limits_total",
            "Rate-limited Provider attempts.",
            ("provider_type",),
            registry=self.registry,
        )
        self._provider_duration = Histogram(
            "rag_provider_request_duration_seconds",
            "Provider network attempt latency.",
            ("provider_type", "status"),
            buckets=_DURATION_BUCKETS,
            registry=self.registry,
        )
        self._provider_tokens = Counter(
            "rag_provider_usage_tokens_total",
            "Provider usage tokens.",
            ("provider_type", "direction"),
            registry=self.registry,
        )
        self._provider_cost = Counter(
            "rag_provider_usage_cost_micros_total",
            "Provider cost in micro-units.",
            ("provider_type",),
            registry=self.registry,
        )

        self._qdrant_upserts = Counter(
            "rag_qdrant_upserts_total",
            "Qdrant upsert attempts.",
            ("outcome",),
            registry=self.registry,
        )
        self._qdrant_upsert_points = Counter(
            "rag_qdrant_upsert_points_total",
            "Points submitted in Qdrant upserts.",
            ("outcome",),
            registry=self.registry,
        )
        self._qdrant_searches = Counter(
            "rag_qdrant_searches_total",
            "Qdrant search attempts.",
            ("outcome",),
            registry=self.registry,
        )
        self._qdrant_search_duration = Histogram(
            "rag_qdrant_search_duration_seconds",
            "Qdrant search latency.",
            ("outcome",),
            buckets=_DURATION_BUCKETS,
            registry=self.registry,
        )
        self._qdrant_visibility_drops = Counter(
            "rag_qdrant_visibility_drops_total",
            "Candidates rejected by the Postgres visibility filter.",
            registry=self.registry,
        )
        self._qdrant_results = Counter(
            "rag_qdrant_results_total",
            "Visible results returned after Qdrant search.",
            registry=self.registry,
        )

    def record_upload(
        self,
        *,
        outcome: str,
        byte_count: int,
        duration_seconds: float,
    ) -> None:
        safe_outcome = _label(outcome, _UPLOAD_OUTCOMES)
        safe_bytes = _count(byte_count)
        safe_duration = _duration(duration_seconds)
        self._uploads.labels(outcome=safe_outcome).inc()
        self._upload_bytes.labels(outcome=safe_outcome).inc(safe_bytes)
        self._upload_duration.labels(outcome=safe_outcome).observe(safe_duration)

    def record_job_state(self, *, state: str) -> None:
        self._job_states.labels(state=_label(state, _JOB_STATES)).inc()

    def record_lease_recovery(self) -> None:
        self._lease_recoveries.inc()

    def record_stage(
        self,
        *,
        stage: str,
        outcome: str,
        duration_seconds: float,
        failure_code: str | None = None,
        character_count: int = 0,
        chunk_count: int = 0,
        batch_count: int = 0,
    ) -> None:
        safe_stage = _label(stage, _STAGES)
        safe_outcome = _label(outcome, _STAGE_OUTCOMES)
        safe_duration = _duration(duration_seconds)
        safe_characters = _count(character_count)
        safe_chunks = _count(chunk_count)
        safe_batches = _count(batch_count)
        if safe_outcome == "failed":
            safe_failure_code = (
                failure_code
                if type(failure_code) is str and failure_code in _STAGE_FAILURE_CODES
                else "OTHER"
            )
        else:
            if failure_code is not None:
                raise ValueError("metric value is invalid")
            safe_failure_code = None
        self._stage_duration.labels(stage=safe_stage, outcome=safe_outcome).observe(safe_duration)
        if safe_failure_code is not None:
            self._stage_errors.labels(
                stage=safe_stage,
                failure_code=safe_failure_code,
            ).inc()
        self._stage_characters.labels(stage=safe_stage).inc(safe_characters)
        self._stage_chunks.labels(stage=safe_stage).inc(safe_chunks)
        self._stage_batches.labels(stage=safe_stage).inc(safe_batches)

    def record_provider_attempt(
        self,
        *,
        provider_type: str,
        status: str,
        duration_seconds: float,
        input_tokens: int,
        output_tokens: int,
        cost_micros: int,
    ) -> None:
        safe_provider_type = _label(provider_type, _PROVIDER_TYPES)
        safe_status = _label(status, _PROVIDER_STATUSES)
        safe_duration = _duration(duration_seconds)
        safe_input_tokens = _count(input_tokens)
        safe_output_tokens = _count(output_tokens)
        safe_cost = _count(cost_micros)
        labels = {"provider_type": safe_provider_type, "status": safe_status}
        self._provider_requests.labels(**labels).inc()
        self._provider_duration.labels(**labels).observe(safe_duration)
        if safe_status == "rate_limited":
            self._provider_rate_limits.labels(provider_type=safe_provider_type).inc()
        self._provider_tokens.labels(
            provider_type=safe_provider_type,
            direction="input",
        ).inc(safe_input_tokens)
        self._provider_tokens.labels(
            provider_type=safe_provider_type,
            direction="output",
        ).inc(safe_output_tokens)
        self._provider_cost.labels(provider_type=safe_provider_type).inc(safe_cost)

    def record_provider_retry(self, *, provider_type: str) -> None:
        self._provider_retries.labels(provider_type=_label(provider_type, _PROVIDER_TYPES)).inc()

    def record_qdrant_upsert(self, *, outcome: str, point_count: int) -> None:
        safe_outcome = _label(outcome, _QDRANT_OUTCOMES)
        safe_points = _count(point_count)
        self._qdrant_upserts.labels(outcome=safe_outcome).inc()
        self._qdrant_upsert_points.labels(outcome=safe_outcome).inc(safe_points)

    def record_qdrant_search(
        self,
        *,
        outcome: str,
        duration_seconds: float,
        candidate_count: int,
        result_count: int,
        visibility_drop_count: int,
    ) -> None:
        safe_outcome = _label(outcome, _QDRANT_OUTCOMES)
        safe_duration = _duration(duration_seconds)
        safe_candidates = _count(candidate_count)
        safe_results = _count(result_count)
        safe_drops = _count(visibility_drop_count)
        if safe_results + safe_drops > safe_candidates:
            raise ValueError("metric value is invalid")
        self._qdrant_searches.labels(outcome=safe_outcome).inc()
        self._qdrant_search_duration.labels(outcome=safe_outcome).observe(safe_duration)
        self._qdrant_visibility_drops.inc(safe_drops)
        self._qdrant_results.inc(safe_results)


METRICS = OperationalMetrics()

__all__ = ["METRICS", "OperationalMetrics"]
