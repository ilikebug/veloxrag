import builtins
import logging
import math
from collections.abc import ItemsView, Iterator, Mapping, Sequence
from typing import SupportsIndex, overload
from uuid import uuid4

import pytest
from prometheus_client import generate_latest


def _render_record(record: logging.LogRecord) -> str:
    return repr(
        (
            record.msg,
            record.args,
            record.__dict__,
            record.exc_info,
            record.exc_text,
            record.stack_info,
        )
    )


class _HostileTuple(tuple[object, ...]):
    def __len__(self) -> int:
        raise RuntimeError("hostile-tuple-secret")

    @overload
    def __getitem__(self, index: SupportsIndex) -> object: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[object, ...]: ...

    def __getitem__(self, index: SupportsIndex | slice) -> object | tuple[object, ...]:
        raise RuntimeError("hostile-tuple-secret")

    def __iter__(self) -> Iterator[object]:
        raise RuntimeError("hostile-tuple-secret")


class _HostileMapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise RuntimeError("hostile-mapping-secret")

    def __iter__(self) -> Iterator[str]:
        raise RuntimeError("hostile-mapping-secret")

    def __len__(self) -> int:
        raise RuntimeError("hostile-mapping-secret")

    def items(self) -> ItemsView[str, object]:
        raise RuntimeError("hostile-mapping-secret")


class _HostileSequence(Sequence[str]):
    @overload
    def __getitem__(self, index: int) -> str: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[str]: ...

    def __getitem__(self, index: int | slice) -> str | Sequence[str]:
        raise RuntimeError("hostile-sequence-secret")

    def __len__(self) -> int:
        raise RuntimeError("hostile-sequence-secret")


class _HostileRepr:
    def __repr__(self) -> str:
        raise RuntimeError("hostile-repr-secret")


class _HostileDictionary(dict[str, object]):
    def get(self, key: str, default: object = None) -> object:
        raise RuntimeError("hostile-dictionary-secret")

    def clear(self) -> None:
        raise RuntimeError("hostile-dictionary-secret")

    def update(self, *args: object, **kwargs: object) -> None:
        raise RuntimeError("hostile-dictionary-secret")


class _HostileLogRecord(logging.LogRecord):
    def getMessage(self) -> str:
        return "overridden-get-message-secret"


class _CollectingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class _BaseExceptionHandler(logging.Handler):
    def __init__(self, error_type: type[BaseException]) -> None:
        super().__init__()
        self._error_type = error_type

    def emit(self, record: logging.LogRecord) -> None:
        del record
        raise self._error_type("hostile-log-handler-secret")


def test_safe_log_context_exports_only_bounded_operational_ids() -> None:
    from rag_service.observability.logging import SafeLogContext

    identifiers = [uuid4() for _ in range(5)]
    context = SafeLogContext(
        request_id="req-observability-1",
        knowledge_base_id=identifiers[0],
        document_id=identifiers[1],
        version_id=identifiers[2],
        job_id=identifiers[3],
        generation_id=identifiers[4],
    )

    assert context.as_extra() == {
        "request_id": "req-observability-1",
        "knowledge_base_id": str(identifiers[0]),
        "document_id": str(identifiers[1]),
        "version_id": str(identifiers[2]),
        "job_id": str(identifiers[3]),
        "generation_id": str(identifiers[4]),
    }
    with pytest.raises(ValueError, match="log context is invalid"):
        SafeLogContext(request_id="x" * 129)


def test_emit_safe_log_delivers_a_fresh_exact_record_before_handler() -> None:
    from rag_service.observability.logging import SafeLogContext, emit_safe_log

    sentinel = "query-chunk-vector-provider-secret-ciphertext-nonce-object-key-raw-body"
    identifiers = [uuid4() for _ in range(5)]
    logger = logging.getLogger("rag_service.tests.safe-emitter")
    previous_level = logger.level
    previous_propagate = logger.propagate
    previous_handlers = list(logger.handlers)
    root_handlers = list(logging.getLogger().handlers)
    handler = _CollectingHandler()
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.ERROR)
    try:
        emit_safe_log(
            logger,
            logging.INFO,
            "upload.completed",
            context=SafeLogContext(request_id="req-disabled"),
            details={"raw_body": sentinel},
        )
        assert handler.records == []

        emit_safe_log(
            logger,
            logging.ERROR,
            "provider.request.completed",
            context=SafeLogContext(
                request_id="req-safe-emitter",
                knowledge_base_id=identifiers[0],
                document_id=identifiers[1],
                version_id=identifiers[2],
                job_id=identifiers[3],
                generation_id=identifiers[4],
            ),
            status="failed",
            error_code="PROVIDER_ERROR",
            details={
                "query": sentinel,
                "retrieved_text": sentinel,
                "chunk_text": sentinel,
                "vector": [0.1, 0.2],
                "provider_secret": sentinel,
                "ciphertext": sentinel,
                "nonce": sentinel,
                "Authorization": f"Bearer {sentinel}",
                "default_headers": {"Authorization": sentinel},
                "object_key": f"tmp/{sentinel}",
                "raw_body": sentinel,
                "exception": ExceptionGroup(sentinel, [RuntimeError(sentinel)]),
            },
        )
    finally:
        logger.handlers = previous_handlers
        logger.propagate = previous_propagate
        logger.setLevel(previous_level)

    assert logging.getLogger().handlers == root_handlers
    assert len(handler.records) == 1
    record = handler.records[0]
    assert type(record) is logging.LogRecord
    assert record.msg == "provider.request.completed"
    assert record.args == ()
    assert record.exc_info is None
    assert record.exc_text is None
    assert record.stack_info is None
    assert record.__dict__["request_id"] == "req-safe-emitter"
    assert record.__dict__["knowledge_base_id"] == str(identifiers[0])
    assert record.__dict__["document_id"] == str(identifiers[1])
    assert record.__dict__["version_id"] == str(identifiers[2])
    assert record.__dict__["job_id"] == str(identifiers[3])
    assert record.__dict__["generation_id"] == str(identifiers[4])
    assert sentinel not in _render_record(record)


@pytest.mark.parametrize("error_type", (KeyboardInterrupt, SystemExit, BaseException))
def test_emit_safe_log_fails_open_when_a_handler_raises_base_exception(
    error_type: type[BaseException],
) -> None:
    from rag_service.observability.logging import emit_safe_log

    logger = logging.getLogger(f"rag_service.tests.base-exception-{error_type.__name__}")
    previous_level = logger.level
    previous_propagate = logger.propagate
    previous_handlers = list(logger.handlers)
    logger.handlers = [_BaseExceptionHandler(error_type)]
    logger.propagate = False
    logger.setLevel(logging.INFO)
    raised: list[type[BaseException]] = []
    try:
        try:
            emit_safe_log(logger, logging.INFO, "safe.event", outcome="succeeded")
        except BaseException as error:
            raised.append(type(error))
    finally:
        logger.handlers = previous_handlers
        logger.propagate = previous_propagate
        logger.setLevel(previous_level)
    assert raised == []


def test_builtin_exception_allowlist_covers_every_unique_python_builtin_type_and_aliases() -> None:
    from rag_service.observability.logging import _BUILTIN_EXCEPTION_TYPES

    builtin_exception_types = {
        value
        for value in vars(builtins).values()
        if isinstance(value, type) and issubclass(value, BaseException)
    }

    mapped = dict(_BUILTIN_EXCEPTION_TYPES)
    assert len(mapped) == len(_BUILTIN_EXCEPTION_TYPES)
    assert set(mapped) == builtin_exception_types
    assert builtins.EnvironmentError is OSError
    assert builtins.IOError is OSError
    assert mapped[builtins.EnvironmentError] == "OSError"
    assert mapped[builtins.IOError] == "OSError"


def test_log_record_sanitizer_removes_recursive_content_and_exception_graphs() -> None:
    from rag_service.observability.logging import sanitize_log_record

    sentinel = "sensitive-query-chunk-vector-secret"
    try:
        raise ExceptionGroup(
            f"upstream {sentinel}",
            [
                RuntimeError(f"Authorization: Bearer {sentinel}"),
                ExceptionGroup("nested", [ValueError(f"nonce={sentinel}")]),
            ],
        )
    except ExceptionGroup as error:
        exc_info = (type(error), error, error.__traceback__)

    record = logging.LogRecord(
        "rag_service.test",
        logging.ERROR,
        __file__,
        1,
        f"unsafe message {sentinel} %s",
        (sentinel,),
        exc_info,
    )
    record.request_id = "req-safe"
    record.job_id = str(uuid4())
    record.details = {
        "status": "failed",
        "query_text": sentinel,
        "nested": {
            "chunk_text": sentinel,
            "vectors": [[0.1, 0.2]],
            "default_headers": {"Authorization": f"Bearer {sentinel}"},
            "object_key": f"tmp/jobs/{sentinel}",
            "raw_upstream_body": sentinel,
        },
    }

    sanitized = sanitize_log_record(record)
    rendered = _render_record(sanitized)

    assert type(sanitized) is logging.LogRecord
    assert sentinel not in rendered
    assert sanitized.msg == "redacted_log_event"
    assert sanitized.args == ()
    assert sanitized.exc_info is None
    assert sanitized.exc_text is None
    assert sanitized.stack_info is None
    assert sanitized.__dict__["request_id"] == "req-safe"
    assert sanitized.__dict__["job_id"] == record.__dict__["job_id"]
    assert sanitized.__dict__["exception_types"] == (
        "ExceptionGroup",
        "RuntimeError",
        "ExceptionGroup",
        "ValueError",
    )


def test_log_record_sanitizer_uses_semantic_allowlists_for_all_string_codes() -> None:
    from rag_service.observability.logging import sanitize_log_record

    record = logging.LogRecord(
        "private-query-123",
        logging.ERROR,
        "customersecret",
        1,
        "private-query-123",
        (),
        None,
        "APISECRET123",
    )
    record.event = "customersecret"
    record.error_code = "APISECRET123"
    record.exception_type = "CustomerSecret"
    record.exception_types = ("RuntimeError", "APISECRET123")

    sanitized = sanitize_log_record(record)

    assert sanitized.msg == "redacted_log_event"
    assert sanitized.__dict__["event"] == "redacted_log_event"
    assert sanitized.__dict__["error_code"] == "OTHER"
    assert sanitized.__dict__["exception_type"] == "Exception"
    assert sanitized.__dict__["exception_types"] == ("RuntimeError", "Exception")
    assert "private-query-123" not in _render_record(sanitized)
    assert "customersecret" not in _render_record(sanitized)
    assert "APISECRET123" not in _render_record(sanitized)


def test_log_record_sanitizer_replaces_all_dynamic_standard_string_fields() -> None:
    from rag_service.observability.logging import sanitize_log_record

    sentinel = "standard-field-secret"
    record = logging.LogRecord(
        sentinel,
        logging.WARNING,
        sentinel,
        17,
        "upload.completed",
        (),
        None,
        sentinel,
    )
    for field in (
        "filename",
        "module",
        "funcName",
        "threadName",
        "processName",
        "taskName",
    ):
        record.__dict__[field] = sentinel

    sanitized = sanitize_log_record(record)

    assert sentinel not in _render_record(sanitized)
    assert sanitized.name == "rag_service"
    assert sanitized.pathname == "redacted"
    assert sanitized.filename == "redacted"
    assert sanitized.module == "redacted"
    assert sanitized.funcName == "redacted"
    assert sanitized.threadName == "worker"
    assert sanitized.processName == "service"
    assert sanitized.__dict__["taskName"] == "task"


@pytest.mark.parametrize(
    ("field", "hostile_value"),
    [
        ("exc_info", _HostileTuple((RuntimeError, RuntimeError("secret"), None))),
        ("details", _HostileMapping()),
        ("exception_types", _HostileSequence()),
        ("details", {"status": "failed", "unknown": _HostileRepr()}),
    ],
)
def test_log_record_sanitizer_is_total_for_hostile_containers(
    field: str,
    hostile_value: object,
) -> None:
    from rag_service.observability.logging import sanitize_log_record

    record = logging.LogRecord(
        "provider.request.completed",
        logging.ERROR,
        __file__,
        1,
        "provider.request.completed",
        (_HostileRepr(),),
        None,
    )
    record.__dict__[field] = hostile_value

    sanitized = sanitize_log_record(record)
    assert type(sanitized) is logging.LogRecord
    assert sanitized.msg == "provider.request.completed"
    assert sanitized.args == ()
    assert "secret" not in _render_record(sanitized)


def test_log_record_sanitizer_replaces_hostile_record_and_dictionary_subclasses() -> None:
    from rag_service.observability.logging import SanitizingLogFilter, sanitize_log_record

    hostile_record = _HostileLogRecord(
        "secret-logger-name",
        logging.ERROR,
        "secret-pathname",
        1,
        "secret-message",
        (),
        None,
    )
    replacement = sanitize_log_record(hostile_record)

    assert type(replacement) is logging.LogRecord
    assert replacement is not hostile_record
    assert "secret" not in logging.Formatter("%(message)s %(name)s %(pathname)s").format(
        replacement
    )

    filtered = SanitizingLogFilter().filter(hostile_record)
    assert type(filtered) is logging.LogRecord
    assert "overridden-get-message-secret" not in logging.Formatter("%(message)s").format(filtered)

    plain_record = logging.LogRecord(
        "secret-logger-name",
        logging.ERROR,
        "secret-pathname",
        1,
        "secret-message",
        (),
        None,
    )
    hostile_dictionary = _HostileDictionary(plain_record.__dict__)
    object.__setattr__(plain_record, "__dict__", hostile_dictionary)

    dictionary_replacement = sanitize_log_record(plain_record)
    assert type(dictionary_replacement) is logging.LogRecord
    assert dictionary_replacement is not plain_record
    assert "secret" not in _render_record(dictionary_replacement)


def test_log_record_sanitizer_maps_custom_exception_type_and_bounds_large_groups() -> None:
    from rag_service.observability.logging import sanitize_log_record

    class CustomerSecret(Exception):
        pass

    custom = CustomerSecret("private custom message")
    custom_record = logging.LogRecord(
        "rag_service.test",
        logging.ERROR,
        __file__,
        1,
        "provider.request.completed",
        (),
        (CustomerSecret, custom, custom.__traceback__),
    )
    sanitized_custom = sanitize_log_record(custom_record)
    assert sanitized_custom.__dict__["exception_types"] == ("Exception",)

    group = ExceptionGroup("large private group", [ValueError(str(index)) for index in range(1000)])
    group_record = logging.LogRecord(
        "rag_service.test",
        logging.ERROR,
        __file__,
        1,
        "ingestion.stage.failed",
        (),
        (ExceptionGroup, group, group.__traceback__),
    )
    sanitized_group = sanitize_log_record(group_record)

    exception_types = sanitized_group.__dict__["exception_types"]
    assert type(exception_types) is tuple
    assert len(exception_types) == 32
    assert exception_types == ("ExceptionGroup", *("ValueError" for _ in range(31)))


@pytest.mark.parametrize(
    ("error", "expected_type"),
    [
        (KeyError("private"), "KeyError"),
        (IndexError("private"), "IndexError"),
        (ZeroDivisionError("private"), "ZeroDivisionError"),
        (OverflowError("private"), "OverflowError"),
    ],
)
def test_log_record_sanitizer_preserves_all_builtin_exception_types(
    error: BaseException,
    expected_type: str,
) -> None:
    from rag_service.observability.logging import sanitize_log_record

    record = logging.LogRecord(
        "rag_service.test",
        logging.ERROR,
        __file__,
        1,
        "provider.request.completed",
        (),
        (type(error), error, error.__traceback__),
    )

    sanitized = sanitize_log_record(record)

    assert sanitized.__dict__["exception_types"] == (expected_type,)
    assert "private" not in _render_record(sanitized)


def test_metrics_use_an_independent_registry_and_bounded_label_sets() -> None:
    from rag_service.observability.metrics import OperationalMetrics

    metrics = OperationalMetrics()
    metrics.record_upload(outcome="succeeded", byte_count=12, duration_seconds=0.01)
    metrics.record_job_state(state="running")
    metrics.record_lease_recovery()
    metrics.record_stage(
        stage="embed_index",
        outcome="succeeded",
        duration_seconds=0.02,
        character_count=42,
        chunk_count=3,
        batch_count=1,
    )
    metrics.record_stage(
        stage="embed_index",
        outcome="failed",
        failure_code="customer-sensitive-failure",
        duration_seconds=0.05,
    )
    metrics.record_provider_attempt(
        provider_type="openrouter",
        status="rate_limited",
        duration_seconds=0.03,
        input_tokens=10,
        output_tokens=0,
        cost_micros=0,
    )
    metrics.record_provider_retry(provider_type="openrouter")
    metrics.record_qdrant_upsert(outcome="succeeded", point_count=3)
    metrics.record_qdrant_search(
        outcome="succeeded",
        duration_seconds=0.04,
        candidate_count=5,
        result_count=3,
        visibility_drop_count=2,
    )

    rendered = generate_latest(metrics.registry).decode()

    for metric_name in (
        "rag_uploads_total",
        "rag_upload_bytes_total",
        "rag_upload_duration_seconds",
        "rag_job_state_transitions_total",
        "rag_job_lease_recoveries_total",
        "rag_ingestion_stage_duration_seconds",
        "rag_ingestion_stage_errors_total",
        "rag_ingestion_characters_total",
        "rag_ingestion_chunks_total",
        "rag_ingestion_batches_total",
        "rag_provider_requests_total",
        "rag_provider_retries_total",
        "rag_provider_rate_limits_total",
        "rag_provider_request_duration_seconds",
        "rag_provider_usage_tokens_total",
        "rag_provider_usage_cost_micros_total",
        "rag_qdrant_upserts_total",
        "rag_qdrant_upsert_points_total",
        "rag_qdrant_searches_total",
        "rag_qdrant_search_duration_seconds",
        "rag_qdrant_visibility_drops_total",
        "rag_qdrant_results_total",
    ):
        assert metric_name in rendered

    assert "request_id" not in rendered
    assert "knowledge_base_id" not in rendered
    assert "filename" not in rendered
    assert "model_name" not in rendered
    assert "error_message" not in rendered
    assert "customer-sensitive-failure" not in rendered
    assert 'failure_code="OTHER"' in rendered

    assert (
        metrics.registry.get_sample_value(
            "rag_uploads_total",
            {"outcome": "succeeded"},
        )
        == 1
    )
    assert (
        metrics.registry.get_sample_value(
            "rag_upload_bytes_total",
            {"outcome": "succeeded"},
        )
        == 12
    )
    assert (
        metrics.registry.get_sample_value(
            "rag_ingestion_stage_errors_total",
            {"stage": "embed_index", "failure_code": "OTHER"},
        )
        == 1
    )
    assert (
        metrics.registry.get_sample_value(
            "rag_provider_requests_total",
            {"provider_type": "openrouter", "status": "rate_limited"},
        )
        == 1
    )
    assert (
        metrics.registry.get_sample_value(
            "rag_provider_rate_limits_total",
            {"provider_type": "openrouter"},
        )
        == 1
    )
    assert metrics.registry.get_sample_value("rag_qdrant_visibility_drops_total") == 2
    assert metrics.registry.get_sample_value("rag_qdrant_results_total") == 3

    bounded_labels = {
        "outcome",
        "state",
        "stage",
        "failure_code",
        "provider_type",
        "status",
        "direction",
        "le",
    }
    for family in metrics.registry.collect():
        for sample in family.samples:
            assert set(sample.labels) <= bounded_labels


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("record_upload", {"outcome": "customer-file.txt", "byte_count": 1, "duration_seconds": 0}),
        ("record_job_state", {"state": "job-id-123"}),
        (
            "record_stage",
            {
                "stage": "dynamic-stage-from-user",
                "outcome": "failed",
                "duration_seconds": 0,
            },
        ),
        (
            "record_provider_attempt",
            {
                "provider_type": "secret-model-input",
                "status": "failed",
                "duration_seconds": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_micros": 0,
            },
        ),
        ("record_qdrant_upsert", {"outcome": "raw-error-message", "point_count": 0}),
    ],
)
def test_metrics_reject_unbounded_or_sensitive_label_values(
    method: str,
    kwargs: dict[str, object],
) -> None:
    from rag_service.observability.metrics import OperationalMetrics

    metrics = OperationalMetrics()

    with pytest.raises(ValueError, match="metric label is invalid"):
        getattr(metrics, method)(**kwargs)


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        (
            "record_upload",
            {"outcome": "succeeded", "byte_count": -1, "duration_seconds": 0},
        ),
        (
            "record_upload",
            {"outcome": "succeeded", "byte_count": True, "duration_seconds": 0},
        ),
        (
            "record_stage",
            {
                "stage": "parse",
                "outcome": "succeeded",
                "duration_seconds": math.inf,
            },
        ),
        (
            "record_provider_attempt",
            {
                "provider_type": "openrouter",
                "status": "failed",
                "duration_seconds": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_micros": -1,
            },
        ),
        (
            "record_qdrant_search",
            {
                "outcome": "succeeded",
                "duration_seconds": 0,
                "candidate_count": 1,
                "result_count": 1,
                "visibility_drop_count": 1,
            },
        ),
    ],
)
def test_metrics_reject_invalid_values_without_partial_mutation(
    method: str,
    kwargs: dict[str, object],
) -> None:
    from rag_service.observability.metrics import OperationalMetrics

    metrics = OperationalMetrics()
    before = generate_latest(metrics.registry)

    with pytest.raises(ValueError, match="metric value is invalid"):
        getattr(metrics, method)(**kwargs)

    assert generate_latest(metrics.registry) == before


def test_metrics_cover_the_complete_bounded_state_and_outcome_matrix() -> None:
    from rag_service.observability.metrics import OperationalMetrics

    metrics = OperationalMetrics()
    upload_outcomes = ("succeeded", "rejected", "failed", "cancelled")
    job_states = ("queued", "running", "retry_wait", "succeeded", "failed", "cancelled")
    stages_and_failure_codes = {
        "parse": "PARSE_FAILED",
        "chunk": "CHUNK_FAILED",
        "embed_index": "PROVIDER_TIMEOUT",
        "validate": "VALIDATION_FAILED",
        "activate": "ACTIVATION_FAILED",
    }
    provider_types = ("openai_compatible", "openrouter", "vendor_specific")
    provider_statuses = ("succeeded", "failed", "rate_limited", "timeout", "cancelled")
    qdrant_outcomes = ("succeeded", "failed", "cancelled")

    for index, outcome in enumerate(upload_outcomes, start=1):
        metrics.record_upload(outcome=outcome, byte_count=index, duration_seconds=index / 100)
    for state in job_states:
        metrics.record_job_state(state=state)
    for stage, failure_code in stages_and_failure_codes.items():
        metrics.record_stage(stage=stage, outcome="succeeded", duration_seconds=0.01)
        metrics.record_stage(
            stage=stage,
            outcome="failed",
            failure_code=failure_code,
            duration_seconds=0.02,
        )
        metrics.record_stage(stage=stage, outcome="cancelled", duration_seconds=0.03)
    for provider_type in provider_types:
        for status in provider_statuses:
            metrics.record_provider_attempt(
                provider_type=provider_type,
                status=status,
                duration_seconds=0.01,
                input_tokens=1,
                output_tokens=2,
                cost_micros=3,
            )
    for outcome in qdrant_outcomes:
        metrics.record_qdrant_upsert(outcome=outcome, point_count=1)
        metrics.record_qdrant_search(
            outcome=outcome,
            duration_seconds=0.01,
            candidate_count=2,
            result_count=1,
            visibility_drop_count=1,
        )

    def label_sets(sample_name: str) -> set[tuple[tuple[str, str], ...]]:
        return {
            tuple(sorted(sample.labels.items()))
            for family in metrics.registry.collect()
            for sample in family.samples
            if sample.name == sample_name
        }

    assert label_sets("rag_uploads_total") == {
        (("outcome", outcome),) for outcome in upload_outcomes
    }
    assert label_sets("rag_job_state_transitions_total") == {
        (("state", state),) for state in job_states
    }
    assert label_sets("rag_ingestion_stage_duration_seconds_count") == {
        (("outcome", outcome), ("stage", stage))
        for stage in stages_and_failure_codes
        for outcome in ("succeeded", "failed", "cancelled")
    }
    assert label_sets("rag_ingestion_stage_errors_total") == {
        (("failure_code", failure_code), ("stage", stage))
        for stage, failure_code in stages_and_failure_codes.items()
    }
    assert label_sets("rag_provider_requests_total") == {
        (("provider_type", provider_type), ("status", status))
        for provider_type in provider_types
        for status in provider_statuses
    }
    assert label_sets("rag_provider_request_duration_seconds_count") == label_sets(
        "rag_provider_requests_total"
    )
    assert label_sets("rag_qdrant_upserts_total") == {
        (("outcome", outcome),) for outcome in qdrant_outcomes
    }
    assert label_sets("rag_qdrant_searches_total") == {
        (("outcome", outcome),) for outcome in qdrant_outcomes
    }

    for index, outcome in enumerate(upload_outcomes, start=1):
        assert (
            metrics.registry.get_sample_value(
                "rag_upload_bytes_total",
                {"outcome": outcome},
            )
            == index
        )
    for stage, failure_code in stages_and_failure_codes.items():
        assert (
            metrics.registry.get_sample_value(
                "rag_ingestion_stage_errors_total",
                {"stage": stage, "failure_code": failure_code},
            )
            == 1
        )
