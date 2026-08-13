"""Shared credential-safe cloud embedding gateway."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import time
from collections import deque
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from types import MappingProxyType
from typing import Final, Literal, Protocol
from uuid import UUID

from rag_service.observability.logging import emit_safe_log
from rag_service.observability.metrics import METRICS, OperationalMetrics
from rag_service.providers.credentials import (
    EncryptedProviderCredential,
    ProviderCredentialKeyring,
)
from rag_service.providers.gateway_failures import status_failure, transport_failure
from rag_service.providers.network_policy import (
    CanonicalProviderEndpoint,
    validate_provider_endpoint_url,
    validate_provider_headers,
)
from rag_service.providers.schemas import ProviderConfigCreate
from rag_service.providers.transport import ProviderHttpResponse

_ADAPTER_SCHEMA_VERSION: Final = "openai-embeddings-v1"
_SUPPORTED_PROVIDER_TYPES: Final = frozenset({"openai_compatible", "openrouter"})
_SUPPORTED_DISTANCES: Final = frozenset({"cosine", "dot", "euclid", "manhattan"})
_SAFE_USAGE_FIELDS: Final = frozenset({"prompt_tokens", "total_tokens", "completion_tokens"})
_MAX_RESPONSE_ITEMS: Final = 10_000
_MAX_EMBEDDING_DIMENSION: Final = 10_000_000
_MAX_PROBE_INPUT_BYTES: Final = 8 * 1024
_MAX_PROBE_INPUT_TOKENS: Final = 8 * 1024
_RATE_WINDOW_SECONDS: Final = 60.0
_DEFAULT_ATTEMPT_OBSERVER_TIMEOUT_SECONDS: Final = 5.0
_MAX_ATTEMPT_OBSERVER_TASKS: Final = 1024
_POSTGRES_BIGINT_MAX: Final = 2**63 - 1
_TELEMETRY_ERROR_CODE: Final = "PROVIDER_TELEMETRY_INVALID"

logger = logging.getLogger(__name__)


class ProviderCredentialReader(Protocol):
    async def get_encrypted(
        self,
        credential_id: UUID,
    ) -> EncryptedProviderCredential | None: ...


class ProviderJsonTransport(Protocol):
    async def post_json(
        self,
        *,
        endpoint: CanonicalProviderEndpoint,
        path: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> ProviderHttpResponse: ...

    async def aclose(self) -> None: ...


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        frozen = {
            key: _freeze_json(member) for key, member in sorted(value.items()) if type(key) is str
        }
        if len(frozen) != len(value):
            raise ValueError
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(member) for member in value)
    if value is None or type(value) in {str, bool, int, float}:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError
        return value
    raise ValueError


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(member) for key, member in sorted(value.items())}
    if isinstance(value, tuple):
        return [_thaw_json(member) for member in value]
    if value is None or type(value) in {str, bool, int, float}:
        return value
    raise ValueError


def _frozen_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    frozen = _freeze_json(deepcopy(dict(value)))
    if not isinstance(frozen, Mapping):
        raise ValueError
    return frozen


def _thawed_mapping(value: Mapping[str, object]) -> dict[str, object]:
    thawed = _thaw_json(value)
    if type(thawed) is not dict:
        raise ValueError
    return thawed


def _contains_control_character(value: str) -> bool:
    return any(ord(character) < 32 or 127 <= ord(character) <= 159 for character in value)


@dataclass(frozen=True, slots=True, repr=False)
class EmbeddingConfigSnapshot:
    """Immutable generation-owned fields that determine vector semantics."""

    adapter_schema_version: str
    provider_type: str
    base_url: str
    credential_id: UUID
    default_headers: Mapping[str, str]
    routing_options: Mapping[str, object]
    model_name: str
    dimension: int
    distance: str
    max_input_tokens: int
    vector_config: Mapping[str, object]

    def __post_init__(self) -> None:
        try:
            if self.adapter_schema_version != _ADAPTER_SCHEMA_VERSION:
                raise ValueError
            if self.provider_type not in _SUPPORTED_PROVIDER_TYPES:
                raise ValueError
            if type(self.credential_id) is not UUID:
                raise ValueError
            endpoint = validate_provider_endpoint_url(self.base_url)
            if (
                type(self.model_name) is not str
                or not self.model_name.strip()
                or len(self.model_name) > 255
                or any(ord(character) < 32 for character in self.model_name)
            ):
                raise ValueError
            if type(self.dimension) is not int or not 1 <= self.dimension <= 10_000_000:
                raise ValueError
            if self.distance not in _SUPPORTED_DISTANCES:
                raise ValueError
            if (
                type(self.max_input_tokens) is not int
                or not 1 <= self.max_input_tokens <= 10_000_000
            ):
                raise ValueError
            if not isinstance(self.default_headers, Mapping) or not isinstance(
                self.routing_options, Mapping
            ):
                raise ValueError
            if not isinstance(self.vector_config, Mapping) or self.vector_config:
                raise ValueError
            canonical_headers = validate_provider_headers(self.default_headers)
            validated = ProviderConfigCreate(
                name="embedding snapshot validation",
                provider_type=self.provider_type,
                base_url=endpoint.url,
                credential_id=self.credential_id,
                default_headers=dict(canonical_headers),
                routing_options=deepcopy(dict(self.routing_options)),
                timeout_seconds=Decimal("1"),
                max_concurrency=1,
                requests_per_minute=1,
                enabled=True,
            )
            object.__setattr__(self, "base_url", endpoint.url)
            object.__setattr__(self, "model_name", self.model_name.strip())
            object.__setattr__(
                self,
                "default_headers",
                MappingProxyType(dict(validated.default_headers)),
            )
            object.__setattr__(
                self,
                "routing_options",
                _frozen_mapping(validated.routing_options),
            )
            object.__setattr__(self, "vector_config", MappingProxyType({}))
        except Exception:
            raise ValueError("Embedding configuration snapshot is invalid") from None

    def __repr__(self) -> str:
        return "EmbeddingConfigSnapshot(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class EmbeddingDimensionProbeRequest:
    """Immutable provider request fields safe for dimension discovery."""

    adapter_schema_version: str
    provider_type: str
    base_url: str
    credential_id: UUID
    default_headers: Mapping[str, str]
    routing_options: Mapping[str, object]
    model_name: str

    def __post_init__(self) -> None:
        try:
            if (
                type(self.adapter_schema_version) is not str
                or self.adapter_schema_version != _ADAPTER_SCHEMA_VERSION
                or type(self.provider_type) is not str
                or self.provider_type not in _SUPPORTED_PROVIDER_TYPES
                or type(self.base_url) is not str
                or type(self.credential_id) is not UUID
                or not isinstance(self.default_headers, Mapping)
                or not isinstance(self.routing_options, Mapping)
                or type(self.model_name) is not str
            ):
                raise ValueError
            endpoint = validate_provider_endpoint_url(self.base_url)
            model_name = self.model_name.strip()
            if not 1 <= len(model_name) <= 255 or _contains_control_character(self.model_name):
                raise ValueError
            canonical_headers = validate_provider_headers(self.default_headers)
            validated = ProviderConfigCreate(
                name="embedding dimension probe validation",
                provider_type=self.provider_type,
                base_url=endpoint.url,
                credential_id=self.credential_id,
                default_headers=dict(canonical_headers),
                routing_options=deepcopy(dict(self.routing_options)),
                timeout_seconds=Decimal("1"),
                max_concurrency=1,
                requests_per_minute=1,
                enabled=True,
            )
            object.__setattr__(self, "base_url", endpoint.url)
            object.__setattr__(self, "model_name", model_name)
            object.__setattr__(
                self,
                "default_headers",
                MappingProxyType(dict(validated.default_headers)),
            )
            object.__setattr__(
                self,
                "routing_options",
                _frozen_mapping(validated.routing_options),
            )
        except Exception:
            raise ValueError("Embedding dimension probe request is invalid") from None

    def __repr__(self) -> str:
        return "EmbeddingDimensionProbeRequest(<redacted>)"


@dataclass(frozen=True, slots=True)
class EmbeddingOperationalConfig:
    """Current non-vector operational controls safe to change in place."""

    provider_config_id: UUID
    provider_enabled: bool
    profile_enabled: bool
    timeout_seconds: Decimal
    max_concurrency: int
    requests_per_minute: int
    batch_size: int

    def __post_init__(self) -> None:
        try:
            timeout = Decimal(self.timeout_seconds)
            if (
                type(self.provider_config_id) is not UUID
                or type(self.provider_enabled) is not bool
                or type(self.profile_enabled) is not bool
                or not timeout.is_finite()
                or not Decimal("0") < timeout <= Decimal("600")
                or type(self.max_concurrency) is not int
                or not 1 <= self.max_concurrency <= 10_000
                or type(self.requests_per_minute) is not int
                or not 1 <= self.requests_per_minute <= 1_000_000
                or type(self.batch_size) is not int
                or not 1 <= self.batch_size <= 10_000
            ):
                raise ValueError
            object.__setattr__(self, "timeout_seconds", timeout)
        except Exception:
            raise ValueError("Embedding operational configuration is invalid") from None


@dataclass(frozen=True, slots=True)
class EmbeddingProbeOperationalConfig:
    """Current provider controls used by a one-input dimension probe."""

    provider_config_id: UUID
    provider_enabled: bool
    timeout_seconds: Decimal
    max_concurrency: int
    requests_per_minute: int
    batch_size: Literal[1] = 1

    def __post_init__(self) -> None:
        try:
            if (
                type(self.provider_config_id) is not UUID
                or type(self.provider_enabled) is not bool
                or type(self.timeout_seconds) is not Decimal
                or not self.timeout_seconds.is_finite()
                or not Decimal("0") < self.timeout_seconds <= Decimal("600")
                or type(self.max_concurrency) is not int
                or not 1 <= self.max_concurrency <= 10_000
                or type(self.requests_per_minute) is not int
                or not 1 <= self.requests_per_minute <= 1_000_000
                or type(self.batch_size) is not int
                or self.batch_size != 1
            ):
                raise ValueError
        except Exception:
            raise ValueError("Embedding probe operational configuration is invalid") from None


type _EmbeddingRequestConfig = EmbeddingConfigSnapshot | EmbeddingDimensionProbeRequest
type _EmbeddingOperationalControls = EmbeddingOperationalConfig | EmbeddingProbeOperationalConfig


@dataclass(slots=True, repr=False)
class _ProviderAdmissionTicket:
    wake: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass(slots=True, repr=False)
class _ProviderLimitState:
    condition: asyncio.Condition
    queue: deque[_ProviderAdmissionTicket] = field(default_factory=deque)
    rate_tokens: float | None = None
    rate_updated_at: float | None = None
    rate_capacity: int = 0
    rate_per_minute: int = 0
    active: int = 0
    waiters: int = 0
    references: int = 0


async def _finish_cleanup(cleanup: Awaitable[None]) -> bool:
    cleanup_future = asyncio.ensure_future(cleanup)
    cancellation_requested = False
    while not cleanup_future.done():
        try:
            await asyncio.shield(cleanup_future)
        except asyncio.CancelledError:
            cancellation_requested = True
    cleanup_future.result()
    return cancellation_requested


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    vectors: tuple[tuple[float, ...], ...]
    usage: Mapping[str, int]
    telemetry_degraded: bool = False

    def __post_init__(self) -> None:
        copied_usage = dict(self.usage)
        if (
            any(
                type(name) is not str
                or name not in _SAFE_USAGE_FIELDS
                or type(value) is not int
                or not 0 <= value <= _POSTGRES_BIGINT_MAX
                for name, value in copied_usage.items()
            )
            or type(self.telemetry_degraded) is not bool
        ):
            raise ValueError("Embedding result usage is invalid")
        object.__setattr__(self, "usage", MappingProxyType(copied_usage))


@dataclass(frozen=True, slots=True, repr=False)
class _ProbeDimensionSentinel:
    pass


_PROBE_DIMENSION: Final = _ProbeDimensionSentinel()
type _ExpectedDimension = int | _ProbeDimensionSentinel


@dataclass(frozen=True, slots=True)
class _ParsedEmbeddingResponse:
    result: EmbeddingResult | None
    dimension: int | None
    usage: Mapping[str, int]
    telemetry_degraded: bool

    def __post_init__(self) -> None:
        copied_usage = dict(self.usage)
        if (
            (self.result is None) == (self.dimension is None)
            or (
                self.dimension is not None
                and (
                    type(self.dimension) is not int
                    or not 1 <= self.dimension <= _MAX_EMBEDDING_DIMENSION
                )
            )
            or any(
                type(name) is not str
                or name not in _SAFE_USAGE_FIELDS
                or type(value) is not int
                or not 0 <= value <= _POSTGRES_BIGINT_MAX
                for name, value in copied_usage.items()
            )
            or type(self.telemetry_degraded) is not bool
        ):
            raise ValueError("Parsed embedding response is invalid")
        object.__setattr__(self, "usage", MappingProxyType(copied_usage))


@dataclass(frozen=True, slots=True, repr=False)
class EmbeddingAttempt:
    """One content-free Provider network attempt safe for persistence."""

    provider_identifier: str
    model_identifier: str
    route_identifier: str | None
    provider_request_id: str | None
    input_tokens: int
    output_tokens: int
    cost_micros: int
    currency: str
    latency_ms: int
    status: Literal["succeeded", "failed", "rate_limited", "timeout", "cancelled"]
    error_code: str | None
    degraded: bool

    def __post_init__(self) -> None:
        if (
            type(self.provider_identifier) is not str
            or not 1 <= len(self.provider_identifier) <= 120
            or type(self.model_identifier) is not str
            or not 1 <= len(self.model_identifier) <= 255
            or (
                self.route_identifier is not None
                and (
                    type(self.route_identifier) is not str
                    or not 1 <= len(self.route_identifier) <= 255
                )
            )
            or (
                self.provider_request_id is not None
                and (
                    type(self.provider_request_id) is not str
                    or not 1 <= len(self.provider_request_id) <= 255
                )
            )
            or any(
                type(value) is not int or not 0 <= value <= _POSTGRES_BIGINT_MAX
                for value in (
                    self.input_tokens,
                    self.output_tokens,
                    self.cost_micros,
                    self.latency_ms,
                )
            )
            or self.currency != "USD"
            or self.status not in {"succeeded", "failed", "rate_limited", "timeout", "cancelled"}
            or (
                self.error_code is not None
                and (type(self.error_code) is not str or not 1 <= len(self.error_code) <= 64)
            )
            or type(self.degraded) is not bool
        ):
            raise ValueError("Embedding attempt is invalid")

    def __repr__(self) -> str:
        return (
            f"EmbeddingAttempt(provider_identifier={self.provider_identifier!r}, "
            f"model_identifier={self.model_identifier!r}, status={self.status!r}, "
            f"error_code={self.error_code!r})"
        )


type EmbeddingAttemptObserver = Callable[
    [EmbeddingAttempt],
    Coroutine[object, object, None] | None,
]


class EmbeddingGatewayError(Exception):
    """Stable sanitized Provider failure classification for jobs and search."""

    __slots__ = ("code", "retryable")

    def __init__(self, code: str, message: str, *, retryable: bool) -> None:
        self.code = code
        self.retryable = retryable
        super().__init__(message)


def _error(code: str, message: str, *, retryable: bool = False) -> EmbeddingGatewayError:
    return EmbeddingGatewayError(code, message, retryable=retryable)


def _input_is_valid(
    value: object,
    *,
    max_input_tokens: int,
    token_counter: Callable[[str], int] | None,
) -> bool:
    encoded = b""
    try:
        if type(value) is not str or not value.strip() or "\x00" in value:
            raise ValueError
        if token_counter is None:
            encoded = value.encode("utf-8", errors="strict")
            input_tokens = len(encoded)
        else:
            input_tokens = token_counter(value)
        if type(input_tokens) is not int or not 0 <= input_tokens <= max_input_tokens:
            raise ValueError
        return True
    except Exception:
        return False
    finally:
        encoded = b""


def _validate_inputs(
    inputs: Sequence[str],
    *,
    snapshot: EmbeddingConfigSnapshot,
    operational: EmbeddingOperationalConfig,
    token_counter: Callable[[str], int] | None,
) -> tuple[str, ...] | EmbeddingGatewayError:
    if (
        isinstance(inputs, (str, bytes, bytearray))
        or not isinstance(inputs, Sequence)
        or not 1 <= len(inputs) <= operational.batch_size
        or len(inputs) > _MAX_RESPONSE_ITEMS
    ):
        return _error("EMBEDDING_BATCH_INVALID", "Embedding batch is invalid")
    copied: list[str] = []
    for value in inputs:
        if not _input_is_valid(
            value,
            max_input_tokens=snapshot.max_input_tokens,
            token_counter=token_counter,
        ):
            copied.clear()
            return _error("EMBEDDING_INPUT_INVALID", "Embedding input is invalid")
        copied.append(value)
    return tuple(copied)


def _validate_probe_input(
    input_text: object,
    *,
    token_counter: Callable[[str], int] | None,
) -> tuple[str, ...] | EmbeddingGatewayError:
    encoded = b""
    try:
        if type(input_text) is not str:
            raise ValueError
        encoded = input_text.encode("utf-8", errors="strict")
        if (
            not input_text.strip()
            or "\x00" in input_text
            or not 1 <= len(encoded) <= _MAX_PROBE_INPUT_BYTES
        ):
            raise ValueError
        input_tokens = len(encoded) if token_counter is None else token_counter(input_text)
        if type(input_tokens) is not int or not 0 <= input_tokens <= _MAX_PROBE_INPUT_TOKENS:
            raise ValueError
    except Exception:
        return _error("EMBEDDING_INPUT_INVALID", "Embedding input is invalid")
    finally:
        encoded = b""
    assert type(input_text) is str
    return (input_text,)


def _status_error(status_code: int) -> EmbeddingGatewayError | None:
    failure = status_failure(
        status_code, input_rejected_message="Provider rejected embedding input"
    )
    if failure is None:
        return None
    return _error(failure.code, failure.message, retryable=failure.retryable)


@dataclass(frozen=True, slots=True)
class _SafeUsage:
    values: Mapping[str, int]
    degraded: bool


@dataclass(frozen=True, slots=True)
class _ResponseTelemetry:
    provider_identifier: str
    model_identifier: str
    route_identifier: str
    cost_micros: int
    degraded: bool


def _safe_usage(value: object) -> _SafeUsage:
    if value is None:
        return _SafeUsage(MappingProxyType({}), False)
    if not isinstance(value, Mapping):
        return _SafeUsage(MappingProxyType({}), True)
    usage: dict[str, int] = {}
    degraded = False
    for name in _SAFE_USAGE_FIELDS:
        if name not in value:
            continue
        count = value[name]
        if type(count) is not int or not 0 <= count <= _POSTGRES_BIGINT_MAX:
            degraded = True
            continue
        usage[name] = count
    return _SafeUsage(MappingProxyType(usage), degraded)


def _safe_optional_identifier(
    document: Mapping[object, object],
    name: str,
    *,
    max_length: int,
) -> tuple[str | None, bool]:
    if name not in document:
        return None, False
    value = document[name]
    if type(value) is not str:
        return None, True
    stripped = value.strip()
    if not 1 <= len(stripped) <= max_length or any(
        ord(character) < 32 or ord(character) > 126 for character in stripped
    ):
        return None, True
    return stripped, False


def _safe_cost_micros(value: object) -> tuple[int, bool]:
    if type(value) not in {int, float, Decimal}:
        return 0, True
    try:
        if isinstance(value, Decimal):
            decimal_value = value
        elif isinstance(value, float):
            decimal_value = Decimal(str(value))
        else:
            assert isinstance(value, int)
            decimal_value = Decimal(value)
        if not decimal_value.is_finite() or decimal_value < 0:
            raise ValueError
        micros = int(
            (decimal_value * Decimal(1_000_000)).quantize(
                Decimal("1"),
                rounding=ROUND_HALF_UP,
            )
        )
        if not 0 <= micros <= _POSTGRES_BIGINT_MAX:
            raise ValueError
        return micros, False
    except (InvalidOperation, OverflowError, TypeError, ValueError):
        return 0, True


def _fallback_response_telemetry(
    snapshot: _EmbeddingRequestConfig,
    *,
    degraded: bool,
) -> _ResponseTelemetry:
    return _ResponseTelemetry(
        provider_identifier=snapshot.provider_type,
        model_identifier=snapshot.model_name,
        route_identifier=(
            "openrouter:unknown" if snapshot.provider_type == "openrouter" else "direct"
        ),
        cost_micros=0,
        degraded=degraded,
    )


def _response_telemetry(
    document: Mapping[object, object],
    *,
    snapshot: _EmbeddingRequestConfig,
) -> _ResponseTelemetry:
    route_fallback = "openrouter:unknown" if snapshot.provider_type == "openrouter" else "direct"
    try:
        model, model_invalid = _safe_optional_identifier(document, "model", max_length=255)
        provider, provider_invalid = _safe_optional_identifier(
            document,
            "provider",
            max_length=120,
        )
        route, route_invalid = _safe_optional_identifier(document, "route", max_length=255)
        usage = document.get("usage")
        cost_micros = 0
        cost_invalid = False
        if isinstance(usage, Mapping) and "cost" in usage:
            cost_micros, cost_invalid = _safe_cost_micros(usage["cost"])
        elif usage is not None and not isinstance(usage, Mapping):
            cost_invalid = True
        return _ResponseTelemetry(
            provider_identifier=snapshot.provider_type if provider is None else provider,
            model_identifier=snapshot.model_name if model is None else model,
            route_identifier=route_fallback if route is None else route,
            cost_micros=cost_micros,
            degraded=model_invalid or provider_invalid or route_invalid or cost_invalid,
        )
    except (ValueError, TypeError, OverflowError):
        return _fallback_response_telemetry(snapshot, degraded=True)


def _decode_success_document(
    response: ProviderHttpResponse,
) -> Mapping[object, object] | EmbeddingGatewayError:
    try:
        document: object = json.loads(response.body, parse_float=Decimal)
        if not isinstance(document, dict):
            return _error("PROVIDER_RESPONSE_INVALID", "Provider response is invalid")
        return document
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError, TypeError, OverflowError):
        return _error("PROVIDER_RESPONSE_INVALID", "Provider response is invalid")


def _safe_latency_ms(started_at: float, finished_at: float) -> tuple[int, bool]:
    try:
        latency_ms = max(0, int((finished_at - started_at) * 1000))
        if latency_ms > _POSTGRES_BIGINT_MAX:
            raise ValueError
        return latency_ms, False
    except (OverflowError, TypeError, ValueError):
        return 0, True


def _safe_provider_request_id(headers: Mapping[str, str]) -> str | None:
    for name in ("x-request-id", "request-id", "openai-request-id"):
        value = headers.get(name)
        if (
            type(value) is str
            and 1 <= len(value) <= 255
            and all(32 <= ord(character) <= 126 for character in value)
        ):
            return value
    return None


def _attempt_status(
    *,
    cancelled: bool,
    failure: EmbeddingGatewayError | None,
) -> Literal["succeeded", "failed", "rate_limited", "timeout", "cancelled"]:
    if cancelled:
        return "cancelled"
    if failure is None:
        return "succeeded"
    if failure.code == "PROVIDER_RATE_LIMITED":
        return "rate_limited"
    if failure.code == "PROVIDER_TIMEOUT":
        return "timeout"
    return "failed"


def _parse_success_response(
    document: Mapping[object, object],
    *,
    input_count: int,
    expected_dimension: _ExpectedDimension,
) -> _ParsedEmbeddingResponse | EmbeddingGatewayError:
    data: object = None
    try:
        probe_mode = expected_dimension is _PROBE_DIMENSION
        if (
            type(input_count) is not int
            or not 1 <= input_count <= _MAX_RESPONSE_ITEMS
            or (probe_mode and input_count != 1)
            or (
                not probe_mode
                and (
                    type(expected_dimension) is not int
                    or not 1 <= expected_dimension <= _MAX_EMBEDDING_DIMENSION
                )
            )
        ):
            return _error("PROVIDER_RESPONSE_INVALID", "Provider response is invalid")
        data = document.get("data")
        if not isinstance(data, list):
            return _error("PROVIDER_RESPONSE_INVALID", "Provider response is invalid")
        if len(data) != input_count:
            return _error(
                "PROVIDER_RESPONSE_COUNT_MISMATCH",
                "Provider response count mismatch",
            )
        vectors_by_index: dict[int, tuple[float, ...]] = {}
        seen_indices: set[int] = set()
        probed_dimension: int | None = None
        for item in data:
            if not isinstance(item, dict):
                return _error("PROVIDER_RESPONSE_INVALID", "Provider response is invalid")
            index = item.get("index")
            raw_vector = item.get("embedding")
            if (
                type(index) is not int
                or not 0 <= index < input_count
                or index in seen_indices
                or not isinstance(raw_vector, list)
            ):
                return _error(
                    "PROVIDER_RESPONSE_COUNT_MISMATCH",
                    "Provider response count mismatch",
                )
            vector_dimension = len(raw_vector)
            if probe_mode:
                if not 1 <= vector_dimension <= _MAX_EMBEDDING_DIMENSION:
                    return _error("PROVIDER_RESPONSE_INVALID", "Provider response is invalid")
                probed_dimension = vector_dimension
            elif vector_dimension != expected_dimension:
                return _error(
                    "PROVIDER_RESPONSE_DIMENSION_MISMATCH",
                    "Provider response dimension mismatch",
                )
            for position, component in enumerate(raw_vector):
                if type(component) not in {int, float, Decimal}:
                    return _error("PROVIDER_RESPONSE_INVALID", "Provider response is invalid")
                try:
                    numeric = float(component)
                except (OverflowError, ValueError):
                    return _error(
                        "PROVIDER_RESPONSE_NONFINITE",
                        "Provider response contains non-finite values",
                    )
                if not math.isfinite(numeric):
                    return _error(
                        "PROVIDER_RESPONSE_NONFINITE",
                        "Provider response contains non-finite values",
                    )
                if not probe_mode:
                    raw_vector[position] = numeric
            seen_indices.add(index)
            if not probe_mode:
                vectors_by_index[index] = tuple(raw_vector)
        if seen_indices != set(range(input_count)):
            return _error(
                "PROVIDER_RESPONSE_COUNT_MISMATCH",
                "Provider response count mismatch",
            )
        safe_usage = _safe_usage(document.get("usage"))
        if probe_mode:
            if probed_dimension is None:
                return _error("PROVIDER_RESPONSE_INVALID", "Provider response is invalid")
            return _ParsedEmbeddingResponse(
                result=None,
                dimension=probed_dimension,
                usage=safe_usage.values,
                telemetry_degraded=safe_usage.degraded,
            )
        result = EmbeddingResult(
            vectors=tuple(vectors_by_index[index] for index in range(input_count)),
            usage=safe_usage.values,
            telemetry_degraded=safe_usage.degraded,
        )
        return _ParsedEmbeddingResponse(
            result=result,
            dimension=None,
            usage=safe_usage.values,
            telemetry_degraded=safe_usage.degraded,
        )
    except (ValueError, TypeError, OverflowError):
        return _error("PROVIDER_RESPONSE_INVALID", "Provider response is invalid")
    finally:
        data = None


def _classified_transport_error(error: Exception) -> EmbeddingGatewayError:
    if isinstance(error, EmbeddingGatewayError):
        return error
    failure = transport_failure(error)
    return _error(failure.code, failure.message, retryable=failure.retryable)


class EmbeddingGateway:
    """One call boundary shared by indexing and query embedding."""

    def __init__(
        self,
        *,
        keyring: ProviderCredentialKeyring,
        credential_reader: ProviderCredentialReader,
        transport: ProviderJsonTransport,
        token_counter: Callable[[str], int] | None = None,
        monotonic_clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
        condition_factory: Callable[[], asyncio.Condition] = asyncio.Condition,
        max_provider_states: int = 1024,
        attempt_observer_timeout_seconds: float = _DEFAULT_ATTEMPT_OBSERVER_TIMEOUT_SECONDS,
        metrics: OperationalMetrics = METRICS,
    ) -> None:
        if (
            type(keyring) is not ProviderCredentialKeyring
            or not callable(getattr(credential_reader, "get_encrypted", None))
            or not callable(getattr(transport, "post_json", None))
            or (token_counter is not None and not callable(token_counter))
            or not callable(monotonic_clock)
            or not callable(sleeper)
            or not callable(condition_factory)
            or type(max_provider_states) is not int
            or not 1 <= max_provider_states <= 100_000
            or type(attempt_observer_timeout_seconds) not in {int, float}
            or not math.isfinite(float(attempt_observer_timeout_seconds))
            or not 0.01 <= float(attempt_observer_timeout_seconds) <= 60.0
        ):
            raise ValueError("Embedding gateway dependencies are invalid")
        self._keyring = keyring
        self._credential_reader = credential_reader
        self._transport = transport
        self._token_counter = token_counter
        self._monotonic_clock = monotonic_clock
        self._sleeper = sleeper
        self._condition_factory = condition_factory
        self._max_provider_states = max_provider_states
        self._attempt_observer_timeout_seconds = float(attempt_observer_timeout_seconds)
        self._metrics = metrics
        self._limit_states: dict[UUID, _ProviderLimitState] = {}
        self._limit_states_condition = condition_factory()
        if not isinstance(self._limit_states_condition, asyncio.Condition):
            raise ValueError("Embedding gateway dependencies are invalid")
        self._limit_states_wake = asyncio.Event()
        self._attempt_observer_tasks: set[asyncio.Future[object]] = set()

    def _observe_attempt(self, provider_type: str, attempt: EmbeddingAttempt) -> None:
        with suppress(BaseException):
            self._metrics.record_provider_attempt(
                provider_type=provider_type,
                status=attempt.status,
                duration_seconds=attempt.latency_ms / 1000,
                input_tokens=attempt.input_tokens,
                output_tokens=attempt.output_tokens,
                cost_micros=attempt.cost_micros,
            )
        fields: dict[str, object] = {
            "operation": "provider_request",
            "provider_type": provider_type,
            "status": attempt.status,
            "latency_ms": attempt.latency_ms,
            "input_tokens": attempt.input_tokens,
            "output_tokens": attempt.output_tokens,
            "cost_micros": attempt.cost_micros,
            "degraded": attempt.degraded,
        }
        if attempt.status == "rate_limited":
            fields["error_code"] = "PROVIDER_RATE_LIMITED"
        elif attempt.status == "timeout":
            fields["error_code"] = "PROVIDER_TIMEOUT"
        elif attempt.status == "failed":
            fields["error_code"] = "PROVIDER_ERROR"
        emit_safe_log(
            logger,
            logging.INFO,
            "provider.request.completed",
            context=None,
            **fields,
        )

    def _consume_attempt_observer_task(self, task: asyncio.Future[object]) -> None:
        self._attempt_observer_tasks.discard(task)
        with suppress(asyncio.CancelledError, Exception):
            task.exception()

    async def _cancel_attempt_observer_tasks(
        self,
        tasks: tuple[asyncio.Future[object], ...],
    ) -> None:
        for task in tasks:
            task.cancel()
        if tasks:
            done, pending = await asyncio.wait(
                tasks,
                timeout=self._attempt_observer_timeout_seconds,
            )
            for task in done:
                self._consume_attempt_observer_task(task)
            # Cancellation-resistant callbacks stay bounded by the global task cap
            # and remain owned until their done callback drains them. Production
            # usage observers are cancellation-propagating repository operations.
            del pending

    async def _notify_attempt(
        self,
        observer: EmbeddingAttemptObserver,
        attempt: EmbeddingAttempt,
    ) -> None:
        if len(self._attempt_observer_tasks) >= _MAX_ATTEMPT_OBSERVER_TASKS:
            return
        try:
            result = observer(attempt)
        except Exception:
            return
        if result is None:
            return
        if isinstance(result, asyncio.Future):
            returned_future = result
            self._attempt_observer_tasks.add(returned_future)
            returned_future.add_done_callback(self._consume_attempt_observer_task)
            await self._cancel_attempt_observer_tasks((returned_future,))
            return
        if not inspect.iscoroutine(result):
            cancel = getattr(result, "cancel", None)
            if callable(cancel):
                with suppress(Exception):
                    cancel()
            close = getattr(result, "close", None)
            if callable(close):
                with suppress(Exception):
                    close()
            return
        observer_task = asyncio.create_task(result, name="rag-embedding-attempt-observer")
        self._attempt_observer_tasks.add(observer_task)
        observer_task.add_done_callback(self._consume_attempt_observer_task)
        try:
            done, _pending = await asyncio.wait(
                {observer_task},
                timeout=self._attempt_observer_timeout_seconds,
            )
            if observer_task not in done:
                await self._cancel_attempt_observer_tasks((observer_task,))
        except asyncio.CancelledError:
            await asyncio.shield(self._cancel_attempt_observer_tasks((observer_task,)))
            raise

    def _now(self) -> float:
        now = float(self._monotonic_clock())
        if not math.isfinite(now):
            raise _error(
                "EMBEDDING_CONFIGURATION_INVALID",
                "Embedding configuration is invalid",
            )
        return now

    @staticmethod
    def _refill_existing_rate(state: _ProviderLimitState, now: float) -> None:
        if state.rate_tokens is None or state.rate_updated_at is None:
            return
        effective_now = max(state.rate_updated_at, now)
        elapsed = effective_now - state.rate_updated_at
        if elapsed > 0 and state.rate_per_minute > 0:
            state.rate_tokens = min(
                float(state.rate_capacity),
                state.rate_tokens + elapsed * float(state.rate_per_minute) / _RATE_WINDOW_SECONDS,
            )
        state.rate_updated_at = effective_now

    @classmethod
    def _refresh_rate(
        cls,
        state: _ProviderLimitState,
        *,
        now: float,
        requests_per_minute: int,
    ) -> None:
        if state.rate_tokens is None:
            state.rate_tokens = float(requests_per_minute)
            state.rate_updated_at = now
        else:
            cls._refill_existing_rate(state, now)
            state.rate_tokens = min(state.rate_tokens, float(requests_per_minute))
        state.rate_capacity = requests_per_minute
        state.rate_per_minute = requests_per_minute

    @classmethod
    def _reclaim_delay(cls, state: _ProviderLimitState, now: float) -> float | None:
        if state.references or state.active or state.waiters or state.queue:
            return None
        if state.rate_tokens is None:
            return 0.0
        cls._refill_existing_rate(state, now)
        missing = float(state.rate_capacity) - state.rate_tokens
        if missing <= 1e-9:
            return 0.0
        if state.rate_per_minute <= 0:
            return None
        return missing * _RATE_WINDOW_SECONDS / float(state.rate_per_minute)

    def _prune_reclaimable_states(self, now: float) -> None:
        reclaimable = [
            provider_config_id
            for provider_config_id, state in self._limit_states.items()
            if self._reclaim_delay(state, now) == 0.0
        ]
        for provider_config_id in reclaimable:
            del self._limit_states[provider_config_id]

    async def _notify_state_registry(self) -> None:
        async with self._limit_states_condition:
            self._limit_states_wake.set()
            self._limit_states_condition.notify_all()

    async def _limit_state(self, provider_config_id: UUID) -> _ProviderLimitState:
        while True:
            delay: float | None = None
            async with self._limit_states_condition:
                state = self._limit_states.get(provider_config_id)
                if state is not None:
                    state.references += 1
                    return state
                now = self._now()
                self._prune_reclaimable_states(now)
                if len(self._limit_states) < self._max_provider_states:
                    state = _ProviderLimitState(condition=self._condition_factory(), references=1)
                    if not isinstance(state.condition, asyncio.Condition):
                        raise _error(
                            "EMBEDDING_CONFIGURATION_INVALID",
                            "Embedding configuration is invalid",
                        )
                    self._limit_states[provider_config_id] = state
                    return state
                self._limit_states_wake.clear()
                delays = [
                    candidate_delay
                    for candidate in self._limit_states.values()
                    if (candidate_delay := self._reclaim_delay(candidate, now)) is not None
                    and candidate_delay > 0
                ]
                if delays:
                    delay = min(delays)
                else:
                    await self._limit_states_condition.wait()
                    continue
            if delay is not None:
                await self._wait_for_registry_or_state_change(delay)

    async def _release_state_reference(self, state: _ProviderLimitState) -> None:
        async with self._limit_states_condition:
            if state.references <= 0:
                raise AssertionError("provider limit state reference is invalid")
            state.references -= 1
            self._limit_states_wake.set()
            self._limit_states_condition.notify_all()

    async def _wait_for_registry_or_state_change(self, delay: float) -> None:
        async def wait_for_reclaim_deadline() -> None:
            await self._sleeper(delay)

        deadline_task = asyncio.create_task(
            wait_for_reclaim_deadline(),
            name="rag-embedding-registry-deadline",
        )
        wake_task = asyncio.create_task(
            self._limit_states_wake.wait(),
            name="rag-embedding-registry-wake",
        )
        cancellation_requested = False

        async def settle_wait_tasks() -> None:
            if not deadline_task.done():
                deadline_task.cancel()
            if not wake_task.done():
                wake_task.cancel()
            results = await asyncio.gather(deadline_task, wake_task, return_exceptions=True)
            for result in results:
                if isinstance(result, asyncio.CancelledError):
                    continue
                if isinstance(result, BaseException):
                    raise result

        try:
            await asyncio.wait(
                (deadline_task, wake_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            cancellation_requested = await _finish_cleanup(settle_wait_tasks())
        if cancellation_requested:
            raise asyncio.CancelledError from None

    async def _admit(
        self,
        state: _ProviderLimitState,
        operational: _EmbeddingOperationalControls,
    ) -> None:
        ticket = _ProviderAdmissionTicket()
        admitted = False
        cancellation_requested = False
        async with state.condition:
            state.queue.append(ticket)
            state.waiters += 1
            self._notify_admission_state(state)
        try:
            while True:
                delay: float | None = None
                async with state.condition:
                    if (
                        state.queue
                        and state.queue[0] is ticket
                        and state.active < operational.max_concurrency
                    ):
                        ticket.wake.clear()
                        self._refresh_rate(
                            state,
                            now=self._now(),
                            requests_per_minute=operational.requests_per_minute,
                        )
                        if state.rate_tokens is not None and state.rate_tokens + 1e-9 >= 1.0:
                            state.queue.popleft()
                            state.waiters -= 1
                            state.active += 1
                            state.rate_tokens = max(0.0, state.rate_tokens - 1.0)
                            admitted = True
                            self._notify_admission_state(state)
                            return
                        available = 0.0 if state.rate_tokens is None else state.rate_tokens
                        delay = (
                            (1.0 - available)
                            * _RATE_WINDOW_SECONDS
                            / float(operational.requests_per_minute)
                        )
                    else:
                        await state.condition.wait()
                if delay is not None:
                    await self._wait_for_rate_or_state_change(ticket, delay)
        finally:
            if not admitted:

                async def remove_unadmitted_ticket() -> None:
                    async with state.condition:
                        with suppress(ValueError):
                            state.queue.remove(ticket)
                            state.waiters -= 1
                        self._notify_admission_state(state)

                cancellation_requested = await _finish_cleanup(remove_unadmitted_ticket())
        if cancellation_requested:
            raise asyncio.CancelledError from None

    @staticmethod
    def _notify_admission_state(state: _ProviderLimitState) -> None:
        if state.queue:
            state.queue[0].wake.set()
        state.condition.notify_all()

    async def _wait_for_rate_or_state_change(
        self,
        ticket: _ProviderAdmissionTicket,
        delay: float,
    ) -> None:
        async def wait_for_rate_deadline() -> None:
            await self._sleeper(delay)

        sleep_task = asyncio.create_task(
            wait_for_rate_deadline(),
            name="rag-embedding-rate-deadline",
        )
        wake_task = asyncio.create_task(
            ticket.wake.wait(),
            name="rag-embedding-admission-wake",
        )
        cancellation_requested = False

        async def settle_wait_tasks() -> None:
            if not sleep_task.done():
                sleep_task.cancel()
            if not wake_task.done():
                wake_task.cancel()
            results = await asyncio.gather(sleep_task, wake_task, return_exceptions=True)
            for result in results:
                if isinstance(result, asyncio.CancelledError):
                    continue
                if isinstance(result, BaseException):
                    raise result

        try:
            await asyncio.wait(
                (sleep_task, wake_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            cancellation_requested = await _finish_cleanup(settle_wait_tasks())
        if cancellation_requested:
            raise asyncio.CancelledError from None

    async def _refund_rate_limit(self, state: _ProviderLimitState) -> None:
        async with state.condition:
            self._refill_existing_rate(state, self._now())
            if state.rate_tokens is None or state.rate_capacity <= 0:
                raise AssertionError("provider rate state is invalid")
            state.rate_tokens = min(float(state.rate_capacity), state.rate_tokens + 1.0)
            self._notify_admission_state(state)
        await self._notify_state_registry()

    async def _release_concurrency(self, state: _ProviderLimitState) -> None:
        async with state.condition:
            if state.active <= 0:
                raise AssertionError("provider concurrency state is invalid")
            state.active -= 1
            self._notify_admission_state(state)
        await self._notify_state_registry()

    async def _request(
        self,
        *,
        snapshot: _EmbeddingRequestConfig,
        operational: _EmbeddingOperationalControls,
        inputs: tuple[str, ...],
        encrypted: EncryptedProviderCredential,
        attempt_started: Callable[[], None],
        attempt_finished: Callable[[bool], None],
    ) -> ProviderHttpResponse:
        async def invoke(secret_buffer: bytearray) -> ProviderHttpResponse:
            headers: dict[str, str] = {}
            payload: dict[str, object] = {}
            routing: dict[str, object] = {}
            authorization = "<redacted>"
            secret = "<redacted>"
            try:
                if (
                    not secret_buffer
                    or len(secret_buffer) > 8192
                    or any(byte < 33 or byte > 126 for byte in secret_buffer)
                ):
                    raise _error(
                        "PROVIDER_CREDENTIAL_INVALID",
                        "Provider credential is invalid",
                    )
                secret = secret_buffer.decode("ascii", errors="strict")
                authorization = f"Bearer {secret}"
                secret = "<redacted>"
                headers = {
                    "Accept": "application/json",
                    "Authorization": authorization,
                    "Content-Type": "application/json",
                    **dict(snapshot.default_headers),
                }
                payload = {
                    "input": list(inputs),
                    "model": snapshot.model_name,
                }
                if snapshot.provider_type == "openrouter" and snapshot.routing_options:
                    routing = _thawed_mapping(snapshot.routing_options)
                    payload["provider"] = routing
                endpoint = validate_provider_endpoint_url(snapshot.base_url)
                attempt_started()
                try:
                    provider_response = await self._transport.post_json(
                        endpoint=endpoint,
                        path="/embeddings",
                        headers=headers,
                        payload=payload,
                        timeout_seconds=float(operational.timeout_seconds),
                    )
                except asyncio.CancelledError:
                    attempt_finished(True)
                    raise
                except BaseException:
                    attempt_finished(False)
                    raise
                attempt_finished(False)
                return provider_response
            finally:
                headers.clear()
                payload.clear()
                routing.clear()
                authorization = "<redacted>"
                secret = "<redacted>"

        return await self._keyring.use_decrypted_async(
            snapshot.credential_id,
            encrypted,
            invoke,
        )

    async def embed(
        self,
        *,
        snapshot: EmbeddingConfigSnapshot,
        operational: EmbeddingOperationalConfig,
        inputs: Sequence[str],
        attempt_observer: EmbeddingAttemptObserver | None = None,
    ) -> EmbeddingResult:
        if (
            type(snapshot) is not EmbeddingConfigSnapshot
            or type(operational) is not EmbeddingOperationalConfig
        ):
            raise _error("EMBEDDING_CONFIGURATION_INVALID", "Embedding configuration is invalid")
        if not operational.provider_enabled:
            raise _error("PROVIDER_DISABLED", "Provider is disabled")
        if not operational.profile_enabled:
            raise _error("MODEL_PROFILE_DISABLED", "Model profile is disabled")
        validated_inputs = _validate_inputs(
            inputs,
            snapshot=snapshot,
            operational=operational,
            token_counter=self._token_counter,
        )
        if isinstance(validated_inputs, EmbeddingGatewayError):
            raise validated_inputs
        parsed = await self._execute(
            snapshot=snapshot,
            operational=operational,
            inputs=validated_inputs,
            expected_dimension=snapshot.dimension,
            attempt_observer=attempt_observer,
        )
        if parsed.result is None or parsed.dimension is not None:
            raise _error("PROVIDER_RESPONSE_INVALID", "Provider response is invalid")
        return parsed.result

    async def probe_dimension(
        self,
        *,
        request: EmbeddingDimensionProbeRequest,
        operational: EmbeddingProbeOperationalConfig,
        input_text: str,
        attempt_observer: EmbeddingAttemptObserver | None = None,
    ) -> int:
        if (
            type(request) is not EmbeddingDimensionProbeRequest
            or type(operational) is not EmbeddingProbeOperationalConfig
        ):
            raise _error("EMBEDDING_CONFIGURATION_INVALID", "Embedding configuration is invalid")
        if not operational.provider_enabled:
            raise _error("PROVIDER_DISABLED", "Provider is disabled")
        validated_inputs = _validate_probe_input(
            input_text,
            token_counter=self._token_counter,
        )
        if isinstance(validated_inputs, EmbeddingGatewayError):
            raise validated_inputs
        parsed = await self._execute(
            snapshot=request,
            operational=operational,
            inputs=validated_inputs,
            expected_dimension=_PROBE_DIMENSION,
            attempt_observer=attempt_observer,
        )
        if parsed.result is not None or parsed.dimension is None:
            raise _error("PROVIDER_RESPONSE_INVALID", "Provider response is invalid")
        return parsed.dimension

    async def _execute(
        self,
        *,
        snapshot: _EmbeddingRequestConfig,
        operational: _EmbeddingOperationalControls,
        inputs: tuple[str, ...],
        expected_dimension: _ExpectedDimension,
        attempt_observer: EmbeddingAttemptObserver | None = None,
    ) -> _ParsedEmbeddingResponse:
        if type(snapshot) is EmbeddingConfigSnapshot:
            if (
                type(operational) is not EmbeddingOperationalConfig
                or type(expected_dimension) is not int
                or expected_dimension != snapshot.dimension
            ):
                raise _error(
                    "EMBEDDING_CONFIGURATION_INVALID",
                    "Embedding configuration is invalid",
                )
        elif type(snapshot) is EmbeddingDimensionProbeRequest:
            if (
                type(operational) is not EmbeddingProbeOperationalConfig
                or expected_dimension is not _PROBE_DIMENSION
                or len(inputs) != 1
            ):
                raise _error(
                    "EMBEDDING_CONFIGURATION_INVALID",
                    "Embedding configuration is invalid",
                )
        else:
            raise _error("EMBEDDING_CONFIGURATION_INVALID", "Embedding configuration is invalid")
        validated_inputs = inputs
        inputs = ()

        encrypted: EncryptedProviderCredential | None = None
        response = ProviderHttpResponse(status_code=500, headers={}, body=b"")
        decoded_document: Mapping[object, object] | EmbeddingGatewayError | None = None
        failure: EmbeddingGatewayError | None = None
        parsed_response: _ParsedEmbeddingResponse | None = None
        limit_state: _ProviderLimitState | None = None
        state_reference_acquired = False
        concurrency_acquired = False
        rate_reserved = False
        provider_attempted = False
        provider_attempt_started_at: float | None = None
        provider_attempt_finished_at: float | None = None
        provider_request_cancelled = False
        provider_attempt_clock_degraded = False
        response_telemetry = _fallback_response_telemetry(snapshot, degraded=False)
        cancelled = False
        cleanup_error: BaseException | None = None

        def mark_provider_attempted() -> None:
            nonlocal provider_attempted, provider_attempt_started_at
            provider_attempt_started_at = self._now()
            provider_attempted = True

        def mark_provider_attempt_finished(request_cancelled: bool) -> None:
            nonlocal provider_attempt_finished_at
            nonlocal provider_request_cancelled
            nonlocal provider_attempt_clock_degraded
            provider_request_cancelled = request_cancelled
            try:
                provider_attempt_finished_at = self._now()
            except BaseException:
                provider_attempt_finished_at = None
                provider_attempt_clock_degraded = True

        try:
            limit_state = await self._limit_state(operational.provider_config_id)
            state_reference_acquired = True
            await self._admit(limit_state, operational)
            concurrency_acquired = True
            rate_reserved = True
            encrypted = await self._credential_reader.get_encrypted(snapshot.credential_id)
            if encrypted is None:
                failure = _error(
                    "PROVIDER_CREDENTIAL_UNAVAILABLE",
                    "Provider credential unavailable",
                )
            else:
                response = await self._request(
                    snapshot=snapshot,
                    operational=operational,
                    inputs=validated_inputs,
                    encrypted=encrypted,
                    attempt_started=mark_provider_attempted,
                    attempt_finished=mark_provider_attempt_finished,
                )
                failure = _status_error(response.status_code)
                if failure is None:
                    decoded_document = _decode_success_document(response)
                    if isinstance(decoded_document, EmbeddingGatewayError):
                        response_telemetry = _fallback_response_telemetry(
                            snapshot,
                            degraded=True,
                        )
                        failure = decoded_document
                    else:
                        response_telemetry = _response_telemetry(
                            decoded_document,
                            snapshot=snapshot,
                        )
                        parsed = _parse_success_response(
                            decoded_document,
                            input_count=len(validated_inputs),
                            expected_dimension=expected_dimension,
                        )
                        if isinstance(parsed, EmbeddingGatewayError):
                            failure = parsed
                        else:
                            parsed_response = parsed
                    decoded_document = None
        except asyncio.CancelledError:
            cancelled = True
        except Exception as error:
            failure = _classified_transport_error(error)
        finally:
            provider_request_id = _safe_provider_request_id(response.headers)
            usage = {} if parsed_response is None else dict(parsed_response.usage)
            response = ProviderHttpResponse(status_code=500, headers={}, body=b"")
            decoded_document = None
            encrypted = None
            validated_inputs = ()
            if rate_reserved and not provider_attempted and limit_state is not None:
                try:
                    cancelled = (
                        await _finish_cleanup(self._refund_rate_limit(limit_state)) or cancelled
                    )
                except BaseException as error:
                    cleanup_error = error
            if concurrency_acquired and limit_state is not None:
                try:
                    cancelled = (
                        await _finish_cleanup(self._release_concurrency(limit_state)) or cancelled
                    )
                except BaseException as error:
                    if cleanup_error is None:
                        cleanup_error = error
            if state_reference_acquired and limit_state is not None:
                try:
                    cancelled = (
                        await _finish_cleanup(self._release_state_reference(limit_state))
                        or cancelled
                    )
                except BaseException as error:
                    if cleanup_error is None:
                        cleanup_error = error
            limit_state = None

        if provider_attempted and provider_attempt_started_at is not None:
            if provider_attempt_finished_at is not None:
                latency_ms, latency_degraded = _safe_latency_ms(
                    provider_attempt_started_at,
                    provider_attempt_finished_at,
                )
            else:
                latency_ms = 0
                latency_degraded = True
            telemetry_degraded = (
                response_telemetry.degraded
                or latency_degraded
                or provider_attempt_clock_degraded
                or (parsed_response is not None and parsed_response.telemetry_degraded)
            )
            attempt: EmbeddingAttempt | None
            try:
                attempt_status = _attempt_status(
                    cancelled=provider_request_cancelled,
                    failure=failure,
                )
                attempt = EmbeddingAttempt(
                    provider_identifier=response_telemetry.provider_identifier,
                    model_identifier=response_telemetry.model_identifier,
                    route_identifier=response_telemetry.route_identifier,
                    provider_request_id=provider_request_id,
                    input_tokens=usage.get("prompt_tokens", usage.get("total_tokens", 0)),
                    output_tokens=usage.get("completion_tokens", 0),
                    cost_micros=response_telemetry.cost_micros,
                    currency="USD",
                    latency_ms=latency_ms,
                    status=attempt_status,
                    error_code=(
                        "PROVIDER_CANCELLED"
                        if attempt_status == "cancelled"
                        else "PROVIDER_RATE_LIMITED"
                        if attempt_status == "rate_limited"
                        else "PROVIDER_TIMEOUT"
                        if attempt_status == "timeout"
                        else "PROVIDER_ERROR"
                        if attempt_status == "failed"
                        else _TELEMETRY_ERROR_CODE
                        if telemetry_degraded
                        else None
                    ),
                    degraded=telemetry_degraded,
                )
            except Exception:
                attempt = None
            if attempt is not None:
                self._observe_attempt(snapshot.provider_type, attempt)
                if attempt_observer is not None:
                    try:
                        await self._notify_attempt(attempt_observer, attempt)
                    except asyncio.CancelledError:
                        if cleanup_error is None:
                            cancelled = True
                    except BaseException:
                        if cleanup_error is None:
                            raise

        if cleanup_error is not None:
            raise cleanup_error

        if cancelled:
            raise asyncio.CancelledError from None
        if failure is not None:
            raise failure from None
        if parsed_response is None:
            raise _error("PROVIDER_RESPONSE_INVALID", "Provider response is invalid")
        return parsed_response

    async def aclose(self) -> None:
        await self._cancel_attempt_observer_tasks(tuple(self._attempt_observer_tasks))
        await self._transport.aclose()


__all__ = [
    "EmbeddingAttempt",
    "EmbeddingAttemptObserver",
    "EmbeddingConfigSnapshot",
    "EmbeddingDimensionProbeRequest",
    "EmbeddingGateway",
    "EmbeddingGatewayError",
    "EmbeddingOperationalConfig",
    "EmbeddingProbeOperationalConfig",
    "EmbeddingResult",
    "ProviderCredentialReader",
]
