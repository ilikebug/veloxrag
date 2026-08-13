from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import FrozenInstanceError, dataclass, fields
from decimal import Decimal
from importlib.metadata import entry_points
from typing import Any, cast
from uuid import UUID, uuid4

import httpcore
import httpx
import pytest

import rag_service.providers.embeddings as embeddings_module
from rag_service.config import Environment
from rag_service.dev import provider_stub
from rag_service.observability.metrics import OperationalMetrics
from rag_service.observability.repositories import ProviderUsageContext
from rag_service.providers.credentials import (
    EncryptedProviderCredential,
    ProviderCredentialKeyring,
    ProviderCredentialUnavailableError,
)
from rag_service.providers.embeddings import (
    EmbeddingAttempt,
    EmbeddingConfigSnapshot,
    EmbeddingDimensionProbeRequest,
    EmbeddingGateway,
    EmbeddingGatewayError,
    EmbeddingOperationalConfig,
    EmbeddingProbeOperationalConfig,
    EmbeddingResult,
)
from rag_service.providers.network_policy import (
    ProviderNetworkPolicyError,
    validate_provider_endpoint_url,
)
from rag_service.providers.services import SqlAlchemyProviderCredentialReader
from rag_service.providers.transport import ProviderHttpResponse

_KEY = b"e" * 32
_SECRET = "embedding-secret-sentinel"
_ROTATED_SECRET = "rotated-embedding-secret-sentinel"
_PROVIDER_CONFIG_ID = UUID("00000000-0000-4000-8000-000000000001")


def _keyring() -> ProviderCredentialKeyring:
    return ProviderCredentialKeyring(keys={"v1": _KEY}, active_key_version="v1")


def _encrypted(credential_id: UUID, secret: str) -> EncryptedProviderCredential:
    return _keyring().encrypt(credential_id, secret.encode("utf-8"))


def _snapshot(
    credential_id: UUID,
    *,
    provider_type: str = "openai_compatible",
    base_url: str = "https://provider.example/v1",
    default_headers: Mapping[str, str] | None = None,
    routing_options: Mapping[str, object] | None = None,
    dimension: int = 3,
    max_input_tokens: int = 8192,
) -> EmbeddingConfigSnapshot:
    return EmbeddingConfigSnapshot(
        adapter_schema_version="openai-embeddings-v1",
        provider_type=provider_type,
        base_url=base_url,
        credential_id=credential_id,
        default_headers={} if default_headers is None else dict(default_headers),
        routing_options={} if routing_options is None else dict(routing_options),
        model_name="text-embedding-test",
        dimension=dimension,
        distance="cosine",
        max_input_tokens=max_input_tokens,
        vector_config={},
    )


def _operational(
    *,
    provider_enabled: bool = True,
    profile_enabled: bool = True,
    timeout_seconds: Decimal = Decimal("5.000"),
    batch_size: int = 8,
) -> EmbeddingOperationalConfig:
    return EmbeddingOperationalConfig(
        provider_config_id=_PROVIDER_CONFIG_ID,
        provider_enabled=provider_enabled,
        profile_enabled=profile_enabled,
        timeout_seconds=timeout_seconds,
        max_concurrency=4,
        requests_per_minute=120,
        batch_size=batch_size,
    )


def _provider_operational(
    provider_config_id: UUID,
    *,
    max_concurrency: int = 1,
    requests_per_minute: int = 60,
) -> EmbeddingOperationalConfig:
    return EmbeddingOperationalConfig(
        provider_config_id=provider_config_id,
        provider_enabled=True,
        profile_enabled=True,
        timeout_seconds=Decimal("5.000"),
        max_concurrency=max_concurrency,
        requests_per_minute=requests_per_minute,
        batch_size=8,
    )


def _probe_request(
    credential_id: UUID,
    *,
    provider_type: str = "openai_compatible",
    base_url: str = "https://provider.example/v1",
    default_headers: Mapping[str, str] | None = None,
    routing_options: Mapping[str, object] | None = None,
    model_name: str = "text-embedding-test",
) -> EmbeddingDimensionProbeRequest:
    return EmbeddingDimensionProbeRequest(
        adapter_schema_version="openai-embeddings-v1",
        provider_type=provider_type,
        base_url=base_url,
        credential_id=credential_id,
        default_headers={} if default_headers is None else dict(default_headers),
        routing_options={} if routing_options is None else dict(routing_options),
        model_name=model_name,
    )


def _probe_operational(
    provider_config_id: UUID = _PROVIDER_CONFIG_ID,
    *,
    provider_enabled: bool = True,
    timeout_seconds: Decimal = Decimal("5.000"),
    max_concurrency: int = 4,
    requests_per_minute: int = 120,
) -> EmbeddingProbeOperationalConfig:
    return EmbeddingProbeOperationalConfig(
        provider_config_id=provider_config_id,
        provider_enabled=provider_enabled,
        timeout_seconds=timeout_seconds,
        max_concurrency=max_concurrency,
        requests_per_minute=requests_per_minute,
        batch_size=1,
    )


@dataclass(slots=True)
class FakeCredentialReader:
    values: dict[UUID, EncryptedProviderCredential]
    calls: list[UUID]

    async def get_encrypted(
        self,
        credential_id: UUID,
    ) -> EncryptedProviderCredential | None:
        self.calls.append(credential_id)
        return self.values.get(credential_id)


class RecordingTransport:
    def __init__(
        self,
        responses: list[ProviderHttpResponse | BaseException],
    ) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, object]] = []
        self.authorization_hashes: list[str] = []
        self.header_references: list[dict[str, str]] = []
        self.payload_references: list[dict[str, object]] = []
        self.closed = False

    async def post_json(
        self,
        *,
        endpoint: object,
        path: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> ProviderHttpResponse:
        authorization = headers.get("Authorization", "")
        self.authorization_hashes.append(hashlib.sha256(authorization.encode()).hexdigest())
        self.requests.append(
            {
                "endpoint": endpoint,
                "path": path,
                "headers": {
                    key: value for key, value in headers.items() if key.lower() != "authorization"
                },
                "payload": json.loads(json.dumps(payload)),
                "timeout_seconds": timeout_seconds,
                "authorization_present": bool(authorization),
            }
        )
        self.header_references.append(headers)
        self.payload_references.append(payload)
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    async def aclose(self) -> None:
        self.closed = True


class _ManualClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleep_calls: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        assert delay > 0
        self.sleep_calls.append(delay)
        self.now += delay


@dataclass(slots=True)
class _ControlledSleepCall:
    delay: float
    release: asyncio.Event
    finished: asyncio.Event
    cancelled: bool = False


class _ControlledClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now
        self.calls: list[_ControlledSleepCall] = []
        self.started: asyncio.Queue[_ControlledSleepCall] = asyncio.Queue()

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, delay: float) -> None:
        assert delay > 0
        call = _ControlledSleepCall(
            delay=delay,
            release=asyncio.Event(),
            finished=asyncio.Event(),
        )
        self.calls.append(call)
        await self.started.put(call)
        try:
            await call.release.wait()
        except asyncio.CancelledError:
            call.cancelled = True
            raise
        finally:
            call.finished.set()

    def pending(self) -> list[_ControlledSleepCall]:
        return [call for call in self.calls if not call.finished.is_set()]

    def release_shortest(self, *, advance: bool = True) -> _ControlledSleepCall:
        call = min(self.pending(), key=lambda candidate: candidate.delay)
        if advance:
            self.now += call.delay
        call.release.set()
        return call

    def release_all(self) -> None:
        for call in self.pending():
            call.release.set()


class _CleanupBlockingCondition(asyncio.Condition):
    def __init__(self) -> None:
        super().__init__()
        self._block_next_context_entry = False
        self.context_entry_started = asyncio.Event()
        self.allow_context_entry = asyncio.Event()

    def block_next_context_entry(self) -> None:
        self._block_next_context_entry = True

    async def __aenter__(self) -> None:
        if self._block_next_context_entry:
            self._block_next_context_entry = False
            self.context_entry_started.set()
            await self.allow_context_entry.wait()
        await self.acquire()


class _CleanupBlockingConditionFactory:
    def __init__(self) -> None:
        self.conditions: list[_CleanupBlockingCondition] = []

    def __call__(self) -> asyncio.Condition:
        condition = _CleanupBlockingCondition()
        self.conditions.append(condition)
        return condition


class _ClockRecordingTransport(RecordingTransport):
    def __init__(
        self,
        responses: list[ProviderHttpResponse | BaseException],
        clock: _ManualClock,
    ) -> None:
        super().__init__(responses)
        self._clock = clock
        self.request_times: list[float] = []

    async def post_json(
        self,
        *,
        endpoint: object,
        path: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> ProviderHttpResponse:
        self.request_times.append(self._clock.monotonic())
        return await super().post_json(
            endpoint=endpoint,
            path=path,
            headers=headers,
            payload=payload,
            timeout_seconds=timeout_seconds,
        )


class _BlockingTransport:
    def __init__(self) -> None:
        self.entered: asyncio.Queue[int] = asyncio.Queue()
        self.release = asyncio.Event()
        self.calls = 0
        self.active = 0
        self.max_active = 0

    async def post_json(
        self,
        *,
        endpoint: object,
        path: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> ProviderHttpResponse:
        del endpoint, path, headers, payload, timeout_seconds
        self.calls += 1
        call_number = self.calls
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await self.entered.put(call_number)
        try:
            await self.release.wait()
            return _response([[0.1, 0.2, 0.3]])
        finally:
            self.active -= 1

    async def aclose(self) -> None:
        return None


class _AuthorizationBlockingTransport(_BlockingTransport):
    def __init__(self) -> None:
        super().__init__()
        self.authorization_hashes: list[str] = []

    async def post_json(
        self,
        *,
        endpoint: object,
        path: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> ProviderHttpResponse:
        self.authorization_hashes.append(
            hashlib.sha256(headers.get("Authorization", "").encode()).hexdigest()
        )
        return await super().post_json(
            endpoint=endpoint,
            path=path,
            headers=headers,
            payload=payload,
            timeout_seconds=timeout_seconds,
        )


class _AdmissionGateCredentialReader:
    def __init__(self, encrypted: EncryptedProviderCredential) -> None:
        self._encrypted = encrypted
        self.calls: list[UUID] = []
        self.first_read_started = asyncio.Event()
        self.release_first_read = asyncio.Event()

    async def get_encrypted(
        self,
        credential_id: UUID,
    ) -> EncryptedProviderCredential | None:
        self.calls.append(credential_id)
        if len(self.calls) == 1:
            self.first_read_started.set()
            await self.release_first_read.wait()
        return self._encrypted


class _GatedSequencedCredentialReader:
    def __init__(
        self,
        first: EncryptedProviderCredential | None,
        subsequent: EncryptedProviderCredential,
    ) -> None:
        self._first = first
        self._subsequent = subsequent
        self.calls: list[UUID] = []
        self.first_read_started = asyncio.Event()
        self.release_first_read = asyncio.Event()

    async def get_encrypted(
        self,
        credential_id: UUID,
    ) -> EncryptedProviderCredential | None:
        self.calls.append(credential_id)
        if len(self.calls) == 1:
            self.first_read_started.set()
            await self.release_first_read.wait()
            return self._first
        return self._subsequent


class _FifoGateTransport:
    def __init__(self) -> None:
        self.entered: asyncio.Queue[str] = asyncio.Queue()
        self._gates: asyncio.Queue[asyncio.Event] = asyncio.Queue()
        self.entry_order: list[str] = []

    async def post_json(
        self,
        *,
        endpoint: object,
        path: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> ProviderHttpResponse:
        del endpoint, path, headers, timeout_seconds
        value = cast(list[str], payload["input"])[0]
        gate = asyncio.Event()
        self.entry_order.append(value)
        await self._gates.put(gate)
        await self.entered.put(value)
        await gate.wait()
        return _response([[0.1, 0.2, 0.3]])

    async def release_next(self) -> None:
        gate = await self._gates.get()
        gate.set()

    async def aclose(self) -> None:
        while not self._gates.empty():
            await self.release_next()


class _CleanupGateEmbeddingGateway(EmbeddingGateway):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.cleanup_started = asyncio.Event()
        self.allow_cleanup = asyncio.Event()

    async def _release_concurrency(self, state: Any) -> None:
        self.cleanup_started.set()
        await self.allow_cleanup.wait()
        await super()._release_concurrency(state)


async def _drain_ready_tasks() -> None:
    await asyncio.sleep(0)
    await asyncio.sleep(0)


def _response(
    vectors: list[list[float]],
    *,
    status_code: int = 200,
    usage: object | None = None,
    indices: list[int] | None = None,
    model: object | None = None,
    provider: object | None = None,
    route: object | None = None,
) -> ProviderHttpResponse:
    if indices is None:
        indices = list(range(len(vectors)))
    document: dict[str, object] = {
        "data": [
            {"object": "embedding", "index": index, "embedding": vector}
            for index, vector in zip(indices, vectors, strict=True)
        ]
    }
    if usage is not None:
        document["usage"] = usage
    if model is not None:
        document["model"] = model
    if provider is not None:
        document["provider"] = provider
    if route is not None:
        document["route"] = route
    return ProviderHttpResponse(
        status_code=status_code,
        headers={"content-type": "application/json"},
        body=json.dumps(document, allow_nan=True).encode(),
    )


def _gateway(
    credential_id: UUID,
    transport: RecordingTransport,
    *,
    secret: str = _SECRET,
    monotonic_clock: Callable[[], float] = time.monotonic,
    metrics: OperationalMetrics | object | None = None,
) -> tuple[EmbeddingGateway, FakeCredentialReader]:
    reader = FakeCredentialReader(
        values={credential_id: _encrypted(credential_id, secret)},
        calls=[],
    )
    kwargs: dict[str, object] = {}
    if metrics is not None:
        kwargs["metrics"] = metrics
    return (
        EmbeddingGateway(
            keyring=_keyring(),
            credential_reader=reader,
            transport=transport,
            monotonic_clock=monotonic_clock,
            attempt_observer_timeout_seconds=0.25,
            **kwargs,  # type: ignore[arg-type]
        ),
        reader,
    )


def test_probe_public_models_are_minimal_immutable_and_safely_represented() -> None:
    credential_id = uuid4()
    request = _probe_request(
        credential_id,
        base_url="https://provider.example/v1/",
        model_name="  text-embedding-test  ",
    )
    operational = _probe_operational()

    assert {field.name for field in fields(request)} == {
        "adapter_schema_version",
        "provider_type",
        "base_url",
        "credential_id",
        "default_headers",
        "routing_options",
        "model_name",
    }
    assert {field.name for field in fields(operational)} == {
        "provider_config_id",
        "provider_enabled",
        "timeout_seconds",
        "max_concurrency",
        "requests_per_minute",
        "batch_size",
    }
    assert request.base_url == "https://provider.example/v1"
    assert request.model_name == "text-embedding-test"
    assert operational.batch_size == 1
    with pytest.raises(FrozenInstanceError):
        cast(Any, request).model_name = "other"
    with pytest.raises(FrozenInstanceError):
        cast(Any, operational).batch_size = 2
    assert repr(request) == "EmbeddingDimensionProbeRequest(<redacted>)"
    assert str(credential_id) not in repr(request)
    assert "provider.example" not in repr(request)
    assert "text-embedding-test" not in repr(request)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("adapter_schema_version", 1),
        ("adapter_schema_version", "other"),
        ("provider_type", 1),
        ("provider_type", "unsupported"),
        ("base_url", 1),
        ("base_url", "not-a-provider-url"),
        ("credential_id", "not-a-uuid"),
        ("default_headers", []),
        ("routing_options", []),
        ("model_name", 1),
        ("model_name", "   "),
        ("model_name", "x" * 256),
        ("model_name", "model\nname"),
        ("model_name", "model\x7fname"),
    ],
)
def test_probe_request_rejects_invalid_exact_field_values(
    field_name: str,
    bad_value: object,
) -> None:
    arguments: dict[str, object] = {
        "adapter_schema_version": "openai-embeddings-v1",
        "provider_type": "openai_compatible",
        "base_url": "https://provider.example/v1",
        "credential_id": uuid4(),
        "default_headers": {},
        "routing_options": {},
        "model_name": "text-embedding-test",
    }
    arguments[field_name] = bad_value

    with pytest.raises(ValueError) as exc_info:
        EmbeddingDimensionProbeRequest(**cast(Any, arguments))

    assert exc_info.value.args == ("Embedding dimension probe request is invalid",)


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("provider_config_id", "not-a-uuid"),
        ("provider_enabled", 1),
        ("timeout_seconds", 1),
        ("timeout_seconds", Decimal("0")),
        ("timeout_seconds", Decimal("600.001")),
        ("timeout_seconds", Decimal("NaN")),
        ("max_concurrency", True),
        ("max_concurrency", 0),
        ("max_concurrency", 10_001),
        ("requests_per_minute", True),
        ("requests_per_minute", 0),
        ("requests_per_minute", 1_000_001),
        ("batch_size", True),
        ("batch_size", 0),
        ("batch_size", 2),
    ],
)
def test_probe_operational_config_rejects_invalid_exact_field_values(
    field_name: str,
    bad_value: object,
) -> None:
    arguments: dict[str, object] = {
        "provider_config_id": uuid4(),
        "provider_enabled": True,
        "timeout_seconds": Decimal("5"),
        "max_concurrency": 4,
        "requests_per_minute": 120,
        "batch_size": 1,
    }
    arguments[field_name] = bad_value

    with pytest.raises(ValueError) as exc_info:
        EmbeddingProbeOperationalConfig(**cast(Any, arguments))

    assert exc_info.value.args == ("Embedding probe operational configuration is invalid",)


def test_probe_request_defensively_freezes_headers_and_nested_routing() -> None:
    source_headers = {"X-OpenRouter-Title": "RAG"}
    source_order = ["openai"]
    source_routing: dict[str, object] = {"order": source_order}
    request = _probe_request(
        uuid4(),
        provider_type="openrouter",
        default_headers=source_headers,
        routing_options=source_routing,
    )

    source_headers["X-OpenRouter-Title"] = "changed"
    source_order.append("mistral")
    source_routing["unknown"] = True

    assert request.default_headers == {"X-OpenRouter-Title": "RAG"}
    assert request.routing_options == {"order": ("openai",)}
    with pytest.raises(TypeError):
        cast(Any, request.default_headers)["X-OpenRouter-Title"] = "changed"
    with pytest.raises((AttributeError, TypeError)):
        cast(Any, request.routing_options["order"]).append("mistral")


def test_probe_public_types_are_exported() -> None:
    assert "EmbeddingDimensionProbeRequest" in embeddings_module.__all__
    assert "EmbeddingProbeOperationalConfig" in embeddings_module.__all__


@pytest.mark.asyncio
async def test_probe_sends_exactly_one_fixed_input_and_returns_only_dimension() -> None:
    credential_id = uuid4()
    transport = RecordingTransport(
        [_response([[0.125, 0.25, 0.5, 1.0]], usage={"prompt_tokens": 2})]
    )
    gateway, reader = _gateway(credential_id, transport)
    input_text = "  fixed dimension probe input  "

    dimension = await gateway.probe_dimension(
        request=_probe_request(credential_id),
        operational=_probe_operational(),
        input_text=input_text,
    )

    assert type(dimension) is int
    assert dimension == 4
    assert not isinstance(dimension, EmbeddingResult)
    assert reader.calls == [credential_id]
    assert transport.requests[0]["payload"] == {
        "input": [input_text],
        "model": "text-embedding-test",
    }
    assert transport._responses == []
    assert "0.125" not in repr(gateway.__dict__)


@pytest.mark.asyncio
async def test_probe_openrouter_reuses_routing_headers_auth_and_canonical_request_path() -> None:
    credential_id = uuid4()
    transport = RecordingTransport([_response([[0.1, 0.2, 0.3]])])
    gateway, _reader = _gateway(credential_id, transport)
    request = _probe_request(
        credential_id,
        provider_type="openrouter",
        default_headers={
            "HTTP-Referer": "https://app.example",
            "X-OpenRouter-Title": "RAG",
        },
        routing_options={"order": ["openai", "mistral"], "allow_fallbacks": False},
    )

    assert (
        await gateway.probe_dimension(
            request=request,
            operational=_probe_operational(),
            input_text="probe",
        )
        == 3
    )

    recorded = transport.requests[0]
    assert recorded["endpoint"] == validate_provider_endpoint_url(request.base_url)
    assert recorded["path"] == "/embeddings"
    assert recorded["payload"] == {
        "input": ["probe"],
        "model": "text-embedding-test",
        "provider": {"allow_fallbacks": False, "order": ["openai", "mistral"]},
    }
    assert recorded["headers"] == {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://app.example",
        "X-OpenRouter-Title": "RAG",
    }
    assert recorded["authorization_present"] is True
    assert transport.authorization_hashes == [
        hashlib.sha256(f"Bearer {_SECRET}".encode()).hexdigest()
    ]


@pytest.mark.asyncio
async def test_success_response_body_is_parsed_once_for_vectors_and_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential_id = uuid4()
    response = _response(
        [[0.1, 0.2, 0.3]],
        model="openai/text-embedding-3-small",
        provider="Azure OpenAI",
        route="azure/eastus",
        usage={"prompt_tokens": 3, "total_tokens": 3, "cost": 0.0000126},
    )
    transport = RecordingTransport([response])
    gateway, _reader = _gateway(credential_id, transport)
    attempts: list[EmbeddingAttempt] = []
    original_loads = json.loads
    response_parse_calls = 0
    response_parse_float_hooks: list[object] = []

    def counting_loads(value: str | bytes | bytearray, *args: Any, **kwargs: Any) -> Any:
        nonlocal response_parse_calls
        if isinstance(value, (bytes, bytearray)):
            response_parse_calls += 1
            response_parse_float_hooks.append(kwargs.get("parse_float"))
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr(json, "loads", counting_loads)

    result = await gateway.embed(
        snapshot=_snapshot(credential_id, provider_type="openrouter"),
        operational=_operational(),
        inputs=("query",),
        attempt_observer=attempts.append,
    )

    assert result.vectors == ((0.1, 0.2, 0.3),)
    assert response_parse_calls == 1
    assert response_parse_float_hooks == [Decimal]
    assert attempts[0].provider_identifier == "Azure OpenAI"
    assert attempts[0].model_identifier == "openai/text-embedding-3-small"
    assert attempts[0].route_identifier == "azure/eastus"
    assert attempts[0].cost_micros == 13
    assert attempts[0].degraded is False


@pytest.mark.parametrize(
    ("cost_literal", "expected_cost_micros", "expected_degraded"),
    [
        ("0.00000049999999999999999999", 0, False),
        ("-1e-10000", 0, True),
        ("0.0000126", 13, False),
    ],
)
@pytest.mark.asyncio
async def test_gateway_preserves_exact_json_cost_lexemes(
    cost_literal: str,
    expected_cost_micros: int,
    expected_degraded: bool,
) -> None:
    credential_id = uuid4()
    response = ProviderHttpResponse(
        status_code=200,
        headers={},
        body=(
            b'{"data":[{"index":0,"embedding":[0.1,0.2,0.3]}],'
            b'"usage":{"prompt_tokens":1,"cost":' + cost_literal.encode("ascii") + b"}}"
        ),
    )
    gateway, _reader = _gateway(credential_id, RecordingTransport([response]))
    attempts: list[EmbeddingAttempt] = []

    result = await gateway.embed(
        snapshot=_snapshot(credential_id, provider_type="openrouter"),
        operational=_operational(),
        inputs=("query",),
        attempt_observer=attempts.append,
    )

    assert result.vectors == ((0.1, 0.2, 0.3),)
    assert attempts[0].cost_micros == expected_cost_micros
    assert attempts[0].degraded is expected_degraded
    assert attempts[0].error_code == ("PROVIDER_TELEMETRY_INVALID" if expected_degraded else None)


@pytest.mark.parametrize("status_code", [429, 503])
@pytest.mark.asyncio
async def test_non_success_response_body_is_not_parsed(
    status_code: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    credential_id = uuid4()
    transport = RecordingTransport(
        [
            ProviderHttpResponse(
                status_code=status_code,
                headers={},
                body=b'{"error":{"message":"untrusted-provider-body"}}',
            )
        ]
    )
    gateway, _reader = _gateway(credential_id, transport)
    original_loads = json.loads
    response_parse_calls = 0

    def counting_loads(value: str | bytes | bytearray, *args: Any, **kwargs: Any) -> Any:
        nonlocal response_parse_calls
        if isinstance(value, (bytes, bytearray)):
            response_parse_calls += 1
        return original_loads(value, *args, **kwargs)

    monkeypatch.setattr(json, "loads", counting_loads)

    with pytest.raises(EmbeddingGatewayError):
        await gateway.probe_dimension(
            request=_probe_request(credential_id),
            operational=_probe_operational(),
            input_text="probe",
        )

    assert response_parse_calls == 0


@pytest.mark.parametrize(
    "bad_input",
    [cast(Any, 1), "", "   ", "nul\x00text"],
)
@pytest.mark.asyncio
async def test_probe_rejects_invalid_input_before_credential_lookup(bad_input: str) -> None:
    credential_id = uuid4()
    transport = RecordingTransport([])
    gateway, reader = _gateway(credential_id, transport)

    with pytest.raises(EmbeddingGatewayError) as exc_info:
        await gateway.probe_dimension(
            request=_probe_request(credential_id),
            operational=_probe_operational(),
            input_text=bad_input,
        )

    assert exc_info.value.code == "EMBEDDING_INPUT_INVALID"
    assert reader.calls == []
    assert transport.requests == []


@pytest.mark.asyncio
async def test_probe_rejects_string_subclass_before_calling_valid_counter() -> None:
    class InputText(str):
        pass

    credential_id = uuid4()
    counter_calls: list[str] = []

    def token_counter(value: str) -> int:
        counter_calls.append(value)
        return 1

    reader = FakeCredentialReader(
        values={credential_id: _encrypted(credential_id, _SECRET)},
        calls=[],
    )
    transport = RecordingTransport([])
    gateway = EmbeddingGateway(
        keyring=_keyring(),
        credential_reader=reader,
        transport=transport,
        token_counter=token_counter,
    )

    with pytest.raises(EmbeddingGatewayError) as exc_info:
        await gateway.probe_dimension(
            request=_probe_request(credential_id),
            operational=_probe_operational(),
            input_text=cast(str, InputText("probe")),
        )

    assert exc_info.value.code == "EMBEDDING_INPUT_INVALID"
    assert counter_calls == []
    assert reader.calls == []
    assert transport.requests == []


@pytest.mark.asyncio
async def test_probe_rejects_token_count_above_independent_limit() -> None:
    credential_id = uuid4()
    transport = RecordingTransport([])
    reader = FakeCredentialReader(
        values={credential_id: _encrypted(credential_id, _SECRET)},
        calls=[],
    )
    gateway = EmbeddingGateway(
        keyring=_keyring(),
        credential_reader=reader,
        transport=transport,
        token_counter=lambda _value: 8 * 1024 + 1,
    )

    with pytest.raises(EmbeddingGatewayError) as exc_info:
        await gateway.probe_dimension(
            request=_probe_request(credential_id),
            operational=_probe_operational(),
            input_text="probe",
        )

    assert exc_info.value.code == "EMBEDDING_INPUT_INVALID"
    assert reader.calls == []
    assert transport.requests == []


@pytest.mark.parametrize("counter_result", [True, 1.5, "1", None])
@pytest.mark.asyncio
async def test_probe_rejects_non_exact_integer_token_counts(counter_result: object) -> None:
    credential_id = uuid4()
    transport = RecordingTransport([])
    reader = FakeCredentialReader(
        values={credential_id: _encrypted(credential_id, _SECRET)},
        calls=[],
    )
    gateway = EmbeddingGateway(
        keyring=_keyring(),
        credential_reader=reader,
        transport=transport,
        token_counter=lambda _value: cast(Any, counter_result),
    )

    with pytest.raises(EmbeddingGatewayError) as exc_info:
        await gateway.probe_dimension(
            request=_probe_request(credential_id),
            operational=_probe_operational(),
            input_text="probe",
        )

    assert exc_info.value.code == "EMBEDDING_INPUT_INVALID"
    assert reader.calls == []
    assert transport.requests == []


@pytest.mark.asyncio
async def test_probe_sanitizes_token_counter_failure() -> None:
    credential_id = uuid4()

    def failing_counter(_value: str) -> int:
        raise RuntimeError("counter-failure-sentinel")

    transport = RecordingTransport([])
    reader = FakeCredentialReader(
        values={credential_id: _encrypted(credential_id, _SECRET)},
        calls=[],
    )
    gateway = EmbeddingGateway(
        keyring=_keyring(),
        credential_reader=reader,
        transport=transport,
        token_counter=failing_counter,
    )

    with pytest.raises(EmbeddingGatewayError) as exc_info:
        await gateway.probe_dimension(
            request=_probe_request(credential_id),
            operational=_probe_operational(),
            input_text="probe",
        )

    assert exc_info.value.code == "EMBEDDING_INPUT_INVALID"
    assert "counter-failure-sentinel" not in str(exc_info.value)
    assert reader.calls == []
    assert transport.requests == []


@pytest.mark.parametrize(
    "input_text",
    ["\ud800", "é" * (4 * 1024 + 1)],
)
@pytest.mark.asyncio
async def test_probe_rejects_invalid_or_oversized_utf8_before_token_counter(
    input_text: str,
) -> None:
    credential_id = uuid4()
    counter_calls: list[str] = []

    def token_counter(value: str) -> int:
        counter_calls.append(value)
        return 1

    transport = RecordingTransport([])
    reader = FakeCredentialReader(
        values={credential_id: _encrypted(credential_id, _SECRET)},
        calls=[],
    )
    gateway = EmbeddingGateway(
        keyring=_keyring(),
        credential_reader=reader,
        transport=transport,
        token_counter=token_counter,
    )

    with pytest.raises(EmbeddingGatewayError) as exc_info:
        await gateway.probe_dimension(
            request=_probe_request(credential_id),
            operational=_probe_operational(),
            input_text=input_text,
        )

    assert exc_info.value.code == "EMBEDDING_INPUT_INVALID"
    assert counter_calls == []
    assert reader.calls == []
    assert transport.requests == []


@pytest.mark.asyncio
async def test_probe_only_checks_current_provider_enabled_state() -> None:
    credential_id = uuid4()
    gateway, reader = _gateway(credential_id, RecordingTransport([]))

    with pytest.raises(EmbeddingGatewayError) as exc_info:
        await gateway.probe_dimension(
            request=_probe_request(credential_id),
            operational=_probe_operational(provider_enabled=False),
            input_text="probe",
        )

    assert exc_info.value.code == "PROVIDER_DISABLED"
    assert reader.calls == []
    assert "profile_enabled" not in {field.name for field in fields(_probe_operational())}


@pytest.mark.parametrize(
    ("document", "expected_code"),
    [
        ({"data": []}, "PROVIDER_RESPONSE_COUNT_MISMATCH"),
        (
            {
                "data": [
                    {"index": 0, "embedding": [0.1]},
                    {"index": 1, "embedding": [0.2]},
                ]
            },
            "PROVIDER_RESPONSE_COUNT_MISMATCH",
        ),
        (
            {
                "data": [
                    {"index": 0, "embedding": [0.1]},
                    {"index": 0, "embedding": [0.2]},
                ]
            },
            "PROVIDER_RESPONSE_COUNT_MISMATCH",
        ),
        ({"data": [{"index": 0, "embedding": {"0": 0.1}}]}, "PROVIDER_RESPONSE_COUNT_MISMATCH"),
        ({"data": [{"index": 0, "embedding": []}]}, "PROVIDER_RESPONSE_INVALID"),
        ({"data": [{"index": 0, "embedding": [True]}]}, "PROVIDER_RESPONSE_INVALID"),
        ({"data": [{"index": 0, "embedding": ["0.1"]}]}, "PROVIDER_RESPONSE_INVALID"),
        ({"data": [{"index": 0, "embedding": [math.nan]}]}, "PROVIDER_RESPONSE_NONFINITE"),
        ({"data": [{"index": 0, "embedding": [math.inf]}]}, "PROVIDER_RESPONSE_NONFINITE"),
    ],
)
@pytest.mark.asyncio
async def test_probe_rejects_malformed_vectors_with_stable_sanitized_codes(
    document: dict[str, object],
    expected_code: str,
) -> None:
    credential_id = uuid4()
    response = ProviderHttpResponse(
        status_code=200,
        headers={},
        body=json.dumps(document, allow_nan=True).encode(),
    )
    gateway, _reader = _gateway(credential_id, RecordingTransport([response]))

    with pytest.raises(EmbeddingGatewayError) as exc_info:
        await gateway.probe_dimension(
            request=_probe_request(credential_id),
            operational=_probe_operational(),
            input_text="probe",
        )

    assert exc_info.value.code == expected_code
    assert exc_info.value.retryable is False
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


def test_probe_decoder_rejects_dimension_above_bound_without_allocating_it() -> None:
    class OversizedVector(list[object]):
        def __len__(self) -> int:
            return 10_000_001

    document: Mapping[object, object] = {
        "data": [{"index": 0, "embedding": OversizedVector([0.1])}]
    }

    parsed = embeddings_module._parse_success_response(
        document,
        input_count=1,
        expected_dimension=embeddings_module._PROBE_DIMENSION,
    )

    assert isinstance(parsed, EmbeddingGatewayError)
    assert parsed.code == "PROVIDER_RESPONSE_INVALID"


def test_probe_decoder_returns_dimension_without_a_vector_bearing_result() -> None:
    parsed = embeddings_module._parse_success_response(
        {
            "data": [
                {
                    "index": 0,
                    "embedding": [Decimal("0.1"), Decimal("0.2"), Decimal("0.3")],
                }
            ],
            "usage": {"prompt_tokens": 1},
        },
        input_count=1,
        expected_dimension=embeddings_module._PROBE_DIMENSION,
    )

    assert not isinstance(parsed, EmbeddingGatewayError)
    assert parsed.result is None
    assert parsed.dimension == 3
    assert parsed.usage == {"prompt_tokens": 1}
    assert parsed.telemetry_degraded is False


def test_embedding_decoder_accepts_finite_decimal_components() -> None:
    parsed = embeddings_module._parse_success_response(
        {
            "data": [
                {
                    "index": 0,
                    "embedding": [Decimal("0.125"), Decimal("-0.5"), Decimal("1")],
                }
            ]
        },
        input_count=1,
        expected_dimension=3,
    )

    assert not isinstance(parsed, EmbeddingGatewayError)
    assert parsed.result is not None
    assert parsed.result.vectors == ((0.125, -0.5, 1.0),)


def test_embedding_decoder_reuses_in_place_floats_for_exact_components() -> None:
    raw_vector: list[object] = [Decimal("0.125"), 1, Decimal("-2")]
    parsed = embeddings_module._parse_success_response(
        {"data": [{"index": 0, "embedding": raw_vector}]},
        input_count=1,
        expected_dimension=3,
    )

    assert not isinstance(parsed, EmbeddingGatewayError)
    assert parsed.result is not None
    assert raw_vector == [0.125, 1.0, -2.0]
    assert all(type(component) is float for component in raw_vector)
    assert parsed.result.vectors == ((0.125, 1.0, -2.0),)
    assert all(
        parsed.result.vectors[0][position] is raw_vector[position]
        for position in range(len(raw_vector))
    )


def test_probe_decoder_does_not_mutate_or_retain_exact_components() -> None:
    decimal_component = Decimal("0.125")
    integer_component = 12345678901234567890
    raw_vector: list[object] = [decimal_component, integer_component]
    parsed = embeddings_module._parse_success_response(
        {"data": [{"index": 0, "embedding": raw_vector}]},
        input_count=1,
        expected_dimension=embeddings_module._PROBE_DIMENSION,
    )

    assert not isinstance(parsed, EmbeddingGatewayError)
    assert parsed.result is None
    assert parsed.dimension == 2
    assert raw_vector[0] is decimal_component
    assert raw_vector[1] is integer_component
    assert raw_vector == [Decimal("0.125"), 12345678901234567890]


@pytest.mark.parametrize(
    "component",
    [
        Decimal("NaN"),
        Decimal("Infinity"),
        Decimal("-Infinity"),
        Decimal("1e1000000"),
    ],
)
def test_embedding_decoder_classifies_nonfinite_or_overflowing_decimal_components(
    component: Decimal,
) -> None:
    parsed = embeddings_module._parse_success_response(
        {"data": [{"index": 0, "embedding": [component]}]},
        input_count=1,
        expected_dimension=1,
    )

    assert isinstance(parsed, EmbeddingGatewayError)
    assert parsed.code == "PROVIDER_RESPONSE_NONFINITE"


@pytest.mark.asyncio
async def test_probe_and_embed_share_concurrency_by_provider_config_id() -> None:
    credential_id = uuid4()
    provider_config_id = uuid4()
    transport = _BlockingTransport()
    reader = FakeCredentialReader(
        values={credential_id: _encrypted(credential_id, _SECRET)},
        calls=[],
    )
    gateway = EmbeddingGateway(
        keyring=_keyring(),
        credential_reader=reader,
        transport=transport,
    )
    embed_task: asyncio.Task[EmbeddingResult] | None = None
    probe_task: asyncio.Task[int] | None = None
    try:
        embed_task = asyncio.create_task(
            gateway.embed(
                snapshot=_snapshot(credential_id),
                operational=_provider_operational(
                    provider_config_id,
                    max_concurrency=1,
                    requests_per_minute=1_000_000,
                ),
                inputs=("embed",),
            )
        )
        async with asyncio.timeout(1):
            assert await transport.entered.get() == 1
        probe_task = asyncio.create_task(
            gateway.probe_dimension(
                request=_probe_request(credential_id),
                operational=_probe_operational(
                    provider_config_id,
                    max_concurrency=1,
                    requests_per_minute=1_000_000,
                ),
                input_text="probe",
            )
        )
        await _drain_ready_tasks()

        assert transport.entered.empty()
        assert transport.max_active == 1
    finally:
        transport.release.set()
        await asyncio.gather(
            *(task for task in (embed_task, probe_task) if task is not None),
            return_exceptions=True,
        )


@pytest.mark.asyncio
async def test_probe_and_embed_share_rate_limit_by_provider_config_id() -> None:
    credential_id = uuid4()
    provider_config_id = uuid4()
    clock = _ManualClock()
    transport = _ClockRecordingTransport(
        [_response([[0.1, 0.2, 0.3]]), _response([[0.1, 0.2, 0.3]])],
        clock,
    )
    reader = FakeCredentialReader(
        values={credential_id: _encrypted(credential_id, _SECRET)},
        calls=[],
    )
    gateway = EmbeddingGateway(
        keyring=_keyring(),
        credential_reader=reader,
        transport=transport,
        monotonic_clock=clock.monotonic,
        sleeper=clock.sleep,
    )

    await gateway.embed(
        snapshot=_snapshot(credential_id),
        operational=_provider_operational(provider_config_id, requests_per_minute=1),
        inputs=("embed",),
    )
    dimension = await gateway.probe_dimension(
        request=_probe_request(credential_id),
        operational=_probe_operational(provider_config_id, requests_per_minute=1),
        input_text="probe",
    )

    assert dimension == 3
    assert transport.request_times == [0.0, 60.0]


@pytest.mark.parametrize(
    ("response_or_error", "expected_code", "retryable"),
    [
        (_response([[0.1]], status_code=401), "PROVIDER_AUTHENTICATION_FAILED", False),
        (_response([[0.1]], status_code=429), "PROVIDER_RATE_LIMITED", True),
        (_response([[0.1]], status_code=503), "PROVIDER_UNAVAILABLE", True),
        (httpx.ReadTimeout("probe-timeout"), "PROVIDER_TIMEOUT", True),
        (
            ProviderNetworkPolicyError(),
            "PROVIDER_ENDPOINT_REJECTED",
            False,
        ),
    ],
)
@pytest.mark.asyncio
async def test_probe_preserves_provider_error_classification(
    response_or_error: ProviderHttpResponse | BaseException,
    expected_code: str,
    retryable: bool,
) -> None:
    credential_id = uuid4()
    gateway, _reader = _gateway(credential_id, RecordingTransport([response_or_error]))

    with pytest.raises(EmbeddingGatewayError) as exc_info:
        await gateway.probe_dimension(
            request=_probe_request(credential_id),
            operational=_probe_operational(),
            input_text="probe",
        )

    assert exc_info.value.code == expected_code
    assert exc_info.value.retryable is retryable
    assert "probe-timeout" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_probe_preserves_missing_credential_classification() -> None:
    credential_id = uuid4()
    reader = FakeCredentialReader(values={}, calls=[])
    transport = RecordingTransport([])
    gateway = EmbeddingGateway(
        keyring=_keyring(),
        credential_reader=reader,
        transport=transport,
    )

    with pytest.raises(EmbeddingGatewayError) as exc_info:
        await gateway.probe_dimension(
            request=_probe_request(credential_id),
            operational=_probe_operational(),
            input_text="probe",
        )

    assert exc_info.value.code == "PROVIDER_CREDENTIAL_UNAVAILABLE"
    assert exc_info.value.retryable is False
    assert reader.calls == [credential_id]
    assert transport.requests == []


@pytest.mark.asyncio
async def test_probe_cancellation_propagates_and_clears_request_containers() -> None:
    credential_id = uuid4()
    started = asyncio.Event()

    class HangingProbeTransport(RecordingTransport):
        async def post_json(
            self,
            *,
            endpoint: object,
            path: str,
            headers: dict[str, str],
            payload: dict[str, object],
            timeout_seconds: float,
        ) -> ProviderHttpResponse:
            del endpoint, path, timeout_seconds
            self.header_references.append(headers)
            self.payload_references.append(payload)
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    transport = HangingProbeTransport([])
    gateway, _reader = _gateway(credential_id, transport)
    task = asyncio.create_task(
        gateway.probe_dimension(
            request=_probe_request(credential_id),
            operational=_probe_operational(),
            input_text="probe",
        )
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert transport.header_references == [{}]
    assert transport.payload_references == [{}]


@pytest.mark.asyncio
async def test_probe_accepts_unknown_dimension_but_embed_remains_strict() -> None:
    credential_id = uuid4()
    transport = RecordingTransport([_response([[0.1, 0.2]]), _response([[0.1, 0.2]])])
    gateway, _reader = _gateway(credential_id, transport)

    dimension = await gateway.probe_dimension(
        request=_probe_request(credential_id),
        operational=_probe_operational(),
        input_text="probe",
    )
    with pytest.raises(EmbeddingGatewayError) as exc_info:
        await gateway.embed(
            snapshot=_snapshot(credential_id, dimension=3),
            operational=_operational(),
            inputs=("embed",),
        )

    assert dimension == 2
    assert exc_info.value.code == "PROVIDER_RESPONSE_DIMENSION_MISMATCH"


class _CollectingProviderLogHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.mark.asyncio
async def test_gateway_observes_each_real_network_attempt_without_a_caller_observer() -> None:
    credential_id = uuid4()
    sentinel = "query-vector-provider-secret-ciphertext-nonce-authorization-default-header"
    response = ProviderHttpResponse(
        status_code=200,
        headers={"x-request-id": f"request-{sentinel}"},
        body=_response(
            [[0.1, 0.2, 0.3]],
            usage={"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
            model=f"model-{sentinel}",
            provider=f"provider-{sentinel}",
            route=f"route-{sentinel}",
        ).body,
    )
    metrics = OperationalMetrics()
    handler = _CollectingProviderLogHandler()
    previous_handlers = list(embeddings_module.logger.handlers)
    previous_propagate = embeddings_module.logger.propagate
    previous_level = embeddings_module.logger.level
    embeddings_module.logger.handlers = [handler]
    embeddings_module.logger.propagate = False
    embeddings_module.logger.setLevel(logging.INFO)
    gateway, _reader = _gateway(
        credential_id,
        RecordingTransport([response]),
        metrics=metrics,
    )
    try:
        result = await gateway.embed(
            snapshot=_snapshot(
                credential_id,
                provider_type="openrouter",
                default_headers={"X-Title": sentinel},
            ),
            operational=_operational(),
            inputs=(sentinel,),
        )
    finally:
        embeddings_module.logger.handlers = previous_handlers
        embeddings_module.logger.propagate = previous_propagate
        embeddings_module.logger.setLevel(previous_level)

    assert len(result.vectors) == 1
    assert (
        metrics.registry.get_sample_value(
            "rag_provider_requests_total",
            {"provider_type": "openrouter", "status": "succeeded"},
        )
        == 1
    )
    assert (
        metrics.registry.get_sample_value(
            "rag_provider_usage_tokens_total",
            {"provider_type": "openrouter", "direction": "input"},
        )
        == 7
    )
    assert (
        metrics.registry.get_sample_value(
            "rag_provider_usage_tokens_total",
            {"provider_type": "openrouter", "direction": "output"},
        )
        == 2
    )
    assert len(handler.records) == 1
    record = handler.records[0]
    assert type(record) is logging.LogRecord
    assert record.msg == "provider.request.completed"
    assert record.__dict__["provider_type"] == "openrouter"
    rendered = repr(record.__dict__)
    assert sentinel not in rendered
    assert "model_name" not in rendered
    assert "provider_request_id" not in rendered


@pytest.mark.asyncio
async def test_gateway_metrics_do_not_double_count_when_caller_observer_is_present() -> None:
    credential_id = uuid4()
    metrics = OperationalMetrics()
    gateway, _reader = _gateway(
        credential_id,
        RecordingTransport([_response([], status_code=429)]),
        metrics=metrics,
    )
    attempts: list[EmbeddingAttempt] = []

    with pytest.raises(EmbeddingGatewayError) as captured:
        await gateway.embed(
            snapshot=_snapshot(credential_id),
            operational=_operational(),
            inputs=("query-secret",),
            attempt_observer=attempts.append,
        )

    assert captured.value.code == "PROVIDER_RATE_LIMITED"
    assert len(attempts) == 1
    assert (
        metrics.registry.get_sample_value(
            "rag_provider_requests_total",
            {"provider_type": "openai_compatible", "status": "rate_limited"},
        )
        == 1
    )
    assert (
        metrics.registry.get_sample_value(
            "rag_provider_rate_limits_total", {"provider_type": "openai_compatible"}
        )
        == 1
    )


@pytest.mark.asyncio
async def test_gateway_attempt_duration_excludes_post_http_cleanup_time() -> None:
    credential_id = uuid4()
    clock = _ManualClock()

    class TimedTransport(RecordingTransport):
        async def post_json(self, **kwargs: Any) -> ProviderHttpResponse:
            clock.now += 0.1
            return await super().post_json(**kwargs)

    class SlowCleanupGateway(EmbeddingGateway):
        async def _release_concurrency(self, state: Any) -> None:
            clock.now += 5.0
            await super()._release_concurrency(state)

    transport = TimedTransport([_response([[0.1, 0.2, 0.3]])])
    reader = FakeCredentialReader(
        values={credential_id: _encrypted(credential_id, _SECRET)},
        calls=[],
    )
    gateway = SlowCleanupGateway(
        keyring=_keyring(),
        credential_reader=reader,
        transport=transport,
        monotonic_clock=clock.monotonic,
    )
    attempts: list[EmbeddingAttempt] = []

    await gateway.embed(
        snapshot=_snapshot(credential_id),
        operational=_operational(),
        inputs=("query",),
        attempt_observer=attempts.append,
    )

    assert len(attempts) == 1
    assert attempts[0].status == "succeeded"
    assert attempts[0].latency_ms == 100


@pytest.mark.asyncio
async def test_gateway_drops_raw_response_and_validated_inputs_before_cleanup_awaits() -> None:
    credential_id = uuid4()
    response = _response([[0.1, 0.2, 0.3]])
    response = ProviderHttpResponse(
        status_code=response.status_code,
        headers={"x-request-id": "provider-request-123"},
        body=response.body,
    )
    transport = RecordingTransport([response])
    reader = FakeCredentialReader(
        values={credential_id: _encrypted(credential_id, _SECRET)},
        calls=[],
    )
    gateway = _CleanupGateEmbeddingGateway(
        keyring=_keyring(),
        credential_reader=reader,
        transport=transport,
    )
    task = asyncio.create_task(
        gateway.embed(
            snapshot=_snapshot(credential_id),
            operational=_operational(),
            inputs=("short-lived-input",),
        )
    )
    try:
        async with asyncio.timeout(1):
            await gateway.cleanup_started.wait()
        current: Any = task.get_coro()
        execute_frame = None
        while inspect.iscoroutine(current):
            if current.cr_code.co_name == "_execute":
                execute_frame = current.cr_frame
                break
            current = current.cr_await

        assert execute_frame is not None
        retained_response = cast(ProviderHttpResponse, execute_frame.f_locals["response"])
        assert retained_response.body == b""
        assert execute_frame.f_locals["validated_inputs"] == ()

        gateway.allow_cleanup.set()
        result = await task
        assert result.vectors == ((0.1, 0.2, 0.3),)
    finally:
        gateway.allow_cleanup.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_gateway_cleanup_cancellation_preserves_completed_http_attempt_status() -> None:
    credential_id = uuid4()
    transport = _BlockingTransport()
    reader = FakeCredentialReader(
        values={credential_id: _encrypted(credential_id, _SECRET)},
        calls=[],
    )
    metrics = OperationalMetrics()
    gateway = _CleanupGateEmbeddingGateway(
        keyring=_keyring(),
        credential_reader=reader,
        transport=transport,
        metrics=metrics,
    )
    attempts: list[EmbeddingAttempt] = []
    task = asyncio.create_task(
        gateway.embed(
            snapshot=_snapshot(credential_id),
            operational=_operational(),
            inputs=("query",),
            attempt_observer=attempts.append,
        )
    )
    try:
        async with asyncio.timeout(1):
            await transport.entered.get()
            transport.release.set()
            await gateway.cleanup_started.wait()
        task.cancel()
        gateway.allow_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        transport.release.set()
        gateway.allow_cleanup.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert len(attempts) == 1
    assert attempts[0].status == "succeeded"
    assert (
        metrics.registry.get_sample_value(
            "rag_provider_requests_total",
            {"provider_type": "openai_compatible", "status": "succeeded"},
        )
        == 1
    )
    assert (
        metrics.registry.get_sample_value(
            "rag_provider_requests_total",
            {"provider_type": "openai_compatible", "status": "cancelled"},
        )
        is None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cleanup_type", "observer_raises"),
    (
        (RuntimeError, False),
        (BaseException, False),
        (RuntimeError, True),
    ),
)
async def test_gateway_records_completed_attempt_before_propagating_cleanup_failure(
    cleanup_type: type[BaseException],
    observer_raises: bool,
) -> None:
    credential_id = uuid4()
    cleanup_message = "cleanup-failure-sentinel"
    metrics = OperationalMetrics()
    remaining_cleanup_calls = 0

    class FailingCleanupGateway(EmbeddingGateway):
        async def _release_concurrency(self, state: Any) -> None:
            await super()._release_concurrency(state)
            raise cleanup_type(cleanup_message)

        async def _release_state_reference(self, state: Any) -> None:
            nonlocal remaining_cleanup_calls
            remaining_cleanup_calls += 1
            await super()._release_state_reference(state)

    reader = FakeCredentialReader(
        values={credential_id: _encrypted(credential_id, _SECRET)},
        calls=[],
    )
    gateway = FailingCleanupGateway(
        keyring=_keyring(),
        credential_reader=reader,
        transport=RecordingTransport([_response([[0.1, 0.2, 0.3]])]),
        metrics=metrics,
    )
    attempts: list[EmbeddingAttempt] = []
    observer_calls = 0

    def observer(attempt: EmbeddingAttempt) -> None:
        nonlocal observer_calls
        observer_calls += 1
        attempts.append(attempt)
        if observer_raises:
            raise BaseException("observer-failure-must-not-mask-cleanup")

    handler = _CollectingProviderLogHandler()
    previous_handlers = list(embeddings_module.logger.handlers)
    previous_propagate = embeddings_module.logger.propagate
    previous_level = embeddings_module.logger.level
    embeddings_module.logger.handlers = [handler]
    embeddings_module.logger.propagate = False
    embeddings_module.logger.setLevel(logging.INFO)
    try:
        with pytest.raises(cleanup_type, match=cleanup_message):
            await gateway.embed(
                snapshot=_snapshot(credential_id),
                operational=_operational(),
                inputs=("query",),
                attempt_observer=observer,
            )
    finally:
        embeddings_module.logger.handlers = previous_handlers
        embeddings_module.logger.propagate = previous_propagate
        embeddings_module.logger.setLevel(previous_level)

    assert remaining_cleanup_calls == 1
    assert observer_calls == 1
    assert len(attempts) == 1
    assert attempts[0].status == "succeeded"
    assert (
        metrics.registry.get_sample_value(
            "rag_provider_requests_total",
            {"provider_type": "openai_compatible", "status": "succeeded"},
        )
        == 1
    )
    provider_records = [
        record for record in handler.records if record.msg == "provider.request.completed"
    ]
    assert len(provider_records) == 1
    assert provider_records[0].__dict__["status"] == "succeeded"
    assert cleanup_message not in repr(provider_records[0].__dict__)


@pytest.mark.asyncio
async def test_gateway_request_cancellation_records_cancelled_attempt_once() -> None:
    credential_id = uuid4()
    transport = _BlockingTransport()
    metrics = OperationalMetrics()
    reader = FakeCredentialReader(
        values={credential_id: _encrypted(credential_id, _SECRET)},
        calls=[],
    )
    gateway = EmbeddingGateway(
        keyring=_keyring(),
        credential_reader=reader,
        transport=transport,
        metrics=metrics,
    )
    attempts: list[EmbeddingAttempt] = []
    task = asyncio.create_task(
        gateway.embed(
            snapshot=_snapshot(credential_id),
            operational=_operational(),
            inputs=("query",),
            attempt_observer=attempts.append,
        )
    )
    try:
        async with asyncio.timeout(1):
            await transport.entered.get()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        transport.release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert len(attempts) == 1
    assert attempts[0].status == "cancelled"
    assert (
        metrics.registry.get_sample_value(
            "rag_provider_requests_total",
            {"provider_type": "openai_compatible", "status": "cancelled"},
        )
        == 1
    )


@pytest.mark.asyncio
async def test_gateway_bounds_long_transport_error_code_without_changing_business_error() -> None:
    credential_id = uuid4()
    long_code = "X" * 65
    metrics = OperationalMetrics()
    gateway, _reader = _gateway(
        credential_id,
        RecordingTransport(
            [EmbeddingGatewayError(long_code, "safe provider failure", retryable=True)]
        ),
        metrics=metrics,
    )
    attempts: list[EmbeddingAttempt] = []

    with pytest.raises(EmbeddingGatewayError) as captured:
        await gateway.embed(
            snapshot=_snapshot(credential_id),
            operational=_operational(),
            inputs=("query",),
            attempt_observer=attempts.append,
        )

    assert captured.value.code == long_code
    assert captured.value.retryable is True
    assert len(attempts) == 1
    assert attempts[0].status == "failed"
    assert attempts[0].error_code == "PROVIDER_ERROR"
    assert (
        metrics.registry.get_sample_value(
            "rag_provider_requests_total",
            {"provider_type": "openai_compatible", "status": "failed"},
        )
        == 1
    )


@pytest.mark.asyncio
async def test_gateway_does_not_observe_validation_failures_without_a_network_attempt() -> None:
    credential_id = uuid4()
    metrics = OperationalMetrics()
    transport = RecordingTransport([])
    gateway, _reader = _gateway(credential_id, transport, metrics=metrics)

    with pytest.raises(EmbeddingGatewayError):
        await gateway.embed(
            snapshot=_snapshot(credential_id),
            operational=_operational(),
            inputs=(),
        )

    assert transport.requests == []
    assert list(metrics.registry.collect())[0].samples == []


@pytest.mark.asyncio
async def test_gateway_observability_failures_do_not_change_provider_result() -> None:
    credential_id = uuid4()

    class FailingMetrics:
        def record_provider_attempt(self, **kwargs: object) -> None:
            del kwargs
            raise BaseException("metrics-secret")

    class FailingHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            del record
            raise BaseException("logging-secret")

    previous_handlers = list(embeddings_module.logger.handlers)
    previous_propagate = embeddings_module.logger.propagate
    previous_level = embeddings_module.logger.level
    embeddings_module.logger.handlers = [FailingHandler()]
    embeddings_module.logger.propagate = False
    embeddings_module.logger.setLevel(logging.INFO)
    gateway, _reader = _gateway(
        credential_id,
        RecordingTransport([_response([[0.1, 0.2, 0.3]])]),
        metrics=FailingMetrics(),
    )
    try:
        result = await gateway.embed(
            snapshot=_snapshot(credential_id),
            operational=_operational(),
            inputs=("query",),
        )
    finally:
        embeddings_module.logger.handlers = previous_handlers
        embeddings_module.logger.propagate = previous_propagate
        embeddings_module.logger.setLevel(previous_level)

    assert result.vectors == ((0.1, 0.2, 0.3),)


@pytest.mark.asyncio
async def test_openai_compatible_adapter_builds_only_the_canonical_request() -> None:
    credential_id = uuid4()
    transport = RecordingTransport(
        [
            _response(
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                usage={"prompt_tokens": 7, "total_tokens": 7},
            )
        ]
    )
    gateway, reader = _gateway(credential_id, transport)
    snapshot = _snapshot(
        credential_id,
        default_headers={"X-Title": "RAG embedding"},
    )

    result = await gateway.embed(
        snapshot=snapshot,
        operational=_operational(),
        inputs=("first", "second"),
    )

    assert result == EmbeddingResult(
        vectors=((0.1, 0.2, 0.3), (0.4, 0.5, 0.6)),
        usage={"prompt_tokens": 7, "total_tokens": 7},
    )
    assert reader.calls == [credential_id]
    request = transport.requests[0]
    assert request["endpoint"] == validate_provider_endpoint_url(snapshot.base_url)
    assert request["path"] == "/embeddings"
    assert request["payload"] == {
        "input": ["first", "second"],
        "model": "text-embedding-test",
    }
    assert request["headers"] == {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-Title": "RAG embedding",
    }
    assert request["authorization_present"] is True
    assert transport.authorization_hashes == [
        hashlib.sha256(f"Bearer {_SECRET}".encode()).hexdigest()
    ]
    assert request["timeout_seconds"] == 5.0
    assert all(headers == {} for headers in transport.header_references)
    assert all(payload == {} for payload in transport.payload_references)
    assert _SECRET not in repr(transport.requests)


@pytest.mark.asyncio
async def test_openrouter_adapter_applies_only_snapshot_routing_options() -> None:
    credential_id = uuid4()
    transport = RecordingTransport([_response([[0.1, 0.2, 0.3]])])
    gateway, _reader = _gateway(credential_id, transport)
    snapshot = _snapshot(
        credential_id,
        provider_type="openrouter",
        default_headers={
            "HTTP-Referer": "https://app.example",
            "X-OpenRouter-Title": "RAG",
        },
        routing_options={
            "order": ["openai", "mistral"],
            "allow_fallbacks": False,
            "zdr": True,
        },
    )

    await gateway.embed(
        snapshot=snapshot,
        operational=_operational(),
        inputs=("query",),
    )

    assert transport.requests[0]["payload"] == {
        "input": ["query"],
        "model": "text-embedding-test",
        "provider": {
            "allow_fallbacks": False,
            "order": ["openai", "mistral"],
            "zdr": True,
        },
    }
    assert transport.requests[0]["headers"] == {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://app.example",
        "X-OpenRouter-Title": "RAG",
    }


@pytest.mark.asyncio
async def test_snapshot_routing_is_detached_recursively_from_mutable_constructor_input() -> None:
    credential_id = uuid4()
    source_order = ["openai"]
    source_sort: dict[str, object] = {"by": "price", "partition": "model"}
    source_routing: dict[str, object] = {
        "order": source_order,
        "sort": source_sort,
    }
    snapshot = _snapshot(
        credential_id,
        provider_type="openrouter",
        routing_options=source_routing,
    )
    source_order.append("attacker-provider")
    source_sort["unknown"] = "attacker-routing-secret-sentinel"
    source_routing["api_key"] = "attacker-routing-secret-sentinel"
    transport = RecordingTransport([_response([[0.1, 0.2, 0.3]])])
    gateway, _reader = _gateway(credential_id, transport)

    await gateway.embed(
        snapshot=snapshot,
        operational=_operational(),
        inputs=("query",),
    )

    assert transport.requests[0]["payload"] == {
        "input": ["query"],
        "model": "text-embedding-test",
        "provider": {
            "order": ["openai"],
            "sort": {"by": "price", "partition": "model"},
        },
    }


def test_snapshot_exposed_nested_routing_arrays_are_immutable() -> None:
    snapshot = _snapshot(
        uuid4(),
        provider_type="openrouter",
        routing_options={"order": ["openai"]},
    )

    with pytest.raises((AttributeError, TypeError)):
        cast(Any, snapshot.routing_options["order"]).append("attacker-provider")

    assert snapshot.routing_options["order"] == ("openai",)


def test_snapshot_exposed_nested_routing_objects_reject_unknown_key_injection() -> None:
    snapshot = _snapshot(
        uuid4(),
        provider_type="openrouter",
        routing_options={"sort": {"by": "price", "partition": "model"}},
    )

    with pytest.raises(TypeError):
        cast(Any, snapshot.routing_options["sort"])["unknown"] = "attacker-routing-secret-sentinel"

    assert snapshot.routing_options["sort"] == {
        "by": "price",
        "partition": "model",
    }


@pytest.mark.asyncio
async def test_gateway_thaws_frozen_routing_to_plain_json_safe_request_values() -> None:
    credential_id = uuid4()
    snapshot = _snapshot(
        credential_id,
        provider_type="openrouter",
        routing_options={
            "order": ["openai", "mistral"],
            "sort": {"by": "price", "partition": "model"},
        },
    )
    transport = RecordingTransport([_response([[0.1, 0.2, 0.3]])])
    gateway, _reader = _gateway(credential_id, transport)

    await gateway.embed(
        snapshot=snapshot,
        operational=_operational(),
        inputs=("query",),
    )

    routing = cast(dict[str, object], transport.requests[0]["payload"])["provider"]
    assert type(routing) is dict
    assert type(cast(dict[str, object], routing)["order"]) is list
    assert type(cast(dict[str, object], routing)["sort"]) is dict


@pytest.mark.parametrize(
    ("provider_type", "routing_options"),
    [
        ("openai_compatible", {"order": ["openai"]}),
        ("openrouter", {"api_key": "routing-secret-sentinel"}),
        ("openrouter", {"headers": {"Authorization": "routing-secret-sentinel"}}),
        ("openrouter", {"unknown": True}),
    ],
)
def test_snapshot_rejects_routing_that_could_leak_or_change_the_wrong_protocol(
    provider_type: str,
    routing_options: dict[str, object],
) -> None:
    sentinel = "routing-secret-sentinel"

    with pytest.raises(ValueError) as exc_info:
        _snapshot(
            uuid4(),
            provider_type=provider_type,
            routing_options=routing_options,
        )

    assert sentinel not in str(exc_info.value)
    assert sentinel not in repr(exc_info.value)


@pytest.mark.asyncio
async def test_gateway_uses_snapshot_semantics_and_only_current_operational_controls() -> None:
    credential_id = uuid4()
    transport = RecordingTransport([_response([[0.1, 0.2, 0.3]])])
    gateway, _reader = _gateway(credential_id, transport)
    immutable_snapshot = _snapshot(
        credential_id,
        base_url="https://snapshot-provider.example/openai/v1",
        default_headers={"X-Title": "snapshot title"},
        dimension=3,
    )
    current_operational = EmbeddingOperationalConfig(
        provider_config_id=_PROVIDER_CONFIG_ID,
        provider_enabled=True,
        profile_enabled=True,
        timeout_seconds=Decimal("2.500"),
        max_concurrency=1,
        requests_per_minute=1,
        batch_size=1,
    )

    await gateway.embed(
        snapshot=immutable_snapshot,
        operational=current_operational,
        inputs=("one",),
    )

    request = transport.requests[0]
    assert request["endpoint"] == validate_provider_endpoint_url(
        "https://snapshot-provider.example/openai/v1"
    )
    assert request["headers"] == {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "X-Title": "snapshot title",
    }
    assert request["timeout_seconds"] == 2.5


@pytest.mark.asyncio
async def test_gateway_reports_one_sanitized_success_attempt_before_returning() -> None:
    credential_id = uuid4()
    sentinel = "embedding-input-secret-sentinel"
    response = _response(
        [[0.1, 0.2, 0.3]],
        usage={"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
    )
    response = ProviderHttpResponse(
        status_code=response.status_code,
        headers={"x-request-id": "provider-request-123", "x-secret": _SECRET},
        body=response.body,
    )
    gateway, _reader = _gateway(credential_id, RecordingTransport([response]))
    attempts: list[EmbeddingAttempt] = []

    result = await gateway.embed(
        snapshot=_snapshot(credential_id),
        operational=_operational(),
        inputs=(sentinel,),
        attempt_observer=attempts.append,
    )

    assert len(result.vectors) == 1
    assert attempts == [
        EmbeddingAttempt(
            provider_identifier="openai_compatible",
            model_identifier="text-embedding-test",
            route_identifier="direct",
            provider_request_id="provider-request-123",
            input_tokens=7,
            output_tokens=2,
            cost_micros=0,
            currency="USD",
            latency_ms=0,
            status="succeeded",
            error_code=None,
            degraded=False,
        )
    ]
    rendered = repr(attempts) + str(attempts)
    assert sentinel not in rendered
    assert _SECRET not in rendered


@pytest.mark.asyncio
async def test_gateway_records_safe_actual_openrouter_route_model_and_decimal_cost() -> None:
    credential_id = uuid4()
    clock = _ManualClock()
    response = _response(
        [[0.1, 0.2, 0.3]],
        model="openai/text-embedding-3-small",
        provider="Azure OpenAI",
        route="azure/eastus",
        usage={
            "prompt_tokens": 17,
            "total_tokens": 17,
            "cost": 0.0000126,
        },
    )
    gateway, _reader = _gateway(
        credential_id,
        RecordingTransport([response]),
        monotonic_clock=clock.monotonic,
    )
    attempts: list[EmbeddingAttempt] = []

    await gateway.embed(
        snapshot=_snapshot(credential_id, provider_type="openrouter"),
        operational=_operational(),
        inputs=("query",),
        attempt_observer=attempts.append,
    )

    assert attempts == [
        EmbeddingAttempt(
            provider_identifier="Azure OpenAI",
            model_identifier="openai/text-embedding-3-small",
            route_identifier="azure/eastus",
            provider_request_id=None,
            input_tokens=17,
            output_tokens=0,
            cost_micros=13,
            currency="USD",
            latency_ms=0,
            status="succeeded",
            error_code=None,
            degraded=False,
        )
    ]


@pytest.mark.asyncio
async def test_gateway_sanitizes_malicious_or_int64_overflow_response_telemetry() -> None:
    credential_id = uuid4()
    response = _response(
        [[0.1, 0.2, 0.3]],
        model="unsafe\x00model-secret-sentinel",
        provider="p" * 121,
        route={"secret": "route-secret-sentinel"},
        usage={
            "prompt_tokens": 2**63,
            "completion_tokens": -1,
            "total_tokens": 2**80,
            "cost": 10**30,
        },
    )
    gateway, _reader = _gateway(credential_id, RecordingTransport([response]))
    attempts: list[EmbeddingAttempt] = []

    await gateway.embed(
        snapshot=_snapshot(credential_id, provider_type="openrouter"),
        operational=_operational(),
        inputs=("query",),
        attempt_observer=attempts.append,
    )

    attempt = attempts[0]
    assert attempt.provider_identifier == "openrouter"
    assert attempt.model_identifier == "text-embedding-test"
    assert attempt.route_identifier == "openrouter:unknown"
    assert attempt.input_tokens == 0
    assert attempt.output_tokens == 0
    assert attempt.cost_micros == 0
    assert attempt.degraded is True
    assert attempt.error_code == "PROVIDER_TELEMETRY_INVALID"
    rendered = repr(attempt) + str(attempt)
    assert "secret-sentinel" not in rendered


def test_embedding_attempt_rejects_values_outside_postgres_bigint() -> None:
    with pytest.raises(ValueError, match="Embedding attempt is invalid"):
        EmbeddingAttempt(
            provider_identifier="provider",
            model_identifier="model",
            route_identifier="direct",
            provider_request_id=None,
            input_tokens=2**63,
            output_tokens=0,
            cost_micros=0,
            currency="USD",
            latency_ms=0,
            status="succeeded",
            error_code=None,
            degraded=False,
        )


@pytest.mark.asyncio
async def test_gateway_does_not_invoke_observer_when_owned_task_cap_is_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embedding_module = __import__(
        "rag_service.providers.embeddings",
        fromlist=["_MAX_ATTEMPT_OBSERVER_TASKS"],
    )
    monkeypatch.setattr(embedding_module, "_MAX_ATTEMPT_OBSERVER_TASKS", 1)
    credential_id = uuid4()
    gateway, _reader = _gateway(
        credential_id,
        RecordingTransport(
            [
                _response([[0.1, 0.2, 0.3]]),
                _response([[0.1, 0.2, 0.3]]),
            ]
        ),
    )
    release = asyncio.Event()
    first_started = asyncio.Event()
    forbidden_callback_calls = 0
    detached_tasks: list[asyncio.Task[None]] = []

    async def cancellation_resistant_observer(_attempt: EmbeddingAttempt) -> None:
        first_started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()

    def forbidden_task_observer(_attempt: EmbeddingAttempt) -> asyncio.Task[None]:
        nonlocal forbidden_callback_calls
        forbidden_callback_calls += 1
        task = asyncio.create_task(release.wait())
        detached_tasks.append(cast(asyncio.Task[None], task))
        return cast(asyncio.Task[None], task)

    await gateway.embed(
        snapshot=_snapshot(credential_id),
        operational=_operational(),
        inputs=("first",),
        attempt_observer=cancellation_resistant_observer,
    )
    assert first_started.is_set()

    await gateway.embed(
        snapshot=_snapshot(credential_id),
        operational=_operational(),
        inputs=("second",),
        attempt_observer=cast(Any, forbidden_task_observer),
    )

    assert forbidden_callback_calls == 0
    assert detached_tasks == []
    release.set()
    await gateway.aclose()


@pytest.mark.asyncio
async def test_gateway_reports_rate_limit_and_ignores_attempt_observer_failure() -> None:
    credential_id = uuid4()
    gateway, _reader = _gateway(
        credential_id,
        RecordingTransport(
            [ProviderHttpResponse(status_code=429, headers={"request-id": "rate-123"}, body=b"{}")]
        ),
    )
    attempts: list[EmbeddingAttempt] = []

    async def observer(attempt: EmbeddingAttempt) -> None:
        attempts.append(attempt)
        raise RuntimeError("usage-store-secret-sentinel")

    with pytest.raises(EmbeddingGatewayError) as exc_info:
        await gateway.embed(
            snapshot=_snapshot(credential_id),
            operational=_operational(),
            inputs=("query",),
            attempt_observer=observer,
        )

    assert exc_info.value.code == "PROVIDER_RATE_LIMITED"
    assert attempts[0].status == "rate_limited"
    assert attempts[0].error_code == "PROVIDER_RATE_LIMITED"
    assert attempts[0].provider_request_id == "rate-123"


@pytest.mark.asyncio
async def test_synchronous_attempt_observer_cancellation_is_not_swallowed() -> None:
    credential_id = uuid4()
    gateway, _reader = _gateway(
        credential_id,
        RecordingTransport([_response([[0.1, 0.2, 0.3]])]),
    )

    def observer(_attempt: EmbeddingAttempt) -> None:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await gateway.embed(
            snapshot=_snapshot(credential_id),
            operational=_operational(),
            inputs=("query",),
            attempt_observer=observer,
        )


@pytest.mark.asyncio
async def test_hung_attempt_observer_is_bounded_and_does_not_delay_provider_result() -> None:
    credential_id = uuid4()
    gateway, _reader = _gateway(
        credential_id,
        RecordingTransport([_response([[0.1, 0.2, 0.3]])]),
    )
    observer_started = asyncio.Event()
    observer_cancelled = asyncio.Event()

    async def observer(_attempt: EmbeddingAttempt) -> None:
        observer_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            observer_cancelled.set()

    async with asyncio.timeout(1):
        result = await gateway.embed(
            snapshot=_snapshot(credential_id),
            operational=_operational(),
            inputs=("query",),
            attempt_observer=observer,
        )

    assert len(result.vectors) == 1
    assert observer_started.is_set()
    await asyncio.sleep(0)
    assert observer_cancelled.is_set()


@pytest.mark.parametrize(
    ("operational", "expected_code"),
    [
        (_operational(provider_enabled=False), "PROVIDER_DISABLED"),
        (_operational(profile_enabled=False), "MODEL_PROFILE_DISABLED"),
    ],
)
@pytest.mark.asyncio
async def test_gateway_fails_before_credential_or_network_when_current_config_is_disabled(
    operational: EmbeddingOperationalConfig,
    expected_code: str,
) -> None:
    credential_id = uuid4()
    transport = RecordingTransport([])
    gateway, reader = _gateway(credential_id, transport)

    with pytest.raises(EmbeddingGatewayError) as exc_info:
        await gateway.embed(
            snapshot=_snapshot(credential_id),
            operational=operational,
            inputs=("query",),
        )

    assert exc_info.value.code == expected_code
    assert exc_info.value.retryable is False
    assert reader.calls == []
    assert transport.requests == []


@pytest.mark.parametrize("inputs", [(), tuple(str(index) for index in range(9))])
@pytest.mark.asyncio
async def test_gateway_enforces_nonempty_current_batch_limit_before_secret_lookup(
    inputs: tuple[str, ...],
) -> None:
    credential_id = uuid4()
    transport = RecordingTransport([])
    gateway, reader = _gateway(credential_id, transport)

    with pytest.raises(EmbeddingGatewayError) as exc_info:
        await gateway.embed(
            snapshot=_snapshot(credential_id),
            operational=_operational(batch_size=8),
            inputs=inputs,
        )

    assert exc_info.value.code == "EMBEDDING_BATCH_INVALID"
    assert exc_info.value.retryable is False
    assert reader.calls == []
    assert transport.requests == []


@pytest.mark.parametrize("bad_input", ["", "   ", "nul\x00text"])
@pytest.mark.asyncio
async def test_gateway_rejects_invalid_input_without_leaking_it(bad_input: str) -> None:
    credential_id = uuid4()
    transport = RecordingTransport([])
    gateway, reader = _gateway(credential_id, transport)

    with pytest.raises(EmbeddingGatewayError) as exc_info:
        await gateway.embed(
            snapshot=_snapshot(credential_id),
            operational=_operational(),
            inputs=(bad_input,),
        )

    assert exc_info.value.code == "EMBEDDING_INPUT_INVALID"
    if bad_input:
        assert bad_input not in str(exc_info.value)
    assert reader.calls == []


@pytest.mark.parametrize(
    "bad_input",
    [
        "fives",
        "ééé",
        "\ud800",
    ],
)
@pytest.mark.asyncio
async def test_default_input_bound_uses_utf8_bytes_as_non_underestimating_token_upper_bound(
    bad_input: str,
) -> None:
    credential_id = uuid4()
    transport = RecordingTransport([])
    gateway, reader = _gateway(credential_id, transport)

    with pytest.raises(EmbeddingGatewayError) as exc_info:
        await gateway.embed(
            snapshot=_snapshot(credential_id, max_input_tokens=4),
            operational=_operational(),
            inputs=(bad_input,),
        )

    assert exc_info.value.code == "EMBEDDING_INPUT_INVALID"
    assert exc_info.value.retryable is False
    assert bad_input not in str(exc_info.value)
    assert reader.calls == []
    assert transport.requests == []


@pytest.mark.asyncio
async def test_explicit_reliable_token_counter_can_replace_default_byte_upper_bound() -> None:
    credential_id = uuid4()
    counter_calls: list[str] = []

    def token_counter(value: str) -> int:
        counter_calls.append(value)
        return len(value)

    transport = RecordingTransport([_response([[0.1, 0.2, 0.3]])])
    reader = FakeCredentialReader(
        values={credential_id: _encrypted(credential_id, _SECRET)},
        calls=[],
    )
    gateway = EmbeddingGateway(
        keyring=_keyring(),
        credential_reader=reader,
        transport=transport,
        token_counter=token_counter,
    )

    result = await gateway.embed(
        snapshot=_snapshot(credential_id, max_input_tokens=2),
        operational=_operational(),
        inputs=("éé",),
    )

    assert result.vectors == ((0.1, 0.2, 0.3),)
    assert counter_calls == ["éé"]


@pytest.mark.asyncio
async def test_token_counter_failure_is_a_permanent_sanitized_input_error() -> None:
    credential_id = uuid4()
    sentinel = "token-counter-input-secret-sentinel"

    def token_counter(value: str) -> int:
        del value
        raise RuntimeError(sentinel)

    transport = RecordingTransport([])
    reader = FakeCredentialReader(
        values={credential_id: _encrypted(credential_id, _SECRET)},
        calls=[],
    )
    gateway = EmbeddingGateway(
        keyring=_keyring(),
        credential_reader=reader,
        transport=transport,
        token_counter=token_counter,
    )

    with pytest.raises(EmbeddingGatewayError) as exc_info:
        await gateway.embed(
            snapshot=_snapshot(credential_id, max_input_tokens=2),
            operational=_operational(),
            inputs=("query",),
        )

    assert exc_info.value.code == "EMBEDDING_INPUT_INVALID"
    assert exc_info.value.retryable is False
    assert sentinel not in str(exc_info.value) + repr(exc_info.value)
    assert reader.calls == []
    assert transport.requests == []


@pytest.mark.asyncio
async def test_gateway_loads_the_current_rotated_secret_by_stable_snapshot_id_each_call() -> None:
    credential_id = uuid4()
    transport = RecordingTransport([_response([[0.1, 0.2, 0.3]]), _response([[0.4, 0.5, 0.6]])])
    gateway, reader = _gateway(credential_id, transport)
    snapshot = _snapshot(credential_id)

    first = await gateway.embed(
        snapshot=snapshot,
        operational=_operational(),
        inputs=("same input",),
    )
    reader.values[credential_id] = _encrypted(credential_id, _ROTATED_SECRET)
    second = await gateway.embed(
        snapshot=snapshot,
        operational=_operational(),
        inputs=("same input",),
    )

    assert first.vectors != second.vectors
    assert reader.calls == [credential_id, credential_id]
    assert transport.authorization_hashes == [
        hashlib.sha256(f"Bearer {_SECRET}".encode()).hexdigest(),
        hashlib.sha256(f"Bearer {_ROTATED_SECRET}".encode()).hexdigest(),
    ]
    rendered = repr(transport.requests)
    assert _SECRET not in rendered
    assert _ROTATED_SECRET not in rendered


@pytest.mark.asyncio
async def test_gateway_missing_credential_is_a_sanitized_permanent_configuration_error() -> None:
    credential_id = uuid4()
    transport = RecordingTransport([])
    reader = FakeCredentialReader(values={}, calls=[])
    gateway = EmbeddingGateway(
        keyring=_keyring(),
        credential_reader=reader,
        transport=transport,
    )

    with pytest.raises(EmbeddingGatewayError) as exc_info:
        await gateway.embed(
            snapshot=_snapshot(credential_id),
            operational=_operational(),
            inputs=("query",),
        )

    assert exc_info.value.code == "PROVIDER_CREDENTIAL_UNAVAILABLE"
    assert exc_info.value.retryable is False
    assert str(credential_id) not in str(exc_info.value)
    assert transport.requests == []


@pytest.mark.parametrize(
    "invalid_secret",
    [
        "",
        "non-ascii-密钥",
        "space injected secret",
        "line\r\ninjected",
        "control\x7fsecret",
    ],
)
@pytest.mark.asyncio
async def test_gateway_rejects_non_ascii_or_header_injectable_credential_before_transport(
    invalid_secret: str,
) -> None:
    credential_id = uuid4()
    transport = RecordingTransport([_response([[0.1, 0.2, 0.3]])])
    gateway, reader = _gateway(
        credential_id,
        transport,
        secret=invalid_secret,
    )

    with pytest.raises(EmbeddingGatewayError) as exc_info:
        await gateway.embed(
            snapshot=_snapshot(credential_id),
            operational=_operational(),
            inputs=("query",),
        )

    assert exc_info.value.code == "PROVIDER_CREDENTIAL_INVALID"
    assert exc_info.value.retryable is False
    if invalid_secret:
        assert invalid_secret not in str(exc_info.value) + repr(exc_info.value)
    assert reader.calls == [credential_id]
    assert transport.requests == []
    assert transport.authorization_hashes == []
    assert transport.header_references == []
    assert transport.payload_references == []


@pytest.mark.parametrize(
    ("response", "expected_code"),
    [
        (_response([[0.1, 0.2, 0.3]], status_code=401), "PROVIDER_AUTHENTICATION_FAILED"),
        (_response([[0.1, 0.2, 0.3]], status_code=403), "PROVIDER_AUTHENTICATION_FAILED"),
        (_response([[0.1, 0.2, 0.3]], status_code=404), "PROVIDER_MODEL_NOT_FOUND"),
        (_response([[0.1, 0.2, 0.3]], status_code=400), "PROVIDER_INPUT_REJECTED"),
        (_response([[0.1, 0.2, 0.3]], status_code=422), "PROVIDER_INPUT_REJECTED"),
    ],
)
@pytest.mark.asyncio
async def test_permanent_upstream_statuses_have_stable_sanitized_classification(
    response: ProviderHttpResponse,
    expected_code: str,
) -> None:
    credential_id = uuid4()
    response = ProviderHttpResponse(
        status_code=response.status_code,
        headers=response.headers,
        body=b'{"error":{"message":"upstream-secret-sentinel"}}',
    )
    transport = RecordingTransport([response])
    gateway, _reader = _gateway(credential_id, transport)

    with pytest.raises(EmbeddingGatewayError) as exc_info:
        await gateway.embed(
            snapshot=_snapshot(credential_id),
            operational=_operational(),
            inputs=("query",),
        )

    assert exc_info.value.code == expected_code
    assert exc_info.value.retryable is False
    assert "upstream-secret-sentinel" not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.parametrize(
    ("response_or_error", "expected_code"),
    [
        (_response([[0.1, 0.2, 0.3]], status_code=429), "PROVIDER_RATE_LIMITED"),
        (_response([[0.1, 0.2, 0.3]], status_code=500), "PROVIDER_UNAVAILABLE"),
        (_response([[0.1, 0.2, 0.3]], status_code=503), "PROVIDER_UNAVAILABLE"),
        (httpx.ConnectTimeout("connect-timeout-secret-sentinel"), "PROVIDER_TIMEOUT"),
        (httpx.ReadTimeout("read-timeout-secret-sentinel"), "PROVIDER_TIMEOUT"),
        (httpcore.ConnectTimeout("core-connect-timeout-secret-sentinel"), "PROVIDER_TIMEOUT"),
        (httpcore.ReadTimeout("core-read-timeout-secret-sentinel"), "PROVIDER_TIMEOUT"),
        (httpcore.WriteTimeout("core-write-timeout-secret-sentinel"), "PROVIDER_TIMEOUT"),
        (httpcore.PoolTimeout("core-pool-timeout-secret-sentinel"), "PROVIDER_TIMEOUT"),
        (httpcore.TimeoutException("core-timeout-secret-sentinel"), "PROVIDER_TIMEOUT"),
        (httpx.ConnectError("connect-error-secret-sentinel"), "PROVIDER_UNAVAILABLE"),
    ],
)
@pytest.mark.asyncio
async def test_transient_provider_failures_are_retryable_and_sanitized(
    response_or_error: ProviderHttpResponse | BaseException,
    expected_code: str,
) -> None:
    credential_id = uuid4()
    transport = RecordingTransport([response_or_error])
    gateway, _reader = _gateway(credential_id, transport)

    with pytest.raises(EmbeddingGatewayError) as exc_info:
        await gateway.embed(
            snapshot=_snapshot(credential_id),
            operational=_operational(),
            inputs=("query",),
        )

    assert exc_info.value.code == expected_code
    assert exc_info.value.retryable is True
    rendered = str(exc_info.value) + repr(exc_info.value)
    assert "secret-sentinel" not in rendered
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.parametrize(
    ("response", "expected_code"),
    [
        (_response([[0.1, 0.2, 0.3]]), "PROVIDER_RESPONSE_COUNT_MISMATCH"),
        (
            _response(
                [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                indices=[0, 0],
            ),
            "PROVIDER_RESPONSE_COUNT_MISMATCH",
        ),
        (
            _response([[0.1, 0.2], [0.4, 0.5, 0.6]]),
            "PROVIDER_RESPONSE_DIMENSION_MISMATCH",
        ),
        (
            _response([[0.1, math.nan, 0.3], [0.4, 0.5, 0.6]]),
            "PROVIDER_RESPONSE_NONFINITE",
        ),
        (
            _response([[0.1, math.inf, 0.3], [0.4, 0.5, 0.6]]),
            "PROVIDER_RESPONSE_NONFINITE",
        ),
    ],
)
@pytest.mark.asyncio
async def test_invalid_provider_vectors_are_permanent_failures(
    response: ProviderHttpResponse,
    expected_code: str,
) -> None:
    credential_id = uuid4()
    transport = RecordingTransport([response])
    gateway, _reader = _gateway(credential_id, transport)

    with pytest.raises(EmbeddingGatewayError) as exc_info:
        await gateway.embed(
            snapshot=_snapshot(credential_id),
            operational=_operational(),
            inputs=("first", "second"),
        )

    assert exc_info.value.code == expected_code
    assert exc_info.value.retryable is False


@pytest.mark.parametrize(
    "body",
    [
        b"not-json-upstream-secret-sentinel",
        b"{}",
        b'{"data":"wrong-shape"}',
        b'{"data":[{"index":0,"embedding":[true,0.2,0.3]}]}',
    ],
)
@pytest.mark.asyncio
async def test_malformed_success_response_is_sanitized_permanent_failure(body: bytes) -> None:
    credential_id = uuid4()
    transport = RecordingTransport([ProviderHttpResponse(status_code=200, headers={}, body=body)])
    gateway, _reader = _gateway(credential_id, transport)

    with pytest.raises(EmbeddingGatewayError) as exc_info:
        await gateway.embed(
            snapshot=_snapshot(credential_id),
            operational=_operational(),
            inputs=("query",),
        )

    assert exc_info.value.code == "PROVIDER_RESPONSE_INVALID"
    assert exc_info.value.retryable is False
    assert "upstream-secret-sentinel" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_oversized_stream_failure_is_permanent_and_sanitized_at_gateway_boundary() -> None:
    credential_id = uuid4()
    sentinel = "upstream-large-response-secret-sentinel"
    transport = RecordingTransport([ValueError(sentinel)])
    gateway, _reader = _gateway(credential_id, transport)

    with pytest.raises(EmbeddingGatewayError) as exc_info:
        await gateway.embed(
            snapshot=_snapshot(credential_id),
            operational=_operational(),
            inputs=("query",),
        )

    assert exc_info.value.code == "PROVIDER_RESPONSE_INVALID"
    assert exc_info.value.retryable is False
    assert sentinel not in str(exc_info.value) + repr(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        (None, {}),
        ({"prompt_tokens": 3, "total_tokens": 3}, {"prompt_tokens": 3, "total_tokens": 3}),
        (
            {"prompt_tokens": 3, "total_tokens": 3, "completion_tokens": 0},
            {"prompt_tokens": 3, "total_tokens": 3, "completion_tokens": 0},
        ),
        ({"prompt_tokens": "unsafe", "secret": "value"}, {}),
    ],
)
@pytest.mark.asyncio
async def test_usage_extraction_is_bounded_to_safe_nonnegative_integer_fields(
    usage: object | None,
    expected: dict[str, int],
) -> None:
    credential_id = uuid4()
    transport = RecordingTransport([_response([[0.1, 0.2, 0.3]], usage=usage)])
    gateway, _reader = _gateway(credential_id, transport)

    result = await gateway.embed(
        snapshot=_snapshot(credential_id),
        operational=_operational(),
        inputs=("query",),
    )

    assert result.usage == expected
    assert "secret" not in result.usage


@pytest.mark.asyncio
async def test_queued_request_loads_rotated_credential_only_after_provider_admission() -> None:
    credential_id = uuid4()
    provider_config_id = uuid4()
    transport = _AuthorizationBlockingTransport()
    reader = FakeCredentialReader(
        values={credential_id: _encrypted(credential_id, _SECRET)},
        calls=[],
    )
    gateway = EmbeddingGateway(
        keyring=_keyring(),
        credential_reader=reader,
        transport=transport,
    )
    operational = _provider_operational(
        provider_config_id,
        max_concurrency=1,
        requests_per_minute=1_000_000,
    )
    first = asyncio.create_task(
        gateway.embed(
            snapshot=_snapshot(credential_id),
            operational=operational,
            inputs=("first",),
        )
    )
    second: asyncio.Task[EmbeddingResult] | None = None
    try:
        async with asyncio.timeout(1):
            assert await transport.entered.get() == 1
        second = asyncio.create_task(
            gateway.embed(
                snapshot=_snapshot(credential_id),
                operational=operational,
                inputs=("second",),
            )
        )
        await _drain_ready_tasks()

        assert reader.calls == [credential_id]
        reader.values[credential_id] = _encrypted(credential_id, _ROTATED_SECRET)
        transport.release.set()
        await asyncio.gather(first, second)

        assert transport.authorization_hashes == [
            hashlib.sha256(f"Bearer {_SECRET}".encode()).hexdigest(),
            hashlib.sha256(f"Bearer {_ROTATED_SECRET}".encode()).hexdigest(),
        ]
    finally:
        transport.release.set()
        for task in (first, second):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *(task for task in (first, second) if task is not None),
            return_exceptions=True,
        )


@pytest.mark.asyncio
async def test_cancelled_credential_read_refunds_admission_before_next_waiter() -> None:
    credential_id = uuid4()
    provider_config_id = uuid4()
    encrypted = _encrypted(credential_id, _SECRET)
    reader = _AdmissionGateCredentialReader(encrypted)
    transport = _BlockingTransport()
    clock = _ManualClock()
    gateway = EmbeddingGateway(
        keyring=_keyring(),
        credential_reader=reader,
        transport=transport,
        monotonic_clock=clock.monotonic,
        sleeper=clock.sleep,
    )
    operational = _provider_operational(
        provider_config_id,
        max_concurrency=1,
        requests_per_minute=1,
    )
    first = asyncio.create_task(
        gateway.embed(
            snapshot=_snapshot(credential_id),
            operational=operational,
            inputs=("first",),
        )
    )
    second: asyncio.Task[EmbeddingResult] | None = None
    try:
        async with asyncio.timeout(1):
            await reader.first_read_started.wait()
        second = asyncio.create_task(
            gateway.embed(
                snapshot=_snapshot(credential_id),
                operational=operational,
                inputs=("second",),
            )
        )
        await _drain_ready_tasks()

        assert reader.calls == [credential_id]
        assert transport.entered.empty()

        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        async with asyncio.timeout(1):
            assert await transport.entered.get() == 1
        assert reader.calls == [credential_id, credential_id]
        assert clock.sleep_calls == []
    finally:
        reader.release_first_read.set()
        transport.release.set()
        for task in (first, second):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *(task for task in (first, second) if task is not None),
            return_exceptions=True,
        )


@pytest.mark.asyncio
async def test_cancelled_credential_read_refund_interrupts_fifo_head_rate_sleep() -> None:
    credential_id = uuid4()
    provider_config_id = uuid4()
    reader = _AdmissionGateCredentialReader(_encrypted(credential_id, _SECRET))
    transport = _BlockingTransport()
    clock = _ControlledClock()
    gateway = EmbeddingGateway(
        keyring=_keyring(),
        credential_reader=reader,
        transport=transport,
        monotonic_clock=clock.monotonic,
        sleeper=clock.sleep,
    )
    operational = _provider_operational(
        provider_config_id,
        max_concurrency=2,
        requests_per_minute=1,
    )
    first = asyncio.create_task(
        gateway.embed(
            snapshot=_snapshot(credential_id),
            operational=operational,
            inputs=("first",),
        )
    )
    second: asyncio.Task[EmbeddingResult] | None = None
    try:
        async with asyncio.timeout(1):
            await reader.first_read_started.wait()
        second = asyncio.create_task(
            gateway.embed(
                snapshot=_snapshot(credential_id),
                operational=operational,
                inputs=("second",),
            )
        )
        async with asyncio.timeout(1):
            rate_sleep = await clock.started.get()
        assert rate_sleep.delay == pytest.approx(60.0)

        first.cancel()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        async with asyncio.timeout(1):
            assert await transport.entered.get() == 1
            await rate_sleep.finished.wait()
        assert rate_sleep.release.is_set() is False
        assert rate_sleep.cancelled is True
        assert reader.calls == [credential_id, credential_id]

        transport.release.set()
        await second
    finally:
        reader.release_first_read.set()
        clock.release_all()
        transport.release.set()
        for task in (first, second):
            if task is not None and not task.done():
                task.cancel()
                task.cancel()
        await asyncio.gather(
            *(task for task in (first, second) if task is not None),
            return_exceptions=True,
        )


@pytest.mark.asyncio
async def test_repeated_cancel_of_fifo_head_rate_wait_cleans_both_internal_tasks() -> None:
    credential_id = uuid4()
    provider_config_id = uuid4()
    reader = _AdmissionGateCredentialReader(_encrypted(credential_id, _SECRET))
    clock = _ControlledClock()
    gateway = EmbeddingGateway(
        keyring=_keyring(),
        credential_reader=reader,
        transport=_BlockingTransport(),
        monotonic_clock=clock.monotonic,
        sleeper=clock.sleep,
    )
    operational = _provider_operational(
        provider_config_id,
        max_concurrency=2,
        requests_per_minute=1,
    )
    first = asyncio.create_task(
        gateway.embed(
            snapshot=_snapshot(credential_id),
            operational=operational,
            inputs=("first",),
        )
    )
    second: asyncio.Task[EmbeddingResult] | None = None
    try:
        async with asyncio.timeout(1):
            await reader.first_read_started.wait()
        second = asyncio.create_task(
            gateway.embed(
                snapshot=_snapshot(credential_id),
                operational=operational,
                inputs=("second",),
            )
        )
        async with asyncio.timeout(1):
            rate_sleep = await clock.started.get()

        second.cancel()
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        async with asyncio.timeout(1):
            await rate_sleep.finished.wait()
        await _drain_ready_tasks()

        assert rate_sleep.cancelled is True
        assert rate_sleep.release.is_set() is False
        orphaned = [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
            and not task.done()
            and task.get_name() in {"rag-embedding-rate-deadline", "rag-embedding-admission-wake"}
        ]
        assert orphaned == []
    finally:
        reader.release_first_read.set()
        clock.release_all()
        for task in (first, second):
            if task is not None and not task.done():
                task.cancel()
                task.cancel()
        await asyncio.gather(
            *(task for task in (first, second) if task is not None),
            return_exceptions=True,
        )


@pytest.mark.asyncio
async def test_double_cancel_during_unadmitted_ticket_cleanup_still_removes_ticket() -> None:
    credential_id = uuid4()
    provider_config_id = uuid4()
    reader = _AdmissionGateCredentialReader(_encrypted(credential_id, _SECRET))
    transport = _BlockingTransport()
    clock = _ControlledClock()
    condition_factory = _CleanupBlockingConditionFactory()
    gateway = EmbeddingGateway(
        keyring=_keyring(),
        credential_reader=reader,
        transport=transport,
        monotonic_clock=clock.monotonic,
        sleeper=clock.sleep,
        condition_factory=condition_factory,
    )
    operational = _provider_operational(
        provider_config_id,
        max_concurrency=2,
        requests_per_minute=1,
    )
    first = asyncio.create_task(
        gateway.embed(
            snapshot=_snapshot(credential_id),
            operational=operational,
            inputs=("first",),
        )
    )
    second: asyncio.Task[EmbeddingResult] | None = None
    third: asyncio.Task[EmbeddingResult] | None = None
    state_condition: _CleanupBlockingCondition | None = None
    try:
        async with asyncio.timeout(1):
            await reader.first_read_started.wait()
        second = asyncio.create_task(
            gateway.embed(
                snapshot=_snapshot(credential_id),
                operational=operational,
                inputs=("second",),
            )
        )
        async with asyncio.timeout(1):
            await clock.started.get()
        state = gateway._limit_states[provider_config_id]
        assert isinstance(state.condition, _CleanupBlockingCondition)
        state_condition = state.condition
        state_condition.block_next_context_entry()

        second.cancel()
        async with asyncio.timeout(1):
            await state_condition.context_entry_started.wait()
        second.cancel()
        state_condition.allow_context_entry.set()
        with pytest.raises(asyncio.CancelledError):
            await second

        assert list(state.queue) == []
        assert state.waiters == 0

        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        third = asyncio.create_task(
            gateway.embed(
                snapshot=_snapshot(credential_id),
                operational=operational,
                inputs=("third",),
            )
        )
        async with asyncio.timeout(1):
            assert await transport.entered.get() == 1
        transport.release.set()
        await third
    finally:
        if state_condition is not None:
            state_condition.allow_context_entry.set()
        reader.release_first_read.set()
        clock.release_all()
        transport.release.set()
        for task in (first, second, third):
            if task is not None and not task.done():
                task.cancel()
                task.cancel()
        await asyncio.gather(
            *(task for task in (first, second, third) if task is not None),
            return_exceptions=True,
        )


@pytest.mark.parametrize("first_credential", ["missing", "decrypt_failure"])
@pytest.mark.asyncio
async def test_pre_attempt_credential_failure_refund_interrupts_fifo_head_rate_sleep(
    first_credential: str,
) -> None:
    credential_id = uuid4()
    provider_config_id = uuid4()
    valid = _encrypted(credential_id, _SECRET)
    invalid = EncryptedProviderCredential(
        ciphertext=b"x" * 16,
        nonce=b"n" * 12,
        key_version="v1",
    )
    reader = _GatedSequencedCredentialReader(
        None if first_credential == "missing" else invalid,
        valid,
    )
    transport = _BlockingTransport()
    clock = _ControlledClock()
    gateway = EmbeddingGateway(
        keyring=_keyring(),
        credential_reader=reader,
        transport=transport,
        monotonic_clock=clock.monotonic,
        sleeper=clock.sleep,
    )
    operational = _provider_operational(
        provider_config_id,
        max_concurrency=2,
        requests_per_minute=1,
    )
    first = asyncio.create_task(
        gateway.embed(
            snapshot=_snapshot(credential_id),
            operational=operational,
            inputs=("first",),
        )
    )
    second: asyncio.Task[EmbeddingResult] | None = None
    try:
        async with asyncio.timeout(1):
            await reader.first_read_started.wait()
        second = asyncio.create_task(
            gateway.embed(
                snapshot=_snapshot(credential_id),
                operational=operational,
                inputs=("second",),
            )
        )
        async with asyncio.timeout(1):
            rate_sleep = await clock.started.get()
        assert rate_sleep.delay == pytest.approx(60.0)

        reader.release_first_read.set()
        with pytest.raises(EmbeddingGatewayError) as exc_info:
            await first
        assert exc_info.value.code == "PROVIDER_CREDENTIAL_UNAVAILABLE"

        async with asyncio.timeout(1):
            assert await transport.entered.get() == 1
            await rate_sleep.finished.wait()
        assert rate_sleep.release.is_set() is False
        assert rate_sleep.cancelled is True
        assert reader.calls == [credential_id, credential_id]

        transport.release.set()
        await second
    finally:
        reader.release_first_read.set()
        clock.release_all()
        transport.release.set()
        for task in (first, second):
            if task is not None and not task.done():
                task.cancel()
                task.cancel()
        await asyncio.gather(
            *(task for task in (first, second) if task is not None),
            return_exceptions=True,
        )


@pytest.mark.asyncio
async def test_provider_admission_is_fifo_under_contention_and_waiter_cancellation() -> None:
    credential_id = uuid4()
    provider_config_id = uuid4()
    transport = _FifoGateTransport()
    reader = FakeCredentialReader(
        values={credential_id: _encrypted(credential_id, _SECRET)},
        calls=[],
    )
    gateway = EmbeddingGateway(
        keyring=_keyring(),
        credential_reader=reader,
        transport=transport,
    )
    operational = _provider_operational(
        provider_config_id,
        max_concurrency=1,
        requests_per_minute=1_000_000,
    )
    tasks: list[asyncio.Task[EmbeddingResult]] = []
    try:
        for index in range(12):
            tasks.append(
                asyncio.create_task(
                    gateway.embed(
                        snapshot=_snapshot(credential_id),
                        operational=operational,
                        inputs=(str(index),),
                    )
                )
            )
            await _drain_ready_tasks()

        async with asyncio.timeout(1):
            assert await transport.entered.get() == "0"
        for index in (3, 7, 9):
            tasks[index].cancel()
        expected = [str(index) for index in range(12) if index not in {3, 7, 9}]
        for value in expected[1:]:
            await transport.release_next()
            async with asyncio.timeout(1):
                assert await transport.entered.get() == value
        await transport.release_next()
        await asyncio.gather(*tasks, return_exceptions=True)

        assert transport.entry_order == expected
    finally:
        await transport.aclose()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_fifo_admission_atomically_reserves_concurrency_and_rate_token() -> None:
    credential_id = uuid4()
    provider_config_id = uuid4()
    clock = _ControlledClock()
    transport = _FifoGateTransport()
    reader = FakeCredentialReader(
        values={credential_id: _encrypted(credential_id, _SECRET)},
        calls=[],
    )
    gateway = EmbeddingGateway(
        keyring=_keyring(),
        credential_reader=reader,
        transport=transport,
        monotonic_clock=clock.monotonic,
        sleeper=clock.sleep,
    )
    first = asyncio.create_task(
        gateway.embed(
            snapshot=_snapshot(credential_id),
            operational=_provider_operational(
                provider_config_id,
                max_concurrency=3,
                requests_per_minute=1,
            ),
            inputs=("first",),
        )
    )
    second: asyncio.Task[EmbeddingResult] | None = None
    third: asyncio.Task[EmbeddingResult] | None = None
    try:
        async with asyncio.timeout(1):
            assert await transport.entered.get() == "first"
        second = asyncio.create_task(
            gateway.embed(
                snapshot=_snapshot(credential_id),
                operational=_provider_operational(
                    provider_config_id,
                    max_concurrency=3,
                    requests_per_minute=1,
                ),
                inputs=("second",),
            )
        )
        async with asyncio.timeout(1):
            second_wait = await clock.started.get()
        assert second_wait.delay == pytest.approx(60.0)

        third = asyncio.create_task(
            gateway.embed(
                snapshot=_snapshot(credential_id),
                operational=_provider_operational(
                    provider_config_id,
                    max_concurrency=3,
                    requests_per_minute=1_000_000,
                ),
                inputs=("third",),
            )
        )
        await _drain_ready_tasks()

        clock.release_shortest()
        async with asyncio.timeout(1):
            assert await transport.entered.get() == "second"
        async with asyncio.timeout(1):
            await clock.started.get()
        clock.release_shortest()
        async with asyncio.timeout(1):
            assert await transport.entered.get() == "third"

        assert transport.entry_order == ["first", "second", "third"]
    finally:
        clock.release_all()
        await transport.aclose()
        for task in (first, second, third):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *(task for task in (first, second, third) if task is not None),
            return_exceptions=True,
        )


@pytest.mark.asyncio
async def test_shared_gateway_enforces_current_concurrency_by_stable_provider_config_id() -> None:
    credential_id = uuid4()
    provider_config_id = uuid4()
    transport = _BlockingTransport()
    reader = FakeCredentialReader(
        values={credential_id: _encrypted(credential_id, _SECRET)},
        calls=[],
    )
    gateway = EmbeddingGateway(
        keyring=_keyring(),
        credential_reader=reader,
        transport=transport,
    )
    operational = _provider_operational(
        provider_config_id,
        max_concurrency=1,
        requests_per_minute=1_000_000,
    )
    first: asyncio.Task[EmbeddingResult] | None = None
    second: asyncio.Task[EmbeddingResult] | None = None
    try:
        first = asyncio.create_task(
            gateway.embed(
                snapshot=_snapshot(credential_id),
                operational=operational,
                inputs=("first",),
            )
        )
        async with asyncio.timeout(1):
            assert await transport.entered.get() == 1
        second = asyncio.create_task(
            gateway.embed(
                snapshot=_snapshot(credential_id),
                operational=operational,
                inputs=("second",),
            )
        )
        await _drain_ready_tasks()

        assert transport.entered.empty()
        assert transport.max_active == 1
    finally:
        transport.release.set()
        for task in (first, second):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *(task for task in (first, second) if task is not None),
            return_exceptions=True,
        )


@pytest.mark.asyncio
async def test_current_higher_concurrency_limit_applies_to_new_requests() -> None:
    credential_id = uuid4()
    provider_config_id = uuid4()
    transport = _BlockingTransport()
    reader = FakeCredentialReader(
        values={credential_id: _encrypted(credential_id, _SECRET)},
        calls=[],
    )
    gateway = EmbeddingGateway(
        keyring=_keyring(),
        credential_reader=reader,
        transport=transport,
    )
    first: asyncio.Task[EmbeddingResult] | None = None
    second: asyncio.Task[EmbeddingResult] | None = None
    try:
        first = asyncio.create_task(
            gateway.embed(
                snapshot=_snapshot(credential_id),
                operational=_provider_operational(
                    provider_config_id,
                    max_concurrency=1,
                    requests_per_minute=1_000_000,
                ),
                inputs=("first",),
            )
        )
        async with asyncio.timeout(1):
            assert await transport.entered.get() == 1
        second = asyncio.create_task(
            gateway.embed(
                snapshot=_snapshot(credential_id),
                operational=_provider_operational(
                    provider_config_id,
                    max_concurrency=2,
                    requests_per_minute=1_000_000,
                ),
                inputs=("second",),
            )
        )

        async with asyncio.timeout(1):
            assert await transport.entered.get() == 2
        assert transport.max_active == 2
    finally:
        transport.release.set()
        await asyncio.gather(
            *(task for task in (first, second) if task is not None),
            return_exceptions=True,
        )


@pytest.mark.asyncio
async def test_cancellation_releases_provider_concurrency_slot_for_waiter() -> None:
    credential_id = uuid4()
    provider_config_id = uuid4()
    transport = _BlockingTransport()
    reader = FakeCredentialReader(
        values={credential_id: _encrypted(credential_id, _SECRET)},
        calls=[],
    )
    gateway = EmbeddingGateway(
        keyring=_keyring(),
        credential_reader=reader,
        transport=transport,
    )
    operational = _provider_operational(
        provider_config_id,
        max_concurrency=1,
        requests_per_minute=1_000_000,
    )
    first = asyncio.create_task(
        gateway.embed(
            snapshot=_snapshot(credential_id),
            operational=operational,
            inputs=("first",),
        )
    )
    second: asyncio.Task[EmbeddingResult] | None = None
    try:
        async with asyncio.timeout(1):
            assert await transport.entered.get() == 1
        second = asyncio.create_task(
            gateway.embed(
                snapshot=_snapshot(credential_id),
                operational=operational,
                inputs=("second",),
            )
        )
        await _drain_ready_tasks()
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first

        async with asyncio.timeout(1):
            assert await transport.entered.get() == 2
        assert transport.max_active == 1
    finally:
        transport.release.set()
        if not first.done():
            first.cancel()
        if second is not None and not second.done():
            await second


@pytest.mark.asyncio
async def test_cancelling_provider_concurrency_waiter_does_not_leak_waiter_state() -> None:
    credential_id = uuid4()
    provider_config_id = uuid4()
    transport = _BlockingTransport()
    reader = FakeCredentialReader(
        values={credential_id: _encrypted(credential_id, _SECRET)},
        calls=[],
    )
    gateway = EmbeddingGateway(
        keyring=_keyring(),
        credential_reader=reader,
        transport=transport,
    )
    operational = _provider_operational(
        provider_config_id,
        max_concurrency=1,
        requests_per_minute=1_000_000,
    )
    first = asyncio.create_task(
        gateway.embed(
            snapshot=_snapshot(credential_id),
            operational=operational,
            inputs=("first",),
        )
    )
    second: asyncio.Task[EmbeddingResult] | None = None
    third: asyncio.Task[EmbeddingResult] | None = None
    try:
        async with asyncio.timeout(1):
            assert await transport.entered.get() == 1
        second = asyncio.create_task(
            gateway.embed(
                snapshot=_snapshot(credential_id),
                operational=operational,
                inputs=("second",),
            )
        )
        await _drain_ready_tasks()
        assert transport.entered.empty()
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        transport.release.set()
        await first

        third = asyncio.create_task(
            gateway.embed(
                snapshot=_snapshot(credential_id),
                operational=operational,
                inputs=("third",),
            )
        )
        async with asyncio.timeout(1):
            assert await transport.entered.get() == 2
        await third
        assert transport.calls == 2
    finally:
        transport.release.set()
        for task in (first, second, third):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *(task for task in (first, second, third) if task is not None),
            return_exceptions=True,
        )


@pytest.mark.asyncio
async def test_repeated_cancellation_cannot_interrupt_concurrency_slot_cleanup() -> None:
    credential_id = uuid4()
    provider_config_id = uuid4()
    transport = _BlockingTransport()
    reader = FakeCredentialReader(
        values={credential_id: _encrypted(credential_id, _SECRET)},
        calls=[],
    )
    gateway = _CleanupGateEmbeddingGateway(
        keyring=_keyring(),
        credential_reader=reader,
        transport=transport,
    )
    operational = _provider_operational(
        provider_config_id,
        max_concurrency=1,
        requests_per_minute=1_000_000,
    )
    state = await gateway._limit_state(provider_config_id)
    first = asyncio.create_task(
        gateway.embed(
            snapshot=_snapshot(credential_id),
            operational=operational,
            inputs=("first",),
        )
    )
    second: asyncio.Task[EmbeddingResult] | None = None
    try:
        async with asyncio.timeout(1):
            assert await transport.entered.get() == 1
        second = asyncio.create_task(
            gateway.embed(
                snapshot=_snapshot(credential_id),
                operational=operational,
                inputs=("second",),
            )
        )
        await _drain_ready_tasks()
        assert state.active == 1
        assert state.waiters == 1

        first.cancel()
        async with asyncio.timeout(1):
            await gateway.cleanup_started.wait()
        first.cancel()
        gateway.allow_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await first

        async with asyncio.timeout(1):
            assert await transport.entered.get() == 2
        assert state.active == 1
        assert state.waiters == 0

        transport.release.set()
        await second
        assert state.active == 0
        assert state.waiters == 0
        await _drain_ready_tasks()
        orphaned = [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task() and not task.done()
        ]
        assert orphaned == []
    finally:
        gateway.allow_cleanup.set()
        transport.release.set()
        for task in (first, second):
            if task is not None and not task.done():
                task.cancel()
        await asyncio.gather(
            *(task for task in (first, second) if task is not None),
            return_exceptions=True,
        )


@pytest.mark.asyncio
async def test_shared_gateway_enforces_rpm_per_provider_with_injected_clock_and_sleeper() -> None:
    credential_id = uuid4()
    provider_config_id = uuid4()
    clock = _ManualClock()
    transport = _ClockRecordingTransport(
        [_response([[0.1, 0.2, 0.3]]) for _ in range(3)],
        clock,
    )
    reader = FakeCredentialReader(
        values={credential_id: _encrypted(credential_id, _SECRET)},
        calls=[],
    )
    gateway = EmbeddingGateway(
        keyring=_keyring(),
        credential_reader=reader,
        transport=transport,
        monotonic_clock=clock.monotonic,
        sleeper=clock.sleep,
    )
    operational = _provider_operational(
        provider_config_id,
        max_concurrency=4,
        requests_per_minute=2,
    )

    for index in range(3):
        await gateway.embed(
            snapshot=_snapshot(credential_id),
            operational=operational,
            inputs=(f"query-{index}",),
        )

    assert transport.request_times == sorted(transport.request_times)
    assert transport.request_times[2] >= 30.0
    assert sum(clock.sleep_calls) >= 30.0


@pytest.mark.asyncio
async def test_rate_refill_clamps_monotonic_clock_rollback() -> None:
    credential_id = uuid4()
    provider_config_id = uuid4()
    clock = _ControlledClock(now=100.0)
    transport = RecordingTransport([_response([[0.1, 0.2, 0.3]]), _response([[0.1, 0.2, 0.3]])])
    reader = FakeCredentialReader(
        values={credential_id: _encrypted(credential_id, _SECRET)},
        calls=[],
    )
    gateway = EmbeddingGateway(
        keyring=_keyring(),
        credential_reader=reader,
        transport=transport,
        monotonic_clock=clock.monotonic,
        sleeper=clock.sleep,
    )
    operational = _provider_operational(
        provider_config_id,
        max_concurrency=2,
        requests_per_minute=1,
    )
    await gateway.embed(
        snapshot=_snapshot(credential_id),
        operational=operational,
        inputs=("first",),
    )

    clock.now = 60.0
    second = asyncio.create_task(
        gateway.embed(
            snapshot=_snapshot(credential_id),
            operational=operational,
            inputs=("second",),
        )
    )
    try:
        async with asyncio.timeout(1):
            rollback_wait = await clock.started.get()
        assert rollback_wait.delay == pytest.approx(60.0)

        clock.now = 100.0
        rollback_wait.release.set()
        async with asyncio.timeout(1):
            retry_wait = await clock.started.get()

        assert retry_wait.delay == pytest.approx(60.0)
        assert len(transport.requests) == 1
    finally:
        clock.release_all()
        if not second.done():
            second.cancel()
        await asyncio.gather(second, return_exceptions=True)


@pytest.mark.asyncio
async def test_current_higher_rpm_limit_applies_to_new_request_for_same_provider() -> None:
    credential_id = uuid4()
    provider_config_id = uuid4()
    clock = _ManualClock()
    transport = _ClockRecordingTransport(
        [_response([[0.1, 0.2, 0.3]]) for _ in range(2)],
        clock,
    )
    reader = FakeCredentialReader(
        values={credential_id: _encrypted(credential_id, _SECRET)},
        calls=[],
    )
    gateway = EmbeddingGateway(
        keyring=_keyring(),
        credential_reader=reader,
        transport=transport,
        monotonic_clock=clock.monotonic,
        sleeper=clock.sleep,
    )

    await gateway.embed(
        snapshot=_snapshot(credential_id),
        operational=_provider_operational(provider_config_id, requests_per_minute=1),
        inputs=("first",),
    )
    await gateway.embed(
        snapshot=_snapshot(credential_id),
        operational=_provider_operational(provider_config_id, requests_per_minute=2),
        inputs=("second",),
    )

    assert transport.request_times[1] <= 30.0
    assert sum(clock.sleep_calls) <= 30.0


@pytest.mark.asyncio
async def test_rpm_state_is_isolated_by_stable_provider_config_id() -> None:
    credential_id = uuid4()
    first_provider_id = uuid4()
    second_provider_id = uuid4()
    clock = _ManualClock()
    transport = _ClockRecordingTransport(
        [_response([[0.1, 0.2, 0.3]]) for _ in range(2)],
        clock,
    )
    reader = FakeCredentialReader(
        values={credential_id: _encrypted(credential_id, _SECRET)},
        calls=[],
    )
    gateway = EmbeddingGateway(
        keyring=_keyring(),
        credential_reader=reader,
        transport=transport,
        monotonic_clock=clock.monotonic,
        sleeper=clock.sleep,
    )

    await gateway.embed(
        snapshot=_snapshot(credential_id),
        operational=_provider_operational(first_provider_id, requests_per_minute=1),
        inputs=("first",),
    )
    await gateway.embed(
        snapshot=_snapshot(credential_id),
        operational=_provider_operational(second_provider_id, requests_per_minute=1),
        inputs=("second",),
    )

    assert transport.request_times == [0.0, 0.0]
    assert clock.sleep_calls == []


@pytest.mark.asyncio
async def test_rate_limit_bookkeeping_is_constant_memory_at_high_rpm() -> None:
    credential_id = uuid4()
    provider_config_id = uuid4()
    clock = _ManualClock()
    request_count = 256
    transport = _ClockRecordingTransport(
        [_response([[0.1, 0.2, 0.3]]) for _ in range(request_count)],
        clock,
    )
    reader = FakeCredentialReader(
        values={credential_id: _encrypted(credential_id, _SECRET)},
        calls=[],
    )
    gateway = EmbeddingGateway(
        keyring=_keyring(),
        credential_reader=reader,
        transport=transport,
        monotonic_clock=clock.monotonic,
        sleeper=clock.sleep,
    )
    operational = _provider_operational(
        provider_config_id,
        max_concurrency=8,
        requests_per_minute=1_000_000,
    )

    for index in range(request_count):
        await gateway.embed(
            snapshot=_snapshot(credential_id),
            operational=operational,
            inputs=(f"query-{index}",),
        )

    state = gateway._limit_states[provider_config_id]
    assert not hasattr(state, "request_times")


@pytest.mark.asyncio
async def test_registry_refund_interrupts_full_capacity_reclaim_deadline() -> None:
    credential_id = uuid4()
    first_provider_id = uuid4()
    second_provider_id = uuid4()
    clock = _ControlledClock()
    transport = RecordingTransport([_response([[0.1, 0.2, 0.3]]), _response([[0.4, 0.5, 0.6]])])
    reader = FakeCredentialReader(
        values={credential_id: _encrypted(credential_id, _SECRET)},
        calls=[],
    )
    gateway = EmbeddingGateway(
        keyring=_keyring(),
        credential_reader=reader,
        transport=transport,
        monotonic_clock=clock.monotonic,
        sleeper=clock.sleep,
        max_provider_states=1,
    )
    await gateway.embed(
        snapshot=_snapshot(credential_id),
        operational=_provider_operational(
            first_provider_id,
            requests_per_minute=1,
        ),
        inputs=("first",),
    )
    old_state = gateway._limit_states[first_provider_id]
    second = asyncio.create_task(
        gateway.embed(
            snapshot=_snapshot(credential_id),
            operational=_provider_operational(
                second_provider_id,
                requests_per_minute=1,
            ),
            inputs=("second",),
        )
    )
    try:
        async with asyncio.timeout(1):
            reclaim_sleep = await clock.started.get()
        assert reclaim_sleep.delay == pytest.approx(60.0)

        await gateway._refund_rate_limit(old_state)

        async with asyncio.timeout(1):
            result = await second
            await reclaim_sleep.finished.wait()
        assert result.vectors == ((0.4, 0.5, 0.6),)
        assert reclaim_sleep.release.is_set() is False
        assert reclaim_sleep.cancelled is True
        assert list(gateway._limit_states) == [second_provider_id]
    finally:
        clock.release_all()
        if not second.done():
            second.cancel()
            second.cancel()
        await asyncio.gather(second, return_exceptions=True)


@pytest.mark.asyncio
async def test_repeated_cancel_of_registry_reclaim_wait_cleans_internal_tasks() -> None:
    credential_id = uuid4()
    first_provider_id = uuid4()
    clock = _ControlledClock()
    transport = RecordingTransport([_response([[0.1, 0.2, 0.3]])])
    reader = FakeCredentialReader(
        values={credential_id: _encrypted(credential_id, _SECRET)},
        calls=[],
    )
    gateway = EmbeddingGateway(
        keyring=_keyring(),
        credential_reader=reader,
        transport=transport,
        monotonic_clock=clock.monotonic,
        sleeper=clock.sleep,
        max_provider_states=1,
    )
    await gateway.embed(
        snapshot=_snapshot(credential_id),
        operational=_provider_operational(
            first_provider_id,
            requests_per_minute=1,
        ),
        inputs=("first",),
    )
    current: asyncio.Task[EmbeddingResult] | None = None
    try:
        for index in range(16):
            current = asyncio.create_task(
                gateway.embed(
                    snapshot=_snapshot(credential_id),
                    operational=_provider_operational(
                        uuid4(),
                        requests_per_minute=1,
                    ),
                    inputs=(f"cancel-{index}",),
                )
            )
            async with asyncio.timeout(1):
                reclaim_sleep = await clock.started.get()

            current.cancel()
            current.cancel()
            with pytest.raises(asyncio.CancelledError):
                await current
            async with asyncio.timeout(1):
                await reclaim_sleep.finished.wait()

            assert reclaim_sleep.cancelled is True
            assert reclaim_sleep.release.is_set() is False
        await _drain_ready_tasks()

        assert list(gateway._limit_states) == [first_provider_id]
        orphaned = [
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
            and not task.done()
            and task.get_name()
            in {"rag-embedding-registry-deadline", "rag-embedding-registry-wake"}
        ]
        assert orphaned == []
    finally:
        clock.release_all()
        if current is not None and not current.done():
            current.cancel()
            current.cancel()
        if current is not None:
            await asyncio.gather(current, return_exceptions=True)


@pytest.mark.asyncio
async def test_provider_limit_state_registry_is_bounded_without_resetting_live_quota() -> None:
    assert "max_provider_states" in inspect.signature(EmbeddingGateway).parameters

    credential_id = uuid4()
    protected_provider_id = uuid4()
    clock = _ManualClock()
    transient_provider_ids = [uuid4() for _ in range(24)]
    transport = _ClockRecordingTransport(
        [_response([[0.1, 0.2, 0.3]]) for _ in range(26)],
        clock,
    )
    reader = FakeCredentialReader(
        values={credential_id: _encrypted(credential_id, _SECRET)},
        calls=[],
    )
    gateway = EmbeddingGateway(
        keyring=_keyring(),
        credential_reader=reader,
        transport=transport,
        monotonic_clock=clock.monotonic,
        sleeper=clock.sleep,
        max_provider_states=4,
    )

    await gateway.embed(
        snapshot=_snapshot(credential_id),
        operational=_provider_operational(
            protected_provider_id,
            requests_per_minute=1,
        ),
        inputs=("protected-first",),
    )
    for provider_config_id in transient_provider_ids:
        clock.now += 0.001
        await gateway.embed(
            snapshot=_snapshot(credential_id),
            operational=_provider_operational(
                provider_config_id,
                requests_per_minute=1_000_000,
            ),
            inputs=("transient",),
        )
        assert len(gateway._limit_states) <= 4

    assert protected_provider_id in gateway._limit_states
    before = clock.now
    await gateway.embed(
        snapshot=_snapshot(credential_id),
        operational=_provider_operational(
            protected_provider_id,
            requests_per_minute=1,
        ),
        inputs=("protected-second",),
    )

    assert clock.now - before >= 59.0
    assert len(gateway._limit_states) <= 4


@pytest.mark.asyncio
async def test_cancellation_propagates_and_clears_plaintext_header_and_payload_containers() -> None:
    credential_id = uuid4()
    started = asyncio.Event()
    release = asyncio.Event()

    class HangingTransport(RecordingTransport):
        async def post_json(
            self,
            *,
            endpoint: object,
            path: str,
            headers: dict[str, str],
            payload: dict[str, object],
            timeout_seconds: float,
        ) -> ProviderHttpResponse:
            self.header_references.append(headers)
            self.payload_references.append(payload)
            started.set()
            await release.wait()
            raise AssertionError("unreachable")

    transport = HangingTransport([])
    gateway, _reader = _gateway(credential_id, transport)
    task = asyncio.create_task(
        gateway.embed(
            snapshot=_snapshot(credential_id),
            operational=_operational(),
            inputs=("query",),
        )
    )
    await started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert transport.header_references == [{}]
    assert transport.payload_references == [{}]
    retained = repr(transport.__dict__)
    assert _SECRET not in retained
    assert f"Bearer {_SECRET}" not in retained


def test_snapshot_and_runtime_models_are_immutable_and_have_safe_representations() -> None:
    credential_id = uuid4()
    snapshot = _snapshot(credential_id)
    operational = _operational()

    with pytest.raises(FrozenInstanceError):
        cast(Any, snapshot).base_url = "https://other.example/v1"
    with pytest.raises(FrozenInstanceError):
        cast(Any, operational).timeout_seconds = Decimal("100")

    for forbidden in (
        str(credential_id),
        "provider.example",
        "text-embedding-test",
    ):
        assert forbidden not in repr(snapshot)
    assert "EmbeddingOperationalConfig" in repr(operational)


@pytest.mark.asyncio
async def test_development_provider_stub_is_deterministic_authenticated_and_bounded() -> None:
    secret = "unit-stub-secret-sentinel"
    expected_hash = hashlib.sha256(f"Bearer {secret}".encode()).hexdigest()
    app = provider_stub.create_provider_stub_app(
        dimension=3,
        max_batch_size=2,
        expected_authorization_sha256=expected_hash,
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="https://provider.test",
    ) as client:
        headers = {"Authorization": f"Bearer {secret}"}
        first = await client.post(
            "/v1/embeddings",
            headers=headers,
            json={"input": ["same input"], "model": "stub-model"},
        )
        second = await client.post(
            "/v1/embeddings",
            headers=headers,
            json={"input": ["same input"], "model": "stub-model"},
        )
        oversized = await client.post(
            "/v1/embeddings",
            headers=headers,
            json={"input": ["one", "two", "three"], "model": "stub-model"},
        )
        unauthorized = await client.post(
            "/v1/embeddings",
            headers={"Authorization": "Bearer wrong-secret-sentinel"},
            json={"input": ["same input"], "model": "stub-model"},
        )

    assert first.status_code == 200
    assert first.json() == second.json()
    assert first.json()["data"][0]["embedding"] == pytest.approx(
        first.json()["data"][0]["embedding"]
    )
    assert len(first.json()["data"][0]["embedding"]) == 3
    assert oversized.status_code == 422
    assert unauthorized.status_code == 401
    rendered = repr(app.state.request_records) + oversized.text + unauthorized.text
    assert secret not in rendered
    assert "wrong-secret-sentinel" not in rendered


@pytest.mark.asyncio
async def test_development_provider_stub_accepts_only_the_fixed_openrouter_base_path() -> None:
    secret = "unit-openrouter-stub-secret-sentinel"
    expected_hash = hashlib.sha256(f"Bearer {secret}".encode()).hexdigest()
    app = provider_stub.create_provider_stub_app(
        dimension=3,
        max_batch_size=2,
        expected_authorization_sha256=expected_hash,
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="https://openrouter.ai",
    ) as client:
        headers = {"Authorization": f"Bearer {secret}"}
        fixed_openrouter = await client.post(
            "/api/v1/embeddings",
            headers=headers,
            json={"input": ["same input"], "model": "stub-model"},
        )
        unsupported_version = await client.post(
            "/api/v2/embeddings",
            headers=headers,
            json={"input": ["same input"], "model": "stub-model"},
        )
        arbitrary_prefix = await client.post(
            "/tenant/api/v1/embeddings",
            headers=headers,
            json={"input": ["same input"], "model": "stub-model"},
        )

    assert fixed_openrouter.status_code == 200
    assert len(fixed_openrouter.json()["data"][0]["embedding"]) == 3
    assert unsupported_version.status_code == 404
    assert arbitrary_prefix.status_code == 404
    assert secret not in fixed_openrouter.text


@pytest.mark.asyncio
async def test_development_provider_stub_bounds_retained_request_records() -> None:
    secret = "bounded-record-stub-secret-sentinel"
    expected_hash = hashlib.sha256(f"Bearer {secret}".encode()).hexdigest()
    app = provider_stub.create_provider_stub_app(
        dimension=3,
        max_batch_size=2,
        expected_authorization_sha256=expected_hash,
        request_record_limit=2,
    )
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(
        transport=transport,
        base_url="https://provider.test",
    ) as client:
        for index in range(3):
            response = await client.post(
                "/v1/embeddings",
                headers={"Authorization": f"Bearer {secret}"},
                json={"input": [f"query-{index}"], "model": f"stub-model-{index}"},
            )
            assert response.status_code == 200

    records = list(app.state.request_records)
    assert len(records) == 2
    assert [record["model"] for record in records] == ["stub-model-1", "stub-model-2"]
    assert secret not in repr(records)


@pytest.mark.parametrize(
    ("environment", "host"),
    [
        (Environment.PRODUCTION, "127.0.0.1"),
        (Environment.PRODUCTION, "localhost"),
        (Environment.LOCAL, "0.0.0.0"),
        (Environment.TEST, "192.168.1.10"),
    ],
)
def test_provider_stub_runtime_rejects_production_or_non_loopback_binding(
    environment: Environment,
    host: str,
) -> None:
    with pytest.raises(provider_stub.ProviderStubConfigurationError) as exc_info:
        provider_stub.validate_provider_stub_runtime(environment=environment, host=host)

    assert exc_info.value.args == ("Development provider stub is not allowed",)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None


@pytest.mark.parametrize(
    ("environment", "host"),
    [
        (Environment.LOCAL, "127.0.0.1"),
        (Environment.LOCAL, "::1"),
        (Environment.TEST, "localhost"),
    ],
)
def test_provider_stub_runtime_accepts_only_explicit_local_test_loopback(
    environment: Environment,
    host: str,
) -> None:
    provider_stub.validate_provider_stub_runtime(environment=environment, host=host)


@pytest.mark.parametrize("environment", [Environment.LOCAL, Environment.TEST])
def test_provider_stub_runtime_requires_explicit_container_bind_permission(
    environment: Environment,
) -> None:
    provider_stub.validate_provider_stub_runtime(
        environment=environment,
        host="0.0.0.0",
        allow_container_bind=True,
    )

    with pytest.raises(provider_stub.ProviderStubConfigurationError):
        provider_stub.validate_provider_stub_runtime(
            environment=environment,
            host="0.0.0.0",
            allow_container_bind=False,
        )


def test_provider_stub_runtime_never_allows_production_container_binding() -> None:
    with pytest.raises(provider_stub.ProviderStubConfigurationError):
        provider_stub.validate_provider_stub_runtime(
            environment=Environment.PRODUCTION,
            host="0.0.0.0",
            allow_container_bind=True,
        )


def test_provider_stub_console_command_is_installed_and_visibly_development_only() -> None:
    scripts = {entry.name: entry.value for entry in entry_points(group="console_scripts")}

    assert scripts["velox-provider-stub"] == "rag_service.dev.provider_stub:main"
    assert "development" in (provider_stub.__doc__ or "").lower()


class _CredentialQueryResult:
    def __init__(self, row: object | None) -> None:
        self._row = row

    def one_or_none(self) -> object | None:
        return self._row


class _CredentialQuerySession:
    def __init__(
        self,
        owner: _CredentialSessionFactory,
        row: object | None = None,
        error: BaseException | None = None,
        execute_gate: asyncio.Event | None = None,
    ) -> None:
        self.owner = owner
        self.row = row
        self.error = error
        self.execute_gate = execute_gate
        self.statements: list[object] = []
        self.entered = False
        self.closed = False

    async def __aenter__(self) -> _CredentialQuerySession:
        self.entered = True
        self.owner.active += 1
        self.owner.max_active = max(self.owner.max_active, self.owner.active)
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        del exc_type, exc, traceback
        self.closed = True
        self.owner.active -= 1

    async def execute(self, statement: object) -> _CredentialQueryResult:
        self.statements.append(statement)
        await self.owner.execute_started.put(self)
        if self.execute_gate is not None:
            await self.execute_gate.wait()
        if self.error is not None:
            raise self.error
        return _CredentialQueryResult(self.row)


class _CredentialSessionFactory:
    def __init__(
        self,
        *,
        rows: list[object | None] | None = None,
        errors: list[BaseException | None] | None = None,
        execute_gate: asyncio.Event | None = None,
    ) -> None:
        self.rows = list(rows or [])
        self.errors = list(errors or [])
        self.execute_gate = execute_gate
        self.sessions: list[_CredentialQuerySession] = []
        self.execute_started: asyncio.Queue[_CredentialQuerySession] = asyncio.Queue()
        self.active = 0
        self.max_active = 0

    def __call__(self) -> _CredentialQuerySession:
        row = self.rows.pop(0) if self.rows else None
        error = self.errors.pop(0) if self.errors else None
        session = _CredentialQuerySession(
            self,
            row=row,
            error=error,
            execute_gate=self.execute_gate,
        )
        self.sessions.append(session)
        return session


@pytest.mark.asyncio
async def test_sqlalchemy_credential_reader_loads_current_ciphertext_by_stable_id() -> None:
    credential_id = uuid4()
    encrypted = _encrypted(credential_id, _ROTATED_SECRET)
    factory = _CredentialSessionFactory(
        rows=[
            (
                encrypted.ciphertext,
                encrypted.nonce,
                encrypted.key_version,
                encrypted.algorithm,
            )
        ]
    )
    reader = SqlAlchemyProviderCredentialReader(cast(Any, factory))

    loaded = await reader.get_encrypted(credential_id)

    assert loaded == encrypted
    session = factory.sessions[0]
    assert len(session.statements) == 1
    assert session.entered is True
    assert session.closed is True
    assert factory.active == 0
    assert repr(reader) == "SqlAlchemyProviderCredentialReader(<redacted>)"
    assert _ROTATED_SECRET not in repr(reader)
    assert encrypted.ciphertext.hex() not in repr(reader)


@pytest.mark.asyncio
async def test_sqlalchemy_credential_reader_returns_none_for_missing_stable_id() -> None:
    factory = _CredentialSessionFactory(rows=[None])
    reader = SqlAlchemyProviderCredentialReader(cast(Any, factory))

    assert await reader.get_encrypted(uuid4()) is None
    assert factory.sessions[0].closed is True
    assert factory.active == 0


@pytest.mark.asyncio
async def test_sqlalchemy_credential_reader_sanitizes_database_failure() -> None:
    sentinel = "credential-query-database-secret-sentinel"
    factory = _CredentialSessionFactory(errors=[RuntimeError(sentinel)])
    reader = SqlAlchemyProviderCredentialReader(cast(Any, factory))

    with pytest.raises(ProviderCredentialUnavailableError) as exc_info:
        await reader.get_encrypted(uuid4())

    assert sentinel not in str(exc_info.value)
    assert sentinel not in repr(exc_info.value)
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert factory.sessions[0].closed is True
    assert factory.active == 0


@pytest.mark.asyncio
async def test_sqlalchemy_credential_reader_propagates_cancellation() -> None:
    factory = _CredentialSessionFactory(errors=[asyncio.CancelledError("cancel-secret-sentinel")])
    reader = SqlAlchemyProviderCredentialReader(cast(Any, factory))

    with pytest.raises(asyncio.CancelledError) as exc_info:
        await reader.get_encrypted(uuid4())

    assert exc_info.value.__cause__ is None
    assert exc_info.value.__context__ is None
    assert "cancel-secret-sentinel" not in str(exc_info.value)
    assert factory.sessions[0].closed is True
    assert factory.active == 0


@pytest.mark.asyncio
async def test_sqlalchemy_credential_reader_uses_independent_short_lived_sessions() -> None:
    first_id = uuid4()
    second_id = uuid4()
    first_encrypted = _encrypted(first_id, _SECRET)
    second_encrypted = _encrypted(second_id, _ROTATED_SECRET)
    gate = asyncio.Event()
    factory = _CredentialSessionFactory(
        rows=[
            (
                first_encrypted.ciphertext,
                first_encrypted.nonce,
                first_encrypted.key_version,
                first_encrypted.algorithm,
            ),
            (
                second_encrypted.ciphertext,
                second_encrypted.nonce,
                second_encrypted.key_version,
                second_encrypted.algorithm,
            ),
        ],
        execute_gate=gate,
    )
    reader = SqlAlchemyProviderCredentialReader(cast(Any, factory))
    first = asyncio.create_task(reader.get_encrypted(first_id))
    second = asyncio.create_task(reader.get_encrypted(second_id))
    try:
        async with asyncio.timeout(1):
            first_session = await factory.execute_started.get()
            second_session = await factory.execute_started.get()
        assert first_session is not second_session
        assert factory.active == 2
        assert factory.max_active == 2

        gate.set()
        assert await first == first_encrypted
        assert await second == second_encrypted
        assert all(session.closed for session in factory.sessions)
        assert factory.active == 0
    finally:
        gate.set()
        for task in (first, second):
            if not task.done():
                task.cancel()
        await asyncio.gather(first, second, return_exceptions=True)


@pytest.mark.asyncio
async def test_sqlalchemy_credential_reader_closes_session_when_task_is_cancelled() -> None:
    credential_id = uuid4()
    encrypted = _encrypted(credential_id, _SECRET)
    gate = asyncio.Event()
    factory = _CredentialSessionFactory(
        rows=[
            (
                encrypted.ciphertext,
                encrypted.nonce,
                encrypted.key_version,
                encrypted.algorithm,
            )
        ],
        execute_gate=gate,
    )
    reader = SqlAlchemyProviderCredentialReader(cast(Any, factory))
    task = asyncio.create_task(reader.get_encrypted(credential_id))
    try:
        async with asyncio.timeout(1):
            session = await factory.execute_started.get()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert session.closed is True
        assert factory.active == 0
    finally:
        gate.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_gateway_does_not_hold_credential_session_during_provider_network_call() -> None:
    credential_id = uuid4()
    encrypted = _encrypted(credential_id, _SECRET)
    factory = _CredentialSessionFactory(
        rows=[
            (
                encrypted.ciphertext,
                encrypted.nonce,
                encrypted.key_version,
                encrypted.algorithm,
            )
        ]
    )
    reader = SqlAlchemyProviderCredentialReader(cast(Any, factory))

    class SessionObservingTransport(RecordingTransport):
        async def post_json(
            self,
            *,
            endpoint: object,
            path: str,
            headers: dict[str, str],
            payload: dict[str, object],
            timeout_seconds: float,
        ) -> ProviderHttpResponse:
            assert factory.active == 0
            assert factory.sessions and all(session.closed for session in factory.sessions)
            return await super().post_json(
                endpoint=endpoint,
                path=path,
                headers=headers,
                payload=payload,
                timeout_seconds=timeout_seconds,
            )

    transport = SessionObservingTransport([_response([[0.1, 0.2, 0.3]])])
    gateway = EmbeddingGateway(
        keyring=_keyring(),
        credential_reader=reader,
        transport=transport,
    )

    result = await gateway.embed(
        snapshot=_snapshot(credential_id),
        operational=_operational(),
        inputs=("query",),
    )

    assert result.vectors == ((0.1, 0.2, 0.3),)
    assert factory.active == 0


@pytest.mark.asyncio
async def test_gateway_uses_injected_condition_factory_for_registry_and_provider_state() -> None:
    credential_id = uuid4()
    created: list[asyncio.Condition] = []

    def condition_factory() -> asyncio.Condition:
        condition = asyncio.Condition()
        created.append(condition)
        return condition

    transport = RecordingTransport([_response([[0.1, 0.2, 0.3]])])
    reader = FakeCredentialReader(
        values={credential_id: _encrypted(credential_id, _SECRET)},
        calls=[],
    )
    gateway = EmbeddingGateway(
        keyring=_keyring(),
        credential_reader=reader,
        transport=transport,
        condition_factory=condition_factory,
    )

    await gateway.embed(
        snapshot=_snapshot(credential_id),
        operational=_operational(),
        inputs=("query",),
    )

    assert len(created) == 2


def test_provider_usage_context_allows_an_unattributed_system_operation() -> None:
    try:
        context = ProviderUsageContext(
            request_id="repair-generation:attempt-1",
            actor_api_key_id=None,
            provider_config_id=uuid4(),
            model_profile_id=uuid4(),
        )
    except ValueError:
        context = None

    assert context is not None
    assert context.actor_api_key_id is None
