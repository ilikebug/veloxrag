import asyncio
import base64
import os
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from types import TracebackType
from typing import Any, cast
from uuid import UUID, uuid4

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from rag_service.api.errors import BusinessError
from rag_service.api.middleware import get_request_id
from rag_service.auth.dependencies import require_admin_principal
from rag_service.auth.policies import AdminPrincipal
from rag_service.config import Settings
from rag_service.db.models.auth import AuditEvent
from rag_service.main import create_app
from rag_service.providers import routes as provider_routes
from rag_service.providers.embedding_probe import _PROBE_INPUT, EmbeddingProbeService
from rag_service.providers.embeddings import (
    EmbeddingDimensionProbeRequest,
    EmbeddingGatewayError,
    EmbeddingProbeOperationalConfig,
)
from rag_service.providers.repositories import (
    AdminActorRecord,
    ProviderConfigRecord,
    ProviderConfigSecretSourceRecord,
    ProviderCredentialRecord,
    ProviderRepositories,
)
from rag_service.providers.schemas import EmbeddingProbeCreate, SafeEmbeddingProbe

NOW = datetime(2026, 8, 1, 8, 0, tzinfo=UTC)


class FakeTransaction:
    def __init__(self, session: "FakeSession") -> None:
        self._session = session
        self._audit_snapshot: list[AuditEvent] = []

    async def __aenter__(self) -> None:
        assert not self._session.in_transaction
        self._audit_snapshot = list(self._session.audits)
        self._session.in_transaction = True

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exception_type is not None:
            self._session.audits = self._audit_snapshot
        self._session.in_transaction = False
        self._session.admin_locked = False


class FakeSession:
    def __init__(self) -> None:
        self.in_transaction = False
        self.admin_locked = False
        self.configs: dict[UUID, ProviderConfigRecord] = {}
        self.sources: dict[UUID, ProviderConfigSecretSourceRecord] = {}
        self.credentials: dict[UUID, ProviderCredentialRecord] = {}
        self.admins: dict[UUID, AdminActorRecord] = {}
        self.audits: list[AuditEvent] = []
        self.audit_error: BaseException | None = None
        self.transaction_count = 0

    def begin(self) -> FakeTransaction:
        self.transaction_count += 1
        return FakeTransaction(self)


class FakeAdminRepository:
    def __init__(self, session: FakeSession) -> None:
        self._session = session

    async def get_for_update(self, key_id: UUID) -> AdminActorRecord | None:
        assert self._session.in_transaction
        self._session.admin_locked = True
        return self._session.admins.get(key_id)


class FakeConfigRepository:
    def __init__(self, session: FakeSession) -> None:
        self._session = session

    async def get_safe(
        self,
        provider_config_id: UUID,
        *,
        for_update: bool = False,
    ) -> ProviderConfigRecord | None:
        assert self._session.in_transaction
        assert not for_update
        return self._session.configs.get(provider_config_id)

    async def get_secret_source(
        self,
        provider_config_id: UUID,
    ) -> ProviderConfigSecretSourceRecord | None:
        assert self._session.in_transaction
        return self._session.sources.get(provider_config_id)


class FakeCredentialRepository:
    def __init__(self, session: FakeSession) -> None:
        self._session = session

    async def get_safe(
        self,
        credential_id: UUID,
        *,
        for_update: bool = False,
    ) -> ProviderCredentialRecord | None:
        assert self._session.in_transaction
        assert not for_update
        return self._session.credentials.get(credential_id)


class FakeAuditRepository:
    def __init__(self, session: FakeSession) -> None:
        self._session = session

    async def add(self, event: AuditEvent) -> None:
        assert self._session.in_transaction
        if self._session.audit_error is not None:
            raise self._session.audit_error
        self._session.audits.append(event)


class FakeGateway:
    def __init__(self, session: FakeSession) -> None:
        self._session = session
        self.calls: list[
            tuple[EmbeddingDimensionProbeRequest, EmbeddingProbeOperationalConfig, str]
        ] = []
        self.dimension = 3
        self.error: BaseException | None = None

    async def probe_dimension(
        self,
        *,
        request: EmbeddingDimensionProbeRequest,
        operational: EmbeddingProbeOperationalConfig,
        input_text: str,
        attempt_observer: object | None = None,
    ) -> int:
        assert attempt_observer is None
        assert not self._session.in_transaction
        assert not self._session.admin_locked
        self.calls.append((request, operational, input_text))
        if self.error is not None:
            raise self.error
        return self.dimension


class SentinelGatewayError(EmbeddingGatewayError):
    __slots__ = ("headers", "raw_upstream_body", "request_body", "traceback_text", "vectors")

    def __init__(self, sentinels: tuple[str, ...]) -> None:
        super().__init__(
            "PROVIDER_RESPONSE_INVALID",
            " ".join(sentinels),
            retryable=False,
        )
        self.vectors = [sentinels[0]]
        self.headers = {"X-Test-Sentinel": sentinels[1]}
        self.request_body = {"input": sentinels[2]}
        self.raw_upstream_body = sentinels[3]
        self.traceback_text = sentinels[4]


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        admin_key_hmac_secret=SecretStr(base64.urlsafe_b64encode(os.urandom(32)).decode()),
        agent_key_hmac_secret=SecretStr(base64.urlsafe_b64encode(os.urandom(32)).decode()),
    )


def _actor() -> AdminPrincipal:
    return AdminPrincipal(key_id=uuid4(), public_id=uuid4().hex)


def _record(
    provider_config_id: UUID,
    credential_id: UUID,
    *,
    provider_type: str = "openrouter",
    enabled: bool = True,
) -> ProviderConfigRecord:
    return ProviderConfigRecord(
        id=provider_config_id,
        name="probe provider",
        provider_type=provider_type,
        base_url="https://provider.example/v1",
        credential_id=credential_id,
        default_headers={},
        routing_options={},
        timeout_seconds=Decimal("12.5"),
        max_concurrency=7,
        requests_per_minute=90,
        enabled=enabled,
        resource_revision=1,
        endpoint_policy_version="v1",
        endpoint_validated_at=NOW,
        created_at=NOW,
        updated_at=NOW,
    )


def _service(
    session: FakeSession,
    gateway: FakeGateway,
    *,
    clock: Any = lambda: NOW,
) -> EmbeddingProbeService:
    def repositories(value: object) -> ProviderRepositories:
        assert value is session
        return ProviderRepositories(
            credentials=cast(Any, FakeCredentialRepository(session)),
            admins=FakeAdminRepository(session),
            idempotency=cast(Any, object()),
            audits=cast(Any, FakeAuditRepository(session)),
            configs=cast(Any, FakeConfigRepository(session)),
        )

    return EmbeddingProbeService(
        session=cast(Any, session),
        settings=_settings(),
        embedding_gateway=cast(Any, gateway),
        repository_factory=cast(Any, repositories),
        clock=clock,
    )


def _seed(session: FakeSession) -> tuple[AdminPrincipal, UUID, UUID]:
    actor = _actor()
    provider_config_id = uuid4()
    credential_id = uuid4()
    session.admins[actor.key_id] = AdminActorRecord(
        id=actor.key_id,
        public_id=actor.public_id,
        key_type="admin",
        status="active",
        not_before=None,
        expires_at=None,
        revoked_at=None,
    )
    session.configs[provider_config_id] = _record(provider_config_id, credential_id)
    session.credentials[credential_id] = ProviderCredentialRecord(
        id=credential_id,
        name="probe credential",
        key_version="active",
        resource_revision=1,
        created_at=NOW,
        updated_at=NOW,
        rotated_at=None,
    )
    session.sources[provider_config_id] = ProviderConfigSecretSourceRecord(
        provider_config_id=provider_config_id,
        credential_id=credential_id,
        secret_ref=None,
    )
    return actor, provider_config_id, credential_id


def _error_tuple(error: BusinessError) -> tuple[int, str, str, bool]:
    return error.status_code, error.code, error.message, error.retryable


def test_embedding_probe_schemas_normalize_and_serialize_only_safe_fields() -> None:
    provider_config_id = uuid4()

    command = EmbeddingProbeCreate(model_name="  vendor/model  ")
    safe = SafeEmbeddingProbe(
        provider_config_id=provider_config_id,
        model_name=command.model_name,
        dimension=3,
    )

    assert command.model_name == "vendor/model"
    assert safe.model_dump(mode="json") == {
        "provider_config_id": str(provider_config_id),
        "model_name": "vendor/model",
        "dimension": 3,
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"model_name": True},
        {"model_name": ""},
        {"model_name": "   "},
        {"model_name": "bad\x00model"},
        {"model_name": "bad\x1fmodel"},
        {"model_name": "bad\x7fmodel"},
        {"model_name": "bad\x80model"},
        {"model_name": "bad\x9fmodel"},
        {"model_name": "m" * 256},
        {"model_name": "valid", "unexpected": "field"},
    ],
)
def test_embedding_probe_create_rejects_invalid_or_extra_input(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        EmbeddingProbeCreate.model_validate(payload)


@pytest.mark.parametrize("dimension", [True, 0, 10_000_001])
def test_safe_embedding_probe_rejects_invalid_dimension(dimension: object) -> None:
    with pytest.raises(ValidationError):
        SafeEmbeddingProbe(
            provider_config_id=uuid4(),
            model_name="vendor/model",
            dimension=cast(Any, dimension),
        )


@pytest.mark.asyncio
async def test_probe_builds_exact_gateway_request_outside_transaction_and_audits_success() -> None:
    session = FakeSession()
    actor, provider_config_id, credential_id = _seed(session)
    gateway = FakeGateway(session)

    result = await _service(session, gateway).probe(
        provider_config_id,
        EmbeddingProbeCreate(model_name=" vendor/model "),
        actor=actor,
        request_id="req-probe-success",
    )

    assert result == SafeEmbeddingProbe(
        provider_config_id=provider_config_id,
        model_name="vendor/model",
        dimension=3,
    )
    assert len(gateway.calls) == 1
    request, operational, input_text = gateway.calls[0]
    assert request == EmbeddingDimensionProbeRequest(
        adapter_schema_version="openai-embeddings-v1",
        provider_type="openrouter",
        base_url="https://provider.example/v1",
        credential_id=credential_id,
        default_headers={},
        routing_options={},
        model_name="vendor/model",
    )
    assert operational == EmbeddingProbeOperationalConfig(
        provider_config_id=provider_config_id,
        provider_enabled=True,
        timeout_seconds=Decimal("12.5"),
        max_concurrency=7,
        requests_per_minute=90,
        batch_size=1,
    )
    assert input_text == _PROBE_INPUT == "RAG embedding configuration dimension probe."
    assert session.transaction_count == 2
    assert len(session.audits) == 1
    audit = session.audits[0]
    assert audit.request_id == "req-probe-success"
    assert audit.actor_api_key_id == actor.key_id
    assert audit.action == "provider_config.embedding_dimension_probed"
    assert audit.target_type == "provider_config"
    assert audit.target_id == provider_config_id
    assert audit.metadata_ == {
        "request_id": "req-probe-success",
        "actor_key_id": str(actor.key_id),
        "provider_config_id": str(provider_config_id),
        "model_name": "vendor/model",
        "outcome": "succeeded",
        "dimension": 3,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("missing", (404, "RESOURCE_NOT_FOUND")),
        ("disabled", (409, "PROVIDER_CONFIG_DISABLED")),
        ("unsupported", (422, "PROVIDER_TYPE_UNSUPPORTED")),
        ("legacy", (422, "PROVIDER_CREDENTIAL_UNSUPPORTED")),
        ("mismatch", (422, "PROVIDER_CREDENTIAL_INVALID")),
        ("missing_credential", (422, "PROVIDER_CREDENTIAL_INVALID")),
        ("missing_source", (422, "PROVIDER_CREDENTIAL_INVALID")),
        ("malformed", (500, "INTERNAL_ERROR")),
    ],
)
async def test_probe_rejects_unsafe_configuration_before_network(
    mutation: str,
    expected: tuple[int, str],
) -> None:
    session = FakeSession()
    actor, provider_config_id, credential_id = _seed(session)
    gateway = FakeGateway(session)
    if mutation == "missing":
        session.configs.clear()
    elif mutation == "disabled":
        session.configs[provider_config_id] = replace(
            session.configs[provider_config_id], enabled=False
        )
    elif mutation == "unsupported":
        session.configs[provider_config_id] = replace(
            session.configs[provider_config_id], provider_type="vendor_specific"
        )
    elif mutation == "legacy":
        session.sources[provider_config_id] = ProviderConfigSecretSourceRecord(
            provider_config_id=provider_config_id,
            credential_id=None,
            secret_ref="legacy-reference",
        )
    elif mutation == "mismatch":
        session.sources[provider_config_id] = ProviderConfigSecretSourceRecord(
            provider_config_id=provider_config_id,
            credential_id=uuid4(),
            secret_ref=None,
        )
    elif mutation == "missing_credential":
        session.credentials.clear()
    elif mutation == "missing_source":
        session.sources.clear()
    elif mutation == "malformed":
        session.configs[provider_config_id] = replace(
            session.configs[provider_config_id], timeout_seconds=Decimal("0")
        )

    with pytest.raises(BusinessError) as captured:
        await _service(session, gateway).probe(
            provider_config_id,
            EmbeddingProbeCreate(model_name="vendor/model"),
            actor=actor,
            request_id="req-probe-preflight",
        )

    assert (captured.value.status_code, captured.value.code) == expected
    assert gateway.calls == []
    assert session.transaction_count == 2
    assert len(session.audits) == 1
    audit = session.audits[0]
    assert audit.request_id == "req-probe-preflight"
    assert audit.actor_api_key_id == actor.key_id
    assert audit.target_id == provider_config_id
    assert audit.metadata_ == {
        "request_id": "req-probe-preflight",
        "actor_key_id": str(actor.key_id),
        "provider_config_id": str(provider_config_id),
        "model_name": "vendor/model",
        "outcome": "failed",
        "error_code": expected[1],
    }
    assert str(credential_id) not in repr(captured.value)
    assert "legacy-reference" not in repr(captured.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("actor_state", ["wrong_type", "missing", "revoked", "mismatch"])
async def test_probe_requires_current_matching_admin_actor(actor_state: str) -> None:
    session = FakeSession()
    actor, provider_config_id, _ = _seed(session)
    gateway = FakeGateway(session)
    supplied: object = actor
    if actor_state == "wrong_type":
        supplied = object()
    elif actor_state == "missing":
        session.admins.clear()
    elif actor_state == "revoked":
        session.admins[actor.key_id] = replace(session.admins[actor.key_id], revoked_at=NOW)
    else:
        session.admins[actor.key_id] = replace(session.admins[actor.key_id], public_id=uuid4().hex)

    with pytest.raises(BusinessError) as captured:
        await _service(session, gateway).probe(
            provider_config_id,
            EmbeddingProbeCreate(model_name="vendor/model"),
            actor=cast(Any, supplied),
            request_id="req-probe-admin",
        )

    assert _error_tuple(captured.value) == (
        401,
        "INVALID_API_KEY",
        "Invalid API key",
        False,
    )
    assert gateway.calls == []
    assert session.audits == []


@pytest.mark.asyncio
async def test_probe_rejects_invalid_request_id_before_transaction_or_network() -> None:
    session = FakeSession()
    actor, provider_config_id, _ = _seed(session)
    gateway = FakeGateway(session)

    with pytest.raises(BusinessError) as captured:
        await _service(session, gateway).probe(
            provider_config_id,
            EmbeddingProbeCreate(model_name="vendor/model"),
            actor=actor,
            request_id="bad\x00request",
        )

    assert (captured.value.status_code, captured.value.code) == (422, "VALIDATION_ERROR")
    assert session.transaction_count == 0
    assert gateway.calls == []
    assert session.audits == []


@pytest.mark.asyncio
async def test_probe_preserves_trusted_preflight_error_when_failure_audit_fails() -> None:
    session = FakeSession()
    actor, provider_config_id, _ = _seed(session)
    gateway = FakeGateway(session)
    session.configs[provider_config_id] = replace(
        session.configs[provider_config_id], enabled=False
    )
    session.audit_error = RuntimeError("audit-failure-test-sentinel")

    with pytest.raises(BusinessError) as captured:
        await _service(session, gateway).probe(
            provider_config_id,
            EmbeddingProbeCreate(model_name="vendor/model"),
            actor=actor,
            request_id="req-probe-preflight-audit-failure",
        )

    assert _error_tuple(captured.value) == (
        409,
        "PROVIDER_CONFIG_DISABLED",
        "Provider configuration is disabled",
        False,
    )
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert gateway.calls == []
    assert session.audits == []


@pytest.mark.asyncio
async def test_probe_replaces_syntactically_valid_unknown_gateway_code_everywhere() -> None:
    session = FakeSession()
    actor, provider_config_id, _ = _seed(session)
    gateway = FakeGateway(session)
    gateway.error = EmbeddingGatewayError(
        "UPSTREAM_VECTOR_DUMP",
        "unknown gateway message test sentinel",
        retryable=True,
    )

    with pytest.raises(BusinessError) as captured:
        await _service(session, gateway).probe(
            provider_config_id,
            EmbeddingProbeCreate(model_name="vendor/model"),
            actor=actor,
            request_id="req-probe-unknown-code",
        )

    assert _error_tuple(captured.value) == (
        503,
        "PROVIDER_FAILURE",
        "Provider request failed",
        True,
    )
    assert session.audits[0].metadata_ == {
        "request_id": "req-probe-unknown-code",
        "actor_key_id": str(actor.key_id),
        "provider_config_id": str(provider_config_id),
        "model_name": "vendor/model",
        "outcome": "failed",
        "error_code": "PROVIDER_FAILURE",
    }
    assert "UPSTREAM_VECTOR_DUMP" not in repr(captured.value)
    assert "UPSTREAM_VECTOR_DUMP" not in repr(session.audits[0].metadata_)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("retryable", "expected"),
    [
        (True, (503, "PROVIDER_UNAVAILABLE", True)),
        (False, (422, "PROVIDER_RESPONSE_INVALID", False)),
    ],
)
async def test_probe_maps_gateway_failure_and_audits_sanitized_code(
    retryable: bool,
    expected: tuple[int, str, bool],
) -> None:
    session = FakeSession()
    actor, provider_config_id, _ = _seed(session)
    gateway = FakeGateway(session)
    gateway.error = EmbeddingGatewayError(
        expected[1],
        "sanitized provider failure",
        retryable=retryable,
    )

    with pytest.raises(BusinessError) as captured:
        await _service(session, gateway).probe(
            provider_config_id,
            EmbeddingProbeCreate(model_name="vendor/model"),
            actor=actor,
            request_id="req-probe-failure",
        )

    assert (
        captured.value.status_code,
        captured.value.code,
        captured.value.retryable,
    ) == expected
    assert captured.value.message in {"Provider unavailable", "Provider response is invalid"}
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert session.audits[0].metadata_ == {
        "request_id": "req-probe-failure",
        "actor_key_id": str(actor.key_id),
        "provider_config_id": str(provider_config_id),
        "model_name": "vendor/model",
        "outcome": "failed",
        "error_code": expected[1],
    }


@pytest.mark.asyncio
async def test_probe_gateway_failure_does_not_leak_operational_sentinels(
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = FakeSession()
    actor, provider_config_id, _ = _seed(session)
    gateway = FakeGateway(session)
    sentinels = (
        "vector-test-sentinel",
        "header-test-sentinel",
        "request-body-test-sentinel",
        "raw-upstream-body-test-sentinel",
        "traceback-test-sentinel",
    )
    source = SentinelGatewayError(sentinels)
    source.__cause__ = RuntimeError("cause-test-sentinel")
    source.__context__ = RuntimeError("context-test-sentinel")
    gateway.error = source

    with pytest.raises(BusinessError) as captured:
        await _service(session, gateway).probe(
            provider_config_id,
            EmbeddingProbeCreate(model_name="vendor/model"),
            actor=actor,
            request_id="req-probe-sentinel-safety",
        )

    assert _error_tuple(captured.value) == (
        422,
        "PROVIDER_RESPONSE_INVALID",
        "Provider response is invalid",
        False,
    )
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert session.audits[0].metadata_ == {
        "request_id": "req-probe-sentinel-safety",
        "actor_key_id": str(actor.key_id),
        "provider_config_id": str(provider_config_id),
        "model_name": "vendor/model",
        "outcome": "failed",
        "error_code": "PROVIDER_RESPONSE_INVALID",
    }
    observable = " ".join(
        (
            repr(captured.value),
            repr(session.audits[0].metadata_),
            " ".join(record.getMessage() for record in caplog.records),
            repr(captured.value.__cause__),
            repr(captured.value.__context__),
        )
    )
    for sentinel in (*sentinels, "cause-test-sentinel", "context-test-sentinel"):
        assert sentinel not in observable


@pytest.mark.asyncio
async def test_probe_preserves_gateway_mapping_when_failure_audit_also_fails() -> None:
    session = FakeSession()
    actor, provider_config_id, _ = _seed(session)
    gateway = FakeGateway(session)
    gateway.error = EmbeddingGatewayError(
        "PROVIDER_TIMEOUT",
        "Provider request timed out",
        retryable=True,
    )
    session.audit_error = RuntimeError("audit-write-sensitive-sentinel")

    with pytest.raises(BusinessError) as captured:
        await _service(session, gateway).probe(
            provider_config_id,
            EmbeddingProbeCreate(model_name="vendor/model"),
            actor=actor,
            request_id="req-probe-audit-failure",
        )

    assert _error_tuple(captured.value) == (
        503,
        "PROVIDER_TIMEOUT",
        "Provider request timed out",
        True,
    )
    assert "sensitive" not in repr(captured.value)


@pytest.mark.asyncio
async def test_probe_success_becomes_sanitized_internal_error_when_audit_fails() -> None:
    session = FakeSession()
    actor, provider_config_id, _ = _seed(session)
    gateway = FakeGateway(session)
    session.audit_error = RuntimeError("audit-write-sensitive-sentinel")

    with pytest.raises(BusinessError) as captured:
        await _service(session, gateway).probe(
            provider_config_id,
            EmbeddingProbeCreate(model_name="vendor/model"),
            actor=actor,
            request_id="req-probe-audit-failure",
        )

    assert _error_tuple(captured.value) == (
        500,
        "INTERNAL_ERROR",
        "Internal server error",
        False,
    )
    assert "sensitive" not in repr(captured.value)


@pytest.mark.asyncio
async def test_probe_propagates_cancellation_without_retaining_sensitive_context() -> None:
    session = FakeSession()
    actor, provider_config_id, _ = _seed(session)
    gateway = FakeGateway(session)
    source = asyncio.CancelledError("cancel-sensitive-sentinel")
    source.__cause__ = RuntimeError("cause-sensitive-sentinel")
    source.__context__ = RuntimeError("context-sensitive-sentinel")
    gateway.error = source

    with pytest.raises(asyncio.CancelledError) as captured:
        await _service(session, gateway).probe(
            provider_config_id,
            EmbeddingProbeCreate(model_name="model-sensitive-sentinel"),
            actor=actor,
            request_id="req-probe-cancel",
        )

    assert captured.value is not source
    assert captured.value.args == ()
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert session.audits == []


class RouteService:
    def __init__(self, provider_config_id: UUID) -> None:
        self.provider_config_id = provider_config_id
        self.calls = 0

    async def probe(
        self,
        provider_config_id: UUID,
        command: EmbeddingProbeCreate,
        *,
        actor: AdminPrincipal,
        request_id: str,
    ) -> SafeEmbeddingProbe:
        assert provider_config_id == self.provider_config_id
        assert command.model_name == "vendor/model"
        assert type(actor) is AdminPrincipal
        assert request_id == "req-route-probe"
        self.calls += 1
        return SafeEmbeddingProbe(
            provider_config_id=provider_config_id,
            model_name=command.model_name,
            dimension=3,
        )


def test_embedding_probe_operation_documents_canonical_uuid_path_pattern() -> None:
    operation = create_app().openapi()["paths"][
        "/v1/admin/provider-configs/{provider_config_id}/embedding-probe"
    ]["post"]
    parameter = next(
        candidate
        for candidate in operation["parameters"]
        if candidate["name"] == "provider_config_id" and candidate["in"] == "path"
    )

    assert parameter["schema"]["format"] == "uuid"
    assert parameter["schema"]["pattern"] == (
        "^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "identifier",
    [
        lambda value: value.hex,
        lambda value: str(value).upper(),
        lambda value: "{" + str(value) + "}",
    ],
)
async def test_embedding_probe_route_rejects_noncanonical_uuid(identifier: Any) -> None:
    provider_config_id = uuid4()
    service = RouteService(provider_config_id)
    app = create_app()
    app.dependency_overrides[provider_routes.get_embedding_probe_service] = lambda: service
    app.dependency_overrides[require_admin_principal] = _actor
    app.dependency_overrides[get_request_id] = lambda: "req-route-probe"
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/v1/admin/provider-configs/{identifier(provider_config_id)}/embedding-probe",
            json={"model_name": "vendor/model"},
        )

    assert response.status_code == 422
    assert service.calls == 0
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.asyncio
@pytest.mark.parametrize("control_character", ["\x80", "\x9f"])
async def test_embedding_probe_route_rejects_c1_control_before_service(
    control_character: str,
) -> None:
    session = FakeSession()
    actor, provider_config_id, _ = _seed(session)
    gateway = FakeGateway(session)
    app = create_app()
    app.dependency_overrides[provider_routes.get_embedding_probe_service] = lambda: _service(
        session, gateway
    )
    app.dependency_overrides[require_admin_principal] = lambda: actor
    app.dependency_overrides[get_request_id] = lambda: "req-route-probe-c1"
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/v1/admin/provider-configs/{provider_config_id}/embedding-probe",
            json={"model_name": f"bad{control_character}model"},
        )

    assert response.status_code == 422
    assert response.status_code != 500
    assert session.transaction_count == 0
    assert gateway.calls == []
    assert session.audits == []
    assert response.headers["cache-control"] == "no-store"


@pytest.mark.asyncio
async def test_embedding_probe_route_returns_no_store_safe_response() -> None:
    provider_config_id = uuid4()
    service = RouteService(provider_config_id)
    app = create_app()
    app.dependency_overrides[provider_routes.get_embedding_probe_service] = lambda: service
    app.dependency_overrides[require_admin_principal] = _actor
    app.dependency_overrides[get_request_id] = lambda: "req-route-probe"
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/v1/admin/provider-configs/{provider_config_id}/embedding-probe",
            json={"model_name": " vendor/model "},
        )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {
        "provider_config_id": str(provider_config_id),
        "model_name": "vendor/model",
        "dimension": 3,
    }
    assert service.calls == 1
