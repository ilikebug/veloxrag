import asyncio
import math
from collections.abc import Collection, Sequence
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID, uuid4

import httpx
import pytest

import rag_service.infrastructure.probes as probes
from rag_service.config import Settings
from rag_service.infrastructure.probes import (
    DatabaseReferencedCredentialLoader,
    LiveReadinessProvider,
)
from rag_service.jobs.worker import WorkerDependencyPreflight
from rag_service.providers.credentials import (
    EncryptedProviderCredential,
    ProviderCredentialKeyring,
)
from rag_service.readiness import (
    INGEST_GENERATION_STATUSES,
    RETRIEVE_GENERATION_STATUSES,
    ComponentStatus,
    ReadinessScope,
    ReadinessSnapshot,
    ReferencedCredential,
    ReferencedCredentialValidator,
)


def status(ok: bool) -> ComponentStatus:
    return ComponentStatus(ok=ok, latency_ms=1.0, error=None if ok else "unavailable")


def healthy_components() -> dict[str, ComponentStatus]:
    return {
        "postgres": status(True),
        "qdrant": status(True),
        "redis": status(True),
        "minio": status(True),
        "ingest_credentials": status(True),
        "retrieve_credentials": status(True),
    }


def test_core_readiness_requires_postgres_and_qdrant() -> None:
    snapshot = ReadinessSnapshot(
        components={
            "postgres": status(True),
            "qdrant": status(True),
            "redis": status(False),
            "minio": status(False),
            "ingest_credentials": status(True),
            "retrieve_credentials": status(True),
        },
        answer_configured=False,
    )

    assert snapshot.core_ready is True
    assert snapshot.ingest_ready is False
    assert snapshot.answer_ready is False


def test_answer_readiness_requires_core_and_configuration() -> None:
    snapshot = ReadinessSnapshot(
        components={
            "postgres": status(True),
            "qdrant": status(True),
            "redis": status(True),
            "minio": status(True),
            "ingest_credentials": status(True),
            "retrieve_credentials": status(True),
        },
        answer_configured=True,
    )

    assert snapshot.core_ready is True
    assert snapshot.ingest_ready is True
    assert snapshot.answer_ready is True


def test_role_readiness_requires_the_exact_referenced_credential_set() -> None:
    components = healthy_components()
    components["ingest_credentials"] = status(False)
    snapshot = ReadinessSnapshot(components=components, answer_configured=True)

    assert snapshot.core_ready is True
    assert snapshot.retrieve_ready is True
    assert snapshot.ingest_ready is False

    components["ingest_credentials"] = status(True)
    components["retrieve_credentials"] = status(False)
    snapshot = ReadinessSnapshot(components=components, answer_configured=True)

    assert snapshot.core_ready is True
    assert snapshot.retrieve_ready is False
    assert snapshot.ingest_ready is True


class TrackingCredentialKeyring:
    def __init__(self) -> None:
        self.authenticated: list[UUID] = []

    def use_decrypted(
        self,
        credential_id: UUID,
        _encrypted: EncryptedProviderCredential,
        callback: Any,
    ) -> None:
        self.authenticated.append(credential_id)
        callback(bytearray(b"transient-secret"))


def referenced_credential(
    credential_id: UUID,
    *,
    resource_revision: int = 1,
    key_version: str = "key-v1",
) -> ReferencedCredential:
    return ReferencedCredential(
        credential_id=credential_id,
        resource_revision=resource_revision,
        encrypted=EncryptedProviderCredential(
            ciphertext=b"ciphertext-with-auth-tag",
            nonce=b"n" * 12,
            key_version=key_version,
        ),
    )


@pytest.mark.asyncio
async def test_referenced_credential_validator_uses_exact_statuses_and_authenticates_all() -> None:
    first_id = uuid4()
    second_id = uuid4()
    requested_statuses: list[frozenset[str]] = []

    async def load(statuses: Collection[str]) -> Sequence[ReferencedCredential]:
        requested_statuses.append(frozenset(statuses))
        return [referenced_credential(first_id), referenced_credential(second_id)]

    keyring = TrackingCredentialKeyring()
    validator = ReferencedCredentialValidator(load_referenced_credentials=load)

    await validator.validate(
        statuses={"active", "building"},
        keyring=cast(ProviderCredentialKeyring, keyring),
        keyring_fingerprint="ring-v1",
    )

    assert requested_statuses == [frozenset({"active", "building"})]
    assert keyring.authenticated == [first_id, second_id]


@pytest.mark.asyncio
async def test_referenced_credential_cache_invalidates_on_revision_key_or_keyring_change() -> None:
    credential_id = uuid4()
    current = referenced_credential(credential_id)

    async def load(_statuses: Collection[str]) -> Sequence[ReferencedCredential]:
        return [current]

    keyring = TrackingCredentialKeyring()
    validator = ReferencedCredentialValidator(
        load_referenced_credentials=load,
        cache_ttl_seconds=30.0,
    )

    async def validate(fingerprint: str = "ring-v1") -> None:
        await validator.validate(
            statuses={"active"},
            keyring=cast(ProviderCredentialKeyring, keyring),
            keyring_fingerprint=fingerprint,
        )

    await validate()
    await validate()
    assert keyring.authenticated == [credential_id]

    current = referenced_credential(credential_id, resource_revision=2)
    await validate()
    current = referenced_credential(credential_id, resource_revision=2, key_version="key-v2")
    await validate()
    await validate("ring-v2")

    assert keyring.authenticated == [credential_id] * 4


@pytest.mark.asyncio
async def test_referenced_credential_cache_expires_after_its_short_ttl() -> None:
    credential_id = uuid4()
    now = 10.0

    async def load(_statuses: Collection[str]) -> Sequence[ReferencedCredential]:
        return [referenced_credential(credential_id)]

    keyring = TrackingCredentialKeyring()
    validator = ReferencedCredentialValidator(
        load_referenced_credentials=load,
        cache_ttl_seconds=5.0,
        clock=lambda: now,
    )

    async def validate() -> None:
        await validator.validate(
            statuses=RETRIEVE_GENERATION_STATUSES,
            keyring=cast(ProviderCredentialKeyring, keyring),
            keyring_fingerprint="ring-v1",
        )

    await validate()
    now = 14.9
    await validate()
    now = 15.1
    await validate()

    assert keyring.authenticated == [credential_id, credential_id]


def test_referenced_credential_cache_rejects_unbounded_configuration() -> None:
    async def load(_statuses: Collection[str]) -> Sequence[ReferencedCredential]:
        return []

    with pytest.raises(ValueError, match="TTL must be finite"):
        ReferencedCredentialValidator(
            load_referenced_credentials=load,
            cache_ttl_seconds=math.inf,
        )
    with pytest.raises(TypeError, match="clock must be callable"):
        ReferencedCredentialValidator(
            load_referenced_credentials=load,
            clock=cast(Any, None),
        )
    with pytest.raises(ValueError, match="cache entry limit must be positive"):
        ReferencedCredentialValidator(
            load_referenced_credentials=load,
            max_cache_entries=0,
        )


@pytest.mark.asyncio
async def test_referenced_credential_cache_has_a_hard_entry_limit() -> None:
    credential_ids = [uuid4(), uuid4(), uuid4()]
    current = referenced_credential(credential_ids[0])

    async def load(_statuses: Collection[str]) -> Sequence[ReferencedCredential]:
        return [current]

    keyring = TrackingCredentialKeyring()
    validator = ReferencedCredentialValidator(
        load_referenced_credentials=load,
        cache_ttl_seconds=30.0,
        max_cache_entries=2,
    )

    async def validate() -> None:
        await validator.validate(
            statuses=RETRIEVE_GENERATION_STATUSES,
            keyring=cast(ProviderCredentialKeyring, keyring),
            keyring_fingerprint="ring-v1",
        )

    await validate()
    current = referenced_credential(credential_ids[1])
    await validate()
    current = referenced_credential(credential_ids[2])
    await validate()
    current = referenced_credential(credential_ids[0])
    await validate()

    assert keyring.authenticated == [*credential_ids, credential_ids[0]]


def test_worker_preflight_requires_an_explicit_keyring_fingerprint() -> None:
    with pytest.raises(TypeError):
        cast(Any, WorkerDependencyPreflight)(
            database=cast(Any, FakeDatabase()),
            keyring=cast(ProviderCredentialKeyring, TrackingCredentialKeyring()),
        )


@pytest.mark.asyncio
async def test_database_credential_loader_uses_only_requested_generations_and_credentials() -> None:
    credential_id = uuid4()
    statements: list[Any] = []
    credential = SimpleNamespace(
        id=credential_id,
        ciphertext=b"ciphertext-with-auth-tag",
        nonce=b"n" * 12,
        key_version="key-v1",
        algorithm="AES-256-GCM",
        resource_revision=7,
    )

    class Rows:
        def __init__(self, values: Sequence[Any]) -> None:
            self._values = values

        def all(self) -> Sequence[Any]:
            return self._values

    class Session:
        async def scalars(self, statement: Any) -> Rows:
            statements.append(statement)
            if len(statements) == 1:
                return Rows([{"credential_id": str(credential_id)}])
            return Rows([credential])

    class Database:
        @asynccontextmanager
        async def sessions(self) -> Any:
            yield Session()

    loader = DatabaseReferencedCredentialLoader(cast(Any, Database()))
    loaded = await loader(RETRIEVE_GENERATION_STATUSES)

    generation_query = statements[0].compile()
    credential_query = statements[1].compile()
    assert generation_query.params["status_1"] == ["active"]
    assert credential_query.params["id_1"] == [credential_id]
    assert all("provider_configs" not in str(statement) for statement in statements)
    assert loaded == (referenced_credential(credential_id, resource_revision=7),)


class TrackingReferencedCredentialValidator:
    def __init__(self, *, fail_statuses: frozenset[str] | None = None) -> None:
        self.fail_statuses = fail_statuses
        self.statuses: list[frozenset[str]] = []

    async def validate(self, *, statuses: Collection[str], **_kwargs: Any) -> None:
        selected = frozenset(statuses)
        self.statuses.append(selected)
        if selected == self.fail_statuses:
            raise RuntimeError("credential unavailable")


@pytest.mark.asyncio
async def test_live_provider_checks_active_building_for_ingest_and_active_for_retrieve() -> None:
    validator = TrackingReferencedCredentialValidator(fail_statuses=INGEST_GENERATION_STATUSES)
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200))
    )
    provider = LiveReadinessProvider(
        Settings(_env_file=None),
        database=FakeDatabase(),
        redis_client=FakeRedis(),
        minio_client=FakeMinio(),
        http_client=http_client,
        credential_validator=cast(Any, validator),
    )
    try:
        snapshot = await provider.snapshot()
    finally:
        await http_client.aclose()

    assert validator.statuses == [INGEST_GENERATION_STATUSES, RETRIEVE_GENERATION_STATUSES]
    assert snapshot.components["ingest_credentials"].ok is False
    assert snapshot.components["retrieve_credentials"].ok is True
    assert snapshot.ingest_ready is False
    assert snapshot.retrieve_ready is True


@pytest.mark.asyncio
async def test_live_provider_reports_keyring_format_failure_without_provider_network_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        return httpx.Response(200)

    def invalid_keyring(_settings: Settings) -> None:
        raise ValueError("keyring detail must remain private")

    monkeypatch.setattr(probes, "provider_credential_keyring_from_settings", invalid_keyring)
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = LiveReadinessProvider(
        Settings(_env_file=None),
        database=FakeDatabase(),
        redis_client=FakeRedis(),
        minio_client=FakeMinio(),
        http_client=http_client,
    )
    try:
        snapshot = await provider.snapshot()
    finally:
        await http_client.aclose()

    assert requested_paths == ["/healthz"]
    assert snapshot.components["ingest_credentials"].error == "ValueError"
    assert snapshot.components["retrieve_credentials"].error == "ValueError"
    assert "keyring detail must remain private" not in repr(snapshot)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "expected_statuses"),
    [
        (ReadinessScope.CORE, []),
        (ReadinessScope.RETRIEVE, [RETRIEVE_GENERATION_STATUSES]),
    ],
)
async def test_scoped_readiness_does_not_call_irrelevant_ingest_probes(
    scope: ReadinessScope,
    expected_statuses: list[frozenset[str]],
) -> None:
    validator = TrackingReferencedCredentialValidator()

    class CountingRedis(FakeRedis):
        def __init__(self) -> None:
            super().__init__()
            self.ping_calls = 0

        async def ping(self) -> bool:
            self.ping_calls += 1
            return await super().ping()

    class CountingMinio(FakeMinio):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def bucket_exists(self, bucket_name: str) -> bool:
            self.calls += 1
            return super().bucket_exists(bucket_name)

    redis = CountingRedis()
    minio = CountingMinio()
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200))
    )
    provider = LiveReadinessProvider(
        Settings(_env_file=None),
        database=FakeDatabase(),
        redis_client=redis,
        minio_client=minio,
        http_client=http_client,
        credential_validator=cast(Any, validator),
    )
    try:
        snapshot = await provider.snapshot(scope)
    finally:
        await http_client.aclose()

    assert redis.ping_calls == 0
    assert minio.calls == 0
    assert validator.statuses == expected_statuses
    assert snapshot.components["redis"] == ComponentStatus(
        ok=False,
        latency_ms=0.0,
        error="NotChecked",
    )
    assert snapshot.components["minio"].error == "NotChecked"
    assert snapshot.components["ingest_credentials"].error == "NotChecked"
    if scope is ReadinessScope.CORE:
        assert snapshot.retrieval_capability is None
        assert snapshot.ingest_capability is None
    else:
        assert snapshot.retrieval_capability is True
        assert snapshot.ingest_capability is None


@pytest.mark.asyncio
async def test_default_all_scope_runs_role_credential_checks_concurrently() -> None:
    both_started = asyncio.Event()
    started: set[frozenset[str]] = set()

    class ConcurrentValidator(TrackingReferencedCredentialValidator):
        async def validate(self, *, statuses: Collection[str], **_kwargs: Any) -> None:
            started.add(frozenset(statuses))
            if len(started) == 2:
                both_started.set()
            await both_started.wait()

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200))
    )
    provider = LiveReadinessProvider(
        Settings(_env_file=None, readiness_timeout_seconds=0.05),
        database=FakeDatabase(),
        redis_client=FakeRedis(),
        minio_client=FakeMinio(),
        http_client=http_client,
        credential_validator=cast(Any, ConcurrentValidator()),
    )
    try:
        snapshot = await provider.snapshot()
    finally:
        await http_client.aclose()

    assert started == {INGEST_GENERATION_STATUSES, RETRIEVE_GENERATION_STATUSES}
    assert snapshot.components["ingest_credentials"].ok is True
    assert snapshot.components["retrieve_credentials"].ok is True


@pytest.mark.asyncio
async def test_worker_preflight_reuses_the_ingest_credential_validation() -> None:
    validator = TrackingReferencedCredentialValidator(fail_statuses=INGEST_GENERATION_STATUSES)
    database = FakeDatabase()
    preflight = WorkerDependencyPreflight(
        database=cast(Any, database),
        keyring=cast(ProviderCredentialKeyring, TrackingCredentialKeyring()),
        credential_validator=cast(Any, validator),
        keyring_fingerprint="test-keyring",
    )

    assert await preflight() is False
    assert database.close_calls == 0
    assert validator.statuses == [INGEST_GENERATION_STATUSES]


def test_constructor_defensively_copies_component_mapping() -> None:
    components = healthy_components()
    components["redis"] = status(False)
    snapshot = ReadinessSnapshot(components=components, answer_configured=True)

    components["redis"] = status(True)

    assert snapshot.core_ready is True
    assert snapshot.ingest_ready is False
    assert snapshot.answer_ready is True


def test_snapshot_component_mapping_rejects_mutation() -> None:
    snapshot = ReadinessSnapshot(
        components=healthy_components(),
        answer_configured=True,
    )

    with pytest.raises(TypeError):
        cast(Any, snapshot.components)["postgres"] = status(False)

    assert snapshot.core_ready is True


def test_missing_required_components_fail_at_construction() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Missing required readiness components: ingest_credentials, minio, qdrant, "
            "redis, retrieve_credentials"
        ),
    ):
        ReadinessSnapshot(
            components={"postgres": status(True)},
            answer_configured=True,
        )


def test_extra_components_are_allowed() -> None:
    components = healthy_components()
    components["metrics"] = status(True)

    snapshot = ReadinessSnapshot(components=components, answer_configured=True)

    assert snapshot.components["metrics"].ok is True


@pytest.mark.parametrize("component", ["postgres", "qdrant"])
def test_each_core_dependency_is_required(component: str) -> None:
    components = healthy_components()
    components[component] = status(False)

    snapshot = ReadinessSnapshot(components=components, answer_configured=True)

    assert snapshot.core_ready is False
    assert snapshot.ingest_ready is False
    assert snapshot.answer_ready is False


@pytest.mark.parametrize("component", ["redis", "minio"])
def test_each_ingest_dependency_is_required(component: str) -> None:
    components = healthy_components()
    components[component] = status(False)

    snapshot = ReadinessSnapshot(components=components, answer_configured=True)

    assert snapshot.core_ready is True
    assert snapshot.ingest_ready is False
    assert snapshot.answer_ready is True


def test_answer_readiness_requires_configuration_with_healthy_core() -> None:
    snapshot = ReadinessSnapshot(
        components=healthy_components(),
        answer_configured=False,
    )

    assert snapshot.core_ready is True
    assert snapshot.ingest_ready is True
    assert snapshot.answer_ready is False


class EmptyRows:
    def all(self) -> list[object]:
        return []


class EmptyCredentialSession:
    async def scalars(self, _statement: object) -> EmptyRows:
        return EmptyRows()


class FakeDatabase:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.close_calls = 0

    async def ping(self) -> None:
        if self.error is not None:
            raise self.error

    @asynccontextmanager
    async def sessions(self) -> Any:
        yield EmptyCredentialSession()

    async def close(self) -> None:
        self.close_calls += 1


class FakeRedis:
    def __init__(self, result: bool = True) -> None:
        self.result = result
        self.close_calls = 0

    async def ping(self) -> bool:
        return self.result

    async def aclose(self) -> None:
        self.close_calls += 1


class FakeMinio:
    def __init__(
        self,
        result: bool = True,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error

    def bucket_exists(self, bucket_name: str) -> bool:
        assert bucket_name == "rag-documents"
        if self.error is not None:
            raise self.error
        return self.result


@pytest.mark.asyncio
async def test_live_provider_reports_all_components() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/healthz":
            return httpx.Response(200)
        return httpx.Response(404)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        provider = LiveReadinessProvider(
            Settings(_env_file=None),
            database=FakeDatabase(),
            redis_client=FakeRedis(),
            minio_client=FakeMinio(),
            http_client=http_client,
        )
        snapshot = await provider.snapshot()
        await provider.close()
    finally:
        await http_client.aclose()

    assert all(component.ok for component in snapshot.components.values())
    assert snapshot.answer_configured is False


@pytest.mark.asyncio
async def test_live_provider_sanitizes_component_failures() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        status = 503 if request.url.path == "/healthz" else 200
        return httpx.Response(status)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        provider = LiveReadinessProvider(
            Settings(_env_file=None),
            database=FakeDatabase(RuntimeError("contains-sensitive-detail")),
            redis_client=FakeRedis(result=False),
            minio_client=FakeMinio(result=False),
            http_client=http_client,
        )
        snapshot = await provider.snapshot()
        await provider.close()
    finally:
        await http_client.aclose()

    assert snapshot.components["postgres"].error == "RuntimeError"
    assert snapshot.components["qdrant"].error == "HTTPStatusError"
    assert snapshot.components["redis"].error == "RuntimeError"
    assert snapshot.components["minio"].error == "RuntimeError"
    assert "contains-sensitive-detail" not in repr(snapshot)


@pytest.mark.asyncio
async def test_live_provider_sanitizes_minio_auth_failure() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        provider = LiveReadinessProvider(
            Settings(_env_file=None),
            database=FakeDatabase(),
            redis_client=FakeRedis(),
            minio_client=FakeMinio(error=PermissionError("invalid-secret-detail")),
            http_client=http_client,
        )
        snapshot = await provider.snapshot()
        await provider.close()
    finally:
        await http_client.aclose()

    assert snapshot.components["minio"].error == "PermissionError"
    assert "invalid-secret-detail" not in repr(snapshot)


class HangingDatabase(FakeDatabase):
    async def ping(self) -> None:
        await asyncio.Event().wait()


@pytest.mark.asyncio
async def test_live_provider_applies_total_component_deadline() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = LiveReadinessProvider(
        Settings(_env_file=None, readiness_timeout_seconds=0.01),
        database=HangingDatabase(),
        redis_client=FakeRedis(),
        minio_client=FakeMinio(),
        http_client=http_client,
    )

    try:
        async with asyncio.timeout(0.2):
            snapshot = await provider.snapshot()
        await provider.close()
    finally:
        await http_client.aclose()

    assert snapshot.components["postgres"].error == "TimeoutError"
    assert snapshot.components["postgres"].latency_ms < 200


@pytest.mark.parametrize(
    "minio_url",
    [
        "htp://localhost:9000",
        "http:///",
        "http://user:password@localhost:9000",
        "http://localhost:9000/documents",
        "http://localhost:9000?region=local",
        "http://localhost:9000#storage",
        "http://localhost:9000/;version=1",
        "http://localhost:not-a-port",
        "http://localhost:99999",
    ],
)
@pytest.mark.asyncio
async def test_invalid_minio_url_is_rejected_before_resource_construction(
    monkeypatch: pytest.MonkeyPatch,
    minio_url: str,
) -> None:
    class ForbiddenDatabase:
        @classmethod
        def from_settings(cls, _settings: Settings) -> FakeDatabase:
            raise AssertionError("database constructed before MinIO URL validation")

    monkeypatch.setattr(probes, "Database", ForbiddenDatabase)

    with pytest.raises(ValueError, match="MinIO URL"):
        await LiveReadinessProvider.create(Settings(_env_file=None, minio_url=minio_url))


class FalseyDatabase(FakeDatabase):
    def __bool__(self) -> bool:
        return False


@pytest.mark.asyncio
async def test_falsey_injected_database_is_retained(monkeypatch: pytest.MonkeyPatch) -> None:
    class ForbiddenDatabase:
        @classmethod
        def from_settings(cls, _settings: Settings) -> FakeDatabase:
            raise AssertionError("falsey injected database was replaced")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200)

    monkeypatch.setattr(probes, "Database", ForbiddenDatabase)
    database = FalseyDatabase()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        provider = LiveReadinessProvider(
            Settings(_env_file=None),
            database=database,
            redis_client=FakeRedis(),
            minio_client=FakeMinio(),
            http_client=http_client,
        )
        snapshot = await provider.snapshot()
    finally:
        await http_client.aclose()

    assert snapshot.components["postgres"].ok is True


@pytest.mark.asyncio
async def test_injected_clients_are_borrowed() -> None:
    database = FakeDatabase()
    redis_client = FakeRedis()
    minio_client = FakeMinio()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200)))
    provider = LiveReadinessProvider(
        Settings(_env_file=None),
        database=database,
        redis_client=redis_client,
        minio_client=minio_client,
        http_client=http_client,
    )

    try:
        await provider.close()

        assert database.close_calls == 0
        assert redis_client.close_calls == 0
        assert http_client.is_closed is False
    finally:
        await http_client.aclose()


class TrackingPool:
    def __init__(self, clear_error: BaseException | None = None) -> None:
        self.clear_calls = 0
        self.clear_error = clear_error

    def clear(self) -> None:
        self.clear_calls += 1
        if self.clear_error is not None:
            raise self.clear_error


@pytest.mark.asyncio
async def test_minio_pool_is_cleared_when_client_construction_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pool = TrackingPool()

    def make_pool(**_kwargs: object) -> TrackingPool:
        return pool

    def fail_minio(*_args: object, **_kwargs: object) -> FakeMinio:
        raise RuntimeError("MinIO construction failed")

    http_client = httpx.AsyncClient()
    monkeypatch.setattr(probes, "PoolManager", make_pool)
    monkeypatch.setattr(probes, "Minio", fail_minio)
    try:
        with pytest.raises(ExceptionGroup) as captured:
            await LiveReadinessProvider.create(
                Settings(_env_file=None),
                database=FakeDatabase(),
                redis_client=FakeRedis(),
                http_client=http_client,
            )
    finally:
        await http_client.aclose()

    assert pool.clear_calls == 1
    assert [str(error) for error in captured.value.exceptions] == [
        "minio construction failed with RuntimeError"
    ]


class OwnedDatabase(FakeDatabase):
    def __init__(self, close_error: BaseException | None = None) -> None:
        super().__init__()
        self.close_error = close_error

    async def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class OwnedRedis(FakeRedis):
    def __init__(self, close_error: BaseException | None = None) -> None:
        super().__init__()
        self.close_error = close_error

    async def aclose(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class CancellationResistantOwnedRedis(OwnedRedis):
    def __init__(self, close_error: BaseException | None = None) -> None:
        super().__init__(close_error)
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.finished = asyncio.Event()

    async def aclose(self) -> None:
        self.close_calls += 1
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            await self.release.wait()
        self.finished.set()
        if self.close_error is not None:
            raise self.close_error


class OwnedHttp:
    def __init__(self, close_error: BaseException | None = None) -> None:
        self.close_error = close_error
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class BlockingOwnedDatabase(OwnedDatabase):
    def __init__(self, close_error: BaseException | None = None) -> None:
        super().__init__(close_error)
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.was_cancelled = False

    async def close(self) -> None:
        self.close_calls += 1
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.was_cancelled = True
            raise
        if self.close_error is not None:
            raise self.close_error


async def make_owned_provider(
    monkeypatch: pytest.MonkeyPatch,
    *,
    database: OwnedDatabase,
    redis_client: OwnedRedis,
    http_client: OwnedHttp,
    minio_pool: TrackingPool,
) -> LiveReadinessProvider:
    class DatabaseFactory:
        @classmethod
        def from_settings(cls, _settings: Settings) -> OwnedDatabase:
            return database

    class RedisFactory:
        @classmethod
        def from_url(cls, *_args: object, **_kwargs: object) -> OwnedRedis:
            return redis_client

    def make_pool(**_kwargs: object) -> TrackingPool:
        return minio_pool

    def make_minio(*_args: object, **_kwargs: object) -> FakeMinio:
        return FakeMinio()

    def make_http_client(**_kwargs: object) -> OwnedHttp:
        return http_client

    monkeypatch.setattr(probes, "Database", DatabaseFactory)
    monkeypatch.setattr(probes, "Redis", RedisFactory)
    monkeypatch.setattr(probes, "PoolManager", make_pool)
    monkeypatch.setattr(probes, "Minio", make_minio)
    monkeypatch.setattr(httpx, "AsyncClient", make_http_client)
    return await LiveReadinessProvider.create(Settings(_env_file=None))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failed_stage", "expected_calls", "expected_redis_closes", "expected_pool_clears"),
    [
        ("redis", (1, 0, 0, 0), 0, 0),
        ("minio_pool", (1, 1, 0, 0), 1, 0),
        ("minio", (1, 1, 1, 0), 1, 1),
        ("http", (1, 1, 1, 1), 1, 1),
    ],
)
async def test_create_rolls_back_each_partially_constructed_resource_once(
    monkeypatch: pytest.MonkeyPatch,
    failed_stage: str,
    expected_calls: tuple[int, int, int, int],
    expected_redis_closes: int,
    expected_pool_clears: int,
) -> None:
    calls = {"redis": 0, "minio_pool": 0, "minio": 0, "http": 0}
    redis_client = OwnedRedis()
    minio_pool = TrackingPool()

    class RedisFactory:
        @classmethod
        def from_url(cls, *_args: object, **_kwargs: object) -> OwnedRedis:
            calls["redis"] += 1
            if failed_stage == "redis":
                raise RuntimeError("redis-construction-secret")
            return redis_client

    def make_pool(**_kwargs: object) -> TrackingPool:
        calls["minio_pool"] += 1
        if failed_stage == "minio_pool":
            raise RuntimeError("pool-construction-secret")
        return minio_pool

    def make_minio(*_args: object, **_kwargs: object) -> FakeMinio:
        calls["minio"] += 1
        if failed_stage == "minio":
            raise RuntimeError("minio-construction-secret")
        return FakeMinio()

    def make_http_client(**_kwargs: object) -> OwnedHttp:
        calls["http"] += 1
        if failed_stage == "http":
            raise RuntimeError("http-construction-secret")
        return OwnedHttp()

    monkeypatch.setattr(probes, "Redis", RedisFactory)
    monkeypatch.setattr(probes, "PoolManager", make_pool)
    monkeypatch.setattr(probes, "Minio", make_minio)
    monkeypatch.setattr(httpx, "AsyncClient", make_http_client)

    with pytest.raises(ExceptionGroup) as captured:
        await LiveReadinessProvider.create(
            Settings(_env_file=None),
            database=FakeDatabase(),
        )

    assert tuple(calls.values()) == expected_calls
    assert redis_client.close_calls == expected_redis_closes
    assert minio_pool.clear_calls == expected_pool_clears
    assert [str(error) for error in captured.value.exceptions] == [
        f"{failed_stage} construction failed with RuntimeError"
    ]
    assert "construction-secret" not in exception_surface(captured.value)
    assert "construction-secret" not in traceback_frame_surface(captured.value, "create")


@pytest.mark.asyncio
async def test_construction_and_rollback_failures_expose_no_secret_details(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis_client = OwnedRedis(ValueError("redis-cleanup-secret"))
    minio_pool = TrackingPool(PermissionError("pool-cleanup-secret"))

    class RedisFactory:
        @classmethod
        def from_url(cls, *_args: object, **_kwargs: object) -> OwnedRedis:
            return redis_client

    def make_pool(**_kwargs: object) -> TrackingPool:
        return minio_pool

    def fail_minio(*_args: object, **_kwargs: object) -> FakeMinio:
        raise RuntimeError("minio-construction-secret")

    monkeypatch.setattr(probes, "Redis", RedisFactory)
    monkeypatch.setattr(probes, "PoolManager", make_pool)
    monkeypatch.setattr(probes, "Minio", fail_minio)

    with pytest.raises(ExceptionGroup) as captured:
        await LiveReadinessProvider.create(
            Settings(_env_file=None),
            database=FakeDatabase(),
        )

    assert {str(error) for error in captured.value.exceptions} == {
        "minio construction failed with RuntimeError",
        "redis cleanup failed with ValueError",
        "minio cleanup failed with PermissionError",
    }
    surface = exception_surface(captured.value)
    assert "minio-construction-secret" not in surface
    assert "redis-cleanup-secret" not in surface
    assert "pool-cleanup-secret" not in surface
    create_frame = traceback_frame_surface(captured.value, "create")
    assert "minio-construction-secret" not in create_frame
    assert "redis-cleanup-secret" not in create_frame
    assert "pool-cleanup-secret" not in create_frame
    assert captured.value.__cause__ is None
    assert captured.value.__context__ is None
    assert all(error.__cause__ is None for error in captured.value.exceptions)
    assert all(error.__context__ is None for error in captured.value.exceptions)
    assert all(error.__traceback__ is None for error in captured.value.exceptions)


@pytest.mark.asyncio
async def test_construction_failure_never_closes_injected_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    injected_database = FakeDatabase()
    injected_redis = OwnedRedis()
    minio_pool = TrackingPool()

    monkeypatch.setattr(probes, "PoolManager", lambda **_kwargs: minio_pool)
    monkeypatch.setattr(probes, "Minio", lambda *_args, **_kwargs: FakeMinio())

    def fail_http(**_kwargs: object) -> OwnedHttp:
        raise RuntimeError("http-construction-secret")

    monkeypatch.setattr(httpx, "AsyncClient", fail_http)

    with pytest.raises(ExceptionGroup):
        await LiveReadinessProvider.create(
            Settings(_env_file=None),
            database=injected_database,
            redis_client=injected_redis,
        )

    assert injected_database.close_calls == 0
    assert injected_redis.close_calls == 0
    assert minio_pool.clear_calls == 1


@pytest.mark.asyncio
async def test_cancelled_construction_rollback_is_bounded_and_attempts_later_closers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = OwnedDatabase()
    redis_client = CancellationResistantOwnedRedis(RuntimeError("late-redis-cleanup-secret"))
    minio_pool = TrackingPool()

    class DatabaseFactory:
        @classmethod
        def from_settings(cls, _settings: Settings) -> OwnedDatabase:
            return database

    class RedisFactory:
        @classmethod
        def from_url(cls, *_args: object, **_kwargs: object) -> CancellationResistantOwnedRedis:
            return redis_client

    def fail_minio(*_args: object, **_kwargs: object) -> FakeMinio:
        raise RuntimeError("minio-construction-secret")

    monkeypatch.setattr(probes, "Database", DatabaseFactory)
    monkeypatch.setattr(probes, "Redis", RedisFactory)
    monkeypatch.setattr(probes, "PoolManager", lambda **_kwargs: minio_pool)
    monkeypatch.setattr(probes, "Minio", fail_minio)
    create_task = asyncio.create_task(
        LiveReadinessProvider.create(Settings(_env_file=None, shutdown_timeout_seconds=0.01))
    )

    try:
        async with asyncio.timeout(0.2):
            await redis_client.started.wait()
        assert minio_pool.clear_calls == 1
        assert database.close_calls == 0
        create_task.cancel("original construction cancellation")
        await asyncio.sleep(0)
        create_task.cancel("repeated construction cancellation")

        async with asyncio.timeout(0.2):
            with pytest.raises(BaseExceptionGroup) as captured:
                await create_task

        cancellation, construction_group = captured.value.exceptions
        assert isinstance(cancellation, asyncio.CancelledError)
        assert cancellation.args == ("original construction cancellation",)
        assert isinstance(construction_group, ExceptionGroup)
        assert [str(error) for error in construction_group.exceptions] == [
            "minio construction failed with RuntimeError",
            "redis cleanup failed with TimeoutError",
        ]
        assert database.close_calls == 1
        assert redis_client.close_calls == 1
        assert minio_pool.clear_calls == 1
        assert redis_client.finished.is_set() is False
        surface = exception_surface(captured.value)
        assert "minio-construction-secret" not in surface
        assert "late-redis-cleanup-secret" not in surface
    finally:
        redis_client.release.set()
        await asyncio.gather(create_task, return_exceptions=True)
        async with asyncio.timeout(0.2):
            await redis_client.finished.wait()
        await asyncio.sleep(0)


def exception_surface(error: BaseException) -> str:
    values = [str(error), repr(error)]
    if isinstance(error, BaseExceptionGroup):
        values.extend(exception_surface(nested) for nested in error.exceptions)
    if error.__cause__ is not None:
        values.append(exception_surface(error.__cause__))
    if error.__context__ is not None:
        values.append(exception_surface(error.__context__))
    return "\n".join(values)


def traceback_frame_surface(error: BaseException, frame_name: str) -> str:
    values: list[str] = []
    traceback = error.__traceback__
    while traceback is not None:
        frame = traceback.tb_frame
        if frame.f_code.co_name == frame_name:
            values.extend(repr(value) for value in frame.f_locals.values())
        traceback = traceback.tb_next
    return "\n".join(values)


@pytest.mark.asyncio
async def test_cancelling_close_caller_does_not_cancel_owned_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = BlockingOwnedDatabase()
    redis_client = OwnedRedis()
    http_client = OwnedHttp()
    minio_pool = TrackingPool()
    provider = await make_owned_provider(
        monkeypatch,
        database=database,
        redis_client=redis_client,
        http_client=http_client,
        minio_pool=minio_pool,
    )

    first_caller = asyncio.create_task(provider.close())
    async with asyncio.timeout(0.2):
        await database.started.wait()
    first_caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_caller

    second_caller = asyncio.create_task(provider.close())
    database.release.set()
    async with asyncio.timeout(0.2):
        await second_caller

    assert database.was_cancelled is False
    assert database.close_calls == 1
    assert redis_client.close_calls == 1
    assert http_client.close_calls == 1
    assert minio_pool.clear_calls == 1


@pytest.mark.asyncio
async def test_child_cancelled_error_becomes_sanitized_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = OwnedDatabase(asyncio.CancelledError("sensitive-sentinel"))
    redis_client = OwnedRedis()
    http_client = OwnedHttp()
    minio_pool = TrackingPool()
    provider = await make_owned_provider(
        monkeypatch,
        database=database,
        redis_client=redis_client,
        http_client=http_client,
        minio_pool=minio_pool,
    )

    with pytest.raises(ExceptionGroup) as captured:
        await provider.close()

    assert "sensitive-sentinel" not in exception_surface(captured.value)
    assert [str(error) for error in captured.value.exceptions] == [
        "postgres cleanup failed with CancelledError"
    ]
    assert minio_pool.clear_calls == 1


@pytest.mark.asyncio
async def test_cleanup_failures_expose_only_safe_resource_and_class_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinels = {
        "postgres-secret-dsn",
        "redis-secret-url",
        "http-secret-response",
        "minio-secret-credential",
    }
    database = OwnedDatabase(RuntimeError("postgres-secret-dsn"))
    redis_client = OwnedRedis(ValueError("redis-secret-url"))
    http_client = OwnedHttp(OSError("http-secret-response"))
    minio_pool = TrackingPool(PermissionError("minio-secret-credential"))
    provider = await make_owned_provider(
        monkeypatch,
        database=database,
        redis_client=redis_client,
        http_client=http_client,
        minio_pool=minio_pool,
    )

    with pytest.raises(ExceptionGroup) as captured:
        await provider.close()

    group = captured.value
    surface = exception_surface(group)
    assert all(sentinel not in surface for sentinel in sentinels)
    cleanup_frame = traceback_frame_surface(group, "_close_owned_resources")
    assert all(sentinel not in cleanup_frame for sentinel in sentinels)
    assert {str(error) for error in group.exceptions} == {
        "postgres cleanup failed with RuntimeError",
        "redis cleanup failed with ValueError",
        "http cleanup failed with OSError",
        "minio cleanup failed with PermissionError",
    }
    assert group.__cause__ is None
    assert group.__context__ is None
    assert all(error.__cause__ is None for error in group.exceptions)
    assert all(error.__context__ is None for error in group.exceptions)
    assert all(error.__traceback__ is None for error in group.exceptions)
    assert all(type(error).__module__ == probes.__name__ for error in group.exceptions)


@pytest.mark.asyncio
async def test_concurrent_and_repeated_close_share_cached_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = BlockingOwnedDatabase(RuntimeError("sensitive-close-detail"))
    redis_client = OwnedRedis()
    http_client = OwnedHttp()
    minio_pool = TrackingPool()
    provider = await make_owned_provider(
        monkeypatch,
        database=database,
        redis_client=redis_client,
        http_client=http_client,
        minio_pool=minio_pool,
    )

    first_caller = asyncio.create_task(provider.close())
    second_caller = asyncio.create_task(provider.close())
    async with asyncio.timeout(0.2):
        await database.started.wait()
    database.release.set()
    results = await asyncio.gather(first_caller, second_caller, return_exceptions=True)

    failures = [result for result in results if isinstance(result, ExceptionGroup)]
    assert len(failures) == 2
    assert failures[0] is failures[1]
    with pytest.raises(ExceptionGroup) as repeated:
        await provider.close()
    assert repeated.value is failures[0]
    assert "sensitive-close-detail" not in exception_surface(repeated.value)
    assert database.close_calls == 1
    assert redis_client.close_calls == 1
    assert http_client.close_calls == 1
    assert minio_pool.clear_calls == 1


class ProbeCoordinator:
    def __init__(self) -> None:
        self.started: set[str] = set()
        self.all_started = asyncio.Event()
        self.release = asyncio.Event()

    async def start(self, component: str) -> None:
        self.started.add(component)
        if len(self.started) == 4:
            self.all_started.set()
        await self.release.wait()


class CoordinatedDatabase(FakeDatabase):
    def __init__(self, coordinator: ProbeCoordinator) -> None:
        super().__init__()
        self.coordinator = coordinator

    async def ping(self) -> None:
        await self.coordinator.start("postgres")


class CoordinatedProvider(LiveReadinessProvider):
    def __init__(
        self,
        coordinator: ProbeCoordinator,
        settings: Settings,
        *,
        database: CoordinatedDatabase,
        redis_client: FakeRedis,
        minio_client: FakeMinio,
        http_client: httpx.AsyncClient,
    ) -> None:
        self.coordinator = coordinator
        super().__init__(
            settings,
            database=database,
            redis_client=redis_client,
            minio_client=minio_client,
            http_client=http_client,
        )

    async def _check_qdrant(self) -> None:
        await self.coordinator.start("qdrant")

    async def _check_redis(self) -> None:
        await self.coordinator.start("redis")

    async def _check_minio(self) -> None:
        await self.coordinator.start("minio")


@pytest.mark.asyncio
async def test_live_provider_starts_all_probes_concurrently() -> None:
    coordinator = ProbeCoordinator()
    http_client = httpx.AsyncClient()
    provider = CoordinatedProvider(
        coordinator,
        Settings(_env_file=None),
        database=CoordinatedDatabase(coordinator),
        redis_client=FakeRedis(),
        minio_client=FakeMinio(),
        http_client=http_client,
    )
    snapshot_task = asyncio.create_task(provider.snapshot())

    try:
        async with asyncio.timeout(0.2):
            await coordinator.all_started.wait()
        assert coordinator.started == {"postgres", "qdrant", "redis", "minio"}
        coordinator.release.set()
        snapshot = await snapshot_task
        await provider.close()
    finally:
        coordinator.release.set()
        if not snapshot_task.done():
            snapshot_task.cancel()
        await asyncio.gather(snapshot_task, return_exceptions=True)
        await http_client.aclose()

    assert all(component.ok for component in snapshot.components.values())
