"""Administrator-only provider embedding dimension discovery."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.api.errors import BusinessError
from rag_service.api.middleware import is_valid_request_id
from rag_service.auth.policies import AdminPrincipal
from rag_service.config import Settings
from rag_service.db.models.auth import AuditEvent
from rag_service.providers.embeddings import (
    EmbeddingDimensionProbeRequest,
    EmbeddingGateway,
    EmbeddingGatewayError,
    EmbeddingProbeOperationalConfig,
)
from rag_service.providers.repositories import (
    AdminActorRecord,
    ProviderConfigRecord,
    ProviderConfigSecretSourceRecord,
    ProviderCredentialRecord,
    ProviderRepositories,
    sqlalchemy_provider_repositories,
)
from rag_service.providers.schemas import EmbeddingProbeCreate, SafeEmbeddingProbe
from rag_service.providers.services import (
    CredentialProviderSecretSource,
    LegacyProviderSecretSource,
    select_provider_config_secret_source,
)

type RepositoryFactory = Callable[[AsyncSession], ProviderRepositories]
type Clock = Callable[[], datetime]

_PROBE_INPUT = "RAG embedding configuration dimension probe."
_SUPPORTED_PROVIDER_TYPES = frozenset({"openai_compatible", "openrouter"})
_GATEWAY_MESSAGES = {
    "EMBEDDING_BATCH_INVALID": "Embedding batch is invalid",
    "EMBEDDING_CONFIGURATION_INVALID": "Embedding configuration is invalid",
    "EMBEDDING_INPUT_INVALID": "Embedding input is invalid",
    "MODEL_PROFILE_DISABLED": "Model profile is disabled",
    "PROVIDER_AUTHENTICATION_FAILED": "Provider authentication failed",
    "PROVIDER_CREDENTIAL_INVALID": "Provider credential is invalid",
    "PROVIDER_CREDENTIAL_UNAVAILABLE": "Provider credential unavailable",
    "PROVIDER_DISABLED": "Provider is disabled",
    "PROVIDER_ENDPOINT_REJECTED": "Provider endpoint rejected",
    "PROVIDER_INPUT_REJECTED": "Provider input rejected",
    "PROVIDER_MODEL_NOT_FOUND": "Provider model not found",
    "PROVIDER_RATE_LIMITED": "Provider rate limited",
    "PROVIDER_REDIRECT_REJECTED": "Provider redirect rejected",
    "PROVIDER_REQUEST_REJECTED": "Provider request rejected",
    "PROVIDER_RESPONSE_COUNT_MISMATCH": "Provider response count mismatch",
    "PROVIDER_RESPONSE_DIMENSION_MISMATCH": "Provider response dimension mismatch",
    "PROVIDER_RESPONSE_INVALID": "Provider response is invalid",
    "PROVIDER_RESPONSE_NONFINITE": "Provider response is invalid",
    "PROVIDER_TIMEOUT": "Provider request timed out",
    "PROVIDER_UNAVAILABLE": "Provider unavailable",
}


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _invalid_api_key_error() -> BusinessError:
    return BusinessError(401, "INVALID_API_KEY", "Invalid API key")


def _not_found_error() -> BusinessError:
    return BusinessError(404, "RESOURCE_NOT_FOUND", "Resource not found")


def _validation_error() -> BusinessError:
    return BusinessError(422, "VALIDATION_ERROR", "Invalid embedding probe request")


def _disabled_error() -> BusinessError:
    return BusinessError(409, "PROVIDER_CONFIG_DISABLED", "Provider configuration is disabled")


def _unsupported_provider_error() -> BusinessError:
    return BusinessError(422, "PROVIDER_TYPE_UNSUPPORTED", "Provider type is unsupported")


def _legacy_credential_error() -> BusinessError:
    return BusinessError(
        422,
        "PROVIDER_CREDENTIAL_UNSUPPORTED",
        "Provider credential source is unsupported",
    )


def _invalid_credential_error() -> BusinessError:
    return BusinessError(422, "PROVIDER_CREDENTIAL_INVALID", "Provider credential is invalid")


def _internal_error() -> BusinessError:
    return BusinessError(500, "INTERNAL_ERROR", "Internal server error")


def _active(row: AdminActorRecord, now: datetime) -> bool:
    if row.status != "active" or row.revoked_at is not None:
        return False
    if row.not_before is not None and row.not_before > now:
        return False
    return row.expires_at is None or row.expires_at > now


def _gateway_error(error: EmbeddingGatewayError) -> BusinessError:
    code = (
        error.code
        if type(error.code) is str and error.code in _GATEWAY_MESSAGES
        else "PROVIDER_FAILURE"
    )
    message = _GATEWAY_MESSAGES.get(code, "Provider request failed")
    retryable = error.retryable is True
    return BusinessError(503 if retryable else 422, code, message, retryable=retryable)


@dataclass(frozen=True, slots=True)
class _ProbeSnapshot:
    request: EmbeddingDimensionProbeRequest
    operational: EmbeddingProbeOperationalConfig
    model_name: str


@dataclass(frozen=True, slots=True, repr=False)
class _TrustedPreflightFailure:
    error: BusinessError

    def __repr__(self) -> str:
        return "_TrustedPreflightFailure(<redacted>)"


class EmbeddingProbeService:
    def __init__(
        self,
        *,
        session: AsyncSession,
        settings: Settings,
        embedding_gateway: EmbeddingGateway,
        repository_factory: RepositoryFactory = sqlalchemy_provider_repositories,
        clock: Clock = _utc_now,
    ) -> None:
        self._session = session
        self._settings = settings
        self._embedding_gateway = embedding_gateway
        self._repository_factory = repository_factory
        self._clock = clock

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() != UTC.utcoffset(value):
            raise _internal_error()
        return value

    async def _require_admin(
        self,
        repositories: ProviderRepositories,
        actor: AdminPrincipal,
    ) -> None:
        if type(actor) is not AdminPrincipal:
            raise _invalid_api_key_error()
        row = await repositories.admins.get_for_update(actor.key_id)
        if (
            row is None
            or row.public_id != actor.public_id
            or row.key_type != "admin"
            or not _active(row, self._now())
        ):
            raise _invalid_api_key_error()

    async def _snapshot(
        self,
        provider_config_id: UUID,
        command: EmbeddingProbeCreate,
        actor: AdminPrincipal,
    ) -> _ProbeSnapshot | _TrustedPreflightFailure:
        repositories: ProviderRepositories | None = None
        row: ProviderConfigRecord | None = None
        source_record: ProviderConfigSecretSourceRecord | None = None
        source: CredentialProviderSecretSource | LegacyProviderSecretSource | None = None
        credential: ProviderCredentialRecord | None = None
        credential_id: UUID | None = None
        request: EmbeddingDimensionProbeRequest | None = None
        operational: EmbeddingProbeOperationalConfig | None = None
        snapshot: _ProbeSnapshot | None = None
        failure: _TrustedPreflightFailure | None = None
        actor_trusted = False
        try:
            async with self._session.begin():
                repositories = self._repository_factory(self._session)
                await self._require_admin(repositories, actor)
                actor_trusted = True
                if repositories.configs is None:
                    raise _internal_error()
                row = await repositories.configs.get_safe(provider_config_id)
                if row is None:
                    raise _not_found_error()
                if not row.enabled:
                    raise _disabled_error()
                if row.provider_type not in _SUPPORTED_PROVIDER_TYPES:
                    raise _unsupported_provider_error()
                source_record = await repositories.configs.get_secret_source(provider_config_id)
                if source_record is None:
                    raise _invalid_credential_error()
                source = select_provider_config_secret_source(source_record)
                source_record = None
                if isinstance(source, LegacyProviderSecretSource):
                    source = None
                    raise _legacy_credential_error()
                if not isinstance(source, CredentialProviderSecretSource):
                    source = None
                    raise _internal_error()
                credential_id = source.credential_id
                source = None
                if row.credential_id is None or row.credential_id != credential_id:
                    raise _invalid_credential_error()
                credential = await repositories.credentials.get_safe(credential_id)
                if credential is None or credential.id != credential_id:
                    credential = None
                    raise _invalid_credential_error()
                credential = None
                request = EmbeddingDimensionProbeRequest(
                    adapter_schema_version="openai-embeddings-v1",
                    provider_type=row.provider_type,
                    base_url=row.base_url,
                    credential_id=credential_id,
                    default_headers=deepcopy(row.default_headers),
                    routing_options=deepcopy(row.routing_options),
                    model_name=command.model_name,
                )
                operational = EmbeddingProbeOperationalConfig(
                    provider_config_id=provider_config_id,
                    provider_enabled=row.enabled,
                    timeout_seconds=row.timeout_seconds,
                    max_concurrency=row.max_concurrency,
                    requests_per_minute=row.requests_per_minute,
                    batch_size=1,
                )
                snapshot = _ProbeSnapshot(
                    request=request,
                    operational=operational,
                    model_name=request.model_name,
                )
            return snapshot
        except BusinessError as error:
            if actor_trusted:
                failure = _TrustedPreflightFailure(error=error)
                return failure
            raise
        except Exception:
            if actor_trusted:
                failure = _TrustedPreflightFailure(error=_internal_error())
                return failure
            raise _internal_error() from None
        finally:
            repositories = None
            row = None
            source_record = None
            source = None
            credential = None
            credential_id = None
            request = None
            operational = None
            snapshot = None
            failure = None
            actor_trusted = False

    async def _audit(
        self,
        *,
        provider_config_id: UUID,
        actor: AdminPrincipal,
        request_id: str,
        model_name: str,
        dimension: int | None,
        error_code: str | None,
    ) -> None:
        repositories: ProviderRepositories | None = None
        event: AuditEvent | None = None
        metadata: dict[str, object] = {
            "request_id": request_id,
            "actor_key_id": str(actor.key_id),
            "provider_config_id": str(provider_config_id),
            "model_name": model_name,
            "outcome": "succeeded" if error_code is None else "failed",
        }
        if error_code is None:
            metadata["dimension"] = dimension
        else:
            metadata["error_code"] = error_code
        try:
            async with self._session.begin():
                repositories = self._repository_factory(self._session)
                event = AuditEvent(
                    id=uuid4(),
                    request_id=request_id,
                    actor_api_key_id=actor.key_id,
                    actor_kind="admin_key",
                    action="provider_config.embedding_dimension_probed",
                    target_type="provider_config",
                    target_id=provider_config_id,
                    metadata_=dict(metadata),
                )
                await repositories.audits.add(event)
        finally:
            metadata.clear()
            repositories = None
            event = None

    async def probe(
        self,
        provider_config_id: UUID,
        command: EmbeddingProbeCreate,
        *,
        actor: AdminPrincipal,
        request_id: str,
    ) -> SafeEmbeddingProbe:
        if type(provider_config_id) is not UUID or type(command) is not EmbeddingProbeCreate:
            raise _validation_error()
        if not is_valid_request_id(request_id, self._settings.max_request_id_length):
            raise _validation_error()
        snapshot: _ProbeSnapshot | None = None
        preflight: _ProbeSnapshot | _TrustedPreflightFailure | None = None
        result: SafeEmbeddingProbe | None = None
        mapped_error: BusinessError | None = None
        dimension: int | None = None
        cancelled = False
        try:
            preflight = await self._snapshot(provider_config_id, command, actor)
            if isinstance(preflight, _TrustedPreflightFailure):
                mapped_error = preflight.error
                try:
                    await self._audit(
                        provider_config_id=provider_config_id,
                        actor=actor,
                        request_id=request_id,
                        model_name=command.model_name,
                        dimension=None,
                        error_code=mapped_error.code,
                    )
                except asyncio.CancelledError:
                    raise asyncio.CancelledError() from None
                except Exception:
                    pass
                raise mapped_error from None
            snapshot = preflight
            try:
                dimension = await self._embedding_gateway.probe_dimension(
                    request=snapshot.request,
                    operational=snapshot.operational,
                    input_text=_PROBE_INPUT,
                )
                result = SafeEmbeddingProbe(
                    provider_config_id=provider_config_id,
                    model_name=snapshot.model_name,
                    dimension=dimension,
                )
            except asyncio.CancelledError:
                cancelled = True
            except EmbeddingGatewayError as error:
                mapped_error = _gateway_error(error)
            except Exception:
                mapped_error = _internal_error()
            if cancelled:
                raise asyncio.CancelledError() from None
            error_code = None if mapped_error is None else mapped_error.code
            try:
                await self._audit(
                    provider_config_id=provider_config_id,
                    actor=actor,
                    request_id=request_id,
                    model_name=snapshot.model_name,
                    dimension=dimension,
                    error_code=error_code,
                )
            except asyncio.CancelledError:
                raise asyncio.CancelledError() from None
            except Exception:
                if mapped_error is None:
                    mapped_error = _internal_error()
            if mapped_error is not None:
                raise mapped_error from None
            if result is None:
                raise _internal_error()
            return result
        except asyncio.CancelledError:
            cancelled = True
        except BusinessError:
            raise
        except Exception:
            raise _internal_error() from None
        finally:
            command = EmbeddingProbeCreate(model_name="<redacted>")
            actor = cast(AdminPrincipal, None)
            request_id = "<redacted>"
            snapshot = None
            preflight = None
            result = None
            mapped_error = None
            dimension = None
        if cancelled:
            raise asyncio.CancelledError() from None
        raise AssertionError("unreachable")


__all__ = ["EmbeddingProbeService", "_PROBE_INPUT"]
