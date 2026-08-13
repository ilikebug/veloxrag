import asyncio
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Query
from fastapi.responses import JSONResponse
from pydantic import BeforeValidator, SecretStr, WithJsonSchema
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.api.errors import error_responses
from rag_service.api.middleware import get_request_id
from rag_service.auth.dependencies import require_admin_principal
from rag_service.auth.policies import AdminPrincipal
from rag_service.config import Settings, get_settings
from rag_service.db.dependencies import get_session
from rag_service.providers.embedding_probe import EmbeddingProbeService
from rag_service.providers.embeddings import EmbeddingGateway
from rag_service.providers.gateway_provider import get_embedding_gateway
from rag_service.providers.schemas import (
    EmbeddingProbeCreate,
    ModelProfileCreate,
    ModelProfilePage,
    ModelProfilePatch,
    ProviderConfigCreate,
    ProviderConfigPage,
    ProviderConfigPatch,
    ProviderCredentialCreate,
    ProviderCredentialPage,
    ProviderCredentialPatch,
    SafeEmbeddingProbe,
    SafeModelProfile,
    SafeProviderConfig,
    SafeProviderCredential,
)
from rag_service.providers.services import (
    ModelProfileService,
    ProviderConfigService,
    ProviderCredentialService,
    model_profile_etag,
    provider_config_etag,
    provider_credential_etag,
    provider_credential_keyring_from_settings,
    provider_endpoint_policy_from_settings,
)

router = APIRouter()
credential_router = APIRouter(
    prefix="/v1/admin/provider-credentials",
    tags=["administrator", "provider credentials"],
)
config_router = APIRouter(
    prefix="/v1/admin/provider-configs",
    tags=["administrator", "provider configurations"],
)
profile_router = APIRouter(
    prefix="/v1/admin/model-profiles",
    tags=["administrator", "model profiles"],
)
_NO_STORE = "no-store"
_CACHE_CONTROL_HEADER = {
    "description": "Prevents storage of provider administration responses.",
    "schema": {"type": "string"},
}
_ETAG_HEADER = {
    "description": "Current provider resource entity tag.",
    "schema": {"type": "string"},
}
_LOCATION_HEADER = {
    "description": "Canonical provider resource path.",
    "schema": {"type": "string"},
}
_SAFE_HEADERS = {
    "ETag": _ETAG_HEADER,
    "Cache-Control": _CACHE_CONTROL_HEADER,
}
_CREATE_HEADERS = {
    **_SAFE_HEADERS,
    "Location": _LOCATION_HEADER,
}


def _canonical_uuid(value: object) -> UUID:
    if type(value) is not str:
        raise ValueError("UUID must use canonical text form")
    try:
        parsed = UUID(value)
    except ValueError:
        raise ValueError("UUID must use canonical text form") from None
    if str(parsed) != value:
        raise ValueError("UUID must use canonical text form")
    return parsed


CanonicalUuid = Annotated[
    UUID,
    BeforeValidator(_canonical_uuid),
    WithJsonSchema(
        {
            "type": "string",
            "format": "uuid",
            "pattern": ("^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"),
        }
    ),
]


def _redacted_create() -> ProviderCredentialCreate:
    return ProviderCredentialCreate(name="<redacted>", secret=SecretStr("<redacted>"))


def _redacted_patch() -> ProviderCredentialPatch:
    return ProviderCredentialPatch(name="<redacted>")


def _redacted_config_create() -> ProviderConfigCreate:
    return ProviderConfigCreate(
        name="<redacted>",
        provider_type="openai_compatible",
        base_url="https://redacted.invalid",
        credential_id=UUID(int=0),
        timeout_seconds=1,
        max_concurrency=1,
        requests_per_minute=1,
    )


def _redacted_config_patch() -> ProviderConfigPatch:
    return ProviderConfigPatch(enabled=False)


def _redacted_profile_create() -> ModelProfileCreate:
    return ModelProfileCreate(
        name="<redacted>",
        capability="embedding",
        provider_config_id=UUID(int=0),
        model_name="<redacted>",
        dimension=1,
        max_input_tokens=1,
        batch_size=1,
        timeout_seconds=1,
        vector_config={},
        enabled=False,
    )


def _redacted_profile_patch() -> ModelProfilePatch:
    return ModelProfilePatch(enabled=False)


def _redacted_probe_create() -> EmbeddingProbeCreate:
    return EmbeddingProbeCreate(model_name="<redacted>")


async def get_provider_credential_service(
    session: Annotated[AsyncSession, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ProviderCredentialService:
    return ProviderCredentialService(
        session=session,
        settings=settings,
        keyring_factory=lambda: provider_credential_keyring_from_settings(settings),
    )


async def get_provider_config_service(
    session: Annotated[AsyncSession, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ProviderConfigService:
    return ProviderConfigService(
        session=session,
        settings=settings,
        endpoint_policy_factory=lambda: provider_endpoint_policy_from_settings(settings),
    )


async def get_model_profile_service(
    session: Annotated[AsyncSession, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ModelProfileService:
    return ModelProfileService(session=session, settings=settings)


async def get_embedding_probe_service(
    session: Annotated[AsyncSession, Depends(get_session)],
    settings: Annotated[Settings, Depends(get_settings)],
    embedding_gateway: Annotated[
        EmbeddingGateway,
        Depends(get_embedding_gateway),
    ],
) -> EmbeddingProbeService:
    return EmbeddingProbeService(
        session=session,
        settings=settings,
        embedding_gateway=embedding_gateway,
    )


def _safe_response(
    document: (
        SafeProviderCredential
        | ProviderCredentialPage
        | SafeProviderConfig
        | ProviderConfigPage
        | SafeEmbeddingProbe
        | SafeModelProfile
        | ModelProfilePage
    ),
    *,
    etag: str | None = None,
    location: str | None = None,
    status_code: int = 200,
) -> JSONResponse:
    headers = {"Cache-Control": _NO_STORE}
    if etag is not None:
        headers["ETag"] = etag
    if location is not None:
        headers["Location"] = location
    return JSONResponse(
        status_code=status_code,
        headers=headers,
        content=document.model_dump(mode="json"),
    )


@credential_router.post(
    "",
    status_code=201,
    response_model=SafeProviderCredential,
    responses={
        201: {
            "model": SafeProviderCredential,
            "description": "A provider credential was created.",
            "headers": _CREATE_HEADERS,
        },
        200: {
            "model": SafeProviderCredential,
            "description": "An equivalent idempotent create was replayed.",
            "headers": _CREATE_HEADERS,
        },
        **error_responses(401, 409, 422, 500, 503),
    },
)
async def create_provider_credential(
    command: ProviderCredentialCreate,
    request_id: Annotated[str, Depends(get_request_id)],
    actor: Annotated[AdminPrincipal, Depends(require_admin_principal)],
    service: Annotated[
        ProviderCredentialService,
        Depends(get_provider_credential_service),
    ],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key")],
) -> JSONResponse:
    result = None
    safe = None
    response = None
    cancelled = False
    try:
        result = await service.create_credential(
            command,
            actor=actor,
            request_id=request_id,
            idempotency_key=idempotency_key,
        )
        safe = result.credential
        response = _safe_response(
            safe,
            status_code=201 if result.created else 200,
            etag=provider_credential_etag(safe.id, safe.resource_revision),
            location=f"/v1/admin/provider-credentials/{safe.id}",
        )
        return response
    except asyncio.CancelledError:
        cancelled = True
    finally:
        command = _redacted_create()
        result = None
        safe = None
        response = None
        request_id = "<redacted>"
        idempotency_key = "<redacted>"
    if cancelled:
        raise asyncio.CancelledError() from None
    raise AssertionError("unreachable")


@credential_router.get(
    "",
    response_model=ProviderCredentialPage,
    responses={
        200: {
            "model": ProviderCredentialPage,
            "description": "A page of provider credential metadata.",
            "headers": {"Cache-Control": _CACHE_CONTROL_HEADER},
        },
        **error_responses(401, 422, 500, 503),
    },
)
async def list_provider_credentials(
    _actor: Annotated[AdminPrincipal, Depends(require_admin_principal)],
    service: Annotated[
        ProviderCredentialService,
        Depends(get_provider_credential_service),
    ],
    cursor: Annotated[str | None, Query()] = None,
    limit: Annotated[int | None, Query()] = None,
) -> JSONResponse:
    return _safe_response(await service.list_credentials(cursor=cursor, limit=limit))


@credential_router.get(
    "/{credential_id}",
    response_model=SafeProviderCredential,
    responses={
        200: {
            "model": SafeProviderCredential,
            "description": "Provider credential metadata.",
            "headers": _SAFE_HEADERS,
        },
        **error_responses(401, 404, 422, 500, 503),
    },
)
async def get_provider_credential(
    credential_id: UUID,
    _actor: Annotated[AdminPrincipal, Depends(require_admin_principal)],
    service: Annotated[
        ProviderCredentialService,
        Depends(get_provider_credential_service),
    ],
) -> JSONResponse:
    safe = await service.get_credential(credential_id)
    return _safe_response(
        safe,
        etag=provider_credential_etag(safe.id, safe.resource_revision),
    )


@credential_router.patch(
    "/{credential_id}",
    response_model=SafeProviderCredential,
    responses={
        200: {
            "model": SafeProviderCredential,
            "description": "Updated provider credential metadata.",
            "headers": _SAFE_HEADERS,
        },
        **error_responses(401, 404, 409, 412, 422, 428, 500, 503),
    },
)
async def update_provider_credential(
    credential_id: UUID,
    command: ProviderCredentialPatch,
    request_id: Annotated[str, Depends(get_request_id)],
    actor: Annotated[AdminPrincipal, Depends(require_admin_principal)],
    service: Annotated[
        ProviderCredentialService,
        Depends(get_provider_credential_service),
    ],
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
) -> JSONResponse:
    safe = None
    response = None
    cancelled = False
    try:
        safe = await service.update_credential(
            credential_id,
            command,
            actor=actor,
            request_id=request_id,
            expected_etag=if_match,
        )
        response = _safe_response(
            safe,
            etag=provider_credential_etag(safe.id, safe.resource_revision),
        )
        return response
    except asyncio.CancelledError:
        cancelled = True
    finally:
        command = _redacted_patch()
        safe = None
        response = None
        request_id = "<redacted>"
        if_match = "<redacted>"
    if cancelled:
        raise asyncio.CancelledError() from None
    raise AssertionError("unreachable")


@config_router.post(
    "",
    status_code=201,
    response_model=SafeProviderConfig,
    responses={
        201: {
            "model": SafeProviderConfig,
            "description": "A provider configuration was created.",
            "headers": _CREATE_HEADERS,
        },
        200: {
            "model": SafeProviderConfig,
            "description": "An equivalent idempotent create was replayed.",
            "headers": _CREATE_HEADERS,
        },
        **error_responses(401, 404, 409, 422, 500),
    },
)
async def create_provider_config(
    command: ProviderConfigCreate,
    request_id: Annotated[str, Depends(get_request_id)],
    actor: Annotated[AdminPrincipal, Depends(require_admin_principal)],
    service: Annotated[ProviderConfigService, Depends(get_provider_config_service)],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key")],
) -> JSONResponse:
    result = None
    safe = None
    response = None
    cancelled = False
    try:
        result = await service.create_provider_config(
            command,
            actor=actor,
            request_id=request_id,
            idempotency_key=idempotency_key,
        )
        safe = result.provider_config
        response = _safe_response(
            safe,
            status_code=201 if result.created else 200,
            etag=provider_config_etag(safe.id, safe.resource_revision),
            location=f"/v1/admin/provider-configs/{safe.id}",
        )
        return response
    except asyncio.CancelledError:
        cancelled = True
    finally:
        command = _redacted_config_create()
        result = None
        safe = None
        response = None
        request_id = "<redacted>"
        idempotency_key = "<redacted>"
    if cancelled:
        raise asyncio.CancelledError() from None
    raise AssertionError("unreachable")


@config_router.get(
    "",
    response_model=ProviderConfigPage,
    responses={
        200: {
            "model": ProviderConfigPage,
            "description": "A page of safe provider configurations.",
            "headers": {"Cache-Control": _CACHE_CONTROL_HEADER},
        },
        **error_responses(401, 422, 500),
    },
)
async def list_provider_configs(
    _actor: Annotated[AdminPrincipal, Depends(require_admin_principal)],
    service: Annotated[ProviderConfigService, Depends(get_provider_config_service)],
    cursor: Annotated[str | None, Query()] = None,
    limit: Annotated[int | None, Query()] = None,
) -> JSONResponse:
    return _safe_response(await service.list_provider_configs(cursor=cursor, limit=limit))


@config_router.get(
    "/{provider_config_id}",
    response_model=SafeProviderConfig,
    responses={
        200: {
            "model": SafeProviderConfig,
            "description": "Safe provider configuration metadata.",
            "headers": _SAFE_HEADERS,
        },
        **error_responses(401, 404, 422, 500),
    },
)
async def get_provider_config(
    provider_config_id: UUID,
    _actor: Annotated[AdminPrincipal, Depends(require_admin_principal)],
    service: Annotated[ProviderConfigService, Depends(get_provider_config_service)],
) -> JSONResponse:
    safe = await service.get_provider_config(provider_config_id)
    return _safe_response(
        safe,
        etag=provider_config_etag(safe.id, safe.resource_revision),
    )


@config_router.post(
    "/{provider_config_id}/embedding-probe",
    response_model=SafeEmbeddingProbe,
    responses={
        200: {
            "model": SafeEmbeddingProbe,
            "description": "The provider model embedding dimension was discovered.",
            "headers": {"Cache-Control": _CACHE_CONTROL_HEADER},
        },
        **error_responses(401, 404, 409, 422, 500, 503),
    },
)
async def probe_embedding_dimension(
    provider_config_id: CanonicalUuid,
    command: EmbeddingProbeCreate,
    request_id: Annotated[str, Depends(get_request_id)],
    actor: Annotated[AdminPrincipal, Depends(require_admin_principal)],
    service: Annotated[EmbeddingProbeService, Depends(get_embedding_probe_service)],
) -> JSONResponse:
    safe = None
    response = None
    cancelled = False
    try:
        safe = await service.probe(
            provider_config_id,
            command,
            actor=actor,
            request_id=request_id,
        )
        response = _safe_response(safe)
        return response
    except asyncio.CancelledError:
        cancelled = True
    finally:
        command = _redacted_probe_create()
        safe = None
        response = None
        request_id = "<redacted>"
    if cancelled:
        raise asyncio.CancelledError() from None
    raise AssertionError("unreachable")


@config_router.patch(
    "/{provider_config_id}",
    response_model=SafeProviderConfig,
    responses={
        200: {
            "model": SafeProviderConfig,
            "description": "Updated safe provider configuration metadata.",
            "headers": _SAFE_HEADERS,
        },
        **error_responses(401, 404, 409, 412, 422, 428, 500),
    },
)
async def update_provider_config(
    provider_config_id: UUID,
    command: ProviderConfigPatch,
    request_id: Annotated[str, Depends(get_request_id)],
    actor: Annotated[AdminPrincipal, Depends(require_admin_principal)],
    service: Annotated[ProviderConfigService, Depends(get_provider_config_service)],
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
) -> JSONResponse:
    safe = None
    response = None
    cancelled = False
    try:
        safe = await service.update_provider_config(
            provider_config_id,
            command,
            actor=actor,
            request_id=request_id,
            expected_etag=if_match,
        )
        response = _safe_response(
            safe,
            etag=provider_config_etag(safe.id, safe.resource_revision),
        )
        return response
    except asyncio.CancelledError:
        cancelled = True
    finally:
        command = _redacted_config_patch()
        safe = None
        response = None
        request_id = "<redacted>"
        if_match = "<redacted>"
    if cancelled:
        raise asyncio.CancelledError() from None
    raise AssertionError("unreachable")


@profile_router.post(
    "",
    status_code=201,
    response_model=SafeModelProfile,
    responses={
        201: {
            "model": SafeModelProfile,
            "description": "An embedding model profile was created.",
            "headers": _CREATE_HEADERS,
        },
        200: {
            "model": SafeModelProfile,
            "description": "An equivalent idempotent create was replayed.",
            "headers": _CREATE_HEADERS,
        },
        **error_responses(401, 404, 409, 422, 500),
    },
)
async def create_model_profile(
    command: ModelProfileCreate,
    request_id: Annotated[str, Depends(get_request_id)],
    actor: Annotated[AdminPrincipal, Depends(require_admin_principal)],
    service: Annotated[ModelProfileService, Depends(get_model_profile_service)],
    idempotency_key: Annotated[str, Header(alias="Idempotency-Key")],
) -> JSONResponse:
    result = None
    safe = None
    response = None
    cancelled = False
    try:
        result = await service.create_model_profile(
            command,
            actor=actor,
            request_id=request_id,
            idempotency_key=idempotency_key,
        )
        safe = result.model_profile
        response = _safe_response(
            safe,
            status_code=201 if result.created else 200,
            etag=model_profile_etag(safe.id, safe.resource_revision),
            location=f"/v1/admin/model-profiles/{safe.id}",
        )
        return response
    except asyncio.CancelledError:
        cancelled = True
    finally:
        command = _redacted_profile_create()
        result = None
        safe = None
        response = None
        request_id = "<redacted>"
        idempotency_key = "<redacted>"
    if cancelled:
        raise asyncio.CancelledError() from None
    raise AssertionError("unreachable")


@profile_router.get(
    "",
    response_model=ModelProfilePage,
    responses={
        200: {
            "model": ModelProfilePage,
            "description": "A page of safe model profiles.",
            "headers": {"Cache-Control": _CACHE_CONTROL_HEADER},
        },
        **error_responses(401, 422, 500),
    },
)
async def list_model_profiles(
    _actor: Annotated[AdminPrincipal, Depends(require_admin_principal)],
    service: Annotated[ModelProfileService, Depends(get_model_profile_service)],
    cursor: Annotated[str | None, Query()] = None,
    limit: Annotated[int | None, Query()] = None,
) -> JSONResponse:
    return _safe_response(await service.list_model_profiles(cursor=cursor, limit=limit))


@profile_router.get(
    "/{model_profile_id}",
    response_model=SafeModelProfile,
    responses={
        200: {
            "model": SafeModelProfile,
            "description": "Safe model profile metadata.",
            "headers": _SAFE_HEADERS,
        },
        **error_responses(401, 404, 422, 500),
    },
)
async def get_model_profile(
    model_profile_id: UUID,
    _actor: Annotated[AdminPrincipal, Depends(require_admin_principal)],
    service: Annotated[ModelProfileService, Depends(get_model_profile_service)],
) -> JSONResponse:
    safe = await service.get_model_profile(model_profile_id)
    return _safe_response(
        safe,
        etag=model_profile_etag(safe.id, safe.resource_revision),
    )


@profile_router.patch(
    "/{model_profile_id}",
    response_model=SafeModelProfile,
    responses={
        200: {
            "model": SafeModelProfile,
            "description": "Updated safe model profile metadata.",
            "headers": _SAFE_HEADERS,
        },
        **error_responses(401, 404, 409, 412, 422, 428, 500),
    },
)
async def update_model_profile(
    model_profile_id: UUID,
    command: ModelProfilePatch,
    request_id: Annotated[str, Depends(get_request_id)],
    actor: Annotated[AdminPrincipal, Depends(require_admin_principal)],
    service: Annotated[ModelProfileService, Depends(get_model_profile_service)],
    if_match: Annotated[str | None, Header(alias="If-Match")] = None,
) -> JSONResponse:
    safe = None
    response = None
    cancelled = False
    try:
        safe = await service.update_model_profile(
            model_profile_id,
            command,
            actor=actor,
            request_id=request_id,
            expected_etag=if_match,
        )
        response = _safe_response(
            safe,
            etag=model_profile_etag(safe.id, safe.resource_revision),
        )
        return response
    except asyncio.CancelledError:
        cancelled = True
    finally:
        command = _redacted_profile_patch()
        safe = None
        response = None
        request_id = "<redacted>"
        if_match = "<redacted>"
    if cancelled:
        raise asyncio.CancelledError() from None
    raise AssertionError("unreachable")


router.include_router(credential_router)
router.include_router(config_router)
router.include_router(profile_router)
