from contextlib import nullcontext
from typing import Annotated, cast

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from rag_service.api.errors import BusinessError
from rag_service.auth.codec import KeyKind
from rag_service.auth.local_trust import (
    ensure_local_keys,
    local_admin_principal,
    local_agent_principal,
)
from rag_service.auth.policies import AdminPrincipal, AgentPrincipal
from rag_service.auth.services import ApiKeyService
from rag_service.config import Settings, get_settings
from rag_service.db.session import Database

_BEARER = HTTPBearer(auto_error=False)


def _invalid_api_key_error() -> BusinessError:
    return BusinessError(
        401,
        "INVALID_API_KEY",
        "Invalid API key",
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_database(request: Request) -> Database:
    return cast(Database, request.app.state.database)


async def _authenticate(
    raw_token: str,
    database: Database,
    settings: Settings,
    expected_kind: KeyKind,
) -> AdminPrincipal | AgentPrincipal:
    authentication_error: BusinessError | None = None
    principal: AdminPrincipal | AgentPrincipal | None = None
    try:
        async with database.session() as authentication_session:
            service = ApiKeyService(
                session=authentication_session,
                authentication_sessions=lambda: nullcontext(authentication_session),
                settings=settings,
            )
            try:
                principal = await service.authenticate(raw_token, expected_kind)
            except BusinessError as error:
                authentication_error = error
        raw_token = "<redacted>"
    except BaseException:
        raw_token = "<redacted>"
        raise

    if authentication_error is not None:
        if (
            authentication_error.status_code == 401
            and authentication_error.code == "INVALID_API_KEY"
        ):
            authentication_error = None
            raise _invalid_api_key_error() from None
        raise authentication_error
    if expected_kind is KeyKind.ADMIN and type(principal) is AdminPrincipal:
        return principal
    if expected_kind is KeyKind.AGENT and type(principal) is AgentPrincipal:
        return principal
    raise _invalid_api_key_error()


async def require_admin_principal(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_BEARER)],
    database: Annotated[Database, Depends(get_database)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AdminPrincipal:
    if credentials is None and settings.local_trusted_auth:
        async with database.sessions() as session, session.begin():
            await ensure_local_keys(session)
        return local_admin_principal()
    if credentials is None:
        raise _invalid_api_key_error()
    raw_token = credentials.credentials
    credentials = None
    try:
        principal = await _authenticate(raw_token, database, settings, KeyKind.ADMIN)
    finally:
        raw_token = "<redacted>"
    if type(principal) is not AdminPrincipal:
        raise _invalid_api_key_error()
    return principal


async def require_agent_principal(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_BEARER)],
    database: Annotated[Database, Depends(get_database)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AgentPrincipal:
    if credentials is None and settings.local_trusted_auth:
        # A supplied token still wins, so an install can be tightened without
        # first tearing this out: the switch only covers the unauthenticated case.
        async with database.sessions() as session, session.begin():
            await ensure_local_keys(session)
            return await local_agent_principal(session)
    if credentials is None:
        raise _invalid_api_key_error()
    raw_token = credentials.credentials
    credentials = None
    try:
        principal = await _authenticate(raw_token, database, settings, KeyKind.AGENT)
    finally:
        raw_token = "<redacted>"
    if type(principal) is not AgentPrincipal:
        raise _invalid_api_key_error()
    return principal
