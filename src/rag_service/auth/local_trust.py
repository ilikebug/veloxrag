"""Bypass bearer authentication for a single-user local install.

The three-tier key model — an admin token minted from a container CLI, agent
keys it signs, capabilities and knowledge base scopes on each — exists to keep
tenants apart. One person running this next to their editor has no tenants to
separate, and the ceremony is the main thing standing between them and using
it: an agent cannot mint its own key, so somebody has to do it by hand first.

Turning it off does not mean removing the actor. Jobs, audit events and
idempotency records carry a non-null foreign key to `api_keys`, so requests
still need a row to be attributed to; this provisions one and hands it to every
request instead of reading a token. Everything downstream — scopes, auditing,
rate limits — keeps working unchanged.

The provisioned rows carry a random digest no token can produce, so enabling
this does not create a credential that could later be replayed against an
instance where the switch is off.

Refused outright when the environment is production (`config.py`), for the same
reason `provider_allow_private_targets` is: a convenience switch that survives
a copied configuration file stops being a convenience.
"""

from __future__ import annotations

import secrets
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.auth.policies import AdminPrincipal, AgentPrincipal, Capability
from rag_service.db.models.auth import ApiKey, ApiKeyKnowledgeBaseScope
from rag_service.db.models.knowledge_bases import KnowledgeBase

# Fixed so a restart attributes work to the same actor rather than accumulating
# a new principal per boot.
LOCAL_ADMIN_KEY_ID = UUID("00000000-0000-4000-8000-0000000010ca")
LOCAL_AGENT_KEY_ID = UUID("00000000-0000-4000-8000-00000000a6e7")
LOCAL_ADMIN_PUBLIC_ID = "local-trusted-admin"
LOCAL_AGENT_PUBLIC_ID = "local-trusted-agent"

_LOCAL_CAPABILITIES = ("manage", "ingest", "retrieve", "answer")
_LOCAL_REQUESTS_PER_MINUTE = 10_000
_LOCAL_MAX_CONCURRENCY = 64


async def ensure_local_keys(session: AsyncSession) -> None:
    """Create the local actor rows if they are missing, and scope them.

    Idempotent: called on every request that needs a principal, because the
    knowledge base set changes underneath it and a scope granted at startup
    would go stale.
    """
    for key_id, public_id, key_type in (
        (LOCAL_ADMIN_KEY_ID, LOCAL_ADMIN_PUBLIC_ID, "admin"),
        (LOCAL_AGENT_KEY_ID, LOCAL_AGENT_PUBLIC_ID, "agent"),
    ):
        existing = await session.get(ApiKey, key_id)
        if existing is not None:
            continue
        session.add(
            ApiKey(
                id=key_id,
                public_id=public_id,
                # Random, never revealed, never compared against: no bearer
                # token can authenticate as this row.
                secret_digest=secrets.token_bytes(32),
                key_type=key_type,
                name=f"local trusted {key_type}",
                status="active",
                capabilities=list(_LOCAL_CAPABILITIES) if key_type == "agent" else [],
                raw_file_read=key_type == "agent",
                requests_per_minute=_LOCAL_REQUESTS_PER_MINUTE if key_type == "agent" else None,
                max_concurrency=_LOCAL_MAX_CONCURRENCY if key_type == "agent" else None,
            )
        )
    await session.flush()
    await _grant_missing_scopes(session)


async def _grant_missing_scopes(session: AsyncSession) -> None:
    knowledge_base_ids = set(
        await session.scalars(select(KnowledgeBase.id).where(KnowledgeBase.status != "deleting"))
    )
    scoped = set(
        await session.scalars(
            select(ApiKeyKnowledgeBaseScope.knowledge_base_id).where(
                ApiKeyKnowledgeBaseScope.api_key_id == LOCAL_AGENT_KEY_ID
            )
        )
    )
    for knowledge_base_id in knowledge_base_ids - scoped:
        session.add(
            ApiKeyKnowledgeBaseScope(
                api_key_id=LOCAL_AGENT_KEY_ID,
                knowledge_base_id=knowledge_base_id,
            )
        )
    if knowledge_base_ids - scoped:
        await session.flush()


def local_admin_principal() -> AdminPrincipal:
    return AdminPrincipal(key_id=LOCAL_ADMIN_KEY_ID, public_id=LOCAL_ADMIN_PUBLIC_ID)


async def local_agent_principal(session: AsyncSession) -> AgentPrincipal:
    knowledge_base_ids = frozenset(
        await session.scalars(
            select(ApiKeyKnowledgeBaseScope.knowledge_base_id).where(
                ApiKeyKnowledgeBaseScope.api_key_id == LOCAL_AGENT_KEY_ID
            )
        )
    )
    return AgentPrincipal(
        key_id=LOCAL_AGENT_KEY_ID,
        public_id=LOCAL_AGENT_PUBLIC_ID,
        capabilities=frozenset(Capability(name) for name in _LOCAL_CAPABILITIES),
        knowledge_base_ids=knowledge_base_ids,
        query_profile_ids=frozenset(),
        default_query_profile_id=None,
        raw_file_read=True,
        requests_per_minute=_LOCAL_REQUESTS_PER_MINUTE,
        max_concurrency=_LOCAL_MAX_CONCURRENCY,
    )


__all__ = [
    "LOCAL_ADMIN_KEY_ID",
    "LOCAL_ADMIN_PUBLIC_ID",
    "LOCAL_AGENT_KEY_ID",
    "LOCAL_AGENT_PUBLIC_ID",
    "ensure_local_keys",
    "local_admin_principal",
    "local_agent_principal",
]
