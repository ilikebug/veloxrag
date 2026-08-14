"""Make a fresh stack usable without eight manual API calls.

Bringing the service up is not the same as being able to search it. A usable
knowledge base needs a provider credential, a provider config, a probed
dimension, a model profile, the knowledge base itself, a filter schema and an
initial index generation — in that order, because the filter schema and the
embedding configuration freeze the moment the first generation exists. Getting
that wrong is not recoverable in place, which makes it a poor first task for
someone who just wants their agent to remember things.

So this does it once, idempotently, over the same HTTP API a human would use, so
there is no second code path that can disagree with the real one about what a
valid setup looks like.

It deliberately does not ingest anything. What to remember is the user's choice,
and an empty knowledge base is honest about being empty.
"""

from __future__ import annotations

import json
import os
import secrets
import time
import urllib.error
import urllib.request
from typing import Any
from uuid import uuid4

DEFAULT_KNOWLEDGE_BASE_NAME = "local-memory"
# Ollama names a model by what was pulled, not by its Hugging Face path.
DEFAULT_EMBEDDING_MODEL = "bge-m3"
DEFAULT_RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# The nine fields the roadmap settled on. Set once here because the filter schema
# is frozen by the first generation: a field left out now cannot be added later
# without rebuilding the knowledge base.
_FILTER_FIELDS = (
    ("source_type", "keyword"),
    ("speaker", "keyword"),
    ("channel", "keyword"),
    ("thread_id", "keyword"),
    ("doc_type", "keyword"),
    ("section", "keyword"),
    ("source_path", "keyword"),
    ("lang", "keyword"),
)
_DATETIME_FIELD = "occurred_at"

_READY_ATTEMPTS = 120
_READY_DELAY_SECONDS = 2.0


class BootstrapError(Exception):
    """Safe setup failure."""


def _request(
    base_url: str,
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | None = None,
    idempotent: bool = False,
    timeout: float = 300.0,
) -> tuple[int, dict[str, Any]]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    if idempotent:
        headers["Idempotency-Key"] = str(uuid4())
    request = urllib.request.Request(f"{base_url}{path}", data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            return response.status, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as error:
        raw = error.read()
        try:
            return error.code, json.loads(raw) if raw else {}
        except ValueError:
            return error.code, {}


def _wait_for_service(base_url: str) -> None:
    for _ in range(_READY_ATTEMPTS):
        try:
            status, _ = _request(base_url, "GET", "/health", timeout=5.0)
        except OSError:
            status = 0
        if status == 200:
            return
        time.sleep(_READY_DELAY_SECONDS)
    raise BootstrapError("the service did not become reachable")


def _find(base_url: str, path: str, name: str) -> str | None:
    status, body = _request(base_url, "GET", f"{path}?page_size=100")
    if status != 200:
        raise BootstrapError(f"could not list {path}")
    for item in body.get("items", ()):
        if item.get("name") == name:
            return str(item["id"])
    return None


def _ensure_credential(base_url: str, name: str) -> str:
    existing = _find(base_url, "/v1/admin/provider-credentials", name)
    if existing is not None:
        return existing
    status, body = _request(
        base_url,
        "POST",
        "/v1/admin/provider-credentials",
        # The local inference server needs no credential, but the schema requires
        # one; a random value is safer than a guessable placeholder in case the
        # endpoint is later pointed somewhere that does check.
        payload={"name": name, "secret": secrets.token_urlsafe(24)},
        idempotent=True,
    )
    if status != 201:
        raise BootstrapError(f"provider credential creation failed with {status}")
    return str(body["id"])


def _ensure_provider_config(base_url: str, name: str, credential_id: str, endpoint: str) -> str:
    existing = _find(base_url, "/v1/admin/provider-configs", name)
    if existing is not None:
        return existing
    status, body = _request(
        base_url,
        "POST",
        "/v1/admin/provider-configs",
        payload={
            "name": name,
            "provider_type": "openai_compatible",
            "base_url": endpoint,
            "credential_id": credential_id,
            "default_headers": {},
            "routing_options": {},
            "timeout_seconds": "300.000",
            "max_concurrency": 2,
            "requests_per_minute": 600,
            "enabled": True,
        },
        idempotent=True,
    )
    if status != 201:
        raise BootstrapError(f"provider config creation failed with {status}")
    return str(body["id"])


def _ensure_embedding_profile(base_url: str, name: str, config_id: str, model: str) -> str:
    existing = _find(base_url, "/v1/admin/model-profiles", name)
    if existing is not None:
        return existing
    # Probed rather than assumed: a wrong dimension is frozen into the first
    # generation and cannot be corrected without rebuilding.
    status, body = _request(
        base_url,
        "POST",
        f"/v1/admin/provider-configs/{config_id}/embedding-probe",
        payload={"model_name": model},
    )
    if status != 200:
        raise BootstrapError(f"embedding probe failed with {status}")
    dimension = body["dimension"]
    status, body = _request(
        base_url,
        "POST",
        "/v1/admin/model-profiles",
        payload={
            "name": name,
            "capability": "embedding",
            "provider_config_id": config_id,
            "model_name": model,
            "dimension": dimension,
            "max_input_tokens": 8192,
            # Matches what a local text-embeddings-inference server accepts; it
            # caps backend batches at 8 and rejects larger ones outright.
            "batch_size": 8,
            "timeout_seconds": "300.000",
            "vector_config": {},
            "enabled": True,
        },
        idempotent=True,
    )
    if status != 201:
        raise BootstrapError(f"embedding profile creation failed with {status}")
    return str(body["id"])


def _ensure_rerank_profile(base_url: str, name: str, config_id: str, model: str) -> str | None:
    existing = _find(base_url, "/v1/admin/model-profiles", name)
    if existing is not None:
        return existing
    status, body = _request(
        base_url,
        "POST",
        "/v1/admin/model-profiles",
        payload={
            "name": name,
            "capability": "rerank",
            "provider_config_id": config_id,
            "model_name": model,
            "dimension": None,
            "max_input_tokens": 8192,
            "batch_size": 8,
            "timeout_seconds": "300.000",
            "vector_config": {},
            "enabled": True,
        },
        idempotent=True,
    )
    if status != 201:
        # Not fatal. Reranking is opt-in per request and the knowledge base is
        # fully usable without it, so a reranker that is not running should not
        # stop the rest of setup.
        return None
    return str(body["id"])


def _etag(base_url: str, knowledge_base_id: str) -> str | None:
    request = urllib.request.Request(
        f"{base_url}/v1/knowledge-bases/{knowledge_base_id}", method="GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=30.0) as response:
            header = response.headers.get("ETag")
    except urllib.error.HTTPError:
        return None
    return None if header is None else str(header)


def _ensure_knowledge_base(base_url: str, name: str) -> tuple[str, bool]:
    existing = _find(base_url, "/v1/knowledge-bases", name)
    if existing is not None:
        return existing, False
    status, body = _request(
        base_url,
        "POST",
        "/v1/knowledge-bases",
        payload={"name": name, "description": "created by velox-bootstrap"},
        idempotent=True,
    )
    if status != 201:
        raise BootstrapError(f"knowledge base creation failed with {status}")
    return str(body["id"]), True


def _apply_filter_schema(base_url: str, knowledge_base_id: str) -> None:
    fields = [
        {
            "name": name,
            "source_path": name,
            "type": kind,
            "operators": ["eq", "in"],
        }
        for name, kind in _FILTER_FIELDS
    ]
    fields.append(
        {
            "name": _DATETIME_FIELD,
            "source_path": _DATETIME_FIELD,
            "type": "datetime",
            "operators": ["eq", "gte", "lte"],
        }
    )
    etag = _etag(base_url, knowledge_base_id)
    request = urllib.request.Request(
        f"{base_url}/v1/knowledge-bases/{knowledge_base_id}/filter-schema",
        data=json.dumps({"fields": fields}).encode("utf-8"),
        headers={"Content-Type": "application/json", **({"If-Match": etag} if etag else {})},
        method="PUT",
    )
    try:
        with urllib.request.urlopen(request, timeout=60.0) as response:
            if response.status != 200:
                raise BootstrapError("filter schema could not be applied")
    except urllib.error.HTTPError as error:
        raise BootstrapError(f"filter schema could not be applied ({error.code})") from None


def _ensure_generation(base_url: str, knowledge_base_id: str, profile_id: str) -> None:
    status, body = _request(
        base_url,
        "GET",
        f"/v1/admin/knowledge-bases/{knowledge_base_id}/index-generations",
    )
    if status == 200 and any(
        item.get("status") in {"active", "building"} for item in body.get("items", ())
    ):
        return
    status, body = _request(
        base_url,
        "POST",
        f"/v1/admin/knowledge-bases/{knowledge_base_id}/index-generations",
        payload={"embedding_profile_id": profile_id, "distance": "cosine"},
        idempotent=True,
    )
    if status != 201:
        code = body.get("error", {}).get("code", status)
        raise BootstrapError(f"initial index generation failed: {code}")


def _set_rerank_profile(base_url: str, knowledge_base_id: str, profile_id: str) -> None:
    etag = _etag(base_url, knowledge_base_id)
    request = urllib.request.Request(
        f"{base_url}/v1/knowledge-bases/{knowledge_base_id}",
        data=json.dumps({"rerank_profile_id": profile_id}).encode("utf-8"),
        headers={"Content-Type": "application/json", **({"If-Match": etag} if etag else {})},
        method="PATCH",
    )
    try:
        urllib.request.urlopen(request, timeout=60.0).close()
    except urllib.error.HTTPError:
        # Same reasoning as a missing rerank profile: searching still works.
        return


def bootstrap(
    *,
    base_url: str,
    embedding_endpoint: str,
    rerank_endpoint: str | None,
    knowledge_base_name: str,
    embedding_model: str,
    rerank_model: str,
) -> dict[str, str]:
    _wait_for_service(base_url)
    credential_id = _ensure_credential(base_url, "local-inference")
    embedding_config = _ensure_provider_config(
        base_url, "local-embedding", credential_id, embedding_endpoint
    )
    embedding_profile = _ensure_embedding_profile(
        base_url, "local-embedding", embedding_config, embedding_model
    )
    knowledge_base_id, _created = _ensure_knowledge_base(base_url, knowledge_base_name)
    # Order matters and is not recoverable: the filter schema must be in place
    # before the first generation freezes it.
    _apply_filter_schema(base_url, knowledge_base_id)
    _ensure_generation(base_url, knowledge_base_id, embedding_profile)

    if rerank_endpoint:
        rerank_config = _ensure_provider_config(
            base_url, "local-rerank", credential_id, rerank_endpoint
        )
        rerank_profile = _ensure_rerank_profile(
            base_url, "local-rerank", rerank_config, rerank_model
        )
        if rerank_profile is not None:
            _set_rerank_profile(base_url, knowledge_base_id, rerank_profile)

    return {"knowledge_base_id": knowledge_base_id}


def main() -> int:
    base_url = os.environ.get("RAG_BOOTSTRAP_BASE_URL", "http://api:8000").rstrip("/")
    embedding_endpoint = os.environ.get("RAG_BOOTSTRAP_EMBEDDING_URL", "https://embedding/v1")
    rerank_endpoint = os.environ.get("RAG_BOOTSTRAP_RERANK_URL", "").strip() or None
    try:
        result = bootstrap(
            base_url=base_url,
            embedding_endpoint=embedding_endpoint,
            rerank_endpoint=rerank_endpoint,
            knowledge_base_name=os.environ.get(
                "RAG_BOOTSTRAP_KNOWLEDGE_BASE", DEFAULT_KNOWLEDGE_BASE_NAME
            ),
            embedding_model=os.environ.get(
                "RAG_BOOTSTRAP_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
            ),
            rerank_model=os.environ.get("RAG_BOOTSTRAP_RERANK_MODEL", DEFAULT_RERANK_MODEL),
        )
    except BootstrapError as error:
        print(f"velox-bootstrap: {error}", flush=True)
        return 1
    print(json.dumps(result), flush=True)
    return 0


__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_KNOWLEDGE_BASE_NAME",
    "DEFAULT_RERANK_MODEL",
    "BootstrapError",
    "bootstrap",
    "main",
]
