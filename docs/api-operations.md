# API operations guide

The service has no configuration UI; the HTTP API is the only entry point. This document is the
authoritative guide to operating it.

The complete machine-readable contract comes from the service itself. **Prefer it** — it is
generated from the code and cannot go stale:

- `GET /openapi.json` — all 40 operations with request/response schemas (feed this one to an AI)
- `GET /docs` — Swagger UI, interactive (frontend assets come from a CDN, so it needs network)
- `GET /redoc` — ReDoc

This document only adds the four kinds of information OpenAPI **cannot express**: call order,
the authentication layering, implicit rules that make a request fail, and traps already stepped in.

**Verification status**: all seven steps in section 3, the readiness probes in section 4, and the
teardown in section 5 were run for real against a local stack on 2026-08-04, including a genuine
provider embedding call (`embedding-probe` returned 1024 dimensions). The real run used equivalent
direct `curl` calls; the examples in the text are the hardened form (token written to a
permission-restricted file, JSON constructed before submission) with identical semantics. Section 7
separately lists the endpoints that were only checked against the OpenAPI contract and not run.

---

## 1. Authentication model

Three layers, each derived from the one above, and none of them skippable:

| Layer | How you get it | What it can do |
| --- | --- | --- |
| Admin token | mintable only by the in-container CLI; there is **no API** | all of `/v1/admin/*`: providers, model configuration, index generations, issuing and revoking Agent keys |
| Agent key | `POST /v1/admin/api-keys` (requires Admin) | determined by capability, see the table below |
| — | — | — |

How Agent capabilities map to endpoints:

| Capability | May call |
| --- | --- |
| `manage` | `POST /v1/knowledge-bases` (create), and `PATCH`/`DELETE` on knowledge bases within its granted scope |
| `ingest` | `POST /v1/knowledge-bases/{kb}/documents`, `GET /v1/jobs/{id}` |
| `retrieve` | `POST /v1/knowledge-bases/{kb}/search` |
| `answer` | the answering endpoint (needs a query profile configured as well, see `/ready/answer`) |

**Scope is a hard constraint**: a `manage` key with an empty `knowledge_base_ids` may **create** new
knowledge bases but returns 404 for any that **already exists**. To operate on an existing knowledge
base, its id has to be in `knowledge_base_ids` at the time the key is issued.

Mint an Admin token (the only entry point; the token is echoed exactly once):

```bash
docker compose exec -T api velox-admin admin-key create --name my-admin
# -> {"api_key": {...}, "token": "..."}
docker compose exec -T api velox-admin admin-key list
docker compose exec -T api velox-admin admin-key revoke <id>
```

---

---

## 2. Three general rules

Violating any one of them gets the request rejected, and the error message will not tell you which
one directly.

**Creates must carry `Idempotency-Key`** (a UUID). This applies to `POST /v1/knowledge-bases`,
`POST /v1/admin/knowledge-bases/{kb}/index-generations`, and provider/model creation. **Reuse the
same key** to retry; once the request body changes, the key has to change too.

**Modifications must carry `If-Match`** (GET the `ETag` response header first). This applies to every
`PATCH`, `DELETE`, and `POST .../revoke`. A stale ETag gets 412.

**Responses are validated by reflection.** A resource returned by the server has to reflect what you
submitted field by field; a mismatch is treated as an invalid response. This is what a hand-rolled
mock or a proxy that rewrites bodies runs into.

The error envelope is uniform:

```json
{"error": {"code": "RESOURCE_NOT_FOUND", "message": "...", "retryable": false, "request_id": "req_..."}}
```

Every response carries `Cache-Control: no-store`.

---

---

## 3. The complete operational flow

### 3.0 Get the stack running first

Two paths, chosen by where the embedding endpoint lives:

| Command | Embedding endpoint | When to use it |
| --- | --- | --- |
| `make start` | in-container `https://embedding/v1` (`BAAI/bge-m3`, 1024 dimensions), and it completes everything from 3.2 onward automatically | the default path. The first start downloads about 4.8 GB of weights; for detail and the throughput trade-offs see [local-embedding.md](local-embedding.md) |
| `make up` | you specify it | using a hosted provider (https already) or an existing internal embedding service; the steps below then have to be done by hand |

Stop with `make stop`. Neither command creates the provider configuration for you — that is what
3.2 onward is about.

A self-hosted local embedding service needs a TLS termination layer in front of it:
`network_policy.py:344` requires provider endpoints to be https, and local inference services
usually speak only http. `make start` already includes that layer.

The flow below assumes:

- `ADMIN_AUTH_CONFIG`, `AGENT_AUTH_CONFIG`, and `RAG_OPERATIONS_DIR` are prepared (for how to mint
  the tokens and write them into permission-restricted config files, see `## Admin and Agent keys`
  in the README);
- the Agent has scope over the target knowledge base and whichever of `manage`/`ingest`/`retrieve`
  the operation needs;
- `KB_ID` is the UUID of an already created knowledge base;
- `RAG_BASE_URL` is an explicit loopback address.

```bash
export RAG_BASE_URL=http://127.0.0.1:8000
export KB_ID='<knowledge-base-id>'
KB_ID=$(python3 - <<'PY'
import os
from uuid import UUID

print(UUID(os.environ["KB_ID"]))
PY
) || exit 1
export KB_ID
```

Do not reuse the example idempotency keys below across different request bodies. The same key with
the same normalized request replays safely; the same key with a different request returns
`409 IDEMPOTENCY_KEY_REUSED`.

### 1. Credential: store the provider secret encrypted

A provider secret enters the service only on a create or PATCH request; the security response
contains no secret, ciphertext, or nonce. Inject `RAG_PROVIDER_SECRET` from a secret manager first,
then write the request file with a JSON serializer, so that untrusted input is never concatenated
into JSON:

```bash
PROVIDER_CREDENTIAL_REQUEST="${RAG_OPERATIONS_DIR}/provider-credential-request.json"
PROVIDER_CREDENTIAL_RESPONSE="${RAG_OPERATIONS_DIR}/provider-credential-response.json"

: "${RAG_PROVIDER_SECRET:?inject the Provider secret through an approved secret source}"
python3 - "${PROVIDER_CREDENTIAL_REQUEST}" <<'PY' || exit 1
import json
import os
import pathlib
import sys

secret = os.environ["RAG_PROVIDER_SECRET"]
if not secret.strip() or len(secret.encode("utf-8")) > 8192:
    raise SystemExit(1)
path = pathlib.Path(sys.argv[1])
path.write_text(
    json.dumps({"name": "primary-embedding", "secret": secret}, separators=(",", ":")),
    encoding="utf-8",
)
path.chmod(0o600)
PY
unset RAG_PROVIDER_SECRET

HTTP_STATUS=$(curl -q --connect-timeout 3 --max-time 30 --noproxy '*' -sS \
  -o "${PROVIDER_CREDENTIAL_RESPONSE}" -w '%{http_code}' \
  -X POST "${RAG_BASE_URL}/v1/admin/provider-credentials" \
  --config "${ADMIN_AUTH_CONFIG}" \
  -H 'Idempotency-Key: provider-credential-create-1' \
  -H 'Content-Type: application/json' \
  --data-binary "@${PROVIDER_CREDENTIAL_REQUEST}")
rm -f -- "${PROVIDER_CREDENTIAL_REQUEST}"
[ "${HTTP_STATUS}" = 201 ] || [ "${HTTP_STATUS}" = 200 ] || exit 1

PROVIDER_CREDENTIAL_ID=$(python3 - "${PROVIDER_CREDENTIAL_RESPONSE}" <<'PY'
import json
import sys
from uuid import UUID

with open(sys.argv[1], encoding="utf-8") as source:
    print(UUID(json.load(source)["id"]))
PY
) || exit 1
export PROVIDER_CREDENTIAL_ID
```

### 2. ProviderConfig: bind the protocol, base URL, and credential

`openai_compatible` and `openrouter` are supported. The service calls `/embeddings` beneath the
canonical base URL, so configure the provider's API base rather than splicing the single embeddings
path into it. The base URL must be HTTPS, and it is checked against the SSRF/DNS policy both when
saved and before every connection. Production refuses loopback, private, link-local, and metadata
targets; `RAG_PROVIDER_ALLOW_PRIVATE_TARGETS=true` is for controlled local testing only, and a
production configuration rejects it.

OpenRouter uses `provider_type: openrouter` and can carry allowlist-restricted routing options and
presentational headers; a self-hosted or other OpenAI-compatible embedding endpoint uses
`provider_type: openai_compatible`. Do not put an authentication header into `default_headers` —
authentication comes from the credential. The base URL below is supplied explicitly through an
environment variable:

```bash
export RAG_PROVIDER_TYPE='openai_compatible'
export RAG_PROVIDER_BASE_URL='https://<provider-host>/v1'
PROVIDER_CONFIG_REQUEST="${RAG_OPERATIONS_DIR}/provider-config-request.json"
PROVIDER_CONFIG_RESPONSE="${RAG_OPERATIONS_DIR}/provider-config-response.json"

python3 - "${PROVIDER_CONFIG_REQUEST}" <<'PY' || exit 1
import json
import os
import pathlib
import sys
from urllib.parse import urlsplit
from uuid import UUID

provider_type = os.environ["RAG_PROVIDER_TYPE"]
base_url = os.environ["RAG_PROVIDER_BASE_URL"]
parsed = urlsplit(base_url)
if provider_type not in {"openai_compatible", "openrouter"}:
    raise SystemExit(1)
if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
    raise SystemExit(1)
document = {
    "name": "primary-provider",
    "provider_type": provider_type,
    "base_url": base_url,
    "credential_id": str(UUID(os.environ["PROVIDER_CREDENTIAL_ID"])),
    "default_headers": {},
    "routing_options": {},
    "timeout_seconds": 30,
    "max_concurrency": 4,
    "requests_per_minute": 60,
    "enabled": True,
}
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8")
path.chmod(0o600)
PY

HTTP_STATUS=$(curl -q --connect-timeout 3 --max-time 30 --noproxy '*' -sS \
  -o "${PROVIDER_CONFIG_RESPONSE}" -w '%{http_code}' \
  -X POST "${RAG_BASE_URL}/v1/admin/provider-configs" \
  --config "${ADMIN_AUTH_CONFIG}" \
  -H 'Idempotency-Key: provider-config-create-1' \
  -H 'Content-Type: application/json' \
  --data-binary "@${PROVIDER_CONFIG_REQUEST}")
rm -f -- "${PROVIDER_CONFIG_REQUEST}"
[ "${HTTP_STATUS}" = 201 ] || [ "${HTTP_STATUS}" = 200 ] || exit 1

PROVIDER_CONFIG_ID=$(python3 - "${PROVIDER_CONFIG_RESPONSE}" <<'PY'
import json
import sys
from uuid import UUID

with open(sys.argv[1], encoding="utf-8") as source:
    print(UUID(json.load(source)["id"]))
PY
) || exit 1
export PROVIDER_CONFIG_ID
```

Once a generation references a ProviderConfig or profile, fields that would change vector semantics
cannot be modified in place; changing the provider type, base URL, request protocol, model, or
dimension means creating a new configuration and a future generation. Runtime parameters such as
timeout, concurrency, RPM, batch size, and enabled can still be PATCHed against the latest ETag.

### 3. ModelProfile: pin the embedding model semantics

`dimension` has to match the dimension the provider actually returns. The first version accepts only
`capability: embedding`, and `vector_config` has to be an empty object:

```bash
export RAG_EMBEDDING_MODEL='<embedding-model-name>'
export RAG_EMBEDDING_DIMENSION='<positive-integer>'
MODEL_PROFILE_REQUEST="${RAG_OPERATIONS_DIR}/model-profile-request.json"
MODEL_PROFILE_RESPONSE="${RAG_OPERATIONS_DIR}/model-profile-response.json"

python3 - "${MODEL_PROFILE_REQUEST}" <<'PY' || exit 1
import json
import os
import pathlib
import sys
from uuid import UUID

dimension = int(os.environ["RAG_EMBEDDING_DIMENSION"])
if not 1 <= dimension <= 10_000_000:
    raise SystemExit(1)
model_name = os.environ["RAG_EMBEDDING_MODEL"].strip()
if not 1 <= len(model_name) <= 255:
    raise SystemExit(1)
document = {
    "name": "primary-embedding-profile",
    "capability": "embedding",
    "provider_config_id": str(UUID(os.environ["PROVIDER_CONFIG_ID"])),
    "model_name": model_name,
    "dimension": dimension,
    "max_input_tokens": 8192,
    "batch_size": 8,
    "timeout_seconds": 30,
    "vector_config": {},
    "enabled": True,
}
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8")
path.chmod(0o600)
PY

HTTP_STATUS=$(curl -q --connect-timeout 3 --max-time 30 --noproxy '*' -sS \
  -o "${MODEL_PROFILE_RESPONSE}" -w '%{http_code}' \
  -X POST "${RAG_BASE_URL}/v1/admin/model-profiles" \
  --config "${ADMIN_AUTH_CONFIG}" \
  -H 'Idempotency-Key: model-profile-create-1' \
  -H 'Content-Type: application/json' \
  --data-binary "@${MODEL_PROFILE_REQUEST}")
rm -f -- "${MODEL_PROFILE_REQUEST}"
[ "${HTTP_STATUS}" = 201 ] || [ "${HTTP_STATUS}" = 200 ] || exit 1

MODEL_PROFILE_ID=$(python3 - "${MODEL_PROFILE_RESPONSE}" <<'PY'
import json
import sys
from uuid import UUID

with open(sys.argv[1], encoding="utf-8") as source:
    print(UUID(json.load(source)["id"]))
PY
) || exit 1
export MODEL_PROFILE_ID
```

### 4. Initial generation: create and activate an empty Qdrant collection

In the first version a knowledge base can have only one initial generation. The service persists a
reservation in PostgreSQL first, then idempotently creates or verifies the Qdrant collection, probes
the provider with fixed non-sensitive text, and finally activates the generation. Replaying with the
same `Idempotency-Key` after a crash continues the same generation and does not create a second
collection.

```bash
GENERATION_REQUEST="${RAG_OPERATIONS_DIR}/generation-request.json"
GENERATION_RESPONSE="${RAG_OPERATIONS_DIR}/generation-response.json"

python3 - "${GENERATION_REQUEST}" <<'PY' || exit 1
import json
import os
import pathlib
import sys
from uuid import UUID

document = {
    "embedding_profile_id": str(UUID(os.environ["MODEL_PROFILE_ID"])),
    "distance": "cosine",
}
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps(document, separators=(",", ":")), encoding="utf-8")
path.chmod(0o600)
PY

HTTP_STATUS=$(curl -q --connect-timeout 3 --max-time 60 --noproxy '*' -sS \
  -o "${GENERATION_RESPONSE}" -w '%{http_code}' \
  -X POST "${RAG_BASE_URL}/v1/admin/knowledge-bases/${KB_ID}/index-generations" \
  --config "${ADMIN_AUTH_CONFIG}" \
  -H 'Idempotency-Key: initial-generation-create-1' \
  -H 'Content-Type: application/json' \
  --data-binary "@${GENERATION_REQUEST}")
rm -f -- "${GENERATION_REQUEST}"
[ "${HTTP_STATUS}" = 201 ] || exit 1

GENERATION_ID=$(python3 - "${GENERATION_RESPONSE}" <<'PY'
import json
import sys
from uuid import UUID

with open(sys.argv[1], encoding="utf-8") as source:
    document = json.load(source)
if document.get("status") != "active":
    raise SystemExit(1)
print(UUID(document["id"]))
PY
) || exit 1
export GENERATION_ID
```

### 5. Upload: submit TXT/Markdown and receive a job

`.txt`, `.md`, and `.markdown` are supported. The content has to be non-empty UTF-8 text, with no NUL
and nothing obviously binary. Uploads are streamed and the 50 MiB maximum (52,428,800 bytes) is
enforced independently of `Content-Length`. A first upload requires an active generation, and
duplicate content within the same knowledge base returns `409 DUPLICATE_DOCUMENT`.

The example below confines the upload to an absolute directory the operator chose explicitly and to a
single basename. It validates through no-follow file descriptors and copies once into this
operation's private directory; curl only ever opens that fixed-name copy and never reopens the
external path. The source directory and file must be owned by the current user and not writable by
group or other, and the source file must additionally be a regular file with a single hard link.
`DOCUMENT_ROOT` has to be a physically canonicalized absolute path; running `pwd -P` in the target
directory and using its output works. A symbolic link as a path component is refused deliberately:

```bash
export DOCUMENT_ROOT='<absolute-directory-containing-document>'
export DOCUMENT_NAME='document.md'
STAGED_DOCUMENT_PATH=$(python3 - "${RAG_OPERATIONS_DIR}" <<'PY'
import os
import re
import stat
import sys

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
SOURCE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
STAGED_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
SUFFIXES = {
    ".md": ("upload-source.md", "text/markdown"),
    ".markdown": ("upload-source.markdown", "text/markdown"),
    ".txt": ("upload-source.txt", "text/plain"),
}

def fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_nlink,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )

def trusted_directory(metadata: os.stat_result, *, private: bool = False) -> bool:
    mode = stat.S_IMODE(metadata.st_mode)
    return (
        stat.S_ISDIR(metadata.st_mode)
        and metadata.st_uid == os.getuid()
        and (mode == 0o700 if private else mode & 0o022 == 0)
    )

def open_directory_path(path: str) -> int:
    if not os.path.isabs(path) or os.path.normpath(path) != path:
        raise ValueError
    descriptor = os.open("/", DIRECTORY_FLAGS)
    try:
        for component in path.split("/")[1:]:
            next_descriptor = os.open(
                component,
                DIRECTORY_FLAGS,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise

def stage_document(operations_path: str) -> str:
    root_path = os.environ["DOCUMENT_ROOT"]
    name = os.environ["DOCUMENT_NAME"]
    if not os.path.isabs(root_path) or os.path.basename(name) != name or name in {"", ".", ".."}:
        raise ValueError
    if (
        not 1 <= len(name.encode("utf-8")) <= 255
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) is None
    ):
        raise ValueError
    suffix = os.path.splitext(name)[1].lower()
    if suffix not in SUFFIXES:
        raise ValueError
    if re.fullmatch(r"/tmp/rag-operations\.[A-Za-z0-9]{6}", operations_path) is None:
        raise ValueError

    staged_name, _content_type = SUFFIXES[suffix]
    tmp_fd = operations_fd = root_fd = source_fd = staged_fd = None
    try:
        tmp_fd = open_directory_path(os.path.realpath("/tmp"))
        operations_fd = os.open(
            os.path.basename(operations_path),
            DIRECTORY_FLAGS,
            dir_fd=tmp_fd,
        )
        if not trusted_directory(os.fstat(operations_fd), private=True):
            raise ValueError

        root_fd = open_directory_path(root_path)
        if not trusted_directory(os.fstat(root_fd)):
            raise ValueError
        source_fd = os.open(name, SOURCE_FLAGS, dir_fd=root_fd)
        before = os.fstat(source_fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != os.getuid()
            or before.st_nlink != 1
            or stat.S_IMODE(before.st_mode) & 0o022 != 0
            or not 1 <= before.st_size <= MAX_UPLOAD_BYTES
        ):
            raise ValueError
        original_fingerprint = fingerprint(before)

        staged_fd = os.open(
            staged_name,
            STAGED_FLAGS,
            0o600,
            dir_fd=operations_fd,
        )
        os.fchmod(staged_fd, 0o600)
        copied = 0
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            copied += len(chunk)
            if copied > MAX_UPLOAD_BYTES:
                raise ValueError
            remaining = memoryview(chunk)
            while remaining:
                written = os.write(staged_fd, remaining)
                if written <= 0:
                    raise OSError
                remaining = remaining[written:]
        os.fsync(staged_fd)

        after = os.fstat(source_fd)
        staged = os.fstat(staged_fd)
        if fingerprint(after) != original_fingerprint or copied != before.st_size:
            raise ValueError
        if (
            not stat.S_ISREG(staged.st_mode)
            or staged.st_uid != os.getuid()
            or staged.st_nlink != 1
            or stat.S_IMODE(staged.st_mode) != 0o600
            or staged.st_size != copied
        ):
            raise ValueError
        return f"{operations_path}/{staged_name}"
    finally:
        for descriptor in (staged_fd, source_fd, root_fd, operations_fd, tmp_fd):
            if descriptor is not None:
                os.close(descriptor)

try:
    print(stage_document(sys.argv[1]))
except (KeyError, OSError, UnicodeError, ValueError):
    raise SystemExit(1) from None
PY
) || exit 1
case "${STAGED_DOCUMENT_PATH}" in
  "${RAG_OPERATIONS_DIR}/upload-source.txt") STAGED_DOCUMENT_MIME='text/plain' ;;
  "${RAG_OPERATIONS_DIR}/upload-source.md"|"${RAG_OPERATIONS_DIR}/upload-source.markdown")
    STAGED_DOCUMENT_MIME='text/markdown'
    ;;
  *) exit 1 ;;
esac
UPLOAD_RESPONSE="${RAG_OPERATIONS_DIR}/upload-response.json"

HTTP_STATUS=$(curl -q --connect-timeout 3 --max-time 330 --noproxy '*' -sS \
  -o "${UPLOAD_RESPONSE}" -w '%{http_code}' \
  -X POST "${RAG_BASE_URL}/v1/knowledge-bases/${KB_ID}/documents" \
  --config "${AGENT_AUTH_CONFIG}" \
  -H 'Idempotency-Key: document-upload-1' \
  -F "file=@${STAGED_DOCUMENT_PATH};type=${STAGED_DOCUMENT_MIME};filename=${DOCUMENT_NAME}" \
  -F 'display_name=Operator guide' \
  -F 'metadata={"category":"guide"}' \
  -F 'tags=["operations"]')
[ "${HTTP_STATUS}" = 202 ] || exit 1

JOB_ID=$(python3 - "${UPLOAD_RESPONSE}" <<'PY'
import json
import sys
from uuid import UUID

with open(sys.argv[1], encoding="utf-8") as source:
    document = json.load(source)
if document.get("status") != "queued":
    raise SystemExit(1)
print(UUID(document["job_id"]))
PY
) || exit 1
export JOB_ID
```

The upload API persists the source to MinIO first, then creates the document, version, index state,
and job in a single PostgreSQL transaction. A failed Redis notification does not revoke an already
successful 202; the worker still discovers the job by polling PostgreSQL.

### 6. Job poll/retry: observe the durable processing state

A job's public response contains only status, stage, progress, attempt count, a redacted error, and
the parent/root links. When polling, save the body to a temporary file and read only allowed fields:

```bash
JOB_RESPONSE="${RAG_OPERATIONS_DIR}/job-response.json"
HTTP_STATUS=$(curl -q --connect-timeout 3 --max-time 15 --noproxy '*' -sS \
  -o "${JOB_RESPONSE}" -w '%{http_code}' \
  "${RAG_BASE_URL}/v1/jobs/${JOB_ID}" \
  --config "${AGENT_AUTH_CONFIG}")
[ "${HTTP_STATUS}" = 200 ] || exit 1

JOB_STATUS=$(python3 - "${JOB_RESPONSE}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    status = json.load(source)["status"]
if status not in {"queued", "running", "retry_wait", "succeeded", "failed", "cancelled"}:
    raise SystemExit(1)
print(status)
PY
) || exit 1
printf 'job_status=%s\n' "${JOB_STATUS}"
```

The worker uses PostgreSQL leases, heartbeats, and fencing. After a worker crashes or its lease
expires, another worker can resume from the committed checkpoint, and the old lease owner cannot
commit any further. A retryable transient failure first moves the same job into `retry_wait`
automatically. Only a job that is terminally `failed` with `retryable: true` can be retried by hand,
and a manual retry needs a new idempotency key and creates a child job with parent/root links:

```bash
RETRY_RESPONSE="${RAG_OPERATIONS_DIR}/retry-response.json"
HTTP_STATUS=$(curl -q --connect-timeout 3 --max-time 15 --noproxy '*' -sS \
  -o "${RETRY_RESPONSE}" -w '%{http_code}' \
  -X POST "${RAG_BASE_URL}/v1/jobs/${JOB_ID}/retry" \
  --config "${AGENT_AUTH_CONFIG}" \
  -H 'Idempotency-Key: document-job-retry-1')
[ "${HTTP_STATUS}" = 202 ] || exit 1
```

### 7. Search: queries the active generation only

Search generates the query vector from the active generation's immutable embedding snapshot, queries
only the corresponding Qdrant collection, and then batch-verifies each candidate's knowledge base,
document, version, and index state visibility against PostgreSQL. `top_k` defaults to 10 with a range
of 1..50; the query, after stripping leading and trailing whitespace, must be 1..8000 Unicode
codepoints. A metadata filter may use only the fields and operators declared in the generation's
filter-schema snapshot.

```bash
export RAG_SEARCH_TEXT='<search-text>'
SEARCH_REQUEST="${RAG_OPERATIONS_DIR}/search-request.json"
SEARCH_RESPONSE="${RAG_OPERATIONS_DIR}/search-response.json"

python3 - "${SEARCH_REQUEST}" <<'PY' || exit 1
import json
import os
import pathlib
import sys

query = os.environ["RAG_SEARCH_TEXT"].strip()
if not 1 <= len(query) <= 8000:
    raise SystemExit(1)
document = {
    "query": query,
    "top_k": 10,
}
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps(document, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
path.chmod(0o600)
PY
unset RAG_SEARCH_TEXT

HTTP_STATUS=$(curl -q --connect-timeout 3 --max-time 60 --noproxy '*' -sS \
  -o "${SEARCH_RESPONSE}" -w '%{http_code}' \
  -X POST "${RAG_BASE_URL}/v1/knowledge-bases/${KB_ID}/search" \
  --config "${AGENT_AUTH_CONFIG}" \
  -H 'Content-Type: application/json' \
  --data-binary "@${SEARCH_REQUEST}")
rm -f -- "${SEARCH_REQUEST}"
[ "${HTTP_STATUS}" = 200 ] || exit 1

python3 - "${SEARCH_RESPONSE}" <<'PY' || exit 1
import json
import sys
from uuid import UUID

with open(sys.argv[1], encoding="utf-8") as source:
    document = json.load(source)
UUID(document["index"]["generation_id"])
print(f'result_count={len(document["results"])}')
PY
```

Delete the temporary responses and curl configs once the work is done:

```bash
cleanup_rag_operations || exit 1
trap - EXIT HUP INT TERM
unset ADMIN_AUTH_CONFIG AGENT_AUTH_CONFIG DOCUMENT_NAME DOCUMENT_ROOT RAG_OPERATIONS_DIR
unset STAGED_DOCUMENT_MIME STAGED_DOCUMENT_PATH
```

---

### 8. Content: read a range of a document's text

A search result identifies a chunk, and the answer it was chosen for often continues past that
chunk's boundary. The offsets on a result address the document's normalized text directly, so
widening a hit is one more call:

```bash
curl -sS --config "${AGENT_AUTH_CONFIG}" \
  "${RAG_BASE_URL}/v1/documents/${DOCUMENT_ID}/content?start=0&end=800"
```

`start` defaults to 0 and `end` to the upload limit, and both are clamped to the document rather
than rejected, so widening around a hit near either end returns what exists. `total_codepoints`
in the response distinguishes a clamped range from an exhausted one, and `version_id` records
which version was read — the endpoint serves the document's current version only, and returns
409 `RESOURCE_STATE_CONFLICT` while a document has no active version yet.

This is the one endpoint gated on `raw_file_read`, which is a per-key flag rather than a
capability because it cuts across them: a key may legitimately search a knowledge base while not
being trusted with the source text behind a hit. Scope is checked before the flag, so a key
outside the scope cannot tell the two refusals apart.

---

## 4. Readiness checks

No authentication required. Use these to decide whether the setup can do work right now; it is more
reliable than aggregating configuration state yourself:

```bash
curl -sS "$BASE/ready"          # overall
curl -sS "$BASE/ready/ingest"   # can it ingest
curl -sS "$BASE/ready/retrieve" # can it retrieve
curl -sS "$BASE/ready/answer"   # can it answer
```

Not ready returns 503 with a reason, for example
`{"ready":false,"reason":"query_profile_not_configured"}`. The response also carries `ok` and latency
for each dependency (postgres / qdrant / redis / minio). `GET /health` is a liveness probe that
performs no dependency checks.

---

---

## 5. Decommissioning and cleanup

**Disable a provider config or model profile.** There is no delete endpoint, by design: existing index
generations reference them, and deleting would break referential integrity.

```bash
ETAG=$(curl -sS -D - -o /dev/null "$BASE/v1/admin/model-profiles/<id>" \
  -H "Authorization: Bearer $ADMIN" | awk 'tolower($1)=="etag:"{print $2}' | tr -d '\r')
curl -sS -X PATCH "$BASE/v1/admin/model-profiles/<id>" \
  -H "Authorization: Bearer $ADMIN" -H "Content-Type: application/json" \
  -H "If-Match: $ETAG" -d '{"enabled":false}'
```

**Revoke an Agent key:**

```bash
ETAG=$(curl -sS -D - -o /dev/null "$BASE/v1/admin/api-keys/<id>" \
  -H "Authorization: Bearer $ADMIN" | awk 'tolower($1)=="etag:"{print $2}' | tr -d '\r')
curl -sS -X POST "$BASE/v1/admin/api-keys/<id>/revoke" \
  -H "Authorization: Bearer $ADMIN" -H "If-Match: $ETAG"
```

**Delete a knowledge base.** Requires a `manage` key whose scope covers it. Returns 204, and the
knowledge base moves to `deleting` for background reclamation:

```bash
curl -sS -X DELETE "$BASE/v1/knowledge-bases/<kb id>" \
  -H "Authorization: Bearer $SCOPED_MANAGE" -H "If-Match: $ETAG"
```

---

---

## 6. List of traps

Ordered by how often they have actually been hit:

1. **`curl` exits 0 even on 4xx/5xx.** In a script, `curl ... && echo ok` reports false success. Take
   the status code with `-o /dev/null -w '%{http_code}'`, or add `--fail`.
2. **`manage` with an empty scope returns 404 for an existing knowledge base.** An empty scope only
   means "may create new ones". To modify or delete an existing one, its id has to be in
   `knowledge_base_ids` when the key is issued.
3. **Forgetting to create the initial index generation.** The knowledge base looks fine, but both
   ingest and search fail.
4. **A model profile `dimension` that does not match the actual model.** Get the exact value from the
   probe in 3.3 first.
5. **Calling `/v1/jobs/{id}` with an Admin token** → 401. Job queries belong to the Agent surface.
6. **Omitting `Idempotency-Key` or `If-Match`** → 422 / 412.
7. **A provider pointing at an unreachable address.** Pointing at the provider-stub that exists only
   in the test stack, for example, reports `Provider endpoint rejected`. When several configurations
   coexist, confirm by `base_url` rather than by name.
8. **Keep the `Idempotency-Key` when creating a generation; if it is lost, recover through the abandon
   endpoint.** This step calls the embedding provider synchronously, and a transient provider outage
   returns 503 `PROVIDER_UNAVAILABLE` and leaves the generation at `building`. The knowledge base then
   has no active generation (so nothing can be ingested) while retrying with a fresh idempotency key
   always returns 409 `INDEX_GENERATION_ALREADY_CONFIGURED`. Replaying with the original key resumes
   it; **if the key is lost, call
   `POST /v1/admin/knowledge-bases/{kb}/index-generations/{generation_id}/abandon`** to mark it
   `failed`, then create it again. Note that only `building` can be abandoned — `active` returns 409
   `GENERATION_NOT_ABANDONABLE`, because abandoning the active generation would make already-indexed
   documents unsearchable. Non-retryable failures such as 422 `PROVIDER_MODEL_NOT_FOUND` clean up
   after themselves and do not get stuck.
9. **With a local embedding service, keep `batch_size` at 8 or below.** TEI hard-limits the backend
   batch to 8 based on model capability and returns 422 for anything larger. Watch concurrency during
   a bulk ingest too: it returns 429 outright rather than queueing once its permits are exhausted.
10. **The reranker's batch limit must be ≥200 or reranking silently does nothing.** Retrieval sends
    `min(max(top_k*4, 20), 200)` candidates to the reranker in one call; when the server's batch limit
    is below that number the whole rerank fails, retrieval still returns the dense ordering, and the
    only trace is `retrieval.rerank.completed outcome=failed` in the log. TEI defaults to 8, and
    measured, `top_k=3` already exceeds it.

---

---

## 7. Endpoints not verified by a real run

These were only checked against the OpenAPI contract; verifying them yourself before use is
recommended:

| Endpoint | Purpose |
| --- | --- |
| `PATCH /v1/admin/provider-credentials/{id}` | rotate a provider secret |
| `GET /v1/knowledge-bases/{kb}/documents` | list documents |
| `GET /v1/documents/{id}` and `/versions` | document and version detail |
| `POST /v1/jobs/{id}/retry` | retry a failed job |
| `PUT /v1/knowledge-bases/{kb}/filter-schema` | set the filter schema |
| the `GET /v1/admin/*` list endpoints | paginated listing (`limit` ≤ 100, with a `cursor`) |

---

---

## 8. Minimal working script

From nothing to a working search, stringing sections 3 and 4 together:

```bash
#!/usr/bin/env bash
set -euo pipefail
BASE=http://127.0.0.1:8000
ADMIN=$(docker compose exec -T api velox-admin admin-key create --name bootstrap \
  | python3 -c 'import json,sys;print(json.load(sys.stdin)["token"])')

post() { curl -sS --fail-with-body -X POST "$BASE$1" \
  -H "Authorization: Bearer $2" -H "Content-Type: application/json" \
  -H "Idempotency-Key: $(uuidgen)" -d "$3"; }

CRED=$(post /v1/admin/provider-credentials "$ADMIN" \
  '{"name":"cred","secret":"'"$PROVIDER_SECRET"'"}' | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])')
CFG=$(post /v1/admin/provider-configs "$ADMIN" \
  '{"name":"p","provider_type":"openrouter","base_url":"https://openrouter.ai/api/v1","credential_id":"'"$CRED"'","default_headers":{},"routing_options":{},"timeout_seconds":"45.000","max_concurrency":3,"requests_per_minute":120,"enabled":true}' \
  | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])')
DIM=$(post "/v1/admin/provider-configs/$CFG/embedding-probe" "$ADMIN" \
  '{"model_name":"baai/bge-m3"}' | python3 -c 'import json,sys;print(json.load(sys.stdin)["dimension"])')
PROF=$(post /v1/admin/model-profiles "$ADMIN" \
  '{"name":"m","capability":"embedding","provider_config_id":"'"$CFG"'","model_name":"baai/bge-m3","dimension":'"$DIM"',"max_input_tokens":8192,"batch_size":8,"timeout_seconds":"60.000","vector_config":{},"enabled":true}' \
  | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])')
MANAGE=$(post /v1/admin/api-keys "$ADMIN" \
  '{"name":"mk","capabilities":["manage"],"knowledge_base_ids":[],"query_profile_ids":[],"default_query_profile_id":null,"raw_file_read":false,"requests_per_minute":60,"max_concurrency":2,"not_before":null,"expires_at":null}' \
  | python3 -c 'import json,sys;print(json.load(sys.stdin)["token"])')
KB=$(post /v1/knowledge-bases "$MANAGE" '{"name":"kb"}' | python3 -c 'import json,sys;print(json.load(sys.stdin)["id"])')
post "/v1/admin/knowledge-bases/$KB/index-generations" "$ADMIN" \
  '{"embedding_profile_id":"'"$PROF"'","distance":"cosine"}' >/dev/null
DATA=$(post /v1/admin/api-keys "$ADMIN" \
  '{"name":"dk","capabilities":["ingest","retrieve"],"knowledge_base_ids":["'"$KB"'"],"query_profile_ids":[],"default_query_profile_id":null,"raw_file_read":false,"requests_per_minute":120,"max_concurrency":4,"not_before":null,"expires_at":null}' \
  | python3 -c 'import json,sys;print(json.load(sys.stdin)["token"])')

curl -sS --fail-with-body -X POST "$BASE/v1/knowledge-bases/$KB/documents" \
  -H "Authorization: Bearer $DATA" -H "Idempotency-Key: $(uuidgen)" \
  -F "file=@article.md;type=text/markdown"

until curl -sS -X POST "$BASE/v1/knowledge-bases/$KB/search" \
  -H "Authorization: Bearer $DATA" -H "Content-Type: application/json" \
  -d '{"query":"test","top_k":1}' | grep -q '"results":\[{'; do sleep 2; done
echo "ready: $KB"
```
