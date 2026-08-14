# VeloxRAG

A provider-neutral RAG service for agents. The current version provides Admin/Agent API keys,
knowledge bases, encrypted provider configuration, an initial index generation, UTF-8
TXT/Markdown upload, a durable worker queue, single-knowledge-base dense retrieval, and
same-generation Qdrant disaster recovery.

## Environment and security baseline

- Python 3.12, uv 0.11.20 or compatible.
- Docker and **Docker Compose 2.23.1 or newer** — compose.yaml carries the embedding proxy's
  nginx configuration inline as a compose config, so that downloading that one file is the
  whole install, and inline `content` only exists from 2.23.1 on. Older versions parse the
  file but mount nothing, which surfaces as the nginx container failing to start.
- **On x86_64 you must start with `RAG_EMBEDDING_IMAGE_TAG=cpu-latest`** — the embedding model
  image, text-embeddings-inference, publishes no multi-architecture tag, and the default in
  compose.yaml is the arm64 one (Apple Silicon). Omitting the variable surfaces as a container
  that fails to start with an exec format error, and the error does not mention the image tag.
  Inside the repository `make start` picks the tag from `uname -m`, so you do not pass it there.
- Every Compose operation sets `COMPOSE_DISABLE_ENV_FILE=1` explicitly. Do not rely on Compose
  reading a dotenv file implicitly; ordinary configuration is passed as explicit environment
  variables, and secrets are injected through environment variables or an approved secret
  manager.
- Do not put bearer tokens, provider secrets, the encryption keyring, or database and object
  storage credentials into command line arguments, shell history, logs, or Git. The examples
  below use variable names and obvious placeholders only.
- Do not log any secret, token, authentication header, or raw response containing one.
- Write curl responses into a permission-restricted temporary directory first, then parse only
  allowed fields. Do not print key creation responses, raw job JSON, or retrieval result text.

Start the trusted local development stack:

```bash
export COMPOSE_DISABLE_ENV_FILE=1
make up
make smoke
```

### Host ports

All of them bind `127.0.0.1` only, and all of them are overridable — every one of these
defaults is a port a developer machine commonly already has taken, so a collision does not
mean editing the file:

| Entry point | Default | Override |
| --- | --- | --- |
| API | `http://127.0.0.1:8000` | `RAG_API_HOST_PORT` |
| PostgreSQL | `127.0.0.1:5432` | `RAG_POSTGRES_HOST_PORT` |
| Qdrant HTTP / gRPC | `127.0.0.1:6333` / `6334` | `RAG_QDRANT_HOST_PORT` / `RAG_QDRANT_GRPC_HOST_PORT` |
| Redis | `127.0.0.1:6379` | `RAG_REDIS_HOST_PORT` |
| MinIO API / console | `127.0.0.1:9000` / `http://127.0.0.1:9001` | `RAG_MINIO_HOST_PORT` / `RAG_MINIO_CONSOLE_HOST_PORT` |

Only the host-side mapping changes; containers reach each other by service name, so overriding
one never has to be matched elsewhere. The single exception is `RAG_API_HOST_PORT`: an MCP
client defaults to `http://127.0.0.1:8000`, so changing the API port means setting
`RAG_MCP_BASE_URL` as well. `make start` probes whether 6379 is taken and falls back to 6380;
it does not probe the other ports.

The MinIO console at `http://127.0.0.1:9001` signs in with the development-only defaults
`rag-dev` / `change-me-local`, which come from `MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD`. They
are published placeholders rather than secrets, and a production deployment has to replace them.
Note which side the startup check actually reads: `RAG_ENVIRONMENT=production` refuses to start
when the *client* credentials are still the defaults — `RAG_MINIO_ACCESS_KEY` still `rag-dev`, or
`RAG_MINIO_SECRET_KEY` carrying a `change-me` marker. `MINIO_ROOT_*` configures the server and is
not covered by that check, so changing only one side gets you a stack that starts and then cannot
authenticate.

MinIO holds original files, normalized text, and canonical chunk manifests;
it is not a vector database. Qdrant holds vectors and retrieval payloads, both
rebuildable from those canonical artifacts. PostgreSQL is the authoritative source for document visibility, jobs, generations,
checkpoints, and authorization facts. Redis only wakes the worker with low latency; it is not
the sole source of truth for the job queue.

Stop the services but keep the data volumes:

```bash
COMPOSE_DISABLE_ENV_FILE=1 docker compose --profile provider-stub down
```

A production deployment must not use the development-only values in `.env.example`, and must
not expose the Compose ports to an untrusted network.

## Admin and Agent keys

To avoid overwriting traps that already exist in your interactive shell, start a dedicated Bash
subshell first (run `bash`, for example) and complete the creation, import, and cleanup steps
below in that dedicated session, in order.

After the first start, create an Admin key with the in-container CLI. The response contains a
one-time token, so write it to a permission-restricted file and move it into a password manager
immediately, without letting it reach the terminal or a log:

```bash
cleanup_admin_bootstrap() {
  local cleanup_file="${ADMIN_BOOTSTRAP_FILE:-}"

  if [[ ! "${cleanup_file}" =~ ^/tmp/velox-admin-created\.[A-Za-z0-9]{6}$ ]]; then
    return 0
  fi
  rm -f -- "${cleanup_file}"
}

ADMIN_BOOTSTRAP_FILE=$(umask 077 && mktemp "/tmp/velox-admin-created.XXXXXX") || exit 1
trap cleanup_admin_bootstrap EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
COMPOSE_DISABLE_ENV_FILE=1 docker compose run --rm api velox-admin admin-key create \
  --name local-admin >"${ADMIN_BOOTSTRAP_FILE}" || exit 1
```

Once the CLI succeeds the response is still in a file with mode `0600`, and the EXIT/signal
traps are still armed. Staying in the same dedicated Bash session, import that one-time response
straight into an approved password manager; do not `cat` it. Only after you have confirmed the
import, run the separate normal cleanup:

```bash
cleanup_admin_bootstrap || exit 1
trap - EXIT HUP INT TERM
unset ADMIN_BOOTSTRAP_FILE
```

Exit the dedicated Bash subshell once cleanup is done.

Inspect security metadata, or revoke an Admin key:

```bash
COMPOSE_DISABLE_ENV_FILE=1 docker compose run --rm api velox-admin admin-key list
COMPOSE_DISABLE_ENV_FILE=1 docker compose run --rm api velox-admin admin-key revoke <key-id>
```

A lost Admin token cannot be recovered or shown again. Use the local CLI `list` to find the old
key id, `revoke` it, and `create` a replacement. When an Agent token is lost, find its security
record through the Admin API's `GET /v1/admin/api-keys`, revoke it against the latest ETag, then
create a replacement Agent key; do not leave a valid key you no longer control.

An Admin bearer token is for the admin API only and cannot stand in for an Agent token on the
KB API. Agent capabilities (`ingest`, `retrieve`, `answer`, `manage`) and KB scope are two
independent checks; `admin` is not an Agent capability. Unauthorized resources and nonexistent
resources both return 404, which denies an enumeration side channel.

Inject the tokens from a password manager or an approved secret manager, then create
configuration files that keep the token out of curl's arguments:

```bash
cleanup_rag_operations() {
  local cleanup_dir="${RAG_OPERATIONS_DIR:-}"
  local cleanup_name

  if [[ ! "${cleanup_dir}" =~ ^/tmp/rag-operations\.[A-Za-z0-9]{6}$ ]]; then
    return 0
  fi
  for cleanup_name in \
    admin-auth.conf \
    agent-auth.conf \
    generation-request.json \
    generation-response.json \
    job-response.json \
    model-profile-request.json \
    model-profile-response.json \
    provider-config-request.json \
    provider-config-response.json \
    provider-credential-request.json \
    provider-credential-response.json \
    retry-response.json \
    search-request.json \
    search-response.json \
    upload-source.markdown \
    upload-source.md \
    upload-source.txt \
    upload-response.json
  do
    rm -f -- "${cleanup_dir}/${cleanup_name}"
  done
  rmdir -- "${cleanup_dir}"
}

RAG_OPERATIONS_DIR=$(umask 077 && mktemp -d "/tmp/rag-operations.XXXXXX") || exit 1
trap cleanup_rag_operations EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
ADMIN_AUTH_CONFIG="${RAG_OPERATIONS_DIR}/admin-auth.conf"
AGENT_AUTH_CONFIG="${RAG_OPERATIONS_DIR}/agent-auth.conf"

: "${RAG_ADMIN_TOKEN:?inject the Admin token through an approved secret source}"
python3 - "${ADMIN_AUTH_CONFIG}" <<'PY' || exit 1
import os
import pathlib
import sys

token = os.environ["RAG_ADMIN_TOKEN"]
if not token or len(token) > 256 or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-" for c in token):
    raise SystemExit(1)
path = pathlib.Path(sys.argv[1])
path.write_text(f'header = "Authorization: Bearer {token}"\n', encoding="utf-8")
path.chmod(0o600)
PY
unset RAG_ADMIN_TOKEN

: "${RAG_AGENT_TOKEN:?inject the Agent token through an approved secret source}"
python3 - "${AGENT_AUTH_CONFIG}" <<'PY' || exit 1
import os
import pathlib
import sys

token = os.environ["RAG_AGENT_TOKEN"]
if not token or len(token) > 256 or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-" for c in token):
    raise SystemExit(1)
path = pathlib.Path(sys.argv[1])
path.write_text(f'header = "Authorization: Bearer {token}"\n', encoding="utf-8")
path.chmod(0o600)
PY
unset RAG_AGENT_TOKEN
```

Agent keys are created by `POST /v1/admin/api-keys`. That response also contains a one-time
token; redirect it to a permission-restricted file and import it into a password manager
immediately. An Agent with the `manage` capability and an initially empty scope receives an
explicit grant for a KB it creates. Every create uses `Idempotency-Key`; every PATCH uses the
latest strong ETag with `If-Match`. Each successful modification produces a new ETag, including
the Agent scope change that follows creating a KB. When updating a KB in sequence, the first
PATCH uses `<etag-from-create-response>`, the following filter schema PUT uses
`<etag-from-patch-response>`, and the next operation uses
`<etag-from-filter-schema-response>`; a missing or stale precondition returns 412. Run the
existing authenticated metadata smoke:

```bash
COMPOSE_DISABLE_ENV_FILE=1 make smoke-auth
```

## How to call it

The service has no UI; everything goes through the HTTP API. **The authoritative contract comes
from the service itself** (generated from the code, so it cannot go stale):

| Endpoint | Purpose |
| --- | --- |
| `GET /openapi.json` | All 39 operations and their schemas. Point an AI or a tool at this one |
| `GET /docs` | Swagger UI, interactive (frontend assets come from a CDN, so it needs network) |
| `GET /redoc` | ReDoc |

Step-by-step operations, field constraints, failure semantics, and a list of traps are in
[docs/api-operations.md](docs/api-operations.md), and **production requirements and the points of no
return are in [docs/deployment.md](docs/deployment.md)**.

To use it as a memory layer for local agents such as Claude Code over MCP, see
[Agent memory over MCP](#agent-memory-over-mcp) below and [docs/mcp.md](docs/mcp.md).

Reranking is supported: set `rerank_profile_id` on the knowledge base
(`PATCH /v1/knowledge-bases/{id}`) and pass `"rerank": true` on the search. On a 70-query
measurement over a real corpus it lifted answer-hit MRR from 0.833 to 0.911 and helped Chinese and
English queries alike. It is a per-request switch rather than always-on because only a minority of
queries change position, and the rest pay for a provider round trip for nothing. It also carries a
deployment requirement — see the reranker batch limit in
[docs/deployment.md](docs/deployment.md).

**`compose.yaml` alone is enough**: `docker compose up -d` brings up the containerized embedding
model (`BAAI/bge-m3`, 1024 dimensions) along with everything else, and completes the provider
configuration, model profile, knowledge base, filter schema, and initial generation on its own;
after that an MCP client needs no configuration to search. The first start downloads about
4.8 GB of weights.

```bash
docker compose up -d                                        # arm64 / Apple Silicon
RAG_EMBEDDING_IMAGE_TAG=cpu-latest docker compose up -d      # x86_64
```

Inside the repository use `make start`, which additionally rebuilds the api image and picks the
architecture. To make the one-file install true, compose defaults two switches that hold only
locally; see [docs/mcp.md](docs/mcp.md). The service refuses to start with either of them in a
production environment. If a host port collides with something already running, see
[Host ports](#host-ports) above.

### Agent memory over MCP

The HTTP API is complete but an agent cannot practically use it: it would have to mint an admin
token from the container CLI, sign an agent key, remember the knowledge base id, and hand-write
requests. The MCP server holds the credential and the knowledge base itself and exposes only what
an agent needs — three read-only tools, `search_memory` / `list_documents` / `memory_status`.

Point a client at it with one command, no checkout required:

```bash
claude mcp add --scope user rag-memory -- uvx --from git+https://github.com/ilikebug/veloxrag velox-mcp
```

Clients that take a config file instead:

```json
{
  "mcpServers": {
    "rag-memory": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/ilikebug/veloxrag", "velox-mcp"]
    }
  }
}
```

Working inside a checkout, run it from the working tree instead:

```bash
claude mcp add --scope user rag-memory -- uv run --project /path/to/VeloxRAG velox-mcp
```

The stack has to be up first — the MCP server is only a client and does not start it. No
environment variables are needed; all three are optional and only for departing from the
defaults:

| Variable | When you need it |
| --- | --- |
| `RAG_MCP_BASE_URL` | The service is not at `http://127.0.0.1:8000`, which includes having overridden `RAG_API_HOST_PORT` |
| `RAG_MCP_TOKEN` | Local trusted auth is switched off; supply an agent key carrying `retrieve` |
| `RAG_MCP_KNOWLEDGE_BASE` | The service holds more than one knowledge base. With exactly one it resolves automatically; with several it refuses to guess, because guessing wrong means searching the wrong memory |

Three things to know before you start calling the HTTP API directly.

**Authentication has three levels and you cannot skip one.** Admin tokens can only be minted by
the in-container CLI (`velox-admin admin-key create`; there is no API); they sign Agent keys; and
an Agent key's capabilities decide what it may call: `manage` creates and modifies knowledge
bases, `ingest` uploads documents and reads jobs, `retrieve` searches, `answer` answers. Scope is
a hard constraint — a `manage` key with an empty `knowledge_base_ids` can create new knowledge
bases but returns 404 for any that already exists.

**The configuration order is fixed and skipping a step fails.** Provider credential →
ProviderConfig → (preferably `POST /v1/admin/provider-configs/{id}/embedding-probe` first, to get
the exact dimension) → ModelProfile → knowledge base → **initial index generation** → Agent key →
upload → search. Without the initial index generation the knowledge base looks fine but both
ingest and search fail.

**Three general rules.** Creates must carry `Idempotency-Key` (reuse the same one to retry, use a
new one once the body changes); `PATCH`/`DELETE`/`revoke` must carry `If-Match` (GET the `ETag`
first); and the service reflects request fields back for validation, so a hand-rolled mock or a
rewritten response body is rejected as invalid.

To decide whether the service can do work right now, use the readiness probes rather than
aggregating configuration state yourself: `GET /ready`, `/ready/ingest`, `/ready/retrieve`,
`/ready/answer` (not ready returns 503 with a `reason`).

## Same-generation Qdrant repair

If the active generation's Qdrant collection is gone, or a compatible and verified empty
collection already exists, you can schedule a `rebuild_generation` job inside the running
controlled stack:

```bash
cleanup_repair_response() {
  local cleanup_file="${REPAIR_RESPONSE:-}"

  if [[ ! "${cleanup_file}" =~ ^/tmp/rag-generation-repair\.[A-Za-z0-9]{6}$ ]]; then
    return 0
  fi
  rm -f -- "${cleanup_file}"
}

REPAIR_RESPONSE=$(umask 077 && mktemp "/tmp/rag-generation-repair.XXXXXX") || exit 1
trap cleanup_repair_response EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
COMPOSE_DISABLE_ENV_FILE=1 docker compose run --rm -T --no-deps api \
  velox-admin repair-generation --generation-id "${GENERATION_ID}" >"${REPAIR_RESPONSE}" || exit 1
REPAIR_JOB_ID=$(python3 - "${REPAIR_RESPONSE}" <<'PY'
import json
import os
import sys
from uuid import UUID

with open(sys.argv[1], encoding="utf-8") as source:
    document = json.load(source)
if not isinstance(document, dict) or set(document) != {"generation_id", "job_id", "status"}:
    raise SystemExit(1)
try:
    generation_id = UUID(document["generation_id"])
    job_id = UUID(document["job_id"])
    status = document["status"]
    if generation_id != UUID(os.environ["GENERATION_ID"]) or status != "queued":
        raise ValueError
except (KeyError, TypeError, ValueError):
    raise SystemExit(1) from None
print(job_id)
PY
) || exit 1
export REPAIR_JOB_ID
cleanup_repair_response || exit 1
trap - EXIT HUP INT TERM
unset REPAIR_RESPONSE
```

You can now request `GET /v1/jobs/${REPAIR_JOB_ID}` with the job poll flow described earlier; the
variable holds only a validated job UUID, and the raw CLI response has been deleted and never
printed.

The repair uses the original generation's immutable snapshot, canonical chunk manifests, and
deterministic point IDs. It does not change the generation ID, the profile, the KB active
pointer, or document visibility. A non-empty conflicting collection, a missing canonical
manifest, a non-active generation, and a concurrent repair are all rejected. This is disaster
recovery within the same vector space, not a cross-profile rebuild or cutover.

## Rotating and losing the provider credential keyring

`RAG_PROVIDER_CREDENTIAL_KEYRING` is a JSON mapping of `key_version -> base64 AES-256 key`, and
`RAG_PROVIDER_CREDENTIAL_ACTIVE_KEY_VERSION` names the version used for new writes. The API and
the worker must load the same keyring; the keyring is injected only through environment variables
or an approved secret manager, and is never written to PostgreSQL, logs, or Git.

A safe rotation order:

1. In the secret manager, add the new version to the keyring while keeping every old version any
   credential still references, and change the active version to the new one.
2. Roll-restart the API and the worker. Check only HTTP status and allowed fields:
   `/ready/ingest` and `/ready/retrieve` must succeed; readiness fails closed on an invalid
   keyring or when a referenced credential cannot be decrypted.
3. For each credential, obtain the current secret again from the provider's trusted secret source
   (or a provider secret you have already rotated) and call
   `PATCH /v1/admin/provider-credentials/{credential_id}` with the latest `If-Match`, submitting
   `secret`. This keeps the credential ID, re-encrypts under the active key version with a new
   nonce, and increments the resource revision.
4. Read the credential's security metadata and confirm `key_version` is now the new version, then
   verify the initial generation probe, document ingest, and the search path. Do not verify
   plaintext through logs or a database export.
5. Only once you have confirmed that no credential still references the old version, and have
   taken independent backups of the database and the keyring, remove the old version from the
   runtime keyring, roll-restart again, and check readiness.

A database backup contains ciphertext only. The keyring backup must be stored separately from the
database backup, authorized separately, and restore-tested independently. If every key that can
decrypt a given old ciphertext is lost, that provider secret is unrecoverable; it has to be
obtained again from the provider or a password manager and PATCHed back in. If the original
provider secret is also unobtainable, the only path is to issue a replacement secret on the
provider side first and then enter it again. Do not try to derive plaintext from ciphertext, a
nonce, or logs when the old key is missing.

The Admin/Agent HMAC secrets and the provider credential keyring are different mechanisms:
changing `RAG_ADMIN_KEY_HMAC_SECRET` immediately invalidates all existing Admin tokens but does
not invalidate Agent tokens; changing `RAG_AGENT_KEY_HMAC_SECRET` invalidates all existing Agent
tokens only. There is no old-HMAC-secret fallback today, so each key class needs its own
changeover and re-issuance plan.

## Readiness and recovery semantics

- `GET /health`: checks only that the API process is alive.
- `GET /ready`: the core PostgreSQL and Qdrant dependencies.
- `GET /ready/ingest`: additionally requires Redis, MinIO, and the provider keyring and
  referenced configuration to satisfy the ingest path.
- `GET /ready/retrieve`: checks the retrieval dependencies and the provider keyring and
  referenced configuration.
- `GET /ready/answer`: returns 503 `query_profile_not_configured` when no query profile is
  configured. Answer generation is outside this layer's responsibility, so this probe never
  becomes ready; use `/ready` or `/ready/retrieve` in a health check.
- The worker periodically scans PostgreSQL for queued jobs, jobs whose retry_wait has come due,
  and running jobs with an expired lease; losing Redis only adds wake-up latency and never loses
  a committed job.
- External side effects in MinIO, Qdrant, and providers all have deterministic identifiers and
  PostgreSQL reconciliation facts; a restarted worker resumes from a committed checkpoint and
  does not trust uncommitted transient external state.

## Current limitations

Not yet supported:

- PDF, DOCX, OCR, or other binary document formats;
- document replacement, new versions, or user-facing delete recovery (a delete is a real delete
  and cannot be undone);
- LLM semantic chunking;
- sparse and hybrid retrieval (reranking is supported, see above);
- answer generation;
- cross-KB search;
- cross-profile rebuild, new generation cutover, or active/pending switching.

## Development, verification, and cleanup

```bash
uv sync --frozen
make check
make verify
make build-arm64
```

`make check` is the Python static checking; `make verify` additionally runs unit,
non-acceptance integration, isolated acceptance, and Compose publication verification.

## Building and publishing the image

```bash
make build TAG=0.1.0
make push VELOX_IMAGE=docker.io/<your namespace>/veloxrag TAG=0.1.0
```

`build` only builds this machine's architecture into the local image store, for running and
inspecting; both targets require `TAG`.

`push` rebuilds rather than pushing what `build` produced: consumers run both amd64 and arm64,
and buildx cannot hold a multi-architecture manifest in the local image store, so the two
platforms have to go straight to the registry from one invocation. `push` needs a prior
`docker login`, and `VELOX_IMAGE` has to match what `compose.yaml` resolves to for consumers —
otherwise whoever downloads `compose.yaml` cannot pull what you pushed.

`make verify` runs unit, non-acceptance integration, and live acceptance separately, then
combines fresh `.coverage.unit`, `.coverage.integration`, and `.coverage.acceptance` with a
branch coverage floor of 80%. The unified live acceptance run uses a random Compose project, a
dynamic loopback API port, temporary volumes, and a temporary image tag, and cleans up on exit:

```bash
COMPOSE_DISABLE_ENV_FILE=1 make acceptance-ingestion-retrieval
```

Compose configuration check:

```bash
COMPOSE_DISABLE_ENV_FILE=1 make compose-config
```

A production environment must replace the development-only database, MinIO, HMAC, and provider
keyring values. Special characters in the database URL need percent-encoding; `POSTGRES_*` and
`RAG_DATABASE_URL`, and `MINIO_ROOT_*` and `RAG_MINIO_*`, are two independent sets of
configuration whose targets and credentials have to stay consistent.

If Testcontainers Ryuk races container startup on an M3 Mac, run:

```bash
TESTCONTAINERS_RYUK_DISABLED=true make test-integration
```

With Ryuk disabled, the test context is still responsible for cleaning up containers.

## License

MIT, see [LICENSE](LICENSE).
