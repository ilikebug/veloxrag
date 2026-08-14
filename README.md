# VeloxRAG

A local RAG service that acts as memory for coding agents. It ingests documents and past
sessions, retrieves passages with citation-grade offsets, and exposes them to an agent over MCP.
Everything runs on your machine: the corpus never leaves it.

It supplies data; composing the answer is the agent's job. That boundary is deliberate — see
[what it does not do](#current-limitations).

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/ilikebug/veloxrag/main/install.sh | bash
```

Installs Ollama if it is missing, pulls `bge-m3` (about 1.2 GB, once), writes `compose.yaml` into
`~/.veloxrag`, starts the stack, and prints the command to connect an agent. Re-running converges
rather than reinstalling, and never touches existing data volumes.

What it needs first:

- **Docker, running**, with **Compose 2.23.1 or newer**. compose.yaml carries the embedding
  proxy's nginx configuration inline, and inline `content` only exists from that version; older
  Compose parses the file and mounts nothing, which surfaces as the nginx container failing to
  start. The installer will not install Docker for you — on macOS that is a choice between
  Docker Desktop and Colima that belongs to you.
- **2 CPUs, 4 GiB of memory and about 4 GB of disk** for the Docker VM — the containers idle at
  roughly 450 MiB in total, so the headroom is for the ingest and the corpus rather than the
  services. [Resources it needs](#resources-it-needs) has the measured figures and where to set
  the limits.
- **Ollama on the host.** The embedding model runs on the host rather than in a container because
  that is the only place it reaches the GPU: Docker on macOS is a Linux VM with no Metal
  passthrough, measured at 3.10 chunks/s against 14.20 on the host, and the flat batch curve says
  the container is compute-bound rather than badly tuned. `install.sh` handles it; by hand it is
  `brew install ollama && brew services start ollama && ollama pull bge-m3`.

After a reboot the containers come back on their own — they carry `restart: unless-stopped` — but
only once the Docker daemon is up, so Docker Desktop or Colima has to start at login. Ollama is a
host process with its own arrangement (`brew services` on macOS, a systemd unit on Linux). If
retrieval fails after a reboot, check Ollama first: `curl http://127.0.0.1:11434/api/version`.

Upgrading from a release that ran the embedding model in a container, pass `--remove-orphans`
once. Compose only removes containers it still knows about, and the retired `embedding-model` is
no longer in the file, so it keeps running and holding memory until told otherwise:

```bash
cd ~/.veloxrag && docker compose up -d --remove-orphans
```

## Connect an agent

```bash
claude mcp add --scope user rag-memory -- uvx --from git+https://github.com/ilikebug/veloxrag velox-mcp
```

No checkout and no token. For a client that takes a config file:

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

Working inside a checkout, run it from the working tree so code changes take effect immediately:

```bash
claude mcp add --scope user rag-memory -- uv run --project /path/to/VeloxRAG velox-mcp
```

Four read-only tools:

| Tool | Purpose |
| --- | --- |
| `search_memory` | Retrieval, with an optional `source_type` filter |
| `read_document` | Read a document's text around a character range, to see what a result was cut off from |
| `list_documents` | See what is indexed, to narrow a search or notice a gap |
| `memory_status` | Which knowledge base is bound, and whether retrieval is ready |

Ingestion, knowledge base creation and key minting are deliberately absent: an agent that can
provision storage can also destroy it, and deletion here is real.

The stack has to be up first — the MCP server is only a client. No environment variables are
needed; all three exist for departing from the defaults:

| Variable | When you need it |
| --- | --- |
| `RAG_MCP_BASE_URL` | The service is not at `http://127.0.0.1:8000`, which includes having overridden `RAG_API_HOST_PORT` |
| `RAG_MCP_TOKEN` | Local trusted auth is switched off; supply an agent key carrying `retrieve` |
| `RAG_MCP_KNOWLEDGE_BASE` | The service holds more than one knowledge base. With exactly one it resolves automatically; with several it refuses to guess, because guessing wrong means searching the wrong memory |

More detail, including why two compose defaults hold only locally, is in
[docs/mcp.md](docs/mcp.md).

## Retrieval

**Relevance judgement is left to the agent, not done by a reranker.** An agent already reads the
passages and reasons about them, which is what a cross-encoder does with a far smaller model. So
the useful thing is not another ranking pass but giving that judgement something to work with:
retrieve more passages than you need, then widen the promising ones before deciding.

A search hit is a chunk, and the answer frequently sits just past its edge. Every result carries
`source.start_offset` / `source.end_offset`, and `GET /v1/documents/{id}/content?start=&end=`
reads that range back out of the document's normalized text — the same offsets, no second
addressing scheme. Measured over 18 queries on a chat-transcript corpus, widening each hit by 300
characters moved answer-level MRR from 0.645 to 0.724 and took @5 and @10 from 0.89 and 0.94 to
1.00. A passage that looks truncated is worth widening rather than discarding.

What that measurement cannot reach: @10 in-chunk was 0.94, so roughly one query in sixteen
returns no candidate holding the answer at all. Closing that needs better retrieval — hybrid
search — rather than better judgement.

Chunking defaults to 600 codepoints with 100 overlap, and those defaults are measured rather than
guessed: on English documentation, moving from 1200 to 600 lifted answer-hit MRR from 0.631 to
0.836, the largest single quality gain found. On a chat-transcript corpus the size barely mattered
(600 scored 0.675 against 300's 0.679, inside the noise at that sample size) because transcript
turns are short already. `RAG_CHUNK_MAX_CODEPOINTS` and `RAG_CHUNK_OVERLAP_CODEPOINTS` change it;
the worker reads them at process start, so a change needs a worker restart.

Reranking exists in the service but has no engine behind it in the default setup: Ollama exposes
no rerank endpoint. Setting `"rerank": true` without a configured rerank profile fails with
`RERANK_NOT_CONFIGURED`.

## What runs, and where the data lives

Every host port binds `127.0.0.1` only, and every one is overridable — these defaults are all
ports a developer machine commonly already has taken:

| Entry point | Default | Override |
| --- | --- | --- |
| API | `http://127.0.0.1:8000` | `RAG_API_HOST_PORT` |
| PostgreSQL | `127.0.0.1:5432` | `RAG_POSTGRES_HOST_PORT` |
| Qdrant HTTP / gRPC | `127.0.0.1:6333` / `6334` | `RAG_QDRANT_HOST_PORT` / `RAG_QDRANT_GRPC_HOST_PORT` |
| Redis | `127.0.0.1:6379` | `RAG_REDIS_HOST_PORT` |
| MinIO API / console | `127.0.0.1:9000` / `http://127.0.0.1:9001` | `RAG_MINIO_HOST_PORT` / `RAG_MINIO_CONSOLE_HOST_PORT` |

Only the host-side mapping changes; containers reach each other by service name. The one exception
is `RAG_API_HOST_PORT`: an MCP client defaults to `http://127.0.0.1:8000`, so changing the API
port means setting `RAG_MCP_BASE_URL` too. `make start` probes whether 6379 is taken and falls
back to 6380; it does not probe the others.

The MinIO console signs in with the development-only defaults `rag-dev` / `change-me-local`, from
`MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD`. They are published placeholders rather than secrets,
and production has to replace them. Note which side the startup check reads:
`RAG_ENVIRONMENT=production` refuses to start when the *client* credentials are still the defaults
— `RAG_MINIO_ACCESS_KEY` still `rag-dev`, or `RAG_MINIO_SECRET_KEY` carrying a `change-me` marker.
`MINIO_ROOT_*` configures the server and is not covered, so changing one side alone gets a stack
that starts and then cannot authenticate.

What each component owns, which is also the backup priority:

- **PostgreSQL** is the authoritative source for document visibility, jobs, generations,
  checkpoints and authorization. Losing it cannot be recovered from the others.
- **MinIO** holds original files, normalized text and canonical chunk manifests;
  it is not a vector database. Vectors can be rebuilt from it, the originals cannot.
- **Qdrant** holds vectors and retrieval payloads, both rebuildable from those canonical
  artifacts.
- **Redis** only wakes the worker with low latency. Losing it adds delay and no data.

Do not log any secret, token, authentication header, or raw response containing one, and keep
credentials out of command line arguments, shell history and Git.

## Resources it needs

Measured on the running stack rather than estimated. Idle, the seven containers hold about
450 MiB between them:

| Container | Idle memory | What it does under load |
| --- | --- | --- |
| api | 137 MiB | CPU spikes to ~15% while accepting an upload |
| worker | 124 MiB | one core to ~70% while chunking; memory flat |
| minio | 77 MiB | — |
| qdrant | 57 MiB | grows with the index, see below |
| postgres | 48 MiB | — |
| redis | 6 MiB | — |
| embedding (nginx) | 2 MiB | proxy only; the model is on the host |

Memory barely moves during ingestion because the expensive part — embedding — runs in Ollama on
the host, not in a container. That is also why a machine that could not previously fit the
containerized model can run this: the container side needs well under 1 GiB.

Storage, measured against a small corpus and linear in the number of chunks:

| What | Size |
| --- | --- |
| Images, all seven | 1.8 GB |
| Ollama plus `bge-m3` (host, outside Docker) | about 1.5 GB |
| Postgres, empty schema | 65 MiB |
| Qdrant | about 30 KB per chunk at 1024 dimensions |
| MinIO | roughly the size of the corpus, plus normalized text and chunk manifests |

A rough total: 4 GB of disk covers the images, the model and a corpus of a few thousand chunks.
The number that grows is Qdrant, and a cutover doubles it until the retired collection is removed
by hand.

### Setting the limits

Nothing in `compose.yaml` caps CPU or memory, deliberately: the ceiling that matters is the one on
the Linux VM your Docker runs in, and a per-container cap below it only turns a slow ingest into a
killed one.

On macOS the VM is where to set it. When Colima is installed but not running, `install.sh` starts
it with 4 CPUs, 8 GiB and 60 GiB, overridable through `VELOX_VM_CPU`, `VELOX_VM_MEMORY` and
`VELOX_VM_DISK`. A **running** VM it leaves alone, and so should you by this route:

```bash
colima stop && colima start --cpu 4 --memory 8 --disk 60
```

The stop matters. `colima start` does not resize a running instance, but `--save-config` defaults
to true, so passing the flags to a live VM rewrites the config without applying it — the machine
keeps its old size until the next restart silently adopts the new one. A Colima disk can also grow
later but not shrink, so err large on that one.

Docker Desktop has the same three under Settings → Resources. 2 CPUs and 4 GiB run the stack; 4
CPUs and 8 GiB leave room for the host Ollama to use the GPU without competing for RAM. On Linux
there is no VM and the containers use the host directly.

**Give the disk more room than the corpus needs.** A full disk fails in two directions at once:
Qdrant refuses writes with `No space left on device`, and — measured, not theorized — a Docker
build in the same state fails without saying why, so the next thing you try appears broken for an
unrelated reason. `docker system df` shows where it went; build cache and old images are usually
most of it.

If you do want a per-container cap, compose takes one:

```yaml
services:
  worker:
    deploy:
      resources:
        limits:
          memory: 1g
```

## HTTP API

The service has no UI. **The authoritative contract comes from the service itself**, generated
from the code so it cannot go stale:

| Endpoint | Purpose |
| --- | --- |
| `GET /openapi.json` | All 40 operations and their schemas. Point an AI or a tool at this one |
| `GET /docs` | Swagger UI, interactive (frontend assets come from a CDN, so it needs network) |
| `GET /redoc` | ReDoc |

Step-by-step operations, including minting keys and the order configuration has to happen in, are
in [docs/api-operations.md](docs/api-operations.md). **Production requirements and the points of
no return are in [docs/deployment.md](docs/deployment.md)**, and the embedding setup is in
[docs/local-embedding.md](docs/local-embedding.md).

Three things that make requests fail in ways the error does not explain:

- **Authentication has three levels and none is skippable.** Admin tokens are minted only by the
  in-container CLI; they sign Agent keys; an Agent key's capabilities decide what it may call.
  Scope is a hard constraint — a `manage` key with an empty `knowledge_base_ids` can create
  knowledge bases but returns 404 for any that already exists.
- **Configuration order is fixed.** Provider credential → ProviderConfig → embedding probe →
  ModelProfile → knowledge base → **initial index generation** → Agent key → upload → search.
  Without that generation the knowledge base looks fine while ingest and search both fail. The
  installer does all of this for you.
- **Creates need `Idempotency-Key`, modifications need `If-Match`.** Reuse the same idempotency
  key to retry and a new one once the body changes; GET the `ETag` before a `PATCH`, `DELETE` or
  `revoke`.

To decide whether the service can work right now, use the readiness probes rather than
aggregating configuration state:

- `GET /health` — the API process is alive.
- `GET /ready` — the core PostgreSQL and Qdrant dependencies.
- `GET /ready/ingest` — additionally Redis, MinIO, and the provider keyring and referenced
  configuration.
- `GET /ready/retrieve` — the retrieval dependencies and the same provider configuration.
- `GET /ready/answer` — always 503. Answer generation is outside this layer's responsibility, so
  this probe never becomes ready; do not wire it into a health check.

The worker scans PostgreSQL for queued jobs, jobs whose retry_wait came due, and running jobs with
an expired lease, so losing Redis adds wake-up latency and never loses a committed job. External
effects in MinIO, Qdrant and providers all carry deterministic identifiers and PostgreSQL
reconciliation facts; a restarted worker resumes from a committed checkpoint rather than trusting
uncommitted external state.

## Changing the index configuration

The embedding model, chunk size, distance metric and filter schema are frozen by the generation
that uses them. Changing one is a cutover: create a second generation on the same knowledge base
and the service swaps to it, enrols the existing documents and queues their backfill in one
transaction, so the knowledge base id never changes and nothing downstream is reconfigured.

```bash
curl -sS -X POST "http://127.0.0.1:8000/v1/admin/knowledge-bases/${KB}/index-generations" \
  -H 'Content-Type: application/json' -H "Idempotency-Key: $(uuidgen)" \
  -d '{"embedding_profile_id":"'"${PROFILE}"'","distance":"cosine"}'
```

Search returns nothing until the backfill finishes, which is a deliberate trade: a single-user
service can be silent for the minutes a rebuild takes, and the alternative — writing to both
generations during the rebuild — is a locked hot-path change. Watch it with
`GET /v1/jobs/{job_id}`.

The retired generation's Qdrant collection is not reclaimed yet, so each cutover leaves one behind.

## Current limitations

Not yet supported:

- PDF, DOCX, OCR or other binary document formats;
- document replacement or new versions, and no user-facing delete recovery — a delete is real and
  cannot be undone;
- LLM semantic chunking, and only one chunking strategy is registered;
- sparse and hybrid retrieval;
- reranking in the default setup, for want of an engine that offers it;
- answer generation, and cross-KB search — both by design, see the boundary above;
- reclaiming a retired generation's collection.

## Development

```bash
uv sync --frozen
make check
make verify
```

`make check` is the static checking; `make verify` adds unit, non-acceptance integration, isolated
acceptance and Compose publication verification, combining fresh coverage files against a branch
floor of 80%.

```bash
COMPOSE_DISABLE_ENV_FILE=1 make acceptance-ingestion-retrieval
COMPOSE_DISABLE_ENV_FILE=1 make compose-config
```

Inside the repository, `make start` builds the api image from the working tree and refuses to start
when Ollama is not answering. Every Compose operation sets `COMPOSE_DISABLE_ENV_FILE=1`
explicitly: configuration arrives as explicit environment variables rather than an implicitly read
dotenv file.

If Testcontainers Ryuk races container startup on an M3 Mac:

```bash
TESTCONTAINERS_RYUK_DISABLED=true make test-integration
```

## Publishing a release

```bash
make build TAG=0.4.0
make push VELOX_IMAGE=docker.io/<your namespace>/veloxrag TAG=0.4.0
```

`build` produces this machine's architecture in the local image store, for running and inspecting.
`push` rebuilds rather than pushing what `build` produced: consumers run both amd64 and arm64, and
buildx cannot hold a multi-architecture manifest locally, so both platforms go straight to the
registry from one invocation. It needs a prior `docker login`, and `VELOX_IMAGE` has to match what
`compose.yaml` resolves to for consumers — otherwise whoever downloads `compose.yaml` cannot pull
what you pushed. Bump the version in `pyproject.toml` before building, so the package version
inside the image matches its tag.

## License

MIT, see [LICENSE](LICENSE).
