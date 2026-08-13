# Agent memory over MCP

What this layer is for: **run a RAG service locally and give it to a local agent such as Claude
Code as its underlying memory**. The agent receives tools over MCP instead of assembling HTTP
requests itself.

## Why it exists

The HTTP API is complete, but an agent cannot use it: you first have to mint an admin token from
the container CLI, sign an agent key, remember the knowledge base id, and hand-write requests.
That ceremony is designed for a multi-tenant deployment; for a single person running this locally
it is pure overhead, and an agent cannot perform it at all.

The MCP server holds the credential and the knowledge base itself, and exposes outward only what
an agent actually needs.

## Using it

`compose.yaml` is all you need (Docker Compose 2.23.1 or newer — the file carries the embedding
proxy's nginx configuration inline as a compose config, and inline `content` only exists from that
version on):

```bash
docker compose up -d
```

That brings up the databases, the object store, and the **containerized embedding model** in one
step, and completes initialization on its own: provider configuration, dimension probe, model
profile, knowledge base, filter schema, initial generation. The first start downloads about
4.8 GB of weights; later starts take a few minutes.

x86_64 machines need one extra variable, because text-embeddings-inference publishes no
multi-architecture tag:

```bash
RAG_EMBEDDING_IMAGE_TAG=cpu-latest docker compose up -d
```

The MCP client then needs one command and no environment variables. **No checkout required** —
`uvx` runs the server straight from the public repository, which is the MCP-side counterpart of
"downloading one compose.yaml is enough":

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

With a checkout already present, run it from the working tree so code changes take effect
immediately:

```json
{
  "mcpServers": {
    "rag-memory": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/VeloxRAG", "velox-mcp"]
    }
  }
}
```

(When developing inside the repository use `make start`, which does two extra things: rebuild the
api image from the working tree, and pick the embedding image tag from `uname -m`.)

All three variables are optional:

| Variable | When you need it |
| --- | --- |
| `RAG_MCP_BASE_URL` | The service is not at `http://127.0.0.1:8000` — including when you have overridden `RAG_API_HOST_PORT` |
| `RAG_MCP_TOKEN` | The service has local trusted auth switched off; supply an agent key carrying `retrieve` |
| `RAG_MCP_KNOWLEDGE_BASE` | The service holds **more than one** knowledge base. With exactly one it resolves automatically; with several it refuses to guess, because guessing wrong means searching the wrong memory |

## The two defaults in compose.yaml that hold only locally

That file *is* the local single-user stack, so two switches default to on. Both make the service
**refuse to start** under `RAG_ENVIRONMENT=production`, so they cannot follow a copied file into a
deployment.

**`RAG_LOCAL_TRUSTED_AUTH`** admits requests that carry no credential and attributes them to an
automatically provisioned local actor. This is not the same as removing the actor: `jobs`,
`audit_events`, and `idempotency_records` all hold non-null foreign keys into `api_keys`, so a
request still needs an attributable row — it just no longer trades a token for one. Scope,
auditing, and rate limiting are unchanged. An **invalid** token is still rejected with 401: local
mode covers only the "no credential at all" case. The two provisioned key digests are random, so
no token can compute them.

**`RAG_PROVIDER_ALLOW_PRIVATE_TARGETS`** is needed because the containerized embedding service is
`https://embedding` on the Docker network, which is a private address. There is an easy mistake to
make here: the provider policy performs **two independent checks** — it must be https, and it must
not be a private IP. The nginx layer in the middle only solves the first. The second is based on
the **resolved IP**, so renaming the network or gathering the components onto one internal network
does not avoid it; "being on an internal network" is precisely the property that gets refused. The
switch opens loopback and private ranges only; cloud metadata addresses, link-local, multicast,
and reserved ranges stay hard-refused.

Any local embedding setup needs it (a host Ollama reached through `host.docker.internal` is
equally a private address). The only way to avoid switching it on is to use a hosted embedding
provider, at the cost of sending your corpus to a third party.

## The tools it exposes

| Tool | Purpose |
| --- | --- |
| `search_memory` | Retrieval. Optional `rerank` (one more provider round trip, in exchange for a better first hit) and a `source_type` filter |
| `list_documents` | See what is in the index, to narrow a search or recognize a gap in the memory |
| `memory_status` | Which knowledge base is bound, and whether the service can retrieve right now |

Ingestion, knowledge base creation, and key minting are **deliberately not offered**. An agent
that can provision its own storage can also destroy it, and deletion in this service
[is a real delete](deployment.md). Ingestion goes through the CLI and the HTTP API, performed by a
person.

## A typical setup

First load the corpus, once, by hand:

```bash
uv run velox-chat-transcripts --source claude-code --root ~/.claude/projects \
  --output-directory ./memory --project=<project directory name>
```

`make start` has already created the knowledge base and the generation, so all that remains is the
upload (`POST /v1/knowledge-bases/{kb}/documents`, no token). After that an agent can simply ask
"why did we decide X back then".

## Known boundaries

- **Read-only.** Updating the memory means ingesting again.
- **One server binds one knowledge base.** Searching across several means running several MCP
  servers; the service itself does not support cross-KB search either.
- **A hit is a fragment, not a whole document.** Results carry `source.start_offset` /
  `end_offset`, but there is currently **no endpoint that reads the original text back by
  range**, so when a fragment is cut off the only remedies are a larger `top_k` or a different
  phrasing.
