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

One command:

```bash
curl -fsSL https://raw.githubusercontent.com/ilikebug/veloxrag/main/install.sh | bash
```

It installs Ollama if missing, pulls `bge-m3` (about 1.2 GB, once), writes `compose.yaml` into
`~/.veloxrag`, starts the stack, and prints the MCP command.

The embedding model runs on the **host**, not in a container, because that is the only place it
reaches the GPU: Docker on macOS is a Linux VM with no Metal passthrough, measured at 3.1
chunks/s against 14.2 on the host. Everything else stays containerized.

With Ollama already running, `compose.yaml` alone is still the whole stack (Docker Compose 2.23.1
or newer — the file carries the embedding proxy's nginx configuration inline as a compose config,
and inline `content` only exists from that version on):

```bash
docker compose up -d
```

It completes initialization on its own: provider configuration, dimension probe, model profile,
knowledge base, filter schema, initial generation.

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
api image from the working tree, and refuse to start when Ollama is not answering.)

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

**`RAG_PROVIDER_ALLOW_PRIVATE_TARGETS`** is needed because the embedding endpoint is
`https://embedding` on the Docker network — an nginx that terminates TLS in front of the host's
Ollama — and that is a private address. There is an easy mistake to
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
| `search_memory` | Retrieval, with a `source_type` filter. `rerank` needs a rerank profile, which the default setup has none of |
| `read_document` | Read a document's text around a character range, to see what a result was cut off from |
| `list_documents` | See what is in the index, to narrow a search or recognize a gap in the memory |
| `memory_status` | Which knowledge base is bound, and whether the service can retrieve right now |

Relevance judgement is left to the agent rather than done by a reranker. An agent already reads
the passages and reasons about them, which is what a cross-encoder does with a far smaller model,
so the useful thing is not another ranking pass but giving that judgement something to work with:
retrieve more passages than you need, then widen the promising ones with `read_document` before
deciding. Measured over 18 queries through these tools, that took @5 from 0.89 to 1.00 and MRR
from 0.645 to 0.724 — the answer frequently sits just outside the matched passage, and a passage
that looks truncated is worth widening rather than discarding.

What this cannot do is rescue a passage retrieval never returned. @10 was 0.94, so about one
query in sixteen has no candidate to promote, and closing that needs better retrieval rather than
better judgement.

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
  `end_offset`, and `GET /v1/documents/{id}/content?start=&end=` reads that range back, so a
  truncated hit can be widened. Measured over transcript queries, widening by 300 characters
  moved answer-level MRR from 0.675 to 0.756 and took @5 and @10 to 1.00. The MCP tools do not
  expose it yet — a consumer that wants it calls the HTTP endpoint directly.
