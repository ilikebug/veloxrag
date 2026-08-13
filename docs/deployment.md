# Production deployment

This document covers only **what running in production requires**. For the day-to-day call flow
see [api-operations.md](api-operations.md); for local embedding see
[local-embedding.md](local-embedding.md).

## 1. What `RAG_ENVIRONMENT=production` enforces

The service validates its configuration at startup and refuses to start when it is not satisfied
(from `config.py:236` on). That is deliberate: nobody notices a production instance running on
development default keys until something goes wrong.

What has to be replaced:

| Setting | Requirement | Consequence if unmet |
| --- | --- | --- |
| `RAG_ADMIN_KEY_HMAC_SECRET` | ≥32 bytes, and **not equal to** `RAG_AGENT_KEY_HMAC_SECRET` | startup fails |
| `RAG_AGENT_KEY_HMAC_SECRET` | same | startup fails |
| `RAG_PROVIDER_CREDENTIAL_KEYRING` | JSON `{version: base64 key}`; a key must not be a single repeated byte | startup fails |
| `RAG_PROVIDER_CREDENTIAL_ACTIVE_KEY_VERSION` | must be a version present in the keyring | startup fails |
| `RAG_DATABASE_URL` / `RAG_MINIO_ACCESS_KEY` / `RAG_MINIO_SECRET_KEY` | must not carry a development marker such as `change-me` | startup fails |
| `RAG_PROVIDER_ALLOW_PRIVATE_TARGETS` | must be `false` | startup fails |
| `RAG_LOCAL_TRUSTED_AUTH` | must be `false` | startup fails |

The last row deserves its own note: with that switch open, a provider endpoint may point at an
internal address, which is an SSRF hole. **It is not allowed in production**, and that means a
**self-hosted embedding or rerank service cannot be used directly** — those live on a private
network. Production either uses a hosted provider that resolves publicly, or puts the self-hosted
service behind an address with public DNS and a real certificate.

The two HMAC secrets must differ because they sign Admin and Agent tokens respectively; were they
the same, an Agent token would validate as an Admin token.

Generate a key (do not copy the example values in this document):

```bash
python3 -c 'import base64,secrets; print(base64.b64encode(secrets.token_bytes(32)).decode())'
```

## 2. Provider endpoints must be HTTPS

`providers/network_policy.py:344` requires https, and additionally resolves the target address and
refuses cloud metadata addresses, link-local, multicast, and reserved ranges. This cannot be
bypassed in production, and should not be.

A self-hosted inference service only speaks http, so it needs TLS termination in front of it and
the service has to trust that certificate (`RAG_PROVIDER_CA_BUNDLE` points at the CA file). Note
that api and worker **load the CA once, at process start**: after rotating the certificate both
processes must be restarted, or they keep trusting the old CA. The failure presents as
`PROVIDER_UNAVAILABLE`, with nothing logged on either the proxy or the model side.

## 3. Components and persistence

| Component | Holds | What losing it means |
| --- | --- | --- |
| PostgreSQL | document visibility, jobs, generations, authorization — **the authoritative facts** | everything is lost and cannot be rebuilt from the other components |
| MinIO | original files, normalized text, chunk manifests | vectors can be rebuilt, the original content cannot |
| Qdrant | vectors and retrieval payloads | **rebuildable from MinIO + Postgres** (see same-generation repair in the README) |
| Redis | only wakes the worker | no impact; the worker falls back to polling |

So the backup priority is **Postgres > MinIO > Qdrant**. Qdrant is the only one that can be
rebuilt completely.

## 4. Readiness probes

| Endpoint | Meaning |
| --- | --- |
| `GET /health` | the process is alive |
| `GET /ready` | the core dependencies are ready |
| `GET /ready/ingest` | ingestion is possible |
| `GET /ready/retrieve` | retrieval is possible |
| `GET /ready/answer` | **always 503** |

`/ready/answer` returning 503 permanently is not a fault: answer generation is outside this
service's responsibility — this layer supplies data and the consuming agent composes the answer —
and `answer_configured` is
hard-coded to `False` at `infrastructure/probes.py:466`. **Do not wire it into a health check**, or
it will alert forever. Use `/ready` or `/ready/retrieve`.

## 5. Four points of no return to know before going live

1. **filter_schema is a one-way door.** It freezes permanently once the first generation is
   created, and the embedding model, dimension, and distance behave the same way. One field you
   failed to define means creating a new knowledge base and ingesting again — so define every field
   you might ever filter on, before creating that first generation. A field has four elements, and
   `operators` is capped at 4 and constrained by type: keyword and boolean allow only `eq` and `in`,
   while integer, float, and datetime additionally allow `gte` and `lte`. Use `datetime` rather than
   a string for timestamps, or range queries are impossible later.

   ```json
   {"fields": [
     {"name":"source_type",  "source_path":"source_type",  "type":"keyword",  "operators":["eq","in"]},
     {"name":"doc_type",     "source_path":"doc_type",     "type":"keyword",  "operators":["eq","in"]},
     {"name":"section",      "source_path":"section",      "type":"keyword",  "operators":["eq","in"]},
     {"name":"source_path",  "source_path":"source_path",  "type":"keyword",  "operators":["eq","in"]},
     {"name":"lang",         "source_path":"lang",         "type":"keyword",  "operators":["eq","in"]},
     {"name":"occurred_at",  "source_path":"occurred_at",  "type":"datetime", "operators":["eq","gte","lte"]}
   ]}
   ```

   The setup order follows from the door being one-way, and cannot be permuted:
   create knowledge base → `PUT /v1/knowledge-bases/{kb}/filter-schema` → create the initial index
   generation → ingest.
2. **There is no generation cutover.** Changing the embedding model, the chunk size, or adding
   sparse all require rebuilding the knowledge base; there is no in-place migration.
3. **A delete is a real delete, and is not recoverable.**
   `DELETE /v1/knowledge-bases/{id}` sets the state to `deleting` and enqueues a
   `purge_knowledge_base` job; the worker then removes that knowledge base's Qdrant collection,
   its MinIO objects, and every database row. Measured at about 8 seconds on a small knowledge
   base, with **no undo**. Uploads are refused while a delete is in progress, because upload
   requires the knowledge base to be `active`.
4. **Documents cannot be replaced.** A revision can only be added, and the old version stays in
   the index.

## 6. Recovering from a failed generation creation

Creating a generation calls the embedding provider synchronously, inside the request. A transient
provider outage (503 `PROVIDER_UNAVAILABLE`) leaves the generation in `building`, and `building`
blocks creating another — so the knowledge base has neither an active generation nor the ability to
create one. To recover:

- replay with the **original** `Idempotency-Key`, which resumes it; or
- if the key is lost, call
  `POST /v1/admin/knowledge-bases/{kb}/index-generations/{generation_id}/abandon`
  to mark it `failed`, then create it again.

Only `building` can be abandoned; `active` returns 409 `GENERATION_NOT_ABANDONABLE`, because
abandoning the active generation would make already-indexed documents unsearchable.
Non-retryable failures such as 422 `PROVIDER_MODEL_NOT_FOUND` clean up after themselves and do not
get stuck.

## 7. Capacity reference

The numbers below are measured on Apple Silicon and are order-of-magnitude guidance only:

- chunking defaults to 600 codepoints with 100 overlap, which is roughly one chunk per 500 bytes
  of English documentation. Those defaults are measured rather than guessed: on a 70-query
  evaluation over a real corpus, moving from 1200 to 600 lifted answer-hit MRR from 0.631 to 0.836,
  and it is the single largest quality gain available. A CJK corpus can go lower still, because a
  codepoint carries far more information in CJK than in English
- embedding throughput: about 13.4 chunks/second on a host Ollama with Metal, about 1.2
  chunks/second on container CPU
- reranking sends `min(max(top_k*4, 20), 200)` candidates to the reranker in one call, so the
  **reranker's batch limit must be ≥200 or reranking silently does nothing** — retrieval still
  returns the dense ordering, and the only trace is
  `retrieval.rerank.completed outcome=failed` in the log

The throughput numbers do not apply when production uses a GPU; re-measure on the actual hardware.
