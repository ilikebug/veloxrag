# Local embedding

The embedding model runs on the **host** through Ollama, not in a container. Everything else in
the stack stays containerized.

```bash
brew install ollama && brew services start ollama   # macOS; Linux: curl -fsSL https://ollama.com/install.sh | sh
ollama pull bge-m3
```

`install.sh` does both. Inside the stack the provider endpoint is `https://embedding/v1`, the
model is `bge-m3`, and the dimension is 1024.

## 1. Why the host and not a container

Docker on macOS is a Linux VM with no Metal passthrough, so a containerized model runs on CPU no
matter what the host has. Measured on the same input, batch of 8:

| Deployment | Throughput | Scales with batch |
| --- | --- | --- |
| Container text-embeddings-inference (CPU) | 3.10 chunks/s | no — 1→8 gained 8% |
| Host Ollama (Metal) | 14.20 chunks/s | yes — 1→8 gained 2.6× |

The flat batch curve is the diagnostic: the container is compute-bound, already saturating the
CPU on a single request, so there is no idle parallelism for a batch to exploit. Raising
`--max-batch-tokens` or the client batch size does nothing, which an earlier round of tuning
confirmed independently.

The host is latency-bound instead, which is why batching pays there.

Two consequences worth stating plainly:

- **The 4.6× only matters for bulk ingest.** A query embeds one string, and 0.07 s against 0.3 s
  is imperceptible. On a corpus of 13,600 chunks it is the difference between 16 minutes and 3
  hours, once.
- **Reranking has no host equivalent.** Ollama exposes no rerank endpoint — `/api/rerank` and
  `/v1/rerank` both return 404 as of 0.13.5 — so reranking would need a container back, which is
  the one thing this arrangement gives up.

## 2. Why bge-m3, and what quantization costs

Ollama serves bge-m3 at F16 in 1.2 GB; the container served fp32 in 4.8 GB. Both produce 1024
dimensions, so an index built against one is readable by the other. Measured cosine between the
two engines on identical text: 1.0000 for an English query, 1.0000 for a Chinese query, and
0.9992 for a 600-character body chunk — noise, in other words, and the reason switching engines
does not require reindexing.

> This covers bge-m3 on these two engines only. Changing model or precision has to be verified
> again; do not assume vectors stay interchangeable.

## 3. Why there is still an nginx in front

`providers/network_policy.py:344` requires provider endpoints to be https, and Ollama speaks only
http. Terminating TLS in front means the local path takes the same code path as a hosted
provider, without carving a hole in the policy. The certificate comes from
`velox-provider-stub-tls`, which ships in the repository, and the CA goes through the existing
`RAG_PROVIDER_CA_BUNDLE` mechanism.

The upstream is referenced through a variable with `resolver` set, so nginx resolves it per
request. A literal upstream is resolved at startup and nginx refuses to boot when the name is
missing, which would turn a stopped Ollama into a stack that will not start at all rather than
one that fails embedding calls.

## 4. The private-target switch

`host.docker.internal` is a private address, so the stack needs
`RAG_PROVIDER_ALLOW_PRIVATE_TARGETS=true`. The switch opens loopback and private ranges only;
cloud metadata addresses, link-local, multicast, and reserved ranges stay hard-refused
(`network_policy.py:500-520`). It makes the service refuse to start under
`RAG_ENVIRONMENT=production`, which is what keeps it from following a copied file into a
deployment.

## 5. After a reboot

The containers carry `restart: unless-stopped` and come back on their own, but only once the
Docker daemon is up — so Docker Desktop or Colima has to start at login. Ollama is a host
process and needs its own arrangement: `brew services start ollama` on macOS, or the systemd
unit the official installer enables on Linux.

If retrieval fails after a reboot, Ollama is the first thing to check:

```bash
curl http://127.0.0.1:11434/api/version
```

A stack that is up while Ollama is down fails every embedding call with `PROVIDER_UNAVAILABLE`,
and neither the proxy nor the model logs anything, because there is nothing listening to log it.
