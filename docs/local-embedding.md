# Local embedding (embedding profile)

Run the embedding model in a container: the user installs nothing on the host, and the corpus never
leaves the machine.

```bash
make start
```

Once it is up, the provider endpoint available inside the stack is `https://embedding/v1`, the model
name is `BAAI/bge-m3`, and the dimension is 1024. Use that base_url directly when creating a
provider config.

The first start downloads about 4.8 GB of model weights, which **can take more than half an hour**.
The weights live in the `embedding_model_cache` volume, so later starts only need a few minutes to
load.

## 1. Why TEI and not Ollama

The obvious approach is to put Ollama in the container, but measurement rules it out. Same batch of
64 real chunks (747 characters on average), batch=16:

| Deployment | Throughput | Estimate for the full corpus (about 88,000 chunks) |
| --- | --- | --- |
| Host Ollama (Apple Silicon, Metal) | 13.4 chunks/s | about 1.8 hours |
| Container TEI (CPU, arm64) | 1.2 chunks/s | about 20 hours |
| Container Ollama (CPU) | 0.07 chunks/s | about 14 days |

That container-Ollama number was re-measured on its own, with TEI stopped to rule out memory
contention, and it was still 0.07. Docker on macOS cannot reach Metal, and Ollama's pure-CPU path is
far too slow for bge-m3, which is XLM-R large scale.

> This number was re-measured. The first suspicion was self-throttling from `--max-batch-tokens`
> being pinned at 2048, so it was raised to 4096 and measured again with 600-codepoint samples, the
> new default chunk size: batch=4 gave 1.19 and batch=8 gave 1.24 chunks/s — **essentially
> unchanged**. So the container really is CPU-bound rather than held back by the batch limit.
> Raising it to 4096 is still right, but the reason is avoiding 429, not throughput.

TEI has one more reason in its favor for the future: it has a `/rerank` endpoint and Ollama does not
(measured on 0.13.5, both `/api/rerank` and `/v1/rerank` return 404). Reranking reuses the same
deployment instead of introducing a third dependency.

## 2. Switching engines does not require reindexing

For the same text, the cosine between vectors from host Ollama and container TEI running the
identically named `bge-m3`:

| Text | Cosine |
| --- | --- |
| English query | 1.0000 |
| Chinese query | 1.0000 |
| 600-character body chunk | 0.9992 |

The difference is at the level of numerical noise, so vectors produced by the two engines can be
mixed within one generation. What that means in practice:

**The container is the answer; host Ollama is only an optional accelerator.** The full initial
ingest is a one-off unattended background job: submit the documents and come back the next day,
without watching it. For everyday queries and incremental ingestion the container is entirely
sufficient.

The genuinely useful corollary: if you **happen to already have** a host Ollama, you can use it for
the one-off initial ingest and go back to the container afterwards — **the same index generation
works for both, with no rebuild**. Conversely, installing Ollama just for the initial ingest is
unnecessary.

> This conclusion covers only bge-m3 on these two engines. It must be re-verified when changing
> model or precision (a different quantization level, for example); do not assume mixing is safe.

## 3. Linux / GPU deployment

The gap in the table above is specific to macOS, where the container cannot reach Metal. On the real
deployment target, a Linux server, "host Ollama with a GPU" is not an option in the first place and
containerization is required anyway; with an NVIDIA card, switching to TEI's GPU image makes the
penalty disappear.

`make start` picks the image tag from `uname -m` (arm64 → `cpu-arm64-latest`, otherwise
→ `cpu-latest`). Set `RAG_EMBEDDING_IMAGE_TAG` to override it, for a GPU image for example.

## 4. Three traps already stepped in

**`--max-batch-tokens` causes trouble at both ends, and fails differently at each.** Left
unrestricted, TEI warms up against the model's 8192-token window and is OOM-killed on a 16 GB
machine (exit 137); set too low, it rejects normal batches with 429.

Measured boundary on a 16 GB Colima VM: 4096 starts cleanly and is ready in about 50 seconds;
**8192 does not OOM cleanly but drags the whole Docker VM into unresponsiveness**, recoverable only
with `colima restart`. Confirm host memory before raising it, and do not experiment on a machine
that is doing work.

Correspondingly, `--max-client-batch-size` has to be tuned together with `--max-batch-tokens` rather
than set independently: 600-codepoint English technical text runs past 256 tokens per item, so
batch=16 overruns 4096 and comes back 429, while batch=8 is measured to work. **Do not set a model
profile `batch_size` above that value.**

**`BAAI/bge-m3` has no `model.safetensors`.** TEI does not find one, falls back to
`pytorch_model.bin`, and warns that loading will be slower. That is why the total download is about
4.8 GB rather than 2.2 GB. This is the state of the official repository; do not point at an
unofficial converted mirror to save download size.

**The CA used to be re-signed on every `up`, and that trap is worth remembering.**
`embedding-tls-init` now takes `--reuse-existing`: a certificate already on disk is reused as-is as
long as it has not expired, its SAN covers the target hostname, and **verification** confirms it was
actually issued by this CA.

It was worth fixing specifically because of how misleadingly it failed: api and worker load the CA
once at process start, so after a re-sign they keep trusting a CA that has been superseded.
Ingestion then reports `PROVIDER_UNAVAILABLE` — flagged `retryable: true`, no less — **with nothing
logged on either the proxy or the model side**, because the client rejects the certificate before
sending a request. It looks like the model is down; the trust chain is what is broken.

Note that the verification step cannot be reduced to comparing issuer names: the CA's subject name
is a fixed string, so a new CA with a different key still compares equal by name while no longer
being able to vouch for the old certificate.

## 5. Why there is an nginx layer in the middle

`providers/network_policy.py:344` requires provider endpoints to be https, and every local inference
service speaks only http. Terminating TLS in a layer in front means the local path takes exactly the
same code path as a hosted provider, without carving a hole in the security policy for the local
case. The certificate is generated by `velox-provider-stub-tls`, which ships in the repository, and
the CA goes through the existing `RAG_PROVIDER_CA_BUNDLE` mechanism.

The `embedding` and `provider-stub` profiles **cannot be used at the same time**: each TLS init
generates its own CA, and sharing one volume means overwriting each other. That is why embedding
uses a separate `embedding_ca` volume, and why `RAG_PROVIDER_CA_BUNDLE` can only point at one of
them.

## 6. The private-target switch

The `embedding` service resolves to a private address on the Docker network, so it needs
`RAG_PROVIDER_ALLOW_PRIVATE_TARGETS=true`, which `make start` already passes. The switch opens
loopback and private ranges only; cloud metadata addresses, link-local, multicast, and reserved
ranges stay hard-refused (`network_policy.py:500-520`). The existing `make up-provider` sets the
same switch for the same reason.
