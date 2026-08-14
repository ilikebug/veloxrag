.PHONY: build push sync fmt check test test-integration test-integration-core test-acceptance acceptance-ingestion-retrieval coverage verify compose-config build-arm64 up up-provider start stop restart down logs migrate smoke smoke-auth

# Local stack knobs used by start/restart.
# 6379 is frequently occupied by a local or tunneled redis; "auto" picks 6380
# when that happens. Pin an explicit port to opt out of the probe.
RAG_REDIS_HOST_PORT ?= auto

# Only needed so that start prints the URL it actually published. The other host
# ports are overridable too and reach compose straight from the environment; see
# the port table in README.
RAG_API_HOST_PORT ?= 8000

# Image coordinates for build/push. The bare default is fine for a local build
# but cannot be pushed, so push rejects it.
VELOX_IMAGE ?= veloxrag

UNIT_COVERAGE_FILE := $(CURDIR)/.coverage.unit
INTEGRATION_COVERAGE_FILE := $(CURDIR)/.coverage.integration
ACCEPTANCE_COVERAGE_FILE := $(CURDIR)/.coverage.acceptance
COMBINED_COVERAGE_FILE := $(CURDIR)/.coverage.combined

sync:
	uv sync --frozen

fmt:
	uv run ruff format .
	uv run ruff check --fix .

check:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src tests migrations

test:
	COVERAGE_FILE=$(UNIT_COVERAGE_FILE) uv run coverage erase
	COVERAGE_FILE=$(UNIT_COVERAGE_FILE) uv run pytest tests/unit --cov=rag_service --cov-branch --cov-report=term-missing --cov-fail-under=0

test-integration:
	COVERAGE_FILE=$(INTEGRATION_COVERAGE_FILE) uv run coverage erase
	COVERAGE_FILE=$(INTEGRATION_COVERAGE_FILE) uv run pytest tests/integration -m integration -q --cov=rag_service --cov-branch --cov-report= --cov-fail-under=0

test-integration-core:
	COVERAGE_FILE=$(INTEGRATION_COVERAGE_FILE) uv run coverage erase
	COVERAGE_FILE=$(INTEGRATION_COVERAGE_FILE) uv run pytest tests/integration -m "integration and not acceptance" -q --cov=rag_service --cov-branch --cov-report= --cov-fail-under=0

test-acceptance:
	uv run pytest -q tests/unit/test_readiness.py::test_live_provider_reports_keyring_format_failure_without_provider_network_probe
	COVERAGE_FILE=$(ACCEPTANCE_COVERAGE_FILE) uv run coverage erase
	COVERAGE_FILE=$(ACCEPTANCE_COVERAGE_FILE) uv run pytest -q -m acceptance tests/integration/test_ingestion_retrieval_e2e.py tests/integration/test_job_recovery.py tests/integration/test_generation_api.py tests/integration/test_generation_repair.py --cov=rag_service --cov-branch --cov-report= --cov-fail-under=0

# Full acceptance gate: focused acceptance tests, live provider stack, smoke, cleanup.
acceptance-ingestion-retrieval:
	$(MAKE) test-acceptance
	COMPOSE_DISABLE_ENV_FILE=1 bash scripts/acceptance_ingestion_retrieval.sh

coverage:
	COVERAGE_FILE=$(COMBINED_COVERAGE_FILE) uv run coverage erase
	COVERAGE_FILE=$(COMBINED_COVERAGE_FILE) uv run coverage combine --keep .coverage.unit .coverage.integration .coverage.acceptance
	COVERAGE_FILE=$(COMBINED_COVERAGE_FILE) uv run coverage report --fail-under=80

compose-config:
	COMPOSE_DISABLE_ENV_FILE=1 docker compose config --quiet
	COMPOSE_DISABLE_ENV_FILE=1 docker compose --profile provider-stub config --quiet
	COMPOSE_DISABLE_ENV_FILE=1 docker compose -f compose.yaml -f compose.build.yaml config --quiet
	COMPOSE_DISABLE_ENV_FILE=1 VELOX_IMAGE_TAG=rag-acceptance-0000000000000000 docker compose --project-name rag-acceptance-0000000000000000 -f compose.yaml -f compose.build.yaml -f compose.acceptance.yaml --profile provider-stub config --quiet

build-arm64:
	docker build --platform linux/arm64 -t rag-service:ingestion-retrieval .

verify:
	$(MAKE) check
	$(MAKE) test
	$(MAKE) test-integration-core
	$(MAKE) acceptance-ingestion-retrieval
	$(MAKE) coverage
	$(MAKE) compose-config

up:
	COMPOSE_DISABLE_ENV_FILE=1 docker compose -f compose.yaml -f compose.build.yaml build api
	COMPOSE_DISABLE_ENV_FILE=1 docker compose -f compose.yaml -f compose.build.yaml up -d --no-build

# The one command for a working local stack: datastores, object store, and a
# bootstrap that creates the provider config, model profile, knowledge base,
# filter schema and initial generation. Embeddings come from the host's Ollama,
# which is the only place the model reaches the GPU.
# After it finishes, point an MCP client at `velox-mcp` and it works.
#
# Only convenience over `docker compose up`: it builds the api image from the
# working tree and checks Ollama first, because a stack that starts without it
# fails every embedding call rather than failing to start.
start:
	@redis_port='$(RAG_REDIS_HOST_PORT)'; \
	if [ "$$redis_port" = "auto" ]; then \
		redis_port=$$(python3 -c 'import socket; probe = socket.socket(); print(6380 if probe.connect_ex(("127.0.0.1", 6379)) == 0 else 6379); probe.close()'); \
	fi; \
	curl -fsS --max-time 3 http://127.0.0.1:11434/api/version >/dev/null 2>&1 \
		|| { echo "Ollama is not answering on 127.0.0.1:11434; start it first (brew services start ollama)"; exit 1; }; \
	echo "starting stack: redis host port $$redis_port, embeddings from the host Ollama"; \
	COMPOSE_DISABLE_ENV_FILE=1 docker compose -f compose.yaml -f compose.build.yaml build api && \
	COMPOSE_DISABLE_ENV_FILE=1 \
		RAG_REDIS_HOST_PORT="$$redis_port" \
		docker compose -f compose.yaml -f compose.build.yaml up -d --no-build && \
	COMPOSE_DISABLE_ENV_FILE=1 RAG_REDIS_HOST_PORT="$$redis_port" \
		docker compose -f compose.yaml -f compose.build.yaml wait bootstrap >/dev/null && \
	echo "api: http://127.0.0.1:$(RAG_API_HOST_PORT)/health" && \
	echo "mcp: uv run velox-mcp   (needs no token and no knowledge base id)"

stop: down

restart:
	$(MAKE) stop
	$(MAKE) start

up-provider:
	COMPOSE_DISABLE_ENV_FILE=1 docker compose -f compose.yaml -f compose.build.yaml build api
	COMPOSE_DISABLE_ENV_FILE=1 RAG_PROVIDER_ALLOW_PRIVATE_TARGETS=true RAG_PROVIDER_CA_BUNDLE=/run/rag/provider-ca/ca.pem docker compose -f compose.yaml -f compose.build.yaml --profile provider-stub up -d --no-build --wait --wait-timeout 120

down:
	COMPOSE_DISABLE_ENV_FILE=1 docker compose -f compose.yaml -f compose.build.yaml --profile provider-stub down

logs:
	COMPOSE_DISABLE_ENV_FILE=1 docker compose -f compose.yaml -f compose.build.yaml logs -f api worker migrate provider-stub

migrate:
	COMPOSE_DISABLE_ENV_FILE=1 docker compose -f compose.yaml -f compose.build.yaml run --rm migrate

# RAG_BASE_URL is derived from the published port so that overriding
# RAG_API_HOST_PORT does not leave these probing 8000 and reporting a dead stack.
# An explicit RAG_BASE_URL still wins.
smoke:
	COMPOSE_DISABLE_ENV_FILE=1 RAG_BASE_URL="$${RAG_BASE_URL:-http://127.0.0.1:$(RAG_API_HOST_PORT)}" bash scripts/smoke_stack.sh

smoke-auth:
	COMPOSE_DISABLE_ENV_FILE=1 RAG_BASE_URL="$${RAG_BASE_URL:-http://127.0.0.1:$(RAG_API_HOST_PORT)}" bash scripts/smoke_auth_metadata.sh

# Build the image into the local Docker image store for this machine's own
# architecture. TAG is required so that every build stays identifiable.
#
#   make build TAG=0.1.0
build:
	@test -n "$(TAG)" || { echo "set TAG, e.g. make build TAG=0.1.0"; exit 1; }
	docker build -t $(VELOX_IMAGE):$(TAG) -t $(VELOX_IMAGE):latest .

# Publish the image so that `docker compose up` works from compose.yaml alone.
# Needs a namespace you own and a prior `docker login`; VELOX_IMAGE must match
# what compose.yaml resolves to for consumers.
#
# This rebuilds rather than pushing what `make build` produced: consumers run
# both architectures, and buildx cannot hold a multi-architecture manifest in
# the local image store, so the two platforms have to go straight to the
# registry from one invocation.
#
#   make push VELOX_IMAGE=docker.io/<you>/veloxrag TAG=0.1.0
push:
	@test -n "$(TAG)" || { echo "set TAG, e.g. make push TAG=0.1.0"; exit 1; }
	@case "$(VELOX_IMAGE)" in */*) ;; *) echo "set VELOX_IMAGE to a namespace you own, e.g. VELOX_IMAGE=docker.io/<you>/veloxrag"; exit 1;; esac
	@docker buildx version >/dev/null 2>&1 || { \
		echo "docker buildx is missing: this target needs it for the two-platform build."; \
		echo "docker reports the unknown flag --platform rather than a missing plugin, so check this first."; \
		echo "  macOS:  brew install docker-buildx && ln -sf /opt/homebrew/lib/docker/cli-plugins/docker-buildx ~/.docker/cli-plugins/docker-buildx"; \
		echo "  other:  https://github.com/docker/buildx#installing"; \
		exit 1; }
	docker buildx build --platform linux/amd64,linux/arm64 \
		-t $(VELOX_IMAGE):$(TAG) -t $(VELOX_IMAGE):latest --push .
