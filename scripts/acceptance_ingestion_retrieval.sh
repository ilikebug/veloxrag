#!/usr/bin/env bash
set -euo pipefail
umask 077

export COMPOSE_DISABLE_ENV_FILE=1

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
acceptance_compose_override="${repo_root}/compose.acceptance.yaml"
cleanup_required=0
compose_project_ready=0
image_cleanup_required=0

fail() {
  printf '%s\n' "RAG ingestion/retrieval acceptance gate failed: $1" >&2
  return 1
}

compose_silent() {
  COMPOSE_DISABLE_ENV_FILE=1 VELOX_IMAGE_TAG="${VELOX_IMAGE_TAG}" VELOX_IMAGE="${VELOX_IMAGE:-veloxrag}" \
    docker compose \
    --project-name "${COMPOSE_PROJECT_NAME}" \
    -f "${repo_root}/compose.yaml" \
    -f "${repo_root}/compose.build.yaml" \
    -f "${acceptance_compose_override}" \
    --profile provider-stub "$@" >/dev/null 2>&1
}

compose_capture() {
  COMPOSE_DISABLE_ENV_FILE=1 VELOX_IMAGE_TAG="${VELOX_IMAGE_TAG}" VELOX_IMAGE="${VELOX_IMAGE:-veloxrag}" \
    docker compose \
    --project-name "${COMPOSE_PROJECT_NAME}" \
    -f "${repo_root}/compose.yaml" \
    -f "${repo_root}/compose.build.yaml" \
    -f "${acceptance_compose_override}" \
    --profile provider-stub "$@" 2>/dev/null
}

initialize_compose_project() {
  local project_suffix

  project_suffix=$(openssl rand -hex 8) || return 1
  if [[ ! "${project_suffix}" =~ ^[0-9a-f]{16}$ ]]; then
    return 1
  fi
  COMPOSE_PROJECT_NAME="rag-acceptance-${project_suffix}"
  VELOX_IMAGE_TAG="${COMPOSE_PROJECT_NAME}"
  VELOX_IMAGE="veloxrag"
  if [[ ! "${COMPOSE_PROJECT_NAME}" =~ ^rag-acceptance-[0-9a-f]{16}$ ]] ||
    [[ ! "${VELOX_IMAGE_TAG}" =~ ^rag-acceptance-[0-9a-f]{16}$ ]]; then
    return 1
  fi
  export COMPOSE_PROJECT_NAME VELOX_IMAGE_TAG VELOX_IMAGE
  compose_project_ready=1
  project_suffix=""
}

verify_expected_one_shot_containers() {
  local exited_services
  local service

  exited_services=$(compose_capture ps -a --status exited --services) || return 1
  while IFS= read -r service; do
    [ -z "${service}" ] && continue
    case "${service}" in
      minio-init | migrate | provider-tls-init) ;;
      *) return 1 ;;
    esac
  done <<<"${exited_services}"
}

finish() {
  local exit_code="$?"
  local cleanup_failed=0
  local remaining=""

  set +e
  trap - EXIT HUP INT TERM
  if [ "${compose_project_ready}" -ne 1 ]; then
    exit "${exit_code}"
  fi
  if [ "${cleanup_required}" -eq 1 ]; then
    compose_silent down --remove-orphans --volumes || cleanup_failed=1
    cleanup_required=0
  fi
  remaining=$(compose_capture ps -a --format json) || cleanup_failed=1
  if [ -n "${remaining}" ]; then
    cleanup_failed=1
  fi
  if [ "${image_cleanup_required}" -eq 1 ]; then
    if docker image inspect "rag-service:${VELOX_IMAGE_TAG}" >/dev/null 2>&1; then
      docker image rm "rag-service:${VELOX_IMAGE_TAG}" >/dev/null 2>&1 || cleanup_failed=1
    fi
    image_cleanup_required=0
  fi
  if [ "${cleanup_failed}" -ne 0 ]; then
    printf '%s\n' \
      "RAG ingestion/retrieval acceptance gate cleanup was incomplete" >&2
    exit 1
  fi
  exit "${exit_code}"
}

on_signal() {
  local exit_code="$1"

  trap - HUP INT TERM
  exit "${exit_code}"
}

main() {
  local api_endpoint
  local api_port
  local provider_secret
  local provider_hash

  trap finish EXIT
  trap 'on_signal 129' HUP
  trap 'on_signal 130' INT
  trap 'on_signal 143' TERM
  cd -- "${repo_root}"

  initialize_compose_project || fail "isolated Compose project generation failed"
  cleanup_required=1
  image_cleanup_required=1
  compose_silent build api ||
    fail "Compose application image build failed"
  provider_secret=$(openssl rand -hex 32) || fail "ephemeral provider material generation failed"
  if [[ ! "${provider_secret}" =~ ^[0-9a-f]{64}$ ]]; then
    fail "ephemeral provider material format was invalid"
  fi
  provider_hash=$(printf 'Bearer %s' "${provider_secret}" | openssl dgst -sha256 -r) ||
    fail "provider authorization digest generation failed"
  provider_hash="${provider_hash%% *}"
  if [[ ! "${provider_hash}" =~ ^[0-9a-f]{64}$ ]]; then
    fail "provider authorization digest format was invalid"
  fi

  COMPOSE_DISABLE_ENV_FILE=1 \
    RAG_PROVIDER_STUB_AUTHORIZATION_SHA256="${provider_hash}" \
    RAG_PROVIDER_ALLOW_PRIVATE_TARGETS=true \
    RAG_PROVIDER_CA_BUNDLE=/run/rag/provider-ca/ca.pem \
    compose_silent up -d --no-build --wait --wait-timeout 120 ||
    fail "provider-stub Compose stack did not become ready"
  provider_hash=""
  verify_expected_one_shot_containers ||
    fail "Compose reported an unexpected one-shot container"

  api_endpoint=$(compose_capture port api 8000) ||
    fail "dynamic API port discovery failed"
  if [[ ! "${api_endpoint}" =~ ^127\.0\.0\.1:[1-9][0-9]{0,4}$ ]]; then
    fail "dynamic API port discovery failed"
  fi
  api_port="${api_endpoint##*:}"
  if [ "$((10#${api_port}))" -gt 65535 ]; then
    fail "dynamic API port discovery failed"
  fi

  RAG_BASE_URL="http://${api_endpoint}" \
    RAG_ACCEPTANCE_EPHEMERAL=1 \
    RAG_ACCEPTANCE_OWNED_PROJECT="${COMPOSE_PROJECT_NAME}" \
    RAG_ACCEPTANCE_OWNED_IMAGE_TAG="${VELOX_IMAGE_TAG}" \
    RAG_ACCEPTANCE_COMPOSE_OVERRIDE="${acceptance_compose_override}" \
    RAG_SMOKE_PROVIDER_SECRET="${provider_secret}" \
    bash scripts/smoke_ingestion_retrieval.sh
  provider_secret=""

  printf '%s\n' "RAG ingestion/retrieval unified acceptance gate passed"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
