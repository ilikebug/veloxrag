#!/usr/bin/env bash
set -euo pipefail

base_url="${RAG_BASE_URL:-http://localhost:8000}"
tmp_root="${TMPDIR:-/tmp}"
lock_dir="${RAG_SMOKE_LOCK_DIR:-/tmp/rag-service-smoke.lock}"
work_dir=""
answer_body=""
ingest_body=""
lock_owned=0
diagnostics_enabled=0
redis_restore_required=0
minio_restore_required=0

acquire_lock() {
  local lock_holder="unknown"

  if ! mkdir "${lock_dir}" 2>/dev/null; then
    if [ -r "${lock_dir}/pid" ]; then
      IFS= read -r lock_holder <"${lock_dir}/pid" || lock_holder="unknown"
    fi
    echo "Another RAG foundation smoke test is running (pid: ${lock_holder})" >&2
    return 1
  fi

  lock_owned=1
  if ! printf '%s\n' "$$" >"${lock_dir}/pid"; then
    echo "Unable to record the RAG foundation smoke test lock owner" >&2
    return 1
  fi
}

release_lock() {
  if [ "${lock_owned}" -eq 1 ]; then
    rm -f "${lock_dir}/pid" >/dev/null 2>&1 || true
    rmdir "${lock_dir}" >/dev/null 2>&1 || true
    lock_owned=0
  fi
}

cleanup_responses() {
  if [ -n "${work_dir}" ]; then
    rm -f "${answer_body}" "${ingest_body}" >/dev/null 2>&1 || true
    rmdir "${work_dir}" >/dev/null 2>&1 || true
    work_dir=""
  fi
}

restore_ingest_dependencies() {
  local restore_status=0

  if [ "${redis_restore_required}" -eq 1 ]; then
    if docker compose start redis >/dev/null 2>&1; then
      redis_restore_required=0
    else
      restore_status=1
    fi
  fi

  if [ "${minio_restore_required}" -eq 1 ]; then
    if docker compose start minio >/dev/null 2>&1; then
      minio_restore_required=0
    else
      restore_status=1
    fi
  fi

  return "${restore_status}"
}

finish() {
  local exit_code="$?"

  set +e
  trap - EXIT HUP INT TERM

  restore_ingest_dependencies
  if [ "${exit_code}" -ne 0 ] && [ "${diagnostics_enabled}" -eq 1 ]; then
    docker compose ps || true
    docker compose logs --tail 200 api migrate minio minio-init redis || true
    echo "RAG foundation smoke test failed" >&2
  fi

  cleanup_responses
  release_lock
  exit "${exit_code}"
}

on_signal() {
  local exit_code="$1"

  trap - HUP INT TERM
  exit "${exit_code}"
}

wait_for_success() {
  local path="$1"
  for _attempt in $(seq 1 90); do
    if curl --connect-timeout 2 --max-time 5 -fsS "${base_url}${path}" >/dev/null; then
      return 0
    fi
    sleep 2
  done
  return 1
}

wait_for_status() {
  local path="$1"
  local expected="$2"
  local output="$3"
  local status
  for _attempt in $(seq 1 45); do
    status=$(
      curl --connect-timeout 2 --max-time 5 -sS \
        -o "${output}" -w "%{http_code}" "${base_url}${path}" || true
    )
    if [ "${status}" = "${expected}" ]; then
      return 0
    fi
    sleep 2
  done
  return 1
}

trap finish EXIT
trap 'on_signal 129' HUP
trap 'on_signal 130' INT
trap 'on_signal 143' TERM

acquire_lock
work_dir=$(mktemp -d "${tmp_root%/}/rag-service-smoke.XXXXXX")
answer_body="${work_dir}/ready-answer.json"
ingest_body="${work_dir}/ready-ingest.json"
diagnostics_enabled=1

wait_for_success "/health"
wait_for_success "/ready"
wait_for_success "/ready/ingest"
wait_for_status "/ready/answer" "503" "${answer_body}"
grep -q "query_profile_not_configured" "${answer_body}"

redis_restore_required=1
docker compose stop redis >/dev/null
minio_restore_required=1
docker compose stop minio >/dev/null
wait_for_success "/ready"
wait_for_status "/ready/ingest" "503" "${ingest_body}"
grep -q "ingest_dependencies_unavailable" "${ingest_body}"

restore_ingest_dependencies
wait_for_success "/ready/ingest"

echo "RAG foundation smoke test passed"
exit 0
