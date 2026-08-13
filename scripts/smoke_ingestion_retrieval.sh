#!/usr/bin/env bash
set -euo pipefail
umask 077

export COMPOSE_DISABLE_ENV_FILE=1

base_url="${RAG_BASE_URL:-http://localhost:8000}"
provider_base_url="${RAG_SMOKE_PROVIDER_BASE_URL:-https://provider-stub:8443/v1}"
compose_arguments=(compose)
repo_root=""
work_dir=""
provider_secret_file=""
admin_id=""
admin_auth_config=""
agent_id=""
agent_auth_config=""
knowledge_base_id=""
provider_id=""
provider_etag=""
profile_id=""
profile_etag=""
redis_restore_required=0
http_status=""

fail() {
  printf '%s\n' "RAG ingestion/retrieval smoke test failed: $1" >&2
  return 1
}

safe_python() {
  python3 "$@" 2>/dev/null
}

validate_configuration() {
  if [ -z "${RAG_SMOKE_PROVIDER_SECRET:-}" ] ||
    [ "${#RAG_SMOKE_PROVIDER_SECRET}" -gt 4096 ]; then
    fail "RAG_SMOKE_PROVIDER_SECRET must be set to a bounded ephemeral value"
    return 1
  fi

  safe_python - "${base_url}" "${provider_base_url}" <<'PY'
import sys
from urllib.parse import urlsplit

base = urlsplit(sys.argv[1])
try:
    base_port = base.port
except ValueError:
    raise SystemExit(1) from None
if (
    base.scheme not in {"http", "https"}
    or base.hostname not in {"localhost", "127.0.0.1", "::1"}
    or base.username is not None
    or base.password is not None
    or base.path not in {"", "/"}
    or base.query
    or base.fragment
    or base_port is None
):
    raise SystemExit(1)

provider = urlsplit(sys.argv[2])
if (
    provider.scheme != "https"
    or provider.hostname != "provider-stub"
    or provider.port != 8443
    or provider.path.rstrip("/") != "/v1"
    or provider.username is not None
    or provider.password is not None
    or provider.query
    or provider.fragment
):
    raise SystemExit(1)
PY
}

validate_acceptance_ownership() {
  local api_port
  local expected_override="${repo_root}/compose.acceptance.yaml"
  local project="${COMPOSE_PROJECT_NAME:-}"
  local image_tag="${RAG_IMAGE_TAG:-}"

  if [ "${RAG_ACCEPTANCE_EPHEMERAL:-}" != "1" ] ||
    [[ ! "${project}" =~ ^rag-acceptance-[0-9a-f]{16}$ ]] ||
    [ "${RAG_ACCEPTANCE_OWNED_PROJECT:-}" != "${project}" ] ||
    [ "${image_tag}" != "${project}" ] ||
    [ "${RAG_ACCEPTANCE_OWNED_IMAGE_TAG:-}" != "${image_tag}" ] ||
    [ "${RAG_ACCEPTANCE_COMPOSE_OVERRIDE:-}" != "${expected_override}" ] ||
    [[ ! "${base_url}" =~ ^http://127\.0\.0\.1:[1-9][0-9]{0,4}$ ]]; then
    return 1
  fi
  api_port="${base_url##*:}"
  if [ "$((10#${api_port}))" -gt 65535 ]; then
    return 1
  fi
  compose_arguments=(
    compose
    --project-name "${project}"
    -f "${repo_root}/compose.yaml"
    -f "${expected_override}"
    --profile provider-stub
  )
}

cleanup_temp_files() {
  if [ -n "${work_dir}" ]; then
    case "${work_dir}" in
      /tmp/rag-service-ingestion-smoke.*)
        rm -rf -- "${work_dir}" >/dev/null 2>&1 || true
        ;;
    esac
    work_dir=""
  fi
}

compose_silent() {
  COMPOSE_DISABLE_ENV_FILE=1 RAG_IMAGE_TAG="${RAG_IMAGE_TAG:-}" \
    docker "${compose_arguments[@]}" "$@" >/dev/null 2>&1
}

compose_capture() {
  COMPOSE_DISABLE_ENV_FILE=1 RAG_IMAGE_TAG="${RAG_IMAGE_TAG:-}" \
    docker "${compose_arguments[@]}" "$@" 2>/dev/null
}

best_effort_delete_knowledge_base() {
  local detail
  local etag
  local response

  [ -n "${knowledge_base_id}" ] || return 0
  [ -n "${agent_auth_config}" ] || return 1
  [ -r "${agent_auth_config}" ] || return 1
  detail="${work_dir}/cleanup-kb-detail.json"
  response="${work_dir}/cleanup-kb-delete.json"
  if ! http_request "GET" "/v1/knowledge-bases/${knowledge_base_id}" \
    "${agent_auth_config}" "${detail}" ""; then
    return 1
  fi
  [ "${http_status}" = "200" ] || return 1
  etag=$(json_etag "${detail}" "etag" "kb" "${knowledge_base_id}") || return 1
  if ! http_request "DELETE" "/v1/knowledge-bases/${knowledge_base_id}" \
    "${agent_auth_config}" "${response}" "" "If-Match: ${etag}"; then
    return 1
  fi
  [ "${http_status}" = "204" ]
}

best_effort_disable_provider_resource() {
  local label="$1"
  local path="$2"
  local resource_id="$3"
  local etag="$4"
  local response

  [ -n "${resource_id}" ] || return 0
  [ -n "${etag}" ] || return 1
  [ -n "${admin_auth_config}" ] || return 1
  [ -r "${admin_auth_config}" ] || return 1
  case "${label}:${path}" in
    profile:/v1/admin/model-profiles | config:/v1/admin/provider-configs) ;;
    *) return 1 ;;
  esac
  response="${work_dir}/cleanup-${label}.json"
  if ! http_request "PATCH" "${path}/${resource_id}" \
    "${admin_auth_config}" "${response}" "${work_dir}/disable-resource.json" \
    "If-Match: ${etag}"; then
    return 1
  fi
  [ "${http_status}" = "200" ] && assert_no_provider_secret "${response}"
}

best_effort_revoke_key() {
  local key_group="$1"
  local key_id="$2"

  [ -n "${key_id}" ] || return 0
  compose_silent run --rm -T --no-deps api velox-admin \
    "${key_group}" revoke "${key_id}"
}

finish() {
  local exit_code="$?"
  local cleanup_failed=0
  local restore_failed=0

  set +e
  trap - EXIT HUP INT TERM
  if [ "${redis_restore_required}" -eq 1 ]; then
    if compose_silent start redis; then
      redis_restore_required=0
    else
      restore_failed=1
    fi
  fi
  best_effort_delete_knowledge_base || cleanup_failed=1
  best_effort_disable_provider_resource \
    "profile" "/v1/admin/model-profiles" "${profile_id}" "${profile_etag}" ||
    cleanup_failed=1
  best_effort_disable_provider_resource \
    "config" "/v1/admin/provider-configs" "${provider_id}" "${provider_etag}" ||
    cleanup_failed=1
  best_effort_revoke_key "agent-key" "${agent_id}" || cleanup_failed=1
  best_effort_revoke_key "admin-key" "${admin_id}" || cleanup_failed=1
  cleanup_temp_files
  if [ "${restore_failed}" -ne 0 ]; then
    printf '%s\n' "RAG ingestion/retrieval smoke cleanup could not restore Redis" >&2
    exit 1
  fi
  if [ "${cleanup_failed}" -ne 0 ]; then
    printf '%s\n' "RAG ingestion/retrieval smoke cleanup was incomplete" >&2
    if [ "${exit_code}" -eq 0 ]; then
      exit 1
    fi
  fi
  exit "${exit_code}"
}

on_signal() {
  local exit_code="$1"
  trap - HUP INT TERM
  exit "${exit_code}"
}

json_scalar() {
  local input="$1"
  local path="$2"

  safe_python - "${input}" "${path}" <<'PY'
import json
import re
import sys

if not re.fullmatch(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*", sys.argv[2]):
    raise SystemExit(1)
with open(sys.argv[1], encoding="utf-8") as source:
    value = json.load(source)
for segment in sys.argv[2].split("."):
    value = value[segment]
if not isinstance(value, (str, int, float)) or isinstance(value, bool):
    raise SystemExit(1)
print(value)
PY
}

json_uuid() {
  local input="$1"
  local path="$2"

  safe_python - "${input}" "${path}" <<'PY'
import json
import re
import sys
from uuid import UUID

if not re.fullmatch(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*", sys.argv[2]):
    raise SystemExit(1)
with open(sys.argv[1], encoding="utf-8") as source:
    value = json.load(source)
for segment in sys.argv[2].split("."):
    value = value[segment]
if not isinstance(value, str):
    raise SystemExit(1)
print(UUID(value))
PY
}

json_etag() {
  local input="$1"
  local path="$2"
  local prefix="$3"
  local resource_id="$4"

  safe_python - "${input}" "${path}" "${prefix}" "${resource_id}" <<'PY'
import json
import re
import sys
from uuid import UUID

if not re.fullmatch(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)*", sys.argv[2]):
    raise SystemExit(1)
if sys.argv[3] not in {"kb", "provider-config", "model-profile"}:
    raise SystemExit(1)
resource_id = str(UUID(sys.argv[4]))
with open(sys.argv[1], encoding="utf-8") as source:
    value = json.load(source)
for segment in sys.argv[2].split("."):
    value = value[segment]
pattern = rf'"{re.escape(sys.argv[3])}:{re.escape(resource_id)}:r[1-9][0-9]*"'
if not isinstance(value, str) or re.fullmatch(pattern, value) is None:
    raise SystemExit(1)
print(value)
PY
}

json_resource_etag() {
  local input="$1"
  local prefix="$2"
  local resource_id="$3"

  safe_python - "${input}" "${prefix}" "${resource_id}" <<'PY'
import json
import sys
from uuid import UUID

if sys.argv[2] not in {"provider-config", "model-profile"}:
    raise SystemExit(1)
resource_id = str(UUID(sys.argv[3]))
with open(sys.argv[1], encoding="utf-8") as source:
    revision = json.load(source)["resource_revision"]
if not isinstance(revision, int) or isinstance(revision, bool) or revision <= 0:
    raise SystemExit(1)
print(f'"{sys.argv[2]}:{resource_id}:r{revision}"')
PY
}

assert_no_provider_secret() {
  local input="$1"

  safe_python - "${provider_secret_file}" "${input}" <<'PY'
import sys

with open(sys.argv[1], "rb") as secret_source:
    secret = secret_source.read()
with open(sys.argv[2], "rb") as response_source:
    response = response_source.read()
if not secret or secret in response:
    raise SystemExit(1)
PY
}

assert_error_code() {
  local input="$1"
  local expected="$2"

  safe_python - "${input}" "${expected}" <<'PY'
import json
import re
import sys

if not re.fullmatch(r"[A-Z][A-Z0-9_]{1,63}", sys.argv[2]):
    raise SystemExit(1)
with open(sys.argv[1], encoding="utf-8") as source:
    body = json.load(source)
if body.get("error", {}).get("code") != sys.argv[2]:
    raise SystemExit(1)
PY
}

json_error_code() {
  local input="$1"

  safe_python - "${input}" <<'PY'
import json
import re
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    code = json.load(source).get("error", {}).get("code")
if not isinstance(code, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]{1,63}", code):
    raise SystemExit(1)
print(code)
PY
}

write_auth_config() {
  local output="$1"
  local token="$2"

  if [ -z "${token}" ] || [ "${#token}" -gt 256 ] ||
    [[ ! "${token}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    fail "API key token format was invalid"
    return 1
  fi
  printf 'header = "Authorization: Bearer %s"\n' "${token}" >"${output}"
  chmod 600 "${output}"
}

safe_search_validation_code() {
  local code="$1"

  if [[ "${code}" =~ ^SEARCH_[A-Z0-9_]+$ ]]; then
    printf '%s\n' "${code}"
  else
    printf '%s\n' "SEARCH_VALIDATION_UNKNOWN"
  fi
}

monotonic_milliseconds() {
  safe_python - <<'PY'
import time

print(time.monotonic_ns() // 1_000_000)
PY
}

bounded_timeout_seconds() {
  local remaining_milliseconds="$1"
  local cap_milliseconds="$2"
  local selected

  if [[ ! "${remaining_milliseconds}" =~ ^[0-9]+$ ]] ||
    [[ ! "${cap_milliseconds}" =~ ^[0-9]+$ ]] ||
    [ "${remaining_milliseconds}" -le 0 ] ||
    [ "${cap_milliseconds}" -le 0 ]; then
    return 1
  fi
  selected="${remaining_milliseconds}"
  if [ "${selected}" -gt "${cap_milliseconds}" ]; then
    selected="${cap_milliseconds}"
  fi
  printf '%d.%03d\n' "$((selected / 1000))" "$((selected % 1000))"
}

sleep_milliseconds() {
  local duration_milliseconds="$1"
  local duration_seconds

  duration_seconds=$(bounded_timeout_seconds \
    "${duration_milliseconds}" "${duration_milliseconds}") || return 1
  sleep "${duration_seconds}"
}

http_request_with_timeout() {
  local request_timeout="$1"
  shift
  local method="$1"
  local path="$2"
  local auth_config="$3"
  local output="$4"
  local body="$5"
  shift 5
  local curl_args=(
    -q
    --connect-timeout "${request_timeout}"
    --max-time "${request_timeout}"
    --noproxy "*"
    -sS
    -X "${method}"
    --config "${auth_config}"
    -o "${output}"
    -w "%{http_code}"
  )
  local header

  if [[ ! "${request_timeout}" =~ ^[0-9]{1,3}\.[0-9]{3}$ ]] ||
    [ "${request_timeout}" = "0.000" ]; then
    return 1
  fi
  case "${path}" in
    /v1/*) ;;
    *) return 1 ;;
  esac
  if [ -n "${body}" ]; then
    curl_args+=(-H "Content-Type: application/json" --data-binary "@${body}")
  fi
  for header in "$@"; do
    curl_args+=(-H "${header}")
  done
  http_status=""
  if ! http_status=$(curl "${curl_args[@]}" "${base_url}${path}" 2>/dev/null); then
    return 1
  fi
  [[ "${http_status}" =~ ^[0-9]{3}$ ]]
}

http_request() {
  http_request_with_timeout "20.000" "$@"
}

expect_status() {
  local label="$1"
  local expected="$2"
  local response="$3"
  local safe_code="UNKNOWN"

  if ! assert_no_provider_secret "${response}"; then
    fail "${label} exposed protected provider material"
    return 1
  fi
  if [ "${http_status}" != "${expected}" ]; then
    safe_code=$(json_error_code "${response}" 2>/dev/null || printf '%s' "UNKNOWN")
    fail "${label} returned HTTP ${http_status} (expected ${expected}, code ${safe_code})"
    return 1
  fi
}

curl_status_with_timeout() {
  local request_timeout="$1"
  local path="$2"

  if [[ ! "${request_timeout}" =~ ^[0-9]{1,3}\.[0-9]{3}$ ]] ||
    [ "${request_timeout}" = "0.000" ]; then
    return 1
  fi
  case "${path}" in
    /health | /ready/ingest) ;;
    *) return 1 ;;
  esac
  curl -q --connect-timeout "${request_timeout}" --max-time "${request_timeout}" \
    --noproxy "*" -sS -o /dev/null -w "%{http_code}" \
    "${base_url}${path}" 2>/dev/null
}

wait_for_http_status() {
  local deadline_seconds="$1"
  local request_cap_seconds="$2"
  local path="$3"
  local failure_message="$4"
  local started_milliseconds
  local deadline_milliseconds
  local now_milliseconds
  local remaining_milliseconds
  local request_timeout
  local sleep_duration
  local status

  started_milliseconds=$(monotonic_milliseconds) || return 1
  [[ "${started_milliseconds}" =~ ^[0-9]+$ ]] || return 1
  deadline_milliseconds=$((started_milliseconds + deadline_seconds * 1000))
  while true; do
    now_milliseconds=$(monotonic_milliseconds) || return 1
    [[ "${now_milliseconds}" =~ ^[0-9]+$ ]] || return 1
    if [ "${now_milliseconds}" -ge "${deadline_milliseconds}" ]; then
      break
    fi
    remaining_milliseconds=$((deadline_milliseconds - now_milliseconds))
    request_timeout=$(bounded_timeout_seconds \
      "${remaining_milliseconds}" "$((request_cap_seconds * 1000))") || return 1
    status=$(curl_status_with_timeout "${request_timeout}" "${path}" || true)
    if [ "${status}" = "200" ]; then
      return 0
    fi
    now_milliseconds=$(monotonic_milliseconds) || return 1
    [[ "${now_milliseconds}" =~ ^[0-9]+$ ]] || return 1
    if [ "${now_milliseconds}" -ge "${deadline_milliseconds}" ]; then
      break
    fi
    remaining_milliseconds=$((deadline_milliseconds - now_milliseconds))
    sleep_duration=1000
    if [ "${remaining_milliseconds}" -lt "${sleep_duration}" ]; then
      sleep_duration="${remaining_milliseconds}"
    fi
    sleep_milliseconds "${sleep_duration}" || return 1
  done
  fail "${failure_message}"
}

wait_for_api() {
  wait_for_http_status \
    30 3 "/health" "API health did not become ready within 30 seconds"
}

wait_for_ingest_readiness() {
  wait_for_http_status \
    30 3 "/ready/ingest" "ingestion readiness did not recover within 30 seconds"
}

wait_for_job() {
  local job_id="$1"
  local response="${work_dir}/job.json"
  local state
  local started_milliseconds
  local deadline_milliseconds
  local now_milliseconds
  local remaining_milliseconds
  local request_timeout
  local sleep_duration

  started_milliseconds=$(monotonic_milliseconds) || return 1
  [[ "${started_milliseconds}" =~ ^[0-9]+$ ]] || return 1
  deadline_milliseconds=$((started_milliseconds + 60000))
  while true; do
    now_milliseconds=$(monotonic_milliseconds) || return 1
    [[ "${now_milliseconds}" =~ ^[0-9]+$ ]] || return 1
    if [ "${now_milliseconds}" -ge "${deadline_milliseconds}" ]; then
      break
    fi
    remaining_milliseconds=$((deadline_milliseconds - now_milliseconds))
    request_timeout=$(bounded_timeout_seconds \
      "${remaining_milliseconds}" 20000) || return 1
    if ! http_request_with_timeout "${request_timeout}" \
      "GET" "/v1/jobs/${job_id}" \
      "${agent_auth_config}" "${response}" ""; then
      fail "job polling could not reach the API"
      return 1
    fi
    expect_status "job polling" "200" "${response}"
    state=$(json_scalar "${response}" "status") || {
      fail "job polling returned an invalid status"
      return 1
    }
    case "${state}" in
      succeeded) return 0 ;;
      queued | running | retry_wait)
        now_milliseconds=$(monotonic_milliseconds) || return 1
        [[ "${now_milliseconds}" =~ ^[0-9]+$ ]] || return 1
        if [ "${now_milliseconds}" -ge "${deadline_milliseconds}" ]; then
          break
        fi
        remaining_milliseconds=$((deadline_milliseconds - now_milliseconds))
        sleep_duration=1000
        if [ "${remaining_milliseconds}" -lt "${sleep_duration}" ]; then
          sleep_duration="${remaining_milliseconds}"
        fi
        sleep_milliseconds "${sleep_duration}" || return 1
        ;;
      failed | cancelled)
        fail "ingestion job reached a terminal failure state"
        return 1
        ;;
      *)
        fail "ingestion job returned an unknown status"
        return 1
        ;;
    esac
  done
  fail "ingestion job did not succeed within 60 seconds"
}

write_request_bodies() {
  local run_tag="$1"

  RUN_TAG="${run_tag}" PROVIDER_BASE_URL="${provider_base_url}" \
    safe_python - "${work_dir}" <<'PY'
import json
import os
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve()
system_tmp = pathlib.Path("/tmp").resolve()
if root.parent != system_tmp or not root.name.startswith(
    "rag-service-ingestion-smoke."
):
    raise SystemExit(1)

requests = {
    "disable-resource.json": {"enabled": False},
    "agent-policy.json": {
        "name": f"{os.environ['RUN_TAG']}-agent",
        "capabilities": ["manage", "ingest", "retrieve"],
        "knowledge_base_ids": [],
        "query_profile_ids": [],
        "default_query_profile_id": None,
        "raw_file_read": False,
        "requests_per_minute": 120,
        "max_concurrency": 4,
    },
    "knowledge-base.json": {
        "name": f"{os.environ['RUN_TAG']}-knowledge-base",
        "description": "Disposable ingestion retrieval acceptance smoke",
        "metadata": {"owner": "acceptance-smoke"},
    },
    "filter-schema.json": {
        "fields": [
            {
                "name": "department",
                "source_path": "attributes.department",
                "type": "keyword",
                "operators": ["in", "eq"],
            }
        ]
    },
    "provider-config.json": {
        "name": f"{os.environ['RUN_TAG']}-provider",
        "provider_type": "openai_compatible",
        "base_url": os.environ["PROVIDER_BASE_URL"],
        "credential_id": None,
        "default_headers": {},
        "routing_options": {},
        "timeout_seconds": 10,
        "max_concurrency": 4,
        "requests_per_minute": 120,
        "enabled": True,
    },
    "model-profile.json": {
        "name": f"{os.environ['RUN_TAG']}-embedding",
        "capability": "embedding",
        "provider_config_id": None,
        "model_name": "acceptance-embedding",
        "dimension": 3,
        "max_input_tokens": 8192,
        "batch_size": 8,
        "timeout_seconds": 10,
        "vector_config": {},
        "enabled": True,
    },
}
for name, body in requests.items():
    target = root / name
    target.write_text(json.dumps(body, separators=(",", ":")), encoding="utf-8")
    target.chmod(0o600)

(root / "document.md").write_text(
    "# Acceptance Guide\n\n"
    "The cerulean orchard phrase confirms end-to-end retrieval.\n\n"
    "## Verification\n\n"
    "This document belongs to the acceptance department.\n",
    encoding="utf-8",
)
(root / "document.md").chmod(0o600)
PY
}

patch_json_uuid() {
  local input="$1"
  local field="$2"
  local value="$3"

  safe_python - "${input}" "${field}" "${value}" <<'PY'
import json
import pathlib
import re
import sys
from uuid import UUID

path = pathlib.Path(sys.argv[1]).resolve()
system_tmp = pathlib.Path("/tmp").resolve()
if path.parent.parent != system_tmp or not path.parent.name.startswith(
    "rag-service-ingestion-smoke."
):
    raise SystemExit(1)
if not re.fullmatch(r"[a-z][a-z0-9_]*", sys.argv[2]):
    raise SystemExit(1)
value = str(UUID(sys.argv[3]))
body = json.loads(path.read_text(encoding="utf-8"))
body[sys.argv[2]] = value
path.write_text(json.dumps(body, separators=(",", ":")), encoding="utf-8")
path.chmod(0o600)
PY
}

write_credential_request() {
  local run_tag="$1"
  local output="${work_dir}/provider-credential.json"

  RUN_TAG="${run_tag}" PROVIDER_SECRET_FILE="${provider_secret_file}" \
    safe_python - "${output}" <<'PY'
import json
import os
import pathlib
import sys

target = pathlib.Path(sys.argv[1]).resolve()
secret_path = pathlib.Path(os.environ["PROVIDER_SECRET_FILE"]).resolve()
if target.parent != secret_path.parent or not target.parent.name.startswith(
    "rag-service-ingestion-smoke."
):
    raise SystemExit(1)
secret = secret_path.read_text(encoding="utf-8")
if not 1 <= len(secret) <= 4096:
    raise SystemExit(1)
target.write_text(
    json.dumps(
        {"name": f"{os.environ['RUN_TAG']}-credential", "secret": secret},
        separators=(",", ":"),
    ),
    encoding="utf-8",
)
target.chmod(0o600)
PY
}

write_generation_request() {
  local profile_id="$1"
  local output="${work_dir}/generation.json"

  safe_python - "${output}" "${profile_id}" <<'PY'
import json
import pathlib
import sys
from uuid import UUID

target = pathlib.Path(sys.argv[1]).resolve()
system_tmp = pathlib.Path("/tmp").resolve()
if target.parent.parent != system_tmp or not target.parent.name.startswith(
    "rag-service-ingestion-smoke."
):
    raise SystemExit(1)
target.write_text(
    json.dumps(
        {"embedding_profile_id": str(UUID(sys.argv[2])), "distance": "cosine"},
        separators=(",", ":"),
    ),
    encoding="utf-8",
)
target.chmod(0o600)
PY
}

assert_search_response() {
  local response="$1"
  local document_id="$2"
  local version_id="$3"
  local generation_id="$4"
  local profile_id="$5"
  local document_file="$6"

  safe_python - "${response}" "${document_id}" "${version_id}" \
    "${generation_id}" "${profile_id}" "${document_file}" <<'PY'
import json
import math
import pathlib
import sys
from uuid import UUID

with open(sys.argv[1], encoding="utf-8") as source:
    body = json.load(source)
expected_document = str(UUID(sys.argv[2]))
expected_version = str(UUID(sys.argv[3]))
expected_generation = str(UUID(sys.argv[4]))
expected_profile = str(UUID(sys.argv[5]))
document_length = len(pathlib.Path(sys.argv[6]).read_text(encoding="utf-8"))

def reject(code: str) -> None:
    print(code)
    raise SystemExit(1)


if set(body) != {"results", "index"}:
    reject("SEARCH_BODY_SHAPE_MISMATCH")
if body["index"] != {
    "generation_id": expected_generation,
    "embedding_profile_id": expected_profile,
}:
    reject("SEARCH_INDEX_MISMATCH")
results = body["results"]
if not isinstance(results, list) or not results:
    reject("SEARCH_RESULTS_MISMATCH")
matches = [item for item in results if "cerulean orchard phrase" in item.get("text", "")]
if len(matches) != 1:
    reject("SEARCH_PHRASE_MISMATCH")
item = matches[0]
score = item.get("score")
if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
    reject("SEARCH_SCORE_MISMATCH")
if item.get("document_id") != expected_document or item.get("version_id") != expected_version:
    reject("SEARCH_IDS_MISMATCH")
if not isinstance(item.get("chunk_index"), int) or item["chunk_index"] < 0:
    reject("SEARCH_CHUNK_INDEX_MISMATCH")
if not isinstance(item.get("title"), str) or not item["title"]:
    reject("SEARCH_TITLE_MISMATCH")
title_path = item.get("title_path")
if not isinstance(title_path, list) or "Acceptance Guide" not in title_path:
    reject("SEARCH_TITLE_PATH_MISMATCH")
source = item.get("source")
if not isinstance(source, dict):
    reject("SEARCH_SOURCE_SHAPE_MISMATCH")
if source.get("filename") != "Acceptance Guide":
    reject("SEARCH_SOURCE_FILENAME_MISMATCH")
start = source.get("start_offset")
end = source.get("end_offset")
if (
    not isinstance(start, int)
    or isinstance(start, bool)
    or not isinstance(end, int)
    or isinstance(end, bool)
    or start < 0
    or end <= start
    or end > document_length
):
    reject("SEARCH_SOURCE_OFFSETS_MISMATCH")
if item.get("metadata", {}).get("department") != "acceptance":
    reject("SEARCH_METADATA_MISMATCH")
PY
}

main() {
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
cd -- "${repo_root}"
if ! validate_acceptance_ownership; then
  fail "ephemeral acceptance ownership contract was invalid"
  return 1
fi

trap finish EXIT
trap 'on_signal 129' HUP
trap 'on_signal 130' INT
trap 'on_signal 143' TERM

validate_configuration
work_dir=$(mktemp -d "/tmp/rag-service-ingestion-smoke.XXXXXX")
chmod 700 "${work_dir}"
provider_secret_file="${work_dir}/provider-secret"
printf '%s' "${RAG_SMOKE_PROVIDER_SECRET}" >"${provider_secret_file}"
chmod 600 "${provider_secret_file}"
unset RAG_SMOKE_PROVIDER_SECRET

run_tag="ingestion-smoke-$$-${RANDOM}"
write_request_bodies "${run_tag}"
write_credential_request "${run_tag}"
wait_for_api
wait_for_ingest_readiness

admin_created="${work_dir}/admin-created.json"
if ! compose_capture run --rm -T --no-deps api velox-admin admin-key create \
  --name "${run_tag}-admin" >"${admin_created}"; then
  fail "Admin API key bootstrap failed"
fi
admin_id=$(json_uuid "${admin_created}" "api_key.id") || fail "Admin key response was invalid"
admin_token=$(json_scalar "${admin_created}" "token") || fail "Admin key response was invalid"
admin_auth_config="${work_dir}/admin-auth.conf"
write_auth_config "${admin_auth_config}" "${admin_token}"
admin_token=""

agent_created="${work_dir}/agent-created.json"
http_request "POST" "/v1/admin/api-keys" "${admin_auth_config}" \
  "${agent_created}" "${work_dir}/agent-policy.json" || fail "Agent key creation could not reach the API"
expect_status "Agent key creation" "201" "${agent_created}"
agent_id=$(json_uuid "${agent_created}" "api_key.id") || fail "Agent key response was invalid"
agent_token=$(json_scalar "${agent_created}" "token") || fail "Agent key response was invalid"
agent_auth_config="${work_dir}/agent-auth.conf"
write_auth_config "${agent_auth_config}" "${agent_token}"
agent_token=""

kb_created="${work_dir}/knowledge-base-created.json"
http_request "POST" "/v1/knowledge-bases" "${agent_auth_config}" \
  "${kb_created}" "${work_dir}/knowledge-base.json" \
  "Idempotency-Key: ${run_tag}-kb" || fail "Knowledge base creation could not reach the API"
expect_status "Knowledge base creation" "201" "${kb_created}"
knowledge_base_id=$(json_uuid "${kb_created}" "id") || fail "Knowledge base response was invalid"
knowledge_base_etag=$(json_etag \
  "${kb_created}" "etag" "kb" "${knowledge_base_id}") ||
  fail "Knowledge base response was invalid"

filter_replaced="${work_dir}/filter-schema-replaced.json"
http_request "PUT" "/v1/knowledge-bases/${knowledge_base_id}/filter-schema" \
  "${agent_auth_config}" "${filter_replaced}" "${work_dir}/filter-schema.json" \
  "If-Match: ${knowledge_base_etag}" || fail "Filter schema update could not reach the API"
expect_status "Filter schema update" "200" "${filter_replaced}"

credential_created="${work_dir}/credential-created.json"
http_request "POST" "/v1/admin/provider-credentials" "${admin_auth_config}" \
  "${credential_created}" "${work_dir}/provider-credential.json" \
  "Idempotency-Key: ${run_tag}-credential" || fail "Provider credential creation could not reach the API"
expect_status "Provider credential creation" "201" "${credential_created}"
credential_id=$(json_uuid "${credential_created}" "id") || fail "Provider credential response was invalid"

patch_json_uuid "${work_dir}/provider-config.json" "credential_id" "${credential_id}"
provider_created="${work_dir}/provider-created.json"
http_request "POST" "/v1/admin/provider-configs" "${admin_auth_config}" \
  "${provider_created}" "${work_dir}/provider-config.json" \
  "Idempotency-Key: ${run_tag}-provider" || fail "Provider configuration creation could not reach the API"
expect_status "Provider configuration creation" "201" "${provider_created}"
provider_id=$(json_uuid "${provider_created}" "id") || fail "Provider configuration response was invalid"
provider_etag=$(json_resource_etag \
  "${provider_created}" "provider-config" "${provider_id}") ||
  fail "Provider configuration response was invalid"

patch_json_uuid "${work_dir}/model-profile.json" "provider_config_id" "${provider_id}"
profile_created="${work_dir}/profile-created.json"
http_request "POST" "/v1/admin/model-profiles" "${admin_auth_config}" \
  "${profile_created}" "${work_dir}/model-profile.json" \
  "Idempotency-Key: ${run_tag}-profile" || fail "Model profile creation could not reach the API"
expect_status "Model profile creation" "201" "${profile_created}"
profile_id=$(json_uuid "${profile_created}" "id") || fail "Model profile response was invalid"
profile_etag=$(json_resource_etag \
  "${profile_created}" "model-profile" "${profile_id}") ||
  fail "Model profile response was invalid"

write_generation_request "${profile_id}"
generation_created="${work_dir}/generation-created.json"
http_request "POST" "/v1/admin/knowledge-bases/${knowledge_base_id}/index-generations" \
  "${admin_auth_config}" "${generation_created}" "${work_dir}/generation.json" \
  "Idempotency-Key: ${run_tag}-generation" || fail "Initial generation creation could not reach the API"
expect_status "Initial generation creation" "201" "${generation_created}"
generation_id=$(json_uuid "${generation_created}" "id") || fail "Generation response was invalid"

redis_restore_required=1
compose_silent stop redis || fail "Redis could not be paused for DB polling verification"

upload_created="${work_dir}/upload-created.json"
http_status=$(curl -q --connect-timeout 2 --max-time 30 --noproxy "*" -sS \
  --config "${agent_auth_config}" \
  -o "${upload_created}" -w "%{http_code}" \
  -H "X-Request-ID: ${run_tag}-upload" \
  -H "Idempotency-Key: ${run_tag}-upload" \
  -F "file=@${work_dir}/document.md;type=text/markdown;filename=acceptance-guide.md" \
  -F "display_name=Acceptance Guide" \
  -F 'metadata={"attributes":{"department":"acceptance"}}' \
  -F 'tags=["acceptance","e2e"]' \
  "${base_url}/v1/knowledge-bases/${knowledge_base_id}/documents" 2>/dev/null) ||
  fail "Document upload could not reach the API"
expect_status "Document upload" "202" "${upload_created}"
document_id=$(json_uuid "${upload_created}" "document_id") || fail "Upload response was invalid"
version_id=$(json_uuid "${upload_created}" "version_id") || fail "Upload response was invalid"
job_id=$(json_uuid "${upload_created}" "job_id") || fail "Upload response was invalid"

wait_for_job "${job_id}"
compose_silent start redis || fail "Redis could not be restored after DB polling verification"
redis_restore_required=0
wait_for_ingest_readiness

search_request="${work_dir}/search.json"
printf '%s\n' '{"query":"cerulean orchard phrase","top_k":5}' >"${search_request}"
chmod 600 "${search_request}"
search_response="${work_dir}/search-response.json"
http_request "POST" "/v1/knowledge-bases/${knowledge_base_id}/search" \
  "${agent_auth_config}" "${search_response}" "${search_request}" ||
  fail "Search could not reach the API"
expect_status "Search" "200" "${search_response}"
search_validation_code=""
if ! search_validation_code=$(assert_search_response \
  "${search_response}" "${document_id}" "${version_id}" \
  "${generation_id}" "${profile_id}" "${work_dir}/document.md"); then
  search_validation_code=$(safe_search_validation_code "${search_validation_code}")
  fail "Search response validation failed (code ${search_validation_code})"
fi

duplicate_response="${work_dir}/duplicate-response.json"
http_status=$(curl -q --connect-timeout 2 --max-time 30 --noproxy "*" -sS \
  --config "${agent_auth_config}" \
  -o "${duplicate_response}" -w "%{http_code}" \
  -H "X-Request-ID: ${run_tag}-duplicate" \
  -H "Idempotency-Key: ${run_tag}-duplicate" \
  -F "file=@${work_dir}/document.md;type=text/markdown;filename=acceptance-guide.md" \
  "${base_url}/v1/knowledge-bases/${knowledge_base_id}/documents" 2>/dev/null) ||
  fail "Duplicate upload check could not reach the API"
expect_status "Duplicate upload" "409" "${duplicate_response}"
assert_error_code "${duplicate_response}" "DUPLICATE_DOCUMENT" ||
  fail "Duplicate upload returned an unexpected safe error code"

printf '%s\n' "RAG ingestion/retrieval smoke test passed"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
