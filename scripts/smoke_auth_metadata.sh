#!/usr/bin/env bash
set -euo pipefail
umask 077

base_url="${RAG_BASE_URL:-http://localhost:8000}"
tmp_root="${TMPDIR:-/tmp}"
lock_dir="${RAG_AUTH_SMOKE_LOCK_DIR:-/tmp/rag-service-auth-smoke.lock}"
work_dir=""
lock_owned=0
http_status=""
admin_id=""
admin_token=""
admin_auth_config=""
admin_name=""
admin_active=0
agent_one_id=""
agent_one_token=""
agent_one_auth_config=""
agent_one_name=""
agent_one_active=0
agent_two_id=""
agent_two_token=""
agent_two_auth_config=""
agent_two_name=""
agent_two_active=0
knowledge_base_id=""
knowledge_base_idempotency_key=""
knowledge_base_active=0

fail() {
  echo "RAG authenticated metadata smoke test failed: $1" >&2
  return 1
}

validate_base_url() {
  python3 - "$1" <<'PY'
import sys
from urllib.parse import urlsplit

try:
    parsed = urlsplit(sys.argv[1])
    port = parsed.port
except ValueError:
    raise SystemExit(1) from None

if (
    parsed.scheme not in {"http", "https"}
    or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
    or parsed.username is not None
    or parsed.password is not None
    or parsed.path not in {"", "/"}
    or parsed.query
    or parsed.fragment
    or port is None
):
    raise SystemExit(1)
PY
}

write_auth_config() {
  local output="$1"
  local token="$2"

  printf 'header = "Authorization: Bearer %s"\n' "${token}" >"${output}"
  chmod 600 "${output}"
}

acquire_lock() {
  local lock_holder="unknown"

  if ! mkdir "${lock_dir}" 2>/dev/null; then
    if [ -r "${lock_dir}/pid" ]; then
      IFS= read -r lock_holder <"${lock_dir}/pid" || lock_holder="unknown"
    fi
    fail "another run owns the lock (pid: ${lock_holder})"
    return 1
  fi

  lock_owned=1
  if ! printf '%s\n' "$$" >"${lock_dir}/pid"; then
    fail "unable to record the lock owner"
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

json_get() {
  local input="$1"
  local path="$2"

  python3 - "${input}" "${path}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    value = json.load(source)
for segment in sys.argv[2].split("."):
    value = value[segment]
if not isinstance(value, (str, int, float)) or isinstance(value, bool):
    raise SystemExit(1)
print(value)
PY
}

json_optional_get() {
  local input="$1"
  local path="$2"

  python3 - "${input}" "${path}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    value = json.load(source)
for segment in sys.argv[2].split("."):
    value = value[segment]
if value is None:
    raise SystemExit(0)
if not isinstance(value, (str, int, float)) or isinstance(value, bool):
    raise SystemExit(1)
print(value)
PY
}

json_find_id_by_name() {
  local input="$1"
  local expected_name="$2"

  python3 - "${input}" "${expected_name}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    document = json.load(source)
for item in document.get("items", []):
    if item.get("name") == sys.argv[2] and isinstance(item.get("id"), str):
        print(item["id"])
        raise SystemExit(0)
raise SystemExit(1)
PY
}

assert_json_equal() {
  python3 - "$1" "$2" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as left_source:
    left = json.load(left_source)
with open(sys.argv[2], encoding="utf-8") as right_source:
    right = json.load(right_source)
if left != right:
    raise SystemExit(1)
PY
}

assert_json_error() {
  local input="$1"
  local expected_code="$2"

  python3 - "${input}" "${expected_code}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    document = json.load(source)
if document.get("error", {}).get("code") != sys.argv[2]:
    raise SystemExit(1)
PY
}

assert_list_contains() {
  local input="$1"
  local expected_id="$2"

  python3 - "${input}" "${expected_id}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    document = json.load(source)
if not any(item.get("id") == sys.argv[2] for item in document.get("items", [])):
    raise SystemExit(1)
PY
}

http_request() {
  local method="$1"
  local path="$2"
  local auth_config="$3"
  local output="$4"
  local body="$5"
  shift 5
  local curl_args=(
    -q
    --connect-timeout 2
    --max-time 15
    --noproxy "*"
    -sS
    -X "${method}"
    --config "${auth_config}"
    -o "${output}"
    -w "%{http_code}"
  )
  local header

  if [ -n "${body}" ]; then
    curl_args+=(
      -H "Content-Type: application/json"
      --data-binary "@${body}"
    )
  fi
  for header in "$@"; do
    curl_args+=(-H "${header}")
  done

  http_status=""
  if ! http_status=$(curl "${curl_args[@]}" "${base_url}${path}"); then
    fail "${method} ${path} could not reach the API"
    return 1
  fi
}

find_cli_key_id_by_name() {
  local key_group="$1"
  local expected_name="$2"
  local cursor=""
  local page=0
  local page_file
  local found_id
  local next_cursor
  local cli_args

  while [ "${page}" -lt 100 ]; do
    page=$((page + 1))
    page_file="${work_dir}/cleanup-${key_group}-page-${page}.json"
    cli_args=(
      compose run --rm -T --no-deps api velox-admin
      "${key_group}" list --limit 100
    )
    if [ -n "${cursor}" ]; then
      cli_args+=(--cursor "${cursor}")
    fi
    if ! docker "${cli_args[@]}" >"${page_file}" 2>/dev/null; then
      return 1
    fi
    if found_id=$(json_find_id_by_name "${page_file}" "${expected_name}" 2>/dev/null); then
      printf '%s\n' "${found_id}"
      return 0
    fi
    next_cursor=$(json_optional_get "${page_file}" "next_cursor" 2>/dev/null) || return 1
    if [ -z "${next_cursor}" ]; then
      return 1
    fi
    cursor="${next_cursor}"
  done
  return 1
}

expect_status() {
  local operation="$1"
  local expected="$2"

  if [ "${http_status}" != "${expected}" ]; then
    fail "${operation} returned HTTP ${http_status}; expected ${expected}"
    return 1
  fi
}

wait_for_api() {
  local status
  local attempt

  for attempt in $(seq 1 60); do
    status=$(curl -q --connect-timeout 2 --max-time 5 --noproxy "*" \
      -sS -o /dev/null -w "%{http_code}" \
      "${base_url}/health" || true)
    if [ "${status}" = "200" ]; then
      return 0
    fi
    sleep 2
  done
  fail "API health endpoint did not become ready"
}

best_effort_revoke_key() {
  local key_group="$1"
  local key_id="$2"
  local key_name="$3"
  local recovered_id=""

  if [ -z "${key_id}" ] && [ -n "${key_name}" ]; then
    recovered_id=$(find_cli_key_id_by_name "${key_group}" "${key_name}" 2>/dev/null) || true
    key_id="${recovered_id}"
  fi
  [ -n "${key_id}" ] || return 0
  docker compose run --rm -T --no-deps api velox-admin \
    "${key_group}" revoke "${key_id}" >/dev/null 2>&1 || true
}

best_effort_cleanup_resources() {
  local kb_replay="${work_dir}/cleanup-kb-replay.json"
  local kb_detail="${work_dir}/cleanup-kb.json"
  local kb_delete="${work_dir}/cleanup-kb-delete.json"
  local current_etag

  if [ "${knowledge_base_active}" -eq 1 ] && [ -r "${agent_one_auth_config}" ]; then
    if [ -z "${knowledge_base_id}" ] && [ -r "${knowledge_base_command:-}" ]; then
      http_request "POST" "/v1/knowledge-bases" "${agent_one_auth_config}" \
        "${kb_replay}" "${knowledge_base_command}" \
        "Idempotency-Key: ${knowledge_base_idempotency_key}" >/dev/null 2>&1 || true
      if [ "${http_status}" = "200" ] || [ "${http_status}" = "201" ]; then
        knowledge_base_id=$(json_get "${kb_replay}" "id" 2>/dev/null) || knowledge_base_id=""
      fi
    fi
    if [ -n "${knowledge_base_id}" ]; then
      http_request "GET" "/v1/knowledge-bases/${knowledge_base_id}" \
        "${agent_one_auth_config}" \
        "${kb_detail}" "" >/dev/null 2>&1 || true
      if [ "${http_status}" = "200" ]; then
        current_etag=$(json_get "${kb_detail}" "etag" 2>/dev/null) || current_etag=""
        if [ -n "${current_etag}" ]; then
          http_request "DELETE" "/v1/knowledge-bases/${knowledge_base_id}" \
            "${agent_one_auth_config}" "${kb_delete}" "" \
            "If-Match: ${current_etag}" >/dev/null 2>&1 || true
        fi
      fi
    fi
  fi
  if [ "${agent_one_active}" -eq 1 ]; then
    best_effort_revoke_key "agent-key" "${agent_one_id}" "${agent_one_name}"
  fi
  if [ "${agent_two_active}" -eq 1 ]; then
    best_effort_revoke_key "agent-key" "${agent_two_id}" "${agent_two_name}"
  fi
  if [ "${admin_active}" -eq 1 ]; then
    best_effort_revoke_key "admin-key" "${admin_id}" "${admin_name}"
  fi
}

cleanup() {
  local exit_code="$?"

  set +e
  trap - EXIT HUP INT TERM
  if [ -n "${work_dir}" ]; then
    best_effort_cleanup_resources
    case "${work_dir}" in
      "${tmp_root%/}"/rag-service-auth-smoke.*)
        rm -rf -- "${work_dir}"
        ;;
    esac
    work_dir=""
  fi
  if [ "${exit_code}" -ne 0 ]; then
    docker compose ps >&2 || true
  fi
  release_lock
  exit "${exit_code}"
}

on_signal() {
  local exit_code="$1"

  trap - HUP INT TERM
  exit "${exit_code}"
}

main() {
if ! validate_base_url "${base_url}"; then
  fail "RAG_BASE_URL must be an explicit loopback http(s) URL with a port"
fi

trap cleanup EXIT
trap 'on_signal 129' HUP
trap 'on_signal 130' INT
trap 'on_signal 143' TERM

acquire_lock
work_dir=$(mktemp -d "${tmp_root%/}/rag-service-auth-smoke.XXXXXX")
run_tag="auth-smoke-$$-${RANDOM}"
admin_name="${run_tag}-admin"
agent_one_name="${run_tag}-owner"
agent_two_name="${run_tag}-outsider"
knowledge_base_idempotency_key="${run_tag}-kb-create"
wait_for_api

admin_created="${work_dir}/admin-created.json"
admin_create_error="${work_dir}/admin-create.stderr"
admin_active=1
if ! docker compose run --rm -T --no-deps api velox-admin admin-key create \
  --name "${admin_name}" >"${admin_created}" 2>"${admin_create_error}"; then
  fail "Admin API key bootstrap failed"
fi
admin_id=$(json_get "${admin_created}" "api_key.id")
admin_token=$(json_get "${admin_created}" "token")
admin_auth_config="${work_dir}/admin-auth.conf"
write_auth_config "${admin_auth_config}" "${admin_token}"
admin_token="<redacted>"

agent_policy="${work_dir}/agent-policy.json"
printf '{"name":"%s","capabilities":["manage"],"knowledge_base_ids":[],"query_profile_ids":[],"default_query_profile_id":null,"raw_file_read":false,"requests_per_minute":60,"max_concurrency":4}\n' \
  "${agent_one_name}" >"${agent_policy}"
agent_one_created="${work_dir}/agent-one-created.json"
# POST /v1/admin/api-keys
agent_one_active=1
http_request "POST" "/v1/admin/api-keys" "${admin_auth_config}" \
  "${agent_one_created}" "${agent_policy}"
expect_status "create first Agent API key" "201"
agent_one_id=$(json_get "${agent_one_created}" "api_key.id")
agent_one_token=$(json_get "${agent_one_created}" "token")
agent_one_auth_config="${work_dir}/agent-one-auth.conf"
write_auth_config "${agent_one_auth_config}" "${agent_one_token}"
agent_one_token="<redacted>"

knowledge_base_command="${work_dir}/knowledge-base-command.json"
printf '%s\n' '{"name":"Authenticated smoke knowledge base","description":"Disposable operations verification","metadata":{"owner":"smoke"}}' >"${knowledge_base_command}"
knowledge_base_created="${work_dir}/knowledge-base-created.json"
knowledge_base_active=1
http_request "POST" "/v1/knowledge-bases" "${agent_one_auth_config}" \
  "${knowledge_base_created}" "${knowledge_base_command}" \
  "Idempotency-Key: ${knowledge_base_idempotency_key}"
expect_status "create knowledge base" "201"
knowledge_base_id=$(json_get "${knowledge_base_created}" "id")
knowledge_base_etag=$(json_get "${knowledge_base_created}" "etag")

knowledge_base_replayed="${work_dir}/knowledge-base-replayed.json"
http_request "POST" "/v1/knowledge-bases" "${agent_one_auth_config}" \
  "${knowledge_base_replayed}" "${knowledge_base_command}" \
  "Idempotency-Key: ${knowledge_base_idempotency_key}"
expect_status "replay knowledge base creation" "200"
assert_json_equal "${knowledge_base_created}" "${knowledge_base_replayed}" || \
  fail "idempotency replay returned a different resource"

knowledge_base_list="${work_dir}/knowledge-base-list.json"
http_request "GET" "/v1/knowledge-bases" "${agent_one_auth_config}" \
  "${knowledge_base_list}" ""
expect_status "list knowledge bases" "200"
assert_list_contains "${knowledge_base_list}" "${knowledge_base_id}" || \
  fail "created knowledge base was absent from the list"

knowledge_base_detail="${work_dir}/knowledge-base-detail.json"
http_request "GET" "/v1/knowledge-bases/${knowledge_base_id}" \
  "${agent_one_auth_config}" \
  "${knowledge_base_detail}" ""
expect_status "get knowledge base" "200"
[ "$(json_get "${knowledge_base_detail}" "id")" = "${knowledge_base_id}" ] || \
  fail "knowledge base detail returned the wrong resource"

filter_command="${work_dir}/filter-command.json"
printf '%s\n' '{"fields":[{"name":"department","source_path":"attributes.department","type":"keyword","operators":["in","eq"]}]}' >"${filter_command}"
filter_replaced="${work_dir}/filter-replaced.json"
http_request "PUT" "/v1/knowledge-bases/${knowledge_base_id}/filter-schema" \
  "${agent_one_auth_config}" "${filter_replaced}" "${filter_command}" \
  "If-Match: ${knowledge_base_etag}"
expect_status "replace filter schema" "200"
filter_etag=$(json_get "${filter_replaced}" "etag")

stale_response="${work_dir}/stale-response.json"
http_request "PUT" "/v1/knowledge-bases/${knowledge_base_id}/filter-schema" \
  "${agent_one_auth_config}" "${stale_response}" "${filter_command}" \
  "If-Match: ${knowledge_base_etag}"
expect_status "reject stale knowledge base ETag" "412"
assert_json_error "${stale_response}" "PRECONDITION_FAILED" || \
  fail "stale ETag did not return PRECONDITION_FAILED"

agent_two_policy="${work_dir}/agent-two-policy.json"
printf '{"name":"%s","capabilities":["manage"],"knowledge_base_ids":[],"query_profile_ids":[],"default_query_profile_id":null,"raw_file_read":false,"requests_per_minute":60,"max_concurrency":4}\n' \
  "${agent_two_name}" >"${agent_two_policy}"
agent_two_created="${work_dir}/agent-two-created.json"
agent_two_active=1
http_request "POST" "/v1/admin/api-keys" "${admin_auth_config}" \
  "${agent_two_created}" "${agent_two_policy}"
expect_status "create unscoped Agent API key" "201"
agent_two_id=$(json_get "${agent_two_created}" "api_key.id")
agent_two_token=$(json_get "${agent_two_created}" "token")
agent_two_auth_config="${work_dir}/agent-two-auth.conf"
write_auth_config "${agent_two_auth_config}" "${agent_two_token}"
agent_two_token="<redacted>"

hidden_response="${work_dir}/hidden-response.json"
http_request "GET" "/v1/knowledge-bases/${knowledge_base_id}" \
  "${agent_two_auth_config}" \
  "${hidden_response}" ""
expect_status "hide out-of-scope knowledge base" "404"
assert_json_error "${hidden_response}" "RESOURCE_NOT_FOUND" || \
  fail "out-of-scope access did not return RESOURCE_NOT_FOUND"

knowledge_base_deleted="${work_dir}/knowledge-base-deleted.json"
http_request "DELETE" "/v1/knowledge-bases/${knowledge_base_id}" \
  "${agent_one_auth_config}" "${knowledge_base_deleted}" "" \
  "If-Match: ${filter_etag}"
expect_status "delete smoke knowledge base" "204"
knowledge_base_active=0

agent_one_detail="${work_dir}/agent-one-detail.json"
http_request "GET" "/v1/admin/api-keys/${agent_one_id}" "${admin_auth_config}" \
  "${agent_one_detail}" ""
expect_status "refresh first Agent API key ETag" "200"
agent_one_etag=$(json_get "${agent_one_detail}" "etag")

agent_one_revoked="${work_dir}/agent-one-revoked.json"
http_request "POST" "/v1/admin/api-keys/${agent_one_id}/revoke" \
  "${admin_auth_config}" \
  "${agent_one_revoked}" "" "If-Match: ${agent_one_etag}"
expect_status "revoke first Agent API key" "200"
agent_one_active=0

revoked_response="${work_dir}/revoked-response.json"
http_request "GET" "/v1/knowledge-bases" "${agent_one_auth_config}" \
  "${revoked_response}" ""
expect_status "reject revoked Agent API key" "401"
assert_json_error "${revoked_response}" "INVALID_API_KEY" || \
  fail "revoked key did not return INVALID_API_KEY"

agent_two_detail="${work_dir}/agent-two-detail.json"
http_request "GET" "/v1/admin/api-keys/${agent_two_id}" "${admin_auth_config}" \
  "${agent_two_detail}" ""
expect_status "refresh second Agent API key ETag" "200"
agent_two_etag=$(json_get "${agent_two_detail}" "etag")

agent_two_revoked="${work_dir}/agent-two-revoked.json"
http_request "POST" "/v1/admin/api-keys/${agent_two_id}/revoke" \
  "${admin_auth_config}" \
  "${agent_two_revoked}" "" "If-Match: ${agent_two_etag}"
expect_status "revoke second Agent API key" "200"
agent_two_active=0

admin_revoked="${work_dir}/admin-revoked.json"
admin_revoke_error="${work_dir}/admin-revoke.stderr"
if ! docker compose run --rm -T --no-deps api velox-admin admin-key revoke "${admin_id}" \
  >"${admin_revoked}" 2>"${admin_revoke_error}"; then
  fail "temporary Admin API key revoke failed"
fi
admin_active=0

echo "RAG authenticated metadata smoke test passed"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main "$@"
fi
