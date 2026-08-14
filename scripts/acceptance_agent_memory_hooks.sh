#!/usr/bin/env bash
# End-to-end acceptance for velox-hook against the live local stack.
#
# Records one turn through `velox-hook record` and polls `velox-hook retrieve`
# until it comes back, reporting the upload-to-searchable latency. This is the
# open question the design left unanswered: how long after a turn is recorded
# can it be retrieved. That number is the input to calibrating the score floor
# and decides whether a later turn in the same session can retrieve an earlier
# one.
#
# Individual documents can never be deleted (`/v1/documents/*` is GET-only), so
# this never writes to the real `local-memory` knowledge base. It creates its
# own knowledge base with `velox-bootstrap` and deletes the whole thing when it
# finishes, on success, on failure and on interrupt.
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
hook_bin="${repo_root}/.venv/bin/velox-hook"
bootstrap_bin="${repo_root}/.venv/bin/velox-bootstrap"

# Deadline for the recorded turn to become searchable. A job stuck in
# retry_wait (Ollama unreachable from the containers) is the usual cause of
# hitting this.
deadline_seconds=120

fail() {
  printf 'agent-memory-hooks acceptance failed: %s\n' "$1" >&2
  exit 1
}

base_url="${VELOX_HOOK_BASE_URL:-http://127.0.0.1:8000}"

[ -x "${hook_bin}" ] || fail "velox-hook console script not found at ${hook_bin}; run 'uv sync'"
[ -x "${bootstrap_bin}" ] || fail "velox-bootstrap console script not found at ${bootstrap_bin}; run 'uv sync'"

# Nothing has been created yet, so a readiness failure needs no cleanup: fail
# directly rather than going through the trap machinery set up below.
#
# No `|| printf '000'` fallback here: curl already writes "000" to %{http_code}
# when the connection itself fails, so a fallback appended on top of that would
# double up into "000000" inside the same command substitution.
if ! ready_status=$(curl -sS --noproxy "*" --connect-timeout 5 --max-time 10 \
  -o /dev/null -w '%{http_code}' "${base_url}/ready/retrieve" 2>/dev/null); then
  ready_status="000"
fi
if [ "${ready_status}" != "200" ]; then
  printf 'agent-memory-hooks acceptance failed: GET %s/ready/retrieve returned %s (expected 200)\n' \
    "${base_url}" "${ready_status}" >&2
  printf 'run `veloxrag start` and retry\n' >&2
  exit 1
fi

# Not a timestamp: repeated runs use the shell's pid and a random number rather
# than `date`, so two runs started in the same second still cannot collide.
run_tag="hook-acceptance-$$-${RANDOM}"
kb_name="${run_tag}"
kb_id=""
work_dir=""

# Installed before the knowledge base exists, so a failure or interrupt during
# bootstrap itself is still followed by a cleanup attempt (a no-op, since
# kb_id is empty until bootstrap reports it).
finish() {
  local exit_code="$?"
  local cleanup_failed=0
  local delete_status
  local etag

  set +e
  trap - EXIT HUP INT TERM
  if [ -n "${kb_id}" ]; then
    # Deletion is optimistic-concurrency-controlled: the endpoint 412s without
    # a matching If-Match, so the current ETag has to be read first.
    etag=$(curl -sS --noproxy "*" --connect-timeout 5 --max-time 20 -D - -o /dev/null \
      "${base_url}/v1/knowledge-bases/${kb_id}" 2>/dev/null |
      grep -i '^etag:' | sed -E 's/^[Ee][Tt]ag: *//' | tr -d '\r\n')
    if [ -n "${etag}" ]; then
      delete_status=$(curl -sS --noproxy "*" --connect-timeout 5 --max-time 20 \
        -o /dev/null -w '%{http_code}' -X DELETE -H "If-Match: ${etag}" \
        "${base_url}/v1/knowledge-bases/${kb_id}" 2>/dev/null)
    else
      delete_status="000"
    fi
    if [ "${delete_status}" = "204" ]; then
      printf 'cleanup: deleted knowledge base %s (%s)\n' "${kb_name}" "${kb_id}"
    else
      printf 'cleanup: FAILED to delete knowledge base %s (%s) -- got HTTP %s, delete it manually\n' \
        "${kb_name}" "${kb_id}" "${delete_status}" >&2
      cleanup_failed=1
    fi
  fi
  if [ -n "${work_dir}" ]; then
    rm -rf -- "${work_dir}" >/dev/null 2>&1 || true
  fi
  if [ "${cleanup_failed}" -ne 0 ] && [ "${exit_code}" -eq 0 ]; then
    exit 1
  fi
  exit "${exit_code}"
}

on_signal() {
  local exit_code="$1"
  trap - HUP INT TERM
  exit "${exit_code}"
}

trap finish EXIT
trap 'on_signal 129' HUP
trap 'on_signal 130' INT
trap 'on_signal 143' TERM

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/velox-hook-acceptance.XXXXXX")
chmod 700 "${work_dir}"

# RAG_BOOTSTRAP_BASE_URL defaults to http://api:8000, the in-container address;
# velox-bootstrap here is the host-installed console script, so it needs the
# host-facing URL instead.
bootstrap_log="${work_dir}/bootstrap.log"
if ! RAG_BOOTSTRAP_BASE_URL="${base_url}" RAG_BOOTSTRAP_KNOWLEDGE_BASE="${kb_name}" \
  "${bootstrap_bin}" 2>&1 | tee "${bootstrap_log}"; then
  fail "velox-bootstrap did not finish successfully; see output above"
fi

# Only the last line: velox-bootstrap's JSON result is its last line of output,
# and nothing guarantees earlier lines (progress, warnings) are also JSON.
#
# Read via a file argument, not a pipe into stdin: `python3 -` already reads
# its own program from stdin (that is what `-` means), so a heredoc program
# combined with piped JSON on stdin would starve one or the other -- the
# heredoc wins, and the JSON read comes back empty.
bootstrap_last_line="${work_dir}/bootstrap-last-line.json"
tail -n1 "${bootstrap_log}" >"${bootstrap_last_line}"
kb_id=$(python3 - "${bootstrap_last_line}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    try:
        data = json.load(source)
    except ValueError:
        sys.exit(1)
value = data.get("knowledge_base_id")
if not isinstance(value, str) or not value:
    sys.exit(1)
print(value)
PY
) || fail "velox-bootstrap output did not include a knowledge_base_id"

printf 'created knowledge base %s (%s)\n' "${kb_name}" "${kb_id}"

# Two knowledge bases now exist (this one and local-memory), so velox-hook's
# auto-resolution deliberately refuses to guess and disables itself. Passing
# VELOX_HOOK_KNOWLEDGE_BASE explicitly is required here -- and is the same fix
# a user needs the moment they create a second knowledge base of their own.
export VELOX_HOOK_BASE_URL="${base_url}"
export VELOX_HOOK_KNOWLEDGE_BASE="${kb_id}"

# A fixed synthetic cwd, distinct from any real project, so the channel this
# test writes to and searches is isolated from real memory.
fake_cwd="/tmp/velox-hook-acceptance"
session_id="${run_tag}"
prompt_id_record="prompt-record-$$-${RANDOM}"
prompt_id_poll="prompt-poll-$$-${RANDOM}"
question="What is the sentinel phrase used by the VeloxRAG agent-memory-hooks acceptance script?"
sentinel="veloxrag-hook-acceptance-sentinel-$$-${RANDOM}"
answer="This is a synthetic answer written by the VeloxRAG agent-memory-hooks acceptance script. It exists only to exercise the record and retrieve hooks end to end against the live local stack, and it deliberately runs well past the two-hundred character recording threshold so that velox-hook record treats it as worth indexing. The sentinel for this run is: ${sentinel}. That token is unique to this run and should not appear anywhere else in any corpus, so once a retrieval finds it we know the whole round trip through record and retrieve completed."

write_retrieve_payload() {
  local output="$1"
  local prompt_id="$2"

  SESSION_ID="${session_id}" PROMPT_ID="${prompt_id}" CWD="${fake_cwd}" QUESTION="${question}" \
    python3 - "${output}" <<'PY'
import json
import os
import sys

payload = {
    "session_id": os.environ["SESSION_ID"],
    "prompt_id": os.environ["PROMPT_ID"],
    "cwd": os.environ["CWD"],
    "user_input": os.environ["QUESTION"],
}
with open(sys.argv[1], "w", encoding="utf-8") as stream:
    json.dump(payload, stream)
PY
}

write_record_payload() {
  local output="$1"

  SESSION_ID="${session_id}" PROMPT_ID="${prompt_id_record}" CWD="${fake_cwd}" ANSWER="${answer}" \
    python3 - "${output}" <<'PY'
import json
import os
import sys

payload = {
    "session_id": os.environ["SESSION_ID"],
    "prompt_id": os.environ["PROMPT_ID"],
    "cwd": os.environ["CWD"],
    "last_assistant_message": os.environ["ANSWER"],
}
with open(sys.argv[1], "w", encoding="utf-8") as stream:
    json.dump(payload, stream)
PY
}

retrieve_initial_json="${work_dir}/retrieve-initial.json"
record_json="${work_dir}/record.json"
retrieve_poll_json="${work_dir}/retrieve-poll.json"

write_retrieve_payload "${retrieve_initial_json}" "${prompt_id_record}"
write_record_payload "${record_json}"
write_retrieve_payload "${retrieve_poll_json}" "${prompt_id_poll}"

# The knowledge base is empty at this point. A non-empty result here would
# mean the channel filter is not scoping the search the way it should.
initial_output=$("${hook_bin}" retrieve <"${retrieve_initial_json}")
if [ -n "${initial_output}" ]; then
  fail "retrieve on an empty, freshly-created knowledge base printed something -- the channel filter is not isolating the search"
fi

"${hook_bin}" record <"${record_json}"

SECONDS=0
found=0
last_retrieve_output=""
while [ "${SECONDS}" -lt "${deadline_seconds}" ]; do
  last_retrieve_output=$("${hook_bin}" retrieve <"${retrieve_poll_json}")
  if printf '%s' "${last_retrieve_output}" | grep -qF -- "${sentinel}"; then
    found=1
    break
  fi
  sleep 2
done
elapsed_seconds="${SECONDS}"

if [ "${found}" -ne 1 ]; then
  {
    printf 'the recorded turn never became retrievable within %s seconds\n' "${deadline_seconds}"
    printf 'last retrieval output:\n%s\n' "${last_retrieve_output:-<empty>}"
    printf 'a job stuck in retry_wait usually means Ollama is not reachable from the containers; try `veloxrag log worker`\n'
  } >&2
  exit 1
fi

printf 'retrievable after %s seconds\n' "${elapsed_seconds}"
printf '%s\n' "${last_retrieve_output}"
