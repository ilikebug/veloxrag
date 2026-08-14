# Recording sessions automatically

What this layer is for: the MCP server ([docs/mcp.md](mcp.md)) already gives the agent tools to
search its own memory, but nothing wrote to that memory, and searching it was left to the agent's
discretion — which it routinely declines to exercise. An instruction to search is not a search.
These two hooks close the loop: one writes to the memory on every turn, the other searches it on
every prompt, so retrieval stops depending on the model remembering to ask.

## Turning it on

Two entries in `~/.claude/settings.json`, at **user** scope rather than project scope, so every
project gets recorded rather than only the one where you happened to add the hook. Retrieval is
still scoped per project regardless of where the hooks are registered — see below.

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "uvx --from git+https://github.com/ilikebug/veloxrag velox-hook retrieve",
            "timeout": 10
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "uvx --from git+https://github.com/ilikebug/veloxrag velox-hook record",
            "timeout": 15
          }
        ]
      }
    ]
  }
}
```

`retrieve` runs on `UserPromptSubmit`: it reads the hook payload on stdin, stores the question in
a state file, searches the knowledge base scoped to the current project, and prints surviving
passages on stdout, which Claude Code adds to that turn's context. `record` runs on `Stop`: it
reads `last_assistant_message` from the payload, pairs it with the stored question by `prompt_id`,
and uploads the turn as one Markdown document.

Both are advisory. Every failure path — service down, Ollama down, a timeout, a non-200 response,
a malformed payload, no knowledge base resolvable — prints nothing and exits 0: the session is
exactly what it would have been without the hooks. `record` in particular never exits 2, because
on `Stop` that would prevent Claude from stopping and loop the session.

Retrieval is scoped to the current project by a `channel` metadata filter, alongside a
`source_type=chat` filter (so it never searches non-chat documents such as an ingested corpus).
`channel` is the working directory with every `/` and `.` replaced by `-`, which is the same
encoding the batch converter `velox-chat-transcripts` uses, and the same directory name Claude
Code assigns under `~/.claude/projects`. One consequence: a git worktree is its own working
directory and therefore its own channel, so a turn recorded inside a worktree is not retrieved by
default from the main checkout. To search across every project regardless of channel, ask the
agent to call the MCP tool `search_memory` directly — it applies no channel filter.

## What gets recorded, and what does not

One document per turn: the question and the answer only, nothing else. Not the reasoning, not
tool output — tool output is mostly the contents of files that are already indexable on their own,
and indexing it again would bury the reasoning under duplicates of the codebase.

A turn is skipped when:

- the prompt starts with `/` — a slash command is an instruction to the tool, not a question with
  an answer worth keeping;
- the answer is under 200 characters **after redaction** (see below) — most turns in a coding
  session are a command and an acknowledgement, and recording those would dominate the document
  count while carrying nothing retrievable.

Redaction runs unconditionally over both the question and the answer, reusing
`chat_transcripts.redact`. It replaces: PEM private keys; `Authorization: Bearer`/`Basic` headers;
bare bearer tokens; provider key shapes (`sk-`, `rk-`, `pk-`); GitHub tokens (`ghp_`, `gho_`,
`ghu_`, `ghs_`, `ghr_`, `github_pat_`); Slack tokens (`xox...`); AWS access key ids (`AKIA...`);
JWTs; `password=`/`token=`/`secret=`/`api_key=`-style assignments; and connection URLs carrying
credentials in the authority (`scheme://user:pass@host`). Length is measured on the redacted text
so an answer that is mostly credentials cannot clear the 200-character floor with characters that
are about to be removed.

Documents carry metadata inside the knowledge base's frozen filter schema: `source_type=chat`,
`doc_type=claude-code`, `section=turn`, `channel`, `thread_id` (the session id), `source_path`
(the working directory), `lang`, `occurred_at`. `section=turn` is what distinguishes these from
whole-session documents written by `velox-chat-transcripts`, which carry no `section` at all — the
field cannot be added to those later, since the schema is frozen.

Measured on this machine: upload to searchable takes 0 to 3 seconds, so a later turn in the same
session can retrieve an earlier one from earlier in that same conversation.

## Tuning it

All six variables are optional and read by both subcommands; an unparseable or out-of-range value
falls back to its default rather than failing, because a hook that refused to start would print a
configuration complaint on every prompt.

| Variable | Default | Meaning |
| --- | --- | --- |
| `VELOX_HOOK_BASE_URL` | `http://127.0.0.1:8000` | Where the service is |
| `VELOX_HOOK_KNOWLEDGE_BASE` | resolved when exactly one exists | Which knowledge base to use |
| `VELOX_HOOK_TOKEN` | empty | Agent token, when local trusted auth is off |
| `VELOX_HOOK_SCORE_FLOOR` | `0.5` | Passages scoring below this are not injected (range 0.0–1.0) |
| `VELOX_HOOK_MIN_ANSWER_CHARACTERS` | `200` | Recording threshold, after redaction (0 or more) |
| `VELOX_HOOK_TOP_K` | `5` | Passages requested per turn (range 1–50, the search endpoint's own range) |

**If the operator ever creates a second knowledge base, auto-resolution stops rather than
guessing, and both hooks go quiet.** Resolving "the" knowledge base only works when exactly one
exists; with several, guessing wrong means searching or writing to the wrong memory, so the hook
disables itself for that invocation instead. Nothing on screen announces this — `retrieve` stops
injecting passages and `record` stops uploading turns, and each writes exactly one line to
`~/.veloxrag/hook.log` per call: `retrieve found no single knowledge base; set
VELOX_HOOK_KNOWLEDGE_BASE` and `record found no single knowledge base; set
VELOX_HOOK_KNOWLEDGE_BASE`, so the log makes clear that both halves stopped, not just one. The fix
is to set `VELOX_HOOK_KNOWLEDGE_BASE` to the intended knowledge base id.

## Turning it off

Remove the two entries from `~/.claude/settings.json`. Recorded turns stay: individual documents
can never be deleted — `/v1/documents/*` is GET-only, and the only deletion the service offers is
of an entire knowledge base ([docs/mcp.md](mcp.md#a-typical-setup)). Turning the hooks off stops
new turns from being written; it does not remove the ones already there.

## Where to look when something seems wrong

- `~/.veloxrag/hook.log` — one line per failure, nothing on success. If a hook seems to be doing
  nothing, this is the first place to check, followed by the second-knowledge-base case above.
- `~/.veloxrag/hook-state/` — one file per session, named by a SHA-256 digest of the session id,
  mode `0600` because it holds the prompt verbatim and un-redacted (redaction happens on the way
  to the knowledge base, not on the way to this file). Entries survive 24 hours and are pruned at
  the start of the next `record`.
- `make acceptance-agent-memory-hooks` — proves the whole loop end to end against the running
  stack: records one turn, polls retrieval until it comes back, and reports the upload-to-
  searchable latency. It creates its own throwaway knowledge base and deletes it when it finishes,
  so it never writes into the real one.
