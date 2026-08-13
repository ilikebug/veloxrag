from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from rag_service.dev.chat_transcripts import (
    REDACTION_PLACEHOLDER,
    ChatTranscriptError,
    build_metadata,
    convert,
    read_claude_code,
    read_codex,
    redact,
    render_markdown,
)

# Values shaped like credentials but not real ones. Each case exists because a
# transcript genuinely produces that shape: minted tokens, printed connection
# strings, pasted keys.
CREDENTIAL_SHAPES = [
    "sk-abcdefghijklmnopqrstuvwxyz012345",
    "ghp_abcdefghijklmnopqrstuvwxyz01",
    "xoxb-1234567890-abcdefghij",
    "AKIAIOSFODNN7EXAMPLE",
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.c2lnbmF0dXJl",
    "Authorization: Bearer abcdefghijklmnopqrstuvwxyz",
    "password=hunter2hunter2",
    "api_key: abcdefghijklmnop",
    "postgresql://rag:supersecretpw@postgres:5432/rag",
    "-----BEGIN RSA PRIVATE KEY-----\nMIIBOgIBAAJBAKj3\n-----END RSA PRIVATE KEY-----",
]


@pytest.mark.parametrize("secret", CREDENTIAL_SHAPES)
def test_redact_removes_credential_shapes(secret: str) -> None:
    redacted = redact(f"the command was: {secret} and it worked")

    assert REDACTION_PLACEHOLDER in redacted
    # The distinctive part of the value must not survive anywhere in the output.
    distinctive = max(secret.replace("\n", " ").split(), key=len)
    assert distinctive not in redacted


def test_redact_keeps_surrounding_prose() -> None:
    redacted = redact("Minting an admin token with password=hunter2hunter2 then retrying")

    assert redacted.startswith("Minting an admin token with")
    assert redacted.endswith("then retrying")


def test_redact_leaves_ordinary_text_untouched() -> None:
    text = "The chunk default moved from 1200 to 600 codepoints and MRR rose to 0.836."

    assert redact(text) == text


def test_redact_rejects_non_text() -> None:
    with pytest.raises(ChatTranscriptError):
        redact(None)  # type: ignore[arg-type]


def _claude_line(role: str, blocks: list[dict[str, object]], stamp: str) -> str:
    return json.dumps(
        {"type": role, "timestamp": stamp, "message": {"role": role, "content": blocks}}
    )


def test_read_claude_code_keeps_only_human_visible_text(tmp_path: Path) -> None:
    path = tmp_path / "project" / "session-1.jsonl"
    path.parent.mkdir()
    path.write_text(
        "\n".join(
            (
                _claude_line(
                    "user",
                    [{"type": "text", "text": "why did retrieval regress"}],
                    "2026-08-01T10:00:00Z",
                ),
                _claude_line(
                    "assistant",
                    [
                        {"type": "thinking", "thinking": "maybe the chunker, maybe the embedder"},
                        {"type": "text", "text": "the chunk size diluted the embedding"},
                        {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}},
                    ],
                    "2026-08-01T10:01:00Z",
                ),
                _claude_line(
                    "user",
                    [{"type": "tool_result", "content": "a whole file listing"}],
                    "2026-08-01T10:02:00Z",
                ),
            )
        ),
        encoding="utf-8",
    )

    transcript = read_claude_code(path)

    assert [turn.speaker for turn in transcript.turns] == ["user", "assistant"]
    assert transcript.turns[1].text == "the chunk size diluted the embedding"
    assert transcript.project == "project"
    assert transcript.session_id == "session-1"
    assert transcript.occurred_at == datetime(2026, 8, 1, 10, 0, tzinfo=UTC)


def test_read_codex_skips_reasoning_tool_output_and_developer_turns(tmp_path: Path) -> None:
    """Codex names its prose blocks input_text/output_text, not text.

    The fixture uses the real names on purpose: an earlier version of this test
    used `text` for both readers, which passed while the converter silently
    extracted zero turns from every real Codex session.
    """
    path = tmp_path / "rollout-2026-08-01T10-00-00-abc.jsonl"
    path.write_text(
        "\n".join(
            (
                json.dumps({"type": "session_meta", "payload": {"cwd": "/Users/x/work/RAG"}}),
                json.dumps(
                    {
                        "type": "response_item",
                        "timestamp": "2026-08-01T10:00:00Z",
                        "payload": {
                            "type": "message",
                            "role": "developer",
                            "content": [{"type": "input_text", "text": "harness scaffolding"}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "response_item",
                        "timestamp": "2026-08-01T10:01:00Z",
                        "payload": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "add hybrid retrieval"}],
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {"type": "reasoning", "content": "long chain"},
                    }
                ),
                json.dumps(
                    {
                        "type": "response_item",
                        "payload": {"type": "function_call_output", "output": "file dump"},
                    }
                ),
                json.dumps(
                    {
                        "type": "response_item",
                        "timestamp": "2026-08-01T10:02:00Z",
                        "payload": {
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {"type": "output_text", "text": "BM25 hurt Chinese queries"}
                            ],
                        },
                    }
                ),
            )
        ),
        encoding="utf-8",
    )

    transcript = read_codex(path)

    assert [turn.text for turn in transcript.turns] == [
        "add hybrid retrieval",
        "BM25 hurt Chinese queries",
    ]
    assert transcript.project == "RAG"


def test_read_claude_code_survives_a_truncated_line(tmp_path: Path) -> None:
    path = tmp_path / "project" / "session-2.jsonl"
    path.parent.mkdir()
    path.write_text(
        _claude_line("user", [{"type": "text", "text": "first"}], "2026-08-01T10:00:00Z")
        + '\n{"type": "assistant", "message"\n'
        + _claude_line("assistant", [{"type": "text", "text": "second"}], "2026-08-01T10:01:00Z"),
        encoding="utf-8",
    )

    # A transcript is an append-only log that can be cut mid-write, so a partial
    # trailing line must not lose the turns before it.
    assert [turn.text for turn in read_claude_code(path).turns] == ["first", "second"]


def test_render_markdown_names_speaker_project_and_date_in_the_body(tmp_path: Path) -> None:
    path = tmp_path / "rag-work" / "session-3.jsonl"
    path.parent.mkdir()
    path.write_text(
        _claude_line(
            "user", [{"type": "text", "text": "why is recall low"}], "2026-08-01T10:00:00Z"
        ),
        encoding="utf-8",
    )

    body = render_markdown(read_claude_code(path))

    # The embedding never sees metadata columns, so a chunk of dialogue has to
    # carry its own identity or it cannot be told apart from any other chunk.
    assert "Turn 1 — User — 2026-08-01 10:00 — rag-work" in body
    assert body.startswith("# claude-code session in rag-work")


def test_render_markdown_redacts_turn_text(tmp_path: Path) -> None:
    path = tmp_path / "project" / "session-4.jsonl"
    path.parent.mkdir()
    path.write_text(
        _claude_line(
            "assistant",
            [{"type": "text", "text": "used ghp_abcdefghijklmnopqrstuvwxyz01 to push"}],
            "2026-08-01T10:00:00Z",
        ),
        encoding="utf-8",
    )

    body = render_markdown(read_claude_code(path))

    assert "ghp_abcdefghijklmnopqrstuvwxyz01" not in body
    assert REDACTION_PLACEHOLDER in body


def test_build_metadata_only_uses_frozen_filter_schema_fields(tmp_path: Path) -> None:
    path = tmp_path / "project" / "session-5.jsonl"
    path.parent.mkdir()
    path.write_text(
        _claude_line(
            "user", [{"type": "text", "text": "为什么中文检索更差"}], "2026-08-01T10:00:00Z"
        ),
        encoding="utf-8",
    )
    transcript = read_claude_code(path)

    metadata = build_metadata(transcript)

    # filter_schema is a one-way door once a generation exists, so a field the
    # schema does not declare would be silently unfilterable.
    allowed = {
        "source_type",
        "speaker",
        "channel",
        "thread_id",
        "doc_type",
        "section",
        "source_path",
        "occurred_at",
        "lang",
    }
    assert set(metadata) <= allowed
    assert metadata["source_type"] == "chat"
    assert metadata["doc_type"] == "claude-code"
    assert metadata["thread_id"] == "session-5"
    assert metadata["lang"] == "zh"


def test_convert_skips_sessions_below_the_turn_floor(tmp_path: Path) -> None:
    root = tmp_path / "projects" / "proj"
    root.mkdir(parents=True)
    (root / "short.jsonl").write_text(
        _claude_line("user", [{"type": "text", "text": "run the tests"}], "2026-08-01T10:00:00Z"),
        encoding="utf-8",
    )
    (root / "long.jsonl").write_text(
        "\n".join(
            _claude_line(
                "user" if index % 2 == 0 else "assistant",
                [{"type": "text", "text": f"turn {index}"}],
                "2026-08-01T10:00:00Z",
            )
            for index in range(6)
        ),
        encoding="utf-8",
    )

    written = convert(tmp_path / "projects", tmp_path / "out", source="claude-code")

    assert [path.name for path in written] == ["claude-code-long.md"]
    metadata = json.loads((tmp_path / "out" / "claude-code-long.metadata.json").read_text())
    assert metadata["channel"] == "proj"


def test_convert_filters_by_start_date(tmp_path: Path) -> None:
    root = tmp_path / "projects" / "proj"
    root.mkdir(parents=True)
    for name, stamp in (("old", "2026-01-01T10:00:00Z"), ("new", "2026-08-01T10:00:00Z")):
        (root / f"{name}.jsonl").write_text(
            "\n".join(
                _claude_line(
                    "user" if index % 2 == 0 else "assistant",
                    [{"type": "text", "text": f"turn {index}"}],
                    stamp,
                )
                for index in range(6)
            ),
            encoding="utf-8",
        )

    written = convert(
        tmp_path / "projects",
        tmp_path / "out",
        source="claude-code",
        since=datetime(2026, 6, 1, tzinfo=UTC),
    )

    assert [path.name for path in written] == ["claude-code-new.md"]


def test_convert_rejects_an_unknown_source(tmp_path: Path) -> None:
    with pytest.raises(ChatTranscriptError):
        convert(tmp_path, tmp_path / "out", source="cursor")


def test_render_markdown_demotes_headings_inside_a_turn(tmp_path: Path) -> None:
    path = tmp_path / "proj" / "session-6.jsonl"
    path.parent.mkdir()
    path.write_text(
        _claude_line(
            "user",
            [{"type": "text", "text": "# Executing Plans\n\n## Overview\n\nfollow the steps"}],
            "2026-08-01T10:00:00Z",
        ),
        encoding="utf-8",
    )

    body = render_markdown(read_claude_code(path))

    # An unmodified `#` would take over the outline, and every chunk after it
    # would inherit "Executing Plans" as its title path instead of the turn.
    assert "\n# Executing Plans" not in body
    assert "### Executing Plans" in body
    assert "#### Overview" in body
    headings = [line for line in body.split("\n") if line.startswith("# ")]
    assert headings == ["# claude-code session in proj"]


def test_render_markdown_leaves_shell_comments_in_fences_alone(tmp_path: Path) -> None:
    path = tmp_path / "proj" / "session-7.jsonl"
    path.parent.mkdir()
    path.write_text(
        _claude_line(
            "assistant",
            [{"type": "text", "text": "run it:\n\n```bash\n# rebuild first\nmake check\n```"}],
            "2026-08-01T10:00:00Z",
        ),
        encoding="utf-8",
    )

    body = render_markdown(read_claude_code(path))

    # Inside a fence `#` is a shell comment; rewriting it would corrupt a command
    # a later session might copy verbatim.
    assert "# rebuild first" in body
    assert "### rebuild first" not in body


def _session(path: Path, turns: list[tuple[str, str]], stamp: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            _claude_line(role, [{"type": "text", "text": text}], stamp) for role, text in turns
        ),
        encoding="utf-8",
    )


def test_convert_drops_turns_replayed_by_a_resumed_session(tmp_path: Path) -> None:
    root = tmp_path / "projects" / "proj"
    shared = [("user", f"turn {i}") for i in range(6)]
    _session(root / "first.jsonl", shared, "2026-08-01T10:00:00Z")
    # A resumed session replays the earlier exchange and adds to it.
    _session(
        root / "second.jsonl",
        shared + [("user", f"later {i}") for i in range(6)],
        "2026-08-02T10:00:00Z",
    )

    convert(tmp_path / "projects", tmp_path / "out", source="claude-code")

    first = (tmp_path / "out" / "claude-code-first.md").read_text()
    second = (tmp_path / "out" / "claude-code-second.md").read_text()
    assert "turn 0" in first
    # The replay is dropped from the later document rather than indexed twice,
    # which would otherwise spend a search's top-k on copies of one answer.
    assert "turn 0" not in second
    assert "later 0" in second


def test_convert_skips_a_session_that_is_entirely_a_replay(tmp_path: Path) -> None:
    root = tmp_path / "projects" / "proj"
    shared = [("user", f"turn {i}") for i in range(6)]
    _session(root / "original.jsonl", shared, "2026-08-01T10:00:00Z")
    _session(root / "replay.jsonl", shared, "2026-08-02T10:00:00Z")

    written = convert(tmp_path / "projects", tmp_path / "out", source="claude-code")

    assert [path.name for path in written] == ["claude-code-original.md"]


def test_convert_keeps_the_earliest_session_when_paths_sort_the_other_way(tmp_path: Path) -> None:
    root = tmp_path / "projects" / "proj"
    shared = [("user", f"turn {i}") for i in range(6)]
    # "aaa" sorts before "zzz" by path but happened later, so path order alone
    # would keep the replay and strip the original.
    _session(root / "zzz.jsonl", shared, "2026-08-01T10:00:00Z")
    _session(root / "aaa.jsonl", shared, "2026-08-02T10:00:00Z")

    written = convert(tmp_path / "projects", tmp_path / "out", source="claude-code")

    assert [path.name for path in written] == ["claude-code-zzz.md"]
