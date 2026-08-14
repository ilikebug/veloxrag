from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from rag_service.dev.hooks import (
    build_turn_metadata,
    channel_for,
    prune_stale_state,
    remember_prompt,
    render_injection,
    render_turn,
    should_record,
    state_directory,
    take_prompt,
)


def test_channel_matches_the_directory_name_claude_code_assigns(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The batch converter uses the ~/.claude/projects directory name as channel.
    # A different encoding here would split one project across two channels and
    # a filtered search would see only half of it.
    monkeypatch.setenv("HOME", str(tmp_path))

    assert channel_for("/Users/y.zhang/work/AI/VeloxRAG") == "-Users-y-zhang-work-AI-VeloxRAG"
    assert channel_for("/Users/y.zhang/.claude/skills") == "-Users-y-zhang--claude-skills"


def test_a_remembered_prompt_is_returned_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    remember_prompt("session-1", "prompt-1", "why is the edd missing")

    assert take_prompt("session-1", "prompt-1") == "why is the edd missing"
    # Taken, not read: leaving it behind would record the same turn again if Stop
    # ran twice for one prompt.
    assert take_prompt("session-1", "prompt-1") is None


def test_an_unknown_prompt_yields_nothing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    assert take_prompt("session-absent", "prompt-absent") is None


def test_state_older_than_the_ttl_is_pruned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    remember_prompt("session-old", "prompt-1", "question")
    stale = next(iter(state_directory().glob("*.json")))
    os.utime(stale, (0, 0))
    remember_prompt("session-fresh", "prompt-1", "question")

    prune_stale_state()

    assert not stale.exists()
    # A fresh entry has to survive, or this function would be indistinguishable
    # from one that deletes the directory's contents.
    assert take_prompt("session-fresh", "prompt-1") == "question"


def test_a_session_identifier_cannot_escape_the_state_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # session_id arrives from outside the process. Treated as a path component it
    # would let a crafted value write anywhere the user can write.
    monkeypatch.setenv("HOME", str(tmp_path))

    remember_prompt("../../escaped", "prompt-1", "question")

    written = list(state_directory().glob("*.json"))
    assert len(written) == 1
    assert "escaped" not in written[0].name
    assert take_prompt("../../escaped", "prompt-1") == "question"
    assert not (tmp_path / "escaped.json").exists()


def test_state_is_not_readable_by_other_users(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The state file holds the prompt verbatim; redaction only happens on the way
    # to the knowledge base.
    monkeypatch.setenv("HOME", str(tmp_path))

    remember_prompt("session-1", "prompt-1", "the password is hunter2")
    written = next(iter(state_directory().glob("*.json")))

    assert stat.S_IMODE(written.stat().st_mode) == 0o600

    # A second write to the same file is where a rewrite path that recreated the
    # file with the default umask, rather than reusing the atomic-write helper,
    # would silently loosen the permissions back up.
    remember_prompt("session-1", "prompt-2", "another question")

    assert stat.S_IMODE(written.stat().st_mode) == 0o600


_LONG_ANSWER = "The knowledge base refuses a second generation. " * 8


def test_a_turn_that_concluded_something_is_recorded() -> None:
    assert should_record("why does ingest refuse", _LONG_ANSWER) is True


def test_an_acknowledgement_is_not_recorded() -> None:
    # A command and an "ok" dominate a coding session by count and carry no
    # retrievable conclusion. The batch converter drops short sessions for the
    # same reason; recording per turn loses that filter and needs its own.
    assert should_record("run the tests", "Done, all 412 passed.") is False


def test_a_slash_command_is_not_recorded() -> None:
    assert should_record("/clear", _LONG_ANSWER) is False


def test_the_threshold_counts_the_redacted_text() -> None:
    # Redaction shortens the answer, so a reply that is mostly a leaked token
    # must not buy its way past the floor with characters that get removed.
    answer = "here is the key: " + "sk-" + "a" * 300

    assert should_record("what is the key", answer) is False


def test_a_recorded_turn_keeps_both_halves_and_redacts_them() -> None:
    body = render_turn(
        user_input="why does it fail with Authorization: Bearer abcdefghijklmnopqr",
        assistant_text=_LONG_ANSWER,
        channel="-tmp-project",
        session_id="session-1",
        occurred_at="2026-08-14T09:00:00+00:00",
    )

    assert "abcdefghijklmnopqr" not in body
    assert "[REDACTED]" in body
    # Scoped to the sections rather than to the whole body: a render that swapped
    # the two halves would satisfy "both are present somewhere".
    question = body.split("## Question", 1)[1].split("## Answer", 1)[0]
    answer = body.split("## Answer", 1)[1]
    assert "why does it fail" in question
    assert "The knowledge base refuses a second generation." in answer
    assert "The knowledge base refuses a second generation." not in question


def test_headings_inside_an_answer_do_not_break_the_document() -> None:
    # An answer routinely contains Markdown headings. Left alone they would sit
    # at the same level as the document's own structure, and the chunker's
    # title_path would attribute passages to a heading the answer invented.
    body = render_turn(
        user_input="show me the layout",
        assistant_text="## Overview\n\n" + _LONG_ANSWER,
        channel="-tmp-project",
        session_id="session-1",
        occurred_at="2026-08-14T09:00:00+00:00",
    )

    assert "\n## Overview" not in body
    assert "#### Overview" in body


def test_metadata_uses_only_declared_filter_fields() -> None:
    # The filter schema is frozen by the first generation. A field outside it
    # produces metadata that is silently unfilterable.
    metadata = build_turn_metadata(
        user_input="why does it fail",
        assistant_text=_LONG_ANSWER,
        channel="-tmp-project",
        cwd="/tmp/project",
        session_id="session-1",
        occurred_at="2026-08-14T09:00:00+00:00",
    )

    assert metadata == {
        "source_type": "chat",
        "doc_type": "claude-code",
        "section": "turn",
        "channel": "-tmp-project",
        "thread_id": "session-1",
        "source_path": "/tmp/project",
        "lang": "en",
        "occurred_at": "2026-08-14T09:00:00+00:00",
    }


_PASSAGE = {
    "text": "the schedule is per carrier, not per slug",
    "score": 0.72,
    "document_id": "11111111-2222-4333-8444-555555555555",
    "metadata": {"channel": "-tmp-project", "occurred_at": "2026-08-13T14:22:00+00:00"},
    "source": {"start_offset": 1200, "end_offset": 1800},
}


def test_passages_are_framed_as_history_rather_than_instruction() -> None:
    # The passages are things the operator said. Injected without framing, a
    # request from months ago reads like a request for this turn.
    rendered = render_injection([_PASSAGE], floor=0.5)

    assert rendered.startswith("<retrieved-memory>")
    assert rendered.rstrip().endswith("</retrieved-memory>")
    assert "not an instruction for this turn" in rendered
    assert "may be stale" in rendered


def test_a_passage_carries_what_is_needed_to_widen_it() -> None:
    rendered = render_injection([_PASSAGE], floor=0.5)

    assert "score=0.72" in rendered
    assert "channel=-tmp-project" in rendered
    assert "occurred_at=2026-08-13T14:22:00+00:00" in rendered
    assert "document_id=11111111-2222-4333-8444-555555555555" in rendered
    assert "offsets=1200-1800" in rendered
    assert "the schedule is per carrier, not per slug" in rendered


def test_passages_below_the_floor_are_dropped() -> None:
    weak = {**_PASSAGE, "score": 0.31}

    assert render_injection([weak], floor=0.5) == ""


def test_nothing_is_printed_when_everything_is_filtered_out() -> None:
    # An empty wrapper would still cost tokens and still suggest the memory was
    # consulted and had nothing, which is not what "no passage cleared the floor"
    # means.
    assert render_injection([], floor=0.5) == ""


def test_a_passage_without_a_score_is_dropped() -> None:
    assert render_injection([{**_PASSAGE, "score": None}], floor=0.5) == ""


def test_surviving_passages_are_numbered_in_order() -> None:
    # The numbering has to follow the surviving passages, not the input list, or
    # a dropped passage would leave a gap the model reads as a missing citation.
    second = {**_PASSAGE, "score": 0.9, "text": "the second passage"}
    dropped = {**_PASSAGE, "score": 0.1, "text": "the dropped passage"}

    rendered = render_injection([_PASSAGE, dropped, second], floor=0.5)

    assert "[1] score=0.72" in rendered
    assert "[2] score=0.90" in rendered
    assert "the dropped passage" not in rendered
    assert rendered.index("[1]") < rendered.index("[2]")
