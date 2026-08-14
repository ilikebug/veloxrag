from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from rag_service.dev.hooks import (
    channel_for,
    prune_stale_state,
    remember_prompt,
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
