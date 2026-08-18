from __future__ import annotations

import io
import json
import os
import stat
import sys
import tomllib
from pathlib import Path

import httpx
import pytest

from rag_service.dev.hooks import (
    HookSettings,
    build_client,
    build_turn_metadata,
    channel_for,
    main,
    prune_stale_state,
    remember_prompt,
    render_injection,
    render_turn,
    resolve_knowledge_base,
    should_record,
    state_directory,
    take_prompt,
)

_KNOWLEDGE_BASE_ID = "1d196c08-303a-4d5c-89b0-235dfe8fc8fc"


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


def test_settings_need_no_environment_at_all() -> None:
    settings = HookSettings({})

    assert settings.base_url == "http://127.0.0.1:8000"
    assert settings.knowledge_base_id is None
    assert settings.score_floor == 0.5
    assert settings.minimum_answer_characters == 200
    assert settings.top_k == 5


def test_settings_read_the_tunable_values() -> None:
    settings = HookSettings(
        {
            "VELOX_HOOK_BASE_URL": "http://127.0.0.1:9000/",
            "VELOX_HOOK_KNOWLEDGE_BASE": _KNOWLEDGE_BASE_ID,
            "VELOX_HOOK_SCORE_FLOOR": "0.62",
            "VELOX_HOOK_MIN_ANSWER_CHARACTERS": "80",
            "VELOX_HOOK_TOP_K": "8",
        }
    )

    assert settings.base_url == "http://127.0.0.1:9000"
    assert settings.knowledge_base_id == _KNOWLEDGE_BASE_ID
    assert settings.score_floor == 0.62
    assert settings.minimum_answer_characters == 80
    assert settings.top_k == 8


def test_an_unusable_value_falls_back_instead_of_failing() -> None:
    # A hook that refused to start over a bad float would take the session's
    # memory away and say so on every prompt. The default is the safer answer.
    settings = HookSettings({"VELOX_HOOK_SCORE_FLOOR": "high", "VELOX_HOOK_TOP_K": ""})

    assert settings.score_floor == 0.5
    assert settings.top_k == 5


def test_a_base_url_that_is_not_http_falls_back() -> None:
    settings = HookSettings({"VELOX_HOOK_BASE_URL": "ftp://elsewhere"})

    assert settings.base_url == "http://127.0.0.1:8000"


def test_a_value_outside_its_usable_range_falls_back() -> None:
    # A parseable but unusable value is the same problem as an unparseable one: a
    # typo should cost the default, never a silently dead hook. top_k=0 would make
    # every search a 422, and retrieve fails silently by design.
    settings = HookSettings(
        {
            "VELOX_HOOK_TOP_K": "0",
            "VELOX_HOOK_MIN_ANSWER_CHARACTERS": "-5",
            "VELOX_HOOK_SCORE_FLOOR": "1.4",
        }
    )

    assert settings.top_k == 5
    assert settings.minimum_answer_characters == 200
    assert settings.score_floor == 0.5


def test_a_top_k_above_the_search_endpoints_range_falls_back() -> None:
    # 50 is the endpoint's own documented ceiling; one past it would be rejected.
    settings = HookSettings({"VELOX_HOOK_TOP_K": "51"})

    assert settings.top_k == 5


def test_boundary_values_are_accepted_rather_than_treated_as_out_of_range() -> None:
    # The bounds themselves are legitimate settings, not values to reject.
    low = HookSettings(
        {
            "VELOX_HOOK_TOP_K": "1",
            "VELOX_HOOK_MIN_ANSWER_CHARACTERS": "0",
            "VELOX_HOOK_SCORE_FLOOR": "0.0",
        }
    )
    high = HookSettings({"VELOX_HOOK_TOP_K": "50", "VELOX_HOOK_SCORE_FLOOR": "1.0"})

    assert low.top_k == 1
    assert low.minimum_answer_characters == 0
    assert low.score_floor == 0.0
    assert high.top_k == 50
    assert high.score_floor == 1.0


def test_a_single_knowledge_base_is_resolved_without_configuration() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"items": [{"id": _KNOWLEDGE_BASE_ID}]})

    client = httpx.Client(base_url="http://127.0.0.1:8000", transport=httpx.MockTransport(handle))

    assert resolve_knowledge_base(HookSettings({}), client) == _KNOWLEDGE_BASE_ID


def test_several_knowledge_bases_are_not_guessed_between() -> None:
    # Guessing would silently search the wrong memory. The operator names one
    # with VELOX_HOOK_KNOWLEDGE_BASE.
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"items": [{"id": _KNOWLEDGE_BASE_ID}, {"id": "other"}]})

    client = httpx.Client(base_url="http://127.0.0.1:8000", transport=httpx.MockTransport(handle))

    assert resolve_knowledge_base(HookSettings({}), client) is None


def test_a_configured_knowledge_base_is_not_looked_up() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be made")

    client = httpx.Client(base_url="http://127.0.0.1:8000", transport=httpx.MockTransport(handle))
    settings = HookSettings({"VELOX_HOOK_KNOWLEDGE_BASE": _KNOWLEDGE_BASE_ID})

    assert resolve_knowledge_base(settings, client) == _KNOWLEDGE_BASE_ID


def test_an_error_listing_knowledge_bases_resolves_to_nothing() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "not ready"})

    client = httpx.Client(base_url="http://127.0.0.1:8000", transport=httpx.MockTransport(handle))

    assert resolve_knowledge_base(HookSettings({}), client) is None


def test_no_authorization_header_is_sent_when_no_token_is_configured() -> None:
    # The service's local-trusted mode only covers requests that carry no
    # credential at all; an empty bearer would be rejected rather than falling
    # through to it.
    client = build_client(HookSettings({}))

    assert "authorization" not in {name.lower() for name in client.headers}


def test_a_configured_token_is_sent_as_a_bearer() -> None:
    client = build_client(HookSettings({"VELOX_HOOK_TOKEN": "agent-token-sentinel"}))

    assert client.headers["authorization"] == "Bearer agent-token-sentinel"


class _Recorder:
    def __init__(self, *responses: httpx.Response) -> None:
        self.responses = list(responses)
        self.requests: list[httpx.Request] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self.responses:
            return httpx.Response(200, json={"results": []})
        return self.responses.pop(0)


def _install(monkeypatch: pytest.MonkeyPatch, recorder: _Recorder) -> None:
    monkeypatch.setattr(
        "rag_service.dev.hooks.build_client",
        lambda settings: httpx.Client(
            base_url=settings.base_url, transport=httpx.MockTransport(recorder.handle)
        ),
    )


def _run(
    monkeypatch: pytest.MonkeyPatch,
    subcommand: str,
    payload: dict[str, object],
    capsys: pytest.CaptureFixture[str],
) -> tuple[int, str]:
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    code = main([subcommand])
    return code, capsys.readouterr().out


_RETRIEVE_PAYLOAD: dict[str, object] = {
    "session_id": "session-1",
    "prompt_id": "prompt-1",
    "cwd": "/tmp/project",
    "prompt": "why is the schedule per carrier",
}


def test_retrieve_prints_the_passages_it_finds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VELOX_HOOK_KNOWLEDGE_BASE", _KNOWLEDGE_BASE_ID)
    recorder = _Recorder(httpx.Response(200, json={"results": [_PASSAGE]}))
    _install(monkeypatch, recorder)

    code, out = _run(monkeypatch, "retrieve", _RETRIEVE_PAYLOAD, capsys)

    assert code == 0
    assert "<retrieved-memory>" in out
    assert "the schedule is per carrier, not per slug" in out


def test_retrieve_scopes_the_search_to_the_current_project(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VELOX_HOOK_KNOWLEDGE_BASE", _KNOWLEDGE_BASE_ID)
    recorder = _Recorder(httpx.Response(200, json={"results": []}))
    _install(monkeypatch, recorder)

    _run(monkeypatch, "retrieve", _RETRIEVE_PAYLOAD, capsys)

    sent = json.loads(recorder.requests[0].content)
    assert sent["query"] == "why is the schedule per carrier"
    assert sent["top_k"] == 5
    assert sent["filters"] == {"metadata": {"channel": "-tmp-project", "source_type": "chat"}}


def test_retrieve_stores_the_question_before_it_searches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Order matters: a service that is down must not also cost the recording of
    # the turn, so the state write cannot be behind the HTTP call.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VELOX_HOOK_KNOWLEDGE_BASE", _KNOWLEDGE_BASE_ID)

    def explode(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("service down")

    monkeypatch.setattr(
        "rag_service.dev.hooks.build_client",
        lambda settings: httpx.Client(
            base_url=settings.base_url, transport=httpx.MockTransport(explode)
        ),
    )

    code, out = _run(monkeypatch, "retrieve", _RETRIEVE_PAYLOAD, capsys)

    assert code == 0
    assert out == ""
    assert take_prompt("session-1", "prompt-1") == "why is the schedule per carrier"


def test_retrieve_says_nothing_when_the_service_answers_with_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VELOX_HOOK_KNOWLEDGE_BASE", _KNOWLEDGE_BASE_ID)
    _install(monkeypatch, _Recorder(httpx.Response(503, json={"detail": "not ready"})))

    code, out = _run(monkeypatch, "retrieve", _RETRIEVE_PAYLOAD, capsys)

    assert code == 0
    assert out == ""


def test_retrieve_skips_a_slash_command_without_calling_the_service(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    recorder = _Recorder()
    _install(monkeypatch, recorder)

    code, out = _run(monkeypatch, "retrieve", {**_RETRIEVE_PAYLOAD, "prompt": "/clear"}, capsys)

    assert code == 0
    assert out == ""
    assert recorder.requests == []


def test_retrieve_truncates_a_question_past_the_api_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VELOX_HOOK_KNOWLEDGE_BASE", _KNOWLEDGE_BASE_ID)
    recorder = _Recorder(httpx.Response(200, json={"results": []}))
    _install(monkeypatch, recorder)

    _run(monkeypatch, "retrieve", {**_RETRIEVE_PAYLOAD, "prompt": "q" * 9000}, capsys)

    assert len(json.loads(recorder.requests[0].content)["query"]) == 8000


def test_retrieve_survives_a_payload_that_is_not_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("sys.stdin", io.StringIO("not json"))

    assert main(["retrieve"]) == 0
    assert capsys.readouterr().out == ""


def test_retrieve_says_nothing_when_no_knowledge_base_can_be_resolved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Two knowledge bases and none named: searching the wrong memory is worse
    # than searching none.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("VELOX_HOOK_KNOWLEDGE_BASE", raising=False)
    recorder = _Recorder(
        httpx.Response(200, json={"items": [{"id": _KNOWLEDGE_BASE_ID}, {"id": "other"}]})
    )
    _install(monkeypatch, recorder)

    code, out = _run(monkeypatch, "retrieve", _RETRIEVE_PAYLOAD, capsys)

    assert code == 0
    assert out == ""


def test_retrieve_logs_when_no_knowledge_base_can_be_resolved(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Retrieval dying silently while record logs the same condition would let a
    # reader of the log fix half the problem.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("VELOX_HOOK_KNOWLEDGE_BASE", raising=False)
    _install(
        monkeypatch,
        _Recorder(
            httpx.Response(200, json={"items": [{"id": _KNOWLEDGE_BASE_ID}, {"id": "other"}]})
        ),
    )

    _run(monkeypatch, "retrieve", _RETRIEVE_PAYLOAD, capsys)

    logged = (tmp_path / ".veloxrag" / "hook.log").read_text(encoding="utf-8")
    assert "VELOX_HOOK_KNOWLEDGE_BASE" in logged


def test_an_unknown_subcommand_is_logged_and_survived(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("sys.stdin", io.StringIO("{}"))

    assert main(["sing"]) == 0
    assert "sing" in (tmp_path / ".veloxrag" / "hook.log").read_text(encoding="utf-8")


_RECORD_PAYLOAD: dict[str, object] = {
    "session_id": "session-1",
    "prompt_id": "prompt-1",
    "cwd": "/tmp/project",
    "last_assistant_message": _LONG_ANSWER,
}


def test_record_uploads_the_paired_turn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VELOX_HOOK_KNOWLEDGE_BASE", _KNOWLEDGE_BASE_ID)
    remember_prompt("session-1", "prompt-1", "why is the schedule per carrier")
    recorder = _Recorder(httpx.Response(202, json={"job_id": "job-1"}))
    _install(monkeypatch, recorder)

    code, out = _run(monkeypatch, "record", _RECORD_PAYLOAD, capsys)

    assert code == 0
    # Stop hook stdout is not context. Printing here would be noise at best.
    assert out == ""
    request = recorder.requests[0]
    assert request.url.path == f"/v1/knowledge-bases/{_KNOWLEDGE_BASE_ID}/documents"
    body = request.content.decode("utf-8")
    assert "why is the schedule per carrier" in body
    assert "The knowledge base refuses a second generation." in body
    assert '"section": "turn"' in body or '"section":"turn"' in body
    assert "claude-code-session-1-prompt-1.md" in body


def test_record_consumes_the_state_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VELOX_HOOK_KNOWLEDGE_BASE", _KNOWLEDGE_BASE_ID)
    remember_prompt("session-1", "prompt-1", "why is the schedule per carrier")
    _install(monkeypatch, _Recorder(httpx.Response(202, json={"job_id": "job-1"})))

    _run(monkeypatch, "record", _RECORD_PAYLOAD, capsys)

    assert take_prompt("session-1", "prompt-1") is None


def test_record_does_nothing_when_the_prompt_was_never_stored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VELOX_HOOK_KNOWLEDGE_BASE", _KNOWLEDGE_BASE_ID)
    recorder = _Recorder()
    _install(monkeypatch, recorder)

    code, _ = _run(monkeypatch, "record", _RECORD_PAYLOAD, capsys)

    assert code == 0
    assert recorder.requests == []


def test_record_drops_a_turn_that_concluded_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VELOX_HOOK_KNOWLEDGE_BASE", _KNOWLEDGE_BASE_ID)
    remember_prompt("session-1", "prompt-1", "run the tests")
    recorder = _Recorder()
    _install(monkeypatch, recorder)

    code, _ = _run(
        monkeypatch, "record", {**_RECORD_PAYLOAD, "last_assistant_message": "Done."}, capsys
    )

    assert code == 0
    assert recorder.requests == []


def test_a_rejected_turn_still_consumes_its_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # take_prompt runs before the threshold check on purpose: a turn that is not
    # worth recording is finished with, and leaving its question behind would
    # keep it around until the pruner collects it a day later.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VELOX_HOOK_KNOWLEDGE_BASE", _KNOWLEDGE_BASE_ID)
    remember_prompt("session-1", "prompt-1", "run the tests")
    _install(monkeypatch, _Recorder())

    _run(monkeypatch, "record", {**_RECORD_PAYLOAD, "last_assistant_message": "Done."}, capsys)

    assert take_prompt("session-1", "prompt-1") is None


def test_record_treats_a_duplicate_as_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Asking the same question twice and getting the same answer is normal.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VELOX_HOOK_KNOWLEDGE_BASE", _KNOWLEDGE_BASE_ID)
    remember_prompt("session-1", "prompt-1", "why is the schedule per carrier")
    _install(
        monkeypatch,
        _Recorder(httpx.Response(409, json={"error": {"code": "DUPLICATE_DOCUMENT"}})),
    )

    code, _ = _run(monkeypatch, "record", _RECORD_PAYLOAD, capsys)

    assert code == 0
    assert not (tmp_path / ".veloxrag" / "hook.log").exists()


def test_record_never_blocks_the_stop_event(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Exit 2 on Stop prevents Claude from stopping and loops the session. Every
    # failure has to come back as 0, and leave a line behind instead.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VELOX_HOOK_KNOWLEDGE_BASE", _KNOWLEDGE_BASE_ID)
    remember_prompt("session-1", "prompt-1", "why is the schedule per carrier")
    _install(monkeypatch, _Recorder(httpx.Response(503, json={"detail": "not ready"})))

    code, _ = _run(monkeypatch, "record", _RECORD_PAYLOAD, capsys)

    assert code == 0
    logged = (tmp_path / ".veloxrag" / "hook.log").read_text(encoding="utf-8")
    assert "503" in logged


def test_record_survives_the_service_being_unreachable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VELOX_HOOK_KNOWLEDGE_BASE", _KNOWLEDGE_BASE_ID)
    remember_prompt("session-1", "prompt-1", "why is the schedule per carrier")

    def explode(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("service down")

    monkeypatch.setattr(
        "rag_service.dev.hooks.build_client",
        lambda settings: httpx.Client(
            base_url=settings.base_url, transport=httpx.MockTransport(explode)
        ),
    )

    code, out = _run(monkeypatch, "record", _RECORD_PAYLOAD, capsys)

    assert code == 0
    assert out == ""


def test_record_sends_the_metadata_as_declared_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The filter schema is frozen; a field outside it would be silently
    # unfilterable, so what actually goes over the wire is worth asserting.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VELOX_HOOK_KNOWLEDGE_BASE", _KNOWLEDGE_BASE_ID)
    remember_prompt("session-1", "prompt-1", "why is the schedule per carrier")
    recorder = _Recorder(httpx.Response(202, json={"job_id": "job-1"}))
    _install(monkeypatch, recorder)

    _run(monkeypatch, "record", _RECORD_PAYLOAD, capsys)

    body = recorder.requests[0].content.decode("utf-8")
    for field in ("source_type", "doc_type", "section", "channel", "thread_id"):
        assert field in body
    assert "-tmp-project" in body


def _settings_path(tmp_path: Path) -> Path:
    return tmp_path / ".claude" / "settings.json"


def _fake_executable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Stand in for the running velox-hook, whose path install() records.

    Under pytest, argv[0] is pytest's own binary, so a test that does not do
    this asserts on whatever happens to be running it.
    """
    executable = tmp_path / "velox-hook"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [str(executable), "install"])
    return executable


def test_install_into_a_missing_file_creates_it_with_both_events(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _fake_executable(monkeypatch, tmp_path)
    settings_path = _settings_path(tmp_path)
    assert not settings_path.exists()

    code = main(["install"])

    assert code == 0
    written = json.loads(settings_path.read_text(encoding="utf-8"))
    retrieve_command = written["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"]
    record_command = written["hooks"]["Stop"][0]["hooks"][0]["command"]
    assert retrieve_command.endswith("velox-hook retrieve")
    assert record_command.endswith("velox-hook record")
    assert written["hooks"]["UserPromptSubmit"][0]["hooks"][0]["timeout"] == 10
    assert written["hooks"]["Stop"][0]["hooks"][0]["timeout"] == 15
    assert "created" in capsys.readouterr().out


def test_install_preserves_unrelated_content_exactly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Asserted on specific keys, not just on length: a merge that dropped and
    # replaced a key with something the same size would still pass a length
    # check, but destroy the operator's permissions or status line.
    monkeypatch.setenv("HOME", str(tmp_path))
    settings_path = _settings_path(tmp_path)
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps(
            {
                "permissions": {"allow": ["Bash(ls:*)"]},
                "statusLine": {"type": "command", "command": "echo hi"},
            }
        ),
        encoding="utf-8",
    )

    code = main(["install"])

    assert code == 0
    written = json.loads(settings_path.read_text(encoding="utf-8"))
    assert written["permissions"] == {"allow": ["Bash(ls:*)"]}
    assert written["statusLine"] == {"type": "command", "command": "echo hi"}
    assert "hooks" in written


def test_install_twice_yields_exactly_one_entry_per_event(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _fake_executable(monkeypatch, tmp_path)
    settings_path = _settings_path(tmp_path)

    main(["install"])
    code = main(["install"])

    assert code == 0
    written = json.loads(settings_path.read_text(encoding="utf-8"))
    assert len(written["hooks"]["UserPromptSubmit"]) == 1
    assert len(written["hooks"]["UserPromptSubmit"][0]["hooks"]) == 1
    assert len(written["hooks"]["Stop"]) == 1
    assert len(written["hooks"]["Stop"][0]["hooks"]) == 1


def test_installing_from_a_different_executable_replaces_rather_than_appends(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Moving from a worktree's executable to an installed one is the expected
    # migration, and it has to leave one entry behind rather than two, or the
    # turn is recorded twice and the stale path fails on every prompt.
    monkeypatch.setenv("HOME", str(tmp_path))
    settings_path = _settings_path(tmp_path)
    first = tmp_path / "velox-hook-worktree"
    second = tmp_path / "velox-hook-installed"
    for executable in (first, second):
        executable.write_text("#!/bin/sh\n", encoding="utf-8")

    monkeypatch.setattr(sys, "argv", [str(first), "install"])
    main(["install"])
    monkeypatch.setattr(sys, "argv", [str(second), "install"])
    code = main(["install"])

    assert code == 0
    written = json.loads(settings_path.read_text(encoding="utf-8"))
    assert len(written["hooks"]["UserPromptSubmit"]) == 1
    assert len(written["hooks"]["Stop"]) == 1
    assert written["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"] == f"{second} retrieve"
    assert written["hooks"]["Stop"][0]["hooks"][0]["command"] == f"{second} record"


def test_install_records_the_executable_without_resolving_symlinks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # `uv tool install` publishes ~/.local/bin/velox-hook as a link into a store
    # directory that moves when the tool is upgraded. The link is the stable
    # name of the two, so it is what belongs in settings.json.
    monkeypatch.setenv("HOME", str(tmp_path))
    settings_path = _settings_path(tmp_path)
    store = tmp_path / "store" / "velox-hook"
    store.parent.mkdir(parents=True)
    store.write_text("#!/bin/sh\n", encoding="utf-8")
    link = tmp_path / "velox-hook"
    link.symlink_to(store)
    monkeypatch.setattr(sys, "argv", [str(link), "install"])

    code = main(["install"])

    assert code == 0
    written = json.loads(settings_path.read_text(encoding="utf-8"))
    command = written["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"]
    assert command == f"{link} retrieve"
    assert str(store) not in command


def test_install_refuses_an_unparseable_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    settings_path = _settings_path(tmp_path)
    settings_path.parent.mkdir(parents=True)
    original = b'{"permissions": {"allow": ["Bash(ls:*)"]}'  # missing closing brace
    settings_path.write_bytes(original)

    code = main(["install"])

    assert code != 0
    assert settings_path.read_bytes() == original
    assert capsys.readouterr().err != ""


def test_install_preserves_an_existing_files_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    settings_path = _settings_path(tmp_path)
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text("{}", encoding="utf-8")
    settings_path.chmod(0o640)

    code = main(["install"])

    assert code == 0
    assert stat.S_IMODE(settings_path.stat().st_mode) == 0o640


def test_uninstall_removes_only_velox_hook_entries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _fake_executable(monkeypatch, tmp_path)
    settings_path = _settings_path(tmp_path)
    settings_path.parent.mkdir(parents=True)
    settings_path.write_text(
        json.dumps(
            {
                "permissions": {"allow": ["Bash(ls:*)"]},
                "hooks": {"Stop": [{"hooks": [{"type": "command", "command": "echo other"}]}]},
            }
        ),
        encoding="utf-8",
    )
    main(["install"])

    code = main(["uninstall"])

    assert code == 0
    written = json.loads(settings_path.read_text(encoding="utf-8"))
    assert written["permissions"] == {"allow": ["Bash(ls:*)"]}
    # The foreign Stop hook survives in the same event a velox-hook entry was
    # also registered in.
    assert written["hooks"]["Stop"] == [{"hooks": [{"type": "command", "command": "echo other"}]}]
    assert "UserPromptSubmit" not in written["hooks"]
    assert "removed" in capsys.readouterr().out


def test_uninstall_with_nothing_to_remove_says_so_and_leaves_the_file_valid(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    settings_path = _settings_path(tmp_path)
    settings_path.parent.mkdir(parents=True)
    original = json.dumps({"permissions": {"allow": ["Bash(ls:*)"]}}).encode("utf-8")
    settings_path.write_bytes(original)

    code = main(["uninstall"])

    assert code == 0
    assert settings_path.read_bytes() == original
    assert "nothing to remove" in capsys.readouterr().out


def test_install_codex_writes_the_same_shape_into_the_codex_hook_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _fake_executable(monkeypatch, tmp_path)
    codex_path = tmp_path / ".codex" / "hooks.json"

    code = main(["install", "--codex"])

    assert code == 0
    assert not _settings_path(tmp_path).exists()
    written = json.loads(codex_path.read_text(encoding="utf-8"))
    assert written["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"].endswith("retrieve")
    assert written["hooks"]["Stop"][0]["hooks"][0]["command"].endswith("record")
    # The trust gate is silent when it bites, so installing must name it.
    assert "trusts hook commands by hash" in capsys.readouterr().out


def test_install_codex_preserves_hooks_another_tool_registered(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _fake_executable(monkeypatch, tmp_path)
    codex_path = tmp_path / ".codex" / "hooks.json"
    codex_path.parent.mkdir(parents=True)
    codex_path.write_text(
        json.dumps({"hooks": {"UserPromptSubmit": [{"hooks": [{"command": "other-tool"}]}]}}),
        encoding="utf-8",
    )

    assert main(["install", "--codex"]) == 0

    written = json.loads(codex_path.read_text(encoding="utf-8"))
    commands = [h["command"] for e in written["hooks"]["UserPromptSubmit"] for h in e["hooks"]]
    assert "other-tool" in commands
    assert any(c.endswith("retrieve") for c in commands)
    capsys.readouterr()


def test_uninstall_codex_touches_only_the_codex_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _fake_executable(monkeypatch, tmp_path)
    assert main(["install"]) == 0
    assert main(["install", "--codex"]) == 0

    assert main(["uninstall", "--codex"]) == 0

    codex = json.loads((tmp_path / ".codex" / "hooks.json").read_text(encoding="utf-8"))
    claude = json.loads(_settings_path(tmp_path).read_text(encoding="utf-8"))
    assert "hooks" not in codex
    assert claude["hooks"]["Stop"][0]["hooks"][0]["command"].endswith("record")
    capsys.readouterr()


def test_install_rejects_an_unknown_agent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    _fake_executable(monkeypatch, tmp_path)

    code = main(["install", "--cursor"])

    assert code == 1
    assert "unknown argument: --cursor" in capsys.readouterr().err
    assert not _settings_path(tmp_path).exists()


def test_a_turn_is_paired_by_turn_id_when_there_is_no_prompt_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Codex sends turn_id and has no prompt_id; the pairing must survive that,
    # or every turn in Codex is dropped exactly the way it was in Claude Code.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VELOX_HOOK_KNOWLEDGE_BASE", _KNOWLEDGE_BASE_ID)
    recorder = _Recorder(httpx.Response(200, json={"results": []}), httpx.Response(202, json={}))
    _install(monkeypatch, recorder)
    codex_prompt: dict[str, object] = {
        "session_id": "codex-session",
        "turn_id": "turn-7",
        "cwd": "/tmp/project",
        "prompt": "why is the schedule per carrier",
    }

    _run(monkeypatch, "retrieve", codex_prompt, capsys)
    _run(
        monkeypatch,
        "record",
        {**codex_prompt, "last_assistant_message": "b" * 400},
        capsys,
    )

    uploaded = recorder.requests[-1]
    assert "documents" in str(uploaded.url)
    assert "codex-session-turn-7" in uploaded.content.decode("utf-8", "replace")


def _write_transcript_pair(directory: Path, stem: str, *, with_metadata: bool = True) -> None:
    (directory / f"{stem}.md").write_text(f"# {stem}\n\nbody text\n", encoding="utf-8")
    if with_metadata:
        (directory / f"{stem}.metadata.json").write_text(
            json.dumps({"source_type": "chat_transcript", "channel": "-tmp-project"}),
            encoding="utf-8",
        )


def test_ingest_skips_a_document_whose_metadata_file_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VELOX_HOOK_KNOWLEDGE_BASE", _KNOWLEDGE_BASE_ID)
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    _write_transcript_pair(transcripts, "claude-code-session-1")
    _write_transcript_pair(transcripts, "claude-code-session-2", with_metadata=False)
    recorder = _Recorder(httpx.Response(202, json={"job_id": "job-1"}))
    _install(monkeypatch, recorder)

    code = main(["ingest", str(transcripts)])

    assert code == 0
    # Only the document with metadata was ever POSTed -- the bare one never
    # reaches the wire, because uploading it without metadata is exactly the
    # unfixable mistake this command exists to prevent.
    assert len(recorder.requests) == 1
    assert b"claude-code-session-1.md" in recorder.requests[0].content
    captured = capsys.readouterr()
    assert "claude-code-session-2.metadata.json" in captured.err
    assert "skipped=1" in captured.out
    assert "accepted=1" in captured.out


def test_ingest_reports_409_as_duplicate_rather_than_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("VELOX_HOOK_KNOWLEDGE_BASE", _KNOWLEDGE_BASE_ID)
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    _write_transcript_pair(transcripts, "claude-code-session-1")
    _install(
        monkeypatch,
        _Recorder(httpx.Response(409, json={"error": {"code": "DUPLICATE_DOCUMENT"}})),
    )

    code = main(["ingest", str(transcripts)])

    assert code == 0
    out = capsys.readouterr().out
    assert "duplicate=1" in out
    assert "failed=0" in out


def test_ingest_fails_loudly_when_no_knowledge_base_resolves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("VELOX_HOOK_KNOWLEDGE_BASE", raising=False)
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    _install(
        monkeypatch,
        _Recorder(
            httpx.Response(200, json={"items": [{"id": _KNOWLEDGE_BASE_ID}, {"id": "other"}]})
        ),
    )

    code = main(["ingest", str(transcripts)])

    assert code != 0
    assert "VELOX_HOOK_KNOWLEDGE_BASE" in capsys.readouterr().err


def test_the_console_script_points_at_this_module() -> None:
    # settings.json holds the path of a `velox-hook` executable, so the entry
    # point is the whole interface. A rename here is a broken hook there, with
    # nothing to say why.
    manifest = tomllib.loads(
        (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert manifest["project"]["scripts"]["velox-hook"] == "rag_service.dev.hooks:main"
