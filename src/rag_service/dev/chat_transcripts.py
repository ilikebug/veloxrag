"""Convert local AI coding session transcripts into ingestible Markdown.

This is a client-side helper, not part of the service's runtime job: the service
takes Markdown and metadata, and turning a vendor's transcript format into that
shape stays outside it (see docs/rag-roadmap.md, batch 0). It lives here because
`dev/` already holds the tooling that ships with the package.

Two things about transcripts drive the whole design.

They are mostly noise. Measured over real sessions, the text a human wrote and
the text the assistant replied with is about 30% of a Claude Code transcript and
about 7% of a Codex one; the rest is chain-of-thought and tool output. Tool
output in particular is the contents of files that are already indexable on their
own, so indexing it again buries the reasoning under duplicates of the codebase.

They contain credentials. Sessions mint tokens, print connection strings, and
paste keys. Indexing a transcript copies whatever it holds into Postgres, Qdrant
and MinIO, where any key with `retrieve` can pull it back out, so redaction runs
unconditionally rather than behind a flag.
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Final, Literal, Never

REDACTION_PLACEHOLDER: Final = "[REDACTED]"

# Ordered longest-context-first so a token inside a header is redacted as part of
# the header rather than leaving `Authorization: Bearer` dangling on its own.
_REDACTIONS: Final = (
    re.compile(r"-----BEGIN[ A-Z]*PRIVATE KEY-----.*?-----END[ A-Z]*PRIVATE KEY-----", re.S),
    re.compile(r"(?i)\b(authorization\s*:\s*)(?:bearer|basic)\s+\S+"),
    re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]{16,}"),
    re.compile(r"\b(?:sk|rk|pk)-[A-Za-z0-9_-]{16,}"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr|github_pat)_[A-Za-z0-9_]{16,}"),
    re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]+"),
    # Assignments are matched by key name because the value shape is unbounded.
    re.compile(
        r"(?i)\b(password|passwd|secret|token|api[_-]?key|access[_-]?key"
        r"|private[_-]?key|credential)s?\s*[=:]\s*[\"']?[^\s\"'&]{8,}"
    ),
    # Postgres/Redis/Mongo URLs carry the password in the authority.
    re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s:/@]+:[^\s/@]+@"),
)


def redact(text: str) -> str:
    """Replace credential-shaped substrings, keeping the surrounding prose.

    Deliberately blunt: it over-redacts (a variable literally named `token` loses
    its value even when harmless) because the cost of a false positive is one
    unreadable line, while the cost of a miss is a live secret in a search index
    that several agent keys can read.
    """
    if type(text) is not str:
        raise ChatTranscriptError("transcript content is invalid")
    for pattern in _REDACTIONS:
        if pattern.groups:
            text = pattern.sub(lambda m: f"{m.group(1)} {REDACTION_PLACEHOLDER}", text)
        else:
            text = pattern.sub(REDACTION_PLACEHOLDER, text)
    return text


class ChatTranscriptError(Exception):
    """Safe conversion failure."""


@dataclass(frozen=True, slots=True)
class Turn:
    speaker: Literal["user", "assistant"]
    text: str
    occurred_at: datetime | None


@dataclass(frozen=True, slots=True)
class Transcript:
    session_id: str
    project: str
    source: Literal["claude-code", "codex"]
    source_path: str
    turns: tuple[Turn, ...]

    @property
    def occurred_at(self) -> datetime | None:
        return next((turn.occurred_at for turn in self.turns if turn.occurred_at), None)


def _parse_timestamp(value: object) -> datetime | None:
    if type(value) is not str:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


# Claude Code labels prose `text`; Codex distinguishes direction with
# `input_text` and `output_text`. Everything else in a content list — thinking,
# reasoning, tool calls, tool output — is deliberately not here.
_TEXT_BLOCK_TYPES: Final = frozenset({"text", "input_text", "output_text"})


def _blocks_text(content: object) -> str:
    """Join the human-visible text of a content field, dropping everything else.

    `thinking` and `tool_result` blocks are skipped here rather than filtered
    later: they are the bulk of the bytes and none of the retrievable answer.
    """
    if type(content) is str:
        return content.strip()
    if type(content) is not list:
        return ""
    parts = []
    for block in content:
        if type(block) is dict and block.get("type") in _TEXT_BLOCK_TYPES:
            text = block.get("text")
            if type(text) is str and text.strip():
                parts.append(text.strip())
    return "\n\n".join(parts)


def read_claude_code(path: Path) -> Transcript:
    turns: list[Turn] = []
    session_id = path.stem
    for line in _lines(path):
        message = line.get("message")
        if type(message) is not dict:
            continue
        role = message.get("role")
        if role not in {"user", "assistant"}:
            continue
        text = _blocks_text(message.get("content"))
        if not text:
            continue
        turns.append(Turn(role, text, _parse_timestamp(line.get("timestamp"))))
    return Transcript(
        session_id=session_id,
        project=path.parent.name,
        source="claude-code",
        source_path=str(path),
        turns=tuple(turns),
    )


def read_codex(path: Path) -> Transcript:
    turns: list[Turn] = []
    session_id = path.stem
    project = ""
    for line in _lines(path):
        kind = line.get("type")
        if kind == "session_meta":
            payload = line.get("payload")
            if type(payload) is dict:
                cwd = payload.get("cwd")
                if type(cwd) is str and cwd:
                    project = Path(cwd).name
            continue
        if kind != "response_item":
            continue
        payload = line.get("payload")
        if type(payload) is not dict or payload.get("type") != "message":
            continue
        role = payload.get("role")
        # `developer` carries harness scaffolding, not conversation.
        if role not in {"user", "assistant"}:
            continue
        text = _blocks_text(payload.get("content"))
        if not text:
            continue
        turns.append(Turn(role, text, _parse_timestamp(line.get("timestamp"))))
    return Transcript(
        session_id=session_id,
        project=project or path.parent.name,
        source="codex",
        source_path=str(path),
        turns=tuple(turns),
    )


def _lines(path: Path) -> Iterator[dict[str, object]]:
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            raw = raw.strip()
            if not raw:
                continue
            try:
                parsed = json.loads(raw)
            except ValueError:
                continue
            if type(parsed) is dict:
                yield parsed


_SPEAKER_LABEL: Final = {"user": "User", "assistant": "Assistant"}
_TURN_HEADING_LEVEL: Final = 2
_MAX_HEADING_LEVEL: Final = 6
_HEADING_LINE = re.compile(r"^(#{1,6})(\s)", re.M)
_FENCE_LINE = re.compile(r"^\s*(?:```|~~~)", re.M)


def _demote_headings(text: str) -> str:
    """Push headings inside a turn below the turn's own heading.

    Turn text routinely carries Markdown of its own — a pasted document, an
    injected skill, a reply that uses headings. Left alone, a `#` in the middle
    of a conversation takes over the document outline, and every chunk after it
    inherits that heading as its title path instead of the turn's. The turn
    heading is the only thing identifying who said this and when, which the
    retrieval baseline showed is what a chunk of otherwise-uniform prose needs
    to be rankable at all.
    """

    def shift(match: re.Match[str]) -> str:
        level = min(len(match.group(1)) + _TURN_HEADING_LEVEL, _MAX_HEADING_LEVEL)
        return "#" * level + match.group(2)

    # Fenced blocks are left alone: `#` at the start of a line inside a fence is
    # a shell comment or a directive, not a heading, and rewriting it corrupts
    # commands that a later session might want to reuse verbatim.
    parts = []
    inside_fence = False
    for line in text.split("\n"):
        if _FENCE_LINE.match(line):
            inside_fence = not inside_fence
            parts.append(line)
            continue
        parts.append(line if inside_fence else _HEADING_LINE.sub(shift, line))
    return "\n".join(parts)


def render_markdown(transcript: Transcript) -> str:
    """Render turns with the speaker and date inside the body, not just metadata.

    The embedding model never sees the metadata columns, so a chunk whose text
    does not name its speaker, project or date is indistinguishable from every
    other chunk of dialogue in the corpus. Turn headings also give the chunker a
    structural boundary to split on, so a chunk carries a heading path that
    identifies it.
    """
    started = transcript.occurred_at
    lines = [
        f"# {transcript.source} session in {transcript.project}",
        "",
        f"- Project: {transcript.project}",
        f"- Session: {transcript.session_id}",
        f"- Started: {started.isoformat() if started else 'unknown'}",
        "",
    ]
    for index, turn in enumerate(transcript.turns, start=1):
        stamp = turn.occurred_at.strftime("%Y-%m-%d %H:%M") if turn.occurred_at else "unknown"
        label = _SPEAKER_LABEL[turn.speaker]
        lines.append(f"## Turn {index} — {label} — {stamp} — {transcript.project}")
        lines.append("")
        lines.append(_demote_headings(redact(turn.text)))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


_CJK = re.compile(r"[\u3400-\u9fff\u3040-\u30ff]")


def _language_of(text: str) -> str:
    cjk = len(_CJK.findall(text))
    if not text:
        return "unknown"
    ratio = cjk / len(text)
    if ratio > 0.1:
        return "zh"
    return "mixed" if cjk else "en"


def build_metadata(transcript: Transcript) -> dict[str, str]:
    """Map onto the knowledge base's frozen filter schema.

    Only fields already in that schema are emitted. filter_schema cannot change
    once a generation exists, so a converter that invented a field would produce
    documents whose metadata is silently unfilterable.
    """
    started = transcript.occurred_at
    metadata = {
        "source_type": "chat",
        "doc_type": transcript.source,
        "channel": transcript.project,
        "thread_id": transcript.session_id,
        "source_path": transcript.source_path,
        # Detected over the turns alone. The rendered document always carries
        # English scaffolding in its headings, so measuring the whole body would
        # report almost every conversation as mixed regardless of its language.
        "lang": _language_of("\n".join(turn.text for turn in transcript.turns)),
    }
    if started:
        metadata["occurred_at"] = started.isoformat()
    return metadata


def _without_seen_turns(transcript: Transcript, seen: set[str]) -> Transcript:
    """Drop turns already emitted by an earlier transcript, recording the rest.

    Resuming a session writes a new transcript that replays the earlier one, so
    the same exchange lands in several files — measured at 47% of turns on this
    machine, with some appearing in eight. Indexed as-is, a search spends most of
    its top-k on copies of one answer.
    """
    kept = []
    for turn in transcript.turns:
        digest = sha256(f"{turn.speaker}\n{turn.text}".encode()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        kept.append(turn)
    return replace(transcript, turns=tuple(kept))


_READERS: Final = {"claude-code": read_claude_code, "codex": read_codex}


def discover(root: Path, source: str) -> list[Path]:
    if source == "claude-code":
        return sorted(root.glob("*/*.jsonl"))
    return sorted(root.glob("**/rollout-*.jsonl"))


def convert(
    root: Path,
    output_directory: Path,
    *,
    source: str,
    since: datetime | None = None,
    project: str | None = None,
    limit: int | None = None,
    min_turns: int = 4,
) -> list[Path]:
    """Convert transcripts and return the Markdown files written.

    `min_turns` drops the many one-exchange sessions that a coding tool
    accumulates: they are almost all "run this command" with no retrievable
    conclusion, and they would dominate the document count.
    """
    if source not in _READERS:
        raise ChatTranscriptError("transcript source is invalid")
    if not root.is_dir():
        raise ChatTranscriptError("transcript root is not a directory")
    output_directory.mkdir(parents=True, exist_ok=True)
    reader = _READERS[source]
    written: list[Path] = []
    seen_turns: set[str] = set()

    candidates = []
    for path in discover(root, source):
        transcript = reader(path)
        if project is not None and transcript.project != project:
            continue
        started = transcript.occurred_at
        if since is not None and (started is None or started < since):
            continue
        candidates.append(transcript)
    # Oldest first so a turn is kept where it was first said, and a session that
    # resumes an earlier one keeps only what it added.
    candidates.sort(key=lambda item: (item.occurred_at is None, item.occurred_at, item.session_id))

    for transcript in candidates:
        if limit is not None and len(written) >= limit:
            break
        transcript = _without_seen_turns(transcript, seen_turns)
        # Applied after deduplication on purpose: a transcript that is entirely a
        # replay of an earlier one has nothing left and should not be written.
        if len(transcript.turns) < min_turns:
            continue
        body = render_markdown(transcript)
        stem = f"{source}-{transcript.session_id}"
        document = output_directory / f"{stem}.md"
        document.write_text(body, encoding="utf-8")
        (output_directory / f"{stem}.metadata.json").write_text(
            json.dumps(build_metadata(transcript), ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        written.append(document)
    return written


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        del message
        raise ChatTranscriptError("transcript conversion configuration is invalid")


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(prog="velox-chat-transcripts")
    parser.add_argument("--source", choices=sorted(_READERS), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--since", type=str)
    parser.add_argument("--project", type=str)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--min-turns", type=int, default=4)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    try:
        parsed = _parser().parse_args(arguments)
        since = None
        if parsed.since:
            since = _parse_timestamp(parsed.since) or _parse_timestamp(f"{parsed.since}T00:00:00Z")
            if since is None:
                raise ChatTranscriptError("transcript conversion configuration is invalid")
        written = convert(
            parsed.root,
            parsed.output_directory,
            source=parsed.source,
            since=since,
            project=parsed.project,
            limit=parsed.limit,
            min_turns=parsed.min_turns,
        )
        print(json.dumps({"documents": len(written)}))
        return 0
    except ChatTranscriptError:
        return 1


__all__ = [
    "ChatTranscriptError",
    "Transcript",
    "Turn",
    "build_metadata",
    "convert",
    "main",
    "read_claude_code",
    "read_codex",
    "redact",
    "render_markdown",
]
