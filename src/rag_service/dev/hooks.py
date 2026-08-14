"""Claude Code hooks that read from and write to the knowledge base.

Two subcommands, bound to two hook events. `retrieve` runs on UserPromptSubmit
and prints the passages it finds, which Claude Code adds to the turn's context.
`record` runs on Stop and uploads the turn that just finished.

Both are advisory. Every failure path prints nothing and exits 0: a memory that
is merely absent leaves the session exactly as it was, while a hook that fails
loudly makes the tool worse than not having it. `record` in particular never
exits 2, which would prevent Claude from stopping and put the session in a loop.

Neither hook reads the transcript. The hook payload carries
`last_assistant_message` for the turn that just ended, and the documentation
says to prefer it because the transcript lags; the two events share a
`prompt_id`, so the question can be handed from one to the other through a state
file. Not reading the transcript also sidesteps its replay duplication — a
resumed session rewrites earlier turns into a new file.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from contextlib import suppress
from hashlib import sha256
from pathlib import Path
from typing import Final

_STATE_TTL_SECONDS: Final = 24 * 60 * 60
_STATE_FILE_MODE: Final = 0o600


def veloxrag_home() -> Path:
    return Path.home() / ".veloxrag"


def state_directory() -> Path:
    directory = veloxrag_home() / "hook-state"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def channel_for(cwd: str) -> str:
    """Encode a working directory the way Claude Code names its project folders.

    Every `/` and `.` becomes `-`, so `/Users/y.zhang/work/AI/VeloxRAG` is
    `-Users-y-zhang-work-AI-VeloxRAG`. This has to agree with
    `chat_transcripts.read_claude_code`, which takes the channel from that folder
    name: two encodings would put whole-session and per-turn documents in
    disjoint channels.
    """
    return re.sub(r"[/.]", "-", cwd)


def _state_path(session_id: str) -> Path:
    # session_id comes from outside the process, so it is hashed rather than used
    # as a path component. A value containing `..` would otherwise choose the
    # file's location.
    digest = sha256(session_id.encode()).hexdigest()[:32]
    return state_directory() / f"{digest}.json"


def _atomic_write(path: Path, payload: bytes) -> None:
    """Write state to disk without a torn or world-readable file.

    Duplicates the pattern in `jobs/worker.write_worker_health` and
    `provider_stub_tls._atomic_write` rather than importing either: this module
    backs a hook that must start cheaply, and both of those pull in machinery
    (job scheduling, TLS) a hook has no use for. Six lines of duplication is
    cheaper than that import.

    The state held here is the user's prompt verbatim and un-redacted —
    redaction only happens later, on the way to the knowledge base — so no one
    else should be able to read this file. `mkstemp` already creates it 0o600
    and umask cannot widen that, so the `fchmod` below changes nothing today;
    it is here so the requirement is stated explicitly rather than inherited
    from a library default that a later change to how the file is created
    could quietly relax.
    """
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, _STATE_FILE_MODE)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            os.unlink(temporary_name)


def _load_state(path: Path) -> dict[str, str]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(loaded, dict):
        return {}
    return {str(key): str(value) for key, value in loaded.items()}


def remember_prompt(session_id: str, prompt_id: str, user_input: str) -> None:
    path = _state_path(session_id)
    state = _load_state(path)
    state[prompt_id] = user_input
    _atomic_write(path, json.dumps(state, ensure_ascii=False).encode("utf-8"))


def take_prompt(session_id: str, prompt_id: str) -> str | None:
    # Not locked: a concurrent `remember_prompt` for the session's next prompt
    # can interleave with this function's read-modify-unlink and lose the newer
    # question. Closing that gap would need a blocking lock, and blocking is the
    # one failure class this feature rules out for a hook that must never stall
    # the session; a non-blocking lock would not help either, since it only
    # succeeds in the uncontended case that was never racy, and degrades to this
    # same unlocked behaviour exactly when contention makes it matter. The cost
    # of leaving it is one unrecorded turn, which the feature already tolerates
    # from other causes — an upload that fails partway has the same effect.
    path = _state_path(session_id)
    state = _load_state(path)
    taken = state.pop(prompt_id, None)
    if taken is None:
        return None
    if state:
        _atomic_write(path, json.dumps(state, ensure_ascii=False).encode("utf-8"))
    else:
        path.unlink(missing_ok=True)
    return taken


def prune_stale_state() -> None:
    """Remove state left by an interrupted turn or a session that died.

    A question whose answer never arrived has no turn to record, and keeping it
    would only grow the directory.
    """
    cutoff = time.time() - _STATE_TTL_SECONDS
    for path in state_directory().glob("*.json"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
        except OSError:
            continue


__all__ = [
    "channel_for",
    "prune_stale_state",
    "remember_prompt",
    "state_directory",
    "take_prompt",
    "veloxrag_home",
]
