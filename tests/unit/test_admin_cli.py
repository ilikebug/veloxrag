import asyncio
import inspect
import json
import os
import subprocess
import sys
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import asynccontextmanager
from importlib import util
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.admin import cli as admin_cli
from rag_service.api.errors import BusinessError
from rag_service.auth.schemas import AgentApiKeyCreate
from rag_service.config import Settings

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


class _PayloadPolicy(BaseModel):
    payload: str


def _retained_exception_text(error: BaseException) -> str:
    pending = [error]
    visited: set[int] = set()
    retained: list[str] = []
    while pending and len(visited) < 64:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        retained.append(repr(current.args))
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        for nested in (current.__cause__, current.__context__):
            if nested is not None:
                pending.append(nested)
        traceback_cursor = current.__traceback__
        while traceback_cursor is not None:
            frame = traceback_cursor.tb_frame
            if "/src/rag_service/" in frame.f_code.co_filename:
                retained.append(repr(frame.f_locals))
            traceback_cursor = traceback_cursor.tb_next
    return "\n".join(retained)


def _raised_with_traceback(error: BaseException) -> BaseException:
    try:
        raise error
    except BaseException as raised:
        return raised


def _write_policy(path: Path, document: Mapping[str, object]) -> None:
    path.write_text(json.dumps(document), encoding="utf-8")


def test_bearer_dependencies_and_rag_admin_entrypoint_are_installed() -> None:
    assert util.find_spec("rag_service.auth.dependencies") is not None
    assert util.find_spec("rag_service.admin.cli") is not None
    scripts = {entry.name: entry.value for entry in entry_points(group="console_scripts")}
    assert scripts["velox-admin"] == "rag_service.admin.cli:main"


def test_policy_loader_is_bounded_and_redacts_rejected_path_and_content(tmp_path: Path) -> None:
    exact_payload_size = admin_cli._MAX_POLICY_BYTES - len(b'{"payload":""}')
    exact_document = b'{"payload":"' + (b"x" * exact_payload_size) + b'"}'
    exact_path = tmp_path / "exact-policy.json"
    exact_path.write_bytes(exact_document)

    exact = admin_cli._load_policy(str(exact_path), _PayloadPolicy)
    assert isinstance(exact, _PayloadPolicy)
    assert len(exact.payload) == exact_payload_size

    sensitive_marker = "OVERSIZED-POLICY-MARKER"
    oversized_path = tmp_path / sensitive_marker
    oversized_path.write_bytes(exact_document + sensitive_marker.encode())
    with pytest.raises(Exception) as raised:
        admin_cli._load_policy(str(oversized_path), _PayloadPolicy)

    retained = _retained_exception_text(raised.value)
    assert sensitive_marker not in retained
    assert str(oversized_path) not in retained


def test_policy_loader_rejects_symlinks_and_pins_the_opened_regular_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "target.json"
    _write_policy(
        target,
        {"name": "symlink-target", "requests_per_minute": 1, "max_concurrency": 1},
    )
    symlink = tmp_path / "policy-link.json"
    symlink.symlink_to(target)
    with pytest.raises(admin_cli._CommandInputError):
        admin_cli._load_policy(str(symlink), AgentApiKeyCreate)

    opened_path = tmp_path / "opened.json"
    replacement_path = tmp_path / "replacement.json"
    _write_policy(
        opened_path,
        {"name": "opened-policy", "requests_per_minute": 1, "max_concurrency": 1},
    )
    _write_policy(
        replacement_path,
        {"name": "replacement-policy", "requests_per_minute": 1, "max_concurrency": 1},
    )
    real_open = os.open
    open_count = 0
    opened_descriptors: list[int] = []

    def replacing_open(path: str, flags: int, mode: int = 0o777) -> int:
        nonlocal open_count
        descriptor = real_open(path, flags, mode)
        if path == str(opened_path):
            open_count += 1
            opened_descriptors.append(descriptor)
            os.replace(replacement_path, opened_path)
        return descriptor

    monkeypatch.setattr(os, "open", replacing_open)
    policy = admin_cli._load_policy(str(opened_path), AgentApiKeyCreate)

    assert isinstance(policy, AgentApiKeyCreate)
    assert policy.name == "opened-policy"
    assert open_count == 1
    with pytest.raises(OSError):
        os.fstat(opened_descriptors[0])


def test_policy_loader_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo_path = tmp_path / "policy.fifo"
    os.mkfifo(fifo_path)
    code = """
import sys
from rag_service.admin.cli import _load_policy
from rag_service.auth.schemas import AgentApiKeyCreate

try:
    _load_policy(sys.argv[1], AgentApiKeyCreate)
except BaseException:
    raise SystemExit(0)
raise SystemExit(3)
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code, str(fifo_path)],
            cwd=REPOSITORY_ROOT,
            text=True,
            capture_output=True,
            timeout=2,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("policy loading blocked on a FIFO")
    assert result.returncode == 0
    with pytest.raises(admin_cli._CommandInputError):
        admin_cli._load_policy("/dev/null", AgentApiKeyCreate)


def test_policy_loader_rejects_a_regular_file_that_grows_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy_path = tmp_path / "growing-policy.json"
    _write_policy(
        policy_path,
        {"name": "growing-policy", "requests_per_minute": 1, "max_concurrency": 1},
    )
    real_read = os.read
    grew = False

    def growing_read(descriptor: int, count: int) -> bytes:
        nonlocal grew
        data = real_read(descriptor, count)
        if data and not grew:
            with policy_path.open("ab") as stream:
                stream.write(b" ")
            grew = True
        return data

    monkeypatch.setattr(os, "read", growing_read)
    with pytest.raises(admin_cli._CommandInputError):
        admin_cli._load_policy(str(policy_path), AgentApiKeyCreate)


@pytest.mark.parametrize("failure_mode", ["write-error", "short-write", "flush-error"])
def test_stdout_failures_do_not_retain_the_secret_document(
    failure_mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = "rag_adm_OUTPUT-RETENTION-MARKER"

    class FailingStdout:
        def write(self, value: str) -> int:
            if failure_mode == "write-error":
                raise OSError("SENSITIVE-WRITE-MARKER")
            if failure_mode == "short-write":
                return len(value) - 1
            return len(value)

        def flush(self) -> None:
            if failure_mode == "flush-error":
                raise OSError("SENSITIVE-FLUSH-MARKER")

    monkeypatch.setattr(sys, "stdout", FailingStdout())
    with pytest.raises(BaseException) as raised:
        admin_cli._write_stdout(json.dumps({"token": token}))

    safe = admin_cli._sanitize_exception(raised.value)
    retained = _retained_exception_text(raised.value) + _retained_exception_text(safe)
    assert token not in retained
    assert "SENSITIVE-WRITE-MARKER" not in retained
    assert "SENSITIVE-FLUSH-MARKER" not in retained


def test_main_normalizes_unsafe_system_exit_payload_without_retaining_it() -> None:
    sensitive_marker = "UNSAFE-SYSTEM-EXIT-PAYLOAD"

    class RaisingArguments:
        def __iter__(self) -> Iterator[str]:
            raise SystemExit(sensitive_marker)

    with pytest.raises(SystemExit) as raised:
        admin_cli.main(cast(Any, RaisingArguments()))

    assert raised.value.code == 1
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert sensitive_marker not in _retained_exception_text(raised.value)


@pytest.mark.parametrize("exit_code", [None, 0, 23])
def test_main_preserves_safe_system_exit_codes(exit_code: int | None) -> None:
    class RaisingArguments:
        def __iter__(self) -> Iterator[str]:
            raise SystemExit(exit_code)

    with pytest.raises(SystemExit) as raised:
        admin_cli.main(cast(Any, RaisingArguments()))

    assert raised.value.code == exit_code


def test_exception_sanitizer_scrubs_nested_base_exception_group_payloads() -> None:
    sensitive_marker = "NESTED-BASE-EXCEPTION-MARKER"
    leaves = tuple(
        _raised_with_traceback(ValueError(f"{sensitive_marker}-{index}")) for index in range(44)
    ) + (
        _raised_with_traceback(asyncio.CancelledError(sensitive_marker)),
        _raised_with_traceback(KeyboardInterrupt(sensitive_marker)),
        _raised_with_traceback(SystemExit(sensitive_marker)),
    )
    nested = BaseExceptionGroup(
        f"{sensitive_marker}-outer-message",
        [BaseExceptionGroup(f"{sensitive_marker}-inner-message", list(leaves))],
    )

    try:
        raise nested
    except BaseExceptionGroup as raised:
        safe = admin_cli._sanitize_exception(raised)

    assert not isinstance(safe, BaseExceptionGroup)
    assert sensitive_marker not in str(safe)
    assert sensitive_marker not in repr(safe)
    assert sensitive_marker not in _retained_exception_text(safe)
    assert sensitive_marker in str(nested)
    for leaf in leaves:
        assert leaf.args == ("<redacted>",)
        assert leaf.__traceback__ is None
        assert leaf.__cause__ is None
        assert leaf.__context__ is None


def test_main_replaces_an_exception_group_before_writing_the_safe_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensitive_marker = "MAIN-GROUP-FRAME-MARKER"
    leaf = _raised_with_traceback(ValueError(sensitive_marker))
    group = BaseExceptionGroup(sensitive_marker, [leaf])
    observed_main_frames: list[str] = []

    def raise_group(_arguments: object) -> int:
        raise group

    def inspect_safe_error(document: dict[str, Any]) -> None:
        caller = inspect.currentframe()
        assert caller is not None
        main_frame = caller.f_back
        assert main_frame is not None
        retained_error = main_frame.f_locals.get("error")
        assert isinstance(retained_error, BaseException)
        assert not isinstance(retained_error, BaseExceptionGroup)
        observed_main_frames.append(
            "\n".join((str(retained_error), repr(retained_error), repr(main_frame.f_locals)))
        )
        assert document["error"]["code"] == "INTERNAL_ERROR"

    monkeypatch.setattr(admin_cli, "_run", raise_group)
    monkeypatch.setattr(admin_cli, "_write_stderr", inspect_safe_error)

    assert admin_cli.main([]) == 1
    assert len(observed_main_frames) == 1
    assert sensitive_marker not in observed_main_frames[0]
    assert leaf.args == ("<redacted>",)
    assert leaf.__traceback__ is None


def test_cli_cancellation_closes_resources_and_prints_one_safe_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeDatabase:
        closed = False

        @classmethod
        def from_settings(cls, _settings: Settings) -> "FakeDatabase":
            del cls
            return database

        @asynccontextmanager
        async def session(self) -> AsyncIterator[AsyncSession]:
            yield cast(AsyncSession, object())

        async def close(self) -> None:
            self.closed = True

    database = FakeDatabase()
    monkeypatch.setattr(admin_cli, "Database", FakeDatabase)

    async def cancel_dispatch(*_arguments: object) -> dict[str, Any]:
        raise asyncio.CancelledError

    monkeypatch.setattr(admin_cli, "_dispatch", cancel_dispatch)
    exit_code = admin_cli.main(["admin-key", "list"])

    captured = capsys.readouterr()
    assert exit_code == 130
    assert captured.out == ""
    assert captured.err == (
        '{"error":{"code":"CANCELLED","message":"Command cancelled","retryable":false}}\n'
    )
    assert database.closed is True


def test_rag_admin_help_lists_only_supported_command_groups() -> None:
    result = subprocess.run(
        [str(Path(sys.executable).with_name("velox-admin")), "--help"],
        cwd=REPOSITORY_ROOT,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0
    assert "admin-key" in result.stdout
    assert "agent-key" in result.stdout
    assert "admin-key update" not in result.stdout


def test_repair_generation_parser_accepts_only_a_generation_uuid() -> None:
    generation_id = uuid4()
    try:
        parsed = admin_cli._parser().parse_args(
            ["repair-generation", "--generation-id", str(generation_id)]
        )
    except admin_cli._CommandInputError:
        parsed = None

    assert parsed is not None
    assert parsed.key_group == "repair-generation"
    assert parsed.generation_id == str(generation_id)


def test_repair_generation_dispatch_returns_only_safe_reservation_facts() -> None:
    generation_id = uuid4()
    job_id = uuid4()

    class FakeRepairService:
        calls: list[object] = []

        async def reserve(self, requested_generation_id: object) -> object:
            self.calls.append(requested_generation_id)
            return {
                "generation_id": str(generation_id),
                "job_id": str(job_id),
                "status": "queued",
            }

    service = FakeRepairService()
    try:
        document = asyncio.run(
            admin_cli._dispatch_generation_repair(
                argparse_namespace=admin_cli._parser().parse_args(
                    ["repair-generation", "--generation-id", str(generation_id)]
                ),
                service=cast(Any, service),
            )
        )
    except AttributeError:
        document = None

    assert document == {
        "generation_id": str(generation_id),
        "job_id": str(job_id),
        "status": "queued",
    }
    assert service.calls == [generation_id]


def test_repair_generation_dispatch_classifies_malformed_service_job_id_as_internal() -> None:
    generation_id = uuid4()

    class MalformedRepairService:
        async def reserve(self, _requested_generation_id: object) -> object:
            return {
                "generation_id": str(generation_id),
                "job_id": "not-a-uuid",
                "status": "queued",
            }

    with pytest.raises(BusinessError) as captured:
        asyncio.run(
            admin_cli._dispatch_generation_repair(
                argparse_namespace=admin_cli._parser().parse_args(
                    ["repair-generation", "--generation-id", str(generation_id)]
                ),
                service=cast(Any, MalformedRepairService()),
            )
        )

    assert captured.value.status_code == 500
    assert captured.value.code == "INTERNAL_ERROR"


def test_repair_generation_execute_closes_qdrant_and_database_on_reservation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generation_id = uuid4()

    class FakeDatabase:
        sessions = cast(Any, object())
        closed = False

        @classmethod
        def from_settings(cls, _settings: Settings) -> "FakeDatabase":
            del cls
            return database

        async def close(self) -> None:
            self.closed = True

    class FakeQdrant:
        closed = False

        async def aclose(self) -> None:
            self.closed = True

    class FailingRepairService:
        def __init__(self, *, session_factory: object, qdrant: object) -> None:
            assert session_factory is database.sessions
            assert qdrant is qdrant_client

        async def reserve(self, requested_generation_id: object) -> object:
            assert requested_generation_id == generation_id
            raise BusinessError(503, "QDRANT_UNAVAILABLE", "Qdrant is unavailable", True)

    database = FakeDatabase()
    qdrant_client = FakeQdrant()
    monkeypatch.setattr(admin_cli, "Database", FakeDatabase)
    monkeypatch.setattr(
        admin_cli,
        "_qdrant_client_from_url",
        lambda _url, *, timeout_seconds: qdrant_client,
    )
    monkeypatch.setattr(
        admin_cli,
        "_generation_repair_service",
        FailingRepairService,
    )

    with pytest.raises(BusinessError, match="Qdrant is unavailable"):
        asyncio.run(
            admin_cli._execute(
                admin_cli._parser().parse_args(
                    ["repair-generation", "--generation-id", str(generation_id)]
                )
            )
        )

    assert qdrant_client.closed is True
    assert database.closed is True
