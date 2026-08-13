from __future__ import annotations

import argparse
import asyncio
import json
import os
import stat
import sys
import traceback
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Never, Protocol, cast
from uuid import UUID

from pydantic import BaseModel, ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.api.errors import BusinessError
from rag_service.api.middleware import request_id_for
from rag_service.auth.schemas import (
    AdminApiKeyCreate,
    AgentApiKeyCreate,
    AgentApiKeyUpdate,
    IssuedApiKey,
)
from rag_service.auth.services import ApiKeyService
from rag_service.config import Settings
from rag_service.db.session import Database

if TYPE_CHECKING:
    from rag_service.indexing.qdrant import QdrantClient
    from rag_service.indexing.repair import GenerationRepairReservation

_MAX_POLICY_BYTES = 64 * 1024


class _CommandInputError(Exception):
    pass


class _OutputDeliveryIndeterminate(Exception):
    pass


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        del message
        raise _CommandInputError from None


class _GenerationRepairReservationService(Protocol):
    async def reserve(self, generation_id: UUID) -> GenerationRepairReservation: ...


def _qdrant_client_from_url(url: str, *, timeout_seconds: float) -> QdrantClient:
    from rag_service.indexing.qdrant import qdrant_client_from_url

    return qdrant_client_from_url(url, timeout_seconds=timeout_seconds)


def _generation_repair_service(
    *,
    session_factory: Any,
    qdrant: QdrantClient,
) -> _GenerationRepairReservationService:
    from rag_service.indexing.repair import GenerationRepairService

    return GenerationRepairService(session_factory=session_factory, qdrant=qdrant)


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(prog="velox-admin")
    groups = parser.add_subparsers(dest="key_group", required=True)

    admin = groups.add_parser("admin-key")
    admin_commands = admin.add_subparsers(dest="command", required=True)
    admin_create = admin_commands.add_parser("create")
    admin_create.add_argument("--name", required=True)
    admin_list = admin_commands.add_parser("list")
    admin_list.add_argument("--cursor")
    admin_list.add_argument("--limit", type=int)
    admin_revoke = admin_commands.add_parser("revoke")
    admin_revoke.add_argument("id")

    agent = groups.add_parser("agent-key")
    agent_commands = agent.add_subparsers(dest="command", required=True)
    agent_create = agent_commands.add_parser("create")
    agent_create.add_argument("--from-file", required=True, dest="policy_file")
    agent_list = agent_commands.add_parser("list")
    agent_list.add_argument("--cursor")
    agent_list.add_argument("--limit", type=int)
    agent_update = agent_commands.add_parser("update")
    agent_update.add_argument("id")
    agent_update.add_argument("--from-file", required=True, dest="policy_file")
    agent_revoke = agent_commands.add_parser("revoke")
    agent_revoke.add_argument("id")

    repair = groups.add_parser("repair-generation")
    repair.add_argument("--generation-id", required=True)
    return parser


def _safe_uuid(value: object) -> UUID:
    if not isinstance(value, str):
        raise _CommandInputError from None
    try:
        return UUID(value)
    except (AttributeError, ValueError):
        value = "<redacted>"
        raise _CommandInputError from None


def _read_policy_bytes(path_value: object) -> bytes:
    if not isinstance(path_value, str):
        raise _CommandInputError from None
    raw_path = path_value
    path_value = "<redacted>"
    descriptor = -1
    chunks: list[bytes] = []
    chunk = b"<redacted>"
    result: bytes | None = None
    payload = b"<redacted>"
    failed = False
    pre_open_identity: tuple[int, int] | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
        before = os.lstat(raw_path)
        if stat.S_ISLNK(before.st_mode):
            failed = True
        pre_open_identity = (before.st_dev, before.st_ino)
        no_follow = getattr(os, "O_NOFOLLOW", 0)
        if no_follow:
            flags |= no_follow
        if not failed:
            descriptor = os.open(raw_path, flags)
            raw_path = "<redacted>"
            opened = os.fstat(descriptor)
            if pre_open_identity is not None and pre_open_identity != (
                opened.st_dev,
                opened.st_ino,
            ):
                failed = True
            if not stat.S_ISREG(opened.st_mode) or opened.st_size > _MAX_POLICY_BYTES:
                failed = True
            remaining = _MAX_POLICY_BYTES + 1
            while not failed and remaining > 0:
                chunk = os.read(descriptor, remaining)
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            after_read = os.fstat(descriptor)
            if (
                after_read.st_size != opened.st_size
                or after_read.st_mtime_ns != opened.st_mtime_ns
                or after_read.st_ctime_ns != opened.st_ctime_ns
            ):
                failed = True
            try:
                os.close(descriptor)
            except OSError:
                failed = True
            finally:
                descriptor = -1
            if not failed:
                payload = b"".join(chunks)
                if len(payload) > _MAX_POLICY_BYTES:
                    failed = True
                else:
                    result = payload
                payload = b"<redacted>"
    except OSError:
        failed = True
    finally:
        raw_path = "<redacted>"
        chunks = [b"<redacted>"]
        chunk = b"<redacted>"
        payload = b"<redacted>"
        if sys.exception() is not None:
            result = None
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                failed = True
            except BaseException:
                result = None
                raise
            finally:
                descriptor = -1
        pre_open_identity = None

    if failed or result is None:
        result = None
        raise _CommandInputError from None
    return result


def _load_policy(path_value: object, model: type[BaseModel]) -> BaseModel:
    raw_path = path_value
    path_value = "<redacted>"
    payload = b"<redacted>"
    policy: BaseModel | None = None
    failed = False
    try:
        payload = _read_policy_bytes(raw_path)
        raw_path = "<redacted>"
        try:
            policy = model.model_validate_json(payload)
        except (ValidationError, ValueError):
            failed = True
    finally:
        raw_path = "<redacted>"
        payload = b"<redacted>"
    if failed or policy is None:
        raise _CommandInputError from None
    return policy


def _model_document(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="json")


def _issued_document(issued: IssuedApiKey) -> dict[str, Any]:
    return {
        "api_key": _model_document(issued.api_key),
        "token": issued.token.get_secret_value(),
    }


async def _dispatch(
    arguments: argparse.Namespace,
    service: ApiKeyService,
    session: AsyncSession,
    settings: Settings,
) -> dict[str, Any]:
    request_id = request_id_for(None, settings.max_request_id_length)
    key_group = arguments.key_group
    command = arguments.command

    if key_group == "admin-key":
        if command == "create":
            issued = await service.create_admin_key(
                AdminApiKeyCreate(name=arguments.name),
                request_id=request_id,
            )
            return _issued_document(issued)
        if command == "list":
            return _model_document(
                await service.list_admin_keys(
                    cursor=arguments.cursor,
                    limit=arguments.limit,
                )
            )
        if command == "revoke":
            return _model_document(
                await service.revoke_admin_key(
                    _safe_uuid(arguments.id),
                    request_id=request_id,
                )
            )

    if key_group == "agent-key":
        if command == "create":
            policy = _load_policy(arguments.policy_file, AgentApiKeyCreate)
            if type(policy) is not AgentApiKeyCreate:
                raise _CommandInputError from None
            issued = await service.create_agent_key(
                policy,
                actor=None,
                request_id=request_id,
            )
            return _issued_document(issued)
        if command == "list":
            return _model_document(
                await service.list_agent_keys(
                    cursor=arguments.cursor,
                    limit=arguments.limit,
                )
            )
        if command == "update":
            key_id = _safe_uuid(arguments.id)
            policy = _load_policy(arguments.policy_file, AgentApiKeyUpdate)
            if type(policy) is not AgentApiKeyUpdate:
                raise _CommandInputError from None
            current = await service.get_agent_key(key_id)
            await session.rollback()
            if current.etag is None:
                raise BusinessError(500, "INTERNAL_ERROR", "Internal server error")
            return _model_document(
                await service.update_agent_key(
                    key_id,
                    policy,
                    actor=None,
                    request_id=request_id,
                    expected_etag=current.etag,
                )
            )
        if command == "revoke":
            return _model_document(
                await service.revoke_agent_key(
                    _safe_uuid(arguments.id),
                    actor=None,
                    request_id=request_id,
                )
            )
    raise _CommandInputError from None


async def _dispatch_generation_repair(
    *,
    argparse_namespace: argparse.Namespace,
    service: _GenerationRepairReservationService,
) -> dict[str, Any]:
    generation_id = _safe_uuid(argparse_namespace.generation_id)
    result = await service.reserve(generation_id)
    if not isinstance(result, Mapping) or set(result) != {
        "generation_id",
        "job_id",
        "status",
    }:
        raise BusinessError(500, "INTERNAL_ERROR", "Internal server error")
    document = dict(result)
    if (
        document["generation_id"] != str(generation_id)
        or type(document["job_id"]) is not str
        or document["status"] != "queued"
    ):
        raise BusinessError(500, "INTERNAL_ERROR", "Internal server error")
    try:
        UUID(document["job_id"])
    except (AttributeError, ValueError):
        raise BusinessError(500, "INTERNAL_ERROR", "Internal server error") from None
    return document


def _compact(document: dict[str, Any]) -> str:
    return json.dumps(document, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


async def _execute(arguments: argparse.Namespace) -> str:
    settings = Settings()
    database = Database.from_settings(settings)
    qdrant: QdrantClient | None = None
    try:
        if arguments.key_group == "repair-generation":
            qdrant = _qdrant_client_from_url(
                settings.qdrant_url,
                timeout_seconds=settings.qdrant_request_timeout_seconds,
            )
            document = await _dispatch_generation_repair(
                argparse_namespace=arguments,
                service=_generation_repair_service(
                    session_factory=database.sessions,
                    qdrant=qdrant,
                ),
            )
            return _compact(document)
        async with database.session() as session:
            service = ApiKeyService(
                session=session,
                authentication_sessions=database.session,
                settings=settings,
            )
            document = await _dispatch(arguments, service, session, settings)
            return _compact(document)
    finally:
        try:
            if qdrant is not None:
                await qdrant.aclose()
        finally:
            await database.close()


def _write_stdout(document: str) -> None:
    payload = "<redacted>"
    try:
        payload = document + "\n"
        written = sys.stdout.write(payload)
        if type(written) is not int or written != len(payload):
            document = "<redacted>"
            payload = "<redacted>"
            raise _OutputDeliveryIndeterminate from None
        sys.stdout.flush()
    except _OutputDeliveryIndeterminate:
        raise
    except BaseException as error:
        document = "<redacted>"
        payload = "<redacted>"
        raise _OutputDeliveryIndeterminate from error
    finally:
        document = "<redacted>"
        payload = "<redacted>"


def _write_stderr(document: dict[str, Any]) -> None:
    try:
        sys.stderr.write(_compact(document) + "\n")
        sys.stderr.flush()
    except BaseException as error:
        error = _sanitize_exception(error)
        del error


def _sanitize_exception(error: BaseException) -> BaseException:
    pending = [error]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        identity = id(current)
        if identity in visited:
            continue
        visited.add(identity)
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        for nested in (current.__cause__, current.__context__):
            if nested is not None:
                pending.append(nested)
        if current.__traceback__ is not None:
            traceback.clear_frames(current.__traceback__)
        if isinstance(current, SystemExit):
            safe_code = _safe_system_exit_code(current.code)
            object.__setattr__(current, "code", safe_code)
        object.__setattr__(current, "args", ("<redacted>",))
        object.__setattr__(current, "__traceback__", None)
        object.__setattr__(current, "__cause__", None)
        object.__setattr__(current, "__context__", None)
    if isinstance(error, SystemExit):
        return SystemExit(_safe_system_exit_code(error.code))
    if isinstance(error, KeyboardInterrupt):
        return KeyboardInterrupt("<redacted>")
    if isinstance(error, asyncio.CancelledError):
        return asyncio.CancelledError("<redacted>")
    if isinstance(error, Exception):
        return Exception("<redacted>")
    return BaseException("<redacted>")


def _safe_system_exit_code(code: object) -> int | None:
    if code is None or type(code) is int:
        return code
    return 1


def _error_document(code: str, message: str, *, retryable: bool = False) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "retryable": retryable,
        }
    }


def _run(arguments: Sequence[str] | None) -> int:
    parsed = _parser().parse_args(arguments)
    _write_stdout(asyncio.run(_execute(parsed)))
    return 0


def main(arguments: Sequence[str] | None = None) -> int:
    reraised_system_exit = False
    system_exit_code: int | None = None
    try:
        return _run(arguments)
    except SystemExit as error:
        error = cast(SystemExit, _sanitize_exception(error))
        system_exit_code = _safe_system_exit_code(error.code)
        arguments = None
        reraised_system_exit = True
    except _OutputDeliveryIndeterminate as error:
        error = cast(_OutputDeliveryIndeterminate, _sanitize_exception(error))
        _write_stderr(
            _error_document(
                "OUTPUT_DELIVERY_INDETERMINATE",
                "Output delivery is uncertain; use list to verify before retrying",
            )
        )
        return 1
    except _CommandInputError as error:
        error = cast(_CommandInputError, _sanitize_exception(error))
        _write_stderr(_error_document("VALIDATION_ERROR", "Invalid command input"))
        return 2
    except BusinessError as error:
        document = _error_document(error.code, error.message, retryable=error.retryable)
        error = cast(BusinessError, _sanitize_exception(error))
        _write_stderr(document)
        return 1
    except (ValidationError, ValueError) as error:
        error = cast(ValidationError | ValueError, _sanitize_exception(error))
        _write_stderr(_error_document("VALIDATION_ERROR", "Invalid command input"))
        return 2
    except (KeyboardInterrupt, asyncio.CancelledError) as error:
        error = cast(KeyboardInterrupt | asyncio.CancelledError, _sanitize_exception(error))
        _write_stderr(_error_document("CANCELLED", "Command cancelled"))
        return 130
    except BaseException as error:
        error = _sanitize_exception(error)
        _write_stderr(_error_document("INTERNAL_ERROR", "Command failed"))
        return 1
    if reraised_system_exit:
        raise SystemExit(system_exit_code) from None
    raise AssertionError("unreachable")
