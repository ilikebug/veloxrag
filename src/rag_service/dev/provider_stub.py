"""Development-only deterministic HTTPS-compatible embedding Provider stub."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import ipaddress
import json
import os
from collections.abc import Sequence
from typing import Final, Never

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse

from rag_service.config import Environment

_MAX_REQUEST_BYTES: Final = 64 * 1024
_ERROR_MESSAGE: Final = "Development provider stub is not allowed"


class ProviderStubConfigurationError(Exception):
    """Safe error for attempts to run the development stub outside local/test."""

    __slots__ = ()

    def __init__(self) -> None:
        super().__init__(_ERROR_MESSAGE)


def validate_provider_stub_runtime(
    *,
    environment: Environment,
    host: str,
    allow_container_bind: bool = False,
) -> None:
    failed = False
    try:
        resolved_environment = Environment(environment)
        if (
            resolved_environment is Environment.PRODUCTION
            or type(host) is not str
            or type(allow_container_bind) is not bool
        ):
            failed = True
        elif host == "0.0.0.0":
            failed = not allow_container_bind
        elif host != "localhost":
            address = ipaddress.ip_address(host)
            if not address.is_loopback:
                failed = True
    except Exception:
        failed = True
    if failed:
        raise ProviderStubConfigurationError from None


def _safe_error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def _deterministic_vector(model: str, text: str, dimension: int) -> list[float]:
    vector: list[float] = []
    counter = 0
    while len(vector) < dimension:
        digest = hashlib.sha256(
            model.encode("utf-8")
            + b"\x00"
            + text.encode("utf-8")
            + b"\x00"
            + counter.to_bytes(4, "big")
        ).digest()
        for offset in range(0, len(digest), 4):
            integer = int.from_bytes(digest[offset : offset + 4], "big")
            vector.append((integer / 2**31) - 1.0)
            if len(vector) == dimension:
                break
        counter += 1
    return vector


async def _bounded_body(request: Request) -> bytes | None:
    chunks: list[bytes] = []
    total = 0
    try:
        async for chunk in request.stream():
            total += len(chunk)
            if total > _MAX_REQUEST_BYTES:
                return None
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        chunks.clear()
        total = 0


def create_provider_stub_app(
    *,
    dimension: int,
    max_batch_size: int,
    expected_authorization_sha256: str,
    request_record_limit: int = 256,
) -> FastAPI:
    if (
        type(dimension) is not int
        or not 1 <= dimension <= 4096
        or type(max_batch_size) is not int
        or not 1 <= max_batch_size <= 256
        or type(expected_authorization_sha256) is not str
        or len(expected_authorization_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_authorization_sha256)
        or type(request_record_limit) is not int
        or not 1 <= request_record_limit <= 10_000
    ):
        raise ProviderStubConfigurationError from None

    app = FastAPI(title="RAG Development Provider Stub", docs_url=None, redoc_url=None)
    app.state.request_records = []

    @app.post("/api/v1/embeddings")
    @app.post("/v1/embeddings")
    async def embeddings(request: Request) -> JSONResponse:
        authorization = request.headers.get("authorization", "")
        authorization_digest = hashlib.sha256(authorization.encode("utf-8")).hexdigest()
        authorization = "<redacted>"
        if not hmac.compare_digest(authorization_digest, expected_authorization_sha256):
            authorization_digest = "<redacted>"
            return _safe_error(401, "UNAUTHORIZED", "Unauthorized")
        authorization_digest = "<redacted>"

        body = await _bounded_body(request)
        if body is None:
            return _safe_error(413, "REQUEST_TOO_LARGE", "Request too large")
        document: object = None
        try:
            document = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _safe_error(422, "INVALID_REQUEST", "Invalid request")
        finally:
            body = b""
        if not isinstance(document, dict) or set(document).difference(
            {"input", "model", "provider"}
        ):
            document = None
            return _safe_error(422, "INVALID_REQUEST", "Invalid request")
        model = document.get("model")
        raw_inputs = document.get("input")
        provider = document.get("provider")
        if (
            type(model) is not str
            or not model
            or type(raw_inputs) is not list
            or not 1 <= len(raw_inputs) <= max_batch_size
            or any(type(value) is not str or not value for value in raw_inputs)
            or (provider is not None and type(provider) is not dict)
        ):
            document = None
            return _safe_error(422, "INVALID_REQUEST", "Invalid request")
        inputs = list(raw_inputs)
        response_data = [
            {
                "object": "embedding",
                "index": index,
                "embedding": _deterministic_vector(model, text, dimension),
            }
            for index, text in enumerate(inputs)
        ]
        prompt_tokens = sum(max(1, len(text.split())) for text in inputs)
        if len(app.state.request_records) >= request_record_limit:
            del app.state.request_records[0]
        app.state.request_records.append(
            {
                "authorized": True,
                "host": request.headers.get("host", ""),
                "input_count": len(inputs),
                "model": model,
                "provider": provider,
            }
        )
        inputs.clear()
        document = None
        return JSONResponse(
            {
                "object": "list",
                "data": response_data,
                "model": model,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "total_tokens": prompt_tokens,
                },
            }
        )

    @app.post("/v1/redirect")
    async def redirect() -> RedirectResponse:
        return RedirectResponse(url="/v1/embeddings", status_code=307)

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    return app


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        del message
        raise ProviderStubConfigurationError from None


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(
        prog="velox-provider-stub",
        description="Run the development-only deterministic Provider HTTPS stub",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument("--max-batch-size", type=int, default=8)
    parser.add_argument("--allow-container-bind", action="store_true")
    parser.add_argument("--cert-file", required=True)
    parser.add_argument("--key-file", required=True)
    parser.add_argument(
        "--authorization-sha256",
        default=os.environ.get("RAG_PROVIDER_STUB_AUTHORIZATION_SHA256"),
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> None:
    try:
        environment = Environment(os.environ.get("RAG_ENVIRONMENT", Environment.LOCAL.value))
        parsed = _parser().parse_args(arguments)
        validate_provider_stub_runtime(
            environment=environment,
            host=parsed.host,
            allow_container_bind=parsed.allow_container_bind,
        )
        app = create_provider_stub_app(
            dimension=parsed.dimension,
            max_batch_size=parsed.max_batch_size,
            expected_authorization_sha256=parsed.authorization_sha256,
        )
        uvicorn.run(
            app,
            host=parsed.host,
            port=parsed.port,
            ssl_certfile=parsed.cert_file,
            ssl_keyfile=parsed.key_file,
            access_log=False,
        )
    except (ProviderStubConfigurationError, ValueError, OSError):
        raise SystemExit(1) from None


__all__ = [
    "ProviderStubConfigurationError",
    "create_provider_stub_app",
    "main",
    "validate_provider_stub_runtime",
]
