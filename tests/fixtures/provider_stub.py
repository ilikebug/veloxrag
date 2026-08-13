from __future__ import annotations

import asyncio
import hashlib
import socket
import ssl
import threading
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
import trustme
import uvicorn

from rag_service.dev.provider_stub import create_provider_stub_app

PROVIDER_STUB_SECRET = "local-provider-stub-secret-sentinel"


class ProviderStubFixtureError(RuntimeError):
    pass


@dataclass(slots=True)
class MutableProviderResolver:
    answers: list[tuple[str, ...]]
    calls: list[tuple[str, int]] = field(default_factory=list)

    def resolve(self, hostname: str, port: int) -> Iterable[str]:
        self.calls.append((hostname, port))
        if not self.answers:
            raise RuntimeError("provider stub resolver answers exhausted")
        return self.answers.pop(0)


@dataclass(frozen=True, slots=True, repr=False)
class RunningProviderStub:
    base_url: str
    loopback_base_url: str
    ca_bundle: str
    hostname: str
    port: int
    resolver: MutableProviderResolver = field(repr=False, compare=False)
    request_records: list[object] = field(repr=False, compare=False)

    def __repr__(self) -> str:
        return "RunningProviderStub(<redacted>)"


def _serve(
    server: uvicorn.Server,
    listener: socket.socket,
    failures: list[str],
) -> None:
    try:
        asyncio.run(server.serve(sockets=[listener]))
    except BaseException as error:
        failures.append(type(error).__name__)


@contextmanager
def running_provider_https_stub(tmp_path: Path) -> Iterator[RunningProviderStub]:
    hostname = "<redacted>"
    authority: Any = None
    certificate: Any = None
    ca_path: Path | None = None
    cert_path: Path | None = None
    key_path: Path | None = None
    expected_authorization_sha256 = "<redacted>"
    app: Any = None
    port = 0
    config: Any = None
    readiness_failure = "<redacted>"
    loopback_base_url = "<redacted>"
    tls_context: ssl.SSLContext | None = None
    response: httpx.Response | None = None
    fixture: RunningProviderStub | None = None
    listener: socket.socket | None = None
    server: uvicorn.Server | None = None
    thread: threading.Thread | None = None
    thread_started = False
    tls_paths: tuple[Path, ...] = ()
    failures: list[str] = []
    primary_error: BaseException | None = None
    yielded = False
    try:
        hostname = "provider.test"
        authority = trustme.CA()
        certificate = authority.issue_cert(hostname, "localhost", "127.0.0.1")
        ca_path = tmp_path / "provider-ca.pem"
        cert_path = tmp_path / "provider-cert.pem"
        key_path = tmp_path / "provider-key.pem"
        tls_paths = (ca_path, cert_path, key_path)
        authority.cert_pem.write_to_path(ca_path)
        certificate.cert_chain_pems[0].write_to_path(cert_path)
        certificate.private_key_pem.write_to_path(key_path)
        key_path.chmod(0o600)

        expected_authorization_sha256 = hashlib.sha256(
            f"Bearer {PROVIDER_STUB_SECRET}".encode()
        ).hexdigest()
        app = create_provider_stub_app(
            dimension=3,
            max_batch_size=4,
            expected_authorization_sha256=expected_authorization_sha256,
        )
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen(128)
        listener.setblocking(False)
        port = int(listener.getsockname()[1])
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_config=None,
            access_log=False,
            lifespan="off",
            ssl_certfile=str(cert_path),
            ssl_keyfile=str(key_path),
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(
            target=_serve,
            args=(server, listener, failures),
            name="rag-provider-test-stub",
            daemon=True,
        )
        thread.start()
        thread_started = True
        deadline = time.monotonic() + 5
        readiness_failure = "timeout"
        loopback_base_url = f"https://127.0.0.1:{port}/v1"
        tls_context = ssl.create_default_context(cafile=str(ca_path))
        while thread.is_alive() and time.monotonic() < deadline:
            if not server.started:
                time.sleep(0.01)
                continue
            try:
                response = httpx.post(
                    f"{loopback_base_url}/embeddings",
                    headers={"Authorization": f"Bearer {PROVIDER_STUB_SECRET}"},
                    json={"model": "fixture-readiness", "input": ["ready"]},
                    verify=tls_context,
                    trust_env=False,
                    timeout=0.5,
                )
                if response.status_code == 200:
                    break
                readiness_failure = f"status_{response.status_code}"
            except httpx.HTTPError as error:
                readiness_failure = type(error).__name__
            time.sleep(0.01)
        else:
            raise ProviderStubFixtureError(
                f"provider HTTPS stub failed to become ready ({failures or [readiness_failure]})"
            )

        app.state.request_records.clear()
        fixture = RunningProviderStub(
            base_url=f"https://{hostname}:{port}/v1",
            loopback_base_url=loopback_base_url,
            ca_bundle=str(ca_path),
            hostname=hostname,
            port=port,
            resolver=MutableProviderResolver(answers=[]),
            request_records=app.state.request_records,
        )
        yielded = True
        yield fixture
    except BaseException as error:
        primary_error = (
            error
            if yielded or not isinstance(error, Exception)
            else ProviderStubFixtureError(
                f"provider HTTPS stub setup failed ({type(error).__name__})"
            )
        )

    teardown_errors: list[BaseException] = []
    if server is not None:
        server.should_exit = True
    if thread is not None and thread_started:
        try:
            thread.join(timeout=5)
            if thread.is_alive():
                teardown_errors.append(
                    ProviderStubFixtureError("provider HTTPS stub failed to stop")
                )
        except BaseException as error:
            teardown_errors.append(
                ProviderStubFixtureError(
                    f"provider HTTPS stub thread cleanup failed ({type(error).__name__})"
                )
            )
    if listener is not None:
        try:
            listener.close()
        except BaseException as error:
            teardown_errors.append(
                ProviderStubFixtureError(
                    f"provider HTTPS stub listener cleanup failed ({type(error).__name__})"
                )
            )
    tls_path: Path | None = None
    for tls_path in reversed(tls_paths):
        try:
            tls_path.unlink(missing_ok=True)
        except BaseException as error:
            teardown_errors.append(
                ProviderStubFixtureError(
                    f"provider HTTPS stub TLS cleanup failed ({type(error).__name__})"
                )
            )
    if failures:
        teardown_errors.append(ProviderStubFixtureError(f"provider HTTPS stub failed ({failures})"))

    # Re-raised fixture errors must not retain raw endpoints or TLS paths in this frame.
    tmp_path = Path(".")
    listener = None
    server = None
    thread = None
    tls_paths = ()
    tls_path = None
    hostname = "<redacted>"
    authority = None
    certificate = None
    ca_path = None
    cert_path = None
    key_path = None
    expected_authorization_sha256 = "<redacted>"
    app = None
    port = 0
    config = None
    readiness_failure = "<redacted>"
    loopback_base_url = "<redacted>"
    tls_context = None
    response = None
    fixture = None

    if primary_error is not None:
        if teardown_errors:
            raise BaseExceptionGroup(
                "provider HTTPS fixture body/setup and teardown failed",
                [primary_error, *teardown_errors],
            ) from None
        raise primary_error.with_traceback(primary_error.__traceback__) from None
    if teardown_errors:
        raise BaseExceptionGroup(
            "provider HTTPS fixture teardown failed",
            teardown_errors,
        ) from None


@pytest.fixture
def provider_https_stub(tmp_path: Path) -> Iterator[RunningProviderStub]:
    with running_provider_https_stub(tmp_path) as provider:
        yield provider
