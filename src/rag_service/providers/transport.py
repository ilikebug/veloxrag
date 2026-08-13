"""Fail-closed HTTPS transport for cloud Provider calls."""

from __future__ import annotations

import asyncio
import inspect
import json
import ssl
from collections import OrderedDict
from collections.abc import Awaitable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Protocol, cast

import httpcore

from rag_service.providers.network_policy import (
    CanonicalProviderEndpoint,
    ProviderDnsExecutor,
    ProviderEndpointPolicy,
    ProviderNetworkPolicyError,
    ResolvedProviderEndpoint,
)

_MAX_RESPONSE_BYTES: Final = 16 * 1024 * 1024
_MAX_REQUEST_PATH_BYTES: Final = 256


def _default_async_network_backend() -> httpcore.AsyncNetworkBackend:
    """Create the supported public httpcore backend or fail closed on API drift."""

    try:
        version = str(httpcore.__version__)
        major = int(version.split(".", 1)[0])
        backend_factory = httpcore.AnyIOBackend
        backend = backend_factory()
        connect_parameters = inspect.signature(backend.connect_tcp).parameters
        if major != 1 or not {
            "host",
            "port",
            "timeout",
            "local_address",
            "socket_options",
        }.issubset(connect_parameters):
            raise ValueError
        if not callable(getattr(backend, "connect_unix_socket", None)) or not callable(
            getattr(backend, "sleep", None)
        ):
            raise ValueError
        return cast(httpcore.AsyncNetworkBackend, backend)
    except Exception:
        raise ProviderNetworkPolicyError from None


@dataclass(frozen=True, slots=True, repr=False)
class ProviderDestination:
    """One pool-safe destination selected from a complete validated DNS answer set."""

    resolved: ResolvedProviderEndpoint
    selected_address: str

    def __post_init__(self) -> None:
        if (
            type(self.resolved) is not ResolvedProviderEndpoint
            or type(self.selected_address) is not str
            or self.selected_address not in self.resolved.addresses
        ):
            raise ProviderNetworkPolicyError from None

    def __repr__(self) -> str:
        return "ProviderDestination(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class ProviderHttpResponse:
    """Bounded response value whose representation never renders upstream content."""

    status_code: int
    headers: Mapping[str, str]
    body: bytes

    def __post_init__(self) -> None:
        if (
            type(self.status_code) is not int
            or not 100 <= self.status_code <= 599
            or not isinstance(self.headers, Mapping)
            or type(self.body) is not bytes
            or len(self.body) > _MAX_RESPONSE_BYTES
        ):
            raise ValueError("Provider response is invalid")
        copied_headers: dict[str, str] = {}
        try:
            for name, value in self.headers.items():
                if type(name) is not str or type(value) is not str:
                    raise ValueError("Provider response is invalid")
                copied_headers[name.lower()] = value
            object.__setattr__(self, "headers", MappingProxyType(copied_headers))
            object.__setattr__(self, "body", bytes(self.body))
        except Exception:
            copied_headers.clear()
            raise ValueError("Provider response is invalid") from None

    def __repr__(self) -> str:
        return f"ProviderHttpResponse(status_code={self.status_code}, <redacted>)"


class ProviderPool(Protocol):
    async def post_json(
        self,
        *,
        path: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> ProviderHttpResponse: ...

    async def aclose(self) -> None: ...


class ProviderPoolFactory(Protocol):
    def create(
        self,
        *,
        destination: ProviderDestination,
        ssl_context: ssl.SSLContext,
    ) -> ProviderPool: ...


@dataclass(slots=True, repr=False)
class _ProviderPoolEntry:
    pool: ProviderPool
    leases: int = 0
    closing: bool = False


async def _finish_cleanup(cleanup: Awaitable[None]) -> bool:
    cleanup_future = asyncio.ensure_future(cleanup)
    cancellation_requested = False
    while not cleanup_future.done():
        try:
            await asyncio.shield(cleanup_future)
        except asyncio.CancelledError:
            cancellation_requested = True
    cleanup_future.result()
    return cancellation_requested


class PinnedProviderNetworkBackend(httpcore.AsyncNetworkBackend):
    """Pin TCP to one validated IP while retaining the original TLS server name."""

    def __init__(
        self,
        *,
        policy: ProviderEndpointPolicy,
        destination: ProviderDestination,
        delegate: httpcore.AsyncNetworkBackend | None = None,
        dns_executor: ProviderDnsExecutor | None = None,
    ) -> None:
        if (
            type(policy) is not ProviderEndpointPolicy
            or type(destination) is not ProviderDestination
        ):
            raise ProviderNetworkPolicyError from None
        if dns_executor is not None and type(dns_executor) is not ProviderDnsExecutor:
            raise ProviderNetworkPolicyError from None
        self._policy = policy
        self._destination = destination
        self._dns_executor = dns_executor
        self._delegate = _default_async_network_backend() if delegate is None else delegate
        if not callable(getattr(self._delegate, "connect_tcp", None)):
            raise ProviderNetworkPolicyError from None

    async def connect_tcp(  # noqa: ASYNC109 -- httpcore interface name
        self,
        host: str,
        port: int,
        timeout: float | None = None,  # noqa: ASYNC109 -- httpcore interface
        local_address: str | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        endpoint = self._destination.resolved.endpoint
        if host != endpoint.hostname or port != endpoint.port:
            raise ProviderNetworkPolicyError from None

        deadline = None
        if timeout is not None:
            deadline = asyncio.get_running_loop().time() + float(timeout)
        try:
            current = await self._policy.validate_for_connection_async(
                endpoint,
                timeout_seconds=timeout,
                dns_executor=self._dns_executor,
            )
        except TimeoutError:
            raise httpcore.ConnectTimeout from None
        if current != self._destination.resolved:
            current = cast(ResolvedProviderEndpoint, None)
            raise ProviderNetworkPolicyError from None
        current = cast(ResolvedProviderEndpoint, None)
        remaining_timeout = timeout
        if deadline is not None:
            remaining_timeout = deadline - asyncio.get_running_loop().time()
            if remaining_timeout <= 0:
                raise httpcore.ConnectTimeout from None
        return await self._delegate.connect_tcp(
            self._destination.selected_address,
            port,
            timeout=remaining_timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    async def connect_unix_socket(  # noqa: ASYNC109 -- httpcore interface name
        self,
        path: str,
        timeout: float | None = None,  # noqa: ASYNC109 -- httpcore interface
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        del path, timeout, socket_options
        raise ProviderNetworkPolicyError from None

    async def sleep(self, seconds: float) -> None:
        await self._delegate.sleep(seconds)


class _HttpcoreProviderPool:
    def __init__(
        self,
        *,
        destination: ProviderDestination,
        policy: ProviderEndpointPolicy,
        ssl_context: ssl.SSLContext,
        dns_executor: ProviderDnsExecutor,
    ) -> None:
        self._destination = destination
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=ssl_context,
            proxy=None,
            max_connections=100,
            max_keepalive_connections=20,
            keepalive_expiry=5.0,
            http1=True,
            http2=False,
            retries=0,
            network_backend=PinnedProviderNetworkBackend(
                policy=policy,
                destination=destination,
                delegate=_default_async_network_backend(),
                dns_executor=dns_executor,
            ),
        )

    async def post_json(
        self,
        *,
        path: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> ProviderHttpResponse:
        endpoint = self._destination.resolved.endpoint
        target = _joined_target(endpoint.path, path)
        request_headers: list[tuple[bytes, bytes]] = []
        content = b""
        response: httpcore.Response | None = None
        response_body = bytearray()
        response_headers: dict[str, str] = {}
        try:
            content = json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            request_headers = [
                (name.encode("ascii"), value.encode("ascii")) for name, value in headers.items()
            ]
            timeout = {
                "connect": timeout_seconds,
                "read": timeout_seconds,
                "write": timeout_seconds,
                "pool": timeout_seconds,
            }
            async with self._pool.stream(
                method=b"POST",
                url=httpcore.URL(
                    scheme=b"https",
                    host=endpoint.hostname.encode("ascii"),
                    port=endpoint.port,
                    target=target.encode("ascii"),
                ),
                headers=request_headers,
                content=content,
                extensions={"timeout": timeout},
            ) as response:
                for raw_name, raw_value in response.headers:
                    name = raw_name.decode("ascii", errors="strict").lower()
                    value = raw_value.decode("latin-1", errors="strict")
                    if name not in response_headers:
                        response_headers[name] = value
                async for chunk in response.aiter_stream():
                    if len(response_body) + len(chunk) > _MAX_RESPONSE_BYTES:
                        raise ValueError("Provider response is invalid")
                    response_body.extend(chunk)
                return ProviderHttpResponse(
                    status_code=response.status,
                    headers=response_headers,
                    body=bytes(response_body),
                )
        finally:
            request_headers.clear()
            content = b""
            response = None
            response_body.clear()
            response_headers.clear()
            payload = {}
            headers = {}
            target = ""

    async def aclose(self) -> None:
        await self._pool.aclose()


class _DefaultProviderPoolFactory:
    def __init__(
        self,
        policy: ProviderEndpointPolicy,
        dns_executor: ProviderDnsExecutor,
    ) -> None:
        self._policy = policy
        self._dns_executor = dns_executor

    def create(
        self,
        *,
        destination: ProviderDestination,
        ssl_context: ssl.SSLContext,
    ) -> ProviderPool:
        return _HttpcoreProviderPool(
            destination=destination,
            policy=self._policy,
            ssl_context=ssl_context,
            dns_executor=self._dns_executor,
        )


def _validated_request_path(path: str) -> str:
    if (
        type(path) is not str
        or not path.startswith("/")
        or path.startswith("//")
        or "?" in path
        or "#" in path
        or "\\" in path
        or len(path.encode("utf-8")) > _MAX_REQUEST_PATH_BYTES
    ):
        raise ProviderNetworkPolicyError from None
    segments = path.split("/")[1:]
    if not segments or any(not segment or segment in {".", ".."} for segment in segments):
        raise ProviderNetworkPolicyError from None
    if any(not segment.isascii() for segment in segments):
        raise ProviderNetworkPolicyError from None
    return path


def _joined_target(base_path: str, request_path: str) -> str:
    path = _validated_request_path(request_path)
    return f"{base_path}{path}" if base_path else path


def _ssl_context(ca_bundle: str | None) -> ssl.SSLContext:
    try:
        if ca_bundle is not None and (type(ca_bundle) is not str or "\x00" in ca_bundle):
            raise ValueError
        context = ssl.create_default_context(cafile=ca_bundle)
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        return context
    except Exception:
        raise ProviderNetworkPolicyError from None


class SecureProviderTransport:
    """SSRF-resistant, proxy-free, redirect-free Provider JSON transport."""

    def __init__(
        self,
        *,
        policy: ProviderEndpointPolicy,
        ca_bundle: str | None = None,
        pool_factory: ProviderPoolFactory | None = None,
        max_destination_pools: int = 32,
        dns_executor: ProviderDnsExecutor | None = None,
    ) -> None:
        if (
            type(policy) is not ProviderEndpointPolicy
            or type(max_destination_pools) is not int
            or not 1 <= max_destination_pools <= 1024
        ):
            raise ProviderNetworkPolicyError from None
        if dns_executor is not None and type(dns_executor) is not ProviderDnsExecutor:
            raise ProviderNetworkPolicyError from None
        self._policy = policy
        self._dns_executor = ProviderDnsExecutor() if dns_executor is None else dns_executor
        self._owns_dns_executor = dns_executor is None
        self._ssl_context = _ssl_context(ca_bundle)
        self._pool_factory = (
            _DefaultProviderPoolFactory(policy, self._dns_executor)
            if pool_factory is None
            else pool_factory
        )
        if not callable(getattr(self._pool_factory, "create", None)):
            raise ProviderNetworkPolicyError from None
        self._max_destination_pools = max_destination_pools
        self._pools: OrderedDict[ProviderDestination, _ProviderPoolEntry] = OrderedDict()
        self._pool_condition = asyncio.Condition()
        self._pool_waiters = 0
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._close_complete = False
        self._close_failure: BaseException | None = None

    async def _pool_for(self, destination: ProviderDestination) -> _ProviderPoolEntry:
        while True:
            evicted: tuple[ProviderDestination, _ProviderPoolEntry] | None = None
            async with self._pool_condition:
                if self._closed:
                    raise RuntimeError("Provider transport is closed")
                entry = self._pools.get(destination)
                if entry is not None and not entry.closing:
                    entry.leases += 1
                    self._pools.move_to_end(destination)
                    return entry
                if entry is None and len(self._pools) < self._max_destination_pools:
                    entry = _ProviderPoolEntry(
                        pool=self._pool_factory.create(
                            destination=destination,
                            ssl_context=self._ssl_context,
                        ),
                        leases=1,
                    )
                    self._pools[destination] = entry
                    return entry
                if entry is None:
                    for candidate_destination, candidate in self._pools.items():
                        if candidate.leases == 0 and not candidate.closing:
                            candidate.closing = True
                            evicted = (candidate_destination, candidate)
                            break
                if evicted is None:
                    self._pool_waiters += 1
                    try:
                        await self._pool_condition.wait()
                    finally:
                        self._pool_waiters -= 1
                    continue

            evicted_destination, evicted_entry = evicted
            cancellation_requested = False
            close_error: BaseException | None = None
            try:
                cancellation_requested = await _finish_cleanup(evicted_entry.pool.aclose())
            except BaseException as error:
                close_error = error
            finally:
                async with self._pool_condition:
                    if self._pools.get(evicted_destination) is evicted_entry:
                        del self._pools[evicted_destination]
                    self._pool_condition.notify_all()
            if cancellation_requested:
                raise asyncio.CancelledError from None
            if close_error is not None:
                raise close_error

    async def _release_pool(self, entry: _ProviderPoolEntry) -> None:
        async with self._pool_condition:
            if entry.leases <= 0:
                raise AssertionError("provider pool lease state is invalid")
            entry.leases -= 1
            if entry.leases == 0:
                self._pool_condition.notify_all()

    async def post_json(
        self,
        *,
        endpoint: CanonicalProviderEndpoint,
        path: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> ProviderHttpResponse:
        path = _validated_request_path(path)
        if (
            type(endpoint) is not CanonicalProviderEndpoint
            or type(headers) is not dict
            or type(payload) is not dict
            or type(timeout_seconds) not in {int, float}
            or isinstance(timeout_seconds, bool)
            or not 0 < float(timeout_seconds) <= 600
        ):
            raise ProviderNetworkPolicyError from None
        try:
            if any(type(name) is not str or name.lower() == "host" for name in headers):
                raise ValueError
        except Exception:
            raise ProviderNetworkPolicyError from None
        timeout_value = float(timeout_seconds)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_value
        try:
            resolved = await self._policy.validate_for_connection_async(
                endpoint,
                timeout_seconds=timeout_value,
                dns_executor=self._dns_executor,
            )
        except TimeoutError:
            raise httpcore.ConnectTimeout from None
        destination = ProviderDestination(
            resolved=resolved,
            selected_address=resolved.addresses[0],
        )
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise httpcore.ConnectTimeout from None
        try:
            async with asyncio.timeout(remaining):
                entry = await self._pool_for(destination)
        except TimeoutError:
            raise httpcore.PoolTimeout from None
        try:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise httpcore.PoolTimeout from None
            try:
                async with asyncio.timeout(remaining):
                    return await entry.pool.post_json(
                        path=path,
                        headers=headers,
                        payload=payload,
                        timeout_seconds=remaining,
                    )
            except TimeoutError:
                raise httpcore.ReadTimeout from None
        finally:
            if await _finish_cleanup(self._release_pool(entry)):
                raise asyncio.CancelledError from None

    @staticmethod
    def _consume_close_task_result(task: asyncio.Task[None]) -> None:
        if not task.cancelled():
            task.exception()

    async def _run_close_cleanup(self) -> None:
        failure: BaseException | None = None
        try:
            await self._close_all_pools()
        except BaseException as error:
            failure = error
        if self._owns_dns_executor:
            try:
                await self._dns_executor.aclose()
            except BaseException as error:
                if failure is None:
                    failure = error
                else:
                    failure = BaseExceptionGroup(
                        "Provider transport cleanup failed",
                        [failure, error],
                    )
        async with self._pool_condition:
            self._close_failure = failure
            self._close_complete = True
            self._close_task = None
            self._pool_condition.notify_all()
        if failure is not None:
            raise failure

    async def _close_all_pools(self) -> None:
        failures: list[BaseException] = []
        cancellation_requested = False
        while True:
            entries_to_close: list[tuple[ProviderDestination, _ProviderPoolEntry]] = []
            async with self._pool_condition:
                for destination, entry in self._pools.items():
                    if entry.leases == 0 and not entry.closing:
                        entry.closing = True
                        entries_to_close.append((destination, entry))
                if not entries_to_close:
                    if not self._pools:
                        break
                    self._pool_waiters += 1
                    try:
                        await self._pool_condition.wait()
                    finally:
                        self._pool_waiters -= 1
                    continue

            for destination, entry in entries_to_close:
                try:
                    cancellation_requested = (
                        await _finish_cleanup(entry.pool.aclose()) or cancellation_requested
                    )
                except BaseException as error:
                    failures.append(error)
                finally:
                    async with self._pool_condition:
                        if self._pools.get(destination) is entry:
                            del self._pools[destination]
                        self._pool_condition.notify_all()
        if cancellation_requested:
            raise asyncio.CancelledError from None
        if failures:
            raise ExceptionGroup(
                "Provider transport cleanup failed",
                [RuntimeError(type(failure).__name__) for failure in failures],
            )

    async def aclose(self) -> None:
        close_task: asyncio.Task[None] | None = None
        close_failure: BaseException | None = None
        async with self._pool_condition:
            self._closed = True
            self._pool_condition.notify_all()
            if self._close_complete:
                close_failure = self._close_failure
            else:
                if self._close_task is None:
                    self._close_task = asyncio.create_task(self._run_close_cleanup())
                    self._close_task.add_done_callback(self._consume_close_task_result)
                close_task = self._close_task
        if close_task is not None:
            await asyncio.shield(close_task)
        elif close_failure is not None:
            raise close_failure


__all__ = [
    "PinnedProviderNetworkBackend",
    "ProviderDestination",
    "ProviderHttpResponse",
    "SecureProviderTransport",
]
