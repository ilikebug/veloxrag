"""Reusable SSRF policy for configured Provider HTTPS endpoints."""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
import unicodedata
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import Future as ConcurrentFuture
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final, Protocol, TypeVar, cast
from urllib.parse import SplitResult, quote, urlsplit

from rag_service.config import Environment

PROVIDER_ENDPOINT_POLICY_VERSION: Final = "provider-endpoint-v1"

_DEFAULT_HTTPS_PORT: Final = 443
_MAX_ENDPOINT_LENGTH: Final = 2048
_MAX_HOSTNAME_LENGTH: Final = 253
_MAX_PATH_LENGTH: Final = 1024
_ERROR_MESSAGE: Final = "Provider endpoint rejected"
_HEADER_NAME_PATTERN: Final = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_DNS_LABEL_PATTERN: Final = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_SECOND_DECODE_DOT_ESCAPE_PATTERN: Final = re.compile(r"%2e", flags=re.IGNORECASE)
_SECOND_DECODE_SEPARATOR_ESCAPE_PATTERN: Final = re.compile(
    r"%(?:2f|5c)",
    flags=re.IGNORECASE,
)
_SECOND_DECODE_CONTROL_OR_DELIMITER_ESCAPE_PATTERN: Final = re.compile(
    r"%(?:[01][0-9a-f]|7f|23|3f)",
    flags=re.IGNORECASE,
)
_ALLOWED_PROVIDER_HEADERS: Final = (
    ("http-referer", "HTTP-Referer"),
    ("x-openrouter-title", "X-OpenRouter-Title"),
    ("x-title", "X-Title"),
)
_CANONICAL_PROVIDER_HEADER_NAMES: Final = MappingProxyType(dict(_ALLOWED_PROVIDER_HEADERS))
_PROVIDER_HEADER_VALUE_LIMIT_BYTES: Final = MappingProxyType(
    {
        "http-referer": 2048,
        "x-openrouter-title": 120,
        "x-title": 120,
    }
)
_MAX_PROVIDER_HEADERS_TOTAL_BYTES: Final = 4096
_MAX_RESOLVER_ANSWERS: Final = 64
_CLOUD_METADATA_ADDRESSES: Final = frozenset(
    {
        ipaddress.ip_address("169.254.169.254"),
        ipaddress.ip_address("169.254.170.2"),
        ipaddress.ip_address("100.100.100.200"),
        ipaddress.ip_address("192.0.0.192"),
        ipaddress.ip_address("fd00:ec2::254"),
    }
)

IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
_DnsResult = TypeVar("_DnsResult")


class ProviderNetworkPolicyError(Exception):
    """Stable sanitized error for rejected Provider network configuration."""

    __slots__ = ()

    code: Final = "PROVIDER_ENDPOINT_REJECTED"

    def __init__(self) -> None:
        super().__init__(_ERROR_MESSAGE)


@dataclass(frozen=True, slots=True, repr=False)
class CanonicalProviderEndpoint:
    """Pure canonical form used for persistence and later connection validation."""

    url: str
    hostname: str
    port: int
    path: str
    policy_version: str = PROVIDER_ENDPOINT_POLICY_VERSION

    def __repr__(self) -> str:
        return "CanonicalProviderEndpoint(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class ResolvedProviderEndpoint:
    """Canonical endpoint plus the complete validated answer set for one resolution."""

    endpoint: CanonicalProviderEndpoint
    addresses: tuple[str, ...]

    def __repr__(self) -> str:
        return "ResolvedProviderEndpoint(<redacted>)"


class ProviderAddressResolver(Protocol):
    """Resolver boundary injectable for deterministic policy and transport tests."""

    def resolve(self, hostname: str, port: int) -> Iterable[str]: ...


class ProviderDnsExecutor:
    """Dedicated bounded executor whose capacity follows actual resolver completion."""

    __slots__ = (
        "_active",
        "_closed",
        "_condition",
        "_executor",
        "_max_workers",
        "_shutdown",
    )

    def __init__(self, *, max_workers: int = 8) -> None:
        if type(max_workers) is not int or not 1 <= max_workers <= 64:
            raise ProviderNetworkPolicyError from None
        self._max_workers = max_workers
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="rag-provider-dns",
        )
        self._condition = asyncio.Condition()
        self._active = 0
        self._closed = False
        self._shutdown = False

    @property
    def closed(self) -> bool:
        return self._closed

    @staticmethod
    def _consume_release_task(task: asyncio.Task[None]) -> None:
        if not task.cancelled():
            task.exception()

    async def _release_capacity(self) -> None:
        async with self._condition:
            if self._active <= 0:
                raise AssertionError("provider DNS capacity state is invalid")
            self._active -= 1
            self._condition.notify_all()

    def _completed_on_loop(self) -> None:
        task = asyncio.create_task(self._release_capacity())
        task.add_done_callback(self._consume_release_task)

    async def run(
        self,
        operation: Callable[[], _DnsResult],
        *,
        timeout_seconds: float,
    ) -> _DnsResult:
        if (
            not callable(operation)
            or type(timeout_seconds) not in {int, float}
            or isinstance(timeout_seconds, bool)
            or not 0 < float(timeout_seconds) <= 600
        ):
            raise ProviderNetworkPolicyError from None
        loop = asyncio.get_running_loop()
        deadline = loop.time() + float(timeout_seconds)
        async with self._condition:
            while self._active >= self._max_workers:
                if self._closed:
                    raise ProviderNetworkPolicyError from None
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise TimeoutError from None
                async with asyncio.timeout(remaining):
                    await self._condition.wait()
            if self._closed:
                raise ProviderNetworkPolicyError from None
            self._active += 1

        future: ConcurrentFuture[_DnsResult] | None = None
        try:
            future = self._executor.submit(operation)
        except Exception:
            await self._release_capacity()
            raise ProviderNetworkPolicyError from None

        def release_when_done(_future: ConcurrentFuture[_DnsResult]) -> None:
            del _future
            with suppress(RuntimeError):
                loop.call_soon_threadsafe(self._completed_on_loop)

        future.add_done_callback(release_when_done)
        remaining = deadline - loop.time()
        if remaining <= 0:
            future.cancel()
            raise TimeoutError from None
        wrapped = asyncio.wrap_future(future)
        async with asyncio.timeout(remaining):
            return await wrapped

    async def aclose(self) -> None:
        async with self._condition:
            self._closed = True
            self._condition.notify_all()
            while self._active > 0:
                await self._condition.wait()
            if self._shutdown:
                return
            self._shutdown = True
        self._executor.shutdown(wait=True, cancel_futures=True)


_SHARED_DNS_EXECUTOR: ProviderDnsExecutor | None = None


def _shared_dns_executor() -> ProviderDnsExecutor:
    global _SHARED_DNS_EXECUTOR
    executor = _SHARED_DNS_EXECUTOR
    if executor is None or executor.closed:
        executor = ProviderDnsExecutor()
        _SHARED_DNS_EXECUTOR = executor
    return executor


@dataclass(frozen=True, slots=True)
class _SystemAddressResolver:
    def resolve(self, hostname: str, port: int) -> Iterable[str]:
        answers = socket.getaddrinfo(
            hostname,
            port,
            family=socket.AF_UNSPEC,
            type=socket.SOCK_STREAM,
            proto=socket.IPPROTO_TCP,
        )
        return tuple(cast(str, answer[4][0]) for answer in answers)


def _canonical_hostname(raw_hostname: str) -> tuple[str, bool]:
    if "%" in raw_hostname:
        raise ValueError
    address: IPAddress | None = None
    with suppress(ValueError):
        address = ipaddress.ip_address(raw_hostname)
    if address is not None:
        return address.compressed.lower(), address.version == 6

    hostname = raw_hostname.rstrip(".")
    if not hostname:
        raise ValueError
    try:
        hostname = hostname.encode("idna").decode("ascii").lower()
    except UnicodeError:
        raise ValueError from None
    if len(hostname) > _MAX_HOSTNAME_LENGTH:
        raise ValueError
    labels = hostname.split(".")
    if any(not _DNS_LABEL_PATTERN.fullmatch(label) for label in labels):
        raise ValueError
    return hostname, False


def _decode_path_segment(raw_segment: str) -> str:
    decoded_bytes = bytearray()
    position = 0
    while position < len(raw_segment):
        character = raw_segment[position]
        if character != "%":
            decoded_bytes.extend(character.encode("utf-8"))
            position += 1
            continue
        if position + 2 >= len(raw_segment):
            raise ValueError
        hexadecimal = raw_segment[position + 1 : position + 3]
        if any(digit not in "0123456789abcdefABCDEF" for digit in hexadecimal):
            raise ValueError
        decoded_bytes.append(int(hexadecimal, 16))
        position += 3
    try:
        return decoded_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        raise ValueError from None


def _has_ambiguous_second_decode_path_semantics(decoded_segment: str) -> bool:
    if _SECOND_DECODE_SEPARATOR_ESCAPE_PATTERN.search(
        decoded_segment
    ) or _SECOND_DECODE_CONTROL_OR_DELIMITER_ESCAPE_PATTERN.search(decoded_segment):
        return True
    dot_decoded_segment = _SECOND_DECODE_DOT_ESCAPE_PATTERN.sub(".", decoded_segment)
    return dot_decoded_segment in {".", ".."}


def _canonical_path(raw_path: str) -> str:
    if len(raw_path) > _MAX_PATH_LENGTH or "\\" in raw_path or "//" in raw_path:
        raise ValueError
    if any(ord(character) < 32 or ord(character) == 127 for character in raw_path):
        raise ValueError
    if raw_path and not raw_path.startswith("/"):
        raise ValueError

    canonical_segments: list[str] = []
    for raw_segment in raw_path.split("/"):
        decoded_segment = unicodedata.normalize("NFC", _decode_path_segment(raw_segment))
        if decoded_segment in {".", ".."}:
            raise ValueError
        if _has_ambiguous_second_decode_path_semantics(decoded_segment):
            raise ValueError
        if "/" in decoded_segment or "\\" in decoded_segment:
            raise ValueError
        if any(ord(character) < 32 or ord(character) == 127 for character in decoded_segment):
            raise ValueError
        canonical_segments.append(
            quote(
                decoded_segment,
                safe="-._~",
                encoding="utf-8",
                errors="strict",
            )
        )

    canonical = "/".join(canonical_segments)
    if canonical == "/":
        return ""
    canonical = canonical.rstrip("/")
    if len(canonical) > _MAX_PATH_LENGTH:
        raise ValueError
    return canonical


def _split_endpoint(raw_url: str) -> SplitResult:
    if type(raw_url) is not str or not raw_url or len(raw_url) > _MAX_ENDPOINT_LENGTH:
        raise ValueError
    if "?" in raw_url or "#" in raw_url or "\\" in raw_url:
        raise ValueError
    if any(character.isspace() or ord(character) < 32 for character in raw_url):
        raise ValueError
    try:
        parsed = urlsplit(raw_url)
    except ValueError:
        raise ValueError from None
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        raise ValueError
    if parsed.username is not None or parsed.password is not None:
        raise ValueError
    if parsed.query or parsed.fragment:
        raise ValueError
    return parsed


def _validate_provider_endpoint_url(raw_url: str) -> CanonicalProviderEndpoint:
    parsed = _split_endpoint(raw_url)
    raw_hostname = parsed.hostname
    if raw_hostname is None:
        raise ValueError
    hostname, is_ipv6 = _canonical_hostname(raw_hostname)
    try:
        parsed_port = parsed.port
    except ValueError:
        raise ValueError from None
    port = _DEFAULT_HTTPS_PORT if parsed_port is None else parsed_port
    if not 1 <= port <= 65535:
        raise ValueError
    if parsed.port is None and parsed.netloc.endswith(":"):
        raise ValueError
    path = _canonical_path(parsed.path)
    authority_host = f"[{hostname}]" if is_ipv6 else hostname
    authority = authority_host if port == _DEFAULT_HTTPS_PORT else f"{authority_host}:{port}"
    canonical_url = f"https://{authority}{path}"
    if len(canonical_url) > _MAX_ENDPOINT_LENGTH:
        raise ValueError
    return CanonicalProviderEndpoint(
        url=canonical_url,
        hostname=hostname,
        port=port,
        path=path,
    )


def validate_provider_endpoint_url(raw_url: str) -> CanonicalProviderEndpoint:
    """Validate and canonicalize a Provider Base URL without performing DNS I/O.

    Query strings are deliberately excluded from the Base URL contract: credentials and
    routing options belong in their dedicated configuration fields, and request paths are
    joined onto an unambiguous canonical base path.
    """

    failed = False
    endpoint: CanonicalProviderEndpoint | None = None
    try:
        endpoint = _validate_provider_endpoint_url(raw_url)
    except Exception:
        failed = True
    raw_url = ""
    if failed or endpoint is None:
        raise ProviderNetworkPolicyError
    return endpoint


def _validate_http_referer(value: str) -> None:
    if "?" in value or "#" in value or " " in value:
        raise ValueError
    try:
        parsed = urlsplit(value)
    except ValueError:
        raise ValueError from None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ValueError
    if parsed.username is not None or parsed.password is not None:
        raise ValueError
    raw_hostname = parsed.hostname
    if raw_hostname is None:
        raise ValueError
    _, is_ipv6 = _canonical_hostname(raw_hostname)
    if is_ipv6 != parsed.netloc.startswith("["):
        raise ValueError
    if is_ipv6:
        port_suffix = parsed.netloc[parsed.netloc.index("]") + 1 :]
        explicit_port_text = port_suffix[1:] if port_suffix.startswith(":") else None
        if port_suffix and explicit_port_text is None:
            raise ValueError
    else:
        _, separator, port_text = parsed.netloc.rpartition(":")
        explicit_port_text = port_text if separator else None
    if explicit_port_text is not None and (
        not 1 <= len(explicit_port_text) <= 5 or not explicit_port_text.isdecimal()
    ):
        raise ValueError
    try:
        port = parsed.port
    except ValueError:
        raise ValueError from None
    if port is not None and not 1 <= port <= 65535:
        raise ValueError


def _validate_provider_headers(headers: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(headers, Mapping):
        raise ValueError
    values_by_normalized_name: dict[str, str] = {}
    seen_names: set[str] = set()
    total_wire_bytes = 0
    for name, value in headers.items():
        if type(name) is not str or type(value) is not str:
            raise ValueError
        normalized_name = name.lower()
        if (
            not _HEADER_NAME_PATTERN.fullmatch(name)
            or normalized_name in seen_names
            or normalized_name not in _CANONICAL_PROVIDER_HEADER_NAMES
        ):
            raise ValueError
        canonical_name = _CANONICAL_PROVIDER_HEADER_NAMES[normalized_name]
        maximum_length = _PROVIDER_HEADER_VALUE_LIMIT_BYTES[normalized_name]
        if not value or len(value) > maximum_length:
            raise ValueError
        if (
            value[0].isspace()
            or value[-1].isspace()
            or not value.isascii()
            or any(ord(character) < 32 or ord(character) == 127 for character in value)
        ):
            raise ValueError
        if canonical_name == "HTTP-Referer":
            _validate_http_referer(value)
        total_wire_bytes += len(canonical_name) + 2 + len(value) + 2
        if total_wire_bytes > _MAX_PROVIDER_HEADERS_TOTAL_BYTES:
            raise ValueError
        seen_names.add(normalized_name)
        values_by_normalized_name[normalized_name] = value
    validated = {
        canonical_name: values_by_normalized_name[normalized_name]
        for normalized_name, canonical_name in _ALLOWED_PROVIDER_HEADERS
        if normalized_name in values_by_normalized_name
    }
    return MappingProxyType(validated)


def validate_provider_headers(headers: Mapping[str, str]) -> Mapping[str, str]:
    """Return canonical configured headers only when every key is explicitly allowed."""

    failed = False
    validated: Mapping[str, str] | None = None
    try:
        validated = _validate_provider_headers(headers)
    except Exception:
        failed = True
    headers = {}
    if failed or validated is None:
        raise ProviderNetworkPolicyError
    return validated


def _address_sort_key(address: IPAddress) -> tuple[int, int]:
    return address.version, int(address)


def _is_permitted_address(address: IPAddress, *, allow_private_targets: bool) -> bool:
    policy_address: IPAddress = address
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped is not None:
            policy_address = address.ipv4_mapped
        elif (
            address.is_site_local
            or address.is_link_local
            or address.sixtofour is not None
            or address.teredo is not None
        ):
            return False
    if policy_address in _CLOUD_METADATA_ADDRESSES:
        return False
    if policy_address.is_link_local or policy_address.is_multicast or policy_address.is_unspecified:
        return False
    if policy_address.is_loopback:
        return allow_private_targets
    if policy_address.is_reserved:
        return False
    if allow_private_targets and policy_address.is_private:
        return True
    return policy_address.is_global


@dataclass(frozen=True, slots=True)
class ProviderEndpointPolicy:
    """Versioned persistence/runtime policy for Provider network destinations."""

    environment: Environment
    allow_private_targets: bool = False
    resolver: ProviderAddressResolver = field(
        default_factory=_SystemAddressResolver,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        failed = False
        environment: Environment | None = None
        try:
            environment = Environment(self.environment)
            if type(self.allow_private_targets) is not bool:
                raise ValueError
            if self.allow_private_targets and environment is Environment.PRODUCTION:
                raise ValueError
            if not callable(getattr(self.resolver, "resolve", None)):
                raise ValueError
        except Exception:
            failed = True
        if failed or environment is None:
            raise ProviderNetworkPolicyError
        object.__setattr__(self, "environment", environment)

    @property
    def policy_version(self) -> str:
        return PROVIDER_ENDPOINT_POLICY_VERSION

    def validate_url(self, raw_url: str) -> CanonicalProviderEndpoint:
        """Run the reusable pure URL-validation phase."""

        return validate_provider_endpoint_url(raw_url)

    def validate_headers(self, headers: Mapping[str, str]) -> Mapping[str, str]:
        """Run reusable Provider configured-header validation."""

        return validate_provider_headers(headers)

    def validate_for_persistence(self, raw_url: str) -> ResolvedProviderEndpoint:
        """Canonicalize and resolve every current address before persisting an endpoint."""

        endpoint = self.validate_url(raw_url)
        raw_url = ""
        return self.validate_for_connection(endpoint)

    def validate_for_connection(
        self,
        endpoint: CanonicalProviderEndpoint,
    ) -> ResolvedProviderEndpoint:
        """Repeat resolution immediately before a transport opens each new connection."""

        failed = False
        resolved: ResolvedProviderEndpoint | None = None
        try:
            resolved = self._resolve_and_validate(endpoint)
        except Exception:
            failed = True
        if failed or resolved is None:
            raise ProviderNetworkPolicyError
        return resolved

    async def validate_for_connection_async(
        self,
        endpoint: CanonicalProviderEndpoint,
        *,
        timeout_seconds: float | None = None,
        dns_executor: ProviderDnsExecutor | None = None,
    ) -> ResolvedProviderEndpoint:
        """Resolve without blocking the event loop, optionally within a connect deadline."""

        if timeout_seconds is not None and (
            type(timeout_seconds) not in {int, float}
            or isinstance(timeout_seconds, bool)
            or not 0 < float(timeout_seconds) <= 600
        ):
            raise ProviderNetworkPolicyError from None
        executor = _shared_dns_executor() if dns_executor is None else dns_executor
        if type(executor) is not ProviderDnsExecutor:
            raise ProviderNetworkPolicyError from None
        effective_timeout = 600.0 if timeout_seconds is None else float(timeout_seconds)
        return await executor.run(
            lambda: self.validate_for_connection(endpoint),
            timeout_seconds=effective_timeout,
        )

    def _resolve_and_validate(
        self,
        endpoint: CanonicalProviderEndpoint,
    ) -> ResolvedProviderEndpoint:
        if (
            type(endpoint) is not CanonicalProviderEndpoint
            or endpoint.policy_version != self.policy_version
            or endpoint != validate_provider_endpoint_url(endpoint.url)
        ):
            raise ValueError

        try:
            literal_address = ipaddress.ip_address(endpoint.hostname)
        except ValueError:
            raw_answers = self._resolve_answers(endpoint.hostname, endpoint.port)
        else:
            raw_answers = (literal_address.compressed,)

        addresses: set[IPAddress] = set()
        for raw_address in raw_answers:
            if type(raw_address) is not str:
                raise ValueError
            try:
                address = ipaddress.ip_address(raw_address)
            except ValueError:
                raise ValueError from None
            if not _is_permitted_address(
                address,
                allow_private_targets=self.allow_private_targets,
            ):
                raise ValueError
            addresses.add(address)
        if not addresses:
            raise ValueError

        canonical_addresses = tuple(
            address.compressed for address in sorted(addresses, key=_address_sort_key)
        )
        return ResolvedProviderEndpoint(
            endpoint=endpoint,
            addresses=canonical_addresses,
        )

    def _resolve_answers(self, hostname: str, port: int) -> tuple[str, ...]:
        failed = False
        answers: list[str] = []
        raw_answers: Iterable[str] = ()
        answer_iterator: Iterator[str] | None = None
        try:
            raw_answers = self.resolver.resolve(hostname, port)
            answer_iterator = iter(raw_answers)
            for _ in range(_MAX_RESOLVER_ANSWERS + 1):
                try:
                    answer = next(answer_iterator)
                except StopIteration:
                    break
                answers.append(answer)
        except Exception:
            failed = True
        hostname = ""
        port = 0
        raw_answers = ()
        answer_iterator = None
        if failed or len(answers) > _MAX_RESOLVER_ANSWERS:
            raise ValueError
        return tuple(answers)


__all__ = [
    "PROVIDER_ENDPOINT_POLICY_VERSION",
    "CanonicalProviderEndpoint",
    "ProviderAddressResolver",
    "ProviderDnsExecutor",
    "ProviderEndpointPolicy",
    "ProviderNetworkPolicyError",
    "ResolvedProviderEndpoint",
    "validate_provider_endpoint_url",
    "validate_provider_headers",
]
