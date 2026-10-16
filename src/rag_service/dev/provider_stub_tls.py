"""Runtime-only TLS material generator for the development Provider stub."""

from __future__ import annotations

import argparse
import ipaddress
import os
import re
import tempfile
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Never

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

_HOSTNAME_PATTERN = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)(?:\.(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?))*$"
)


_AUTHORITY_NAME = x509.Name(
    [x509.NameAttribute(NameOID.COMMON_NAME, "RAG development Provider CA")]
)
_CERTIFICATE_LIFETIME = timedelta(days=7)
# Without a kept CA key nothing can re-sign under the CA, so it only has to
# outlive one certificate. With one, it is what running clients keep trusting
# across renewals, and its expiry would be a client restart nobody schedules.
_SHORT_AUTHORITY_LIFETIME = timedelta(days=30)
_RENEWABLE_AUTHORITY_LIFETIME = timedelta(days=3650)


def _utc_now() -> datetime:
    return datetime.now(UTC)


class ProviderStubTlsError(Exception):
    """Safe configuration failure for development TLS generation."""


@dataclass(frozen=True, slots=True)
class ProviderStubTlsPaths:
    ca_certificate: Path
    server_certificate: Path
    server_private_key: Path


def _atomic_write(
    path: Path,
    payload: bytes,
    *,
    mode: int,
    owner_uid: int | None,
    owner_gid: int | None,
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, mode)
        if owner_uid is not None and owner_gid is not None:
            os.fchown(descriptor, owner_uid, owner_gid)
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        os.chmod(path, mode)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            temporary.unlink()


def _existing_material_is_usable(
    paths: ProviderStubTlsPaths, hostname: str, now: datetime, *, renewing: bool
) -> bool:
    """Report whether the material on disk can still serve `hostname`.

    Regenerating on every start is what makes reuse worth having: the CA is
    loaded once by long-running clients, so a fresh CA silently invalidates
    every already-running consumer while the server keeps answering. The check
    is deliberately stricter than "the files exist" — a half-written, expired,
    or foreign-signed pair has to be replaced rather than trusted. When
    `renewing`, a certificate past half its life counts as used up, so the
    renewal loop replaces it well before a missed run could let it lapse.
    """
    try:
        authority = x509.load_pem_x509_certificate(paths.ca_certificate.read_bytes())
        certificate = x509.load_pem_x509_certificate(paths.server_certificate.read_bytes())
        serialization.load_pem_private_key(paths.server_private_key.read_bytes(), password=None)
    except Exception:
        return False
    if not (certificate.not_valid_before_utc <= now < certificate.not_valid_after_utc):
        return False
    if renewing:
        lifetime = certificate.not_valid_after_utc - certificate.not_valid_before_utc
        if certificate.not_valid_after_utc - now < lifetime / 2:
            return False
    if not (authority.not_valid_before_utc <= now < authority.not_valid_after_utc):
        return False
    try:
        # Signature verification, not an issuer-name comparison: the CA subject
        # is a fixed string, so a regenerated CA with a new key still matches by
        # name while being unable to vouch for the old certificate.
        certificate.verify_directly_issued_by(authority)
    except Exception:
        return False
    try:
        names = certificate.extensions.get_extension_for_class(
            x509.SubjectAlternativeName
        ).value.get_values_for_type(x509.DNSName)
    except x509.ExtensionNotFound:
        return False
    return hostname in names


def _load_signing_authority(
    ca_certificate: Path, ca_private_key: Path, now: datetime
) -> tuple[x509.Certificate, RSAPrivateKey] | None:
    """Return the kept CA and its key when they can still sign, else None."""
    try:
        authority = x509.load_pem_x509_certificate(ca_certificate.read_bytes())
        key = serialization.load_pem_private_key(ca_private_key.read_bytes(), password=None)
    except Exception:
        return None
    if not isinstance(key, RSAPrivateKey):
        return None
    if not (authority.not_valid_before_utc <= now < authority.not_valid_after_utc):
        return None
    if key.public_key().public_numbers() != authority.public_key().public_numbers():  # type: ignore[union-attr]
        return None
    return authority, key


def _build_authority(key: RSAPrivateKey, now: datetime, lifetime: timedelta) -> x509.Certificate:
    return (
        x509.CertificateBuilder()
        .subject_name(_AUTHORITY_NAME)
        .issuer_name(_AUTHORITY_NAME)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + lifetime)
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(key, hashes.SHA256())
    )


def _build_certificate(
    hostname: str, server_key: RSAPrivateKey, authority_key: RSAPrivateKey, now: datetime
) -> x509.Certificate:
    return (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hostname)]))
        .issuer_name(_AUTHORITY_NAME)
        .public_key(server_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=1))
        .not_valid_after(now + _CERTIFICATE_LIFETIME)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.DNSName(hostname),
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                ]
            ),
            critical=False,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                content_commitment=False,
                key_encipherment=True,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .sign(authority_key, hashes.SHA256())
    )


def _private_key_bytes(key: RSAPrivateKey) -> bytes:
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


def generate_provider_stub_tls(
    ca_output_directory: Path,
    server_output_directory: Path,
    *,
    hostname: str,
    owner_uid: int | None = None,
    owner_gid: int | None = None,
    reuse_existing: bool = False,
    ca_key_directory: Path | None = None,
    clock: Callable[[], datetime] = _utc_now,
) -> ProviderStubTlsPaths:
    if (
        not isinstance(ca_output_directory, Path)
        or not isinstance(server_output_directory, Path)
        or ca_output_directory == server_output_directory
        or type(hostname) is not str
        or _HOSTNAME_PATTERN.fullmatch(hostname) is None
        or (owner_uid is None) != (owner_gid is None)
        or (owner_uid is not None and (type(owner_uid) is not int or owner_uid < 0))
        or (owner_gid is not None and (type(owner_gid) is not int or owner_gid < 0))
        or type(reuse_existing) is not bool
        or (
            ca_key_directory is not None
            and (
                not isinstance(ca_key_directory, Path)
                or ca_key_directory in (ca_output_directory, server_output_directory)
            )
        )
    ):
        raise ProviderStubTlsError("Provider stub TLS configuration is invalid")
    paths = ProviderStubTlsPaths(
        ca_certificate=ca_output_directory / "ca.pem",
        server_certificate=server_output_directory / "cert.pem",
        server_private_key=server_output_directory / "key.pem",
    )
    renewing = ca_key_directory is not None
    now = clock()
    # A CA with no usable key on disk cannot renew anything, so it is replaced
    # now rather than at the first renewal, when it would surprise clients.
    kept = (
        _load_signing_authority(paths.ca_certificate, ca_key_directory / "ca-key.pem", now)
        if reuse_existing and ca_key_directory is not None
        else None
    )
    if (
        reuse_existing
        and (kept is not None or not renewing)
        and _existing_material_is_usable(paths, hostname, now, renewing=renewing)
    ):
        return paths
    write_options = {"owner_uid": owner_uid, "owner_gid": owner_gid}
    try:
        server_output_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        server_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        if kept is None:
            ca_output_directory.mkdir(parents=True, exist_ok=True, mode=0o755)
            authority_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            authority = _build_authority(
                authority_key,
                now,
                _RENEWABLE_AUTHORITY_LIFETIME if renewing else _SHORT_AUTHORITY_LIFETIME,
            )
            if ca_key_directory is not None:
                ca_key_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
                _atomic_write(
                    ca_key_directory / "ca-key.pem",
                    _private_key_bytes(authority_key),
                    mode=0o600,
                    **write_options,
                )
            _atomic_write(
                paths.ca_certificate,
                authority.public_bytes(serialization.Encoding.PEM),
                mode=0o644,
                **write_options,
            )
        else:
            # Renewal: the CA file is left untouched, byte for byte, because
            # running clients loaded it once and will not load it again.
            _, authority_key = kept
        certificate = _build_certificate(hostname, server_key, authority_key, now)
        # Key before certificate: nginx reads both on reload, and a new key
        # beside the old certificate fails the reload, which leaves the running
        # workers serving, rather than a pair that loads and mismatches.
        _atomic_write(
            paths.server_private_key,
            _private_key_bytes(server_key),
            mode=0o600,
            **write_options,
        )
        _atomic_write(
            paths.server_certificate,
            certificate.public_bytes(serialization.Encoding.PEM),
            mode=0o644,
            **write_options,
        )
        return paths
    except ProviderStubTlsError:
        raise
    except Exception:
        raise ProviderStubTlsError("Provider stub TLS generation failed") from None


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Never:
        del message
        raise ProviderStubTlsError("Provider stub TLS configuration is invalid") from None


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(prog="velox-provider-stub-tls")
    parser.add_argument("--ca-output-directory", type=Path, required=True)
    parser.add_argument("--server-output-directory", type=Path, required=True)
    parser.add_argument("--hostname", default="provider-stub")
    parser.add_argument("--owner-uid", type=int)
    parser.add_argument("--owner-gid", type=int)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--ca-key-directory", type=Path)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    try:
        parsed = _parser().parse_args(arguments)
        generate_provider_stub_tls(
            parsed.ca_output_directory,
            parsed.server_output_directory,
            hostname=parsed.hostname,
            owner_uid=parsed.owner_uid,
            owner_gid=parsed.owner_gid,
            reuse_existing=parsed.reuse_existing,
            ca_key_directory=parsed.ca_key_directory,
        )
        return 0
    except ProviderStubTlsError:
        return 1


__all__ = [
    "ProviderStubTlsError",
    "ProviderStubTlsPaths",
    "generate_provider_stub_tls",
    "main",
]
