"""Runtime-only TLS material generator for the development Provider stub."""

from __future__ import annotations

import argparse
import ipaddress
import os
import re
import tempfile
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Never

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

_HOSTNAME_PATTERN = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)(?:\.(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?))*$"
)


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


def _existing_material_is_usable(paths: ProviderStubTlsPaths, hostname: str) -> bool:
    """Report whether the material on disk can still serve `hostname`.

    Regenerating on every start is what makes reuse worth having: the CA is
    loaded once by long-running clients, so a fresh CA silently invalidates
    every already-running consumer while the server keeps answering. The check
    is deliberately stricter than "the files exist" — a half-written, expired,
    or foreign-signed pair has to be replaced rather than trusted.
    """
    try:
        authority = x509.load_pem_x509_certificate(paths.ca_certificate.read_bytes())
        certificate = x509.load_pem_x509_certificate(paths.server_certificate.read_bytes())
        serialization.load_pem_private_key(paths.server_private_key.read_bytes(), password=None)
    except Exception:
        return False
    now = datetime.now(UTC)
    if not (certificate.not_valid_before_utc <= now < certificate.not_valid_after_utc):
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


def generate_provider_stub_tls(
    ca_output_directory: Path,
    server_output_directory: Path,
    *,
    hostname: str,
    owner_uid: int | None = None,
    owner_gid: int | None = None,
    reuse_existing: bool = False,
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
    ):
        raise ProviderStubTlsError("Provider stub TLS configuration is invalid")
    if reuse_existing:
        existing = ProviderStubTlsPaths(
            ca_certificate=ca_output_directory / "ca.pem",
            server_certificate=server_output_directory / "cert.pem",
            server_private_key=server_output_directory / "key.pem",
        )
        if _existing_material_is_usable(existing, hostname):
            return existing
    try:
        ca_output_directory.mkdir(parents=True, exist_ok=True, mode=0o755)
        server_output_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        now = datetime.now(UTC)
        authority_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        server_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        authority_name = x509.Name(
            [x509.NameAttribute(NameOID.COMMON_NAME, "RAG development Provider CA")]
        )
        server_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, hostname)])
        authority = (
            x509.CertificateBuilder()
            .subject_name(authority_name)
            .issuer_name(authority_name)
            .public_key(authority_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=1))
            .not_valid_after(now + timedelta(days=30))
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
            .sign(authority_key, hashes.SHA256())
        )
        certificate = (
            x509.CertificateBuilder()
            .subject_name(server_name)
            .issuer_name(authority_name)
            .public_key(server_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=1))
            .not_valid_after(now + timedelta(days=7))
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
        paths = ProviderStubTlsPaths(
            ca_certificate=ca_output_directory / "ca.pem",
            server_certificate=server_output_directory / "cert.pem",
            server_private_key=server_output_directory / "key.pem",
        )
        _atomic_write(
            paths.ca_certificate,
            authority.public_bytes(serialization.Encoding.PEM),
            mode=0o644,
            owner_uid=owner_uid,
            owner_gid=owner_gid,
        )
        _atomic_write(
            paths.server_certificate,
            certificate.public_bytes(serialization.Encoding.PEM),
            mode=0o644,
            owner_uid=owner_uid,
            owner_gid=owner_gid,
        )
        _atomic_write(
            paths.server_private_key,
            server_key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            ),
            mode=0o600,
            owner_uid=owner_uid,
            owner_gid=owner_gid,
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
