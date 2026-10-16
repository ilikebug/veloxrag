import ipaddress
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import serialization

from rag_service.dev.provider_stub_tls import ProviderStubTlsPaths, generate_provider_stub_tls


def test_provider_stub_tls_generation_writes_private_runtime_material(tmp_path: Path) -> None:
    ca_directory = tmp_path / "ca"
    server_directory = tmp_path / "server"

    paths = generate_provider_stub_tls(
        ca_directory,
        server_directory,
        hostname="provider-stub",
    )

    assert paths.ca_certificate == ca_directory / "ca.pem"
    assert paths.server_certificate == server_directory / "cert.pem"
    assert paths.server_private_key == server_directory / "key.pem"
    assert not (ca_directory / "key.pem").exists()
    assert not (ca_directory / "cert.pem").exists()
    assert not (server_directory / "ca.pem").exists()
    assert stat.S_IMODE(paths.server_private_key.stat().st_mode) == 0o600
    assert stat.S_IMODE(paths.ca_certificate.stat().st_mode) == 0o644
    assert stat.S_IMODE(paths.server_certificate.stat().st_mode) == 0o644

    authority = x509.load_pem_x509_certificate(paths.ca_certificate.read_bytes())
    certificate = x509.load_pem_x509_certificate(paths.server_certificate.read_bytes())
    private_key = serialization.load_pem_private_key(
        paths.server_private_key.read_bytes(),
        password=None,
    )
    san = certificate.extensions.get_extension_for_class(x509.SubjectAlternativeName).value

    assert authority.subject == certificate.issuer
    assert certificate.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    ) == private_key.public_key().public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    assert set(san.get_values_for_type(x509.DNSName)) == {"provider-stub", "localhost"}
    assert san.get_values_for_type(x509.IPAddress) == [ipaddress.ip_address("127.0.0.1")]


def test_provider_stub_tls_generation_replaces_existing_files_without_leaking_permissions(
    tmp_path: Path,
) -> None:
    ca_directory = tmp_path / "ca"
    server_directory = tmp_path / "server"
    first = generate_provider_stub_tls(
        ca_directory,
        server_directory,
        hostname="provider-stub",
    )
    first_key = first.server_private_key.read_bytes()

    second = generate_provider_stub_tls(
        ca_directory,
        server_directory,
        hostname="provider-stub",
    )

    assert second.server_private_key.read_bytes() != first_key
    assert stat.S_IMODE(second.server_private_key.stat().st_mode) == 0o600


def test_reuse_existing_keeps_material_that_still_serves_the_hostname(tmp_path: Path) -> None:
    ca_directory = tmp_path / "ca"
    server_directory = tmp_path / "server"
    first = generate_provider_stub_tls(ca_directory, server_directory, hostname="embedding")
    original_authority = first.ca_certificate.read_bytes()
    original_certificate = first.server_certificate.read_bytes()

    generate_provider_stub_tls(
        ca_directory, server_directory, hostname="embedding", reuse_existing=True
    )

    # Byte-identical, not merely valid: a regenerated CA would still verify its
    # own new certificate while silently breaking clients that loaded the old one.
    assert first.ca_certificate.read_bytes() == original_authority
    assert first.server_certificate.read_bytes() == original_certificate


def test_reuse_existing_regenerates_for_a_different_hostname(tmp_path: Path) -> None:
    ca_directory = tmp_path / "ca"
    server_directory = tmp_path / "server"
    first = generate_provider_stub_tls(ca_directory, server_directory, hostname="provider-stub")
    original_certificate = first.server_certificate.read_bytes()

    generate_provider_stub_tls(
        ca_directory, server_directory, hostname="embedding", reuse_existing=True
    )

    certificate = x509.load_pem_x509_certificate(first.server_certificate.read_bytes())
    names = certificate.extensions.get_extension_for_class(
        x509.SubjectAlternativeName
    ).value.get_values_for_type(x509.DNSName)
    assert first.server_certificate.read_bytes() != original_certificate
    assert "embedding" in names


def test_reuse_existing_regenerates_when_the_certificate_is_unreadable(tmp_path: Path) -> None:
    ca_directory = tmp_path / "ca"
    server_directory = tmp_path / "server"
    first = generate_provider_stub_tls(ca_directory, server_directory, hostname="embedding")
    first.server_certificate.write_bytes(b"truncated")

    generate_provider_stub_tls(
        ca_directory, server_directory, hostname="embedding", reuse_existing=True
    )

    assert x509.load_pem_x509_certificate(first.server_certificate.read_bytes())


def test_reuse_existing_regenerates_when_the_certificate_is_foreign_signed(tmp_path: Path) -> None:
    ca_directory = tmp_path / "ca"
    server_directory = tmp_path / "server"
    first = generate_provider_stub_tls(ca_directory, server_directory, hostname="embedding")
    # Replace only the CA, as a partially rebuilt volume would.
    generate_provider_stub_tls(ca_directory, tmp_path / "other", hostname="embedding")
    stale_certificate = first.server_certificate.read_bytes()

    generate_provider_stub_tls(
        ca_directory, server_directory, hostname="embedding", reuse_existing=True
    )

    authority = x509.load_pem_x509_certificate(first.ca_certificate.read_bytes())
    certificate = x509.load_pem_x509_certificate(first.server_certificate.read_bytes())
    assert first.server_certificate.read_bytes() != stale_certificate
    # Verified rather than name-matched: the CA subject is a fixed string, so a
    # re-keyed CA compares equal by name while failing to vouch for the cert.
    certificate.verify_directly_issued_by(authority)


def _renew(tmp_path: Path, moment: datetime) -> ProviderStubTlsPaths:
    return generate_provider_stub_tls(
        tmp_path / "ca",
        tmp_path / "server",
        hostname="embedding",
        ca_key_directory=tmp_path / "ca-key",
        reuse_existing=True,
        clock=lambda: moment,
    )


def test_ca_key_directory_keeps_the_ca_key_out_of_the_shared_ca_directory(
    tmp_path: Path,
) -> None:
    paths = generate_provider_stub_tls(
        tmp_path / "ca",
        tmp_path / "server",
        hostname="embedding",
        ca_key_directory=tmp_path / "ca-key",
    )

    ca_key = tmp_path / "ca-key" / "ca-key.pem"
    assert stat.S_IMODE(ca_key.stat().st_mode) == 0o600
    assert sorted(path.name for path in (tmp_path / "ca").iterdir()) == ["ca.pem"]
    authority = x509.load_pem_x509_certificate(paths.ca_certificate.read_bytes())
    # Long-lived because renewal re-signs under it: clients load the CA once at
    # start, so a CA that expires is a restart nobody schedules.
    assert authority.not_valid_after_utc - authority.not_valid_before_utc > timedelta(days=3000)


def test_renewal_re_signs_the_certificate_under_the_same_ca_once_half_its_life_is_gone(
    tmp_path: Path,
) -> None:
    start = datetime.now(UTC)
    first = _renew(tmp_path, start)
    original_authority = first.ca_certificate.read_bytes()
    original_certificate = first.server_certificate.read_bytes()

    later = start + timedelta(days=5)
    _renew(tmp_path, later)

    # Byte-identical CA is the whole point: running clients keep trusting it.
    assert first.ca_certificate.read_bytes() == original_authority
    assert first.server_certificate.read_bytes() != original_certificate
    authority = x509.load_pem_x509_certificate(original_authority)
    certificate = x509.load_pem_x509_certificate(first.server_certificate.read_bytes())
    certificate.verify_directly_issued_by(authority)
    assert certificate.not_valid_after_utc > later + timedelta(days=3)


def test_renewal_leaves_a_certificate_with_more_than_half_its_life_alone(
    tmp_path: Path,
) -> None:
    start = datetime.now(UTC)
    first = _renew(tmp_path, start)
    original_certificate = first.server_certificate.read_bytes()

    _renew(tmp_path, start + timedelta(days=1))

    assert first.server_certificate.read_bytes() == original_certificate


def test_renewal_regenerates_everything_when_the_ca_key_is_missing(tmp_path: Path) -> None:
    # An install from before the CA key was kept has a CA nothing can sign under.
    first = generate_provider_stub_tls(tmp_path / "ca", tmp_path / "server", hostname="embedding")
    original_authority = first.ca_certificate.read_bytes()

    generate_provider_stub_tls(
        tmp_path / "ca",
        tmp_path / "server",
        hostname="embedding",
        ca_key_directory=tmp_path / "ca-key",
        reuse_existing=True,
    )

    assert first.ca_certificate.read_bytes() != original_authority
    authority = x509.load_pem_x509_certificate(first.ca_certificate.read_bytes())
    certificate = x509.load_pem_x509_certificate(first.server_certificate.read_bytes())
    certificate.verify_directly_issued_by(authority)
    assert (tmp_path / "ca-key" / "ca-key.pem").exists()


def test_without_a_ca_key_directory_the_ca_stays_short_lived(tmp_path: Path) -> None:
    paths = generate_provider_stub_tls(tmp_path / "ca", tmp_path / "server", hostname="embedding")

    authority = x509.load_pem_x509_certificate(paths.ca_certificate.read_bytes())
    assert authority.not_valid_after_utc - authority.not_valid_before_utc < timedelta(days=31)
