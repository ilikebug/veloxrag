import base64
import json
import os
import re
import stat
import subprocess
import tomllib
from pathlib import Path
from uuid import uuid4

import pytest
from pydantic import SecretStr, ValidationError

from rag_service.config import Environment, Settings

ROOT = Path(__file__).resolve().parents[2]


def _readme_bash_block(marker: str) -> str:
    readme = _operations_doc()
    marker_index = readme.index(marker)
    start = readme.rindex("```bash\n", 0, marker_index) + len("```bash\n")
    end = readme.index("\n```", marker_index)
    return readme[start:end]


def _readme_upload_stager() -> str:
    upload = (
        _operations_doc()
        .split("### 5. Upload", maxsplit=1)[1]
        .split("### 6. Job poll/retry", maxsplit=1)[0]
    )
    marker = "STAGED_DOCUMENT_PATH=$(python3 - \"${RAG_OPERATIONS_DIR}\" <<'PY'"
    assert marker in upload
    return upload.split(f"{marker}\n", maxsplit=1)[1].split("\nPY\n)", maxsplit=1)[0]


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(0o700)


def _dotenv_values() -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in (ROOT / ".env.example").read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, value = line.split("=", maxsplit=1)
        values[name] = value
    return values


def _smoke_library(tmp_path: Path) -> Path:
    script = (ROOT / "scripts/smoke_auth_metadata.sh").read_text()
    execution_markers = ("\ntrap cleanup EXIT", "\nmain() {")
    cut_points = [script.index(marker) for marker in execution_markers if marker in script]
    library = tmp_path / "smoke-auth-library.sh"
    library.write_text(script[: min(cut_points)])
    return library


def test_compose_uses_required_pinned_images() -> None:
    compose = (ROOT / "compose.yaml").read_text()

    assert "postgres:18.4-alpine" in compose
    assert "redis:8.8.0-alpine" in compose
    assert "qdrant/qdrant:v1.18.3" in compose
    assert "minio/minio:RELEASE.2025-09-07T16-13-09Z" in compose
    # `latest` is now the published default for the consumer-facing file; pinned
    # tags are what `make push` produces alongside it.
    assert "${VELOX_IMAGE_TAG:-latest}" in compose


def test_compose_runs_migration_api_and_worker_with_explicit_process_contracts() -> None:
    compose = (ROOT / "compose.yaml").read_text()

    # A published reference, not a local tag: downloading compose.yaml is meant to
    # be the whole install, so it must not depend on a source tree being present.
    # compose.build.yaml adds the build stanza back for development.
    assert (
        "image: ${VELOX_IMAGE:-docker.io/ilikebug/veloxrag}:${VELOX_IMAGE_TAG:-latest}" in compose
    )
    assert "build:" not in compose
    assert "build:" in (ROOT / "compose.build.yaml").read_text()
    assert 'command: ["alembic", "upgrade", "head"]' in compose
    assert 'command: ["uvicorn", "rag_service.main:app"' in compose
    assert 'command: ["velox-worker", "--health-file", "/tmp/velox-worker-health.json"' in compose
    assert '"velox-worker-health"' in compose
    assert '"--max-age-seconds"' in compose
    assert compose.count("condition: service_completed_successfully") >= 3
    assert "api:\n" in compose
    assert "worker:\n" in compose
    assert (
        "worker:\n" not in compose.split("api:\n", maxsplit=1)[1].split("worker:\n", maxsplit=1)[0]
    )


def test_compose_provider_stub_is_profiled_tls_only_and_not_host_published() -> None:
    compose = (ROOT / "compose.yaml").read_text()
    runtime_section = compose.split("x-rag-runtime: &rag-runtime\n", maxsplit=1)[1].split(
        "\nservices:", maxsplit=1
    )[0]
    provider_section = compose.split("  provider-stub:\n", maxsplit=1)[1].split(
        "\n  api:", maxsplit=1
    )[0]

    assert "provider-tls-init:" in compose
    assert "provider-stub:" in compose
    assert compose.count('profiles: ["provider-stub"]') >= 2
    assert "velox-provider-stub-tls" in compose
    assert "velox-provider-stub" in compose
    # This file is the local single-user stack, where downloading it and running
    # `docker compose up` is the whole setup, so both local-only switches default
    # on. What keeps that safe is the production refusal asserted below, not the
    # default itself.
    assert (
        "RAG_PROVIDER_ALLOW_PRIVATE_TARGETS: ${RAG_PROVIDER_ALLOW_PRIVATE_TARGETS:-true}" in compose
    )
    assert "RAG_LOCAL_TRUSTED_AUTH: ${RAG_LOCAL_TRUSTED_AUTH:-true}" in compose
    assert (
        "RAG_PROVIDER_CA_BUNDLE: ${RAG_PROVIDER_CA_BUNDLE:-/run/rag/embedding-ca/ca.pem}" in compose
    )
    assert "required: false" in compose
    assert '"127.0.0.1:8443:8443"' not in compose
    assert '"8443:8443"' not in compose
    assert "provider_ca:/run/rag/provider-ca:ro" in runtime_section
    assert "provider_server_tls" not in runtime_section
    assert "provider_ca:/run/rag/provider-ca:ro" in provider_section
    assert "provider_server_tls:/run/rag/provider-server-tls:ro" in provider_section
    assert "provider_tls:" not in compose


def test_minio_initialization_uses_the_safe_bounded_application_entrypoint() -> None:
    compose = (ROOT / "compose.yaml").read_text()
    init_section = compose.split("\n  minio-init:\n", maxsplit=1)[1].split(
        "\n  migrate:", maxsplit=1
    )[0]

    assert "velox-minio-init" in init_section
    assert "RAG_MINIO_ACCESS_KEY:" in init_section
    assert "RAG_MINIO_SECRET_KEY:" in init_section
    assert "RAG_MINIO_INIT_MAX_ATTEMPTS:" in init_section
    assert "MC_HOST_local:" not in init_section
    assert "mc alias set" not in init_section


def test_compose_persists_datastores_and_keeps_api_independent_from_worker() -> None:
    compose = (ROOT / "compose.yaml").read_text()

    for volume in ("postgres_data", "redis_data", "qdrant_data", "minio_data"):
        assert f"  {volume}:" in compose
    # Anchored on a newline: the bare two-space form also matches a nested
    # `api:` key such as another service's depends_on, which would extend this
    # section over unrelated services.
    api_section = compose.split("\n  api:\n", maxsplit=1)[1].split("\n  worker:\n", maxsplit=1)[0]
    assert "worker" not in api_section
    assert 'command: ["redis-server", "--appendonly", "yes"]' in compose


def test_compose_api_environment_respects_rag_overrides() -> None:
    compose = (ROOT / "compose.yaml").read_text()

    expected_environment = {
        "RAG_ENVIRONMENT": "${RAG_ENVIRONMENT:-local}",
        "RAG_API_HOST": "${RAG_API_HOST:-0.0.0.0}",
        "RAG_API_PORT": "${RAG_API_PORT:-8000}",
        "RAG_DATABASE_URL": (
            "${RAG_DATABASE_URL:-postgresql+psycopg://rag:change-me-local@postgres:5432/rag}"
        ),
        "RAG_REDIS_URL": "${RAG_REDIS_URL:-redis://redis:6379/0}",
        "RAG_QDRANT_URL": "${RAG_QDRANT_URL:-http://qdrant:6333}",
        "RAG_MINIO_URL": "${RAG_MINIO_URL:-http://minio:9000}",
        "RAG_MINIO_ACCESS_KEY": "${RAG_MINIO_ACCESS_KEY:-rag-dev}",
        "RAG_MINIO_SECRET_KEY": "${RAG_MINIO_SECRET_KEY:-change-me-local}",
        "RAG_MINIO_BUCKET": "${RAG_MINIO_BUCKET:-rag-documents}",
        "RAG_MAX_UPLOAD_BYTES": "${RAG_MAX_UPLOAD_BYTES:-52428800}",
        "RAG_UPLOAD_BUFFER_BYTES": "${RAG_UPLOAD_BUFFER_BYTES:-1048576}",
        "RAG_MINIO_MULTIPART_PART_SIZE_BYTES": ("${RAG_MINIO_MULTIPART_PART_SIZE_BYTES:-5242880}"),
        "RAG_MINIO_OPERATION_TIMEOUT_SECONDS": "${RAG_MINIO_OPERATION_TIMEOUT_SECONDS:-300.0}",
        "RAG_READINESS_TIMEOUT_SECONDS": "${RAG_READINESS_TIMEOUT_SECONDS:-2.0}",
        "RAG_SHUTDOWN_TIMEOUT_SECONDS": "${RAG_SHUTDOWN_TIMEOUT_SECONDS:-2.0}",
        "RAG_ADMIN_KEY_HMAC_SECRET": (
            "${RAG_ADMIN_KEY_HMAC_SECRET:-local-dev-admin-hmac-secret-not-for-production-01}"
        ),
        "RAG_AGENT_KEY_HMAC_SECRET": (
            "${RAG_AGENT_KEY_HMAC_SECRET:-local-dev-agent-hmac-secret-not-for-production-02}"
        ),
        "RAG_DEFAULT_PAGE_SIZE": "${RAG_DEFAULT_PAGE_SIZE:-20}",
        "RAG_MAX_PAGE_SIZE": "${RAG_MAX_PAGE_SIZE:-100}",
        "RAG_MAX_API_KEY_REQUESTS_PER_MINUTE": ("${RAG_MAX_API_KEY_REQUESTS_PER_MINUTE:-10000}"),
        "RAG_MAX_API_KEY_CONCURRENCY": "${RAG_MAX_API_KEY_CONCURRENCY:-1000}",
    }

    for name, interpolation in expected_environment.items():
        assert f"  {name}: {interpolation}" in compose


def test_compose_publishes_ports_on_loopback_only() -> None:
    compose = (ROOT / "compose.yaml").read_text()

    # Asserted over every publication the file contains rather than a fixed list,
    # so that adding a service cannot quietly publish on all interfaces.
    publications = re.findall(r"^\s+- \"(.+?:\d+)\"$", compose, re.MULTILINE)
    assert publications
    for publication in publications:
        assert publication.startswith("127.0.0.1:"), publication

    # Each host port is overridable: every one of these defaults is a port a
    # developer machine commonly already has taken.
    expected = {
        "RAG_API_HOST_PORT": 8000,
        "RAG_POSTGRES_HOST_PORT": 5432,
        "RAG_QDRANT_HOST_PORT": 6333,
        "RAG_QDRANT_GRPC_HOST_PORT": 6334,
        "RAG_REDIS_HOST_PORT": 6379,
        "RAG_MINIO_HOST_PORT": 9000,
        "RAG_MINIO_CONSOLE_HOST_PORT": 9001,
    }
    for name, port in expected.items():
        assert f'"127.0.0.1:${{{name}:-{port}}}:{port}"' in compose
        # The container side stays fixed, so a literal mapping would mean the
        # override was dropped on one of the two sides.
        assert f'"{port}:{port}"' not in compose
        assert f'"127.0.0.1:{port}:{port}"' not in compose


def test_dockerfile_uses_pinned_python_and_uv_images() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text()

    assert "ghcr.io/astral-sh/uv:0.11.20" in dockerfile
    assert "python:3.12.13-slim-bookworm" in dockerfile
    assert ":latest" not in dockerfile


def test_dockerfile_caches_locked_dependencies_before_copying_application_source() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text()

    dependency_sync = "RUN uv sync --frozen --no-dev --no-install-project"
    project_sync = "RUN uv sync --frozen --no-dev"
    assert dockerfile.index(dependency_sync) < dockerfile.index("COPY src ./src")
    assert dockerfile.index("COPY src ./src") < dockerfile.rindex(project_sync)


def test_ci_runs_quality_integration_and_stack_smoke_checks() -> None:
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()

    assert "make check" in workflow
    assert "make test" in workflow
    assert "make test-integration" in workflow
    assert "make coverage" in workflow
    assert workflow.index("make test-integration") < workflow.index("make coverage")
    assert "make up-provider" in workflow
    assert "make smoke" in workflow
    assert "uv-version: 0.11.20" in workflow
    assert "docker/setup-qemu-action@v3" in workflow
    assert "--platform linux/arm64" in workflow


def test_make_starts_shared_image_services_without_parallel_build_races() -> None:
    makefile = (ROOT / "Makefile").read_text()

    assert makefile.count("docker compose -f compose.yaml -f compose.build.yaml build api") >= 2
    # Both start paths pass the build override, so the image they run is the one
    # just built rather than the published tag compose.yaml points at.
    assert "-f compose.yaml -f compose.build.yaml up -d --no-build" in makefile
    assert (
        "-f compose.yaml -f compose.build.yaml --profile provider-stub up -d --no-build" in makefile
    )


def test_coverage_gate_combines_fresh_unit_and_integration_data() -> None:
    makefile = (ROOT / "Makefile").read_text()

    assert "COVERAGE_FILE=$(UNIT_COVERAGE_FILE) uv run coverage erase" in makefile
    assert "COVERAGE_FILE=$(INTEGRATION_COVERAGE_FILE) uv run coverage erase" in makefile
    assert "COVERAGE_FILE=$(COMBINED_COVERAGE_FILE) uv run coverage erase" in makefile
    assert "coverage combine --keep .coverage.unit .coverage.integration" in makefile
    assert "coverage report --fail-under=80" in makefile
    assert "verify:" in makefile
    assert "$(MAKE) coverage" in makefile


def test_local_auth_secrets_are_distinct_and_not_application_defaults() -> None:
    values = _dotenv_values()
    defaults = Settings(_env_file=None)

    admin_secret = values["RAG_ADMIN_KEY_HMAC_SECRET"]
    agent_secret = values["RAG_AGENT_KEY_HMAC_SECRET"]

    assert len(admin_secret) >= 32
    assert len(agent_secret) >= 32
    assert admin_secret != agent_secret
    assert admin_secret != defaults.admin_key_hmac_secret.get_secret_value()
    assert agent_secret != defaults.agent_key_hmac_secret.get_secret_value()

    with pytest.raises(ValidationError, match="production secrets"):
        Settings(
            environment=Environment.PRODUCTION,
            database_url=SecretStr("postgresql+psycopg://rag:strong-password@postgres:5432/rag"),
            minio_access_key="prod-access-key",
            minio_secret_key=SecretStr("prod-secret-key"),
            admin_key_hmac_secret=SecretStr(admin_secret),
            agent_key_hmac_secret=SecretStr(agent_secret),
            provider_credential_keyring=SecretStr(
                json.dumps({"2026-07": base64.b64encode(bytes(range(32))).decode()})
            ),
            provider_credential_active_key_version="2026-07",
            _env_file=None,
        )


def test_env_example_documents_auth_limits_and_development_only_secrets() -> None:
    env_example = (ROOT / ".env.example").read_text()
    values = _dotenv_values()

    assert "development only" in env_example.lower()
    assert "RAG_ADMIN_KEY_HMAC_SECRET" in values
    assert "RAG_AGENT_KEY_HMAC_SECRET" in values
    assert values["RAG_DEFAULT_PAGE_SIZE"] == "20"
    assert values["RAG_MAX_PAGE_SIZE"] == "100"
    assert values["RAG_MAX_API_KEY_REQUESTS_PER_MINUTE"] == "10000"
    assert values["RAG_MAX_API_KEY_CONCURRENCY"] == "1000"


def test_runtime_environment_example_documents_ingestion_settings() -> None:
    values = _dotenv_values()

    assert values["RAG_MAX_UPLOAD_BYTES"] == str(50 * 1024 * 1024)
    assert values["RAG_UPLOAD_BUFFER_BYTES"] == str(1024 * 1024)
    assert values["RAG_MINIO_MULTIPART_PART_SIZE_BYTES"] == str(5 * 1024 * 1024)
    assert values["RAG_MINIO_OPERATION_TIMEOUT_SECONDS"] == "300.0"
    assert values["RAG_MINIO_INIT_MAX_ATTEMPTS"] == "30"
    assert values["RAG_WORKER_POLL_INTERVAL_SECONDS"] == "1.0"
    assert values["RAG_WORKER_LEASE_SECONDS"] == "60.0"
    assert values["RAG_WORKER_HEARTBEAT_SECONDS"] == "15.0"
    assert values["RAG_WORKER_MAX_ATTEMPTS"] == "5"
    assert values["RAG_WORKER_RETRY_INITIAL_SECONDS"] == "5.0"
    assert values["RAG_WORKER_RETRY_MAX_SECONDS"] == "300.0"
    assert values["RAG_WORKER_MAX_CONCURRENCY"] == "4"
    assert values["RAG_WORKER_HEALTH_FILE"] == "/tmp/velox-worker-health.json"
    assert values["RAG_ORPHAN_OBJECT_GRACE_SECONDS"] == "86400"
    assert values["RAG_QDRANT_CONNECT_TIMEOUT_SECONDS"] == "5.0"
    assert values["RAG_QDRANT_REQUEST_TIMEOUT_SECONDS"] == "30.0"
    assert values["RAG_PROVIDER_ALLOW_PRIVATE_TARGETS"] == "false"
    assert "RAG_PROVIDER_CA_BUNDLE" in values
    assert "RAG_PROVIDER_STUB_AUTHORIZATION_SHA256" in values
    assert "RAG_PROVIDER_CREDENTIAL_KEYRING" in values
    assert values["RAG_PROVIDER_CREDENTIAL_ACTIVE_KEY_VERSION"] == "local-v1"
    assert values["VELOX_IMAGE_TAG"] == "local"


def test_project_locks_runtime_dependencies_and_future_process_entry_points() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dependencies = project["project"]["dependencies"]
    dev_dependencies = project["dependency-groups"]["dev"]
    scripts = project["project"]["scripts"]

    for dependency in dependencies:
        assert "==" in dependency
    assert {
        "cryptography",
        "python-multipart",
        "qdrant-client",
        "prometheus-client",
    } <= {dependency.split("==", maxsplit=1)[0] for dependency in dependencies}
    assert "trustme" in {dependency.split("==", maxsplit=1)[0] for dependency in dev_dependencies}
    assert scripts["velox-worker"] == "rag_service.jobs.worker:main"
    assert scripts["velox-worker-health"] == "rag_service.jobs.worker:health_main"
    assert scripts["velox-provider-stub"] == "rag_service.dev.provider_stub:main"
    assert scripts["velox-provider-stub-tls"] == "rag_service.dev.provider_stub_tls:main"
    assert scripts["velox-minio-init"] == "rag_service.dev.minio_init:main"


def _operations_doc() -> str:
    """The operator documentation these contracts guard spans two files.

    Credential preparation stayed in the README while the step-by-step flow
    moved to the operations guide, so both are checked together: asserting
    against either one alone would silently stop covering half the commands.
    """

    return "\n".join(
        (
            (ROOT / "README.md").read_text(),
            (ROOT / "docs" / "api-operations.md").read_text(),
        )
    )


def test_readme_documents_authenticated_metadata_operations() -> None:
    readme = _operations_doc()
    readme_lower = readme.lower()
    readme_compact = " ".join(readme.split())

    required_fragments = (
        "docker compose run --rm api velox-admin admin-key create",
        "docker compose run --rm api velox-admin admin-key list",
        "docker compose run --rm api velox-admin admin-key revoke",
        "authorization: bearer",
        "idempotency-key",
        "if-match",
        "etag",
        "/v1/admin/api-keys",
        "/v1/knowledge-bases",
        "rag_admin_key_hmac_secret",
        "rag_agent_key_hmac_secret",
        "minio",
        "qdrant",
    )
    for fragment in required_fragments:
        assert fragment in readme_lower

    # Each of these guards a security fact the documentation has to keep stating.
    assert "one-time" in readme_lower
    assert "cannot be recovered" in readme_lower
    assert "do not log" in readme_lower
    assert "independent" in readme_lower
    assert "invalidates all existing Admin tokens" in readme_compact
    assert "does not invalidate Agent tokens" in readme_compact
    assert "invalidates all existing Agent tokens only" in readme_compact
    assert "`ingest`, `retrieve`, `answer`, `manage`" in readme
    assert "`admin` is not an Agent capability" in readme
    assert "<etag-from-patch-response>" in readme
    assert "<etag-from-filter-schema-response>" in readme
    assert "not a vector database" in readme_lower
    assert "not yet supported" in readme_lower


def test_readme_normalizes_kb_id_before_url_interpolation() -> None:
    readme = _operations_doc()
    normalization = """KB_ID=$(python3 - <<'PY'
import os
from uuid import UUID

print(UUID(os.environ["KB_ID"]))
PY
) || exit 1
export KB_ID"""

    assert normalization in readme
    assert readme.index(normalization) < readme.index("${KB_ID}")


def test_readme_upload_stages_from_no_follow_descriptors_before_curl() -> None:
    readme = _operations_doc()
    upload = readme.split("### 5. Upload", maxsplit=1)[1].split(
        "### 6. Job poll/retry", maxsplit=1
    )[0]
    curl_form = (
        '-F "file=@${STAGED_DOCUMENT_PATH};type=${STAGED_DOCUMENT_MIME};filename=${DOCUMENT_NAME}"'
    )
    curl_form_index = upload.index(curl_form)
    required_validation = (
        "os.O_DIRECTORY",
        "os.O_NOFOLLOW",
        "dir_fd=root_fd",
        "os.fstat",
        "stat.S_ISREG",
        "st_nlink",
        "st_mtime_ns",
        "st_ctime_ns",
        "os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW",
        "0o600",
        'not 1 <= len(name.encode("utf-8")) <= 255',
        're.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) is None',
        '".md": ("upload-source.md", "text/markdown")',
        '".markdown": ("upload-source.markdown", "text/markdown")',
        '".txt": ("upload-source.txt", "text/plain")',
    )

    for fragment in required_validation:
        assert fragment in upload
        assert upload.index(fragment) < curl_form_index
    assert "${DOCUMENT_PATH}" not in curl_form
    assert "${DOCUMENT_ROOT}" not in curl_form
    assert "\nPY\n) || exit 1\n" in upload
    assert "pwd -P" in upload
    # The guide has to say that a symlink path component is refused deliberately,
    # so nobody "fixes" it by resolving the link.
    assert "symbolic link as a path component is refused" in upload

    compile(_readme_upload_stager(), "README upload stager", "exec")
    subprocess.run(
        ["bash", "-n"],
        input=_readme_bash_block("STAGED_DOCUMENT_PATH="),
        text=True,
        check=True,
    )


def test_readme_upload_stager_copies_once_and_rejects_unsafe_sources(
    tmp_path: Path,
) -> None:
    stager = _readme_upload_stager()
    staged_names = ("upload-source.md", "upload-source.markdown", "upload-source.txt")

    def run_stager(
        root: Path, name: str, *, precreate: str | None = None
    ) -> tuple[subprocess.CompletedProcess[str], Path]:
        operations_dir = Path(f"/tmp/rag-operations.{uuid4().hex[:6]}")
        operations_dir.mkdir(mode=0o700)
        if precreate is not None:
            staged_file = operations_dir / precreate
            staged_file.write_text("sentinel")
            staged_file.chmod(0o600)
        try:
            result = subprocess.run(
                ["python3", "-", str(operations_dir)],
                input=stager,
                text=True,
                capture_output=True,
                env={
                    "PATH": os.defpath,
                    "DOCUMENT_ROOT": str(root),
                    "DOCUMENT_NAME": name,
                },
                check=False,
            )
            return result, operations_dir
        except BaseException:
            for staged_name in staged_names:
                (operations_dir / staged_name).unlink(missing_ok=True)
            operations_dir.rmdir()
            raise

    def cleanup(operations_dir: Path) -> None:
        for staged_name in staged_names:
            (operations_dir / staged_name).unlink(missing_ok=True)
        operations_dir.rmdir()

    trusted_root = tmp_path / "trusted"
    trusted_root.mkdir(mode=0o700)
    source = trusted_root / "document.md"
    source.write_bytes(b"validated source\n")
    source.chmod(0o600)
    valid_result, valid_operations_dir = run_stager(trusted_root, source.name)
    try:
        assert valid_result.returncode == 0, valid_result.stderr
        staged = Path(valid_result.stdout.strip())
        assert staged == valid_operations_dir / "upload-source.md"
        assert stat.S_IMODE(staged.stat().st_mode) == 0o600
        source.write_bytes(b"replacement after staging\n")
        assert staged.read_bytes() == b"validated source\n"
    finally:
        cleanup(valid_operations_dir)

    target = trusted_root / "target.md"
    target.write_text("target")
    symlink = trusted_root / "symlink.md"
    symlink.symlink_to(target.name)
    symlink_result, symlink_operations_dir = run_stager(trusted_root, symlink.name)
    try:
        assert symlink_result.returncode != 0
    finally:
        cleanup(symlink_operations_dir)

    symlink_root = tmp_path / "trusted-link"
    symlink_root.symlink_to(trusted_root, target_is_directory=True)
    symlink_root_result, symlink_root_operations_dir = run_stager(symlink_root, source.name)
    try:
        assert symlink_root_result.returncode != 0
    finally:
        cleanup(symlink_root_operations_dir)

    hardlink_source = trusted_root / "hardlink-source.md"
    hardlink_source.write_text("hard linked")
    hardlink = trusted_root / "hardlink.md"
    os.link(hardlink_source, hardlink)
    hardlink_result, hardlink_operations_dir = run_stager(trusted_root, hardlink.name)
    try:
        assert hardlink_result.returncode != 0
    finally:
        cleanup(hardlink_operations_dir)

    oversized = trusted_root / "oversized.txt"
    with oversized.open("wb") as output:
        output.truncate((50 * 1024 * 1024) + 1)
    oversized_result, oversized_operations_dir = run_stager(trusted_root, oversized.name)
    try:
        assert oversized_result.returncode != 0
    finally:
        cleanup(oversized_operations_dir)

    source.write_bytes(b"must not replace staged content")
    collision_result, collision_operations_dir = run_stager(
        trusted_root,
        source.name,
        precreate="upload-source.md",
    )
    try:
        assert collision_result.returncode != 0
        assert (collision_operations_dir / "upload-source.md").read_text() == "sentinel"
    finally:
        cleanup(collision_operations_dir)

    trusted_root.chmod(0o770)
    writable_result, writable_operations_dir = run_stager(trusted_root, source.name)
    try:
        assert writable_result.returncode != 0
    finally:
        cleanup(writable_operations_dir)


def test_readme_python_heredocs_and_mktemp_assignments_fail_closed() -> None:
    readme = _operations_doc()
    lines = readme.splitlines()

    for index, line in enumerate(lines):
        if "python3 " not in line or "<<'PY'" not in line:
            continue
        closing_index = lines.index("PY", index + 1)
        if "=$(python3 " in line:
            assert lines[closing_index + 1] == ") || exit 1"
        else:
            assert line.endswith("<<'PY' || exit 1")

    mktemp_assignments = [line for line in lines if "$(umask 077 && mktemp" in line]
    assert len(mktemp_assignments) >= 3
    assert all(line.endswith(") || exit 1") for line in mktemp_assignments)


def test_readme_single_file_cli_outputs_cleanup_and_propagate_failure(
    tmp_path: Path,
) -> None:
    readme = _operations_doc()
    expected_contracts = (
        (
            "ADMIN_BOOTSTRAP_FILE",
            "cleanup_admin_bootstrap",
            r"^/tmp/velox-admin-created\.[A-Za-z0-9]{6}$",
        ),
        (
            "REPAIR_RESPONSE",
            "cleanup_repair_response",
            r"^/tmp/rag-generation-repair\.[A-Za-z0-9]{6}$",
        ),
    )
    for variable, cleanup_function, safe_pattern in expected_contracts:
        block = _readme_bash_block(f"{variable}=")
        lines = block.splitlines()
        assignment_index = next(
            index for index, line in enumerate(lines) if line.startswith(f"{variable}=")
        )
        assert lines[assignment_index].endswith(") || exit 1")
        assert lines[assignment_index + 1] == f"trap {cleanup_function} EXIT"
        assert safe_pattern in block
        assert "*" not in "\n".join(
            line for line in lines if line.lstrip().startswith(("rm ", "rmdir "))
        )

    admin_creation = _readme_bash_block("ADMIN_BOOTSTRAP_FILE=")
    admin_cleanup = _readme_bash_block("cleanup_admin_bootstrap || exit 1")
    assert admin_creation != admin_cleanup
    assert "cleanup_admin_bootstrap || exit 1" not in admin_creation
    assert "trap - EXIT HUP INT TERM" not in admin_creation
    assert "cleanup_admin_bootstrap || exit 1" in admin_cleanup
    assert "trap - EXIT HUP INT TERM" in admin_cleanup
    admin_section = readme.split("## Admin and Agent keys", maxsplit=1)[1].split(
        "Inspect security metadata, or revoke an Admin key", maxsplit=1
    )[0]
    assert "dedicated Bash" in admin_section
    assert "0600" in admin_section
    # The ordering is the point: the one-time response has to reach the password
    # manager before the cleanup that deletes it.
    assert "approved password manager" in admin_section
    assert admin_section.index("ADMIN_BOOTSTRAP_FILE=") < admin_section.index(
        "approved password manager"
    )
    assert admin_section.index("approved password manager") < admin_section.index(
        "cleanup_admin_bootstrap || exit 1"
    )

    repair_block = _readme_bash_block("REPAIR_RESPONSE=")
    assert "cleanup_repair_response || exit 1" in repair_block
    assert "trap - EXIT HUP INT TERM" in repair_block

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    admin_path = Path(f"/tmp/velox-admin-created.{uuid4().hex[:6]}")
    repair_path = Path(f"/tmp/rag-generation-repair.{uuid4().hex[:6]}")
    assert not admin_path.exists()
    assert not repair_path.exists()
    _write_executable(
        fake_bin / "mktemp",
        """#!/bin/sh
case "$1" in
  /tmp/velox-admin-created.XXXXXX) target="$FAKE_ADMIN_PATH" ;;
  /tmp/rag-generation-repair.XXXXXX) target="$FAKE_REPAIR_PATH" ;;
  *) exit 2 ;;
esac
umask 077
: >"$target" || exit 1
printf '%s\n' "$target"
""",
    )
    _write_executable(fake_bin / "docker", "#!/bin/sh\nprintf 'nonsecret-response'\nexit 1\n")
    environment = {
        "PATH": f"{fake_bin}{os.pathsep}{os.defpath}",
        "FAKE_ADMIN_PATH": str(admin_path),
        "FAKE_REPAIR_PATH": str(repair_path),
    }

    cases = (
        ("ADMIN_BOOTSTRAP_FILE=", admin_path),
        ("REPAIR_RESPONSE=", repair_path),
    )
    for marker, exact_path in cases:
        try:
            result = subprocess.run(
                ["bash"],
                input=_readme_bash_block(marker),
                text=True,
                capture_output=True,
                env=environment,
                check=False,
            )
            assert result.returncode != 0
            assert not exact_path.exists()
            assert "nonsecret-response" not in result.stdout
            assert "nonsecret-response" not in result.stderr
        finally:
            exact_path.unlink(missing_ok=True)


def test_readme_repair_extracts_a_validated_job_id_before_cleanup(tmp_path: Path) -> None:
    repair_block = _readme_bash_block("REPAIR_RESPONSE=")
    parser_start = "REPAIR_JOB_ID=$(python3 - \"${REPAIR_RESPONSE}\" <<'PY'"
    required_parser_fragments = (
        'generation_id = UUID(document["generation_id"])',
        'job_id = UUID(document["job_id"])',
        'status = document["status"]',
        'generation_id != UUID(os.environ["GENERATION_ID"])',
        'status != "queued"',
        "print(job_id)",
        ") || exit 1",
        "export REPAIR_JOB_ID",
    )
    assert parser_start in repair_block
    for fragment in required_parser_fragments:
        assert fragment in repair_block
    assert repair_block.index(parser_start) < repair_block.index(
        "cleanup_repair_response || exit 1"
    )
    assert "print(document)" not in repair_block
    assert "cat " not in repair_block

    readme = _operations_doc()
    repair_section = readme.split("## Same-generation Qdrant repair", maxsplit=1)[1].split(
        "## Rotating and losing the provider credential keyring", maxsplit=1
    )[0]
    assert "/v1/jobs/${REPAIR_JOB_ID}" in repair_section

    generation_id = "11111111-1111-4111-8111-111111111111"
    job_id = "22222222-2222-4222-8222-222222222222"
    exact_path = Path(f"/tmp/rag-generation-repair.{uuid4().hex[:6]}")
    assert not exact_path.exists()
    fake_bin = tmp_path / "repair-bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "mktemp",
        """#!/bin/sh
umask 077
: >"$FAKE_REPAIR_PATH" || exit 1
printf '%s\n' "$FAKE_REPAIR_PATH"
""",
    )
    _write_executable(fake_bin / "docker", "#!/bin/sh\nprintf '%s\n' \"$FAKE_REPAIR_JSON\"\n")
    base_environment = {
        "PATH": f"{fake_bin}{os.pathsep}{os.defpath}",
        "FAKE_REPAIR_PATH": str(exact_path),
        "GENERATION_ID": generation_id,
    }

    cases = (
        (
            json.dumps({"generation_id": generation_id, "job_id": job_id, "status": "queued"}),
            0,
            f"repair_job_id={job_id}\n",
        ),
        (
            json.dumps({"generation_id": generation_id, "job_id": job_id, "status": "failed"}),
            1,
            "",
        ),
    )
    for response, expected_returncode, expected_stdout in cases:
        try:
            result = subprocess.run(
                ["bash"],
                input=(repair_block + "\nprintf 'repair_job_id=%s\\n' \"${REPAIR_JOB_ID}\"\n"),
                text=True,
                capture_output=True,
                env={**base_environment, "FAKE_REPAIR_JSON": response},
                check=False,
            )
            assert result.returncode == expected_returncode
            assert result.stdout == expected_stdout
            assert response not in result.stdout
            assert response not in result.stderr
            assert not exact_path.exists()
        finally:
            exact_path.unlink(missing_ok=True)


def test_readme_operations_setup_stops_when_mktemp_fails(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "mktemp", "#!/bin/sh\nexit 1\n")
    _write_executable(fake_bin / "python3", "#!/bin/sh\ncat >/dev/null\nexit 0\n")
    result = subprocess.run(
        ["bash"],
        input=_readme_bash_block("cleanup_rag_operations() {"),
        text=True,
        capture_output=True,
        env={
            "PATH": f"{fake_bin}{os.pathsep}{os.defpath}",
            "RAG_ADMIN_TOKEN": "test-admin-placeholder",
            "RAG_AGENT_TOKEN": "test-agent-placeholder",
        },
        check=False,
    )

    assert result.returncode != 0


def test_readme_registers_scoped_operations_cleanup_immediately() -> None:
    readme = _operations_doc()
    cleanup = readme.split("cleanup_rag_operations() {", maxsplit=1)[1].split(
        "\n}\n\nRAG_OPERATIONS_DIR=", maxsplit=1
    )[0]
    creation = 'RAG_OPERATIONS_DIR=$(umask 077 && mktemp -d "/tmp/rag-operations.XXXXXX") || exit 1'
    exit_trap = "trap cleanup_rag_operations EXIT"
    known_files = (
        "admin-auth.conf",
        "agent-auth.conf",
        "generation-request.json",
        "generation-response.json",
        "job-response.json",
        "model-profile-request.json",
        "model-profile-response.json",
        "provider-config-request.json",
        "provider-config-response.json",
        "provider-credential-request.json",
        "provider-credential-response.json",
        "retry-response.json",
        "search-request.json",
        "search-response.json",
        "upload-source.markdown",
        "upload-source.md",
        "upload-source.txt",
        "upload-response.json",
    )

    assert "cleanup_rag_operations() {" in readme
    assert f"{creation}\n{exit_trap}" in readme
    assert readme.index(exit_trap) < readme.index(
        ': "${RAG_ADMIN_TOKEN:?inject the Admin token through an approved secret source}"'
    )
    assert '[[ ! "${cleanup_dir}" =~ ^/tmp/rag-operations\\.[A-Za-z0-9]{6}$ ]]' in cleanup
    assert 'rm -f -- "${cleanup_dir}/${cleanup_name}"' in cleanup
    assert 'rmdir -- "${cleanup_dir}"' in cleanup
    assert 'rm -f -- "${RAG_OPERATIONS_DIR}"/*' not in readme
    assert "cleanup_rag_operations || exit 1" in readme
    assert "trap - EXIT HUP INT TERM" in readme
    for filename in known_files:
        assert filename in cleanup

    auth_block = (
        "cleanup_rag_operations() {"
        + readme.split("```bash\ncleanup_rag_operations() {", maxsplit=1)[1].split(
            "\n```", maxsplit=1
        )[0]
    )
    subprocess.run(["bash", "-n"], input=auth_block, text=True, check=True)


def test_operational_examples_do_not_commit_api_tokens() -> None:
    operational_text = "\n".join(
        (ROOT / path).read_text()
        for path in (
            ".env.example",
            "compose.yaml",
            "README.md",
            "scripts/smoke_auth_metadata.sh",
        )
    )

    token_pattern = r"rag_(?:adm|agent)_[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"
    assert re.search(token_pattern, operational_text) is None
    assert re.search(r"\bsk-[A-Za-z0-9_-]{20,}", operational_text) is None
    assert re.search(r"\bAIza[A-Za-z0-9_-]{20,}", operational_text) is None
    provider_assignment = (
        r"(?im)^(?:export\s+)?(?:OPENAI|OPENROUTER|ANTHROPIC|GOOGLE|GEMINI|COHERE)"
        r"[A-Z0-9_]*(?:API_KEY|TOKEN|SECRET)\s*=\s*['\"]?"
        r"(?!<|change-me|replace-me)[A-Za-z0-9]"
    )
    assert re.search(provider_assignment, operational_text) is None


def test_authenticated_smoke_has_required_safety_and_api_contracts() -> None:
    script_path = ROOT / "scripts/smoke_auth_metadata.sh"
    assert script_path.is_file()

    script = script_path.read_text()
    required_fragments = (
        "set -euo pipefail",
        "umask 077",
        "mktemp -d",
        "trap cleanup EXIT",
        "trap 'on_signal 129' HUP",
        "trap 'on_signal 130' INT",
        "trap 'on_signal 143' TERM",
        "velox-admin admin-key create",
        "POST /v1/admin/api-keys",
        "/v1/knowledge-bases",
        "Idempotency-Key:",
        "If-Match:",
        "PRECONDITION_FAILED",
        "RESOURCE_NOT_FOUND",
        "INVALID_API_KEY",
        "python3",
        "RAG authenticated metadata smoke test passed",
    )
    for fragment in required_fragments:
        assert fragment in script

    assert "jq" not in script
    assert "set -x" not in script
    assert '-H "Authorization: Bearer ${token}"' not in script
    assert '--config "${auth_config}"' in script
    assert "/tmp/rag-service-auth-smoke.lock" in script
    assert re.search(r"rag_(?:adm|agent)_[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", script) is None

    subprocess.run(["bash", "-n", str(script_path)], check=True)


def test_authenticated_smoke_keeps_bearer_token_out_of_curl_argv(tmp_path: Path) -> None:
    library = _smoke_library(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_curl = fake_bin / "curl"
    fake_curl.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        'printf \'%s\\n\' "$@" >"${FAKE_CURL_ARGV_FILE}"\n'
        "printf '200'\n"
    )
    fake_curl.chmod(0o700)
    argv_file = tmp_path / "curl-argv.txt"
    auth_file = tmp_path / "curl-auth.conf"
    output_file = tmp_path / "response.json"
    token = "rag_adm_public.secret-must-not-enter-curl-argv"
    environment = {
        "PATH": f"{fake_bin}:{os.environ.get('PATH', '/usr/bin:/bin')}",
        "FAKE_CURL_ARGV_FILE": str(argv_file),
    }

    subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; work_dir="$2"; write_auth_config "$3" "$4"; '
            'http_request GET /v1/example "$3" "$5" ""',
            "bash",
            str(library),
            str(tmp_path),
            str(auth_file),
            token,
            str(output_file),
        ],
        check=True,
        env=environment,
    )

    curl_argv = argv_file.read_text().splitlines()
    assert token not in argv_file.read_text()
    assert curl_argv[0] == "-q"
    assert "--config" in curl_argv
    assert auth_file.stat().st_mode & 0o777 == 0o600
    assert token in auth_file.read_text()


def test_authenticated_smoke_restricts_base_url_to_loopback(tmp_path: Path) -> None:
    library = _smoke_library(tmp_path)

    subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; validate_base_url http://localhost:8123; '
            "validate_base_url http://127.0.0.1:8000; "
            "validate_base_url 'http://[::1]:8000'; "
            "if validate_base_url https://example.com; then exit 1; fi",
            "bash",
            str(library),
        ],
        check=True,
    )


def test_authenticated_smoke_recovers_paginated_key_by_unique_name(tmp_path: Path) -> None:
    library = _smoke_library(tmp_path)
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_docker = fake_bin / "docker"
    fake_docker.write_text(
        r"""#!/usr/bin/env bash
set -eu
case " $* " in
  *" --cursor cursor-1 "*)
    printf '%s%s\n' \
      '{"items":[{"id":"00000000-0000-0000-0000-000000000099",' \
      '"name":"target-smoke-key"}],"next_cursor":null}'
    ;;
  *)
    printf '%s\n' '{"items":[],"next_cursor":"cursor-1"}'
    ;;
esac
"""
    )
    fake_docker.chmod(0o700)
    environment = {"PATH": f"{fake_bin}:{os.environ.get('PATH', '/usr/bin:/bin')}"}

    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; work_dir="$2"; find_cli_key_id_by_name agent-key target-smoke-key',
            "bash",
            str(library),
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.stdout.strip() == "00000000-0000-0000-0000-000000000099"


def test_makefile_exposes_authenticated_smoke_target() -> None:
    makefile = (ROOT / "Makefile").read_text()

    assert "smoke-auth:" in makefile
    assert "bash scripts/smoke_auth_metadata.sh" in makefile
    assert "smoke-auth" in makefile.splitlines()[0]


def test_makefile_disables_implicit_env_files_for_smoke_scripts() -> None:
    makefile = (ROOT / "Makefile").read_text()

    # Matched on the two properties that matter — compose never reads a dotenv
    # implicitly, and the target runs the script it claims to — rather than on the
    # whole recipe line, which also carries the RAG_BASE_URL derivation.
    for target, script in (
        ("smoke", "scripts/smoke_stack.sh"),
        ("smoke-auth", "scripts/smoke_auth_metadata.sh"),
    ):
        recipe = re.search(rf"^{target}:\n\t(.+)$", makefile, re.MULTILINE)
        assert recipe, target
        command = recipe.group(1)
        assert command.startswith("COMPOSE_DISABLE_ENV_FILE=1 "), command
        assert command.endswith(f"bash {script}"), command


def test_admin_cli_is_installed_as_project_entrypoint() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text()

    assert 'velox-admin = "rag_service.admin.cli:main"' in pyproject
