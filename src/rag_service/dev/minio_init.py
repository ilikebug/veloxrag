"""Bounded, secret-safe MinIO bucket initialization for Compose."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Protocol

from minio import Minio
from urllib3 import PoolManager, Timeout

from rag_service.config import Settings
from rag_service.infrastructure.probes import _validate_minio_url


class MinioInitializationError(Exception):
    """Safe failure raised when the local bucket cannot be initialized."""


class MinioInitializationClient(Protocol):
    def bucket_exists(self, bucket_name: str) -> bool: ...

    def make_bucket(self, bucket_name: str) -> None: ...

    def delete_bucket_policy(self, bucket_name: str) -> None: ...


def initialize_minio_bucket(
    client: MinioInitializationClient,
    *,
    bucket_name: str,
    max_attempts: int,
    retry_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    if (
        type(bucket_name) is not str
        or not bucket_name
        or type(max_attempts) is not int
        or max_attempts < 1
        or type(retry_seconds) not in {int, float}
        or isinstance(retry_seconds, bool)
        or retry_seconds < 0
        or not callable(sleep)
    ):
        raise MinioInitializationError("MinIO initialization configuration is invalid")

    for attempt in range(1, max_attempts + 1):
        try:
            if not client.bucket_exists(bucket_name):
                client.make_bucket(bucket_name)
            client.delete_bucket_policy(bucket_name)
            return
        except Exception:
            if attempt >= max_attempts:
                break
            sleep(float(retry_seconds))
    raise MinioInitializationError("MinIO initialization failed before the deadline")


def _max_attempts() -> int:
    try:
        value = int(os.environ.get("RAG_MINIO_INIT_MAX_ATTEMPTS", "30"))
    except ValueError:
        raise MinioInitializationError("MinIO initialization configuration is invalid") from None
    if not 1 <= value <= 3600:
        raise MinioInitializationError("MinIO initialization configuration is invalid")
    return value


def main() -> int:
    http_client: PoolManager | None = None
    try:
        settings = Settings()
        endpoint, secure = _validate_minio_url(settings.minio_url)
        http_client = PoolManager(
            timeout=Timeout(
                connect=settings.readiness_timeout_seconds,
                read=settings.readiness_timeout_seconds,
            ),
            retries=False,
        )
        client = Minio(
            endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key.get_secret_value(),
            secure=secure,
            http_client=http_client,
        )
        initialize_minio_bucket(
            client,
            bucket_name=settings.minio_bucket,
            max_attempts=_max_attempts(),
            retry_seconds=1,
        )
        return 0
    except Exception:
        return 1
    finally:
        if http_client is not None:
            http_client.clear()


__all__ = [
    "MinioInitializationClient",
    "MinioInitializationError",
    "initialize_minio_bucket",
    "main",
]
