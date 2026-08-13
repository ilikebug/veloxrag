import pytest

from rag_service.dev.minio_init import MinioInitializationError, initialize_minio_bucket


class FakeMinio:
    def __init__(self, *, readiness_failures: int, bucket_exists: bool) -> None:
        self.readiness_failures = readiness_failures
        self.exists = bucket_exists
        self.bucket_exists_calls = 0
        self.make_calls: list[str] = []
        self.delete_policy_calls: list[str] = []

    def bucket_exists(self, bucket_name: str) -> bool:
        self.bucket_exists_calls += 1
        if self.bucket_exists_calls <= self.readiness_failures:
            raise RuntimeError("connection detail must stay private")
        return self.exists

    def make_bucket(self, bucket_name: str) -> None:
        self.make_calls.append(bucket_name)
        self.exists = True

    def delete_bucket_policy(self, bucket_name: str) -> None:
        self.delete_policy_calls.append(bucket_name)


def test_minio_initializer_retries_then_creates_a_private_bucket() -> None:
    client = FakeMinio(readiness_failures=2, bucket_exists=False)
    sleeps: list[float] = []

    initialize_minio_bucket(
        client,
        bucket_name="rag-documents",
        max_attempts=3,
        retry_seconds=0.25,
        sleep=sleeps.append,
    )

    assert client.bucket_exists_calls == 3
    assert client.make_calls == ["rag-documents"]
    assert client.delete_policy_calls == ["rag-documents"]
    assert sleeps == [0.25, 0.25]


@pytest.mark.parametrize("maximum", (0, -1, True))
def test_minio_initializer_rejects_invalid_attempt_bounds(maximum: int) -> None:
    client = FakeMinio(readiness_failures=0, bucket_exists=True)

    with pytest.raises(MinioInitializationError):
        initialize_minio_bucket(
            client,
            bucket_name="rag-documents",
            max_attempts=maximum,
            retry_seconds=0,
        )


def test_minio_initializer_fails_safely_after_the_deadline() -> None:
    client = FakeMinio(readiness_failures=3, bucket_exists=False)

    def sleep(_seconds: float) -> None:
        return None

    with pytest.raises(MinioInitializationError, match="initialization failed") as captured:
        initialize_minio_bucket(
            client,
            bucket_name="rag-documents",
            max_attempts=2,
            retry_seconds=0,
            sleep=sleep,
        )

    assert "connection detail" not in str(captured.value)
    assert client.bucket_exists_calls == 2
