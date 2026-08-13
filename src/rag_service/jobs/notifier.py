"""Best-effort Redis wakeups for PostgreSQL-authoritative Jobs."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

JOB_NOTIFICATION_CHANNEL = "rag:jobs"


class RedisPublisher(Protocol):
    async def publish(self, channel: str, payload: str) -> int: ...


class RedisSubscription(Protocol):
    async def subscribe(self, channel: str) -> None: ...

    async def unsubscribe(self, channel: str) -> None: ...

    async def get_message(
        self,
        ignore_subscribe_messages: bool,
        wait_seconds: float,
        /,
    ) -> dict[str, object] | None: ...

    async def aclose(self) -> None: ...


class RedisSubscriberClient(Protocol):
    def pubsub(self) -> RedisSubscription: ...


class RedisJobNotifier:
    def __init__(
        self,
        redis_client: RedisPublisher,
        *,
        channel: str = JOB_NOTIFICATION_CHANNEL,
    ) -> None:
        self._redis = redis_client
        self._channel = channel

    async def notify(self, job_id: UUID) -> bool:
        """Publish only the UUID; failures are wakeup loss, not Job loss."""

        try:
            await self._redis.publish(self._channel, str(job_id))
        except Exception:
            return False
        return True


class RedisJobSubscriber:
    def __init__(
        self,
        redis_client: RedisSubscriberClient,
        *,
        channel: str = JOB_NOTIFICATION_CHANNEL,
    ) -> None:
        self._subscription = redis_client.pubsub()
        self._channel = channel
        self._started = False

    async def start(self) -> None:
        if not self._started:
            await self._subscription.subscribe(self._channel)
            self._started = True

    async def get(self, timeout_seconds: float) -> UUID | None:
        if timeout_seconds < 0:
            raise ValueError("notification timeout must be nonnegative")
        await self.start()
        message = await self._subscription.get_message(True, timeout_seconds)
        if message is None:
            return None
        payload = message.get("data")
        if isinstance(payload, bytes):
            try:
                payload = payload.decode("ascii")
            except UnicodeDecodeError:
                return None
        if type(payload) is not str:
            return None
        try:
            job_id = UUID(payload)
        except ValueError:
            return None
        return job_id if payload == str(job_id) else None

    async def close(self) -> None:
        if self._started:
            await self._subscription.unsubscribe(self._channel)
            self._started = False
        await self._subscription.aclose()


__all__ = [
    "JOB_NOTIFICATION_CHANNEL",
    "RedisJobNotifier",
    "RedisJobSubscriber",
]
