from contextlib import AbstractAsyncContextManager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from rag_service.config import Settings


class Database:
    sessions: async_sessionmaker[AsyncSession]

    def __init__(self, database_url: str) -> None:
        self.engine: AsyncEngine = create_async_engine(
            database_url,
            pool_pre_ping=True,
        )
        self.sessions = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> "Database":
        return cls(settings.database_url.get_secret_value())

    def session(self) -> AbstractAsyncContextManager[AsyncSession]:
        """Return a request-scoped session; application services own transactions."""
        return self.sessions()

    async def ping(self) -> None:
        async with self.engine.connect() as connection:
            await connection.execute(text("SELECT 1"))

    async def close(self) -> None:
        await self.engine.dispose()
