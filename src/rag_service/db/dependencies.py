from collections.abc import AsyncIterator
from typing import cast

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from rag_service.db.session import Database


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    """Yield a business session without beginning or committing a transaction."""
    database = cast(Database, request.app.state.database)
    async with database.sessions() as session:
        yield session
