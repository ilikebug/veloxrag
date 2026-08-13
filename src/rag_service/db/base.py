from datetime import datetime
from typing import Annotated
from uuid import UUID, uuid4

from sqlalchemy import func
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import DeclarativeBase, mapped_column


class Base(DeclarativeBase):
    """Base class for SQLAlchemy models introduced in later increments."""


type UUIDPrimaryKey = Annotated[
    UUID,
    mapped_column(PostgreSQLUUID(as_uuid=True), primary_key=True, default=uuid4),
]
type CreatedAt = Annotated[
    datetime,
    mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now()),
]
type UpdatedAt = Annotated[
    datetime,
    mapped_column(TIMESTAMP(timezone=True), nullable=False, server_default=func.now()),
]
