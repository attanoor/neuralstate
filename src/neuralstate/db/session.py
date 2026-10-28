from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from neuralstate.config import Settings


def make_engine(settings: Settings) -> AsyncEngine:
    # No asyncpg vector codec is registered: pgvector's SQLAlchemy `Vector` type already
    # converts to and from the text format, and the two conflict if both are active.
    connect_args: dict = {}
    if settings.db_schema:
        connect_args["server_settings"] = {"search_path": f"{settings.db_schema},public"}
    return create_async_engine(settings.database_url, pool_pre_ping=True, connect_args=connect_args)


def make_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


@asynccontextmanager
async def session_scope(factory: async_sessionmaker[AsyncSession]) -> AsyncIterator[AsyncSession]:
    """A transaction that commits on success and rolls back on error."""
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except BaseException:
            await session.rollback()
            raise
