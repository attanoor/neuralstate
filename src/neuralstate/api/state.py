from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from neuralstate.config import Settings
from neuralstate.providers.registry import Providers


@dataclass
class AppState:
    settings: Settings
    engine: AsyncEngine
    sessions: async_sessionmaker[AsyncSession]
    providers: Providers


def get_state(request: Request) -> AppState:
    return request.app.state.ns


async def get_session(state: AppState = Depends(get_state)) -> AsyncIterator[AsyncSession]:
    async with state.sessions() as session:
        try:
            yield session
            await session.commit()
        except BaseException:
            await session.rollback()
            raise


def key_matches(expected: str | None, given: str | None) -> bool:
    return expected is None or (given is not None and secrets.compare_digest(expected, given))


def require_api_key(state: AppState = Depends(get_state), x_api_key: str | None = Header(None)) -> None:
    if not key_matches(state.settings.api_key, x_api_key):
        raise HTTPException(status_code=401, detail="missing or invalid X-API-Key")
