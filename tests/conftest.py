from __future__ import annotations

import os
import uuid

import pytest
from sqlalchemy import text

from neuralstate.config import Settings
from neuralstate.consolidation.reasoner import HeuristicReasoner
from neuralstate.providers.embed import HashEmbedder
from neuralstate.providers.registry import Providers

TEST_DATABASE_URL = os.environ.get(
    "NS_TEST_DATABASE_URL", "postgresql+asyncpg://neuralstate:neuralstate@127.0.0.1:5432/neuralstate_test")


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, database_url=TEST_DATABASE_URL, db_schema=f"t_{uuid.uuid4().hex[:12]}",
                    llm_provider="fake", embed_provider="fake", api_key=None,
                    recall_min_similarity=0.25)  # hash embeddings score lower than real models


@pytest.fixture
def providers(settings: Settings) -> Providers:
    return Providers(embedder=HashEmbedder(settings.embed_dim), reasoner=HeuristicReasoner())


@pytest.fixture
async def db(settings: Settings):
    """A fresh schema per test, created from the models and dropped afterwards."""
    from neuralstate.db.models import Base
    from neuralstate.db.session import make_engine, make_sessionmaker

    engine = make_engine(settings)
    try:
        async with engine.begin() as conn:
            await conn.execute(text(f'CREATE SCHEMA "{settings.db_schema}"'))
    except (OSError, ConnectionError) as exc:
        await engine.dispose()
        pytest.skip(f"Postgres not reachable at {TEST_DATABASE_URL}: {exc}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield engine, make_sessionmaker(engine)
    finally:
        async with engine.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA "{settings.db_schema}" CASCADE'))
        await engine.dispose()
