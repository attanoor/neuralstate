from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from neuralstate.config import Settings
from neuralstate.consolidation.reasoner import Reasoner
from neuralstate.providers.base import Embedder


@dataclass
class CycleContext:
    """Everything a consolidation stage needs. Stages never hold a transaction open across LLM calls."""

    sessions: async_sessionmaker[AsyncSession]
    embedder: Embedder
    reasoner: Reasoner
    settings: Settings
    agent_id: uuid.UUID
    cycle_id: uuid.UUID | None = None
    now: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    # Fact pairs flagged during extraction for the reconcile stage: (new_fact_id, existing_fact_id).
    candidate_pairs: set[tuple[uuid.UUID, uuid.UUID]] = field(default_factory=set)
