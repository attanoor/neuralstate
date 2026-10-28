"""Runs a consolidation cycle ("sleep"): extract -> reconcile -> synthesize -> decay."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from neuralstate.config import Settings
from neuralstate.consolidation import decay, extract, reconcile, synthesize
from neuralstate.consolidation.context import CycleContext
from neuralstate.db.models import ConsolidationCycle, utcnow
from neuralstate.db.session import session_scope
from neuralstate.providers.registry import Providers

log = logging.getLogger(__name__)

STAGES = {
    "cycle": [("extract", extract.run), ("reconcile", reconcile.run), ("synthesize", synthesize.run),
              ("decay", decay.run)],
    # Cheap check queued after failures; picks up any rule that is now ready to be learned.
    "synthesize": [("synthesize", synthesize.run)],
}


async def run_cycle(sessions: async_sessionmaker[AsyncSession], providers: Providers, settings: Settings,
                    agent_id: uuid.UUID, kind: str = "cycle", job_id: uuid.UUID | None = None) -> dict[str, Any]:
    """Run one cycle for one agent. Callers must hold the agent's advisory lock (see worker)."""
    async with session_scope(sessions) as s:
        cycle = ConsolidationCycle(agent_id=agent_id, job_id=job_id, kind=kind)
        s.add(cycle)
        await s.flush()
        cycle_id = cycle.id

    ctx = CycleContext(sessions=sessions, embedder=providers.embedder, reasoner=providers.reasoner,
                       settings=settings, agent_id=agent_id, cycle_id=cycle_id)
    stats: dict[str, Any] = {"reasoner": providers.reasoner.name}
    error: str | None = None
    started = time.perf_counter()
    try:
        for name, stage in STAGES[kind]:
            t0 = time.perf_counter()
            stats[name] = await stage(ctx)
            stats[name]["seconds"] = round(time.perf_counter() - t0, 3)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        log.exception("consolidation %s failed for agent %s", kind, agent_id)
        raise
    finally:
        stats["seconds"] = round(time.perf_counter() - started, 3)
        async with session_scope(sessions) as s:
            cycle = await s.get(ConsolidationCycle, cycle_id)
            cycle.finished_at = utcnow()
            cycle.stats = stats
            cycle.error = error
    stats["cycle_id"] = str(cycle_id)
    return stats
