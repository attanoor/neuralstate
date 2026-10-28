"""The background consolidation worker.

Run as many workers as you like: jobs are claimed with SKIP LOCKED and each agent is
guarded by a Postgres advisory lock, so no two cycles for the same agent overlap.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from neuralstate.config import Settings
from neuralstate.consolidation.engine import run_cycle
from neuralstate.db.models import ConsolidationCycle, Episode
from neuralstate.db.session import session_scope
from neuralstate.providers.registry import Providers
from neuralstate.worker import jobs

log = logging.getLogger("neuralstate.worker")


async def schedule_due_cycles(sessions: async_sessionmaker[AsyncSession], settings: Settings) -> int:
    """Queue a full cycle for every agent with enough backlog, or any backlog and a stale last cycle."""
    queued = 0
    now = datetime.now(timezone.utc)
    async with session_scope(sessions) as s:
        backlog = dict((await s.execute(
            select(Episode.agent_id, func.count()).where(Episode.consolidated_at.is_(None))
            .group_by(Episode.agent_id))).all())
        if not backlog:
            return 0
        last = dict((await s.execute(
            select(ConsolidationCycle.agent_id, func.max(ConsolidationCycle.started_at))
            .where(ConsolidationCycle.kind == "cycle", ConsolidationCycle.agent_id.in_(list(backlog)))
            .group_by(ConsolidationCycle.agent_id))).all())
        for agent_id, count in backlog.items():
            stale = last.get(agent_id) is None or now - last[agent_id] > timedelta(seconds=settings.cycle_interval_seconds)
            if count >= settings.cycle_episode_threshold or stale:
                await jobs.enqueue(s, agent_id, "cycle")
                queued += 1
    return queued


async def process_one(engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], providers: Providers,
                      settings: Settings) -> bool:
    """Claim and run one job. Returns False when the queue had nothing due."""
    async with session_scope(sessions) as s:
        job = await jobs.claim(s)
    if job is None:
        return False

    async with engine.connect() as lock_conn:
        if not await jobs.try_agent_lock(lock_conn, job.agent_id):
            async with session_scope(sessions) as s:
                await jobs.defer(s, job.id, 15)
            await lock_conn.commit()
            return True
        await lock_conn.commit()  # end the implicit transaction; the session-level lock persists
        try:
            stats = await run_cycle(sessions, providers, settings, job.agent_id, job.kind, job.id)
            async with session_scope(sessions) as s:
                await jobs.finish(s, job.id)
            log.info("%s for agent %s done in %ss", job.kind, job.agent_id, stats.get("seconds"))
        except Exception as exc:  # noqa: BLE001 - every failure is recorded and retried
            async with session_scope(sessions) as s:
                await jobs.fail(s, job.id, job.attempts, f"{type(exc).__name__}: {exc}", settings.job_max_attempts)
        finally:
            await jobs.release_agent_lock(lock_conn, job.agent_id)
            await lock_conn.commit()
    return True


class AgentBusy(RuntimeError):
    pass


async def run_now(engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], providers: Providers,
                  settings: Settings, agent_id, kind: str = "cycle") -> dict:
    """Run a cycle in the calling process (CLI / `POST /consolidate?wait=true`), honoring the agent lock."""
    async with engine.connect() as lock_conn:
        if not await jobs.try_agent_lock(lock_conn, agent_id):
            raise AgentBusy("a consolidation cycle is already running for this agent")
        await lock_conn.commit()
        try:
            return await run_cycle(sessions, providers, settings, agent_id, kind)
        finally:
            await jobs.release_agent_lock(lock_conn, agent_id)
            await lock_conn.commit()


async def run_worker(engine: AsyncEngine, sessions: async_sessionmaker[AsyncSession], providers: Providers,
                     settings: Settings, *, once: bool = False) -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):  # Windows: Ctrl+C raises KeyboardInterrupt instead
            pass

    log.info("worker started (reasoner=%s, embedder=%s)", providers.reasoner.name, providers.embedder.name)
    last_schedule = 0.0
    while not stop.is_set():
        if loop.time() - last_schedule > 30 or once:
            async with session_scope(sessions) as s:
                recovered = await jobs.requeue_stale(s, settings.job_stale_seconds)
            if recovered:
                log.warning("requeued %d stale jobs", recovered)
            await schedule_due_cycles(sessions, settings)
            last_schedule = loop.time()

        worked = await process_one(engine, sessions, providers, settings)
        if once and not worked:
            return
        if not worked:
            try:
                await asyncio.wait_for(stop.wait(), timeout=settings.worker_poll_seconds)
            except asyncio.TimeoutError:
                pass
    log.info("worker stopped")

