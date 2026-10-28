"""Postgres-backed job queue: idempotent enqueue, SKIP LOCKED claiming, per-agent advisory locks."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncSession

from neuralstate.db.models import ConsolidationJob, JobStatus


async def enqueue(session: AsyncSession, agent_id: uuid.UUID, kind: str, delay_seconds: float = 0.0,
                  payload: dict | None = None) -> None:
    """Queue a job unless one of the same kind is already queued for this agent."""
    stmt = insert(ConsolidationJob).values(
        id=uuid.uuid4(), agent_id=agent_id, kind=kind, status=JobStatus.queued.value, payload=payload or {},
        attempts=0, run_after=datetime.now(timezone.utc) + timedelta(seconds=delay_seconds),
    ).on_conflict_do_nothing(index_elements=["agent_id", "kind"], index_where=text("status = 'queued'"))
    await session.execute(stmt)


_CLAIM = text("""
UPDATE consolidation_jobs SET status = 'running', locked_at = now(), attempts = attempts + 1
WHERE id = (
    SELECT id FROM consolidation_jobs
    WHERE status = 'queued' AND run_after <= now()
    ORDER BY run_after
    FOR UPDATE SKIP LOCKED
    LIMIT 1
)
RETURNING id, agent_id, kind, attempts
""")


async def claim(session: AsyncSession):
    """Atomically take the next due job, or return None. Safe with any number of workers."""
    return (await session.execute(_CLAIM)).first()


async def finish(session: AsyncSession, job_id: uuid.UUID) -> None:
    await session.execute(text(
        "UPDATE consolidation_jobs SET status = 'done', finished_at = now(), locked_at = NULL WHERE id = :id"
    ), {"id": job_id})


async def fail(session: AsyncSession, job_id: uuid.UUID, attempts: int, error: str, max_attempts: int) -> None:
    """Requeue with exponential backoff, or mark failed after `max_attempts`.

    If an identical job was queued meanwhile, the unique index rejects the requeue and this
    one is marked failed instead (the queued one will do the work).
    """
    if attempts < max_attempts:
        backoff = min(3600, 10 * 2 ** attempts)
        await session.execute(text("""
            UPDATE consolidation_jobs j SET status = 'queued', locked_at = NULL, last_error = :err,
                   run_after = now() + make_interval(secs => :backoff)
            WHERE j.id = :id AND NOT EXISTS (
                SELECT 1 FROM consolidation_jobs q
                WHERE q.agent_id = j.agent_id AND q.kind = j.kind AND q.status = 'queued')
        """), {"id": job_id, "err": error[:4000], "backoff": backoff})
    await session.execute(text("""
        UPDATE consolidation_jobs SET status = 'failed', finished_at = now(), locked_at = NULL,
               last_error = :err
        WHERE id = :id AND status = 'running'
    """), {"id": job_id, "err": error[:4000]})


async def defer(session: AsyncSession, job_id: uuid.UUID, seconds: float) -> None:
    """Put a claimed job back without counting the attempt (e.g. agent busy in another worker)."""
    await session.execute(text("""
        UPDATE consolidation_jobs j SET status = 'queued', locked_at = NULL, attempts = attempts - 1,
               run_after = now() + make_interval(secs => :s)
        WHERE j.id = :id AND NOT EXISTS (
            SELECT 1 FROM consolidation_jobs q
            WHERE q.agent_id = j.agent_id AND q.kind = j.kind AND q.status = 'queued')
    """), {"id": job_id, "s": seconds})
    await session.execute(text(
        "UPDATE consolidation_jobs SET status = 'done', finished_at = now(), locked_at = NULL, "
        "last_error = 'superseded by an already-queued job' WHERE id = :id AND status = 'running'"
    ), {"id": job_id})


async def requeue_stale(session: AsyncSession, stale_seconds: int) -> int:
    """Recover jobs whose worker died mid-run."""
    result = await session.execute(text("""
        UPDATE consolidation_jobs j SET status = 'queued', locked_at = NULL
        WHERE j.status = 'running' AND j.locked_at < now() - make_interval(secs => :s)
          AND NOT EXISTS (SELECT 1 FROM consolidation_jobs q
                          WHERE q.agent_id = j.agent_id AND q.kind = j.kind AND q.status = 'queued')
    """), {"s": stale_seconds})
    return result.rowcount or 0


def agent_lock_key(agent_id: uuid.UUID) -> int:
    return int.from_bytes(agent_id.bytes[:8], "big", signed=True)


async def try_agent_lock(conn: AsyncConnection, agent_id: uuid.UUID) -> bool:
    """Session-level advisory lock so only one cycle per agent runs at a time across workers."""
    return bool((await conn.execute(text("SELECT pg_try_advisory_lock(:k)"),
                                    {"k": agent_lock_key(agent_id)})).scalar())


async def release_agent_lock(conn: AsyncConnection, agent_id: uuid.UUID) -> None:
    await conn.execute(text("SELECT pg_advisory_unlock(:k)"), {"k": agent_lock_key(agent_id)})
