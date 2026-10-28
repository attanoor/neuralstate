"""Audit log helpers: every memory mutation is recorded with a before/after snapshot."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import inspect as sa_inspect
from sqlalchemy.ext.asyncio import AsyncSession

from neuralstate.db.models import Base, MemoryEvent

# Columns never captured in snapshots (large, derived, or bookkeeping).
_SKIP = {"embedding", "updated_at"}


def _jsonable(value: Any) -> Any:
    if isinstance(value, (uuid.UUID,)):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def snapshot(obj: Base, fields: list[str] | None = None) -> dict[str, Any]:
    """JSON-safe dict of an ORM object's column values (optionally only `fields`)."""
    mapper = sa_inspect(type(obj))
    names = fields or [c.key for c in mapper.column_attrs if c.key not in _SKIP]
    return {name: _jsonable(getattr(obj, name)) for name in names}


def record(
    session: AsyncSession,
    *,
    agent_id: uuid.UUID,
    op: str,
    target: Base,
    target_id: uuid.UUID,
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    cycle_id: uuid.UUID | None = None,
    note: str | None = None,
) -> MemoryEvent:
    event = MemoryEvent(
        agent_id=agent_id,
        cycle_id=cycle_id,
        op=op,
        target_type=type(target).__tablename__,
        target_id=target_id,
        before=before,
        after=after,
        note=note,
    )
    session.add(event)
    return event


class Changes:
    """Capture an object's fields, mutate it, then record exactly what changed.

    >>> with Changes(session, node, agent_id=a, op="supersede") as c:
    ...     node.status = "superseded"
    """

    def __init__(self, session: AsyncSession, obj: Base, *, agent_id: uuid.UUID, op: str,
                 cycle_id: uuid.UUID | None = None, note: str | None = None, target_id: uuid.UUID | None = None):
        self.session, self.obj, self.agent_id, self.op = session, obj, agent_id, op
        self.cycle_id, self.note = cycle_id, note
        self.target_id = target_id if target_id is not None else getattr(obj, "id", None) or getattr(obj, "node_id")
        self.before: dict[str, Any] = {}

    def __enter__(self) -> "Changes":
        self.before = snapshot(self.obj)
        return self

    def __exit__(self, exc_type, *_):
        if exc_type is not None:
            return False
        after = snapshot(self.obj)
        changed = [k for k in after if after[k] != self.before.get(k)]
        if changed:
            record(self.session, agent_id=self.agent_id, op=self.op, target=self.obj, target_id=self.target_id,
                   before={k: self.before[k] for k in changed}, after={k: after[k] for k in changed},
                   cycle_id=self.cycle_id, note=self.note)
        return False
