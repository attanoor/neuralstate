"""Undo a memory mutation recorded in the audit log."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from neuralstate.db.audit import Changes
from neuralstate.db.models import Episode, HeuristicRule, MemoryEvent, Node, NodeStatus

_TABLES = {"nodes": Node, "episodes": Episode, "heuristic_rules": HeuristicRule}
_DATETIME_FIELDS = {"last_accessed_at", "last_evidence_at", "created_at", "consolidated_at"}


class RevertError(ValueError):
    pass


def _restore(value: Any, field: str) -> Any:
    if field in _DATETIME_FIELDS and isinstance(value, str):
        return datetime.fromisoformat(value)
    return value


async def revert_event(session: AsyncSession, event: MemoryEvent) -> MemoryEvent:
    """Apply the inverse of `event` and log that as a new event.

    A `create` is undone by archiving the created node (nothing is ever hard-deleted);
    every other op restores the fields captured in `before`.
    """
    if event.reverted:
        raise RevertError("event already reverted")
    model = _TABLES.get(event.target_type)
    if model is None:
        raise RevertError(f"cannot revert changes to {event.target_type}")
    target = await session.get(model, event.target_id)
    if target is None:
        raise RevertError("target no longer exists")

    with Changes(session, target, agent_id=event.agent_id, op="revert", note=f"revert of event #{event.id}"):
        if event.op == "create":
            if not isinstance(target, Node):
                raise RevertError("only node creation can be reverted")
            target.status = NodeStatus.archived.value
        else:
            for field, value in (event.before or {}).items():
                setattr(target, field, _restore(value, field))
    event.reverted = True
    return event
