"""Read models and operator actions shared by the REST API and the dashboard."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from neuralstate.consolidation.reconcile import settle
from neuralstate.db.audit import Changes, snapshot
from neuralstate.db.models import (
    Agent,
    ConsolidationCycle,
    ConsolidationJob,
    Contradiction,
    ContradictionStatus,
    Edge,
    Episode,
    HeuristicRule,
    MemoryEvent,
    Node,
    NodeKind,
    NodeStatus,
    Provenance,
    utcnow,
)


class NotFound(LookupError):
    pass


def node_dict(n: Node) -> dict[str, Any]:
    return {
        "id": str(n.id), "kind": n.kind, "label": n.label, "body": n.body, "status": n.status,
        "confidence": round(n.confidence, 3), "evidence_count": n.evidence_count, "access_count": n.access_count,
        "attrs": n.attrs, "created_at": n.created_at.isoformat(), "last_evidence_at": n.last_evidence_at.isoformat(),
        "last_accessed_at": n.last_accessed_at.isoformat() if n.last_accessed_at else None,
    }


def episode_dict(e: Episode, full: bool = False) -> dict[str, Any]:
    d = {"id": str(e.id), "agent_id": str(e.agent_id), "summary": e.summary, "task_type": e.task_type,
         "session_id": e.session_id, "outcome": e.outcome, "outcome_detail": e.outcome_detail,
         "applied_rule_ids": e.applied_rule_ids, "consolidated": e.consolidated_at is not None,
         "compacted": e.compacted, "created_at": e.created_at.isoformat()}
    if full:
        d["content"] = e.content
    return d


def rule_dict(n: Node, r: HeuristicRule) -> dict[str, Any]:
    from neuralstate.consolidation.scoring import rule_effectiveness

    return {**node_dict(n), "task_type": r.task_type, "condition": r.condition, "avoid": r.avoid,
            "action": r.action, "rationale": r.rationale, "success_after": r.success_after,
            "failure_after": r.failure_after,
            "effectiveness": round(rule_effectiveness(r.success_after, r.failure_after), 3)}


def event_dict(e: MemoryEvent) -> dict[str, Any]:
    return {"id": e.id, "agent_id": str(e.agent_id), "cycle_id": str(e.cycle_id) if e.cycle_id else None,
            "op": e.op, "target_type": e.target_type, "target_id": str(e.target_id), "before": e.before,
            "after": e.after, "note": e.note, "reverted": e.reverted, "created_at": e.created_at.isoformat()}


async def resolve_agent(s: AsyncSession, ref: str | uuid.UUID) -> Agent:
    """Look an agent up by id or by name."""
    try:
        agent = await s.get(Agent, uuid.UUID(str(ref)))
    except ValueError:
        agent = (await s.execute(select(Agent).where(Agent.name == str(ref)))).scalar_one_or_none()
    if agent is None:
        raise NotFound(f"agent {ref} not found")
    return agent


async def list_agents(s: AsyncSession) -> list[dict[str, Any]]:
    agents = (await s.execute(select(Agent).order_by(Agent.created_at))).scalars().all()
    return [{"id": str(a.id), "name": a.name, "created_at": a.created_at.isoformat()} for a in agents]


async def overview(s: AsyncSession, agent_id: uuid.UUID) -> dict[str, Any]:
    by_kind = dict((await s.execute(
        select(Node.kind, func.count()).where(Node.agent_id == agent_id, Node.status == NodeStatus.active.value)
        .group_by(Node.kind))).all())
    by_outcome = dict((await s.execute(
        select(Episode.outcome, func.count()).where(Episode.agent_id == agent_id).group_by(Episode.outcome))).all())
    backlog = (await s.execute(select(func.count()).where(
        Episode.agent_id == agent_id, Episode.consolidated_at.is_(None)))).scalar()
    pending = (await s.execute(select(func.count()).select_from(Contradiction).where(
        Contradiction.agent_id == agent_id, Contradiction.status == ContradictionStatus.pending.value))).scalar()
    cycles = (await s.execute(select(ConsolidationCycle).where(ConsolidationCycle.agent_id == agent_id)
                              .order_by(ConsolidationCycle.started_at.desc()).limit(10))).scalars().all()
    jobs = dict((await s.execute(select(ConsolidationJob.status, func.count()).where(
        ConsolidationJob.agent_id == agent_id).group_by(ConsolidationJob.status))).all())
    rule_growth = (await s.execute(
        select(func.date_trunc("day", Node.created_at).label("day"), func.count())
        .where(Node.agent_id == agent_id, Node.kind == NodeKind.rule.value)
        .group_by("day").order_by("day"))).all()
    return {
        "nodes": {k.value: by_kind.get(k.value, 0) for k in NodeKind},
        "episodes": {"total": sum(by_outcome.values()), **by_outcome, "unconsolidated": backlog},
        "pending_review": pending,
        "jobs": jobs,
        "rule_growth": [{"day": d.date().isoformat(), "rules": c} for d, c in rule_growth],
        "cycles": [{"id": str(c.id), "kind": c.kind, "started_at": c.started_at.isoformat(),
                    "finished_at": c.finished_at.isoformat() if c.finished_at else None,
                    "error": c.error, "stats": c.stats} for c in cycles],
    }


async def list_episodes(s: AsyncSession, agent_id: uuid.UUID, outcome: str | None = None,
                        limit: int = 50) -> list[dict[str, Any]]:
    q = select(Episode).where(Episode.agent_id == agent_id)
    if outcome:
        q = q.where(Episode.outcome == outcome)
    rows = (await s.execute(q.order_by(Episode.created_at.desc()).limit(limit))).scalars().all()
    return [episode_dict(e) for e in rows]


async def list_rules(s: AsyncSession, agent_id: uuid.UUID, status: str | None = None) -> list[dict[str, Any]]:
    q = (select(Node, HeuristicRule).join(HeuristicRule, HeuristicRule.node_id == Node.id)
         .where(Node.agent_id == agent_id))
    if status:
        q = q.where(Node.status == status)
    rows = (await s.execute(q.order_by(Node.confidence.desc(), Node.created_at.desc()))).all()
    return [rule_dict(n, r) for n, r in rows]


async def node_detail(s: AsyncSession, node_id: uuid.UUID) -> dict[str, Any]:
    node = await s.get(Node, node_id)
    if node is None:
        raise NotFound("node not found")
    d = node_dict(node)
    if node.kind == NodeKind.rule.value:
        rule = await s.get(HeuristicRule, node_id)
        if rule is not None:
            d = rule_dict(node, rule)
    evidence = (await s.execute(
        select(Episode, Provenance.role).join(Provenance, Provenance.episode_id == Episode.id)
        .where(Provenance.node_id == node_id).order_by(Episode.created_at.desc()).limit(50))).all()
    d["evidence"] = [{**episode_dict(e), "role": role} for e, role in evidence]
    events = (await s.execute(select(MemoryEvent).where(MemoryEvent.target_id == node_id)
                              .order_by(MemoryEvent.id.desc()).limit(50))).scalars().all()
    d["history"] = [event_dict(e) for e in events]
    return d


async def graph(s: AsyncSession, agent_id: uuid.UUID, center: uuid.UUID | None = None, depth: int = 1,
                limit: int = 300, include_inactive: bool = False) -> dict[str, Any]:
    statuses = [NodeStatus.active.value, NodeStatus.pending_review.value]
    if include_inactive:
        statuses += [NodeStatus.superseded.value, NodeStatus.archived.value, NodeStatus.disabled.value]
    if center is None:
        nodes = (await s.execute(
            select(Node).where(Node.agent_id == agent_id, Node.status.in_(statuses), Node.kind != NodeKind.rule.value)
            .order_by(Node.evidence_count.desc(), Node.confidence.desc()).limit(limit))).scalars().all()
        ids = {n.id for n in nodes}
    else:
        ids, frontier = {center}, {center}
        for _ in range(max(0, depth)):
            rows = (await s.execute(select(Edge.src_id, Edge.dst_id).where(
                Edge.agent_id == agent_id, or_(Edge.src_id.in_(list(frontier)), Edge.dst_id.in_(list(frontier)))
            ).limit(limit))).all()
            frontier = {x for r in rows for x in r} - ids
            ids |= frontier
            if len(ids) >= limit:
                break
        nodes = (await s.execute(select(Node).where(Node.id.in_(list(ids)), Node.status.in_(statuses)))).scalars().all()
        ids = {n.id for n in nodes}
    edges = (await s.execute(select(Edge).where(Edge.agent_id == agent_id, Edge.src_id.in_(list(ids)),
                                                Edge.dst_id.in_(list(ids))))).scalars().all() if ids else []
    return {
        "nodes": [{"id": str(n.id), "kind": n.kind, "label": n.label, "status": n.status,
                   "confidence": round(n.confidence, 3), "evidence_count": n.evidence_count} for n in nodes],
        "edges": [{"id": str(e.id), "source": str(e.src_id), "target": str(e.dst_id), "relation": e.relation}
                  for e in edges],
    }


async def update_rule(s: AsyncSession, node_id: uuid.UUID, changes: dict[str, Any]) -> dict[str, Any]:
    """Operator edit: enable/disable a rule or rewrite its text. Logged like any other mutation."""
    node, rule = await s.get(Node, node_id), await s.get(HeuristicRule, node_id)
    if node is None or rule is None:
        raise NotFound("rule not found")
    status = changes.get("status")
    if status is not None and status not in (NodeStatus.active.value, NodeStatus.disabled.value):
        raise ValueError("status must be 'active' or 'disabled'")
    text_fields = {k: v for k, v in changes.items() if k in ("condition", "avoid", "action", "rationale") and v}
    with Changes(s, rule, agent_id=node.agent_id, op="operator_edit", target_id=node.id):
        for k, v in text_fields.items():
            setattr(rule, k, v)
    with Changes(s, node, agent_id=node.agent_id, op="operator_edit"):
        if status is not None:
            node.status = status
        if text_fields:
            node.label = f"When {rule.condition}: avoid {rule.avoid}; instead {rule.action}."
            node.body = rule.rationale
            node.attrs = {**node.attrs, "condition": rule.condition, "avoid": rule.avoid, "action": rule.action}
    return rule_dict(node, rule)


async def list_review(s: AsyncSession, agent_id: uuid.UUID, status: str = "pending") -> list[dict[str, Any]]:
    rows = (await s.execute(select(Contradiction).where(
        Contradiction.agent_id == agent_id, Contradiction.status == status)
        .order_by(Contradiction.created_at.desc()).limit(200))).scalars().all()
    ids = {i for c in rows for i in (c.node_a, c.node_b)}
    nodes = {n.id: n for n in (await s.execute(select(Node).where(Node.id.in_(list(ids))))).scalars()} if ids else {}
    out = []
    for c in rows:
        a, b = nodes.get(c.node_a), nodes.get(c.node_b)
        out.append({"id": str(c.id), "status": c.status, "margin": round(c.margin, 3), "reason": c.reason,
                    "resolution": c.resolution, "created_at": c.created_at.isoformat(),
                    "winner_id": str(c.winner_id) if c.winner_id else None,
                    "a": node_dict(a) if a else None, "b": node_dict(b) if b else None})
    return out


async def resolve_review(s: AsyncSession, contradiction_id: uuid.UUID, winner: str,
                         note: str | None) -> dict[str, Any]:
    """Operator decision: 'a' or 'b' wins, or 'both' are true (not a real contradiction)."""
    c = await s.get(Contradiction, contradiction_id)
    if c is None:
        raise NotFound("contradiction not found")
    if c.status != ContradictionStatus.pending.value:
        raise ValueError(f"already {c.status}")
    a, b = await s.get(Node, c.node_a), await s.get(Node, c.node_b)
    resolution = f"operator: {note}" if note else "operator decision"
    if winner == "both":
        for n in (a, b):
            if n.status == NodeStatus.pending_review.value:
                with Changes(s, n, agent_id=c.agent_id, op="restore", note="operator: both facts hold"):
                    n.status = NodeStatus.active.value
        c.status, c.resolution, c.resolved_at = ContradictionStatus.resolved.value, resolution, utcnow()
    elif winner in ("a", "b"):
        w, l = (a, b) if winner == "a" else (b, a)
        await settle(s, c, w, l, ContradictionStatus.resolved.value, resolution)
    else:
        raise ValueError("winner must be 'a', 'b' or 'both'")
    return {"id": str(c.id), "status": c.status, "resolution": c.resolution,
            "a": snapshot(a, ["id", "status"]), "b": snapshot(b, ["id", "status"])}


async def list_events(s: AsyncSession, agent_id: uuid.UUID, op: str | None = None, limit: int = 100,
                      before_id: int | None = None) -> list[dict[str, Any]]:
    q = select(MemoryEvent).where(MemoryEvent.agent_id == agent_id)
    if op:
        q = q.where(MemoryEvent.op == op)
    if before_id:
        q = q.where(MemoryEvent.id < before_id)
    rows = (await s.execute(q.order_by(MemoryEvent.id.desc()).limit(limit))).scalars().all()
    return [event_dict(e) for e in rows]
