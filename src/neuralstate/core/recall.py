"""Recall: hybrid retrieval over rules, the fact graph and raw episodes, packed to a token budget."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from neuralstate.config import Settings
from neuralstate.consolidation import scoring
from neuralstate.db.models import Edge, Episode, HeuristicRule, Node, NodeKind, NodeStatus
from neuralstate.providers.base import Embedder

SECTIONS = ("rules", "facts", "episodes")
HEADINGS = {
    "rules": "Learned rules (from your past mistakes; follow them):",
    "facts": "Known facts:",
    "episodes": "Related past episodes:",
}


def estimate_tokens(text: str) -> int:
    """Cheap provider-neutral estimate (about 4 characters per token)."""
    return max(1, (len(text) + 3) // 4)


@dataclass
class Item:
    section: str
    id: uuid.UUID
    text: str
    score: float
    data: dict[str, Any] = field(default_factory=dict)


def pack(items: list[Item], budget_tokens: int) -> tuple[list[Item], int]:
    """Greedy packing: sections in priority order (rules, facts, episodes), best score first.

    An item that does not fit is skipped (a shorter one later may still fit), so a single
    long episode can't starve the rest of the context.
    """
    chosen: list[Item] = []
    used = 0
    for section in SECTIONS:
        section_items = sorted((i for i in items if i.section == section), key=lambda i: i.score, reverse=True)
        header_cost = estimate_tokens(HEADINGS[section]) + 1
        header_paid = False
        for item in section_items:
            cost = estimate_tokens(item.text) + 1 + (0 if header_paid else header_cost)
            if used + cost > budget_tokens:
                continue
            used += cost
            header_paid = True
            chosen.append(item)
    return chosen, used


def render(items: list[Item]) -> str:
    blocks = []
    for section in SECTIONS:
        lines = [f"- {i.text}" for i in items if i.section == section]
        if lines:
            blocks.append(HEADINGS[section] + "\n" + "\n".join(lines))
    return "\n\n".join(blocks)


def _days(since: datetime | None, now: datetime) -> float:
    return 0.0 if since is None else max(0.0, (now - since).total_seconds() / 86400)


async def recall(session: AsyncSession, embedder: Embedder, settings: Settings, *, agent_id: uuid.UUID,
                 query: str, task_type: str | None = None, budget_tokens: int = 2000,
                 k: int | None = None) -> dict[str, Any]:
    k = k or settings.recall_k
    now = datetime.now(timezone.utc)
    [qv] = await embedder.embed([f"{task_type}: {query}" if task_type else query])
    min_sim = settings.recall_min_similarity
    items: list[Item] = []

    # Rules: vector match, boosted when the task type matches exactly.
    dist = Node.embedding.cosine_distance(qv)
    rows = (await session.execute(
        select(Node, HeuristicRule, dist.label("d")).join(HeuristicRule, HeuristicRule.node_id == Node.id)
        .where(Node.agent_id == agent_id, Node.kind == NodeKind.rule.value, Node.status == NodeStatus.active.value)
        .order_by(dist).limit(k)
    )).all()
    for node, rule, d in rows:
        sim = 1.0 - float(d) + (0.15 if task_type and rule.task_type == task_type else 0.0)
        if sim < min_sim:
            continue
        items.append(Item("rules", node.id, node.label, sim * node.confidence, {
            "rule_id": str(node.id), "condition": rule.condition, "avoid": rule.avoid, "action": rule.action,
            "rationale": rule.rationale, "confidence": round(node.confidence, 3), "similarity": round(sim, 3),
            "evidence_count": node.evidence_count, "task_type": rule.task_type,
        }))

    # Facts: vector hits on facts and entities, then one hop along the graph.
    live = [NodeStatus.active.value, NodeStatus.pending_review.value]
    rows = (await session.execute(
        select(Node, dist.label("d"))
        .where(Node.agent_id == agent_id, Node.kind.in_([NodeKind.fact.value, NodeKind.entity.value]),
               Node.status.in_(live), Node.embedding.is_not(None))
        .order_by(dist).limit(k)
    )).all()
    hits = {node.id: (node, 1.0 - float(d)) for node, d in rows if 1.0 - float(d) >= min_sim}
    if hits:
        edges = (await session.execute(
            select(Edge).where(Edge.agent_id == agent_id, Edge.status == NodeStatus.active.value,
                               or_(Edge.src_id.in_(list(hits)), Edge.dst_id.in_(list(hits)))).limit(k * 6)
        )).scalars().all()
        neighbor_sim: dict[uuid.UUID, float] = {}
        for e in edges:
            for a, b in ((e.src_id, e.dst_id), (e.dst_id, e.src_id)):
                if a in hits and b not in hits:
                    neighbor_sim[b] = max(neighbor_sim.get(b, 0.0), hits[a][1] * 0.7)  # one hop away: discounted
        if neighbor_sim:
            neighbors = (await session.execute(
                select(Node).where(Node.id.in_(list(neighbor_sim)), Node.kind == NodeKind.fact.value,
                                   Node.status.in_(live))
            )).scalars().all()
            for n in neighbors:
                hits[n.id] = (n, neighbor_sim[n.id])
    for node, sim in hits.values():
        if node.kind != NodeKind.fact.value:
            continue  # entities only steer the hop; facts carry the content
        disputed = node.status == NodeStatus.pending_review.value
        text = node.label + (" (disputed: conflicting evidence)" if disputed else "")
        score = sim * node.confidence * scoring.recency(_days(node.last_evidence_at, now), 90)
        items.append(Item("facts", node.id, text, score, {
            "fact_id": str(node.id), **node.attrs, "confidence": round(node.confidence, 3),
            "similarity": round(sim, 3), "disputed": disputed, "evidence_count": node.evidence_count,
        }))

    # Episodes: raw fallback for things consolidation has not captured (yet).
    edist = Episode.embedding.cosine_distance(qv)
    rows = (await session.execute(
        select(Episode, edist.label("d"))
        .where(Episode.agent_id == agent_id, Episode.embedding.is_not(None)).order_by(edist).limit(k)
    )).all()
    for ep, d in rows:
        sim = 1.0 - float(d)
        if sim < min_sim:
            continue
        detail = f" ({ep.outcome_detail})" if ep.outcome_detail else ""
        text = f"[{ep.outcome}] {ep.summary}{detail}"
        items.append(Item("episodes", ep.id, text, sim * scoring.recency(_days(ep.created_at, now), 30), {
            "episode_id": str(ep.id), "summary": ep.summary, "outcome": ep.outcome,
            "outcome_detail": ep.outcome_detail, "task_type": ep.task_type, "similarity": round(sim, 3),
            "created_at": ep.created_at.isoformat(),
        }))

    chosen, used = pack(items, budget_tokens)

    node_ids = [i.id for i in chosen if i.section in ("rules", "facts")]
    if node_ids:
        await session.execute(update(Node).where(Node.id.in_(node_ids)).values(
            access_count=Node.access_count + 1, last_accessed_at=now))

    return {
        "rules": [i.data for i in chosen if i.section == "rules"],
        "facts": [i.data for i in chosen if i.section == "facts"],
        "episodes": [i.data for i in chosen if i.section == "episodes"],
        "applied_rule_ids": [str(i.id) for i in chosen if i.section == "rules"],
        "context": render(chosen),
        "tokens_used": used,
        "budget_tokens": budget_tokens,
    }
