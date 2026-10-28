"""Stage 2: merge duplicates and settle contradictions between facts.

A contradiction is settled automatically only when one side is clearly better supported
(score margin >= `contradiction_margin`). Otherwise both facts are marked `pending_review`
and the pair is re-scored every cycle, so it resolves itself once evidence accumulates on
one side — or an operator settles it from the review queue first.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from neuralstate.consolidation import scoring
from neuralstate.consolidation.context import CycleContext
from neuralstate.consolidation.types import FactView
from neuralstate.core.graph import add_edge, copy_provenance
from neuralstate.db.audit import Changes
from neuralstate.db.models import Contradiction, ContradictionStatus, Node, NodeKind, NodeStatus, utcnow
from neuralstate.db.session import session_scope

LIVE = (NodeStatus.active.value, NodeStatus.pending_review.value)
JUDGE_CHUNK = 20


def node_strength(node: Node, now) -> float:
    days = (now - node.last_evidence_at).total_seconds() / 86400
    return scoring.strength(node.evidence_count, node.base_confidence, days)


def _view(n: Node) -> FactView:
    return FactView(id=n.id, subject=n.attrs.get("subject", ""), predicate=n.attrs.get("predicate", ""),
                    object=n.attrs.get("object", n.label))


async def _set_status(s: AsyncSession, agent_id: uuid.UUID, cycle_id: uuid.UUID | None, node: Node, status: str,
                      op: str, note: str) -> None:
    if node.status == status:
        return
    with Changes(s, node, agent_id=agent_id, op=op, cycle_id=cycle_id, note=note):
        node.status = status


async def _has_other_pending(s: AsyncSession, node_id: uuid.UUID, exclude: uuid.UUID) -> bool:
    row = (await s.execute(select(Contradiction.id).where(
        Contradiction.status == ContradictionStatus.pending.value, Contradiction.id != exclude,
        or_(Contradiction.node_a == node_id, Contradiction.node_b == node_id)).limit(1))).first()
    return row is not None


async def settle(s: AsyncSession, c: Contradiction, winner: Node, loser: Node, status: str, resolution: str,
                 cycle_id: uuid.UUID | None = None) -> None:
    """Supersede `loser` in favor of `winner` and close the contradiction."""
    await _set_status(s, c.agent_id, cycle_id, loser, NodeStatus.superseded.value, "supersede",
                      f"superseded by {winner.label!r}: {resolution}")
    if winner.status == NodeStatus.pending_review.value and not await _has_other_pending(s, winner.id, c.id):
        await _set_status(s, c.agent_id, cycle_id, winner, NodeStatus.active.value, "restore",
                          f"won contradiction: {resolution}")
    await add_edge(s, c.agent_id, winner.id, loser.id, "supersedes")
    c.winner_id = winner.id
    c.status = status
    c.resolution = resolution
    c.resolved_at = utcnow()


async def _merge(ctx: CycleContext, s: AsyncSession, keep: Node, drop: Node, reason: str) -> None:
    keep.evidence_count += drop.evidence_count
    keep.base_confidence = scoring.reinforce(keep.base_confidence, drop.base_confidence)
    keep.confidence = keep.base_confidence
    keep.last_evidence_at = max(keep.last_evidence_at, drop.last_evidence_at)
    await copy_provenance(s, drop.id, keep.id)
    with Changes(s, drop, agent_id=ctx.agent_id, op="merge", cycle_id=ctx.cycle_id,
                 note=f"duplicate of {keep.label!r}: {reason}"):
        drop.status = NodeStatus.superseded.value
        drop.attrs = {**drop.attrs, "merged_into": str(keep.id)}


async def _judge_new_pairs(ctx: CycleContext, stats: dict[str, int]) -> None:
    pairs = {tuple(sorted(p, key=str)) for p in ctx.candidate_pairs if p[0] != p[1]}
    if not pairs:
        return
    async with session_scope(ctx.sessions) as s:
        ids = {i for p in pairs for i in p}
        nodes = {n.id: n for n in (await s.execute(select(Node).where(
            Node.id.in_(ids), Node.kind == NodeKind.fact.value, Node.status.in_(LIVE)))).scalars()}
        known = {(a, b) for a, b in (await s.execute(select(Contradiction.node_a, Contradiction.node_b).where(
            Contradiction.agent_id == ctx.agent_id))).all()}
    todo = [p for p in pairs if p[0] in nodes and p[1] in nodes and p not in known and (p[1], p[0]) not in known]

    for start in range(0, len(todo), JUDGE_CHUNK):
        chunk = todo[start:start + JUDGE_CHUNK]
        verdicts = await ctx.reasoner.judge_pairs([(_view(nodes[a]), _view(nodes[b])) for a, b in chunk])
        async with session_scope(ctx.sessions) as s:
            for v in verdicts:
                a_id, b_id = chunk[v.pair_index]
                a, b = await s.get(Node, a_id), await s.get(Node, b_id)
                if a is None or b is None or a.status not in LIVE or b.status not in LIVE:
                    continue
                stats["pairs_judged"] += 1
                if v.verdict == "duplicate":
                    keep, drop = (a, b) if a.created_at <= b.created_at else (b, a)
                    await _merge(ctx, s, keep, drop, v.reason)
                    stats["duplicates_merged"] += 1
                elif v.verdict == "contradiction":
                    await _contradiction(ctx, s, a, b, v.reason, stats)


async def _contradiction(ctx: CycleContext, s: AsyncSession, a: Node, b: Node, reason: str,
                         stats: dict[str, int]) -> None:
    side, margin = scoring.contradiction_margin(node_strength(a, ctx.now), node_strength(b, ctx.now))
    winner, loser = (a, b) if side == "a" else (b, a)
    c = Contradiction(agent_id=ctx.agent_id, node_a=a.id, node_b=b.id, margin=margin, reason=reason,
                      status=ContradictionStatus.pending.value)
    s.add(c)
    await s.flush()
    if margin >= ctx.settings.contradiction_margin:
        await settle(s, c, winner, loser, ContradictionStatus.auto.value,
                     f"auto: better supported (margin {margin:.2f})", ctx.cycle_id)
        stats["contradictions_auto_resolved"] += 1
    else:
        for n in (a, b):
            await _set_status(s, ctx.agent_id, ctx.cycle_id, n, NodeStatus.pending_review.value, "flag",
                              f"conflicts with another fact: {reason}")
        stats["contradictions_queued"] += 1


async def _rescore_pending(ctx: CycleContext, stats: dict[str, int]) -> None:
    async with session_scope(ctx.sessions) as s:
        pending = (await s.execute(select(Contradiction).where(
            Contradiction.agent_id == ctx.agent_id,
            Contradiction.status == ContradictionStatus.pending.value))).scalars().all()
        for c in pending:
            a, b = await s.get(Node, c.node_a), await s.get(Node, c.node_b)
            if a is None or b is None:
                continue
            side, margin = scoring.contradiction_margin(node_strength(a, ctx.now), node_strength(b, ctx.now))
            c.margin = margin
            if margin >= ctx.settings.contradiction_margin:
                winner, loser = (a, b) if side == "a" else (b, a)
                await settle(s, c, winner, loser, ContradictionStatus.auto.value,
                             f"auto: evidence accumulated (margin {margin:.2f})", ctx.cycle_id)
                stats["contradictions_auto_resolved"] += 1


async def run(ctx: CycleContext) -> dict[str, Any]:
    stats = {"pairs_judged": 0, "duplicates_merged": 0, "contradictions_auto_resolved": 0,
             "contradictions_queued": 0}
    await _judge_new_pairs(ctx, stats)
    await _rescore_pending(ctx, stats)
    ctx.candidate_pairs.clear()
    return stats
