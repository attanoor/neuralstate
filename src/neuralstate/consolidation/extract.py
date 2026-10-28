"""Stage 1: turn unconsolidated episodes into entities, facts, edges and provenance."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from neuralstate.consolidation import scoring
from neuralstate.consolidation.context import CycleContext
from neuralstate.consolidation.reasoner import norm
from neuralstate.consolidation.types import EpisodeView, Extraction, ExtractedFact
from neuralstate.core.graph import add_edge, add_provenance
from neuralstate.db import audit
from neuralstate.db.models import Episode, Node, NodeKind, NodeStatus
from neuralstate.db.session import session_scope

LIVE = (NodeStatus.active.value, NodeStatus.pending_review.value)


def fact_key(subject: str, predicate: str) -> str:
    return f"{norm(subject)}|{norm(predicate)}"[:500]


def fact_label(f: ExtractedFact) -> str:
    return f"{f.subject.strip()} {f.predicate.strip()} {f.object.strip()}"


def to_view(ep: Episode) -> EpisodeView:
    return EpisodeView(id=ep.id, task_type=ep.task_type, summary=ep.summary, outcome=ep.outcome,
                       outcome_detail=ep.outcome_detail, content=ep.content or {})


class _Applier:
    """Writes one batch of extractions inside a single transaction."""

    def __init__(self, ctx: CycleContext, session: AsyncSession, vectors: dict[str, list[float]]):
        self.ctx, self.s, self.vectors = ctx, session, vectors
        self.stats = {"entities_created": 0, "facts_created": 0, "facts_reinforced": 0}

    def _event_create(self, node: Node) -> None:
        audit.record(self.s, agent_id=self.ctx.agent_id, op="create", target=node, target_id=node.id,
                     before=None, after=audit.snapshot(node), cycle_id=self.ctx.cycle_id)

    async def entity(self, name: str, etype: str, episode_id: uuid.UUID) -> Node:
        key = norm(name)[:500]
        node = (await self.s.execute(select(Node).where(
            Node.agent_id == self.ctx.agent_id, Node.kind == NodeKind.entity.value, Node.key == key,
            Node.status.in_(LIVE)).limit(1))).scalar_one_or_none()
        if node is None:
            node = Node(id=uuid.uuid4(), agent_id=self.ctx.agent_id, kind=NodeKind.entity.value, key=key,
                        label=name.strip(), attrs={"type": etype}, embedding=self.vectors.get(name),
                        base_confidence=0.6, confidence=0.6, evidence_count=1, last_evidence_at=self.ctx.now)
            self.s.add(node)
            await self.s.flush()
            self._event_create(node)
            self.stats["entities_created"] += 1
        else:
            node.evidence_count += 1
            node.last_evidence_at = self.ctx.now
        await add_provenance(self.s, node.id, [episode_id])
        return node

    def _reinforce(self, node: Node, confidence: float) -> None:
        node.evidence_count += 1
        node.base_confidence = scoring.reinforce(node.base_confidence, confidence)
        node.confidence = node.base_confidence
        node.last_evidence_at = self.ctx.now
        self.stats["facts_reinforced"] += 1

    async def _merge_target(self, key: str, obj: str) -> Node | None:
        """A paraphrase already judged duplicate: follow `merged_into` to the surviving fact."""
        merged = (await self.s.execute(select(Node).where(
            Node.agent_id == self.ctx.agent_id, Node.kind == NodeKind.fact.value, Node.key == key,
            Node.status == NodeStatus.superseded.value, Node.attrs.has_key("merged_into")))).scalars().all()
        for m in merged:
            if norm(m.attrs.get("object", "")) == norm(obj):
                target = await self.s.get(Node, uuid.UUID(m.attrs["merged_into"]))
                if target is not None and target.status in LIVE:
                    return target
        return None

    async def fact(self, f: ExtractedFact, episode_id: uuid.UUID) -> None:
        settings = self.ctx.settings
        subject = await self.entity(f.subject, "thing", episode_id)
        if f.object_is_entity:
            obj = await self.entity(f.object, "thing", episode_id)
            await add_edge(self.s, self.ctx.agent_id, subject.id, obj.id, norm(f.predicate))

        key, label = fact_key(f.subject, f.predicate), fact_label(f)
        vector = self.vectors.get(label)
        same_key = (await self.s.execute(select(Node).where(
            Node.agent_id == self.ctx.agent_id, Node.kind == NodeKind.fact.value, Node.key == key,
            Node.status.in_(LIVE)))).scalars().all()

        # Same triple already known: this episode is more evidence for it.
        target = next((n for n in same_key if norm(n.attrs.get("object", "")) == norm(f.object)), None)
        if target is None:
            target = await self._merge_target(key, f.object)
        if target is not None:
            self._reinforce(target, f.confidence)
            await add_provenance(self.s, target.id, [episode_id])
            return

        # Facts that merely embed close together are NOT merged here: "acme uses mysql" and
        # "acme uses postgres" are near-identical vectors yet contradict. Close neighbors go to
        # the reconcile stage, where the reasoner decides duplicate / contradiction / consistent.
        neighbors: list[tuple[Node, float]] = []
        if vector is not None:
            dist = Node.embedding.cosine_distance(vector)
            rows = (await self.s.execute(select(Node, dist.label("d")).where(
                Node.agent_id == self.ctx.agent_id, Node.kind == NodeKind.fact.value, Node.status.in_(LIVE),
                Node.embedding.is_not(None)).order_by(dist).limit(5))).all()
            neighbors = [(n, 1.0 - float(d)) for n, d in rows]

        conf = max(0.05, min(1.0, f.confidence))
        node = Node(id=uuid.uuid4(), agent_id=self.ctx.agent_id, kind=NodeKind.fact.value, key=key, label=label,
                    attrs={"subject": f.subject.strip(), "predicate": f.predicate.strip(), "object": f.object.strip()},
                    embedding=vector, base_confidence=conf, confidence=conf, evidence_count=1,
                    last_evidence_at=self.ctx.now)
        self.s.add(node)
        await self.s.flush()
        self._event_create(node)
        self.stats["facts_created"] += 1
        await add_provenance(self.s, node.id, [episode_id])
        await add_edge(self.s, self.ctx.agent_id, subject.id, node.id, "has_fact")

        # Anything that might conflict goes to the reconcile stage.
        for existing in same_key:
            self.ctx.candidate_pairs.add((node.id, existing.id))
        for other, sim in neighbors:
            if sim >= settings.reconcile_neighbor_threshold and other.id != node.id:
                self.ctx.candidate_pairs.add((node.id, other.id))


async def run(ctx: CycleContext) -> dict[str, Any]:
    stats = {"episodes_processed": 0, "entities_created": 0, "facts_created": 0, "facts_reinforced": 0}
    batch = ctx.settings.extract_batch_size
    while stats["episodes_processed"] < ctx.settings.max_episodes_per_cycle:
        async with session_scope(ctx.sessions) as s:
            episodes = (await s.execute(
                select(Episode).where(Episode.agent_id == ctx.agent_id, Episode.consolidated_at.is_(None))
                .order_by(Episode.created_at).limit(batch)
            )).scalars().all()
        if not episodes:
            break
        views = [to_view(e) for e in episodes]
        extractions: list[Extraction] = await ctx.reasoner.extract(views)

        names: list[str] = []
        for x in extractions:
            names += [e.name for e in x.entities]
            for f in x.facts:
                names += [f.subject, fact_label(f)] + ([f.object] if f.object_is_entity else [])
        names = list(dict.fromkeys(n for n in names if n.strip()))
        vectors = dict(zip(names, await ctx.embedder.embed(names))) if names else {}

        async with session_scope(ctx.sessions) as s:
            applier = _Applier(ctx, s, vectors)
            for x in extractions:
                episode_id = views[x.episode_index].id
                for e in x.entities:
                    await applier.entity(e.name, e.type, episode_id)
                for f in x.facts:
                    if f.subject.strip() and f.predicate.strip() and f.object.strip():
                        await applier.fact(f, episode_id)
            for ep in episodes:
                ep_db = await s.get(Episode, ep.id)
                if ep_db is not None:
                    ep_db.consolidated_at = ctx.now
        for k, v in applier.stats.items():
            stats[k] += v
        stats["episodes_processed"] += len(episodes)
    return stats
