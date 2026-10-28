"""Stage 3: learn heuristic rules from repeated failures.

Failures that no rule covers yet are clustered by embedding within each task type. A
cluster with at least `min_rule_evidence` episodes becomes a candidate rule, which must
pass a validation check (does it explain the failures without contradicting known
successes?) before it is stored. A candidate that matches an existing rule reinforces it
instead; one that matches a retired or disabled rule is suppressed, since memory already
knows that rule didn't help or wasn't wanted.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import timedelta
from typing import Any

import numpy as np
from sqlalchemy import select

from neuralstate.consolidation import scoring
from neuralstate.consolidation.cluster import centroid, cluster
from neuralstate.consolidation.context import CycleContext
from neuralstate.consolidation.extract import to_view
from neuralstate.consolidation.reasoner import norm
from neuralstate.consolidation.types import RuleDraft
from neuralstate.core.graph import add_provenance
from neuralstate.db import audit
from neuralstate.db.audit import Changes
from neuralstate.db.models import Episode, HeuristicRule, Node, NodeKind, NodeStatus, Outcome, Provenance
from neuralstate.db.session import session_scope

MAX_FAILURES_PER_TYPE = 300
PROMPT_FAILURES = 12
PROMPT_SUCCESSES = 5


async def _uncovered_failures(ctx: CycleContext) -> list[Episode]:
    since = ctx.now - timedelta(days=ctx.settings.synthesis_window_days)
    covered = (
        select(Provenance.episode_id).join(Node, Node.id == Provenance.node_id)
        .where(Node.agent_id == ctx.agent_id, Node.kind == NodeKind.rule.value)
    )
    async with session_scope(ctx.sessions) as s:
        return list((await s.execute(
            select(Episode).where(
                Episode.agent_id == ctx.agent_id, Episode.outcome == Outcome.failure.value,
                Episode.created_at >= since, Episode.embedding.is_not(None), Episode.id.not_in(covered))
            .order_by(Episode.created_at.desc()).limit(MAX_FAILURES_PER_TYPE * 10)
        )).scalars().all())


async def _similar_successes(ctx: CycleContext, task_type: str | None, center: list[float]) -> list[Episode]:
    dist = Episode.embedding.cosine_distance(center)
    q = select(Episode).where(Episode.agent_id == ctx.agent_id, Episode.outcome == Outcome.success.value,
                              Episode.embedding.is_not(None))
    if task_type is not None:
        q = q.where(Episode.task_type == task_type)
    async with session_scope(ctx.sessions) as s:
        return list((await s.execute(q.order_by(dist).limit(PROMPT_SUCCESSES))).scalars().all())


def rule_vector(situation: list[float], text_vector: list[float], situation_weight: float = 0.7) -> list[float]:
    """Embed a rule mostly by *where it applies*: the centroid of the failures it came from.

    Recall queries describe the task about to be done, which looks like those episodes far
    more than it looks like the rule's own wording.
    """
    v = situation_weight * np.asarray(situation) + (1 - situation_weight) * np.asarray(text_vector)
    return (v / (np.linalg.norm(v) or 1.0)).tolist()


async def _store(ctx: CycleContext, draft: RuleDraft, task_type: str | None, members: list[Episode],
                 situation: list[float], stats: dict[str, Any]) -> None:
    text = draft.text()
    [text_vector] = await ctx.embedder.embed([text])
    vector = rule_vector(situation, text_vector)
    episode_ids = [e.id for e in members]
    s_min = ctx.settings.min_rule_evidence
    async with session_scope(ctx.sessions) as s:
        dist = Node.embedding.cosine_distance(vector)
        nearest = (await s.execute(
            select(Node, HeuristicRule, dist.label("d")).join(HeuristicRule, HeuristicRule.node_id == Node.id)
            .where(Node.agent_id == ctx.agent_id, Node.kind == NodeKind.rule.value).order_by(dist).limit(1)
        )).first()
        if nearest is not None and 1.0 - float(nearest.d) >= ctx.settings.rule_merge_threshold:
            node, rule = nearest.Node, nearest.HeuristicRule
            # Either way these failures are now accounted for and won't be re-clustered.
            await add_provenance(s, node.id, episode_ids)
            if node.status in (NodeStatus.archived.value, NodeStatus.disabled.value):
                stats["rules_suppressed"] += 1
                return
            with Changes(s, node, agent_id=ctx.agent_id, op="reinforce_rule", cycle_id=ctx.cycle_id,
                         note=f"{len(members)} more failures matched this rule"):
                node.evidence_count += len(members)
                node.base_confidence = scoring.evidence_confidence(node.evidence_count, s_min)
                node.confidence = scoring.rule_confidence(node.evidence_count, rule.success_after,
                                                          rule.failure_after, s_min)
                node.last_evidence_at = ctx.now
            stats["rules_reinforced"] += 1
            return

        evidence = len(members)
        node = Node(
            id=uuid.uuid4(), agent_id=ctx.agent_id, kind=NodeKind.rule.value, key=norm(draft.condition)[:500],
            label=text, body=draft.rationale,
            attrs={"condition": draft.condition, "avoid": draft.avoid, "action": draft.action,
                   "task_type": task_type, "reasoner": ctx.reasoner.name},
            embedding=vector, base_confidence=scoring.evidence_confidence(evidence, s_min),
            confidence=scoring.rule_confidence(evidence, 0, 0, s_min), evidence_count=evidence,
            last_evidence_at=ctx.now,
        )
        s.add(node)
        await s.flush()
        s.add(HeuristicRule(node_id=node.id, task_type=task_type, condition=draft.condition, avoid=draft.avoid,
                            action=draft.action, rationale=draft.rationale))
        await add_provenance(s, node.id, episode_ids)
        audit.record(s, agent_id=ctx.agent_id, op="create", target=node, target_id=node.id, before=None,
                     after=audit.snapshot(node), cycle_id=ctx.cycle_id,
                     note=f"synthesized from {evidence} failures")
        stats["rules_created"] += 1
        stats.setdefault("new_rules", []).append(text)


async def run(ctx: CycleContext) -> dict[str, Any]:
    stats: dict[str, Any] = {"failures_considered": 0, "clusters": 0, "rules_created": 0, "rules_reinforced": 0,
                             "rules_suppressed": 0, "rules_rejected": 0, "no_rule": 0}
    failures = await _uncovered_failures(ctx)
    stats["failures_considered"] = len(failures)
    by_type: dict[str | None, list[Episode]] = defaultdict(list)
    for ep in failures:
        if len(by_type[ep.task_type]) < MAX_FAILURES_PER_TYPE:
            by_type[ep.task_type].append(ep)

    for task_type, eps in by_type.items():
        if len(eps) < ctx.settings.min_rule_evidence:
            continue
        vectors = np.asarray([e.embedding for e in eps], dtype=np.float64)
        for idx in cluster(vectors, ctx.settings.cluster_threshold):
            if len(idx) < ctx.settings.min_rule_evidence:
                continue
            stats["clusters"] += 1
            members = [eps[i] for i in idx]
            fail_views = [to_view(e) for e in sorted(members, key=lambda e: e.created_at, reverse=True)]
            fail_views = fail_views[:PROMPT_FAILURES]
            center = centroid(vectors[idx])
            ok_views = [to_view(e) for e in await _similar_successes(ctx, task_type, center)]

            draft = await ctx.reasoner.synthesize_rule(fail_views, ok_views)
            if draft is None:
                stats["no_rule"] += 1
                continue
            check = await ctx.reasoner.validate_rule(draft, fail_views, ok_views)
            if not check.explains_failures or check.contradicts_success:
                stats["rules_rejected"] += 1
                stats.setdefault("rejections", []).append({"rule": draft.text(), "notes": check.notes})
                continue
            await _store(ctx, draft, task_type, members, center, stats)
    return stats
