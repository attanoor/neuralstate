"""Stage 4: forget gracefully.

Confidence of unused facts decays from their evidence-based `base_confidence` (recomputed,
never compounded). Cold, low-confidence memories are archived, never deleted. Rules are
exempt from time decay; they retire only when outcomes show they don't help.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from sqlalchemy import extract, func, literal, select, update

from neuralstate.consolidation import scoring
from neuralstate.consolidation.context import CycleContext
from neuralstate.db.audit import Changes
from neuralstate.db.models import Episode, HeuristicRule, Node, NodeKind, NodeStatus
from neuralstate.db.session import session_scope

LIVE = (NodeStatus.active.value, NodeStatus.pending_review.value)


async def run(ctx: CycleContext) -> dict[str, Any]:
    st = ctx.settings
    stats = {"nodes_decayed": 0, "nodes_archived": 0, "rules_retired": 0, "episodes_compacted": 0}
    async with session_scope(ctx.sessions) as s:
        last_touch = func.greatest(Node.last_evidence_at, func.coalesce(Node.last_accessed_at, Node.created_at))
        idle_days = extract("epoch", literal(ctx.now) - last_touch) / 86400.0
        new_conf = Node.base_confidence * func.exp(-st.decay_lambda * func.greatest(0.0, idle_days - st.decay_grace_days))
        result = await s.execute(
            update(Node).where(Node.agent_id == ctx.agent_id, Node.kind != NodeKind.rule.value,
                               Node.status.in_(LIVE), func.abs(Node.confidence - new_conf) > 0.005)
            .values(confidence=new_conf).execution_options(synchronize_session=False))
        stats["nodes_decayed"] = result.rowcount or 0

        cold_before = ctx.now - timedelta(days=st.archive_after_days)
        cold = (await s.execute(select(Node).where(
            Node.agent_id == ctx.agent_id, Node.kind != NodeKind.rule.value, Node.status.in_(LIVE),
            Node.confidence < st.archive_threshold, last_touch < cold_before))).scalars().all()
        for node in cold:
            with Changes(s, node, agent_id=ctx.agent_id, op="archive", cycle_id=ctx.cycle_id,
                         note=f"unused and low confidence ({node.confidence:.2f})"):
                node.status = NodeStatus.archived.value
            stats["nodes_archived"] += 1

        rules = (await s.execute(select(Node, HeuristicRule).join(HeuristicRule, HeuristicRule.node_id == Node.id)
                                 .where(Node.agent_id == ctx.agent_id, Node.status == NodeStatus.active.value))).all()
        for node, rule in rules:
            if scoring.rule_should_retire(rule.success_after, rule.failure_after, st.rule_archive_min_trials,
                                          st.rule_archive_effectiveness):
                eff = scoring.rule_effectiveness(rule.success_after, rule.failure_after)
                with Changes(s, node, agent_id=ctx.agent_id, op="retire_rule", cycle_id=ctx.cycle_id,
                             note=f"not helping: {rule.success_after} successes / {rule.failure_after} failures "
                                  f"while applied (effectiveness {eff:.2f})"):
                    node.status = NodeStatus.archived.value
                stats["rules_retired"] += 1

        if st.compact_episodes:
            # Raw content of old, consolidated episodes moves into the audit log (still revertible).
            old = (await s.execute(select(Episode).where(
                Episode.agent_id == ctx.agent_id, Episode.compacted.is_(False),
                Episode.consolidated_at.is_not(None),
                Episode.created_at < ctx.now - timedelta(days=st.compact_after_days)).limit(500))).scalars().all()
            for ep in old:
                with Changes(s, ep, agent_id=ctx.agent_id, op="compact", cycle_id=ctx.cycle_id):
                    ep.content = {}
                    ep.compacted = True
                stats["episodes_compacted"] += 1
    return stats
