"""End-to-end memory behavior against a real Postgres + pgvector (heuristic reasoner, hash embeddings)."""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import func, select

from neuralstate.consolidation.engine import run_cycle
from neuralstate.core.ingest import EpisodeIn, get_or_create_agent, record_episode
from neuralstate.core.recall import recall
from neuralstate.core.revert import revert_event
from neuralstate.db.models import (
    ConsolidationJob,
    Contradiction,
    HeuristicRule,
    MemoryEvent,
    Node,
    NodeKind,
    NodeStatus,
    Outcome,
)
from neuralstate.db.session import session_scope
from neuralstate.worker import jobs
from neuralstate.worker.runner import process_one

pytestmark = pytest.mark.integration

BAD = "run npm install against the stale lockfile"
GOOD = "regenerate the lockfile with npm ci before deploying"


async def _agent(sessions, name="tester"):
    async with session_scope(sessions) as s:
        return (await get_or_create_agent(s, name)).id


async def _ingest(sessions, providers, settings, agent_id, n, **kw):
    ids = []
    for i in range(n):
        async with session_scope(sessions) as s:
            ep = await record_episode(s, providers.embedder, settings, EpisodeIn(agent_id=agent_id, **kw))
            ids.append(ep.id)
    return ids


async def _deploy_failures(sessions, providers, settings, agent_id, n=5):
    for client in [f"client-{i}" for i in range(n)]:
        await _ingest(sessions, providers, settings, agent_id, 1,
                      summary=f"Deploy web app for {client}", task_type="deploy",
                      outcome=Outcome.failure, outcome_detail="build broke: lockfile out of date",
                      content={"approach": BAD, "facts": [
                          {"subject": client, "predicate": "deploys with", "object": "npm"}]})


async def test_rule_emerges_from_repeated_failures(db, settings, providers):
    engine, sessions = db
    agent_id = await _agent(sessions)
    await _deploy_failures(sessions, providers, settings, agent_id, 5)
    await _ingest(sessions, providers, settings, agent_id, 2, summary="Deploy web app for client-9",
                  task_type="deploy", outcome=Outcome.success, content={"approach": GOOD})

    stats = await run_cycle(sessions, providers, settings, agent_id)
    assert stats["extract"]["episodes_processed"] == 7
    assert stats["extract"]["facts_created"] == 5
    assert stats["synthesize"]["rules_created"] == 1, stats

    async with session_scope(sessions) as s:
        rule = (await s.execute(select(HeuristicRule))).scalar_one()
        assert rule.avoid == BAD and rule.action == GOOD

    async with session_scope(sessions) as s:
        result = await recall(s, providers.embedder, settings, agent_id=agent_id,
                              query="Deploy web app for client-42", task_type="deploy")
    assert result["rules"] and result["rules"][0]["avoid"] == BAD
    assert result["context"].startswith("Learned rules")
    rule_id = result["applied_rule_ids"][0]

    # The rule is credited when the agent succeeds with it in context ...
    await _ingest(sessions, providers, settings, agent_id, 1, summary="Deploy for client-42", task_type="deploy",
                  outcome=Outcome.success, applied_rule_ids=[rule_id], content={"approach": GOOD})
    async with session_scope(sessions) as s:
        rule = (await s.execute(select(HeuristicRule))).scalar_one()
        assert (rule.success_after, rule.failure_after) == (1, 0)

    # ... and the same failures don't produce a second rule next cycle.
    stats = await run_cycle(sessions, providers, settings, agent_id)
    assert stats["synthesize"]["rules_created"] == 0
    assert stats["synthesize"]["failures_considered"] == 0


async def test_too_few_failures_make_no_rule(db, settings, providers):
    _, sessions = db
    agent_id = await _agent(sessions)
    await _deploy_failures(sessions, providers, settings, agent_id, settings.min_rule_evidence - 1)
    stats = await run_cycle(sessions, providers, settings, agent_id)
    assert stats["synthesize"]["rules_created"] == 0


async def test_ineffective_rule_retires(db, settings, providers):
    _, sessions = db
    agent_id = await _agent(sessions)
    await _deploy_failures(sessions, providers, settings, agent_id, 5)
    await run_cycle(sessions, providers, settings, agent_id)
    async with session_scope(sessions) as s:
        rule_id = (await s.execute(select(HeuristicRule.node_id))).scalar_one()
    await _ingest(sessions, providers, settings, agent_id, settings.rule_archive_min_trials,
                  summary="Deploy again", task_type="deploy", outcome=Outcome.failure,
                  outcome_detail="still broken", applied_rule_ids=[rule_id], content={"approach": "something else"})
    stats = await run_cycle(sessions, providers, settings, agent_id)
    assert stats["decay"]["rules_retired"] == 1
    async with session_scope(sessions) as s:
        assert (await s.get(Node, rule_id)).status == NodeStatus.archived.value


def _db_fact(obj):
    return {"facts": [{"subject": "acme-corp", "predicate": "primary database", "object": obj}]}


async def test_contradiction_queues_then_resolves_with_evidence(db, settings, providers):
    _, sessions = db
    agent_id = await _agent(sessions)
    await _ingest(sessions, providers, settings, agent_id, 1, summary="Checked acme db", content=_db_fact("postgres"))
    await run_cycle(sessions, providers, settings, agent_id)
    await _ingest(sessions, providers, settings, agent_id, 1, summary="Checked acme db", content=_db_fact("mysql"))
    stats = await run_cycle(sessions, providers, settings, agent_id)
    assert stats["reconcile"]["contradictions_queued"] == 1

    async with session_scope(sessions) as s:
        statuses = dict((await s.execute(select(Node.attrs["object"].astext, Node.status)
                                         .where(Node.kind == NodeKind.fact.value))).all())
        assert statuses == {"postgres": "pending_review", "mysql": "pending_review"}
        result = await recall(s, providers.embedder, settings, agent_id=agent_id, query="acme-corp primary database")
        assert any(f["disputed"] for f in result["facts"])

    # Three more observations of mysql tip the balance; the pair resolves without an operator.
    await _ingest(sessions, providers, settings, agent_id, 3, summary="Checked acme db", content=_db_fact("mysql"))
    stats = await run_cycle(sessions, providers, settings, agent_id)
    assert stats["reconcile"]["contradictions_auto_resolved"] == 1, stats
    async with session_scope(sessions) as s:
        statuses = dict((await s.execute(select(Node.attrs["object"].astext, Node.status)
                                         .where(Node.kind == NodeKind.fact.value))).all())
        assert statuses == {"postgres": "superseded", "mysql": "active"}
        c = (await s.execute(select(Contradiction))).scalar_one()
        assert c.status == "auto"

    # Every change is in the audit log and can be undone.
    async with session_scope(sessions) as s:
        event = (await s.execute(select(MemoryEvent).where(MemoryEvent.op == "supersede"))).scalar_one()
        await revert_event(s, event)
    async with session_scope(sessions) as s:
        postgres = (await s.execute(select(Node).where(Node.attrs["object"].astext == "postgres"))).scalar_one()
        assert postgres.status == NodeStatus.pending_review.value


class SameVectorEmbedder:
    """Worst case for vector-based merging: every text embeds identically."""

    name = "same"

    def __init__(self, dim):
        self.dim = dim

    async def embed(self, texts):
        return [[1.0] + [0.0] * (self.dim - 1) for _ in texts]


async def test_near_identical_embeddings_do_not_hide_a_contradiction(db, settings, providers):
    _, sessions = db
    providers.embedder = SameVectorEmbedder(settings.embed_dim)
    agent_id = await _agent(sessions)
    await _ingest(sessions, providers, settings, agent_id, 1, summary="Checked acme db", content=_db_fact("postgres"))
    await _ingest(sessions, providers, settings, agent_id, 1, summary="Checked acme db", content=_db_fact("mysql"))
    stats = await run_cycle(sessions, providers, settings, agent_id)
    assert stats["extract"]["facts_created"] == 2
    assert stats["reconcile"]["contradictions_queued"] == 1


async def test_enqueue_is_idempotent_and_workers_do_not_overlap(db, settings, providers):
    engine, sessions = db
    agent_id = await _agent(sessions)
    await _deploy_failures(sessions, providers, settings, agent_id, 5)  # queues one debounced synthesize job
    async with session_scope(sessions) as s:
        await jobs.enqueue(s, agent_id, "cycle")
        await jobs.enqueue(s, agent_id, "cycle")
        await s.execute(ConsolidationJob.__table__.update().values(run_after=func.now()))
        assert (await s.execute(select(func.count()).select_from(ConsolidationJob))).scalar() == 2

    # Two workers race for the two jobs of the same agent: the advisory lock serializes them.
    results = await asyncio.gather(process_one(engine, sessions, providers, settings),
                                   process_one(engine, sessions, providers, settings))
    assert all(results)
    async with session_scope(sessions) as s:  # a deferred job waits 15s; make it due now
        await s.execute(ConsolidationJob.__table__.update().values(run_after=func.now()))
    while await process_one(engine, sessions, providers, settings):
        pass
    async with session_scope(sessions) as s:
        statuses = (await s.execute(select(ConsolidationJob.status))).scalars().all()
        assert set(statuses) == {"done"}
        assert (await s.execute(select(func.count()).select_from(HeuristicRule))).scalar() == 1
