"""Recording episodes: the fast, append-only write path agents call after every interaction."""

from __future__ import annotations

import json
import uuid
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from neuralstate.config import Settings
from neuralstate.consolidation import scoring
from neuralstate.db.models import Agent, Episode, HeuristicRule, Node, Outcome
from neuralstate.providers.base import Embedder
from neuralstate.worker import jobs


class EpisodeIn(BaseModel):
    agent_id: uuid.UUID
    summary: str | None = Field(None, description="One-line description; derived from content when omitted")
    content: dict[str, Any] = Field(default_factory=dict,
                                    description="Raw interaction log: messages, actions, tool calls, …")
    task_type: str | None = None
    session_id: str | None = None
    outcome: Outcome = Outcome.unknown
    outcome_detail: str | None = None
    applied_rule_ids: list[uuid.UUID] = Field(default_factory=list,
                                              description="Rule ids from recall() that were in the agent's context")


def episode_text(summary: str, task_type: str | None, outcome: str, outcome_detail: str | None) -> str:
    """Text that gets embedded. Includes the outcome so similar failures cluster together."""
    parts = [f"{task_type}:" if task_type else "", summary]
    if outcome != Outcome.unknown:
        parts.append(f"Outcome: {outcome}.")
    if outcome_detail:
        parts.append(outcome_detail)
    return " ".join(p for p in parts if p)


def derive_summary(content: dict[str, Any], limit: int = 300) -> str:
    for key in ("summary", "task", "goal", "query", "input"):
        if isinstance(content.get(key), str) and content[key].strip():
            return content[key].strip()[:limit]
    return json.dumps(content, ensure_ascii=False, default=str)[:limit] or "(empty episode)"


async def get_or_create_agent(session: AsyncSession, name: str) -> Agent:
    agent = (await session.execute(select(Agent).where(Agent.name == name))).scalar_one_or_none()
    if agent is None:
        agent = Agent(name=name)
        session.add(agent)
        await session.flush()
    return agent


async def apply_rule_feedback(session: AsyncSession, settings: Settings, agent_id: uuid.UUID,
                              rule_ids: list[Any], old_outcome: str, new_outcome: str) -> None:
    """Credit or blame the rules that were in context when the agent acted."""
    delta = {Outcome.success.value: (1, 0), Outcome.failure.value: (0, 1)}
    old, new = delta.get(old_outcome, (0, 0)), delta.get(new_outcome, (0, 0))
    ds, df = new[0] - old[0], new[1] - old[1]
    if not rule_ids or (ds == 0 and df == 0):
        return
    ids = [uuid.UUID(str(r)) for r in rule_ids]
    rows = (await session.execute(
        select(Node, HeuristicRule).join(HeuristicRule, HeuristicRule.node_id == Node.id)
        .where(Node.agent_id == agent_id, Node.id.in_(ids))
    )).all()
    for node, rule in rows:
        rule.success_after = max(0, rule.success_after + ds)
        rule.failure_after = max(0, rule.failure_after + df)
        node.confidence = scoring.rule_confidence(node.evidence_count, rule.success_after, rule.failure_after,
                                                  settings.min_rule_evidence)


async def record_episode(session: AsyncSession, embedder: Embedder, settings: Settings, data: EpisodeIn) -> Episode:
    summary = data.summary or derive_summary(data.content)
    [vector] = await embedder.embed([episode_text(summary, data.task_type, data.outcome, data.outcome_detail)])
    episode = Episode(
        agent_id=data.agent_id, session_id=data.session_id, task_type=data.task_type, content=data.content,
        summary=summary, outcome=data.outcome.value, outcome_detail=data.outcome_detail,
        applied_rule_ids=[str(r) for r in data.applied_rule_ids], embedding=vector,
    )
    session.add(episode)
    await session.flush()
    await apply_rule_feedback(session, settings, data.agent_id, episode.applied_rule_ids,
                              Outcome.unknown.value, episode.outcome)
    if episode.outcome == Outcome.failure.value:
        # Failures are what rules are made of; check soon, but debounce bursts.
        await jobs.enqueue(session, data.agent_id, "synthesize", delay_seconds=30)
    return episode


async def set_outcome(session: AsyncSession, embedder: Embedder, settings: Settings, episode: Episode,
                      outcome: Outcome, detail: str | None) -> Episode:
    """Attach (or correct) an outcome after the fact, e.g. when a deploy finishes hours later."""
    old = episode.outcome
    episode.outcome = outcome.value
    episode.outcome_detail = detail if detail is not None else episode.outcome_detail
    [episode.embedding] = await embedder.embed(
        [episode_text(episode.summary, episode.task_type, episode.outcome, episode.outcome_detail)])
    await apply_rule_feedback(session, settings, episode.agent_id, episode.applied_rule_ids, old, episode.outcome)
    if episode.outcome == Outcome.failure.value and old != episode.outcome:
        await jobs.enqueue(session, episode.agent_id, "synthesize", delay_seconds=30)
    return episode

