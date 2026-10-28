"""REST API, version 1."""

from __future__ import annotations

import uuid
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from neuralstate.api.state import AppState, get_session, get_state, require_api_key
from neuralstate.core import queries
from neuralstate.core.ingest import EpisodeIn, get_or_create_agent, record_episode, set_outcome
from neuralstate.core.recall import recall
from neuralstate.core.revert import RevertError, revert_event
from neuralstate.db.models import Episode, MemoryEvent, Outcome
from neuralstate.worker import jobs
from neuralstate.worker.runner import AgentBusy, run_now

router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])


async def _agent_id(s: AsyncSession, ref: str) -> uuid.UUID:
    try:
        return (await queries.resolve_agent(s, ref)).id
    except queries.NotFound as exc:
        raise HTTPException(404, str(exc)) from exc


# ------------------------------------------------------------------ agents
class AgentIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)


@router.post("/agents", summary="Create an agent (or return the existing one with this name)")
async def create_agent(body: AgentIn, s: AsyncSession = Depends(get_session)) -> dict[str, Any]:
    agent = await get_or_create_agent(s, body.name)
    return {"id": str(agent.id), "name": agent.name}


@router.get("/agents")
async def agents(s: AsyncSession = Depends(get_session)) -> list[dict[str, Any]]:
    return await queries.list_agents(s)


@router.get("/agents/{agent}/overview")
async def agent_overview(agent: str, s: AsyncSession = Depends(get_session)) -> dict[str, Any]:
    return await queries.overview(s, await _agent_id(s, agent))


# ------------------------------------------------------------------ episodes
@router.post("/episodes", status_code=201, summary="Record an interaction episode")
async def create_episode(body: EpisodeIn, s: AsyncSession = Depends(get_session),
                         st: AppState = Depends(get_state)) -> dict[str, Any]:
    await _agent_id(s, str(body.agent_id))
    ep = await record_episode(s, st.providers.embedder, st.settings, body)
    return queries.episode_dict(ep)


class OutcomeIn(BaseModel):
    outcome: Outcome
    outcome_detail: str | None = None


@router.post("/episodes/{episode_id}/outcome", summary="Attach or correct an episode's outcome")
async def episode_outcome(episode_id: uuid.UUID, body: OutcomeIn, s: AsyncSession = Depends(get_session),
                          st: AppState = Depends(get_state)) -> dict[str, Any]:
    ep = await s.get(Episode, episode_id)
    if ep is None:
        raise HTTPException(404, "episode not found")
    await set_outcome(s, st.providers.embedder, st.settings, ep, body.outcome, body.outcome_detail)
    return queries.episode_dict(ep)


@router.get("/episodes")
async def episodes(agent: str, outcome: Outcome | None = None, limit: int = Query(50, le=500),
                   s: AsyncSession = Depends(get_session)) -> list[dict[str, Any]]:
    return await queries.list_episodes(s, await _agent_id(s, agent), outcome.value if outcome else None, limit)


@router.get("/episodes/{episode_id}")
async def episode(episode_id: uuid.UUID, s: AsyncSession = Depends(get_session)) -> dict[str, Any]:
    ep = await s.get(Episode, episode_id)
    if ep is None:
        raise HTTPException(404, "episode not found")
    return queries.episode_dict(ep, full=True)


# ------------------------------------------------------------------ recall
class RecallIn(BaseModel):
    agent_id: str = Field(..., description="Agent id or name")
    query: str = Field(..., min_length=1, description="What the agent is about to do")
    task_type: str | None = None
    budget_tokens: int = Field(2000, ge=50, le=200_000)
    k: int | None = Field(None, ge=1, le=100)


@router.post("/recall", summary="Retrieve memory for a task, packed into a token budget")
async def do_recall(body: RecallIn, s: AsyncSession = Depends(get_session),
                    st: AppState = Depends(get_state)) -> dict[str, Any]:
    return await recall(s, st.providers.embedder, st.settings, agent_id=await _agent_id(s, body.agent_id),
                        query=body.query, task_type=body.task_type, budget_tokens=body.budget_tokens, k=body.k)


# ------------------------------------------------------------------ graph
@router.get("/graph")
async def get_graph(agent: str, center: uuid.UUID | None = None, depth: int = Query(1, ge=0, le=4),
                    limit: int = Query(300, le=2000), include_inactive: bool = False,
                    s: AsyncSession = Depends(get_session)) -> dict[str, Any]:
    return await queries.graph(s, await _agent_id(s, agent), center, depth, limit, include_inactive)


@router.get("/nodes/{node_id}", summary="A node with its evidence episodes and change history")
async def node(node_id: uuid.UUID, s: AsyncSession = Depends(get_session)) -> dict[str, Any]:
    try:
        return await queries.node_detail(s, node_id)
    except queries.NotFound as exc:
        raise HTTPException(404, str(exc)) from exc


# ------------------------------------------------------------------ rules
@router.get("/rules")
async def rules(agent: str, status: str | None = None, s: AsyncSession = Depends(get_session)) -> list[dict[str, Any]]:
    return await queries.list_rules(s, await _agent_id(s, agent), status)


class RulePatch(BaseModel):
    status: Literal["active", "disabled"] | None = None
    condition: str | None = None
    avoid: str | None = None
    action: str | None = None
    rationale: str | None = None


@router.patch("/rules/{rule_id}", summary="Enable, disable or edit a rule (audited)")
async def patch_rule(rule_id: uuid.UUID, body: RulePatch, s: AsyncSession = Depends(get_session)) -> dict[str, Any]:
    try:
        return await queries.update_rule(s, rule_id, body.model_dump(exclude_none=True))
    except queries.NotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


# ------------------------------------------------------------------ review queue
@router.get("/review", summary="Contradictions waiting for evidence or an operator")
async def review(agent: str, status: Literal["pending", "auto", "resolved"] = "pending",
                 s: AsyncSession = Depends(get_session)) -> list[dict[str, Any]]:
    return await queries.list_review(s, await _agent_id(s, agent), status)


class ResolveIn(BaseModel):
    winner: Literal["a", "b", "both"]
    note: str | None = None


@router.post("/review/{contradiction_id}/resolve")
async def resolve(contradiction_id: uuid.UUID, body: ResolveIn,
                  s: AsyncSession = Depends(get_session)) -> dict[str, Any]:
    try:
        return await queries.resolve_review(s, contradiction_id, body.winner, body.note)
    except queries.NotFound as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


# ------------------------------------------------------------------ audit log
@router.get("/events", summary="Audit log of memory mutations, newest first")
async def events(agent: str, op: str | None = None, limit: int = Query(100, le=1000), before_id: int | None = None,
                 s: AsyncSession = Depends(get_session)) -> list[dict[str, Any]]:
    return await queries.list_events(s, await _agent_id(s, agent), op, limit, before_id)


@router.post("/events/{event_id}/revert", summary="Undo one memory mutation")
async def revert(event_id: int, s: AsyncSession = Depends(get_session)) -> dict[str, Any]:
    event = await s.get(MemoryEvent, event_id)
    if event is None:
        raise HTTPException(404, "event not found")
    try:
        await revert_event(s, event)
    except RevertError as exc:
        raise HTTPException(409, str(exc)) from exc
    return queries.event_dict(event)


# ------------------------------------------------------------------ consolidation
class ConsolidateIn(BaseModel):
    agent_id: str
    kind: Literal["cycle", "synthesize"] = "cycle"
    wait: bool = Field(False, description="Run now in the API process and return stats, instead of queueing")


@router.post("/consolidate", summary="Trigger a consolidation cycle")
async def consolidate(body: ConsolidateIn, s: AsyncSession = Depends(get_session),
                      st: AppState = Depends(get_state)) -> dict[str, Any]:
    agent_id = await _agent_id(s, body.agent_id)
    if not body.wait:
        await jobs.enqueue(s, agent_id, body.kind)
        return {"queued": True, "kind": body.kind}
    await s.commit()
    try:
        stats = await run_now(st.engine, st.sessions, st.providers, st.settings, agent_id, body.kind)
    except AgentBusy as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"queued": False, "kind": body.kind, "stats": stats}
