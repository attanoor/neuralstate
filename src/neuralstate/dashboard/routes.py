"""Server-rendered dashboard (Jinja2 + HTMX). Shares every query with the REST API."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from neuralstate.api.state import AppState, get_session, get_state, key_matches
from neuralstate.core import queries
from neuralstate.core.revert import RevertError, revert_event
from neuralstate.db.models import MemoryEvent

HERE = Path(__file__).parent
templates = Jinja2Templates(directory=HERE / "templates")
templates.env.filters["pct"] = lambda v: f"{round(float(v) * 100)}%"
templates.env.filters["tojson_pretty"] = lambda v: json.dumps(v, indent=2, ensure_ascii=False, default=str)
templates.env.filters["short"] = lambda v, n=80: (v[: n - 1] + "…") if v and len(v) > n else (v or "")
templates.env.filters["when"] = lambda v: (v or "")[:19].replace("T", " ")

COOKIE = "ns_key"


class LoginRequired(Exception):
    pass


def ui_auth(request: Request, state: AppState = Depends(get_state)) -> None:
    if not key_matches(state.settings.api_key, request.cookies.get(COOKIE)):
        raise LoginRequired


router = APIRouter(include_in_schema=False)  # public: login only
protected = APIRouter(prefix="/ui", include_in_schema=False, dependencies=[Depends(ui_auth)])


def render(request: Request, name: str, **ctx: Any) -> HTMLResponse:
    return templates.TemplateResponse(request, name, ctx)


async def _page_ctx(s: AsyncSession, agent: str | None, page: str) -> dict[str, Any]:
    agents = await queries.list_agents(s)
    current = None
    if agent:
        try:
            a = await queries.resolve_agent(s, agent)
            current = {"id": str(a.id), "name": a.name}
        except queries.NotFound:
            raise HTTPException(404, f"agent {agent} not found")
    elif agents:
        current = agents[0]
    return {"agents": agents, "agent": current, "page": page}


# ---------------------------------------------------------------- auth
@router.get("/ui/login")
async def login_form(request: Request) -> HTMLResponse:
    return render(request, "login.html", error=None, page="login", agent={"name": ""})


@router.post("/ui/login")
async def login(request: Request, key: str = Form(...), state: AppState = Depends(get_state)) -> Response:
    if not key_matches(state.settings.api_key, key):
        return render(request, "login.html", error="That key is not valid.", page="login",
                      agent={"name": ""})
    response = RedirectResponse("/ui", status_code=303)
    response.set_cookie(COOKIE, key, httponly=True, samesite="strict")
    return response


# ---------------------------------------------------------------- pages
@protected.get("")
async def overview_page(request: Request, agent: str | None = None, s: AsyncSession = Depends(get_session)):
    ctx = await _page_ctx(s, agent, "overview")
    if ctx["agent"]:
        ctx["data"] = await queries.overview(s, uuid.UUID(ctx["agent"]["id"]))
    return render(request, "overview.html", **ctx)


@protected.get("/graph")
async def graph_page(request: Request, agent: str | None = None, s: AsyncSession = Depends(get_session)):
    return render(request, "graph.html", **await _page_ctx(s, agent, "graph"))


@protected.get("/graph.json")
async def graph_json(agent: str, center: uuid.UUID | None = None, depth: int = 1, include_inactive: bool = False,
                     s: AsyncSession = Depends(get_session)) -> JSONResponse:
    a = await queries.resolve_agent(s, agent)
    return JSONResponse(await queries.graph(s, a.id, center, min(depth, 4), 400, include_inactive))


@protected.get("/nodes/{node_id}")
async def node_panel(request: Request, node_id: uuid.UUID, s: AsyncSession = Depends(get_session)):
    try:
        node = await queries.node_detail(s, node_id)
    except queries.NotFound:
        raise HTTPException(404, "node not found")
    return render(request, "_node.html", node=node)


@protected.get("/rules")
async def rules_page(request: Request, agent: str | None = None, status: str | None = None,
                     s: AsyncSession = Depends(get_session)):
    ctx = await _page_ctx(s, agent, "rules")
    ctx["status"] = status
    ctx["rules"] = await queries.list_rules(s, uuid.UUID(ctx["agent"]["id"]), status) if ctx["agent"] else []
    return render(request, "rules.html", **ctx)


@protected.post("/rules/{rule_id}/status")
async def rule_status(request: Request, rule_id: uuid.UUID, status: str = Form(...),
                      s: AsyncSession = Depends(get_session)):
    try:
        rule = await queries.update_rule(s, rule_id, {"status": status})
    except (queries.NotFound, ValueError) as exc:
        raise HTTPException(422, str(exc))
    return render(request, "_rule_row.html", rule=rule)


@protected.get("/review")
async def review_page(request: Request, agent: str | None = None, status: str = "pending",
                      s: AsyncSession = Depends(get_session)):
    ctx = await _page_ctx(s, agent, "review")
    ctx["status"] = status
    ctx["items"] = await queries.list_review(s, uuid.UUID(ctx["agent"]["id"]), status) if ctx["agent"] else []
    return render(request, "review.html", **ctx)


@protected.post("/review/{contradiction_id}/resolve")
async def review_resolve(request: Request, contradiction_id: uuid.UUID, winner: str = Form(...),
                         note: str | None = Form(None), s: AsyncSession = Depends(get_session)):
    try:
        result = await queries.resolve_review(s, contradiction_id, winner, note)
    except (queries.NotFound, ValueError) as exc:
        return HTMLResponse(f'<p class="notice error">{exc}</p>', status_code=200)
    return render(request, "_resolved.html", result=result, winner=winner)


@protected.get("/events")
async def events_page(request: Request, agent: str | None = None, op: str | None = None,
                      before_id: int | None = None, s: AsyncSession = Depends(get_session)):
    ctx = await _page_ctx(s, agent, "events")
    ctx["op"] = op
    ctx["events"] = (await queries.list_events(s, uuid.UUID(ctx["agent"]["id"]), op, 100, before_id)
                     if ctx["agent"] else [])
    return render(request, "events.html", **ctx)


@protected.post("/events/{event_id}/revert")
async def event_revert(request: Request, event_id: int, s: AsyncSession = Depends(get_session)):
    event = await s.get(MemoryEvent, event_id)
    if event is None:
        raise HTTPException(404, "event not found")
    try:
        await revert_event(s, event)
    except RevertError as exc:
        return HTMLResponse(f'<td colspan="5" class="notice error">{exc}</td>')
    return render(request, "_event_row.html", e=queries.event_dict(event))


router.include_router(protected)
