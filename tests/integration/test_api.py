"""REST API and dashboard through the ASGI app (real Postgres, offline providers)."""

from __future__ import annotations

import httpx
import pytest

from neuralstate.api.app import create_app

pytestmark = pytest.mark.integration


@pytest.fixture
async def client(db, settings, providers):
    settings.api_key = "secret"
    app = create_app(settings, providers)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test",
                                     headers={"X-API-Key": "secret"}) as c:
            yield c


def _failure(agent_id, client):
    return {"agent_id": agent_id, "summary": f"Deploy storefront for {client}", "task_type": "deploy",
            "outcome": "failure", "outcome_detail": "lockfile out of date",
            "content": {"approach": "npm install", "facts": [
                {"subject": client, "predicate": "runs on", "object": "node 18"}]}}


async def test_full_api_flow(client):
    assert (await client.get("/health")).json()["status"] == "ok"
    agent = (await client.post("/v1/agents", json={"name": "api-bot"})).json()
    assert (await client.post("/v1/agents", json={"name": "api-bot"})).json()["id"] == agent["id"]

    for c in ["a", "b", "c", "d", "e"]:
        r = await client.post("/v1/episodes", json=_failure(agent["id"], c))
        assert r.status_code == 201, r.text

    r = await client.post("/v1/consolidate", json={"agent_id": "api-bot", "wait": True})
    assert r.json()["stats"]["synthesize"]["rules_created"] == 1

    rules = (await client.get("/v1/rules", params={"agent": "api-bot"})).json()
    assert len(rules) == 1 and rules[0]["avoid"] == "npm install"
    rule_id = rules[0]["id"]

    memory = (await client.post("/v1/recall", json={"agent_id": "api-bot", "query": "Deploy storefront for zeta",
                                                     "task_type": "deploy"})).json()
    assert memory["applied_rule_ids"] == [rule_id]

    ep = (await client.post("/v1/episodes", json={"agent_id": agent["id"], "summary": "Deploy for zeta",
                                                   "task_type": "deploy", "applied_rule_ids": [rule_id]})).json()
    r = await client.post(f"/v1/episodes/{ep['id']}/outcome", json={"outcome": "success"})
    assert r.json()["outcome"] == "success"
    detail = (await client.get(f"/v1/nodes/{rule_id}")).json()
    assert detail["success_after"] == 1 and len(detail["evidence"]) == 5

    r = await client.patch(f"/v1/rules/{rule_id}", json={"status": "disabled"})
    assert r.json()["status"] == "disabled"
    memory = (await client.post("/v1/recall", json={"agent_id": "api-bot", "query": "Deploy storefront"})).json()
    assert memory["rules"] == []

    events = (await client.get("/v1/events", params={"agent": "api-bot", "op": "operator_edit"})).json()
    r = await client.post(f"/v1/events/{events[0]['id']}/revert")
    assert r.status_code == 200 and r.json()["reverted"]
    assert (await client.get(f"/v1/nodes/{rule_id}")).json()["status"] == "active"
    assert (await client.post(f"/v1/events/{events[0]['id']}/revert")).status_code == 409

    graph = (await client.get("/v1/graph", params={"agent": "api-bot"})).json()
    assert graph["nodes"] and graph["edges"]
    overview = (await client.get(f"/v1/agents/api-bot/overview")).json()
    assert overview["nodes"]["rule"] == 1 and overview["episodes"]["failure"] == 5


async def test_api_key_required(client):
    r = await client.get("/v1/agents", headers={"X-API-Key": "wrong"})
    assert r.status_code == 401


async def test_dashboard_pages_render(client):
    await client.post("/v1/agents", json={"name": "ui-bot"})
    agent = (await client.get("/v1/agents")).json()[0]
    for c in "abcde":
        await client.post("/v1/episodes", json=_failure(agent["id"], c))
    await client.post("/v1/consolidate", json={"agent_id": "ui-bot", "wait": True})

    r = await client.get("/ui", follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/ui/login"
    r = await client.post("/ui/login", data={"key": "secret"}, follow_redirects=False)
    assert r.status_code == 303
    client.cookies.set("ns_key", "secret")

    for path in ["/ui", "/ui/graph", "/ui/rules", "/ui/review", "/ui/events", "/ui/static/app.css"]:
        r = await client.get(path, params={"agent": "ui-bot"})
        assert r.status_code == 200, (path, r.text[:500])
    assert "npm install" in (await client.get("/ui/rules", params={"agent": "ui-bot"})).text

    graph = (await client.get("/ui/graph.json", params={"agent": "ui-bot"})).json()
    node_id = graph["nodes"][0]["id"]
    assert (await client.get(f"/ui/nodes/{node_id}")).status_code == 200

    rule_id = (await client.get("/v1/rules", params={"agent": "ui-bot"})).json()[0]["id"]
    r = await client.post(f"/ui/rules/{rule_id}/status", data={"status": "disabled"})
    assert r.status_code == 200 and "Enable" in r.text
