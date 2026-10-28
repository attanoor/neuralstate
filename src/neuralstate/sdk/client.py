"""Thin clients for agents: record what happened, recall what was learned.

    from neuralstate.sdk import NeuralStateClient

    ns = NeuralStateClient("http://127.0.0.1:8000", agent="support-bot")
    memory = ns.recall("Refund a duplicated charge for acme-corp", task_type="billing")
    prompt = memory["context"] + "\\n\\n" + task                 # inject into the agent's context
    ...
    ns.record(summary="Refunded duplicate charge", task_type="billing", outcome="success",
              content={"messages": transcript}, applied_rule_ids=memory["applied_rule_ids"])
"""

from __future__ import annotations

from typing import Any

import httpx


class NeuralStateError(RuntimeError):
    def __init__(self, status: int, detail: Any):
        super().__init__(f"{status}: {detail}")
        self.status, self.detail = status, detail


def _check(r: httpx.Response) -> Any:
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise NeuralStateError(r.status_code, detail)
    return r.json()


def _episode_body(agent_id: str, summary: str | None, content: dict | None, task_type: str | None,
                  outcome: str, outcome_detail: str | None, session_id: str | None,
                  applied_rule_ids: list[str] | None) -> dict[str, Any]:
    return {"agent_id": agent_id, "summary": summary, "content": content or {}, "task_type": task_type,
            "outcome": outcome, "outcome_detail": outcome_detail, "session_id": session_id,
            "applied_rule_ids": applied_rule_ids or []}


class NeuralStateClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", agent: str | None = None,
                 api_key: str | None = None, timeout: float = 60.0):
        headers = {"X-API-Key": api_key} if api_key else {}
        self.http = httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)
        self._agent_name, self._agent_id = agent, None

    def __enter__(self) -> "NeuralStateClient":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def close(self) -> None:
        self.http.close()

    @property
    def agent_id(self) -> str:
        if self._agent_id is None:
            if not self._agent_name:
                raise ValueError("pass agent=... to the client")
            self._agent_id = self.create_agent(self._agent_name)["id"]
        return self._agent_id

    def create_agent(self, name: str) -> dict[str, Any]:
        return _check(self.http.post("/v1/agents", json={"name": name}))

    def record(self, *, summary: str | None = None, content: dict | None = None, task_type: str | None = None,
               outcome: str = "unknown", outcome_detail: str | None = None, session_id: str | None = None,
               applied_rule_ids: list[str] | None = None) -> dict[str, Any]:
        """Record one episode. Pass `applied_rule_ids` from `recall()` so rules get credit or blame."""
        return _check(self.http.post("/v1/episodes", json=_episode_body(
            self.agent_id, summary, content, task_type, outcome, outcome_detail, session_id, applied_rule_ids)))

    def set_outcome(self, episode_id: str, outcome: str, outcome_detail: str | None = None) -> dict[str, Any]:
        return _check(self.http.post(f"/v1/episodes/{episode_id}/outcome",
                                     json={"outcome": outcome, "outcome_detail": outcome_detail}))

    def recall(self, query: str, task_type: str | None = None, budget_tokens: int = 2000) -> dict[str, Any]:
        return _check(self.http.post("/v1/recall", json={"agent_id": self.agent_id, "query": query,
                                                         "task_type": task_type, "budget_tokens": budget_tokens}))

    def rules(self, status: str | None = None) -> list[dict[str, Any]]:
        params = {"agent": self.agent_id, **({"status": status} if status else {})}
        return _check(self.http.get("/v1/rules", params=params))

    def overview(self) -> dict[str, Any]:
        return _check(self.http.get(f"/v1/agents/{self.agent_id}/overview"))

    def consolidate(self, wait: bool = False, kind: str = "cycle") -> dict[str, Any]:
        r = self.http.post("/v1/consolidate", json={"agent_id": self.agent_id, "kind": kind, "wait": wait},
                           timeout=None if wait else self.http.timeout)
        return _check(r)


class AsyncNeuralStateClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", agent: str | None = None,
                 api_key: str | None = None, timeout: float = 60.0):
        headers = {"X-API-Key": api_key} if api_key else {}
        self.http = httpx.AsyncClient(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)
        self._agent_name, self._agent_id = agent, None

    async def __aenter__(self) -> "AsyncNeuralStateClient":
        return self

    async def __aexit__(self, *_) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self.http.aclose()

    async def agent_id(self) -> str:
        if self._agent_id is None:
            if not self._agent_name:
                raise ValueError("pass agent=... to the client")
            self._agent_id = (await self.create_agent(self._agent_name))["id"]
        return self._agent_id

    async def create_agent(self, name: str) -> dict[str, Any]:
        return _check(await self.http.post("/v1/agents", json={"name": name}))

    async def record(self, *, summary: str | None = None, content: dict | None = None, task_type: str | None = None,
                     outcome: str = "unknown", outcome_detail: str | None = None, session_id: str | None = None,
                     applied_rule_ids: list[str] | None = None) -> dict[str, Any]:
        return _check(await self.http.post("/v1/episodes", json=_episode_body(
            await self.agent_id(), summary, content, task_type, outcome, outcome_detail, session_id,
            applied_rule_ids)))

    async def set_outcome(self, episode_id: str, outcome: str, outcome_detail: str | None = None) -> dict[str, Any]:
        return _check(await self.http.post(f"/v1/episodes/{episode_id}/outcome",
                                           json={"outcome": outcome, "outcome_detail": outcome_detail}))

    async def recall(self, query: str, task_type: str | None = None, budget_tokens: int = 2000) -> dict[str, Any]:
        return _check(await self.http.post("/v1/recall", json={
            "agent_id": await self.agent_id(), "query": query, "task_type": task_type,
            "budget_tokens": budget_tokens}))

    async def rules(self, status: str | None = None) -> list[dict[str, Any]]:
        params = {"agent": await self.agent_id(), **({"status": status} if status else {})}
        return _check(await self.http.get("/v1/rules", params=params))

    async def consolidate(self, wait: bool = False, kind: str = "cycle") -> dict[str, Any]:
        r = await self.http.post("/v1/consolidate", json={"agent_id": await self.agent_id(), "kind": kind,
                                                          "wait": wait},
                                 timeout=None if wait else self.http.timeout)
        return _check(r)
