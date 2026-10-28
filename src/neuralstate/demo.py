"""A scripted agent that learns from its mistakes, end to end through the REST API.

The simulated ops agent deploys web apps for clients. Its default habit (`npm install` on
a stale lockfile) breaks the build. After a few failures, a consolidation cycle turns them
into a heuristic rule; on the next deployments the agent recalls that rule, changes its
approach and succeeds, and reports the rule as applied so its effectiveness is tracked.
A second thread shows contradictory facts being settled as evidence accumulates.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Callable

from neuralstate.sdk import NeuralStateClient, NeuralStateError

BAD = "run npm install against the committed lockfile"
GOOD = "regenerate the lockfile with npm ci on the target Node version, then build"
FAIL_DETAIL = "build failed: lockfile out of date (npm ERR! ERESOLVE could not resolve dependency tree)"


class OpsAgent:
    """Picks an approach. Without memory it always uses its habit; with memory it obeys rules."""

    def __init__(self, ns: NeuralStateClient, say: Callable[[str], None]):
        self.ns, self.say = ns, say

    def deploy(self, client: str, node_version: str) -> str:
        task = f"Deploy the web storefront for {client} to production"
        memory = self.ns.recall(task, task_type="deploy", budget_tokens=800)
        approach, applied = BAD, []
        if memory["rules"]:  # a real agent would put memory["context"] in its prompt; this one just obeys
            rule = memory["rules"][0]
            approach, applied = rule["action"], [rule["rule_id"]]
            self.say(f"    recalled rule ({rule['confidence']:.0%} confidence): avoid {rule['avoid']!r}")
        ok = approach != BAD  # the world: the habit always breaks, anything that rebuilds the lockfile works
        self.ns.record(
            summary=task, task_type="deploy", session_id=f"{client}-{int(time.time())}",
            outcome="success" if ok else "failure", outcome_detail=None if ok else FAIL_DETAIL,
            applied_rule_ids=applied,
            content={
                "approach": approach,
                "messages": [
                    {"role": "user", "content": task},
                    {"role": "assistant", "content": f"Plan: {approach}."},
                    {"role": "tool", "content": "Deployed." if ok else FAIL_DETAIL},
                ],
                "facts": [
                    {"subject": client, "predicate": "storefront runs on", "object": f"node {node_version}"},
                    {"subject": client, "predicate": "deploys to", "object": "production cluster",
                     "object_is_entity": True},
                ],
            },
        )
        self.say(f"  {'SUCCESS' if ok else 'FAILURE'}  {client}: {approach}")
        return "success" if ok else "failure"


def consolidate(ns: NeuralStateClient) -> dict:
    """Run a cycle now; if a background worker is mid-cycle for this agent, wait for it."""
    for _ in range(60):
        try:
            return ns.consolidate(wait=True)["stats"]
        except NeuralStateError as exc:
            if exc.status != 409:
                raise
            time.sleep(2)
    raise TimeoutError("agent stayed busy for 2 minutes")


def run(base_url: str, api_key: str | None = None, agent_name: str | None = None,
        say: Callable[[str], None] = print) -> dict:
    name = agent_name or f"demo-ops-{datetime.now():%Y%m%d-%H%M%S}"
    with NeuralStateClient(base_url, agent=name, api_key=api_key, timeout=600) as ns:
        agent = OpsAgent(ns, say)
        say(f"Agent {name!r}\n")

        say("1) Five deployments with no learned memory:")
        for i, client in enumerate(["acme", "globex", "initech", "umbrella", "hooli"]):
            agent.deploy(client, f"{18 + i % 3}")

        say("\n2) A consolidation cycle (the 'sleep'):")
        stats = consolidate(ns)
        syn = stats["synthesize"]
        say(f"  extracted {stats['extract']['facts_created']} facts from "
            f"{stats['extract']['episodes_processed']} episodes; "
            f"learned {syn['rules_created']} rule(s) from {syn['failures_considered']} failures")
        for text in syn.get("new_rules", []):
            say(f"  NEW RULE: {text}")

        say("\n3) New clients, same task, with memory:")
        results = [agent.deploy(c, "20") for c in ["stark", "wayne", "wonka"]]

        say("\n4) Conflicting facts settle themselves as evidence accumulates:")
        def observe(db: str, n: int = 1) -> None:
            for _ in range(n):
                ns.record(summary=f"Inspected acme's infrastructure: primary database is {db}",
                          content={"facts": [{"subject": "acme", "predicate": "primary database", "object": db}]})
        observe("postgres")
        consolidate(ns)
        observe("mysql")
        r = consolidate(ns)["reconcile"]
        say(f"  'acme primary database postgres' vs 'mysql': {r['contradictions_queued']} queued for review")
        observe("mysql", 3)
        r = consolidate(ns)["reconcile"]
        say(f"  three more mysql observations later: {r['contradictions_auto_resolved']} auto-resolved")

        rules = ns.rules()
        say("\nResult:")
        for rule in rules:
            say(f"  rule effectiveness {rule['effectiveness']:.0%} "
                f"({rule['success_after']} successes / {rule['failure_after']} failures while applied), "
                f"confidence {rule['confidence']:.0%}")
        say(f"  post-rule deployments: {results.count('success')}/{len(results)} succeeded")
        say(f"\nExplore it: {base_url.rstrip('/')}/ui?agent={name}")
        return {"agent": name, "rules": rules, "post_rule_results": results}
